import re
import json
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ==================== 配置 ====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

MD_FILE = r"c:\Users\x1359\Desktop\rag_project\datas\full.md"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# ==============================================

def parse_markdown(filepath):
    """
    解析Markdown文件，提取标题层级和内容块。
    返回一个列表，每个元素是一个dict：
    {
        "level1": str,
        "level2": str,
        "level3": str,
        "level4": str,
        "content": str
    }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entries = []
    current_level1 = ""
    current_level2 = ""
    current_level3 = ""
    current_level4 = ""
    current_content = []

    def save_entry():
        nonlocal current_content
        text = '\n'.join(current_content).strip()
        if text:
            entries.append({
                "level1": current_level1,
                "level2": current_level2,
                "level3": current_level3,
                "level4": current_level4,
                "content": text
            })
        current_content = []

    for line in lines:
        stripped = line.rstrip('\n')

        # 一级标题
        m = re.match(r'^#\s+(.+)', stripped)
        if m:
            save_entry()
            current_level1 = m.group(1).strip()
            current_level2 = ""
            current_level3 = ""
            current_level4 = ""
            continue

        # 二级标题
        m = re.match(r'^##\s+(.+)', stripped)
        if m:
            save_entry()
            current_level2 = m.group(1).strip()
            current_level3 = ""
            current_level4 = ""
            continue

        # 三级标题
        m = re.match(r'^###\s+(.+)', stripped)
        if m:
            save_entry()
            current_level3 = m.group(1).strip()
            current_level4 = ""
            continue

        # 四级标题
        m = re.match(r'^####\s+(.+)', stripped)
        if m:
            save_entry()
            current_level4 = m.group(1).strip()
            continue

        current_content.append(stripped)

    # 保存最后一个条目
    save_entry()

    return entries


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    将文本按字符数分块，支持重叠。
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    """
    调用BGE-M3 API获取向量。
    """
    payload = {
        "model": BGE_M3_MODEL,
        "input": text
    }
    response = requests.post(BGE_M3_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        # 兼容不同的API响应格式
        if "data" in data:
            return data["data"][0]["embedding"]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unexpected API response: {data}")
    else:
        raise Exception(f"API request failed: {response.status_code} {response.text}")


def main():
    # 1. 解析Markdown
    print("Parsing markdown file...")
    entries = parse_markdown(MD_FILE)
    print(f"Found {len(entries)} entries.")

    # 2. 连接Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 3. 创建Collection
    # 先获取一个样本向量来确定维度
    sample_text = "测试文本"
    sample_embedding = get_embedding(sample_text)
    vector_size = len(sample_embedding)
    print(f"Vector size: {vector_size}")

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists, deleting...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Collection {COLLECTION_NAME} created.")

    # 4. 分块并插入Qdrant
    points = []
    total_chunks = 0

    for entry in entries:
        # 构建完整文本（包含标题信息）
        header_parts = []
        if entry["level1"]:
            header_parts.append(entry["level1"])
        if entry["level2"]:
            header_parts.append(entry["level2"])
        if entry["level3"]:
            header_parts.append(entry["level3"])
        if entry["level4"]:
            header_parts.append(entry["level4"])

        header = " > ".join(header_parts)
        full_text = entry["content"]

        # 分块
        chunks = chunk_text(full_text)

        for chunk in chunks:
            # 获取向量
            try:
                embedding = get_embedding(chunk)
            except Exception as e:
                print(f"Error getting embedding: {e}")
                continue

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "level1": entry["level1"],
                    "level2": entry["level2"],
                    "level3": entry["level3"],
                    "level4": entry["level4"],
                    "content": chunk,
                    "header": header
                }
            )
            points.append(point)
            total_chunks += 1

            # 批量插入（每100条）
            if len(points) >= 100:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"Inserted {total_chunks} chunks...")
                points = []

    # 插入剩余数据
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Inserted {total_chunks} chunks...")

    print(f"\nDone! Total chunks inserted: {total_chunks}")


if __name__ == "__main__":
    main()
