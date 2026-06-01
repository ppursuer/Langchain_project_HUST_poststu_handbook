import requests
import json
from qdrant_client import QdrantClient

# ==================== 配置 ====================
# Qdrant配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

# 向量模型配置（BGE-M3）
BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

# LLM配置（Qwen2.5-7B-Instruct）
LLM_API_URL = "http://10.154.22.10:34525/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# RAG参数
TOP_K = 5  # 检索相关文档数量
MAX_TOKENS = 1024  # 最大生成token数
TEMPERATURE = 0.3  # 温度参数
# ==============================================


def get_embedding(text):
    """调用BGE-M3 API获取向量"""
    payload = {
        "model": BGE_M3_MODEL,
        "input": text
    }
    response = requests.post(BGE_M3_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            return data["data"][0]["embedding"]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unexpected API response: {data}")
    else:
        raise Exception(f"API request failed: {response.status_code} {response.text}")


def search_in_qdrant(query, top_k=TOP_K):
    """在Qdrant中搜索相关文档"""
    query_vector = get_embedding(query)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    return results.points


def build_context(results):
    """构建上下文文本"""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        level1 = result.payload.get('level1', '')
        level2 = result.payload.get('level2', '')
        level3 = result.payload.get('level3', '')
        level4 = result.payload.get('level4', '')
        content = result.payload.get('content', '')
        
        # 构建标题
        header_parts = [p for p in [level1, level2, level3, level4] if p]
        header = " > ".join(header_parts)
        
        context_parts.append(f"【参考资料 {i}】\n{header}\n{content}")
    
    return "\n\n".join(context_parts)


def generate_answer(query, context):
    """调用LLM生成答案"""
    prompt = f"""你是华中科技大学研究生手册的智能助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{query}

请根据参考资料提供准确、完整的回答。如果参考资料中没有相关信息，请说明"根据现有资料无法回答该问题"。"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是华中科技大学研究生手册的智能助手，负责回答关于研究生培养、管理和学位授予的相关规定。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    
    response = requests.post(LLM_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected API response: {data}")
    else:
        raise Exception(f"LLM API request failed: {response.status_code} {response.text}")


def rag_query(query):
    """RAG查询主函数"""
    print("=" * 80)
    print(f"用户问题: {query}")
    print("=" * 80)
    
    # 步骤1: 检索相关文档
    print("\n[1/3] 正在检索相关文档...")
    results = search_in_qdrant(query)
    print(f"找到 {len(results)} 条相关文档")
    
    # 打印检索结果详情
    print("\n检索结果详情:")
    for i, result in enumerate(results, 1):
        level1 = result.payload.get('level1', '')
        level2 = result.payload.get('level2', '')
        level3 = result.payload.get('level3', '')
        level4 = result.payload.get('level4', '')
        content = result.payload.get('content', '')
        score = result.score
        
        header_parts = [p for p in [level1, level2, level3, level4] if p]
        header = " > ".join(header_parts)
        
        print(f"\n【文档 {i}】")
        print(f"相似度分数: {score:.4f}")
        print(f"一级标题: {level1}")
        print(f"二级标题: {level2}")
        print(f"三级标题: {level3}")
        print(f"四级标题: {level4}")
        print(f"内容:\n{content[:300]}...")
        print("-" * 80)
    
    # 步骤2: 构建上下文
    print("\n[2/3] 正在构建上下文...")
    context = build_context(results)
    
    # 步骤3: 生成答案
    print("\n[3/3] 正在生成答案...")
    answer = generate_answer(query, context)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("AI助手回答:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    return answer


def main():
    print("华中科技大学研究生手册 RAG 系统 (V1 - 向量检索)")
    print("=" * 80)
    
    query = "华中科技大学博士生培养目标中，对学术能力和专业能力分别提出哪些具体要求？"
    print(f"\n用户问题: {query}\n")
    
    try:
        rag_query(query)
    except Exception as e:
        print(f"\n查询失败: {e}\n")


if __name__ == "__main__":
    main()
