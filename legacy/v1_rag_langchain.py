import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# ==================== 配置 ====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

LLM_API_URL = "http://10.154.22.10:34525/v1"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 5
# ==============================================


class CustomEmbeddings(Embeddings):
    """自定义Embeddings类，调用BGE-M3 API"""
    
    def embed_documents(self, texts):
        payload = {"model": BGE_M3_MODEL, "input": texts}
        response = requests.post(BGE_M3_API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_vector_store():
    """获取Qdrant向量存储"""
    embeddings = CustomEmbeddings()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="content",
    )
    return vector_store


def search_with_metadata(query, top_k=TOP_K):
    """搜索并返回完整metadata"""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=top_k)
    
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    for doc in docs:
        doc_id = doc.metadata.get("_id", "")
        if doc_id:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[doc_id],
            )
            if points:
                point = points[0]
                doc.metadata["level1"] = point.payload.get("level1", "")
                doc.metadata["level2"] = point.payload.get("level2", "")
                doc.metadata["level3"] = point.payload.get("level3", "")
                doc.metadata["level4"] = point.payload.get("level4", "")
                doc.metadata["header"] = point.payload.get("header", "")
    
    return docs


def format_docs(docs):
    """格式化文档为上下文"""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        header = " > ".join([p for p in [
            doc.metadata.get('level1', ''),
            doc.metadata.get('level2', ''),
            doc.metadata.get('level3', ''),
            doc.metadata.get('level4', '')
        ] if p])
        context_parts.append(f"【参考资料 {i}】\n{header}\n{doc.page_content}")
    return "\n\n".join(context_parts)


def build_rag_chain():
    """构建RAG链"""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    
    prompt = ChatPromptTemplate.from_template("""你是华中科技大学研究生手册的智能助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

回答要求：
1. 直接回答用户的问题，不要提及"根据参考资料"、"根据资料"、"参考资料X"等类似表述。
2. 严格基于参考资料的内容进行回答，准确理解原文的语义和上下文，不要曲解或错误归类信息。
3. 如果参考资料中没有相关信息，请说明"根据现有资料无法回答该问题"。

请在回答的最后，另起一行，添加"参考资料来源："，并列出所有参考资料的标题路径（格式：level1 > level2 > level3 > level4）。""")
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key="not-needed",
        openai_api_base=LLM_API_URL,
        temperature=0.3,
        max_tokens=1024,
    )
    
    def enrich_metadata(docs):
        """为文档补充完整metadata"""
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        for doc in docs:
            doc_id = doc.metadata.get("_id", "")
            if doc_id:
                points = client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[doc_id],
                )
                if points:
                    point = points[0]
                    doc.metadata["level1"] = point.payload.get("level1", "")
                    doc.metadata["level2"] = point.payload.get("level2", "")
                    doc.metadata["level3"] = point.payload.get("level3", "")
                    doc.metadata["level4"] = point.payload.get("level4", "")
                    doc.metadata["header"] = point.payload.get("header", "")
        return docs
    
    rag_chain = (
        {"context": retriever | RunnableLambda(enrich_metadata) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def rag_query(query):
    """RAG查询"""
    rag_chain, retriever = build_rag_chain()
    
    print("=" * 80)
    print(f"用户问题: {query}")
    print("=" * 80)
    
    print("\n[1/2] 正在检索相关文档...")
    docs = retriever.invoke(query)
    print(f"找到 {len(docs)} 条相关文档")
    
    # 补充metadata
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    for doc in docs:
        doc_id = doc.metadata.get("_id", "")
        if doc_id:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[doc_id],
            )
            if points:
                point = points[0]
                doc.metadata["level1"] = point.payload.get("level1", "")
                doc.metadata["level2"] = point.payload.get("level2", "")
                doc.metadata["level3"] = point.payload.get("level3", "")
                doc.metadata["level4"] = point.payload.get("level4", "")
                doc.metadata["header"] = point.payload.get("header", "")
    
    print("\n检索结果详情:")
    for i, doc in enumerate(docs, 1):
        header = " > ".join([p for p in [
            doc.metadata.get('level1', ''),
            doc.metadata.get('level2', ''),
            doc.metadata.get('level3', ''),
            doc.metadata.get('level4', '')
        ] if p])
        print(f"\n【文档 {i}】")
        print(f"标题路径: {header}")
        print(f"内容:\n{doc.page_content[:300]}...")
        print("-" * 80)
    
    print("\n[2/2] 正在生成答案...")
    answer = rag_chain.invoke(query)
    
    print("\n" + "=" * 80)
    print("AI助手回答:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    return answer


def main():
    print("华中科技大学研究生手册 RAG 系统 (V1 - LangChain向量检索)")
    print("=" * 80)
    
    query = "华中科技大学博士生培养目标中，对学术能力和专业能力分别提出哪些具体要求？"
    print(f"\n用户问题: {query}\n")
    
    try:
        rag_query(query)
    except Exception as e:
        print(f"\n查询失败: {e}\n")


if __name__ == "__main__":
    main()
