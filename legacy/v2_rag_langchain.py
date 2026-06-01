import requests
import jieba
import jieba.analyse
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage

# ==================== 配置 ====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

RERANK_API_URL = "http://10.154.22.10:34523/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_API_URL = "http://10.154.22.10:34525/v1"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 5
VECTOR_TOP_K = 10
BM25_TOP_K = 10
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


def enrich_docs_metadata(docs):
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


def get_bm25_retriever(vector_store):
    """构建BM25检索器"""
    docs = vector_store.similarity_search("", k=1000)
    
    for doc in docs:
        doc.page_content = ' '.join(jieba.lcut(doc.page_content))
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = BM25_TOP_K
    return bm25_retriever


def hybrid_retrieve(query, vector_store, bm25_retriever):
    """混合检索：向量检索 + BM25，去重合并"""
    vector_docs = vector_store.similarity_search(query, k=VECTOR_TOP_K)
    for doc in vector_docs:
        doc.metadata["vector_score"] = 0
    
    bm25_docs = bm25_retriever.invoke(query)
    for doc in bm25_docs:
        doc.metadata["bm25_score"] = 0
    
    doc_map = {}
    
    for doc in vector_docs:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        doc.metadata["vector_score"] = 1
        doc_map[doc_id] = doc
    
    for doc in bm25_docs:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        doc.metadata["bm25_score"] = 1
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
    
    return list(doc_map.values())[:TOP_K * 2]


def rerank_documents(query, docs):
    """使用Rerank模型重排文档"""
    if not docs:
        return []
    
    texts = [doc.page_content for doc in docs]
    
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": texts
    }
    
    response = requests.post(RERANK_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            results = data["results"]
            results.sort(key=lambda x: x.get("relevance_score", x.get("score", 0)), reverse=True)
            
            reranked_docs = []
            for item in results:
                idx = item.get("index", 0)
                if idx < len(docs):
                    doc = docs[idx]
                    doc.metadata["rerank_score"] = item.get("relevance_score", item.get("score", 0))
                    reranked_docs.append(doc)
            return reranked_docs
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


# 会话存储
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = []
    return store[session_id]


def build_rag_chain(session_id: str = "default"):
    """构建RAG链"""
    vector_store = get_vector_store()
    bm25_retriever = get_bm25_retriever(vector_store)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是华中科技大学研究生手册的智能助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

回答要求：
1. 直接回答用户的问题，不要提及"根据参考资料"、"根据资料"、"参考资料X"等类似表述。
2. 严格基于参考资料的内容进行回答，准确理解原文的语义和上下文，不要曲解或错误归类信息。
3. 如果参考资料中没有相关信息，请说明"根据现有资料无法回答该问题"。
4. 结合对话历史，理解用户的后续问题或追问。

请在回答的最后，另起一行，添加"参考资料来源："，并列出所有参考资料的标题路径（格式：level1 > level2 > level3 > level4）。"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key="not-needed",
        openai_api_base=LLM_API_URL,
        temperature=0.3,
        max_tokens=1024,
    )
    
    def retrieve_and_rerank(input_data):
        """检索并重排文档"""
        query = input_data.get("question", "")
        docs = hybrid_retrieve(query, vector_store, bm25_retriever)
        print(f"  混合检索找到 {len(docs)} 条文档")
        
        docs = enrich_docs_metadata(docs)
        
        reranked_docs = rerank_documents(query, docs)
        print(f"  Rerank后保留 {len(reranked_docs[:TOP_K])} 条文档")
        
        result = {"context": format_docs(reranked_docs[:TOP_K]), "question": query}
        if "history" in input_data:
            result["history"] = input_data["history"]
        return result
    
    rag_chain = (
        RunnableLambda(retrieve_and_rerank)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, vector_store, bm25_retriever


def rag_query(query, session_id: str = "default"):
    """RAG查询"""
    rag_chain, vector_store, bm25_retriever = build_rag_chain(session_id)
    
    print("=" * 80)
    print(f"用户问题: {query}")
    print("=" * 80)
    
    print("\n[1/3] 正在混合检索相关文档...")
    docs = hybrid_retrieve(query, vector_store, bm25_retriever)
    print(f"找到 {len(docs)} 条相关文档")
    
    print("\n[2/3] 正在Rerank重排...")
    docs = enrich_docs_metadata(docs)
    reranked_docs = rerank_documents(query, docs)[:TOP_K]
    
    print("\n检索结果详情:")
    for i, doc in enumerate(reranked_docs, 1):
        header = " > ".join([p for p in [
            doc.metadata.get('level1', ''),
            doc.metadata.get('level2', ''),
            doc.metadata.get('level3', ''),
            doc.metadata.get('level4', '')
        ] if p])
        print(f"\n【文档 {i}】")
        print(f"Rerank分数: {doc.metadata.get('rerank_score', 'N/A')}")
        print(f"标题路径: {header}")
        print(f"内容:\n{doc.page_content[:300]}...")
        print("-" * 80)
    
    print("\n[3/3] 正在生成答案...")
    
    history = get_session_history(session_id)
    history_messages = []
    for msg in history[-4:]:
        if msg["role"] == "human":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))
    
    answer = rag_chain.invoke({
        "question": query,
        "history": history_messages,
    })
    
    store[session_id].append({"role": "human", "content": query})
    store[session_id].append({"role": "ai", "content": answer})
    
    print("\n" + "=" * 80)
    print("AI助手回答:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    return answer


def main():
    print("华中科技大学研究生手册 RAG 系统 (V2 - LangChain混合检索)")
    print("=" * 80)
    
    session_id = "session_001"
    
    queries = [
        "华中科技大学博士生培养目标中，对学术能力和专业能力分别提出哪些具体要求？", 
        "那硕士生的培养目标呢？",
        "他们的学习年限有什么规定？",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"对话轮次")
        print(f"{'='*80}\n")
        
        try:
            rag_query(query, session_id)
        except Exception as e:
            print(f"\n查询失败: {e}\n")


if __name__ == "__main__":
    main()
