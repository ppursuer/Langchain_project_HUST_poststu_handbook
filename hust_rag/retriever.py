import jieba
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever
from config.settings import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    VECTOR_TOP_K, BM25_TOP_K, TOP_K
)
from .embeddings import CustomEmbeddings


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
