import requests
from config.settings import RERANK_API_URL, RERANK_MODEL


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
