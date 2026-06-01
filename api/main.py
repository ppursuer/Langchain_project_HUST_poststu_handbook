from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os

from hust_rag.chain import build_rag_chain, get_session_history, store
from hust_rag.retriever import hybrid_retrieve, enrich_docs_metadata
from hust_rag.reranker import rerank_documents
from config.settings import TOP_K
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="华中科技大学研究生手册智能助手")

# 挂载静态文件和模板目录
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    references: list


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    rag_chain, vector_store, bm25_retriever = build_rag_chain(request.session_id)

    docs = hybrid_retrieve(request.message, vector_store, bm25_retriever)
    docs = enrich_docs_metadata(docs)
    reranked_docs = rerank_documents(request.message, docs)[:TOP_K]

    history = get_session_history(request.session_id)
    history_messages = []
    for msg in history[-4:]:
        if msg["role"] == "human":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))

    answer = rag_chain.invoke({
        "question": request.message,
        "history": history_messages,
    })

    store[request.session_id].append({"role": "human", "content": request.message})
    store[request.session_id].append({"role": "ai", "content": answer})

    references = []
    for i, doc in enumerate(reranked_docs, 1):
        header = " > ".join([p for p in [
            doc.metadata.get('level1', ''),
            doc.metadata.get('level2', ''),
            doc.metadata.get('level3', ''),
            doc.metadata.get('level4', '')
        ] if p])
        references.append({
            "index": i,
            "header": header,
            "content": doc.page_content,
            "rerank_score": doc.metadata.get('rerank_score', 0),
        })

    return ChatResponse(answer=answer, references=references)


@app.post("/api/clear")
async def clear_session(session_id: Optional[str] = "default"):
    """清空会话"""
    store[session_id] = []
    return {"status": "ok"}
