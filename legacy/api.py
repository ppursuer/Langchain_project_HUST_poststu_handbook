from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json
import os

from v2_rag_langchain import (
    build_rag_chain, hybrid_retrieve, enrich_docs_metadata,
    rerank_documents, get_session_history, store, TOP_K
)
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="华中科技大学研究生手册智能助手")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    references: list


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
    store[session_id] = []
    return {"status": "ok"}
