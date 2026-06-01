from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
import re

from hust_rag.chain import build_rag_chain, get_session_history, store
from hust_rag.retriever import hybrid_retrieve, enrich_docs_metadata
from hust_rag.reranker import rerank_documents
from config.settings import TOP_K, LLM_API_URL, LLM_MODEL
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

app = FastAPI(title="华中科技大学研究生手册智能助手")

# 挂载静态文件和模板目录
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")


def rewrite_query(current_query: str, history_messages: list) -> str:
    """使用LLM重写查询，将简短追问扩展为完整查询"""
    rewrite_llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key="not-needed",
        openai_api_base=LLM_API_URL,
        temperature=0.1,
        max_tokens=128,
    )
    
    history_context = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else '助手'}: {msg.content}"
        for msg in history_messages[-4:]
    ])
    
    prompt = f"""你是一个查询重写助手。根据对话历史，将用户的简短追问重写为完整的、可独立理解的查询语句。

对话历史：
{history_context}

当前追问：{current_query}

请直接输出重写后的完整查询，不要添加任何其他内容。如果当前查询已经很完整，请直接返回原查询。"""
    
    try:
        response = rewrite_llm.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten if rewritten else current_query
    except Exception:
        return current_query


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    references: list
    used_references: list = []


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

    # 获取对话历史
    history = get_session_history(request.session_id)
    history_messages = []
    for msg in history[-4:]:
        if msg["role"] == "human":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))

    # Query重写：将简短追问扩展为完整查询
    rewritten_query = request.message
    if history_messages:
        rewritten_query = rewrite_query(request.message, history_messages)
        print(f"原始问题: {request.message}")
        print(f"重写后: {rewritten_query}")

    # 使用重写后的query进行检索
    docs = hybrid_retrieve(rewritten_query, vector_store, bm25_retriever)
    docs = enrich_docs_metadata(docs)
    reranked_docs = rerank_documents(rewritten_query, docs)[:TOP_K]

    answer = rag_chain.invoke({
        "question": request.message,
        "history": history_messages,
    })

    store[request.session_id].append({"role": "human", "content": request.message})
    store[request.session_id].append({"role": "ai", "content": answer})

    # 构建所有检索到的参考资料（用于"参考资料"板块显示）
    all_references = []
    for i, doc in enumerate(reranked_docs, 1):
        header = " > ".join([p for p in [
            doc.metadata.get('level1', ''),
            doc.metadata.get('level2', ''),
            doc.metadata.get('level3', ''),
            doc.metadata.get('level4', '')
        ] if p])
        all_references.append({
            "index": i,
            "header": header,
            "content": doc.page_content,
            "rerank_score": doc.metadata.get('rerank_score', 0),
        })

    # 判断LLM回答中实际使用了哪些参考资料
    # 方法：检查回答中是否包含参考资料的关键内容
    used_references = []
    for ref in all_references:
        # 提取参考资料中的关键短语（前100字）
        content_snippet = ref["content"][:100].strip()
        if content_snippet and content_snippet in answer:
            used_references.append(ref)

    return ChatResponse(answer=answer, references=all_references, used_references=used_references)


@app.post("/api/clear")
async def clear_session(session_id: Optional[str] = "default"):
    """清空会话"""
    store[session_id] = []
    return {"status": "ok"}
