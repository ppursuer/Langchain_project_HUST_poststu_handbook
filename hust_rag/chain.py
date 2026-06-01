from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from config.settings import LLM_API_URL, LLM_MODEL, TOP_K
from .retriever import get_vector_store, get_bm25_retriever, hybrid_retrieve, enrich_docs_metadata
from .reranker import rerank_documents
from .formatter import format_docs

# 会话存储
store = {}


def get_session_history(session_id: str):
    """获取会话历史"""
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
