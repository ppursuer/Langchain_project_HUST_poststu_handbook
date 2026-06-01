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
