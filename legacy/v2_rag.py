import requests
import json
import math
from qdrant_client import QdrantClient
from collections import Counter
import jieba
import jieba.analyse

# ==================== 配置 ====================
# Qdrant配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

# 向量模型配置（BGE-M3）
BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

# Rerank模型配置（BGE-reranker-v2-m3）
RERANK_API_URL = "http://10.154.22.10:34523/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# LLM配置（Qwen2.5-7B-Instruct）
LLM_API_URL = "http://10.154.22.10:34525/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# RAG参数
TOP_K = 5  # 最终返回的相关文档数量
BM25_TOP_K = 10  # BM25检索数量
VECTOR_TOP_K = 10  # 向量检索数量
MAX_TOKENS = 1024  # 最大生成token数
TEMPERATURE = 0.3  # 温度参数
# ==============================================


class BM25Searcher:
    """BM25关键词搜索引擎"""
    
    def __init__(self):
        self.documents = []
        self.doc_freq = Counter()
        self.term_freq = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0
        self.k1 = 1.5
        self.b = 0.75
        self.idf = {}
        
        # 停用词表
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '啊', '呢', '吧', '吗', '什么', '怎么', '为什么', '哪',
            '哪些', '如何', '怎样', '是否', '能否', '可以', '需要', '应该', '必须',
            '中', '对', '分别', '提出', '哪些', '具体', '要求', '根据', '规定', '进行',
            '以及', '通过', '关于', '对于', '从', '向', '与', '或', '等', '其', '该',
            '此', '这些', '那些', '另', '另外', '其他', '各', '每', '本', '本学科',
            '掌握', '具备', '具有', '能够', '能', '可', '应', '需', '须', '要'
        ])
    
    def tokenize(self, text):
        """使用jieba分词，过滤停用词和短词"""
        import re
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        
        # jieba精确模式分词
        words = jieba.lcut(text)
        
        # 过滤停用词、单字和数字
        filtered_words = [
            w.lower() for w in words 
            if w.lower() not in self.stopwords 
            and len(w.strip()) > 1
            and not w.strip().isdigit()
        ]
        
        return filtered_words
    
    def extract_keywords(self, query, top_n=10):
        """提取query的核心关键词，带权重排序"""
        import re
        # 清理文本
        query_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', query)
        
        # 方法1: TF-IDF提取关键词
        tfidf_keywords = jieba.analyse.extract_tags(query_clean, topK=top_n, withWeight=True)
        
        # 方法2: TextRank提取关键词
        textrank_keywords = jieba.analyse.textrank(query_clean, topK=top_n, withWeight=True)
        
        # 合并两种方法的结果，取平均权重
        keyword_scores = {}
        for kw, weight in tfidf_keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.stopwords and len(kw_lower) > 1:
                keyword_scores[kw_lower] = weight * 0.6  # TF-IDF权重60%
        
        for kw, weight in textrank_keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.stopwords and len(kw_lower) > 1:
                if kw_lower in keyword_scores:
                    keyword_scores[kw_lower] += weight * 0.4  # TextRank权重40%
                else:
                    keyword_scores[kw_lower] = weight * 0.4
        
        # 按权重排序
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords
    
    def extract_key_phrases(self, query):
        """提取核心短语（适合长文本、复杂问句）"""
        import re
        
        # 清理文本
        query_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9？?]', ' ', query)
        
        # 使用jieba分词
        words = jieba.lcut(query_clean)
        
        # 过滤停用词
        filtered_words = [
            w for w in words 
            if w not in self.stopwords 
            and len(w.strip()) > 1
            and not w.strip().isdigit()
        ]
        
        # 提取名词短语（连续的名词组合）
        pos_tags = jieba.posseg.cut(query_clean)
        noun_phrases = []
        current_phrase = []
        
        for word, flag in pos_tags:
            if flag.startswith('n') and word not in self.stopwords:  # 名词
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:
                    phrase = ''.join(current_phrase)
                    if len(phrase) > 1:
                        noun_phrases.append(phrase)
                current_phrase = []
        
        if current_phrase:
            phrase = ''.join(current_phrase)
            if len(phrase) > 1:
                noun_phrases.append(phrase)
        
        return filtered_words, noun_phrases
    
    def add_document(self, doc_id, content):
        """添加文档到索引"""
        tokens = self.tokenize(content)
        self.documents.append({
            'doc_id': doc_id,
            'content': content,
            'tokens': tokens
        })
        
        tf = Counter(tokens)
        self.term_freq.append(tf)
        self.doc_lengths.append(len(tokens))
        
        for term in set(tokens):
            self.doc_freq[term] += 1
        
        self.N = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        self._calculate_idf()
    
    def _calculate_idf(self):
        """计算IDF值"""
        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query, top_k=10):
        """BM25搜索（使用加权关键词）"""
        # 提取带权重的关键词
        keywords_with_weights = self.extract_keywords(query)
        
        if not keywords_with_weights:
            # 如果提取不到关键词，使用普通分词
            query_tokens = self.tokenize(query)
            keyword_dict = {token: 1.0 for token in query_tokens}
        else:
            keyword_dict = {kw: weight for kw, weight in keywords_with_weights}
        
        scores = []
        for i, doc in enumerate(self.documents):
            score = 0
            doc_len = self.doc_lengths[i]
            tf = self.term_freq[i]
            
            for token, weight in keyword_dict.items():
                if token in tf:
                    term_freq = tf[token]
                    idf = self.idf.get(token, 0)
                    
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                    score += idf * numerator / denominator * weight  # 乘以关键词权重
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


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


def search_in_qdrant(query, top_k=VECTOR_TOP_K):
    """在Qdrant中搜索相关文档"""
    query_vector = get_embedding(query)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    return results.points


def rerank_documents(query, documents):
    """使用rerank模型对文档进行重排"""
    if not documents:
        return []
    
    # 构建文本列表
    texts = []
    for doc in documents:
        content = doc.get('content', '')
        level1 = doc.get('level1', '')
        level2 = doc.get('level2', '')
        level3 = doc.get('level3', '')
        level4 = doc.get('level4', '')
        
        header_parts = [p for p in [level1, level2, level3, level4] if p]
        header = " > ".join(header_parts)
        
        text = f"{header}\n{content}"
        texts.append(text)
    
    # 调用rerank API
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": texts
    }
    
    response = requests.post(RERANK_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            return data["results"]
        else:
            raise ValueError(f"Unexpected API response: {data}")
    else:
        raise Exception(f"Rerank API request failed: {response.status_code} {response.text}")


def build_bm25_index():
    """构建BM25索引"""
    print("正在从Qdrant加载文档...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # 获取所有文档
    all_docs = []
    offset = None
    while True:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True
        )
        points, next_offset = response
        
        for point in points:
            all_docs.append(point)
        
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"加载了 {len(all_docs)} 个文档")
    
    # 构建BM25索引
    searcher = BM25Searcher()
    for doc in all_docs:
        content = doc.payload.get('content', '')
        doc_id = doc.id
        searcher.add_document(doc_id, content)
    
    print("BM25索引构建完成")
    return searcher, all_docs


def hybrid_search(query, bm25_searcher, all_docs, top_k=TOP_K):
    """混合搜索：向量检索 + BM25 + Rerank"""
    # 1. 向量检索
    print("  [1/4] 向量检索...")
    vector_results = search_in_qdrant(query, top_k=VECTOR_TOP_K)
    print(f"    找到 {len(vector_results)} 条向量检索结果")
    
    # 打印query的关键词（带权重）
    keywords_with_weights = bm25_searcher.extract_keywords(query)
    print(f"    核心关键词（带权重）:")
    for kw, weight in keywords_with_weights[:8]:
        print(f"      {kw}: {weight:.4f}")
    
    # 打印核心短语
    filtered_words, noun_phrases = bm25_searcher.extract_key_phrases(query)
    print(f"    分词结果: {', '.join(filtered_words)}")
    if noun_phrases:
        print(f"    核心短语: {', '.join(noun_phrases)}")
    
    # 2. BM25检索
    print("  [2/4] BM25关键词检索...")
    bm25_results = bm25_searcher.search(query, top_k=BM25_TOP_K)
    print(f"    找到 {len(bm25_results)} 条BM25检索结果")
    
    # 3. 合并结果
    print("  [3/4] 合并检索结果...")
    doc_map = {}
    
    # 添加向量检索结果
    for result in vector_results:
        doc_id = result.id
        doc_map[doc_id] = {
            'doc_id': doc_id,
            'content': result.payload.get('content', ''),
            'level1': result.payload.get('level1', ''),
            'level2': result.payload.get('level2', ''),
            'level3': result.payload.get('level3', ''),
            'level4': result.payload.get('level4', ''),
            'vector_score': result.score,
            'bm25_score': 0
        }
    
    # 添加BM25检索结果
    for idx, bm25_score in bm25_results:
        doc = all_docs[idx]
        doc_id = doc.id
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                'doc_id': doc_id,
                'content': doc.payload.get('content', ''),
                'level1': doc.payload.get('level1', ''),
                'level2': doc.payload.get('level2', ''),
                'level3': doc.payload.get('level3', ''),
                'level4': doc.payload.get('level4', ''),
                'vector_score': 0,
                'bm25_score': bm25_score
            }
        else:
            doc_map[doc_id]['bm25_score'] = bm25_score
    
    # 4. Rerank
    print("  [4/4] Rerank重排...")
    documents_list = list(doc_map.values())
    rerank_results = rerank_documents(query, documents_list)
    
    # 按rerank分数排序
    if isinstance(rerank_results, list):
        # 如果返回的是列表，按index排序
        reranked_docs = []
        for item in rerank_results:
            if isinstance(item, dict) and 'index' in item:
                idx = item['index']
                score = item.get('relevance_score', item.get('score', 0))
                if idx < len(documents_list):
                    doc = documents_list[idx].copy()
                    doc['rerank_score'] = score
                    reranked_docs.append(doc)
        
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs[:top_k]
    else:
        raise ValueError(f"Unexpected rerank results format: {type(rerank_results)}")


def build_context(results):
    """构建上下文文本"""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        level1 = result.get('level1', '')
        level2 = result.get('level2', '')
        level3 = result.get('level3', '')
        level4 = result.get('level4', '')
        content = result.get('content', '')
        
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


def rag_query(query, bm25_searcher, all_docs):
    """RAG查询主函数"""
    print("=" * 80)
    print(f"用户问题: {query}")
    print("=" * 80)
    
    # 步骤1: 混合检索
    print("\n[1/3] 正在混合检索相关文档（向量+BM25+Rerank）...")
    results = hybrid_search(query, bm25_searcher, all_docs)
    print(f"找到 {len(results)} 条相关文档")
    
    # 打印检索结果详情
    print("\n检索结果详情:")
    for i, result in enumerate(results, 1):
        print(f"\n【文档 {i}】")
        print(f"Rerank分数: {result.get('rerank_score', 0):.4f}")
        print(f"向量分数: {result.get('vector_score', 0):.4f}")
        print(f"BM25分数: {result.get('bm25_score', 0):.4f}")
        print(f"一级标题: {result.get('level1', 'N/A')}")
        print(f"二级标题: {result.get('level2', 'N/A')}")
        print(f"三级标题: {result.get('level3', 'N/A')}")
        print(f"四级标题: {result.get('level4', 'N/A')}")
        print(f"内容:\n{result.get('content', 'N/A')[:300]}...")
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
    print("华中科技大学研究生手册 RAG 系统 (V2 - 混合检索: 向量+BM25+Rerank)")
    print("=" * 80)
    
    # 初始化BM25索引
    print("\n初始化BM25索引...")
    bm25_searcher, all_docs = build_bm25_index()
    
    query = "华中科技大学博士生培养目标中，对学术能力和专业能力分别提出哪些具体要求？"
    print(f"\n用户问题: {query}\n")
    
    try:
        rag_query(query, bm25_searcher, all_docs)
    except Exception as e:
        print(f"\n查询失败: {e}\n")


if __name__ == "__main__":
    main()
