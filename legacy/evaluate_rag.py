import json
import requests
import math
import time
from qdrant_client import QdrantClient
from collections import Counter
import jieba
import jieba.analyse

# ==================== 配置 ====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6335
COLLECTION_NAME = "HUST_poststu_handbook"

BGE_M3_API_URL = "http://10.154.22.10:34520/v1/embeddings"
BGE_M3_MODEL = "BAAI/bge-m3"

RERANK_API_URL = "http://10.154.22.10:34523/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_API_URL = "http://10.154.22.10:34525/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

BM25_TOP_K = 20
VECTOR_TOP_K = 20

FAQ_FILE = r'c:\Users\x1359\Desktop\rag_project\datas\lora_hust_student_handbookt_faq.json'
EVAL_OUTPUT_FILE = r'c:\Users\x1359\Desktop\rag_project\eval_results.json'
# ==============================================


def get_embedding(text):
    payload = {"model": BGE_M3_MODEL, "input": text}
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
    query_vector = get_embedding(query)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    return results.points


class BM25Searcher:
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
        import re
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        words = jieba.lcut(text)
        filtered_words = [
            w.lower() for w in words 
            if w.lower() not in self.stopwords 
            and len(w.strip()) > 1
            and not w.strip().isdigit()
        ]
        return filtered_words
    
    def extract_keywords(self, query, top_n=10):
        import re
        query_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', query)
        tfidf_keywords = jieba.analyse.extract_tags(query_clean, topK=top_n, withWeight=True)
        textrank_keywords = jieba.analyse.textrank(query_clean, topK=top_n, withWeight=True)
        keyword_scores = {}
        for kw, weight in tfidf_keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.stopwords and len(kw_lower) > 1:
                keyword_scores[kw_lower] = weight * 0.6
        for kw, weight in textrank_keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.stopwords and len(kw_lower) > 1:
                if kw_lower in keyword_scores:
                    keyword_scores[kw_lower] += weight * 0.4
                else:
                    keyword_scores[kw_lower] = weight * 0.4
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords
    
    def add_document(self, doc_id, content):
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
        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query, top_k=10):
        keywords_with_weights = self.extract_keywords(query)
        if not keywords_with_weights:
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
                    score += idf * numerator / denominator * weight
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def rerank_documents(query, documents):
    if not documents:
        return []
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
    print("正在从Qdrant加载文档...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
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
    searcher = BM25Searcher()
    for doc in all_docs:
        content = doc.payload.get('content', '')
        doc_id = doc.id
        searcher.add_document(doc_id, content)
    print("BM25索引构建完成")
    return searcher, all_docs


def v1_search(query, top_k=1):
    results = search_in_qdrant(query, top_k=top_k)
    return results


def v2_search(query, bm25_searcher, all_docs, top_k=1):
    vector_results = search_in_qdrant(query, top_k=VECTOR_TOP_K)
    bm25_results = bm25_searcher.search(query, top_k=BM25_TOP_K)
    
    doc_map = {}
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
    
    documents_list = list(doc_map.values())
    rerank_results = rerank_documents(query, documents_list)
    
    top_docs = []
    if isinstance(rerank_results, list):
        for item in rerank_results[:top_k]:
            if isinstance(item, dict) and 'index' in item:
                idx = item['index']
                score = item.get('relevance_score', item.get('score', 0))
                if idx < len(documents_list):
                    doc = documents_list[idx].copy()
                    doc['rerank_score'] = score
                    top_docs.append(doc)
    
    return top_docs


def check_if_can_answer(query, doc_content):
    prompt = f"""请判断以下文档内容是否能回答用户的问题。只需回答"是"或"否"。

用户问题：{query}

文档内容：
{doc_content}

是否能回答该问题？（是/否）"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content'].strip()
            return "是" in answer
        else:
            return False
    except:
        return False


def get_header(doc):
    if isinstance(doc, dict):
        return " > ".join([p for p in [doc.get('level1', ''), doc.get('level2', ''), doc.get('level3', ''), doc.get('level4', '')] if p])
    else:
        return " > ".join([p for p in [doc.payload.get('level1', ''), doc.payload.get('level2', ''), doc.payload.get('level3', ''), doc.payload.get('level4', '')] if p])


def evaluate_top1(faq_data, bm25_searcher, all_docs):
    print("\n" + "=" * 60)
    print("Top-1 评估")
    print("=" * 60)
    
    v1_can_answer = 0
    v2_can_answer = 0
    v1_better = 0
    v2_better = 0
    same = 0
    detailed = []
    
    total = len(faq_data)
    
    for i, item in enumerate(faq_data, 1):
        query = item['query']
        
        try:
            v1_results = v1_search(query, top_k=1)
            v1_top1 = v1_results[0] if v1_results else None
            v1_header = get_header(v1_top1) if v1_top1 else ""
            v1_content = v1_top1.payload.get('content', '') if v1_top1 else ""
            v1_can = check_if_can_answer(query, v1_content) if v1_top1 else False
            
            v2_results = v2_search(query, bm25_searcher, all_docs, top_k=1)
            v2_top1 = v2_results[0] if v2_results else None
            v2_header = get_header(v2_top1) if v2_top1 else ""
            v2_content = v2_top1.get('content', '') if v2_top1 else ""
            v2_can = check_if_can_answer(query, v2_content) if v2_top1 else False
            
            if v1_can: v1_can_answer += 1
            if v2_can: v2_can_answer += 1
            
            if v1_header == v2_header:
                same += 1
            elif v1_can and not v2_can:
                v1_better += 1
            elif v2_can and not v1_can:
                v2_better += 1
            
            detailed.append({
                "query": query,
                "v1_header": v1_header,
                "v1_can_answer": v1_can,
                "v2_header": v2_header,
                "v2_can_answer": v2_can
            })
            
            if i % 50 == 0:
                print(f"[{i}/{total}] V1={v1_can_answer}, V2={v2_can_answer}")
            
            time.sleep(0.5)
        except Exception as e:
            print(f"[{i}/{total}] 错误: {e}")
    
    print(f"\nV1能回答: {v1_can_answer}/{total} = {v1_can_answer/total*100:.2f}%")
    print(f"V2能回答: {v2_can_answer}/{total} = {v2_can_answer/total*100:.2f}%")
    print(f"相同: {same}, V1更好: {v1_better}, V2更好: {v2_better}")
    
    return {
        "v1_accuracy": v1_can_answer / total,
        "v2_accuracy": v2_can_answer / total,
        "v1_can_answer": v1_can_answer,
        "v2_can_answer": v2_can_answer,
        "same": same,
        "v1_better": v1_better,
        "v2_better": v2_better,
        "detailed": detailed
    }


def evaluate_top5(faq_data, bm25_searcher, all_docs):
    print("\n" + "=" * 60)
    print("Top-5 评估")
    print("=" * 60)
    
    v1_recall = 0
    v2_recall = 0
    v1_rank_sum = 0
    v2_rank_sum = 0
    detailed = []
    
    total = len(faq_data)
    
    for i, item in enumerate(faq_data, 1):
        query = item['query']
        
        try:
            v1_results = v1_search(query, top_k=5)
            v1_first_rank = None
            v1_can_list = []
            for rank, doc in enumerate(v1_results, 1):
                can = check_if_can_answer(query, doc.payload.get('content', ''))
                v1_can_list.append(can)
                if can and v1_first_rank is None:
                    v1_first_rank = rank
            
            if v1_first_rank:
                v1_recall += 1
                v1_rank_sum += v1_first_rank
            
            v2_results = v2_search(query, bm25_searcher, all_docs, top_k=5)
            v2_first_rank = None
            v2_can_list = []
            for rank, doc in enumerate(v2_results, 1):
                can = check_if_can_answer(query, doc.get('content', ''))
                v2_can_list.append(can)
                if can and v2_first_rank is None:
                    v2_first_rank = rank
            
            if v2_first_rank:
                v2_recall += 1
                v2_rank_sum += v2_first_rank
            
            detailed.append({
                "query": query,
                "v1_first_rank": v1_first_rank,
                "v2_first_rank": v2_first_rank
            })
            
            if i % 50 == 0:
                v1_avg = v1_rank_sum / v1_recall if v1_recall > 0 else 0
                v2_avg = v2_rank_sum / v2_recall if v2_recall > 0 else 0
                print(f"[{i}/{total}] V1召回率={v1_recall/i*100:.1f}%, 平均排名={v1_avg:.2f} | V2召回率={v2_recall/i*100:.1f}%, 平均排名={v2_avg:.2f}")
            
            time.sleep(0.5)
        except Exception as e:
            print(f"[{i}/{total}] 错误: {e}")
    
    v1_avg_rank = v1_rank_sum / v1_recall if v1_recall > 0 else 0
    v2_avg_rank = v2_rank_sum / v2_recall if v2_recall > 0 else 0
    
    print(f"\nV1召回率: {v1_recall}/{total} = {v1_recall/total*100:.2f}%, 平均排名={v1_avg_rank:.2f}")
    print(f"V2召回率: {v2_recall}/{total} = {v2_recall/total*100:.2f}%, 平均排名={v2_avg_rank:.2f}")
    
    return {
        "v1_recall": v1_recall / total,
        "v2_recall": v2_recall / total,
        "v1_avg_rank": v1_avg_rank,
        "v2_avg_rank": v2_avg_rank,
        "v1_recall_count": v1_recall,
        "v2_recall_count": v2_recall,
        "detailed": detailed
    }


def main():
    with open(FAQ_FILE, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    print(f"加载了 {len(faq_data)} 条FAQ数据")
    print("\n初始化BM25索引...")
    bm25_searcher, all_docs = build_bm25_index()
    
    top1_results = evaluate_top1(faq_data, bm25_searcher, all_docs)
    top5_results = evaluate_top5(faq_data, bm25_searcher, all_docs)
    
    output = {
        "total_count": len(faq_data),
        "top1": top1_results,
        "top5": top5_results
    }
    
    with open(EVAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {EVAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
