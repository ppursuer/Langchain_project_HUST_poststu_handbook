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
