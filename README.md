# 华中科技大学研究生手册智能助手 - RAG系统

基于Retrieval-Augmented Generation (RAG) 技术的智能问答系统，用于回答华中科技大学研究生手册相关问题。

## 📁 项目结构

```
rag_project/
├── hust_rag/                    # 核心RAG系统包
│   ├── __init__.py             # 包初始化
│   ├── embeddings.py           # 自定义Embeddings实现
│   ├── retriever.py            # 检索逻辑（向量+BM25混合检索）
│   ├── reranker.py             # Rerank重排逻辑
│   ├── formatter.py            # 文档格式化
│   └── chain.py                # RAG链构建和查询
│
├── api/                        # FastAPI后端
│   └── main.py                 # API路由和处理
│
├── frontend/                   # 前端界面
│   └── index.html              # 主页面
│
├── scripts/                    # 工具脚本
│   ├── ingest.py               # 数据导入逻辑
│   └── run_ingest.py           # 导入运行脚本
│
├── config/                     # 配置文件
│   └── settings.py             # 所有配置集中管理
│
├── data/                       # 数据目录
│   └── full.md                 # 原始文档
│
├── legacy/                     # 旧版本代码（存档）
│
├── requirements.txt            # Python依赖
├── README.md                   # 项目文档
└── run_api.py                  # 启动入口
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 导入数据

```bash
python scripts/run_ingest.py --file data/full.md
```

### 3. 启动API服务

```bash
python run_api.py
```

访问 http://localhost:8080 使用Web界面

## 🛠️ 技术栈

- **向量数据库**: Qdrant
- **Embedding模型**: BGE-M3
- **Rerank模型**: BAAI/bge-reranker-v2-m3
- **LLM**: Qwen2.5-7B-Instruct
- **检索方式**: 混合检索（向量检索 + BM25）
- **后端框架**: FastAPI
- **前端**: HTML + JavaScript + Marked.js

## � Legacy代码存档

以下文件保留在 `legacy/` 目录中，作为历史版本参考：

| 文件 | 说明 |
|------|------|
| `v1_rag.py` | V1版本 - 纯向量检索RAG系统（原生实现，未使用LangChain） |
| `v1_rag_langchain.py` | V1版本 - 纯向量检索RAG系统（基于LangChain框架） |
| `v2_rag.py` | V2版本 - 混合检索+Rerank RAG系统（原生实现，未使用LangChain） |
| `v2_rag_langchain.py` | V2版本 - 混合检索+Rerank RAG系统（基于LangChain框架，当前重构版本的来源） |
| `api.py` | 旧版FastAPI后端入口（已迁移至 `api/main.py`） |
| `ingest_qdrant.py` | 旧版数据导入脚本（已迁移至 `scripts/ingest.py`） |
| `evaluate_rag.py` | RAG系统评估脚本 |
| `convert_to_faq.py` | FAQ数据转换脚本 |
| `process_md.py` | Markdown文档预处理脚本 |

## �📖 功能特性

- ✅ 混合检索（向量 + BM25）
- ✅ Rerank重排
- ✅ 多轮对话（上下文记忆）
- ✅ Markdown格式化输出
- ✅ 参考资料来源显示
- ✅ 参考资料折叠展开
- ✅ Web界面交互

## ⚙️ 配置说明

所有配置项在 `config/settings.py` 中管理：

- Qdrant向量库地址
- BGE-M3 Embedding API地址
- Rerank API地址
- LLM API地址
- 检索参数（TOP_K等）

## 📝 使用示例

### 命令行使用

```python
from hust_rag.chain import rag_query

answer = rag_query("博士生培养目标中，对学术能力和专业能力分别提出哪些具体要求？")
print(answer)
```

### API调用

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "博士生学习年限有什么规定？", "session_id": "my_session"}'
```
