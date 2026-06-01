import requests
from langchain_core.embeddings import Embeddings
from config.settings import BGE_M3_API_URL, BGE_M3_MODEL


class CustomEmbeddings(Embeddings):
    """自定义Embeddings类，调用BGE-M3 API"""
    
    def embed_documents(self, texts):
        payload = {"model": BGE_M3_MODEL, "input": texts}
        response = requests.post(BGE_M3_API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]
