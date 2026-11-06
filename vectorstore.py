import config_data as config

from langchain_chroma import Chroma
from langchain_core.documents import Document
import jieba
from typing import List
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict
from reranker import get_reranker

def chinese_tokenizer(text: str) -> List[str]:
    return list(jieba.cut_for_search(text))

def docs_from_get_result(get_result):
    """提取 Chroma.get() 返回字典中的文档文本列表"""
    return (get_result or {}).get("documents") or []



class Vectorstore:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_hybrid_retriever(self):
        """得到混合检索的检索器用于后续加入链中"""
        corpus = self.vectorstore.get()
        keyword_retriever = BM25Retriever.from_texts(
        texts=docs_from_get_result(corpus),
        metadatas=corpus.get("metadatas") or None,
        preprocess_func=chinese_tokenizer,k=config.rerank_candidates)

        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.rerank_candidates}
        )
         # 2. 定义 RRF 融合逻辑
        def hybrid_search(query: str) -> List[Document]:
            # 分别获取两路检索结果
            vector_docs = vector_retriever.invoke(query)
            keyword_docs = keyword_retriever.invoke(query)
            
            # RRF 算法核心：计算每个文档的 RRF 分数
            # 公式: RRF(d) = sum(1 / (k + rank_i(d)))，通常 k 取 60
            rrf_scores = defaultdict(float)
            doc_map = {}  # 用于存储文档对象，避免重复存储
            
            for rank, doc in enumerate(vector_docs):
                cid=doc.metadata["chunk_id"]
                rrf_scores[cid] += 1.0 / (60 + rank + 1)
                doc_map[cid] = doc
                
            for rank, doc in enumerate(keyword_docs):
                cid=doc.metadata["chunk_id"]
                rrf_scores[cid] += 1.0 / (60 + rank + 1)
                doc_map[cid] = doc
                
            # 3. 按照 RRF 分数降序排序，取候选集交给精排
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            candidates = [doc_map[cid] for cid in sorted_ids[:config.rerank_candidates]]

            # 4. 精排后截取 Top-K
            return get_reranker().rerank(query, candidates, config.top_k)
    
        return hybrid_search



       
            


