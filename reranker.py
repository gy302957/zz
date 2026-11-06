import torch
from sentence_transformers import CrossEncoder

import config_data as config
from langchain_core.documents import Document


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            config.reranker_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """按与 query 的相关性对 docs 重排，截取前 top_n 条"""
        if not docs:
            return []
        scores = self.model.predict([(query, doc.page_content) for doc in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """全局单例，首次查询时才加载模型"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
