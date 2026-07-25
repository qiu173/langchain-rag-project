"""
检索模块：从 Chroma 向量数据库中检索相关文档 Chunk，可选支持 Reranker 二次重排序。

支持作为独立模块调用，方便在评估脚本 (如 eval/evaluate.py) 中进行是否启用 Rerank 的 A/B 对比实验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

# 以当前文件路径反推项目根目录，避免因执行目录不同导致数据路径解析错误
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")


@dataclass
class RetrieveResult:
    documents: list[Document]
    scores: list[float]


class Retriever:
    def __init__(
        self,
        collection_name: str,
        embedding_model: Any,
        use_rerank: bool = False,
    ):
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        self.use_rerank = use_rerank
        self._reranker = None
        if use_rerank:
            self._reranker = self._load_reranker()

    def _load_reranker(self):
        """懒加载 Reranker 模型，仅在启用 Rerank 时加载以优化内存/显存占用"""
        from sentence_transformers import CrossEncoder

        return CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int | None = None,
    ) -> RetrieveResult:
        """
        根据查询语句检索相关文档

        Args:
            query: 用户查询字符串
            top_k: 最终返回的文档数量
            initial_k: 启用 Rerank 时初检索召回的数量；若未指定则默认为 top_k 的 4 倍
        """
        if self.use_rerank:
            initial_k = initial_k or top_k * 4
            candidates = self.vectorstore.similarity_search_with_score(
                query, k=initial_k
            )
            docs = [d for d, _ in candidates]

            pairs = [[query, d.page_content] for d in docs]
            rerank_scores = self._reranker.predict(pairs)

            ranked = sorted(
                zip(docs, rerank_scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            return RetrieveResult(
                documents=[d for d, _ in ranked],
                scores=[float(s) for _, s in ranked],
            )
        else:
            candidates = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
            return RetrieveResult(
                documents=[d for d, _ in candidates],
                scores=[float(s) for _, s in candidates],
            )