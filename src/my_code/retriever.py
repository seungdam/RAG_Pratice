from __future__ import annotations

# rag.py에서 HybridRetriever를 분리한 파일
# 로직 변경 없음 — 파일만 분리

from typing import Optional

from langchain_core.documents import Document

if __package__:
    from . import structure
    from .embedding import VectorIndexRepository
    from .database import MetadataRepository
else:
    import structure
    from embedding import VectorIndexRepository
    from database import MetadataRepository


class HybridRetriever:  # Hybrid Searching
    def __init__(
        self,
        vector_repo: VectorIndexRepository,
        metadata_repo: Optional[MetadataRepository] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.vector_repo   = vector_repo
        self.metadata_repo = metadata_repo
        self.dense_weight  = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        k: int = 6,
        dense_k: int = 8,
        sparse_k: int = 8,
        rrf_k: int = 60,
    ) -> list[structure.RetrievalResult]:
        dense_results  = self.vector_repo.search(query, k=dense_k)
        sparse_results = self.metadata_repo.keyword_search(query, k=sparse_k) if self.metadata_repo else []

        fused_scores:  dict[str, float]    = {}
        fused_docs:    dict[str, Document] = {}
        fused_sources: dict[str, str]      = {}

        for rank, item in enumerate(dense_results):
            chunk_id = str(item.document.metadata.get("chunk_id"))
            fused_scores[chunk_id]  = fused_scores.get(chunk_id, 0.0) + self.dense_weight / (rank + 1 + rrf_k)
            fused_docs[chunk_id]    = item.document
            fused_sources[chunk_id] = "dense"

        for rank, item in enumerate(sparse_results):
            chunk_id = str(item.document.metadata.get("chunk_id"))
            fused_scores[chunk_id]  = fused_scores.get(chunk_id, 0.0) + self.sparse_weight / (rank + 1 + rrf_k)
            if chunk_id not in fused_docs:
                fused_docs[chunk_id]    = item.document
                fused_sources[chunk_id] = "sparse"
            else:
                fused_sources[chunk_id] = "hybrid"

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            structure.RetrievalResult(
                document=fused_docs[chunk_id],
                score=score,
                retriever=fused_sources.get(chunk_id, "hybrid"),
            )
            for chunk_id, score in ranked
        ]
