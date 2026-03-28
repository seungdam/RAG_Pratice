from __future__ import annotations

# rag.py에서 EmbeddingService, VectorIndexRepository, Embedder를 분리한 파일
# 기존 로직은 100% 유지, Embedder에만 get_or_build 메서드 추가

import os
from typing import Optional, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
try:
    # Preferred in recent LangChain versions.
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Backward-compatible fallback for older environments.
    from langchain_community.embeddings import HuggingFaceEmbeddings

if __package__:
    from . import structure
else:
    import structure


class EmbeddingService:  # Text to Embedding Vector
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        """HuggingFace 기반 모델 호출"""
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )

    def get_model(self) -> HuggingFaceEmbeddings:
        return self.model


class VectorIndexRepository:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vector_store: Optional[FAISS] = None

    def build(self, docs: Sequence[Document]) -> None:
        if not docs:
            raise ValueError("No documents to index.")
        self.vector_store = FAISS.from_documents(list(docs), self.embedding_service.get_model())

    def save_local(self, path: str) -> None:
        if self.vector_store is None:
            raise ValueError("Vector index is not built yet.")
        self.vector_store.save_local(path)

    def load_local(self, path: str) -> None:
        self.vector_store = FAISS.load_local(
            path,
            self.embedding_service.get_model(),
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int = 6) -> list[structure.RetrievalResult]:
        if self.vector_store is None:
            raise ValueError("Vector index is not ready.")

        raw = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            structure.RetrievalResult(
                document=doc,
                score=(1.0 / (1.0 + max(float(distance), 0.0))),
                retriever="dense",
            )
            for doc, distance in raw
        ]


class Embedder:
    def __init__(self, mpath: str):
        self.embedding_service = EmbeddingService(model_name=mpath)
        self.vector_repo = VectorIndexRepository(self.embedding_service)

    def set_vector(self, docs: Sequence[Document]) -> FAISS:
        self.vector_repo.build(docs)
        if self.vector_repo.vector_store is None:
            raise RuntimeError("Failed to build FAISS vector store.")
        return self.vector_repo.vector_store

    # 캐시 활용 빌더
    def get_or_build(
        self,
        docs: Sequence[Document],
        save_path: str | None = None,
    ) -> VectorIndexRepository:
        """
        FAISS 인덱스 캐시가 있으면 로드, 없으면 새로 빌드 후 저장.
        Args:
            docs:      ChunkService.split()이 반환한 청크 리스트
            save_path: FAISS 인덱스 저장/로드 경로 (None이면 저장 안 함)
        Returns:
            VectorIndexRepository (search() 바로 사용 가능한 상태)
        """
        if save_path and os.path.exists(save_path):
            print(f"FAISS 캐시 로드: {save_path}")
            self.vector_repo.load_local(save_path)
        else:
            print("FAISS 인덱스 신규 빌드 중...")
            self.vector_repo.build(docs)
            if save_path:
                self.vector_repo.save_local(save_path)
                print(f"FAISS 인덱스 저장: {save_path}")

        return self.vector_repo
