from __future__ import annotations

import re
from enum import Enum
from typing import Any, Sequence

# 지원할 LangChain 파서들을 임포트합니다.
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyMuPDFLoader,
)
from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. define parser
class ParserType(Enum):
    UNSTRUCTURED = "unstructured"
    PYMUPDF = "pymupdf"
    LLAMA_PARSE = "llama_parse"


class PDFIngestor:
    def __init__(self, file_path: str, parser_type: ParserType = ParserType.PYMUPDF):
        self.file_path = file_path
        self.parser_type = parser_type
        # 2. call factory function to load accurate loader
        self.loader = self._create_loader(file_path, parser_type)

    def _create_loader(self, file_path: str, parser_type: ParserType) -> BaseLoader:
        """파서 팩토리 함수."""
        if parser_type == ParserType.PYMUPDF:
            return PyMuPDFLoader(file_path=file_path)
            
        elif parser_type == ParserType.UNSTRUCTURED:
            return UnstructuredPDFLoader(file_path=file_path)
            
        elif parser_type == ParserType.LLAMA_PARSE:
            # LlamaParse 구현 예시 (필요시 활성화)
            from llama_parse import LlamaParse
            return LlamaParse(result_type="markdown",  # "markdown"과 "text" 사용 가능
                            num_workers=8,  # worker 수 (기본값: 4)
                            verbose=True,
                            language="ko",)
        else:
            raise ValueError(f"지원하지 않는 파서 타입입니다: {parser_type}")

    def load_documents(self) -> list[Document]:
        # 3. 다형성을 활용하여 어떤 로더가 할당되었든 동일한 .load() 메서드를 호출합니다.
        docs = self.loader.load()
        if not docs:
            raise ValueError(f"Failed to load PDF: {self.file_path}")
        return docs


class ChunkService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: Sequence[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(list(docs))
        processed: list[Document] = []

        for idx, chunk in enumerate(chunks):
            chunk.page_content = self._normalize_text(chunk.page_content)
            
            # consider of meta data diff between parsers
            source   = str(chunk.metadata.get("source", "unknown"))
            page     = self._safe_int(chunk.metadata.get("page"), default=-1)
            section  = str(chunk.metadata.get("section") or chunk.metadata.get("category") or "unknown")
            chunk_id = str(chunk.metadata.get("chunk_id") or f"{source}:{page}:{idx}")

            chunk.metadata.update({
                "chunk_id": chunk_id,
                "source":   source,
                "page":     page,
                "section":  section,
            })
            processed.append(chunk)

        return processed

    @staticmethod
    def _normalize_text(text: str) -> str:
        one_line_break_to_space = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        collapsed_space = re.sub(r"[ \t]+", " ", one_line_break_to_space)
        return collapsed_space.strip()

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default