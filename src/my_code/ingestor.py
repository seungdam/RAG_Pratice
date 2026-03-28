from __future__ import annotations

import re
from enum import Enum
from typing import Any, Sequence
from collections import defaultdict
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

class PDFIngestor:
    def __init__(self, file_path: str, parser_type: ParserType = ParserType.PYMUPDF):
        self.file_path = file_path
        self.parser_type = parser_type
        self.loader = self._create_loader(file_path, parser_type)

    def _create_loader(self, file_path: str, parser_type: ParserType) -> BaseLoader:
        """파서 팩토리 함수."""
        if parser_type == ParserType.PYMUPDF:
            return PyMuPDFLoader(file_path=file_path)
        elif parser_type == ParserType.UNSTRUCTURED:
            return UnstructuredPDFLoader(file_path=file_path,
                                         mode="elements", # Parsing option
                                         strategy="hi_res") # PDF/Slide
        else:
            raise ValueError(f"지원하지 않는 파서 타입입니다: {parser_type}")

    def load_documents(self) -> list[Document]:
        # 3. 다형성을 활용하여 어떤 로더가 할당되었든 동일한 .load() 메서드를 호출합니다.
        docs = self.loader.load()
        if not docs:
            raise ValueError(f"Failed to load PDF: {self.file_path}")
        for doc in docs:
            doc.metadata.setdefault("parser", self.parser_type.value)
        return docs


class ChunkService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: Sequence[Document]) -> list[Document]:
        if not docs:
            return []

        # parser별로 처리 분기
        first_parser = self._detect_parser(docs[0])

        if first_parser == ParserType.UNSTRUCTURED:
            return self._split_unstructured_by_page(docs)
        else:
            # PyMuPDF는 보통 이미 page 단위 Document로 들어오는 경우가 많음
            return self._split_generic_documents(docs)

    def _split_unstructured_by_page(self, docs: Sequence[Document]) -> list[Document]:
        """
        Unstructured element들을 page_number 기준으로 묶은 뒤,
        각 page 내부에서만 split 수행
        """
        page_buckets: dict[int, list[Document]] = defaultdict(list)

        for doc in docs:
            page = self._extract_page(doc.metadata)
            page_buckets[page].append(doc)

        page_docs: list[Document] = []

        for page, elements in sorted(page_buckets.items(), key=lambda x: x[0]):
            merged_texts: list[str] = []
            categories: list[str] = []

            for el in elements:
                text = self._normalize_text(el.page_content)
                if not text:
                    continue

                merged_texts.append(text)

                category = el.metadata.get("category")
                if category:
                    categories.append(str(category))

            merged_text = "\n".join(merged_texts).strip()
            if not merged_text:
                continue

            first_meta = dict(elements[0].metadata)

            # source / section / parser metadata 정리
            source = str(first_meta.get("source", "unknown"))
            section = str(first_meta.get("section", "unknown"))

            page_docs.append(
                Document(
                    page_content=merged_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "page_number": page,
                        "section": section,
                        "element_types": list(sorted(set(categories))),
                        "parser": ParserType.UNSTRUCTURED.value,
                    },
                )
            )

        # 이제 page_docs는 "페이지 단위 문서"
        # splitter가 돌더라도 페이지 경계 밖으로는 안 넘어감
        return self._post_split(page_docs)

    def _split_generic_documents(self, docs: Sequence[Document]) -> list[Document]:
        """
        PyMuPDF 등 일반 문서 처리.
        이미 page 단위 문서라고 가정하고 split 수행
        """
        normalized_docs: list[Document] = []

        for doc in docs:
            page = self._extract_page(doc.metadata)

            normalized_docs.append(
                Document(
                    page_content=self._normalize_text(doc.page_content),
                    metadata={
                        **doc.metadata,
                        "page": page,
                        "page_number": page,
                        "source": str(doc.metadata.get("source", "unknown")),
                        "section": str(
                            doc.metadata.get("section")
                            or doc.metadata.get("category")
                            or "unknown"
                        ),
                        "parser": str(doc.metadata.get("parser", ParserType.PYMUPDF.value)),
                    },
                )
            )

        return self._post_split(normalized_docs)

    def _post_split(self, docs: Sequence[Document]) -> list[Document]:
        """
        최종 split + metadata 정리
        """
        chunks = self.splitter.split_documents(list(docs))
        processed: list[Document] = []

        for idx, chunk in enumerate(chunks):
            chunk.page_content = self._normalize_text(chunk.page_content)

            source = str(chunk.metadata.get("source", "unknown"))
            page = self._safe_int(
                chunk.metadata.get("page", chunk.metadata.get("page_number")),
                default=-1,
            )
            section = str(chunk.metadata.get("section", "unknown"))
            parser = str(chunk.metadata.get("parser", "unknown"))

            chunk_id = str(chunk.metadata.get("chunk_id") or f"{source}:p{page}:{idx}")

            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "page_number": page,
                    "section": section,
                    "parser": parser,
                }
            )
            processed.append(chunk)

        return processed

    @staticmethod
    def _extract_page(metadata: dict[str, Any]) -> int:
        raw = metadata.get("page_number", metadata.get("page", -1))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _detect_parser(doc: Document) -> ParserType:
        parser = str(doc.metadata.get("parser", "")).lower()

        if parser == ParserType.UNSTRUCTURED.value:
            return ParserType.UNSTRUCTURED

        # Unstructured element 문서들은 category/page_number를 자주 가짐
        if "page_number" in doc.metadata or "category" in doc.metadata:
            return ParserType.UNSTRUCTURED

        return ParserType.PYMUPDF

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
