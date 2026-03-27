from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

@dataclass
class QuestionType(str, Enum):
    """시험 문제 유형 열거형"""
    TRUE_FALSE      = "True/False"
    FILL_BLANK      = "Fill in the Blank"
    SHORT_ANSWER    = "Short Answer"
    MULTIPLE_CHOICE = "Multiple Choice"


@dataclass
class RetrievalResult:
    """검색 결과 하나를 나타내는 구조체"""
    document: Document
    score: float
    retriever: str  # "dense" | "sparse" | "hybrid" | "reranked"


@dataclass
class EvaluationItem:
    """자동 평가 데이터셋의 항목 하나"""
    question: str
    expected_answer: str
    required_keywords: list[str] = field(default_factory=list)
    question_type: str = QuestionType.SHORT_ANSWER
