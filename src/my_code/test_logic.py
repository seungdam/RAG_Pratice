from __future__ import annotations
import os
from pathlib import Path

"""
test_logic.py — RAG 시험 에이전트 진입점
실행:
    PYTHONPATH=src python -m my_code.main
또는:
    cd src/my_code && python test_logic.py

사전 준비:
    pip install langchain langchain-community langchain-ollama
    pip install unstructured faiss-cpu sentence-transformers
    ollama pull gemma3:4b
"""

if __package__:
    from . import structure
    from .ingestor import PDFIngestor, ChunkService
    from .embedding import Embedder
    from .database import try_build_metadata_repo
    from .retriever import HybridRetriever
    from .reranker import try_build_reranker
    from .pipeline import RAGPipeline
else:
    import structure
    from ingestor import PDFIngestor, ChunkService
    from embedding import Embedder
    from database import try_build_metadata_repo
    from retriever import HybridRetriever
    from reranker import try_build_reranker
    from pipeline import RAGPipeline


# =============================================
# 설정값 (한 곳에서 관리)
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent  # /.../RAG_Pratice/src
PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "7_TransferLearning.pdf"))
DB_DSN          = None          # PostgreSQL 없으면 None
FAISS_CACHE_PATH = str(BASE_DIR / "faiss_index")  # 재실행 시 임베딩 재생성 방지

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "gemma3:270n"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 100


# =============================================
# 시험 문제
# =============================================
EXAM_QUESTIONS: list[dict] = [
    {
        "id":       1,
        "points":   2,
        "type":     structure.QuestionType.TRUE_FALSE,
        "question": (
            "In transfer learning, pre-training is typically conducted on a large-scale dataset, "
            "whereas fine-tuning is performed on a relatively small dataset for the target task."
        ),
        "answer":   "True",
        "keywords": ["true"],
    },
    {
        "id":       2,
        "points":   2,
        "type":     structure.QuestionType.TRUE_FALSE,
        "question": (
            "In multi-task learning, knowledge transfer between tasks is always positive, "
            "meaning that jointly training multiple tasks always leads to performance improvements for all tasks."
        ),
        "answer":   "False",
        "keywords": ["false"],
    },
    {
        "id":       3,
        "points":   3,
        "type":     structure.QuestionType.FILL_BLANK,
        "question": (
            "In the Delta Tuning framework for parameter-efficient fine-tuning, "
            "the three categories of methods are: "
            "(a) _____________ -based methods (e.g., inserting new modules like adapters), "
            "(b) _____________ -based methods (e.g., selecting a subset of existing parameters to update), "
            "and (c) _____________ -based methods (e.g., transforming parameters into a lower-dimensional form)."
        ),
        "answer":   "(a) Addition  (b) Specification  (c) Reparameterization",
        "keywords": ["addition", "specification", "reparameterization"],
    },
    {
        "id":       4,
        "points":   3,
        "type":     structure.QuestionType.SHORT_ANSWER,
        "question": (
            "List three benefits that transfer learning provides for the learning of target tasks, "
            "as discussed in the lecture."
        ),
        "answer":   "(1) Better generalization  (2) Sample efficiency  (3) Faster convergence",
        "keywords": ["generalization", "sample efficiency", "convergence"],
    },
]


# =============================================
# 채점 로직 (exam_rag.py)
# =============================================
def grade(q_info: dict, ai_answer: str) -> dict:
    answer   = ai_answer.strip().lower()
    keywords = q_info.get("keywords", [])
    q_id     = q_info["id"]

    if q_id in (1, 2):  # True/False: 정답 키워드 포함 시 만점
        key    = keywords[0] if keywords else ""
        earned = q_info["points"] if key in answer else 0
        feedback = [("O" if earned else "X") + f" 정답: {q_info['answer']}"]
    else:               # Fill/Short: 키워드별 1점씩
        earned   = 0
        feedback = []
        for kw in keywords:
            if kw.lower() in answer:
                feedback.append(f"  O '{kw}' 확인됨")
                earned += 1
            else:
                feedback.append(f"  X '{kw}' 미확인")

    return {"earned": earned, "max": q_info["points"], "feedback": feedback}
