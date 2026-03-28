from __future__ import annotations
import os
from pathlib import Path

"""
main.py — RAG 시험 에이전트 진입점
실행:
    PYTHONPATH=src python -m my_code.main
또는:
    cd src/my_code && python main.py

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
DB_DSN          = None          # PostgreSQL 없으면 None → Dense 전용 모드
FAISS_CACHE_PATH = str(BASE_DIR / "faiss_index")  # 재실행 시 임베딩 재생성 방지

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "gemma3:4b"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 100


# =============================================
# 시험 문제 정의
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
# 채점 로직 (샘플 exam_rag.py 방식 유지 + 부분점수)
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


# =============================================
# 결과 출력
# =============================================
def print_result(q_info: dict, response: dict, grading: dict) -> None:
    LINE = "=" * 65
    print(f"\n{LINE}")
    print(f"  Q{q_info['id']}. [{q_info['points']}pts] ({q_info['type']})")
    print(LINE)

    print(f"\n[문제]\n  {q_info['question']}\n")

    print("📄 [참조한 강의자료]")
    for i, c in enumerate(response["citations"], 1):
        print(f"  [{i}] p.{c['page']} | score={c['score']} | via={c['retriever']}")
    print()

    print("[AI 답변]")
    for line in response["answer"].strip().splitlines():
        print(f"  {line}")
    print()

    print("[실제 정답]")
    print(f"  {q_info['answer']}")
    print()

    print(f"[채점]  {grading['earned']} / {grading['max']} 점")
    for fb in grading["feedback"]:
        print(fb)
    print(f"\n{LINE}")


# =============================================
# 메인 실행: 모든 클래스 조립
# =============================================
def main() -> None:
    from langchain_ollama import ChatOllama

    raw_docs = PDFIngestor(PDF_PATH).load_documents()
    chunks   = ChunkService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split(raw_docs)

    embedder    = Embedder(mpath=EMBEDDING_MODEL)
    vector_repo = embedder.get_or_build(chunks, save_path=FAISS_CACHE_PATH)

    # ── 3. DB (없으면 None → Dense 전용) ─────────────────
    metadata_repo = try_build_metadata_repo(DB_DSN)
    if metadata_repo:
        metadata_repo.initialize_schema()
        metadata_repo.upsert_documents(chunks)

    # ── 4. 검색기 ─────────────────────────────────────────
    retriever = HybridRetriever(
        vector_repo=vector_repo,
        metadata_repo=metadata_repo,
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    # ── 5. Reranker (없거나 실패하면 None) ────────────────
    reranker = try_build_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=4,
        enabled=True,
    )

    print(f"LLM 로딩: {LLM_MODEL}")
    llm = ChatOllama(model=LLM_MODEL)


    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        max_context_chunks=4,
    )

    print("\n시험 문제 풀이 시작!\n")
    total_max    = sum(q["points"] for q in EXAM_QUESTIONS)
    total_earned = 0

    for q_info in EXAM_QUESTIONS:
        response = pipeline.answer(q_info["question"], question_type=q_info["type"])
        grading  = grade(q_info, response["answer"])
        print_result(q_info, response, grading)
        total_earned += grading["earned"]

    print(f"\n최종 점수: {total_earned} / {total_max} 점")


if __name__ == "__main__":
    main()
