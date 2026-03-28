from __future__ import annotations

# rag.py의 RAGPipeline을 기반으로 고도화한 파일
#
# 기존 RAGPipeline과의 차이:
#   기존: answer(question, k) → 단일 범용 프롬프트, 상위 N개 단순 선택
#   개선: answer(question, question_type, k) → 유형별 프롬프트 + Reranker 정렬

from typing import Any, Callable, Optional, Protocol, Sequence

if __package__:
    from . import structure
    from . import prompt as prompt_module
    from .retriever import HybridRetriever
    from .reranker import CrossEncoderReranker
else:
    import structure
    import prompt as prompt_module
    from retriever import HybridRetriever
    from reranker import CrossEncoderReranker


class SupportsInvoke(Protocol):
    """rag.py의 기존 Protocol 유지"""
    def invoke(self, input: str, **kwargs: Any) -> Any:
        ...


class RAGPipeline:  # Prompting Pipeline
    """
    고도화된 RAG 파이프라인.

    기존 코드에서 변경된 부분:
    1. answer()에 question_type 파라미터 추가
       → prompt.py의 build_prompt()를 통해 유형별 프롬프트 자동 선택
    
    2. reranker 파라미터 추가
       → CrossEncoderReranker가 있으면 Rerank 후 선택,
          없으면 기존 방식(retrieved[:max_context_chunks])으로 fallback
    
    3. _build_context()에 retriever, score 정보 추가
       → 어떤 방식으로 검색됐는지 컨텍스트에 명시

    4. citations에 retriever 필드 추가
       → dense/sparse/hybrid/+reranked 추적 가능
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: SupportsInvoke | Callable[[str], Any],
        reranker: Optional[CrossEncoderReranker] = None,  # 신규
        max_context_chunks: int = 4,
    ):
        self.retriever          = retriever
        self.llm                = llm
        self.reranker           = reranker
        self.max_context_chunks = max_context_chunks

    # ── 기존 answer() 시그니처 확장 ──────────────────────
    def answer(
        self,
        question: str,
        question_type: str = structure.QuestionType.SHORT_ANSWER,  # 신규
        k: int = 6,
    ) -> dict[str, Any]:
        # [1] Retrieve (기존 로직 유지)
        retrieved = self.retriever.retrieve(query=question, k=k)

        # [2] 신규: Reranker로 정밀 정렬, 없으면 기존 방식
        selected = self._select_chunks(question, retrieved)

        # [3] Context 구성 (retriever 정보 추가됨)
        context = self._build_context(selected)

        # [4] 신규: 문제 유형별 프롬프트 (기존은 _build_prompt 단일 함수)
        final_prompt = prompt_module.build_prompt(question, context, question_type)

        # [5] LLM 호출 (기존 로직 유지)
        answer_text = self._invoke_llm(final_prompt)

        # citations에 retriever 필드 추가
        citations = [
            {
                "chunk_id":  str(item.document.metadata.get("chunk_id", "")),
                "source":    str(item.document.metadata.get("source", "unknown")),
                "page":      int(item.document.metadata.get("page", -1)),
                "score":     round(item.score, 4),
                "retriever": item.retriever,  # 신규: 검색 방식 추적
            }
            for item in selected
        ]

        return {
            "question":        question,
            "answer":          answer_text,
            "citations":       citations,
            "retrieved_count": len(retrieved),
            "selected_count":  len(selected),   # 신규
        }

    # ── 신규: 청크 선택 로직 분리 ─────────────────────────
    def _select_chunks(
        self,
        question: str,
        retrieved: list[structure.RetrievalResult],
    ) -> list[structure.RetrievalResult]:
        """
        Reranker 유무에 따라 자동으로 선택 방식을 결정합니다.
        - Reranker 있음 → CrossEncoder로 재정렬 후 top_k 반환
        - Reranker 없음 → 기존 방식: retrieved[:max_context_chunks]
        """
        if self.reranker and self.reranker.available:
            return self.reranker.rerank(question, retrieved)
        return retrieved[:self.max_context_chunks]

    # ── _build_context: retriever/score 정보 추가 ────────
    def _build_context(self, items: Sequence[structure.RetrievalResult]) -> str:
        """기존 대비 retriever(dense/sparse/hybrid) 및 score 정보 헤더에 추가"""
        blocks = []
        for idx, item in enumerate(items, start=1):
            source    = item.document.metadata.get("source", "unknown")
            page      = item.document.metadata.get("page", -1)
            chunk_id  = item.document.metadata.get("chunk_id", f"chunk-{idx}")
            score     = round(item.score, 4)
            retriever = item.retriever
            blocks.append(
                f"[{idx}] source={source}, page={page}, chunk_id={chunk_id}, "
                f"score={score}, via={retriever}\n"
                f"{item.document.page_content}"
            )
        return "\n\n".join(blocks)

    # ── _invoke_llm: 기존 로직 100% 유지 ─────────────────
    def _invoke_llm(self, prompt: str) -> str:
        if hasattr(self.llm, "invoke"):
            raw = self.llm.invoke(prompt)
        elif callable(self.llm):
            raw = self.llm(prompt)
        else:
            raise TypeError("LLM must be callable or support invoke().")

        if hasattr(raw, "content"):
            return str(raw.content).strip()
        return str(raw).strip()
