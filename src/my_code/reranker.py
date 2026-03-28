from __future__ import annotations

# rag.py에 없던 기능 — 신규 추가
# CrossEncoder 기반 Reranker

from typing import Optional

if __package__:
    from . import structure
else:
    import structure


class CrossEncoderReranker:
    """
    CrossEncoder 기반 Reranker.

    HybridRetriever(Bi-Encoder)와의 차이:
    ┌────────────────┬─────────────────────────────────────┐
    │  Bi-Encoder   │ 질문, 청크를 각각 벡터화 후 유사도 계산  │
    │  (FAISS)      │ 빠르지만 문맥 이해 없이 단순 거리 측정   │
    ├────────────────┼─────────────────────────────────────┤
    │  CrossEncoder  │ (질문, 청크) 쌍을 함께 입력해 직접 점수화│
    │  (Reranker)   │ 느리지만 실제 관련성 정밀 평가 가능      │
    └────────────────┴─────────────────────────────────────┘

    RAG 표준 2단계 전략:
      1단계: HybridRetriever → 빠르게 후보 k개 추출
      2단계: CrossEncoderReranker → 후보를 정밀하게 재정렬
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 4,
    ):
        self.top_k      = top_k
        self._available = False
        self._model     = None

        try:
            from sentence_transformers import CrossEncoder
            print(f"🔁 Reranker 로딩: {model_name}")
            self._model     = CrossEncoder(model_name)
            self._available = True
            print("   → 완료")
        except ImportError:
            print("⚠️  sentence-transformers 없음 → Reranker 비활성화")
            print("   설치: pip install sentence-transformers")
        except Exception as e:
            print(f"⚠️  Reranker 로딩 실패 ({e}) → 비활성화")

    @property
    def available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        results: list[structure.RetrievalResult],
    ) -> list[structure.RetrievalResult]:
        """
        HybridRetriever 결과를 CrossEncoder 점수로 재정렬.

        available=False면 원본 리스트 상위 top_k개를 그대로 반환합니다.
        (pipeline.py에서 None 체크 없이 항상 이 메서드를 호출할 수 있습니다)
        """
        if not self._available or not results:
            return results[:self.top_k]

        # (질문, 청크 내용) 쌍 리스트를 CrossEncoder에 한번에 전달
        pairs  = [(query, item.document.page_content) for item in results]
        scores = self._model.predict(pairs)  # numpy array 반환

        reranked = sorted(
            zip(scores, results),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        return [
            structure.RetrievalResult(
                document=item.document,
                score=float(score),
                retriever=item.retriever + "+reranked",  # 출처 추적용 표시
            )
            for score, item in reranked[:self.top_k]
        ]


def try_build_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 4,
    enabled: bool = True,
) -> Optional[CrossEncoderReranker]:
    """
    Reranker를 안전하게 생성하는 팩토리.
    enabled=False거나 로딩 실패 시 None 반환.
    pipeline.py는 None이면 자동으로 단순 상위 N개 선택으로 대체합니다.
    """
    if not enabled:
        print("ℹ️  Reranker 비활성화 설정")
        return None
    reranker = CrossEncoderReranker(model_name=model_name, top_k=top_k)
    return reranker if reranker.available else None
