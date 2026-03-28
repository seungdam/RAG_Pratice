from __future__ import annotations


import re
from typing import Any, Sequence

if __package__:
    from . import structure
    from .pipeline import RAGPipeline
else:
    import structure
    from pipeline import RAGPipeline


class Evaluator:  # Response Evaluator
    def evaluate(
        self,
        pipeline: RAGPipeline,
        dataset: Sequence[structure.EvaluationItem],
        pass_threshold: float = 0.7,
    ) -> dict[str, Any]:
        details: list[dict[str, Any]] = []
        total_score = 0.0
        pass_count  = 0

        for item in dataset:
        
            #question_type 전달로 문제 유형별 프롬프트 활성화
            response  = pipeline.answer(item.question, question_type=item.question_type)
            predicted = str(response["answer"])
            score     = self._score_answer(item, predicted)
            total_score += score
            if score >= pass_threshold:
                pass_count += 1

            details.append({
                "question":         item.question,
                "expected_answer":  item.expected_answer,
                "predicted_answer": predicted,
                "score":            round(score, 4),
                "citations":        response.get("citations", []),
            })

        count     = len(dataset)
        accuracy  = (total_score / count) if count else 0.0
        pass_rate = (pass_count / count)  if count else 0.0

        return {
            "count":            count,
            "accuracy":         round(accuracy, 4),
            "pass_rate_at_0_7": round(pass_rate, 4),
            "details":          details,
        }


    def _score_answer(self, item: structure.EvaluationItem, predicted: str) -> float:
        expected = self._normalize(item.expected_answer)
        pred     = self._normalize(predicted)

        if not pred:
            return 0.0
        if expected and expected in pred:
            return 1.0

        if item.required_keywords:
            keywords = [self._normalize(k) for k in item.required_keywords]
            matched  = sum(1 for kw in keywords if kw and kw in pred)
            return matched / len(keywords)

        expected_tokens = set(expected.split())
        pred_tokens     = set(pred.split())
        if not expected_tokens:
            return 0.0

        overlap = len(expected_tokens & pred_tokens) / len(expected_tokens)
        if overlap >= 0.7:
            return 0.7
        if overlap >= 0.4:
            return 0.4
        return 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        lowered   = text.lower().strip()
        collapsed = re.sub(r"\s+", " ", lowered)
        return re.sub(r"[^\w\s가-힣]", "", collapsed)
