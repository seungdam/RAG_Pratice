from __future__ import annotations

# rag.py의 RAGPipeline._build_prompt()에서 분리 + 고도화
# 기존: 문제 유형 관계없이 프롬프트가 하나뿐이었음
# 개선: QuestionType별로 전용 프롬프트 사용

if __package__:
    from .structure import QuestionType
else:
    from structure import QuestionType

_TEMPLATES: dict[str, str] = {

    QuestionType.TRUE_FALSE: """
You are a True/False question answering assistant.
Answer based ONLY on the provided context below.
Start your answer with exactly "True" or "False".
Then add one or two sentences explaining why, referencing the context.

[Context]
{context}
[/Context]

Question: {question}

Answer (must start with True or False):
""".strip(),

    QuestionType.FILL_BLANK: """
You are a fill-in-the-blank answering assistant.
Read the context carefully and find the EXACT words that belong in each blank.
Use ONLY terminology found in the context. Do NOT guess or invent words.

[Context]
{context}
[/Context]

Question: {question}

Instructions:
- Each blank (a), (b), (c) must be filled with a single specific term from the context.
- Format your answer exactly as:
  (a) [term]-based methods
  (b) [term]-based methods
  (c) [term]-based methods

Your answer:
""".strip(),

    QuestionType.SHORT_ANSWER: """
You are a short-answer question answering assistant.
Answer based ONLY on the provided context below.
Be concise. If the question asks for a list, number your points: (1) ..., (2) ..., (3) ...
Do NOT include information outside the context.

[Context]
{context}
[/Context]

Question: {question}

Answer:
""".strip(),

    QuestionType.MULTIPLE_CHOICE: """
You are a multiple-choice question answering assistant.
Choose the single best answer based ONLY on the provided context.
State the letter of your choice first, then briefly explain using evidence from the context.

[Context]
{context}
[/Context]

Question: {question}

Answer (letter first, then explanation):
""".strip(),
}

# 등록되지 않은 유형의 fallback
_FALLBACK = _TEMPLATES[QuestionType.SHORT_ANSWER]


def build_prompt(question: str, context: str, question_type: str = QuestionType.SHORT_ANSWER) -> str:
    """
    문제 유형에 맞는 완성된 프롬프트 문자열 반환.

    기존 RAGPipeline._build_prompt()는 단일 범용 프롬프트였고
    question_type을 아예 인식하지 못했습니다.
    이 함수는 QuestionType별로 최적화된 지시문을 사용합니다.
    """
    template = _TEMPLATES.get(question_type, _FALLBACK)
    return template.format(question=question, context=context)
