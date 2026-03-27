from rag_engine import build_retriever, build_llm, ask

# =============================================
# 시험 문제 및 정답 정의
# =============================================
EXAM_QUESTIONS = [
    {
        "id": 1,
        "points": 2,
        "type": "True/False",
        "question": "In transfer learning, pre-training is typically conducted on a large-scale dataset, whereas fine-tuning is performed on a relatively small dataset for the target task.",
        "answer": "True",
    },
    {
        "id": 2,
        "points": 2,
        "type": "True/False",
        "question": "In multi-task learning, knowledge transfer between tasks is always positive, meaning that jointly training multiple tasks always leads to performance improvements for all tasks.",
        "answer": "False",
    },
    {
        "id": 3,
        "points": 3,
        "type": "Fill in the Blank",
        "question": (
            "In the Delta Tuning framework for parameter-efficient fine-tuning, "
            "the three categories of methods are: "
            "(a) _____________ -based methods (e.g., inserting new modules like adapters), "
            "(b) _____________ -based methods (e.g., selecting a subset of existing parameters to update), "
            "and (c) _____________ -based methods (e.g., transforming parameters into a lower-dimensional form)."
        ),
        "answer": "(a) Addition  (b) Specification  (c) Reparameterization",
    },
    {
        "id": 4,
        "points": 3,
        "type": "Short Answer",
        "question": "List three benefits that transfer learning provides for the learning of target tasks, as discussed in the lecture.",
        "answer": "(1) Better generalization  (2) Sample efficiency (less labeled data needed)  (3) Faster convergence",
    },
]

# =============================================
# 문제 유형별 프롬프트 템플릿
# =============================================
PROMPT_BY_TYPE = {
    "True/False": """
Answer the following True/False question based ONLY on the provided context.
Reply with "True" or "False", followed by a brief one or two sentence explanation referencing the context.

[context]
{context}
[/context]

Question: {input}
""",
    "Fill in the Blank": """
Read the context carefully and find the specific words that fill in the blanks.

[context]
{context}
[/context]

Question: {input}

Each blank (a), (b), (c) requires a single specific word found in the context above.
Replace each "_____________ " with the actual word from the context.

For example, if the answer were "Adapter", you would write:
(a) Adapter-based methods

Now write your answers:
(a)
(b)
(c)
""",
    "Short Answer": """
Answer the following question based ONLY on the provided context.
Be concise and list your points clearly (e.g., (1) ..., (2) ..., (3) ...).

[context]
{context}
[/context]

Question: {input}
""",
}


# =============================================
# 시험 문제 풀기
# =============================================
def solve_exam_question(question_info: dict, llm, retriever) -> dict:
    """RAG를 사용해 문제를 풀고 결과 dict를 반환합니다."""
    q_type = question_info["type"]
    prompt_template = PROMPT_BY_TYPE.get(q_type, PROMPT_BY_TYPE["Short Answer"])

    response = ask(question_info["question"], prompt_template, llm, retriever)

    return {
        "id": question_info["id"],
        "points": question_info["points"],
        "type": q_type,
        "question": question_info["question"],
        "ai_answer": response["answer"],
        "correct_answer": question_info["answer"],
        "context_docs": response["context"],
    }


# =============================================
# 채점 로직
# =============================================
def grade(result: dict) -> dict:
    """AI 답변을 채점하여 획득 점수와 피드백을 반환합니다."""
    q_id = result["id"]
    answer = result["ai_answer"].strip().lower()

    if q_id == 1:
        earned = 2 if "true" in answer else 0
        feedback = ["O 정답 (True)" if earned == 2 else "X 오답 (True 여야 함)"]

    elif q_id == 2:
        earned = 2 if "false" in answer else 0
        feedback = ["O 정답 (False)" if earned == 2 else "X 오답 (False 여야 함)"]

    elif q_id == 3:
        checks = [
            ("(a) Addition",           ["addition"]),
            ("(b) Specification",      ["specification"]),
            ("(c) Reparameterization", ["reparameterization"]),
        ]
        earned, feedback = 0, []
        for label, keywords in checks:
            if any(kw in answer for kw in keywords):
                feedback.append(f"O {label}")
                earned += 1
            else:
                feedback.append(f"X {label} (언급 없음)")

    elif q_id == 4:
        checks = [
            ("(1) Better generalization", ["better generalization", "generalization"]),
            ("(2) Sample efficiency",     ["sample efficiency", "less labeled", "fewer labeled", "labeled data"]),
            ("(3) Faster convergence",    ["faster convergence", "convergence"]),
        ]
        earned, feedback = 0, []
        for label, keywords in checks:
            if any(kw in answer for kw in keywords):
                feedback.append(f"O {label}")
                earned += 1
            else:
                feedback.append(f"X {label} (언급 없음)")

    else:
        earned = 0
        feedback = ["채점 기준 없음"]

    return {"earned": earned, "max": result["points"], "feedback": feedback}


# =============================================
# 출력
# =============================================
def print_result(result: dict):
    LINE = "=" * 65
    grading = grade(result)

    print(f"\n{LINE}")
    print(f"  Q{result['id']}. [{result['points']}pts] ({result['type']})")
    print(LINE)

    print(f"\n📌 [문제]\n  {result['question']}\n")

    print("📄 [참조한 강의자료 내용]")
    for i, doc in enumerate(result["context_docs"]):
        page = doc.metadata.get("page", "?")
        snippet = " ".join(doc.page_content.split())[:150]
        print(f"  [{i+1}] (p.{page}) {snippet} ...")
    print()

    print("🤖 [AI 답변]")
    for line in result["ai_answer"].strip().splitlines():
        print(f"  {line}")
    print()

    print("✅ [실제 정답]")
    print(f"  {result['correct_answer']}")
    print()

    print(f"📊 [채점 결과]  {grading['earned']} / {grading['max']} 점")
    for fb in grading["feedback"]:
        print(f"  {fb}")
    print(f"\n{LINE}")

    result["grading"] = grading


# =============================================
# 메인 실행
# =============================================
def main():
    retriever = build_retriever("./7_TransferLearning.pdf")
    llm = build_llm("gemma3:270m")

    print("\n📝 시험 문제 풀이 시작!\n")
    total_max = sum(q["points"] for q in EXAM_QUESTIONS)
    results = []

    for question_info in EXAM_QUESTIONS:
        result = solve_exam_question(question_info, llm, retriever)
        print_result(result)
        results.append(result)

    total_earned = sum(r["grading"]["earned"] for r in results)
    print(f"\n🏆 최종 점수: {total_earned} / {total_max} 점")


if __name__ == "__main__":
    main()
