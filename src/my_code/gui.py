from __future__ import annotations

import asyncio
import html
import os
from typing import Any, Iterable

import gradio as gr

if __package__:
    from . import structure
    from .database import try_build_metadata_repo
    from .embedding import Embedder
    from .ingestor import PDFIngestor, ChunkService
    from .test_logic import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        EMBEDDING_MODEL,
        EXAM_QUESTIONS,
        FAISS_CACHE_PATH,
        LLM_MODEL,
        PDF_PATH,
        grade,
    )
    from .pipeline import RAGPipeline
    from .reranker import try_build_reranker
    from .retriever import HybridRetriever
else:
    import structure
    from database import try_build_metadata_repo
    from embedding import Embedder
    from ingestor import PDFIngestor, ChunkService
    from test_logic import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        EMBEDDING_MODEL,
        EXAM_QUESTIONS,
        FAISS_CACHE_PATH,
        LLM_MODEL,
        PDF_PATH,
        grade,
    )
    from pipeline import RAGPipeline
    from reranker import try_build_reranker
    from retriever import HybridRetriever


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUESTION_TYPE_CHOICES = [qtype.value for qtype in structure.QuestionType]

_state: dict[str, Any] = {
    "retriever": None,
    "embedder": None,
    "pipeline": None,
    "chunks": [],
    "chunk_count": 0,
    "config": {},
    "last_exam_records": [],
}


def run_indexing(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
    emb_model: str,
    llm_model: str,
    dense_weight: float,
    sparse_weight: float,
    max_context_chunks: int,
    use_cache: bool,
    faiss_cache_path: str,
    use_reranker: bool,
    reranker_model: str,
):
    if not pdf_path.strip():
        yield "Please provide a PDF path.", "", _render_pipeline_summary()
        return

    preview_html = ""
    yield "Loading PDF...", preview_html, _render_pipeline_summary()

    try:
        raw_docs = PDFIngestor(pdf_path.strip()).load_documents()
        yield f"Loaded {len(raw_docs)} pages. Splitting chunks...", preview_html, _render_pipeline_summary()

        chunks = ChunkService(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        ).split(raw_docs)

        _state["chunks"] = chunks
        _state["chunk_count"] = len(chunks)
        preview_html = _render_chunk_preview(chunks[:5])
        yield f"Created {len(chunks)} chunks. Building vector index...", preview_html, _render_pipeline_summary()

        embedder = Embedder(mpath=emb_model.strip())
        save_path = faiss_cache_path.strip() if (use_cache and faiss_cache_path.strip()) else None
        vector_repo = embedder.get_or_build(chunks, save_path=save_path)

        metadata_repo = try_build_metadata_repo(None)
        retriever = HybridRetriever(
            vector_repo=vector_repo,
            metadata_repo=metadata_repo,
            dense_weight=float(dense_weight),
            sparse_weight=float(sparse_weight),
        )

        yield "Vector index ready. Loading LLM and pipeline...", preview_html, _render_pipeline_summary()

        llm = _build_llm(llm_model.strip())
        reranker = try_build_reranker(
            model_name=reranker_model.strip() or DEFAULT_RERANKER_MODEL,
            top_k=int(max_context_chunks),
            enabled=bool(use_reranker),
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
            max_context_chunks=int(max_context_chunks),
        )

        _state["embedder"] = embedder
        _state["retriever"] = retriever
        _state["pipeline"] = pipeline
        _state["config"] = {
            "pdf_path": pdf_path.strip(),
            "embedding_model": emb_model.strip(),
            "llm_model": llm_model.strip(),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "dense_weight": float(dense_weight),
            "sparse_weight": float(sparse_weight),
            "max_context_chunks": int(max_context_chunks),
            "faiss_cache_path": save_path or "-",
            "reranker": "enabled" if reranker else "disabled",
            "retrieval_mode": "Hybrid" if metadata_repo else "Dense (FAISS)",
        }

        yield (
            f"Indexing complete. Chunks={len(chunks)}, LLM={llm_model.strip()}",
            preview_html,
            _render_pipeline_summary(),
        )
    except Exception as exc:
        yield f"Error: {exc}", preview_html, _render_pipeline_summary()


def run_query(query: str, question_type: str, top_k: int) -> tuple[str, str]:
    pipeline: RAGPipeline | None = _state.get("pipeline")
    retriever: HybridRetriever | None = _state.get("retriever")

    if pipeline is None or retriever is None:
        return _render_notice("Run indexing first to initialize the pipeline."), ""
    if not query.strip():
        return _render_notice("Please enter a question."), ""

    try:
        response = pipeline.answer(
            question=query.strip(),
            question_type=question_type,
            k=int(top_k),
        )
        answer_html = _render_query_answer(query.strip(), question_type, response)

        retrieved = retriever.retrieve(query=query.strip(), k=int(top_k))
        retrieval_html = _render_retrieval_results(retrieved)
        return answer_html, retrieval_html
    except Exception as exc:
        return _render_notice(f"Error: {exc}"), ""


def run_exam_single(question_id: str, top_k: int) -> tuple[str, str, list[list[Any]], str]:
    pipeline: RAGPipeline | None = _state.get("pipeline")
    if pipeline is None:
        return (
            "Run indexing first to initialize the pipeline.",
            _render_exam_dashboard([]),
            [],
            "",
        )

    try:
        qid = int(str(question_id).strip())
    except ValueError:
        return "Invalid question id.", _render_exam_dashboard([]), [], ""

    q_info = _find_question(qid)
    if q_info is None:
        return f"Question Q{qid} not found.", _render_exam_dashboard([]), [], ""

    try:
        response = pipeline.answer(q_info["question"], question_type=q_info["type"], k=int(top_k))
        grading = grade(q_info, response["answer"])
        record = _build_exam_record(q_info, response, grading)
        _state["last_exam_records"] = [record]

        status = f"Q{qid} finished: {grading['earned']} / {grading['max']}"
        return (
            status,
            _render_exam_dashboard([record]),
            _build_exam_score_rows([record]),
            _render_exam_details([record]),
        )
    except Exception as exc:
        return f"Error while running Q{qid}: {exc}", _render_exam_dashboard([]), [], ""


def run_exam_all(top_k: int):
    pipeline: RAGPipeline | None = _state.get("pipeline")
    if pipeline is None:
        yield (
            "Run indexing first to initialize the pipeline.",
            _render_exam_dashboard([]),
            [],
            "",
        )
        return

    records: list[dict[str, Any]] = []
    question_count = len(EXAM_QUESTIONS)
    yield (
        f"Starting exam run ({question_count} questions)...",
        _render_exam_dashboard(records),
        _build_exam_score_rows(records),
        _render_exam_details(records),
    )

    for index, q_info in enumerate(EXAM_QUESTIONS, start=1):
        yield (
            f"Running Q{q_info['id']} ({index}/{question_count})...",
            _render_exam_dashboard(records),
            _build_exam_score_rows(records),
            _render_exam_details(records),
        )

        try:
            response = pipeline.answer(q_info["question"], question_type=q_info["type"], k=int(top_k))
            grading = grade(q_info, response["answer"])
            records.append(_build_exam_record(q_info, response, grading))
        except Exception as exc:
            records.append(
                {
                    "id": int(q_info["id"]),
                    "type": _qtype_to_text(q_info["type"]),
                    "points": int(q_info["points"]),
                    "question": str(q_info["question"]),
                    "correct_answer": str(q_info["answer"]),
                    "ai_answer": f"[ERROR] {exc}",
                    "citations": [],
                    "earned": 0,
                    "max": int(q_info["points"]),
                    "feedback": [f"Execution error: {exc}"],
                }
            )

    _state["last_exam_records"] = records
    total_earned = sum(r["earned"] for r in records)
    total_max = sum(r["max"] for r in records)
    yield (
        f"Exam finished: {total_earned} / {total_max}",
        _render_exam_dashboard(records),
        _build_exam_score_rows(records),
        _render_exam_details(records),
    )


def _build_llm(model_name: str):
    if not model_name:
        raise ValueError("LLM model name is empty.")

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name)


def _find_question(question_id: int) -> dict[str, Any] | None:
    for item in EXAM_QUESTIONS:
        if int(item["id"]) == question_id:
            return item
    return None


def _build_exam_record(q_info: dict[str, Any], response: dict[str, Any], grading: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(q_info["id"]),
        "type": _qtype_to_text(q_info["type"]),
        "points": int(q_info["points"]),
        "question": str(q_info["question"]),
        "correct_answer": str(q_info["answer"]),
        "ai_answer": str(response.get("answer", "")),
        "citations": list(response.get("citations", [])),
        "earned": int(grading["earned"]),
        "max": int(grading["max"]),
        "feedback": list(grading.get("feedback", [])),
    }


def _build_exam_score_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for rec in records:
        if rec["earned"] == rec["max"]:
            status = "Correct"
        elif rec["earned"] > 0:
            status = "Partial"
        else:
            status = "Wrong"
        rows.append(
            [
                f"Q{rec['id']}",
                rec["type"],
                rec["earned"],
                rec["max"],
                status,
            ]
        )
    return rows


def _render_pipeline_summary() -> str:
    cfg = _state.get("config") or {}
    if not cfg:
        return _render_notice("Pipeline not initialized.")

    return f"""
    <div style="border:1px solid #ddd;border-radius:10px;padding:12px">
      <div style="font-size:13px;font-weight:600;margin-bottom:8px">Runtime Configuration</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        {_kv_row("PDF", cfg.get("pdf_path", "-"))}
        {_kv_row("Embedding", cfg.get("embedding_model", "-"))}
        {_kv_row("LLM", cfg.get("llm_model", "-"))}
        {_kv_row("Chunk", f"{cfg.get('chunk_size', '-')} / overlap {cfg.get('chunk_overlap', '-')}")}
        {_kv_row("Retriever", cfg.get("retrieval_mode", "-"))}
        {_kv_row("Weights", f"dense={cfg.get('dense_weight', '-')} sparse={cfg.get('sparse_weight', '-')}")}
        {_kv_row("Max Context", cfg.get("max_context_chunks", "-"))}
        {_kv_row("Reranker", cfg.get("reranker", "-"))}
        {_kv_row("FAISS cache", cfg.get("faiss_cache_path", "-"))}
      </table>
    </div>
    """


def _kv_row(key: str, value: Any) -> str:
    return (
        "<tr>"
        f"<td style='padding:4px 6px;color:#666;width:120px'>{_esc(str(key))}</td>"
        f"<td style='padding:4px 6px;font-family:monospace'>{_esc(str(value))}</td>"
        "</tr>"
    )


def _render_chunk_preview(chunks: list[Any]) -> str:
    if not chunks:
        return _render_notice("No chunks.")

    rows = []
    for idx, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "?")
        chunk_id = str(chunk.metadata.get("chunk_id", "-"))
        text = chunk.page_content[:140]
        if len(chunk.page_content) > 140:
            text += "..."
        rows.append(
            "<tr>"
            f"<td style='padding:6px'>{idx}</td>"
            f"<td style='padding:6px'>{_esc(str(page))}</td>"
            f"<td style='padding:6px'>{_esc(text)}</td>"
            f"<td style='padding:6px;font-family:monospace;color:#666'>{_esc(chunk_id[-30:])}</td>"
            "</tr>"
        )

    return f"""
    <div style="margin-top:8px">
      <div style="font-size:13px;font-weight:600;margin-bottom:8px">Chunk Preview (top {len(chunks)})</div>
      <table style="width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:12px">
        <thead>
          <tr style="background:#f7f7f7">
            <th style="padding:6px;width:40px;text-align:left">#</th>
            <th style="padding:6px;width:60px;text-align:left">Page</th>
            <th style="padding:6px;text-align:left">Text</th>
            <th style="padding:6px;width:180px;text-align:left">Chunk ID</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """


def _render_query_answer(query: str, question_type: str, response: dict[str, Any]) -> str:
    citations = response.get("citations", [])
    citation_rows = _render_citation_rows(citations)
    return f"""
    <div style="border:1px solid #ddd;border-radius:10px;padding:12px">
      <div style="font-size:13px;font-weight:600;margin-bottom:8px">Answer</div>
      <div style="font-size:12px;color:#666;margin-bottom:8px">Type: {_esc(str(question_type))}</div>
      <div style="font-size:12px;margin-bottom:8px"><strong>Q:</strong> {_esc(query)}</div>
      <div style="font-size:12px;white-space:pre-wrap;background:#fafafa;border:1px solid #eee;border-radius:8px;padding:10px">{_esc(str(response.get("answer", "")))}</div>
      <div style="margin-top:10px;font-size:12px;color:#666">Retrieved: {response.get('retrieved_count', 0)}, Selected: {response.get('selected_count', 0)}</div>
      <div style="margin-top:10px">
        <div style="font-size:12px;font-weight:600;margin-bottom:6px">Citations</div>
        {citation_rows}
      </div>
    </div>
    """


def _render_retrieval_results(results: list[structure.RetrievalResult]) -> str:
    if not results:
        return _render_notice("No retrieval results.")

    cards: list[str] = []
    for idx, item in enumerate(results, start=1):
        meta = item.document.metadata
        source = str(meta.get("source", "unknown")).split("/")[-1]
        page = meta.get("page", "?")
        chunk_id = str(meta.get("chunk_id", "-"))
        text = item.document.page_content[:280]
        if len(item.document.page_content) > 280:
            text += "..."
        cards.append(
            f"""
            <div style="border:1px solid #e1e1e1;border-radius:10px;padding:10px;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
                <strong>#{idx} score={item.score:.4f}</strong>
                <span>{_esc(item.retriever)}</span>
              </div>
              <div style="font-size:11px;color:#666;margin-bottom:6px">
                source={_esc(source)} | page={_esc(str(page))} | chunk={_esc(chunk_id[-30:])}
              </div>
              <div style="font-size:12px;white-space:pre-wrap">{_esc(text)}</div>
            </div>
            """
        )

    return "".join(cards)


def _render_citation_rows(citations: Iterable[dict[str, Any]]) -> str:
    rows: list[str] = []
    for idx, c in enumerate(citations, start=1):
        rows.append(
            "<tr>"
            f"<td style='padding:6px'>{idx}</td>"
            f"<td style='padding:6px'>{_esc(str(c.get('page', '?')))}</td>"
            f"<td style='padding:6px'>{_esc(str(c.get('score', '?')))}</td>"
            f"<td style='padding:6px'>{_esc(str(c.get('retriever', '?')))}</td>"
            f"<td style='padding:6px;font-family:monospace'>{_esc(str(c.get('chunk_id', ''))[-30:])}</td>"
            "</tr>"
        )

    if not rows:
        return _render_notice("No citations.")

    return (
        "<table style='width:100%;border-collapse:collapse;border:1px solid #ddd;font-size:12px'>"
        "<thead><tr style='background:#f7f7f7'>"
        "<th style='padding:6px;text-align:left'>#</th>"
        "<th style='padding:6px;text-align:left'>Page</th>"
        "<th style='padding:6px;text-align:left'>Score</th>"
        "<th style='padding:6px;text-align:left'>Retriever</th>"
        "<th style='padding:6px;text-align:left'>Chunk ID</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_exam_dashboard(records: list[dict[str, Any]]) -> str:
    if not records:
        return _render_notice("No exam result yet.")

    total_earned = sum(r["earned"] for r in records)
    total_max = sum(r["max"] for r in records)
    accuracy = (100.0 * total_earned / total_max) if total_max else 0.0
    full_correct = sum(1 for r in records if r["earned"] == r["max"])

    return f"""
    <div style="border:1px solid #ddd;border-radius:10px;padding:12px">
      <div style="font-size:13px;font-weight:600;margin-bottom:8px">Exam Dashboard</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        {_metric_card("Total Score", f"{total_earned} / {total_max}")}
        {_metric_card("Accuracy", f"{accuracy:.1f}%")}
        {_metric_card("Questions", str(len(records)))}
        {_metric_card("Fully Correct", str(full_correct))}
      </div>
    </div>
    """


def _metric_card(title: str, value: str) -> str:
    return (
        "<div style='border:1px solid #e3e3e3;border-radius:8px;padding:10px;min-width:140px'>"
        f"<div style='font-size:11px;color:#666'>{_esc(title)}</div>"
        f"<div style='font-size:16px;font-weight:700'>{_esc(value)}</div>"
        "</div>"
    )


def _render_exam_details(records: list[dict[str, Any]]) -> str:
    if not records:
        return _render_notice("No exam result detail yet.")

    cards: list[str] = []
    for rec in records:
        feedback_items = "".join(f"<li>{_esc(str(item))}</li>" for item in rec["feedback"])
        citations = _render_citation_rows(rec.get("citations", []))
        cards.append(
            f"""
            <div style="border:1px solid #e1e1e1;border-radius:10px;padding:12px;margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px">
                <div style="font-size:13px;font-weight:700">Q{rec['id']} [{_esc(rec['type'])}]</div>
                <div style="font-size:12px;font-weight:600">{rec['earned']} / {rec['max']}</div>
              </div>
              <div style="font-size:12px;margin-bottom:8px"><strong>Question</strong><br>{_esc(rec['question'])}</div>
              <div style="font-size:12px;margin-bottom:8px"><strong>AI Answer</strong>
                <div style="white-space:pre-wrap;background:#fafafa;border:1px solid #eee;border-radius:8px;padding:8px;margin-top:4px">{_esc(rec['ai_answer'])}</div>
              </div>
              <div style="font-size:12px;margin-bottom:8px"><strong>Expected</strong><br>{_esc(rec['correct_answer'])}</div>
              <div style="font-size:12px;margin-bottom:8px"><strong>Feedback</strong><ul style="margin:6px 0 0 18px">{feedback_items}</ul></div>
              <div style="font-size:12px"><strong>Citations</strong>{citations}</div>
            </div>
            """
        )
    return "".join(cards)


def _render_notice(message: str) -> str:
    return f"<p style='color:#666'>{_esc(message)}</p>"


def _qtype_to_text(qtype: Any) -> str:
    if hasattr(qtype, "value"):
        return str(qtype.value)
    return str(qtype)


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _question_bank_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for q in EXAM_QUESTIONS:
        preview = str(q["question"]).replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        rows.append(
            [
                f"Q{q['id']}",
                _qtype_to_text(q["type"]),
                str(q["points"]),
                preview,
            ]
        )
    return rows


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Client") as app:
        gr.Markdown("## RAG Client")
        gr.Markdown("This UI reuses `test_logic.py` exam questions and grading logic.")

        with gr.Tab("1) Indexing"):
            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.Textbox(label="PDF Path", value=PDF_PATH, placeholder="./data/7_TransferLearning.pdf")
                    emb_model = gr.Textbox(label="Embedding Model", value=EMBEDDING_MODEL)
                    llm_model = gr.Textbox(label="LLM Model (Ollama)", value=LLM_MODEL)
                    cache_path = gr.Textbox(label="FAISS Cache Path", value=FAISS_CACHE_PATH)
                    reranker_model = gr.Textbox(label="Reranker Model", value=DEFAULT_RERANKER_MODEL)
                with gr.Column(scale=1):
                    chunk_size = gr.Slider(100, 2000, value=CHUNK_SIZE, step=50, label="Chunk Size")
                    chunk_overlap = gr.Slider(0, 300, value=CHUNK_OVERLAP, step=10, label="Chunk Overlap")
                    dense_weight = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Dense Weight")
                    sparse_weight = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Sparse Weight")
                    max_context = gr.Slider(1, 8, value=4, step=1, label="Max Context Chunks")
                    use_cache = gr.Checkbox(label="Use FAISS Cache", value=True)
                    use_reranker = gr.Checkbox(label="Enable Reranker", value=True)
                    index_btn = gr.Button("Run Indexing", variant="primary")

            index_log = gr.Textbox(label="Indexing Log", lines=6, interactive=False)
            pipeline_summary = gr.HTML(label="Runtime Status", value=_render_pipeline_summary())
            chunk_preview = gr.HTML(label="Chunk Preview")

            index_btn.click(
                fn=run_indexing,
                inputs=[
                    pdf_input,
                    chunk_size,
                    chunk_overlap,
                    emb_model,
                    llm_model,
                    dense_weight,
                    sparse_weight,
                    max_context,
                    use_cache,
                    cache_path,
                    use_reranker,
                    reranker_model,
                ],
                outputs=[index_log, chunk_preview, pipeline_summary],
            )

        with gr.Tab("2) Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(label="Question", lines=2, placeholder="What is transfer learning?")
                with gr.Column(scale=1):
                    qtype = gr.Dropdown(label="Question Type", choices=QUESTION_TYPE_CHOICES, value=structure.QuestionType.SHORT_ANSWER.value)
                    top_k = gr.Slider(1, 10, value=6, step=1, label="Top-K")
                    ask_btn = gr.Button("Ask", variant="primary")

            answer_panel = gr.HTML(label="LLM Answer")
            retrieval_panel = gr.HTML(label="Retrieved Chunks")

            ask_btn.click(
                fn=run_query,
                inputs=[query_input, qtype, top_k],
                outputs=[answer_panel, retrieval_panel],
            )
            query_input.submit(
                fn=run_query,
                inputs=[query_input, qtype, top_k],
                outputs=[answer_panel, retrieval_panel],
            )

        with gr.Tab("3) Exam"):
            gr.Markdown("Question bank and grading dashboard based on `test_logic.py`.")

            question_bank = gr.Dataframe(
                headers=["ID", "Type", "Points", "Question Preview"],
                value=_question_bank_rows(),
                interactive=False,
                label="Exam Questions",
            )

            with gr.Row():
                with gr.Column(scale=2):
                    question_id = gr.Dropdown(
                        label="Question ID",
                        choices=[str(q["id"]) for q in EXAM_QUESTIONS],
                        value=str(EXAM_QUESTIONS[0]["id"]),
                    )
                    exam_top_k = gr.Slider(1, 10, value=6, step=1, label="Top-K for Exam")
                with gr.Column(scale=1):
                    run_single_btn = gr.Button("Run Selected Question")
                    run_all_btn = gr.Button("Run Full Exam", variant="primary")

            exam_status = gr.Textbox(label="Exam Status", lines=2, interactive=False)
            exam_dashboard = gr.HTML(label="Score Dashboard", value=_render_exam_dashboard([]))
            exam_scores = gr.Dataframe(
                headers=["QID", "Type", "Score", "Max", "Status"],
                value=[],
                interactive=False,
                label="Score Table",
            )
            exam_details = gr.HTML(label="Detailed Results")

            run_single_btn.click(
                fn=run_exam_single,
                inputs=[question_id, exam_top_k],
                outputs=[exam_status, exam_dashboard, exam_scores, exam_details],
            )
            run_all_btn.click(
                fn=run_exam_all,
                inputs=[exam_top_k],
                outputs=[exam_status, exam_dashboard, exam_scores, exam_details],
            )

    return app


if __name__ == "__main__":
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    build_app().launch(server_name="127.0.0.1", server_port=7861, share=False)
