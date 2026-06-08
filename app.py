"""
app.py — Gradio Web Interface
ai201-project1-unofficial-guide-starter

Per Milestone 5: Build a usable query interface.
A viewer can understand how to use it without narration.

Run: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from retrieve import retrieve_chunks
from generate import generate_answer, GROQ_MODEL


def ask(question: str) -> tuple[str, str, str]:
    """
    Full pipeline: question → retrieve → generate → return.

    Returns:
      answer:  generated answer with inline citations
      sources: list of retrieved sources
      debug:   chunk details for inspection
    """
    if not question.strip():
        return (
            "Please enter a question.",
            "",
            "",
        )

    try:
        # Stage 1: Retrieve top-5 relevant chunks
        chunks = retrieve_chunks(question)

        if not chunks:
            return (
                "No relevant student reviews found for that question.\n\n"
                "Try including a professor name (Barry, DeLeon, Goodey, "
                "Clendenin, Herman) or a course code (HLTH 320, BIMS 301, "
                "CHEM 227, EPIB 301).",
                "No sources retrieved.",
                "No chunks matched above similarity threshold.",
            )

        # Stage 2: Generate grounded cited answer
        result = generate_answer(question, chunks)

        # Format sources for display
        sources_text = "\n".join(
            f"[{i+1}] {c.get('professor_name','Unknown')} — "
            f"{c.get('source','Unknown')} "
            f"(similarity: {c.get('similarity_score',0):.3f}"
            f"{', ⚠ possibly outdated' if c.get('is_stale') else ''})"
            for i, c in enumerate(chunks)
        )

        # Format debug info
        debug_text = "\n\n".join(
            f"Chunk {i+1} | {c.get('source')} | "
            f"Prof: {c.get('professor_name','N/A')} | "
            f"dist: {c.get('distance_score',0):.3f}\n"
            f"{c['text'][:300]}..."
            for i, c in enumerate(chunks)
        )

        return result["answer"], sources_text, debug_text

    except Exception as e:
        return (
            f"Error: {str(e)}\n\n"
            "Make sure:\n"
            "1. You ran: python main.py --build\n"
            "2. GROQ_API_KEY is set in your .env file",
            "",
            "",
        )


# ── Example questions ──
EXAMPLE_QUESTIONS = [
    "What do students say about Prof Adam Barry in TAMU Public Health?",
    "Is Joy DeLeon a good professor for HLTH 215?",
    "Which professor should I take for Orgo 1 at TAMU?",
    "How hard are exams in EPIB 301 Epidemiology at TAMU?",
    "Do students recommend Jim Herman for BIMS 301?",
    "What do Reddit students say about CHEM 227 professor recommendations?",
    "What is the workload like in Angela Clendenin epidemiology course?",
    "What are the required pre-med courses at TAMU?",
]

CSS = """
* { font-family: 'Segoe UI', Arial, sans-serif !important; }

* { font-family: 'Inter', sans-serif !important; }
.gr-button-primary { background: #500000 !important; border-color: #500000 !important; }
.gr-button-primary:hover { background: #3a0000 !important; }
footer { display: none !important; }
"""

# ── Build UI ──
with gr.Blocks(title="The Unofficial TAMU Guide") as demo:

    gr.Markdown("""
    # 🤘 The Unofficial TAMU Guide
    ### Pre-Med & Public Health Professor Reviews — Texas A&M University

    Ask any question about TAMU pre-med or public health professors.
    Answers are grounded in real student reviews from 10 sources.
    Every claim is cited with a source number [1][2].

    **Sources:** Rate My Professors · Reddit · Niche · Coursicle · Discord ·
    Uloop · Professors.directory · TAMU SPH Catalog · Pre-Med Society · OPSA

    **Model:** `llama-3.3-70b-versatile` via Groq · **Embeddings:** `all-MiniLM-L6-v2` · **Vector store:** ChromaDB
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What do students say about Prof Barry's exams?",
                lines=2,
            )
            with gr.Row():
                ask_btn   = gr.Button("Ask →", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Markdown("**Example questions — click to fill:**")
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question_box,
                label="",
            )

    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Answer",
                lines=12,
            )
        with gr.Column(scale=2):
            sources_box = gr.Textbox(
                label="Retrieved sources",
                lines=8,
            )

    with gr.Accordion("Debug — retrieved chunks (inspect for grounding)", open=False):
        debug_box = gr.Textbox(
            label="Raw retrieved chunks",
            lines=15,
        )
        gr.Markdown(
            "**Milestone 5 grounding check:** Every claim in the answer above "
            "should trace to one of these chunks. If the answer contains "
            "information not in any chunk — that is a grounding failure."
        )

    gr.Markdown("""
    ---
    **How to use:**
    - Type a question and press **Ask** or hit Enter
    - Every answer includes source citations [1][2]
    - Check **Retrieved sources** to see where each claim came from
    - Expand **Debug** to inspect raw retrieved chunks
    - If the system says it does not have enough information — that is correct behavior, not a bug
    """)

    # ── Wire up events ──
    ask_btn.click(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )
    question_box.submit(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, sources_box, debug_box],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[question_box, answer_box, sources_box, debug_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
