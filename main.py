"""
main.py — ai201-project1-unofficial-guide-starter
Full entry point — wires all 6 stages together

Commands:
  python main.py --build              Build index from scratch (run once)
  python main.py --build --overwrite  Rebuild index completely
  python main.py                      Interactive CLI mode
  python main.py "your question"      Single question mode
  python main.py --eval               Run 5 evaluation questions + score
  python main.py --grounding-test     Test grounding (Milestone 5)
  python main.py --inspect-chunks     Print 5 chunks for inspection
  python app.py                       Launch Gradio web UI
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY from .env file
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

VECTOR_DIR  = "data/vector_store"
CHUNKS_PATH = "data/chunks/chunks.jsonl"

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     ai201-project1-unofficial-guide-starter               ║
║     Pre-Med & Public Health Professor Reviews            ║
║     Groq LLaMA 3.3 70B · ChromaDB · all-MiniLM-L6-v2    ║
╚══════════════════════════════════════════════════════════╝
Ask anything about TAMU pre-med or public health professors.
Type  help   for example questions.  Type  quit  to exit.
Type  web    to launch the Gradio web interface.
"""

HELP_TEXT = """
Example questions:
  • Which public health professor gives the most useful feedback?
  • Is Prof Barry good for HLTH 320?
  • What do students say about Orgo 1 at TAMU?
  • How hard are Joy DeLeon exams in Public Health?
  • Do students recommend Clendenin for Epidemiology at TAMU?
  • Which BIMS 301 professor do students prefer, Herman or Ory?
  • What does OPSA say about pre-med prerequisites at TAMU?
  • What do Reddit students say about CHEM 227 at TAMU?

Commands:
  help     show this message
  web      launch Gradio web UI at http://localhost:7860
  eval     run 5 evaluation questions
  quit     exit
"""


def pipeline(query: str, top_k: int = 5) -> dict:
    """Run one query through the full retrieve → generate pipeline."""
    from retrieve import retrieve_chunks
    from generate import generate_answer

    chunks = retrieve_chunks(query, k=top_k)
    if not chunks:
        return {
            "answer": (
                "No relevant student reviews found for that question.\n"
                "Try a professor name (Barry, DeLeon, Goodey, Clendenin, Herman)\n"
                "or a course code (HLTH 320, BIMS 301, CHEM 227, EPIB 301)."
            ),
            "query":       query,
            "num_sources": 0,
            "has_stale":   False,
            "sources":     [],
        }
    return generate_answer(query, chunks)


def build_index(overwrite: bool = False) -> None:
    """Run ingestion → cleaning → chunking → embedding."""
    from ingest import run_all_ingestion
    from clean  import clean_all_documents, verify_cleaning
    from chunk  import chunk_documents, print_5_chunks, verify_chunks
    from embed  import embed_and_store, verify_index

    print("\n=== Building ai201-project1 Index ===\n")

    # Stage 1: Ingest
    print("Stage 1/4 — Ingesting from all 10 sources...")
    raw_docs = run_all_ingestion()
    print(f"  {len(raw_docs)} raw documents\n")

    # Stage 2: Clean
    print("Stage 2/4 — Cleaning...")
    clean_docs = clean_all_documents(raw_docs)
    verify_cleaning(clean_docs)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open("data/raw/all_documents.jsonl", "w", encoding="utf-8") as f:
        for doc in clean_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  {len(clean_docs)} clean documents saved\n")

    # Stage 3: Chunk
    print("Stage 3/4 — Chunking (200 tokens / 50 overlap)...")
    chunks = chunk_documents(clean_docs)

    # Milestone 3: inspect 5 chunks before embedding
    print_5_chunks(chunks)
    verify_chunks(chunks)

    cont = input("\nInspect the 5 chunks above. Press Enter to continue to embedding, "
                 "or Ctrl+C to stop and adjust chunking settings...")

    # Stage 4: Embed
    print("\nStage 4/4 — Embedding with all-MiniLM-L6-v2...")
    total = embed_and_store(overwrite=overwrite)
    verify_index()
    print(f"\n  {total} vectors stored in ChromaDB\n")

    print("=== Index build complete ===")
    print("Run  python main.py        for CLI mode")
    print("Run  python app.py         for web UI at http://localhost:7860\n")


def is_index_ready() -> bool:
    if not Path(VECTOR_DIR).exists():
        return False
    try:
        import chromadb
        col = chromadb.PersistentClient(path=VECTOR_DIR).get_collection("tamu_reviews")
        return col.count() > 0
    except Exception:
        return False


def print_result(result: dict) -> None:
    print("\n" + "─"*60)
    print(result["answer"])
    if result["num_sources"] > 0:
        print(f"\n[{result['num_sources']} source(s) retrieved]")
    if result.get("has_stale"):
        print("⚠ Note: one or more sources may be outdated (>2 years)")
    print("─"*60 + "\n")


def main():
    args = sys.argv[1:]

    if "--build" in args:
        build_index(overwrite="--overwrite" in args)
        return

    if "--eval" in args:
        from generate import run_evaluation
        run_evaluation()
        return

    if "--grounding-test" in args:
        from generate import test_grounding
        test_grounding()
        return

    if "--inspect-chunks" in args:
        from chunk import load_chunks, print_5_chunks, diagnose_bad_chunks
        if not Path(CHUNKS_PATH).exists():
            print(f"No chunks found. Run: python main.py --build")
            return
        chunks = load_chunks(CHUNKS_PATH)
        print_5_chunks(chunks)
        diagnose_bad_chunks(chunks)
        return

    if not is_index_ready():
        print("\n⚠  Vector index not found.")
        print("Build it first:\n")
        print("  python main.py --build\n")
        sys.exit(1)

    # Single query from command line
    query_args = [a for a in args if not a.startswith("--")]
    if query_args:
        query  = " ".join(query_args)
        result = pipeline(query)
        print_result(result)
        return

    # Interactive CLI mode
    print(BANNER)

    while True:
        try:
            query = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGig 'em! 🤘")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Gig 'em! 🤘")
            break
        if query.lower() in ("help", "?", "h"):
            print(HELP_TEXT)
            continue
        if query.lower() == "web":
            print("Launching web UI at http://localhost:7860 ...")
            import subprocess
            subprocess.Popen([sys.executable, "app.py"])
            continue
        if query.lower() == "eval":
            from generate import run_evaluation
            run_evaluation()
            continue

        try:
            result = pipeline(query)
            print_result(result)
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure GROQ_API_KEY is set.\n")


if __name__ == "__main__":
    main()
