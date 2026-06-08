"""
chunk.py — Stage 3: Chunking
ai201-project1-unofficial-guide-starter

Fixed-size chunking per planning.md:
  Chunk size:  200 tokens (~800 chars)
  Overlap:     50 tokens  (~200 chars)
  Tool:        LangChain RecursiveCharacterTextSplitter

Every chunk gets a metadata prefix prepended:
  [Professor: Adam Barry | Course: HLTH 320 | Dept: Public Health]

This ensures even a mid-review boundary chunk identifies
the professor and course without relying on metadata alone.

Per Milestone 3:
  - Prints 5 representative chunks for manual inspection
  - Reports total chunk count (target: 50–2000)
  - Diagnoses bad chunk types (empty, HTML, fragments)

Run standalone: python chunk.py
"""

import json
import random
import logging
from pathlib import Path
from collections import defaultdict

log = logging.getLogger(__name__)

CHUNKS_DIR       = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_CHARS = 800    # 200 tokens × ~4 chars/token
OVERLAP_CHARS    = 200    # 50 tokens  × ~4 chars/token
REVIEW_SEP       = " [REVIEW_END] "


def build_prefix(doc: dict) -> str:
    """
    Build prefix prepended to every chunk.
    Format: [Professor: X | Course: Y | Dept: Z]
    Ensures boundary chunks always carry identifying context.
    """
    parts = []
    if doc.get("professor_name"):
        parts.append(f"Professor: {doc['professor_name']}")
    if doc.get("course"):
        parts.append(f"Course: {doc['course']}")
    if doc.get("department"):
        parts.append(f"Dept: {doc['department']}")
    return ("[" + " | ".join(parts) + "] ") if parts else ""


def chunk_document(doc: dict, splitter) -> list[dict]:
    """
    Split one document into chunks with full metadata on each.

    1. Build prefix from professor/course/department
    2. Prepend prefix to document text
    3. Split with RecursiveCharacterTextSplitter
    4. Re-add prefix to any boundary chunks that lost it
    5. Attach all metadata fields to every chunk dict
    """
    prefix    = build_prefix(doc)
    full_text = prefix + doc.get("text", "")

    raw_chunks = splitter.split_text(full_text)
    chunks = []

    for i, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue
        # Re-add prefix if split removed it from boundary chunk
        if prefix and not text.startswith("["):
            text = prefix + text

        chunk = {
            "text":           text,
            "chunk_index":    i,
            "professor_name": doc.get("professor_name"),
            "department":     doc.get("department"),
            "course":         doc.get("course"),
            "source":         doc.get("source"),
            "rating":         doc.get("rating"),
            "review_date":    doc.get("review_date"),
            "is_stale":       doc.get("is_stale", False),
            "url":            doc.get("url"),
        }
        for key in ("thread_title", "subreddit", "channel"):
            if doc.get(key):
                chunk[key] = doc[key]
        chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = OVERLAP_CHARS,
    output_path: str = "data/chunks/chunks.jsonl",
) -> list[dict]:
    """
    Chunk all documents and save to chunks.jsonl.

    Uses [REVIEW_END] as first separator so reviews are never
    split mid-boundary into the same chunk as a different review.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "LangChain not installed.\n"
            "Run: pip install langchain langchain-text-splitters"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            REVIEW_SEP,  # review boundaries first
            "\n\n",      # paragraph breaks
            "\n",        # line breaks
            ". ",        # sentence endings
            " ",         # word boundaries
            "",          # last resort
        ],
    )

    log.info(
        f"Chunking {len(documents)} documents "
        f"(size={chunk_size}chars/{chunk_size//4}tokens, "
        f"overlap={chunk_overlap}chars/{chunk_overlap//4}tokens)..."
    )

    all_chunks, single, split_docs = [], 0, 0
    for doc in documents:
        chunks = chunk_document(doc, splitter)
        single    += 1 if len(chunks) <= 1 else 0
        split_docs += 1 if len(chunks) > 1 else 0
        all_chunks.extend(chunks)

    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"chunk_{idx:05d}"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info(f"Chunking complete:")
    log.info(f"  Input:        {len(documents)} documents")
    log.info(f"  Single-chunk: {single}")
    log.info(f"  Split:        {split_docs}")
    log.info(f"  Total chunks: {len(all_chunks)}")
    log.info(f"  Saved → {output_path}")
    return all_chunks


# ─────────────────────────────────────────────
# Inspection — Milestone 3 required
# ─────────────────────────────────────────────

def print_5_chunks(chunks: list[dict]) -> None:
    """
    Print 5 representative chunks for manual inspection.

    Per Milestone 3: For each chunk ask:
      - Does this make sense on its own?
      - Could someone answer a question from this chunk alone?
      - Is it self-contained without needing context from before/after?

    GOOD chunk:  complete opinion, clearly attributed to one professor
    BAD (small): fragment like "Professor Barry exams are heavily"
    BAD (HTML):  <div class="review">Professor Barry&#39;s exams
    BAD (large): 600 words covering 3 professors and unrelated topics
    """
    if not chunks:
        print("No chunks to inspect.")
        return

    print("\n" + "="*60)
    print("5 CHUNKS — MANUAL INSPECTION (Milestone 3)")
    print("For each chunk: is it self-contained and answerable?")
    print("="*60)

    indices = []
    indices.append(0)
    indices.append(len(chunks) - 1)
    indices.append(len(chunks) // 2)
    if len(chunks) > 5:
        rng = random.Random(42)
        indices += rng.sample(range(1, len(chunks) - 1), min(2, len(chunks) - 2))
    indices = list(dict.fromkeys(indices))[:5]

    for pos, i in enumerate(indices, 1):
        c = chunks[i]
        words = len(c.get("text", "").split())
        print(f"\n── Chunk {pos}/5  (id: {c.get('chunk_id')}) ──")
        print(f"  Professor:  {c.get('professor_name', 'N/A')}")
        print(f"  Department: {c.get('department', 'N/A')}")
        print(f"  Course:     {c.get('course', 'N/A')}")
        print(f"  Source:     {c.get('source', 'N/A')}")
        print(f"  Date:       {c.get('review_date', 'N/A')}")
        print(f"  Words:      {words}")
        print(f"\n  TEXT:")
        print(f"  {c.get('text', '')}")
        print(f"\n  → Self-contained? YES / NO  ← decide manually")
        print(f"  {'-'*50}")


def diagnose_bad_chunks(chunks: list[dict]) -> None:
    """Check for the four bad chunk types from Milestone 3."""
    import re
    print("\n=== Chunk Diagnosis ===")

    empty = [c for c in chunks if not c.get("text","").strip()]
    print(f"Empty chunks:        {len(empty)}"
          + (" ← add len > 0 filter" if empty else " ✓"))

    html = [c for c in chunks if re.search(r"<[^>]+>|&\w+;", c.get("text",""))]
    print(f"HTML artifacts:      {len(html)}"
          + (" ← cleaning missed some tags" if html else " ✓"))
    if html:
        print(f"  Sample: {html[0]['text'][:100]}")

    fragments = [c for c in chunks if 0 < len(c.get("text","").split()) < 10]
    print(f"Fragments (<10 wds): {len(fragments)}"
          + (" ← chunk size too small" if fragments else " ✓"))

    large = [c for c in chunks if len(c.get("text","").split()) > 300]
    print(f"Very large (>300):   {len(large)}"
          + (" ← consider smaller chunk_size" if large else " ✓"))

    no_source = [c for c in chunks if not c.get("source")]
    print(f"Missing source:      {len(no_source)}"
          + (" ← check metadata attachment" if no_source else " ✓"))

    no_prefix = [c for c in chunks
                 if c.get("professor_name") and not c.get("text","").startswith("[")]
    print(f"Missing prefix:      {len(no_prefix)}"
          + (" ← prefix re-attachment failed" if no_prefix else " ✓"))


# ─────────────────────────────────────────────
# Verification — Milestone 3 checkpoint
# ─────────────────────────────────────────────

def verify_chunks(chunks: list[dict]) -> bool:
    """
    Run spec assertions. Must pass before embedding.

    Target: 50–2000 chunks
    Under 50:  chunks too large, queries won't match precisely
    Over 2000: chunks too small, embeddings carry too little signal
    """
    print("\n=== Chunk Verification (Milestone 3) ===")
    passed = True
    n = len(chunks)

    if n < 50:
        print(f"[WARN] Only {n} chunks — add more source documents")
    elif n > 2000:
        print(f"[WARN] {n} chunks — may be too granular")
    else:
        print(f"[PASS] {n} chunks — within target range (50–2000)")

    empty = [c for c in chunks if not c.get("text","").strip()]
    if empty:
        print(f"[FAIL] {len(empty)} empty chunks")
        passed = False
    else:
        print("[PASS] No empty chunks")

    too_long = [c for c in chunks if len(c.get("text","").split()) > 220]
    if too_long:
        print(f"[WARN] {len(too_long)} chunks over ~220 words")
    else:
        print("[PASS] All chunks within 200-token target")

    no_prefix = [c for c in chunks
                 if c.get("professor_name") and not c.get("text","").startswith("[")]
    if no_prefix:
        print(f"[WARN] {len(no_prefix)} professor-tagged chunks missing prefix")
    else:
        print("[PASS] All professor-tagged chunks have prefix")

    no_id = [c for c in chunks if not c.get("chunk_id")]
    if no_id:
        print(f"[FAIL] {len(no_id)} chunks missing chunk_id")
        passed = False
    else:
        print("[PASS] All chunks have chunk_id")

    src_counts = defaultdict(int)
    for c in chunks:
        src_counts[c.get("source") or "Unknown"] += 1
    print(f"\nChunks by source:")
    for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<35} {count}")

    print(f"\n{'[PASS]' if passed else '[FAIL]'} Chunk verification — {n} total chunks")
    return passed


def load_documents(path: str = "data/raw/all_documents.jsonl") -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    log.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def load_chunks(path: str = "data/chunks/chunks.jsonl") -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    for input_path in [
        "data/raw/cleaned_documents.jsonl",
        "data/raw/all_documents.jsonl",
    ]:
        if Path(input_path).exists():
            break
    else:
        print("No input found. Run ingest.py first.")
        sys.exit(1)

    print(f"Loading from: {input_path}")
    docs   = load_documents(input_path)
    chunks = chunk_documents(docs)

    # ── Milestone 3 required steps ──
    print_5_chunks(chunks)
    diagnose_bad_chunks(chunks)
    verify_chunks(chunks)

    print(f"\n{'='*55}")
    print(f"NEXT STEP: Inspect the 5 chunks above carefully.")
    print(f"Each should be readable and self-contained.")
    print(f"If you see fragments, HTML, or empty strings — fix before embedding.")
    print(f"When chunks look good → run: python embed.py")
