"""
generate.py — Stage 6: Generation
ai201-project1-unofficial-guide-starter

LLM: Groq llama-3.3-70b-versatile (free tier)
Get a free key at: https://console.groq.com

Grounding is enforced — the system prompt instructs the model
to answer ONLY from retrieved chunks, never from training knowledge.

Per Milestone 5:
  - Every answer must be traceable to retrieved chunks
  - When documents don't cover a question, the system says so
  - Source attribution is explicit in every response

Run: python generate.py "your question"
Eval: python generate.py --eval
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY from .env file
import logging

log = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000

# ── Grounding system prompt ──
# Per Milestone 5: must explicitly instruct the model to answer ONLY
# from provided documents — not from general training knowledge.
SYSTEM_PROMPT = """You are the Unofficial Guide assistant, a RAG system that answers
questions about Texas A&M pre-med and public health professors using ONLY
the student review documents provided to you.

STRICT GROUNDING RULES — follow every one:
1. Answer ONLY using information from the numbered source chunks below.
   Do NOT use your general training knowledge about TAMU, professors, or courses.
2. If the provided chunks do not contain enough information to answer the question,
   respond with: "The available student reviews don't have enough information to
   answer this question. Try asking about a specific professor or course."
3. Cite every factual claim with its source number in brackets: [1], [2], etc.
4. If a source is marked POSSIBLY OUTDATED, note this in your answer.
5. Never invent professor names, ratings, course numbers, or student opinions
   that are not present in the source chunks.
6. Write like a senior student giving honest, direct advice to a freshman.
7. End every answer with a Sources section listing each cited source.

GROUNDING TEST: Before writing your answer, ask yourself:
"Can I point to a specific chunk that supports every claim I'm making?"
If the answer is no for any claim — remove that claim.

FORMAT:
[2-4 sentence grounded answer with inline citations like [1] and [2]]

Sources:
[1] Professor Name — Platform (date)
[2] Professor Name — Platform (date)
"""


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered context blocks."""
    if not chunks:
        return (
            f"Student question: {query}\n\n"
            "No relevant source chunks were retrieved for this query."
        )
    lines = [f"Student question: {query}\n", "Source chunks:"]
    for i, c in enumerate(chunks, 1):
        parts = []
        if c.get("professor_name"):
            parts.append(f"Professor: {c['professor_name']}")
        if c.get("department"):
            parts.append(f"Dept: {c['department']}")
        if c.get("course"):
            parts.append(f"Course: {c['course']}")
        if c.get("source"):
            parts.append(f"Source: {c['source']}")
        if c.get("review_date"):
            parts.append(f"Date: {c['review_date']}")
        header = " | ".join(parts) if parts else "Source unknown"
        stale  = " ⚠ POSSIBLY OUTDATED (>2 years old)" if c.get("is_stale") else ""
        lines.append(f"\n[{i}] {header}{stale}")
        lines.append(c["text"])
    return "\n".join(lines)


def call_groq(user_message: str) -> str:
    """
    Call Groq API with llama-3.3-70b-versatile.

    Groq runs LLaMA on dedicated LPU hardware — very fast (<1 second),
    generous free tier, follows system prompt grounding rules reliably.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "Groq library not installed.\n"
            "Run: pip install groq"
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at: https://console.groq.com\n"
            "Then set it:\n"
            "  Windows:   $env:GROQ_API_KEY='your_key_here'\n"
            "  Mac/Linux: export GROQ_API_KEY=your_key_here"
        )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded cited answer from retrieved chunks.

    Returns dict with answer, sources, model info, and stale flag.
    """
    log.info(f"Generating answer ({GROQ_MODEL}) for: '{query[:60]}'")
    prompt = build_user_prompt(query, chunks)
    answer = call_groq(prompt)
    return {
        "answer":      answer,
        "query":       query,
        "model":       GROQ_MODEL,
        "num_sources": len(chunks),
        "has_stale":   any(c.get("is_stale") for c in chunks),
        "sources": [
            f"{c.get('professor_name', 'Unknown')} — "
            f"{c.get('source', 'Unknown')} "
            f"({c.get('review_date', 'no date')})"
            for c in chunks
        ],
        "source_details": [{
            "rank":           i + 1,
            "professor_name": c.get("professor_name"),
            "department":     c.get("department"),
            "source":         c.get("source"),
            "review_date":    c.get("review_date"),
            "similarity":     c.get("similarity_score"),
            "is_stale":       c.get("is_stale", False),
        } for i, c in enumerate(chunks)],
    }


def test_grounding() -> None:
    """
    Per Milestone 5: test that the system is grounded.

    Tests:
    1. A question covered by documents → grounded answer with citations
    2. A question NOT covered → system declines, does not hallucinate
    """
    from retrieve import retrieve_chunks

    print("\n=== Grounding Test (Milestone 5) ===\n")

    # Test 1: covered question
    print("Test 1 — Question covered by documents:")
    query  = "What do students say about Adam Barry teaching quality at TAMU?"
    chunks = retrieve_chunks(query)
    result = generate_answer(query, chunks)
    print(f"Query: {query}")
    print(f"Answer:\n{result['answer']}")
    print(f"\nGrounding check: Does every claim trace to a retrieved chunk?")
    print(f"Retrieved {result['num_sources']} chunks from: "
          f"{[c.get('source') for c in chunks]}")

    print("\n" + "─"*55)

    # Test 2: question NOT in documents
    print("\nTest 2 — Question NOT covered by documents:")
    query2  = "What is the parking situation near the TAMU SPH building?"
    chunks2 = retrieve_chunks(query2)
    result2 = generate_answer(query2, chunks2)
    print(f"Query: {query2}")
    print(f"Answer:\n{result2['answer']}")
    print(f"\nExpected: system should decline, not hallucinate.")
    print(f"Retrieved {result2['num_sources']} chunks")

    print("\n=== Grounding Test Complete ===")


def run_evaluation() -> None:
    """
    Run all 5 evaluation plan questions.

    Rubric (from planning.md):
      3 = correct professor + 2 citations + source + specific detail
      2 = correct topic but missing citations or too vague
      1 = wrong professor, fabricated info, or no reviews cited

    Target: 13-15 / 15
    """
    from retrieve import retrieve_chunks

    questions = [
        "What do students say about Prof Joy DeLeon exam difficulty in Public Health at TAMU?",
        "Do students recommend taking BIMS 301 with a specific professor at TAMU and why?",
        "What do TAMU pre-med students on Reddit say about which chemistry professor for Orgo 1?",
        "How do students rate Prof Adam Barry teaching quality in TAMU School of Public Health?",
        "What do students say about workload and grading in TAMU Epidemiology and Biostatistics?",
    ]

    print(f"\n=== Evaluation — 5 Test Questions ===")
    print(f"Model: {GROQ_MODEL}\n")
    total = 0

    for i, query in enumerate(questions, 1):
        print(f"{'='*60}")
        print(f"Q{i}: {query}")
        print(f"{'='*60}")
        try:
            chunks = retrieve_chunks(query)
            result = generate_answer(query, chunks)
            print(f"\n{result['answer']}")
            print(f"\n[{result['num_sources']} sources | model: {result['model']}]")
            if result["has_stale"]:
                print("⚠ One or more sources may be outdated (>2 years)")
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"\nSCORING RUBRIC:")
        print(f"  3 = Correct professor + 2+ citations + source + specific detail")
        print(f"  2 = Correct topic but vague or missing citations")
        print(f"  1 = Wrong professor, fabricated info, no reviews cited")
        score_input = input(f"\nYour score for Q{i} [1/2/3]: ").strip()
        score = int(score_input) if score_input in ("1","2","3") else 2
        total += score
        print(f"Recorded: {score}/3")

    print(f"\n{'='*60}")
    print(f"FINAL SCORE: {total}/15  (target: 13-15)")
    if total >= 13:
        print("✓ Production-ready — ready to submit")
    elif total >= 10:
        print("~ Partial — tighten system prompt or improve retrieval")
    else:
        print("✗ Below threshold — check chunking and embedding stages")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if "--eval" in sys.argv:
        run_evaluation()
    elif "--grounding-test" in sys.argv:
        test_grounding()
    elif len(sys.argv) > 1:
        from retrieve import retrieve_chunks
        query  = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
        chunks = retrieve_chunks(query)
        result = generate_answer(query, chunks)
        print(f"\n{result['answer']}")
        print(f"\n[model: {result['model']} | {result['num_sources']} sources]")
        if result["has_stale"]:
            print("\n⚠ One or more reviews may be outdated")
    else:
        print("Usage:")
        print("  python generate.py 'your question'")
        print("  python generate.py --eval")
        print("  python generate.py --grounding-test")
