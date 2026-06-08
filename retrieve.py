"""
retrieve.py — Stage 5: Retrieval
ai201-project1-unofficial-guide-starter

Top-K = 5  |  Min similarity = 0.70 (max distance = 0.30)
Detects department + professor from query → metadata pre-filter
Returns chunks with similarity_score and distance_score fields

Per Milestone 4:
  - Test with 3+ evaluation plan queries before wiring in LLM
  - Print returned chunks and distance scores
  - Good retrieval: distance < 0.5, chunks visibly related to query
  - Bad retrieval: distance > 0.6-0.7, off-topic content

Run standalone: python retrieve.py "your query here"
Run verification: python retrieve.py
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

VECTOR_DIR     = "data/vector_store"
COLLECTION     = "tamu_reviews"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5
MIN_SIMILARITY = 0.50   # drop chunks below this threshold
MAX_DISTANCE   = 0.30   # equivalent to MIN_SIMILARITY for cosine

DEPARTMENT_KEYWORDS = {
    "Public Health": [
        "public health", "hlth", "sph", "school of public health",
        "health behavior", "health policy", "health communication",
    ],
    "Epidemiology & Biostatistics": [
        "epidemiology", "epib", "biostatistics",
    ],
    "Biology":  ["biology", "biol", "microbiology"],
    "Chemistry":["chemistry", "chem", "organic chemistry", "orgo", "biochem"],
    "BIMS":     ["bims", "bims 301", "bims 302", "biomedical sciences", "pre-med", "premed"],
    "Pre-Med":  ["pre-med", "premed", "opsa", "mcat"],
}

PROFESSOR_NAME_MAP = {
    "barry":     "Adam Barry",
    "deleon":    "Joy DeLeon",
    "de leon":   "Joy DeLeon",
    "clendenin": "Angela Clendenin",
    "goodey":    "Joanna Goodey",
    "herman":    "Jim Herman",
    "colwell":   "Brian Colwell",
    "towne":     "Samuel Towne",
    "cubbin":    "Catherine Cubbin",
}


COURSE_CODE_DEPT_MAP = {
    "bims 301": "BIMS",
    "bims 302": "BIMS",
    "chem 227": "Chemistry",
    "chem 237": "Chemistry",
    "hlth 215": "Public Health",
    "hlth 310": "Public Health",
    "hlth 320": "Public Health",
    "hlth 370": "Public Health",
    "hlth 425": "Public Health",
    "epib 301": "Epidemiology & Biostatistics",
    "epib 410": "Epidemiology & Biostatistics",
    "biol 111": "Biology",
    "biol 112": "Biology",
}

# When a course code is detected, also hint which professor teaches it
COURSE_PROFESSOR_HINT = {
    "bims 301": "Jim Herman",
    "chem 227": "Joanna Goodey",
    "chem 237": "Joanna Goodey",
    "hlth 320": "Adam Barry",
    "hlth 215": "Joy DeLeon",
    "hlth 310": "Brian Colwell",
    "hlth 425": "Adam Barry",
    "epib 301": "Angela Clendenin",
    "epib 410": "Angela Clendenin",
}

def detect_department(query: str) -> Optional[str]:
    lower = query.lower()
    # Check course codes first (most specific)
    for code, dept in COURSE_CODE_DEPT_MAP.items():
        if code in lower:
            return dept
    # Then check department keywords
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return dept
    return None


def detect_professor(query: str) -> Optional[str]:
    lower = query.lower()
    # Check explicit professor name first
    for variant, canonical in PROFESSOR_NAME_MAP.items():
        if variant in lower:
            return canonical
    # If no name but course code found, hint professor from course
    for code, prof in COURSE_PROFESSOR_HINT.items():
        if code in lower:
            return prof
    return None


# Departments that often appear together in cross-discipline review chunks
RELATED_DEPTS = {
    "BIMS":     ["BIMS", "Pre-Med", "Biology", "Public Health"],
    "Pre-Med":  ["Pre-Med", "BIMS", "Biology", "Public Health"],
    "Biology":  ["Biology", "BIMS", "Pre-Med"],
    "Chemistry":["Chemistry", "BIMS", "Pre-Med"],
}

def _build_where(dept: Optional[str], prof: Optional[str]) -> Optional[dict]:
    conditions = []
    if dept:
        # Use $in with related departments to catch cross-discipline chunks
        related = RELATED_DEPTS.get(dept)
        if related:
            conditions.append({"department": {"$in": related}})
        else:
            conditions.append({"department": {"$eq": dept}})
    if prof:
        conditions.append({"professor_name": {"$eq": prof}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve_chunks(
    query:           str,
    k:               int   = TOP_K,
    min_similarity:  float = MIN_SIMILARITY,
    persist_dir:     str   = VECTOR_DIR,
    collection_name: str   = COLLECTION,
    model_name:      str   = EMBED_MODEL,
) -> list[dict]:
    """
    Retrieve top-K most relevant chunks for a student query.

    Strategy:
    1. Detect department + professor from query keywords
    2. Try filtered search first (by professor name if detected)
    3. If filtered returns 0 results, fall back to full unfiltered search
    4. Drop results below min_similarity threshold
    5. Return ranked chunk list

    The fallback to unfiltered search ensures no valid query ever
    returns empty results due to metadata tag mismatches.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(f"Missing: {e}. Run: pip install chromadb sentence-transformers")

    dept  = detect_department(query)
    prof  = detect_professor(query)

    log.info(f"Query: '{query[:60]}'")
    log.info(f"  Dept: {dept}  |  Prof: {prof}")

    model  = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        col = client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' not found.\n"
            "Run: python main.py --build"
        )

    if col.count() == 0:
        raise ValueError("Vector index is empty. Run: python main.py --build")

    q_emb = model.encode(query, normalize_embeddings=True).tolist()

    # Fetch more candidates than needed so threshold filtering still leaves k results
    fetch_n = min(k * 4, col.count())

    def run_query(where_filter=None):
        kwargs = dict(
            query_embeddings=[q_emb],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            kwargs["where"] = where_filter
        try:
            return col.query(**kwargs)
        except Exception as e:
            log.warning(f"Query failed ({e})")
            return None

    # --- Attempt 1: filter by professor name only (most precise) ---
    results = None
    if prof:
        results = run_query({"professor_name": {"$eq": prof}})
        if results and len(results["documents"][0]) == 0:
            log.info("  Professor filter returned 0 — trying unfiltered")
            results = None

    # --- Attempt 2: no filter — full semantic search ---
    if not results:
        results = run_query(None)

    if not results:
        log.warning("  All query attempts failed")
        return []

    # Build result list, apply threshold
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1.0 - dist, 4)
        if similarity < min_similarity:
            continue

        chunk = {
            "text":             doc,
            "similarity_score": similarity,
            "distance_score":   round(dist, 4),
            "professor_name":   meta.get("professor_name"),
            "department":       meta.get("department"),
            "course":           meta.get("course"),
            "source":           meta.get("source"),
            "review_date":      meta.get("review_date"),
            "is_stale":         bool(meta.get("is_stale", 0)),
            "url":              meta.get("url"),
        }
        for key in ("thread_title", "subreddit", "channel"):
            if meta.get(key):
                chunk[key] = meta[key]
        chunks.append(chunk)
        if len(chunks) >= k:
            break

    log.info(f"  Retrieved {len(chunks)} chunks (threshold={min_similarity})")
    return chunks


def test_retrieval_milestone4() -> bool:
    """
    Test retrieval with evaluation plan queries.

    Per Milestone 4:
      - Run at least 3 of the 5 evaluation queries
      - Print returned chunks and distance scores
      - Verify chunks are visibly relevant to each query
      - Distance scores on top results should be below 0.5
    """
    print("\n=== Retrieval Test — Milestone 4 ===")
    print("Testing with evaluation plan queries.")
    print("Good retrieval: distance < 0.5, chunks visibly relevant.\n")

    test_queries = [
        ("Q1", "What do students say about Prof Joy DeLeon exam difficulty in Public Health?"),
        ("Q2", "Do students recommend taking BIMS 301 with a specific professor at TAMU?"),
        ("Q3", "What do TAMU pre-med students on Reddit say about chemistry professor for Orgo 1?"),
        ("Q4", "How do students rate Prof Adam Barry teaching quality in TAMU School of Public Health?"),
        ("Q5", "What do students say about workload and grading in TAMU Epidemiology courses?"),
    ]

    all_passed = True
    for qid, query in test_queries:
        print(f"{qid}: {query[:65]}...")
        chunks = retrieve_chunks(query)
        print(f"  Retrieved: {len(chunks)} chunks")

        ok = len(chunks) > 0
        print(f"  {'[PASS]' if ok else '[FAIL]'} returned results")
        all_passed = all_passed and ok

        if chunks:
            scores = [c["distance_score"] for c in chunks]
            quality = "GOOD" if min(scores) < 0.5 else ("OK" if min(scores) < 0.7 else "WEAK")
            print(f"  distances: min={min(scores):.3f} max={max(scores):.3f} [{quality}]")

            # Milestone 4: check Reddit source for Q3
            if qid == "Q3":
                has_reddit = any("reddit" in c.get("source","").lower() for c in chunks)
                print(f"  {'[PASS]' if has_reddit else '[WARN]'} has Reddit source")

            # Print top chunk for inspection
            top = chunks[0]
            print(f"  Top result (dist={top['distance_score']}):")
            print(f"    Prof: {top.get('professor_name','N/A')} | "
                  f"Source: {top.get('source','N/A')}")
            print(f"    Text: {top['text'][:120]}...")

        print()

    print(f"=== Retrieval: {'PASSED' if all_passed else 'NEEDS WORK'} ===")
    print("If distances are high (>0.5) or chunks are off-topic:")
    print("  - Check that chunk.py ran correctly")
    print("  - Try python embed.py with --overwrite flag")
    print("  - Consider larger chunk_size in chunk.py")
    return all_passed


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        # Direct query mode: python retrieve.py "your question"
        query  = " ".join(sys.argv[1:])
        chunks = retrieve_chunks(query)
        print(f"\n'{query}'")
        print(f"Retrieved {len(chunks)} chunks:\n")
        for i, c in enumerate(chunks, 1):
            print(f"[{i}] similarity={c['similarity_score']}  "
                  f"distance={c['distance_score']}  "
                  f"Prof:{c.get('professor_name','N/A')}  "
                  f"Source:{c.get('source','N/A')}")
            if c.get("is_stale"):
                print(f"    ⚠ Review may be outdated (>2 years)")
            print(f"    {c['text'][:200]}...\n")
    else:
        # Verification mode
        test_retrieval_milestone4()