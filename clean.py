"""
clean.py — Stage 2: Text Cleaning
ai201-project1-unofficial-guide-starter

Cleans raw documents before chunking.

REMOVES: HTML tags, HTML entities (&amp; &nbsp;), boilerplate,
         short reviews under 20 words, deleted Reddit comments
KEEPS:   Review text, professor names, course numbers, opinions,
         ratings, and all context needed to understand the content

Per Milestone 3: After cleaning, print one document and read it.
If you still see HTML, nav text, or boilerplate — clean further.

Run standalone: python clean.py
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

log = logging.getLogger(__name__)

MIN_WORDS = 20

PROFESSOR_NAME_MAP = {
    "barry":            "Adam Barry",
    "adam barry":       "Adam Barry",
    "prof barry":       "Adam Barry",
    "dr barry":         "Adam Barry",
    "dr. barry":        "Adam Barry",
    "deleon":           "Joy DeLeon",
    "de leon":          "Joy DeLeon",
    "joy deleon":       "Joy DeLeon",
    "dr deleon":        "Joy DeLeon",
    "clendenin":        "Angela Clendenin",
    "angela clendenin": "Angela Clendenin",
    "dr clendenin":     "Angela Clendenin",
    "goodey":           "Joanna Goodey",
    "joanna goodey":    "Joanna Goodey",
    "dr goodey":        "Joanna Goodey",
    "herman":           "Jim Herman",
    "jim herman":       "Jim Herman",
    "dr herman":        "Jim Herman",
    "colwell":          "Brian Colwell",
    "brian colwell":    "Brian Colwell",
    "towne":            "Samuel Towne",
    "cubbin":           "Catherine Cubbin",
}

DEPARTMENT_KEYWORDS = {
    "Public Health": [
        "public health", "hlth", "sph", "school of public health",
        "health behavior", "health policy", "health promotion",
        "health communication",
    ],
    "Epidemiology & Biostatistics": [
        "epidemiology", "epib", "biostatistics",
    ],
    "Biology":  ["biology", "biol", "microbiology", "cell biology"],
    "Chemistry":["chemistry", "chem", "organic chemistry", "orgo", "biochem"],
    "BIMS":     ["bims", "biomedical sciences", "pre-med", "premed", "pre med"],
    "Pre-Med":  ["pre-med", "premed", "opsa", "mcat", "medical school"],
}

HTML_ENTITIES = {
    "&amp;": "&",  "&lt;": "<",   "&gt;": ">",   "&quot;": '"',
    "&#39;": "'",  "&apos;": "'", "&nbsp;": " ",  "&mdash;": "—",
    "&ndash;": "–","&hellip;": "...","&rsquo;": "'","&lsquo;": "'",
    "&rdquo;": '"', "&ldquo;": '"',
}

BOILERPLATE = [
    r"read more\.?", r"see more\.?", r"share this\.?",
    r"was this helpful\?", r"helpful \d+", r"not helpful \d+",
    r"report\b", r"sign in to.*", r"log in to.*",
    r"cookie.*policy", r"privacy policy",
    r"copyright \d{4}", r"all rights reserved",
]


# ─────────────────────────────────────────────
# Core text cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw text.

    REMOVES: HTML tags, entities, non-printable chars, boilerplate
    KEEPS:   Review content, opinions, names, course numbers
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace HTML entities
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse newlines
    text = re.sub(r"\n+", " ", text)
    # Remove boilerplate phrases
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_useful(text: str, min_words: int = MIN_WORDS) -> bool:
    return bool(text) and len(text.split()) >= min_words


def normalize_professor_name(name: str) -> str:
    if not name:
        return name
    key = re.sub(r"\b(dr\.?|prof\.?|professor|mr\.?|ms\.?)\s*", "",
                 name.lower().strip())
    key = re.sub(r"\s+", " ", key).strip()
    return PROFESSOR_NAME_MAP.get(key, name.strip())


def detect_professor_in_text(text: str) -> str | None:
    lower = text.lower()
    for variant, canonical in PROFESSOR_NAME_MAP.items():
        if variant in lower:
            return canonical
    return None


def detect_department_in_text(text: str) -> str | None:
    lower = text.lower()
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return dept
    return None


def flag_stale(review_date: str | None, max_years: int = 2) -> bool:
    if not review_date:
        return False
    try:
        date = datetime.fromisoformat(review_date[:10])
        return (datetime.now() - date).days > (max_years * 365)
    except Exception:
        return False


# ─────────────────────────────────────────────
# Per-document cleaning
# ─────────────────────────────────────────────

def clean_document(doc: dict) -> dict | None:
    """
    Clean one document. Returns None if it should be discarded.

    1. Clean text
    2. Discard if under MIN_WORDS
    3. Skip deleted Reddit comments
    4. Normalize professor name
    5. Fill missing department/professor from text
    6. Flag stale reviews
    """
    # Skip deleted Reddit comments
    if doc.get("text", "") in ("[deleted]", "[removed]"):
        return None

    text = clean_text(doc.get("text", ""))
    if not is_useful(text):
        return None

    doc = doc.copy()
    doc["text"] = text

    if doc.get("professor_name"):
        doc["professor_name"] = normalize_professor_name(doc["professor_name"])
    else:
        doc["professor_name"] = detect_professor_in_text(text)

    if not doc.get("department"):
        doc["department"] = detect_department_in_text(text)

    doc["is_stale"] = flag_stale(doc.get("review_date"))
    return doc


def deduplicate(documents: list[dict], chars: int = 80) -> list[dict]:
    seen, unique = set(), []
    for doc in documents:
        key = doc.get("text", "")[:chars].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def clean_all_documents(documents: list[dict]) -> list[dict]:
    log.info(f"Cleaning {len(documents)} documents...")
    cleaned, discarded = [], 0
    for doc in documents:
        result = clean_document(doc)
        if result is None:
            discarded += 1
        else:
            cleaned.append(result)
    cleaned = deduplicate(cleaned)

    stale = sum(1 for d in cleaned if d.get("is_stale"))
    log.info(f"  Input {len(documents)} → Output {len(cleaned)} "
             f"(discarded {discarded}, stale {stale})")

    dept_counts = defaultdict(int)
    for d in cleaned:
        dept_counts[d.get("department") or "Unknown"] += 1
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {dept}: {count}")

    return cleaned


# ─────────────────────────────────────────────
# Inspection helpers — Milestone 3
# ─────────────────────────────────────────────

def print_before_after(raw_doc: dict) -> None:
    """
    Print one document before and after cleaning.
    Per Milestone 3: read it. If you see HTML, nav text,
    or &amp; entities — your cleaning is incomplete.
    """
    cleaned = clean_document(raw_doc.copy())
    print(f"\n{'='*55}")
    print("BEFORE CLEANING:")
    print(f"  {raw_doc.get('text','')[:300]}")
    print("\nAFTER CLEANING:")
    if cleaned:
        print(f"  {cleaned.get('text','')[:300]}")
        print(f"  Professor:  {cleaned.get('professor_name')}")
        print(f"  Department: {cleaned.get('department')}")
        print(f"  Stale:      {cleaned.get('is_stale')}")
    else:
        print("  DISCARDED (too short or empty)")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# Verification — Milestone 3 checkpoint
# ─────────────────────────────────────────────

def verify_cleaning(documents: list[dict]) -> bool:
    """
    Run spec assertions. Must pass before chunking.
    Per Milestone 3: fix any failures before moving to chunk.py
    """
    print("\n=== Cleaning Verification (Milestone 3) ===")
    passed = True

    empty = [d for d in documents if not d.get("text", "").strip()]
    if empty:
        print(f"[FAIL] {len(empty)} documents with empty text")
        passed = False
    else:
        print("[PASS] No empty text")

    short = [d for d in documents if len(d.get("text","").split()) < MIN_WORDS]
    if short:
        print(f"[FAIL] {len(short)} reviews under {MIN_WORDS} words")
        passed = False
    else:
        print(f"[PASS] All reviews >= {MIN_WORDS} words")

    html = [d for d in documents if re.search(r"<[^>]+>", d.get("text",""))]
    if html:
        print(f"[FAIL] {len(html)} documents still contain HTML tags")
        print(f"       Sample: {html[0]['text'][:100]}")
        passed = False
    else:
        print("[PASS] No HTML tags")

    entities = [d for d in documents if re.search(r"&\w+;|&#\d+;", d.get("text",""))]
    if entities:
        print(f"[FAIL] {len(entities)} documents contain HTML entities (&amp; etc.)")
        passed = False
    else:
        print("[PASS] No HTML entities")

    sources = set(d.get("source") for d in documents)
    print(f"\n[INFO] {len(sources)} sources present in cleaned corpus")
    for s in sorted(sources):
        count = sum(1 for d in documents if d.get("source") == s)
        print(f"       {s}: {count}")

    print(f"\n{'[PASS]' if passed else '[FAIL]'} "
          f"Cleaning check — {len(documents)} documents ready")
    return passed


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    input_path = Path("data/raw/all_documents.jsonl")
    if not input_path.exists():
        print(f"Not found: {input_path}\nRun ingest.py first.")
        sys.exit(1)

    raw_docs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(raw_docs)} raw documents")

    # Milestone 3: print one document before and after cleaning
    print("\n--- Before / After Cleaning (doc 0) ---")
    print_before_after(raw_docs[0])

    cleaned = clean_all_documents(raw_docs)
    verify_cleaning(cleaned)

    out_path = Path("data/raw/cleaned_documents.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in cleaned:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(cleaned)} clean documents → {out_path}")
