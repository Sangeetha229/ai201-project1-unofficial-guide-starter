"""
ingest.py — Stage 1: Document Ingestion
ai201-project1-unofficial-guide-starter

Loads all 10 sources from plain .txt files in documents/.
No scraping required to run — everything works out of the box.
Live scraping functions included for when you want real data.

Sources:
  1.  Rate My Professors    → documents/rmp_reviews.txt
  2.  Reddit r/aggies       → documents/reddit_reviews.txt
  3.  Niche                 → documents/niche_reviews.txt
  4.  Coursicle             → documents/coursicle_reviews.txt
  5.  Discord               → documents/discord_export.txt
  6.  Uloop                 → documents/uloop_reviews.txt
  7.  Professors.directory  → documents/professors_directory.txt
  8.  TAMU SPH Catalog      → documents/sph_catalog.txt
  9.  TAMU Pre-Med Society  → documents/premed_society.txt
  10. TAMU OPSA             → documents/opsa.txt

Output: data/raw/<source>.jsonl  (one per source)
        data/raw/all_documents.jsonl  (combined)

HOW TO ADD YOUR OWN DATA:
  Open any .txt file in documents/ and add more reviews.
  Separate each review block with [REVIEW_END] on its own line.
  Use metadata lines: Professor: Name, Department: X, Course: Y, Rating: Z, Date: YYYY-MM-DD
  Then run: python ingest.py
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

RAW_DIR    = Path("data/raw")
SAMPLE_DIR = Path("documents")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def save_jsonl(records: list[dict], filename: str) -> None:
    path = RAW_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(records)} documents → {path}")


def make_doc(
    text: str,
    professor_name: str = None,
    department: str = None,
    course: str = None,
    source: str = None,
    rating: float = None,
    review_date: str = None,
    url: str = None,
    **kwargs,
) -> dict:
    doc = {
        "text": text,
        "professor_name": professor_name,
        "department": department,
        "course": course,
        "source": source,
        "rating": rating,
        "review_date": review_date,
        "url": url,
    }
    doc.update(kwargs)
    return doc


def parse_txt_file(path: Path, default_source: str = None) -> list[dict]:
    """
    Parse a plain .txt file into document dicts.

    Expected format (each block separated by [REVIEW_END]):
      Professor: Name
      Department: Public Health
      Course: HLTH 320
      Rating: 4.5
      Date: 2025-01-15
      Review: Full review text here...
      [REVIEW_END]

    Metadata fields are optional — text can appear without them.
    """
    if not path.exists():
        log.warning(f"File not found: {path}")
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    blocks  = content.split("[REVIEW_END]")
    documents = []

    for block in blocks:
        block = block.strip()
        if not block or len(block.split()) < 10:
            continue

        # Skip header-only blocks (first block with source info)
        lines_count = len([l for l in block.split("\n") if l.strip()])
        if lines_count <= 4 and ("Source:" in block or "Collected:" in block):
            continue

        lines     = block.strip().split("\n")
        meta      = {}
        text_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Professor:"):
                meta["professor_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Department:"):
                meta["department"] = line.split(":", 1)[1].strip()
            elif line.startswith("Course:"):
                meta["course"] = line.split(":", 1)[1].strip()
            elif line.startswith("Rating:"):
                try:
                    meta["rating"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Date:"):
                meta["review_date"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                meta["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Subreddit:"):
                meta["subreddit"] = line.split(":", 1)[1].strip()
            elif line.startswith("Thread:"):
                meta["thread_title"] = line.split(":", 1)[1].strip()
            elif line.startswith("Channel:"):
                meta["channel"] = line.split(":", 1)[1].strip()
            elif line.startswith("URL:"):
                meta["url"] = line.split(":", 1)[1].strip()
            elif line.startswith(("Review:", "Comment:", "Content:", "Message:")):
                text_lines.append(line.split(":", 1)[1].strip())
            elif line.startswith(("Faculty:", "Credits:", "Collected:")):
                text_lines.append(line)
            else:
                text_lines.append(line)

        text = " ".join(text_lines).strip()
        if not text or len(text.split()) < 10:
            continue

        if not meta.get("source") and default_source:
            meta["source"] = default_source

        documents.append(make_doc(text=text, **meta))

    return documents


# ─────────────────────────────────────────────
# All 10 sources
# ─────────────────────────────────────────────

def load_rmp() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "rmp_reviews.txt",
                          default_source="Rate My Professors")
    if not docs:
        log.warning("RMP: sample file empty — add reviews to documents/rmp_reviews.txt")
    log.info(f"Rate My Professors: {len(docs)} documents")
    save_jsonl(docs, "rmp_reviews.jsonl")
    return docs


def load_reddit() -> list[dict]:
    # Try live PRAW first if credentials set
    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        live = _scrape_reddit_live(client_id, client_secret)
        if live:
            log.info(f"Reddit (live): {len(live)} documents")
            save_jsonl(live, "reddit_reviews.jsonl")
            return live

    docs = parse_txt_file(SAMPLE_DIR / "reddit_reviews.txt",
                          default_source="Reddit")
    log.info(f"Reddit: {len(docs)} documents")
    save_jsonl(docs, "reddit_reviews.jsonl")
    return docs


def _scrape_reddit_live(client_id: str, client_secret: str) -> list[dict]:
    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="ai201-project1/1.0",
        )
        docs = []
        for sub_name in ["aggies", "premed"]:
            sub = reddit.subreddit(sub_name)
            for term in ["professor TAMU pre-med", "BIMS 301 professor",
                         "public health professor TAMU", "CHEM 227 professor TAMU"]:
                for post in sub.search(term, limit=50, sort="relevance"):
                    post.comments.replace_more(limit=0)
                    for c in post.comments.list():
                        if c.body in ("[deleted]", "[removed]"): continue
                        if len(c.body.split()) < 15: continue
                        docs.append(make_doc(
                            text=c.body, source="Reddit",
                            thread_title=post.title, subreddit=sub_name,
                        ))
                time.sleep(1)
        seen, unique = set(), []
        for d in docs:
            k = d["text"][:80]
            if k not in seen:
                seen.add(k)
                unique.append(d)
        return unique
    except Exception as e:
        log.warning(f"Reddit live scrape failed: {e}")
        return []


def load_niche() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "niche_reviews.txt", default_source="Niche")
    log.info(f"Niche: {len(docs)} documents")
    save_jsonl(docs, "niche_reviews.jsonl")
    return docs


def load_coursicle() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "coursicle_reviews.txt", default_source="Coursicle")
    log.info(f"Coursicle: {len(docs)} documents")
    save_jsonl(docs, "coursicle_reviews.jsonl")
    return docs


def load_discord() -> list[dict]:
    # Try JSON exports first (DiscordChatExporter format)
    discord_dir = Path("data/raw/discord")
    if discord_dir.exists():
        json_files = list(discord_dir.glob("*.json"))
        if json_files:
            docs = _load_discord_json(json_files)
            if docs:
                log.info(f"Discord (JSON): {len(docs)} documents")
                save_jsonl(docs, "discord_reviews.jsonl")
                return docs

    docs = parse_txt_file(SAMPLE_DIR / "discord_export.txt", default_source="Discord")
    log.info(f"Discord: {len(docs)} documents")
    save_jsonl(docs, "discord_reviews.jsonl")
    return docs


def _load_discord_json(json_files: list[Path]) -> list[dict]:
    docs = []
    for path in json_files:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            channel  = data.get("channel", {}).get("name", "unknown")
            messages = data.get("messages", [])
            group, group_time = [], None
            for msg in messages:
                content = msg.get("content", "").strip()
                if not content:
                    continue
                try:
                    ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                    if group_time and abs((ts - group_time).total_seconds()) > 300:
                        text = " ".join(group)
                        if len(text.split()) >= 15:
                            docs.append(make_doc(text=text, source="Discord", channel=channel))
                        group = []
                    group_time = ts
                except Exception:
                    pass
                group.append(content)
            if group:
                text = " ".join(group)
                if len(text.split()) >= 15:
                    docs.append(make_doc(text=text, source="Discord", channel=channel))
        except Exception as e:
            log.warning(f"Discord JSON {path} failed: {e}")
    return docs


def load_uloop() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "uloop_reviews.txt", default_source="Uloop")
    log.info(f"Uloop: {len(docs)} documents")
    save_jsonl(docs, "uloop_reviews.jsonl")
    return docs


def load_professors_directory() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "professors_directory.txt",
                          default_source="Professors.directory")
    log.info(f"Professors.directory: {len(docs)} documents")
    save_jsonl(docs, "professors_directory.jsonl")
    return docs


def load_sph_catalog() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "sph_catalog.txt",
                          default_source="TAMU SPH Catalog")
    log.info(f"TAMU SPH Catalog: {len(docs)} documents")
    save_jsonl(docs, "sph_catalog.jsonl")
    return docs


def load_premed_society() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "premed_society.txt",
                          default_source="TAMU Pre-Medical Society")
    log.info(f"TAMU Pre-Medical Society: {len(docs)} documents")
    save_jsonl(docs, "premed_society.jsonl")
    return docs


def load_opsa() -> list[dict]:
    docs = parse_txt_file(SAMPLE_DIR / "opsa.txt", default_source="TAMU OPSA")
    log.info(f"TAMU OPSA: {len(docs)} documents")
    save_jsonl(docs, "opsa.jsonl")
    return docs


# ─────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────

def run_all_ingestion() -> list[dict]:
    """
    Load all 10 sources and save combined all_documents.jsonl.
    Each source prints a document count.
    Falls back to sample data if live scraping is blocked.
    """
    log.info("=== ai201-project1 — Ingesting 10 sources ===")
    all_docs = []

    sources = [
        ("1.  Rate My Professors",       load_rmp),
        ("2.  Reddit",                   load_reddit),
        ("3.  Niche",                    load_niche),
        ("4.  Coursicle",                load_coursicle),
        ("5.  Discord",                  load_discord),
        ("6.  Uloop",                    load_uloop),
        ("7.  Professors.directory",     load_professors_directory),
        ("8.  TAMU SPH Catalog",         load_sph_catalog),
        ("9.  TAMU Pre-Med Society",     load_premed_society),
        ("10. TAMU OPSA",                load_opsa),
    ]

    for label, fn in sources:
        try:
            docs = fn()
            all_docs.extend(docs)
        except Exception as e:
            log.error(f"{label} failed: {e}")

    save_jsonl(all_docs, "all_documents.jsonl")
    log.info(f"=== Ingestion complete: {len(all_docs)} total documents ===")
    return all_docs


def print_sample_document(documents: list[dict], index: int = 0) -> None:
    """Print one raw document — use this to verify before cleaning."""
    if not documents:
        print("No documents loaded.")
        return
    doc = documents[min(index, len(documents) - 1)]
    print(f"\n{'='*55}")
    print(f"RAW DOCUMENT (index {index}) — read before cleaning")
    print(f"{'='*55}")
    print(f"Source:     {doc.get('source')}")
    print(f"Professor:  {doc.get('professor_name')}")
    print(f"Department: {doc.get('department')}")
    print(f"Course:     {doc.get('course')}")
    print(f"Rating:     {doc.get('rating')}")
    print(f"Date:       {doc.get('review_date')}")
    print(f"Word count: {len(doc.get('text','').split())}")
    print(f"\nText:\n{doc.get('text','')}")
    print(f"{'='*55}")


if __name__ == "__main__":
    docs = run_all_ingestion()

    print(f"\n{'='*55}")
    print(f"Total documents: {len(docs)}")
    print(f"\nBy source:")
    for source, count in sorted(
        Counter(d.get("source") for d in docs).items(),
        key=lambda x: -x[1]
    ):
        print(f"  {source:<35} {count}")

    print(f"\nChecking data/raw/ files:")
    for f in [
        "rmp_reviews.jsonl", "reddit_reviews.jsonl", "niche_reviews.jsonl",
        "coursicle_reviews.jsonl", "discord_reviews.jsonl", "uloop_reviews.jsonl",
        "professors_directory.jsonl", "sph_catalog.jsonl",
        "premed_society.jsonl", "opsa.jsonl", "all_documents.jsonl",
    ]:
        path = RAW_DIR / f
        if path.exists():
            n = sum(1 for _ in open(path))
            print(f"  ✓  {f} ({n} records)")
        else:
            print(f"  ✗  {f} MISSING")

    print(f"\nSample document (index 0):")
    print_sample_document(docs, 0)
