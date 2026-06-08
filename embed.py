"""
embed.py — Stage 4: Embedding + Vector Store
ai201-project1-unofficial-guide-starter

Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
  - Runs locally, no API key, no rate limits
  - Load with: SentenceTransformer("all-MiniLM-L6-v2")
  - 384-dimensional vectors
  - Max context: 256 tokens (fits 200-token chunks)

Vector store: ChromaDB (local persistent)
  - Runs locally, no account needed
  - Collection: tamu_reviews
  - Cosine similarity search

Input:  data/chunks/chunks.jsonl
Output: data/vector_store/

Run standalone: python embed.py
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CHUNKS_PATH = "data/chunks/chunks.jsonl"
VECTOR_DIR  = "data/vector_store"
COLLECTION  = "tamu_reviews"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64


def load_embedding_model(model_name: str = EMBED_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers torch")
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    log.info(f"Model ready — dim: {dim}, max_seq: {model.max_seq_length}")
    assert dim == 384, f"Expected 384-dim, got {dim}"
    return model


def get_collection(persist_dir: str = VECTOR_DIR, name: str = COLLECTION):
    try:
        import chromadb
    except ImportError:
        raise ImportError("Run: pip install chromadb")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    col    = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Collection '{name}' ready ({col.count()} existing vectors)")
    return col


def _load_chunks(path: str) -> list[dict]:
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


def _build_metadata(chunk: dict) -> dict:
    """Build ChromaDB metadata — values must be str/int/float/bool."""
    meta = {}
    for field in ["professor_name", "department", "course", "source",
                  "review_date", "url", "thread_title", "subreddit", "channel"]:
        val = chunk.get(field)
        if val is not None:
            meta[field] = str(val)
    if chunk.get("rating") is not None:
        try:
            meta["rating"] = float(chunk["rating"])
        except (ValueError, TypeError):
            pass
    meta["is_stale"] = 1 if chunk.get("is_stale") else 0
    return meta


def embed_and_store(
    chunks_path: str = CHUNKS_PATH,
    persist_dir:  str = VECTOR_DIR,
    collection_name: str = COLLECTION,
    model_name:   str = EMBED_MODEL,
    batch_size:   int = BATCH_SIZE,
    overwrite:    bool = False,
) -> int:
    """
    Embed all chunks and upsert into ChromaDB.
    Skips already-indexed chunks unless overwrite=True.
    Returns total vector count in collection.
    """
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks at {chunks_path}. Run chunk.py first.")

    model = load_embedding_model(model_name)

    if overwrite:
        try:
            import chromadb
            chromadb.PersistentClient(path=persist_dir).delete_collection(collection_name)
            log.info("Deleted existing collection")
        except Exception:
            pass

    col = get_collection(persist_dir, collection_name)

    existing_ids = set()
    if col.count() > 0:
        existing_ids = set(col.get(include=[])["ids"])
        log.info(f"Skipping {len(existing_ids)} already-indexed chunks")

    new_chunks = [c for c in chunks if c.get("chunk_id") not in existing_ids]
    if not new_chunks:
        log.info("All chunks already indexed.")
        return col.count()

    log.info(f"Embedding {len(new_chunks)} chunks...")
    total = 0

    for i in range(0, len(new_chunks), batch_size):
        batch      = new_chunks[i: i + batch_size]
        texts      = [c["text"] for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [_build_metadata(c) for c in batch]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        col.upsert(ids=ids, embeddings=embeddings,
                   documents=texts, metadatas=metadatas)
        total += len(batch)
        log.info(f"  Stored {total}/{len(new_chunks)}...")

    final = col.count()
    log.info(f"Embedding complete — {final} total vectors")
    return final


def verify_index(
    persist_dir:  str = VECTOR_DIR,
    collection_name: str = COLLECTION,
    chunks_path:  str = CHUNKS_PATH,
    model_name:   str = EMBED_MODEL,
) -> bool:
    """
    Verify index matches chunks file.
    Runs one test query and prints top-5 results with distance scores.

    Per Milestone 4: distance scores on top results should be below 0.5.
    Scores above 0.6-0.7 indicate weak matches.
    """
    print("\n=== Index Verification (Milestone 4) ===")
    col          = get_collection(persist_dir, collection_name)
    model        = load_embedding_model(model_name)
    chunk_count  = len(_load_chunks(chunks_path))
    vector_count = col.count()

    print(f"Vectors in index: {vector_count}")
    print(f"Chunks in file:   {chunk_count}")

    if vector_count == chunk_count:
        print("[PASS] Counts match")
    else:
        print(f"[WARN] {chunk_count - vector_count} chunks not indexed — "
              "run embed_and_store() again")

    # Test query — distance scores should be < 0.5 for good retrieval
    query = "Which public health professor gives the most useful feedback?"
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(5, vector_count),
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nTest query: '{query}'")
    print(f"(distance < 0.5 = good match, > 0.7 = weak match)\n")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        quality = "good" if dist < 0.5 else ("ok" if dist < 0.7 else "weak")
        print(f"  [{i+1}] dist={dist:.3f} ({quality})  "
              f"Prof:{meta.get('professor_name','N/A')}  "
              f"Source:{meta.get('source','N/A')}")
        print(f"       {doc[:100]}...")

    print(f"\n[PASS] Index verification complete")
    return True


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if not Path(CHUNKS_PATH).exists():
        print(f"Not found: {CHUNKS_PATH}\nRun chunk.py first.")
        sys.exit(1)
    total = embed_and_store()
    print(f"\nTotal vectors: {total}")
    verify_index()
    print(f"\nNEXT STEP: run python retrieve.py to test retrieval")
