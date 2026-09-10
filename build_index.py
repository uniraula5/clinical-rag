"""
Embeds every chunk with a local sentence-transformers model and saves the
vectors in a ChromaDB collection on disk (the chroma_db/ folder).

Run this once before searching. The full build takes a few minutes.
    python build_index.py              # full index
    python build_index.py --limit 500  # quick test with 500 chunks
"""

import argparse
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunking import build_chunks
from clean_data import clean_medquad

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "medquad"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 500

METADATA_COLUMNS = ["doc_id", "source", "question_type", "question", "chunk_index", "num_chunks"]


def get_embedding_function():
    # runs locally, no API key needed. The same function must be used when
    # searching, or the query vectors won't match the stored vectors.
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def build_index(chroma_dir=CHROMA_DIR, limit=None):
    chunks = build_chunks(clean_medquad(verbose=False))
    if limit:
        chunks = chunks.head(limit)
    print(f"Chunks to index: {len(chunks)}")

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # start from scratch each time so old chunks never mix with new ones
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete the first time

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        # compare vectors by angle (cosine), not straight-line distance
        metadata={"hnsw:space": "cosine"},
    )

    start_time = time.time()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks.iloc[start:start + BATCH_SIZE]
        collection.add(
            ids=batch["chunk_id"].tolist(),
            documents=batch["text"].tolist(),
            metadatas=batch[METADATA_COLUMNS].to_dict("records"),
        )
        done = start + len(batch)
        print(f"  {done}/{len(chunks)} chunks ({time.time() - start_time:.0f}s)")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only index the first N chunks")
    args = parser.parse_args()
    build_index(limit=args.limit)
