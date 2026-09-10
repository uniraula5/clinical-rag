"""
Semantic search over the MedQuAD index: turns a question into a vector and
returns the closest answers, best match first.

    python search.py "What are the symptoms of type 2 diabetes?"
"""

import sys

import chromadb

from build_index import CHROMA_DIR, COLLECTION_NAME, get_embedding_function

# several chunks from one answer can all match a query, so ask for extra
# chunks and keep only the best chunk per answer
CHUNKS_PER_RESULT = 4


def get_collection(chroma_dir=CHROMA_DIR):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(COLLECTION_NAME, embedding_function=get_embedding_function())


def semantic_search(query, n_results=5, collection=None):
    if collection is None:
        collection = get_collection()

    raw = collection.query(query_texts=[query], n_results=n_results * CHUNKS_PER_RESULT)

    results = []
    seen_docs = set()
    for chunk_id, distance, meta, text in zip(
        raw["ids"][0], raw["distances"][0], raw["metadatas"][0], raw["documents"][0]
    ):
        if meta["doc_id"] in seen_docs:
            continue
        seen_docs.add(meta["doc_id"])
        results.append({
            "rank": len(results) + 1,
            "score": round(1 - distance, 3),  # cosine distance -> similarity
            "doc_id": meta["doc_id"],
            "chunk_id": chunk_id,
            "source": meta["source"],
            "question_type": meta["question_type"],
            "question": meta["question"],
            "text": text,
        })
        if len(results) == n_results:
            break
    return results


def print_results(query, results):
    print(f'\nQuery: "{query}"\n')
    for r in results:
        print(f"{r['rank']}. [{r['score']:.3f}] {r['source']} | {r['question']}")
        answer = r["text"].split("\nAnswer: ", 1)[-1]
        print(f"   {answer[:200]}...\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python search.py "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print_results(query, semantic_search(query))
