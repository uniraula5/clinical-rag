"""
Keyword search over the same chunks using BM25, the classic ranking method
behind most search engines. It scores a chunk by the exact words it shares
with the query, and gives rare words (like a specific disease name) much
more weight than common ones (like "symptoms").

    python keyword_search.py "Is MELAS inherited?"
"""

import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

from chunking import build_chunks
from clean_data import clean_medquad
from search import print_results

# same idea as search.py: grab extra chunks, keep the best one per answer
CHUNKS_PER_RESULT = 4


def tokenize(text):
    # lowercase words and numbers: "Prader-Willi syndrome?" -> ["prader", "willi", "syndrome"]
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordIndex:
    def __init__(self, chunks=None):
        if chunks is None:
            chunks = build_chunks(clean_medquad(verbose=False))
        self.chunks = chunks.reset_index(drop=True)
        # BM25 is built in memory from the same chunk text that was embedded
        self.bm25 = BM25Okapi([tokenize(text) for text in self.chunks["text"]])

    def search(self, query, n_results=5):
        scores = self.bm25.get_scores(tokenize(query))
        best_first = np.argsort(scores)[::-1][: n_results * CHUNKS_PER_RESULT]

        results = []
        seen_docs = set()
        for i in best_first:
            if scores[i] <= 0:
                break  # no shared words, nothing useful below this
            row = self.chunks.iloc[i]
            if row["doc_id"] in seen_docs:
                continue
            seen_docs.add(row["doc_id"])
            results.append({
                "rank": len(results) + 1,
                "score": round(float(scores[i]), 3),
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "question_type": row["question_type"],
                "question": row["question"],
                "text": row["text"],
            })
            if len(results) == n_results:
                break
        return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python keyword_search.py "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    start = time.time()
    index = KeywordIndex()
    print(f"Built keyword index in {time.time() - start:.1f}s")
    print_results(query, index.search(query))
