"""
Hybrid search: runs semantic search and keyword (BM25) search, then merges
the two ranked lists with Reciprocal Rank Fusion (RRF).

Semantic search is good at meaning ("wheezing" -> asthma). Keyword search is
good at exact names ("Wolff-Parkinson-White", "MELAS"). RRF rewards answers
that rank well in either list, and most of all in both.

    python hybrid_search.py "Is MELAS inherited?"
"""

import sys

from keyword_search import KeywordIndex
from search import get_collection, print_results, semantic_search

RRF_K = 60        # standard value from the original RRF paper
CANDIDATES = 20   # answers taken from each method before merging


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    # ranked_lists looks like {"semantic": [...], "keyword": [...]}
    # every answer earns 1 / (k + rank) from each list it shows up in
    fused = {}
    for method, results in ranked_lists.items():
        for r in results:
            # if an answer is in both lists, keep the first copy's text
            entry = fused.setdefault(r["doc_id"], {"result": r, "score": 0.0, "found_by": []})
            entry["score"] += 1 / (k + r["rank"])
            entry["found_by"].append(method)
    return sorted(fused.values(), key=lambda e: e["score"], reverse=True)


def hybrid_search(query, n_results=5, collection=None, keyword_index=None):
    if collection is None:
        collection = get_collection()
    if keyword_index is None:
        keyword_index = KeywordIndex()

    ranked_lists = {
        "semantic": semantic_search(query, n_results=CANDIDATES, collection=collection),
        "keyword": keyword_index.search(query, n_results=CANDIDATES),
    }

    results = []
    for entry in reciprocal_rank_fusion(ranked_lists)[:n_results]:
        r = dict(entry["result"])  # copy, so the original lists aren't changed
        r["rank"] = len(results) + 1
        r["score"] = round(entry["score"], 4)
        r["found_by"] = entry["found_by"]
        results.append(r)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python hybrid_search.py "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    results = hybrid_search(query)
    print_results(query, results)
    for r in results:
        print(f"{r['rank']}. found by: {' + '.join(r['found_by'])}")
