"""
Measures how well search finds the right answers, using the hand-written
test questions in eval/test_cases.json.

Each test question lists the answers (doc_ids) that correctly answer it.
A search "hits" if at least one of them shows up in the top K results.
Runs semantic, keyword and hybrid search on the same questions to compare them.

    python evaluate.py
"""

import json
from collections import defaultdict
from pathlib import Path

from hybrid_search import hybrid_search
from keyword_search import KeywordIndex
from search import get_collection, semantic_search

TEST_CASES_PATH = Path(__file__).parent / "eval" / "test_cases.json"
K = 5


def load_test_cases(path=TEST_CASES_PATH):
    with open(path) as f:
        return json.load(f)


def first_hit_rank(results, relevant_ids):
    # rank (1, 2, 3...) of the first correct answer, or None if none showed up
    for r in results:
        if r["doc_id"] in relevant_ids:
            return r["rank"]
    return None


def evaluate(search_fn, test_cases, k=K):
    rows = []
    for case in test_cases:
        results = search_fn(case["query"], k)
        rank = first_hit_rank(results, set(case["relevant_doc_ids"]))
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "hit": rank is not None,
            "rank": rank,
            "top_result": results[0]["question"] if results else "",
        })
    return rows


def summarize(rows):
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    # reciprocal rank: 1 if the right answer is first, 1/2 if second, ... 0 if missed
    mrr = sum(1 / r["rank"] for r in rows if r["hit"]) / n
    return {"cases": n, "hit_rate": hits / n, "mrr": mrr}


def print_report(name, rows, k=K):
    overall = summarize(rows)
    print(f"\n=== {name} ===")
    print(f"All {overall['cases']} cases: hit@{k} = {overall['hit_rate']:.2f}, MRR = {overall['mrr']:.2f}")

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    for category, cat_rows in sorted(by_category.items()):
        s = summarize(cat_rows)
        print(f"  {category:12s} ({s['cases']:2d} cases): hit@{k} = {s['hit_rate']:.2f}, MRR = {s['mrr']:.2f}")

    misses = [r for r in rows if not r["hit"]]
    if misses:
        print(f"\nMissed ({len(misses)}):")
        for r in misses:
            print(f"  [{r['id']}] {r['query']}")
            print(f"      top result was: {r['top_result']}")


if __name__ == "__main__":
    test_cases = load_test_cases()
    collection = get_collection()
    keyword_index = KeywordIndex()

    # every method gets the same (query, k) shape so evaluate() can run all three
    methods = {
        "Semantic search": lambda q, k: semantic_search(q, n_results=k, collection=collection),
        "Keyword search (BM25)": lambda q, k: keyword_index.search(q, k),
        "Hybrid search (RRF)": lambda q, k: hybrid_search(
            q, n_results=k, collection=collection, keyword_index=keyword_index
        ),
    }

    summaries = {}
    for name, search_fn in methods.items():
        rows = evaluate(search_fn, test_cases)
        print_report(name, rows)
        summaries[name] = summarize(rows)

    print(f"\n=== Comparison: hit@{K} / MRR ===")
    for name, s in summaries.items():
        print(f"  {name:22s} {s['hit_rate']:.2f} / {s['mrr']:.2f}")
