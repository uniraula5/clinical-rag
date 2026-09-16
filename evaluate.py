"""
Measures how well search finds the right answers, using the hand-written
test questions in eval/test_cases.json.

Each test question lists the answers (doc_ids) that correctly answer it.
A search "hits" if at least one of them shows up in the top K results.
Runs semantic, keyword and hybrid search on the same questions to compare them.

    python evaluate.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from clean_data import clean_medquad
from hybrid_search import hybrid_search
from keyword_search import KeywordIndex
from search import get_collection, semantic_search

TEST_CASES_PATH = Path(__file__).parent / "eval" / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "eval" / "retrieval_results.csv"
K = 5


def load_test_cases(path=TEST_CASES_PATH):
    with open(path) as f:
        return json.load(f)


def words_in(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_overlap(query, titles):
    """How much of the question is already in the title of a correct answer.

    1.00 means every word of the question also appears in the title, so finding
    it is close to looking up a heading. This is the honest way to say how hard
    a test question really is, instead of trusting the label I gave it.
    """
    query_words = words_in(query)
    if not query_words:
        return 0.0
    return max((len(query_words & words_in(t)) / len(query_words) for t in titles), default=0.0)


def print_overlap_table(test_cases, titles_by_doc_id):
    by_category = defaultdict(list)
    skipped = 0
    for case in test_cases:
        titles = [titles_by_doc_id[d] for d in case["relevant_doc_ids"] if d in titles_by_doc_id]
        if not titles:
            skipped += 1
            continue
        by_category[case["category"]].append(title_overlap(case["query"], titles))

    print("\n=== How much of each question is already in the answer's title ===")
    print(f"  {'category':13s}{'median':>9s}{'near-copies':>13s}")
    for category, scores in sorted(by_category.items()):
        near = sum(1 for s in scores if s >= 0.8)
        print(f"  {category:13s}{median(scores):>9.0%}{near:>8d}/{len(scores):<4d}")
    if skipped:
        print(f"  ({skipped} question(s) skipped: no labelled answer is in the data)")


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


def print_comparison(all_rows, k=K):
    # one line per method, one column per category, each cell "hit@k / MRR"
    categories = sorted({r["category"] for rows in all_rows.values() for r in rows})
    columns = ["all"] + categories
    print(f"\n=== Comparison: hit@{k} / MRR ===")
    print(f"  {'':22s}" + "".join(f"{c:>14s}" for c in columns))
    for name, rows in all_rows.items():
        cells = []
        for column in columns:
            subset = rows if column == "all" else [r for r in rows if r["category"] == column]
            s = summarize(subset)
            cells.append(f"{s['hit_rate']:.2f} / {s['mrr']:.2f}")
        print(f"  {name:22s}" + "".join(f"{c:>14s}" for c in cells))


def hit_at_k(rows, k):
    """Share of questions with a correct answer in the top k."""
    return sum(1 for r in rows if r["rank"] and r["rank"] <= k) / len(rows)


def print_hit_at_k_table(all_rows, max_k=K):
    # The answer pipeline only sends its top few sources to the LLM, so this table
    # is what decides how many sources it should send.
    categories = sorted({r["category"] for rows in all_rows.values() for r in rows})
    print("\n=== hit@k: how often a correct answer is in the top k ===")
    print(f"  {'method':22s}{'category':13s}" + "".join(f"{'hit@' + str(k):>8s}" for k in range(1, max_k + 1)))
    for name, rows in all_rows.items():
        for category in ["all"] + categories:
            subset = rows if category == "all" else [r for r in rows if r["category"] == category]
            cells = "".join(f"{hit_at_k(subset, k):>8.2f}" for k in range(1, max_k + 1))
            print(f"  {name:22s}{category:13s}{cells}")


def save_ranks(all_rows, test_cases, path=RESULTS_PATH):
    # one row per question: the rank of the first correct answer for each method (blank = missed)
    names = list(all_rows)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "query"] + names)
        for i, case in enumerate(test_cases):
            ranks = [all_rows[name][i]["rank"] or "" for name in names]
            writer.writerow([case["id"], case["category"], case["query"]] + ranks)
    print(f"\nSaved per-question ranks to {path}")


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

    all_rows = {}
    for name, search_fn in methods.items():
        rows = evaluate(search_fn, test_cases)
        print_report(name, rows)
        all_rows[name] = rows

    print_comparison(all_rows)
    print_hit_at_k_table(all_rows)

    data = clean_medquad(verbose=False)
    print_overlap_table(test_cases, dict(zip(data["doc_id"], data["question"])))

    save_ranks(all_rows, test_cases)
