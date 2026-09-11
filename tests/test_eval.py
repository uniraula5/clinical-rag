"""Tests for the eval scoring (evaluate.py) and the test question file itself."""

from collections import Counter
from pathlib import Path

import pytest

from evaluate import TEST_CASES_PATH, evaluate, first_hit_rank, load_test_cases, summarize


def test_first_hit_rank():
    results = [{"rank": 1, "doc_id": "A"}, {"rank": 2, "doc_id": "B"}]
    assert first_hit_rank(results, {"B", "Z"}) == 2
    assert first_hit_rank(results, {"Z"}) is None


def test_hit_rate_and_mrr_math():
    ranked = {"q1": ["A", "B"], "q2": ["X", "Y", "C"], "q3": ["X", "Y"]}

    def fake_search(query, k):
        return [{"rank": i, "doc_id": d, "question": d} for i, d in enumerate(ranked[query][:k], start=1)]

    cases = [
        {"id": "1", "category": "paraphrase", "query": "q1", "relevant_doc_ids": ["A"]},  # rank 1
        {"id": "2", "category": "paraphrase", "query": "q2", "relevant_doc_ids": ["C"]},  # rank 3
        {"id": "3", "category": "exact_term", "query": "q3", "relevant_doc_ids": ["C"]},  # miss
    ]
    rows = evaluate(fake_search, cases, k=5)
    summary = summarize(rows)

    assert [r["rank"] for r in rows] == [1, 3, None]
    assert summary["hit_rate"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_test_cases_file_is_well_formed():
    cases = load_test_cases()
    assert len(cases) == 32
    assert len({c["id"] for c in cases}) == 32
    assert Counter(c["category"] for c in cases) == {"paraphrase": 16, "exact_term": 16}
    assert all(c["query"].strip() and c["relevant_doc_ids"] for c in cases)


@pytest.mark.skipif(not any((Path(__file__).parent.parent / "data" / "raw").glob("*.csv")),
                    reason="needs the MedQuAD data (python download_data.py)")
def test_every_labeled_answer_exists_in_the_clean_data():
    from clean_data import clean_medquad

    known_ids = set(clean_medquad(verbose=False)["doc_id"])
    missing = {doc_id for case in load_test_cases() for doc_id in case["relevant_doc_ids"]} - known_ids
    assert not missing, f"labels point to answers that don't exist: {missing}"
