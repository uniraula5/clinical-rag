"""Tests for the eval scoring (evaluate.py) and the test question file itself."""

import re
from collections import Counter
from pathlib import Path

import pytest

from evaluate import (evaluate, first_hit_rank, hit_at_k, load_test_cases, print_hit_at_k_table,
                      print_overlap_table, save_ranks, summarize, title_overlap)
from evaluate_answers import HOLDOUT_PATH, RESULTS_PATH, results_path_for, select_cases

HAS_DATA = any((Path(__file__).parent.parent / "data" / "raw").glob("*.csv"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs the MedQuAD data (python download_data.py)")


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
    assert len({c["id"] for c in cases}) == len(cases) == 48
    assert Counter(c["category"] for c in cases) == {"paraphrase": 16, "exact_term": 16, "gene_symbol": 16}
    assert all(c["query"].strip() and c["relevant_doc_ids"] for c in cases)


@needs_data
def test_every_labeled_answer_exists_in_the_clean_data():
    from clean_data import clean_medquad

    known_ids = set(clean_medquad(verbose=False)["doc_id"])
    missing = {doc_id for case in load_test_cases() for doc_id in case["relevant_doc_ids"]} - known_ids
    assert not missing, f"labels point to answers that don't exist: {missing}"


@needs_data
def test_gene_questions_point_to_answers_that_mention_the_gene():
    from clean_data import clean_medquad

    data = clean_medquad(verbose=False)
    answers = dict(zip(data["doc_id"], data["answer"]))
    for case in load_test_cases():
        if case["category"] != "gene_symbol":
            continue
        gene = re.search(r"in the (\S+) gene", case["query"]).group(1)
        for doc_id in case["relevant_doc_ids"]:
            assert f"{gene} gene" in answers[doc_id], (case["id"], doc_id)


def test_save_ranks_writes_one_row_per_question(tmp_path):
    cases = [{"id": "1", "category": "c", "query": "q1"}, {"id": "2", "category": "c", "query": "q2"}]
    all_rows = {"Semantic": [{"rank": 1}, {"rank": None}], "Keyword": [{"rank": 3}, {"rank": 2}]}
    path = tmp_path / "ranks.csv"
    save_ranks(all_rows, cases, path)
    # a miss is written as an empty cell
    assert path.read_text().splitlines() == ["id,category,query,Semantic,Keyword", "1,c,q1,1,3", "2,c,q2,,2"]


def test_hit_at_k_counts_ranks_up_to_k():
    rows = [{"rank": 1}, {"rank": 3}, {"rank": None}, {"rank": 5}]
    assert hit_at_k(rows, 1) == 0.25   # only the rank-1 question
    assert hit_at_k(rows, 3) == 0.50   # ranks 1 and 3
    assert hit_at_k(rows, 5) == 0.75   # a miss never counts, whatever k is


def test_hit_at_k_table_shows_every_method_and_category(capsys):
    all_rows = {
        "Semantic": [{"rank": 1, "category": "paraphrase"}, {"rank": None, "category": "gene_symbol"}],
        "Hybrid": [{"rank": 2, "category": "paraphrase"}, {"rank": 1, "category": "gene_symbol"}],
    }
    print_hit_at_k_table(all_rows, max_k=3)
    printed = capsys.readouterr().out

    assert "hit@1" in printed and "hit@3" in printed
    for label in ["Semantic", "Hybrid", "all", "paraphrase", "gene_symbol"]:
        assert label in printed
    # semantic: 1 of 2 questions found by rank 3; hybrid: both
    assert "    0.50" in printed and "    1.00" in printed


def test_select_cases_takes_every_fourth_question():
    cases = [{"id": str(i)} for i in range(12)]
    assert [c["id"] for c in select_cases(cases, 0)] == ["0", "4", "8"]
    assert [c["id"] for c in select_cases(cases, 2)] == ["2", "6", "10"]


def test_tuning_and_holdout_sets_never_overlap():
    cases = load_test_cases()
    tuning = {c["id"] for c in select_cases(cases, 0)}
    holdout = {c["id"] for c in select_cases(cases, 2)}
    assert len(tuning) == len(holdout) == 12
    assert not tuning & holdout
    # both samples should still cover all three kinds of question
    for sample in (select_cases(cases, 0), select_cases(cases, 2)):
        assert len({c["category"] for c in sample}) == 3


def test_results_go_to_separate_files():
    assert results_path_for(0) == RESULTS_PATH
    assert results_path_for(2) == HOLDOUT_PATH
    assert results_path_for(0) != results_path_for(2)


def test_title_overlap_scores_how_much_of_the_question_is_in_the_title():
    # every word of the question is in the title: finding it is close to a lookup
    assert title_overlap("Is Huntington disease inherited?",
                         ["Is Huntington disease inherited?"]) == 1.0
    # a real paraphrase shares almost nothing with the title
    assert title_overlap("keep bones from breaking",
                         ["What is (are) Osteoporosis?"]) == 0.0
    # the easiest of the labelled answers is the one that counts
    assert title_overlap("What causes gout?",
                         ["Something else entirely", "What causes gout?"]) == 1.0


def test_title_overlap_of_a_partly_matching_question():
    # words are matched whole, with no stemming: of "how is gout treated" only
    # "gout" is in the title, because "treated" and "treatments" are different words
    assert title_overlap("how is gout treated",
                         ["What are the treatments for gout?"]) == pytest.approx(0.25)
    assert title_overlap("symptoms of gout", ["What are the symptoms of gout?"]) == 1.0


def test_overlap_table_skips_questions_whose_answer_is_not_in_the_data(capsys):
    cases = [
        {"id": "a", "category": "paraphrase", "query": "what is gout", "relevant_doc_ids": ["D1"]},
        {"id": "b", "category": "paraphrase", "query": "what is gout", "relevant_doc_ids": ["GONE"]},
    ]
    print_overlap_table(cases, {"D1": "What is (are) gout?"})
    out = capsys.readouterr().out
    assert "1 question(s) skipped" in out
