"""Tests for semantic, keyword and hybrid search logic, using small fake data
so no embedding model or database is needed."""

import pandas as pd
import pytest

import search
from hybrid_search import hybrid_search, reciprocal_rank_fusion
from keyword_search import KeywordIndex, tokenize
from search import semantic_search


class FakeCollection:
    """Stands in for a Chroma collection: returns chunks already sorted by distance."""

    def __init__(self, rows):
        self.rows = rows  # (chunk_id, distance, doc_id)

    def query(self, query_texts, n_results):
        rows = self.rows[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "distances": [[r[1] for r in rows]],
            "metadatas": [[{"doc_id": r[2], "source": "S", "question_type": "t", "question": f"about {r[2]}"}
                           for r in rows]],
            "documents": [[f"Question: about {r[2]}\nAnswer: text of {r[0]}" for r in rows]],
        }


def test_semantic_search_keeps_best_chunk_per_answer():
    collection = FakeCollection([("A#0", 0.1, "A"), ("A#1", 0.2, "A"), ("B#0", 0.3, "B"), ("C#0", 0.4, "C")])
    results = semantic_search("anything", n_results=3, collection=collection)

    assert [r["doc_id"] for r in results] == ["A", "B", "C"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert results[0]["chunk_id"] == "A#0"          # the closer chunk of A wins
    assert results[0]["score"] == pytest.approx(0.9)  # cosine distance 0.1 -> similarity 0.9


def test_tokenize():
    assert tokenize("Is Prader-Willi syndrome inherited?") == ["is", "prader", "willi", "syndrome", "inherited"]


@pytest.fixture
def keyword_index():
    rows = [
        ("X", "X#0", "Is MELAS inherited?", "MELAS is passed down from the mother."),
        ("X", "X#1", "Is MELAS inherited?", "Mitochondrial DNA details."),
        ("Y", "Y#0", "What is asthma?", "Asthma narrows the airways."),
        ("Z", "Z#0", "What causes gout?", "Uric acid crystals."),
        ("H", "H#0", "Is Huntington disease inherited?", "It is autosomal dominant."),
    ]
    chunks = pd.DataFrame([{
        "doc_id": doc_id, "chunk_id": chunk_id, "source": "S", "question_type": "t",
        "question": question, "text": f"Question: {question}\nAnswer: {answer}",
    } for doc_id, chunk_id, question, answer in rows])
    return KeywordIndex(chunks)


def test_keyword_search_rare_name_ranks_first(keyword_index):
    results = keyword_index.search("Is Huntington disease inherited?", n_results=3)
    assert results[0]["doc_id"] == "H"


def test_keyword_search_one_result_per_answer_and_stops_at_zero(keyword_index):
    results = keyword_index.search("MELAS", n_results=5)
    # both chunks of X match, but X appears once, and nothing else shares a word
    assert [r["doc_id"] for r in results] == ["X"]


def test_keyword_search_no_shared_words_returns_nothing(keyword_index):
    assert keyword_index.search("zebra", n_results=5) == []


def test_rrf_rewards_agreement():
    semantic = [{"doc_id": "A", "rank": 1}, {"doc_id": "B", "rank": 2}, {"doc_id": "C", "rank": 3}]
    keyword = [{"doc_id": "C", "rank": 1}, {"doc_id": "D", "rank": 2}]
    fused = reciprocal_rank_fusion({"semantic": semantic, "keyword": keyword})

    assert fused[0]["result"]["doc_id"] == "C"  # found by both methods
    assert fused[0]["score"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[0]["found_by"] == ["semantic", "keyword"]
    assert fused[1]["result"]["doc_id"] == "A"


class FakeKeywordIndex:
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids

    def search(self, query, n_results=5):
        return [{"rank": i, "doc_id": d, "chunk_id": f"{d}#0", "score": 1.0, "source": "S",
                 "question_type": "t", "question": f"about {d}", "text": f"kw text {d}"}
                for i, d in enumerate(self.doc_ids[:n_results], start=1)]


def test_hybrid_search_renumbers_and_labels():
    collection = FakeCollection([("A#0", 0.1, "A"), ("B#0", 0.2, "B")])
    keyword = FakeKeywordIndex(["B", "C"])
    results = hybrid_search("q", n_results=3, collection=collection, keyword_index=keyword)

    assert [r["doc_id"] for r in results] == ["B", "A", "C"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert results[0]["found_by"] == ["semantic", "keyword"]
    assert results[2]["found_by"] == ["keyword"]


def test_hybrid_search_returns_no_more_than_asked_for():
    # the pipeline relies on this: it asks for 4 sources and must not get 5
    collection = FakeCollection([("A#0", 0.1, "A"), ("B#0", 0.2, "B")])
    results = hybrid_search("q", n_results=2, collection=collection,
                            keyword_index=FakeKeywordIndex(["C", "D", "E"]))
    assert len(results) == 2
    assert [r["rank"] for r in results] == [1, 2]


def test_missing_index_says_which_command_to_run(monkeypatch):
    class Client:
        def __init__(self, path):
            pass

        def get_collection(self, name, embedding_function):
            raise RuntimeError("Collection [medquad] does not exist")

    monkeypatch.setattr(search, "chromadb", type("m", (), {"PersistentClient": Client}))
    monkeypatch.setattr(search, "get_embedding_function", lambda: None)

    with pytest.raises(RuntimeError, match="python build_index.py"):
        search.get_collection(chroma_dir="nowhere")


def test_print_results_does_not_add_dots_to_a_short_answer(capsys):
    search.print_results("q", [{"rank": 1, "score": 0.5, "source": "S", "question": "q?",
                                "text": "Question: q?\nAnswer: Short answer."}])
    assert "Short answer.\n" in capsys.readouterr().out
