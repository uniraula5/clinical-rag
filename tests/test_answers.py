"""Tests for the answer, fact-check and revision logic (answer.py, verify.py,
pipeline.py). The LLM is replaced with fakes, so no API key is needed."""

import pytest

import pipeline
import verify
from answer import clean_citations, format_sources
from pipeline import QAPipeline, find_position, passage_for
from verify import check_citations, is_no_answer, split_sentences, verify_answer


def test_clean_citations_converts_model_style_and_joins_lines():
    raw = "Bronchitis causes wheezing【2†L1-L4】【4†L1-L4】.  \nAsthma narrows airways【3】.\nIt is common [1]."
    assert clean_citations(raw) == "Bronchitis causes wheezing[2][4]. Asthma narrows airways[3]. It is common [1]."


def test_format_sources():
    sources = [{"number": 1, "question": "What is gout?", "source": "NIDDK", "text": "Gout is arthritis."}]
    assert format_sources(sources) == "[1] What is gout? (source: NIDDK)\nGout is arthritis."


@pytest.mark.parametrize("text, expected", [
    ("It is rare [1]. It is inherited [2].", ["It is rare [1].", "It is inherited [2]."]),
    ("It is rare. [1] It is inherited [2].", ["It is rare. [1]", "It is inherited [2]."]),
    ("Dose is 2.5 mg [1].", ["Dose is 2.5 mg [1]."]),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_check_citations():
    sentences = ["Cited [1].", "Not cited.", "Too high [9]."]
    problems = check_citations(sentences, n_sources=3)
    assert len(problems) == 2
    assert problems[0].startswith("No citation")
    assert "does not exist [9]" in problems[1]


def test_is_no_answer_tolerates_small_differences():
    assert is_no_answer("The sources I found don’t answer this question")
    assert not is_no_answer("Zinc helps [1].")


def test_checker_reply_with_string_false_is_not_supported(monkeypatch):
    # "false" as a string would be True if passed to bool(), which is the bug this guards against
    reply = {"results": [{"sentence": "1", "supported": "false", "reason": "not in source"},
                         {"sentence": 2, "supported": True, "reason": "ok"}]}
    monkeypatch.setattr(verify, "chat_json", lambda system, user: reply)
    result = verify_answer("First claim [1]. Second claim [1].", "sources", n_sources=1)

    assert [c["supported"] for c in result["claims"]] == [False, True]
    assert not result["passed"]


def test_sentence_skipped_by_checker_counts_as_unsupported(monkeypatch):
    monkeypatch.setattr(verify, "chat_json", lambda system, user: {"results": [{"sentence": 1, "supported": True}]})
    result = verify_answer("Checked [1]. Skipped [1].", "sources", n_sources=1)
    assert result["claims"][1] == {"sentence": "Skipped [1].", "supported": False, "reason": "not checked"}
    assert not result["passed"]


def test_no_answer_reply_passes_without_calling_the_llm(monkeypatch):
    def fail(*args):
        raise AssertionError("the checker should not be called")
    monkeypatch.setattr(verify, "chat_json", fail)
    assert verify_answer("The sources I found don't answer this question.", "sources", 3)["passed"]


def test_find_position():
    assert find_position(["a", "b", "c", "d"], ["c", "d"]) == 2
    assert find_position(["a"], ["x"]) is None


def test_passage_for_short_answer_is_used_whole():
    assert passage_for("a b c", "Question: q\nAnswer: b", max_words=350) == "a b c"


def test_passage_for_long_answer_is_centered_on_matched_chunk():
    full = " ".join(f"w{i}" for i in range(1000))
    chunk = "Question: q?\nAnswer: " + " ".join(f"w{i}" for i in range(600, 700))
    passage = passage_for(full, chunk, max_words=350)
    words = passage.replace("... ", "").replace(" ...", "").split()

    assert len(words) == 350
    assert "w600" in words and "w699" in words
    assert passage.startswith("... ") and passage.endswith(" ...")


def make_pipeline(monkeypatch, checks):
    """A QAPipeline with fake sources and a fake LLM. `checks` is the list of
    pass/fail results the fact-checker will return, in order."""
    pipe = object.__new__(QAPipeline)  # skip __init__, which loads the real index
    pipe.cache = {}
    pipe.get_sources = lambda question: [
        {"number": 1, "doc_id": "D", "source": "S", "question": "q", "score": 0.9, "text": "source text"}
    ]
    results = iter(checks)
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, sources_text: "draft [1].")
    monkeypatch.setattr(pipeline, "revise_answer", lambda question, sources_text, draft, problems: "revised [1].")
    monkeypatch.setattr(pipeline, "verify_answer", lambda answer, sources_text, n: next(results))
    return pipe


PASS = {"passed": True, "claims": [], "problems": []}
FAIL = {"passed": False, "claims": [], "problems": ["Not supported"]}


def test_ask_revises_once_when_the_draft_fails(monkeypatch):
    result = make_pipeline(monkeypatch, [FAIL, PASS]).ask("question")
    assert [s["step"] for s in result["steps"]] == ["draft", "revision"]
    assert result["answer"] == "revised [1]."
    assert result["passed"]


def test_ask_stops_after_max_revisions_and_reports_failure(monkeypatch):
    result = make_pipeline(monkeypatch, [FAIL, FAIL, FAIL]).ask("question")
    assert len(result["steps"]) == 1 + pipeline.MAX_REVISIONS
    assert not result["passed"]


def test_ask_skips_revision_when_draft_passes(monkeypatch):
    pipe = make_pipeline(monkeypatch, [PASS])
    monkeypatch.setattr(pipeline, "revise_answer", lambda *args: pytest.fail("should not revise"))
    result = pipe.ask("question")
    assert [s["step"] for s in result["steps"]] == ["draft"]
    assert result["answer"] == "draft [1]."


def test_ask_answers_the_same_question_from_memory(monkeypatch):
    pipe = make_pipeline(monkeypatch, [PASS, PASS])
    writes = []
    monkeypatch.setattr(pipeline, "generate_answer",
                        lambda question, sources_text: writes.append(question) or "draft [1].")

    first = pipe.ask("Same question?")
    second = pipe.ask("  same QUESTION?  ")  # spacing and capitals shouldn't matter

    assert second is first          # the saved result comes straight back
    assert len(writes) == 1         # the LLM ran only once
    assert pipe.ask("Different question?", use_cache=False) is not first


def test_get_sources_uses_hybrid_search_with_both_indexes(monkeypatch):
    pipe = object.__new__(QAPipeline)
    pipe.collection = "fake-collection"
    pipe.keyword_index = "fake-keyword-index"
    pipe.full_answers = {"D": "the full answer text"}
    calls = {}

    def fake_hybrid(question, n_results, collection, keyword_index):
        calls.update(question=question, n_results=n_results, collection=collection, keyword_index=keyword_index)
        return [{"doc_id": "D", "source": "S", "question": "q", "score": 0.03,
                 "found_by": ["semantic", "keyword"], "text": "Question: q\nAnswer: the full answer text"}]

    monkeypatch.setattr(pipeline, "hybrid_search", fake_hybrid)
    sources = pipe.get_sources("a question")

    assert calls["collection"] == "fake-collection"
    assert calls["keyword_index"] == "fake-keyword-index"
    assert calls["n_results"] == pipeline.N_SOURCES
    assert sources[0]["number"] == 1
    assert sources[0]["found_by"] == ["semantic", "keyword"]
    assert sources[0]["text"] == "the full answer text"
