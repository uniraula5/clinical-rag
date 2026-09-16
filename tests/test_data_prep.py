"""Tests for loading and cleaning the MedQuAD data (load_data.py, clean_data.py)."""

import pandas as pd
import pytest

import clean_data
from clean_data import clean_question, clean_text, is_only_a_question
from load_data import REQUIRED_COLUMNS, load_medquad, source_from_filename


@pytest.mark.parametrize("filename, expected", [
    ("5_NIDDK_QA.csv", "NIDDK"),
    ("8_NHLBI_QA_XML.csv", "NHLBI"),
    ("10_MPlus_ADAM_QA.csv", "MPlus_ADAM"),
    ("4_MPlus_Health_Topics_QA.csv", "MPlus_Health_Topics"),
])
def test_source_from_filename(tmp_path, filename, expected):
    assert source_from_filename(tmp_path / filename) == expected


def test_source_from_unexpected_filename_fails_loudly(tmp_path):
    with pytest.raises(ValueError):
        source_from_filename(tmp_path / "notes.csv")


def test_load_medquad_tags_sources_and_marks_empty_answers(tmp_path):
    pd.DataFrame({
        "question": ["What is A ?", "What is B ?"],
        "question_id": ["1", "2"],
        "question_type": ["information", "information"],
        "answer": ["A is a thing.", "   "],
    }).to_csv(tmp_path / "1_CancerGov_QA.csv")  # index gets saved as "Unnamed: 0", like the real files
    pd.DataFrame({
        "question": ["What is C ?"],
        "question_id": ["1"],
        "question_type": ["symptoms"],
        "answer": [None],
    }).to_csv(tmp_path / "2_GARD_QA.csv")

    data = load_medquad(tmp_path)

    assert len(data) == 3
    assert "Unnamed: 0" not in data.columns
    assert list(data["source"]) == ["CancerGov", "CancerGov", "GARD"]
    # a missing answer and a whitespace-only answer both count as empty
    assert list(data["has_answer"]) == [True, False, False]


def test_load_medquad_with_no_files_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_medquad(tmp_path)


def test_clean_text_keeps_paragraphs_and_squashes_spaces():
    raw = "First   line\n  still first.\n                \nSecond paragraph."
    assert clean_text(raw) == "First line still first.\n\nSecond paragraph."


@pytest.mark.parametrize("raw, expected", [
    ("What is (are) Acromegaly ?", "What is (are) Acromegaly?"),
    ("Who is at risk for Neuroblastoma? ?", "Who is at risk for Neuroblastoma?"),
    ("Is it  inherited?", "Is it inherited?"),
])
def test_clean_question(raw, expected):
    assert clean_question(raw) == expected


@pytest.mark.parametrize("answer, is_junk", [
    # the question copied where the answer should be - nothing to learn from it
    ("Is Williams syndrome inherited?", True),
    ("How might Bell's palsy be treated?", True),
    # a real answer that happens to finish on a question. Checking only for a
    # "?" at the end used to throw these away.
    ("A desirable level is under 200 mg/dL. Do you know how yours compares?", False),
    ("What causes it? It can arise from errors in cell division.", False),
    ("Atherosclerosis is a slow, complex disease.", False),
])
def test_is_only_a_question(answer, is_junk):
    assert is_only_a_question(answer) is is_junk


def test_clean_medquad_applies_every_rule(monkeypatch):
    fake_raw = pd.DataFrame([
        # question, question_id, question_type, answer, source, has_answer
        ("What is A ?", "1", "information", "A is a thing.", "S1", True),      # kept
        ("What is B ?", "2", "information", "", "S1", False),                  # no answer
        ("What causes C ?", "3", "causes", "What causes C?", "S1", True),      # answer is just a question
        ("What is A ?", "4", "information", "A is a thing.", "S1", True),      # duplicate question + answer
        ("What is D ?", "5", "information", "A is   a thing.", "S1", True),    # same answer once spaces are cleaned
        ("What is E ?", "1", "information", "E is different.", "S1", True),    # kept, repeats question_id 1
    ], columns=["question", "question_id", "question_type", "answer", "source", "has_answer"])
    monkeypatch.setattr(clean_data, "load_medquad", lambda: fake_raw.copy())

    clean = clean_data.clean_medquad(verbose=False)

    assert list(clean["question"]) == ["What is A?", "What is E?"]
    # question_id 1 appears twice in the same source, so the second one gets a counter
    assert list(clean["doc_id"]) == ["S1_1", "S1_1_1"]
    assert list(clean.columns) == ["doc_id", "source", "question_type", "question", "answer"]


def test_a_corrupted_csv_says_which_file_to_delete(tmp_path):
    # a blocked download can still be saved with a .csv name
    (tmp_path / "1_CancerGov_QA.csv").write_text("<html>404: Not Found</html>\n")
    with pytest.raises(ValueError) as error:
        load_medquad(raw_dir=tmp_path)

    message = str(error.value)
    assert "1_CancerGov_QA.csv" in message
    assert "python download_data.py" in message
    assert all(column in message for column in REQUIRED_COLUMNS)
