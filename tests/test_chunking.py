"""Tests for splitting answers into chunks (chunking.py)."""

import pandas as pd

from chunking import MAX_WORDS, OVERLAP_WORDS, build_chunks, chunk_answer


def numbered_words(count, start=0):
    return [f"w{i}" for i in range(start, start + count)]


def test_short_answer_is_one_chunk():
    answer = "Short answer. Only a few words."
    assert chunk_answer(answer) == ["Short answer. Only a few words."]


def test_long_paragraph_chunks_stay_under_limit_and_overlap():
    # one 300-word paragraph made of 30 ten-word sentences
    words = numbered_words(300)
    sentences = [" ".join(words[i:i + 10]) + "." for i in range(0, 300, 10)]
    chunks = chunk_answer(" ".join(sentences))

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= MAX_WORDS
    # each chunk starts with the last OVERLAP_WORDS words of the chunk before it
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split()[:OVERLAP_WORDS] == previous.split()[-OVERLAP_WORDS:]


def test_no_words_are_lost():
    words = numbered_words(250)
    answer = " ".join(words[:120]) + "\n\n" + " ".join(words[120:])
    chunk_words = {w.rstrip(".") for chunk in chunk_answer(answer) for w in chunk.split()}
    assert set(words) <= chunk_words


def test_one_huge_sentence_gets_hard_cut():
    # 250 words with no punctuation at all
    chunks = chunk_answer(" ".join(numbered_words(250)))
    assert all(len(chunk.split()) <= MAX_WORDS for chunk in chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w249"


def test_build_chunks_ids_and_question_prefix():
    data = pd.DataFrame([{
        "doc_id": "NIDDK_1",
        "source": "NIDDK",
        "question_type": "treatment",
        "question": "How is it treated?",
        "answer": " ".join(numbered_words(250)),
    }])
    chunks = build_chunks(data)

    n = len(chunks)
    assert n > 1
    assert list(chunks["chunk_id"]) == [f"NIDDK_1#{i}" for i in range(n)]
    assert (chunks["num_chunks"] == n).all()
    # every chunk carries the question, so a middle chunk still says what it's about
    assert chunks["text"].str.startswith("Question: How is it treated?\nAnswer: ").all()
