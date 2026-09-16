"""
Cleans the raw MedQuAD data so only real, usable Q&A pairs go into the
search index. Each cleaning step prints how many rows it removed, so
nothing gets thrown away silently.
"""

import re

from load_data import load_medquad


def clean_text(text):
    # keep paragraph breaks (blank lines) but squash every other run of
    # spaces, tabs and newlines into one space
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def clean_question(text):
    # "What is (are) Acromegaly ?" -> "What is (are) Acromegaly?"
    # "Who is at risk for Neuroblastoma? ?" -> "... Neuroblastoma?"
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[\s?]*\?$", "?", text)


def is_only_a_question(answer):
    """True when the whole answer is one question and nothing else.

    Some GARD rows have the question repeated where the answer should be
    ("Is Williams syndrome inherited?"). Checking only for a "?" at the end
    threw away real answers too, because a long answer can finish on a
    question ("Do you know how your cholesterol compares?"). So an answer
    counts as junk only if it is a single sentence ending in "?".
    """
    if not answer.endswith("?"):
        return False
    return len(re.split(r"(?<=[.!?])\s+", answer)) == 1


def keep_rows(data, mask, reason, verbose):
    if verbose:
        print(f"  {reason}: removed {(~mask).sum()}")
    return data[mask]


def clean_medquad(verbose=True):
    data = load_medquad()
    if verbose:
        print(f"Starting rows: {len(data)}")

    data = keep_rows(data, data["has_answer"], "no answer", verbose)

    data = data.copy()
    data["answer"] = data["answer"].apply(clean_text)
    data["question"] = data["question"].astype(str).apply(clean_question)

    only_a_question = data["answer"].apply(is_only_a_question)
    data = keep_rows(data, ~only_a_question, "answer is just a question", verbose)

    same_pair = data.duplicated(["question", "answer"])
    data = keep_rows(data, ~same_pair, "duplicate question + answer", verbose)

    # the same answer text under different questions would take up several
    # of the top 5 search results with identical text, so keep one copy
    same_answer = data.duplicated(["answer"])
    data = keep_rows(data, ~same_answer, "duplicate answer text", verbose)

    # question_id repeats across files (and even inside a file), so build
    # our own id: source + question_id, plus a counter if still repeated
    base_id = data["source"] + "_" + data["question_id"].astype(str)
    repeat = data.groupby(base_id).cumcount()
    data["doc_id"] = base_id.where(repeat == 0, base_id + "_" + repeat.astype(str))
    assert data["doc_id"].is_unique

    columns = ["doc_id", "source", "question_type", "question", "answer"]
    return data[columns].reset_index(drop=True)


if __name__ == "__main__":
    data = clean_medquad()
    print(f"Clean rows: {len(data)}\n")
    print(data["source"].value_counts())
    print("\nExample row:")
    print(data.iloc[0].to_dict())
