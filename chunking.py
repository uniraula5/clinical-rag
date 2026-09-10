"""
Splits long answers into smaller, overlapping chunks before embedding.

The embedding model (all-MiniLM-L6-v2) only reads the first 256 tokens of
a text and silently ignores the rest. Without chunking, a long answer would
be searched using only its first paragraph.
"""

import re

import pandas as pd

from clean_data import clean_medquad

# checked with the model's tokenizer: at 100 words, 99.8% of chunks (question
# included) fit in the 256 token limit. At 150 words, 3.7% got cut off.
MAX_WORDS = 100
OVERLAP_WORDS = 20   # words carried over from the end of the previous chunk


def split_sentences(paragraph):
    # break after . ! or ? when a space follows
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s]


def split_into_units(answer):
    # pieces no longer than MAX_WORDS: whole paragraphs when they fit,
    # otherwise sentences, otherwise a hard cut every MAX_WORDS words
    units = []
    for paragraph in answer.split("\n\n"):
        if len(paragraph.split()) <= MAX_WORDS:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            words = sentence.split()
            for start in range(0, len(words), MAX_WORDS):
                units.append(" ".join(words[start:start + MAX_WORDS]))
    return units


def chunk_answer(answer):
    chunks = []
    current = []
    for unit in split_into_units(answer):
        unit_words = unit.split()
        if current and len(current) + len(unit_words) > MAX_WORDS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_WORDS:]
            # if the overlap plus the next unit is still too long, drop the overlap
            if len(current) + len(unit_words) > MAX_WORDS:
                current = []
        current = current + unit_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(data):
    rows = []
    for row in data.itertuples(index=False):
        pieces = chunk_answer(row.answer)
        for i, piece in enumerate(pieces):
            rows.append({
                "chunk_id": f"{row.doc_id}#{i}",
                "doc_id": row.doc_id,
                "source": row.source,
                "question_type": row.question_type,
                "question": row.question,
                "chunk_index": i,
                "num_chunks": len(pieces),
                # the question goes in every chunk so a chunk from the middle
                # of an answer still says which disease it is about
                "text": f"Question: {row.question}\nAnswer: {piece}",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = clean_medquad(verbose=False)
    chunks = build_chunks(data)

    print(f"Documents: {len(data)}")
    print(f"Chunks:    {len(chunks)}\n")

    per_doc = chunks.groupby("doc_id").size()
    print("Chunks per document:")
    print(per_doc.describe(percentiles=[0.5, 0.9, 0.99]).round(1))

    words = chunks["text"].str.split().str.len()
    print("\nWords per chunk (including the question):")
    print(words.describe(percentiles=[0.5, 0.9, 0.99]).round(0))

    example = chunks[chunks["num_chunks"] == 3].iloc[:3]
    print("\nExample: one answer split into 3 chunks")
    for text in example["text"]:
        print("-" * 60)
        print(text[:300] + ("..." if len(text) > 300 else ""))
