"""
The full question-answering pipeline:

  1. retrieve - semantic search finds the 3 most relevant answers (search.py)
  2. answer   - the LLM writes a cited answer from those sources only (answer.py)
  3. check    - citations and every claim are checked against the sources (verify.py)
  4. revise   - if the check found problems, the LLM fixes them once and it's checked again

Semantic search is used because it scored best on the retrieval eval (evaluate.py).
Only its top 3 are used: every correct answer it found in the eval was already in
the top 3, and fewer, shorter sources keep each LLM call well under Groq's free-tier
limit of 8,000 tokens per minute.

    python pipeline.py "How is Wilson disease treated?"
"""

import sys

from answer import format_sources, generate_answer, revise_answer
from clean_data import clean_medquad
from search import get_collection, semantic_search
from verify import verify_answer

N_SOURCES = 3
MAX_WORDS_PER_SOURCE = 350
MAX_REVISIONS = 1


def find_position(words, target):
    # index where the list `target` starts inside the list `words`, or None
    for i in range(len(words) - len(target) + 1):
        if words[i:i + len(target)] == target:
            return i
    return None


def passage_for(full_answer, chunk_text, max_words=MAX_WORDS_PER_SOURCE):
    # short answers are used whole. For long ones, take max_words centered on
    # the chunk that matched, so the LLM sees the relevant part plus context.
    words = full_answer.split()
    if len(words) <= max_words:
        return full_answer
    chunk_words = chunk_text.split("\nAnswer: ", 1)[-1].split()
    start = find_position(words, chunk_words[:8]) or 0
    begin = start - (max_words - len(chunk_words)) // 2
    begin = max(0, min(begin, len(words) - max_words))
    passage = " ".join(words[begin:begin + max_words])
    if begin > 0:
        passage = "... " + passage
    if begin + max_words < len(words):
        passage += " ..."
    return passage


class QAPipeline:
    def __init__(self, collection=None):
        self.collection = collection if collection is not None else get_collection()
        # full answer text for every doc_id, so the LLM reads more than one 100-word chunk
        data = clean_medquad(verbose=False)
        self.full_answers = dict(zip(data["doc_id"], data["answer"]))

    def get_sources(self, question):
        results = semantic_search(question, n_results=N_SOURCES, collection=self.collection)
        return [
            {
                "number": i,
                "doc_id": r["doc_id"],
                "source": r["source"],
                "question": r["question"],
                "score": r["score"],
                "text": passage_for(self.full_answers[r["doc_id"]], r["text"]),
            }
            for i, r in enumerate(results, start=1)
        ]

    def ask(self, question):
        sources = self.get_sources(question)
        sources_text = format_sources(sources)

        answer = generate_answer(question, sources_text)
        check = verify_answer(answer, sources_text, len(sources))
        steps = [{"step": "draft", "answer": answer, **check}]

        for _ in range(MAX_REVISIONS):
            if check["passed"]:
                break
            answer = revise_answer(question, sources_text, answer, check["problems"])
            check = verify_answer(answer, sources_text, len(sources))
            steps.append({"step": "revision", "answer": answer, **check})

        return {
            "question": question,
            "answer": answer,
            "passed": check["passed"],
            "sources": sources,
            "steps": steps,
        }


def print_result(result):
    print(f"\nQuestion: {result['question']}\n")
    for step in result["steps"]:
        status = "passed" if step["passed"] else f"{len(step['problems'])} problem(s)"
        print(f"--- {step['step']} ({status}) ---")
        print(step["answer"])
        for problem in step["problems"]:
            print(f"  ! {problem}")
        print()
    print("Sources:")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['source']} | {s['question']} ({s['doc_id']})")
    if not result["passed"]:
        print("\nWarning: the final answer still has claims the checker could not verify.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python pipeline.py "your question"')
        sys.exit(1)
    print_result(QAPipeline().ask(" ".join(sys.argv[1:])))
