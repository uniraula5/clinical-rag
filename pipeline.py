"""
The full question-answering pipeline:

  1. retrieve - hybrid search finds the 4 most relevant answers (hybrid_search.py)
  2. answer   - the LLM writes a cited answer from those sources only (answer.py)
  3. check    - citations and every claim are checked against the sources (verify.py)
  4. revise   - if the check found problems, the LLM fixes them once and it's checked again

Hybrid search (semantic + keyword) is used because it scored best on the retrieval
eval: 0.94 hit@5 against 0.83 for semantic search alone, mostly because semantic
search misses gene symbols like PAH.

Four sources are sent to the LLM. With three, hybrid search only found a correct
answer for 62% of the plain-language questions, because one wrong keyword result
can take a slot. A fourth slot lifts that to 81% and gene symbols to 100%, and
five sources score no better while costing more tokens (run `python evaluate.py`
to see the hit@k table this came from).

The same question is answered from memory the second time, so clicking an example
twice in the demo does not spend tokens again.

    python pipeline.py "How is Wilson disease treated?"
"""

import sys

from answer import format_sources, generate_answer, revise_answer
from clean_data import clean_medquad
from hybrid_search import hybrid_search
from keyword_search import KeywordIndex
from search import get_collection
from verify import verify_answer

N_SOURCES = 4
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
    def __init__(self, collection=None, keyword_index=None):
        self.collection = collection if collection is not None else get_collection()
        self.keyword_index = keyword_index if keyword_index is not None else KeywordIndex()
        # full answer text for every doc_id, so the LLM reads more than one 100-word chunk
        data = clean_medquad(verbose=False)
        self.full_answers = dict(zip(data["doc_id"], data["answer"]))
        self.cache = {}

    def full_answer_for(self, doc_id):
        if doc_id not in self.full_answers:
            # the index was built from a different version of the data, so a
            # bare KeyError here would say nothing about how to fix it
            raise RuntimeError(
                f"The index has an answer ({doc_id}) that is not in the cleaned data. "
                "Rebuild it with: python build_index.py"
            )
        return self.full_answers[doc_id]

    def get_sources(self, question):
        results = hybrid_search(question, n_results=N_SOURCES, collection=self.collection,
                                keyword_index=self.keyword_index)
        return [
            {
                "number": i,
                "doc_id": r["doc_id"],
                "source": r["source"],
                "question": r["question"],
                "score": r["score"],
                "found_by": r.get("found_by", []),
                "text": passage_for(self.full_answer_for(r["doc_id"]), r["text"]),
            }
            for i, r in enumerate(results, start=1)
        ]

    def ask(self, question, use_cache=True):
        key = question.strip().lower()
        if use_cache and key in self.cache:
            return self.cache[key]

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

        result = {
            "question": question,
            "answer": answer,
            "passed": check["passed"],
            "sources": sources,
            "steps": steps,
        }
        self.cache[key] = result
        return result


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
