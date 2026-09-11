"""
Checks the full pipeline's answers (not just retrieval) on 8 of the 32 test
questions: did the right source get retrieved, did the first draft pass the
fact check, how often a revision was needed, and did the final answer pass.

LLM output varies a little between runs, so treat these as rough numbers.
Also note the same model writes and checks the answers, so this measures
self-consistency with the sources, not medical correctness.

    python evaluate_answers.py
"""

import json
import time
from pathlib import Path

from evaluate import load_test_cases
from pipeline import QAPipeline

RESULTS_PATH = Path(__file__).parent / "eval" / "answer_results.json"


def supported_share(step):
    # share of sentences the checker marked supported (None for a "no answer" reply)
    if not step["claims"]:
        return None
    return sum(c["supported"] for c in step["claims"]) / len(step["claims"])


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    cases = load_test_cases()[::4]  # every 4th question: 8 of 32, from both categories
    pipeline = QAPipeline()

    records = []
    for case in cases:
        start = time.time()
        result = pipeline.ask(case["query"])
        draft, final = result["steps"][0], result["steps"][-1]
        retrieved_ids = {s["doc_id"] for s in result["sources"]}
        record = {
            "id": case["id"],
            "query": case["query"],
            "correct_source_retrieved": bool(retrieved_ids & set(case["relevant_doc_ids"])),
            "draft_passed": draft["passed"],
            "revised": len(result["steps"]) > 1,
            "final_passed": final["passed"],
            "draft_supported_share": supported_share(draft),
            "final_supported_share": supported_share(final),
            "seconds": round(time.time() - start, 1),
            "final_answer": result["answer"],
            "draft_problems": draft["problems"],
            "final_problems": final["problems"],
        }
        records.append(record)
        draft_status = "pass" if draft["passed"] else "FAIL"
        final_status = "pass" if final["passed"] else "FAIL"
        print(f"[{case['id']}] draft {draft_status} -> final {final_status} ({record['seconds']}s)  {case['query']}")

    n = len(records)
    print(f"\nQuestions:                          {n}")
    print(f"Correct source among the 5 used:    {sum(r['correct_source_retrieved'] for r in records)}/{n}")
    print(f"First draft passed the check:       {sum(r['draft_passed'] for r in records)}/{n}")
    print(f"Needed a revision:                  {sum(r['revised'] for r in records)}/{n}")
    print(f"Final answer passed the check:      {sum(r['final_passed'] for r in records)}/{n}")
    print(f"Supported sentences, first draft:   {average(r['draft_supported_share'] for r in records):.0%}")
    print(f"Supported sentences, final answer:  {average(r['final_supported_share'] for r in records):.0%}")

    RESULTS_PATH.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nSaved details to {RESULTS_PATH}")
