"""
Checks the full pipeline's answers (not just retrieval) on a quarter of the test
questions: did the right source get retrieved, did the first draft pass the
fact check, how often a revision was needed, and did the final answer pass.

LLM output varies a little between runs, so treat these as rough numbers.
Also note the same model writes and checks the answers, so this measures
self-consistency with the sources, not medical correctness.

    python evaluate_answers.py
"""

import argparse
import json
import time
from pathlib import Path

from evaluate import load_test_cases
from pipeline import QAPipeline

RESULTS_PATH = Path(__file__).parent / "eval" / "answer_results.json"
HOLDOUT_PATH = Path(__file__).parent / "eval" / "answer_results_holdout.json"
STEP = 4  # take every 4th question, so the sample covers all three categories


def select_cases(cases, offset):
    """Every 4th question starting at `offset`. Offset 0 is the set used while
    writing the prompts; any other offset gives questions those never saw."""
    return cases[offset::STEP]


def results_path_for(offset):
    return RESULTS_PATH if offset == 0 else HOLDOUT_PATH


def supported_share(step):
    # share of sentences the checker marked supported (None for a "no answer" reply)
    if not step["claims"]:
        return None
    return sum(c["supported"] for c in step["claims"]) / len(step["claims"])


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def shares(records, field):
    """The supported-sentence shares that exist. A refusal has no sentences to
    score, so it is None here - and that makes the draft and final averages run
    over different numbers of answers, which the summary prints out loud."""
    return [r[field] for r in records if r[field] is not None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, choices=range(STEP),
                        help="0 = the questions used while writing the prompts, "
                             "2 = a held-out set the prompts never saw. Offsets "
                             "stop at 3 because every 4th question repeats after that.")
    args = parser.parse_args()

    cases = select_cases(load_test_cases(), args.offset)
    results_path = results_path_for(args.offset)
    print(f"{len(cases)} questions (offset {args.offset}): {', '.join(c['id'] for c in cases)}\n")
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
    draft_shares = shares(records, "draft_supported_share")
    final_shares = shares(records, "final_supported_share")

    def line(label, value):
        print(f"  {label:34s}{value}")

    print()
    line("Questions:", n)
    line("Correct source among those used:", f"{sum(r['correct_source_retrieved'] for r in records)}/{n}")
    line("First draft passed the check:", f"{sum(r['draft_passed'] for r in records)}/{n}")
    line("Needed a revision:", f"{sum(r['revised'] for r in records)}/{n}")
    line("Final answer passed the check:", f"{sum(r['final_passed'] for r in records)}/{n}")
    line("Supported sentences, first draft:", f"{average(draft_shares):.0%} of {len(draft_shares)} answers")
    line("Supported sentences, final answer:", f"{average(final_shares):.0%} of {len(final_shares)} answers")

    results_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nSaved details to {results_path}")
