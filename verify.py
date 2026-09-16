"""
Checks an answer against its sources before anyone sees it, in two ways:

1. check_citations - plain Python, no LLM. Every sentence must cite a source,
   and every citation number must point to a source that exists.
2. check_claims - the LLM acts as a strict fact-checker and decides, sentence
   by sentence, whether the cited sources really say what the sentence says.

The answer passes only if both checks find no problems.
"""

import re

from answer import NO_ANSWER
from llm import chat_json

CHECKER_RULES = """You are a strict fact-checker for medical answers.
You get numbered sources and a numbered list of sentences. Each sentence cites sources like [1].

For each sentence, decide if it is supported: every fact in it is stated in the source(s) it cites.
Mark it NOT supported if it adds details the sources don't have, makes a stronger or broader claim
than the source, or cites a source that doesn't contain that information.
Being true in general is not enough. It has to be in the cited source.

Reply with JSON only, in this shape:
{"results": [{"sentence": 1, "supported": true, "reason": "short reason"}]}"""


# Every sentence in an answer ends with a citation, so "[2]" or "[2]." is the
# reliable place to split. The capital-letter rule below is the backup for a
# sentence the writer forgot to cite. Splitting only on a capital letter used to
# glue "... [1]. 2.5 mg is typical [2]." into one sentence, which let an uncited
# sentence ride on the previous sentence's citation.
SENTENCE_END = re.compile(r"(?<=\])\s+|(?<=\]\.)\s+|(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text):
    parts = SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_citations(sentences, n_sources):
    problems = []
    for sentence in sentences:
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if not numbers:
            problems.append(f'No citation like [1]: "{sentence}"')
        bad = [n for n in numbers if n < 1 or n > n_sources]
        if bad:
            problems.append(f'Cites a source that does not exist {bad}: "{sentence}"')
    return problems


def check_claims(sentences, sources_text):
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    reply = chat_json(CHECKER_RULES, f"Sources:\n{sources_text}\n\nSentences:\n{numbered}")

    by_number = {}
    for item in reply.get("results", []):
        try:
            by_number[int(item["sentence"])] = item
        except (KeyError, TypeError, ValueError):
            continue

    claims = []
    for i, sentence in enumerate(sentences, start=1):
        item = by_number.get(i)
        if item is None:
            # the checker skipped this sentence, so it can't count as supported
            claims.append({"sentence": sentence, "supported": False, "reason": "not checked"})
            continue
        # guard against "false" coming back as a string, which bool() would treat as True
        supported = item.get("supported") in (True, "true", "True")
        claims.append({"sentence": sentence, "supported": supported, "reason": item.get("reason", "")})
    return claims


def is_no_answer(answer):
    normalize = lambda text: text.replace("’", "'").strip().rstrip(".")
    return normalize(answer) == normalize(NO_ANSWER)


def verify_answer(answer, sources_text, n_sources):
    if is_no_answer(answer):
        return {"passed": True, "claims": [], "problems": []}  # nothing claimed, nothing to check

    sentences = split_sentences(answer)
    if not sentences:
        # an empty reply used to pass: no sentences meant no problems to find
        return {"passed": False, "claims": [], "problems": ["The answer was empty."]}

    problems = check_citations(sentences, n_sources)
    claims = check_claims(sentences, sources_text)
    for c in claims:
        if not c["supported"]:
            problems.append(f'Not supported by its source: "{c["sentence"]}" ({c["reason"]})')
    return {"passed": not problems, "claims": claims, "problems": problems}
