"""
Writes an answer to a health question using ONLY the retrieved sources, with
a citation like [2] on every sentence. Also rewrites an answer when the
fact-checker (verify.py) finds problems.
"""

import re

from llm import chat

NO_ANSWER = "The sources I found don't answer this question."

ANSWER_RULES = f"""You answer health questions for the general public using ONLY the numbered sources you are given.

Rules:
- End every sentence with the citation(s) it came from, in square brackets like [1] or [2][3].
  Do not use any other citation format.
- Use only what the sources actually say. Do not add outside medical knowledge, even if you know it is true.
- If the sources do not answer the question, reply with exactly: {NO_ANSWER}
- Write 3 to 6 sentences in plain language.
- Do not diagnose anyone or tell the reader what to do about their own health.
- Do not add headings, bullet points, a reference list, or a disclaimer."""


def format_sources(sources):
    # the numbered list the LLM cites from: [1] question (source), then the passage
    return "\n\n".join(
        f"[{s['number']}] {s['question']} (source: {s['source']})\n{s['text']}" for s in sources
    )


def clean_citations(text):
    # some models write citations like 【4†L1-L4】 instead of [4], so convert them
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    # join everything into one paragraph so every sentence gets checked the same way
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def generate_answer(question, sources_text):
    return clean_citations(chat(ANSWER_RULES, f"Question: {question}\n\nSources:\n{sources_text}"))


def revise_answer(question, sources_text, draft, problems):
    problem_list = "\n".join(f"- {p}" for p in problems)
    prompt = (
        f"Question: {question}\n\nSources:\n{sources_text}\n\n"
        f"Your previous answer:\n{draft}\n\n"
        f"A fact-checker found these problems:\n{problem_list}\n\n"
        "Write a corrected answer. For each problem, fix the citation, rewrite the sentence "
        "so it only says what the source says, or remove the sentence. Keep the sentences "
        "that had no problems. Follow all the original rules."
    )
    return clean_citations(chat(ANSWER_RULES, prompt))
