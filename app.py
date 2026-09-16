"""
Web demo with two tabs:
  Ask    - a short answer written only from NIH/CDC sources, with a citation on
           every sentence, fact-checked before it's shown
  Search - try semantic, keyword and hybrid search side by side

    python app.py
Then open the local link it prints (usually http://127.0.0.1:7860).
Needs the index (python build_index.py) and, for the Ask tab, a Groq key in .env.
"""

import gradio as gr
from openai import RateLimitError

from hybrid_search import hybrid_search
from keyword_search import KeywordIndex
from pipeline import QAPipeline
from search import get_collection, semantic_search
from verify import is_no_answer

EXAMPLE_QUESTIONS = [
    "What are the early signs of type 2 diabetes?",
    "How is high blood pressure treated?",
    "Is Huntington disease inherited?",
    "What causes kidney stones?",
    "What are the symptoms of a stroke?",
    # a gene symbol: keyword search finds this one, semantic search alone does not
    "What condition is linked to mutations in the PAH gene?",
]

SOURCE_NAMES = {
    "GARD": "Genetic and Rare Diseases Information Center (NIH)",
    "GHR": "Genetics Home Reference (NIH)",
    "NIDDK": "National Institute of Diabetes and Digestive and Kidney Diseases",
    "NINDS": "National Institute of Neurological Disorders and Stroke",
    "MPlus_Health_Topics": "MedlinePlus Health Topics (NIH)",
    "SeniorHealth": "NIHSeniorHealth",
    "CancerGov": "National Cancer Institute",
    "NHLBI": "National Heart, Lung, and Blood Institute",
    "CDC": "Centers for Disease Control and Prevention",
}

# load the index, keyword index and full answers once when the app starts
collection = get_collection()
keyword_index = KeywordIndex()
# pass the keyword index in, or QAPipeline builds a second copy of the same
# 48,978-chunk BM25 index (about 2.5 seconds and a few hundred MB wasted)
pipeline = QAPipeline(collection=collection, keyword_index=keyword_index)

SEARCH_METHODS = {
    "Semantic": lambda q, k: semantic_search(q, n_results=k, collection=collection),
    "Keyword (BM25)": lambda q, k: keyword_index.search(q, k),
    "Hybrid": lambda q, k: hybrid_search(q, n_results=k, collection=collection, keyword_index=keyword_index),
}

# the three methods score on scales that have nothing to do with each other, so
# the number is labelled with the scale it is on instead of a bare "Score"
SCORE_LABELS = {
    "Semantic": "Cosine similarity",
    "Keyword (BM25)": "BM25 score",
    "Hybrid": "RRF score",
}


def source_name(code):
    return SOURCE_NAMES.get(code, code)


# ---------- Ask tab ----------

def found_by(source):
    # hybrid search records which method(s) found each answer
    methods = source.get("found_by")
    return f" · found by {' + '.join(methods)}" if methods else ""


def format_sources(sources):
    blocks = []
    for s in sources:
        excerpt = s["text"] if len(s["text"]) <= 600 else s["text"][:600] + "..."
        blocks.append(
            f"**[{s['number']}] {s['question']}**  \n"
            f"{source_name(s['source'])} · RRF score {s['score']:g}{found_by(s)} · `{s['doc_id']}`\n\n"
            f"> {excerpt}"
        )
    return "\n\n".join(blocks)


def format_check(result):
    lines = []
    for step in result["steps"]:
        label = "First draft" if step["step"] == "draft" else "After revision"
        if step["passed"]:
            lines.append(f"**{label}: passed.** Every sentence is cited and supported by its source.")
        else:
            lines.append(f"**{label}: {len(step['problems'])} problem(s) found**")
            lines += [f"- {p}" for p in step["problems"]]
        lines.append("")
    if len(result["steps"]) > 1:
        lines.append("_The checker's problems were sent back to the writer, which revised the answer once._")
    return "\n".join(lines)


def sources_or_nothing(question):
    # the answer already failed; if search fails too, say so instead of
    # raising a second error out of the error handler
    try:
        return format_sources(pipeline.get_sources(question))
    except Exception as error:
        return f"_Search could not run either ({type(error).__name__}). Try `python check_setup.py`._"


def run_ask(question):
    question = (question or "").strip()
    if not question:
        return "Type a question.", "", ""
    try:
        result = pipeline.ask(question)
    except RateLimitError:
        # the free tier allows 8,000 tokens a minute and 200,000 a day
        message = ("**Groq's free limit was reached.** Wait a minute and try again, or try "
                   "tomorrow if the daily limit is used up. Search still works without the LLM, "
                   "and the sources this question found are below.")
        return message, "", sources_or_nothing(question)
    except Exception as error:
        # usually the key or the API address in .env; the error type is shown, never its text
        message = (f"**Couldn't write an answer ({type(error).__name__}).** Run "
                   "`python check_setup.py` to see whether .env is complete. "
                   "The sources search found are below.")
        return message, "", sources_or_nothing(question)

    if is_no_answer(result["answer"]):
        # a refusal passes the fact check because it claims nothing, so the
        # green "every sentence is supported" badge would be misleading here
        badge = "ℹ️ **No answer given.** Search found no source that answers this, so nothing was written."
    elif result["passed"]:
        badge = "✅ **Fact-checked:** every sentence is supported by the cited source."
    else:
        badge = "⚠️ **Some sentences could not be verified.** Check them against the sources below."
    return f"{result['answer']}\n\n{badge}", format_check(result), format_sources(result["sources"])


# ---------- Search tab ----------

def format_results(results, score_label="Score"):
    if not results:
        return "No results found. Try different words."
    blocks = []
    for r in results:
        answer = r["text"].split("\nAnswer: ", 1)[-1]
        found_by = f" · found by: {' + '.join(r['found_by'])}" if "found_by" in r else ""
        blocks.append(
            f"### {r['rank']}. {r['question']}\n"
            f"**Source:** {source_name(r['source'])} · **Type:** {r['question_type']} · "
            f"**{score_label}:** {r['score']:g}{found_by} · `{r['doc_id']}`\n\n"
            f"> {answer}\n"
        )
    return "\n---\n".join(blocks)


def run_search(query, method, n_results):
    query = (query or "").strip()
    if not query:
        return "Type a question to search."
    results = SEARCH_METHODS[method](query, int(n_results))
    return format_results(results, SCORE_LABELS[method])


# ---------- Layout ----------

with gr.Blocks(title="Clinical Q&A Search") as demo:
    gr.Markdown(
        "# Clinical Q&A\n"
        "Answers health questions using only 15,795 answers from NIH and CDC websites "
        "(the MedQuAD dataset). Every sentence cites its source, and a fact-checking step "
        "verifies each sentence against that source before the answer is shown.\n\n"
        "_Research demo only. Not medical advice._"
    )

    with gr.Tab("Ask"):
        ask_box = gr.Textbox(label="Question", placeholder="e.g. How is Wilson disease treated?")
        ask_button = gr.Button("Ask", variant="primary")
        gr.Examples(EXAMPLE_QUESTIONS, inputs=ask_box)
        answer_out = gr.Markdown()
        with gr.Accordion("Fact check details", open=False):
            check_out = gr.Markdown()
        with gr.Accordion("Sources", open=True):
            sources_out = gr.Markdown()
        ask_button.click(run_ask, inputs=ask_box, outputs=[answer_out, check_out, sources_out])
        ask_box.submit(run_ask, inputs=ask_box, outputs=[answer_out, check_out, sources_out])

    with gr.Tab("Search"):
        with gr.Row():
            query_box = gr.Textbox(label="Question", placeholder="e.g. HEXA gene", scale=4)
            n_slider = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)
        method_radio = gr.Radio(list(SEARCH_METHODS), value="Semantic", label="Search method")
        search_button = gr.Button("Search", variant="primary")
        gr.Examples(EXAMPLE_QUESTIONS, inputs=query_box)
        search_out = gr.Markdown()
        search_inputs = [query_box, method_radio, n_slider]
        search_button.click(run_search, inputs=search_inputs, outputs=search_out)
        query_box.submit(run_search, inputs=search_inputs, outputs=search_out)


if __name__ == "__main__":
    demo.launch()
