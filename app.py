"""
Simple web demo: type a health question, get the closest answers from NIH
and CDC sources, each labeled with where it came from.

    python app.py
Then open the local link it prints (usually http://127.0.0.1:7860).
Build the index first with: python build_index.py
"""

import gradio as gr

from search import get_collection, semantic_search

EXAMPLE_QUESTIONS = [
    "What are the early signs of type 2 diabetes?",
    "How is high blood pressure treated?",
    "Is Huntington disease inherited?",
    "What causes kidney stones?",
    "What are the symptoms of a stroke?",
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

# load the index and the embedding model once when the app starts
collection = get_collection()


def format_results(results):
    if not results:
        return "No results found."
    blocks = []
    for r in results:
        answer = r["text"].split("\nAnswer: ", 1)[-1]
        source = SOURCE_NAMES.get(r["source"], r["source"])
        blocks.append(
            f"### {r['rank']}. {r['question']}\n"
            f"**Source:** {source} · **Type:** {r['question_type']} · "
            f"**Similarity:** {r['score']:.2f} · `{r['doc_id']}`\n\n"
            f"> {answer}\n"
        )
    return "\n---\n".join(blocks)


def run_search(query, n_results):
    query = (query or "").strip()
    if not query:
        return "Type a question to search."
    results = semantic_search(query, n_results=int(n_results), collection=collection)
    return format_results(results)


with gr.Blocks(title="Clinical Q&A Search") as demo:
    gr.Markdown(
        "# Clinical Q&A Search\n"
        "Searches 15,795 answers from NIH and CDC health websites (the MedQuAD "
        "dataset) by meaning, not just exact words. Every result shows its source.\n\n"
        "_Research demo only. Not medical advice._"
    )
    with gr.Row():
        query_box = gr.Textbox(
            label="Question",
            placeholder="e.g. What are the symptoms of diabetes?",
            scale=4,
        )
        n_slider = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)
    search_button = gr.Button("Search", variant="primary")
    gr.Examples(EXAMPLE_QUESTIONS, inputs=query_box)
    output = gr.Markdown()

    search_button.click(run_search, inputs=[query_box, n_slider], outputs=output)
    query_box.submit(run_search, inputs=[query_box, n_slider], outputs=output)


if __name__ == "__main__":
    demo.launch()
