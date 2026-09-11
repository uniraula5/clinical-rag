# Clinical Q&A

Ask a health question and get a short answer written **only** from NIH and CDC
sources, with a citation on every sentence and a fact-check step that runs
before the answer is shown.

This is a retrieval-augmented generation (RAG) system built on
[MedQuAD](https://github.com/abachaa/MedQuAD), a public medical Q&A dataset.
The goal is answers you can trace back to trusted clinical text, instead of
whatever a language model happens to remember.

## How it works

```
question
  -> semantic search (search.py)             top 5 of 15,795 answers
  -> write a cited answer (answer.py)        the LLM may only use those 5 sources
  -> fact-check it (verify.py)               valid citations? every sentence supported?
  -> problems found? revise once (pipeline.py)
  -> answer + fact-check result + sources (app.py)
```

Index built once, ahead of time:

```
data/raw/*.csv        12 MedQuAD source files          47,457 rows
  -> clean_data.py    drop empty/duplicate answers     15,795 Q&A pairs
  -> chunking.py      split into <=100-word chunks     48,978 chunks
  -> build_index.py   embed locally, store in Chroma   48,978 vectors
```

## Results

### Retrieval: 32 curated test questions (`python evaluate.py`)

16 questions are phrased the way a patient would ask ("keep bones from
breaking"). 16 use exact condition names with near-identical traps
("Wolff-Parkinson-White" vs. Parkinson disease, "Aicardi-Goutieres type 3" vs.
types 1, 2, 4, 5).

| Method | hit@5 | MRR | Exact names (hit@5 / MRR) | Patient phrasing (hit@5 / MRR) |
|---|---|---|---|---|
| **Semantic** | **0.91** | **0.75** | 1.00 / 0.90 | 0.81 / 0.60 |
| Keyword (BM25) | 0.78 | 0.58 | 0.88 / 0.66 | 0.69 / 0.49 |
| Hybrid (RRF) | 0.91 | 0.69 | 1.00 / 0.88 | 0.81 / 0.51 |

hit@5 = share of questions with a correct answer in the top 5.
MRR = mean of 1/rank of the first correct answer.

**The hypothesis was that hybrid search would beat semantic search on exact
clinical names. It didn't.** Semantic search got all 16 exact-name questions.
Each chunk is embedded together with its question title, which already anchors
the vector to the condition name. Equal-weight fusion helped on a few
questions but pulled good semantic results down when keyword search was wrong.
So the answer pipeline uses semantic search, and hybrid stays available in the
app's Search tab.

A small follow-up probe suggests keyword search still matters for identifiers
that don't appear in question titles, such as gene symbols: "HEXA gene" was
missed by semantic search but ranked first by keyword search.

### Answers: 8 of the test questions (`python evaluate_answers.py`)

| | |
|---|---|
| Correct source among the 5 used | 7 / 8 |
| First draft passed the fact check | 4 / 8 |
| Needed a revision | 4 / 8 |
| Final answer passed the fact check | 8 / 8 |
| Supported sentences, first draft → final | 77% → 100% |

Typical catch: asked about wheezing and chest tightness, the first draft said
these are "typical of asthma" and cited two sources that describe asthma but
never list those symptoms. The checker flagged it, and the revision kept only
what the sources say. Per-question details are in `eval/answer_results.json`.

## Setup

Requires Python 3.11 and a free [Groq](https://console.groq.com) API key.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then paste your Groq key into .env
python download_data.py      # MedQuAD CSVs -> data/raw/
python build_index.py        # about 4 minutes, one time
```

## Usage

```bash
python app.py                                         # web demo: Ask + Search tabs
python pipeline.py "How is Wilson disease treated?"   # full pipeline in the terminal
python search.py "symptoms of leukemia"               # semantic search only
python hybrid_search.py "HEXA gene"                   # semantic + keyword search
python evaluate.py                                    # retrieval eval
python evaluate_answers.py                            # answer eval (calls the LLM)
```

## Project files

| File | Purpose |
|---|---|
| `download_data.py` | Downloads the MedQuAD CSVs (pinned to one commit) |
| `load_data.py` | Loads all sources into one table and profiles it |
| `clean_data.py` | Removes unusable rows, assigns unique IDs |
| `chunking.py` | Splits long answers into overlapping chunks |
| `build_index.py` | Embeds chunks locally and saves them to ChromaDB |
| `search.py` | Semantic search, one best chunk per answer |
| `keyword_search.py` | BM25 keyword search over the same chunks |
| `hybrid_search.py` | Merges both with Reciprocal Rank Fusion |
| `llm.py` | Groq API client (OpenAI-compatible) |
| `answer.py` | Writes and revises cited answers from sources only |
| `verify.py` | Citation check (code) + claim check (LLM) |
| `pipeline.py` | Retrieve → answer → check → revise once |
| `app.py` | Gradio web demo |
| `evaluate.py` | Retrieval eval: hit@5 and MRR for all three search methods |
| `evaluate_answers.py` | Answer eval: fact-check pass rates |
| `eval/test_cases.json` | 32 test questions with their correct answer IDs |

## Design decisions

- **Only 15,795 of 47,457 rows are usable.** MedQuAD's authors removed the
  answers from its three MedlinePlus sources for copyright reasons.
- **Chunk size was measured, not guessed.** The embedding model silently
  ignores anything past 256 tokens. At 150 words per chunk, 3.7% of chunks were
  cut off; at 100 words, 99.8% fit.
- **Embeddings run locally** (`all-MiniLM-L6-v2`). Only the question and the 5
  retrieved public passages are sent to the LLM.
- **The LLM reads up to 350 words per source,** centered on the chunk that
  matched, not just the 100-word chunk, so it has enough context to answer.
- **Two layers of checking.** Plain code checks that every sentence has a
  citation and that each citation number exists. The LLM then checks whether
  each sentence is actually supported by the source it cites.
- **Citations are normalized in code.** The model sometimes writes its own
  citation style (`【4†L1-L4】`) instead of `[4]`, which made every sentence
  fail the citation check.
- **Revision is capped at one round,** so a question can't loop forever. If the
  revised answer still fails, it's shown with a warning, not hidden.
- **Temperature 0** for the most repeatable output.

## Limitations

- **The same model writes and checks answers.** The check measures whether
  answers stick to their sources, not whether the sources are medically
  current. The checker can also be wrong in either direction.
- **The eval sets are small and curated** (32 retrieval questions, 8 answer
  questions). One question moves hit@5 by about 3 points. LLM results vary a
  little between runs.
- **Answers are slow on Groq's free tier:** 28 to 113 seconds per question in
  testing, likely mostly waiting on rate limits. A revision adds two more LLM calls.
- Hybrid search uses equal weights. It wasn't tuned, to avoid fitting the test set.
- About 0.2% of chunks are still longer than the embedding model's token limit.
- MedQuAD content was collected around 2017 and may be out of date.
- **Research demo only. Not medical advice.**

## Data credit

Ben Abacha, A. & Demner-Fushman, D. (2019). A question-entailment approach to
question answering. *BMC Bioinformatics*, 20, 511.
CSV conversion by [avery-lockwood/MedQuAD-CSVs](https://github.com/avery-lockwood/MedQuAD-CSVs).
