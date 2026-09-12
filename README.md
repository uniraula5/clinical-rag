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
  -> semantic search (search.py)             top 3 of 15,795 answers
  -> write a cited answer (answer.py)        the LLM may only use those 3 sources
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

### Retrieval: 48 curated test questions (`python evaluate.py`)

Three kinds of question, 16 each:
- **Patient phrasing:** everyday wording ("keep bones from breaking").
- **Exact names:** condition names with near-identical traps
  ("Wolff-Parkinson-White" vs. Parkinson disease, "Aicardi-Goutieres type 3"
  vs. types 1, 2, 4, 5).
- **Gene symbols:** only a gene symbol that appears in answer text but never
  in a question title ("What condition is linked to mutations in the PAH gene?").

| Method | All 48 (hit@5 / MRR) | Patient phrasing | Exact names | Gene symbols |
|---|---|---|---|---|
| Semantic | 0.83 / 0.67 | **0.81 / 0.60** | **1.00 / 0.90** | 0.69 / 0.52 |
| Keyword (BM25) | 0.85 / 0.68 | 0.69 / 0.49 | 0.88 / 0.66 | **1.00 / 0.89** |
| **Hybrid (RRF)** | **0.94 / 0.75** | 0.81 / 0.51 | 1.00 / 0.88 | 1.00 / 0.86 |

hit@5 = share of questions with a correct answer in the top 5.
MRR = mean of 1/rank of the first correct answer.

**Exact names:** the hypothesis was that hybrid search would beat semantic
search here. It didn't. Semantic search got all 16, because each chunk is
embedded together with its question title, which anchors the vector to the
condition name.

**Gene symbols:** this is where keyword search matters. Semantic search found
a correct answer for only 11 of 16; keyword and hybrid search found all 16. A
rare token like `PAH` or `F9` means little to a small embedding model, but
it's exactly what BM25 weights most. This category was added after the first
32 questions showed no hybrid benefit, to test identifiers directly. The 6
genes from an earlier informal check were left out, and correct answers were
labeled by a fixed rule: "genetic changes" or "causes" answers containing
"<SYMBOL> gene".

**Overall, hybrid search is best** (0.94 hit@5 vs. 0.83 for semantic). It keeps
semantic search's strength on phrasing and names and keyword search's strength
on identifiers. On patient phrasing its MRR is still lower than semantic alone
(0.51 vs. 0.60), because equal-weight fusion can push a good semantic result
down when keyword search is wrong.

The answer pipeline still uses semantic search (chosen from the first 32
questions), so gene-symbol questions in the Ask tab currently get weaker
sources. Switching the pipeline to hybrid search is the next step.
Per-question ranks for every method are in `eval/retrieval_results.csv`.

### Answers: 8 of the test questions (`python evaluate_answers.py`)

One run per setting. The current setting gives the LLM 3 sources instead of 5,
to stay under Groq's free-tier token limits (see Design decisions).

| | 5 sources (first version) | **3 sources (current)** |
|---|---|---|
| Correct source among those used | 7 / 8 | 7 / 8 |
| First draft passed the fact check | 4 / 8 | 5 / 8 |
| Needed a revision | 4 / 8 | 3 / 8 |
| Final answer passed the fact check | 8 / 8 | 6 / 8 |
| Supported sentences, first draft → final | 77% → 100% | 74% → 93% |

With 8 questions, one question moves a row by 1/8, and the same question can
pass on one run and fail on the next, so the two columns are within
run-to-run noise. In testing, the failures that remain are small over-reaches
the strict checker catches, such as adding "which often involves needles" to a
source that only says "injected illegal drugs".

Typical catch: asked about wheezing and chest tightness, the first draft said
these are "typical of asthma" and cited two sources that describe asthma but
never list those symptoms. The checker flagged it, and the revision kept only
what the sources say. Per-question details from the 5-source run are in
`eval/answer_results.json`.

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

## Tests

```bash
python -m pytest
```

46 tests, about 2 seconds. They run offline, with no API key, no index and
no LLM calls: the LLM and the database are replaced with small fakes. They
cover the cleaning rules, the 100-word chunk limit and overlap, one result
per answer, BM25 and rank-fusion scoring, the citation and claim checks, the
one-revision cap, and the eval math. One test also confirms that every
labeled answer in `eval/test_cases.json` exists in the cleaned data (it's
skipped if the data hasn't been downloaded).

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
| `eval/test_cases.json` | 48 test questions with their correct answer IDs |
| `eval/retrieval_results.csv` | Rank of the first correct answer per question, per method |
| `tests/` | Offline pytest suite (fake LLM and fake database) |
| `conftest.py` | Lets the tests import the project's modules |

## Design decisions

- **Only 15,795 of 47,457 rows are usable.** MedQuAD's authors removed the
  answers from its three MedlinePlus sources for copyright reasons.
- **Chunk size was measured, not guessed.** The embedding model silently
  ignores anything past 256 tokens. At 150 words per chunk, 3.7% of chunks were
  cut off; at 100 words, 99.8% fit.
- **Embeddings run locally** (`all-MiniLM-L6-v2`). Only the question and the 3
  retrieved public passages are sent to the LLM.
- **The LLM reads 3 sources of up to 350 words each,** centered on the chunk
  that matched, not just the 100-word chunk. The first version sent 5 sources
  (~1,600 prompt tokens per call). Groq's free tier allows 8,000 tokens per
  minute, so a question needing a revision (4 calls) hit the limit and waited on
  retries. 3 sources is ~1,050 tokens per call, with no loss in retrieval on the
  first 32 eval questions (hit@3 = hit@5).
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
- **The eval sets are small and curated** (48 retrieval questions, 8 answer
  questions). One question moves a category's hit@5 by about 6 points. LLM results vary a
  little between runs.
- **Groq's free tier limits how much it can be used.** For this model it allows
  8,000 tokens per minute and 200,000 per day. One question uses roughly 3,000
  to 6,000 tokens: a single question takes about 8 seconds, but several in a row
  wait on rate-limit retries (4 to 53 seconds each in the eval), and the daily
  limit allows a few dozen questions. Running the answer eval several times in
  one day used up the daily limit.
- Hybrid search uses equal weights. It wasn't tuned, to avoid fitting the test set.
- About 0.2% of chunks are still longer than the embedding model's token limit.
- MedQuAD content was collected around 2017 and may be out of date.
- **Research demo only. Not medical advice.**

## Data credit

Ben Abacha, A. & Demner-Fushman, D. (2019). A question-entailment approach to
question answering. *BMC Bioinformatics*, 20, 511.
CSV conversion by [avery-lockwood/MedQuAD-CSVs](https://github.com/avery-lockwood/MedQuAD-CSVs).
