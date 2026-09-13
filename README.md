# Clinical Q&A

Ask a health question and get a short answer built only from NIH and CDC
sources. Every sentence has a citation like `[2]`, and the answer is
fact-checked against those sources before you see it.

The data is [MedQuAD](https://github.com/abachaa/MedQuAD), a public set of
medical questions and answers from government health websites. Nothing here
is medical advice.

## What it looks like

```bash
python pipeline.py "How is Wilson disease treated?"
```

```
--- draft (1 problem(s)) ---
Wilson disease has no cure, but treatments focus on lowering the copper that
builds up in the body [1][3]. People usually need to take medications that help
remove excess copper and may also follow a low-copper diet, and these measures
must be continued for life [1]. ...
  ! Not supported by its source: "...may also follow a low-copper diet..." [1]
    (Source mentions medications and dietary modifications, but does not
    explicitly specify a low-copper diet.)

--- revision (passed) ---
Wilson disease has no cure, but treatments aim to reduce or control the copper
that builds up in the body [1][3]. Treatment may include certain medications and
dietary modifications, and it must be continued for life [1]. If treatment is not
effective or liver failure develops, a liver transplant may be necessary [1].
Without treatment, Wilson disease can cause brain damage, liver failure, and
death, so lifelong therapy is required [2].

Sources:
  [1] GARD  | What are the treatments for Wilson disease?
  [2] NINDS | What is the outlook for Wilson Disease?
  [3] GARD  | What is (are) Wilson disease?
```

The first answer said "low-copper diet", which the source doesn't actually say.
The checker caught it and the second version stuck to the source.

The web demo shows the same thing with the sources underneath, plus a second
tab for comparing search methods.

## Quick start

You need Python 3.11 and a free API key from [Groq](https://console.groq.com)
(only for writing answers, search works without one).

**1. Get the code**

```bash
git clone https://github.com/uniraula5/clinical-rag.git
cd clinical-rag
```

**2. Make a virtual environment and install the packages**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(On Windows the activate line is `.venv\Scripts\activate`.)

**3. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and paste your Groq key after `OPENAI_API_KEY=`.

**4. Download the data**

```bash
python download_data.py
```

This pulls 12 CSV files (~25 MB) into `data/raw/`.

**5. Build the search index**

```bash
python build_index.py
```

This takes about 4 minutes. It reads every answer, splits it into chunks, and
turns each chunk into a vector. It only has to be done once.

**6. Check that everything is ready**

```bash
python check_setup.py
```

```
Checking setup...

  [OK  ] API key (.env): key found
  [OK  ] MedQuAD data: 12 CSV files
  [OK  ] Search index: 48,978 chunks

Everything is ready. Start the demo with:  python app.py
```

If something is missing it tells you the exact command to fix it.

**7. Start the demo**

```bash
python app.py
```

Open the link it prints, usually http://127.0.0.1:7860.

## Using it

The demo has two tabs.

**Ask** — type a question, get an answer with citations. Underneath you can
open "Fact check details" to see what the checker flagged and whether the
answer had to be rewritten, and "Sources" to read the passages it used.

**Search** — the same questions, but showing the matching answers directly
with no LLM. You can switch between Semantic, Keyword and Hybrid search to see
how they differ. This tab works without an API key.

Try `What condition is linked to mutations in the PAH gene?` in the Search tab
and switch between the three methods:

| Method | Top result |
|---|---|
| Semantic | Lesch-Nyhan syndrome (wrong) |
| Keyword | Pulmonary arterial hypertension (also abbreviated PAH, wrong) |
| Hybrid | Phenylketonuria (correct) |

You can also run things from the command line:

```bash
python pipeline.py "How is Wilson disease treated?"   # full answer + fact check
python search.py "symptoms of leukemia"               # semantic search only
python keyword_search.py "Is MELAS inherited?"        # keyword search only
python hybrid_search.py "HEXA gene"                   # both combined
python evaluate.py                                    # how good is the search
python evaluate_answers.py                            # how good are the answers (uses the LLM)
```

## How it works

Answering a question:

```
question
  -> hybrid search (hybrid_search.py)        top 3 of 15,795 answers
  -> write an answer (answer.py)             the LLM may only use those 3 sources
  -> fact-check it (verify.py)               is every sentence really in its source?
  -> if not, rewrite once (pipeline.py)      the checker's complaints go back to the writer
  -> answer + fact check + sources (app.py)
```

Building the index, which happens once before any of that:

```
data/raw/*.csv        12 MedQuAD files                 47,457 rows
  -> clean_data.py    drop empty and duplicate answers 15,795 questions
  -> chunking.py      split into pieces of <=100 words 48,978 chunks
  -> build_index.py   turn each chunk into a vector    48,978 vectors
```

## Results

### How good is the search? (48 test questions, `python evaluate.py`)

I wrote 48 questions and marked which answers are correct for each one, in
three groups of 16:

- **Patient phrasing** — how a person actually talks: "keep bones from breaking"
- **Exact names** — condition names with lookalikes: "Wolff-Parkinson-White"
  (not Parkinson disease), "Aicardi-Goutieres type 3" (not types 1, 2, 4, 5)
- **Gene symbols** — only a gene code, which never appears in a question title:
  "What condition is linked to mutations in the PAH gene?"

| Method | All 48 (hit@5 / MRR) | Patient phrasing | Exact names | Gene symbols |
|---|---|---|---|---|
| Semantic | 0.83 / 0.67 | **0.81 / 0.60** | **1.00 / 0.90** | 0.69 / 0.52 |
| Keyword (BM25) | 0.85 / 0.68 | 0.69 / 0.49 | 0.88 / 0.66 | **1.00 / 0.89** |
| **Hybrid (RRF)** | **0.94 / 0.75** | 0.81 / 0.51 | 1.00 / 0.88 | 1.00 / 0.86 |

hit@5 means a correct answer was in the top 5. MRR is higher when the correct
answer is nearer the top.

What I found:

- On **exact names** I expected hybrid search to win, and it didn't. Semantic
  search already got all 16, because every chunk is stored together with its
  question title, so the condition name is part of the vector.
- On **gene symbols** semantic search only found 11 of 16. A short code like
  `PAH` or `F9` means little to a small embedding model, but it is exactly what
  keyword search weights most. Keyword and hybrid both got all 16.
- **Hybrid is best overall** (0.94 vs 0.83), so the Ask tab uses it. It is still
  slightly worse than semantic alone at putting the best patient-phrasing answer
  first (MRR 0.51 vs 0.60), because when keyword search is wrong it drags good
  results down.

The gene group was added after the first 32 questions showed no benefit from
hybrid search. To keep it fair, the 6 genes I had already tried informally were
left out, every question uses the same wording, and correct answers were picked
by a rule instead of by hand: any "genetic changes" or "causes" answer whose
text contains "<SYMBOL> gene".

Per-question ranks for all three methods are in `eval/retrieval_results.csv`.

### How good are the answers? (8 questions, `python evaluate_answers.py`)

This runs the whole pipeline on 12 of the 48 questions (every 4th one), so it
scores the answers and not just the search.

| | |
|---|---|
| A correct source was among the 3 used | 9 / 12 |
| First draft passed the fact check | 8 / 12 |
| Needed one rewrite | 4 / 12 |
| Final answer passed the fact check | 9 / 12 |
| Sentences supported by their source | 81% → 91% |

The three that still failed are worth knowing about. In two of them the search
never found the right answer (kidney stone treatment, hepatitis C spread), so the
writer only had loosely related sources to work from. In the third the checker
objected to the word "benign" in front of "tumors" because the source doesn't say
it. That is the checker being strict rather than the answer being wrong, which is
the direction I would rather it erred in for medical text.

The numbers move a little between runs, because the LLM doesn't produce identical
output twice. Per-question details are in `eval/answer_results.json`.

## Tests

```bash
python -m pytest
```

55 tests, about 2 seconds. They don't need an API key, the index, or any LLM
calls, because fake versions of the LLM and database are used. They check the
cleaning rules, the chunk size limit and overlap, one result per answer, the
search and ranking math, the citation and claim checks, the rewrite limit, the
answer cache, the eval scoring, and the setup checker.

## Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `No API key found` | `.env` has no Groq key | `cp .env.example .env` and paste your key |
| "Groq's free limit was reached" | 8,000 tokens a minute or 200,000 a day used up | Wait a minute, or try tomorrow. The Search tab still works |
| `chroma_db` errors, or no results | The index isn't built | `python build_index.py` |
| `FileNotFoundError: No CSV files` | The data isn't downloaded | `python download_data.py` |
| The first question is slow | The model loads on the first run | Normal, later questions are faster |
| Anything else | Not sure what's missing | `python check_setup.py` |

## What's in each file

| File | What it does |
|---|---|
| `download_data.py` | Downloads the MedQuAD CSV files |
| `load_data.py` | Loads all 12 files into one table and prints a summary |
| `clean_data.py` | Throws out unusable rows and gives every answer an ID |
| `chunking.py` | Splits long answers into overlapping chunks |
| `build_index.py` | Turns chunks into vectors and saves them |
| `search.py` | Semantic search |
| `keyword_search.py` | Keyword (BM25) search |
| `hybrid_search.py` | Combines both |
| `llm.py` | Talks to Groq |
| `answer.py` | Writes and rewrites the answer |
| `verify.py` | Checks citations and claims |
| `pipeline.py` | Runs the whole thing: search, write, check, rewrite |
| `app.py` | The web demo |
| `check_setup.py` | Tells you what's missing before you start |
| `evaluate.py` | Scores the search |
| `evaluate_answers.py` | Scores the answers |
| `eval/test_cases.json` | The 48 test questions and their correct answers |
| `eval/retrieval_results.csv` | Where each method ranked the correct answer |
| `eval/answer_results.json` | Details from the answer scoring run |
| `tests/` | The test suite |

## Choices I made

- **15,795 of 47,457 rows are usable.** MedQuAD had to remove the answers from
  its three MedlinePlus sources for copyright reasons, so those rows are
  questions with nothing attached.
- **Chunks are 100 words.** The embedding model stops reading after 256 tokens
  and says nothing about it. At 150 words per chunk, 3.7% of chunks were being
  cut off. At 100 words, 99.8% fit.
- **Embeddings run on your own machine.** Only the question and the 3 retrieved
  passages are sent to Groq.
- **The LLM gets 3 sources, up to 350 words each,** centred on the part that
  matched. Sending 5 sources of 350 words used ~1,600 tokens per call, which
  went over Groq's per-minute limit when an answer had to be rewritten.
- **Two kinds of checking.** Plain Python checks that every sentence has a
  citation and that the number exists. The LLM then checks whether the sentence
  is actually supported by the source it cites.
- **Citations get fixed in code.** The model sometimes writes them in its own
  style (`【4†L1-L4】`), which made every sentence fail the citation check.
- **Only one rewrite.** If the second version still fails, it is shown with a
  warning rather than hidden.
- **The same question is answered from memory** the second time, so clicking an
  example twice doesn't spend tokens.

## Limitations

- **The same model writes the answer and checks it.** That measures whether the
  answer sticks to its sources, not whether it is medically right, and the
  checker can be wrong in both directions.
- **The test sets are small** (48 search questions, 8 answer questions) and I
  wrote them myself, so one question moves a group's score by about 6 points.
  LLM results also move a little between runs.
- **Groq's free tier is limited:** 8,000 tokens a minute and 200,000 a day, which
  is a few dozen questions. Running the answer scoring uses a big chunk of that.
- Hybrid search weights both methods equally. I did not tune it, so the score
  isn't fitted to my own test questions.
- About 0.2% of chunks are still longer than the embedding model's limit.
- MedQuAD was collected around 2017, so some content is out of date.
- **This is a demo, not medical advice.**

## Where the data comes from

Ben Abacha, A. & Demner-Fushman, D. (2019). A question-entailment approach to
question answering. *BMC Bioinformatics*, 20, 511.
CSVs converted by [avery-lockwood/MedQuAD-CSVs](https://github.com/avery-lockwood/MedQuAD-CSVs).
