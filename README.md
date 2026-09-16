# Clinical Q&A

Ask a health question and get a short answer built only from NIH and CDC
sources. Every sentence has a citation like `[2]`, and the answer is
fact-checked against those sources before you see it.

The data is [MedQuAD](https://github.com/abachaa/MedQuAD), a public set of
medical questions and answers from government health websites. Nothing here
is medical advice.

## If you only have five minutes

- **The result worth reading** is under "How good is the search?": semantic search
  wins on plain language, loses badly on gene symbols, and hybrid search wins
  overall. That comparison is the point of the project.
- **The evidence:** `eval/test_cases.json` (48 questions and their correct
  answers), `evaluate.py` (scores them), `eval/retrieval_results.csv` (where each
  method ranked the correct answer for every question).
- **The whole answer flow** is `pipeline.py`, about 150 lines: search, write,
  fact-check, rewrite once.
- **`python -m pytest`** runs 74 tests in about 2 seconds with no API key, no
  index and no LLM calls. (Two of them check the eval questions against the real
  data, so they skip until you run `python download_data.py`.)

## What it looks like

```bash
python pipeline.py "How is Wilson disease treated?"
```

```
--- draft (1 problem(s)) ---
Wilson disease is treated with therapies that aim to reduce or control the amount
of copper that builds up in the body [1][3]. Affected individuals usually need
lifelong treatment that includes medications and changes to their diet to lower
copper intake [1][4]. If medication and diet are not enough or if liver failure
develops, a liver transplant may be required [1][4]. When the disease is identified
early and treatment is followed, most people can lead normal lives and have a normal
lifespan [2].
  ! Not supported by its source: "If medication and diet are not enough or if liver
    failure develops, a liver transplant may be required [1][4]." (Source 4 does not
    specify transplant is required when meds/diet fail; condition not in cited source)

--- revision (passed) ---
Therapies for Wilson disease aim to reduce or control the amount of copper that
builds up in the body [1]. Affected individuals need lifelong treatment, which may
include medications and dietary changes to lower copper intake [1]. If treatment is
not effective or if liver failure develops, a liver transplant may be necessary [1].
When the disorder is detected early and treated appropriately, most people can enjoy
normal health and a normal lifespan [2].

Sources:
  [1] GARD | What are the treatments for Wilson disease? (GARD_0006449-5)
  [2] NINDS | What is the outlook for Wilson Disease? (NINDS_0000276-3)
  [3] GARD | What is (are) Wilson disease? (GARD_0006449-1)
  [4] NIDDK | What to do for Wilson Disease? (NIDDK_0000133-13)
```

The draft leaned on two sources at once. It cited `[1][4]` for the transplant
sentence, but source 4 is a diet page that never mentions transplants, so the
checker rejected it. The rewrite says the same thing from source 1 alone, which is
the "one fact per sentence, from one source" rule doing its job.

This is a real run, so your own output will differ a little — the model does not
write the same words twice.

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

This takes about 4 minutes and writes a `chroma_db/` folder of roughly 300 MB.
It reads every answer, splits it into chunks, and turns each chunk into a vector.
It only has to be done once.

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

Each result shows a score, but the three methods are not on the same scale, so
the label says which one you are looking at: **cosine similarity** (0 to 1) for
semantic, an unbounded **BM25 score** for keyword, and an **RRF score** for hybrid.
RRF adds up `1 / (60 + rank)`, so its numbers are always near 0.03 no matter how
good the match is. Compare results within a method, not across them.

Try `What condition is linked to mutations in the PAH gene?` in the Search tab
and switch between the three methods:

| Method | Top result |
|---|---|
| Semantic | Lesch-Nyhan syndrome (wrong) |
| Keyword | Pulmonary arterial hypertension (also abbreviated PAH, wrong) |
| Hybrid | Phenylketonuria (correct) |

You can also run things from the command line:

```bash
python pipeline.py "How is Wilson disease treated?"     # full answer + fact check
python search.py "symptoms of leukemia"                 # semantic search only
python keyword_search.py "Is MELAS inherited?"          # keyword search only
python evaluate.py                                      # how good is the search
python evaluate_answers.py                              # how good are the answers (uses the LLM)
python evaluate_answers.py --offset 2                   # the same check on held-out questions

# both searches combined
python hybrid_search.py "What condition is linked to mutations in the PAH gene?"
```

## How it works

Answering a question:

```
question
  -> hybrid search (hybrid_search.py)        top 4 of 15,795 answers
  -> write an answer (answer.py)             the LLM may only use those 4 sources
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

The eval has 48 questions with the correct answers marked, in three groups of 16:

- **Patient phrasing** — how a person actually talks: "keep bones from breaking"
- **Exact names** — condition names with lookalikes: "Wolff-Parkinson-White"
  (not Parkinson disease), "Aicardi-Goutieres type 3" (not types 1, 2, 4, 5)
- **Gene symbols** — only a gene code, which almost never appears in a question
  title: "What condition is linked to mutations in the PAH gene?" (`DMD` is the one
  exception of the 16: five titles mention it, which is why keyword search finds it
  so easily.)

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

### How many sources should the Ask tab send?

The pipeline only hands its top few answers to the LLM, so what matters there is
hit@3 or hit@4, not hit@5. `python evaluate.py` prints a hit@k table, and it
showed a problem I had missed:

| Sources sent | All | Patient phrasing | Exact names | Gene symbols |
|---|---|---|---|---|
| 3 | 0.85 | 0.62 | 1.00 | 0.94 |
| **4** | **0.94** | **0.81** | **1.00** | **1.00** |
| 5 | 0.94 | 0.81 | 1.00 | 1.00 |

With only three slots, one wrong keyword result can push out a correct semantic
one. Three questions that semantic search ranked first — the kidney stone, shingles
and breast cancer ones — fell to fourth place under hybrid search, so a fourth slot
rescues all three for about 250 extra tokens per call, and a fifth adds nothing. So
the pipeline sends **4 sources**.

A fourth slot does not rescue everything. Hybrid search loses three questions
outright, and one of them, "Can you catch hepatitis C from sharing needles?",
is a question semantic search had at rank 1. No number of slots brings back an
answer the merged ranking never returned. That is the same question the answer
eval below reports as a refusal, and it is the clearest cost of using hybrid
search: it wins on 45 of 48 questions and pays for it on this one.

Per-question ranks for all three methods are in `eval/retrieval_results.csv`.

### How good are the answers? (12 questions, `python evaluate_answers.py`)

This runs the whole pipeline on 12 of the 48 questions (every 4th one), so it
scores the answers and not just the search.

| | 3 sources | 4 sources | 4 + stricter rules (now) |
|---|---|---|---|
| A correct source was among those used | 9 / 12 | 11 / 12 | **11 / 12** |
| First draft passed the fact check | 8 / 12 | 6 / 12 | 7 / 12 |
| Needed one rewrite | 4 / 12 | 6 / 12 | 5 / 12 |
| Final answer passed the fact check | 9 / 12 | 9 / 12 | **12 / 12** |
| Sentences supported by their source | 81% → 91% | 77% → 92% | 82% → **100%** \* |

\* The two percentages do not cover the same answers. Every draft made claims, so
the first number is over all 12. The final answer for the hepatitis C question is a
refusal, which has no sentences to score, so the second is over the 11 that made a
claim. `python evaluate_answers.py` now prints the count next to each percentage
instead of leaving that to be discovered.

The failures were never random. Once search improved, what was left was the writer
adding detail the sources don't have, in two shapes: gluing a fact from one source
to a fact from another in one sentence and citing both, and explaining a mechanism
the source never explains. So the writer now has two extra rules: one fact per
sentence from a single source, and no explaining how something works unless the
source explains it. That took the final answers from 9 of 12 to 12 of 12.

Two honest notes about that 12 of 12:

- **One of them is a refusal.** For "Can you catch hepatitis C from sharing
  needles?" search still finds no source that mentions needles, so the answer is
  "The sources I found don't answer this question." It passes because it claims
  nothing. That is the behaviour I want, but it isn't an answer.
- **The rules were written after reading the failures from these same 12
  questions,** which is a mild form of fitting the test. So I checked them on
  questions they had never seen, below.

#### Checking the rules on questions they never saw

The eval takes every 4th question. Shifting the starting point gives a second
sample of 12 from the same 48, with no overlap and 4 from each category:

```bash
python evaluate_answers.py --offset 2
```

| | tuning set | held-out set |
|---|---|---|
| A correct source was among those used | 11 / 12 | 11 / 12 |
| First draft passed the fact check | 7 / 12 | 10 / 12 |
| Final answer passed the fact check | 12 / 12 | 12 / 12 |
| Sentences supported by their source | 82% of 12 → 100% of 11 | 92% of 12 → 100% of 12 |

The held-out questions did slightly better, and none of those answers was a
refusal, so the rules are doing general work rather than fitting the questions I
looked at. Results are saved separately in `eval/answer_results_holdout.json` so a
held-out run can't overwrite the other one.

One held-out question (lupus) counts as a retrieval miss because the answer I
marked correct wasn't retrieved, but the answer it did write is properly grounded
in other lupus pages. Strict labels undercount cases like that.

Answers didn't get shorter to please the checker. On the tuning set they run 43 to
111 words and on the held-out set 32 to 130, 3 to 5 sentences either way, inside the
3-to-6 rule the writer is given. What changed is one fact per sentence, with one
citation.

The numbers move a little between runs, because the LLM doesn't produce identical
output twice. Per-question details are in `eval/answer_results.json`.

## Tests

```bash
python -m pytest
```

74 tests, about 2 seconds. They don't need an API key, the index, or any LLM
calls, because fake versions of the LLM and database are used. Two of them read the
real MedQuAD files to check the eval questions, so those two skip until you run
`python download_data.py`; the other 72 run on a fresh clone.

They check the cleaning rules, the chunk size limit and overlap, one result per
answer, the search and ranking math, where a sentence ends, the citation and claim
checks, the rewrite limit, the answer cache, the eval scoring, and the setup checker.

I also checked that the tests can actually fail. Putting each bug back on purpose —
the old sentence splitter, the removed empty-answer guard, an unbounded result list,
a passage window that no longer centres on the match, the raw database error — made
exactly the test that covers it fail, five for five.

## Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `No API key found` | `.env` has no Groq key | `cp .env.example .env` and paste your key |
| `Couldn't write an answer (AuthenticationError)` | `.env` has a key but no `OPENAI_API_BASE`, so the request went to OpenAI instead of Groq | Add `OPENAI_API_BASE=https://api.groq.com/openai/v1` to `.env`. `check_setup.py` now flags this |
| `No search index found` | The index isn't built yet | `python build_index.py` |
| "Groq's free limit was reached" | 8,000 tokens a minute or 200,000 a day used up | Wait a minute, or try tomorrow. The Search tab still works |
| `No CSV files found` | The data isn't downloaded | `python download_data.py` |
| `Warning: only N of 12 CSV files` | The download stopped part way | `python download_data.py` again; it skips what you already have |
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
| `eval/answer_results_holdout.json` | The same, for the held-out questions |
| `tests/` | The test suite |

## Choices I made

- **15,795 of 47,457 rows are usable.** MedQuAD had to remove the answers from
  its three MedlinePlus sources for copyright reasons, so those rows are
  questions with nothing attached.
- **Chunks are 100 words.** The embedding model stops reading after 256 tokens
  and says nothing about it. At 150 words per chunk, 3.7% of chunks were being
  cut off. At 100 words, 99.8% fit.
- **Embeddings run on your own machine.** Only the question and the 4 retrieved
  passages are sent to Groq.
- **The LLM gets 4 sources, up to 350 words each,** centred on the part that
  matched. Five sources used ~1,600 tokens per call, which went over Groq's
  per-minute limit when an answer had to be rewritten. Three saved tokens but
  lost plain-language questions. Four is ~1,250 tokens and scores the same as
  five on every category.
- **Two kinds of checking.** Plain Python checks that every sentence has a
  citation and that the number exists. The LLM then checks whether the sentence
  is actually supported by the source it cites.
- **Citations get fixed in code.** The model sometimes writes them in its own
  style (`【4†L1-L4】`), which made every sentence fail the citation check.
- **One fact per sentence, from one source.** The writer used to merge two sources
  into a sentence and cite both, and explain mechanisms the sources never state.
  Two rules against that took the answer score from 9 of 12 to 12 of 12.
- **Only one rewrite.** If the second version still fails, it is shown with a
  warning rather than hidden.
- **The same question is answered from memory** the second time, so clicking an
  example twice doesn't spend tokens.

## Limitations

- **The same model writes the answer and checks it.** That measures whether the
  answer sticks to its sources, not whether it is medically right, and the
  checker can be wrong in both directions.
- **The test sets are small** (48 search questions, 12 answer questions), so one
  question moves a group's score by about 6 points. LLM results also move a little
  between runs. The gene questions were labelled by a rule rather than by hand,
  which is stated where they're described; the other two groups were labelled by
  matching the condition name and question type.
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
