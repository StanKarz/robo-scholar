# NOTES — decisions, tradeoffs, reasoning

How RoboScholar works and why it's built this way (README.md is the pitch).
One entry per decision, newest at the bottom, except the orientation section
right below, which is meant to be read first.

---

## How it all fits together

![RoboScholar architecture](docs/architecture.svg)

Three flows share one Chroma index. Flow 1 writes it; flows 2 and 3 only read
it. Solid boxes exist, dashed ones don't yet.

### Flow 1: index (once per document)

`rs fetch` downloads the papers listed in `data/raw/papers.json`; blog posts
are committed as markdown. `ingest.py` splits each document into sections and
each section into chunks of whole sentences, tagged with `paper_id`, `section`
and a page range. `retrieval.py` embeds every chunk once (one dense vector per
chunk) and stores vector, text and metadata in Chroma.

A new paper goes through this flow once, on its own, and nothing already
indexed is touched. Changing chunk size, overlap or embedder reissues every
chunk ID, so the whole index has to be rebuilt (see "Chunk IDs").

### Flow 2: ask (every question)

Claude gets the question plus a `search_corpus` tool. When it calls the tool,
`rank_chunks` embeds the query with the same model (plus bge's query prefix)
and Chroma returns the k nearest chunks. They go back to Claude as a
`tool_result`, metadata included. Claude can search again or answer, and the
citations come from the chunk metadata rather than from Claude's memory of the
paper. `rs compare` is the same loop over two papers: Claude searches each
(filtered by `paper_id`) before writing the comparison. Nothing here is scored.

![Inside the agent loop](docs/agent-loop.svg)

Claude never touches Chroma: a `tool_use` block is a request, and `agent.py`
runs the function and sends the result back. The API keeps no state, so each
call resends the whole message list, two messages longer per search (about
1k, 3k, 5k input tokens in the example above).

**Search at scale.** Chroma doesn't compare the query with every chunk: it
keeps an HNSW graph (approximate nearest neighbour), so query cost grows
roughly with log N. At this corpus size it doesn't matter either way: a brute
scan of 215 × 384 floats takes about 40 µs, and embedding the query about
12 ms. What grows with the corpus is competition for the top k. For "temporal
ensembling ablation", a π0.5 co-training chunk already ranks third among ACT's.

### Flow 3: evaluate (once per config)

| Score                          | Question it answers                                                  | How                                               | Cost                                |
| ------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------- |
| A: retrieval (hit-rate@k, MRR) | Did search find a chunk from the *labelled* section?                 | Compare retrieved chunk metadata to golden labels | Free, instant, no LLM               |
| B: answer (LLM-as-judge)       | Given what was found, was the *written answer* correct and grounded? | Claude reads generated vs. reference answer       | Agent run + judge call per question |

Cosine similarity only answers the first question. "X causes Y" and "X
prevents Y" sit close together in embedding space, so correctness needs the
judge. The two scores fail independently: retrieval can succeed while the
answer is still wrong, or retrieval can fail while the model answers correctly
from its training data. The second is a real failure (the system isn't using
the corpus), and only a groundedness check catches it.

**A config** is every setting that changes which chunks come back. Chunk
size, overlap and embedder change the stored vectors, so each needs a rebuilt
index; the query prefix and, later, the ranking method (dense or hybrid) only
change the query. k isn't swept: one ranked list of 5 gives hit-rate@1, @3
and @5 at once. The grid (decided 2026-11-12, see "Embedding model choice")
is 2 sizes × 3 overlaps × 2 embedders = 12 index builds, plus prefix off for
the 6 bge builds: 18 scored rows, a few minutes in all. Hybrid search would
double it.

**The sweep** is a grid search over configs, as you'd sweep hyperparameters
in MLflow. For each config: rebuild the index if needed, run score A over the
34 golden questions, store config plus hit-rate@k and MRR as one SQLite row.
`rs eval report` compares rows. The objective is hit-rate, not cosine
similarity: cosine ranks chunks within one config, its scale differs between
embedders, and a high similarity to the wrong chunk scores as well as one to
the right chunk. Only the golden labels know which is which. Score A is free,
so the grid is cheap; score B costs API calls, so it runs once, on the
winning config. After that, day to day is flows 1 and 2. Re-run the eval when
something could move the score: chunking, embedder, ranking, the agent's
prompt or model, or a batch of new papers.

**Questions outside the golden set** are never scored one by one. The golden
set is a test set: a config that does well on 34 hand-labelled questions is
trusted on new ones, as a model is trusted on unseen data because of its
held-out score. Of the judge's two checks, correctness needs a reference
answer, so it only runs on golden questions. Groundedness only needs the
answer and the chunks it came from, so it could run live on any question
(v0.2). Real questions worth keeping can be labelled by hand and added.

**Why not paste the paper into a long-context chat?** For one paper in one
sitting, that's a fair alternative and quicker to set up. This system pays off
as the corpus grows (embed once, query cheaply), in checkable citations, and
in persistence across sessions. Building and evaluating the infrastructure is
the point; studying with it is the useful side effect.

---

## Stack choices

| Layer                | Choice                                                            | Why / tradeoff                                                                                |
| -------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| LLM                  | Anthropic SDK direct, no LangChain/LlamaIndex                     | Frameworks hide the loop you need to be able to explain                                       |
| Vector DB            | Chroma (embedded, local)                                          | Zero infra, persists to disk; fewer prod features than Qdrant, not needed for v1              |
| Embeddings           | voyage-4-lite (API) vs. bge-small (local)                         | Decided by the eval harness — see "Embedding model choice"                                    |
| PDF parsing          | PyMuPDF first, Docling as plan B                                  | Academic PDFs are hostile; never silently degrade to a naive text dump                        |
| Search               | Dense first; BM25 + reciprocal rank fusion is the planned upgrade | Hybrid retrieval is common in practice and RRF is ~20 lines                                   |
| Interface            | Typer + Rich CLI                                                  | Ships fast; a web UI is an explicit non-goal (README)                                         |
| Storage (non-vector) | SQLite via stdlib                                                 | Eval run scores now; quiz history in v0.2                                                     |

### Anthropic SDK — the agent loop mechanics

One endpoint: `client.messages.create(model=..., tools=..., messages=...)`.
While `response.stop_reason == "tool_use"`, run each `tool_use` block and send
back a `tool_result` matched by `tool_use_id`; all results from one response
go in a single user message. Model split: `claude-sonnet-5` for the agent and
judge, `claude-haiku-4-5` for cheap tasks. Cache the stable prefix (system
prompt and tool schemas) with `cache_control`, since the loop re-sends it every
iteration. Docs:
[tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
[structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
(for `JudgeScore`),
[prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### Chroma — the retrieval mechanics

`chromadb.PersistentClient(path="data/chroma/")`, no server. `retrieval.py`
upserts into one collection, `chunks`, with `paper_id`/`section`/`page_start`/
`page_end` as metadata. `collection.query(query_embeddings=..., n_results=k,
where={...})` returns top-k with distances; `where` is the paper filter.
Embeddings are computed explicitly and passed in, because passing only
`documents` makes Chroma embed with its own default (all-MiniLM). That keeps
the embedder a config value and guarantees query and chunk vectors come from
the same model. A collection's distance function is fixed at creation and
defaults to L2. Docs: [Chroma](https://docs.trychroma.com/docs/overview/getting-started).

Open in `retrieval.py` as of 2026-11-10 (marked `TODO(4a)`–`TODO(4d)` in the file):

1. One collection holds every config. Upserting a config with fewer chunks
   leaves the previous config's extras searchable: ACT at 1,000 words
   overwrites `act:0000`–`act:0023` and leaves `act:0024`–`act:0033`.
2. The bge query prefix is in the `rank_chunks` docstring but never applied.
3. The embedder is hard-coded to bge-small in `index_chroma`.
4. The collection uses L2. Rankings match cosine today only because bge
   returns unit vectors, where ‖a−b‖² = 2 − 2cos θ.

Still to look up: reciprocal rank fusion
([rank-bm25](https://github.com/dorianbrown/rank_bm25) for the BM25 half);
LLM-as-judge pitfalls (position bias, verbosity bias, few-shot rubrics).

---

## Workflow

**Notebook-first, then productionise.** Extraction and chunking are written in
marimo notebooks (`notebooks/`), checked by eye, then moved into `roboscholar/`.
Chunk quality can only be judged by looking at chunks. Once ported, the package
is the source of truth.

**Data layout.** The manifests (`data/raw/papers.json`, `data/raw/blogs.json`)
define the corpus: `rs ingest` and the tests read them through
`fetch.corpus_files()`, so a PDF on disk without a manifest entry is ignored.
PDFs are fetched and checksummed by `rs fetch`, not committed; blog markdown is
committed so the corpus stays stable if a post changes upstream. Generated data
(`data/chroma/`, `*.db`) is gitignored and rebuildable with one command.

---

## Ingestion (notebooks/ingest.py, 2026-07)

**PyMuPDF blocks, not plain `get_text()`.** The golden set labels answers by
section and page, and the retrieval metrics score against those labels, so
metadata extraction is load-bearing. `page.get_text("blocks")` keeps page
numbers and lets us drop image blocks, the arXiv sidebar watermark and bare
page-number footers.

**Reading order: trust the content stream.** LaTeX two-column PDFs emit text
column by column. PyMuPDF's `sort=True` sorts by y-coordinate, which
*interleaves* columns, so it's deliberately off. Verified by eye on all five
papers.

**Headings: ToC first, font heuristic as fallback.** 4 of 5 papers ship PDF
bookmarks (`doc.get_toc()`): exact titles with page numbers. Fallback (used by
world-models): a short (<80 chars) bold line in a recurring (≥3 lines) font
size larger than body text, with font-size rank giving the level. If both
fail, everything lands in one "Front matter" section, and Docling is plan B.
`split_sections` returns an `unmatched` list per document: it should be empty,
and it's the first thing to check when adding a paper.

**Heading↔block matching is a normalized character-stream prefix match**
(`consume_key`), not string equality. Every complication was found empirically:

| Problem                                      | Example                                    | Fix                                                             |
| -------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| Small caps extract with spaces               | `"I NTRODUCTION"`                          | normalize to `[a-z0-9]` only                                    |
| ToC and page number headings differently     | ToC `IV-A …` vs page `A. …`                | strip leading numbering from both sides                         |
| Heading merged with first paragraph          | `"IV. ACTION CHUNKING … As we will see"`   | prefix-match, keep remainder as body text                       |
| Heading split across two blocks              | `"III. ALOHA: A LOW-COST…"` + next block   | pending-continuation state                                      |
| Equation fragment impersonating a heading    | `"×K"` normalizing to `"k"`                | partial match requires ≥8 normalized chars                      |
| References missing from ToC                  | ACT bibliography polluting Sec. VII        | any `References`/`Bibliography` block forces a section boundary |
| Non-ASCII symbol in heading silently dropped | `"π0 Model"` → emitted as `"The 0 Model"`  | check the emitted label, not the source heading (pi-0/pi-0.5)   |
| Figure label impersonating a heading line    | ACT p.4 Fig. 5 label `"7 Action Chunking"` | undo a partial match that the next block doesn't finish         |

**Fixed 2026-11-22: a figure label opened ACT's Sec. IV early.** Fig. 5's
`"7 Action Chunking"` (an axis tick merged with a label) is a ≥8-character
prefix of `IV Action Chunking with Transformers`, so the partial-match rule
opened section IV there. The next block didn't continue the heading, and the
old code gave up on the continuation but kept the section open. Section III's
last paragraph (ALOHA's task list) and the Fig. 4/5 label text were labelled
section IV, and the real heading block (`"IV. ACTION CHUNKING … As we will
see"`, the merged-heading example in the table above) was read as body text.
Found while reading act-002's misses: chunk act:0009 was labelled section IV
but held section III's text. Now a partial match records the state before it,
and a block that doesn't finish the heading undoes it. A block-level diff of
all five PDFs moved 8 blocks, all on ACT p.4; `unmatched` is still empty
everywhere; 2–5 ACT chunks change per grid config. `TestSplitSections` and an
ACT integration assertion cover it, and both fail on the old code.

**What's dropped.** Figure/table caption blocks (pattern `Fig./Figure/Table +
number + [.:]`; the punctuation separates a caption from body text saying
"Figure 3 shows…", which is kept), and References/Bibliography
(`DROP_SECTIONS` in `ingest.py`), which are retrieval noise. Appendices stay:
they hold hyperparameters and task details. Title, authors and abstract stay
together in Front matter.

**Known noise, not fixed yet: text drawn inside figures.** Captions are
dropped, but labels inside a figure (axis ticks, legends, diagram boxes) come
out as ordinary text blocks and land in chunks. world-models:0000 carries
"PPO Rainbow MuZero Dreamer 57 tasks, 200M steps Atari…". Measured 2026-11-22
as blocks under 8 words with no closing punctuation: 3.1–4.5% of words in
ACT, Diffusion Policy, π0 and π0.5, and 16.6% in DreamerV3, which is mostly
plots. π0 also has runs of undecodable glyphs (control characters) from figure
fonts, and the blog chunks keep trafilatura's stray `**` bold markers. A
candidate fix is to drop text blocks inside a figure's region (image bboxes,
`page.get_drawings()` clusters, or the area above a caption). It changes chunk
text in every paper, so it gets its own eval run, separate from the
2026-11-22 rerun.

**Chunking: sentence-accumulation within sections.** Whole sentences up to
`n_words`, never crossing a section boundary, so no chunk holds a broken
statement or mixes topics, and every chunk's `section` label is exact (the hit
rule relies on this). Garbled-equation "sentences" are hard-split at
`n_words`; chunks under 25 words are dropped. The next chunk is seeded with
the previous chunk's trailing sentences up to `overlap_words` (default 50,
about 10% of 500). Both numbers are for the sweep to settle.

**Word-based sizing, not token-based.** Words keep the chunker
dependency-free, but the embedder's token limit is the real constraint: text
past it is silently dropped from the embedding. Measured on this corpus
(2026-11): about 1.45 tokens per word with the bge tokenizer, so bge-small's
512 tokens hold roughly 350 words and all-MiniLM's 256 roughly 175.

**Blogs: same chunker, markdown sectioning.** HTML→markdown via trafilatura
(it strips nav/footer boilerplate; html2text and markdownify keep it). h1/h2
both map to level 1, since a blog's single h1 is its title. No pages, so
`page_start`/`page_end` = 0. Settled empirically (2026-09) by dumping real
emitted labels:

- Content between the h1 and the first h2 opens a section named by the h1
  title. It isn't dropped and doesn't attach forward to the next h2.
- h3s are their own sections with compound `Parent › Child` labels. Without
  this, GAIA-3's five h3s collapse into one section.
- PDF sections store the ToC-qualified form too (`VI Ablations › VI-A Action
  Chunking and Temporal Ensembling`). ACT has `IV-A Action Chunking and
  Temporal Ensemble` (the method) and `VI-A Action Chunking and Temporal
  Ensembling` (the ablation); qualifying by parent keeps them distinct.

**Chunk IDs: `{doc_id}:{index:04d}`** (e.g. `act:0012`), where `doc_id` is the
file stem. Re-ingesting the same config overwrites rather than duplicates. IDs
are NOT stable across chunk-parameter changes, and overwriting isn't enough
when a config produces fewer chunks: the leftovers stay. So results are only
comparable within one config (the runner stores the config with the scores),
and the index must be rebuilt, not upserted into, when the config changes.

**Notebook → package (roboscholar/ingest.py).** Same pipeline plus TypedDicts,
a *computed* body font size for the heading fallback (the notebook's
hard-coded 12.0 would miss headings in a 10pt paper), and `ingest_file`
dispatching by suffix. Test-fixture lesson: the chunker's first tests passed
spuriously because the synthetic sentences were all lowercase (the sentence
regex needs a capital after the full stop) and identical (so equality checks
compared equal strings). Fixtures must be *distinct and realistic*.

---

## Embedding model choice (decided by evals)

| Model             | Where      | Cost                                     | Input limit               | Notes                    |
| ----------------- | ---------- | ---------------------------------------- | ------------------------- | ------------------------ |
| voyage-4-lite     | API        | 200M tokens free, then $0.02/M           | 32k tok                   | needs network + key      |
| bge-small-en-v1.5 | local (M1) | free                                     | 512 tok (~350 words here) | the real local candidate |
| all-MiniLM-L6-v2  | local      | free                                     | **256 tok (~175 words)**  | truncates silently       |

Process: fixed golden set, small full grid, every run stored in SQLite,
decided on hit-rate@k then MRR, with cost, offline use and latency in the run
notes.

**voyage-3-lite → voyage-4-lite (2026-11-11).** Voyage's docs now list
voyage-3-lite under "older models", with no free tokens. voyage-4-lite costs
the same ($0.02/M) and comes with 200M free tokens; a full index build is
roughly 90k tokens. It returns 1024 numbers per text against bge-small's
384. Its `input_type="query"` / `"document"` does the same job as bge's query
prefix, except Voyage prepends the prompt on its side. Order of work: every
retrieval fix on bge first, Voyage added after.

**Truncation measured (2026-11), and it changes the grid.** Chunks over
bge-small's 512 tokens across the 8 labelled documents: 2 of 386 at 200 words,
25 of 261 at 350, 77 of 215 at 500 (longest 1,212). The originally planned
500 vs 1,000 words would mostly measure truncation. Instead: 200 and 350
words × 0/10/20% overlap × bge-small and voyage-4-lite (12 builds), plus the
bge query prefix on and off (18 rows), with MiniLM dropped.

**Grid decided 2026-11-12: all 18 rows in one sweep.** The alternatives were
bge's 12 rows first with Voyage's 6 later, or one factor at a time from a
baseline (~5 rows). Rows are nearly free (score A makes no LLM calls; six
Voyage builds use ~540k of the 200M free tokens), the Voyage branch is close,
and only a full grid shows interactions: Voyage never truncates, so it may
prefer 350 words where bge starts to clip.

---

## Golden set structure

One `golden.json`: 34 hand-written entries across the 8 documents (ACT 10,
GAIA-3 8, Diffusion Policy 3, π0 3, π0.5 3, GAIA-2 3, world models 2, GAIA-1
2). `source_paper` scopes each entry, so metrics can slice per document or
papers vs. blogs. Blogs have no pages and match on section only. Papers and
blogs overlap on world models (DreamerV3 vs. the GAIA posts), which gives
natural hard negatives.

### Label audit after the first sweep (2026-11-22)

Sweep `20261117T234037Z` left four voyage-4-lite configs tied: 350w/70 hit
32 of 34, and 200w/40, 200w/0 and 350w/35 hit 31. `notebooks/eval_check.py`
rebuilt all four and listed every miss. Three questions explain all of them.
Ranks below are from a 40-deep search on the 350w/35 index.

| question  | missed by                     | labelled chunk's rank | cause                                                                                                                           |
| --------- | ----------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| pi05-001  | 3 of 4 (350w/70: rank 4)      | not in top 40         | "What is the main contribution of the authors?" never names π0.5; "authors" and "contribution" matched π0's and π0.5's credits appendices |
| act-002   | all 4                         | 9                     | label too narrow: only the intro, while IV-A, IV-B and VI's opening paragraph each state a component and the problem it solves |
| gaia2-003 | all 4                         | 9                     | the question covers the whole "Advancing Safety and Validation" heading (5 subsections); label and reference answer cover one   |

So 350w/70's one-question lead was pi05-001: a question with no paper in it
landing an intro chunk at rank 4, which is luck. Without it all four configs
hit the same 31 questions, with MRR between 0.672 and 0.686. The configs are
tied.

**Rule adopted: a question that points at its source indirectly names it.**
"The authors", "the model", "in this case" and "the learning algorithm" all
assume a document. Score A embeds the question with nothing around it, so an
unnamed question measures luck. Topic questions that assume no document
(act-001, gaia1-001, dp-002) stay as written, and so does wm-002, since "actor
and critic" already points at DreamerV3.

Edits, with every reference answer left as written:

| id        | change                                                                                                     |
| --------- | ---------------------------------------------------------------------------------------------------------- |
| pi05-001  | question → "What is the main contribution of the π0.5 paper?"                                              |
| pi05-002  | "…post-training in this case?" → "…post-training in π0.5?"                                                 |
| pi0-001   | "…into the model?" → "…into π0?"                                                                           |
| pi0-002   | "Describe the pre-training phase…" → "Describe π0's pre-training phase…" (π0.5 has one too)                |
| wm-001    | "Describe how the learning algorithm works." → "Describe how DreamerV3's learning algorithm works."        |
| act-002   | sections + IV-A, IV-B and `VI Ablations` (the parent paragraph, not VI-A/VI-B); pages + 4, 5, 9            |
| gaia2-003 | question narrowed to "How does GAIA-2's scenario diversity improve safety and validation?"                 |

For gaia2-003 the other option was widening the label to all five
subsections, which would also mean rewriting the reference answer. Narrowing
the question keeps the hand-written answer and gives the question one right
place.

Labels were chosen from the section text, not from what retrieval returned.
One overlap to know about: act-002's new `VI Ablations` label is the section
the 350w/35 search ranked first. It's included because its opening paragraph
lists each component with its problem ("action chunking and temporal
ensembling to mitigate compounding errors and better handle non-Markovian
demonstrations … a conditional VAE to model the noisy human
demonstrations"). VI-A and VI-B are the experiments and stay out, as do IV's
parent section (the training pipeline, act-003's answer) and IV-C (the
architecture, act-007's).

Sweep rows from before 2026-11-22 were scored on the old labels and the old
ingest, so compare configs only within one sweep. Follow-up, 2026-11-23: the
naming rule backfired for π0, and pi05-001's label was widened (see "Chunk
config: 350w / 70" below). Next: rerun `rs eval run`,
then apply the decision rule below to the new rows.

---

## Reading queue — candidate papers, not in the corpus

Found 2026-09. The first three PDFs sit in `data/raw/papers/` without manifest
entries, so ingest and the eval ignore them:

- LeCun, "A Path Towards Autonomous Machine Intelligence" (v0.9.2, 2022). Two
  unmatched headings ("A Model Architecture for Autonomous Intelligence" p7,
  "Designing the Configurator" p38) file their content under the preceding
  section. Fix before writing golden entries for it.
- Apple, "The Illusion of Thinking"
- "LeWorldModel" (arXiv 2603.19312)
- Rohit Bandaru, "JEPA Deep Dive" (blog post)

Adding one: manifest entry → `rs fetch` → `rs ingest` (check `unmatched` is
empty) → read it properly before writing questions. Every golden entry comes
from actually reading the paper. An unlabelled paper still tests something: it
adds distractors, so re-running the eval shows how retrieval degrades as the
corpus grows.

---

## Retrieval metrics — how they work, how to read them

A retrieved chunk counts as **correct** when its `paper_id` matches
`source_paper` and its `section` is one of `source_sections` (exact match).
Pages don't count. The old OR rule (section or page overlap) is reported
alongside as a sanity column; if the two pick different winners, read the
failing questions.

**OR, not AND — decided 2026-09, superseded 2026-11-12.** Page ranges are
reliable; section names come from ToC matching, which can fail. OR gives each
axis a fallback; AND
turns every labelling slip into a silent zero that looks like a worse config.

**Reopened 2026-11-10: OR is lenient.** At 500 words, a question has on
average 1.68 correct chunks under section match, 1.59 under section AND page,
and 3.85 under OR. The extras are neighbouring sections sharing a page. For
act-010 (the CVAE ablation, Sec. VI-B, pages 9–10), OR also accepts V-C, VI-A,
VI-C and the conclusion: 7 of ACT's 34 chunks where section match accepts 1.
A config that ranks the wrong section first still scores a hit, so configs
bunch near the ceiling. Meanwhile the failure OR guarded against, a label that
doesn't match what ingest emits, is now caught by `tests/test_golden_labels.py`.

**Section match decided 2026-11-12.** Measured over all 34 questions at 0
overlap, the share of a paper's chunks counted correct is 10% (200 words) and
9% (350) under section match, about the same under AND, and 15% at both sizes
under OR. So OR is half as lenient again, but not biased by chunk size (the
worry was that longer chunks span more pages). AND barely differs from
section match and adds a second way for a labelling slip to score zero.

**Validate labels before embedding anything.** `tests/test_golden_labels.py`
asserts every `source_sections` value appears in emitted chunk metadata, with
near-miss suggestions. It should also gate `eval run`: section names can drift
if the heading logic changes, and a gate turns "quietly lower scores" into
"the run refuses to start".

**hit-rate@k**: the fraction of questions where at least one of the top-k
chunks is correct. The one that matters most for RAG: the agent gets all k
chunks, so what counts is whether the answer is in there at all. **MRR**: mean
of 1/rank of the first correct chunk (0 if absent). Rewards ranking, and
shows improvements once hit-rate@k saturates.

**Picking a config (fixed 2026-11-12, before any results).** Hit-rate@5
under section match decides. With 34 questions each is worth about 3 points,
so gaps under 2 questions (~6 points) are ties. Ties go to MRR, then to the
cheaper, offline option (bge). Prefer configs that win across slices (per
paper, papers vs. blogs), and eyeball the failures, since one mislabelled
entry moves a comparison more than most real effects. Writing the rule down
first matters: the best of 18 noisy scores is flattered by luck, and choosing
the rule after seeing results lets it drift towards the favourite.

---

## Chunk config: 350w / 70 (decided 2026-11-23)

Sweep `20261122T234428Z`, the first after the label audit and the ACT ingest
fix. Voyage rows, plus the best bge row:

| config                 | hit@5 (section) | OR | MRR    |
| ---------------------- | --------------- | -- | ------ |
| voyage 350w / 70       | 33              | 33 | 0.6926 |
| voyage 350w / 35       | 32              | 32 | 0.6892 |
| voyage 200w / 40       | 32              | 32 | 0.6843 |
| voyage 200w / 0        | 31              | 31 | 0.6946 |
| voyage 350w / 0        | 31              | 32 | 0.6833 |
| voyage 200w / 20       | 30              | 31 | 0.6765 |
| bge 350w / 70 (prefix) | 27              | 29 | 0.579  |

Applying the rule from "Picking a config": 350w/70, 350w/35 and 200w/40 are
within 2 questions, so they tie; 200w/0 and 350w/0 sit 2 behind and drop out.
The tie goes to MRR, and 350w/70 has the highest. bge trails Voyage by 6 or
more questions, so the bge tiebreak never applies. `eval_check.py` showed the
one-question gaps inside the tie band come from π questions flipping between
ranks 4 and 6, so 350w/70 is ahead by a margin of noise on both measures,
which is the case the rule's tiebreak exists for.

**π0 vs π0.5 is the remaining failure, and it isn't about chunking.** The two
papers share a lab, a structure (pre-training, post-training, architecture
sections) and a name: the π0.5 paper mentions π0 throughout ("builds on the
π0 VLA"), and "π0" is the first half of "π0.5". A chunk's embedding only
encodes its text, so nothing tells it which paper it came from. On the
200w/40 index, adding "π0" to two questions in the audit made them worse:

| question | Stan's wording | named | top 10 by paper, named          |
| -------- | -------------- | ----- | ------------------------------- |
| pi0-001  | rank 1         | 5     | 7 π0.5, 3 π0 (was 5 / 5)        |
| pi0-002  | rank 3         | 16    | 9 π0, 1 π0.5                    |
| pi05-001 | rank 11        | 9     | 7 π0.5 (was 1 π0.5)             |

The names stay anyway. "What does π0 do" is how a user asks, it exposes a real
weakness, and reverting to recover the score would be fitting questions to
retrieval. pi0-002 fails inside π0 (9 of its top 10 are π0 chunks), so a
paper name can't fix it. See the open label question below.

**pi05-001 label widened (2026-11-23):** + `Front matter` and `Discussion and
Future Work`, pages + 1, 11. The abstract ("We describe π0.5, a new model
based on π0 that uses co-training on heterogeneous tasks to enable broad
generalization") and the Discussion's opening ("We described π0.5, a
co-trained model that builds on the π0 VLA…") both state the contribution.

**Locked in.** `data/chroma` rebuilt at voyage-4-lite 350w/70 (266 chunks),
orphaned segment folders deleted. Baseline on the current labels (including
pi0-002's, below), plain search: 34/34, MRR 0.7113. hit@5 is now at the ceiling, so from here Score A
can only show changes through MRR (or a stricter hit@1/hit@3). New golden
entries with one right section, and harder cross-paper ones, would give it
room again.

**Paper filter in the agent's search, decided over contextual chunk headers.**
Simulated on 350w/70 by taking each question's top 40 and keeping only its
labelled paper's chunks: still 34/34, MRR 0.7113 → 0.773. All of the gain is
on five questions: pi0-001 rank 4 → 1, pi0-003 5 → 2, pi0-002 2 → 1, pi05-003
5 → 4, gaia2-002 2 → 1. That's an upper bound, since it assumes Claude always
picks the right paper. Why the filter first:

- paper identity is a metadata fact, so filtering on `paper_id` is an exact
  lookup, while headers ask the embedding to separate identifiers, which is
  what embeddings are worst at ("π0" and "π0.5" differ by two characters, and
  both titles contain "Vision-Language-Action Model");
- it needs no re-embedding and no change to chunks, and it lives in
  `search_corpus`, which is the next thing to build;
- the agent's real number is measurable by logging each search's query and
  filter, and scoring those searches against the labels.

What it gives up: plain search (Score A) is unchanged, and questions that
don't name a paper get no help. Cross-paper questions (`rs compare`) need the
filter to take a list of papers or be left off. Contextual headers would help
both cases, so they're parked until Score B shows the agent failing on paper
identity (see Ideas).

**pi0-002 label widened (2026-11-23):** + `Data Collection and Training
Recipe` and its `› Pre-training and post-training` subsection, pages + 5, 6.
The reference answer comes from π0's Overview, but the Overview itself points
on ("The pre-training phase (Section V-A)"), and section V opens with the
purpose of each phase. Same narrow-label pattern as act-002 and pi05-001.
Plain-search rank 5 → 2 on 350w/70.

---

## v0.2: quiz, and how to evaluate generated questions

Quiz moved to v0.2 on 2026-11-10. Open problem, raised by a friend: "how do
you evaluate the model's ability to generate good quiz questions for new
text?" Both current scores check fixed golden questions; a generated question
has no reference. Failure modes: ungrounded (doesn't trace back to the
retrieved chunk), trivial (restates a sentence verbatim), wrong difficulty
label. Two directions (judge.py is Stan's, so this is a design note only): a
second judge prompt scoring a question against its source chunk for
groundedness and non-triviality, and a thumbs-up/down logged when a quiz is
actually taken.

### Ideas from Jane Street's socratic-tutor skill (2026-11-20)

Source: [janestreet/agent_skills,
socratic-tutor/SKILL.md](https://github.com/janestreet/agent_skills/blob/master/socratic-tutor/SKILL.md),
MIT licence, a single 5 KB prompt file for a coding agent. It teaches by
asking. A diagnostic of 3–5 questions comes before any teaching, a skill map
rates each sub-skill from 0 to 5, each message ends on one question with
nothing after it, and grading is strict. Four parts of it answer open
questions in the paragraph above.

For generating questions, it rotates through five types: predict an
outcome, explain why, find the flaw in a plausible but wrong claim, apply the
idea to a new case, and connect two ideas. If the generator has to pick one
type per question and name it, the judge gets a test for the "trivial"
failure mode: a question that one sentence of the source chunk answers fits
none of the five. Find-the-flaw suits papers. Take a claim from the chunk and
change one detail, e.g. "ACT's temporal ensemble weights the newest
prediction most" (it's the oldest: w_i = exp(−m·i), w_0 the oldest action,
act:0012), then ask what's wrong with it.

For grading answers, its rules are: a vague or partially correct answer is
not correct; don't credit the learner with insight they didn't actually
express; don't flatter. These go into whatever prompt grades a quiz answer
against its source chunk (Stan's, like judge.py).

For wrong answers, it first asks a narrower question that exposes the
contradiction. It explains directly only after 2–3 misses, then re-tests the
same idea from a different angle. This is the shape for `rs quiz` once it's
interactive.

For state across sessions, the skill prints a text block to paste into the
next session. RoboScholar already plans to track wrong answers in SQLite
(README, Quiz), so storing a level and a last-tested date per (paper,
section) would give spaced review with no pasting. Its promotion rule is
worth copying: a level rises only after two checks on separate occasions,
one of them unprompted recall.

Not borrowed for v0.2: the full tutor (diagnostic, skill map, mastery at
4/5, a final mixed test). That's the parked "Socratic tutor with a reader
profile" below, and it's a new feature. Until then the skill works unchanged
for studying the corpus papers: copy SKILL.md into
`~/.claude/skills/socratic-tutor/`.

---

## Ideas — parked, not built

Parked until retrieval.py and the eval harness are done (CLAUDE.md: no new
features after Day 10).

**Contextual chunk headers (parked 2026-11-23).** Prepend the paper title and
section to the text that gets embedded (not to what Claude reads), so each
chunk's embedding carries its document. It targets the π0/π0.5 confusion in
plain search and helps questions that don't name a paper, which the agent's
paper filter can't. Costs a re-embed and a sweep. Revisit if Score B shows
the agent still failing on paper identity with the filter in place. See
"Chunk config: 350w / 70".

**Glossary: dropped 2026-09-14.** Marking terms while reading, then resolving
them into cited definitions, made a second place to write things down and a
term database nobody would reread. Highlight-to-ask serves the need in the
moment. The unwired code was deleted 2026-11-07.

**Socratic tutor with a reader profile (raised 2026-11-10).** Stan's framing
of the longer-term goal: a Socratic learning tool for papers and technical
posts that gets more useful as the corpus grows and adapts to the reader's
strengths, gaps and interests. Quiz (v0.2) is the first piece, since wrong
answers logged per concept are the raw material for a profile. A Textual front
end fits here (interactive quiz, a reader view showing the cited page or
figure). None of it is v0.1, which is the engine plus the evals that make
claims about it credible.

**Concept visualisation / simulation.** Floated 2026-09: show how a concept
like temporal ensembling works instead of describing it. One data point: the
π0.5 policy-factorisation walkthrough, an Artifact with terms that highlight
per step and a hover-synced symbol glossary, built on demand for one question
and judged exactly the right shape. Still open: numeric simulation of the
method or illustrative animation, which are very different scopes.

**Highlight-to-ask.** Floated 2026-09: highlight a passage while reading and
ask about it in place. Closest precedent: [Clicky](https://www.heyclicky.com/)
(hotkey → screenshot → vision model → answer), which works in any app and so
sidesteps PDFs having no DOM. Minimum build: hotkey, screenshot, one
vision-capable `messages.create` call. The RoboScholar part is grounding:
knowing which corpus document is on screen, so the answer cites a retrieved
chunk. Second data point: wanting "representation collapse" in LeWorldModel
explained in the context of its sentence, which a screenshot captures for
free.
