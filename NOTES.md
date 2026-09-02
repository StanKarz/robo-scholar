# NOTES — decisions, tradeoffs, reasoning

Running log of the non-obvious decisions in this project and why they were
made, plus the architecture itself — this is the file that explains how
RoboScholar actually works, not just the pitch (that's README.md) or personal
learning notes (that's COMPREHENSION.md). One entry per decision; newest
sections at the bottom, except the two orientation sections right below,
which are meant to be read first.

---

## How it all fits together

```
PDF / blog post
      │  ingest.py: parse, detect sections, chunk
      ▼
chunks + metadata  (text, paper_id, section, page)
      │  embed each chunk once
      ▼
Chroma  (vector store — chunk vector + text + metadata)

question ──embed──▶ query vector ──cosine similarity──▶ top-k chunks
                                                              │
                        ┌─────────────────────────────────────┤
                        ▼                                     ▼
          agent.py uses them to answer /          eval harness scores the
          quiz / compare, with citations          pipeline (see table below)
```

**Chunks get vector embeddings, not word embeddings.** Each chunk (a
section-aware slice of a document, a few hundred words) becomes *one* dense
vector — not one vector per word, which is the older word2vec-style approach
and isn't what's used here. The vector's job is purely geometric: texts with
similar meaning end up as vectors that sit close together, measured by cosine
similarity, so "find relevant chunks" becomes "find nearby vectors" with no
LLM call involved. Chroma is the storage + nearest-neighbour search over
those vectors; `search_corpus` is the tool wrapping that search.

**Two separate scoring stages — this is the part that's easy to tangle up:**

| Stage | Question it answers | How | Cost |
| --- | --- | --- | --- |
| Retrieval metrics (hit-rate@k, MRR) | Did search find a chunk from the *labelled* section/page? | Compare retrieved chunk metadata to golden labels | Free, instant, no LLM |
| LLM-as-judge | Given what was found, was the *written answer* actually correct? | Sonnet reads generated vs. reference answer | One real LLM call per question |

Cosine similarity only ever answers the first question — it is **not** a
correctness check. Two sentences can sit close together in embedding space
while asserting opposite things ("X causes Y" vs. "X prevents Y" are
topically near-identical vectors). Judging whether a written answer is
actually correct needs real reading comprehension, which is what the judge
is for. The two stages can fail independently: retrieval can succeed while
the written answer is still bad, or retrieval can fail while the model
answers correctly anyway from its own training data — the second case is a
real failure, since it means the system isn't actually using the corpus.

**Why this instead of pasting the paper into a long-context chat?** For one
paper in one sitting, pasting it in and asking for a quiz is a fair
alternative and probably faster to set up — modern context windows are
genuinely good now, and that's not a hole in the argument, it's just a
different tradeoff. This system's edge shows up as the corpus grows (no
re-pasting every document on every question — embed once, query cheaply),
in checkable citations (an answer points at a specific section/page instead
of the model's fuzzy memory of a popular paper's internet-summary version),
and in persistence (quiz history survives across sessions in SQLite, a chat
window doesn't). The plainest honest framing: comprehension alone doesn't
require this infrastructure — building and evaluating the infrastructure is
the actual point, and the quizzing is a genuinely useful side effect of
having it.

---

## Stack choices

| Layer | Choice | Why / tradeoff |
| --- | --- | --- |
| LLM | Anthropic SDK direct, no LangChain/LlamaIndex | Frameworks hide the loop you need to be able to explain; more code, but the code is the point |
| Vector DB | Chroma (embedded, local) | Zero infra, persists to disk; fewer prod features than Qdrant, but v1 doesn't need them |
| Embeddings | voyage-3-lite (API) vs. bge-small / all-MiniLM (local) | Decided by the eval harness, not taste — see "Embedding model choice" below |
| PDF parsing | PyMuPDF first, Docling as plan B | Academic PDFs are hostile (two columns, equations, figures); never silently degrade to a naive text dump |
| Search | Dense first; BM25 + reciprocal rank fusion is the planned upgrade | Hybrid retrieval is common in practice and RRF is ~20 lines to implement directly |
| Interface | Typer + Rich CLI | Ships fast; a web UI is an explicit non-goal (README) |
| Storage (non-vector) | SQLite via stdlib | Quiz history + eval run scores |

### Anthropic SDK — the agent loop mechanics

`client = anthropic.Anthropic()` picks up `ANTHROPIC_API_KEY` from the env.
One endpoint does everything: `client.messages.create(model=..., tools=...,
messages=...)`. The loop: while `response.stop_reason == "tool_use"`, execute
each `tool_use` block with a real Python function and append a `tool_result`
block (matched by `tool_use_id`) as the next user message — multiple tool
calls in one response all go back in a single user message. Model split:
`claude-sonnet-5` for the agent loop and the judge, `claude-haiku-4-5` for
quiz generation and other cheap tasks. Use prompt caching
(`cache_control: {"type": "ephemeral"}`) on the stable prefix (system prompt
+ tool schemas) since the loop re-sends them every iteration. Docs:
[tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
[structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
(for `JudgeScore` / `QuizQuestion`),
[prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### Chroma — the retrieval mechanics

Embedded mode, persisted to disk: `chromadb.PersistentClient(path="data/chroma")`
— no server, no infra. `collection = client.get_or_create_collection(name="papers")`,
then `collection.add(ids=..., documents=..., embeddings=..., metadatas=...)`
where metadata carries `paper_id`/`section`/`page` for citations and eval
labels, and `collection.query(query_embeddings=..., n_results=k, where={...})`
returns top-k with distances (`where` implements the paper filter). Passing
only `documents` makes Chroma silently embed with its default model
(all-MiniLM via ONNX) — instead, compute embeddings explicitly and pass
`embeddings=`, so the embedding model stays a swappable config value the eval
harness can sweep, and query/ingest embeddings are guaranteed consistent.
Docs: [Chroma getting started](https://docs.trychroma.com/docs/overview/getting-started).

### Still to look up

Reciprocal rank fusion (the week-2 hybrid-search upgrade —
[rank-bm25](https://github.com/dorianbrown/rank_bm25)); LLM-as-judge pitfalls
(position bias, verbosity bias, why few-shot rubrics help); the
[arXiv API](https://info.arxiv.org/help/api/index.html) for `fetch_arxiv`.

---

## Workflow

**Notebook-first, then productionise.** Exploratory code (extraction, chunking)
is written in marimo notebooks (`notebooks/`), inspected by eye, approved, then
moved into `roboscholar/` as plain modules. Rationale: chunk quality can only be
judged by looking at chunks; a notebook makes the intermediate output of every
pipeline stage visible. Tradeoff: code exists in two places briefly — the
notebook is the draft, the package is the source of truth once ported.

**Data layout.** Raw sources under `data/raw/` (committed), everything generated
under `data/` (gitignored: `data/chroma/`, `*.db`) — the test is "can it be
rebuilt with one command?". Blog posts are fetched once from the
`data/raw/blogs.json` manifest into `data/raw/blogs/*.md` and committed, so the
corpus is stable even if a post changes or disappears upstream.

---

## Ingestion (notebooks/ingest.py, 2026-07)

**PyMuPDF blocks, not plain `get_text()`.** A flat text dump loses page numbers
and structure — but the golden set labels answers by section+page, and the
retrieval metrics (hit-rate@k, MRR) score against those labels. So metadata
extraction is load-bearing for the whole eval harness, not a nice-to-have.
`page.get_text("blocks")` keeps page numbers and lets us drop image blocks, the
arXiv sidebar watermark, and bare page-number footers.

**Reading order: trust the content stream.** LaTeX-generated two-column PDFs
emit text in reading order (column by column). PyMuPDF's `sort=True` was
deliberately NOT used — it sorts blocks by y-coordinate, which *interleaves*
columns. Verified by eye on all five papers.

**Headings: ToC first, font heuristic as fallback.** 4 of 5 papers ship PDF
bookmarks (`doc.get_toc()`) — free, exact section titles with page numbers.
Fallback (used by world-models): a heading is a short (<80 chars), bold line in
a recurring font size clearly larger than body text; font-size rank gives the
heading level. "Recurring" (≥3 lines) stops the one-off title line from
claiming level 1. If both fail, everything lands in one "Front matter" section
— degraded but functional; Docling remains plan B if a future paper defeats
this. The `unmatched` headings list returned by `split_sections` is the
per-paper diagnostic: it should be empty, and it's the first thing to check
when adding a new paper.

**Heading↔block matching is a normalized character-stream prefix match**
(`consume_key`), not string equality. Every complication was found empirically,
not speculatively:

| Problem | Example | Fix |
| --- | --- | --- |
| Small caps extract with spaces | `"I NTRODUCTION"` | normalize to `[a-z0-9]` only |
| ToC and page number headings differently | ToC `IV-A …` vs page `A. …` | strip leading numbering from both sides |
| Heading merged with first paragraph | `"IV. ACTION CHUNKING … As we will see"` | prefix-match, keep remainder as body text |
| Heading split across two blocks | `"III. ALOHA: A LOW-COST…"` + next block | pending-continuation state |
| Equation fragment impersonating a heading | `"×K"` normalizing to `"k"` | partial match requires ≥8 normalized chars |
| References missing from ToC | ACT bibliography polluting §VII | any `References`/`Bibliography` block forces a section boundary |

**Figure/table captions dropped.** Chunk review showed captions ("Fig. 1:
ALOHA…") mixed into content chunks — layout description, not comprehension
material, and noise for quiz generation. Filter is block-level on the caption
pattern `Fig./Figure/Table + number + [.:]` — the punctuation right after the
number is what separates a caption block from body text saying "Figure 3
shows…" (kept). Captions embedded mid-block aren't caught; acceptable.

**Title/authors/abstract stay together in Front matter.** Separating them
wasn't worth the code — the abstract is valuable retrieval content and the
title/author prefix doesn't hurt it.

**References/bibliography dropped.** Q&A questions don't resolve to citation
lists; reference chunks are retrieval noise. `DROP_SECTIONS` in the notebook.
Appendices are kept — they contain real content (hyperparameters, task details).

**Chunking: sentence-accumulation within sections.** Whole sentences up to
`n_words`, never crossing a section boundary. Rationale: cutting mid-sentence
guarantees some chunk contains a broken statement; cutting across sections mixes
topics and breaks section-level citations. Pathological "sentences" (garbled
equations) are hard-split at `n_words` so they can't create giant chunks.
Chunks under 25 words are dropped (stray date lines, orphaned fragments).

**Overlap: trailing sentences, budgeted by words.** The next chunk is seeded
with the previous chunk's trailing sentences up to `overlap_words`. Overlap is
damage control for boundary cuts; because boundaries here are already
sentence/section-aware, the default is modest (50 words ≈ 10% of 500). Both
`n_words` and `overlap_words` are parameters **to be settled by the eval
harness, not by taste** — planned sweep: 500/1000 words × 0/10%/20% overlap.

**Word-based sizing, not token-based.** Words are a good-enough proxy
(~1.3 tokens/word) and keep the chunker dependency-free. The unit is not the
interesting axis — effective chunk length is, and the sweep covers that.
The real token constraint is the embedder input limit: **all-MiniLM truncates
silently at 256 tokens (~190 words), bge-small at 512 (~380 words)**. Decision
deferred to the embedding A/B: either cap chunk size for a fair local-model
comparison or accept/report the truncation confound explicitly.

**Blogs: fetch step + markdown path, same chunker.** HTML→markdown via
trafilatura (chosen because it strips nav/footer boilerplate; html2text and
markdownify keep it). Markdown headings are unambiguous, so sectioning is a
regex; `chunk_section` and the record schema are shared with the PDF path.
h1/h2 both map to level 1 because a blog's single h1 is its title. Markdown has
no pages → `page_start`/`page_end` = 0.

**Chunk IDs: `{doc_id}:{index:04d}`** (e.g. `act:0012`), where `doc_id` is the
file stem. Deterministic for a given (document, chunk params), so re-ingesting
overwrites rather than duplicates in Chroma. Consequence: IDs are NOT stable
across chunk-parameter changes — an eval run's results are only comparable
within one ingest config, which is why the runner stores the config with the
scores.

---

## Productionising ingest (roboscholar/ingest.py, 2026-07)

**Notebook → package changes.** Same pipeline, plus: type hints throughout
(TypedDicts for records/headings/sections — the Pydantic `Chunk` model in
models.py can wrap `ChunkRecord` later); the font-fallback's body size is now
*computed* (char-weighted modal span size) instead of the notebook's hardcoded
12.0, which happened to work for world-models but would miss headings in a
10pt-body paper; `fetch_blogs` raises on fetch failure instead of printing;
`ingest_file` dispatches by suffix for the CLI. `rs ingest` parses and reports
chunk stats — Chroma storage lands with retrieval.py.

**Test fixture lesson.** The chunker's first test suite passed spuriously:
synthetic sentences were all-lowercase (so the sentence regex never split — it
requires a capital after the period) and identical (so cross-chunk equality
checks compared equal strings regardless). Fixtures must produce *distinct,
realistic* inputs or they test nothing.

---

## Embedding model choice (decision process, decided by evals)

Candidates and their hard constraints:

| Model | Where | Cost | Input limit | Notes |
| --- | --- | --- | --- | --- |
| voyage-3-lite | API | ~free at this corpus size (token-priced) | 32k tok | needs network + key |
| bge-small-en-v1.5 | local (M1) | free | 512 tok (~380 words) | the real local candidate |
| all-MiniLM-L6-v2 | local | free | **256 tok (~190 words)** | truncates silently |

Process: fixed golden set, full (small) grid — 2–3 embedders × 2 chunk sizes ×
3 overlaps ≈ 12–18 ingest+metric runs, each seconds and ~free — stored in the
runner's SQLite, decided on hit-rate@k / MRR. Numbers first, but the decision
also weighs cost, offline use, and latency (write these in the run notes).
MiniLM is effectively disqualified at 500-word chunks unless chunks shrink to
~190 words — either test it only at a small chunk size or report the
truncation confound explicitly.

---

## Golden set structure

One `golden.json` for the whole corpus, not one per source — `source_paper`
already scopes each entry, and metrics can slice per document or per source
type (papers vs blogs). Include blog entries (e.g. ~5 for GAIA-1): blogs are
first-class corpus members and their retrieval must be tested too. Blog labels
have no pages (`source_pages: []` / page 0) — metrics match blogs on section
only. Bonus: papers + blogs overlap on "world models" (Dreamer paper vs GAIA
posts), which gives the golden set natural hard negatives — a GAIA question
should retrieve the blog, not the paper.

---

## Retrieval metrics — how they work, how to read them

Both metrics run per golden question against the top-k chunks retrieved for it.
A retrieved chunk counts as **correct** when its `paper_id` matches the label
and its section matches `source_sections` (or its page range overlaps
`source_pages` — the exact matching rule is a metrics.py implementation
decision; blogs match on section only since they have no pages).

**hit-rate@k** (a.k.a. recall@k at question level): the fraction of golden
questions where *at least one* of the top-k chunks is correct. Binary per
question. This is the metric that matters most for RAG: the agent stuffs all
k chunks into context, so what counts is whether the answer is in there at all
— the LLM doesn't care if it was rank 1 or rank 4.

**MRR** (mean reciprocal rank): for each question, 1/rank of the *first*
correct chunk (1.0 if rank 1, 0.5 if rank 2, …, 0 if absent from top-k),
averaged over questions. Rewards putting the right chunk early — a proxy for
ranking quality, and it's where improvements show up when hit-rate@k has
already saturated.

**How to pick a config — not "combined highest".** The metrics are
complementary, not summable:

1. Primary: **hit-rate@k at the k the agent will actually use** in
   `search_corpus` (e.g. k=5). This is the retrieval floor for answer quality.
2. Tiebreaker: MRR, then cost/latency/offline notes from the run log.
3. Sanity rules: with ~25–30 questions each question is worth ~3–4 points of
   hit-rate, so differences of a few percent are noise — prefer configs that
   win consistently across slices (per-paper, papers-vs-blogs) over one big
   aggregate winner. And eyeball the failures: one mislabelled golden entry
   moves a config comparison more than most real effects.

---

## Ideas — parked, not built

Written down so they aren't lost, not commitments. CLAUDE.md's "no new
features after Day 10" guardrail exists specifically to fight scope creep;
these stay parked until retrieval.py + the eval harness (the actual
differentiator per README) are done.

**Glossary.** Reader marks a term while reading; the system resolves it
against the corpus into a grounded, cited definition, splitting senses when
the same term means different things in different documents (e.g.
"embodiment" in a GAIA-3 post vs. in pi-0). Full design in
`GLOSSARY_DESIGN.md`. The input side is already built — `glossary/terms.txt`,
`roboscholar/glossary.py`, `rs glossary status` — resolution needs
retrieval.py to exist first.

**Concept visualisation / simulation.** Floated 2026-09: for a concept like
action chunking or temporal ensembling, generate a small visual or
simulation of how it works instead of only prose. Undesigned — open
questions before this goes further: output format (an Artifact? a notebook
cell?), generated on-demand per question or pre-built per concept, and
whether "simulation" means an actual numeric simulation of the method or an
illustrative animation — those are very different scopes and worth pinning
down before any code gets written.
