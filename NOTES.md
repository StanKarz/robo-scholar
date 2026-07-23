# NOTES — decisions, tradeoffs, reasoning

Running log of the non-obvious decisions in this project and why they were made.
One entry per decision; newest sections at the bottom. Spec-level decisions
(no frameworks, Chroma, CLI-first, model split) live in README §3 — this file
covers the choices made *while building*.

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

## Evals (planned flow — for orientation)

The golden set is labels + reference answers written while reading the papers.
Retrieval metrics need only the labels: embed each golden question → query
Chroma top-k → check whether retrieved chunks' section/page match the label →
hit-rate@k and MRR. No LLM, free, seconds per run. This is the workhorse that
settles chunk size, overlap, and the embedding model choice: change ONE
variable → re-ingest → re-run metrics → compare in the runner's SQLite history.
The LLM-as-judge (faithfulness/correctness of generated answers) comes later
and needs the reference answers + a working agent; it's for judging the whole
pipeline, not for tuning retrieval knobs.
