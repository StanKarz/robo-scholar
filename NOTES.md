# NOTES — decisions, tradeoffs, reasoning

Running log of the non-obvious decisions in this project and why they were
made, plus the architecture itself — this is the file that explains how
RoboScholar actually works, not just the pitch (that's README.md). One entry per decision; newest
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

| Stage                               | Question it answers                                               | How                                               | Cost                           |
| ----------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------- | ------------------------------ |
| Retrieval metrics (hit-rate@k, MRR) | Did search find a chunk from the*labelled* section/page?        | Compare retrieved chunk metadata to golden labels | Free, instant, no LLM          |
| LLM-as-judge                        | Given what was found, was the*written answer* actually correct? | Sonnet reads generated vs. reference answer       | One real LLM call per question |

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

| Layer                | Choice                                                            | Why / tradeoff                                                                                           |
| -------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| LLM                  | Anthropic SDK direct, no LangChain/LlamaIndex                     | Frameworks hide the loop you need to be able to explain; more code, but the code is the point            |
| Vector DB            | Chroma (embedded, local)                                          | Zero infra, persists to disk; fewer prod features than Qdrant, but v1 doesn't need them                  |
| Embeddings           | voyage-3-lite (API) vs. bge-small / all-MiniLM (local)            | Decided by the eval harness, not taste — see "Embedding model choice" below                             |
| PDF parsing          | PyMuPDF first, Docling as plan B                                  | Academic PDFs are hostile (two columns, equations, figures); never silently degrade to a naive text dump |
| Search               | Dense first; BM25 + reciprocal rank fusion is the planned upgrade | Hybrid retrieval is common in practice and RRF is ~20 lines to implement directly                        |
| Interface            | Typer + Rich CLI                                                  | Ships fast; a web UI is an explicit non-goal (README)                                                    |
| Storage (non-vector) | SQLite via stdlib                                                 | Quiz history + eval run scores                                                                           |

### Anthropic SDK — the agent loop mechanics

`client = anthropic.Anthropic()` picks up `ANTHROPIC_API_KEY` from the env.
One endpoint does everything: `client.messages.create(model=..., tools=..., messages=...)`. The loop: while `response.stop_reason == "tool_use"`, execute
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

| Problem                                   | Example                                     | Fix                                                                |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| Small caps extract with spaces            | `"I NTRODUCTION"`                         | normalize to`[a-z0-9]` only                                      |
| ToC and page number headings differently  | ToC`IV-A …` vs page `A. …`            | strip leading numbering from both sides                            |
| Heading merged with first paragraph       | `"IV. ACTION CHUNKING … As we will see"` | prefix-match, keep remainder as body text                          |
| Heading split across two blocks           | `"III. ALOHA: A LOW-COST…"` + next block | pending-continuation state                                         |
| Equation fragment impersonating a heading | `"×K"` normalizing to `"k"`            | partial match requires ≥8 normalized chars                        |
| References missing from ToC               | ACT bibliography polluting §VII            | any`References`/`Bibliography` block forces a section boundary |
| Non-ASCII symbol in heading silently dropped | `"π0 Model"` → emitted as `"The 0 Model"` | Normalization strips non-`[a-z0-9]` chars, Greek letters included — always check the actual emitted label rather than assuming the source heading text survives (found on pi-0/pi-0.5, which also emit no numeric ToC prefixes at all, unlike ACT/Diffusion Policy) |

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

**Markdown sectioning, settled empirically (2026-09).** Three questions were
open long enough to be worth recording the answers, all confirmed by dumping
real emitted labels rather than reasoning about the regex:

- *Content between the h1 and the first h2* opens a section named by the h1
  title text (so GAIA-3's five opening paragraphs live under
  `"GAIA-3: Scaling World Models to Power Safety and Evaluation"`). It isn't
  dropped and doesn't attach forward to the following h2 — the two dangerous
  cases.
- *h3s are their own sections*, emitted as compound `Parent › Child` labels.
  Without this, GAIA-3's five h3s would have collapsed into one enormous
  section and three golden entries would have carried labels for sections
  that don't exist.
- *PDF sections store the ToC-qualified form* (`VI Ablations › VI-A Action
  Chunking and Temporal Ensembling`). This is load-bearing on ACT
  specifically, which has three `A.` subsections, two of which differ by one
  word: `IV-A Action Chunking and Temporal Ensemble` (the method) and
  `VI-A Action Chunking and Temporal Ensembling` (the ablation). Qualifying
  by parent makes subsection names unique by construction.

`unmatched` stays the per-document diagnostic and should be empty. As of
2026-09-14 every document ingests clean except `path_towards_autonomous_mach`
(LeCun), which reports two unmatched level-1 headings — "A Model Architecture
for Autonomous Intelligence" (p7) and "Designing the Configurator" (p38).
Content from those regions is currently filed under whatever section precedes
it, so any golden entry drawn from there would be mislabelled. Harmless while
that paper has no golden entries; fix before it gets any.

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

| Model             | Where      | Cost                                     | Input limit                    | Notes                    |
| ----------------- | ---------- | ---------------------------------------- | ------------------------------ | ------------------------ |
| voyage-3-lite     | API        | ~free at this corpus size (token-priced) | 32k tok                        | needs network + key      |
| bge-small-en-v1.5 | local (M1) | free                                     | 512 tok (~380 words)           | the real local candidate |
| all-MiniLM-L6-v2  | local      | free                                     | **256 tok (~190 words)** | truncates silently       |

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

## Reading queue — candidate papers, not yet in golden.json

Found 2026-09 while researching a separate Transformers blog post. More
LLM/world-model adjacent than the current embodied-AI corpus, but relevant
enough to add 2-3 golden questions each once actually read:

- LeCun, "A Path Towards Autonomous Machine Intelligence" (v0.9.2, 2022-06-27)
- Apple, "The Illusion of Thinking" (machinelearning.apple.com)
- "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
  from Pixels" (arXiv 2603.19312)
- Rohit Bandaru, "JEPA Deep Dive" (blog post)

None ingested yet. Adding one means: drop the PDF or fetch the blog per the
existing `data/raw/` pattern, ingest, check `unmatched` is empty, then read it
properly before writing questions. This is a read-before-write list, not a
task queue — the golden set's value is that every entry came from Stan
actually reading, and that doesn't change because the backlog is longer.

---

## Retrieval metrics — how they work, how to read them

Both metrics run per golden question against the top-k chunks retrieved for it.
A retrieved chunk counts as **correct** when its `paper_id` matches the label
and its section matches `source_sections` **or** its page range overlaps
`source_pages` (blogs match on section only since they have no pages).

**OR, not AND — decided, and it matters.** Page ranges come from PyMuPDF
block metadata and are reliable. Section names come from ToC-to-block
matching, which has a documented list of ways it can fail (see the table in
"Ingestion" above). OR gives each label axis a fallback when the other
misses. AND conjoins two things that can each fail independently and turns
every labelling slip into a silent zero — and a silent zero doesn't look
like a bug, it looks like a config that scored slightly worse.

**Validate labels before embedding anything.** `tests/test_golden_labels.py`
asserts every `source_sections` value in `golden.json` actually appears in
emitted chunk metadata, with near-miss suggestions on failure. It should run
at the top of `eval run` too, not only under pytest: chunk IDs aren't stable
across chunk-parameter changes and section names aren't guaranteed stable if
the heading logic changes, so a label that was correct during one sweep can
rot by the next. Running it as a gate converts "quietly lower scores" into
"the run refuses to start", which is the difference between a bug you find
and one you don't.

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

## Open question: evaluating quiz question quality

Not handled by the current eval design, and it's a real gap, not a maybe.
Raised by a friend while Stan was explaining the project: "how do you evaluate
the model's ability to generate good quiz questions for new text?"

Retrieval metrics (above) score whether search finds the right chunk. The
LLM-judge scores whether a *generated answer* matches a hand-written reference.
Both are checked against `golden.json`'s fixed, pre-written questions. Neither
touches `quiz_me`, which generates *novel* questions on the fly from whatever
chunks come back for a topic — there's no reference to compare a fresh
question against, so nothing currently checks whether a generated quiz
question is any good. `quiz_me` is core v1 (README) and currently has zero
eval coverage — this belongs in scope now, not as a someday improvement.

Three distinct failure modes, needing different checks:

- **Ungrounded** — the question or its expected answer doesn't trace back to
  the retrieved chunk; the model filled in from its own knowledge rather than
  from what was retrieved.
- **Trivial** — a fill-in-the-blank restating a sentence verbatim, testing
  recall of wording rather than understanding.
- **Wrong difficulty label** — tagged "easy" but actually needs synthesis
  across the chunk, or vice versa.

Two directions worth weighing when `judge.py` gets built (judge.py is Stan's
per CLAUDE.md — this is a design note, not an implementation):

1. Extend the LLM-as-judge pattern to questions: judge a generated question
   against its source chunk for groundedness + non-triviality, reusing the
   same Sonnet + structured-output infrastructure already planned for
   answer grading — not net-new infrastructure, an additional judge prompt.
2. Cheap human-in-the-loop signal: log a thumbs-up/down when a quiz is
   actually taken. Consistent with how this project already treats ground
   truth — hand-written golden set, hand-marked glossary terms — rather than
   defaulting to auto-generated labels for something this subjective.

---

## Ideas — parked, not built

Written down so they aren't lost, not commitments. CLAUDE.md's "no new
features after Day 10" guardrail exists specifically to fight scope creep;
these stay parked until retrieval.py + the eval harness (the actual
differentiator per README) are done.

**Glossary — dropped 2026-09-14, not parked.** The design was: mark unfamiliar
terms in `glossary/terms.txt` while reading, then batch-resolve each one
against the corpus into a grounded, cited definition, splitting senses where
documents disagree (e.g. "embodiment" in GAIA-3 vs. in pi-0). Killed on
Stan's call: the marking step is a second place to write things down, and
the payoff is a separate term database he'd have to go back and read, which
is work that doesn't pay for itself. The need it was meant to serve — "I hit
an unfamiliar term mid-paper and want it explained" — is better served in the
moment by highlight-to-ask below, which needs no marking step, no separate
store, and grounds the explanation in the passage actually on screen rather
than in a corpus-wide sense merge. `roboscholar/glossary.py` and
`glossary/terms.txt` are left on disk, unwired and uncommitted.

**Concept visualisation / simulation.** Floated 2026-09: for a concept like
action chunking or temporal ensembling, generate a small visual or
simulation of how it works instead of only prose. Undesigned — open
questions before this goes further: output format (an Artifact? a notebook
cell?), generated on-demand per question or pre-built per concept, and
whether "simulation" means an actual numeric simulation of the method or an
illustrative animation — those are very different scopes and worth pinning
down before any code gets written.

First real data point, 2026-09: while writing `pi05-003`, Stan got stuck on
the π0.5 policy-factorisation equation and asked for an on-demand walkthrough
as an Artifact rather than a chat explanation — a persistent equation with
terms that highlight/dim per step, a hover-synced symbol glossary, and a
small flow diagram for the "why ℓ is dropped" insight. Confirmed as exactly
the right shape ("this is exactly the sort we'd want to create in future for
when I don't understand the math"). Answers two of the open questions above:
on-demand (not pre-built) and per-question, not per-concept library. Doesn't
resolve the "numeric simulation vs. illustrative diagram" question — this was
firmly the latter, a notation walkthrough, not a simulation of anything
computing. Practical note: LaTeX (`$$...$$`) doesn't render in the CLI
terminal at all — chat replies with maths need plain notation, actual
equations need an Artifact.

**Browser highlight-to-ask.** Floated 2026-09: while reading a paper,
highlight an equation or passage and ask the agent about it in place, rather
than switching windows to paste it into a chat. Reference point was Benji's
[agentation](https://benji.org/agentation), but the mechanism doesn't
transfer directly — Agentation is a React component with DOM access to your
*own* app: annotate an element, get its selector and bounding box, paste
that into an agent. No vision model involved, because it never needs one; it
always has structured access to the page it's embedded in. RoboScholar's
target content is the opposite case — PDFs (no DOM at all) and arbitrary
external blog pages (no control over their markup) — which is why this
actually does need a screenshot-plus-vision path rather than DOM scraping,
plus a way to tell the agent which corpus document is on screen so the
answer grounds against that entry instead of answering cold. A real
extension build (content script, screenshot capture, vision call), not a CLI
feature, and a materially bigger scope than glossary or concept
visualisation. Stays parked behind retrieval.py and the eval harness like
the rest of this section.

Update 2026-09: [Clicky](https://www.heyclicky.com/) is a closer precedent
than Agentation — hotkey → full-screen screenshot → vision-capable frontier
model → spoken/on-screen answer, no browser extension, no persistent
screenshot storage, works identically across any app because it never
touches the DOM at all. That sidesteps the PDF-has-no-DOM problem above
entirely: the minimum build is a hotkey listener, a screenshot, and one
vision-capable `messages.create` call (Claude and GPT-4o-class models take
images natively, no separate VLM to source or host). What Clicky doesn't do
is grounding — knowing which corpus document is on screen so the answer
cites a retrieved chunk instead of the model's own reading of the pixels —
which is the actual RoboScholar contribution, not the capture mechanism.

Update 2026-09, second data point: while reading LeWorldModel, Stan hit
"representation collapse" and wanted to highlight the term and get it
explained *in the context of that specific sentence*, not a generic
definition. This is the same feature, not a third idea — and it argues for
the screenshot approach over pure text/DOM selection, because a screenshot
captures the surrounding paragraph for free, which a bare highlighted string
would lose. This use case is also what killed the glossary feature above —
a single in-the-moment lookup grounded in what's on screen beats marking a
term now to read a definition later.

Also floated: a "highlight this, generate an artifact" mode extending the
math-walkthrough pattern in [[feedback_math_walkthrough_artifacts]] to be
on-demand rather than something Claude Code offers proactively. Correctly
flagged as expensive — building the π0.5 artifact took real tool calls
(source extraction, layout design, several hundred lines of HTML/CSS/JS),
nothing like the cost of an ordinary `ask` call, so this should stay a
deliberate, occasional action rather than a per-highlight default. On model
choice: Sonnet built that artifact and it held up, so there's no evidence
yet that Opus is required — default to Sonnet (same tier as the agent loop
and judge) and only reconsider with real examples of Sonnet falling short on
a harder equation, not pre-emptively.
