# RoboScholar — Project Spec

> An agentic research assistant for studying embodied AI (ideally extensible to AI papers generally).
> A hand-written tool-use loop over a local vector store of robotics papers, with a real eval harness.
> **Timebox: 2 weeks to v1.** After that: maintenance mode — use it, don't develop it.

---

## Status — 22 July 2026

**In progress, built in the open.** This README is the spec I am building against, not a description
of finished software. What exists today:

**Built and tested**

- **Ingestion pipeline** (`roboscholar/ingest.py`, ~470 LOC, 20 passing tests) — PDF parsing via
  PyMuPDF block extraction that preserves section titles and page numbers, ToC-first heading
  detection with a font-size fallback for papers without bookmarks, section-aware chunking, and
  markdown ingestion. The section/page metadata is load-bearing rather than decorative: the eval
  harness scores retrieval against ground-truth locations, so it has to be right.
- **`rs ingest`** — runs end to end. Parses the five committed papers plus the Wayve GAIA posts and
  reports chunk counts, section counts and unmatched-heading warnings per document. Chunks are not
  persisted yet.

**Scaffolded, not implemented**

- `agent.py` — the hand-written tool-use loop (deliberately no framework)
- `retrieval.py` — embedding + Chroma storage
- `tools.py` — `search_corpus`, `compare_methods`, `quiz_me`
- `evals/` — retrieval metrics, LLM-as-judge, regression tracking. `golden.json` is a populated
  schema with placeholder questions; the real set is written by hand while reading each paper.

`rs ask`, `rs compare`, `rs quiz` and `rs eval` are registered and documented, but each exits with a
pointer to the file where the work lands. Nothing here pretends to work.

[`NOTES.md`](NOTES.md) is the running log of the non-obvious decisions and why they were made — why
PyMuPDF blocks rather than a flat text dump, why `sort=True` was rejected for two-column papers, why
the golden set is written by hand. §5 below explains why the eval harness is the part I will not skip.

---

## 1. What It Does (v1 scope)

- **Ingest** PDFs (the four papers first: ACT, Diffusion Policy, pi0, pi0.5) + markdown (LeRobot docs, own blog posts) into a local vector DB
- **Ask**: grounded Q&A over the corpus with citations to paper + section
- **Compare**: multi-step agentic synthesis across two papers ("how does ACT's action chunking differ from Diffusion Policy's?")
- **Quiz**: generate questions FROM retrieved chunks (grounded, not from model memory), track what you get wrong
- **Eval**: a real eval harness with golden set, retrieval metrics, LLM-as-judge, regression tracking

**Explicit non-goals for v1** (to resist creep): web UI, multi-user anything, agent memory across sessions, fine-tuning, streaming UI polish, supporting every document format.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│ CLI (Typer + Rich)  —  thin adapter             │
│   rs ingest | rs ask | rs compare | rs quiz |   │
│   rs eval run | rs eval report                  │
├─────────────────────────────────────────────────┤
│ Core library (roboscholar/)                     │
│                                                 │
│  agent.py      — the tool-use loop (~30-50 loc, │
│                  hand-written, no framework)    │
│  tools.py      — search_corpus, fetch_arxiv,    │
│                  compare_methods, quiz_me       │
│  ingest.py     — PDF/MD → structured chunks     │
│  retrieval.py  — embed, store, hybrid search    │
│  evals/        — golden.json, metrics.py,       │
│                  judge.py, runner.py            │
│  models.py     — Pydantic schemas (chunks,      │
│                  quiz questions, judge scores)  │
├─────────────────────────────────────────────────┤
│ Storage: Chroma (local, embedded) + SQLite      │
│ (quiz history, eval runs)                       │
│ LLM: Anthropic SDK — Sonnet (agent/judge),      │
│ Haiku (quiz gen, cheap tasks)                   │
│ Embeddings: voyage-3-lite or local              │
│ sentence-transformers (see tradeoffs)           │
└─────────────────────────────────────────────────┘
```

### The agent loop (the part you hand-write)

```Python
messages = [user_msg]
while True:
    response = client.messages.create(model=..., tools=TOOL_SCHEMAS, messages=messages)
    if response.stop_reason != "tool_use": return final_text
    for block in tool_use_blocks(response):
        result = dispatch(block.name, block.input)   # your Python functions
        messages.append(tool_result(block.id, result))
```

Tools are plain Python functions + a JSON schema each. The model chooses when to call them; your code executes.

---

## 3. Stack & Key Decisions

| Layer                | Choice                                                                                                                          | Why / Tradeoff                                                                                                                                                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language/env         | Python 3.13, uv, Ruff, Pydantic, marimo                                                                                         | Existing stack; consistency with CLAUDE.md                                                                                                                                                                                                                                                            |
| LLM                  | Anthropic SDK direct (no LangChain/LlamaIndex)                                                                                  | Depth signal — frameworks hide the loop you need to be able to explain. Tradeoff: more code, but the code IS the portfolio.**Note: API billing is separate from the Claude Code subscription** — budget single-digit £/month; use Haiku where possible, prompt caching for repeated contexts |
| Vector DB            | Chroma (embedded, local)                                                                                                        | Zero infra, persists to disk. Tradeoff vs Qdrant: fewer prod features, but v1 doesn't need them; swapping later is a good extension story                                                                                                                                                             |
| Embeddings           | Decide in week 1: Voyage AV (API, better quality) vs sentence-transformers all-MiniLM / bge-small locally on M1 (free, offline) | Run both on the golden set in the eval harness and let the numbers decide — this comparison is itself a blog-post section                                                                                                                                                                            |
| PDF parsing          | PyMuPDF (fitz) first; Docling if section structure comes out mangled                                                            | Academic PDFs are hostile (two columns, equations, figures). Do NOT silently degrade to naive text dump — chunk quality caps the whole system                                                                                                                                                        |
| Chunking             | ~500–1,000 tokens, respect section boundaries, small overlap; metadata: paper_id, section, page                                | The single highest-leverage RAG decision; make it a tunable parameter so evals can test variants                                                                                                                                                                                                      |
| Search               | Dense first; add BM25 (rank-bm25) + reciprocal rank fusion as the week-2 upgrade                                                | Hybrid search is a common interview topic — implementing RRF yourself is ~20 lines and great signal                                                                                                                                                                                                  |
| Interface            | Typer + Rich CLI; core-library-first layout                                                                                     | Ships in a day; FastAPI becomes a thin optional adapter later. Textual TUI explicitly rejected: complexity budget belongs to RAG/agent/evals                                                                                                                                                          |
| Storage (non-vector) | SQLite via stdlib                                                                                                               | Quiz history + eval run scores; trivial, durable                                                                                                                                                                                                                                                      |

---

## 4. Tools (v1)

| Tool                | Signature (conceptual)                                                    | Notes                                                                                                   |
| ------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `search_corpus`   | (query, k, paper_filter?) → chunks with citations                        | The RAG workhorse; hybrid search lives here                                                             |
| `fetch_arxiv`     | (arxiv_id or search terms) → abstract/metadata, optional download+ingest | Uses the free arXiv API; turns the corpus extensible                                                    |
| `compare_methods` | (topic, paper_a, paper_b) → structured comparison                        | Agentic: internally calls search_corpus per paper, synthesises. Multi-step reasoning demo               |
| `quiz_me`         | (paper/topic, n, difficulty) → Pydantic QuizQuestion[]                   | Generates FROM retrieved chunks; wrong answers logged to SQLite → later quizzes bias toward weak areas |

---

## 5. Eval Harness (the differentiator — do not skip)

1. **Golden set** (`evals/golden.json`): 25–30 entries: question, reference answer, source paper, section/page of the answer. Written BY HAND while reading the four papers (reading and eval-building are the same activity — this is why the timebox works)
2. **Retrieval metrics** (`metrics.py`, no LLM): hit-rate@k, MRR against labelled sources. Runs in seconds, free
3. **LLM-as-judge** (`judge.py`): Sonnet scores generated answers vs references on faithfulness (grounded in retrieved context?) + correctness, returning a Pydantic score object (structured output). Include 2–3 few-shot scored examples in the judge prompt for consistency
4. **Runner + regression** (`runner.py`): every run stores config (chunk size, k, embed model, hybrid on/off) + scores in SQLite. `rs eval report` renders a comparison table; marimo notebook charts runs over time
5. **Method**: change ONE variable → run evals → compare. The write-up of 3–4 such experiments (chunk size sweep, dense vs hybrid, embedding model A/B) is the heart of the blog post

---

## 6. Build Plan (2 weeks)

**Week 1 — vertical slice, end-to-end ugly:**

- D1–2: repo, uv, CLAUDE.md; ingest.py for ONE paper (ACT); chunks visible and sane
- D3: retrieval.py — embed, store in Chroma, `rs ask` with naive top-k RAG (no agent yet)
- D4: hand-write the agent loop + `search_corpus` as first tool; `rs ask` now agentic
- D5: golden set started (10 Qs from ACT); retrieval metrics running
- Weekend: read Diffusion Policy, ingest it, extend golden set

**Week 2 — depth + evals + ship:**

- D1: `fetch_arxiv` + `compare_methods`; ingest pi0 + pi0.5
- D2: `quiz_me` with structured outputs + SQLite history
- D3: LLM-as-judge + eval runner + regression storage
- D4: run the experiments (chunk sweep, hybrid vs dense, embedding A/B); marimo analysis
- D5: README, demo GIF/video, blog post draft: "I Built a Research Agent to Learn Robotics — With Evals"
- Hard stop Sunday. Whatever exists, ships.

---

## 7. Things to Research (short list, as-needed)

- Anthropic tool use docs (messages API, tool_use/tool_result blocks, structured outputs, prompt caching)
- Chunking strategies for academic PDFs (section-aware splitting)
- Reciprocal rank fusion (it's one formula)
- LLM-as-judge pitfalls: position bias, verbosity bias, why few-shot rubrics help
- arXiv API basics

---

## 8. CLAUDE.md Additions for This Repo

Append to the existing learning-first rules:

```markdown
## RoboScholar-Specific Rules

### I write myself (Track A interview material)
- The agent tool-use loop (agent.py) — every line
- Chunking logic and retrieval scoring (incl. RRF)
- The eval harness: metrics, judge prompts, runner
- All Pydantic schemas

### Claude Code handles
- Typer CLI boilerplate and Rich formatting
- PDF parsing plumbing once I've chosen the strategy
- SQLite schema/queries
- Test scaffolding, README structure

### Guardrails
- No LangChain, LlamaIndex, or agent frameworks — raw Anthropic SDK only
- No new features after Day 10; polish and evals only
- Every retrieval change must be justified by an eval run, not vibes
- If I propose a web UI, remind me of the v1 non-goals
```

---

## 9. Project Structure

```
robo-scholar/
├── roboscholar/              # core library (importable, CLI is a thin adapter)
│   ├── cli.py                # Typer app — installed as `rs` (boilerplate, done)
│   ├── agent.py              # the hand-written tool-use loop        [I write]
│   ├── tools.py              # search_corpus, fetch_arxiv,
│   │                         # compare_methods, quiz_me + schemas    [I write]
│   ├── ingest.py             # PDF/MD → structured chunks            [I write]
│   ├── retrieval.py          # embed, Chroma store, hybrid search    [I write]
│   ├── models.py             # Pydantic schemas                      [I write]
│   └── evals/
│       ├── golden.json       # hand-written golden set (skeleton in place)
│       ├── metrics.py        # hit-rate@k, MRR                       [I write]
│       ├── judge.py          # LLM-as-judge                          [I write]
│       └── runner.py         # run configs + scores → SQLite         [I write]
├── data/
│   ├── raw/
│   │   ├── papers/           # source PDFs (ACT, DP, pi0, pi0.5, world-models)
│   │   ├── blogs/            # downloaded blog posts as markdown
│   │   └── blogs.json        # manifest of blog sources (title + url)
│   ├── chroma/               # Chroma persistence dir     (gitignored)
│   └── roboscholar.db        # SQLite: quiz + eval runs   (gitignored)
├── notebooks/                # marimo analysis of eval runs
├── CLAUDE.md                 # working rules for Claude Code in this repo
└── pyproject.toml            # uv-managed; `rs` script entry point
```

Everything generated (Chroma index, SQLite) lives under `data/` and is rebuildable
from `data/raw/` — only raw sources and code are committed.

---

## 10. Getting Started

```bash
uv sync                      # install everything incl. the `rs` script
export ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com (billing is separate from Claude Code)
uv run rs --help             # CLI skeleton — commands exist, all raise "not implemented"
```

New dependencies: `uv add <package>` (never edit pyproject deps by hand).

### Setting up the Anthropic SDK (read, then write agent.py)

1. `client = anthropic.Anthropic()` — picks up `ANTHROPIC_API_KEY` from the env.
2. One endpoint does everything: `client.messages.create(model=..., max_tokens=..., tools=..., messages=...)`.
3. The loop: while `response.stop_reason == "tool_use"`, execute each `tool_use`
   block with your own functions and append a `tool_result` block (matched by
   `tool_use_id`) as the next user message. Multiple tool calls in one response →
   all results go back in a **single** user message.
4. Model IDs (current): `claude-sonnet-5` (agent + judge), `claude-haiku-4-5`
   (quiz gen, cheap tasks).
5. Cost control: use prompt caching (`cache_control: {"type": "ephemeral"}`) on the
   stable prefix — system prompt + tool schemas — since the agent re-sends them
   every loop iteration. Structured outputs (`client.messages.parse()` with a
   Pydantic model) is the clean way to get JudgeScore / QuizQuestion objects back.

### Setting up Chroma (read, then write retrieval.py)

1. Embedded mode, persisted to disk:
   `client = chromadb.PersistentClient(path="data/chroma")` — no server, no infra.
2. `collection = client.get_or_create_collection(name="papers")`.
3. `collection.add(ids=..., documents=..., embeddings=..., metadatas=...)` —
   metadatas carry `paper_id`, `section`, `page` for citations and eval labels.
4. `collection.query(query_embeddings=..., n_results=k, where={"paper_id": ...})`
   gives top-k with distances; `where` implements the paper_filter tool arg.
5. **Decision for the embedding A/B:** if you pass only `documents`, Chroma silently
   embeds with its default model (all-MiniLM-L6-v2 via ONNX). Instead, compute
   embeddings yourself and pass `embeddings=` explicitly — that makes the embedding
   model a swappable config value the eval harness can sweep (Voyage API vs local
   sentence-transformers), and keeps query/ingest embedding guaranteed-consistent.

### Embeddings — what they are and which we test

An embedding model maps text to a fixed-length vector such that semantically
similar texts land near each other (cosine similarity / small angle) in that
space. That's the whole trick behind dense retrieval: embed every chunk once at
ingest time, embed the question at query time, return the nearest chunks.

Embedders differ not in *what* they output (vectors) but in what their training
taught them to consider "similar":

- **Training objective + data** — modern embedders are transformer encoders
  trained contrastively: pull (query, relevant passage) pairs together, push
  hard negatives apart. The pair-mining and domains in that training data
  define the model's notion of similarity — a model trained on web Q&A pairs
  behaves differently on dense academic prose than one trained on scientific
  text.
- **Capacity + dimensionality** — more parameters and wider vectors (e.g. 384
  dims for the small local models vs 512+ for Voyage) can encode finer
  distinctions, at higher compute/storage cost.
- **Input limit** — the silent killer: MiniLM truncates at 256 tokens, bge-small
  at 512, Voyage handles 32k. A chunk longer than the limit is *partially*
  embedded with no error (see NOTES.md).

Our candidates (from §3): **voyage-3-lite** (API, strong quality tier at low
cost) vs **bge-small-en-v1.5 / all-MiniLM-L6-v2** (free, offline,
sentence-transformers on M1). None of these are state-of-the-art in absolute
terms — the MTEB leaderboard (https://huggingface.co/spaces/mteb/leaderboard)
is where SOTA lives and it churns monthly — they're the pragmatic band for a
small local corpus on a budget, and the A/B on our own golden set beats any
leaderboard number for *this* corpus anyway.

### Ingestion flow (once ingest.py + retrieval.py exist)

`rs ingest data/raw/papers` → PyMuPDF parse → section-aware chunks (~500–1,000
tokens, chunk size as a parameter) → embed → `collection.add(...)`. Sanity-check
chunks by eye before trusting anything downstream — chunk quality caps the system.

---

## 11. Documentation Reading List

| Topic                    | Link                                                                     | For                      |
| ------------------------ | ------------------------------------------------------------------------ | ------------------------ |
| Anthropic tool use       | https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview   | agent.py, tools.py       |
| Messages API + streaming | https://platform.claude.com/docs/en/build-with-claude/streaming          | agent.py                 |
| Structured outputs       | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | judge.py, quiz_me        |
| Prompt caching           | https://platform.claude.com/docs/en/build-with-claude/prompt-caching     | cost control in the loop |
| Models + pricing         | https://platform.claude.com/docs/en/about-claude/models/overview         | Sonnet/Haiku split       |
| Chroma getting started   | https://docs.trychroma.com/docs/overview/getting-started                 | retrieval.py             |
| PyMuPDF                  | https://pymupdf.readthedocs.io/                                          | ingest.py                |
| sentence-transformers    | https://sbert.net/                                                       | local embedding option   |
| Voyage AI                | https://docs.voyageai.com/                                               | API embedding option     |
| rank-bm25                | https://github.com/dorianbrown/rank_bm25                                 | week-2 hybrid search     |
| arXiv API                | https://info.arxiv.org/help/api/index.html                               | fetch_arxiv tool         |

---

## 12. Definition of Done (v1)

- [ ] Four papers + LeRobot docs ingested; `rs ask` answers with citations
- [ ] Agent loop hand-written, ≥3 tools working
- [ ] Quiz mode with history
- [ ] Golden set ≥25 Qs; retrieval + judge evals runnable with one command
- [ ] ≥3 documented experiments with before/after numbers
- [ ] README + demo video + blog post published
- [ ] I use it the next morning to study — and stop developing it
