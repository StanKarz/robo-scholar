# RoboScholar

An agentic research assistant for studying embodied-AI papers: grounded Q&A,
multi-paper comparison, and quizzing over a local corpus of papers and blog
posts — with a hand-written tool-use loop (no agent framework) and a real
eval harness instead of vibes.

Built in the open as a learning project: the point is to understand and be
able to explain every part of it, not just to have it work. `CLAUDE.md` has
the working rules and division of labour; `NOTES.md` has the pipeline
walkthrough and the running decision log — that's where "why is it built
this way" actually gets answered.

## What it does

- **Ingest** — PDFs and blog posts → section-aware chunks with page/section
  metadata, embedded and stored locally (Chroma)
- **Ask** — grounded Q&A with citations to paper + section, via a
  hand-written agent loop over the Anthropic SDK
- **Compare** — multi-step synthesis across two papers, e.g. "how does ACT's
  action chunking differ from Diffusion Policy's?"
- **Quiz** — questions generated from retrieved chunks, not model memory;
  wrong answers tracked over time
- **Eval** — golden set + retrieval metrics (hit-rate@k, MRR) + LLM-as-judge,
  so every retrieval change is justified by a number, not a feeling

**Parked ideas** — written down, not built (see `NOTES.md` → Ideas): a
concept glossary grounded in the corpus; visualising or simulating concepts
from a paper.

**Non-goals**: web UI, multi-user anything, fine-tuning, supporting every
document format. This is a CLI over a local corpus, on purpose.

## Status

Ingestion is built and tested (`roboscholar/ingest.py`, PDF + markdown, ToC
and font-based section detection, section-aware chunking). Retrieval
(embeddings + Chroma) and the eval harness are the current focus — see
`NOTES.md` for what's built vs. scaffolded.

## Quick start

```bash
uv sync
uv run rs fetch                       # download the paper corpus, verify checksums
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com — billing is separate
uv run rs --help
```

## Project structure

```
roboscholar/
├── cli.py         # rs — Typer CLI
├── ingest.py      # PDF/MD → chunks                       (built)
├── retrieval.py   # embed, Chroma store, hybrid search     (next)
├── agent.py       # hand-written tool-use loop
├── tools.py       # search_corpus, compare_methods, quiz_me
├── models.py      # Pydantic schemas
└── evals/         # golden.json, metrics.py, judge.py, runner.py
data/raw/          # papers.json + blogs.json manifests, fetched corpus
```

## Docs

- `NOTES.md` — how the pipeline fits together, and why each decision was made
- `CLAUDE.md` — working rules for Claude Code in this repo
