# Comprehension log

My own running notes on understanding this project — separate from
`NOTES.md` (the decision log: why things were built a certain way) and
`CLAUDE.md` (working rules). This one's personal and informal: what clicked,
what's still fuzzy, what to revisit. Newest entries at the bottom.

Shape for a new entry:

```
## <date> — <topic>
Question: what was confusing
Answer: what resolved it, in my own words
Still fuzzy: anything left over
```

---

## 2026-09-02 — resurrecting after 5 weeks; why RAG at all

Prompt for this: tried explaining the project to a friend and couldn't
answer "why not just paste the paper into an LLM and ask it to quiz you?"
Fair question. Full pipeline explanation now lives in `NOTES.md` → "How it
all fits together" — this is the short version, in my own words.

**Why RAG here at all, honestly:** for one paper in one sitting, pasting it
into a long-context chat and asking for a quiz would work fine, maybe
faster. That's not a flaw in the argument, it's just a different tradeoff.
This project's actual reason to exist is the infrastructure — chunking,
retrieval, eval harness — and being able to explain how it works, which is
the Track-A-interview-relevant skill. The quizzing is a real side benefit of
having built it, not the sole justification for building it. Worth
remembering next time I lose the thread on why I'm doing this.

**What clicked:**

- Chunks get *vector* embeddings — one dense vector per chunk (a few hundred
  words), not one vector per word (that's the older word2vec approach).
  Cosine similarity between those vectors is what "search" means here.
- Cosine similarity only ever answers "is this chunk relevant?" — never
  "is the generated answer correct?" Those are different questions. Two
  sentences can sit close in embedding space while saying opposite things
  ("X causes Y" vs. "X prevents Y" are topically near-identical vectors).
- So the eval harness is two independent scores, not one: retrieval metrics
  (hit-rate@k, MRR — did search find the *labelled* chunk? free, instant,
  no LLM) and LLM-as-judge (given that chunk, was the *written answer*
  actually right? costs a real LLM call). They diagnose different failure
  modes and aren't meant to combine into one number.
- Section-aware chunking isn't just tidiness — the golden set's ground truth
  literally *is* a section/page label, so sectioning is what the eval
  harness scores retrieval against.

**Still fuzzy — revisit once retrieval.py exists:**

- What hybrid search (dense + BM25 + RRF) actually buys over dense-only,
  concretely, once there are real numbers on the golden set to look at.
- How `compare_methods` will orchestrate two `search_corpus` calls and
  synthesise a comparison — haven't traced the agent loop shape yet since
  `agent.py` isn't written.
