"""Retrieval metrics (score A): hit-rate@k and MRR against golden.json labels.

Pure functions: no Chroma, no LLM, no file reading. The caller ranks chunks
for each golden question and passes them in. Definitions and the decision
rule: NOTES.md, "Retrieval metrics". Tests: tests/test_metrics.py.
"""

from __future__ import annotations

from typing import Literal

from roboscholar.models import GoldenQuestion, RetrievedChunk

# Which chunks count as correct. "section" decides the sweep; "or" is a
# diagnostic column (NOTES.md, "Section match decided 2026-11-12").
Rule = Literal["section", "or"]


def is_correct(
    chunk: RetrievedChunk, question: GoldenQuestion, rule: Rule = "section"
) -> bool:
    """Whether one retrieved chunk counts as correct for a golden question.

    The chunk's paper must be the question's `source_paper`. Under "section"
    its section must also be one of `source_sections`, as an exact string.
    "or" accepts that, or a page range that includes a labelled page. Blog
    labels have no pages, so for blogs the two rules agree.

    Args:
        chunk: One result from rank_chunks.
        question: The question it was retrieved for.
        rule: Which definition of correct to apply.

    Returns:
        True if the chunk counts as correct under `rule`.
    """
    if chunk.paper_id != question.source_paper:
        return False
    elif chunk.section in question.source_sections:
        return True
    elif rule == "or":
        for p in question.source_pages:
            if chunk.page_start <= p <= chunk.page_end:
                return True
    return False


def first_correct_rank(
    chunks: list[RetrievedChunk], question: GoldenQuestion, rule: Rule = "section"
) -> int | None:
    """The rank of the first correct chunk, counting from 1, or None.

    Args:
        chunks: rank_chunks' results for the question, nearest first.
        question: The question they were retrieved for.
        rule: Passed on to is_correct.

    Returns:
        The 1-based rank of the first chunk is_correct accepts, or None.
    """
    for idx, chunk in enumerate(chunks, start=1):
        if is_correct(chunk, question, rule):
            return idx
    return None


def hit_rate(ranks: list[int | None]) -> float:
    """The share of questions with a correct chunk in the top k.

    k is however many chunks rank_chunks returned per question.

    Args:
        ranks: One first_correct_rank result per question; None is a miss.

    Returns:
        A fraction in [0, 1].

    Raises:
        ValueError: If `ranks` is empty.
    """
    if not ranks:
        raise ValueError("Can't get hit rate on empty list")
    n_hits = []
    for rank in ranks:
        if rank is not None:
            n_hits.append(True)
    return sum(n_hits) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: the average of 1 / rank, a miss counting as 0.

    Rank 1 scores 1, rank 2 scores 0.5, rank 5 scores 0.2.

    Args:
        ranks: One first_correct_rank result per question; None is a miss.

    Returns:
        A value in [0, 1].

    Raises:
        ValueError: If `ranks` is empty.
    """
    if not ranks:
        raise ValueError("Can't get MRR on empty list")
    score = 0
    for rank in ranks:
        if rank is not None:
            score += 1 / rank
    return score / len(ranks)
