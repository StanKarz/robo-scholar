"""Eval runner: the retrieval sweep. Rebuilds the index for each config, scores
every golden question against it, and returns one EvalRun per row; `rs eval run`
saves the rows to SQLite (data/roboscholar.db).
"""

from __future__ import annotations

from itertools import product
from typing import Literal

from voyageai.error import VoyageError

from roboscholar.evals.metrics import first_correct_rank, hit_rate, mrr
from roboscholar.fetch import corpus_files
from roboscholar.ingest import ingest_file
from roboscholar.models import EvalRun, GoldenQuestion
from roboscholar.retrieval import index_chroma, rank_chunks

# The grid fixed in NOTES.md (12 Nov): 2 models x 2 sizes x 3 overlaps is 12
# builds, and prefix off for bge only makes 18 rows. Tuples keep the order
# fixed, and bge comes first so the local builds finish before any Voyage call.
MODELS = ("bge-small", "voyage-4-lite")
N_WORDS = (200, 350)
OVERLAP_PCTS = (0, 10, 20)
PREFIX_SETTINGS = {"bge-small": (True, False), "voyage-4-lite": (True,)}


def run_sweep(
    questions: list[GoldenQuestion], k: int = 5
) -> tuple[list[EvalRun], list[str]]:
    """Build and score every config in the grid.

    Each (model, chunk size, overlap) combination is ingested and indexed once,
    then scored once per prefix setting that model has. A build that fails to
    index is reported and skipped, never scored.

    Args:
        questions: The golden questions every config is scored on.
        k: How many chunks each question retrieves.

    Returns:
        The scored rows (18 for the full grid), and one message per failed
        build naming its config and the error.
    """
    runs = []
    failures = []
    for model, n_words, pct in product(MODELS, N_WORDS, OVERLAP_PCTS):
        overlap_words = n_words * pct // 100
        records = []
        for path in corpus_files():
            chunk_records, _ = ingest_file(
                path, n_words=n_words, overlap_words=overlap_words
            )
            records.extend(chunk_records)
        try:
            index_chroma(records, model)
        except (ValueError, VoyageError) as e:
            failed_config = (
                f"{model} {n_words}w / {overlap_words} ({len(records)} chunks): "
                f"{type(e).__name__}: {e}"
            )
            print(failed_config)
            failures.append(failed_config)
            # data/chroma still holds the previous build: scoring it here would
            # save that build's numbers under this config's label
            continue
        print(
            f"{model} {n_words}w / {overlap_words}  indexed {len(records)} chunks: COMPLETED"
        )

        for has_query_prefix in PREFIX_SETTINGS[model]:
            runs.append(
                score_config(
                    questions,
                    embedding_model=model,
                    n_words=n_words,
                    overlap_words=overlap_words,
                    has_query_prefix=has_query_prefix,
                    k=k,
                )
            )

    return runs, failures


def score_config(
    questions: list[GoldenQuestion],
    *,
    embedding_model: Literal["bge-small", "voyage-4-lite"],
    n_words: int,
    overlap_words: int,
    has_query_prefix: bool,
    k: int,
) -> EvalRun:
    """Score the index in data/chroma on the golden questions.

    Retrieves once per question and judges the same chunks under both rules.
    Nothing is rebuilt here: the config arguments only label the row, so they
    must describe the index that is actually loaded.

    Args:
        questions: The golden questions to score.
        embedding_model: The model the loaded index was built with.
        n_words: Chunk size of the loaded index, in words.
        overlap_words: Chunk overlap of the loaded index, in words.
        has_query_prefix: Whether bge queries get their instruction prefix
            (Voyage ignores it).
        k: How many chunks to retrieve per question.

    Returns:
        One row: hit-rate@k under section match and under OR, and MRR under
        section match.
    """
    section_ranks = []
    or_ranks = []
    for q in questions:
        chunks = rank_chunks(
            query_str=q.question, k=k, has_query_prefix=has_query_prefix
        )
        section_ranks.append(first_correct_rank(chunks, q, rule="section"))
        or_ranks.append(first_correct_rank(chunks, q, rule="or"))

    return EvalRun(
        embedding_model=embedding_model,
        n_words=n_words,
        has_query_prefix=has_query_prefix,
        k=k,
        overlap_words=overlap_words,
        n_questions=len(questions),
        hit_rate_at_k=hit_rate(section_ranks),
        hit_rate_or_at_k=hit_rate(or_ranks),
        mrr=mrr(section_ranks),
    )
