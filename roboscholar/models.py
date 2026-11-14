"""Pydantic schemas: Chunk, RetrievedChunk, QuizQuestion, JudgeScore, EvalRun."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """One indexed chunk: its text plus where it came from in the source."""

    id: str
    text: str
    paper_id: str
    section: str
    page_start: int
    page_end: int
    n_words: int


class RetrievedChunk(Chunk):
    """A Chunk returned by rank_chunks, with its cosine distance to the query."""

    distance: float


class QuizQuestion(BaseModel):
    id: int
    paper_id: str
    question: str
    answer: str
    section: str
    chunk_ids: list[str] = Field(
        description="IDs of the retrieved chunks this question was generated from."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty level of the question being asked"
    )


class JudgeScore(BaseModel):
    """One judged answer, scored on two axes against two different references.

    Field order matters: structured outputs are generated in declaration order,
    so reasoning is written before either score is committed to.
    """

    question_id: str
    reasoning: str = Field(
        description="Justification for both scores below, written before scoring."
    )
    correctness: int = Field(
        ge=1,
        le=5,
        description=(
            "How well the generated answer matches the hand-written reference answer. "
            "1 means it contradicts the reference, 5 means it fully matches."
        ),
    )
    groundedness: int = Field(
        ge=1,
        le=5,
        description=(
            "How well the generated answer is supported by the retrieved chunks it was "
            "given. 1 means it asserts things absent from those chunks, 5 means every "
            "claim traces back to them."
        ),
    )


class EvalRun(BaseModel):
    """One row of the sweep: a config and its score-A retrieval metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # config - every setting that changes which chunks come back, so rows are
    # comparable. When hybrid search lands, add a ranking-method field.
    embedding_model: Literal["bge-small", "voyage-4-lite"] = Field(
        description="Embedding model used for this run"
    )
    has_query_prefix: bool = Field(
        description=(
            "Whether queries got the model's search instruction: bge's prefix, "
            'or input_type="query" for Voyage'
        )
    )

    n_words: int = Field(gt=0, description="Chunk size in words")
    n_questions: int = Field(
        gt=0,
        description="Golden questions scored in this run; comparisons are only valid across runs with the same value",
    )
    overlap_words: int = Field(ge=0, description="Word overlap between adjacent chunks")
    k: int = Field(gt=0, description="k used for retrieval (top k results considered)")

    # aggregate metrics
    hit_rate_at_k: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of golden questions with a correct chunk in the top k, where "
            "correct means paper and section match. This one decides the sweep."
        ),
    )
    hit_rate_or_at_k: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The same under the lenient rule (section match OR page overlap). "
            "A sanity column: if it picks a different winner, read the failures."
        ),
    )
    mrr: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean reciprocal rank of the first correct chunk (section match)",
    )

    notes: str = Field(
        default="",
        description="Free-text notes, e.g. cost, latency, or rationale for this config choice",
    )

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> EvalRun:
        """Reject a config whose overlap is as long as the chunk itself."""
        if self.overlap_words >= self.n_words:
            raise ValueError("overlap_words must be smaller than n_words")
        return self
