"""Tests for the score-A metrics. Pure: no index, no embedder, no API calls.

The ACT rank lists are real results: ACT-only index (500 words / 50 overlap,
bge-small), the 10 ACT questions, k=5, checked by hand on 2026-11-15.
"""

from pathlib import Path

import pytest

from roboscholar.evals.metrics import first_correct_rank, hit_rate, is_correct, mrr
from roboscholar.models import GoldenQuestion, GoldenSet, RetrievedChunk

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "roboscholar" / "evals" / "golden.json"
)

# act-001 to act-010. Under "or", act-008's rank-2 chunk is on a labelled page.
ACT_RANKS_SECTION = [1, None, None, 1, 1, 2, 1, None, 1, 1]
ACT_RANKS_OR = [1, None, None, 1, 1, 2, 1, 2, 1, 1]


def make_chunk(
    paper_id: str = "act",
    section: str = "IV-A",
    page_start: int = 4,
    page_end: int = 5,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{paper_id}:0000",
        text="...",
        paper_id=paper_id,
        section=section,
        page_start=page_start,
        page_end=page_end,
        n_words=300,
        distance=0.2,
    )


def make_question(
    source_paper: str = "act",
    source_sections: list[str] | None = None,
    source_pages: list[int] | None = None,
) -> GoldenQuestion:
    return GoldenQuestion(
        id="act-006",
        question="What does temporal ensembling do?",
        reference_answer="...",
        source_paper=source_paper,
        source_sections=["IV-A"] if source_sections is None else source_sections,
        source_pages=[5] if source_pages is None else source_pages,
        difficulty="medium",
        tags=[],
    )


class TestIsCorrect:
    @pytest.mark.parametrize("rule", ["section", "or"])
    def test_section_match_counts_under_both_rules(self, rule):
        assert is_correct(make_chunk(), make_question(), rule)

    @pytest.mark.parametrize("rule", ["section", "or"])
    def test_wrong_paper_never_counts(self, rule):
        # Same section name and pages, different paper: "Introduction" exists everywhere.
        chunk = make_chunk(paper_id="diffusion-policy")
        assert not is_correct(chunk, make_question(), rule)

    def test_section_compared_as_exact_string(self):
        question = make_question(source_sections=["IV › IV-A"], source_pages=[])
        assert not is_correct(make_chunk(section="IV-A"), question)
        assert is_correct(make_chunk(section="IV › IV-A"), question)

    def test_page_overlap_counts_only_under_or(self):
        chunk = make_chunk(section="VI")  # pages 4-5, label page 5
        assert not is_correct(chunk, make_question(), "section")
        assert is_correct(chunk, make_question(), "or")

    def test_label_page_strictly_inside_chunk_range(self):
        chunk = make_chunk(section="Appendix", page_start=15, page_end=17)
        assert is_correct(chunk, make_question(source_pages=[16]), "or")

    def test_label_page_outside_chunk_range(self):
        chunk = make_chunk(section="VI")
        assert not is_correct(chunk, make_question(source_pages=[3, 6]), "or")

    def test_blog_falls_back_to_section_match(self):
        question = make_question(source_paper="gaia-2", source_pages=[])
        blog = make_chunk(paper_id="gaia-2", page_start=0, page_end=0)
        assert is_correct(blog, question, "or")
        other = make_chunk(
            paper_id="gaia-2", section="Results", page_start=0, page_end=0
        )
        assert not is_correct(other, question, "or")


class TestFirstCorrectRank:
    def test_rank_counts_from_one(self):
        chunks = [make_chunk(section="VI", page_start=9, page_end=9), make_chunk()]
        assert first_correct_rank(chunks, make_question()) == 2

    def test_only_the_first_correct_chunk_matters(self):
        assert first_correct_rank([make_chunk(), make_chunk()], make_question()) == 1

    def test_no_correct_chunk(self):
        chunks = [make_chunk(section="VI", page_start=9, page_end=9)] * 5
        assert first_correct_rank(chunks, make_question()) is None

    def test_no_chunks(self):
        assert first_correct_rank([], make_question()) is None

    def test_rule_is_passed_through(self):
        page_only = make_chunk(section="VI")  # right page, wrong section
        chunks = [page_only, make_chunk()]
        assert first_correct_rank(chunks, make_question(), "section") == 2
        assert first_correct_rank(chunks, make_question(), "or") == 1


class TestHitRateAndMrr:
    def test_act_known_answers_section(self):
        assert hit_rate(ACT_RANKS_SECTION) == pytest.approx(0.70)
        assert mrr(ACT_RANKS_SECTION) == pytest.approx(0.65)

    def test_act_known_answers_or(self):
        assert hit_rate(ACT_RANKS_OR) == pytest.approx(0.80)
        assert mrr(ACT_RANKS_OR) == pytest.approx(0.70)

    def test_hit_rate_ignores_position_mrr_does_not(self):
        ranks: list[int | None] = [1, 2, 5]
        assert hit_rate(ranks) == pytest.approx(1.0)
        assert mrr(ranks) == pytest.approx((1 + 0.5 + 0.2) / 3)

    def test_misses_count_in_the_divisor(self):
        # Dividing by hits instead of questions would give 1.0 and 0.5.
        assert hit_rate([2, None]) == pytest.approx(0.5)
        assert mrr([2, None]) == pytest.approx(0.25)

    @pytest.mark.parametrize("metric", [hit_rate, mrr])
    def test_all_misses_score_zero(self, metric):
        assert metric([None, None, None]) == 0.0

    @pytest.mark.parametrize("metric", [hit_rate, mrr])
    def test_empty_raises(self, metric):
        with pytest.raises(ValueError):
            metric([])


def test_golden_set_parses():
    assert GoldenSet.load(GOLDEN_PATH).questions
