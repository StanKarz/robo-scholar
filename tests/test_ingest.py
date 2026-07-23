"""Tests for the ingest pipeline. Unit tests are corpus-independent; the
integration test uses the committed ACT paper."""

from pathlib import Path

import pytest

from roboscholar.ingest import (
    CAPTION_RE,
    chunk_section,
    consume_key,
    ingest_markdown,
    ingest_pdf,
    md_sections,
    norm_key,
    strip_numbering,
)

ACT_PDF = Path(__file__).parent.parent / "data" / "raw" / "papers" / "ACT.pdf"


class TestNormalization:
    def test_small_caps_extraction(self):
        assert norm_key("I NTRODUCTION") == norm_key("Introduction")

    def test_strip_roman_numbering(self):
        assert strip_numbering("IV. Action Chunking") == "Action Chunking"
        assert strip_numbering("IV-A Action Chunking") == "Action Chunking"

    def test_strip_decimal_and_appendix_numbering(self):
        assert strip_numbering("2.1 DDPM Training") == "DDPM Training"
        assert strip_numbering("D.0.1 UR5 robot station") == "UR5 robot station"

    def test_plain_text_untouched(self):
        assert strip_numbering("Introduction") == "Introduction"


class TestConsumeKey:
    def test_full_match_returns_remainder(self):
        rest = consume_key("ACTION CHUNKING As we will see", norm_key("Action Chunking"))
        assert rest == "As we will see"

    def test_exact_match_returns_empty(self):
        assert consume_key("Action Chunking", norm_key("Action Chunking")) == ""

    def test_mismatch_returns_none(self):
        assert consume_key("Something else", norm_key("Action Chunking")) is None

    def test_partial_text_returns_none(self):
        assert consume_key("Action", norm_key("Action Chunking")) is None


class TestCaptionFilter:
    @pytest.mark.parametrize(
        "caption",
        ["Fig. 1: ALOHA setup", "Figure 2. Results", "TABLE I: Success rates", "Table 3: Ablations"],
    )
    def test_captions_match(self, caption):
        assert CAPTION_RE.match(caption)

    def test_body_text_mentioning_figure_kept(self):
        assert not CAPTION_RE.match("Figure 3 shows the results of our ablation")


class TestChunking:
    def _section(self, n_sentences, words_per_sentence=10):
        # distinct, capitalized sentences so SENT_RE splits them and equality
        # checks between chunks are meaningful
        sents = [
            f"Sentence{i} " + " ".join(["word"] * (words_per_sentence - 2)) + " end."
            for i in range(n_sentences)
        ]
        return {"title": "T", "level": 1, "blocks": [(1, " ".join(sents))]}

    def test_respects_word_budget(self):
        chunks = chunk_section(self._section(20), n_words=50, overlap_words=0)
        assert all(sum(len(s.split()) for s, _ in c) <= 50 for c in chunks)

    def test_overlap_repeats_trailing_sentences(self):
        chunks = chunk_section(self._section(20), n_words=50, overlap_words=10)
        assert len(chunks) >= 2
        assert chunks[0][-1] == chunks[1][0]  # last sentence of A seeds B

    def test_no_overlap_no_repeats(self):
        chunks = chunk_section(self._section(20), n_words=50, overlap_words=0)
        flat = [s for c in chunks for s, _ in c]
        assert len(flat) == 20

    def test_giant_sentence_hard_split(self):
        sec = {"title": "T", "level": 1, "blocks": [(1, " ".join(["eq"] * 300))]}
        chunks = chunk_section(sec, n_words=100, overlap_words=0)
        assert all(sum(len(s.split()) for s, _ in c) <= 100 for c in chunks)


class TestMarkdown:
    MD = "# Title\n\nIntro para one two three.\n\n## Section A\n\nBody of section A.\n\n### Sub\n\nNested body."

    def test_sections_and_levels(self):
        secs = md_sections(self.MD)
        titles = [(s["title"], s["level"]) for s in secs]
        assert ("Title", 1) in titles
        assert ("Section A", 1) in titles  # h2 is top-level: h1 is the post title
        assert ("Sub", 2) in titles

    def test_ingest_markdown_schema(self, tmp_path):
        md = tmp_path / "post.md"
        md.write_text("# Post\n\n" + "This is a sentence with several words in it. " * 20)
        records = ingest_markdown(md, n_words=100, overlap_words=0)
        assert records, "expected at least one chunk"
        r = records[0]
        assert r["id"] == "post:0000"
        assert r["paper_id"] == "post"
        assert r["page_start"] == 0  # markdown has no pages


@pytest.mark.skipif(not ACT_PDF.exists(), reason="corpus not present")
class TestActIntegration:
    def test_act_ingests_cleanly(self):
        records, unmatched = ingest_pdf(ACT_PDF)
        assert not unmatched, f"unmatched headings: {[h['title'] for h in unmatched]}"
        assert len(records) > 20
        sections = {r["section"] for r in records}
        assert any("Action Chunking" in s for s in sections)
        assert not any("References" in s for s in sections)
        # pages must be plausible labels for the golden set
        assert all(1 <= r["page_start"] <= r["page_end"] for r in records)
