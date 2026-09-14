"""Golden-set labels must match what ingest actually emits.

Section names are the join key between golden.json and chunk metadata: a label
that drifts from the emitted form is a guaranteed silent miss on hit-rate@k and
MRR, for every config (see SECTIONING_QUESTIONS.md). This suite turns that
failure mode from "quietly lower scores" into a red test that lists near-miss
candidates. Placeholder entries (containing REPLACE ME) are skipped.

Needs the fetched corpus: run `uv run rs fetch` first if data/raw/papers is empty.
"""

import difflib
import json
from pathlib import Path

import pytest

from roboscholar.ingest import ingest_file

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "roboscholar" / "evals" / "golden.json"
CORPUS_DIRS = (ROOT / "data" / "raw" / "papers", ROOT / "data" / "raw" / "blogs")
DIFFICULTIES = {"easy", "medium", "hard"}


def _is_placeholder(question: dict) -> bool:
    return "REPLACE ME" in json.dumps(question)


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text())["questions"]


@pytest.fixture(scope="module")
def emitted() -> dict[str, set[str]]:
    """doc_id -> section labels the ingest pipeline emits, whole corpus."""
    files = [
        p
        for d in CORPUS_DIRS
        for suffix in ("*.pdf", "*.md")
        for p in sorted(d.glob(suffix))
    ]
    if not files:
        pytest.skip("corpus not fetched - run `uv run rs fetch`")
    sections: dict[str, set[str]] = {}
    for f in files:
        records, _ = ingest_file(f, 500, 50)
        for r in records:
            sections.setdefault(r["paper_id"], set()).add(r["section"])
    return sections


def test_ids_unique(golden):
    ids = [q["id"] for q in golden]
    assert len(ids) == len(set(ids))


def test_difficulties_valid(golden):
    bad = [(q["id"], q["difficulty"]) for q in golden if q["difficulty"] not in DIFFICULTIES]
    assert not bad, f"invalid difficulty values: {bad}"


def test_source_papers_exist(golden, emitted):
    bad = [
        (q["id"], q["source_paper"])
        for q in golden
        if not _is_placeholder(q) and q["source_paper"] not in emitted
    ]
    assert not bad, (
        f"source_paper not among ingested doc_ids {sorted(emitted)}: {bad}"
    )


def test_source_sections_match_emitted(golden, emitted):
    failures = []
    for q in golden:
        if _is_placeholder(q) or q["source_paper"] not in emitted:
            continue
        doc_sections = emitted[q["source_paper"]]
        for label in q["source_sections"]:
            if label in doc_sections:
                continue
            near = difflib.get_close_matches(label, doc_sections, n=2, cutoff=0.4)
            near += [
                s
                for s in doc_sections
                if (label.lower() in s.lower() or s.lower() in label.lower())
                and s not in near
            ]
            failures.append(f"  {q['id']}: {label!r}\n    near misses: {near!r}")
    assert not failures, (
        "golden source_sections not found in ingested chunk metadata "
        "(labels must EXACTLY equal the emitted form):\n" + "\n".join(failures)
    )
