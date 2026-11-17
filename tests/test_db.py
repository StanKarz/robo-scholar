"""Tests for the SQLite layer (roboscholar/db.py). Each test uses its own .db file."""

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from roboscholar import db
from roboscholar.models import EvalRun


def make_run(**overrides) -> EvalRun:
    fields = {
        "embedding_model": "bge-small",
        "has_query_prefix": True,
        "n_words": 200,
        "overlap_words": 20,
        "k": 5,
        "n_questions": 34,
        "hit_rate_at_k": 0.75,
        "hit_rate_or_at_k": 0.8,
        "mrr": 0.6,
    }
    return EvalRun.model_validate(fields | overrides)


def test_round_trip(tmp_path):
    path = tmp_path / "rs.db"
    runs = [
        make_run(),
        make_run(embedding_model="voyage-4-lite", has_query_prefix=False, notes="x"),
    ]
    db.save_runs(runs, "s1", path)
    assert db.load_sweep("s1", path) == runs


def test_sweeps_are_kept_apart(tmp_path):
    path = tmp_path / "rs.db"
    db.save_runs([make_run()], "s1", path)
    db.save_runs(
        [make_run(n_words=350), make_run(n_words=350, overlap_words=35)], "s2", path
    )
    assert len(db.load_sweep("s1", path)) == 1
    assert [run.overlap_words for run in db.load_sweep("s2", path)] == [20, 35]


def test_sweeps_listed_newest_first(tmp_path):
    path = tmp_path / "rs.db"
    earlier = datetime(2026, 11, 15, 9, 0, tzinfo=timezone.utc)
    later = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)
    db.save_runs([make_run(timestamp=earlier)], "old", path)
    db.save_runs([make_run(timestamp=later), make_run(timestamp=later)], "new", path)
    assert [(sweep_id, n) for sweep_id, _, n in db.list_sweeps(path)] == [
        ("new", 2),
        ("old", 1),
    ]


def test_unknown_sweep_is_empty(tmp_path):
    assert db.load_sweep("nope", tmp_path / "rs.db") == []
    assert db.list_sweeps(tmp_path / "rs.db") == []


def test_table_matches_eval_run(tmp_path):
    # Adding a field to EvalRun without a column here would lose it silently.
    path = tmp_path / "rs.db"
    with closing(db.connect(path)) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    assert columns - set(db.TABLE_ONLY_COLUMNS) == set(EvalRun.model_fields)


def test_failed_insert_saves_nothing(tmp_path):
    path = tmp_path / "rs.db"
    good = make_run()
    bad = make_run()
    object.__setattr__(bad, "notes", None)  # violates NOT NULL
    try:
        db.save_runs([good, bad], "s1", path)
    except sqlite3.IntegrityError:
        pass
    assert db.load_sweep("s1", path) == []


def test_sweep_id_format():
    assert re.fullmatch(r"\d{8}T\d{6}Z", db.new_sweep_id())
