"""Tests for `rs eval run` and `rs eval report`, with a fake sweep and a temporary .db."""

import pytest
from rich.console import Console
from typer.testing import CliRunner

from roboscholar import cli, db
from roboscholar.evals import runner
from tests.test_db import make_run


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temporary database, and a console wide enough that no column wraps."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rs.db")
    monkeypatch.setattr(cli, "console", Console(width=200))


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def test_report_without_sweeps():
    result = invoke("eval", "report")
    assert result.exit_code == 1
    assert "No sweeps saved yet" in result.output


def test_report_ranks_best_hit_rate_first():
    db.save_runs(
        [
            make_run(n_words=200, hit_rate_at_k=0.5),
            make_run(n_words=350, overlap_words=35, hit_rate_at_k=26 / 34),
        ],
        "20261116T100000Z",
    )
    result = invoke("eval", "report")
    assert result.exit_code == 0
    assert "20261116T100000Z" in result.output
    assert result.output.index("350w / 35") < result.output.index("200w / 20")
    assert "(26/34)" in result.output
    assert "-9" in result.output  # 17 hits is 9 behind the top row's 26


def test_report_list_and_pick_sweep():
    db.save_runs([make_run(n_words=200)], "20261115T100000Z")
    db.save_runs([make_run(n_words=350, overlap_words=35)], "20261116T100000Z")
    listed = invoke("eval", "report", "--list")
    assert "20261115T100000Z" in listed.output and "20261116T100000Z" in listed.output
    older = invoke("eval", "report", "--sweep", "20261115T100000Z")
    assert "200w / 20" in older.output and "350w" not in older.output
    missing = invoke("eval", "report", "--sweep", "nope")
    assert missing.exit_code == 1


def test_run_before_run_sweep_exists(monkeypatch):
    monkeypatch.delattr(runner, "run_sweep", raising=False)
    result = invoke("eval", "run")
    assert result.exit_code == 1
    assert "run_sweep isn't written yet" in result.output


def test_run_saves_rows_and_flags_failed_builds(monkeypatch):
    def fake_sweep(questions, k):
        assert len(questions) > 0 and k == 5
        return [make_run()], ["voyage-4-lite 350w / 70: RateLimitError: slow down"]

    monkeypatch.setattr(runner, "run_sweep", fake_sweep, raising=False)
    result = invoke("eval", "run")
    assert result.exit_code == 1  # a failed build fails the command
    assert "1 build(s) failed" in result.output
    assert "RateLimitError" in result.output
    ((sweep_id, _, n_rows),) = db.list_sweeps()
    assert n_rows == 1  # the rows that did score are still saved
    assert sweep_id in result.output
