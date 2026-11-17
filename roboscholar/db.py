"""SQLite storage for eval runs: data/roboscholar.db, one row per EvalRun.

Rows are grouped by `sweep_id`, one per `rs eval run`, so a report compares
rows that were scored together. The file is gitignored and rebuilt by
rerunning the sweep. The table mirrors EvalRun's fields, and
tests/test_db.py fails if the two drift apart. If EvalRun gains a field,
delete the .db file so the table is recreated with the new column.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from roboscholar.models import EvalRun

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "roboscholar.db"

# Columns that belong to the table, not to EvalRun.
TABLE_ONLY_COLUMNS = ("id", "sweep_id")

CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id         TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    embedding_model  TEXT    NOT NULL,
    has_query_prefix INTEGER NOT NULL,
    n_words          INTEGER NOT NULL,
    overlap_words    INTEGER NOT NULL,
    k                INTEGER NOT NULL,
    n_questions      INTEGER NOT NULL,
    hit_rate_at_k    REAL    NOT NULL,
    hit_rate_or_at_k REAL    NOT NULL,
    mrr              REAL    NOT NULL,
    notes            TEXT    NOT NULL DEFAULT ''
)
"""

INSERT_EVAL_RUN = """
INSERT INTO eval_runs (
    sweep_id, timestamp, embedding_model, has_query_prefix, n_words,
    overlap_words, k, n_questions, hit_rate_at_k, hit_rate_or_at_k, mrr, notes
) VALUES (
    :sweep_id, :timestamp, :embedding_model, :has_query_prefix, :n_words,
    :overlap_words, :k, :n_questions, :hit_rate_at_k, :hit_rate_or_at_k, :mrr, :notes
)
"""

SELECT_SWEEP = "SELECT * FROM eval_runs WHERE sweep_id = ? ORDER BY id"

SELECT_SWEEPS = """
SELECT sweep_id, MIN(timestamp) AS started, COUNT(*) AS n_rows
FROM eval_runs
GROUP BY sweep_id
ORDER BY started DESC
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database, creating the file and the table if they don't exist.

    Args:
        path: The .db file. Defaults to DB_PATH; tests pass a temporary one.
    """
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_EVAL_RUNS)
    return conn


def new_sweep_id() -> str:
    """A sortable, readable id for one sweep, e.g. "20261116T193012Z" (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_runs(runs: list[EvalRun], sweep_id: str, path: Path | None = None) -> None:
    """Insert one row per run, all under `sweep_id`, in a single transaction."""
    rows = [{"sweep_id": sweep_id, **run.model_dump(mode="json")} for run in runs]
    with closing(connect(path)) as conn, conn:
        conn.executemany(INSERT_EVAL_RUN, rows)


def load_sweep(sweep_id: str, path: Path | None = None) -> list[EvalRun]:
    """Every run saved under `sweep_id`, in the order they were saved.

    Returns an empty list if there is no such sweep.
    """
    with closing(connect(path)) as conn:
        rows = conn.execute(SELECT_SWEEP, (sweep_id,)).fetchall()
    return [
        EvalRun.model_validate(
            {key: row[key] for key in row.keys() if key not in TABLE_ONLY_COLUMNS}
        )
        for row in rows
    ]


def list_sweeps(path: Path | None = None) -> list[tuple[str, str, int]]:
    """(sweep_id, started, n_rows) for every sweep, newest first."""
    with closing(connect(path)) as conn:
        rows = conn.execute(SELECT_SWEEPS).fetchall()
    return [(row["sweep_id"], row["started"], row["n_rows"]) for row in rows]
