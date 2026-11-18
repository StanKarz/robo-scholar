"""Typer CLI — thin adapter over the core library. Installed as `rs`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from dotenv import load_dotenv
from rich.console import Console

if TYPE_CHECKING:
    from roboscholar.models import EvalRun

# API keys (VOYAGE_API_KEY, ANTHROPIC_API_KEY) live in .env, which is gitignored.
load_dotenv()

app = typer.Typer(
    no_args_is_help=True,
    help="RoboScholar — agentic research assistant for embodied AI papers.",
)
eval_app = typer.Typer(no_args_is_help=True, help="Run and report evals.")
app.add_typer(eval_app, name="eval")

console = Console()


@app.command()
def fetch(
    manifest: str = typer.Option(
        "data/raw/papers.json", help="Manifest listing the papers."
    ),
    dest: str = typer.Option("data/raw/papers", help="Where to write the PDFs."),
    force: bool = typer.Option(
        False, "--force", help="Re-download even if a valid copy exists."
    ),
):
    """Download the paper corpus listed in the manifest and verify checksums."""

    from rich.table import Table

    from roboscholar.fetch import fetch_all

    results = fetch_all(Path(manifest), Path(dest), force)

    table = Table(title=f"Fetched into {dest}")
    for col in ("paper", "arxiv", "status", "detail"):
        table.add_column(col)
    colours = {
        "ok": "green",
        "cached": "cyan",
        "checksum-mismatch": "yellow",
        "error": "red",
    }
    for r in results:
        table.add_row(
            r.paper.id,
            r.paper.arxiv_id,
            f"[{colours.get(r.status, 'white')}]{r.status}[/]",
            r.detail,
        )
    console.print(table)

    failed = [r for r in results if r.status not in ("ok", "cached")]
    if failed:
        console.print(f"[red]{len(failed)} of {len(results)} failed.[/red]")
        raise typer.Exit(1)


@app.command()
def ingest(
    path: str | None = typer.Argument(
        None,
        help="PDF file, markdown file, or directory. Default: every document in the manifests.",
    ),
    n_words: int = typer.Option(500, help="Target chunk size in words."),
    overlap_words: int = typer.Option(50, help="Chunk overlap in words."),
    index: bool = typer.Option(
        False,
        "--index",
        help="Embed the chunks and rebuild the Chroma index from empty.",
    ),
    model: str = typer.Option(
        "bge-small", help="Embedder for --index: bge-small or voyage-4-lite."
    ),
):
    """Parse documents into chunks; with --index, embed them and rebuild data/chroma/."""

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    from roboscholar.fetch import corpus_files
    from roboscholar.ingest import ingest_file

    if path is None:
        files = corpus_files()
        missing = [f for f in files if not f.exists()]
        if missing:
            names = ", ".join(f.name for f in missing)
            console.print(
                f"[red]Listed in the manifests but not on disk: {names}[/red]"
            )
            console.print("Run `uv run rs fetch` first.")
            raise typer.Exit(1)
    else:
        target = Path(path)
        files = (
            sorted(p for suffix in ("*.pdf", "*.md") for p in target.rglob(suffix))
            if target.is_dir()
            else [target]
        )
        if not files:
            console.print(f"[red]No .pdf or .md files found under {target}[/red]")
            raise typer.Exit(1)

    table = Table(title=f"Chunks ({n_words}w / {overlap_words}w overlap)")
    for col in ("document", "chunks", "sections", "avg words", "warnings"):
        table.add_column(col)
    all_records = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task = progress.add_task("Parsing", total=len(files))
        for f in files:
            progress.update(task, description=f"Parsing {f.name}")
            records, unmatched = ingest_file(f, n_words, overlap_words)
            all_records.extend(records)
            warn = (
                f"[red]{len(unmatched)} unmatched headings[/red]" if unmatched else "-"
            )
            table.add_row(
                f.stem,
                str(len(records)),
                str(len({r["section"] for r in records})),
                str(round(sum(r["n_words"] for r in records) / max(len(records), 1))),
                warn,
            )
            progress.advance(task)
    console.print(table)
    if not index:
        console.print(
            f"{len(all_records)} chunks total. "
            "[yellow]Not stored: add --index to embed them and rebuild data/chroma/.[/yellow]"
        )
        return

    from voyageai.error import VoyageError

    from roboscholar.retrieval import client, index_chroma

    try:
        with console.status(f"Embedding {len(all_records)} chunks with {model}…"):
            index_chroma(all_records, model)
    except (ValueError, VoyageError) as e:
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        console.print("The previous index is unchanged.")
        raise typer.Exit(1) from None

    # The rebuild checks: every chunk stored exactly once, cosine distance, and
    # the embedder recorded on the collection for rank_chunks to read back.
    collection = client.get_collection("chunks")
    count = collection.count()
    space = collection.configuration_json["hnsw"]["space"]
    is_ok = count == len(all_records) and space == "cosine"
    colour = "green" if is_ok else "red"
    console.print(
        f"[{colour}]Indexed: count() = {count} of {len(all_records)} chunks · "
        f"space = {space} · metadata = {collection.metadata}[/{colour}]"
    )
    if not is_ok:
        raise typer.Exit(1)


@app.command()
def search(
    question: str = typer.Argument(..., help="Question to search the index with."),
    k: int = typer.Option(5, "--k", "-k", help="Number of chunks to return."),
):
    """Show the top-k chunks for a question, nearest first. For checking retrieval by eye."""
    from chromadb.errors import NotFoundError
    from rich.table import Table

    from roboscholar.retrieval import rank_chunks

    try:
        chunks = rank_chunks(question, k)
    except NotFoundError:
        console.print("[red]No index yet.[/red] Run `uv run rs ingest --index` first.")
        raise typer.Exit(1) from None

    table = Table(title=f"Top {k}: {question}")
    table.add_column("rank", justify="right")
    table.add_column("distance", justify="right")
    table.add_column("id")
    table.add_column("section", max_width=36, overflow="fold")
    table.add_column("pages")
    table.add_column("text", max_width=60, overflow="fold")
    for rank, c in enumerate(chunks, start=1):
        if c.page_start == 0:  # markdown blogs have no pages
            pages = "-"
        elif c.page_start == c.page_end:
            pages = str(c.page_start)
        else:
            pages = f"{c.page_start}–{c.page_end}"
        table.add_row(
            str(rank), f"{c.distance:.3f}", c.id, c.section, pages, c.text[:120] + "…"
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer over the corpus."),
    k: int = typer.Option(5, help="Number of chunks to retrieve."),
):
    """Grounded Q&A with citations (agentic once agent.py exists)."""
    console.print("[yellow]Not implemented yet — see roboscholar/agent.py[/yellow]")
    raise typer.Exit(1)


@app.command()
def compare(
    topic: str = typer.Argument(..., help="Topic to compare, e.g. 'action chunking'."),
    paper_a: str = typer.Option(..., help="First paper_id."),
    paper_b: str = typer.Option(..., help="Second paper_id."),
):
    """Multi-step agentic comparison across two papers."""
    console.print(
        "[yellow]Not implemented yet — see roboscholar/tools.py (compare_methods)[/yellow]"
    )
    raise typer.Exit(1)


@app.command()
def quiz(
    topic: str = typer.Argument(..., help="Paper or topic to be quizzed on."),
    n: int = typer.Option(5, help="Number of questions."),
    difficulty: str = typer.Option("medium", help="easy | medium | hard"),
):
    """Grounded quiz generated from retrieved chunks; wrong answers logged to SQLite."""
    console.print(
        "[yellow]Not implemented yet — see roboscholar/tools.py (quiz_me)[/yellow]"
    )
    raise typer.Exit(1)


GOLDEN_PATH = Path(__file__).resolve().parent / "evals" / "golden.json"


@eval_app.command("run")
def eval_run(
    k: int = typer.Option(5, "--k", "-k", help="Chunks retrieved per question."),
):
    """Run the retrieval sweep over the golden set and save every row to SQLite."""
    from roboscholar import db
    from roboscholar.evals import runner
    from roboscholar.models import GoldenSet

    run_sweep = getattr(runner, "run_sweep", None)
    if run_sweep is None:
        console.print(
            "[yellow]runner.run_sweep isn't written yet (roboscholar/evals/runner.py).[/yellow]"
        )
        raise typer.Exit(1)

    questions = GoldenSet.load(GOLDEN_PATH).questions
    with console.status(f"Running the sweep over {len(questions)} questions…"):
        runs, failures = run_sweep(questions, k)

    if runs:
        sweep_id = db.new_sweep_id()
        db.save_runs(runs, sweep_id)
        _print_sweep(sweep_id, runs)
    else:
        console.print("[red]No rows to save: every build failed.[/red]")

    if failures:
        console.print(
            f"[red]{len(failures)} build(s) failed and were not scored:[/red]"
        )
        for failure in failures:
            console.print(f"  [red]•[/red] {failure}")

    # The sweep rebuilt data/chroma once per config, so it now holds the last
    # build, not necessarily the one the table picks.
    from chromadb.errors import NotFoundError

    from roboscholar.retrieval import client

    try:
        collection = client.get_collection("chunks")
        live = f"{collection.metadata['embedding_model']}, {collection.count()} chunks"
    except NotFoundError:
        live = "no index"
    console.print(
        f"\ndata/chroma holds the last successful build ({live}). "
        "Rebuild the config you choose with\n"
        "  uv run rs ingest --n-words N --overlap-words M --index --model MODEL"
    )
    if failures or not runs:
        raise typer.Exit(1)


@eval_app.command("report")
def eval_report(
    sweep: str | None = typer.Option(
        None, help="Sweep id to show. Default: the latest."
    ),
    list_all: bool = typer.Option(False, "--list", help="List every saved sweep."),
):
    """Show one sweep's rows, best hit-rate first."""
    from rich.table import Table

    from roboscholar import db

    sweeps = db.list_sweeps()
    if not sweeps:
        console.print("[yellow]No sweeps saved yet.[/yellow] Run `uv run rs eval run`.")
        raise typer.Exit(1)

    if list_all:
        table = Table(title="Saved sweeps, newest first")
        for col in ("sweep", "started (UTC)", "rows"):
            table.add_column(col)
        for sweep_id, started, n_rows in sweeps:
            table.add_row(sweep_id, started[:19].replace("T", " "), str(n_rows))
        console.print(table)
        return

    sweep_id = sweep or sweeps[0][0]
    runs = db.load_sweep(sweep_id)
    if not runs:
        console.print(f"[red]No sweep with id {sweep_id!r}.[/red] Try --list.")
        raise typer.Exit(1)
    _print_sweep(sweep_id, runs)
    if sweep is None and len(sweeps) > 1:
        console.print(
            f"{len(sweeps) - 1} earlier sweep(s): `uv run rs eval report --list`."
        )


def _print_sweep(sweep_id: str, runs: list[EvalRun]) -> None:
    """One sweep as a table, best hit-rate first, MRR breaking ties.

    "gap" is how many questions behind the top row each row is, the unit the
    decision rule's tie threshold is written in (NOTES.md, "Retrieval metrics").
    """
    from rich.table import Table

    n_questions = {run.n_questions for run in runs}
    ks = {run.k for run in runs}
    ranked = sorted(runs, key=lambda run: (run.hit_rate_at_k, run.mrr), reverse=True)
    top_hits = round(ranked[0].hit_rate_at_k * ranked[0].n_questions)

    table = Table(
        title=f"Sweep {sweep_id} · {len(runs)} rows · "
        f"{'/'.join(map(str, sorted(n_questions)))} questions · k={'/'.join(map(str, sorted(ks)))}",
        caption="hit@k, MRR: section rule · OR: section or page · gap: questions behind #1",
    )
    for col in ("#", "embedder", "prefix", "chunks", "hit@k", "OR hit@k", "MRR", "gap"):
        is_number = col not in ("embedder", "prefix", "chunks")
        table.add_column(col, justify="right" if is_number else "left")
    for i, run in enumerate(ranked, start=1):
        hits = round(run.hit_rate_at_k * run.n_questions)
        or_hits = round(run.hit_rate_or_at_k * run.n_questions)
        table.add_row(
            str(i),
            run.embedding_model,
            "on" if run.has_query_prefix else "off",
            f"{run.n_words}w / {run.overlap_words}",
            f"{run.hit_rate_at_k:.3f} ({hits}/{run.n_questions})",
            f"{run.hit_rate_or_at_k:.3f} ({or_hits})",
            f"{run.mrr:.3f}",
            "-" if hits == top_hits else str(hits - top_hits),
        )
    console.print(table)
    if len(n_questions) > 1 or len(ks) > 1:
        console.print(
            "[yellow]Rows were scored on different question counts or k; "
            "they aren't comparable.[/yellow]"
        )


if __name__ == "__main__":
    app()
