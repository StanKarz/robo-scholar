"""Typer CLI — thin adapter over the core library. Installed as `rs`."""

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="RoboScholar — agentic research assistant for embodied AI papers.")
eval_app = typer.Typer(no_args_is_help=True, help="Run and report evals.")
app.add_typer(eval_app, name="eval")

console = Console()


@app.command()
def fetch(
    manifest: str = typer.Option("data/raw/papers.json", help="Manifest listing the papers."),
    dest: str = typer.Option("data/raw/papers", help="Where to write the PDFs."),
    force: bool = typer.Option(False, "--force", help="Re-download even if a valid copy exists."),
):
    """Download the paper corpus listed in the manifest and verify checksums."""
    from pathlib import Path

    from rich.table import Table

    from roboscholar.fetch import fetch_all

    results = fetch_all(Path(manifest), Path(dest), force)

    table = Table(title=f"Fetched into {dest}")
    for col in ("paper", "arxiv", "status", "detail"):
        table.add_column(col)
    colours = {"ok": "green", "cached": "cyan", "checksum-mismatch": "yellow", "error": "red"}
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
    path: str = typer.Argument("data/raw", help="PDF file, markdown file, or directory to ingest."),
    n_words: int = typer.Option(500, help="Target chunk size in words."),
    overlap_words: int = typer.Option(50, help="Chunk overlap in words."),
):
    """Parse documents into chunks (storage in Chroma lands with retrieval.py)."""
    from pathlib import Path

    from rich.table import Table

    from roboscholar.ingest import ingest_file

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
    for f in files:
        records, unmatched = ingest_file(f, n_words, overlap_words)
        all_records.extend(records)
        warn = f"[red]{len(unmatched)} unmatched headings[/red]" if unmatched else "-"
        table.add_row(
            f.stem,
            str(len(records)),
            str(len({r["section"] for r in records})),
            str(round(sum(r["n_words"] for r in records) / max(len(records), 1))),
            warn,
        )
    console.print(table)
    console.print(
        f"{len(all_records)} chunks total. "
        "[yellow]Not stored yet — embedding + Chroma writes land in retrieval.py.[/yellow]"
    )


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
    console.print("[yellow]Not implemented yet — see roboscholar/tools.py (compare_methods)[/yellow]")
    raise typer.Exit(1)


@app.command()
def quiz(
    topic: str = typer.Argument(..., help="Paper or topic to be quizzed on."),
    n: int = typer.Option(5, help="Number of questions."),
    difficulty: str = typer.Option("medium", help="easy | medium | hard"),
):
    """Grounded quiz generated from retrieved chunks; wrong answers logged to SQLite."""
    console.print("[yellow]Not implemented yet — see roboscholar/tools.py (quiz_me)[/yellow]")
    raise typer.Exit(1)


@eval_app.command("run")
def eval_run():
    """Run retrieval metrics + LLM-as-judge against the golden set; store results."""
    console.print("[yellow]Not implemented yet — see roboscholar/evals/runner.py[/yellow]")
    raise typer.Exit(1)


@eval_app.command("report")
def eval_report():
    """Render a comparison table of past eval runs."""
    console.print("[yellow]Not implemented yet — see roboscholar/evals/runner.py[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
