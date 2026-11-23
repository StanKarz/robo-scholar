import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter

    import marimo as mo

    from roboscholar.cli import GOLDEN_PATH
    from roboscholar.evals.metrics import first_correct_rank, is_correct, mrr
    from roboscholar.fetch import corpus_files
    from roboscholar.ingest import ingest_file
    from roboscholar.models import GoldenSet
    from roboscholar.retrieval import index_chroma, rank_chunks

    return (
        Counter,
        GOLDEN_PATH,
        GoldenSet,
        corpus_files,
        first_correct_rank,
        index_chroma,
        ingest_file,
        is_correct,
        mo,
        mrr,
        rank_chunks,
    )


@app.cell
def _(mo):
    mo.md("""
    # Choosing the chunk config

    Three voyage-4-lite configs tied in sweep `20261122T234428Z`, the first one
    after the label audit (NOTES.md, 2026-11-22). The rule from NOTES.md: hit@5
    under section match decides, and gaps under 2 questions are ties. Ties go
    to MRR. Prefer a config that wins across slices (per paper, papers vs
    blogs), and read the failures.

    350w / 70 leads by one question. In the last sweep a one-question lead was
    luck (pi05-001 at rank 4), so check which question it is this time.

    The button rebuilds all three indexes. Afterwards `data/chroma` holds the
    last one (200w / 40), so rebuild the winner once you've picked it.
    """)
    return


@app.cell
def _(GOLDEN_PATH, GoldenSet):
    all_qs = GoldenSet.load(GOLDEN_PATH).questions
    len(all_qs)
    return (all_qs,)


@app.cell
def _():
    MODEL = "voyage-4-lite"
    # (n_words, overlap_words) for the three tied rows, in the sweep table's
    # order. 200w / 0 and 350w / 0 hit 31, two behind the leader, which the
    # rule counts as a real gap.
    TIED_CONFIGS = [(350, 70), (350, 35), (200, 40)]
    LABELS = {c: f"{c[0]}w / {c[1]}" for c in TIED_CONFIGS}
    # What the sweep scored each config, read from data/roboscholar.db:
    # (questions hit out of 34, MRR). The summary below recomputes both, so a
    # mismatch means this notebook scored something different from the sweep.
    SWEEP = {
        (350, 70): (33, 0.693),
        (350, 35): (32, 0.689),
        (200, 40): (32, 0.684),
    }
    BLOGS = {"introducing-gaia1", "gaia-2", "gaia-3"}  # every other source is a paper
    return BLOGS, LABELS, MODEL, SWEEP, TIED_CONFIGS


@app.cell
def _(mo):
    # Gates the build so it only runs on a click, not whenever an upstream cell
    # changes. Each click spends about 290k Voyage tokens.
    build_button = mo.ui.run_button(label="Build and query the 3 tied configs")
    build_button
    return (build_button,)


@app.cell
def _(
    LABELS,
    MODEL,
    TIED_CONFIGS,
    all_qs,
    build_button,
    corpus_files,
    index_chroma,
    ingest_file,
    mo,
    rank_chunks,
):
    mo.stop(not build_button.value, mo.md("Click the button above to build."))

    # retrieved[config][question id] is that question's top 5 (a list of
    # RetrievedChunk) against that config's index. The chunks are kept, not
    # only their ranks, because each build replaces the last one in
    # data/chroma, and the misses section reads every config's top 5.
    retrieved = {}
    _chunk_counts = {}
    for _config in TIED_CONFIGS:
        _n_words, _overlap_words = _config

        # Chunk all 8 documents with this config into one flat list.
        _records = []
        for _path in corpus_files():
            _chunk_records, _ = ingest_file(
                _path, n_words=_n_words, overlap_words=_overlap_words
            )
            _records.extend(_chunk_records)

        # Index once, after every file is chunked. This replaces whatever
        # data/chroma held before.
        index_chroma(_records, MODEL)
        _chunk_counts[_config] = len(_records)

        # Query that index for every question while it's the one loaded.
        retrieved[_config] = {}
        for _q in all_qs:
            retrieved[_config][_q.id] = rank_chunks(_q.question, k=5)

    # Should read 266 / 260 / 404 chunks, the same as the sweep's builds (one
    # fewer or more than before the 2026-11-22 ingest fix).
    mo.md(
        "\n".join(
            f"- **{LABELS[_c]}**: {_n} chunks indexed, {len(retrieved[_c])} questions queried"
            for _c, _n in _chunk_counts.items()
        )
    )
    return (retrieved,)


@app.cell
def _(LABELS, TIED_CONFIGS, all_qs, first_correct_rank, mo, retrieved):
    # ranks[config][question id] is the rank (1-5) of the first chunk from the
    # right paper and a labelled section, or None for a miss: the same rule the
    # sweep scored with.
    ranks = {
        c: {q.id: first_correct_rank(retrieved[c][q.id], q, rule="section") for q in all_qs}
        for c in TIED_CONFIGS
    }

    # One row per question, one rank column per config. Sort by any column.
    # "found by" is how many of the configs got it into the top 5.
    _rows = []
    for _q in all_qs:
        _row = {"question": _q.id, "source": _q.source_paper}
        for _c in TIED_CONFIGS:
            _row[LABELS[_c]] = ranks[_c][_q.id]
        _row["found by"] = sum(ranks[_c][_q.id] is not None for _c in TIED_CONFIGS)
        _row["text"] = _q.question
        _rows.append(_row)

    mo.vstack(
        [
            mo.md("## Rank of the first correct chunk, per question"),
            mo.ui.table(_rows, selection=None, pagination=False),
        ]
    )
    return (ranks,)


@app.cell
def _(BLOGS, LABELS, SWEEP, TIED_CONFIGS, all_qs, mo, mrr, ranks, retrieved):
    _papers = [q for q in all_qs if q.source_paper not in BLOGS]
    _blogs = [q for q in all_qs if q.source_paper in BLOGS]

    def _hits(config, questions):
        """How many of these questions the config found in its top 5."""
        return sum(ranks[config][q.id] is not None for q in questions)

    _summary = []
    for _c in TIED_CONFIGS:
        _sweep_hits, _sweep_mrr = SWEEP[_c]
        # Words Claude would read per search: the 5 chunks' lengths, summed,
        # averaged over the 34 questions. Bigger chunks mean more context per
        # search in the agent loop, and more tokens.
        _top5_words = [sum(ch.n_words for ch in retrieved[_c][q.id]) for q in all_qs]
        _summary.append(
            {
                "config": LABELS[_c],
                f"hits /{len(all_qs)}": _hits(_c, all_qs),
                f"papers /{len(_papers)}": _hits(_c, _papers),
                f"blogs /{len(_blogs)}": _hits(_c, _blogs),
                "MRR": round(mrr(list(ranks[_c].values())), 3),
                "sweep hits": _sweep_hits,
                "sweep MRR": _sweep_mrr,
                "words per top 5": round(sum(_top5_words) / len(_top5_words)),
            }
        )

    mo.vstack(
        [
            mo.md(
                "## Summary per config\n\n"
                "`hits` and `MRR` should equal the `sweep` columns."
            ),
            mo.ui.table(_summary, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(BLOGS, Counter, LABELS, TIED_CONFIGS, all_qs, mo, ranks):
    # Hits split by source document, for the rule's "wins across slices".
    # Each cell is questions hit out of that source's count, most-asked first.
    _by_source = []
    for _source, _n in Counter(q.source_paper for q in all_qs).most_common():
        _qs = [q for q in all_qs if q.source_paper == _source]
        _row = {
            "source": _source,
            "kind": "blog" if _source in BLOGS else "paper",
            "questions": _n,
        }
        for _c in TIED_CONFIGS:
            _row[LABELS[_c]] = sum(ranks[_c][q.id] is not None for q in _qs)
        _by_source.append(_row)

    mo.vstack(
        [
            mo.md("## Hits per source"),
            mo.ui.table(_by_source, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(LABELS, TIED_CONFIGS, all_qs, is_correct, mo, ranks, retrieved):
    def _top5_table(q, chunks):
        """One config's top 5 for a question, marking which chunks count.

        "section" is the scoring rule. "OR" also accepts a chunk whose pages
        cover a labelled page, so OR yes / section no usually means the right
        passage came back under a section the label doesn't list.
        """
        rows = []
        for rank, chunk in enumerate(chunks, start=1):
            pages = (
                str(chunk.page_start)
                if chunk.page_start == chunk.page_end
                else f"{chunk.page_start}–{chunk.page_end}"
            )
            rows.append(
                {
                    "rank": rank,
                    "chunk": chunk.id,
                    "section": chunk.section,
                    "pages": pages,
                    "distance": round(chunk.distance, 3),
                    "section match": is_correct(chunk, q, rule="section"),
                    "OR match": is_correct(chunk, q, rule="or"),
                    "text": chunk.text[:200],
                }
            )
        return mo.ui.table(rows, selection=None, pagination=False)

    # Every question that at least one config misses. Each one opens to the
    # question, its label and reference answer, then one tab per config with
    # that config's top 5.
    _items = {}
    for _q in all_qs:
        _missed_by = [LABELS[_c] for _c in TIED_CONFIGS if ranks[_c][_q.id] is None]
        if not _missed_by:
            continue
        _ranks_line = " · ".join(
            f"{LABELS[_c]}: {ranks[_c][_q.id] or 'miss'}" for _c in TIED_CONFIGS
        )
        _header = mo.md(
            f"**{_q.question}**\n\n"
            f"Label: `{_q.source_paper}`, sections {', '.join(_q.source_sections)},"
            f" pages {', '.join(str(p) for p in _q.source_pages) or 'none (blog)'}\n\n"
            f"Reference answer: {_q.reference_answer}\n\n"
            f"Ranks: {_ranks_line}"
        )
        _tabs = mo.ui.tabs(
            {
                f"{LABELS[_c]} ({ranks[_c][_q.id] or 'miss'})": _top5_table(
                    _q, retrieved[_c][_q.id]
                )
                for _c in TIED_CONFIGS
            }
        )
        _items[f"{_q.id}: missed by {', '.join(_missed_by)}"] = mo.vstack(
            [_header, _tabs]
        )

    mo.vstack(
        [
            mo.md(
                f"## Misses\n\n{len(_items)} questions missed by at least one config."
                " For each: is the label wrong, is the question vague, or did"
                " retrieval fail?"
            ),
            mo.accordion(_items, multiple=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
