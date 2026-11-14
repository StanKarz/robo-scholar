import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import tempfile

    import chromadb
    import marimo as mo
    from chromadb.errors import NotFoundError

    return NotFoundError, chromadb, math, mo, tempfile


@app.cell
def _(mo):
    mo.md("""
    # Chroma sandbox: the calls retrieval.py needs

    Toy data, throwaway folders: nothing here touches `data/chroma/`. Each
    section starts from a fresh, empty Chroma so you can rerun any of it.

    The "chunks" have **2-D vectors** instead of bge's 384 numbers, so you can
    draw them. Each one is an arrow at some angle, and the cosine similarity of
    two arrows is just the cosine of the angle between them.

    One thing that differs from the Chroma docs: their examples pass
    `embedding_function=...` so that Chroma embeds text for you. We don't. We
    embed ourselves (`create_embeddings`) and hand Chroma the vectors through
    `embeddings=`, so the same model is guaranteed for chunks and queries.
    Leave `embedding_function` out.
    """)
    return


@app.cell
def _(chromadb, math, tempfile):
    def fresh_client():
        """A Chroma client on a brand-new temp folder."""
        return chromadb.PersistentClient(
            path=tempfile.mkdtemp(prefix="chroma-sandbox-")
        )

    def toy_vec(degrees):
        """A length-1 arrow at `degrees`: the toy stand-in for an embedding."""
        rad = math.radians(degrees)
        return [math.cos(rad), math.sin(rad)]

    # (id, text, metadata, angle of its vector). Metadata mirrors what
    # index_chroma stores, including n_words.
    TOY = [
        (
            "toy:0000",
            "Action chunking predicts k actions at once.",
            {
                "paper_id": "act",
                "section": "IV-A",
                "page_start": 4,
                "page_end": 5,
                "n_words": 7,
            },
            10,
        ),
        (
            "toy:0001",
            "Temporal ensembling averages overlapping predictions.",
            {
                "paper_id": "act",
                "section": "IV-A",
                "page_start": 5,
                "page_end": 5,
                "n_words": 6,
            },
            35,
        ),
        (
            "toy:0002",
            "Diffusion policies denoise whole action sequences.",
            {
                "paper_id": "diffusion-policy",
                "section": "III",
                "page_start": 3,
                "page_end": 3,
                "n_words": 6,
            },
            80,
        ),
        (
            "toy:0003",
            "World models predict future observations.",
            {
                "paper_id": "world-models",
                "section": "2",
                "page_start": 2,
                "page_end": 2,
                "n_words": 5,
            },
            140,
        ),
    ]
    return TOY, fresh_client, toy_vec


@app.cell
def _(mo):
    mo.md("""
    ## 1. Create a collection, then add rows

    A **collection** is the container: a name, settings (`configuration`, which
    is Chroma's: how to search) and your own labels (`metadata`: what's inside).
    **Rows** go in with `add`, as four lists of the same length, where position
    *i* in each list belongs to the same chunk.
    """)
    return


@app.cell
def _(TOY):
    TOY
    return


@app.cell
def _(TOY):
    for t in TOY:
        print(t[3])
    return


@app.cell
def _(TOY):
    TOY[0][3]
    return


@app.cell
def _(TOY, fresh_client, mo, toy_vec):
    client1 = fresh_client()
    col1 = client1.create_collection(
        "chunks",
        configuration={"hnsw": {"space": "cosine"}},  # Chroma's setting
        metadata={"embedding_model": "toy-2d"},  # your label (option B)
    )
    col1.add(
        ids=[t[0] for t in TOY],
        documents=[t[1] for t in TOY],
        metadatas=[t[2] for t in TOY],
        embeddings=[toy_vec(t[3]) for t in TOY],
    )
    _rows = col1.get(include=["documents", "metadatas"])  # get: flat lists
    mo.vstack(
        [
            mo.md(
                f"`count()` = **{col1.count()}** · `configuration_json['hnsw']['space']` = "
                f"**{col1.configuration_json['hnsw']['space']}** · `metadata` = **{col1.metadata}**"
            ),
            mo.ui.table(
                [
                    {"id": i, "text": d, **m}
                    for i, d, m in zip(
                        _rows["ids"], _rows["documents"], _rows["metadatas"]
                    )
                ],
                selection=None,
            ),
        ]
    )
    return client1, col1


@app.cell
def _(mo):
    mo.md("""
    ## 2. Query: drag the slider

    The slider sets the query's angle. `query` returns the nearest rows, best
    first. Under cosine, `distance = 1 − cos(angle between them)`: the last
    column computes that by hand and matches Chroma's number (up to float32
    rounding). Note every value comes back as a list inside a list: `[0]` is
    "the first (only) query".
    """)
    return


@app.cell
def _(mo):
    angle = mo.ui.slider(
        0, 180, value=25, step=1, label="query angle (degrees)", show_value=True
    )
    angle
    return (angle,)


@app.cell
def _(TOY, angle, col1, math, mo, toy_vec):
    res = col1.query(
        query_embeddings=[toy_vec(angle.value)],
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )
    _angles = {t[0]: t[3] for t in TOY}
    _table = [
        {
            "rank": r + 1,
            "id": res["ids"][0][r],
            "section": res["metadatas"][0][r]["section"],
            "Chroma distance": round(res["distances"][0][r], 4),
            "1 − cos, by hand": round(
                1 - math.cos(math.radians(angle.value - _angles[res["ids"][0][r]])), 4
            ),
        }
        for r in range(len(res["ids"][0]))
    ]

    # Drawing: each chunk is an arrow from the centre; the query is the green one.
    _cx, _cy, _R = 210, 190, 150

    def _pt(deg, scale=1.0):
        rad = math.radians(deg)
        return _cx + _R * scale * math.cos(rad), _cy - _R * scale * math.sin(rad)

    _parts = [
        f'<path d="M{_cx - _R},{_cy} A{_R},{_R} 0 0 1 {_cx + _R},{_cy}" fill="none" stroke="#c9ced4"/>',
        f'<line x1="{_cx - _R - 10}" y1="{_cy}" x2="{_cx + _R + 10}" y2="{_cy}" stroke="#c9ced4"/>',
    ]
    _best = res["ids"][0][0]
    for _id, _text, _meta, _deg in TOY:
        _x, _y = _pt(_deg)
        _lx, _ly = _pt(_deg, 1.13)
        _colour = "#b5651d" if _id == _best else "#6e7781"
        _parts.append(
            f'<line x1="{_cx}" y1="{_cy}" x2="{_x:.1f}" y2="{_y:.1f}" stroke="{_colour}" stroke-width="2"/>'
        )
        _parts.append(
            f'<text x="{_lx:.1f}" y="{_ly:.1f}" font-size="12" fill="{_colour}" text-anchor="middle" '
            f'font-family="ui-monospace, Menlo, monospace">{_id}</text>'
        )
    _qx, _qy = _pt(angle.value)
    _parts.append(
        f'<line x1="{_cx}" y1="{_cy}" x2="{_qx:.1f}" y2="{_qy:.1f}" stroke="#2e7d4f" stroke-width="3.5"/>'
    )
    _parts.append(f'<circle cx="{_qx:.1f}" cy="{_qy:.1f}" r="5" fill="#2e7d4f"/>')
    _parts.append(
        f'<text x="{_cx}" y="{_cy + 24}" font-size="12" fill="#2e7d4f" text-anchor="middle">query (green) · nearest chunk in orange</text>'
    )
    _svg = f'<svg viewBox="0 0 420 225" width="420" xmlns="http://www.w3.org/2000/svg">{"".join(_parts)}</svg>'

    mo.hstack([mo.Html(_svg), mo.ui.table(_table, selection=None)], widths=[1, 1.4])
    return (res,)


@app.cell
def _(mo, res):
    mo.vstack(
        [
            mo.md("The raw dict `query` returned (what `rank_chunks` returns today):"),
            mo.tree(res),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Get an existing collection by name

    `rank_chunks` should never create anything: it fetches by name. The
    metadata you stored at creation comes back with it, which is how option B
    knows which embedder built the index.
    """)
    return


@app.cell
def _(client1, mo):
    _again = client1.get_collection("chunks")
    mo.md(
        f"`get_collection('chunks').metadata` = **{_again.metadata}** · `count()` = **{_again.count()}**"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Why we never update in place (bug 4a)

    Index four chunks (config A), then upsert a smaller config B that makes
    only two chunks. B overwrites the ids it shares with A and leaves the rest:
    two of A's rows survive, still searchable, describing a config that no
    longer exists.
    """)
    return


@app.cell
def _(TOY, fresh_client, mo, toy_vec):
    _col = fresh_client().create_collection(
        "chunks", configuration={"hnsw": {"space": "cosine"}}
    )
    _col.add(
        ids=[t[0] for t in TOY],
        documents=[f"[config A] {t[1]}" for t in TOY],
        metadatas=[t[2] for t in TOY],
        embeddings=[toy_vec(t[3]) for t in TOY],
    )
    _col.upsert(
        ids=["toy:0000", "toy:0001"],
        documents=[
            "[config B] first half, re-chunked",
            "[config B] second half, re-chunked",
        ],
        metadatas=[TOY[0][2], TOY[1][2]],
        embeddings=[toy_vec(20), toy_vec(60)],
    )
    _rows = _col.get(include=["documents"])
    mo.vstack(
        [
            mo.md(
                f"`count()` = **{_col.count()}**, not 2. Rows starting `[config A]` are leftovers."
            ),
            mo.ui.table(
                [
                    {"id": i, "text": d}
                    for i, d in zip(_rows["ids"], _rows["documents"])
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. `add` quietly skips ids that already exist

    So swapping `upsert` for `add` doesn't fix 4a on its own: adding into a
    collection that wasn't emptied first silently keeps the old rows.
    """)
    return


@app.cell
def _(fresh_client, mo, toy_vec):
    _col = fresh_client().create_collection(
        "chunks", configuration={"hnsw": {"space": "cosine"}}
    )
    _col.add(ids=["toy:0000"], documents=["old text"], embeddings=[toy_vec(10)])
    _col.add(
        ids=["toy:0000"], documents=["new text"], embeddings=[toy_vec(50)]
    )  # no error
    mo.md(
        f"After two adds: `count()` = **{_col.count()}**, text = **{_col.get(ids=['toy:0000'])['documents'][0]!r}**"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Delete the collection, and the first-run error

    You don't delete chunks one by one: you delete the **collection**, which
    takes every row and the settings with it. Then you create a new, empty one
    (section 1) with whatever settings this build needs. That's the whole of
    4a and 4d.

    Deleting a collection that doesn't exist raises `NotFoundError`. On a fresh
    clone there's no index yet, so `index_chroma`'s first step has to survive
    exactly that error.
    """)
    return


@app.cell
def _(NotFoundError, fresh_client, mo):
    _client = fresh_client()
    _client.create_collection("chunks")
    _client.delete_collection("chunks")
    _log = ["created 'chunks', then `delete_collection('chunks')`: ok"]
    for _label, _call in [
        (
            "`get_collection('chunks')` after deleting",
            lambda: _client.get_collection("chunks"),
        ),
        (
            "`delete_collection('chunks')` again",
            lambda: _client.delete_collection("chunks"),
        ),
    ]:
        try:
            _call()
            _log.append(f"{_label}: ok")
        except NotFoundError as _e:
            _log.append(f"{_label}: raises `NotFoundError`: {_e}")
    mo.md("\n\n".join(f"- {line}" for line in _log))
    return


if __name__ == "__main__":
    app.run()
