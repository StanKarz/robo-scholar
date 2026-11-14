"""Embed chunks, store them in Chroma (data/chroma/), and search them.

Dense search first; BM25 + RRF in week 2.

Background: NOTES.md, "Chroma — the retrieval mechanics". Every Chroma call
used here is tried out on toy 2-D vectors in notebooks/chroma_sandbox.py.
Check by eye with `uv run rs search "<question>"`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import chromadb
import voyageai
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from roboscholar.models import RetrievedChunk

# Only the type checker reads this block; at runtime TYPE_CHECKING is False.
# ChunkRecord is needed for a type hint only, and importing ingest for real
# would load PyMuPDF and trafilatura on every `rs search`.
if TYPE_CHECKING:
    from roboscholar.ingest import ChunkRecord

HF_MODEL_NAMES = {"bge-small": "BAAI/bge-small-en-v1.5"}

# Loaded embedders, keyed by model name. Constructing a SentenceTransformer
# takes seconds, so each one is built at most once per process.
_EMBEDDERS: dict[str, SentenceTransformer] = {}

# The index lives at <project root>/data/chroma. Built from this file's location
# rather than the current directory, so a notebook started inside notebooks/
# opens the same index `rs` builds instead of creating an empty one there.
# parents[0] is roboscholar/, parents[1] the project root.
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"

# One client for the whole module. Collection handles are fetched by name inside
# each function instead: index_chroma deletes and recreates the collection, so a
# handle held here would point at a deleted one.
client = chromadb.PersistentClient(path=CHROMA_DIR)


def load_embedder(model: str) -> SentenceTransformer:
    """Return the embedder for `model`, loading it on first use.

    Args:
        model: A key of HF_MODEL_NAMES, e.g. "bge-small".

    Raises:
        ValueError: If the model name isn't one this project knows about.
    """
    if model not in _EMBEDDERS:
        if model not in HF_MODEL_NAMES:
            raise ValueError(f"Unknown embedding model: {model!r}")
        _EMBEDDERS[model] = SentenceTransformer(HF_MODEL_NAMES[model])
    return _EMBEDDERS[model]


def create_embeddings(
    texts: list[str],
    query_type: Literal["query", "document"],
    model: str,
    *,
    has_query_prefix: bool = True,
) -> list[Sequence[float]]:
    """Turn text into vectors. The only place an embedding model is invoked.

    Both the indexing path and the query path go through here, which is what
    guarantees stored chunks and incoming queries land in the same vector
    space. Embedding a query with a different model than the chunks produces
    plausible-looking nonsense rather than an error.

    Args:
        texts: Raw strings to embed: chunk text when indexing, the question
            when querying.
        query_type: "document" for chunks being indexed, "query" for a
            question being searched. Both models embed the two differently:
            bge wants an instruction prefix on queries only; Voyage takes this
            value as its `input_type`.
        model: Which embedder to use. This is the swap point the config sweep
            turns: the value stored in `EvalRun.embedding_model`.
        has_query_prefix: Whether a query gets bge's search instruction.
            False only for the sweep's six prefix-off rows.

    Returns:
        One vector per input string, in the same order, as plain Python lists
        whichever model made them. Declared as `Sequence[float]` because that
        is what Chroma's add() and query() declare, and a list type only
        matches its exact element type.

    Raises:
        ValueError: If `model` isn't one this project knows about.
        voyageai.error.VoyageError: Voyage's own errors pass through, e.g.
            AuthenticationError (no key) or RateLimitError.
    """
    if model == "bge-small":
        if query_type == "query" and has_query_prefix:
            embeddings = (
                load_embedder(model)
                .encode(
                    texts,
                    prompt="Represent this sentence for searching relevant passages: ",
                )
                .tolist()
            )
        else:
            embeddings = load_embedder(model).encode(texts, prompt=None).tolist()

    elif model == "voyage-4-lite":
        # Built per call, not at import: the client reads VOYAGE_API_KEY once,
        # when constructed, so one built before load_dotenv() stays keyless for
        # good. It holds only the key and URL (~1 µs to build); the model runs
        # on Voyage's servers, so there's nothing worth caching here.
        vo = voyageai.Client()
        # Voyage types .embeddings as "floats or ints". Ints only come back when
        # you ask for quantised output (embed's output_dtype), which we never do,
        # so cast tells the type checker what we know. It does nothing at runtime.
        embeddings = cast(
            list[Sequence[float]],
            vo.embed(texts, model=model, input_type=query_type).embeddings,
        )

    else:
        raise ValueError(
            f"Unknown embedding model: {model!r}. Use 'bge-small' or 'voyage-4-lite'."
        )

    return embeddings


def index_chroma(records: list[ChunkRecord], model: str) -> None:
    """Rebuild the Chroma collection from empty and fill it with `records`.

    Runs once per ingest config, not per query: the expensive, do-it-upfront
    half of retrieval. Each build deletes the old collection and creates a new
    one, so no chunk from a previous config survives (upsert or add into the
    old collection would leave them behind: sandbox sections 4 and 5). The new
    collection uses cosine distance and records `model` in its metadata, so
    rank_chunks embeds queries with the model that built the index.

    The chunks are embedded before anything is deleted, so a build that fails
    (a mistyped model, a Voyage rate limit) leaves the previous index as it was.

    Each chunk becomes one row: id, text, vector and metadata. `paper_id`,
    `section` and the page range are what `metrics.py` compares against the
    golden set's labels and what an answer cites; `n_words` is there so a
    result can be rebuilt as a `RetrievedChunk`.

    Args:
        records: Chunk dicts straight from `ingest_file`.
        model: The embedder for this build, e.g. "bge-small".
    """
    # The step that can fail runs first, before the step that destroys anything.
    embeddings = create_embeddings([r["text"] for r in records], "document", model)

    # Delete the previous build if there is one. On a fresh clone there isn't,
    # and delete_collection raises NotFoundError, which is fine.
    try:
        client.delete_collection("chunks")
    except NotFoundError:
        pass

    collection = client.create_collection(
        "chunks",
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model},
    )
    # Four parallel lists: position i in each one is chunk i.
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "paper_id": r["paper_id"],
                "section": r["section"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "n_words": r["n_words"],
            }
            for r in records
        ],
        embeddings=embeddings,
    )


def rank_chunks(
    query_str: str, k: int = 5, *, has_query_prefix: bool = True
) -> list[RetrievedChunk]:
    """Retrieve the k chunks closest to a question, best match first.

    The cheap, per-question half of retrieval. Chroma does the ranking
    (nearest-neighbour search over the stored vectors), so nothing here
    computes a similarity by hand. Results come back already ordered, which
    is what makes MRR meaningful downstream.

    The question is embedded as a "query" with the model named in the
    collection's metadata, the one index_chroma built it with.

    Args:
        query_str: The question, in natural language.
        k: How many chunks to return. Defaults to the 5 the agent uses.
        has_query_prefix: Passed to create_embeddings. False only for the
            sweep's prefix-off rows.

    Returns:
        Up to k `RetrievedChunk`s, nearest first. `distance` is cosine
        distance, 1 - cos θ: 0 means same direction, smaller is closer.

    Raises:
        NotFoundError: If there is no index yet (run `rs ingest --index`).
        ValueError: If Chroma's result is missing a field asked for in
            include=. Shouldn't happen; it's there so the type checker knows
            the three lists aren't None.
    """
    collection = client.get_collection("chunks")
    model = collection.metadata["embedding_model"]
    query_embeddings = create_embeddings(
        [query_str], "query", model, has_query_prefix=has_query_prefix
    )
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma answers in columns: one list of ids, one of texts, one of metadata
    # dicts, one of distances. Each column is wrapped in an outer list with one
    # entry per query vector; we sent one, so [0] below is our only answer.
    # The three columns are typed Optional (only filled when named in include=),
    # and the check narrows them to non-None for the type checker.
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]
    if documents is None or metadatas is None or distances is None:
        raise ValueError("Chroma left out a field named in include=")

    # zip reads the four columns side by side, one chunk per turn: position i of
    # each. ** then copies that chunk's metadata dict (paper_id, section, pages,
    # n_words) into row, giving the eight fields RetrievedChunk needs.
    chunks = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0], documents[0], metadatas[0], distances[0]
    ):
        row = {"id": chunk_id, "text": text, "distance": distance, **metadata}
        chunks.append(RetrievedChunk.model_validate(row))
    return chunks
