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

# <project root>/data/chroma (parents[1] is the project root). Built from this
# file's location, so a notebook started in notebooks/ opens the same index.
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
    """Turn text into vectors. The only place an embedding model is called.

    Indexing and querying both come through here, so chunks and queries are
    always embedded by the same model. A mismatch wouldn't raise an error;
    it would return plausible-looking nonsense.

    Args:
        texts: Chunk texts when indexing, the question when querying.
        query_type: "document" for chunks, "query" for a question. bge adds a
            search instruction to queries only; Voyage takes this as its
            `input_type`.
        model: The embedder, as stored in `EvalRun.embedding_model`.
        has_query_prefix: Whether a bge query gets its search instruction.
            False only for the sweep's six prefix-off rows.

    Returns:
        One vector per input, in order, typed `Sequence[float]` to match
        Chroma's add() and query().

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
        # Built per call: the client reads VOYAGE_API_KEY when it's constructed,
        # so one built at import, before load_dotenv(), would stay keyless.
        # Building one takes ~1 µs.
        vo = voyageai.Client()
        # .embeddings is typed "floats or ints"; ints only come back for
        # quantised output (output_dtype), which we never ask for. cast only
        # informs the type checker.
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

    Runs once per ingest config. Deleting and recreating the collection means
    no chunk from a previous config survives (upsert or add would leave them
    behind: sandbox sections 4 and 5). The collection uses cosine distance and
    records `model` in its metadata, so rank_chunks embeds queries with the
    same model.

    Embedding runs before the delete, so a failed build (a mistyped model, a
    Voyage rate limit) leaves the previous index as it was.

    Args:
        records: Chunk dicts straight from `ingest_file`.
        model: The embedder for this build, e.g. "bge-small".
    """
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
    """Retrieve the k chunks closest to a question, nearest first.

    The question is embedded as a "query" with the model named in the
    collection's metadata, the one index_chroma built it with. Chroma does
    the nearest-neighbour search and returns the results in order.

    Args:
        query_str: The question, in natural language.
        k: How many chunks to return.
        has_query_prefix: Passed to create_embeddings.

    Returns:
        Up to k `RetrievedChunk`s. `distance` is cosine distance, 1 - cos θ:
        smaller is closer.

    Raises:
        NotFoundError: If there is no index yet (run `rs ingest --index`).
        ValueError: If Chroma leaves out a field named in include=.
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
    # The three columns are typed Optional; the check narrows them for ty.
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
