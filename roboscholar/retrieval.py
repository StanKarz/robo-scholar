"""Embed, store in Chroma (data/chroma/), search. Dense first; BM25 + RRF week 2.

Docs to read first:
- https://docs.trychroma.com/docs/overview/getting-started
- README "Setting up Chroma" section for the persistence/embedding decisions.
"""

import chromadb
from sentence_transformers import SentenceTransformer

HF_MODEL_NAMES = {"bge-small": "BAAI/bge-small-en-v1.5"}

# Loaded embedders, keyed by model name. Constructing a SentenceTransformer
# takes seconds, so each one is built at most once per process.
_EMBEDDERS: dict[str, SentenceTransformer] = {}

client = chromadb.PersistentClient(path="data/chroma/")
collection = client.get_or_create_collection(name="chunks")


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


def create_embeddings(texts: list[str], model: str = "bge-small") -> list[list[float]]:
    """Turn text into vectors. The only place an embedding model is invoked.

    Both the indexing path and the query path go through here, which is what
    guarantees stored chunks and incoming queries land in the same vector
    space. Embedding a query with a different model than the chunks produces
    plausible-looking nonsense rather than an error.

    Args:
        texts: Raw strings to embed — chunk text when indexing, the question
            when querying.
        model: Which embedder to use. This is the swap point the config sweep
            turns: the value stored in `EvalRun.embedding_model`.

    Returns:
        One vector per input string, in the same order, as plain lists
        (Chroma will not accept numpy arrays).
    """
    return load_embedder(model).encode(texts).tolist()


def index_chroma(records):
    """Embed every chunk and write it into the Chroma collection.

    Runs once per ingest config, not per query — this is the expensive,
    do-it-upfront half of retrieval. Each chunk becomes one Chroma row
    holding four things: its id, its text, its vector, and its metadata.
    The metadata is not decoration — `paper_id`, `section` and the page range
    are what `metrics.py` later compares against the golden set's labels, and
    what an answer cites when the agent quotes a chunk.

    Uses upsert rather than add so re-indexing overwrites in place. Chunk ids
    are deterministic (`act:0012`), so the same chunk always lands on the same
    row instead of duplicating on every re-run.

    Args:
        records: Chunk dicts straight from `ingest_file`.

    """
    collection.upsert(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "paper_id": r["paper_id"],
                "section": r["section"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
            }
            for r in records
        ],
        embeddings=create_embeddings([r["text"] for r in records], model="bge-small"),
    )


def rank_chunks(query_str: str, k: int = 5):
    """Retrieve the k chunks closest to a question, best match first.

    The cheap, per-question half of retrieval. Chroma does the ranking —
    nearest-neighbour search over the stored vectors — so nothing here
    computes a similarity by hand. Results come back already ordered, which
    is what makes MRR meaningful downstream.

    Note the BGE models are asymmetric: queries want an instruction prefix
    ("Represent this sentence for searching relevant passages: ") that stored
    chunks must not have. Prepend it here, not in `index_chroma`.

    Args:
        query_str: The question, in natural language.
        k: How many chunks to return. Defaults to the 5 the agent uses.

    Returns:
        Chroma's result dict with documents, metadatas and distances.
    """

    results = collection.query(
        query_embeddings=create_embeddings([query_str], model="bge-small"),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results
