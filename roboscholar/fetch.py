"""Fetch source documents listed in the data manifests.

The corpus is not committed. `data/raw/papers.json` records what each paper is and where
it came from (arXiv id, versioned URL, size, sha256); this module downloads them into
`data/raw/papers/` and verifies each file against its recorded checksum.

Versioned arXiv URLs (…v1, …v5) are deliberate: an unversioned URL follows the latest
revision, so its bytes drift and the checksum stops meaning anything.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "robo-scholar/0.1 (+https://github.com/StanKarz/robo-scholar)"
CHUNK = 1 << 16


@dataclass
class Paper:
    id: str
    title: str
    arxiv_id: str
    url: str
    filename: str
    bytes: int
    sha256: str


@dataclass
class Result:
    paper: Paper
    status: str  # "ok" | "cached" | "checksum-mismatch" | "error"
    detail: str = ""


def load_manifest(path: Path) -> list[Paper]:
    entries = json.loads(Path(path).read_text())
    return [Paper(**e) for e in entries]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(CHUNK):
            f.write(block)
    tmp.replace(dest)


def fetch_paper(paper: Paper, dest_dir: Path, force: bool = False) -> Result:
    dest = dest_dir / paper.filename

    if dest.exists() and not force:
        if sha256_of(dest) == paper.sha256:
            return Result(paper, "cached")
        return Result(paper, "checksum-mismatch", "on-disk copy differs; re-run with --force")

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        _download(paper.url, dest)
    except Exception as e:  # network, 404, timeout
        return Result(paper, "error", str(e))

    actual = sha256_of(dest)
    if actual != paper.sha256:
        return Result(
            paper,
            "checksum-mismatch",
            f"expected {paper.sha256[:12]}… got {actual[:12]}…",
        )
    return Result(paper, "ok")


def fetch_all(
    manifest: Path = Path("data/raw/papers.json"),
    dest_dir: Path = Path("data/raw/papers"),
    force: bool = False,
) -> list[Result]:
    return [fetch_paper(p, dest_dir, force) for p in load_manifest(manifest)]
