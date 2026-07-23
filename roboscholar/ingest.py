"""PDF/Markdown -> structured chunk records (paper_id, section, page metadata).

Pipeline (see NOTES.md for the reasoning behind each stage):

    PDF:      extract_blocks -> get_headings -> split_sections -> chunk_section
    Markdown: md_sections -> chunk_section

Both paths share build_records and produce the same record shape, ready for
Chroma: metadata carries paper_id, section, and page range, which the golden
set labels and retrieval metrics score against. Notebook draft (kept for the
experiment write-up): notebooks/ingest.py.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import TypedDict

import fitz
import trafilatura

Block = tuple[int, str]
"""(1-based page number, whitespace-normalized text). Page 0 for markdown."""


class Heading(TypedDict):
    level: int
    title: str
    page: int


class Section(TypedDict):
    title: str
    level: int
    blocks: list[Block]


class ChunkRecord(TypedDict):
    id: str
    text: str
    paper_id: str
    section: str
    page_start: int
    page_end: int
    n_words: int


# --------------------------------------------------------------------------- #
# PDF block extraction
# --------------------------------------------------------------------------- #

# figure/table captions: "Fig. 1:", "Figure 2.", "TABLE I:" — the punctuation
# right after the number separates a caption block from body text that merely
# says "Figure 3 shows..."
CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?|table)\s+[ivxlcdm\d]+\s*[.:]", re.IGNORECASE)


def extract_blocks(pdf_path: str | Path) -> list[Block]:
    """Extract text blocks in reading order, junk filtered.

    Content-stream order is trusted (LaTeX PDFs emit column-by-column);
    sort=True would interleave two-column layouts. Drops image blocks, the
    arXiv sidebar watermark, bare page-number footers, and figure/table
    captions.
    """
    doc = fitz.open(pdf_path)
    blocks: list[Block] = []
    for page_no, page in enumerate(doc, start=1):
        for _x0, _y0, _x1, _y1, text, _bno, btype in page.get_text("blocks"):
            if btype != 0:  # 1 = image block
                continue
            t = " ".join(text.split())
            if not t:
                continue
            if t.startswith("arXiv:") and len(t) < 60:  # sidebar watermark
                continue
            if t.isdigit():  # bare page-number footer
                continue
            if CAPTION_RE.match(t):
                continue
            blocks.append((page_no, t))
    doc.close()
    return blocks


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #

# leading section numbering: "IV.", "IV-A", "2.1", "D.0.1", "A." — ToC titles
# and page blocks number headings differently, so strip before comparing
HEAD_NUM_RE = re.compile(
    r"^\s*(?:[IVXLCDM]+(?:-[A-Z])?|[A-Z](?:\.\d+)+|\d+(?:\.\d+)*|[A-Z])[.):]?\s+"
)


def norm_key(s: str) -> str:
    """Normalize for heading matching: small-caps extract as 'I NTRODUCTION',
    so strip everything but [a-z0-9]."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def strip_numbering(s: str) -> str:
    return HEAD_NUM_RE.sub("", s, count=1)


def get_headings(pdf_path: str | Path, max_level: int = 2) -> list[Heading]:
    """Headings in document order — PDF ToC if present, else font heuristic."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    if toc:
        doc.close()
        return [
            {"level": lvl, "title": " ".join(title.split()), "page": page}
            for lvl, title, page in toc
            if lvl <= max_level
        ]
    headings = _font_fallback_headings(doc, max_level)
    doc.close()
    return headings


def _font_fallback_headings(doc: fitz.Document, max_level: int) -> list[Heading]:
    """A heading is a short, bold line clearly larger than body text.

    Level = rank of its font size among recurring heading sizes (>= 3 lines,
    so a one-off title line can't claim level 1).
    """
    body_size = _body_font_size(doc)
    candidates: list[tuple[float, str, int]] = []
    for page_no, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                text = " ".join(s["text"] for s in spans).strip()
                if not text or len(text) > 80:
                    continue
                bold = bool(spans[0]["flags"] & 16)
                if bold:
                    candidates.append((round(spans[0]["size"], 1), text, page_no))

    counts = collections.Counter(size for size, _, _ in candidates)
    heading_sizes = sorted(
        (s for s, n in counts.items() if n >= 3 and s > body_size + 1.5),
        reverse=True,
    )
    level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}
    return [
        {"level": level_of[size], "title": title, "page": page}
        for size, title, page in candidates
        if size in level_of and level_of[size] <= max_level
    ]


def _body_font_size(doc: fitz.Document) -> float:
    """Dominant font size, weighted by characters — the body text size."""
    sizes: collections.Counter[float] = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #


def consume_key(text: str, key: str) -> str | None:
    """If text's alphanumeric stream starts with normalized `key`, return the
    remainder of text after it (may be ""); else None.

    Matches headings case/punctuation/small-caps-insensitively AND recovers
    trailing body text when a heading block merged with its first paragraph.
    `key` must already be normalized (norm_key output).
    """
    j = 0
    for i, ch in enumerate(text):
        if j == len(key):
            return text[i:].strip()
        if ch.isalnum():
            if ch.lower() != key[j]:
                return None
            j += 1
    return "" if j == len(key) else None


def split_sections(
    blocks: list[Block], headings: list[Heading]
) -> tuple[list[Section], list[Heading]]:
    """Assign blocks to sections by walking both lists in document order.

    Handles the failure modes found empirically (see NOTES.md): headings merged
    with body text (remainder kept as body), headings split across blocks
    (pending-key continuation), ToC/page numbering mismatches, short fragments
    impersonating headings, and bibliographies missing from the ToC.

    Returns (sections, unmatched). `unmatched` should be empty — eyeball it
    whenever a new paper is added.
    """
    sections: list[Section] = [{"title": "Front matter", "level": 1, "blocks": []}]
    hi = 0
    unmatched: list[Heading] = []
    pending = ""  # rest of a heading whose first line ended the previous block
    for page, text in blocks:
        if pending:
            rest = consume_key(text, pending)
            if rest is not None:  # block completes the heading
                pending = ""
                if rest:
                    sections[-1]["blocks"].append((page, rest))
                continue
            np_ = norm_key(text)
            if np_ and pending.startswith(np_):  # yet another heading line
                pending = pending[len(np_):]
                continue
            pending = ""  # give up, treat as body

        # references often aren't a ToC entry — force a boundary so the
        # bibliography never pollutes the paper's final section
        if norm_key(strip_numbering(text)) in ("references", "bibliography"):
            sections.append({"title": "References", "level": 1, "blocks": []})
            if hi < len(headings) and norm_key(
                strip_numbering(headings[hi]["title"])
            ) in ("references", "bibliography"):
                hi += 1  # consume it so the unmatched diagnostic stays clean
            continue

        stripped = strip_numbering(text)
        matched = None
        remainder = ""
        for j in range(hi, len(headings)):
            h = headings[j]
            if h["page"] > page:  # headings are in page order; stop looking ahead
                break
            nt = norm_key(strip_numbering(h["title"]))
            if not nt:
                continue
            rest = consume_key(stripped, nt)
            if rest is not None:  # block starts with the full heading
                matched, remainder = j, rest
                break
            nb = norm_key(stripped)
            # partial heading line: block is a *long* prefix of the title
            # (>= 8 chars, so equation fragments like "×K" can't claim a match)
            if len(nb) >= 8 and len(text) <= 90 and nt.startswith(nb):
                matched, remainder = j, ""
                pending = nt[len(nb):]
                break
        if matched is not None:
            unmatched.extend(headings[hi:matched])  # skipped-over headings never found
            h = headings[matched]
            sections.append({"title": h["title"], "level": h["level"], "blocks": []})
            hi = matched + 1
            if remainder:  # body text that shared the heading's block
                sections[-1]["blocks"].append((page, remainder))
        else:
            sections[-1]["blocks"].append((page, text))
    unmatched.extend(headings[hi:])
    return sections, unmatched


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def chunk_section(
    section: Section, n_words: int, overlap_words: int
) -> list[list[tuple[str, int]]]:
    """Chunk one section into lists of (sentence, page).

    Accumulates whole sentences until adding the next would exceed n_words,
    then emits and seeds the next chunk with trailing sentences totalling at
    most overlap_words. Never crosses the section boundary. Pathological
    "sentences" (garbled equations, tables) are hard-split at n_words.
    """
    sentences: list[tuple[str, int]] = []
    for page, text in section["blocks"]:
        for sent in SENT_RE.split(text):
            words = sent.split()
            for k in range(0, len(words), n_words):
                sentences.append((" ".join(words[k : k + n_words]), page))

    chunks: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_words, seed_len = 0, 0
    for sent, page in sentences:
        w = len(sent.split())
        if cur and cur_words + w > n_words:
            chunks.append(cur)
            # overlap: carry trailing sentences into the next chunk
            tail: list[tuple[str, int]] = []
            tail_words = 0
            for s, p in reversed(cur):
                sw = len(s.split())
                if tail_words + sw > overlap_words:
                    break
                tail.insert(0, (s, p))
                tail_words += sw
            cur, cur_words, seed_len = list(tail), tail_words, len(tail)
        cur.append((sent, page))
        cur_words += w
    if len(cur) > seed_len:  # final chunk only if it has non-overlap content
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

DROP_SECTIONS = {"references", "bibliography"}


def build_records(
    doc_id: str,
    sections: list[Section],
    n_words: int,
    overlap_words: int,
    min_words: int = 25,
) -> list[ChunkRecord]:
    """Sections -> chunk records. Shared by the PDF and markdown paths.

    Chunks under min_words (stray date lines, orphaned fragments) are pure
    retrieval noise and get dropped. IDs are deterministic for a given
    (document, chunk params) but NOT stable across param changes — eval runs
    are only comparable within one ingest config.
    """
    records: list[ChunkRecord] = []
    idx = 0
    parent: str | None = None
    for sec in sections:
        if sec["level"] == 1:
            parent = sec["title"]
        if norm_key(sec["title"]) in DROP_SECTIONS:
            continue
        label = (
            sec["title"]
            if sec["level"] == 1 or parent is None or parent == sec["title"]
            else f"{parent} › {sec['title']}"
        )
        for chunk in chunk_section(sec, n_words, overlap_words):
            text = " ".join(s for s, _ in chunk)
            if len(text.split()) < min_words:
                continue
            pages = [p for _, p in chunk]
            records.append(
                {
                    "id": f"{doc_id}:{idx:04d}",
                    "text": text,
                    "paper_id": doc_id,
                    "section": label,
                    "page_start": min(pages),  # 0 for markdown (no pages)
                    "page_end": max(pages),
                    "n_words": len(text.split()),
                }
            )
            idx += 1
    return records


# --------------------------------------------------------------------------- #
# Markdown path
# --------------------------------------------------------------------------- #

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")


def md_sections(md_text: str) -> list[Section]:
    """Markdown -> sections in the same shape split_sections produces.

    h1 is usually the post title, so h1/h2 both map to level 1 (top-level
    sections) and h3+ to level 2 — keeps section labels short.
    """
    sections: list[Section] = [{"title": "Front matter", "level": 1, "blocks": []}]
    for para in re.split(r"\n\s*\n", md_text):
        lines = [ln.strip() for ln in para.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        m = MD_HEADING_RE.match(lines[0])
        if m:
            level = 1 if len(m.group(1)) <= 2 else 2
            sections.append({"title": m.group(2).strip(), "level": level, "blocks": []})
            lines = lines[1:]
        body = " ".join(lines)
        if body:
            sections[-1]["blocks"].append((0, body))  # markdown has no pages
    return sections


# --------------------------------------------------------------------------- #
# Blog fetching
# --------------------------------------------------------------------------- #


def fetch_blogs(manifest_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Download manifest entries ({title, url}) as markdown via trafilatura
    (strips nav/footer boilerplate). Idempotent — skips existing files, so the
    committed corpus stays stable even if a post changes upstream.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for entry in json.loads(Path(manifest_path).read_text()):
        slug = entry["url"].rstrip("/").split("/")[-1]
        out = out_dir / f"{slug}.md"
        if not out.exists():
            html = trafilatura.fetch_url(entry["url"])
            md = (
                trafilatura.extract(html, output_format="markdown", include_links=False)
                if html
                else None
            )
            if not md:
                raise RuntimeError(f"failed to fetch/extract {entry['url']}")
            out.write_text(md)
        paths.append(out)
    return paths


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def ingest_pdf(
    pdf_path: str | Path, n_words: int = 500, overlap_words: int = 50
) -> tuple[list[ChunkRecord], list[Heading]]:
    """PDF -> chunk records + unmatched-heading diagnostic (should be empty)."""
    paper_id = Path(pdf_path).stem.lower()
    blocks = extract_blocks(pdf_path)
    headings = get_headings(pdf_path)
    sections, unmatched = split_sections(blocks, headings)
    return build_records(paper_id, sections, n_words, overlap_words), unmatched


def ingest_markdown(
    md_path: str | Path, n_words: int = 500, overlap_words: int = 50
) -> list[ChunkRecord]:
    """Markdown file -> chunk records, same schema as ingest_pdf."""
    doc_id = Path(md_path).stem.lower()
    sections = md_sections(Path(md_path).read_text())
    return build_records(doc_id, sections, n_words, overlap_words)


def ingest_file(
    path: str | Path, n_words: int = 500, overlap_words: int = 50
) -> tuple[list[ChunkRecord], list[Heading]]:
    """Dispatch on suffix. Unmatched headings are always [] for markdown."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return ingest_pdf(path, n_words, overlap_words)
    if path.suffix.lower() in (".md", ".markdown"):
        return ingest_markdown(path, n_words, overlap_words), []
    raise ValueError(f"unsupported file type: {path.name} (expected .pdf or .md)")
