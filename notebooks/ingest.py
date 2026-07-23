import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import collections
    import re
    from pathlib import Path

    import fitz
    import marimo as mo

    return Path, collections, fitz, mo, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ingest pipeline (notebook draft)

    PDF → filtered blocks → sections (via ToC, font-size fallback) → sentence-aware
    chunks with overlap + metadata (`paper_id`, `section`, `page_start/end`).

    Pipeline stages, each its own cell so intermediate output is inspectable:

    1. **`extract_blocks`** — `page.get_text("blocks")`, keeping page numbers,
       dropping images, the arXiv watermark, and bare page-number blocks.
    2. **`get_headings`** — 4/5 papers have PDF bookmarks (`doc.get_toc()`);
       `world-models` doesn't, so we fall back to a font heuristic (bold + larger
       than body text).
    3. **`split_sections`** — walk blocks in reading order, start a new section
       whenever a block matches the next ToC title (normalized, because small-caps
       headings extract as `"I NTRODUCTION"`).
    4. **`chunk_section`** — accumulate whole sentences up to `n_words`, seed the
       next chunk with trailing sentences up to `overlap_words`. Chunks never cross
       section boundaries. References/bibliography sections are dropped.
    """)
    return


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data" / "raw"
    PAPERS_DIR = DATA_DIR / "papers"
    BLOGS_DIR = DATA_DIR / "blogs"
    BLOGS_MANIFEST = DATA_DIR / "blogs.json"
    ACT_PATH = PAPERS_DIR / "ACT.pdf"
    return ACT_PATH, BLOGS_DIR, BLOGS_MANIFEST, PAPERS_DIR


@app.cell
def _(fitz, re):
    # figure/table captions: "Fig. 1:", "Figure 2.", "TABLE I:" — the punctuation
    # right after the number is what separates a caption block from body text
    # that merely says "Figure 3 shows..."
    CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?|table)\s+[ivxlcdm\d]+\s*[.:]", re.IGNORECASE)

    def extract_blocks(pdf_path):
        """Extract text blocks as (page_number, text), 1-based pages, junk filtered."""
        doc = fitz.open(pdf_path)
        blocks = []
        for page_no, page in enumerate(doc, start=1):
            for x0, y0, x1, y1, text, _bno, btype in page.get_text("blocks"):
                if btype != 0:  # 1 = image block
                    continue
                t = " ".join(text.split())
                if not t:
                    continue
                if t.startswith("arXiv:") and len(t) < 60:  # sidebar watermark
                    continue
                if t.isdigit():  # bare page-number footer
                    continue
                if CAPTION_RE.match(t):  # figure/table captions: layout, not comprehension
                    continue
                blocks.append((page_no, t))
        doc.close()
        return blocks

    return (extract_blocks,)


@app.cell
def _(collections, fitz, re):
    def norm_key(s):
        """Normalize for heading matching: small-caps extract as 'I NTRODUCTION',
        ToC titles may or may not carry numbering — strip everything but [a-z0-9]."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def get_headings(pdf_path, max_level=2):
        """[{level, title, page}] in document order. ToC if present, else font heuristic."""
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        if toc:
            headings = [
                {"level": lvl, "title": " ".join(title.split()), "page": page}
                for lvl, title, page in toc
                if lvl <= max_level
            ]
            doc.close()
            return headings

        # Fallback: a heading is a short, bold line, clearly larger than body text.
        # Level = rank of its font size (biggest recurring size = level 1).
        candidates = []
        for page_no, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    text = " ".join(s["text"] for s in spans).strip()
                    if not text or len(text) > 80:
                        continue
                    size = round(spans[0]["size"], 1)
                    bold = bool(spans[0]["flags"] & 16)
                    if bold:
                        candidates.append({"size": size, "title": text, "page": page_no})
        doc.close()

        body_size = 12.0  # anything <= body-ish is not a heading
        counts = collections.Counter(c["size"] for c in candidates)
        # recurring sizes only (>= 3 lines) so the one-off title line doesn't claim level 1
        heading_sizes = sorted(
            (s for s, n in counts.items() if n >= 3 and s > body_size + 1.5),
            reverse=True,
        )
        level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}
        return [
            {"level": level_of[c["size"]], "title": c["title"], "page": c["page"]}
            for c in candidates
            if c["size"] in level_of and level_of[c["size"]] <= max_level
        ]

    return get_headings, norm_key


@app.cell
def _(norm_key, re):
    # leading section numbering: "IV.", "IV-A", "2.1", "D.0.1", "A." — ToC titles
    # and page blocks number headings differently, so strip before comparing
    HEAD_NUM_RE = re.compile(r"^\s*(?:[IVXLCDM]+(?:-[A-Z])?|[A-Z](?:\.\d+)+|\d+(?:\.\d+)*|[A-Z])[.):]?\s+")

    def strip_numbering(s):
        return HEAD_NUM_RE.sub("", s, count=1)

    def consume_key(text, key):
        """If text's alphanumeric stream starts with normalized `key`, return the
        remainder of text after it (may be ""); else None. Lets us match headings
        case/punctuation/small-caps-insensitively AND recover trailing body text
        when a heading block merged with its first paragraph."""
        j = 0
        for i, ch in enumerate(text):
            if j == len(key):
                return text[i:].strip()
            if ch.isalnum():
                if ch.lower() != key[j]:
                    return None
                j += 1
        return "" if j == len(key) else None

    def split_sections(blocks, headings):
        """Assign blocks to sections by walking both lists in document order.

        Handles the three real-world failure modes: heading merged with body text
        (remainder is kept as body), heading split across blocks (pending-key
        continuation), and numbering mismatches between ToC and page. Returns
        (sections, unmatched) — eyeball unmatched per paper as the diagnostic.
        """
        sections = [{"title": "Front matter", "level": 1, "blocks": []}]
        hi = 0
        unmatched = []
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
                # if References also exists as a heading, consume it so the
                # unmatched diagnostic stays clean
                if hi < len(headings) and norm_key(
                    strip_numbering(headings[hi]["title"])
                ) in ("references", "bibliography"):
                    hi += 1
                continue

            stripped = strip_numbering(text)
            matched = None
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

    return (split_sections,)


@app.cell
def _(re):
    SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

    def chunk_section(section, n_words, overlap_words):
        """Chunk one section into lists of (sentence, page).

        Accumulate whole sentences until adding the next would exceed n_words,
        then emit and seed the next chunk with trailing sentences totalling at
        most overlap_words. Never crosses the section boundary.
        """
        sentences = []
        for page, text in section["blocks"]:
            for sent in SENT_RE.split(text):
                words = sent.split()
                # hard-split pathological "sentences" (garbled equations, tables)
                for k in range(0, len(words), n_words):
                    sentences.append((" ".join(words[k : k + n_words]), page))

        chunks = []
        cur, cur_words, seed_len = [], 0, 0
        for sent, page in sentences:
            w = len(sent.split())
            if cur and cur_words + w > n_words:
                chunks.append(cur)
                # overlap: carry trailing sentences into the next chunk
                tail, tail_words = [], 0
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

    return (chunk_section,)


@app.cell
def _(chunk_section, norm_key):
    DROP_SECTIONS = {"references", "bibliography"}

    def build_records(doc_id, sections, n_words, overlap_words, min_words=25):
        """Sections → chunk records. Shared by the PDF and markdown paths.

        Chunks under min_words (stray date lines, orphaned captions) are pure
        retrieval noise and get dropped.
        """
        records = []
        idx = 0
        parent = None
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

    return (build_records,)


@app.cell
def _(Path, build_records, extract_blocks, get_headings, split_sections):
    def ingest_pdf(pdf_path, n_words=500, overlap_words=50):
        """PDF → list of chunk records ready for Chroma (+ unmatched-heading diagnostic)."""
        paper_id = Path(pdf_path).stem.lower()
        blocks = extract_blocks(pdf_path)
        headings = get_headings(pdf_path)
        sections, unmatched = split_sections(blocks, headings)
        return build_records(paper_id, sections, n_words, overlap_words), unmatched

    return (ingest_pdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ACT — chunk browser
    """)
    return


@app.cell
def _(ACT_PATH, ingest_pdf, mo):
    act_chunks, act_unmatched = ingest_pdf(ACT_PATH, n_words=500, overlap_words=50)
    print(f"{len(act_chunks)} chunks; unmatched headings: {[h['title'] for h in act_unmatched]}")
    mo.ui.table(
        [
            {
                "id": c["id"],
                "section": c["section"],
                "pages": f"{c['page_start']}–{c['page_end']}",
                "words": c["n_words"],
                "text": c["text"][:120] + "…",
            }
            for c in act_chunks
        ],
        page_size=15,
    )
    return (act_chunks,)


@app.cell
def _(act_chunks, mo):
    chunk_picker = mo.ui.number(
        start=0, stop=len(act_chunks) - 1, value=0, label="inspect chunk #"
    )
    chunk_picker
    return (chunk_picker,)


@app.cell
def _(act_chunks, chunk_picker, mo):
    sel = act_chunks[chunk_picker.value]
    mo.md(
        f"**`{sel['id']}`** — *{sel['section']}* "
        f"(pp. {sel['page_start']}–{sel['page_end']}, {sel['n_words']} words)\n\n---\n\n"
        f"{sel['text']}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All papers — sanity summary
    """)
    return


@app.cell
def _(PAPERS_DIR, ingest_pdf, mo):
    summary = []
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        recs, unmatched = ingest_pdf(pdf, n_words=500, overlap_words=50)
        summary.append(
            {
                "paper": pdf.stem,
                "chunks": len(recs),
                "sections": len({r["section"] for r in recs}),
                "avg_words": round(sum(r["n_words"] for r in recs) / len(recs)),
                "unmatched_headings": len(unmatched),
            }
        )
        print(summary[-1])
        if unmatched:
            print(f"   unmatched: {[h['title'] for h in unmatched][:8]}")
    mo.ui.table(summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Blogs — same chunker, markdown path

    Blogs are fetched once from `data/raw/blogs.json` into `data/raw/blogs/*.md`
    (trafilatura strips nav/footer boilerplate and keeps `#`/`##` headings).
    Markdown headings are unambiguous, so sectioning is a regex — no ToC/font
    heuristics needed. Chunking is the exact same `chunk_section`. Markdown has
    no pages, so `page_start`/`page_end` are 0.
    """
    )
    return


@app.cell
def _(BLOGS_DIR, BLOGS_MANIFEST):
    import json

    import trafilatura

    def fetch_blogs(manifest_path=BLOGS_MANIFEST, out_dir=BLOGS_DIR):
        """Download manifest entries as markdown. Idempotent — skips existing files."""
        paths = []
        for entry in json.loads(manifest_path.read_text()):
            slug = entry["url"].rstrip("/").split("/")[-1]
            out = out_dir / f"{slug}.md"
            if not out.exists():
                html = trafilatura.fetch_url(entry["url"])
                md = trafilatura.extract(html, output_format="markdown", include_links=False)
                if not md:
                    print(f"FAILED to extract: {entry['url']}")
                    continue
                out.write_text(md)
            paths.append(out)
        return paths

    return (fetch_blogs,)


@app.cell
def _(Path, build_records, re):
    MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")

    def md_sections(md_text):
        """Markdown → sections in the same shape split_sections produces.

        h1 is usually the post title, so h1/h2 both map to level 1 (top-level
        sections) and h3+ to level 2 — keeps section labels short.
        """
        sections = [{"title": "Front matter", "level": 1, "blocks": []}]
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

    def ingest_markdown(md_path, n_words=500, overlap_words=50):
        """Markdown file → chunk records, same schema as ingest_pdf."""
        doc_id = Path(md_path).stem.lower()
        sections = md_sections(Path(md_path).read_text())
        return build_records(doc_id, sections, n_words, overlap_words)

    return (ingest_markdown,)


@app.cell
def _(fetch_blogs, ingest_markdown, mo):
    blog_chunks = []
    for _p in fetch_blogs():
        blog_chunks.extend(ingest_markdown(_p))
    print(f"{len(blog_chunks)} blog chunks from {len({c['paper_id'] for c in blog_chunks})} posts")
    mo.ui.table(
        [
            {
                "id": c["id"],
                "section": c["section"],
                "words": c["n_words"],
                "text": c["text"][:120] + "…",
            }
            for c in blog_chunks
        ],
        page_size=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment
    - With 500 & 1000 word size (maybe token size too)
    - With 0, 10% & 20% token size
    - Figure out which combination gives the best results using metrics.py
    """)
    return


if __name__ == "__main__":
    app.run()
