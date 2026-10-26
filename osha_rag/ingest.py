"""Download 29 CFR (OSHA) from the eCFR API and parse it into sections and chunks.

Supports both eCFR "full" XML (DIV8 TYPE="SECTION") and GovInfo bulk CFR XML
(<SECTION><SECTNO>), so a manually downloaded file also works.
"""
import json
import re
from pathlib import Path

import requests
from lxml import etree

ECFR = "https://www.ecfr.gov/api/versioner/v1"
SECTION_RE = re.compile(r"(\d{4}\.\d+[a-z]?)")
WS = re.compile(r"\s+")


def download_part(part: str, out_dir: Path, title: int = 29) -> Path:
    """Fetch the latest version of one CFR part as XML (cached)."""
    out = out_dir / f"title{title}_part{part}.xml"
    if out.exists():
        return out
    titles = requests.get(f"{ECFR}/titles.json", timeout=60).json()["titles"]
    date = next(t["up_to_date_as_of"] for t in titles if t["number"] == title)
    r = requests.get(f"{ECFR}/full/{date}/title-{title}.xml", params={"part": part}, timeout=300)
    r.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    (out_dir / f"title{title}_part{part}.meta.json").write_text(json.dumps({"as_of": date, "part": part}))
    return out


def _text(el) -> str:
    return WS.sub(" ", "".join(el.itertext())).strip()


def parse_sections(xml_path: Path) -> list[dict]:
    """Return [{section, heading, paragraphs}] for every section and appendix."""
    root = etree.parse(str(xml_path), etree.XMLParser(recover=True, huge_tree=True)).getroot()
    sections = []
    # eCFR format
    for div in root.iter("DIV8", "DIV9"):
        head = div.find("HEAD")
        heading = _text(head) if head is not None else ""
        num = div.get("N", "") if div.tag == "DIV8" else ""
        m = SECTION_RE.search(num) or SECTION_RE.search(heading)
        if not m:
            continue
        paras = [_text(c) for c in div if c.tag not in ("HEAD",) and not c.tag.startswith("DIV")]
        sections.append({"section": m.group(1), "heading": heading,
                         "is_appendix": div.tag == "DIV9", "paragraphs": [p for p in paras if p]})
    # GovInfo bulk format
    for sec in root.iter("SECTION"):
        sectno, subject = sec.find("SECTNO"), sec.find("SUBJECT")
        m = SECTION_RE.search(_text(sectno)) if sectno is not None else None
        if not m:
            continue
        paras = [_text(c) for c in sec if c.tag not in ("SECTNO", "SUBJECT")]
        sections.append({"section": m.group(1),
                         "heading": f"§ {m.group(1)} {_text(subject) if subject is not None else ''}".strip(),
                         "is_appendix": False, "paragraphs": [p for p in paras if p]})
    return sections


def chunk_sections(sections: list[dict], max_words: int = 220, overlap_paras: int = 1) -> list[dict]:
    """Greedy paragraph packing within a section; the heading is attached to every chunk."""
    chunks = []
    for s in sections:
        paras, start = s["paragraphs"], 0
        while start < len(paras):
            end, words = start, 0
            while end < len(paras) and (words == 0 or words + len(paras[end].split()) <= max_words):
                words += len(paras[end].split())
                end += 1
            body = " ".join(paras[start:end])
            chunks.append({"id": f"{s['section']}#{len(chunks)}", "section": s["section"],
                           "heading": s["heading"], "text": body,
                           "first_para": paras[start][:40]})
            if end >= len(paras):
                break
            start = max(end - overlap_paras, start + 1)
    return chunks


def build_corpus(parts: list[str], data_dir: Path, xml_files: list[Path] | None = None) -> list[dict]:
    files = xml_files or [download_part(p, data_dir) for p in parts]
    sections = [s for f in files for s in parse_sections(f)]
    chunks = chunk_sections(sections)
    (data_dir / "sections.jsonl").write_text("\n".join(json.dumps(s) for s in sections))
    (data_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    return chunks


def load_chunks(data_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (data_dir / "chunks.jsonl").read_text().splitlines() if l]
