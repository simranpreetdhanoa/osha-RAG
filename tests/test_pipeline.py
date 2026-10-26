"""Mechanics tests on a tiny synthetic XML fixture (placeholder text, not real regulation text)."""
from pathlib import Path

from osha_rag.evaluate import evaluate, score_one
from osha_rag.ingest import chunk_sections, parse_sections
from osha_rag.retrieve import BM25, Hybrid, search

ECFR_XML = """<DIV5 N="1910" TYPE="PART"><DIV6 N="L" TYPE="SUBPART">
<DIV8 N="1910.157" TYPE="SECTION"><HEAD>§ 1910.157 Portable fire extinguishers.</HEAD>
<P>(a) Placeholder: extinguishers are visually inspected monthly.</P>
<P>(b) Placeholder: annual maintenance check of extinguishers.</P></DIV8>
<DIV8 N="1910.95" TYPE="SECTION"><HEAD>§ 1910.95 Occupational noise exposure.</HEAD>
<P>(a) Placeholder: hearing conservation program at noise action level.</P></DIV8>
<DIV9 N="Appendix A" TYPE="APPENDIX"><HEAD>Appendix A to § 1910.95 - Noise dose</HEAD>
<P>Placeholder appendix text about dosimeter.</P></DIV9>
</DIV6></DIV5>"""

GOVINFO_XML = """<CFRDOC><PART><SECTION><SECTNO>§ 1910.178</SECTNO><SUBJECT>Powered industrial trucks.</SUBJECT>
<P>Placeholder: operator evaluation every three years.</P></SECTION></PART></CFRDOC>"""


def _write(tmp_path, name, txt):
    p = Path(tmp_path) / name
    p.write_text(txt)
    return p


def test_parse_both_formats(tmp_path):
    secs = parse_sections(_write(tmp_path, "e.xml", ECFR_XML)) + parse_sections(_write(tmp_path, "g.xml", GOVINFO_XML))
    assert [s["section"] for s in secs] == ["1910.157", "1910.95", "1910.95", "1910.178"]
    assert secs[2]["is_appendix"] and len(secs[0]["paragraphs"]) == 2


def test_chunking_respects_word_limit():
    sec = [{"section": "1910.1", "heading": "h", "paragraphs": ["w " * 100] * 5}]
    chunks = chunk_sections(sec, max_words=220, overlap_paras=1)
    assert all(len(c["text"].split()) <= 220 for c in chunks) and len(chunks) == 4


def test_retrieval_and_synonyms(tmp_path):
    chunks = chunk_sections(parse_sections(_write(tmp_path, "e.xml", ECFR_XML)) +
                            parse_sections(_write(tmp_path, "g.xml", GOVINFO_XML)))
    assert search(BM25(chunks), chunks, "how often inspect fire extinguishers")[0]["section"] == "1910.157"
    assert search(Hybrid(chunks), chunks, "forklift operator evaluation")[0]["section"] == "1910.178"
    hits = search(BM25(chunks), chunks, "noise hearing dosimeter", k=10)
    assert [h["section"] for h in hits].count("1910.95") == 1  # deduped per section


def test_metrics_math():
    m = score_one(["a", "b", "c"], {"b"})
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr@10"] == 0.5
    assert abs(m["ndcg@10"] - 1 / __import__("math").log2(3)) < 1e-9


def test_evaluate_reports_missing_gold(tmp_path):
    chunks = chunk_sections(parse_sections(_write(tmp_path, "e.xml", ECFR_XML)))
    rep = evaluate(chunks, [{"id": 1, "topic": "t", "question": "fire extinguisher", "gold": ["1910.157"]},
                            {"id": 2, "topic": "t", "question": "x", "gold": ["1910.999"]}])
    assert rep["gold_missing_from_corpus"] == ["1910.999"]
    assert rep["systems"]["bm25"]["hit@1"] == 0.5


def test_faiss_dense_and_grounding(tmp_path):
    from osha_rag.embed import Dense, HybridDense, LSAEmbedder
    from osha_rag.grounding import Grounding
    chunks = chunk_sections(parse_sections(_write(tmp_path, "e.xml", ECFR_XML)) +
                            parse_sections(_write(tmp_path, "g.xml", GOVINFO_XML)))
    dense = Dense(chunks, LSAEmbedder(dim=3))
    assert dense.index.ntotal == len(chunks)
    g = Grounding(chunks, HybridDense(chunks, dense), dense.emb)
    ok = g.check_claim("Placeholder: extinguishers are visually inspected monthly.", "1910.157")
    assert ok["grounded"]
    assert g.check_claim("Placeholder: extinguishers are visually inspected 12 times.", "1910.157")["reason"] == "number_not_in_source"
    assert g.check_claim("anything at all here", "1910.9999")["reason"] == "citation_not_found"
