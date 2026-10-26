"""CLI for the OSHA regulations assistant.

  python run.py build [--parts 1910] [--xml FILE ...]       download + index
  python run.py ask "how often inspect fire extinguishers"  answer + hallucination check
  python run.py eval [--embedder lsa|minilm]                 all metrics -> results/
"""
import argparse
import json
from pathlib import Path

from osha_rag.embed import Dense, HybridDense
from osha_rag.evaluate import evaluate, to_markdown, write
from osha_rag.grounding import Grounding, evaluate_grounding
from osha_rag.ingest import build_corpus, load_chunks
from osha_rag.retrieve import BM25, TFIDF, Hybrid

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parts", nargs="+", default=["1910"])
    b.add_argument("--xml", nargs="*", type=Path, help="use local CFR XML instead of downloading")
    for name in ("ask", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--embedder", default="lsa", choices=["lsa", "minilm"])
    sub.choices["ask"].add_argument("question")
    args = ap.parse_args()

    if args.cmd == "build":
        chunks = build_corpus(args.parts, DATA, args.xml)
        print(f"indexed {len(chunks)} chunks from {len({c['section'] for c in chunks})} sections")
        return

    chunks = load_chunks(DATA)
    dense = Dense(chunks, args.embedder, cache_dir=DATA)
    hybrid = HybridDense(chunks, dense)
    g = Grounding(chunks, hybrid, dense.emb)

    if args.cmd == "ask":
        ans = g.answer(args.question)
        chk = g.check(ans)
        for c, r in zip(ans["claims"], chk["claims"]):
            status = "GROUNDED" if r["grounded"] else f"UNSUPPORTED ({r['reason']})"
            print(f"\n- {c['text']}\n  [29 CFR {c['cite']}]  {status}  sim={r['sim']}")
        print(f"\nhallucination detected: {chk['hallucination']}")
    elif args.cmd == "eval":
        qs = [json.loads(l) for l in (ROOT / "eval/questions.jsonl").read_text().splitlines() if l]
        rep = evaluate(chunks, qs, (BM25, TFIDF, Hybrid, lambda c: dense, lambda c: hybrid))
        rep["grounding"] = evaluate_grounding(g, qs)
        write(rep, RESULTS)
        print(to_markdown(rep))
        print(json.dumps({k: v for k, v in rep["grounding"].items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
