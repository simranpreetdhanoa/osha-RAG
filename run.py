"""CLI for the OSHA regulations assistant.

  python run.py build [--parts 1910] [--xml FILE ...]       download + index
  python run.py ask "how often inspect fire extinguishers"  answer + hallucination check
  python run.py eval [--embedder minilm|lsa]                 all metrics -> results/<embedder>/
  python run.py compare                                      results/comparison.md (both techniques)
"""
import argparse
import json
from pathlib import Path

from osha_rag.embed import Dense
from osha_rag.evaluate import evaluate, to_markdown, write
from osha_rag.golden import score_answers, verify_golden
from osha_rag.grounding import Grounding, evaluate_grounding
from osha_rag.ingest import build_corpus, load_chunks
from osha_rag import summary

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
        p.add_argument("--embedder", default="minilm", choices=["minilm", "lsa"])
        p.add_argument("--sentences", type=int, default=5, help="sentences per answer")
    sub.choices["ask"].add_argument("question")
    sub.add_parser("summary", help="headline metrics from the last eval").add_argument(
        "--embedder", default="minilm", choices=["minilm", "lsa"])
    sub.add_parser("compare", help="compare the techniques' headline metrics")
    args = ap.parse_args()

    if args.cmd == "build":
        chunks = build_corpus(args.parts, DATA, args.xml)
        print(f"indexed {len(chunks)} chunks from {len({c['section'] for c in chunks})} sections")
        return

    if args.cmd == "compare":
        md = summary.compare({e: json.loads((RESULTS / e / "summary.json").read_text()) for e in ("lsa", "minilm")})
        (RESULTS / "comparison.md").write_text(md)
        print(md)
        return

    out = RESULTS / args.embedder
    if args.cmd == "summary":
        s = summary.summarize(out, ROOT / "eval/golden_answers_corrected.json")
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        (out / "summary.md").write_text(summary.to_markdown(s))
        print(summary.to_markdown(s))
        return

    chunks = load_chunks(DATA)
    dense = Dense(chunks, args.embedder, cache_dir=DATA)
    g = Grounding(chunks, dense, dense.emb, args.sentences)

    if args.cmd == "ask":
        ans = g.answer(args.question)
        chk = g.check(ans)
        for c, r in zip(ans["claims"], chk["claims"]):
            status = "GROUNDED" if r["grounded"] else f"UNSUPPORTED ({r['reason']})"
            print(f"\n- {c['text']}\n  [29 CFR {c['cite']}]  {status}  sim={r['sim']}")
        print(f"\nhallucination detected: {chk['hallucination']}")
    elif args.cmd == "eval":
        qs = [json.loads(l) for l in (ROOT / "eval/questions.jsonl").read_text().splitlines() if l]
        rep = evaluate(chunks, qs, (lambda c: dense,))
        rep["grounding"] = evaluate_grounding(g, qs)
        rep["golden"] = {}
        answerers = {dense.name: g}
        for label, fname in (("as_given", "golden_answers.json"), ("corrected", "golden_answers_corrected.json")):
            if (ROOT / "eval" / fname).exists():
                gold = json.loads((ROOT / "eval" / fname).read_text())
                ver = verify_golden(gold, qs, g.src_text)
                rep["golden"][label] = {"verification": ver, "systems": {
                    n: score_answers(a, gold, ver, qs) for n, a in answerers.items()}}
        write(rep, out)
        print(to_markdown(rep))
        s = summary.summarize(out, ROOT / "eval/golden_answers_corrected.json")
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        (out / "summary.md").write_text(summary.to_markdown(s))
        print(summary.to_markdown(s))


if __name__ == "__main__":
    main()
