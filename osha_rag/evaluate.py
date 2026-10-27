"""Retrieval evaluation: Hit@k, Recall@k, MRR@10, nDCG@10, latency, per-topic breakdown."""
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .retrieve import BM25, TFIDF, Hybrid, search

KS = (1, 3, 5, 10)


def score_one(ranked: list[str], gold: set[str]) -> dict:
    m = {f"hit@{k}": float(any(s in gold for s in ranked[:k])) for k in KS}
    m.update({f"recall@{k}": len(gold & set(ranked[:k])) / len(gold) for k in KS})
    rr = next((1 / (i + 1) for i, s in enumerate(ranked[:10]) if s in gold), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    m.update({"mrr@10": rr, "ndcg@10": dcg / idcg})
    return m


def evaluate(chunks, questions, retrievers=(BM25, TFIDF, Hybrid)):
    corpus_sections = {c["section"] for c in chunks}
    report = {"n_questions": len(questions), "n_chunks": len(chunks),
              "n_sections": len(corpus_sections),
              "gold_missing_from_corpus": sorted({g for q in questions for g in q["gold"]} - corpus_sections),
              "systems": {}, "per_question": {}}
    for make in retrievers:
        t0 = time.perf_counter()
        r = make(chunks)
        build_s = time.perf_counter() - t0
        rows, lat, by_topic = [], [], defaultdict(list)
        for q in questions:
            t = time.perf_counter()
            hits = search(r, chunks, q["question"], k=10)
            lat.append((time.perf_counter() - t) * 1000)
            ranked = [h["section"] for h in hits]
            m = score_one(ranked, set(q["gold"]))
            rows.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "top5": ranked[:5], **m})
            by_topic[q["topic"]].append(m["hit@1"])
        agg = {k: round(statistics.mean(r_[k] for r_ in rows), 4) for k in rows[0] if "@" in k}
        agg.update({"latency_ms_p50": round(statistics.median(lat), 2),
                    "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2),
                    "index_build_s": round(build_s, 2),
                    "hit@1_by_topic": {t: round(statistics.mean(v), 3) for t, v in sorted(by_topic.items())}})
        report["systems"][r.name] = agg
        report["per_question"][r.name] = rows
    return report


def to_markdown(rep: dict) -> str:
    cols = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10", "latency_ms_p50", "latency_ms_p95"]
    out = [f"# OSHA Assistant — retrieval evaluation\n",
           f"Corpus: {rep['n_sections']} sections, {rep['n_chunks']} chunks. Questions: {rep['n_questions']}.\n",
           "| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, a in rep["systems"].items():
        out.append(f"| {name} | " + " | ".join(str(a[c]) for c in cols) + " |")
    best = max(rep["systems"], key=lambda n: rep["systems"][n]["mrr@10"])
    out.append(f"\n## Misses at rank 1 ({best})\n")
    for r in rep["per_question"][best]:
        if not r["hit@1"]:
            out.append(f"- Q{r['id']}: {r['question']} — gold {r['gold']}, got {r['top5'][:3]}")
    if "grounding" in rep:
        g = rep["grounding"]
        out += ["\n## Answers + hallucination check\n",
                *(f"- {k}: {v}" for k, v in g["answers"].items()),
                "\n### Checker stress test (injected hallucinations)\n",
                f"- false positives on clean answers: {g['checker_stress_test']['false_positive_rate_on_clean']}",
                f"- overall detection: {g['checker_stress_test']['overall_detection_rate']} "
                f"of {g['checker_stress_test']['n_injected']}",
                *(f"- {k}: {v['detection_rate']} (n={v['n']})" for k, v in g["checker_stress_test"]["by_type"].items())]
    for label, gr in rep.get("golden", {}).items():
        v = gr["verification"]
        out += [f"\n## Scored against golden answers ({label})\n",
                f"Golden check vs regulation text: {sum(x['verified'] for x in v)}/{len(v)} fully verified, "
                f"{sum(x['quote_verbatim'] for x in v)}/{len(v)} quotes verbatim, "
                f"{sum(f['in_source'] for x in v for f in x['key_facts'])}/{sum(len(x['key_facts']) for x in v)} key facts found.\n"]
        cols = ["key_fact_recall", "answers_with_all_facts", "token_f1", "rouge_l", "semantic_sim",
                "semantic_sim_random_baseline", "context_fact_recall@3", "cites_gold_section"]
        out += ["| answerer | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for n, sc in gr["systems"].items():
            out.append(f"| {n} | " + " | ".join(str(sc["all_questions"][c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write(rep: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "report.md").write_text(to_markdown(rep))
