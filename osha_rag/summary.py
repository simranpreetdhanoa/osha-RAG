"""Headline metrics from results/metrics.json: Recall@5, MRR, semantic similarity, hallucination, numbers."""
import json
import re
import statistics
from pathlib import Path

from .golden import NUM, norm


def numeric_accuracy(per_question, golden):
    """Share of the golden answer's key-fact numbers that appear in the system answer."""
    gmap = {g["id"]: g for g in golden}
    found = total = 0
    exact = []
    for r in per_question:
        nums = {n for f in gmap[r["id"]]["key_facts"] for n in NUM.findall(norm(f))}
        if not nums:
            continue
        hit = nums & set(NUM.findall(norm(r["system_answer"])))
        found, total = found + len(hit), total + len(nums)
        exact.append(hit == nums)
    return {"questions_with_numbers": len(exact), "numbers_found": f"{found}/{total}",
            "numeric_recall": round(found / total, 3), "all_numbers_correct": round(statistics.mean(exact), 3)}


def summarize(results: Path, golden_path: Path) -> dict:
    rep = json.loads((results / "metrics.json").read_text())
    golden = json.loads(golden_path.read_text())
    method, = rep["systems"]
    gold_rows = rep["golden"]["corrected"]["systems"][method]
    stress = rep["grounding"]["checker_stress_test"]
    return {
        "method": method,
        "retrieval": {"recall@5": rep["systems"][method]["recall@5"], "mrr@10": rep["systems"][method]["mrr@10"]},
        "semantic_similarity": {"to_golden_answer": gold_rows["all_questions"]["semantic_sim"],
                                "random_baseline": gold_rows["all_questions"]["semantic_sim_random_baseline"]},
        "hallucination": {"answers_flagged": rep["grounding"]["answers"]["flagged_as_hallucination"],
                          "checker_detection_rate": stress["overall_detection_rate"],
                          "checker_false_positive_rate": stress["false_positive_rate_on_clean"],
                          "n_injected": stress["n_injected"],
                          "by_type": {k: v["detection_rate"] for k, v in stress["by_type"].items()}},
        "numerical": numeric_accuracy(gold_rows["per_question"], golden),
    }


def to_markdown(s: dict) -> str:
    sem, hal, num, ret = s["semantic_similarity"], s["hallucination"], s["numerical"], s["retrieval"]
    out = ["# OSHA Assistant — headline metrics\n", f"Method: {s['method']} (corrected golden answers)\n",
            f"- Recall@5: {ret['recall@5']}", f"- MRR: {ret['mrr@10']}",
            f"- Semantic similarity to golden answer: {sem['to_golden_answer']} (random baseline {sem['random_baseline']})",
            f"- Hallucination: {hal['answers_flagged']} of answers flagged; checker caught "
            f"{hal['checker_detection_rate']} of {hal['n_injected']} injected errors, "
            f"false-positive rate {hal['checker_false_positive_rate']}",
            f"- Numerical accuracy: {num['numbers_found']} golden numbers found ({num['numeric_recall']}); "
            f"all numbers correct in {num['all_numbers_correct']} of {num['questions_with_numbers']} numeric questions"]
    return "\n".join(out) + "\n"


NAMES = {"lsa": "LSA (TF-IDF + SVD, 256-d)", "minilm": "all-MiniLM-L6-v2 (384-d)"}


def compare(runs: dict) -> str:
    """Side-by-side headline metrics for each embedding technique."""
    pct = lambda x: f"{100 * x:.1f}%"
    rows = [("Recall@5", lambda s: pct(s["retrieval"]["recall@5"])),
            ("MRR", lambda s: f"{s['retrieval']['mrr@10']:.3f}"),
            ("Semantic similarity to golden answer", lambda s: f"{s['semantic_similarity']['to_golden_answer']:.3f}"),
            ("  random-answer baseline", lambda s: f"{s['semantic_similarity']['random_baseline']:.3f}"),
            ("  margin over baseline", lambda s: f"{s['semantic_similarity']['to_golden_answer'] - s['semantic_similarity']['random_baseline']:.3f}"),
            ("Answers flagged as hallucination", lambda s: pct(s["hallucination"]["answers_flagged"])),
            ("Checker: injected errors caught", lambda s: f"{pct(s['hallucination']['checker_detection_rate'])} of {s['hallucination']['n_injected']}"),
            ("Checker: false positives", lambda s: pct(s["hallucination"]["checker_false_positive_rate"]))]
    kinds = sorted({k for s in runs.values() for k in s["hallucination"].get("by_type", {})})
    rows += [(f"  caught: {k}", lambda s, k=k: pct(s["hallucination"]["by_type"][k])) for k in kinds]
    rows += [("Numerical accuracy: golden numbers found", lambda s: f"{s['numerical']['numbers_found']} ({pct(s['numerical']['numeric_recall'])})"),
             ("Numerical: answers with every number right", lambda s: pct(s["numerical"]["all_numbers_correct"]))]
    out = ["# Embedding techniques compared\n", "FAISS semantic search, 60 questions, 5-sentence answers, corrected golden answers.\n",
           "| Metric | " + " | ".join(NAMES.get(e, e) for e in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    out += [f"| {label} | " + " | ".join(f(s) for s in runs.values()) + " |" for label, f in rows]
    return "\n".join(out) + "\n"
