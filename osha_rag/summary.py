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
                          "n_injected": stress["n_injected"]},
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
