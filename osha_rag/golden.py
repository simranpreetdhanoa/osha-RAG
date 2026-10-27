"""Golden-answer evaluation.

1. verify_golden: check every golden answer against the downloaded regulation text
   (is the quote really there? are the key facts and numbers in the cited section? does the
   cited paragraph exist?). Golden answers come from another model and can be wrong.
2. score_answers: compare the system's answers with the golden answers (key-fact recall,
   token F1, ROUGE-L, semantic similarity, evidence recall of the golden quote).
"""
import random
import re
import statistics

import numpy as np

from .retrieve import search, tokenize

NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "twenty": "20",
            "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "one-eighth": "1/8",
            "one-half": "1/2", "one-fourth": "1/4", "one-quarter": "1/4"}
NUM = re.compile(r"\d+(?:\.\d+)?")
DESIG = re.compile(r"\(([a-z]+|\d+)\)", re.I)


def norm(t: str) -> str:
    t = re.sub(r"\b\d{4}\.\d+\b", " ", t.lower())  # drop section references like 1910.37
    for a, b in (("“", '"'), ("”", '"'), ("’", "'"), ("—", " "), ("–", "-"), ("μ", "µ"), ("³", "3"),
                 ("…", " ... ")):
        t = t.replace(a, b)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"\b(" + "|".join(sorted(NUMWORDS, key=len, reverse=True)) + r")\b",
               lambda m: NUMWORDS[m.group(1)], t)
    return re.sub(r"\s+", " ", t)


def words(t):
    return re.findall(r"[a-z0-9/.]+", norm(t))


def quote_coverage(quote: str, text_ngrams: set, n=3) -> float:
    """Share of the quote's word 3-grams found in the source (each '...' segment separately)."""
    segs = [words(s) for s in re.split(r"\.\.\.|…", quote)]
    grams = [tuple(s[i:i + n]) for s in segs if len(s) >= n for i in range(len(s) - n + 1)]
    grams += [tuple(s) for s in segs if 0 < len(s) < n]
    return sum(g in text_ngrams for g in grams) / max(len(grams), 1)


def ngrams(text, n=3):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} | {tuple(w[i:i + k]) for k in (1, 2) for i in range(len(w) - k + 1)}


def fact_in(fact: str, text_norm: str, text_tokens: set) -> bool:
    """A fact matches if all its numbers appear and >= 60% of its content words appear."""
    nums = set(NUM.findall(norm(fact)))
    if nums - set(NUM.findall(text_norm)):
        return False
    toks = set(tokenize(norm(fact)))
    return not toks or len(toks & text_tokens) / len(toks) >= 0.6


def paragraph_exists(citation: str, section_text: str) -> bool | None:
    """Approximate: the first cited designators, e.g. (e)(2), appear in order as paragraph markers."""
    m = re.search(r"\d{4}\.\d+((?:\([a-z0-9]+\))+)", citation, re.I)
    if not m:
        return None
    pos = 0
    for d in DESIG.findall(m.group(1)):
        hit = re.compile(r"(?:^|\s|\)|—)\(" + re.escape(d) + r"\)\s").search(section_text, pos)
        if not hit:
            return False
        pos = hit.end()
    return True


def verify_golden(golden: list[dict], questions: list[dict], src_text: dict) -> list[dict]:
    qmap = {q["id"]: q for q in questions}
    out = []
    for g in golden:
        sec = qmap[g["id"]]["gold"][0]
        text = src_text.get(sec, "")
        tn, tt, tg = norm(text), set(tokenize(norm(text))), ngrams(text)
        qcov = quote_coverage(g["quote"], tg)
        facts = [{"fact": f, "in_source": fact_in(f, tn, tt)} for f in g["key_facts"]]
        bad_nums = sorted(set(NUM.findall(norm(g["answer"]))) - set(NUM.findall(tn)))
        cited_sec = re.search(r"\d{4}\.\d+", g["citation"])
        out.append({
            "id": g["id"], "section": sec,
            "citation_section_matches": bool(cited_sec and cited_sec.group() == sec),
            "paragraph_found": paragraph_exists(g["citation"], text),
            "quote_coverage": round(qcov, 3),
            "quote_verbatim": qcov >= 0.9,
            "answer_numbers_not_in_source": bad_nums,
            "key_facts": facts,
            "verified": qcov >= 0.9 and not bad_nums and all(f["in_source"] for f in facts),
        })
    return out


# ---------- scoring the system against golden answers ----------
def token_f1(pred, ref):
    p, r = tokenize(norm(pred)), tokenize(norm(ref))
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred, ref):
    p, r = tokenize(norm(pred)), tokenize(norm(ref))
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 0.0 if not lcs else 2 * prec * rec / (prec + rec)


def score_answers(g, golden, verification, questions, seed=0):
    """g is a grounding.Grounding (answerer + embedder)."""
    vmap = {v["id"]: v for v in verification}
    gmap = {x["id"]: x for x in golden}
    rng = random.Random(seed)
    rows = []
    for q in questions:
        gold = gmap[q["id"]]
        ans = g.answer(q["question"])
        pred = " ".join(c["text"] for c in ans["claims"])
        pn, pt = norm(pred), set(tokenize(norm(pred)))
        facts = [f["fact"] for f in vmap[q["id"]]["key_facts"] if f["in_source"]]  # verified facts only
        hit_facts = [f for f in facts if fact_in(f, pn, pt)]
        ctx = " ".join(h["text"] for h in search(g.r, g.chunks, q["question"], k=3))
        e = g.emb.encode([pred, gold["answer"], gmap[rng.choice([i for i in gmap if i != q["id"]])]["answer"]])
        rows.append({
            "id": q["id"], "question": q["question"], "system_answer": pred,
            "system_cites": [c["cite"] for c in ans["claims"]], "golden_answer": gold["answer"],
            "golden_verified": vmap[q["id"]]["verified"],
            "key_fact_recall": len(hit_facts) / len(facts) if facts else None,
            "facts_found": hit_facts, "facts_missed": [f for f in facts if f not in hit_facts],
            "token_f1": token_f1(pred, gold["answer"]), "rouge_l": rouge_l(pred, gold["answer"]),
            "semantic_sim": float(e[0] @ e[1]), "semantic_sim_random_golden": float(e[0] @ e[2]),
            "context_fact_recall@3": (sum(fact_in(f, norm(ctx), set(tokenize(norm(ctx)))) for f in facts) / len(facts)
                                      if facts else None),
            "cites_gold_section": any(c["cite"] in q["gold"] for c in ans["claims"]),
        })

    def agg(rs):
        m = lambda k: round(statistics.mean(r[k] for r in rs if r[k] is not None), 3)
        return {"n": len(rs), "key_fact_recall": m("key_fact_recall"),
                "answers_with_all_facts": round(statistics.mean(r["key_fact_recall"] == 1 for r in rs if r["key_fact_recall"] is not None), 3),
                "token_f1": m("token_f1"), "rouge_l": m("rouge_l"), "semantic_sim": m("semantic_sim"),
                "semantic_sim_random_baseline": m("semantic_sim_random_golden"),
                "context_fact_recall@3": m("context_fact_recall@3"), "cites_gold_section": m("cites_gold_section")}

    return {"all_questions": agg(rows),
            "verified_golden_only": agg([r for r in rows if r["golden_verified"]]),
            "per_question": rows}
