"""Answer generation (extractive, cited) + hallucination / grounding checker + checker stress test.

A sentence in an answer counts as GROUNDED in its cited section only if all four checks pass:
  1. citation  - the cited section exists in the corpus
  2. numbers   - every number in the sentence appears in the cited section
  3. support   - semantic similarity to the closest source sentence >= SIM_MIN
                 and >= CONTAIN_MIN of its content words appear in the cited section
  4. negation  - the sentence has no more negations (not/no/never) than its closest source sentence
"""
import random
import re
import statistics
from collections import defaultdict

import numpy as np

from .retrieve import search, tokenize

SENT = re.compile(r"(?<=[.;:])\s+(?=\(?[A-Z0-9(])")
NUM = re.compile(r"\d+(?:\.\d+)?")
NEG = re.compile(r"\b(not|no|never|neither|nor)\b", re.I)
SIM_MIN, CONTAIN_MIN = 0.5, 0.75


def sentences(text):
    return [s.strip() for s in SENT.split(text) if len(s.split()) >= 5]


class Grounding:
    def __init__(self, chunks, retriever, embedder, n_sentences=5):
        self.chunks, self.r, self.emb, self.n_sentences = chunks, retriever, embedder, n_sentences
        self.src_text = defaultdict(str)
        for c in chunks:
            self.src_text[c["section"]] += " " + c["text"]
        self.src_sents = {s: sentences(t) for s, t in self.src_text.items()}
        self.src_tokens = {s: set(tokenize(t)) for s, t in self.src_text.items()}
        self.src_nums = {s: set(NUM.findall(t)) for s, t in self.src_text.items()}
        self._vec_cache = {}

    def _src_vecs(self, sec):
        if sec not in self._vec_cache:
            self._vec_cache[sec] = self.emb.encode(self.src_sents[sec] or [self.src_text[sec]])
        return self._vec_cache[sec]

    # ---------- answering ----------
    def answer(self, question, n_sections=3, n_sentences=None):
        n_sentences = n_sentences or self.n_sentences
        """Pick the source sentences most similar to the question; each keeps its citation."""
        hits = search(self.r, self.chunks, question, k=n_sections)
        cands = [(h["section"], s, rank) for rank, h in enumerate(hits) for s in sentences(h["text"])]
        if not cands:
            return {"question": question, "claims": []}
        q = self.emb.encode([question])[0]
        sims = self.emb.encode([c[1] for c in cands]) @ q
        q_tok = set(tokenize(question, expand=True))
        scored = [(sims[i] + 0.5 * len(q_tok & set(tokenize(s))) / max(len(q_tok), 1) - 0.05 * rank, sec, s)
                  for i, (sec, s, rank) in enumerate(cands)]
        top = sorted(scored, key=lambda x: -x[0])[:n_sentences]
        return {"question": question, "claims": [{"text": s, "cite": sec} for _, sec, s in top]}

    # ---------- hallucination check ----------
    def check_claim(self, text, cite):
        if cite not in self.src_text:
            return {"grounded": False, "reason": "citation_not_found", "sim": 0.0}
        missing = set(NUM.findall(text)) - self.src_nums[cite]
        toks = set(tokenize(text))
        contain = len(toks & self.src_tokens[cite]) / max(len(toks), 1)
        sims = self._src_vecs(cite) @ self.emb.encode([text])[0]
        best = int(np.argmax(sims))
        sim = float(sims[best])
        src_best = (self.src_sents[cite] or [self.src_text[cite]])[best]
        neg_added = len(NEG.findall(text)) > len(NEG.findall(src_best))
        reason = ("number_not_in_source" if missing else "low_support" if sim < SIM_MIN or contain < CONTAIN_MIN
                  else "negation_added" if neg_added else None)
        return {"grounded": reason is None, "reason": reason, "sim": round(sim, 3),
                "contain": round(contain, 3), "missing_numbers": sorted(missing)}

    def check(self, ans):
        res = [self.check_claim(c["text"], c["cite"]) for c in ans["claims"]]
        return {"hallucination": any(not r["grounded"] for r in res), "claims": res}


# ---------- stress test: inject known hallucinations, measure what the checker catches ----------
def perturb(ans, g: Grounding, kind, rng):
    claims = [dict(c) for c in ans["claims"]]
    if not claims:
        return None
    i = 0
    if kind == "number_swap":
        idx = [j for j, c in enumerate(claims) if NUM.search(c["text"])]
        if not idx:
            return None
        i = idx[0]
        n = NUM.search(claims[i]["text"]).group()
        new = next(str(int(float(n)) + d) for d in (7, 13, 29, 101, 997)
                   if str(int(float(n)) + d) not in g.src_nums[claims[i]["cite"]])
        claims[i]["text"] = claims[i]["text"].replace(n, new, 1)
    elif kind == "foreign_sentence":
        other = rng.choice([s for s in g.src_sents if s != claims[i]["cite"] and g.src_sents[s]])
        claims[i]["text"] = rng.choice(g.src_sents[other])
    elif kind == "wrong_citation":
        claims[i]["cite"] = rng.choice([s for s in g.src_text if s != claims[i]["cite"]])
    elif kind == "fake_citation":
        claims[i]["cite"] = "1910.9999"
    elif kind == "negation":
        t = claims[i]["text"]
        for a, b in ((" shall ", " shall not "), (" must ", " must not "), (" is ", " is not "),
                     (" are ", " are not "), (" may ", " may not ")):
            if a in t:
                claims[i]["text"] = t.replace(a, b, 1)
                break
        else:
            return None
    return {**ans, "claims": claims}


KINDS = ("number_swap", "foreign_sentence", "wrong_citation", "fake_citation", "negation")


def evaluate_grounding(g: Grounding, questions, seed=0):
    rng = random.Random(seed)
    rows, caught, sims_gold, cites_gold = [], defaultdict(list), [], []
    for q in questions:
        ans = g.answer(q["question"])
        chk = g.check(ans)
        cites_gold.append(float(any(c["cite"] in q["gold"] for c in ans["claims"])))
        # semantic similarity of the answer to the correct (gold) section text
        a_vec = g.emb.encode([" ".join(c["text"] for c in ans["claims"])])[0]
        sims_gold.append(max(float(np.max(g._src_vecs(s) @ a_vec)) for s in q["gold"] if s in g.src_text))
        rows.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "answer": ans["claims"],
                     "flagged": chk["hallucination"], "checks": chk["claims"]})
        for k in KINDS:
            bad = perturb(ans, g, k, rng)
            if bad is not None:
                caught[k].append(float(g.check(bad)["hallucination"]))
    clean_fp = statistics.mean(r["flagged"] for r in rows)
    det = {k: {"n": len(v), "detection_rate": round(statistics.mean(v), 3)} for k, v in caught.items()}
    all_bad = [x for v in caught.values() for x in v]
    return {
        "answers": {"n": len(rows),
                    "cites_correct_section": round(statistics.mean(cites_gold), 3),
                    "semantic_sim_to_gold_section": round(statistics.mean(sims_gold), 3),
                    "flagged_as_hallucination": round(clean_fp, 3)},
        "checker_stress_test": {"false_positive_rate_on_clean": round(clean_fp, 3),
                                "overall_detection_rate": round(statistics.mean(all_bad), 3),
                                "n_injected": len(all_bad), "by_type": det},
        "thresholds": {"sim_min": SIM_MIN, "contain_min": CONTAIN_MIN},
        "per_question": rows,
    }
