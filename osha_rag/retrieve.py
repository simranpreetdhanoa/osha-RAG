"""Lexical retrievers over regulation chunks: BM25, TF-IDF cosine, and RRF hybrid."""
import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"\d{4}\.\d+|[a-z0-9]+")
STOP = set("""a an and are as at be by can do does for from has have how i if in is it its
must of on or our should that the their them there these this to was we what when where which
who will with my me you your any shall may not no all each per such than""".split())
SYNONYMS = {  # common plain-language -> regulatory vocabulary
    "forklift": ["powered", "industrial", "truck"], "forklifts": ["powered", "industrial", "truck"],
    "hardhat": ["head", "protection", "helmet"], "earplug": ["hearing", "protector"],
    "msds": ["safety", "data", "sheet"], "sds": ["safety", "data", "sheet"],
    "lockout": ["lockout", "energy", "control"], "loto": ["lockout", "tagout", "energy", "control"],
    "osha": [], "employer": ["employer"], "worker": ["employee"], "workers": ["employee"],
    "gfci": ["ground", "fault", "circuit", "interrupter"], "ppe": ["personal", "protective", "equipment"],
}


def stem(t: str) -> str:
    if t[0].isdigit():
        return t
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return t


def tokenize(text: str, expand: bool = False) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        out.extend([t] + (SYNONYMS.get(t, []) if expand else []))
    return [stem(t) for t in out if t]


def doc_text(c: dict, heading_weight: int = 2) -> str:
    return " ".join([c["heading"]] * heading_weight + [c["text"]])


class BM25:
    name = "bm25"

    def __init__(self, chunks, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.post = defaultdict(list)
        self.len = []
        for i, c in enumerate(chunks):
            tf = Counter(tokenize(doc_text(c)))
            self.len.append(sum(tf.values()))
            for t, f in tf.items():
                self.post[t].append((i, f))
        self.n, self.avg = len(chunks), sum(self.len) / max(len(chunks), 1)
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, query):
        s = defaultdict(float)
        for t in tokenize(query, expand=True):
            for i, f in self.post.get(t, ()):
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class TFIDF:
    name = "tfidf"

    def __init__(self, chunks):
        df, tfs = Counter(), []
        for c in chunks:
            tf = Counter(tokenize(doc_text(c)))
            tfs.append(tf)
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.post = defaultdict(list)
        for i, tf in enumerate(tfs):
            w = {t: (1 + math.log(f)) * self.idf[t] for t, f in tf.items()}
            norm = math.sqrt(sum(v * v for v in w.values())) or 1
            for t, v in w.items():
                self.post[t].append((i, v / norm))

    def scores(self, query):
        q = Counter(tokenize(query, expand=True))
        s = defaultdict(float)
        for t, f in q.items():
            if t in self.idf:
                qw = (1 + math.log(f)) * self.idf[t]
                for i, v in self.post[t]:
                    s[i] += qw * v
        return s


class Hybrid:
    """Reciprocal-rank fusion of BM25 and TF-IDF."""
    name = "hybrid_rrf"

    def __init__(self, chunks, k=60):
        self.parts, self.k = [BM25(chunks), TFIDF(chunks)], k

    def scores(self, query):
        s = defaultdict(float)
        for r in self.parts:
            ranked = sorted(r.scores(query).items(), key=lambda x: -x[1])[:200]
            for rank, (i, _) in enumerate(ranked):
                s[i] += 1 / (self.k + rank + 1)
        return s


def search(retriever, chunks, query, k=10):
    """Top-k chunks, then collapse to distinct sections (best chunk per section)."""
    ranked = sorted(retriever.scores(query).items(), key=lambda x: -x[1])
    hits, seen = [], set()
    for i, sc in ranked:
        sec = chunks[i]["section"]
        if sec in seen:
            continue
        seen.add(sec)
        hits.append({**chunks[i], "score": sc})
        if len(hits) == k:
            break
    return hits
