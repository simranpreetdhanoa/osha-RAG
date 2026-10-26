"""Dense retrieval: small embedding models + a FAISS vector index.

Embedders
  minilm : sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-d). Needs huggingface.co access.
  lsa    : TF-IDF -> TruncatedSVD (256-d) trained on the OSHA corpus itself. No download needed.
"""
from pathlib import Path

import faiss
import numpy as np

from .retrieve import BM25, doc_text, tokenize


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim=256, seed=0):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(tokenizer=lambda t: tokenize(t, expand=True), lowercase=False,
                                   token_pattern=None, sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        self.svd = TruncatedSVD(self.dim, random_state=self.seed)
        self.svd.fit(self.vec.fit_transform(texts))
        return self

    def encode(self, texts):
        return _norm(self.svd.transform(self.vec.transform(texts)))


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return _norm(self.model.encode(list(texts), batch_size=64, show_progress_bar=False))


EMBEDDERS = {"lsa": LSAEmbedder, "minilm": MiniLMEmbedder}


class Dense:
    """Cosine search over chunk embeddings with a FAISS inner-product index."""

    def __init__(self, chunks, embedder="lsa", cache_dir: Path | None = None, depth=200):
        self.emb = EMBEDDERS[embedder]() if isinstance(embedder, str) else embedder
        self.name, self.depth = f"dense_{self.emb.name}", depth
        texts = [doc_text(c, heading_weight=1) for c in chunks]
        self.emb.fit(texts)
        path = cache_dir / f"faiss_{self.emb.name}.index" if cache_dir else None
        if path and path.exists() and faiss.read_index(str(path)).ntotal == len(chunks) and self.emb.name != "lsa":
            self.index = faiss.read_index(str(path))
        else:
            vecs = self.emb.encode(texts)
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            if path:
                faiss.write_index(self.index, str(path))

    def scores(self, query):
        d, i = self.index.search(self.emb.encode([query]), self.depth)
        return {int(j): float(s) for j, s in zip(i[0], d[0]) if j >= 0}


class HybridDense:
    """Reciprocal-rank fusion of BM25 (keywords) and the dense retriever (meaning)."""

    def __init__(self, chunks, dense: Dense, k=60):
        self.parts, self.k, self.name = [BM25(chunks), dense], k, f"hybrid_bm25+{dense.name}"

    def scores(self, query):
        s = {}
        for r in self.parts:
            for rank, (i, _) in enumerate(sorted(r.scores(query).items(), key=lambda x: -x[1])[:200]):
                s[i] = s.get(i, 0.0) + 1 / (self.k + rank + 1)
        return s
