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


ONNX_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
ONNX_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"  # as pinned by chromadb
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def fetch_onnx_minilm(model_dir: Path = MODEL_DIR) -> Path:
    """Download the ONNX export of all-MiniLM-L6-v2 once and verify its checksum."""
    import hashlib
    import tarfile

    import requests
    onnx_dir = model_dir / "onnx"
    if (onnx_dir / "model.onnx").exists():
        return onnx_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / "onnx.tar.gz"
    archive.write_bytes(requests.get(ONNX_URL, timeout=600).content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ONNX_SHA256:
        archive.unlink()
        raise RuntimeError("all-MiniLM-L6-v2 download failed checksum verification")
    with tarfile.open(archive) as tar:
        tar.extractall(model_dir, filter="data")
    archive.unlink()
    return onnx_dir


class MiniLMEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2 (384-d): mean pooling over tokens, then L2 normalization.

    Uses the sentence-transformers library when installed; otherwise runs the same model's ONNX export
    with onnxruntime (no PyTorch needed).
    """
    name = "minilm"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", max_tokens=256, batch_size=32):
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(model)
        except ImportError:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self.st = None
            onnx_dir = fetch_onnx_minilm()
            self.tok = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
            self.tok.enable_truncation(max_tokens)
            self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
            self.sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    def fit(self, texts):
        return self

    def encode(self, texts):
        texts = list(texts)
        if self.st is not None:
            return _norm(self.st.encode(texts, batch_size=self.batch_size, show_progress_bar=False))
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch(texts[i:i + self.batch_size])
            ids = np.array([e.ids for e in enc], dtype="int64")
            mask = np.array([e.attention_mask for e in enc], dtype="int64")
            hidden = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                          "token_type_ids": np.zeros_like(ids)})[0]
            m = mask[..., None].astype("float32")
            out.append((hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9))
        return _norm(np.vstack(out))


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
