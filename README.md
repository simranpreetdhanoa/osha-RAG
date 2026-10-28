# OSHA Safety Regulations Assistant

Ask plain-language safety questions and get the relevant 29 CFR 1910 (OSHA General Industry)
section back, with the text and the citation. Source data is the public-domain eCFR.

## Pipeline
1. **Ingest** (`osha_rag/ingest.py`): download the latest Part 1910 XML from the eCFR API. Parse each
   section and appendix (appendices are linked to their parent section). Split the text into
   chunks of at most 220 words that keep paragraph boundaries, overlap by one paragraph, and carry
   the section heading.
2. **Embed + index** (`osha_rag/embed.py`): every passage becomes a vector using one of two embedding
   techniques (below) and is stored in a FAISS `IndexFlatIP` (cosine similarity).
3. **Search**: the question is embedded the same way; FAISS returns the closest passages, collapsed to
   one hit per section.
4. **Answer + hallucination check** (`osha_rag/grounding.py`): the 5 sentences closest to the question
   (from the top 3 sections) are returned with their section citations. Each sentence is checked:
   the citation exists, every number appears in the source, it matches a source sentence in meaning
   and wording, and it adds no negation. A stress test plants known errors and measures how many
   the checker catches.
5. **Evaluate** (`osha_rag/evaluate.py`, `golden.py`, `summary.py`): 60 questions
   (`eval/questions.jsonl`) across 13 topics, each labeled with its correct section, plus golden
   answers checked against the regulation text. Metrics: Recall@5, MRR, semantic similarity to the
   golden answer, hallucination detection, numerical accuracy.

## Embedding techniques
| | **all-MiniLM-L6-v2** (default, `--embedder minilm`) | **LSA** (`--embedder lsa`) |
|---|---|---|
| What it is | Pre-trained neural sentence-embedding model (6-layer transformer, 22M parameters) | Latent Semantic Analysis: TF-IDF word counts compressed with truncated SVD |
| Trained on | ~1 billion general English sentence pairs | Only the OSHA corpus itself |
| Vector size | 384 | 256 |
| How a text becomes a vector | Token vectors from the transformer, mean-pooled, L2-normalized | TF-IDF weights (1-2 word terms) projected onto 256 SVD topics, L2-normalized |
| Knows everyday synonyms (hard hat = head protection) | Yes, from pre-training | Only if the words co-occur in the regulations |
| Setup | Downloads a 90 MB model once (SHA-256 checked); runs with `sentence-transformers` or `onnxruntime` | Nothing to download; fits in seconds |

## Usage
```bash
pip install -r requirements.txt
python run.py build                 # or: python run.py build --xml path/to/title-29.xml
python run.py ask "how often must fire extinguishers be inspected?"
python run.py eval                  # MiniLM -> results/minilm/ (metrics.json, report.md, summary.md)
python run.py eval --embedder lsa   # LSA    -> results/lsa/
python run.py compare               # results/comparison.md
python -m pytest tests
```
`--parts 1910 1926` adds the Construction standards. Chunks overlap by one paragraph (MRR 0.899 vs 0.856 with no overlap). `--xml` also accepts GovInfo bulk CFR XML.

## Results (Part 1910 as of 2026-10-22, 60 questions)

FAISS semantic search, 5-sentence answers, corrected golden answers. Full table: `results/comparison.md`.

| Metric | all-MiniLM-L6-v2 | LSA |
|---|---|---|
| Recall@5 | **100.0%** | 93.3% |
| MRR | **0.923** | 0.899 |
| Semantic similarity to golden answer | 0.730 (random baseline 0.209) | 0.664 (random baseline 0.055) |
| Answers flagged as hallucination | 0% | 0% |
| Hallucination checker: injected errors caught (0% false positives) | 96.5% of 289 | **98.6%** of 276 |
| Numerical accuracy: golden numbers found | **47/69 (68.1%)** | 19/69 (27.5%) |

**What differs:**
- **Retrieval:** MiniLM puts the correct section in the top 5 for every question (LSA misses 4). It fixes
  everyday-language questions LSA could not match: "hard hats" (1910.135), "ways out" (exit routes,
  1910.36), "color for fire protection" (1910.144), "eyewash" (1910.151). Both rank the right section
  first for 52/60; MiniLM drops four questions from rank 1 to rank 2-3.
- **Answers:** MiniLM picks more precise sentences, so answers contain 68% of the needed numbers vs 27.5%.
- **Semantic similarity:** the scales differ. MiniLM rates all regulatory text as somewhat similar
  (random baseline 0.209), so its margin over the baseline (0.52) is smaller than LSA's (0.61).
- **Hallucination checker:** its 0.5 similarity threshold was set with LSA. Under MiniLM, a few unrelated
  sentences clear it, so detection of foreign sentences (93.3% vs 96.7%) and wrong citations
  (90.0% vs 96.7%) is lower. Numbers, negations and fake citations are caught 100% with both.

**Golden answers checked against the regulation:** the model-written golden answers had 3 factual
errors (Q41, Q47, Q55) and 4 partial errors (Q5, Q22, Q33, Q50); 33 of 60 "verbatim" quotes are not
in the text. `golden_answers_corrected.json` fixes them; each change has a `review` note.

**Caveats:** answers are extractive, so the 0% flagged rate measures false alarms, not model
honesty. Key-fact matching is lenient (all numbers + 60% of words).
