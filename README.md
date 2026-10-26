# OSHA Safety Regulations Assistant

Ask plain-language safety questions and get the relevant 29 CFR 1910 (OSHA General Industry)
section back, with the text and the citation. Source data is the public-domain eCFR.

## Pipeline
1. **Ingest** (`osha_rag/ingest.py`): download the latest Part 1910 XML from the eCFR API. Parse each
   section and appendix (appendices are linked to their parent section). Split the text into
   chunks of at most 220 words that keep paragraph boundaries, overlap by one paragraph, and carry
   the section heading.
2. **Retrieve** (`osha_rag/retrieve.py`): BM25, TF-IDF cosine, and a hybrid that merges the two with
   reciprocal rank fusion (RRF). Section headings count double, and a small synonym map turns
   plain words into regulatory terms (forklift → powered industrial truck, SDS, LOTO, PPE…).
   Results are collapsed to one hit per section.
3. **Semantic search** (`osha_rag/embed.py`): passages embedded and stored in a FAISS
   `IndexFlatIP` (cosine). Embedders: `minilm` (all-MiniLM-L6-v2) or `lsa` (TF-IDF + SVD trained on
   the corpus, no download). Also a BM25 + dense hybrid.
4. **Answers + hallucination check** (`osha_rag/grounding.py`): extractive 2-sentence answers with
   section citations. Each sentence is checked: citation exists, numbers appear in the source,
   semantic + word overlap with a source sentence, no added negation. A stress test injects
   known errors (changed numbers, negations, wrong/fake citations, foreign sentences) and
   measures how many the checker catches.
5. **Evaluate** (`osha_rag/evaluate.py`): 60 questions (`eval/questions.jsonl`) across 13 topics,
   each labeled with the section(s) that answer it. Metrics: Hit@1/3/5/10, Recall@k, MRR@10,
   nDCG@10, p50/p95 query latency, and Hit@1 per topic.

## Usage
```bash
pip install -r requirements.txt
python run.py build                 # or: python run.py build --xml path/to/title-29.xml
python run.py ask "how often must fire extinguishers be inspected?"   # add --embedder minilm
python run.py eval                  # writes results/metrics.json and results/report.md
python -m pytest tests
```
`--parts 1910 1926` adds the Construction standards. `--xml` also accepts GovInfo bulk CFR XML.

## Results (Part 1910 as of 2026-10-22, 60 questions, `lsa` embedder)

| Method | Hit@1 | Hit@10 | MRR@10 | p50 latency |
|---|---|---|---|---|
| BM25 | 0.883 | 0.983 | 0.912 | 2 ms |
| TF-IDF | 0.867 | 0.967 | 0.902 | 2 ms |
| FAISS dense (LSA) | 0.833 | 0.950 | 0.875 | 59 ms |
| BM25 + dense hybrid | 0.867 | 0.950 | 0.901 | 58 ms |

Hallucination checker: 0% false positives on clean answers; 98.5% of 273 injected errors caught.
Answers: 54/60 have at least one sentence from the correct section (32/60 both sentences).
Full details in `results/report.md` and `results/metrics.json`.

**Caveats:** answers are extractive, so the 0% flagged rate measures false alarms, not model
honesty. The "similarity to gold section" metric is inflated because answers are copied from the
corpus. There are no reference answers yet, so answer correctness is only measured at section level.
