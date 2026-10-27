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
| BM25 | 0.867 | 0.983 | 0.903 | 2 ms |
| TF-IDF | 0.867 | 0.967 | 0.902 | 1 ms |
| FAISS dense (LSA) | 0.867 | 0.950 | 0.899 | 54 ms |
| BM25 + dense hybrid | 0.883 | 0.950 | 0.910 | 58 ms |

**Against golden answers** (`eval/golden_answers_corrected.json`, BM25 answerer): key-fact recall 0.32,
6/60 answers contain every key fact, token F1 0.27, ROUGE-L 0.22, semantic similarity 0.71
(0.05 for a random golden answer), and 67% of key facts appear in the top-3 retrieved passages.

**Golden answers checked against the regulation:** the model-written golden answers had 3 factual
errors (Q41, Q47, Q55) and 4 partial errors (Q5, Q22, Q33, Q50); 33 of 60 "verbatim" quotes are not
in the text. `golden_answers_corrected.json` fixes them; each change has a `review` note.

**Hallucination checker:** 0% false positives on clean answers; 98.1% of 270 injected errors caught.

**Caveats:** answers are extractive, so the 0% flagged rate measures false alarms, not model
honesty. Key-fact matching is lenient (all numbers + 60% of words).
