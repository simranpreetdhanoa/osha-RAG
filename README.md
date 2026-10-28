# OSHA Safety Regulations Assistant

Ask plain-language safety questions and get the relevant 29 CFR 1910 (OSHA General Industry)
section back, with the text and the citation. Source data is the public-domain eCFR.

## Pipeline
1. **Ingest** (`osha_rag/ingest.py`): download the latest Part 1910 XML from the eCFR API. Parse each
   section and appendix (appendices are linked to their parent section). Split the text into
   chunks of at most 220 words that keep paragraph boundaries, overlap by one paragraph, and carry
   the section heading.
2. **Embed + index** (`osha_rag/embed.py`): every passage becomes a 384-dimension vector with
   **all-MiniLM-L6-v2** (mean pooling + L2 normalization) and is stored in a FAISS `IndexFlatIP`
   (cosine similarity). Uses `sentence-transformers` if installed; otherwise runs the model's ONNX
   export with `onnxruntime`, downloaded on first use and checked against a pinned SHA-256.
   `--embedder lsa` uses a corpus-trained LSA model instead.
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

## Usage
```bash
pip install -r requirements.txt
python run.py build                 # or: python run.py build --xml path/to/title-29.xml
python run.py ask "how often must fire extinguishers be inspected?"
python run.py eval                  # writes results/metrics.json, report.md, summary.md
python run.py summary               # headline metrics only
python -m pytest tests
```
`--parts 1910 1926` adds the Construction standards. Chunks overlap by one paragraph (MRR 0.899 vs 0.856 with no overlap). `--xml` also accepts GovInfo bulk CFR XML.

## Results (Part 1910 as of 2026-10-22, 60 questions)

Method: **FAISS semantic search with all-MiniLM-L6-v2**, 5-sentence answers, corrected golden answers.
`python run.py summary` prints these after `python run.py eval`.

| Metric | all-MiniLM-L6-v2 | LSA (previous) |
|---|---|---|
| Recall@5 | **1.000** | 0.933 |
| MRR | **0.923** | 0.899 |
| Semantic similarity to golden answer | **0.730** (random baseline 0.209) | 0.664 (random 0.055) |
| Answers flagged as hallucination | 0% | 0% |
| Hallucination checker: injected errors caught | 96.5% of 289 (0% false positives) | 98.6% of 276 |
| Numerical accuracy: golden numbers found | **47/69 (68.1%)** | 19/69 (27.5%) |

Similarity scores from the two models are not on the same scale; compare each to its own random baseline.

**Golden answers checked against the regulation:** the model-written golden answers had 3 factual
errors (Q41, Q47, Q55) and 4 partial errors (Q5, Q22, Q33, Q50); 33 of 60 "verbatim" quotes are not
in the text. `golden_answers_corrected.json` fixes them; each change has a `review` note.

**Caveats:** answers are extractive, so the 0% flagged rate measures false alarms, not model
honesty. Key-fact matching is lenient (all numbers + 60% of words).
