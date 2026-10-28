# Embedding techniques compared

FAISS semantic search, 60 questions, 5-sentence answers, corrected golden answers.

| Metric | LSA (TF-IDF + SVD, 256-d) | all-MiniLM-L6-v2 (384-d) |
|---|---|---|
| Recall@5 | 93.3% | 100.0% |
| MRR | 0.899 | 0.923 |
| Semantic similarity to golden answer | 0.664 | 0.730 |
|   random-answer baseline | 0.055 | 0.209 |
|   margin over baseline | 0.609 | 0.521 |
| Answers flagged as hallucination | 0.0% | 0.0% |
| Checker: injected errors caught | 98.6% of 276 | 96.5% of 289 |
| Checker: false positives | 0.0% | 0.0% |
|   caught: fake_citation | 100.0% | 100.0% |
|   caught: foreign_sentence | 96.7% | 93.3% |
|   caught: negation | 100.0% | 100.0% |
|   caught: number_swap | 100.0% | 100.0% |
|   caught: wrong_citation | 96.7% | 90.0% |
| Numerical accuracy: golden numbers found | 19/69 (27.5%) | 47/69 (68.1%) |
| Numerical: answers with every number right | 7.7% | 38.5% |
