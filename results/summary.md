# OSHA Assistant — headline metrics

## Retrieval

| method | Recall@5 | MRR |
|---|---|---|
| bm25 | 0.95 | 0.9033 |
| tfidf | 0.9333 | 0.9019 |
| hybrid_rrf | 0.95 | 0.9015 |
| dense_lsa | 0.9333 | 0.8993 |
| hybrid_bm25+dense_lsa | 0.95 | 0.9097 |

## Answers (BM25 answerer, corrected golden answers)

- Semantic similarity to golden answer: 0.704 (random baseline 0.053)
- Hallucination: 0 of answers flagged; checker caught 0.982 of 281 injected errors, false-positive rate 0
- Numerical accuracy: 41/69 golden numbers found (0.594); all numbers correct in 0.462 of 26 numeric questions
