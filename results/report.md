# OSHA Assistant — retrieval evaluation

Corpus: 188 sections, 3501 chunks. Questions: 60.

| system | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.8833 | 0.9167 | 0.95 | 0.9833 | 0.9123 | 0.929 | 1.82 | 3.04 |
| tfidf | 0.8667 | 0.9333 | 0.95 | 0.9667 | 0.9024 | 0.918 | 1.86 | 3.22 |
| hybrid_rrf | 0.8833 | 0.9167 | 0.95 | 0.9667 | 0.9107 | 0.9243 | 3.99 | 7.27 |
| dense_lsa | 0.8333 | 0.9 | 0.9333 | 0.95 | 0.875 | 0.8935 | 58.66 | 70.89 |
| hybrid_bm25+dense_lsa | 0.8667 | 0.9333 | 0.95 | 0.95 | 0.9014 | 0.9137 | 58.21 | 70.88 |

## Misses at rank 1 (bm25)

- Q1: How many ways out does a workplace need in case of an emergency? — gold ['1910.36'], got ['1910.1200', '1910.1048', '1910.261']
- Q10: At what height do workers on an unprotected edge need fall protection in general industry? — gold ['1910.28'], got ['1910.29', '1910.28', '1910.21']
- Q21: When are safety glasses with side shields or face shields required? — gold ['1910.133'], got ['1910.1030', '1910.252', '1910.502']
- Q22: When do workers have to wear hard hats? — gold ['1910.135'], got ['1910.502', '1910.120', '1910.262']
- Q23: When is protective footwear like steel toe boots required? — gold ['1910.136'], got ['1910.132', '1910.156', '1910.120']
- Q40: When is an emergency eyewash or shower required at the workplace? — gold ['1910.151'], got ['1910.1003', '1910.1048', '1910.1050']
- Q56: Ventilation and fire rules for spray painting booths — gold ['1910.107'], got ['1910.94', '1910.107', '1910.126']

## Answers + hallucination check

- n: 60
- cites_correct_section: 0.9
- semantic_sim_to_gold_section: 0.907
- flagged_as_hallucination: 0

### Checker stress test (injected hallucinations)

- false positives on clean answers: 0
- overall detection: 0.985 of 273
- foreign_sentence: 0.95 (n=60)
- wrong_citation: 0.983 (n=60)
- fake_citation: 1.0 (n=60)
- negation: 1.0 (n=49)
- number_swap: 1.0 (n=44)
