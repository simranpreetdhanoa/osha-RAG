# OSHA Assistant — retrieval evaluation

Corpus: 188 sections, 3728 chunks. Questions: 60.

| system | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.8667 | 0.9167 | 0.95 | 0.9833 | 0.9033 | 0.9224 | 1.72 | 4.8 |
| tfidf | 0.8667 | 0.9333 | 0.9333 | 0.9667 | 0.9019 | 0.9175 | 1.49 | 2.73 |
| hybrid_rrf | 0.8667 | 0.9167 | 0.95 | 0.9667 | 0.9015 | 0.9174 | 3.69 | 7.04 |
| dense_lsa | 0.8667 | 0.9333 | 0.9333 | 0.95 | 0.8993 | 0.9118 | 54.39 | 75.57 |
| hybrid_bm25+dense_lsa | 0.8833 | 0.9333 | 0.95 | 0.95 | 0.9097 | 0.9199 | 57.84 | 69.54 |

## Misses at rank 1 (hybrid_bm25+dense_lsa)

- Q1: How many ways out does a workplace need in case of an emergency? — gold ['1910.36'], got ['1910.120', '1910.261', '1910.165']
- Q21: When are safety glasses with side shields or face shields required? — gold ['1910.133'], got ['1910.502', '1910.252', '1910.1030']
- Q22: When do workers have to wear hard hats? — gold ['1910.135'], got ['1910.502', '1910.134', '1910.120']
- Q23: When is protective footwear like steel toe boots required? — gold ['1910.136'], got ['1910.132', '1910.156', '1910.136']
- Q40: When is an emergency eyewash or shower required at the workplace? — gold ['1910.151'], got ['1910.1003', '1910.1025', '1910.1050']
- Q56: Ventilation and fire rules for spray painting booths — gold ['1910.107'], got ['1910.94', '1910.107', '1910.126']
- Q59: What color must be used to mark fire protection equipment and danger? — gold ['1910.144'], got ['1910.145', '1910.144', '1910.305']

## Answers + hallucination check

- n: 60
- cites_correct_section: 0.9
- semantic_sim_to_gold_section: 0.9
- flagged_as_hallucination: 0

### Checker stress test (injected hallucinations)

- false positives on clean answers: 0
- overall detection: 0.981 of 270
- foreign_sentence: 0.967 (n=60)
- wrong_citation: 0.95 (n=60)
- fake_citation: 1.0 (n=60)
- negation: 1.0 (n=48)
- number_swap: 1.0 (n=42)

## Scored against golden answers (as_given)

Golden check vs regulation text: 20/60 fully verified, 27/60 quotes verbatim, 171/180 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.331 | 0.1 | 0.264 | 0.216 | 0.71 | 0.049 | 0.686 | 0.883 |
| hybrid_bm25+dense_lsa | 0.306 | 0.1 | 0.257 | 0.208 | 0.699 | 0.049 | 0.681 | 0.9 |

## Scored against golden answers (corrected)

Golden check vs regulation text: 24/60 fully verified, 28/60 quotes verbatim, 174/178 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.322 | 0.1 | 0.269 | 0.224 | 0.711 | 0.049 | 0.667 | 0.883 |
| hybrid_bm25+dense_lsa | 0.3 | 0.1 | 0.264 | 0.221 | 0.701 | 0.05 | 0.661 | 0.9 |
