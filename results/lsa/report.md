# OSHA Assistant — retrieval evaluation

Corpus: 188 sections, 3728 chunks. Questions: 60.

| system | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| dense_lsa | 0.8667 | 0.9333 | 0.9333 | 0.95 | 0.8993 | 0.9118 | 50.58 | 69.42 |

## Misses at rank 1 (dense_lsa)

- Q1: How many ways out does a workplace need in case of an emergency? — gold ['1910.36'], got ['1910.165', '1910.38', '1910.120']
- Q21: When are safety glasses with side shields or face shields required? — gold ['1910.133'], got ['1910.502', '1910.133', '1910.252']
- Q22: When do workers have to wear hard hats? — gold ['1910.135'], got ['1910.502', '1910.134', '1910.1027']
- Q23: When is protective footwear like steel toe boots required? — gold ['1910.136'], got ['1910.132', '1910.335', '1910.136']
- Q24: Requirements for gloves to protect hands from cuts, chemicals and burns — gold ['1910.138'], got ['1910.1200', '1910.138', '1910.335']
- Q40: When is an emergency eyewash or shower required at the workplace? — gold ['1910.151'], got ['1910.1003', '1910.1044', '1910.1029']
- Q56: Ventilation and fire rules for spray painting booths — gold ['1910.107'], got ['1910.94', '1910.107', '1910.252']
- Q59: What color must be used to mark fire protection equipment and danger? — gold ['1910.144'], got ['1910.145', '1910.156', '1910.252']

## Answers + hallucination check

- n: 60
- cites_correct_section: 0.883
- semantic_sim_to_gold_section: 0.858
- flagged_as_hallucination: 0

### Checker stress test (injected hallucinations)

- false positives on clean answers: 0
- overall detection: 0.986 of 276
- number_swap: 1.0 (n=51)
- foreign_sentence: 0.967 (n=60)
- wrong_citation: 0.967 (n=60)
- fake_citation: 1.0 (n=60)
- negation: 1.0 (n=45)

## Scored against golden answers (as_given)

Golden check vs regulation text: 20/60 fully verified, 27/60 quotes verbatim, 171/180 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| dense_lsa | 0.289 | 0.1 | 0.167 | 0.135 | 0.662 | 0.054 | 0.586 | 0.883 |

## Scored against golden answers (corrected)

Golden check vs regulation text: 24/60 fully verified, 28/60 quotes verbatim, 174/178 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| dense_lsa | 0.283 | 0.1 | 0.171 | 0.138 | 0.664 | 0.055 | 0.581 | 0.883 |
