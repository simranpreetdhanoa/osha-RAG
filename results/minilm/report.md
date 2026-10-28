# OSHA Assistant — retrieval evaluation

Corpus: 188 sections, 3728 chunks. Questions: 60.

| system | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@10 | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| dense_minilm | 0.8667 | 0.9833 | 1.0 | 1.0 | 0.9228 | 0.9424 | 3.56 | 7.51 |

## Misses at rank 1 (dense_minilm)

- Q10: At what height do workers on an unprotected edge need fall protection in general industry? — gold ['1910.28'], got ['1910.29', '1910.28', '1910.269']
- Q14: Rules for using rope descent systems to clean building windows — gold ['1910.27'], got ['1910.140', '1910.27', '1910.66']
- Q18: How often do employees need to be fit tested for a tight-fitting respirator? — gold ['1910.134'], got ['1910.1048', '1910.134', '1910.504']
- Q22: When do workers have to wear hard hats? — gold ['1910.135'], got ['1910.269', '1910.1029', '1910.135']
- Q24: Requirements for gloves to protect hands from cuts, chemicals and burns — gold ['1910.138'], got ['1910.156', '1910.138', '1910.6']
- Q38: Guards required on table saws and woodworking machinery — gold ['1910.213'], got ['1910.261', '1910.219', '1910.213']
- Q40: When is an emergency eyewash or shower required at the workplace? — gold ['1910.151'], got ['1910.1052', '1910.1048', '1910.1045']
- Q56: Ventilation and fire rules for spray painting booths — gold ['1910.107'], got ['1910.94', '1910.107', '1910.269']

## Answers + hallucination check

- n: 60
- cites_correct_section: 0.95
- semantic_sim_to_gold_section: 0.82
- flagged_as_hallucination: 0

### Checker stress test (injected hallucinations)

- false positives on clean answers: 0
- overall detection: 0.965 of 289
- number_swap: 1.0 (n=57)
- foreign_sentence: 0.933 (n=60)
- wrong_citation: 0.9 (n=60)
- fake_citation: 1.0 (n=60)
- negation: 1.0 (n=52)

## Scored against golden answers (as_given)

Golden check vs regulation text: 20/60 fully verified, 27/60 quotes verbatim, 171/180 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| dense_minilm | 0.483 | 0.25 | 0.208 | 0.161 | 0.729 | 0.212 | 0.714 | 0.95 |

## Scored against golden answers (corrected)

Golden check vs regulation text: 24/60 fully verified, 28/60 quotes verbatim, 174/178 key facts found.

| answerer | key_fact_recall | answers_with_all_facts | token_f1 | rouge_l | semantic_sim | semantic_sim_random_baseline | context_fact_recall@3 | cites_gold_section |
|---|---|---|---|---|---|---|---|---|
| dense_minilm | 0.492 | 0.25 | 0.216 | 0.172 | 0.73 | 0.209 | 0.722 | 0.95 |
