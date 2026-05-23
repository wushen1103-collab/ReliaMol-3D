# DiffSBDD Aligned Vina Diagnostic Analysis

Input: official DiffSBDD public samples restricted to 39 public-PDB aligned pockets.

## Status

- Candidates: 3900
- Vina successes: 3862 (0.9903)
- Ligand prep failures: 38
- All-candidate reliable proxy rate: 0.9400
- Success-only reliable proxy rate: 0.9407

## Vina As Reliability Ranker

- ROC-AUC using -Vina affinity: 0.8658
- PR-AUC for reliable class: 0.9866
- PR-AUC for unreliable class: 0.4129
- Median affinity, reliable: -4.087
- Median affinity, unreliable: 17.311

## Pocketwise Top-10% Behavior

- Mean pocket top-10% reliable proxy rate: 0.9923
- Minimum pocket top-10% reliable proxy rate: 0.8000

Interpretation: Vina score-only is useful as a conservative post-generation filter on this aligned subset. The gain is driven mostly by deprioritizing clash/geometric failures, not by chemical validity, because the official DiffSBDD samples are already chemically clean.
