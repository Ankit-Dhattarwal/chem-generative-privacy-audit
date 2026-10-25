# Table 5: Generative Memorization Audit (E6)

Evaluated on 5,000 generated molecules per model on Apple Silicon (MPS). Exact counts are derived from each model's sample count. Validity, uniqueness, and novelty are reported as the unweighted mean of the three per-model rates; hit counts and the binomial test use pooled counts. One-sided binomial tests ($p=0.5$ null hypothesis) assess whether member matches exceed holdout matches, conditional on a match.

| Model ID | Validity (%) | Uniqueness (%) | Novelty vs Members (%) | Training Matches (Count) | Holdout Matches (Count) | Binomial p-value ($p=0.5$) | Significant at α=0.05? |
|---|---|---|---|---|---|---|---|
| Model 0 | 60.24% | 99.97% | 99.97% | 1 | 1 | 0.7500 | No ($p > 0.05$) |
| Model 15 | 62.26% | 99.87% | 99.90% | 3 | 2 | 0.5000 | No ($p > 0.05$) |
| Model 31 | 59.10% | 99.90% | 99.73% | 8 | 4 | 0.1938 | No ($p > 0.05$) |
| **Mean rates; pooled hits (15,000 samples)** | **60.53%** | **99.91%** | **99.87%** | **12** | **7** | **0.1796** | **No ($p > 0.05$)** |
