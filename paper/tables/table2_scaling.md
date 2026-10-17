# Table 2: Membership Leakage Scaling Across Dataset Sizes (E3)

Note: 8 shadows per size; LiRA unstable at small $N$. We report empirical metrics without claiming a power law.

| Private Size ($N$) | Candidate Universe ($2N$) | LOSS AUC | LOSS TPR@1% (%) | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) |
|---|---|---|---|---|---|---|---|
| 1,000 | 2,000 | 0.998 ± 0.001 | 93.99 ± 4.35 | 0.957 ± 0.006 | 20.75 ± 6.51 | 0.975 ± 0.004 | 45.49 ± 33.14 |
| 2,500 | 5,000 | 0.941 ± 0.006 | 27.68 ± 3.97 | 0.936 ± 0.003 | 12.85 ± 2.97 | 0.939 ± 0.003 | 6.53 ± 1.60 |
| 5,000 | 10,000 | 0.820 ± 0.010 | 7.99 ± 1.55 | 0.898 ± 0.008 | 8.19 ± 1.05 | 0.882 ± 0.006 | 3.23 ± 0.60 |
| 10,000 | 20,000 | 0.689 ± 0.003 | 2.84 ± 0.09 | 0.827 ± 0.006 | 4.81 ± 0.63 | 0.785 ± 0.003 | 2.07 ± 0.28 |
