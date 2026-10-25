# Table 4: Defense Comparison at Epoch 60

*Note: preliminary, utility not measured.*

### Panel (a): Calibrated Likelihood-Ratio Attacks (LiRA)

| Defense Configuration | Shadow Setup | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) | Strongest Attack |
|---|---|---|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.880 ± 0.004 | 27.15 ± 1.63 | 0.898 ± 0.003 | 27.82 ± 1.47 | **Online LiRA** |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.528 ± 0.004 | 1.38 ± 0.21 | 0.508 ± 0.004 | 1.09 ± 0.17 | **Offline LiRA** |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.552 ± 0.005 | 1.56 ± 0.17 | 0.519 ± 0.005 | 1.14 ± 0.18 | **Offline LiRA** |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.528 ± 0.003 | 1.20 ± 0.14 | 0.501 ± 0.005 | 1.09 ± 0.16 | **Offline LiRA** |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.520 ± 0.003 | 1.12 ± 0.06 | 0.500 ± 0.002 | 1.01 ± 0.10 | **Offline LiRA** |

### Panel (b): Uncalibrated Loss Thresholding Baseline

| Defense Configuration | Shadow Setup | LOSS AUC | LOSS TPR@1% (%) |
|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.689 ± 0.005 | 2.96 ± 0.24 |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.506 ± 0.004 | 1.08 ± 0.16 |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.506 ± 0.004 | 1.10 ± 0.16 |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.505 ± 0.004 | 1.13 ± 0.15 |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.503 ± 0.004 | 1.10 ± 0.06 |
