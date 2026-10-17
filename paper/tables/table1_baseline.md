# Table 1: Baseline Membership Inference Vulnerability Across Epochs

Evaluated over 32 shadow models on 20,000 candidate targets (10,000 members, 10,000 non-members). Values denote mean ± standard deviation across 32 leave-one-out target models (not a bootstrap CI). Consistently evaluated using `nll_sum`, with randomized variants scoring (up to 10 distinct SMILES strings per molecule, the canonical form plus randomized forms, mean NLL) also reported at epoch 60.

| Epochs | Scoring Representation | Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|---|---|
| 5 | Canonical (`nll_sum`) | Loss Thresholding (LOSS) | 0.535 ± 0.004 | 1.23 ± 0.18 | 0.14 ± 0.07 |
| 5 | Canonical (`nll_sum`) | Offline LiRA (Out-only) | 0.630 ± 0.008 | 3.15 ± 0.36 | 0.42 ± 0.15 |
| 5 | Canonical (`nll_sum`) | Online LiRA (In vs. Out) | 0.595 ± 0.006 | 2.12 ± 0.35 | 0.19 ± 0.09 |
| 5 | Canonical (`nll_sum`) | **Strongest: offline LiRA** | **0.630 ± 0.008** | **3.15 ± 0.36** | **0.42 ± 0.15** |
| 20 | Canonical (`nll_sum`) | Loss Thresholding (LOSS) | 0.591 ± 0.004 | 1.66 ± 0.21 | 0.21 ± 0.08 |
| 20 | Canonical (`nll_sum`) | Offline LiRA (Out-only) | 0.763 ± 0.007 | 9.58 ± 0.84 | 1.86 ± 0.47 |
| 20 | Canonical (`nll_sum`) | Online LiRA (In vs. Out) | 0.760 ± 0.006 | 8.66 ± 0.62 | 1.34 ± 0.33 |
| 20 | Canonical (`nll_sum`) | **Strongest: offline LiRA** | **0.763 ± 0.007** | **9.58 ± 0.84** | **1.86 ± 0.47** |
| 60 | Canonical (`nll_sum`) | Loss Thresholding (LOSS) | 0.689 ± 0.005 | 2.96 ± 0.24 | 0.41 ± 0.15 |
| 60 | Canonical (`nll_sum`) | Offline LiRA (Out-only) | 0.880 ± 0.004 | 27.15 ± 1.63 | 8.22 ± 1.32 |
| 60 | Canonical (`nll_sum`) | Online LiRA (In vs. Out) | 0.898 ± 0.003 | 27.82 ± 1.47 | 8.15 ± 1.58 |
| 60 | Canonical (`nll_sum`) | **Strongest: online LiRA** | **0.898 ± 0.003** | **27.82 ± 1.47** | **8.15 ± 1.58** |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Loss Thresholding (LOSS) | 0.511 ± 0.004 | 1.09 ± 0.12 | 0.12 ± 0.05 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Offline LiRA (Out-only) | 0.560 ± 0.007 | 2.01 ± 0.29 | 0.30 ± 0.10 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Online LiRA (In vs. Out) | 0.528 ± 0.006 | 1.45 ± 0.47 | 0.18 ± 0.12 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | **Strongest: offline LiRA** | **0.560 ± 0.007** | **2.01 ± 0.29** | **0.30 ± 0.10** |
