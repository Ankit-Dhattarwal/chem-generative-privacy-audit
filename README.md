# Auditing Membership Leakage in Small SMILES Language Models: LiRA, Randomized-SMILES Training and DP-SGD (Preliminary Study)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Differential Privacy](https://img.shields.io/badge/Differential%20Privacy-Opacus%201.6.0-green.svg)](https://opacus.ai/)
[![Cheminformatics](https://img.shields.io/badge/Cheminformatics-RDKit-orange.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical audit of membership inference vulnerability, randomized-SMILES scoring sensitivity, and defensive interventions in small autoregressive molecular language models (2 layers, $d=128$, $\sim 0.4\text{M}$ parameters) trained on the MOSES molecular benchmark.

---

## Overview

In computer-aided drug design, language models are frequently trained on chemical libraries containing proprietary chemical structures and patented scaffolds. This repository implements an empirical audit framework evaluating:
1. **Membership Inference Attacks (MIA)**: Standard sequence loss thresholding (`LOSS`) and calibrated Likelihood-Ratio Attacks (`LiRA`, both offline and online variants) across training duration (epochs 5, 20, 60).
2. **Private Dataset Scaling**: Exposure scaling across cohort sizes $N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$ (8 shadow models per size).
3. **String-Form Sensitivity**: Evaluating canonical-trained models with up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms, with mean NLL) to assess whether memorization is syntax-dependent.
4. **Defensive Interventions**: Randomized-SMILES data augmentation and Differentially Private SGD (`DP-SGD`, targeting $\epsilon \in \{8.0, 3.0\}$, $\delta = 10^{-5}$).
5. **Generative Sampling Audit**: Evaluating verbatim replication across 15,000 generated samples (5,000 per model) against member versus non-member holdout sets.

---

## Key Empirical Findings

All metrics are evaluated using negative log-likelihood sum (`nll_sum`) and traced directly to `results_summary/*.csv`:

- **Baseline Exposure (Epoch 60)**: Online LiRA achieves a ROC-AUC of **0.898 ± 0.003** with a True Positive Rate (TPR) of **27.82 ± 1.47%** at 1% False Positive Rate (FPR) and **8.15 ± 1.58%** at 0.1% FPR. Standard loss thresholding substantially underestimates exposure (AUC **0.689 ± 0.005**, TPR@1% **2.96 ± 0.24%**).
- **String-Form Finding**: Scoring the epoch-60 canonical baseline models using up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms, with mean NLL) drops online LiRA AUC to **0.528 ± 0.006** (LOSS AUC drops to **0.511 ± 0.004**), indicating that under mean aggregation, alternative string forms remove most of the signal (individual non-canonical forms were not scored separately).
- **Dataset Size Scaling**: On small cohorts ($N = 1,000$), standard LOSS achieves an AUC of **0.998 ± 0.001** (TPR@1% **93.99 ± 4.35%**), while online LiRA reaches AUC **0.975 ± 0.004** (TPR@1% **45.49 ± 33.14%**; note that LiRA exhibits high estimation variance at small $N$).
- **Transfer Learning**: Private fine-tuning on 10,000 molecules after public pretraining leaks private membership at offline LiRA AUC **0.801 ± 0.003** (TPR@1% **4.36 ± 0.51%**).
- **Defenses**:
  - *Randomized-SMILES Training*: Suppresses canonical attack AUC to **0.528 ± 0.004** (TPR@1% **1.38 ± 0.21%**).
  - *DP-SGD*: Suppresses attack AUC to **0.500 – 0.528**, but preliminary runs suffered severe utility collapse and generated strings failed standard valence checks (chemical validity evaluations for DP-SGD were not recorded in the released summary; AUC ~0.5 for a model that does not generate valid molecules is not evidence of a useful defense).
- **Generative Replication**: In a 15,000-sample audit, models produced 12 exact member copies vs. 7 holdout copies; a one-sided binomial test ($p = 0.1796 > 0.05$) and Fisher exact test ($p = 0.359 > 0.05$) indicate that this difference is **not statistically significant** under the evaluated sample size.

---

## Summary Tables

### Table 1: Baseline Membership Leakage Across Epochs

Evaluated over 32 shadow models on 20,000 candidate targets (10,000 members, 10,000 non-members). Values denote mean ± standard deviation across 32 leave-one-out target models (not a bootstrap CI). Consistently evaluated using `nll_sum`, with randomized variants scoring ($K=10$) also reported at epoch 60.

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
| 60 (variants) | Randomized-SMILES ($K=10$) | Loss Thresholding (LOSS) | 0.511 ± 0.004 | 1.09 ± 0.12 | 0.12 ± 0.05 |
| 60 (variants) | Randomized-SMILES ($K=10$) | Offline LiRA (Out-only) | 0.560 ± 0.007 | 2.01 ± 0.29 | 0.30 ± 0.10 |
| 60 (variants) | Randomized-SMILES ($K=10$) | Online LiRA (In vs. Out) | 0.528 ± 0.006 | 1.45 ± 0.47 | 0.18 ± 0.12 |
| 60 (variants) | Randomized-SMILES ($K=10$) | **Strongest: offline LiRA** | **0.560 ± 0.007** | **2.01 ± 0.29** | **0.30 ± 0.10** |

### Table 3: Defense Benchmark at Epoch 60

*Note on DP rows: preliminary run with utility collapse; no privacy-utility conclusion. AUC ~0.5 for a model that does not generate valid molecules is not evidence of a useful defense.*

| Defense Configuration | Shadow Setup | LOSS AUC | LOSS TPR@1% (%) | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) | Strongest Attack (Method: AUC, TPR@1%) |
|---|---|---|---|---|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.689 ± 0.005 | 2.96 ± 0.24 | 0.880 ± 0.004 | 27.15 ± 1.63 | 0.898 ± 0.003 | 27.82 ± 1.47 | **online LiRA**: 0.898 (TPR 27.82%) |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.506 ± 0.004 | 1.08 ± 0.16 | 0.528 ± 0.004 | 1.38 ± 0.21 | 0.508 ± 0.004 | 1.09 ± 0.17 | **offline LiRA**: 0.528 (TPR 1.38%) |
| Randomized-SMILES Training (Variants Scoring, K=10) | 32 shadows | 0.506 ± 0.004 | 1.10 ± 0.16 | 0.552 ± 0.005 | 1.56 ± 0.17 | 0.519 ± 0.005 | 1.14 ± 0.18 | **offline LiRA**: 0.552 (TPR 1.56%) |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.505 ± 0.004 | 1.13 ± 0.15 | 0.528 ± 0.003 | 1.20 ± 0.14 | 0.501 ± 0.005 | 1.09 ± 0.16 | **offline LiRA**: 0.528 (TPR 1.20%) |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.503 ± 0.004 | 1.10 ± 0.06 | 0.520 ± 0.003 | 1.12 ± 0.06 | 0.500 ± 0.002 | 1.01 ± 0.10 | **offline LiRA**: 0.520 (TPR 1.12%) |

---|---|---|---|---|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.689 ± 0.005 | 2.96 ± 0.24 | 0.880 ± 0.004 | 27.15 ± 1.63 | 0.898 ± 0.003 | 27.82 ± 1.47 | **online LiRA**: 0.898 (TPR 27.82%) |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.506 ± 0.004 | 1.08 ± 0.16 | 0.528 ± 0.004 | 1.38 ± 0.21 | 0.508 ± 0.004 | 1.09 ± 0.17 | **offline LiRA**: 0.528 (TPR 1.38%) |
| Randomized-SMILES Training (Variants Scoring, K=10) | 32 shadows | 0.506 ± 0.004 | 1.10 ± 0.16 | 0.552 ± 0.005 | 1.56 ± 0.17 | 0.519 ± 0.005 | 1.14 ± 0.18 | **offline LiRA**: 0.552 (TPR 1.56%) |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.505 ± 0.004 | 1.13 ± 0.15 | 0.528 ± 0.003 | 1.20 ± 0.14 | 0.501 ± 0.005 | 1.09 ± 0.16 | **offline LiRA**: 0.528 (TPR 1.20%) |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.503 ± 0.004 | 1.10 ± 0.06 | 0.520 ± 0.003 | 1.12 ± 0.06 | 0.500 ± 0.002 | 1.01 ± 0.10 | **offline LiRA**: 0.520 (TPR 1.12%) |

---

## Repository Structure

```
├── README.md                      # Project overview and audited findings
├── REPRODUCE.md                   # Reproduction instructions and runtimes
├── CHANGELOG.md                   # Comprehensive audit change log
├── CLAIMS_CHECK.md                # Verification report mapping claims to CSVs
├── TODO_for_v2.md                 # Methodological roadmap for version 2
├── LICENSE                        # MIT License (Ankit Dhattarwal)
├── requirements.txt               # Pinned dependencies
├── colab_pipeline.py              # End-to-end experiment pipeline
├── check_claims.py                # Automated claims verification script
├── results_summary/               # Primary CSV summaries (canonical truth)
│   ├── README.md                  # Column definitions and data dictionary
│   ├── baseline_attack_summary.csv
│   ├── E3_scaling_summary.csv
│   ├── E5a_augmentation_summary.csv
│   ├── DP_SGD_summary.csv
│   ├── pretrain_ft_summary.csv
│   ├── E6_generative_audit_summary.csv
│   └── manifest.json
├── paper/
│   ├── main.tex                   # LaTeX manuscript
│   ├── main.pdf                   # Compiled publication PDF
│   ├── MANUSCRIPT.md              # Markdown manuscript (synchronized with main.tex)
│   ├── references.bib             # Verified bibliography
│   ├── make_tables.py             # Script to regenerate LaTeX/MD tables
│   ├── generate_figures.py        # Script to regenerate figures from CSVs
│   ├── tables/                    # Generated LaTeX and Markdown tables
│   └── figures/                   # Generated publication figures (.png and .pdf)
└── src/                           # Model architectures, attacks, and tokenizers
```

---

## Verification and Reproduction

To reproduce all tables, figures, verification checks, and the manuscript PDF without retraining:

```bash
# 1. Regenerate tables
python paper/make_tables.py

# 2. Regenerate figures
python paper/generate_figures.py

# 3. Verify all numerical claims against CSVs
python check_claims.py

# 4. Compile manuscript PDF
tectonic paper/main.tex --outdir paper/
```

See [REPRODUCE.md](REPRODUCE.md) for full training and execution details.
