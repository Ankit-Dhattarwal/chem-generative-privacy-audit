# Auditing Membership Leakage in Small SMILES Language Models: LiRA, Randomized-SMILES Training and DP-SGD (Preliminary Study)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Differential Privacy](https://img.shields.io/badge/Differential%20Privacy-Opacus%201.6.0-green.svg)](https://opacus.ai/)
[![Cheminformatics](https://img.shields.io/badge/Cheminformatics-RDKit-orange.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical audit of membership inference vulnerability, randomized-SMILES scoring sensitivity, and defensive interventions in small autoregressive molecular language models (2 layers, $d = 128$, with 419,584 parameters) trained on the MOSES molecular benchmark.

---

## Overview

In computer-aided drug design, language models are frequently trained on chemical libraries containing proprietary chemical structures and patented scaffolds. This repository implements an empirical audit framework evaluating:
1. **Membership Inference Attacks (MIA)**: Standard sequence loss thresholding (`LOSS`) and calibrated Likelihood-Ratio Attacks (`LiRA`, both offline and online variants) across training duration (epochs 5, 20, 60).
2. **Private Dataset Scaling**: Exposure scaling across cohort sizes $N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$ (8 shadow models per size).
3. **String-Form Sensitivity**: Evaluating canonical-trained models with up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms, with mean NLL) to assess whether memorization is syntax-dependent.
4. **Defensive Interventions**: Randomized-SMILES data augmentation and Differentially Private SGD (`DP-SGD`, targeting $\epsilon \in \{8.0, 3.0\}$, $\delta = 10^{-5}$).
5. **Generative Sampling Audit**: Evaluating canonicalized structure matches across 15,000 generated samples (5,000 per model) against training and holdout sets.

---

## Key Empirical Findings

All metrics are evaluated using negative log-likelihood sum (`nll_sum`) and traced directly to `results_summary/*.csv`:

- **Baseline Exposure (Epoch 60)**: Online LiRA achieves a ROC-AUC of **0.898 ± 0.003** with a True Positive Rate (TPR) of **27.82 ± 1.47%** at 1% False Positive Rate (FPR) and **8.15 ± 1.58%** at 0.1% FPR. Standard loss thresholding substantially underestimates exposure (AUC **0.689 ± 0.005**, TPR@1% **2.96 ± 0.24%**).
- **String-Form Finding**: Scoring the epoch-60 canonical baseline models using up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms, with mean NLL) drops online LiRA AUC to **0.528 ± 0.006** (LOSS AUC drops to **0.511 ± 0.004**), indicating that under mean aggregation, alternative string forms remove most of the signal (individual non-canonical forms were not scored separately).
- **Dataset Size Scaling**: On small cohorts ($N = 1,000$), standard LOSS achieves an AUC of **0.998 ± 0.001** (TPR@1% **93.99 ± 4.35%**), while online LiRA reaches AUC **0.975 ± 0.004** (TPR@1% **45.49 ± 33.14%**; note that LiRA exhibits high estimation variance at small $N$).
- **Transfer Learning**: Private fine-tuning on 10,000 molecules after public pretraining leaks private membership at offline LiRA AUC **0.801 ± 0.003** (TPR@1% **4.36 ± 0.51%**).
- **Defenses**:
  - *Randomized-SMILES Training*: Suppresses canonical attack AUC to **0.528 ± 0.004** (TPR@1% **1.38 ± 0.21%**).
  - *DP-SGD*: Suppresses attack AUC to **0.500 – 0.528**, but preliminary runs exhibited possible loss of generative utility; an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense.
- **Generative Replication**: Of 15,000 generated samples, 12 canonicalized outputs matched molecules in the training set and 7 matched molecules in the holdout set. Conditioning on these 19 matches and assuming equal match probability for the two equally sized reference sets (H0: p = 0.5), the one-sided exact binomial p-value is 0.1796 and the two-sided exact binomial p-value is 0.3593. Neither result is significant at the 0.05 level.

---

## Summary Tables

### Table 1: Baseline Membership Leakage Across Epochs

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

### Table 4: Defense Comparison at Epoch 60

*Note: preliminary, utility not measured.*

#### Panel (a): Calibrated Likelihood-Ratio Attacks (LiRA)

| Defense Configuration | Shadow Setup | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) | Strongest Attack |
|---|---|---|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.880 ± 0.004 | 27.15 ± 1.63 | 0.898 ± 0.003 | 27.82 ± 1.47 | **Online LiRA** |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.528 ± 0.004 | 1.38 ± 0.21 | 0.508 ± 0.004 | 1.09 ± 0.17 | **Offline LiRA** |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.552 ± 0.005 | 1.56 ± 0.17 | 0.519 ± 0.005 | 1.14 ± 0.18 | **Offline LiRA** |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.528 ± 0.003 | 1.20 ± 0.14 | 0.501 ± 0.005 | 1.09 ± 0.16 | **Offline LiRA** |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.520 ± 0.003 | 1.12 ± 0.06 | 0.500 ± 0.002 | 1.01 ± 0.10 | **Offline LiRA** |

#### Panel (b): Uncalibrated Loss Thresholding Baseline

| Defense Configuration | Shadow Setup | LOSS AUC | LOSS TPR@1% (%) |
|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.689 ± 0.005 | 2.96 ± 0.24 |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.506 ± 0.004 | 1.08 ± 0.16 |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.506 ± 0.004 | 1.10 ± 0.16 |
| DP-SGD (ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.505 ± 0.004 | 1.13 ± 0.15 |
| DP-SGD (ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.503 ± 0.004 | 1.10 ± 0.06 |

---

## Repository Structure

```
├── README.md                      # Project overview and audited findings
├── REPRODUCE.md                   # Reproduction instructions and runtimes
├── CHANGELOG.md                   # Comprehensive audit change log
├── data/
│   ├── download_moses.py          # Downloader for official upstream MOSES benchmark
│   ├── pool.csv                   # Candidate evaluation pool (20,000 compounds)
│   ├── val.csv                    # Held-out validation split
│   └── pool_metadata.csv          # Scaffolds and randomized SMILES variants
├── claims.yaml                    # Formal claim registry mapping claims to CSVs
├── LICENSE                        # MIT License (Ankit Dhattarwal)
├── requirements.txt               # Pinned dependencies
├── colab_pipeline.py              # End-to-end experiment pipeline
├── scripts/
│   ├── verify_claims.py           # CSV claim re-derivation and numeric-token audit
│   └── mutation_test.py           # Mutation testing suite for claims verifier
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
│   ├── references.bib             # BibTeX bibliography
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

# 3. Verify all numerical claims across manuscripts, tables, and PDF
python scripts/verify_claims.py

# 4. Run mutation testing suite (must detect all corruptions)
python scripts/mutation_test.py

# 5. Compile manuscript PDF
tectonic paper/main.tex --outdir paper/
```

### Claims Verification Architecture
- **CSV derivation:** `scripts/verify_claims.py` checks registered metrics against `results_summary/*.csv`, recomputes the E6 binomial tests, checks the pooled E6 p-values in the abstract sources, and audits numerical tokens against registered values or a constants whitelist.
- **Table generation:** `paper/make_tables.py` regenerates the published table sources directly from the CSV summaries.
- **Mutation coverage:** `scripts/mutation_test.py` checks ten specified CSV and abstract mutations; it is a targeted regression suite, not a proof that every possible text error is detected.

See [REPRODUCE.md](REPRODUCE.md) for full training and execution details.
