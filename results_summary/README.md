# Results Summary: Column Definitions and Data Dictionary

This directory contains aggregated CSV result summaries and execution manifests for the empirical audit. All metrics are read-only and serve as the single source of truth for all tables and figures in the manuscript.

---

## 1. `baseline_attack_summary.csv`
Summary metrics for baseline causal SMILES models across training epochs (5, 20, 60) evaluated using 32 leave-one-out shadow target models.

- **`arm`**: Hardware execution environment (`CUDA`).
- **`epoch`**: Training epoch checkpoint evaluated (`5`, `20`, `60`).
- **`stat`**: Sequence loss reduction metric across tokens:
  - `nll_sum`: Sum of negative log-likelihood across all sequence tokens (primary metric).
  - `nll_mean`: Average negative log-likelihood per token (length-normalized).
  - `nll_variants_mean`: Average negative log-likelihood across up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms), mean NLL (evaluated at epoch 60).
- **`method`**: Attack algorithm:
  - `LOSS`: Standard sequence loss thresholding.
  - `offline_LiRA`: Calibrated Likelihood-Ratio Attack using out-distribution mean and variance ($\mu_{\text{out}}, \sigma_{\text{out}}$).
  - `online_LiRA`: Calibrated Likelihood-Ratio Attack modeling both in-distribution and out-distribution parametric Gaussians.
- **`auc_mean`**: Mean Area Under the ROC Curve across 32 leave-one-out target models.
- **`auc_std`**: Sample standard deviation of ROC-AUC across 32 leave-one-out targets.
- **`tpr1_mean`**: Mean True Positive Rate at a False Positive Rate of 1.0% ($\text{FPR} \le 0.01$).
- **`tpr1_std`**: Sample standard deviation of TPR@1% FPR across 32 targets.
- **`tpr01_mean`**: Mean True Positive Rate at a False Positive Rate of 0.1% ($\text{FPR} \le 0.001$).
- **`tpr01_std`**: Sample standard deviation of TPR@0.1% FPR across 32 targets.
- **`balanced_accuracy_mean`**: Mean balanced classification accuracy at the median threshold.
- **`balanced_accuracy_std`**: Sample standard deviation of balanced accuracy.

---

## 2. `E3_scaling_summary.csv`
Summary metrics evaluating membership leakage scaling across private dataset sizes ($N \in \{1000, 2500, 5000, 10000\}$) trained for 60 epochs.

- **`arm`**: Hardware arm (`CUDA`).
- **`member_size`**: Number of private training molecules $N$. The evaluation universe is $2N$ molecules ($N$ members and $N$ non-members).
- **`method`**: Attack algorithm (`LOSS`, `offline_LiRA`, `online_LiRA`).
- **`auc_mean`**: Mean ROC-AUC across 8 shadow models per size.
- **`auc_std`**: Sample standard deviation of ROC-AUC across 8 shadow models.
- **`tpr1_mean`**: Mean TPR at 1.0% FPR across 8 shadow models.
- **`tpr1_std`**: Sample standard deviation of TPR at 1.0% FPR across 8 shadow models.

---

## 3. `E5a_augmentation_summary.csv`
Summary metrics for models trained with randomized-SMILES data augmentation (E5a) over 60 epochs with 32 shadow models. Columns match `baseline_attack_summary.csv`.

---

## 4. `DP_SGD_summary.csv`
Summary metrics for models trained with Differentially Private SGD (Opacus 1.6.0) across 8 shadow models per privacy budget.

- **`arm`**: Hardware arm (`CUDA`).
- **`epsilon_target`**: Target differential privacy parameter $\epsilon \in \{8, 3\}$ at $\delta = 10^{-5}$.
- **`method`**: Attack algorithm (`LOSS`, `offline_LiRA`, `online_LiRA`).
- **`auc_mean`**: Mean ROC-AUC across 8 shadow models.
- **`auc_std`**: Sample standard deviation of ROC-AUC across 8 shadow models.
- **`tpr1_mean`**: Mean TPR at 1.0% FPR across 8 shadow models.
- **`tpr1_std`**: Sample standard deviation of TPR at 1.0% FPR across 8 shadow models.

*Note: Results are preliminary. Serialized Opacus accountant states from all 16 latest DP-SGD checkpoints were reloaded and their ε values recomputed at δ=10⁻⁵; the recomputed values agree closely with per-model logs, with small numeric differences. This does not independently establish end-to-end correctness of the training implementation or its assumptions; downstream generative chemical utility was not measured.*

---

## 5. `pretrain_ft_summary.csv`
Per-target attack results for the transfer learning experiment (public pretraining on 10,000 molecules for 60 epochs; fine-tuning on 10,000 private molecules for 20 epochs across 8 shadow models).

- **`target`**: Target shadow model index ($0 \dots 7$).
- **`method`**: Attack algorithm (`LOSS`, `offline_LiRA`, `online_LiRA`).
- **`stat`**: Scoring loss reduction metric (`nll_sum`).
- **`auc`**: ROC-AUC for this target model.
- **`tpr_fpr_1pct`**: True Positive Rate at 1% FPR.
- **`tpr_fpr_0_1pct`**: True Positive Rate at 0.1% FPR.
- **`balanced_accuracy`**: Balanced accuracy at median attack score.
- **`arm`**: Hardware execution arm (`CUDA`).
- **`scenario`**: Experimental scenario description.

---

## 6. `E6_generative_audit_summary.csv`
Downstream generative sampling metrics auditing canonicalized structure matches across 5,000 samples per baseline model (Models 0, 15, 31; total 15,000 samples).

- **`arm`**: Execution hardware (`MPS` on Apple Silicon).
- **`model`**: Evaluated baseline model index (`0`, `15`, `31`).
- **`samples`**: Number of generated SMILES sequences ($5,000$).
- **`validity`**: Fraction of samples that are chemically valid molecules parsable by RDKit.
- **`uniqueness_valid`**: Fraction of valid samples that are structurally unique.
- **`novelty_vs_members`**: Fraction of valid samples that do not match any training member compound.
- **`exact_member_copy_fraction_all_samples`**: Fraction of all samples whose canonicalized structure matches a training member molecule ($Count = \text{Fraction} \times 5,000$).
- **`exact_out_copy_fraction_all_samples`**: Fraction of all samples whose canonicalized structure matches an unseen holdout molecule ($Count = \text{Fraction} \times 5,000$).

---

## 7. `manifest.json`
Run stage status dictionary tracking timestamps, compute devices, software versions, and stage completion states. Note that a stale initialization error record (`ImportError`) is present from an initial module launch attempt.
