# TODO for v2 (Future Research Roadmap)

This document tracks identified methodological, statistical, and computational extensions planned for version 2 (v2) of this research project.

---

### 1. Utility Metrics for DP-SGD and Randomized-SMILES Models
- **Issue:** The DP arm represents a preliminary run with possible loss of generative utility (not evaluated): chemical validity evaluations for DP-SGD were not recorded in the released summary, and we did not evaluate the generative utility of the DP models. Similarly, validation NLL, validity, and uniqueness of randomized-SMILES-trained models were not evaluated. An attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense, and no privacy-utility conclusion can be drawn.
- **Action for v2:**
  - Record chemical utility metrics (chemical validity, uniqueness, novelty vs. training set, Fréchet ChemNet Distance, and validation NLL) across all checkpoints for both DP-SGD and randomized-SMILES models.
  - Implement adaptive gradient clipping and tune learning rate schedules with larger batch sizes or gradient accumulation to preserve generative utility under DP-SGD.
  - Track and verify RDP privacy accountant state steps continuously across checkpoints without reset on resume.

### 2. Full 32 Shadow Models for E3 Scaling, DP-SGD, and Pretraining-Fine-Tuning
- **Issue:** While the baseline and E5a arms used 32 shadow models, compute limits restricted E3 (dataset size scaling), DP-SGD, and pretrain-fine-tuning arms to 8 shadow models each, causing high estimation variance in LiRA Gaussian parameters (e.g., online LiRA TPR@1% standard deviation was $\pm 33.14\%$ at $N=1,000$).
- **Action for v2:**
  - Scale from 8 to 32 shadow models across all cohort sizes ($N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$), DP-SGD target epsilons ($\epsilon \in \{8, 3\}$), and the pretrain-FT condition.
  - Stabilize out-distribution parameter estimation to evaluate empirical scaling behavior with reduced estimation uncertainty.

### 3. High-Power Generative Sampling Audit (E6 with $\ge 100,000$ Samples on CUDA)
- **Issue:** Experiment E6 audited 5,000 samples per model (15,000 total) on Apple Silicon (MPS). At this budget, the 12 member copies vs. 7 out copies observed were not statistically significant ($p = 0.1796 > 0.05$ one-sided binomial test; $p = 0.359 > 0.05$ Fisher exact test).
- **Action for v2:**
  - Increase sampling budget to $\ge 100,000$ generated molecules per model across all 32 baseline models.
  - Conduct hardware-matched sampling on CUDA (e.g., NVIDIA Tesla T4 or A100 matching shadow training, rather than Apple Silicon MPS).
  - Test statistical significance of exact verbatim replications and scaffold-level memorization at high statistical power.

### 4. Per-Molecule Bootstrap Confidence Intervals Across All Arms
- **Issue:** The paper currently reports standard deviation across leave-one-out shadow target models. While 95% bootstrap CIs over molecules were computed for baseline epoch 60 online LiRA on 3 targets, comprehensive bootstrap intervals are needed across all configurations.
- **Action for v2:**
  - Implement an automated non-parametric bootstrap routine over candidate molecules ($B = 2,000$ resamples) for every reported configuration in Tables 1–4.
  - Report 95% percentile confidence intervals alongside cross-target standard deviations.

### 5. From-Scratch Baseline Control for Pretraining-Fine-Tuning
- **Issue:** In the pretrain $\to$ fine-tune scenario, models were pretrained for 60 epochs on 10,000 compounds and fine-tuned for 20 epochs on 10,000 compounds. There was no control model trained from scratch on 10,000 compounds for 20 epochs.
- **Action for v2:**
  - Train an exact control arm from random initialization on the same 10,000 private compounds for exactly 20 epochs without pretraining.
  - Disentangle how much residual leakage originates from private exposure duration versus pretraining representation alignment.

### 6. Scoring Non-Canonical Variants Individually
- **Issue:** In v1, variants were scored using mean aggregation across up to 10 distinct SMILES strings; individual non-canonical forms were not scored separately.
- **Action for v2:**
  - Score individual non-canonical variants separately to evaluate worst-case single-representation leakage.
  - Compare alternative aggregation functions such as minimum loss or logsumexp pooling across string representations.

### 7. Evaluation on a Second Molecular Dataset
- **Issue:** All experiments in this preliminary study were conducted on a single filtered subset of the MOSES benchmark (clean, drug-like molecules).
- **Action for v2:**
  - Replicate the audit on a second distinct chemical dataset with different scaffold distributions and property profiles (e.g., ChEMBL bioactivity subsets or ZINC lead-like compounds).
  - Assess whether scaffold diversity and library composition alter membership exposure dynamics.

### 8. Larger Model Architectures
- **Issue:** The evaluated architecture is restricted to a small 2-layer, $d=128$, $\sim 0.4\text{M}$ parameter causal transformer with modest chemical validity ($\sim 60\%$).
- **Action for v2:**
  - Scale evaluation to larger causal and masked architectures (e.g., 6–12 layers, $d=512–768$, 10M–100M parameters, such as ChemBERTa-scale or MolGPT-scale models).
  - Measure how increased parameter capacity affects token memorization, chemical validity, and LiRA susceptibility.
