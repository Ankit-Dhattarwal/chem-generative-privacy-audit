# TODO for v2 (Future Research Roadmap)

This document tracks identified methodological, statistical, and computational extensions planned for version 2 (v2) of this research project.

---

### 1. Differential Privacy (DP-SGD) Rerun with Utility Preservation & Accounting Verification
- **Issue:** The DP arm represents a preliminary run with utility collapse: chemical validity evaluations for DP-SGD were not recorded in the released summary; preliminary runs suffered severe utility degradation and generated strings failed standard valence checks. Furthermore, serialized accountant state was not independently verified. An attack AUC of $\sim 0.5$ for a model that does not generate valid molecules is not evidence of a useful defense, and no privacy-utility conclusion can be drawn.
- **Action for v2:**
  - Implement adaptive gradient clipping and tune learning rate schedules with larger batch sizes or gradient accumulation to preserve generative validity.
  - Track and verify RDP privacy accountant state steps continuously across checkpoints without reset on resume.
  - Record chemical validity evaluations systematically in CSV summaries alongside Fréchet ChemNet Distance and scaffold diversity to identify a viable operating point.

### 2. Dataset Scaling (E3) with 32 Shadow Models per Size
- **Issue:** The E3 experiment used 8 shadow models per size ($N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$), causing high estimation variance in LiRA Gaussian parameters at small sample sizes (e.g., online LiRA TPR@1% std was $\pm 33.14\%$ at $N=1,000$).
- **Action for v2:**
  - Scale from 8 to 32 shadow models across all cohort sizes ($N = 1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}$).
  - Stabilize out-distribution parameter estimation to test whether empirical scaling follows a formal power law.

### 3. High-Power Generative Sampling Audit (E6 with $\ge 100,000$ Samples)
- **Issue:** Experiment E6 audited 5,000 samples per model (15,000 total). At this budget, the 12 member copies vs. 7 out copies observed were not statistically significant ($p = 0.1796 > 0.05$ one-sided binomial test; $p = 0.359 > 0.05$ Fisher exact test).
- **Action for v2:**
  - Increase sampling budget to $\ge 100,000$ generated molecules per model across all 32 baseline models.
  - Conduct hardware-matched sampling (CUDA Tesla T4 or A100 matching shadow training, rather than Apple Silicon MPS).
  - Test statistical significance of exact and approximate (Tanimoto $\ge 0.95$) verbatim replications at high statistical power.

### 4. Per-Molecule Bootstrap Confidence Intervals Across All Arms
- **Issue:** The paper currently reports standard deviation across leave-one-out shadow target models. While 95% bootstrap CIs over molecules were computed for baseline epoch 60 online LiRA on 3 targets, comprehensive bootstrap intervals are needed across all configurations.
- **Action for v2:**
  - Implement an automated non-parametric bootstrap routine over candidate molecules ($B = 2,000$ resamples) for every reported configuration in Tables 1–4.
  - Report 95% percentile confidence intervals alongside cross-target standard deviations.

### 5. From-Scratch Baseline Control for Pretraining-Fine-Tuning Experiment
- **Issue:** In the pretrain $\to$ fine-tune scenario, models were pretrained for 60 epochs on 10,000 compounds and fine-tuned for 20 epochs on 10,000 compounds. There was no control model trained from scratch on 10,000 compounds for 20 epochs.
- **Action for v2:**
  - Train an exact control arm from random initialization on the same 10,000 private compounds for exactly 20 epochs.
  - Disentangle how much residual leakage originates from private exposure duration versus pretraining representation alignment.

### 6. Scoring Non-Canonical Variants Individually
- **Issue:** In v1, variants were scored using mean aggregation across up to 10 distinct SMILES strings; individual non-canonical forms were not scored separately.
- **Action for v2:**
  - Score non-canonical variants individually to evaluate worst-case single-representation leakage.
  - Compare alternative aggregation functions such as minimum loss or logsumexp pooling across string representations.
