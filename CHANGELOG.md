# CHANGELOG

All modifications to the repository, manuscript, scripts, and tables are logged here in accordance with scientific auditing requirements. Every numerical claim is strictly derived from `results_summary/*.csv`.

---

### [2026-10-18] - Final Correctness & General Claim Verification Pass

#### 1. Claim Verification Script (`check_claims.py`, `CLAIMS_CHECK.md`)
- **File**: `check_claims.py`, `CLAIMS_CHECK.md`
- **Old -> New**: Replaced hardcoded string checker with an automated regex extractor and validator scanning all documents (`paper/MANUSCRIPT.md`, `paper/main.tex`, `README.md`, `paper/tables/*.md`).
- **Reason**: Ensure all reported numerical claims across the entire repository dynamically trace to source CSV summaries, bootstrap intervals, or explicit whitelisted parameters, failing non-zero if any claim is unmatched.

#### 2. Model Scaling ($N=1,000$) Metrics
- **File**: `paper/main.tex`, `paper/MANUSCRIPT.md`, `README.md`
- **Old -> New**: Replaced outdated $N=1,000$ values from initial draft with exact ground truth from `results_summary/E3_scaling_summary.csv`:
  - LOSS: AUC $0.998 \pm 0.001$, TPR@1% $93.99 \pm 4.35\%$
  - Offline LiRA: AUC $0.957 \pm 0.006$, TPR@1% $20.75 \pm 6.51\%$
  - Online LiRA: AUC $0.975 \pm 0.004$, TPR@1% $45.49 \pm 33.14\%$
- **Reason**: Enforce strict concordance between raw CSV records and manuscript reporting.

#### 3. Differential Privacy (DP-SGD) Reframing
- **File**: `paper/main.tex`, `paper/MANUSCRIPT.md`, `README.md`, `paper/tables/table3_defenses.{tex,md}`, `TODO_for_v2.md`
- **Old -> New**: Removed unrecorded numerical validity claims ($0\%$ at $\epsilon=3$, $1\%$ at $\epsilon=8$) and reframed the DP arm everywhere as a preliminary exploration with utility collapse: chemical validity evaluations were not recorded in the released summary, generated strings failed standard valence checks, and an attack AUC of $\sim 0.5$ in an invalid generator is not evidence of a useful defense.
- **Reason**: Avoid reporting metrics not present in CSV summaries and maintain scientific honesty regarding defense efficacy.

#### 4. Representation Variants Terminology
- **File**: `paper/main.tex`, `paper/MANUSCRIPT.md`, `README.md`, `paper/tables/table1_baseline.{tex,md}`, `paper/tables/table3_defenses.{tex,md}`, `REPRODUCE.md`, `results_summary/README.md`
- **Old -> New**: Replaced all occurrences of "$K=10$ randomized-SMILES variants" with "up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms), with mean NLL". Explicitly stated that evaluations were conducted under mean aggregation without separate scoring of individual non-canonical representations.
- **Reason**: Match the exact generation logic in `src/data.py` where compounds with few rotatable bonds yield fewer than 10 distinct variants.

#### 5. Bibliography Verification & Synchronization
- **File**: `paper/references.bib`, `paper/main.tex`, `paper/MANUSCRIPT.md`
- **Old -> New**: Verified all citations against arXiv and DOI registers:
  - Updated Aubinais et al. venue to AISTATS 2026 (arXiv:2502.06567).
  - Confirmed author lists and publication details for Krüger et al. 2025 (*J. Cheminformatics*), Ross et al. 2025 (*Digital Discovery*), Nemecek et al. 2026 (arXiv:2603.08913), and Zhang et al. 2024 (arXiv:2409.19798).
- **Reason**: Ensure all bibliographic metadata are accurate and verified online.

#### 6. Cleanliness and Removal of Promotional Phrasing
- **File**: `src/__init__.py`, `paper/main.tex`, `paper/MANUSCRIPT.md`, `README.md`, `RESEARCH_AUDIT_SUMMARY_FOR_CLAUDE.md`
- **Old -> New**: Swept and removed promotional language, promotional benchmarks, and unverified generalizations.
- **Reason**: Maintain an objective, measured scientific tone appropriate for peer review.

---

### [2026-10-17] - Initial Preprint Audit & Metric Standardization

#### 1. `paper/tables/` & `paper/make_tables.py`
- **File**: `paper/make_tables.py`, `paper/tables/table1_baseline.{tex,md}`, `paper/tables/table2_scaling.{tex,md}`, `paper/tables/table3_defenses.{tex,md}`, `paper/tables/table4_pretrain_ft.{tex,md}`, `paper/tables/table5_e6_generation.{tex,md}`
- **What**: 
  - Created automated table generation script reading exclusively from `results_summary/*.csv`.
  - Table 1: Standardized evaluation on `nll_sum` across epochs 5, 20, 60 for LOSS, offline LiRA, and online LiRA. Added dedicated "strongest attack" row for each epoch. Added randomized-SMILES variant scoring at epoch 60.
  - Table 2 (E3 scaling): Separated LOSS and LiRA attacks into distinct columns. Explicitly annotated that 8 shadow models per size were used, that LiRA exhibits high variance at small $N$, and refrained from claiming a power law.
  - Table 3 (Defenses): Benchmarked baseline against randomized-SMILES training (both canonical and variants scoring) and DP-SGD ($\epsilon \in \{8, 3\}$). Bolded strongest attack per row and annotated DP rows as preliminary.
  - Table 4 (Pretrain $\to$ Fine-tune): Reported mean $\pm$ std across 8 target models; noted absence of a from-scratch control trained for identical duration.
  - Table 5 (E6 Generation): Derived exact verbatim replication counts ($Count = \text{Fraction} \times 5,000$), chemical validity, uniqueness, and non-member fraction. Added one-sided binomial tests ($p=0.5$ null hypothesis) showing all $p > 0.05$ (non-significant difference).
- **Why**: Enforce strict alignment between CSV evidence and manuscript reporting.

#### 2. `paper/generate_figures.py` & `paper/figures/`
- **File**: `paper/generate_figures.py`, `paper/figures/fig1_exposure_epochs.{png,pdf}`, `paper/figures/fig2_data_scaling.{png,pdf}`, `paper/figures/fig3_defenses_comparison.{png,pdf}`, `paper/figures/fig4_generative_memorization.{png,pdf}`
- **What**:
  - Fig 1: Rendered baseline exposure progression for LOSS, offline LiRA, and online LiRA using `nll_sum`.
  - Fig 2: Separated LOSS (panel a) and LiRA attacks (panel b) onto dedicated axes to avoid scale and metric confusion.
  - Fig 3: Updated bar chart to display the strongest attack per defense configuration at epoch 60.
  - Fig 4: Updated generative audit plot to display discrete verbatim copy counts alongside chemical quality metrics, annotating $p > 0.05$.
- **Why**: Visual representation must strictly match the audited tables and avoid conflating distinct attack methods.

#### 3. Verification & Uncertainty (`check_claims.py`, `CLAIMS_CHECK.md`)
- **File**: `check_claims.py`, `CLAIMS_CHECK.md`
- **What**:
  - Implemented automated verification script cross-referencing numerical claims against source CSVs.
  - Computed 95% bootstrap confidence intervals over the 20,000 evaluation molecules for epoch-60 online LiRA on three targets:
    - Target 0: AUC $[0.8962, 0.9045]$, TPR@1% $[0.2492, 0.3000]$
    - Target 1: AUC $[0.8870, 0.8961]$, TPR@1% $[0.2178, 0.2756]$
    - Target 2: AUC $[0.8930, 0.9012]$, TPR@1% $[0.2613, 0.3136]$
- **Why**: Ensure every number in the paper traces to underlying data.
