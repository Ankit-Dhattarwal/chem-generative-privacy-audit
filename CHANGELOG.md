# CHANGELOG

All modifications to the repository, manuscript, scripts, and tables are logged here in accordance with scientific auditing requirements. Every numerical claim is strictly derived from `results_summary/*.csv`.

---

### [2026-10-18] - Pre-Release Cleanup & Reviewer Audit Verification (v1.0.0)

#### 1. Randomized SMILES Variants Terminology (V1 / F1)
- **File**: `paper/make_tables.py`, `paper/tables/table1_baseline.{tex,md}`, `paper/tables/table3_defenses.{tex,md}`, `paper/main.tex`, `paper/MANUSCRIPT.md`, `README.md`, `REPRODUCE.md`, `results_summary/README.md`
- **Old -> New**: Replaced all remaining occurrences of "$K=10$" with "up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms), mean NLL".
- **Reason**: Aligns with `src/data.py` where variants start with canonical SMILES and molecules with high symmetry produce fewer than 10 distinct strings.

#### 2. Removal of Unsupported Valence Checks Phrasing (V2 / F2)
- **File**: `README.md`, `TODO_for_v2.md`, `CHANGELOG.md`
- **Old -> New**: Deleted all mentions of "failed standard valence checks". Replaced with: "we did not evaluate the generative utility of the DP models; an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense."
- **Reason**: No valence check logs or generation outputs exist for DP models in the repository.

#### 3. Stray Table Fragment Deletion (V3 / F3)
- **File**: `README.md`
- **Old -> New**: Removed duplicated Markdown table block following Table 3. Synchronized Table 1 and Table 3 blocks directly from `paper/tables/*.md`.
- **Reason**: Clean up typographical duplication.

#### 4. Claims Checker Scope & Mutation Testing (V4 / F4)
- **File**: `README.md`, `REPRODUCE.md`, `scripts/mutation_test_checker.py`
- **Old -> New**: Added dedicated documentation section "What `check_claims.py` Does and Does Not Verify". Created `scripts/mutation_test_checker.py` proving that swapping metrics across arms still exits 0, demonstrating set-membership verification vs semantic context. Audited whitelist-only numbers.
- **Reason**: Transparently document the verification boundaries and limitations of automated token checking.

#### 5. Defense Range & Statistical Non-Significance Phrasing (V5, V6 / F5)
- **File**: `paper/main.tex`, `paper/MANUSCRIPT.md`
- **Old -> New**: Updated Conclusion defense range from "$0.51 - 0.55$" to "$0.51 - 0.56$" (matching baseline variants offline LiRA AUC $0.560 \pm 0.007$). In Limitations #5, replaced "confirm that verbatim replication differences are not statistically significant" with "tests do not show a significant difference at this sample size ($p > 0.05$)".
- **Reason**: Precision in empirical ranges and formal statistical hypothesis testing terminology.

#### 6. Cross-Entropy Reduction & LiRA Invariance Justification (V7 / F6)
- **File**: `paper/main.tex`, `paper/MANUSCRIPT.md`
- **Old -> New**: Updated Section 2.1 to clarify that token-mean vs token-sum cross-entropy yields identical LiRA z-scores across all epochs due to per-target sequence length normalization.
- **Reason**: Accurately explain why `nll_sum` and `nll_mean` yield identical LiRA attack metrics.

#### 7. Parquet Directory CLI Parameter (V8 / F7)
- **File**: `scripts/make_bootstrap_ci.py`
- **Old -> New**: Replaced hardcoded absolute path `/Users/ankit/...` with `--parquet-dir` CLI argument defaulting to relative repository path. Added raw scoring parquet omission note to Limitations #4.
- **Reason**: Ensure script executes on any environment and document parquet omission.

#### 8. Table & Manuscript Synchronization (V11 / F10)
- **File**: `paper/MANUSCRIPT.md`, `README.md`, `paper/tables/*.md`
- **Old -> New**: Regenerated all tables via `paper/make_tables.py` and synchronized markdown tables across documents.
- **Reason**: Guarantee byte-for-byte consistency across all table representations.

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
- **Old -> New**: Removed unrecorded numerical validity claims ($0\%$ at $\epsilon=3$, $1\%$ at $\epsilon=8$) and reframed the DP arm everywhere as a preliminary exploration with utility collapse: chemical validity evaluations were not recorded in the released summary, and we did not evaluate the generative utility of the DP models (an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense).
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
