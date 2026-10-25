# Auditing Membership Leakage in Small SMILES Language Models: LiRA, Randomized-SMILES Training and DP-SGD (Preliminary Study)

**Author:** Ankit Dhattarwal  
**Affiliation:** Independent Research, Machine Learning and Computational Chemistry  
**Repository:** [github.com/Ankit-Dhattarwal/chem-generative-privacy-audit](https://github.com/Ankit-Dhattarwal/chem-generative-privacy-audit)  
**Date:** October 2026  

---

## Abstract

Autoregressive models trained on molecular SMILES strings are used to explore generative chemistry; some drug-discovery models may be trained on confidential compound collections. In this study, we empirically evaluate **Membership Inference Attacks (MIA)** against small causal SMILES language models (2 layers, $d = 128$, with 419,584 parameters, vocab size 25) on a subset of the MOSES benchmark across training epochs (5, 20, 60), dataset sizes ($N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$), and defenses.

Under canonical SMILES queries and sequence negative log-likelihood sum (`nll_sum`), models exhibit substantial membership leakage by epoch 60: online LiRA reaches a ROC-AUC of $0.898 \pm 0.003$ and a True Positive Rate (TPR) of $27.82 \pm 1.47\%$ at a 1% False Positive Rate (FPR), compared to $0.689 \pm 0.005$ AUC and $2.96 \pm 0.24\%$ TPR@1% FPR for standard LOSS. In smaller training cohorts ($N = 1,000$), attack sensitivity increases: LOSS AUC reaches $0.998 \pm 0.001$ (TPR@1% = $93.99 \pm 4.35\%$) and online LiRA AUC reaches $0.975 \pm 0.004$ (TPR@1% = $45.49 \pm 33.14\%$).

Mean-aggregated scoring over up to 10 string forms (canonical included) removes most of the signal (online LiRA AUC drops to $0.528 \pm 0.006$); this reflects a change in the query/scoring method for a model trained on canonical SMILES and is not a defense. Finally, in our preliminary DP-SGD arm (target $\epsilon \in \{8.0, 3.0\}$, $\delta = 10^{-5}$), generative utility was not evaluated, so the low attack AUC ($0.500 - 0.528$) cannot be interpreted as an effective defense. Training with randomized SMILES reduces canonical-query offline LiRA AUC to 0.528 ± 0.004 (variants scoring: 0.552 ± 0.005); validation NLL and generative validity of these models were not evaluated. Of 15,000 generated samples, 12 canonicalized outputs matched molecules in the training set and 7 matched molecules in the holdout set. Conditioning on these 19 matches, assuming equal match probability for the two equally sized reference sets, and treating the 19 match outcomes as independent under H0 (p = 0.5), the one-sided exact binomial p-value is 0.1796 and the two-sided exact binomial p-value is 0.3593. Neither result is significant at the 0.05 level.

---

## 1. Introduction

Autoregressive generative sequence models parameterize discrete token distributions to explore chemical space. By modeling the Simplified Molecular Input Line Entry System (SMILES), causal transformers can generate candidate structures and explore molecular libraries.

Some drug-discovery models are trained or fine-tuned on confidential compound collections. If a model is shared, an adversary with likelihood access may use membership inference to assess whether a target compound occurred in its training data, creating a potential confidentiality risk (Krüger et al., 2025; Shokri et al., 2017).

While membership inference has been widely investigated in computer vision and natural language processing, auditing molecular generative sequence models presents distinctive factors:
1. Molecules admit multiple non-unique equivalent SMILES representations (non-canonical string enumeration).
2. Chemical libraries often contain recurring core frameworks (Bemis-Murcko scaffolds) that induce structural correlation across compounds.
3. Language models may memorize specific surface syntax without learning generalized semantic invariants.

In this work, we present an empirical audit of membership leakage in small SMILES autoregressive models. Using a shadow-model ensemble framework based on Carlini et al.'s Likelihood-Ratio Attack (LiRA), we evaluate exposure across training duration, dataset scaling, transfer learning, and defensive interventions.

---

## 2. Threat Model and Formulation

### 2.1 Autoregressive Chemical Sequence Modeling
Let a molecular SMILES string be represented as a token sequence $x = (x_1, \dots, x_T) \in \mathcal{V}^T$, where $\mathcal{V}$ is a discrete vocabulary of chemical tokens (atoms, bonds, branches, ring closures). An autoregressive transformer parameterizes the likelihood:
$$P_\theta(x) = \prod_{t=1}^T P_\theta(x_t \mid x_{<t})$$

The sequence negative log-likelihood (NLL) sum is defined as:
$$\text{NLL}_{\text{sum}}(\theta; x) = - \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

Throughout this study, we consistently report and evaluate metrics using sequence NLL sum (`nll_sum`). In our experiments, `nll_sum` and `nll_mean` yield near-identical LiRA results across epochs 5, 20, and 60 (LiRA z-scores are mathematically invariant to per-molecule length scaling), while uncalibrated LOSS differs only marginally ($\Delta \text{AUC} \le 0.004$); we report `nll_sum` consistently throughout. Where indicated, we also report randomized-SMILES variant scoring (`nll_variants_mean`), which averages NLL across up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms), under mean aggregation (individual non-canonical forms were not scored separately).

### 2.2 Attack Formulations: LOSS vs. LiRA
Given a query molecule $x$, the adversary computes a test statistic $S(x)$ to predict whether $x \in \mathcal{D}_{\text{train}}$:

1. **Loss Thresholding (LOSS):**
   $$S_{\text{LOSS}}(x) = - \text{NLL}_{\text{sum}}(\mathcal{M}_{\text{target}}; x)$$
   LOSS assumes that training examples achieve lower sequence cross-entropy loss than non-training examples. However, intrinsic molecular complexity varies substantially: complex macrocycles or long branched chains exhibit higher baseline loss than simple aromatics regardless of membership.

2. **Calibrated Likelihood-Ratio Attack (LiRA):**
   To adjust for molecular complexity, LiRA trains an ensemble of $K$ shadow models $\{\mathcal{M}_k\}_{k=1}^K$ on random subsets of the data pool. For each query $x$, the models are partitioned into in-models $\mathcal{S}_{\text{in}}(x) = \{k : x \in \mathcal{D}_k\}$ and out-models $\mathcal{S}_{\text{out}}(x) = \{k : x \notin \mathcal{D}_k\}$.
   
   - **Offline LiRA (Out-only):** Estimates the out-distribution parameters $\mu_{\text{out}}(x)$ and $\sigma_{\text{out}}(x)$ over out-models:
     $$\Lambda_{\text{offline}}(x) = - \frac{\text{NLL}_{\text{sum}}(\mathcal{M}_{\text{target}}; x) - \mu_{\text{out}}(x)}{\sigma_{\text{out}}(x) + 10^{-6}}$$
   - **Online LiRA (In vs. Out):** Estimates both in-distribution $(\mu_{\text{in}}, \sigma_{\text{in}})$ and out-distribution $(\mu_{\text{out}}, \sigma_{\text{out}})$ parameters, computing the log-likelihood ratio under parametric Gaussian assumptions:
     $$\Lambda_{\text{online}}(x) = -\frac{1}{2}\left(\frac{\text{obs} - \mu_{\text{in}}}{\sigma_{\text{in}}}\right)^2 + \log\left(\frac{\sigma_{\text{out}}}{\sigma_{\text{in}}}\right) + \frac{1}{2}\left(\frac{\text{obs} - \mu_{\text{out}}}{\sigma_{\text{out}}}\right)^2$$

### 2.3 Threat Model Assumptions
We evaluate our models under an explicit, optimistic threat model:
- **Distribution Match:** Shadow models are sampled from the exact same marginal distribution as the target model (MOSES benchmark pool).
- **Likelihood Access:** The adversary has white-box or gray-box query access enabling exact calculation of token-level sequence log-likelihoods (`nll_sum`).
- **Canonical-String Queries:** The adversary queries the target and shadow models using standard canonical SMILES representations.

These represent strong assumptions favoring the attacker. In restricted black-box deployments (e.g., API rate limits, generation-only access, or query filtering), adversary effectiveness may be substantially lower.

---

## 3. Experimental Setup and Architecture

### 3.1 Model Architecture and Training Setup
All evaluations utilize a small causal GPT architecture:
- Number of transformer layers: 2
- Hidden embedding dimension ($d_{\text{model}}$): 128
- Attention heads: 4
- Dropout: 0.1
- Maximum context window: 128 tokens
- Parameter count: 419,584 parameters (exact count from `len(tokenizer.itos)`)
- Vocabulary size: 25 tokens

Models are trained using AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 0.01$), batch size 32, with gradient clipping norm 1.0 and a fixed sequence length of 128 tokens. Differentially private models (`DPCausalGPT`) evaluate the identical architecture (419,584 parameters) and utilize the same optimizer configuration (AdamW, $\text{lr} = 10^{-3}$, $\text{weight\_decay} = 0.01$, batch size 32, dropout $0.1$, $\text{max\_grad\_norm} = 1.0$) wrapped via Opacus.

### 3.2 Dataset and Shadow Model Partitioning
All experiments use the public MOSES benchmark (clean, drug-like molecules); “member” and “private” refer to an experimental split, not confidential real-world records.
- Evaluation pool size: $\text{POOL\_N} = 20,000$ molecules.
- Member selection: For each shadow model $k$, exactly $N = 10,000$ molecules are sampled uniformly at random without replacement using seed $k$ (`_membership(pool, k)` in `colab_pipeline.py`).
- Evaluation candidate universe: Evaluated across all 20,000 candidate compounds (10,000 members, 10,000 non-members for that shadow model).
- In the baseline and randomized-augmentation (E5a) experiments, 32 shadow models are trained ($k \in \{0, \dots, 31\}$). In data scaling (E3), DP-SGD, and pretrain-finetune experiments, 8 shadow models are trained per condition.

### 3.3 Opacus DP-SGD Implementation
Differentially private training is implemented using Opacus (version 1.6.0).
- Noise scale $\sigma$ is computed dynamically using `opacus.accountants.utils.get_noise_multiplier(target_epsilon=eps, target_delta=1e-5, sample_rate=32/N, epochs=60)`.
- Gradient clipping: `max_grad_norm = 1.0`.
- Target privacy parameters: $\epsilon \in \{8.0, 3.0\}$ with target $\delta = 10^{-5}$. The final saved accountant values were $\epsilon=7.158$ for target 8 and $\epsilon=2.750$ for target 3 across the eight models per setting.
- Recorded accountant totals: $ε=7.158$ for target 8 and $ε=2.750$ for target 3 at $δ=10^{-5}$, for all eight models in each setting.
- Implementation notes: The causal attention layer avoids fused operations incompatible with Opacus per-sample gradient hooks. If training resumes from a checkpoint, privacy accounting restarts from zero unless serialized accountant state is restored. These totals check the saved accountant state, not end-to-end correctness of the implementation or its assumptions.

---

## 4. Empirical Results and Findings

### 4.1 Baseline Leakage vs. Training Duration
Table 1 and Figure 1 present membership inference performance across 5, 20, and 60 training epochs on canonical SMILES representations evaluated using `nll_sum`.

| Epochs | Scoring Representation | Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|---|---|
| 5 | Canonical (nll_sum) | Loss Thresholding (LOSS) | 0.535 ± 0.004 | 1.23 ± 0.18 | 0.14 ± 0.07 |
| 5 | Canonical (nll_sum) | Offline LiRA (Out-only) | 0.630 ± 0.008 | 3.15 ± 0.36 | 0.42 ± 0.15 |
| 5 | Canonical (nll_sum) | Online LiRA (In vs. Out) | 0.595 ± 0.006 | 2.12 ± 0.35 | 0.19 ± 0.09 |
| 5 | Canonical (nll_sum) | **Strongest: offline LiRA** | **0.630 ± 0.008** | **3.15 ± 0.36** | **0.42 ± 0.15** |
| 20 | Canonical (nll_sum) | Loss Thresholding (LOSS) | 0.591 ± 0.004 | 1.66 ± 0.21 | 0.21 ± 0.08 |
| 20 | Canonical (nll_sum) | Offline LiRA (Out-only) | 0.763 ± 0.007 | 9.58 ± 0.84 | 1.86 ± 0.47 |
| 20 | Canonical (nll_sum) | Online LiRA (In vs. Out) | 0.760 ± 0.006 | 8.66 ± 0.62 | 1.34 ± 0.33 |
| 20 | Canonical (nll_sum) | **Strongest: offline LiRA** | **0.763 ± 0.007** | **9.58 ± 0.84** | **1.86 ± 0.47** |
| 60 | Canonical (nll_sum) | Loss Thresholding (LOSS) | 0.689 ± 0.005 | 2.96 ± 0.24 | 0.41 ± 0.15 |
| 60 | Canonical (nll_sum) | Offline LiRA (Out-only) | 0.880 ± 0.004 | 27.15 ± 1.63 | 8.22 ± 1.32 |
| 60 | Canonical (nll_sum) | Online LiRA (In vs. Out) | 0.898 ± 0.003 | 27.82 ± 1.47 | 8.15 ± 1.58 |
| 60 | Canonical (nll_sum) | **Strongest: online LiRA** | **0.898 ± 0.003** | **27.82 ± 1.47** | **8.15 ± 1.58** |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Loss Thresholding (LOSS) | 0.511 ± 0.004 | 1.09 ± 0.12 | 0.12 ± 0.05 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Offline LiRA (Out-only) | 0.560 ± 0.007 | 2.01 ± 0.29 | 0.30 ± 0.10 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | Online LiRA (In vs. Out) | 0.528 ± 0.006 | 1.45 ± 0.47 | 0.18 ± 0.12 |
| 60 (variants) | Randomized-SMILES (up to 10 strings, mean NLL) | **Strongest: offline LiRA** | **0.560 ± 0.007** | **2.01 ± 0.29** | **0.30 ± 0.10** |

*Table 1: Baseline membership inference vulnerability across epochs. Reported values denote mean ± standard deviation across 32 leave-one-out shadow target evaluations (not bootstrap confidence intervals).*

**Findings:**
- At epoch 5, membership leakage is modest (strongest attack: offline LiRA AUC $0.630 \pm 0.008$).
- By epoch 60, extensive exposure occurs: online LiRA achieves ROC-AUC $0.898 \pm 0.003$, with TPR@1% FPR reaching $27.82 \pm 1.47\%$. At a stricter threshold of 0.1% FPR, online LiRA still identifies $8.15 \pm 1.58\%$ of member molecules.
- Standard loss thresholding substantially underestimates vulnerability (AUC 0.689 vs. 0.898 for online LiRA at epoch 60).
- **Statistical Uncertainty:** To supplement the cross-target standard deviations, we computed 95% bootstrap confidence intervals over the 20,000 evaluation molecules for epoch-60 online LiRA on three target models (not independent: they share shadow models and overlapping training pools):
  - Target 0: AUC $0.9004$, 95% CI $[0.8962, 0.9045]$; TPR@1% $0.2684$, 95% CI $[0.2492, 0.3000]$.
  - Target 1: AUC $0.8916$, 95% CI $[0.8870, 0.8961]$; TPR@1% $0.2510$, 95% CI $[0.2178, 0.2756]$.
  - Target 2: AUC $0.8972$, 95% CI $[0.8930, 0.9012]$; TPR@1% $0.2838$, 95% CI $[0.2613, 0.3136]$.

### 4.2 String-Form Sensitivity: Randomized-SMILES Scoring
In our setting, mean-aggregated scoring over up to 10 string forms (canonical included) removes most of the signal. In `colab_pipeline.py`, each molecule was scored at epoch 60 using up to 10 distinct SMILES strings per molecule (the canonical form plus randomized forms generated via RDKit `Chem.MolToSmiles(mol, doRandom=True, canonical=False)`), with the average NLL recorded as `nll_variants_mean`.

As reported in Table 1 (bottom section):
- Scoring canonical models with randomized variants drops LOSS AUC from $0.689$ to $0.511 \pm 0.004$.
- Offline LiRA AUC drops from $0.880$ to $0.560 \pm 0.007$.
- Online LiRA AUC drops from $0.898$ to $0.528 \pm 0.006$.

We did not score individual non-canonical strings, so we cannot say whether single randomized forms leak.

### 4.3 Private Dataset Scaling (Experiment E3)
We examine vulnerability across dataset sizes $N \in \{1\text{k}, 2.5\text{k}, 5\text{k}, 10\text{k}\}$ trained for 60 epochs, with 8 shadow models per size ($2N$ candidate universe). Results are presented in Table 2 and Figure 2, separating LOSS and LiRA attacks.

| Private Size ($N$) | Candidate Universe ($2N$) | LOSS AUC | LOSS TPR@1% (%) | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) |
|---|---|---|---|---|---|---|---|
| 1,000 | 2,000 | 0.998 ± 0.001 | 93.99 ± 4.35 | 0.957 ± 0.006 | 20.75 ± 6.51 | 0.975 ± 0.004 | 45.49 ± 33.14 |
| 2,500 | 5,000 | 0.941 ± 0.006 | 27.68 ± 3.97 | 0.936 ± 0.003 | 12.85 ± 2.97 | 0.939 ± 0.003 | 6.53 ± 1.60 |
| 5,000 | 10,000 | 0.820 ± 0.010 | 7.99 ± 1.55 | 0.898 ± 0.008 | 8.19 ± 1.05 | 0.882 ± 0.006 | 3.23 ± 0.60 |
| 10,000 | 20,000 | 0.689 ± 0.003 | 2.84 ± 0.09 | 0.827 ± 0.006 | 4.81 ± 0.63 | 0.785 ± 0.003 | 2.07 ± 0.28 |

*Table 2: Dataset size scaling (E3). Evaluated across 8 shadow models per size. Note: LiRA exhibits estimation instability at small $N$. We report empirical observations without claiming a parametric power law.*

**Observations:**
- Smaller training cohorts experience pronounced overfitting at 60 epochs. At $N=1,000$, standard LOSS reaches AUC $0.998 \pm 0.001$ with TPR@1% = $93.99 \pm 4.35\%$, while offline LiRA reaches AUC $0.957 \pm 0.006$ (TPR@1% = $20.75 \pm 6.51\%$) and online LiRA reaches AUC $0.975 \pm 0.004$ (TPR@1% = $45.49 \pm 33.14\%$).
- With only 8 shadow models, LiRA density estimation at small $N$ suffers from high variance (e.g., online LiRA TPR@1% standard deviation is $\pm 33.14\%$). We report these empirical numbers directly and do not claim a formal parametric scaling law.

### 4.4 Transfer Learning: Pretraining then Fine-Tuning
We evaluated a transfer learning setting where a model was pretrained on 10,000 public molecules for 60 epochs, then fine-tuned on 10,000 private molecules for 20 epochs across 8 shadow targets (Table 3).

| Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|
| Loss Thresholding (LOSS) | 0.640 ± 0.003 | 2.31 ± 0.26 | 0.34 ± 0.08 |
| Offline LiRA | 0.801 ± 0.003 | 4.36 ± 0.51 | 0.43 ± 0.26 |
| Online LiRA | 0.751 ± 0.004 | 2.00 ± 0.22 | 0.22 ± 0.05 |

*Table 3: Pretrain $\to$ fine-tune scenario (8 target models (not independent: they share shadow models and overlapping training pools), `nll_sum`). Note: this experiment lacks an isolated from-scratch control at the same size and epoch count.*

Offline LiRA achieves an AUC of $0.801 \pm 0.003$ and TPR@1% of $4.36 \pm 0.51\%$. While lower than baseline epoch 60 ($0.880 \pm 0.004$ / $27.15 \pm 1.63\%$), offline LiRA AUC is higher than the baseline trained from scratch for 20 epochs ($0.763 \pm 0.007$ / $9.58 \pm 0.84\%$, though TPR@1% is lower), indicating that fine-tuning on private data remains susceptible to membership inference. However, there is no matched from-scratch control trained on the identical 10,000 private compounds for 20 epochs without pretraining, so the isolated effect of pretraining cannot be disentangled.

### 4.5 Defense Comparison at Epoch 60
Table 4 and Figure 3 compare two defensive approaches at epoch 60:
1. **Randomized-SMILES Training (E5a):** Models are trained with random-SMILES data augmentation (`Chem.MolToSmiles(mol, doRandom=True, canonical=False)` applied per batch).
2. **DP-SGD:** Differentially private training targeting $\epsilon = 8.0$ and $\epsilon = 3.0$ with $\delta = 10^{-5}$.

#### Panel (a): Calibrated Likelihood-Ratio Attacks (LiRA)

| Defense Configuration | Shadow Setup | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) | Strongest Attack |
|---|---|---|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.880 ± 0.004 | 27.15 ± 1.63 | 0.898 ± 0.003 | 27.82 ± 1.47 | **Online LiRA** |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.528 ± 0.004 | 1.38 ± 0.21 | 0.508 ± 0.004 | 1.09 ± 0.17 | **Offline LiRA** |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.552 ± 0.005 | 1.56 ± 0.17 | 0.519 ± 0.005 | 1.14 ± 0.18 | **Offline LiRA** |
| DP-SGD (target ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.528 ± 0.003 | 1.20 ± 0.14 | 0.501 ± 0.005 | 1.09 ± 0.16 | **Offline LiRA** |
| DP-SGD (target ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.520 ± 0.003 | 1.12 ± 0.06 | 0.500 ± 0.002 | 1.01 ± 0.10 | **Offline LiRA** |

#### Panel (b): Uncalibrated Loss Thresholding Baseline

| Defense Configuration | Shadow Setup | LOSS AUC | LOSS TPR@1% (%) |
|---|---|---|---|
| Baseline (Unmitigated) | 32 shadows | 0.689 ± 0.005 | 2.96 ± 0.24 |
| Randomized-SMILES Training (Canonical Scoring) | 32 shadows | 0.506 ± 0.004 | 1.08 ± 0.16 |
| Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL) | 32 shadows | 0.506 ± 0.004 | 1.10 ± 0.16 |
| DP-SGD (target ε=8.0, δ=1e-5)* | 8 shadows; prelim. | 0.505 ± 0.004 | 1.13 ± 0.15 |
| DP-SGD (target ε=3.0, δ=1e-5)* | 8 shadows; prelim. | 0.503 ± 0.004 | 1.10 ± 0.06 |

*Table 4: Defense comparison at epoch 60. Evaluated with `nll_sum` (except variants scoring). *Note: preliminary, utility not measured.*

**Defensive Analysis:**
- Randomized-SMILES training reduces canonical attack performance: offline LiRA drops from $0.880$ to $0.528 \pm 0.004$, and TPR@1% drops from $27.15\%$ to $1.38\%$. When attacked via randomized variants scoring, offline LiRA AUC remains low at $0.552 \pm 0.005$.
- In preliminary runs, DP-SGD reduces attack AUC near random guess ($0.500 - 0.528$). However, generative utility was not evaluated: an attack AUC near 0.5 can arise from an under-trained model rather than an effective defense, and privacy-utility tradeoffs cannot be established without verifying chemical validity. This represents a methodological caution rather than a measured finding of this paper.

### 4.6 Generative Memorization Audit (Experiment E6)
To audit whether models generate molecules matching the canonicalized structures in the training set, we sampled 5,000 molecules per model for three baseline models (Models 0, 15, 31; total 15,000 samples) on Apple Silicon (MPS). Generated molecules were evaluated for chemical validity, uniqueness, and canonicalized structure matches against 10,000 training members and 10,000 holdout molecules (Table 5).

| Model ID | Validity (%) | Uniqueness (%) | Novelty vs Members (%) | Training Matches (Count) | Holdout Matches (Count) | Binomial p-value ($p=0.5$) | Significant at α=0.05? |
|---|---|---|---|---|---|---|---|
| Model 0 | 60.24% | 99.97% | 99.97% | 1 | 1 | 0.7500 | No ($p > 0.05$) |
| Model 15 | 62.26% | 99.87% | 99.90% | 3 | 2 | 0.5000 | No ($p > 0.05$) |
| Model 31 | 59.10% | 99.90% | 99.73% | 8 | 4 | 0.1938 | No ($p > 0.05$) |
| **Mean rates; pooled hits (15,000 samples)** | **60.53%** | **99.91%** | **99.87%** | **12** | **7** | **0.1796** | **No ($p > 0.05$)** |

*Table 5: Quality metrics in the summary row are unweighted means of the three per-model rates; match counts are pooled. Of 15,000 generated samples, 12 canonicalized outputs matched molecules in the training set and 7 matched molecules in the holdout set. Conditioning on these 19 matches, assuming equal match probability for the two equally sized reference sets, and treating the 19 match outcomes as independent under H0 (p = 0.5), the one-sided exact binomial p-value is 0.1796 and the two-sided exact binomial p-value is 0.3593. Neither result is significant at the 0.05 level.*

**Findings:**
- Of 15,000 generated samples, 12 canonicalized outputs matched molecules in the training set and 7 matched molecules in the holdout set. Conditioning on these 19 matches, assuming equal match probability for the two equally sized reference sets, and treating the 19 match outcomes as independent under H0 (p = 0.5), the one-sided exact binomial p-value is 0.1796 and the two-sided exact binomial p-value is 0.3593. Neither result is significant at the 0.05 level. Individual model one-sided binomial p-values are $0.750$, $0.500$, and $0.194$. The pooled test treats the 19 observed matches as independent Bernoulli observations under the stated null and should be read as exploratory.
- Under the evaluated sample budget (5,000 per model), the difference between member copies and holdout copies is **not statistically significant**. The non-significant result does not establish that the model did not memorize training molecules.

---

## 5. Related Work

Our work interfaces with privacy auditing, generative chemistry, and membership inference:

- **Membership Inference Attacks:** Carlini et al. (2022) established Likelihood-Ratio Attacks (LiRA) to overcome sequence hardness confounding in classification and NLP. Shokri et al. (2017) originally formulated shadow-model attacks. Recent theoretical analysis by Zhang et al. (2024; arXiv:2409.19798) emphasized that empirical membership inference attacks face significant evidentiary caveats and cannot formally prove that a specific sample was present in a model's training set without known reference baselines.
- **Privacy in Cheminformatics & Molecular Models:** Krüger et al. (2025; *J. Cheminformatics* 17:38, `molprivacy`) investigated membership leakage in molecular property predictors, demonstrating that publishing neural network weights poses data confidentiality risks. Aubinais et al. (2026; AISTATS, PMLR 300:649–657) proposed an attack-independent indicator for quantized models and evaluated it on molecular modeling tasks. Ross et al. (2025; *Digital Discovery* 4:2684–2696) developed GP-MoLFormer, pretrained on a corpus of 1.1 billion SMILES strings, including 650 million unique strings.
- **Privacy Auditing in Biological Sequence Models:** Nemecek et al. (2026; ACM BCB, DOI: 10.1145/3807503.3819470) evaluated genomic language models using perplexity-based detection, canary extraction, and membership inference, illustrating domain-specific privacy risks.
- **SMILES Randomization & Data Augmentation:** Arús-Pous et al. (2019; *J. Cheminformatics* 11:71) and Bjerrum (2017) demonstrated that randomized SMILES enumeration expands the effective representation space and regularizes generative models against overfitting.
- **Differential Privacy in Deep Learning:** Abadi et al. (2016) and the Opacus library (Yousefpour et al., 2021) formalized differential privacy via DP-SGD and R\'enyi Differential Privacy (RDP) accounting.

Our contribution provides a focused empirical audit of small causal SMILES language models across training epochs, dataset sizes, randomized representation scoring, and DP-SGD training. We do not claim to present the first or definitive study, but rather a transparent, reproducible empirical audit of these dynamics in small chemical generative LMs.

---

## 6. Limitations

This preliminary study has several explicit limitations:
1. **Model Capacity and Architecture:** Models evaluate a 2-layer, $d=128$, $\sim 0.4\text{M}$ parameter GPT with modest chemical validity ($\sim 60\%$). These findings may not generalize directly to larger models (e.g., ChemBERTa or MegaMolBART).
2. **Dataset Scope:** Experiments are restricted to a single filtered subset of the MOSES benchmark (clean, drug-like compounds). Diverse proprietary libraries with differing scaffold distributions may exhibit different leakage profiles.
3. **Shadow Ensemble Scale Asymmetry:** While baseline and E5a arms used 32 shadow models, compute limits restricted E3, DP-SGD, and pretrain-FT to 8 shadow models, leading to higher estimation variance.
4. **Reported Uncertainty:** Metrics report standard deviation across leave-one-out shadow target models, rather than per-molecule bootstrap confidence intervals. Our computed 95% bootstrap CIs across molecules for epoch 60 online LiRA on three targets span roughly $\pm 0.005$ in AUC and $\pm 0.03$ in TPR@1% (raw per-molecule scoring parquets are omitted from the release repository due to storage constraints and are available on request).
5. **Generative Sample Size and Hardware:** Experiment E6 sampled 5,000 molecules per model (15,000 total) and ran on Apple Silicon (MPS), whereas shadow training was conducted on CUDA Tesla T4. Statistical tests do not show a significant difference at this sample size ($p > 0.05$).
6. **DP-SGD Generative Utility:** The DP-SGD arm represents a preliminary exploration where utility was not evaluated for DP checkpoints; an attack AUC near 0.5 may reflect an under-trained model rather than an effective defense. The recorded accountant totals are ε=7.158 for target 8 and ε=2.750 for target 3 at δ=10⁻⁵. This checks the saved accountant totals, not the end-to-end correctness of the training implementation or its assumptions.
7. **Lack of From-Scratch Control:** The pretrain $\to$ fine-tune experiment lacks an exact from-scratch baseline trained for 20 epochs on 10,000 molecules, preventing full disentanglement of pretraining effects from training duration.
8. **Randomized-SMILES Utility Evaluation:** Validation NLL and generative chemical validity of randomized-SMILES-trained models were not assessed.

---

## 7. Conclusion
 
This preliminary study audited membership leakage in small SMILES language models. We observed that under canonical queries and likelihood access, models overfit over 60 epochs, enabling calibrated LiRA attacks to reach an AUC of 0.898. In our setting, scoring with mean NLL over up to 10 string forms (canonical included) removes most of the signal; training with randomized SMILES also reduces attack AUC to $0.51 - 0.56$. While DP-SGD is designed to bound the influence of individual training examples, preliminary implementations exhibited possible loss of generative utility (not evaluated); an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense. Future work must investigate scaling laws on larger architectures, higher-power generative sampling, and privacy-preserving training that maintains chemical validity.

---

## Acknowledgments

AI coding and writing assistants were used during code development, table generation and manuscript editing. The author designed the experiments, reviewed all code, results and text, and takes full responsibility for the content.

---

## References

1. N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramèr. "Membership Inference Attacks From First Principles." *IEEE S&P*, 2022.
2. M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. "Deep Learning with Differential Privacy." *ACM CCS*, 2016.
3. D. Polykovskiy et al. "Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models." *Frontiers in Pharmacology*, 11:565644, 2020.
4. D. Weininger. "SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules." *JCICS*, 28(1):31–36, 1988.
5. R. Shokri, M. Stronati, C. Song, and V. Shmatikov. "Membership inference attacks against machine learning models." *IEEE S&P*, 2017.
6. G. W. Bemis and M. A. Murcko. "The properties of known drugs. 1. Molecular frameworks." *J. Med. Chem.*, 39(15):2887–2893, 1996.
7. E. J. Bjerrum. "SMILES enumeration as data augmentation for neural network modeling of molecules." *arXiv:1703.07076*, 2017.
8. A. Yousefpour et al. "Opacus: User-friendly differential privacy library in PyTorch." *arXiv:2109.12298*, 2021.
9. F. P. Krüger, J. Östman, L. Mervin, I. V. Tetko, and O. Engkvist. "Publishing neural networks in drug discovery might compromise training data privacy." *Journal of Cheminformatics*, 17:38, 2025.
10. E. Aubinais, P. Formont, P. Piantanida, and E. Gassiat. "An Indicator of Membership Inference Security in Post-Training Quantized Models." *Proceedings of the 29th International Conference on Artificial Intelligence and Statistics*, PMLR 300:649–657, 2026.
11. J. Ross, B. Belgodere, S. C. Hoffman, V. Chenthamarakshan, J. Navratil, Y. Mroueh, and P. Das. "GP-MoLFormer: a foundation model for molecular generation." *Digital Discovery*, 4(10):2684–2696, 2025.
12. A. Nemecek, W. Li, X. Jiang, J. Vaidya, and E. Ayday. "Quantifying Memorization and Privacy Risks in Genomic Language Models." *Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics*, 2026. https://doi.org/10.1145/3807503.3819470.
13. J. Arús-Pous, S. V. Johansson, O. Prykhodko, E. J. Bjerrum, C. Tyrchan, J.-L. Reymond, H. Chen, and O. Engkvist. "Randomized SMILES strings improve the quality of molecular generative models." *Journal of Cheminformatics*, 11:71, 2019.
14. J. Zhang, D. Das, G. Kamath, and F. Tramèr. "Membership Inference Attacks Cannot Prove that a Model Was Trained On Your Data." *arXiv:2409.19798*, 2024.
