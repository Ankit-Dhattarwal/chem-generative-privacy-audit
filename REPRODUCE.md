# Reproduction Guide: SMILES Language Model Membership Audit

This document describes how to verify, analyze, and reproduce the experiments, tables, figures, and manuscript artifacts in this repository.

---

## 1. System Requirements & Environment Setup

### 1.1 Software & Hardware Prerequisites
- **Python**: Python 3.10 or 3.11 recommended.
- **Operating System**: Linux (Ubuntu 22.04 LTS / Google Colab) or macOS (Apple Silicon supported for evaluation / MPS).
- **GPU Acceleration**: NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (e.g., Tesla T4, V100, A100) and CUDA $\ge 12.0$ for the training stages.
- **TeX Engine**: `tectonic` (version $\ge 0.15$) for standalone PDF compilation.

### 1.2 Installation
```bash
# Clone the repository
git clone https://github.com/Ankit-Dhattarwal/chem-generative-privacy-audit.git
cd chem-generative-privacy-audit

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 2. Directory Structure and Data Locations

- `data/`:
  - `moses.csv`: Filtered subset of the MOSES molecular benchmark.
  - `pool.csv`: Evaluation universe pool (35,000 total candidate molecules; 20,000 active evaluation pool).
  - `val.csv`: Held-out validation split (5,000 molecules).
  - `pool_metadata.csv`: Scaffold definitions, frequencies, and randomized-SMILES variants (up to 10 distinct strings per compound, canonical included).
- `results_summary/`:
  - Primary CSV summaries serving as the canonical source of truth for all manuscript numbers.
  - See `results_summary/README.md` for column definitions.
- `results/cuda_arm/` (or Colab run outputs):
  - Model scoring parquets (`model_{k}_epoch{ep}.parquet`) containing per-molecule sequence loss values.
  - `split_control.csv`: Deterministic membership partition checks.
- `runs/cuda_arm/`:
  - Checkpoint files (`epoch5.pt`, `epoch20.pt`, `epoch60.pt`), RNG states, and `membership_mask.npy`.
- `paper/`:
  - `main.tex`: LaTeX source for the manuscript.
  - `MANUSCRIPT.md`: Markdown version kept in strict synchronization with `main.tex`.
  - `make_tables.py`: Automated table generator outputting to `paper/tables/`.
  - `generate_figures.py`: Publication figure generator outputting to `paper/figures/`.

---

## 3. Rerunning Experiment Stages (`colab_pipeline.py`)

The end-to-end training and scoring pipeline is orchestrated via `colab_pipeline.py`.

```bash
# Set environment variables (if running in Colab or on a remote workstation)
export PROJECT_ROOT="$(pwd)"

# Run baseline shadow models (32 shadows, 60 epochs)
python colab_pipeline.py --stages cuda_baseline

# Run dataset size scaling E3 (1k, 2.5k, 5k, 10k; 8 shadows per size)
python colab_pipeline.py --stages E3

# Run randomized-SMILES augmentation defense E5a (32 shadows, 60 epochs)
python colab_pipeline.py --stages E5a

# Run public pretraining then private fine-tuning
python colab_pipeline.py --stages pretrain_ft

# Run Differentially Private SGD (DP-SGD; eps=8 and eps=3; 8 shadows per eps)
python colab_pipeline.py --stages DP_SGD

# Run generative memorization audit (5,000 samples per model on MPS/CUDA)
python colab_pipeline.py --stages E6
```

### Estimated Compute Runtimes (NVIDIA Tesla T4)
| Stage | Description | Shadows | Epochs | Estimated Runtime |
|---|---|---|---|---|
| `cuda_baseline` | Baseline canonical models | 32 | 60 | ~40 min |
| `E3` | Dataset size scaling | 32 (4 sizes × 8) | 60 | ~25 min |
| `E5a` | Randomized SMILES training | 32 | 60 | ~45 min |
| `pretrain_ft` | Public pretrain (10k) $\to$ private FT | 8 | 60 + 20 | ~30 min |
| `DP_SGD` | Opacus DP-SGD ($\epsilon \in \{8, 3\}$) | 16 (2 eps × 8) | 60 | ~75 min |
| `E6` | Generative sampling audit | 3 models | 5k samples/model | ~15 min |
| **Total** | Full experimental suite | **121 models** | — | **~3.5–4.5 GPU hours** |

---

## 4. Regenerating Tables, Figures, and Verification Checks

To regenerate all tables and figures directly from the source CSVs without retraining:

```bash
# 1. Regenerate LaTeX and Markdown tables
python paper/make_tables.py

# 2. Regenerate publication-quality figures
python paper/generate_figures.py

# 3. Execute independent claims verifier (checks exact match across paper and PDF)
python scripts/verify_claims.py

# 4. Execute mutation testing suite (must detect all corrupted values)
python scripts/mutation_test.py

# 5. Compile the paper manuscript PDF
tectonic paper/main.tex --outdir paper/
```

### 4.1 Independent Claims Verification
- **What it verifies:** `scripts/verify_claims.py` loads `claims.yaml` and independently re-derives every expected string from the ground-truth CSVs in `results_summary/` using `pandas` exclusively. It checks that each exact string appears at its expected location in `paper/MANUSCRIPT.md`, `paper/main.tex`, `README.md`, `paper/tables/*`, and in the text extracted from `paper/main.pdf`. It also verifies that all numbers in these files are either registered claims or part of a strict whitelist of constants (e.g. model hyperparameter dimensions, epoch counts, table numbers, statistical significance thresholds).
- **Mutation Resistance:** `scripts/mutation_test.py` validates the verification pipeline by injecting synthetic mutations into the data and claim registry (swapping values between LiRA and LOSS, altering dataset sizes, modifying counts, changing standard deviations, and perturbing bootstrap CIs). The verification harness must catch and fail on all of these mutations.
