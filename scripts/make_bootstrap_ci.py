#!/usr/bin/env python3
"""
Compute 95% bootstrap confidence intervals over candidate molecules
for baseline epoch-60 online LiRA on three target models (0, 1, 2).
Uses the raw per-molecule scoring parquets from results/cuda_arm/baseline/.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_PARQUETS = Path(__file__).resolve().parent.parent / "results" / "cuda_arm" / "baseline"
OUT_CSV = Path(__file__).resolve().parent.parent / "results_summary" / "bootstrap_ci.csv"

parser = argparse.ArgumentParser(description="Compute bootstrap CIs for baseline online LiRA.")
parser.add_argument(
    "--parquet-dir",
    type=Path,
    default=DEFAULT_PARQUETS,
    help="Path to directory containing per-molecule scoring parquets (model_{k}_epoch60.parquet)",
)
args = parser.parse_args()
BASE_PARQUETS = args.parquet_dir

N_SHADOWS = 32
POOL_N = 20000

def _membership(n, k):
    rng = np.random.default_rng(k)
    idx = rng.choice(n, n // 2, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

masks = np.stack([_membership(POOL_N, k) for k in range(N_SHADOWS)])

# Load nll_sum from parquets
if not (BASE_PARQUETS / "model_0_epoch60.parquet").exists():
    raise FileNotFoundError(
        f"Parquets not found at {BASE_PARQUETS}. Raw per-molecule scoring parquets are omitted from "
        "the release repository due to storage constraints and are available on request."
    )

parquets = [pd.read_parquet(BASE_PARQUETS / f'model_{k}_epoch60.parquet') for k in range(N_SHADOWS)]
nll = np.stack([p['nll_sum'].to_numpy() for p in parquets])
K, N = nll.shape

def get_online_lira_scores(t):
    sh = np.arange(K) != t
    obs = nll[t]
    ins = masks[sh]
    vals = nll[sh]
    muout = np.array([np.mean(vals[~ins[:, i], i]) if (~ins[:, i]).any() else np.mean(vals[:, i]) for i in range(N)])
    mout = np.array([np.std(vals[~ins[:, i], i]) if (~ins[:, i]).sum() >= 2 else np.std(vals) for i in range(N)]) + 1e-6
    muin = np.array([np.mean(vals[ins[:, i], i]) if ins[:, i].any() else np.mean(vals[:, i]) for i in range(N)])
    min_ = np.array([np.std(vals[ins[:, i], i]) if ins[:, i].sum() >= 2 else np.std(vals) for i in range(N)]) + 1e-6
    score = -0.5 * ((obs - muin) / min_) ** 2 + np.log(mout / min_) + 0.5 * ((obs - muout) / mout) ** 2
    return score, masks[t]

def eval_metrics(labels, scores):
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    ix = np.where(fpr <= 0.01)[0]
    tpr1 = float(tpr[ix[-1]]) if len(ix) else 0.0
    return auc, tpr1

rng = np.random.default_rng(42)
B = 1000
records = []

for t in [0, 1, 2]:
    score, y = get_online_lira_scores(t)
    auc_pt, tpr1_pt = eval_metrics(y, score)
    boot_aucs = []
    boot_tpr1s = []
    for _ in range(B):
        idx = rng.choice(N, N, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        auc_b, tpr1_b = eval_metrics(y[idx], score[idx])
        boot_aucs.append(auc_b)
        boot_tpr1s.append(tpr1_b)
    
    auc_ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))
    tpr1_ci = (float(np.percentile(boot_tpr1s, 2.5)), float(np.percentile(boot_tpr1s, 97.5)))
    
    records.append({
        'target': t,
        'metric': 'online_LiRA_AUC',
        'point_estimate': round(auc_pt, 4),
        'ci_95_low': round(auc_ci[0], 4),
        'ci_95_high': round(auc_ci[1], 4),
        'ci_format': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"
    })
    records.append({
        'target': t,
        'metric': 'online_LiRA_TPR1',
        'point_estimate': round(tpr1_pt, 4),
        'ci_95_low': round(tpr1_ci[0], 4),
        'ci_95_high': round(tpr1_ci[1], 4),
        'ci_format': f"[{tpr1_ci[0]:.4f}, {tpr1_ci[1]:.4f}]"
    })

df = pd.DataFrame(records)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Generated {OUT_CSV}")
print(df)
