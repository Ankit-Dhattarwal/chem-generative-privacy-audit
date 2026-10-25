#!/usr/bin/env python3
"""
Generates claims.yaml mapping all numbers in:
- Abstract
- Sections 2-7
- Tables 1-5
- Figure Captions
- README tables
to the ground truth CSVs.
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_summary"

df_base = pd.read_csv(RESULTS_DIR / "baseline_attack_summary.csv")
df_e3 = pd.read_csv(RESULTS_DIR / "E3_scaling_summary.csv")
df_e5a = pd.read_csv(RESULTS_DIR / "E5a_augmentation_summary.csv")
df_dp = pd.read_csv(RESULTS_DIR / "DP_SGD_summary.csv")
df_pt = pd.read_csv(RESULTS_DIR / "pretrain_ft_summary.csv")
df_e6 = pd.read_csv(RESULTS_DIR / "E6_generative_audit_summary.csv")
df_ci = pd.read_csv(RESULTS_DIR / "bootstrap_ci.csv")

claims = []

# ==============================================================================
# 1. Baseline Leakage (Table 1, Abstract, Sec 4.1, README Table 1)
# ==============================================================================
for _, r in df_base.iterrows():
    ep = int(r["epoch"])
    stat = str(r["stat"])
    method = str(r["method"])
    for metric in ["auc", "tpr1", "tpr01"]:
        is_pct = metric.startswith("tpr")
        cid = f"base_ep{ep}_{stat}_{method}_{metric}"
        claims.append({
            "id": cid,
            "section": "Sec 4.1 / Table 1",
            "source_csv": "results_summary/baseline_attack_summary.csv",
            "row_filters": {"epoch": ep, "stat": stat, "method": method},
            "metric": metric,
            "mean_col": f"{metric}_mean",
            "std_col": f"{metric}_std",
            "is_percentage": is_pct,
            "mean_val": float(r[f"{metric}_mean"]),
            "std_val": float(r[f"{metric}_std"])
        })

# ==============================================================================
# 2. Bootstrap CIs (Sec 4.1)
# ==============================================================================
for _, r in df_ci.iterrows():
    tgt = int(r["target"])
    metric = str(r["metric"])
    cid = f"bootstrap_target_{tgt}_{metric}"
    claims.append({
        "id": cid,
        "section": "Sec 4.1",
        "source_csv": "results_summary/bootstrap_ci.csv",
        "row_filters": {"target": tgt, "metric": metric},
        "metric": metric,
        "point_estimate": float(r["point_estimate"]),
        "ci_low": float(r["ci_95_low"]),
        "ci_high": float(r["ci_95_high"]),
        "ci_format": str(r["ci_format"])
    })

# ==============================================================================
# 3. Dataset Scaling (Table 2, Abstract, Sec 4.3)
# ==============================================================================
for _, r in df_e3.iterrows():
    size = int(r["member_size"])
    method = str(r["method"])
    for metric in ["auc", "tpr1"]:
        is_pct = metric.startswith("tpr")
        cid = f"e3_scaling_size_{size}_{method}_{metric}"
        claims.append({
            "id": cid,
            "section": "Sec 4.3 / Table 2",
            "source_csv": "results_summary/E3_scaling_summary.csv",
            "row_filters": {"member_size": size, "method": method},
            "metric": metric,
            "mean_col": f"{metric}_mean",
            "std_col": f"{metric}_std",
            "is_percentage": is_pct,
            "mean_val": float(r[f"{metric}_mean"]),
            "std_val": float(r[f"{metric}_std"])
        })

# ==============================================================================
# 4. Transfer Learning (Table 3 / Table 4 in make_tables, Sec 4.4, Abstract)
# ==============================================================================
for method in ["LOSS", "offline_LiRA", "online_LiRA"]:
    sub = df_pt[(df_pt["method"] == method) & (df_pt["stat"] == "nll_sum")]
    for metric, col_name in [("auc", "auc"), ("tpr1", "tpr_fpr_1pct"), ("tpr01", "tpr_fpr_0_1pct")]:
        is_pct = metric.startswith("tpr")
        cid = f"pretrain_ft_{method}_{metric}"
        claims.append({
            "id": cid,
            "section": "Sec 4.4 / Table 3",
            "source_csv": "results_summary/pretrain_ft_summary.csv",
            "row_filters": {"method": method, "stat": "nll_sum"},
            "metric": metric,
            "column": col_name,
            "is_percentage": is_pct,
            "mean_val": float(sub[col_name].mean()),
            "std_val": float(sub[col_name].std(ddof=1))
        })

# ==============================================================================
# 5. Defenses Comparison (Table 4, Abstract, Sec 4.5, README Table 4)
# ==============================================================================
# E5a Randomized-SMILES
for _, r in df_e5a.iterrows():
    ep = int(r["epoch"])
    stat = str(r["stat"])
    method = str(r["method"])
    for metric in ["auc", "tpr1"]:
        is_pct = metric.startswith("tpr")
        cid = f"e5a_defense_ep{ep}_{stat}_{method}_{metric}"
        claims.append({
            "id": cid,
            "section": "Sec 4.5 / Table 4",
            "source_csv": "results_summary/E5a_augmentation_summary.csv",
            "row_filters": {"epoch": ep, "stat": stat, "method": method},
            "metric": metric,
            "mean_col": f"{metric}_mean",
            "std_col": f"{metric}_std",
            "is_percentage": is_pct,
            "mean_val": float(r[f"{metric}_mean"]),
            "std_val": float(r[f"{metric}_std"])
        })

# DP-SGD
for _, r in df_dp.iterrows():
    eps = float(r["epsilon_target"])
    method = str(r["method"])
    for metric in ["auc", "tpr1"]:
        is_pct = metric.startswith("tpr")
        cid = f"dp_sgd_defense_eps{eps}_{method}_{metric}"
        claims.append({
            "id": cid,
            "section": "Sec 4.5 / Table 4",
            "source_csv": "results_summary/DP_SGD_summary.csv",
            "row_filters": {"epsilon_target": eps, "method": method},
            "metric": metric,
            "mean_col": f"{metric}_mean",
            "std_col": f"{metric}_std",
            "is_percentage": is_pct,
            "mean_val": float(r[f"{metric}_mean"]),
            "std_val": float(r[f"{metric}_std"])
        })

# ==============================================================================
# 6. Generative Memorization Audit (Table 5, Sec 4.6, Abstract)
# ==============================================================================
tot_m = 0
tot_o = 0
vals, uniqs, novs = [], [], []

from scipy.stats import binomtest

for _, r in df_e6.iterrows():
    m = int(r["model"])
    mc = int(round(r["exact_member_copy_fraction_all_samples"] * int(r["samples"])))
    oc = int(round(r["exact_out_copy_fraction_all_samples"] * int(r["samples"])))
    pval = float(binomtest(mc, mc + oc, 0.5, alternative="greater").pvalue)
    tot_m += mc
    tot_o += oc
    vals.append(r["validity"] * 100)
    uniqs.append(r["uniqueness_valid"] * 100)
    novs.append(r["novelty_vs_members"] * 100)
    claims.append({
        "id": f"e6_model_{m}",
        "section": "Sec 4.6 / Table 5",
        "source_csv": "results_summary/E6_generative_audit_summary.csv",
        "row_filters": {"model": m},
        "validity_pct": float(r["validity"] * 100),
        "uniqueness_pct": float(r["uniqueness_valid"] * 100),
        "novelty_pct": float(r["novelty_vs_members"] * 100),
        "member_copies": mc,
        "out_copies": oc,
        "binomial_pvalue": pval
    })

pooled_pval = float(binomtest(tot_m, tot_m + tot_o, 0.5, alternative="greater").pvalue)

claims.append({
    "id": "e6_pooled",
    "section": "Sec 4.6 / Table 5",
    "source_csv": "results_summary/E6_generative_audit_summary.csv",
    "row_filters": {"pooled": True},
    "validity_pct": float(np.mean(vals)),
    "uniqueness_pct": float(np.mean(uniqs)),
    "novelty_pct": float(np.mean(novs)),
    "member_copies": tot_m,
    "out_copies": tot_o,
    "binomial_pvalue": pooled_pval,
})

registry = {
    "version": "1.0",
    "description": "Exhaustive Claim Registry mapping all empirical numbers to source CSVs",
    "claims": claims
}

with open(REPO_ROOT / "claims.yaml", "w") as f:
    yaml.dump(registry, f, sort_keys=False)

print(f"Generated claims.yaml with {len(claims)} claim entries.")
