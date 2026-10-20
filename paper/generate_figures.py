#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for SMILES MIA Audit.
Strictly generated from results_summary/*.csv files:
- Fig 1: Baseline attack vulnerability vs epochs (LOSS, offline LiRA, online LiRA using nll_sum)
- Fig 2: Dataset scaling (E3: 1k, 2.5k, 5k, 10k) with LOSS and LiRA shown separately
- Fig 3: Defense comparison at epoch 60 showing the strongest attack per defense
- Fig 4: Generative audit (E6) exact member copies vs out copies and chemical quality
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font and aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results_summary"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: Baseline Attack Vulnerability vs Epochs
# -------------------------------------------------------------
df_base = pd.read_csv(RESULTS_DIR / "baseline_attack_summary.csv")
df_base_sum = df_base[df_base['stat'] == 'nll_sum']

epochs = [5, 20, 60]
methods = ['LOSS', 'offline_LiRA', 'online_LiRA']
method_labels = {
    'LOSS': 'Loss Thresholding (LOSS)',
    'offline_LiRA': 'Offline LiRA (Out-only)',
    'online_LiRA': 'Online LiRA (In vs. Out)'
}
colors = {'LOSS': '#555555', 'offline_LiRA': '#1f77b4', 'online_LiRA': '#d62728'}
markers = {'LOSS': 'o', 'offline_LiRA': 's', 'online_LiRA': '^'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for m in methods:
    sub = df_base_sum[df_base_sum['method'] == m].sort_values('epoch')
    ax1.errorbar(sub['epoch'], sub['auc_mean'], yerr=sub['auc_std'], label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.0, capsize=4, markersize=6.5)
    ax2.errorbar(sub['epoch'], sub['tpr1_mean'] * 100, yerr=sub['tpr1_std'] * 100, label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.0, capsize=4, markersize=6.5)

ax1.set_xlabel('Training Epochs')
ax1.set_ylabel('Attack ROC-AUC')
ax1.set_title('(a) Membership Leakage (AUC) vs Training Epochs')
ax1.set_ylim(0.48, 0.95)
ax1.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Random Baseline (0.50)')
ax1.set_xticks(epochs)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2.set_xlabel('Training Epochs')
ax2.set_ylabel('True Positive Rate @ 1% FPR (%)')
ax2.set_title('(b) High-Confidence Leakage (TPR @ 1% FPR)')
ax2.set_ylim(-1, 32)
ax2.set_xticks(epochs)
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_exposure_epochs.png")
fig.savefig(FIG_DIR / "fig1_exposure_epochs.pdf")
plt.close(fig)
print("Figure 1 generated.")

# -------------------------------------------------------------
# FIGURE 2: Dataset Scaling (E3) - LOSS and LiRA Shown Separately
# -------------------------------------------------------------
df_e3 = pd.read_csv(RESULTS_DIR / "E3_scaling_summary.csv")
sizes = [1000, 2500, 5000, 10000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Subplot (a): LOSS Attack Scaling (AUC and TPR@1%)
sub_loss = df_e3[df_e3['method'] == 'LOSS'].sort_values('member_size')
color_loss_auc = '#1b9e77'
color_loss_tpr = '#d95f02'

ax1.plot(sub_loss['member_size'], sub_loss['auc_mean'], color=color_loss_auc, marker='o', lw=2.2, label='LOSS ROC-AUC (Left Axis)')
ax1.errorbar(sub_loss['member_size'], sub_loss['auc_mean'], yerr=sub_loss['auc_std'], color=color_loss_auc, fmt='none', capsize=4)
ax1.set_xlabel('Private Training Set Size ($N$)')
ax1.set_ylabel('LOSS ROC-AUC', color=color_loss_auc)
ax1.tick_params(axis='y', labelcolor=color_loss_auc)
ax1.set_ylim(0.55, 1.05)
ax1.set_xticks(sizes)
ax1.set_xticklabels(['1k', '2.5k', '5k', '10k'])
ax1.set_title('(a) Loss Thresholding Attack Scaling (LOSS)')
ax1.grid(True, linestyle=':', alpha=0.5)

ax1_tpr = ax1.twinx()
ax1_tpr.plot(sub_loss['member_size'], sub_loss['tpr1_mean'] * 100, color=color_loss_tpr, marker='s', linestyle='--', lw=2.0, label='LOSS TPR @ 1% FPR (Right Axis)')
ax1_tpr.errorbar(sub_loss['member_size'], sub_loss['tpr1_mean'] * 100, yerr=sub_loss['tpr1_std'] * 100, color=color_loss_tpr, fmt='none', capsize=4)
ax1_tpr.set_ylabel('LOSS TPR @ 1% FPR (%)', color=color_loss_tpr)
ax1_tpr.tick_params(axis='y', labelcolor=color_loss_tpr)
ax1_tpr.set_ylim(-5, 105)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_tpr.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)

# Subplot (b): LiRA Attack Scaling (Offline vs Online LiRA)
sub_off = df_e3[df_e3['method'] == 'offline_LiRA'].sort_values('member_size')
sub_on = df_e3[df_e3['method'] == 'online_LiRA'].sort_values('member_size')

ax2.errorbar(sub_off['member_size'], sub_off['auc_mean'], yerr=sub_off['auc_std'], label='Offline LiRA AUC',
             color='#1f77b4', marker='s', lw=2.0, capsize=4)
ax2.errorbar(sub_on['member_size'], sub_on['auc_mean'], yerr=sub_on['auc_std'], label='Online LiRA AUC',
             color='#d62728', marker='^', lw=2.0, capsize=4)

ax2.set_xlabel('Private Training Set Size ($N$)')
ax2.set_ylabel('Attack ROC-AUC')
ax2.set_title('(b) Calibrated Likelihood-Ratio Attacks Scaling (LiRA)')
ax2.set_ylim(0.55, 1.05)
ax2.set_xticks(sizes)
ax2.set_xticklabels(['1k', '2.5k', '5k', '10k'])
ax2.axhline(0.50, color='gray', linestyle='--', alpha=0.7)
ax2.grid(True, linestyle=':', alpha=0.5)

ax2_tpr = ax2.twinx()
ax2_tpr.plot(sub_off['member_size'], sub_off['tpr1_mean'] * 100, color='#1f77b4', marker='s', linestyle=':', lw=1.5, alpha=0.75, label='Offline LiRA TPR@1%')
ax2_tpr.plot(sub_on['member_size'], sub_on['tpr1_mean'] * 100, color='#d62728', marker='^', linestyle=':', lw=1.5, alpha=0.75, label='Online LiRA TPR@1%')
ax2_tpr.set_ylabel('LiRA TPR @ 1% FPR (%)', color='#444444')
ax2_tpr.tick_params(axis='y', labelcolor='#444444')
ax2_tpr.set_ylim(-3, 60)

lines_b1, labels_b1 = ax2.get_legend_handles_labels()
lines_b2, labels_b2 = ax2_tpr.get_legend_handles_labels()
ax2.legend(lines_b1 + lines_b2, labels_b1 + labels_b2, loc='upper right', frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_data_scaling.png")
fig.savefig(FIG_DIR / "fig2_data_scaling.pdf")
plt.close(fig)
print("Figure 2 generated (LOSS and LiRA shown separately).")

# -------------------------------------------------------------
# FIGURE 3: Defense Comparison (Strongest Attack per Defense)
# -------------------------------------------------------------
df_dp = pd.read_csv(RESULTS_DIR / "DP_SGD_summary.csv")
df_e5a = pd.read_csv(RESULTS_DIR / "E5a_augmentation_summary.csv")

# Strongest attack per defense at epoch 60
# Baseline: Online LiRA (0.898, TPR 27.82%)
base_row = df_base[(df_base['epoch'] == 60) & (df_base['stat'] == 'nll_sum') & (df_base['method'] == 'online_LiRA')].iloc[0]

# Randomized SMILES (Canonical): Offline LiRA (0.528, TPR 1.38%)
e5a_canon_row = df_e5a[(df_e5a['epoch'] == 60) & (df_e5a['stat'] == 'nll_sum') & (df_e5a['method'] == 'offline_LiRA')].iloc[0]

# Randomized SMILES (Variants): Offline LiRA (0.552, TPR 1.56%)
e5a_var_row = df_e5a[(df_e5a['epoch'] == 60) & (df_e5a['stat'] == 'nll_variants_mean') & (df_e5a['method'] == 'offline_LiRA')].iloc[0]

# DP-SGD eps=8: Offline LiRA (0.528, TPR 1.20%)
dp_8_row = df_dp[(df_dp['epsilon_target'] == 8) & (df_dp['method'] == 'offline_LiRA')].iloc[0]

# DP-SGD eps=3: Offline LiRA (0.520, TPR 1.12%)
dp_3_row = df_dp[(df_dp['epsilon_target'] == 3) & (df_dp['method'] == 'offline_LiRA')].iloc[0]

categories = [
    'Baseline\n[Online LiRA]',
    'Rand-SMILES\n(Canonical)\n[Offline LiRA]',
    'Rand-SMILES\n(Variants)\n[Offline LiRA]',
    'DP-SGD (ε=8)*\n[Offline LiRA]',
    'DP-SGD (ε=3)*\n[Offline LiRA]'
]
auc_vals = [base_row['auc_mean'], e5a_canon_row['auc_mean'], e5a_var_row['auc_mean'], dp_8_row['auc_mean'], dp_3_row['auc_mean']]
auc_errs = [base_row['auc_std'], e5a_canon_row['auc_std'], e5a_var_row['auc_std'], dp_8_row['auc_std'], dp_3_row['auc_std']]
tpr_vals = [base_row['tpr1_mean'] * 100, e5a_canon_row['tpr1_mean'] * 100, e5a_var_row['tpr1_mean'] * 100, dp_8_row['tpr1_mean'] * 100, dp_3_row['tpr1_mean'] * 100]
tpr_errs = [base_row['tpr1_std'] * 100, e5a_canon_row['tpr1_std'] * 100, e5a_var_row['tpr1_std'] * 100, dp_8_row['tpr1_std'] * 100, dp_3_row['tpr1_std'] * 100]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

bar_colors = ['#d62728', '#2ca02c', '#31a354', '#1f77b4', '#17becf']
bars1 = ax1.bar(categories, auc_vals, yerr=auc_errs, capsize=4, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
ax1.set_ylabel('Strongest Attack ROC-AUC')
ax1.set_title('(a) Strongest Attack ROC-AUC per Defense (Epoch 60)')
ax1.set_ylim(0.40, 1.0)
ax1.axhline(0.50, color='gray', linestyle='--', label='Random Guess (0.50)')
for bar, val in zip(bars1, auc_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=8.5)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')

bars2 = ax2.bar(categories, tpr_vals, yerr=tpr_errs, capsize=4, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
ax2.set_ylabel('TPR @ 1% FPR (%)')
ax2.set_title('(b) High-Confidence Leakage Rate (TPR @ 1% FPR)')
ax2.set_ylim(0, 32)
for bar, val in zip(bars2, tpr_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=8.5)
ax2.grid(True, linestyle=':', alpha=0.5, axis='y')

plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_defenses_comparison.png")
fig.savefig(FIG_DIR / "fig3_defenses_comparison.pdf")
plt.close(fig)
print("Figure 3 generated (strongest attack per defense).")

# -------------------------------------------------------------
# FIGURE 4: Generative Memorization Audit (E6)
# -------------------------------------------------------------
df_e6 = pd.read_csv(RESULTS_DIR / "E6_generative_audit_summary.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
models = [f"Model {int(k)}" for k in df_e6['model']]

# Subplot 1: Validity, Uniqueness, Novelty
x = np.arange(len(models))
width = 0.25
ax1.bar(x - width, df_e6['validity'] * 100, width, label='Validity', color='#2ca02c', alpha=0.85, edgecolor='black')
ax1.bar(x, df_e6['uniqueness_valid'] * 100, width, label='Uniqueness', color='#1f77b4', alpha=0.85, edgecolor='black')
ax1.bar(x + width, df_e6['novelty_vs_members'] * 100, width, label='Novelty vs Members', color='#ff7f0e', alpha=0.85, edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel('Percentage (%)')
ax1.set_ylim(0, 115)
ax1.set_title('(a) Chemical Quality of Generated Library')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')

# Subplot 2: Exact Copy Counts (Fraction * 5000)
mem_counts = [int(round(f * 5000)) for f in df_e6['exact_member_copy_fraction_all_samples']]
out_counts = [int(round(f * 5000)) for f in df_e6['exact_out_copy_fraction_all_samples']]

ax2.bar(x - width/2, mem_counts, width, label='Member Copies (Count)', color='#d62728', alpha=0.85, edgecolor='black')
ax2.bar(x + width/2, out_counts, width, label='Non-Member Copies (Count)', color='#7f7f7f', alpha=0.85, edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('Exact Verbatim Replications (out of 5,000)')
ax2.set_title('(b) Generative Replications: Member vs Out (p > 0.05)')
ax2.set_ylim(0, 10)
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.5, axis='y')

for i, (mc, oc) in enumerate(zip(mem_counts, out_counts)):
    ax2.text(x[i] - width/2, mc + 0.2, str(mc), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(x[i] + width/2, oc + 0.2, str(oc), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_generative_memorization.png")
fig.savefig(FIG_DIR / "fig4_generative_memorization.pdf")
plt.close(fig)
print("Figure 4 generated.")
print("All 4 figures successfully generated.")
