import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PAPER_DIR = Path("paper")
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: Baseline Attack Vulnerability vs Epochs (LiRA vs LOSS)
# -------------------------------------------------------------
df_base = pd.read_csv("results/cuda_arm/baseline/attack_summary.csv")
# Filter nll_sum
df_base_sum = df_base[df_base['stat'] == 'nll_sum']

epochs = [5, 20, 60]
methods = ['LOSS', 'offline_LiRA', 'online_LiRA']
method_labels = {'LOSS': 'Standard Loss Thresholding', 'offline_LiRA': 'Offline LiRA (Ours)', 'online_LiRA': 'Online LiRA (Full)'}
colors = {'LOSS': '#7f7f7f', 'offline_LiRA': '#1f77b4', 'online_LiRA': '#d62728'}
markers = {'LOSS': 'o', 'offline_LiRA': 's', 'online_LiRA': '^'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for m in methods:
    sub = df_base_sum[df_base_sum['method'] == m].sort_values('epoch')
    ax1.errorbar(sub['epoch'], sub['auc_mean'], yerr=sub['auc_std'], label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.2, capsize=4, markersize=7)
    ax2.errorbar(sub['epoch'], sub['tpr1_mean'] * 100, yerr=sub['tpr1_std'] * 100, label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.2, capsize=4, markersize=7)

ax1.set_xlabel('Training Epochs')
ax1.set_ylabel('Attack ROC-AUC')
ax1.set_title('(a) Membership Leakage (AUC) vs Training Duration')
ax1.set_ylim(0.48, 0.95)
ax1.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Random Guess (0.50)')
ax1.set_xticks(epochs)
ax1.legend(loc='upper left', frameon=True)

ax2.set_xlabel('Training Epochs')
ax2.set_ylabel('True Positive Rate @ 1% FPR (%)')
ax2.set_title('(b) High-Confidence Leakage (TPR @ 1% FPR)')
ax2.set_ylim(-1, 32)
ax2.set_xticks(epochs)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_exposure_epochs.png")
fig.savefig(FIG_DIR / "fig1_exposure_epochs.pdf")
plt.close(fig)
print("✅ Figure 1 saved.")

# -------------------------------------------------------------
# FIGURE 2: Dataset Scaling Law (E3: 1k, 2.5k, 5k, 10k)
# -------------------------------------------------------------
df_e3 = pd.read_csv("results/cuda_arm/E3/summary.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

sizes = [1000, 2500, 5000, 10000]
for m in methods:
    sub = df_e3[df_e3['method'] == m].sort_values('member_size')
    ax1.errorbar(sub['member_size'], sub['auc_mean'], yerr=sub['auc_std'], label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.2, capsize=4, markersize=7)
    ax2.errorbar(sub['member_size'], sub['tpr1_mean'] * 100, yerr=sub['tpr1_std'] * 100, label=method_labels[m],
                 color=colors[m], marker=markers[m], lw=2.2, capsize=4, markersize=7)

ax1.set_xlabel('Private Training Set Size ($N$)')
ax1.set_ylabel('Attack ROC-AUC')
ax1.set_title('(a) Attack Vulnerability vs Dataset Size')
ax1.set_ylim(0.50, 1.02)
ax1.set_xticks(sizes)
ax1.set_xticklabels(['1k', '2.5k', '5k', '10k'])
ax1.axhline(0.50, color='gray', linestyle='--', alpha=0.7)
ax1.legend(loc='upper right', frameon=True)

ax2.set_xlabel('Private Training Set Size ($N$)')
ax2.set_ylabel('TPR @ 1% FPR (%)')
ax2.set_title('(b) Target Identification Rate at Low FPR')
ax2.set_ylim(-2, 105)
ax2.set_xticks(sizes)
ax2.set_xticklabels(['1k', '2.5k', '5k', '10k'])
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_data_scaling.png")
fig.savefig(FIG_DIR / "fig2_data_scaling.pdf")
plt.close(fig)
print("✅ Figure 2 saved.")

# -------------------------------------------------------------
# FIGURE 3: Defense Benchmark Comparison (Baseline vs E5a vs DP-SGD)
# -------------------------------------------------------------
df_dp = pd.read_csv("results/cuda_arm/DP_SGD/summary.csv")
df_e5a = pd.read_csv("results/cuda_arm/E5a/attack_summary.csv")
df_e5a_60 = df_e5a[(df_e5a['epoch'] == 60) & (df_e5a['stat'] == 'nll_sum') & (df_e5a['method'] == 'online_LiRA')].iloc[0]
base_60 = df_base[(df_base['epoch'] == 60) & (df_base['stat'] == 'nll_sum') & (df_base['method'] == 'online_LiRA')].iloc[0]
dp_8 = df_dp[(df_dp['epsilon_target'] == 8) & (df_dp['method'] == 'online_LiRA')].iloc[0]
dp_3 = df_dp[(df_dp['epsilon_target'] == 3) & (df_dp['method'] == 'online_LiRA')].iloc[0]

categories = [
    'Baseline\n(No Defense)',
    'Randomized SMILES\n(Invariance Defense)',
    'DP-SGD\n(ε = 8.0, δ = 1e-5)',
    'DP-SGD\n(ε = 3.0, δ = 1e-5)'
]
auc_vals = [base_60['auc_mean'], df_e5a_60['auc_mean'], dp_8['auc_mean'], dp_3['auc_mean']]
auc_errs = [base_60['auc_std'], df_e5a_60['auc_std'], dp_8['auc_std'], dp_3['auc_std']]
tpr_vals = [base_60['tpr1_mean'] * 100, df_e5a_60['tpr1_mean'] * 100, dp_8['tpr1_mean'] * 100, dp_3['tpr1_mean'] * 100]
tpr_errs = [base_60['tpr1_std'] * 100, df_e5a_60['tpr1_std'] * 100, dp_8['tpr1_std'] * 100, dp_3['tpr1_std'] * 100]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

bar_colors = ['#d62728', '#2ca02c', '#1f77b4', '#17becf']
bars1 = ax1.bar(categories, auc_vals, yerr=auc_errs, capsize=5, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
ax1.set_ylabel('Online LiRA ROC-AUC')
ax1.set_title('(a) Membership Attack Resistance')
ax1.set_ylim(0.40, 1.0)
ax1.axhline(0.50, color='black', linestyle='--', label='Ideal Privacy (0.50)')
for bar, val in zip(bars1, auc_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax1.legend(loc='upper right')

bars2 = ax2.bar(categories, tpr_vals, yerr=tpr_errs, capsize=5, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
ax2.set_ylabel('TPR @ 1% FPR (%)')
ax2.set_title('(b) High-Confidence Leakage Rate')
ax2.set_ylim(0, 32)
for bar, val in zip(bars2, tpr_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_defenses_comparison.png")
fig.savefig(FIG_DIR / "fig3_defenses_comparison.pdf")
plt.close(fig)
print("✅ Figure 3 saved.")

# -------------------------------------------------------------
# FIGURE 4: Generative Memorization Audit (E6)
# -------------------------------------------------------------
df_e6 = pd.read_csv("results/cuda_arm/E6/summary.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
models = [f"Model {k}" for k in df_e6['model']]

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

# Subplot 2: Exact Copy Fractions (Member vs Out)
ax2.bar(x - width/2, df_e6['exact_member_copy_fraction_all_samples'] * 100, width, label='Exact Member Copy', color='#d62728', alpha=0.85, edgecolor='black')
ax2.bar(x + width/2, df_e6['exact_out_copy_fraction_all_samples'] * 100, width, label='Exact Out Copy', color='#7f7f7f', alpha=0.85, edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('Exact Replication Rate (%)')
ax2.set_title('(b) Unintentional Verbatim Memorization')
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_generative_memorization.png")
fig.savefig(FIG_DIR / "fig4_generative_memorization.pdf")
plt.close(fig)
print("✅ Figure 4 saved.")
print("🎉 All 4 publication-quality figures successfully generated!")
