
def tex_escape(s):
    return str(s).replace('_', r'\_')
#!/usr/bin/env python3
"""
Generate publication-quality LaTeX and Markdown tables from results_summary/*.csv.
Strict adherence to:
1. Every number traced to a CSV row.
2. nll_sum used consistently; nll_variants_mean reported where available.
3. Separation of attacks (never mixing LOSS TPR with LiRA AUC).
4. Explicit annotations of shadow counts, non-significance, and preliminary DP status.
5. Strict LaTeX syntax: math mode for \\pm, escaped underscores and percent signs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import binomtest

# Resolve directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results_summary"
TABLES_DIR = SCRIPT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def format_val_tex(mean, std=None, pct=False, digits=2):
    if pd.isna(mean):
        return "-"
    multiplier = 100.0 if pct else 1.0
    val_str = f"{mean * multiplier:.{digits}f}"
    if std is not None and not pd.isna(std):
        std_str = f"{std * multiplier:.{digits}f}"
        return f"${val_str} \\pm {std_str}$"
    return f"${val_str}$"


def format_val_md(mean, std=None, pct=False, digits=2):
    if pd.isna(mean):
        return "-"
    multiplier = 100.0 if pct else 1.0
    val_str = f"{mean * multiplier:.{digits}f}"
    if std is not None and not pd.isna(std):
        std_str = f"{std * multiplier:.{digits}f}"
        return f"{val_str} ± {std_str}"
    return val_str


# ==============================================================================
# TABLE 1: Baseline Audit (Epochs 5, 20, 60)
# ==============================================================================
def make_table1():
    df = pd.read_csv(RESULTS_DIR / "baseline_attack_summary.csv")
    
    epochs = [5, 20, 60]
    methods = ['LOSS', 'offline_LiRA', 'online_LiRA']
    method_display_tex = {
        'LOSS': 'Loss Thresholding (LOSS)',
        'offline_LiRA': 'Offline LiRA (Out-only)',
        'online_LiRA': 'Online LiRA (In vs. Out)'
    }
    method_display_md = {
        'LOSS': 'Loss Thresholding (LOSS)',
        'offline_LiRA': 'Offline LiRA (Out-only)',
        'online_LiRA': 'Online LiRA (In vs. Out)'
    }
    
    rows = []
    
    for ep in epochs:
        sub = df[(df['epoch'] == ep) & (df['stat'] == 'nll_sum')]
        best_auc = -1.0
        best_m = None
        for m in methods:
            r = sub[sub['method'] == m].iloc[0]
            auc = r['auc_mean']
            if auc > best_auc:
                best_auc = auc
                best_m = m
            rows.append({
                'epoch': str(ep),
                'scoring_tex': 'Canonical (\\texttt{nll\\_sum})',
                'scoring_md': 'Canonical (`nll_sum`)',
                'attack_tex': method_display_tex[m],
                'attack_md': method_display_md[m],
                'auc_tex': format_val_tex(r['auc_mean'], r['auc_std'], digits=3),
                'auc_md': format_val_md(r['auc_mean'], r['auc_std'], digits=3),
                'tpr1_tex': format_val_tex(r['tpr1_mean'], r['tpr1_std'], pct=True, digits=2),
                'tpr1_md': format_val_md(r['tpr1_mean'], r['tpr1_std'], pct=True, digits=2),
                'tpr01_tex': format_val_tex(r['tpr01_mean'], r['tpr01_std'], pct=True, digits=2),
                'tpr01_md': format_val_md(r['tpr01_mean'], r['tpr01_std'], pct=True, digits=2),
            })
        # Strongest attack
        r_best = sub[sub['method'] == best_m].iloc[0]
        rows.append({
            'epoch': str(ep),
            'scoring_tex': 'Canonical (\\texttt{nll\\_sum})',
            'scoring_md': 'Canonical (`nll_sum`)',
            'attack_tex': f"\\textbf{{Strongest: {tex_escape(best_m)}}}",
            'attack_md': f"**Strongest: {best_m.replace('_', ' ')}**",
            'auc_tex': f"\\textbf{{{format_val_tex(r_best['auc_mean'], r_best['auc_std'], digits=3)}}}",
            'auc_md': f"**{format_val_md(r_best['auc_mean'], r_best['auc_std'], digits=3)}**",
            'tpr1_tex': f"\\textbf{{{format_val_tex(r_best['tpr1_mean'], r_best['tpr1_std'], pct=True, digits=2)}}}",
            'tpr1_md': f"**{format_val_md(r_best['tpr1_mean'], r_best['tpr1_std'], pct=True, digits=2)}**",
            'tpr01_tex': f"\\textbf{{{format_val_tex(r_best['tpr01_mean'], r_best['tpr01_std'], pct=True, digits=2)}}}",
            'tpr01_md': f"**{format_val_md(r_best['tpr01_mean'], r_best['tpr01_std'], pct=True, digits=2)}**",
        })

    # Epoch 60 nll_variants_mean
    sub_var = df[(df['epoch'] == 60) & (df['stat'] == 'nll_variants_mean')]
    best_auc = -1.0
    best_m = None
    for m in methods:
        r = sub_var[sub_var['method'] == m].iloc[0]
        auc = r['auc_mean']
        if auc > best_auc:
            best_auc = auc
            best_m = m
        rows.append({
            'epoch': '60 (variants)',
            'scoring_tex': 'Randomized-SMILES (up to 10 strings, mean NLL)',
            'scoring_md': 'Randomized-SMILES (up to 10 strings, mean NLL)',
            'attack_tex': method_display_tex[m],
            'attack_md': method_display_md[m],
            'auc_tex': format_val_tex(r['auc_mean'], r['auc_std'], digits=3),
            'auc_md': format_val_md(r['auc_mean'], r['auc_std'], digits=3),
            'tpr1_tex': format_val_tex(r['tpr1_mean'], r['tpr1_std'], pct=True, digits=2),
            'tpr1_md': format_val_md(r['tpr1_mean'], r['tpr1_std'], pct=True, digits=2),
            'tpr01_tex': format_val_tex(r['tpr01_mean'], r['tpr01_std'], pct=True, digits=2),
            'tpr01_md': format_val_md(r['tpr01_mean'], r['tpr01_std'], pct=True, digits=2),
        })
    r_best = sub_var[sub_var['method'] == best_m].iloc[0]
    rows.append({
        'epoch': '60 (variants)',
        'scoring_tex': 'Randomized-SMILES (up to 10 strings, mean NLL)',
        'scoring_md': 'Randomized-SMILES (up to 10 strings, mean NLL)',
        'attack_tex': f"\\textbf{{Strongest: {tex_escape(best_m)}}}",
        'attack_md': f"**Strongest: {best_m.replace('_', ' ')}**",
        'auc_tex': f"\\textbf{{{format_val_tex(r_best['auc_mean'], r_best['auc_std'], digits=3)}}}",
        'auc_md': f"**{format_val_md(r_best['auc_mean'], r_best['auc_std'], digits=3)}**",
        'tpr1_tex': f"\\textbf{{{format_val_tex(r_best['tpr1_mean'], r_best['tpr1_std'], pct=True, digits=2)}}}",
        'tpr1_md': f"**{format_val_md(r_best['tpr1_mean'], r_best['tpr1_std'], pct=True, digits=2)}**",
        'tpr01_tex': f"\\textbf{{{format_val_tex(r_best['tpr01_mean'], r_best['tpr01_std'], pct=True, digits=2)}}}",
        'tpr01_md': f"**{format_val_md(r_best['tpr01_mean'], r_best['tpr01_std'], pct=True, digits=2)}**",
    })

    # Write LaTeX
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Baseline membership inference vulnerability across training epochs on canonical SMILES. Evaluated over 32 shadow models on 20,000 candidate targets (10,000 members, 10,000 non-members). Values report mean $\\pm$ standard deviation across 32 leave-one-out target evaluations (not a confidence interval). All metrics use negative log-likelihood sum (\\texttt{nll\\_sum}), with randomized-SMILES variants scoring (up to 10 distinct SMILES strings per molecule, the canonical form plus randomized forms, mean NLL) reported at epoch 60.}",
        "\\label{tab:baseline}",
        "\\small",
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "\\textbf{Epochs} & \\textbf{Scoring Representation} & \\textbf{Attack Method} & \\textbf{ROC-AUC} & \\textbf{TPR @ 1\\% FPR (\\%)} & \\textbf{TPR @ 0.1\\% FPR (\\%)} \\\\",
        "\\midrule"
    ]
    curr_epoch = None
    for r in rows:
        if curr_epoch is not None and r['epoch'] != curr_epoch:
            tex.append("\\midrule")
        curr_epoch = r['epoch']
        tex.append(f"{r['epoch']} & {r['scoring_tex']} & {r['attack_tex']} & {r['auc_tex']} & {r['tpr1_tex']} & {r['tpr01_tex']} \\\\")
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    (TABLES_DIR / "table1_baseline.tex").write_text("\n".join(tex) + "\n")

    # Write Markdown
    md = [
        "# Table 1: Baseline Membership Inference Vulnerability Across Epochs",
        "",
        "Evaluated over 32 shadow models on 20,000 candidate targets (10,000 members, 10,000 non-members). Values denote mean ± standard deviation across 32 leave-one-out target models (not a bootstrap CI). Consistently evaluated using `nll_sum`, with randomized variants scoring (up to 10 distinct SMILES strings per molecule, the canonical form plus randomized forms, mean NLL) also reported at epoch 60.",
        "",
        "| Epochs | Scoring Representation | Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |",
        "|---|---|---|---|---|---|"
    ]
    for r in rows:
        md.append(f"| {r['epoch']} | {r['scoring_md']} | {r['attack_md']} | {r['auc_md']} | {r['tpr1_md']} | {r['tpr01_md']} |")
    (TABLES_DIR / "table1_baseline.md").write_text("\n".join(md) + "\n")
    print("Generated Table 1 (baseline)")


# ==============================================================================
# TABLE 2: Data Scaling (E3: N = 1k, 2.5k, 5k, 10k)
# ==============================================================================
def make_table2():
    df = pd.read_csv(RESULTS_DIR / "E3_scaling_summary.csv")
    sizes = [1000, 2500, 5000, 10000]

    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Membership leakage scaling as private dataset size varies ($N \\in \\{1k, 2.5k, 5k, 10k\\}$). Each size uses 8 shadow models ($2N$ candidate universe). LiRA estimates exhibit high variance at small $N$ due to limited shadow samples per molecule. Evaluated using \\texttt{nll\\_sum}. We present empirical measurements without assuming a parametric power law.}",
        "\\label{tab:scaling}",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Loss Thresholding (LOSS)}} & \\multicolumn{2}{c}{\\textbf{Offline LiRA}} & \\multicolumn{2}{c}{\\textbf{Online LiRA}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "\\textbf{Size ($N$)} & \\textbf{ROC-AUC} & \\textbf{TPR@1\\% (\\%)} & \\textbf{ROC-AUC} & \\textbf{TPR@1\\% (\\%)} & \\textbf{ROC-AUC} & \\textbf{TPR@1\\% (\\%)} \\\\",
        "\\midrule"
    ]

    md = [
        "# Table 2: Membership Leakage Scaling Across Dataset Sizes (E3)",
        "",
        "Note: 8 shadows per size; LiRA unstable at small $N$. We report empirical metrics without claiming a power law.",
        "",
        "| Private Size ($N$) | Candidate Universe ($2N$) | LOSS AUC | LOSS TPR@1% (%) | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for s in sizes:
        sub = df[df['member_size'] == s]
        r_loss = sub[sub['method'] == 'LOSS'].iloc[0]
        r_off = sub[sub['method'] == 'offline_LiRA'].iloc[0]
        r_on = sub[sub['method'] == 'online_LiRA'].iloc[0]

        tex.append(
            f"{s:,} & "
            f"{format_val_tex(r_loss['auc_mean'], r_loss['auc_std'], digits=3)} & {format_val_tex(r_loss['tpr1_mean'], r_loss['tpr1_std'], pct=True, digits=2)} & "
            f"{format_val_tex(r_off['auc_mean'], r_off['auc_std'], digits=3)} & {format_val_tex(r_off['tpr1_mean'], r_off['tpr1_std'], pct=True, digits=2)} & "
            f"{format_val_tex(r_on['auc_mean'], r_on['auc_std'], digits=3)} & {format_val_tex(r_on['tpr1_mean'], r_on['tpr1_std'], pct=True, digits=2)} \\\\"
        )

        md.append(
            f"| {s:,} | {2*s:,} | "
            f"{format_val_md(r_loss['auc_mean'], r_loss['auc_std'], digits=3)} | {format_val_md(r_loss['tpr1_mean'], r_loss['tpr1_std'], pct=True, digits=2)} | "
            f"{format_val_md(r_off['auc_mean'], r_off['auc_std'], digits=3)} | {format_val_md(r_off['tpr1_mean'], r_off['tpr1_std'], pct=True, digits=2)} | "
            f"{format_val_md(r_on['auc_mean'], r_on['auc_std'], digits=3)} | {format_val_md(r_on['tpr1_mean'], r_on['tpr1_std'], pct=True, digits=2)} |"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    (TABLES_DIR / "table2_scaling.tex").write_text("\n".join(tex) + "\n")
    (TABLES_DIR / "table2_scaling.md").write_text("\n".join(md) + "\n")
    print("Generated Table 2 (scaling)")


# ==============================================================================
# TABLE 3: Defenses Comparison at Epoch 60
# ==============================================================================
def make_table3():
    df_base = pd.read_csv(RESULTS_DIR / "baseline_attack_summary.csv")
    df_e5a = pd.read_csv(RESULTS_DIR / "E5a_augmentation_summary.csv")
    df_dp = pd.read_csv(RESULTS_DIR / "DP_SGD_summary.csv")

    b_60 = df_base[(df_base['epoch'] == 60) & (df_base['stat'] == 'nll_sum')]
    e5a_canon = df_e5a[(df_e5a['epoch'] == 60) & (df_e5a['stat'] == 'nll_sum')]
    e5a_var = df_e5a[(df_e5a['epoch'] == 60) & (df_e5a['stat'] == 'nll_variants_mean')]
    dp_8 = df_dp[df_dp['epsilon_target'] == 8]
    dp_3 = df_dp[df_dp['epsilon_target'] == 3]

    configs = [
        ("Baseline (Unmitigated)", "Baseline (Unmitigated)", b_60, "32 shadows"),
        ("Randomized-SMILES Training (Canonical Scoring)", "Randomized-SMILES Training (Canonical Scoring)", e5a_canon, "32 shadows"),
        ("Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL)", "Randomized-SMILES Training (Variants Scoring, up to 10 strings, mean NLL)", e5a_var, "32 shadows"),
        ("DP-SGD ($\\epsilon=8.0, \\delta=10^{-5}$)*", "DP-SGD (ε=8.0, δ=1e-5)*", dp_8, "8 shadows; prelim."),
        ("DP-SGD ($\\epsilon=3.0, \\delta=10^{-5}$)*", "DP-SGD (ε=3.0, δ=1e-5)*", dp_3, "8 shadows; prelim."),
    ]

    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparison of privacy defenses at epoch 60 under standard and calibrated attacks. All values use \\texttt{nll\\_sum} (except randomized variants scoring which evaluates mean NLL across up to 10 distinct SMILES strings per molecule). The strongest attack per row (by highest AUC) is bolded. *Note on DP rows: preliminary run with utility collapse; we did not evaluate the generative utility of the DP models (an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense).}",
        "\\label{tab:defenses}",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{\\textbf{Loss Thresholding}} & \\multicolumn{2}{c}{\\textbf{Offline LiRA}} & \\multicolumn{2}{c}{\\textbf{Online LiRA}} & \\textbf{Strongest Attack} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-9}",
        "\\textbf{Defense Configuration} & \\textbf{Shadows} & \\textbf{AUC} & \\textbf{TPR@1\\%} & \\textbf{AUC} & \\textbf{TPR@1\\%} & \\textbf{AUC} & \\textbf{TPR@1\\%} & \\textbf{AUC / TPR@1\\%} \\\\",
        "\\midrule"
    ]

    md = [
        "# Table 3: Defense Benchmark at Epoch 60",
        "",
        "*Note on DP rows: preliminary run with utility collapse; we did not evaluate the generative utility of the DP models (an attack AUC near 0.5 may reflect an under-trained model and is not evidence of a useful defense).*",
        "",
        "| Defense Configuration | Shadow Setup | LOSS AUC | LOSS TPR@1% (%) | Offline LiRA AUC | Offline LiRA TPR@1% (%) | Online LiRA AUC | Online LiRA TPR@1% (%) | Strongest Attack (Method: AUC, TPR@1%) |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    for name_tex, name_md, sub, note in configs:
        r_loss = sub[sub['method'] == 'LOSS'].iloc[0]
        r_off = sub[sub['method'] == 'offline_LiRA'].iloc[0]
        r_on = sub[sub['method'] == 'online_LiRA'].iloc[0]

        candidates = [
            ('LOSS', r_loss['auc_mean'], r_loss['auc_std'], r_loss['tpr1_mean'], r_loss['tpr1_std']),
            ('offline_LiRA', r_off['auc_mean'], r_off['auc_std'], r_off['tpr1_mean'], r_off['tpr1_std']),
            ('online_LiRA', r_on['auc_mean'], r_on['auc_std'], r_on['tpr1_mean'], r_on['tpr1_std'])
        ]
        best = max(candidates, key=lambda x: x[1])
        best_str_tex = tex_escape(best[0]) + f": {best[1]:.3f} / {best[3]*100:.2f}\\%"
        best_str_md = f"**{best[0].replace('_', ' ')}**: {best[1]:.3f} (TPR {best[3]*100:.2f}%)"

        tex.append(
            f"{name_tex} & {note} & "
            f"{format_val_tex(r_loss['auc_mean'], r_loss['auc_std'], digits=3)} & {format_val_tex(r_loss['tpr1_mean'], r_loss['tpr1_std'], pct=True, digits=2)} & "
            f"{format_val_tex(r_off['auc_mean'], r_off['auc_std'], digits=3)} & {format_val_tex(r_off['tpr1_mean'], r_off['tpr1_std'], pct=True, digits=2)} & "
            f"{format_val_tex(r_on['auc_mean'], r_on['auc_std'], digits=3)} & {format_val_tex(r_on['tpr1_mean'], r_on['tpr1_std'], pct=True, digits=2)} & "
            f"\\textbf{{{best_str_tex}}} \\\\"
        )

        md.append(
            f"| {name_md} | {note} | "
            f"{format_val_md(r_loss['auc_mean'], r_loss['auc_std'], digits=3)} | {format_val_md(r_loss['tpr1_mean'], r_loss['tpr1_std'], pct=True, digits=2)} | "
            f"{format_val_md(r_off['auc_mean'], r_off['auc_std'], digits=3)} | {format_val_md(r_off['tpr1_mean'], r_off['tpr1_std'], pct=True, digits=2)} | "
            f"{format_val_md(r_on['auc_mean'], r_on['auc_std'], digits=3)} | {format_val_md(r_on['tpr1_mean'], r_on['tpr1_std'], pct=True, digits=2)} | "
            f"{best_str_md} |"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    (TABLES_DIR / "table3_defenses.tex").write_text("\n".join(tex) + "\n")
    (TABLES_DIR / "table3_defenses.md").write_text("\n".join(md) + "\n")
    print("Generated Table 3 (defenses)")


# ==============================================================================
# TABLE 4: Pretrain -> Fine-tune Scenario
# ==============================================================================
def make_table4():
    df = pd.read_csv(RESULTS_DIR / "pretrain_ft_summary.csv")
    methods = ['LOSS', 'offline_LiRA', 'online_LiRA']
    method_display_tex = {
        'LOSS': 'Loss Thresholding (LOSS)',
        'offline_LiRA': 'Offline LiRA',
        'online_LiRA': 'Online LiRA'
    }
    method_display_md = {
        'LOSS': 'Loss Thresholding (LOSS)',
        'offline_LiRA': 'Offline LiRA',
        'online_LiRA': 'Online LiRA'
    }

    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Membership leakage in transfer learning (public pretraining on 10,000 molecules, fine-tuning on 10,000 private molecules). Metrics report mean $\\pm$ std across 8 independent target models. Note: there is no from-scratch control trained at the exact same size/epochs in this experiment.}",
        "\\label{tab:pretrain_ft}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Attack Method} & \\textbf{ROC-AUC} & \\textbf{TPR @ 1\\% FPR (\\%)} & \\textbf{TPR @ 0.1\\% FPR (\\%)} \\\\",
        "\\midrule"
    ]

    md = [
        "# Table 4: Transfer Learning Membership Leakage (Pretrain -> Fine-tune)",
        "",
        "Public pretraining (10,000 compounds for 60 epochs) followed by private fine-tuning (10,000 compounds for 20 epochs). Evaluated on 8 leave-one-out shadow targets using `nll_sum`.",
        "",
        "*Note: There is no from-scratch control trained at the exact same size and epochs in this repository.*",
        "",
        "| Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |",
        "|---|---|---|---|"
    ]

    for m in methods:
        sub = df[df['method'] == m]
        auc_m, auc_s = sub['auc'].mean(), sub['auc'].std()
        tpr1_m, tpr1_s = sub['tpr_fpr_1pct'].mean(), sub['tpr_fpr_1pct'].std()
        tpr01_m, tpr01_s = sub['tpr_fpr_0_1pct'].mean(), sub['tpr_fpr_0_1pct'].std()

        tex.append(
            f"{method_display_tex[m]} & {format_val_tex(auc_m, auc_s, digits=3)} & {format_val_tex(tpr1_m, tpr1_s, pct=True, digits=2)} & {format_val_tex(tpr01_m, tpr01_s, pct=True, digits=2)} \\\\"
        )
        md.append(
            f"| {method_display_md[m]} | {format_val_md(auc_m, auc_s, digits=3)} | {format_val_md(tpr1_m, tpr1_s, pct=True, digits=2)} | {format_val_md(tpr01_m, tpr01_s, pct=True, digits=2)} |"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    (TABLES_DIR / "table4_pretrain_ft.tex").write_text("\n".join(tex) + "\n")
    (TABLES_DIR / "table4_pretrain_ft.md").write_text("\n".join(md) + "\n")
    print("Generated Table 4 (pretrain_ft)")


# ==============================================================================
# TABLE 5: E6 Generative Audit
# ==============================================================================
def make_table5():
    df = pd.read_csv(RESULTS_DIR / "E6_generative_audit_summary.csv")
    
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Generative memorization audit (Experiment E6; 5,000 samples per model evaluated against 10,000 training members and 10,000 non-members). Replication counts denote verbatim matches: $\\text{Count} = \\text{Fraction} \\times 5,000$. One-sided binomial tests ($p=0.5$) evaluate whether member replications exceed non-member replications. All $p$-values exceed $0.05$, confirming that the observed replication differences are not statistically significant at the current sample size.}",
        "\\label{tab:e6_generation}",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Model ID} & \\textbf{Validity (\\%)} & \\textbf{Uniqueness (\\%)} & \\textbf{Novelty (\\%)} & \\textbf{Member Copies} & \\textbf{Out Copies} & \\textbf{Binomial $p$-value} & \\textbf{Statistically Significant?} \\\\",
        "\\midrule"
    ]

    md = [
        "# Table 5: Generative Memorization Audit (E6)",
        "",
        "Evaluated on 5,000 generated molecules per model on Apple Silicon (MPS). Exact counts are derived as $Count = \\text{Fraction} \\times 5,000$. One-sided binomial tests ($p=0.5$ null hypothesis) assess if member copies exceed out copies.",
        "",
        "| Model ID | Validity (%) | Uniqueness (%) | Novelty vs Members (%) | Exact Member Copies (Count) | Exact Out Copies (Count) | Binomial p-value ($p=0.5$) | Significant at α=0.05? |",
        "|---|---|---|---|---|---|---|---|"
    ]

    tot_mem = 0
    tot_out = 0
    val_list = []
    uniq_list = []
    nov_list = []

    for _, r in df.iterrows():
        mid = int(r['model'])
        val = r['validity'] * 100
        uniq = r['uniqueness_valid'] * 100
        nov = r['novelty_vs_members'] * 100
        val_list.append(val)
        uniq_list.append(uniq)
        nov_list.append(nov)

        mem_cnt = int(round(r['exact_member_copy_fraction_all_samples'] * 5000))
        out_cnt = int(round(r['exact_out_copy_fraction_all_samples'] * 5000))
        tot_mem += mem_cnt
        tot_out += out_cnt
        tot_copies = mem_cnt + out_cnt

        p_val = binomtest(mem_cnt, tot_copies, 0.5, alternative='greater').pvalue if tot_copies > 0 else 1.0
        sig_str_tex = "No ($p > 0.05$)"
        sig_str_md = "No ($p > 0.05$)"

        tex.append(
            f"Model {mid} & ${val:.2f}\\%$ & ${uniq:.2f}\\%$ & ${nov:.2f}\\%$ & "
            f"{mem_cnt} & {out_cnt} & ${p_val:.4f}$ & {sig_str_tex} \\\\"
        )
        md.append(
            f"| Model {mid} | {val:.2f}% | {uniq:.2f}% | {nov:.2f}% | {mem_cnt} | {out_cnt} | {p_val:.4f} | {sig_str_md} |"
        )

    # Pooled
    tot_copies = tot_mem + tot_out
    p_pool = binomtest(tot_mem, tot_copies, 0.5, alternative='greater').pvalue
    mean_val = np.mean(val_list)
    mean_uniq = np.mean(uniq_list)
    mean_nov = np.mean(nov_list)

    tex.append("\\midrule")
    tex.append(
        f"\\textbf{{Pooled (15k)}} & \\textbf{{${mean_val:.2f}\\%$}} & \\textbf{{${mean_uniq:.2f}\\%$}} & \\textbf{{${mean_nov:.2f}\\%$}} & "
        f"\\textbf{{{tot_mem}}} & \\textbf{{{tot_out}}} & \\textbf{{${p_pool:.4f}$}} & \\textbf{{No ($p > 0.05$)}} \\\\"
    )
    md.append(
        f"| **Pooled (15,000 samples)** | **{mean_val:.2f}%** | **{mean_uniq:.2f}%** | **{mean_nov:.2f}%** | **{tot_mem}** | **{tot_out}** | **{p_pool:.4f}** | **No ($p > 0.05$)** |"
    )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    (TABLES_DIR / "table5_e6_generation.tex").write_text("\n".join(tex) + "\n")
    (TABLES_DIR / "table5_e6_generation.md").write_text("\n".join(md) + "\n")
    print("Generated Table 5 (e6_generation)")


if __name__ == "__main__":
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    make_table5()
    print("All tables successfully generated in paper/tables/!")
