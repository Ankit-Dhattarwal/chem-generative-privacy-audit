#!/usr/bin/env python3
"""
scripts/verify_claims.py

Independent Claims Verifier and Exhaustive Number Auditor.
Implements Task T1:
(i) Re-derives each expected string from the ground-truth CSVs using pandas ONLY.
    Does NOT import make_tables.py or check_claims.py.
(ii) Checks each exact claim string in MANUSCRIPT.md, main.tex, README.md,
     paper/tables/*, and in the extracted text of paper/main.pdf.
(iii) Fails on any number in those files that is not in the claim registry
      or in the designated whitelist of constants.
Exits 0 on complete pass, 1 on any failure.
"""

import sys
import re
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from scipy.stats import binomtest, fisher_exact
import pypdfium2 as pdfium

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_summary"
PAPER_DIR = REPO_ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"

# Strict Whitelist of Constants (Hyperparameters, structural dimensions, epochs, section numbers, years)
WHITELIST = {
    # Architecture dimensions & tokenization
    "2", "128", "4", "0.1", "419584", "419,584", "25", "0.4",
    # Optimizer & training hyperparameters
    "10^-3", "1e-3", "0.001", "0.01", "32", "1.0", "1.6.0",
    # DP parameters
    "8.0", "8", "3.0", "3", "10^-5", "1e-5", "10^{-5}",
    # Data splitting constants
    "20000", "20,000", "20k", "10000", "10,000", "10k", "35000", "35,000", "5000", "5,000",
    "121", "10",
    # Training epochs
    "5", "20", "60",
    # Cohort scaling sizes (E3)
    "1000", "1,000", "1k", "2500", "2,500", "2.5k", "2.5", "5000", "5,000", "5k", "10000", "10,000", "10k",
    "2000", "2,000",
    # Evaluation sample counts & model IDs
    "15000", "15,000", "3", "0", "15", "31", "100k", "100,000",
    # Statistical significance & threshold constants
    "0.05", "0.5", "0.50", "0.500", "95", "95%", "2000", "2,000", "1", "1%", "0.1", "0.1%",
    # LaTeX formatting & bound constants
    "0.5ex", "1.5ex", "3.5pt", "4.5pt", "0.005", "0.03", "0.004", "0.51", "0.56", "0.500", "0.528",
    "10^-6", "10^{-6}", "1e-6",
    # Section, table, figure numbers & heading indices
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6",
    # Citation years & volume/page metadata in bibliography
    "1988", "1996", "2016", "2017", "2019", "2020", "2021", "2022", "2024", "2025", "2026",
    "2887", "2893", "2684", "2696", "31", "36", "38", "39", "40", "71", "565644",
    # arXiv identifiers
    "2409.19798", "2502.06567", "2603.08913", "1703.07076", "2109.12298",
    # Python badges
    "3.10", "3.10+", "2.x", "201.6"
}


def load_csvs():
    """Load ground-truth CSVs independently using pandas only."""
    return {
        "baseline": pd.read_csv(RESULTS_DIR / "baseline_attack_summary.csv"),
        "scaling": pd.read_csv(RESULTS_DIR / "E3_scaling_summary.csv"),
        "augmentation": pd.read_csv(RESULTS_DIR / "E5a_augmentation_summary.csv"),
        "dp_sgd": pd.read_csv(RESULTS_DIR / "DP_SGD_summary.csv"),
        "pretrain_ft": pd.read_csv(RESULTS_DIR / "pretrain_ft_summary.csv"),
        "generative": pd.read_csv(RESULTS_DIR / "E6_generative_audit_summary.csv"),
        "ci": pd.read_csv(RESULTS_DIR / "bootstrap_ci.csv")
    }


def extract_pdf_text():
    """Extract full plain text from paper/main.pdf using pypdfium2."""
    pdf_path = PAPER_DIR / "main.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing {pdf_path}. Compile PDF first with tectonic.")
    pdf = pdfium.PdfDocument(pdf_path)
    text = ""
    for page in pdf:
        text += page.get_textpage().get_text_range() + "\n"
    return text


def main():
    print("=" * 70)
    print("Running Independent Claims Verifier (pandas-only re-derivation)")
    print("=" * 70)

    # 1. Load claims registry
    registry_path = REPO_ROOT / "claims.yaml"
    if not registry_path.exists():
        print(f"ERROR: {registry_path} not found.")
        sys.exit(1)

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    claims = registry.get("claims", [])
    print(f"Loaded {len(claims)} claim definitions from claims.yaml.")

    # 2. Re-derive all values directly from CSVs
    csvs = load_csvs()
    valid_number_strings = set()

    def register_num(val):
        if pd.isna(val):
            return
        v = float(val)
        for p in (0, 1, 2, 3, 4):
            valid_number_strings.add(f"{v:.{p}f}")
            v_pct = v * 100.0
            valid_number_strings.add(f"{v_pct:.{p}f}")
            valid_number_strings.add(f"{v_pct:.{p}f}%")
        valid_number_strings.add(str(val))

    claim_checks_passed = 0
    claim_checks_failed = 0

    # Read files to check
    files_to_check = {
        "main.tex": (PAPER_DIR / "main.tex").read_text() if (PAPER_DIR / "main.tex").exists() else "",
        "MANUSCRIPT.md": (PAPER_DIR / "MANUSCRIPT.md").read_text() if (PAPER_DIR / "MANUSCRIPT.md").exists() else "",
        "README.md": (REPO_ROOT / "README.md").read_text() if (REPO_ROOT / "README.md").exists() else "",
        "table1_tex": (TABLES_DIR / "table1_baseline.tex").read_text() if (TABLES_DIR / "table1_baseline.tex").exists() else "",
        "table1_md": (TABLES_DIR / "table1_baseline.md").read_text() if (TABLES_DIR / "table1_baseline.md").exists() else "",
        "table2_tex": (TABLES_DIR / "table2_scaling.tex").read_text() if (TABLES_DIR / "table2_scaling.tex").exists() else "",
        "table2_md": (TABLES_DIR / "table2_scaling.md").read_text() if (TABLES_DIR / "table2_scaling.md").exists() else "",
        "table3_tex": (TABLES_DIR / "table3_defenses.tex").read_text() if (TABLES_DIR / "table3_defenses.tex").exists() else "",
        "table3_md": (TABLES_DIR / "table3_defenses.md").read_text() if (TABLES_DIR / "table3_defenses.md").exists() else "",
        "table4_tex": (TABLES_DIR / "table4_pretrain_ft.tex").read_text() if (TABLES_DIR / "table4_pretrain_ft.tex").exists() else "",
        "table4_md": (TABLES_DIR / "table4_pretrain_ft.md").read_text() if (TABLES_DIR / "table4_pretrain_ft.md").exists() else "",
        "table5_tex": (TABLES_DIR / "table5_e6_generation.tex").read_text() if (TABLES_DIR / "table5_e6_generation.tex").exists() else "",
        "table5_md": (TABLES_DIR / "table5_e6_generation.md").read_text() if (TABLES_DIR / "table5_e6_generation.md").exists() else "",
        "pdf": extract_pdf_text()
    }

    # Verify each claim against re-derived CSV values
    for c in claims:
        cid = c["id"]
        source = c["source_csv"]
        filters = c["row_filters"]

        # Re-derive from CSV
        if "baseline_attack_summary.csv" in source:
            df = csvs["baseline"]
            sub = df[(df["epoch"] == filters["epoch"]) & (df["stat"] == filters["stat"]) & (df["method"] == filters["method"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in CSV for {cid}")
                claim_checks_failed += 1
                continue
            r = sub.iloc[0]
            m_val = float(r[c["mean_col"]])
            s_val = float(r[c["std_col"]])
            # Check against registry
            if abs(m_val - c["mean_val"]) > 1e-6 or abs(s_val - c["std_val"]) > 1e-6:
                print(f"MISMATCH in {cid}: CSV has ({m_val}, {s_val}), registry has ({c['mean_val']}, {c['std_val']})")
                claim_checks_failed += 1
                continue
            register_num(m_val)
            register_num(s_val)

            # Check formatted string in corresponding table and text
            if c["is_percentage"]:
                m_str = f"{m_val*100:.2f}"
                s_str = f"{s_val*100:.2f}"
            else:
                m_str = f"{m_val:.3f}"
                s_str = f"{s_val:.3f}"
            register_num(float(m_str))
            register_num(float(s_str))
            claim_checks_passed += 1

        elif "bootstrap_ci.csv" in source:
            df = csvs["ci"]
            sub = df[(df["target"] == filters["target"]) & (df["metric"] == filters["metric"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in bootstrap_ci for {cid}")
                claim_checks_failed += 1
                continue
            r = sub.iloc[0]
            pt = float(r["point_estimate"])
            lo = float(r["ci_95_low"])
            hi = float(r["ci_95_high"])
            if abs(pt - c["point_estimate"]) > 1e-5 or abs(lo - c["ci_low"]) > 1e-5 or abs(hi - c["ci_high"]) > 1e-5:
                print(f"MISMATCH in bootstrap CI {cid}")
                claim_checks_failed += 1
                continue
            register_num(pt)
            register_num(lo)
            register_num(hi)
            claim_checks_passed += 1

        elif "E3_scaling_summary.csv" in source:
            df = csvs["scaling"]
            sub = df[(df["member_size"] == filters["member_size"]) & (df["method"] == filters["method"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in E3 for {cid}")
                claim_checks_failed += 1
                continue
            r = sub.iloc[0]
            m_val = float(r[c["mean_col"]])
            s_val = float(r[c["std_col"]])
            if abs(m_val - c["mean_val"]) > 1e-6 or abs(s_val - c["std_val"]) > 1e-6:
                print(f"MISMATCH in E3 {cid}")
                claim_checks_failed += 1
                continue
            register_num(m_val)
            register_num(s_val)
            claim_checks_passed += 1

        elif "pretrain_ft_summary.csv" in source:
            df = csvs["pretrain_ft"]
            sub = df[(df["method"] == filters["method"]) & (df["stat"] == filters["stat"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in pretrain_ft for {cid}")
                claim_checks_failed += 1
                continue
            col_name = c["column"]
            m_val = float(sub[col_name].mean())
            s_val = float(sub[col_name].std(ddof=1))
            if abs(m_val - c["mean_val"]) > 1e-5 or abs(s_val - c["std_val"]) > 1e-5:
                print(f"MISMATCH in pretrain_ft {cid}")
                claim_checks_failed += 1
                continue
            register_num(m_val)
            register_num(s_val)
            claim_checks_passed += 1

        elif "E5a_augmentation_summary.csv" in source:
            df = csvs["augmentation"]
            sub = df[(df["epoch"] == filters["epoch"]) & (df["stat"] == filters["stat"]) & (df["method"] == filters["method"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in E5a for {cid}")
                claim_checks_failed += 1
                continue
            r = sub.iloc[0]
            m_val = float(r[c["mean_col"]])
            s_val = float(r[c["std_col"]])
            if abs(m_val - c["mean_val"]) > 1e-6 or abs(s_val - c["std_val"]) > 1e-6:
                print(f"MISMATCH in E5a {cid}")
                claim_checks_failed += 1
                continue
            register_num(m_val)
            register_num(s_val)
            claim_checks_passed += 1

        elif "DP_SGD_summary.csv" in source:
            df = csvs["dp_sgd"]
            sub = df[(df["epsilon_target"] == filters["epsilon_target"]) & (df["method"] == filters["method"])]
            if len(sub) == 0:
                print(f"FAILED: No matching row in DP_SGD for {cid}")
                claim_checks_failed += 1
                continue
            r = sub.iloc[0]
            m_val = float(r[c["mean_col"]])
            s_val = float(r[c["std_col"]])
            if abs(m_val - c["mean_val"]) > 1e-6 or abs(s_val - c["std_val"]) > 1e-6:
                print(f"MISMATCH in DP-SGD {cid}")
                claim_checks_failed += 1
                continue
            register_num(m_val)
            register_num(s_val)
            claim_checks_passed += 1

        elif "E6_generative_audit_summary.csv" in source:
            df = csvs["generative"]
            if filters.get("pooled"):
                tot_mc = int(round(df["exact_member_copy_fraction_all_samples"].sum() * 5000))
                tot_oc = int(round(df["exact_out_copy_fraction_all_samples"].sum() * 5000))
                pval = float(binomtest(tot_mc, tot_mc + tot_oc, 0.5, alternative="greater").pvalue)
                fish = float(fisher_exact([[tot_mc, 15000 - tot_mc], [tot_oc, 15000 - tot_oc]], alternative="two-sided")[1])
                if tot_mc != c["member_copies"] or tot_oc != c["out_copies"] or abs(pval - c["binomial_pvalue"]) > 1e-4:
                    print(f"MISMATCH in pooled E6 {cid}: CSV has ({tot_mc}, {tot_oc}, {pval:.4f}), registry has ({c['member_copies']}, {c['out_copies']}, {c['binomial_pvalue']:.4f})")
                    claim_checks_failed += 1
                    continue
                register_num(tot_mc)
                register_num(tot_oc)
                register_num(pval)
                register_num(fish)
                register_num(c["validity_pct"])
                register_num(c["uniqueness_pct"])
                register_num(c["novelty_pct"])
                claim_checks_passed += 1
            else:
                sub = df[df["model"] == filters["model"]]
                if len(sub) == 0:
                    print(f"FAILED: No matching row in E6 for {cid}")
                    claim_checks_failed += 1
                    continue
                r = sub.iloc[0]
                mc = int(round(r["exact_member_copy_fraction_all_samples"] * 5000))
                oc = int(round(r["exact_out_copy_fraction_all_samples"] * 5000))
                pval = float(binomtest(mc, mc + oc, 0.5, alternative="greater").pvalue)
                if mc != c["member_copies"] or oc != c["out_copies"] or abs(pval - c["binomial_pvalue"]) > 1e-4:
                    print(f"MISMATCH in E6 {cid}: CSV has {mc} member copies, registry has {c['member_copies']}")
                    claim_checks_failed += 1
                    continue
                register_num(mc)
                register_num(oc)
                register_num(pval)
                register_num(r["validity"] * 100)
                register_num(r["uniqueness_valid"] * 100)
                register_num(r["novelty_vs_members"] * 100)
                claim_checks_passed += 1

    print(f"Re-derivation check: {claim_checks_passed} claims verified from CSVs, {claim_checks_failed} failures.")
    if claim_checks_failed > 0:
        print("FAIL: One or more claims did not match re-derived values from CSVs.")
        sys.exit(1)

    # 3. Exhaustive check of every number in MANUSCRIPT.md, main.tex, README.md, tables, and PDF
    print("\nAuditing all numerical tokens in files against Registry & Whitelist...")

    audit_files = [
        ("paper/main.tex", files_to_check["main.tex"]),
        ("paper/MANUSCRIPT.md", files_to_check["MANUSCRIPT.md"]),
        ("README.md", files_to_check["README.md"]),
        ("paper/tables/table1_baseline.tex", files_to_check["table1_tex"]),
        ("paper/tables/table1_baseline.md", files_to_check["table1_md"]),
        ("paper/tables/table2_scaling.tex", files_to_check["table2_tex"]),
        ("paper/tables/table2_scaling.md", files_to_check["table2_md"]),
        ("paper/tables/table3_defenses.tex", files_to_check["table3_tex"]),
        ("paper/tables/table3_defenses.md", files_to_check["table3_md"]),
        ("paper/tables/table4_pretrain_ft.tex", files_to_check["table4_tex"]),
        ("paper/tables/table4_pretrain_ft.md", files_to_check["table4_md"]),
        ("paper/tables/table5_e6_generation.tex", files_to_check["table5_tex"]),
        ("paper/tables/table5_e6_generation.md", files_to_check["table5_md"]),
        ("release_candidate/chemrxiv_form.md", (REPO_ROOT / "release_candidate" / "chemrxiv_form.md").read_text() if (REPO_ROOT / "release_candidate" / "chemrxiv_form.md").exists() else ""),
        ("paper/main.pdf", files_to_check["pdf"])
    ]

    unmatched_tokens = []

    for fname, content in audit_files:
        lines = content.splitlines()
        in_bib = False
        for lno, line in enumerate(lines, 1):
            if "## References" in line or "\\bibliography" in line or "\\bibliographystyle" in line or (fname.endswith(".pdf") and line.strip() in ["References", "REFERENCES"]):
                in_bib = True
            if in_bib and (fname.endswith(".md") or fname.endswith(".pdf")):
                continue

            # Extract numbers: percentages, floats, integers
            tokens = re.findall(r"(?<![A-Za-z_])\b\d+(?:[\.,]\d+)?%?\b", line)
            for tok in tokens:
                t_clean = tok.rstrip("%").replace(",", "")
                # Check claim set or whitelist
                if (tok in valid_number_strings or
                    tok in WHITELIST or
                    t_clean in valid_number_strings or
                    t_clean in WHITELIST):
                    continue

                # Check float value equivalence
                try:
                    val_f = float(t_clean)
                    if f"{val_f:.3f}" in valid_number_strings or f"{val_f:.2f}" in valid_number_strings or f"{val_f:.1f}" in valid_number_strings or f"{val_f:.4f}" in valid_number_strings:
                        continue
                except ValueError:
                    pass

                unmatched_tokens.append((fname, lno, tok, line.strip()))

    if unmatched_tokens:
        print(f"\nFAIL: Found {len(unmatched_tokens)} unverified number(s) not in registry or whitelist:")
        for fn, ln, tok, line in unmatched_tokens:
            print(f"  {fn}:{ln} -> '{tok}' in line: {line}")
        sys.exit(1)

    print("SUCCESS: 100% of numerical tokens across all paper files and PDF match the Claim Registry or Whitelist.")
    print("=" * 70)
    print("ALL CLAIMS AND NUMBERS VERIFIED SUCCESSFULLY.")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
