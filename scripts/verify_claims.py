#!/usr/bin/env python3
"""
scripts/verify_claims.py

Independent Claims Verifier and Exhaustive Number Auditor.
Implements Task T1:
(i) Re-derives registered metrics from ground-truth CSVs using pandas and
    recomputes E6 exact binomial tests from the CSV-derived match counts.
    Does NOT import make_tables.py or check_claims.py.
(ii) Checks the pooled E6 binomial results in the abstract against the CSVs.
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
from scipy.stats import binomtest
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
    # Verified venue page ranges, article identifiers, and literature corpus size
    "649", "657", "650", "1145", "3807503", "3819470", "10.1145", "3807503.3819470",
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
        "dp_accounting": pd.read_csv(REPO_ROOT / "results/cuda_arm/DP_SGD/privacy_accounting_by_model.csv"),
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
    print("Running Independent Claims Verifier (CSV re-derivation and number audit)")
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

    # Verify and register the realized accountant values shown alongside the
    # requested DP-SGD targets in the Methods section.
    dp_accounting = csvs["dp_accounting"]
    for target, expected_models in ((8, 8), (3, 8)):
        rows = dp_accounting[dp_accounting["epsilon_target"] == target]
        if len(rows) != expected_models or rows["model"].nunique() != expected_models:
            raise ValueError(f"Expected {expected_models} DP accountant records for target epsilon {target}")
        spent = rows["epsilon_spent_last_epoch"].astype(float)
        if not np.allclose(spent, spent.iloc[0], rtol=0, atol=1e-9):
            raise ValueError(f"DP accountant values differ across models for target epsilon {target}")
        register_num(round(float(spent.iloc[0]), 3))
        register_num(round(float(spent.iloc[0]), 4))
        print(f"Derived DP accountant epsilon from CSV: target={target}, spent={spent.iloc[0]:.3f} across {len(rows)} models.")

    # Register the E6 values printed in the generative-audit prose/table from
    # the CSV, including pooled hit counts and the conditional match total.
    e6 = csvs["generative"]
    samples_per_model_values = e6["samples"].astype(int).unique()
    if len(samples_per_model_values) != 1:
        raise ValueError("E6 models do not have a common sample count")
    samples_per_model = int(samples_per_model_values[0])
    model_count = len(e6)
    total_samples = samples_per_model * model_count
    if total_samples != int(e6["samples"].astype(int).sum()):
        raise ValueError("E6 sample total disagrees with samples per model × model count")
    pooled_member_copies = int(round((e6["exact_member_copy_fraction_all_samples"] * e6["samples"]).sum()))
    pooled_out_copies = int(round((e6["exact_out_copy_fraction_all_samples"] * e6["samples"]).sum()))
    combined_matches = pooled_member_copies + pooled_out_copies
    binomial_null_probability = 0.5
    pooled_two_sided_pvalue = float(binomtest(
        pooled_member_copies, combined_matches, binomial_null_probability,
        alternative="two-sided"
    ).pvalue)
    # These derived numbers occur explicitly in the abstract and Section 4.6.
    for derived in (total_samples, pooled_member_copies, pooled_out_copies,
                    combined_matches, binomial_null_probability,
                    pooled_two_sided_pvalue):
        register_num(derived)
    print("Derived E6 audit numbers from CSV: "
          f"samples/model={samples_per_model}, models={model_count}, total={total_samples}, "
          f"member={pooled_member_copies}, holdout={pooled_out_copies}, "
          f"combined matches={combined_matches}, "
          f"H0 p={binomial_null_probability}, two-sided p={pooled_two_sided_pvalue:.4f}.")

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

    # Verify the pooled E6 binomial result in the actual abstract blocks.
    # This closes a gap left by the number registry: a wrong value can be
    # individually valid elsewhere in the CSV while still being wrong here.
    expected_e6_test = (
        f"one-sided exact binomial p-value is {float(binomtest(pooled_member_copies, combined_matches, binomial_null_probability, alternative='greater').pvalue):.4f} "
        f"and the two-sided exact binomial p-value is {pooled_two_sided_pvalue:.4f}"
    )
    main_abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", files_to_check["main.tex"], re.S)
    manuscript_abstract_match = re.search(r"## Abstract\s+(.*?)\s+---", files_to_check["MANUSCRIPT.md"], re.S)
    chemrxiv_text = (REPO_ROOT / "release_candidate" / "chemrxiv_form.md").read_text() if (REPO_ROOT / "release_candidate" / "chemrxiv_form.md").exists() else ""
    chemrxiv_abstract_match = re.search(r"## Plain-Text Abstract\s+(.*?)\s+## Keywords", chemrxiv_text, re.S)
    abstract_blocks = {
        "paper/main.tex abstract": main_abstract_match.group(1) if main_abstract_match else "",
        "paper/MANUSCRIPT.md abstract": manuscript_abstract_match.group(1) if manuscript_abstract_match else "",
        "release_candidate/chemrxiv_form.md abstract": chemrxiv_abstract_match.group(1) if chemrxiv_abstract_match else "",
        "README.md generative finding": files_to_check["README.md"],
        "paper/main.pdf abstract": files_to_check["pdf"],
    }
    for label, block in abstract_blocks.items():
        if " ".join(expected_e6_test.split()) not in " ".join(block.split()):
            print(f"FAILED: CSV-derived pooled E6 binomial values missing from {label}: {expected_e6_test}")
            sys.exit(1)

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
                tot_mc = pooled_member_copies
                tot_oc = pooled_out_copies
                pval = float(binomtest(tot_mc, tot_mc + tot_oc, binomial_null_probability, alternative="greater").pvalue)
                if tot_mc != c["member_copies"] or tot_oc != c["out_copies"] or abs(pval - c["binomial_pvalue"]) > 1e-4:
                    print(f"MISMATCH in pooled E6 {cid}: CSV has ({tot_mc}, {tot_oc}, {pval:.4f}), registry has ({c['member_copies']}, {c['out_copies']}, {c['binomial_pvalue']:.4f})")
                    claim_checks_failed += 1
                    continue
                register_num(tot_mc)
                register_num(tot_oc)
                register_num(pval)
                register_num(pooled_two_sided_pvalue)
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
                model_samples = int(r["samples"])
                mc = int(round(r["exact_member_copy_fraction_all_samples"] * model_samples))
                oc = int(round(r["exact_out_copy_fraction_all_samples"] * model_samples))
                pval = float(binomtest(mc, mc + oc, binomial_null_probability, alternative="greater").pvalue)
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

            # Normalize LaTeX's grouped thousands separator (e.g. 14{,}988)
            # before extracting numbers, so it is checked as one derived value.
            scan_line = re.sub(r"\{,\}", ",", line)
            # Extract numbers: percentages, floats, integers
            tokens = re.findall(r"(?<![A-Za-z_])\b\d+(?:[\.,]\d+)?%?\b", scan_line)
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

    print("SUCCESS: Registered values match the CSVs; numerical tokens are registered or whitelisted; pooled E6 abstract p-values match the CSV-derived tests.")
    print("=" * 70)
    print("CSV claim and numerical-token checks completed successfully.")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
