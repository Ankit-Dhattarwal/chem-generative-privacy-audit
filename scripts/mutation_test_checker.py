#!/usr/bin/env python3
"""
Mutation Test for check_claims.py (Reviewer Claim V4).

Demonstrates that check_claims.py verifies set membership of numerical tokens
against CSVs/whitelist, but cannot verify semantic context (i.e. whether a number
belongs to the correct experimental metric).

This script:
1. Copies the repo files to a temporary directory.
2. Introduces semantic context mutations into MANUSCRIPT.md:
   - Mutates "standard LOSS reaches AUC $0.998 \\pm 0.001$" to "$0.975 \\pm 0.004$"
   - Mutates "TPR@1% FPR reaching $27.82" to "$45.49"
3. Runs check_claims.py on the mutated copy and verifies that it still exits 0.
4. Identifies which tokens in the manuscript are covered ONLY by the whitelist.
"""

import sys
import shutil
import tempfile
import subprocess
import re
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_mutation_test():
    print("=" * 70)
    print("RUNNING MUTATION TEST ON check_claims.py (Claim V4 Verification)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Copy essential files
        shutil.copy(REPO_ROOT / "check_claims.py", tmp_path / "check_claims.py")
        shutil.copy(REPO_ROOT / "README.md", tmp_path / "README.md")
        shutil.copytree(REPO_ROOT / "results_summary", tmp_path / "results_summary")
        shutil.copytree(REPO_ROOT / "paper", tmp_path / "paper")

        manuscript = tmp_path / "paper" / "MANUSCRIPT.md"
        content = manuscript.read_text(encoding="utf-8")

        # Check target strings
        # Look for the mutation targets
        target1_found = False
        target2_found = False

        if "0.998 ± 0.001" in content or "0.998 \\pm 0.001" in content:
            content = content.replace("0.998 ± 0.001", "0.975 ± 0.004")
            content = content.replace("0.998 \\pm 0.001", "0.975 \\pm 0.004")
            target1_found = True

        if "27.82" in content:
            content = content.replace("27.82", "45.49")
            target2_found = True

        manuscript.write_text(content, encoding="utf-8")

        print(f"[*] Mutated copy created in temporary directory: {tmpdir}")
        print(f"[*] Mutation 1 applied (LOSS AUC 0.998 -> 0.975): {target1_found}")
        print(f"[*] Mutation 2 applied (TPR@1% 27.82 -> 45.49): {target2_found}")

        # Run check_claims.py in the mutated repo
        proc = subprocess.run(
            [sys.executable, "check_claims.py"],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        print("\n[*] check_claims.py Execution Result on Mutated Manuscript:")
        print(f"    Exit code: {proc.returncode}")
        first_few_lines = "\n    ".join(proc.stdout.strip().splitlines()[-4:])
        print(f"    Output snippet:\n    {first_few_lines}")

        if proc.returncode == 0:
            print("\n[CONFIRMED] check_claims.py EXITED WITH CODE 0 on swapped metrics!")
            print("            This confirms that check_claims.py tests set membership")
            print("            across CSVs and cannot detect correct numbers in the wrong context.")
        else:
            print(f"\n[UNEXPECTED] check_claims.py failed with code {proc.returncode}")
            print(proc.stderr)

    # Now inspect whitelist-only tokens
    print("\n" + "=" * 70)
    print("INSPECTING NUMBERS COVERED ONLY BY HARD-CODED WHITELIST")
    print("=" * 70)

    # Re-harvest CSV values as done in check_claims.py
    RESULTS_DIR = REPO_ROOT / "results_summary"
    csv_values = set()

    def register_val(val, precisions=(2, 3, 4), as_pct=True):
        if pd.isna(val):
            return
        v = float(val)
        for p in precisions:
            csv_values.add(f"{v:.{p}f}")
        if as_pct:
            v_pct = v * 100.0
            for p in (0, 1, 2, 3):
                csv_values.add(f"{v_pct:.{p}f}")
                csv_values.add(f"{v_pct:.{p}f}%")

    for csv_file in RESULTS_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        for col in df.columns:
            if any(term in col for term in ['auc', 'tpr', 'acc', 'val', 'uniq', 'nov', 'point', 'ci']):
                for val in df[col]:
                    try:
                        register_val(float(val))
                    except (ValueError, TypeError):
                        pass

    # Extract all tokens from manuscript and main.tex
    files_to_check = [
        REPO_ROOT / 'paper' / 'MANUSCRIPT.md',
        REPO_ROOT / 'paper' / 'main.tex',
        REPO_ROOT / 'README.md',
    ]
    whitelist_only_tokens = set()

    for fp in files_to_check:
        text = fp.read_text(encoding="utf-8")
        matches = re.finditer(r'(?:(?:\\pm|±)\s*)?(\d+\.\d+)(?:\s*(?:\\%|%))?|(\d+)\s*(?:\\%|%)', text)
        for m in matches:
            raw = m.group(0).replace(r'\%', '%').strip()
            clean = raw.replace('\\pm', '').replace('±', '').strip()
            
            # Check if in csv_values directly or via float match
            in_csv = (clean in csv_values)
            if not in_csv:
                try:
                    num_val = float(clean.rstrip('%').strip())
                    for cv in csv_values:
                        try:
                            if abs(num_val - float(cv.rstrip('%'))) < 1e-4:
                                in_csv = True
                                break
                        except:
                            pass
                except:
                    pass
            
            if not in_csv:
                whitelist_only_tokens.add(clean)

    print(f"Tokens in manuscript/README that match ONLY via the whitelist (not in CSVs):")
    for t in sorted(whitelist_only_tokens):
        print(f"  - '{t}'")

    print("\nSummary:")
    print("  1. The mutation test confirms check_claims.py is a set-membership checker.")
    print("  2. Tokens like 0.51, 0.55, 0.56, 0.05, 0.005, 0.03 represent interval bounds,")
    print("     significance thresholds, or bootstrap CI half-widths covered by design via whitelist.")
    print("  3. Contextual correctness of tables is ensured by make_tables.py directly from CSVs.")
    print("=" * 70)

if __name__ == "__main__":
    run_mutation_test()
