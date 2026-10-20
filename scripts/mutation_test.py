#!/usr/bin/env python3
"""
scripts/mutation_test.py

Mutation Testing Suite for Claims Verifier.
Validates that scripts/verify_claims.py strictly detects and rejects corrupted data.
Implements at least 8 distinct mutations across experimental arms.
Ensures zero false negatives (every mutation MUST cause verify_claims.py to exit non-zero).
All files are restored safely via try...finally blocks.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_summary"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_claims.py"
PYTHON_BIN = sys.executable


def run_verifier():
    """Runs verify_claims.py and returns exit code."""
    res = subprocess.run(
        [PYTHON_BIN, str(VERIFY_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    return res.returncode, res.stdout, res.stderr


class MutationRunner:
    def __init__(self):
        self.passed_mutations = 0
        self.failed_mutations = 0

    def test_mutation(self, name, target_csv, mutate_fn):
        print(f"\n[MUTATION TEST] {name} ...")
        csv_path = RESULTS_DIR / target_csv
        original_content = csv_path.read_bytes()

        try:
            df = pd.read_csv(csv_path)
            mutated_df = mutate_fn(df)
            mutated_df.to_csv(csv_path, index=False)

            code, stdout, stderr = run_verifier()
            if code != 0:
                print(f"  -> DETECTED (Exit code {code})")
                self.passed_mutations += 1
            else:
                print(f"  -> MISSED! Verifier returned code 0 on corrupted data!")
                print(stdout)
                self.failed_mutations += 1
        finally:
            csv_path.write_bytes(original_content)


def main():
    print("=" * 70)
    print("Running Mutation Testing Suite for Independent Claims Verifier")
    print("=" * 70)

    # Sanity check: baseline unmutated verifier must exit 0
    code, stdout, stderr = run_verifier()
    if code != 0:
        print("ERROR: Unmutated baseline verifier did not exit 0!")
        print(stdout)
        print(stderr)
        sys.exit(1)
    print("Baseline verification: PASSED (clean ground truth exits 0)\n")

    runner = MutationRunner()

    # Mutation 1: Swap LOSS and Online LiRA AUC values in baseline_attack_summary.csv at epoch 60
    def m1(df):
        m_loss = (df["epoch"] == 60) & (df["stat"] == "nll_sum") & (df["method"] == "LOSS")
        m_lira = (df["epoch"] == 60) & (df["stat"] == "nll_sum") & (df["method"] == "online_LiRA")
        val_loss = df.loc[m_loss, "auc_mean"].values[0]
        val_lira = df.loc[m_lira, "auc_mean"].values[0]
        df.loc[m_loss, "auc_mean"] = val_lira
        df.loc[m_lira, "auc_mean"] = val_loss
        return df
    runner.test_mutation("1. Swap LOSS and Online LiRA AUC at Epoch 60", "baseline_attack_summary.csv", m1)

    # Mutation 2: Swap N=1,000 and N=10,000 AUC values in E3_scaling_summary.csv
    def m2(df):
        m_1k = (df["member_size"] == 1000) & (df["method"] == "LOSS")
        m_10k = (df["member_size"] == 10000) & (df["method"] == "LOSS")
        v_1k = df.loc[m_1k, "auc_mean"].values[0]
        v_10k = df.loc[m_10k, "auc_mean"].values[0]
        df.loc[m_1k, "auc_mean"] = v_10k
        df.loc[m_10k, "auc_mean"] = v_1k
        return df
    runner.test_mutation("2. Swap N=1,000 and N=10,000 LOSS AUC in E3", "E3_scaling_summary.csv", m2)

    # Mutation 3: Swap Epoch 5 and Epoch 20 Offline LiRA AUC in baseline_attack_summary.csv
    def m3(df):
        m_5 = (df["epoch"] == 5) & (df["stat"] == "nll_sum") & (df["method"] == "offline_LiRA")
        m_20 = (df["epoch"] == 20) & (df["stat"] == "nll_sum") & (df["method"] == "offline_LiRA")
        v_5 = df.loc[m_5, "auc_mean"].values[0]
        v_20 = df.loc[m_20, "auc_mean"].values[0]
        df.loc[m_5, "auc_mean"] = v_20
        df.loc[m_20, "auc_mean"] = v_5
        return df
    runner.test_mutation("3. Swap Epoch 5 and Epoch 20 Offline LiRA AUC", "baseline_attack_summary.csv", m3)

    # Mutation 4: Change standard deviation of Epoch 60 Online LiRA
    def m4(df):
        m = (df["epoch"] == 60) & (df["stat"] == "nll_sum") & (df["method"] == "online_LiRA")
        df.loc[m, "auc_std"] = 0.045
        return df
    runner.test_mutation("4. Change Online LiRA AUC std from 0.003 to 0.045", "baseline_attack_summary.csv", m4)

    # Mutation 5: Change copy count in Table 5 (E6)
    def m5(df):
        m = df["model"] == 31
        df.loc[m, "exact_member_copy_fraction_all_samples"] = 0.0080  # was 0.0016 (40 copies instead of 8)
        return df
    runner.test_mutation("5. Corrupt Model 31 member copy count in E6", "E6_generative_audit_summary.csv", m5)

    # Mutation 6: Change bootstrap CI upper bound
    def m6(df):
        m = (df["target"] == 0) & (df["metric"] == "online_LiRA_AUC")
        df.loc[m, "ci_95_high"] = 0.9500  # was 0.9045
        return df
    runner.test_mutation("6. Corrupt Target 0 online LiRA AUC bootstrap CI upper bound", "bootstrap_ci.csv", m6)

    # Mutation 7: Corrupt transfer learning Pretrain-FT AUC
    def m7(df):
        m = (df["target"] == 0) & (df["method"] == "offline_LiRA")
        df.loc[m, "auc"] = 0.5000  # was 0.799
        return df
    runner.test_mutation("7. Corrupt Target 0 offline LiRA AUC in pretrain-FT", "pretrain_ft_summary.csv", m7)

    # Mutation 8: Swap DP-SGD epsilon 8.0 and 3.0 AUC
    def m8(df):
        m_8 = (df["epsilon_target"] == 8) & (df["method"] == "offline_LiRA")
        m_3 = (df["epsilon_target"] == 3) & (df["method"] == "offline_LiRA")
        v_8 = df.loc[m_8, "auc_mean"].values[0]
        v_3 = df.loc[m_3, "auc_mean"].values[0]
        df.loc[m_8, "auc_mean"] = v_3
        df.loc[m_3, "auc_mean"] = v_8
        return df
    runner.test_mutation("8. Swap DP-SGD eps=8 and eps=3 offline LiRA AUC", "DP_SGD_summary.csv", m8)

    # Mutation 9: Corrupt E5a randomized variants scoring AUC
    def m9(df):
        m = (df["epoch"] == 60) & (df["stat"] == "nll_variants_mean") & (df["method"] == "offline_LiRA")
        df.loc[m, "auc_mean"] = 0.750  # was 0.552
        return df
    runner.test_mutation("9. Corrupt E5a variants scoring offline LiRA AUC", "E5a_augmentation_summary.csv", m9)

    print("\n" + "=" * 70)
    print(f"Mutation Testing Results: {runner.passed_mutations} detected, {runner.failed_mutations} missed.")
    print("=" * 70)

    if runner.failed_mutations > 0:
        print("FAIL: One or more mutations escaped detection!")
        sys.exit(1)

    print("ALL MUTATIONS SUCCESSFULLY DETECTED BY VERIFIER.")
    sys.exit(0)


if __name__ == "__main__":
    main()
