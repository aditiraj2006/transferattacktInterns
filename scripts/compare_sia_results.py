"""
scripts/compare_sia_results.py
Merge baseline + SIA results and print comparison.
Usage: python scripts/compare_sia_results.py
"""
import pandas as pd, os

BASE = "results_baseline/subset_attack_summary.csv"
NEW  = "experiments/results_new/SIA_attack_summary.csv"
BASE_EVAL = "results_baseline/subset_attack_eval_long.csv"
NEW_EVAL  = "experiments/results_new/SIA_attack_eval_long.csv"

def normalize_summary(df):
    if "attack_method" in df.columns:
        df = df.rename(columns={"attack_method": "attack"})
    if "breach_rate" in df.columns:
        if df["breach_rate"].max() <= 1.0:
            df["breach_rate_%"] = (df["breach_rate"] * 100).round(2)
        else:
            df["breach_rate_%"] = df["breach_rate"].round(2)
    elif "breach_rate_pct" in df.columns:
        df["breach_rate_%"] = df["breach_rate_pct"].round(2)
    return df


def main():
    if not os.path.exists(NEW):
        print("Run run_sia_attack.py first"); return
    b = normalize_summary(pd.read_csv(BASE))
    n = normalize_summary(pd.read_csv(NEW))
    comb = pd.concat([b[["attack","breach_rate_%"]], n[["attack","breach_rate_%"]]])
    comb = comb.sort_values("breach_rate_%", ascending=False)
    print("\n=== OVERALL BREACH RATE ===")
    for _, r in comb.iterrows():
        m = " <- SIA (new)" if r["attack"]=="SIA" else ""
        print(f"  {r['attack']:<22} {r['breach_rate_%']:>7.2f}%{m}")
    comb.to_csv("experiments/results_new/final_comparison.csv", index=False)

    if os.path.exists(BASE_EVAL) and os.path.exists(NEW_EVAL):
        be = pd.read_csv(BASE_EVAL)
        ne = pd.read_csv(NEW_EVAL)
        if "attack_method" in be.columns:
            be = be.rename(columns={"attack_method": "attack"})
        all_e = pd.concat([be, ne])
        bg = (all_e.groupby(["attack","goal"])["breach"]
              .agg(total="count", breaches="sum")
              .assign(breach_rate=lambda d:(d["breaches"]/d["total"]*100).round(2))
              .reset_index())
        print("\n=== BY GOAL ===")
        print(bg.to_string(index=False))
        bg.to_csv("experiments/results_new/comparison_by_goal.csv", index=False)
    print("\nDone. Check experiments/results_new/")

if __name__ == "__main__":
    main()