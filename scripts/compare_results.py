"""
scripts/compare_results.py
===========================
Merge the baseline results with VMI-FGSM results and produce:
  - A comparison table (printed + saved as CSV)
  - A bar chart (saved as PNG)
  - A per-goal breakdown table
  - An attacker-victim heatmap CSV

Usage
-----
python scripts/compare_results.py \
    --baseline_eval  results_baseline/subset_attack_eval_long.csv \
    --new_eval       results_new/new_attack_eval_long.csv \
    --output_dir     results_new/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Try importing matplotlib; gracefully degrade if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    print("[WARN] matplotlib not available – skipping plot generation.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_eval",  required=True)
    p.add_argument("--new_eval",       required=True)
    p.add_argument("--output_dir",     default="results_new/")
    return p.parse_args()


def load_and_validate(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"attack", "breach", "goal", "attacker_model", "victim_model"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] {label} is missing columns: {missing}")
        sys.exit(1)
    print(f"Loaded {label}: {len(df)} rows, attacks={df['attack'].unique().tolist()}")
    return df


def breach_rate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("attack")["breach"]
        .agg(total="count", breaches="sum")
        .assign(breach_rate=lambda d: (d["breaches"] / d["total"] * 100).round(2))
        .reset_index()
        .rename(columns={"breach_rate": "breach_rate_%"})
    )


def breach_rate_by_goal(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["attack", "goal"])["breach"]
        .agg(total="count", breaches="sum")
        .assign(breach_rate=lambda d: (d["breaches"] / d["total"] * 100).round(2))
        .reset_index()
        .rename(columns={"breach_rate": "breach_rate_%"})
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    base_df = load_and_validate(args.baseline_eval, "baseline")
    new_df  = load_and_validate(args.new_eval,      "new (VMI-FGSM)")

    combined = pd.concat([base_df, new_df], ignore_index=True)

    # ---- Overall comparison table ----
    overall = breach_rate(combined).sort_values("breach_rate_%", ascending=False)
    print("\n=== Overall Breach Rate Comparison ===")
    print(overall.to_string(index=False))
    overall.to_csv(os.path.join(args.output_dir, "comparison_overall.csv"), index=False)

    # ---- By-goal comparison ----
    by_goal = breach_rate_by_goal(combined)
    print("\n=== Breach Rate by Goal ===")
    print(by_goal.to_string(index=False))
    by_goal.to_csv(os.path.join(args.output_dir, "comparison_by_goal.csv"), index=False)

    # ---- Attacker-victim heatmap data ----
    av = (
        combined.groupby(["attack", "attacker_model", "victim_model"])["breach"]
        .agg(total="count", breaches="sum")
        .assign(breach_rate=lambda d: (d["breaches"] / d["total"] * 100).round(2))
        .reset_index()
        .rename(columns={"breach_rate": "breach_rate_%"})
    )
    av.to_csv(os.path.join(args.output_dir, "comparison_attacker_victim.csv"), index=False)
    print(f"\nAttacker-victim breakdown saved.")

    # ---- Delta table: VMI-FGSM vs each baseline ----
    vmi_rate = overall.loc[overall["attack"] == "VMI-FGSM", "breach_rate_%"].values
    if len(vmi_rate) > 0:
        vmi_val = vmi_rate[0]
        deltas = overall.copy()
        deltas["delta_vs_VMI-FGSM"] = (vmi_val - deltas["breach_rate_%"]).round(2)
        print("\n=== Delta vs VMI-FGSM (positive = VMI-FGSM better) ===")
        print(deltas[["attack", "breach_rate_%", "delta_vs_VMI-FGSM"]].to_string(index=False))
        deltas.to_csv(os.path.join(args.output_dir, "comparison_delta.csv"), index=False)

    # ---- Bar chart ----
    if _MPL:
        _make_bar_chart(overall, args.output_dir)
        _make_goal_chart(by_goal, args.output_dir)
    else:
        print("Skipping charts (matplotlib unavailable).")

    print(f"\nAll comparison files saved to {args.output_dir}")


def _make_bar_chart(overall: pd.DataFrame, out_dir: str):
    """Horizontal bar chart: breach rate per attack, VMI-FGSM highlighted."""
    fig, ax = plt.subplots(figsize=(8, 5))

    attacks = overall["attack"].tolist()
    rates   = overall["breach_rate_%"].tolist()
    colors  = ["#d62728" if a == "VMI-FGSM" else "#4878d0" for a in attacks]

    bars = ax.barh(attacks, rates, color=colors, height=0.6, edgecolor="white")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%",
            va="center", ha="left", fontsize=10,
        )

    ax.set_xlabel("Breach Rate (%)", fontsize=11)
    ax.set_title("Transfer Attack Comparison — Subset Results\n(Red = VMI-FGSM new attack)", fontsize=12)
    ax.set_xlim(0, max(rates) * 1.18 if rates else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_bar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved: {path}")


def _make_goal_chart(by_goal: pd.DataFrame, out_dir: str):
    """Grouped bar chart: breach rate per attack split by goal."""
    attacks = by_goal["attack"].unique().tolist()
    goals   = ["impersonation", "dodging"]
    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, goal in enumerate(goals):
        sub = by_goal[by_goal["goal"] == goal].set_index("attack")
        rates = [sub.loc[a, "breach_rate_%"] if a in sub.index else 0.0 for a in attacks]
        bars = ax.bar(x + i * width, rates, width, label=goal.capitalize(),
                      color="#4878d0" if goal == "impersonation" else "#ee854a",
                      edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(attacks, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Breach Rate (%)", fontsize=11)
    ax.set_title("Breach Rate by Goal per Attack", fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_by_goal_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"By-goal chart saved: {path}")


if __name__ == "__main__":
    run(parse_args())
