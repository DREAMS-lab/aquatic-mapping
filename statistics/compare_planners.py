#!/usr/bin/env python3
"""
Statistical Comparison: Exact vs Pose-Aware Planner

Auto-discovers all trials, all fields. Publication-quality figures.

Usage:
    python3 compare_planners.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Config
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
})

SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE = SCRIPT_DIR.parent
TRIALS_DIR = WORKSPACE / "data" / "trials"
OUTPUT_DIR = WORKSPACE / "data" / "statistics"

FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
FIELD_LABELS = ["Radial", "X-Compress", "Y-Compress", "X-Tilt", "Y-Tilt"]
def _discover_trial_range():
    """Auto-discover available trial numbers from disk."""
    trials_dir = Path(__file__).parent.absolute().parent / "data" / "trials"
    trial_nums = set()
    for planner in ["exact", "pose_aware"]:
        for field_dir in (trials_dir / planner).glob("*/"):
            for trial_dir in field_dir.glob("trial_*"):
                if (trial_dir / "summary.json").exists():
                    try:
                        num = int(trial_dir.name.split("_")[-1])
                        trial_nums.add(num)
                    except ValueError:
                        pass
    return range(1, max(trial_nums) + 1) if trial_nums else range(1, 11)

TRIAL_RANGE = _discover_trial_range()


def load_trials():
    """Load all paired trial data."""
    rows = []
    for field in FIELDS:
        for trial_num in TRIAL_RANGE:
            trial_str = f"trial_{trial_num:03d}"
            exact_file = TRIALS_DIR / "exact" / field / trial_str / "summary.json"
            pose_file = TRIALS_DIR / "pose_aware" / field / trial_str / "summary.json"

            if not (exact_file.exists() and pose_file.exists()):
                continue

            for planner, f in [("exact", exact_file), ("pose_aware", pose_file)]:
                with open(f) as fp:
                    s = json.load(fp)
                rows.append({
                    "field": field,
                    "trial": trial_num,
                    "planner": planner,
                    "rmse": s.get("reconstruction_rmse"),
                    "mae": s.get("reconstruction_mae"),
                    "travel": s.get("total_travel_cost"),
                    "info_gain": s.get("cumulative_info_gain"),
                })
    df = pd.DataFrame(rows)
    df["efficiency"] = df["info_gain"] / df["travel"]
    return df


def cohens_d(x1, x2):
    diff = x1 - x2
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else 0


def cohens_d_ci(x1, x2, confidence=0.95):
    """Cohen's d with confidence interval (paired formula)."""
    n = len(x1)
    d = cohens_d(x1, x2)
    # Paired SE: Hedges & Olkin (1985), correct for dependent samples
    se = np.sqrt(1.0 / n + d**2 / (2 * (n - 1)))
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci_low = d - t_crit * se
    ci_high = d + t_crit * se
    return d, ci_low, ci_high


def effect_label(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def plot_paired_lines(df, output_dir):
    """
    Slope graph: lines connecting exact to pose-aware for each trial.
    Shows individual trial improvements.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = df.pivot_table(index=["field", "trial"], columns="planner", values="rmse").dropna()

    # Plot each trial as a line
    for idx in pivot.index:
        exact_val = pivot.loc[idx, "exact"]
        pose_val = pivot.loc[idx, "pose_aware"]
        color = "green" if pose_val < exact_val else "red"
        alpha = 0.4
        ax.plot([0, 1], [exact_val, pose_val], color=color, alpha=alpha, linewidth=1)

    # Add means
    exact_mean = pivot["exact"].mean()
    pose_mean = pivot["pose_aware"].mean()
    ax.plot([0, 1], [exact_mean, pose_mean], color="black", linewidth=3, marker="o", markersize=10, label=f"Mean")

    # Styling
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Exact Planner", "Pose-Aware Planner"], fontsize=12)
    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_title(f"Paired Trial Comparison (n={len(pivot)})\nGreen = Pose-Aware better, Red = Exact better", fontsize=12)

    # Add counts
    n_pose_better = np.sum(pivot["pose_aware"] < pivot["exact"])
    n_exact_better = np.sum(pivot["pose_aware"] > pivot["exact"])
    ax.text(0.5, 0.98, f"Pose-Aware better: {n_pose_better}/{len(pivot)} ({100*n_pose_better/len(pivot):.0f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    ax.set_xlim(-0.3, 1.3)
    plt.tight_layout()
    plt.savefig(output_dir / "paired_lines.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_bars(df, output_dir):
    """Bar chart with error bars (SEM) by field."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(FIELDS))
    width = 0.35

    exact_means, exact_sems = [], []
    pose_means, pose_sems = [], []

    for field in FIELDS:
        exact_vals = df[(df["field"] == field) & (df["planner"] == "exact")]["rmse"].values
        pose_vals = df[(df["field"] == field) & (df["planner"] == "pose_aware")]["rmse"].values
        exact_means.append(np.mean(exact_vals))
        exact_sems.append(stats.sem(exact_vals))
        pose_means.append(np.mean(pose_vals))
        pose_sems.append(stats.sem(pose_vals))

    bars1 = ax.bar(x - width/2, exact_means, width, yerr=exact_sems, label="Exact",
                   color="#d62728", alpha=0.8, capsize=4, error_kw={"linewidth": 1.5})
    bars2 = ax.bar(x + width/2, pose_means, width, yerr=pose_sems, label="Pose-Aware",
                   color="#2ca02c", alpha=0.8, capsize=4, error_kw={"linewidth": 1.5})

    ax.set_ylabel("RMSE (°C)")
    ax.set_xlabel("Field Type")
    ax.set_title("Reconstruction Error by Field (mean ± SEM)")
    ax.set_xticks(x)
    ax.set_xticklabels(FIELD_LABELS)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # Add significance stars
    for i, field in enumerate(FIELDS):
        exact_vals = df[(df["field"] == field) & (df["planner"] == "exact")]["rmse"].values
        pose_vals = df[(df["field"] == field) & (df["planner"] == "pose_aware")]["rmse"].values
        # Match by trial
        e_df = df[(df["field"] == field) & (df["planner"] == "exact")].set_index("trial")["rmse"]
        p_df = df[(df["field"] == field) & (df["planner"] == "pose_aware")].set_index("trial")["rmse"]
        common = e_df.index.intersection(p_df.index)
        if len(common) >= 2:
            _, p_val = stats.ttest_rel(e_df.loc[common], p_df.loc[common])
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                max_height = max(exact_means[i] + exact_sems[i], pose_means[i] + pose_sems[i])
                ax.text(i, max_height + 0.1, star, ha="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "field_bars.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_forest(df, output_dir):
    """Forest plot: Cohen's d with 95% CI for each field."""
    fig, ax = plt.subplots(figsize=(8, 5))

    results = []
    for i, field in enumerate(FIELDS):
        e_df = df[(df["field"] == field) & (df["planner"] == "exact")].set_index("trial")["rmse"]
        p_df = df[(df["field"] == field) & (df["planner"] == "pose_aware")].set_index("trial")["rmse"]
        common = e_df.index.intersection(p_df.index)
        if len(common) >= 2:
            d, ci_low, ci_high = cohens_d_ci(e_df.loc[common].values, p_df.loc[common].values)
            results.append({"field": field, "d": d, "ci_low": ci_low, "ci_high": ci_high, "n": len(common)})

    # Add overall
    pivot = df.pivot_table(index=["field", "trial"], columns="planner", values="rmse").dropna()
    d, ci_low, ci_high = cohens_d_ci(pivot["exact"].values, pivot["pose_aware"].values)
    results.append({"field": "OVERALL", "d": d, "ci_low": ci_low, "ci_high": ci_high, "n": len(pivot)})

    # Plot
    y_positions = list(range(len(results)))
    for i, r in enumerate(results):
        color = "black" if r["field"] == "OVERALL" else "steelblue"
        weight = "bold" if r["field"] == "OVERALL" else "normal"
        markersize = 10 if r["field"] == "OVERALL" else 8

        ax.errorbar(r["d"], i, xerr=[[r["d"] - r["ci_low"]], [r["ci_high"] - r["d"]]],
                    fmt="o", color=color, markersize=markersize, capsize=5, capthick=2, linewidth=2)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=0.8, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Large effect (0.8)")
    ax.axvline(x=0.5, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="Medium effect (0.5)")

    ax.set_yticks(y_positions)
    labels = [FIELD_LABELS[FIELDS.index(r["field"])] if r["field"] != "OVERALL" else "OVERALL" for r in results]
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (with 95% CI)")
    ax.set_title("Effect Size by Field\n(Positive = Pose-Aware better)")
    ax.legend(loc="lower right", fontsize=9)

    # Add text annotations
    for i, r in enumerate(results):
        ax.text(r["ci_high"] + 0.1, i, f"d={r['d']:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_improvement_distribution(df, output_dir):
    """Histogram of RMSE improvement (exact - pose_aware)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pivot = df.pivot_table(index=["field", "trial"], columns="planner", values="rmse").dropna()
    improvement = pivot["exact"] - pivot["pose_aware"]

    # Histogram
    n, bins, patches = ax.hist(improvement, bins=15, edgecolor="black", alpha=0.7, color="steelblue")

    # Color bars by sign
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0:
            patch.set_facecolor("green")
        else:
            patch.set_facecolor("red")

    ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
    ax.axvline(x=improvement.mean(), color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {improvement.mean():.3f}°C")

    ax.set_xlabel("RMSE Improvement (Exact − Pose-Aware) [°C]")
    ax.set_ylabel("Number of Trials")
    ax.set_title("Distribution of RMSE Improvement\nGreen = Pose-Aware better, Red = Exact better")
    ax.legend(loc="upper right")

    # Add summary text
    n_improved = np.sum(improvement > 0)
    n_total = len(improvement)
    ax.text(0.02, 0.98, f"Pose-Aware better: {n_improved}/{n_total} ({100*n_improved/n_total:.0f}%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_hist.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_figure(df, output_dir):
    """Combined 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    pivot = df.pivot_table(index=["field", "trial"], columns="planner", values="rmse").dropna()

    # 1. Paired scatter
    ax = axes[0, 0]
    ax.scatter(pivot["exact"], pivot["pose_aware"], alpha=0.6, s=50, c="steelblue", edgecolors="black", linewidth=0.5)
    lims = [min(pivot.min().min(), 1.5), max(pivot.max().max(), 4.5)]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5)
    ax.fill_between(lims, lims, [lims[0], lims[0]], alpha=0.1, color="green", label="Pose-Aware better")
    ax.set_xlabel("Exact Planner RMSE (°C)")
    ax.set_ylabel("Pose-Aware Planner RMSE (°C)")
    ax.set_title("A) Paired Comparison")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

    # 2. Field means
    ax = axes[0, 1]
    x = np.arange(len(FIELDS))
    width = 0.35
    exact_means = [df[(df["field"]==f) & (df["planner"]=="exact")]["rmse"].mean() for f in FIELDS]
    pose_means = [df[(df["field"]==f) & (df["planner"]=="pose_aware")]["rmse"].mean() for f in FIELDS]
    exact_sems = [stats.sem(df[(df["field"]==f) & (df["planner"]=="exact")]["rmse"]) for f in FIELDS]
    pose_sems = [stats.sem(df[(df["field"]==f) & (df["planner"]=="pose_aware")]["rmse"]) for f in FIELDS]

    ax.bar(x - width/2, exact_means, width, yerr=exact_sems, label="Exact", color="#d62728", alpha=0.8, capsize=3)
    ax.bar(x + width/2, pose_means, width, yerr=pose_sems, label="Pose-Aware", color="#2ca02c", alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(FIELD_LABELS, fontsize=9)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("B) RMSE by Field (mean ± SEM)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)

    # 3. Improvement histogram
    ax = axes[1, 0]
    improvement = pivot["exact"] - pivot["pose_aware"]
    n, bins, patches = ax.hist(improvement, bins=12, edgecolor="black", alpha=0.7)
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor("green" if left_edge >= 0 else "red")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
    ax.axvline(x=improvement.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {improvement.mean():.2f}°C")
    ax.set_xlabel("RMSE Improvement (°C)")
    ax.set_ylabel("Count")
    ax.set_title("C) Improvement Distribution")
    ax.legend(loc="upper right", fontsize=9)

    # 4. Effect sizes
    ax = axes[1, 1]
    effect_data = []
    for field in FIELDS:
        e_df = df[(df["field"] == field) & (df["planner"] == "exact")].set_index("trial")["rmse"]
        p_df = df[(df["field"] == field) & (df["planner"] == "pose_aware")].set_index("trial")["rmse"]
        common = e_df.index.intersection(p_df.index)
        if len(common) >= 2:
            d = cohens_d(e_df.loc[common].values, p_df.loc[common].values)
            effect_data.append(d)

    # Overall
    d_overall = cohens_d(pivot["exact"].values, pivot["pose_aware"].values)

    colors = ["steelblue"] * len(FIELDS) + ["darkblue"]
    y_pos = np.arange(len(FIELDS) + 1)
    bars = ax.barh(y_pos, effect_data + [d_overall], color=colors, alpha=0.8, edgecolor="black")
    ax.axvline(x=0.8, color="red", linestyle=":", linewidth=1.5, label="Large (0.8)")
    ax.axvline(x=0.5, color="orange", linestyle=":", linewidth=1.5, label="Medium (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FIELD_LABELS + ["OVERALL"], fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title("D) Effect Size by Field")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_figure.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_results(df):
    """Print statistical summary."""
    print("=" * 70)
    print(f"PLANNER COMPARISON: Trials {TRIAL_RANGE.start}-{TRIAL_RANGE.stop - 1}")
    print("=" * 70)

    pivot = df.pivot_table(index=["field", "trial"], columns="planner", values="rmse").dropna()
    n_total = len(pivot)

    print(f"\nTotal paired trials: {n_total}")
    print(f"Fields: {len(FIELDS)}")
    print(f"Trials per field: {len(TRIAL_RANGE)}")

    # Overall
    print("\n" + "-" * 70)
    print("OVERALL RESULTS")
    print("-" * 70)

    exact = pivot["exact"].values
    pose = pivot["pose_aware"].values
    diff = exact - pose

    print(f"Exact mean RMSE:      {np.mean(exact):.3f} ± {np.std(exact, ddof=1):.3f} °C")
    print(f"Pose-Aware mean RMSE: {np.mean(pose):.3f} ± {np.std(pose, ddof=1):.3f} °C")
    print(f"Mean improvement:     {np.mean(diff):.3f} °C ({100*np.mean(diff)/np.mean(exact):.1f}%)")

    d = cohens_d(exact, pose)
    t_stat, t_pval = stats.ttest_rel(exact, pose)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(exact, pose)
    except ValueError:
        w_stat, w_pval = np.nan, np.nan

    print(f"\nCohen's d: {d:.3f} ({effect_label(d)})")
    print(f"Paired t-test: t={t_stat:.3f}, p={t_pval:.6f}")
    print(f"Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.6f}" if not np.isnan(w_pval) else "Wilcoxon: N/A")

    n_pose_wins = np.sum(diff > 0)
    print(f"\nPose-Aware wins: {n_pose_wins}/{n_total} ({100*n_pose_wins/n_total:.1f}%)")

    # Per field
    print("\n" + "-" * 70)
    print("PER-FIELD BREAKDOWN")
    print("-" * 70)
    print(f"{'Field':<15} {'n':>4} {'Exact':>8} {'Pose':>8} {'Δ':>8} {'d':>8} {'Effect':>10} {'Wins':>8}")
    print("-" * 70)

    for field, label in zip(FIELDS, FIELD_LABELS):
        e_df = df[(df["field"] == field) & (df["planner"] == "exact")].set_index("trial")["rmse"]
        p_df = df[(df["field"] == field) & (df["planner"] == "pose_aware")].set_index("trial")["rmse"]
        common = e_df.index.intersection(p_df.index)

        e = e_df.loc[common].values
        p = p_df.loc[common].values
        d = cohens_d(e, p)
        wins = np.sum(e > p)

        print(f"{label:<15} {len(common):>4} {np.mean(e):>8.3f} {np.mean(p):>8.3f} "
              f"{np.mean(e-p):>+8.3f} {d:>+8.2f} {effect_label(d):>10} {wins:>4}/{len(common)}")

    print("-" * 70)

    sig_t = t_pval < 0.05
    sig_w = (not np.isnan(w_pval)) and w_pval < 0.05
    if sig_t and sig_w:
        print("\n✓ SIGNIFICANT: Pose-aware produces significantly lower RMSE")
        print(f"  (paired t-test p={t_pval:.6f}, Wilcoxon p={w_pval:.6f})")
    elif sig_t:
        print(f"\n✓ Significant by paired t-test (p={t_pval:.6f}), not by Wilcoxon (p={w_pval:.6f})")
    elif sig_w:
        print(f"\n✓ Significant by Wilcoxon (p={w_pval:.6f}), not by paired t-test (p={t_pval:.6f})")
    else:
        print("\n✗ NOT SIGNIFICANT: No significant difference (p >= 0.05)")


def main():
    df = load_trials()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_results(df)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_paired_lines(df, OUTPUT_DIR)
    print(f"Saved: paired_lines.png")

    plot_field_bars(df, OUTPUT_DIR)
    print(f"Saved: field_bars.png")

    plot_forest(df, OUTPUT_DIR)
    print(f"Saved: forest_plot.png")

    plot_improvement_distribution(df, OUTPUT_DIR)
    print(f"Saved: improvement_hist.png")

    plot_summary_figure(df, OUTPUT_DIR)
    print(f"Saved: summary_figure.png")

    # Save data
    df.to_csv(OUTPUT_DIR / "all_data.csv", index=False)
    print(f"Saved: all_data.csv")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
