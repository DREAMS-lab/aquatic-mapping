#!/usr/bin/env python3
"""
Unified 5-Planner Statistical Comparison

Auto-discovers all trials for all 5 planners across all fields.
Generates publication-quality figures and statistical tests.

Planners: exact, pose_aware, analytical, nonstationary_exact, nonstationary_pose_aware

Usage:
    python3 compare_all_planners.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations
try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTITEST = True
except ImportError:
    HAS_MULTITEST = False

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

PLANNERS = [
    "exact", "pose_aware", "analytical",
    "nonstationary_exact", "nonstationary_pose_aware",
    "nonstationary_hotspot_exact", "nonstationary_hotspot_pose_aware",
]
PLANNER_LABELS = {
    "exact": "Exact",
    "pose_aware": "Pose-Aware",
    "analytical": "Analytical",
    "nonstationary_exact": "NS-Exact",
    "nonstationary_pose_aware": "NS-PoseAware",
    "nonstationary_hotspot_exact": "NS-HS-Exact",
    "nonstationary_hotspot_pose_aware": "NS-HS-PoseAware",
}
PLANNER_COLORS = {
    "exact": "#d62728",
    "pose_aware": "#2ca02c",
    "analytical": "#1f77b4",
    "nonstationary_exact": "#9467bd",
    "nonstationary_pose_aware": "#ff7f0e",
    "nonstationary_hotspot_exact": "#8c564b",
    "nonstationary_hotspot_pose_aware": "#e377c2",
}

FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
FIELD_LABELS = {
    "radial": "Radial",
    "x_compress": "X-Compress",
    "y_compress": "Y-Compress",
    "x_compress_tilt": "X-Tilt",
    "y_compress_tilt": "Y-Tilt",
}


def discover_trials():
    """Auto-discover available trial numbers from disk."""
    trial_nums = set()
    for planner in PLANNERS:
        planner_dir = TRIALS_DIR / planner
        if not planner_dir.exists():
            continue
        for field_dir in planner_dir.glob("*/"):
            for trial_dir in field_dir.glob("trial_*"):
                if (trial_dir / "summary.json").exists():
                    try:
                        num = int(trial_dir.name.split("_")[-1])
                        trial_nums.add(num)
                    except ValueError:
                        pass
    return sorted(trial_nums)


def load_all_trials():
    """Load all trial data for all planners."""
    trial_nums = discover_trials()
    rows = []
    for planner in PLANNERS:
        for field in FIELDS:
            for trial_num in trial_nums:
                trial_str = f"trial_{trial_num:03d}"
                summary_file = TRIALS_DIR / planner / field / trial_str / "summary.json"
                if not summary_file.exists():
                    continue
                with open(summary_file) as fp:
                    s = json.load(fp)

                # Read samples CSV for unique location count
                samples_file = TRIALS_DIR / planner / field / trial_str / "samples.csv"
                n_unique = s.get("n_samples", 100)
                if samples_file.exists():
                    try:
                        sdf = pd.read_csv(samples_file)
                        coords = sdf[['x', 'y']].round(1)
                        n_unique = len(coords.drop_duplicates())
                    except Exception:
                        pass

                rows.append({
                    "field": field,
                    "trial": trial_num,
                    "planner": planner,
                    "rmse": s.get("reconstruction_rmse"),
                    "mae": s.get("reconstruction_mae"),
                    "max_error": s.get("reconstruction_max_error"),
                    "travel": s.get("total_travel_cost"),
                    "info_gain": s.get("cumulative_info_gain"),
                    "n_samples": s.get("n_samples", 100),
                    "n_unique": n_unique,
                })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["sample_efficiency"] = df["n_unique"] / df["n_samples"]
        df["efficiency"] = df["info_gain"] / df["travel"].replace(0, np.nan)
    return df


def cohens_d(x1, x2):
    diff = x1 - x2
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else 0


def cohens_d_ci(x1, x2, confidence=0.95):
    n = len(x1)
    d = cohens_d(x1, x2)
    # Paired SE: Hedges & Olkin (1985), correct for dependent samples
    se = np.sqrt(1.0 / n + d**2 / (2 * (n - 1)))
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    return d, d - t_crit * se, d + t_crit * se


def effect_label(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


# ---- Plotting Functions ----

def plot_rmse_bars_all(df, output_dir):
    """RMSE bar chart for all planners across all fields."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]
    n_planners = len(planners)

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(FIELDS))
    width = 0.8 / n_planners

    for i, planner in enumerate(planners):
        means, sems = [], []
        for field in FIELDS:
            vals = df[(df["field"] == field) & (df["planner"] == planner)]["rmse"].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0)

        offset = (i - (n_planners - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems,
               label=PLANNER_LABELS[planner], color=PLANNER_COLORS[planner],
               alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_ylabel("RMSE (°C)")
    ax.set_xlabel("Field Type")
    ax.set_title("Reconstruction Error by Field and Planner (mean ± SEM)")
    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_LABELS[f] for f in FIELDS])
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_all_planners.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_efficiency(df, output_dir):
    """Unique locations vs total samples for each planner."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(planners))

    means = []
    sems = []
    for planner in planners:
        vals = df[df["planner"] == planner]["sample_efficiency"].dropna()
        means.append(vals.mean() if len(vals) > 0 else 0)
        sems.append(stats.sem(vals) if len(vals) > 1 else 0)

    ax.bar(x, means, yerr=sems, color=[PLANNER_COLORS[p] for p in planners],
           alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Sample Efficiency (unique / total)")
    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS[p] for p in planners], rotation=15)
    ax.set_title("Sample Efficiency (higher = fewer duplicates)")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_travel_comparison(df, output_dir):
    """Travel cost comparison across planners."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(planners))

    means = []
    sems = []
    for planner in planners:
        vals = df[df["planner"] == planner]["travel"].dropna()
        means.append(vals.mean() if len(vals) > 0 else 0)
        sems.append(stats.sem(vals) if len(vals) > 1 else 0)

    ax.bar(x, means, yerr=sems, color=[PLANNER_COLORS[p] for p in planners],
           alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Total Travel Cost (m)")
    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS[p] for p in planners], rotation=15)
    ax.set_title("Total Travel Cost by Planner (mean ± SEM)")

    plt.tight_layout()
    plt.savefig(output_dir / "travel_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_forest_all(df, output_dir):
    """Forest plot: Cohen's d with 95% CI for all planner pairs."""
    # Use exact as baseline
    baseline = "exact"
    available = df["planner"].unique()
    others = [p for p in PLANNERS if p in available and p != baseline]

    if baseline not in available or len(others) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(others) * len(FIELDS) * 0.3 + 2)))

    results = []
    for planner in others:
        for field in FIELDS:
            e_df = df[(df["field"] == field) & (df["planner"] == baseline)].set_index("trial")["rmse"]
            p_df = df[(df["field"] == field) & (df["planner"] == planner)].set_index("trial")["rmse"]
            common = e_df.index.intersection(p_df.index)
            if len(common) >= 2:
                d, ci_low, ci_high = cohens_d_ci(e_df.loc[common].values, p_df.loc[common].values)
                results.append({
                    "label": f"{PLANNER_LABELS[planner]} / {FIELD_LABELS[field]}",
                    "d": d, "ci_low": ci_low, "ci_high": ci_high,
                    "n": len(common), "planner": planner,
                })

        # Overall for this planner
        e_all = df[df["planner"] == baseline].set_index(["field", "trial"])["rmse"]
        p_all = df[df["planner"] == planner].set_index(["field", "trial"])["rmse"]
        common_all = e_all.index.intersection(p_all.index)
        if len(common_all) >= 2:
            d, ci_low, ci_high = cohens_d_ci(e_all.loc[common_all].values, p_all.loc[common_all].values)
            results.append({
                "label": f"{PLANNER_LABELS[planner]} OVERALL",
                "d": d, "ci_low": ci_low, "ci_high": ci_high,
                "n": len(common_all), "planner": planner,
            })

    if not results:
        plt.close()
        return

    for i, r in enumerate(results):
        color = PLANNER_COLORS.get(r["planner"], "gray")
        weight = "bold" if "OVERALL" in r["label"] else "normal"
        ms = 10 if "OVERALL" in r["label"] else 7
        ax.errorbar(r["d"], i, xerr=[[r["d"] - r["ci_low"]], [r["ci_high"] - r["d"]]],
                    fmt="o", color=color, markersize=ms, capsize=4, capthick=1.5, linewidth=1.5)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=0.8, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Large (0.8)")
    ax.axvline(x=0.5, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="Medium (0.5)")

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["label"] for r in results], fontsize=9)
    ax.set_xlabel("Cohen's d vs Exact (positive = better than Exact)")
    ax.set_title("Effect Size: All Planners vs Exact Baseline")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "forest_all_planners.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pairwise_matrix(df, output_dir):
    """Pairwise win-rate matrix between all planners."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]
    n = len(planners)

    win_matrix = np.full((n, n), np.nan)

    for i, p1 in enumerate(planners):
        for j, p2 in enumerate(planners):
            if i == j:
                continue
            d1 = df[df["planner"] == p1].set_index(["field", "trial"])["rmse"]
            d2 = df[df["planner"] == p2].set_index(["field", "trial"])["rmse"]
            common = d1.index.intersection(d2.index)
            if len(common) > 0:
                wins = (d1.loc[common] < d2.loc[common]).sum()
                win_matrix[i, j] = 100 * wins / len(common)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    labels = [PLANNER_LABELS[p] for p in planners]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Planner")
    ax.set_title("Win Rate (% trials with lower RMSE)")

    for i in range(n):
        for j in range(n):
            if not np.isnan(win_matrix[i, j]):
                ax.text(j, i, f"{win_matrix[i, j]:.0f}%",
                        ha="center", va="center", fontsize=10,
                        color="black" if 30 < win_matrix[i, j] < 70 else "white")

    plt.colorbar(im, ax=ax, label="Win Rate (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_wins.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(df, output_dir):
    """Combined 2x2 summary figure for all planners."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Overall RMSE boxplot
    ax = axes[0, 0]
    data_for_box = [df[df["planner"] == p]["rmse"].dropna().values for p in planners]
    bp = ax.boxplot(data_for_box, labels=[PLANNER_LABELS[p] for p in planners],
                    patch_artist=True, widths=0.6)
    for patch, planner in zip(bp['boxes'], planners):
        patch.set_facecolor(PLANNER_COLORS[planner])
        patch.set_alpha(0.7)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("A) RMSE Distribution by Planner")
    ax.tick_params(axis='x', rotation=15)

    # 2. RMSE by field (grouped bars)
    ax = axes[0, 1]
    x = np.arange(len(FIELDS))
    n_p = len(planners)
    width = 0.8 / n_p
    for i, planner in enumerate(planners):
        means = [df[(df["field"] == f) & (df["planner"] == planner)]["rmse"].mean() for f in FIELDS]
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, means, width, label=PLANNER_LABELS[planner],
               color=PLANNER_COLORS[planner], alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_LABELS[f] for f in FIELDS], fontsize=9)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("B) Mean RMSE by Field")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)

    # 3. Travel cost boxplot
    ax = axes[1, 0]
    data_for_box = [df[df["planner"] == p]["travel"].dropna().values for p in planners]
    bp = ax.boxplot(data_for_box, labels=[PLANNER_LABELS[p] for p in planners],
                    patch_artist=True, widths=0.6)
    for patch, planner in zip(bp['boxes'], planners):
        patch.set_facecolor(PLANNER_COLORS[planner])
        patch.set_alpha(0.7)
    ax.set_ylabel("Travel Cost (m)")
    ax.set_title("C) Travel Cost Distribution")
    ax.tick_params(axis='x', rotation=15)

    # 4. Sample efficiency
    ax = axes[1, 1]
    eff_means = [df[df["planner"] == p]["sample_efficiency"].mean() for p in planners]
    ax.bar(range(len(planners)), eff_means,
           color=[PLANNER_COLORS[p] for p in planners], alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels([PLANNER_LABELS[p] for p in planners], fontsize=9, rotation=15)
    ax.set_ylabel("Unique / Total Samples")
    ax.set_title("D) Sample Efficiency")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_all_planners.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---- Statistical Tests ----

def run_statistical_tests(df, output_dir):
    """Run pairwise statistical tests between all planners."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]

    test_rows = []
    for p1, p2 in combinations(planners, 2):
        d1 = df[df["planner"] == p1].set_index(["field", "trial"])["rmse"]
        d2 = df[df["planner"] == p2].set_index(["field", "trial"])["rmse"]
        common = d1.index.intersection(d2.index)
        if len(common) < 2:
            continue

        v1 = d1.loc[common].values
        v2 = d2.loc[common].values

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(v1, v2)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(v1, v2)

        # Cohen's d
        d, ci_low, ci_high = cohens_d_ci(v1, v2)

        # Win rate
        wins_p1 = (v1 < v2).sum()
        wins_p2 = (v2 < v1).sum()

        test_rows.append({
            "planner_1": p1,
            "planner_2": p2,
            "n_paired": len(common),
            "mean_1": np.mean(v1),
            "mean_2": np.mean(v2),
            "mean_diff": np.mean(v1 - v2),
            "pct_improvement": 100 * np.mean(v1 - v2) / np.mean(v1) if np.mean(v1) > 0 else 0,
            "cohens_d": d,
            "cohens_d_ci_low": ci_low,
            "cohens_d_ci_high": ci_high,
            "effect_size": effect_label(d),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_pval,
            "ttest_stat": t_stat,
            "ttest_p": t_pval,
            "wins_p1": wins_p1,
            "wins_p2": wins_p2,
        })

    test_df = pd.DataFrame(test_rows)

    # Apply Holm-Bonferroni correction for multiple comparisons
    if len(test_df) > 1 and HAS_MULTITEST:
        # Correct Wilcoxon p-values
        w_pvals = test_df["wilcoxon_p"].values
        valid_w = ~np.isnan(w_pvals)
        if valid_w.sum() > 1:
            _, corrected_w, _, _ = multipletests(w_pvals[valid_w], method='holm')
            test_df.loc[valid_w, "wilcoxon_p_corrected"] = corrected_w
        else:
            test_df["wilcoxon_p_corrected"] = test_df["wilcoxon_p"]

        # Correct t-test p-values
        _, corrected_t, _, _ = multipletests(test_df["ttest_p"].values, method='holm')
        test_df["ttest_p_corrected"] = corrected_t
    else:
        test_df["wilcoxon_p_corrected"] = test_df["wilcoxon_p"]
        test_df["ttest_p_corrected"] = test_df["ttest_p"]

    test_df.to_csv(output_dir / "statistical_tests.csv", index=False)
    return test_df


def print_results(df, test_df):
    """Print statistical summary."""
    available = df["planner"].unique()
    planners = [p for p in PLANNERS if p in available]

    print("=" * 80)
    print("5-PLANNER COMPARISON")
    print("=" * 80)

    # Overall stats per planner
    print(f"\n{'Planner':<20} {'n':>5} {'RMSE':>10} {'Travel':>10} {'Efficiency':>10} {'Unique%':>10}")
    print("-" * 80)
    for planner in planners:
        sub = df[df["planner"] == planner]
        n = len(sub)
        rmse_str = f"{sub['rmse'].mean():.3f}±{sub['rmse'].std():.3f}" if n > 0 else "N/A"
        travel_str = f"{sub['travel'].mean():.1f}" if n > 0 else "N/A"
        eff_str = f"{sub['efficiency'].mean():.4f}" if n > 0 and 'efficiency' in sub else "N/A"
        uniq_str = f"{100*sub['sample_efficiency'].mean():.1f}%" if n > 0 else "N/A"
        print(f"{PLANNER_LABELS[planner]:<20} {n:>5} {rmse_str:>10} {travel_str:>10} {eff_str:>10} {uniq_str:>10}")

    # Pairwise tests
    if len(test_df) > 0:
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS (Cohen's d: positive = planner_1 has HIGHER RMSE)")
        print("=" * 80)
        print(f"{'Pair':<35} {'n':>4} {'d':>8} {'Effect':>10} {'p(W)':>10} {'p(W)holm':>10} {'Wins':>10}")
        print("-" * 90)
        for _, row in test_df.iterrows():
            pair = f"{PLANNER_LABELS[row['planner_1']]} vs {PLANNER_LABELS[row['planner_2']]}"
            p_raw = f"{row['wilcoxon_p']:.4f}" if not np.isnan(row['wilcoxon_p']) else "N/A"
            p_corr = f"{row['wilcoxon_p_corrected']:.4f}" if not np.isnan(row.get('wilcoxon_p_corrected', np.nan)) else "N/A"
            wins = f"{int(row['wins_p1'])}/{int(row['wins_p2'])}"
            print(f"{pair:<35} {int(row['n_paired']):>4} {row['cohens_d']:>+8.3f} "
                  f"{row['effect_size']:>10} {p_raw:>10} {p_corr:>10} {wins:>10}")
        print("-" * 90)
        if HAS_MULTITEST:
            print("  p(W)holm = Holm-Bonferroni corrected Wilcoxon p-value")
        else:
            print("  [!] statsmodels not available — p-values are uncorrected")


def main():
    print("Loading trial data...")
    df = load_all_trials()

    if len(df) == 0:
        print("No trial data found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tests
    test_df = run_statistical_tests(df, OUTPUT_DIR)

    # Print results
    print_results(df, test_df)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_rmse_bars_all(df, OUTPUT_DIR)
    print("Saved: rmse_all_planners.png")

    plot_sample_efficiency(df, OUTPUT_DIR)
    print("Saved: sample_efficiency.png")

    plot_travel_comparison(df, OUTPUT_DIR)
    print("Saved: travel_comparison.png")

    plot_forest_all(df, OUTPUT_DIR)
    print("Saved: forest_all_planners.png")

    plot_pairwise_matrix(df, OUTPUT_DIR)
    print("Saved: pairwise_wins.png")

    plot_summary(df, OUTPUT_DIR)
    print("Saved: summary_all_planners.png")

    # Save data
    df.to_csv(OUTPUT_DIR / "all_planners_data.csv", index=False)
    print("Saved: all_planners_data.csv")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
