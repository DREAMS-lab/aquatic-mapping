#!/usr/bin/env python3
"""
Statistical Analysis: Exact vs Pose-Aware Planner Comparison

Computes comprehensive statistics across all trials for thesis.
Generates tables and figures suitable for publication.

Usage:
    cd /home/blazair/workspaces/aquatic-mapping/reconstruction
    source venv/bin/activate
    python3 statistical_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = SCRIPT_DIR.parent
TRIALS_DIR = WORKSPACE_ROOT / "data" / "trials"
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "reconstruction" / "statistical_analysis"

FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
PLANNERS = ["exact", "pose_aware"]


def load_all_trials() -> pd.DataFrame:
    """Load all trial data into a single DataFrame."""
    rows = []

    for field in FIELDS:
        for trial_num in range(1, 50):
            trial_str = f"trial_{trial_num:03d}"

            # Check both planners exist for this trial
            exact_dir = TRIALS_DIR / "exact" / field / trial_str
            pose_dir = TRIALS_DIR / "pose_aware" / field / trial_str

            for planner, trial_dir in [("exact", exact_dir), ("pose_aware", pose_dir)]:
                summary_file = trial_dir / "summary.json"
                samples_file = trial_dir / "samples.csv"

                if not summary_file.exists():
                    continue

                with open(summary_file) as f:
                    summary = json.load(f)

                row = {
                    "field": field,
                    "trial": trial_num,
                    "planner": planner,
                    "rmse": summary.get("reconstruction_rmse", np.nan),
                    "mae": summary.get("reconstruction_mae", np.nan),
                    "max_error": summary.get("reconstruction_max_error", np.nan),
                    "total_travel": summary.get("total_travel_cost", np.nan),
                    "total_info_gain": summary.get("cumulative_info_gain", np.nan),
                    "mean_gp_variance": summary.get("mean_gp_variance", np.nan),
                    "total_samples": summary.get("total_samples", 100),
                }

                # Compute info efficiency
                if row["total_travel"] > 0:
                    row["info_efficiency"] = row["total_info_gain"] / row["total_travel"]
                else:
                    row["info_efficiency"] = np.nan

                # Load samples.csv for additional metrics
                if samples_file.exists():
                    try:
                        samples_df = pd.read_csv(samples_file)
                        # Average position variance
                        if "pos_var_x" in samples_df.columns:
                            row["mean_pos_var_x"] = samples_df["pos_var_x"].mean()
                            row["mean_pos_var_y"] = samples_df["pos_var_y"].mean()
                            row["mean_pos_std"] = np.sqrt(
                                (samples_df["pos_var_x"] + samples_df["pos_var_y"]) / 2
                            ).mean()
                    except:
                        pass

                rows.append(row)

    return pd.DataFrame(rows)


def cohens_d_paired(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = x1 - x2
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "LARGE"


def paired_comparison(df: pd.DataFrame, metric: str, group_by: str = None) -> pd.DataFrame:
    """
    Perform paired comparison between exact and pose_aware planners.

    Returns DataFrame with statistics for each group.
    """
    results = []

    if group_by:
        groups = df[group_by].unique()
    else:
        groups = [None]

    for group in groups:
        if group is not None:
            subset = df[df[group_by] == group]
        else:
            subset = df

        # Get paired trials (both planners completed)
        exact_trials = subset[subset["planner"] == "exact"].set_index("trial")[metric]
        pose_trials = subset[subset["planner"] == "pose_aware"].set_index("trial")[metric]

        # Find common trials
        common_trials = exact_trials.index.intersection(pose_trials.index)

        if len(common_trials) < 2:
            continue

        exact = exact_trials.loc[common_trials].values
        pose = pose_trials.loc[common_trials].values

        # Statistics
        n = len(common_trials)
        exact_mean = np.mean(exact)
        exact_std = np.std(exact, ddof=1)
        pose_mean = np.mean(pose)
        pose_std = np.std(pose, ddof=1)

        # Difference
        diff_mean = exact_mean - pose_mean

        # Cohen's d
        d = cohens_d_paired(exact, pose)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(exact, pose)

        # Wilcoxon signed-rank (non-parametric alternative)
        try:
            w_stat, w_pvalue = stats.wilcoxon(exact, pose)
        except:
            w_stat, w_pvalue = np.nan, np.nan

        results.append({
            "group": group if group else "overall",
            "metric": metric,
            "n_pairs": n,
            "exact_mean": exact_mean,
            "exact_std": exact_std,
            "pose_mean": pose_mean,
            "pose_std": pose_std,
            "diff_mean": diff_mean,
            "cohens_d": d,
            "effect_size": interpret_cohens_d(d),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "wilcoxon_p": w_pvalue,
        })

    return pd.DataFrame(results)


def analyze_all_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze all metrics of interest."""
    metrics = {
        "rmse": "Reconstruction RMSE",
        "mae": "Reconstruction MAE",
        "max_error": "Max Error",
        "total_travel": "Total Travel Cost",
        "total_info_gain": "Total Information Gain",
        "info_efficiency": "Information Efficiency",
        "mean_gp_variance": "Mean GP Variance",
    }

    results = {}

    for metric, name in metrics.items():
        # Overall comparison
        overall = paired_comparison(df, metric)
        overall["analysis"] = "overall"

        # Per-field comparison
        per_field = paired_comparison(df, metric, group_by="field")
        per_field["analysis"] = "per_field"

        results[metric] = pd.concat([overall, per_field], ignore_index=True)

    return results


def print_summary_table(results: Dict[str, pd.DataFrame]):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("STATISTICAL ANALYSIS: Exact vs Pose-Aware Planner")
    print("=" * 90)

    # Overall summary for key metrics
    print("\n### OVERALL COMPARISON (All Fields Pooled)")
    print("-" * 90)
    print(f"{'Metric':<25} {'n':>4} {'Exact':>12} {'Pose':>12} {'Diff':>10} {'Cohen d':>10} {'p-value':>10}")
    print("-" * 90)

    for metric in ["rmse", "mae", "total_travel", "total_info_gain", "info_efficiency"]:
        if metric in results:
            row = results[metric][results[metric]["analysis"] == "overall"].iloc[0]
            sig = "*" if row["significant"] else ""
            print(f"{metric:<25} {row['n_pairs']:>4} {row['exact_mean']:>12.3f} {row['pose_mean']:>12.3f} "
                  f"{row['diff_mean']:>+10.3f} {row['cohens_d']:>+10.3f} {row['p_value']:>9.4f}{sig}")

    print("-" * 90)
    print("* p < 0.05")

    # Per-field RMSE breakdown
    print("\n### RMSE BY FIELD")
    print("-" * 90)
    print(f"{'Field':<20} {'n':>4} {'Exact':>12} {'Pose':>12} {'Cohen d':>10} {'Effect':>12} {'p-value':>10}")
    print("-" * 90)

    rmse_results = results["rmse"]
    per_field = rmse_results[rmse_results["analysis"] == "per_field"]

    for _, row in per_field.iterrows():
        sig = "*" if row["significant"] else ""
        print(f"{row['group']:<20} {row['n_pairs']:>4} {row['exact_mean']:>12.3f} {row['pose_mean']:>12.3f} "
              f"{row['cohens_d']:>+10.3f} {row['effect_size']:>12} {row['p_value']:>9.4f}{sig}")

    print("-" * 90)


def plot_comparison_boxplots(df: pd.DataFrame, output_dir: Path):
    """Generate boxplot comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    metrics = [
        ("rmse", "RMSE (°C)"),
        ("mae", "MAE (°C)"),
        ("total_travel", "Travel Cost (m)"),
        ("total_info_gain", "Info Gain"),
        ("info_efficiency", "Info Efficiency"),
        ("mean_gp_variance", "Mean GP Variance"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        # Prepare data for boxplot
        exact_data = df[df["planner"] == "exact"][metric].dropna()
        pose_data = df[df["planner"] == "pose_aware"][metric].dropna()

        bp = ax.boxplot([exact_data, pose_data], labels=["Exact", "Pose-Aware"])
        ax.set_ylabel(label)
        ax.set_title(label)

        # Add individual points
        for i, data in enumerate([exact_data, pose_data], 1):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=20)

    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'boxplot_comparison.png'}")


def plot_field_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate per-field comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE by field
    ax = axes[0]
    fields_order = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
    x = np.arange(len(fields_order))
    width = 0.35

    exact_means = []
    exact_stds = []
    pose_means = []
    pose_stds = []

    for field in fields_order:
        exact_vals = df[(df["field"] == field) & (df["planner"] == "exact")]["rmse"]
        pose_vals = df[(df["field"] == field) & (df["planner"] == "pose_aware")]["rmse"]
        exact_means.append(exact_vals.mean())
        exact_stds.append(exact_vals.std())
        pose_means.append(pose_vals.mean())
        pose_stds.append(pose_vals.std())

    ax.bar(x - width/2, exact_means, width, yerr=exact_stds, label='Exact', capsize=3, alpha=0.8)
    ax.bar(x + width/2, pose_means, width, yerr=pose_stds, label='Pose-Aware', capsize=3, alpha=0.8)
    ax.set_ylabel('RMSE (°C)')
    ax.set_xlabel('Field Type')
    ax.set_title('Reconstruction RMSE by Field')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fields_order], fontsize=8)
    ax.legend()

    # Cohen's d by field
    ax = axes[1]
    cohens_d_vals = []
    for field in fields_order:
        exact_vals = df[(df["field"] == field) & (df["planner"] == "exact")]["rmse"].values
        pose_vals = df[(df["field"] == field) & (df["planner"] == "pose_aware")]["rmse"].values

        # Match by trial
        exact_df = df[(df["field"] == field) & (df["planner"] == "exact")].set_index("trial")["rmse"]
        pose_df = df[(df["field"] == field) & (df["planner"] == "pose_aware")].set_index("trial")["rmse"]
        common = exact_df.index.intersection(pose_df.index)

        if len(common) >= 2:
            d = cohens_d_paired(exact_df.loc[common].values, pose_df.loc[common].values)
        else:
            d = 0
        cohens_d_vals.append(d)

    colors = ['green' if d > 0 else 'red' for d in cohens_d_vals]
    ax.bar(x, cohens_d_vals, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='gray', linestyle='--', label='Large effect (0.8)')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Medium effect (0.5)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel("Cohen's d")
    ax.set_xlabel('Field Type')
    ax.set_title("Effect Size by Field (d > 0 = Pose-Aware Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fields_order], fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "field_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'field_comparison.png'}")


def plot_paired_scatter(df: pd.DataFrame, output_dir: Path):
    """Scatter plot showing paired trial results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get paired data
    exact_df = df[df["planner"] == "exact"].set_index(["field", "trial"])
    pose_df = df[df["planner"] == "pose_aware"].set_index(["field", "trial"])

    common_idx = exact_df.index.intersection(pose_df.index)

    exact_rmse = exact_df.loc[common_idx, "rmse"].values
    pose_rmse = pose_df.loc[common_idx, "rmse"].values
    fields = [idx[0] for idx in common_idx]

    # Scatter plot
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(FIELDS)))
    field_colors = {f: c for f, c in zip(FIELDS, colors)}

    for field in FIELDS:
        mask = [f == field for f in fields]
        ax.scatter(
            np.array(exact_rmse)[mask],
            np.array(pose_rmse)[mask],
            c=[field_colors[field]],
            label=field,
            alpha=0.7,
            s=60
        )

    # Diagonal line (equal performance)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Exact Planner RMSE (°C)')
    ax.set_ylabel('Pose-Aware Planner RMSE (°C)')
    ax.set_title('Paired Trial Comparison\n(Below diagonal = Pose-Aware better)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # Difference histogram
    ax = axes[1]
    diff = exact_rmse - pose_rmse
    ax.hist(diff, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.axvline(x=np.mean(diff), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(diff):.3f}')
    ax.set_xlabel('RMSE Difference (Exact - Pose-Aware)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of RMSE Differences\n(Positive = Pose-Aware better)')
    ax.legend()

    # Count how many favor each
    n_pose_better = np.sum(diff > 0)
    n_exact_better = np.sum(diff < 0)
    ax.text(0.95, 0.95, f'Pose-Aware better: {n_pose_better}/{len(diff)}\nExact better: {n_exact_better}/{len(diff)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "paired_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'paired_scatter.png'}")


def generate_latex_table(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate LaTeX table for thesis."""

    # Overall comparison table
    latex = """\\begin{table}[h]
\\centering
\\caption{Statistical Comparison: Exact vs Pose-Aware Planner}
\\label{tab:planner_comparison}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Metric} & \\textbf{n} & \\textbf{Exact} & \\textbf{Pose-Aware} & \\textbf{Cohen's d} & \\textbf{p-value} \\\\
\\midrule
"""

    for metric in ["rmse", "mae", "total_travel", "total_info_gain"]:
        if metric in results:
            row = results[metric][results[metric]["analysis"] == "overall"].iloc[0]
            metric_name = {
                "rmse": "RMSE",
                "mae": "MAE",
                "total_travel": "Travel Cost",
                "total_info_gain": "Info Gain"
            }[metric]
            sig = "*" if row["significant"] else ""
            latex += f"{metric_name} & {row['n_pairs']:.0f} & {row['exact_mean']:.2f} & {row['pose_mean']:.2f} & {row['cohens_d']:+.2f} & {row['p_value']:.4f}{sig} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # Per-field RMSE table
    latex += """
\\begin{table}[h]
\\centering
\\caption{RMSE Comparison by Field Type}
\\label{tab:rmse_by_field}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Field} & \\textbf{n} & \\textbf{Exact} & \\textbf{Pose-Aware} & \\textbf{Cohen's d} & \\textbf{Effect} \\\\
\\midrule
"""

    rmse_results = results["rmse"]
    per_field = rmse_results[rmse_results["analysis"] == "per_field"]

    for _, row in per_field.iterrows():
        field_name = row['group'].replace("_", " ").title()
        latex += f"{field_name} & {row['n_pairs']:.0f} & {row['exact_mean']:.2f} & {row['pose_mean']:.2f} & {row['cohens_d']:+.2f} & {row['effect_size']} \\\\\n"

    # Add overall
    overall = rmse_results[rmse_results["analysis"] == "overall"].iloc[0]
    latex += f"\\midrule\n\\textbf{{Overall}} & {overall['n_pairs']:.0f} & {overall['exact_mean']:.2f} & {overall['pose_mean']:.2f} & {overall['cohens_d']:+.2f} & {overall['effect_size']} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(output_dir / "tables.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'tables.tex'}")


def main():
    """Run complete statistical analysis."""
    print("Loading trial data...")
    df = load_all_trials()

    print(f"Loaded {len(df)} trial records")
    print(f"  Exact trials: {len(df[df['planner'] == 'exact'])}")
    print(f"  Pose-aware trials: {len(df[df['planner'] == 'pose_aware'])}")
    print(f"  Fields: {df['field'].unique().tolist()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze all metrics
    print("\nAnalyzing metrics...")
    results = analyze_all_metrics(df)

    # Print summary
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison_boxplots(df, OUTPUT_DIR)
    plot_field_comparison(df, OUTPUT_DIR)
    plot_paired_scatter(df, OUTPUT_DIR)

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table(results, OUTPUT_DIR)

    # Save raw results to CSV
    all_results = pd.concat(results.values(), ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "statistical_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'statistical_results.csv'}")

    # Save raw data
    df.to_csv(OUTPUT_DIR / "all_trials_data.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'all_trials_data.csv'}")

    print("\n" + "=" * 90)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
