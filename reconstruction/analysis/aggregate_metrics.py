#!/usr/bin/env python3
"""
Aggregate Metrics Analysis Across All Trials

Generates:
1. Mean posterior variance as aggregated metric
2. Boxplots across trials (RMSE, MAE, NRMSE, Mean Variance)
3. Variance vs error comparison plots
4. Summary statistics CSV

Auto-detects all existing trials.

Usage:
    python aggregate_metrics.py
"""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIELDS = ['radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt']
KERNELS = ['rbf', 'exponential', 'matern15', 'matern25']
METHODS = ['standard_gp', 'girard', 'mchutchon_nigp']

# Girard only supports RBF
METHOD_KERNELS = {
    'standard_gp': KERNELS,
    'girard': ['rbf'],
    'mchutchon_nigp': KERNELS,
}


def discover_trials():
    """Auto-detect all trial directories."""
    trials = []
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith('trial_'):
            try:
                trial_num = int(d.name.split('_')[1])
                trials.append(trial_num)
            except ValueError:
                continue
    return sorted(trials)


def load_metrics(trial, method, field, kernel):
    """Load metrics CSV for a specific configuration."""
    metrics_path = RESULTS_DIR / f'trial_{trial}' / method / field / kernel / f'{field}_{kernel}_metrics.csv'
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {
                'mse': float(row['mse']),
                'rmse': float(row['rmse']),
                'mae': float(row['mae']),
                'nrmse': float(row['nrmse']),
            }
    return None


def load_variance(trial, method, field, kernel):
    """Load variance npy file and compute mean posterior variance."""
    var_path = RESULTS_DIR / f'trial_{trial}' / method / field / kernel / f'{field}_{kernel}_variance.npy'
    if not var_path.exists():
        return None, None

    variance = np.load(var_path)
    mean_var = float(np.mean(variance))
    max_var = float(np.max(variance))
    return mean_var, max_var


def load_predictions_and_ground_truth(trial, method, field, kernel):
    """Load predictions for variance vs error analysis."""
    pred_path = RESULTS_DIR / f'trial_{trial}' / method / field / kernel / f'{field}_{kernel}_predictions.npy'
    var_path = RESULTS_DIR / f'trial_{trial}' / method / field / kernel / f'{field}_{kernel}_variance.npy'

    if not pred_path.exists() or not var_path.exists():
        return None, None

    predictions = np.load(pred_path)
    variance = np.load(var_path)
    return predictions, variance


def collect_all_data(trials):
    """Collect all metrics across trials, methods, fields, kernels."""
    data = []

    for trial in trials:
        for method in METHODS:
            for field in FIELDS:
                for kernel in METHOD_KERNELS[method]:
                    metrics = load_metrics(trial, method, field, kernel)
                    if metrics is None:
                        continue

                    mean_var, max_var = load_variance(trial, method, field, kernel)

                    data.append({
                        'trial': trial,
                        'method': method,
                        'field': field,
                        'kernel': kernel,
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'nrmse': metrics['nrmse'],
                        'mse': metrics['mse'],
                        'mean_variance': mean_var,
                        'max_variance': max_var,
                    })

    return data


def create_boxplots(data, output_dir):
    """Create boxplots across trials for each metric."""
    print("\nGenerating boxplots...")

    # Group by method+kernel
    method_kernel_data = defaultdict(lambda: {'rmse': [], 'mae': [], 'nrmse': [], 'mean_variance': []})

    for row in data:
        key = f"{row['method']} ({row['kernel']})"
        method_kernel_data[key]['rmse'].append(row['rmse'])
        method_kernel_data[key]['mae'].append(row['mae'])
        method_kernel_data[key]['nrmse'].append(row['nrmse'])
        if row['mean_variance'] is not None:
            method_kernel_data[key]['mean_variance'].append(row['mean_variance'])

    # Sort methods for consistent ordering
    methods_sorted = sorted(method_kernel_data.keys())

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ('rmse', 'RMSE [°C]', axes[0, 0]),
        ('mae', 'MAE [°C]', axes[0, 1]),
        ('nrmse', 'NRMSE', axes[1, 0]),
        ('mean_variance', 'Mean Posterior Variance [°C²]', axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics_to_plot:
        box_data = [method_kernel_data[m][metric] for m in methods_sorted]

        # Filter out empty lists
        valid_data = []
        valid_labels = []
        for m, d in zip(methods_sorted, box_data):
            if len(d) > 0:
                valid_data.append(d)
                valid_labels.append(m.replace('standard_gp', 'Std GP').replace('mchutchon_nigp', 'NIGP').replace('girard', 'Girard'))

        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        bp = ax.boxplot(valid_data, patch_artist=True)

        # Color by method type
        colors = []
        for label in valid_labels:
            if 'Std GP' in label:
                colors.append('#3498db')  # Blue
            elif 'Girard' in label:
                colors.append('#2ecc71')  # Green
            elif 'NIGP' in label:
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#95a5a6')  # Gray

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.upper()} Across All Trials', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'boxplots_all_trials.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_boxplots_by_field(data, output_dir):
    """Create boxplots grouped by field type."""
    print("\nGenerating boxplots by field...")

    for field in FIELDS:
        field_data = [d for d in data if d['field'] == field]
        if not field_data:
            continue

        # Group by method+kernel
        method_kernel_data = defaultdict(lambda: {'rmse': [], 'mae': [], 'nrmse': [], 'mean_variance': []})

        for row in field_data:
            key = f"{row['method']} ({row['kernel']})"
            method_kernel_data[key]['rmse'].append(row['rmse'])
            method_kernel_data[key]['mae'].append(row['mae'])
            method_kernel_data[key]['nrmse'].append(row['nrmse'])
            if row['mean_variance'] is not None:
                method_kernel_data[key]['mean_variance'].append(row['mean_variance'])

        methods_sorted = sorted(method_kernel_data.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = [
            ('rmse', 'RMSE [°C]', axes[0, 0]),
            ('mae', 'MAE [°C]', axes[0, 1]),
            ('nrmse', 'NRMSE', axes[1, 0]),
            ('mean_variance', 'Mean Posterior Variance [°C²]', axes[1, 1]),
        ]

        for metric, ylabel, ax in metrics_to_plot:
            box_data = [method_kernel_data[m][metric] for m in methods_sorted]

            valid_data = []
            valid_labels = []
            for m, d in zip(methods_sorted, box_data):
                if len(d) > 0:
                    valid_data.append(d)
                    valid_labels.append(m.replace('standard_gp', 'Std').replace('mchutchon_nigp', 'NIGP').replace('girard', 'Gir'))

            if len(valid_data) == 0:
                continue

            bp = ax.boxplot(valid_data, patch_artist=True)

            colors = []
            for label in valid_labels:
                if 'Std' in label:
                    colors.append('#3498db')
                elif 'Gir' in label:
                    colors.append('#2ecc71')
                elif 'NIGP' in label:
                    colors.append('#e74c3c')
                else:
                    colors.append('#95a5a6')

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Field: {field}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f'boxplots_{field}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def create_variance_vs_error_plots(data, output_dir):
    """Create variance vs error scatter plots."""
    print("\nGenerating variance vs error plots...")

    # Group by method
    method_data = defaultdict(lambda: {'mean_variance': [], 'rmse': [], 'mae': []})

    for row in data:
        if row['mean_variance'] is None:
            continue
        method = row['method']
        method_data[method]['mean_variance'].append(row['mean_variance'])
        method_data[method]['rmse'].append(row['rmse'])
        method_data[method]['mae'].append(row['mae'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'standard_gp': '#3498db', 'girard': '#2ecc71', 'mchutchon_nigp': '#e74c3c'}
    labels = {'standard_gp': 'Standard GP', 'girard': 'Girard', 'mchutchon_nigp': 'McHutchon NIGP'}

    # Variance vs RMSE
    ax = axes[0]
    for method in METHODS:
        if method not in method_data or len(method_data[method]['mean_variance']) == 0:
            continue
        ax.scatter(
            method_data[method]['mean_variance'],
            method_data[method]['rmse'],
            c=colors[method],
            label=labels[method],
            alpha=0.6,
            s=50
        )
    ax.set_xlabel('Mean Posterior Variance [°C²]', fontsize=11)
    ax.set_ylabel('RMSE [°C]', fontsize=11)
    ax.set_title('Variance vs RMSE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance vs MAE
    ax = axes[1]
    for method in METHODS:
        if method not in method_data or len(method_data[method]['mean_variance']) == 0:
            continue
        ax.scatter(
            method_data[method]['mean_variance'],
            method_data[method]['mae'],
            c=colors[method],
            label=labels[method],
            alpha=0.6,
            s=50
        )
    ax.set_xlabel('Mean Posterior Variance [°C²]', fontsize=11)
    ax.set_ylabel('MAE [°C]', fontsize=11)
    ax.set_title('Variance vs MAE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'variance_vs_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_variance_vs_error_by_field(data, output_dir):
    """Create variance vs error scatter plots per field."""
    print("\nGenerating variance vs error plots by field...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = {'standard_gp': '#3498db', 'girard': '#2ecc71', 'mchutchon_nigp': '#e74c3c'}
    labels = {'standard_gp': 'Standard GP', 'girard': 'Girard', 'mchutchon_nigp': 'McHutchon NIGP'}

    for idx, field in enumerate(FIELDS):
        ax = axes[idx]
        field_data = [d for d in data if d['field'] == field and d['mean_variance'] is not None]

        for method in METHODS:
            method_field_data = [d for d in field_data if d['method'] == method]
            if not method_field_data:
                continue

            variances = [d['mean_variance'] for d in method_field_data]
            rmses = [d['rmse'] for d in method_field_data]

            ax.scatter(variances, rmses, c=colors[method], label=labels[method], alpha=0.7, s=40)

        ax.set_xlabel('Mean Variance [°C²]', fontsize=9)
        ax.set_ylabel('RMSE [°C]', fontsize=9)
        ax.set_title(field, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide last subplot if odd number
    if len(FIELDS) < 6:
        axes[5].axis('off')

    fig.suptitle('Variance vs RMSE by Field', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'variance_vs_error_by_field.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_summary_statistics(data, output_dir):
    """Create summary statistics CSV with mean variance included."""
    print("\nGenerating summary statistics...")

    # Group by method+kernel
    method_kernel_stats = defaultdict(lambda: {
        'rmse': [], 'mae': [], 'nrmse': [], 'mean_variance': []
    })

    for row in data:
        key = (row['method'], row['kernel'])
        method_kernel_stats[key]['rmse'].append(row['rmse'])
        method_kernel_stats[key]['mae'].append(row['mae'])
        method_kernel_stats[key]['nrmse'].append(row['nrmse'])
        if row['mean_variance'] is not None:
            method_kernel_stats[key]['mean_variance'].append(row['mean_variance'])

    # Write summary CSV
    summary_path = output_dir / 'summary_statistics.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'method', 'kernel', 'n_samples',
            'rmse_mean', 'rmse_std', 'rmse_min', 'rmse_max',
            'mae_mean', 'mae_std', 'mae_min', 'mae_max',
            'nrmse_mean', 'nrmse_std', 'nrmse_min', 'nrmse_max',
            'mean_var_mean', 'mean_var_std', 'mean_var_min', 'mean_var_max',
        ])

        for (method, kernel), stats in sorted(method_kernel_stats.items()):
            n = len(stats['rmse'])
            row = [method, kernel, n]

            for metric in ['rmse', 'mae', 'nrmse', 'mean_variance']:
                vals = stats[metric]
                if len(vals) > 0:
                    row.extend([
                        np.mean(vals),
                        np.std(vals),
                        np.min(vals),
                        np.max(vals),
                    ])
                else:
                    row.extend([None, None, None, None])

            writer.writerow(row)

    print(f"  Saved: {summary_path}")

    # Write detailed CSV with all data points
    detailed_path = output_dir / 'all_metrics_with_variance.csv'
    with open(detailed_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial', 'method', 'field', 'kernel',
            'rmse', 'mae', 'nrmse', 'mse', 'mean_variance', 'max_variance'
        ])

        for row in sorted(data, key=lambda x: (x['trial'], x['method'], x['field'], x['kernel'])):
            writer.writerow([
                row['trial'], row['method'], row['field'], row['kernel'],
                row['rmse'], row['mae'], row['nrmse'], row['mse'],
                row['mean_variance'], row['max_variance']
            ])

    print(f"  Saved: {detailed_path}")


def create_method_comparison_table(data, output_dir):
    """Create a clean comparison table image."""
    print("\nGenerating method comparison table...")

    # Aggregate by method (across all kernels, fields, trials)
    method_stats = defaultdict(lambda: {'rmse': [], 'mae': [], 'nrmse': [], 'mean_variance': []})

    for row in data:
        method = row['method']
        method_stats[method]['rmse'].append(row['rmse'])
        method_stats[method]['mae'].append(row['mae'])
        method_stats[method]['nrmse'].append(row['nrmse'])
        if row['mean_variance'] is not None:
            method_stats[method]['mean_variance'].append(row['mean_variance'])

    # Create table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    methods_order = ['standard_gp', 'girard', 'mchutchon_nigp']
    method_names = {'standard_gp': 'Standard GP', 'girard': 'Girard', 'mchutchon_nigp': 'McHutchon NIGP'}

    table_data = []
    for method in methods_order:
        if method not in method_stats:
            continue
        stats = method_stats[method]
        row = [
            method_names[method],
            f"{np.mean(stats['rmse']):.3f} ± {np.std(stats['rmse']):.3f}",
            f"{np.mean(stats['mae']):.3f} ± {np.std(stats['mae']):.3f}",
            f"{np.mean(stats['nrmse']):.4f} ± {np.std(stats['nrmse']):.4f}",
            f"{np.mean(stats['mean_variance']):.3f} ± {np.std(stats['mean_variance']):.3f}" if stats['mean_variance'] else 'N/A',
        ]
        table_data.append(row)

    columns = ['Method', 'RMSE [°C]', 'MAE [°C]', 'NRMSE', 'Mean Variance [°C²]']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#ecf0f1')
            else:
                table[(i + 1, j)].set_facecolor('#ffffff')

    plt.title('Method Comparison (Mean ± Std across all trials, fields, kernels)',
              fontsize=14, fontweight='bold', pad=20)

    output_path = output_dir / 'method_comparison_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    print("="*80)
    print("AGGREGATE METRICS ANALYSIS")
    print("="*80)

    # Discover trials
    trials = discover_trials()
    print(f"\nFound {len(trials)} trials: {trials}")

    if not trials:
        print("ERROR: No trial directories found!")
        return

    # Create output directory
    output_dir = RESULTS_DIR / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Collect all data
    print("\nCollecting data from all trials...")
    data = collect_all_data(trials)
    print(f"Collected {len(data)} data points")

    if not data:
        print("ERROR: No data found!")
        return

    # Generate all outputs
    create_boxplots(data, output_dir)
    create_boxplots_by_field(data, output_dir)
    create_variance_vs_error_plots(data, output_dir)
    create_variance_vs_error_by_field(data, output_dir)
    create_summary_statistics(data, output_dir)
    create_method_comparison_table(data, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        if f.is_file():
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
