#!/usr/bin/env python3
"""
Compare ALL methods with heatmap visualizations.
Includes: Standard GP, McHutchon NIGP, Girard uncertain-input GP

Usage:
    python compare_all_methods.py [trial_number]

Examples:
    python compare_all_methods.py 1
    python compare_all_methods.py 2
    python compare_all_methods.py 3
"""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Parse command-line argument
trial = int(sys.argv[1]) if len(sys.argv) > 1 else 1

fields = ['radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt']
kernels = ['rbf', 'exponential', 'matern15', 'matern25']

print("\n" + "="*100)
print(f"RECONSTRUCTION COMPARISON (Trial {trial})")
print("="*100 + "\n")

# Collect all results
all_results = {}

for field in fields:
    all_results[field] = {}

    # Standard GP all kernels
    for kernel in kernels:
        std_path = Path(f'results/trial_{trial}/standard_gp/{field}/{kernel}/{field}_{kernel}_metrics.csv')
        if std_path.exists():
            with open(std_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results[field][f'Standard GP ({kernel})'] = {
                        'rmse': float(row['rmse']),
                        'nrmse': float(row['nrmse']),
                    }

    # Girard RBF-only
    girard_path = Path(f'results/trial_{trial}/girard/{field}/rbf/{field}_rbf_metrics.csv')
    if girard_path.exists():
        with open(girard_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results[field]['Girard (RBF)'] = {
                    'rmse': float(row['rmse']),
                    'nrmse': float(row['nrmse']),
                }

    # McHutchon all kernels
    for kernel in kernels:
        mchutchon_path = Path(f'results/trial_{trial}/mchutchon_nigp/{field}/{kernel}/{field}_{kernel}_metrics.csv')
        if mchutchon_path.exists():
            with open(mchutchon_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results[field][f'McHutchon ({kernel})'] = {
                        'rmse': float(row['rmse']),
                        'nrmse': float(row['nrmse']),
                    }

# Print text comparison
for field in fields:
    if field in all_results and all_results[field]:
        print(f"\n{'─'*120}")
        print(f"FIELD: {field.upper()}")
        print(f"{'─'*120}")
        print(f"{'Method':<40} {'RMSE':>20} {'NRMSE':>20}")
        print(f"{'-'*120}")

        # Sort by NRMSE (lower is better)
        sorted_methods = sorted(all_results[field].items(),
                               key=lambda x: x[1]['nrmse'])

        for method, metrics in sorted_methods:
            rmse = metrics['rmse']
            nrmse = metrics['nrmse']
            print(f"{method:<40} {rmse:>20.4f} {nrmse:>20.4f}")

print("\n" + "="*100 + "\n")

# Calculate stats
method_stats = {}
for field in fields:
    if field in all_results:
        for method, metrics in all_results[field].items():
            if method not in method_stats:
                method_stats[method] = {'rmse_values': [], 'nrmse_values': []}
            method_stats[method]['rmse_values'].append(metrics['rmse'])
            method_stats[method]['nrmse_values'].append(metrics['nrmse'])

# Create heatmap visualizations
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 1, figure=fig, hspace=0.35)

# Prepare data
heatmap_methods = sorted(method_stats.keys())
heatmap_rmse = []
heatmap_nrmse = []

for method in heatmap_methods:
    rmse_row = []
    nrmse_row = []
    for field in fields:
        if field in all_results and method in all_results[field]:
            rmse_row.append(all_results[field][method]['rmse'])
            nrmse_row.append(all_results[field][method]['nrmse'])
        else:
            rmse_row.append(0)
            nrmse_row.append(0)
    heatmap_rmse.append(rmse_row)
    heatmap_nrmse.append(nrmse_row)

# Plot 1: RMSE Heatmap
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(heatmap_rmse, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max([max(row) for row in heatmap_rmse if max(row) > 0]))
ax1.set_xticks(np.arange(len(fields)))
ax1.set_yticks(np.arange(len(heatmap_methods)))
ax1.set_xticklabels(fields, fontsize=12, fontweight='bold')
ax1.set_yticklabels(heatmap_methods, fontsize=11)
ax1.set_title('RMSE by Method & Field', fontsize=14, fontweight='bold', pad=15)

# Add values to RMSE heatmap
for i in range(len(heatmap_methods)):
    for j in range(len(fields)):
        if heatmap_rmse[i][j] > 0:
            text = ax1.text(j, i, f'{heatmap_rmse[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('RMSE', fontsize=11, fontweight='bold')

# Plot 2: NRMSE Heatmap
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(heatmap_nrmse, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=max([max(row) for row in heatmap_nrmse if max(row) > 0]))
ax2.set_xticks(np.arange(len(fields)))
ax2.set_yticks(np.arange(len(heatmap_methods)))
ax2.set_xticklabels(fields, fontsize=12, fontweight='bold')
ax2.set_yticklabels(heatmap_methods, fontsize=11)
ax2.set_title('NRMSE by Method & Field (Lower is Better)', fontsize=14, fontweight='bold', pad=15)

# Add values to NRMSE heatmap
for i in range(len(heatmap_methods)):
    for j in range(len(fields)):
        if heatmap_nrmse[i][j] > 0:
            text = ax2.text(j, i, f'{heatmap_nrmse[i][j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('NRMSE', fontsize=11, fontweight='bold')

# Overall title
fig.suptitle(f'Trial {trial} - All Methods Comparison', fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_dir = Path('results') / f'trial_{trial}' / 'comparison'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'comparison_heatmaps.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Comparison figure saved to: {output_path}\n")

# Save detailed metrics to CSV
metrics_path = output_dir / 'all_methods_metrics.csv'
with open(metrics_path, 'w') as f:
    f.write('Field,Method,RMSE,NRMSE\n')
    for field in fields:
        if field in all_results:
            for method in sorted(all_results[field].keys()):
                metrics = all_results[field][method]
                f.write(f'{field},{method},{metrics["rmse"]:.4f},{metrics["nrmse"]:.4f}\n')

print(f"✓ Detailed metrics saved to: {metrics_path}\n")

# Show the plot
plt.show()
