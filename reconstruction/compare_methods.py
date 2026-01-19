#!/usr/bin/env python3
"""Compare Girard and McHutchon RBF results"""
import csv
from pathlib import Path

trial = 1
field = 'radial'

results = {}

# Read Girard RBF results
girard_path = Path(f'results/trial_{trial}/girard/{field}/rbf/{field}_rbf_metrics.csv')
if girard_path.exists():
    with open(girard_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results['Girard (RBF-only, analytic)'] = {
                'RMSE': float(row['rmse']),
                'NRMSE': float(row['nrmse']),
            }

# Read McHutchon RBF results
mchutchon_rbf_path = Path(f'results/trial_{trial}/mchutchon_nigp/{field}/rbf/{field}_rbf_metrics.csv')
if mchutchon_rbf_path.exists():
    with open(mchutchon_rbf_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results['McHutchon (RBF)'] = {
                'RMSE': float(row['rmse']),
                'NRMSE': float(row['nrmse']),
            }

# Read McHutchon Matern25 results (best)
mchutchon_m25_path = Path(f'results/trial_{trial}/mchutchon_nigp/{field}/matern25/{field}_matern25_metrics.csv')
if mchutchon_m25_path.exists():
    with open(mchutchon_m25_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results['McHutchon (Matern2.5)'] = {
                'RMSE': float(row['rmse']),
                'NRMSE': float(row['nrmse']),
            }

# Print comparison
print("\n" + "="*70)
print(f"RECONSTRUCTION COMPARISON: {field} (trial {trial})")
print("="*70 + "\n")
print(f"{'Method':<40} {'RMSE':>12} {'NRMSE':>12}")
print("-"*70)
for method, metrics in sorted(results.items()):
    print(f"{method:<40} {metrics['RMSE']:>12.4f} {metrics['NRMSE']:>12.4f}")
print()
