#!/usr/bin/env python3
"""
Informative Path Planning - Analysis and Graph Generation

Automatically finds all existing trials and generates graphs.

Usage:
    python generate_graphs.py              # Auto-detect and generate all
    python generate_graphs.py --list       # Just list available trials

Output: reconstruction/analysis/results/{field_type}/{planner}/trial_NNN/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PLANNER_NAMES = {
    'exact': 'Exact (No Uncertainty)',
    'pose_aware': 'Planner-Aware (Expected Info Gain)',
    'model_aware': 'Model-Aware (NIGP)',
    'both_aware': 'Both-Aware (NIGP + Expected Info)',
}

PLANNER_COLORS = {
    'exact': '#1f77b4',
    'pose_aware': '#ff7f0e',
    'model_aware': '#2ca02c',
    'both_aware': '#d62728',
}

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = WORKSPACE_ROOT / 'src' / 'info_gain' / 'data' / 'trials'
OUTPUT_ROOT = Path(__file__).parent / 'results'


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def find_all_trials():
    """Find all existing trials in the data directory."""
    trials = []

    if not DATA_ROOT.exists():
        print(f"Data directory not found: {DATA_ROOT}")
        return trials

    for planner_dir in DATA_ROOT.iterdir():
        if not planner_dir.is_dir():
            continue
        planner = planner_dir.name

        for field_dir in planner_dir.iterdir():
            if not field_dir.is_dir():
                continue
            field_type = field_dir.name

            for trial_dir in field_dir.iterdir():
                if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
                    continue

                # Check if it has required files
                if (trial_dir / 'samples.csv').exists():
                    trial_num = int(trial_dir.name.split('_')[1])
                    trials.append({
                        'planner': planner,
                        'field_type': field_type,
                        'trial': trial_num,
                        'path': trial_dir
                    })

    return sorted(trials, key=lambda x: (x['field_type'], x['planner'], x['trial']))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trial_data(trial_info: Dict) -> Dict:
    """Load all data from a trial."""
    trial_path = trial_info['path']

    data = {
        'path': trial_path,
        'planner': trial_info['planner'],
        'field_type': trial_info['field_type'],
        'trial': trial_info['trial']
    }

    # Load config
    config_file = trial_path / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            data['config'] = json.load(f)

    # Load samples CSV
    samples_file = trial_path / 'samples.csv'
    if samples_file.exists():
        data['samples'] = pd.read_csv(samples_file)

    # Load summary
    summary_file = trial_path / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            data['summary'] = json.load(f)

    # Load ground truth
    gt_file = trial_path / 'ground_truth.npz'
    if gt_file.exists():
        gt = np.load(gt_file)
        data['ground_truth'] = {'X': gt['X'], 'Y': gt['Y'], 'field': gt['field']}

    # Load GP reconstruction
    gp_file = trial_path / 'gp_reconstruction.npz'
    if gp_file.exists():
        gp = np.load(gp_file)
        data['gp_reconstruction'] = {'mean': gp['mean'], 'variance': gp['variance'], 'error': gp['error']}

    # Load metrics
    metrics_file = trial_path / 'reconstruction_metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            data['metrics'] = json.load(f)

    return data


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def plot_cumulative_info_gain(data: Dict, output_dir: Path):
    """Cumulative information gain over samples."""
    samples = data.get('samples')
    if samples is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    planner = data['planner']
    color = PLANNER_COLORS.get(planner, 'blue')
    label = PLANNER_NAMES.get(planner, planner)

    steps = samples['step'].values
    cum_info = samples['cumulative_info'].values

    ax.plot(steps, cum_info, '-o', color=color, linewidth=2, markersize=4)
    ax.fill_between(steps, cum_info, alpha=0.2, color=color)

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Cumulative Information Gain (nats)', fontsize=12)
    ax.set_title(f'Information Accumulation - {label}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_info_gain.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_travel_cost(data: Dict, output_dir: Path):
    """Cumulative travel cost."""
    samples = data.get('samples')
    if samples is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    planner = data['planner']
    color = PLANNER_COLORS.get(planner, 'blue')
    label = PLANNER_NAMES.get(planner, planner)

    steps = samples['step'].values
    travel = samples['travel_cost'].values

    ax.plot(steps, travel, '-o', color=color, linewidth=2, markersize=4)
    ax.fill_between(steps, travel, alpha=0.2, color=color)

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Cumulative Travel Cost (m)', fontsize=12)
    ax.set_title(f'Travel Cost - {label}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'travel_cost.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_position_uncertainty(data: Dict, output_dir: Path):
    """Position uncertainty from PX4 EKF over time."""
    samples = data.get('samples')
    if samples is None or 'pos_std_x' not in samples.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    planner = data['planner']
    label = PLANNER_NAMES.get(planner, planner)

    steps = samples['step'].values
    std_x = samples['pos_std_x'].values
    std_y = samples['pos_std_y'].values

    ax.plot(steps, std_x, '-o', color='blue', label='σ_x (East)', linewidth=2, markersize=4)
    ax.plot(steps, std_y, '-s', color='red', label='σ_y (North)', linewidth=2, markersize=4)

    std_combined = np.sqrt(std_x**2 + std_y**2)
    ax.plot(steps, std_combined, '--', color='purple', label='||σ|| (combined)', linewidth=2)

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Position Std Deviation (m)', fontsize=12)
    ax.set_title(f'PX4 EKF Position Uncertainty - {label}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if 'pose_aware' in planner:
        ax.annotate('USED for expected info gain', xy=(0.02, 0.98),
                   xycoords='axes fraction', fontsize=10, color='green', va='top')
    else:
        ax.annotate('Logged but NOT used', xy=(0.02, 0.98),
                   xycoords='axes fraction', fontsize=10, color='gray', va='top')

    plt.tight_layout()
    plt.savefig(output_dir / 'position_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory(data: Dict, output_dir: Path):
    """Sampling trajectory on field."""
    samples = data.get('samples')
    gt = data.get('ground_truth')
    if samples is None:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    planner = data['planner']
    label = PLANNER_NAMES.get(planner, planner)

    if gt is not None:
        im = ax.contourf(gt['X'], gt['Y'], gt['field'], levels=20, cmap='RdYlBu_r', alpha=0.6)
        plt.colorbar(im, ax=ax, label='Temperature (°C)')

    initial = samples[samples['phase'] == 'initial']
    adaptive = samples[samples['phase'] == 'adaptive']

    all_x = samples['x'].values
    all_y = samples['y'].values
    ax.plot(all_x, all_y, 'k-', alpha=0.3, linewidth=1)

    if len(initial) > 0:
        ax.scatter(initial['x'], initial['y'], c='blue', s=80, marker='o',
                  edgecolors='black', label=f'Initial ({len(initial)})', zorder=5)
    if len(adaptive) > 0:
        ax.scatter(adaptive['x'], adaptive['y'], c='red', s=80, marker='s',
                  edgecolors='black', label=f'Adaptive ({len(adaptive)})', zorder=5)

    ax.scatter(all_x[0], all_y[0], c='green', s=150, marker='^',
              edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter(all_x[-1], all_y[-1], c='purple', s=150, marker='v',
              edgecolors='black', linewidth=2, label='End', zorder=10)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Sampling Trajectory - {label}', fontsize=14)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction(data: Dict, output_dir: Path):
    """4-panel reconstruction comparison."""
    gt = data.get('ground_truth')
    gp = data.get('gp_reconstruction')
    metrics = data.get('metrics', {})

    if gt is None or gp is None:
        return

    planner = data['planner']
    label = PLANNER_NAMES.get(planner, planner)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    X, Y = gt['X'], gt['Y']

    im1 = axes[0, 0].contourf(X, Y, gt['field'], levels=20, cmap='RdYlBu_r')
    axes[0, 0].set_title('Ground Truth', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], label='Temp (°C)')

    im2 = axes[0, 1].contourf(X, Y, gp['mean'], levels=20, cmap='RdYlBu_r')
    axes[0, 1].set_title('GP Reconstruction', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1], label='Temp (°C)')

    im3 = axes[1, 0].contourf(X, Y, np.abs(gp['error']), levels=20, cmap='Reds')
    axes[1, 0].set_title('Absolute Error', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], label='Error (°C)')
    if 'rmse' in metrics:
        axes[1, 0].text(0.05, 0.95, f"RMSE: {metrics['rmse']:.3f}°C",
                       transform=axes[1, 0].transAxes, fontsize=12, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    im4 = axes[1, 1].contourf(X, Y, gp['variance'], levels=20, cmap='viridis')
    axes[1, 1].set_title('GP Variance', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1], label='Variance')

    for ax in axes.flat:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

    fig.suptitle(f'Reconstruction - {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_info_per_step(data: Dict, output_dir: Path):
    """Per-step information gain."""
    samples = data.get('samples')
    if samples is None or 'info_gain' not in samples.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    planner = data['planner']
    color = PLANNER_COLORS.get(planner, 'blue')
    label = PLANNER_NAMES.get(planner, planner)

    steps = samples['step'].values
    info = samples['info_gain'].values

    ax.bar(steps, info, color=color, alpha=0.7)

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Information Gain (nats)', fontsize=12)
    ax.set_title(f'Per-Step Info Gain - {label}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'info_per_step.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_csv(data: Dict, output_dir: Path):
    """Generate summary CSV."""
    config = data.get('config', {})
    metrics = data.get('metrics', {})
    samples = data.get('samples')

    summary = {
        'planner': data['planner'],
        'field_type': data['field_type'],
        'trial': data['trial'],
        'method': config.get('method', 'N/A'),
        'n_samples': len(samples) if samples is not None else 0,
        'horizon': config.get('horizon', 'N/A'),
        'lambda_cost': config.get('lambda_cost', 'N/A'),
        'rmse': metrics.get('rmse', 'N/A'),
        'mae': metrics.get('mae', 'N/A'),
        'max_error': metrics.get('max_error', 'N/A'),
        'total_travel': samples['travel_cost'].iloc[-1] if samples is not None else 'N/A',
        'total_info': samples['cumulative_info'].iloc[-1] if samples is not None else 'N/A',
    }

    if samples is not None and 'pos_std_x' in samples.columns:
        summary['mean_pos_std'] = np.sqrt(samples['pos_std_x'].mean()**2 + samples['pos_std_y'].mean()**2)

    df = pd.DataFrame([summary])
    df.to_csv(output_dir / 'summary.csv', index=False)


# ============================================================================
# COMPARISON GRAPHS
# ============================================================================

def plot_comparison_bar(all_data: List[Dict], output_dir: Path):
    """RMSE comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    planners = []
    rmses = []
    colors = []

    for data in all_data:
        planner = data['planner']
        metrics = data.get('metrics', {})
        if 'rmse' in metrics:
            planners.append(PLANNER_NAMES.get(planner, planner))
            rmses.append(metrics['rmse'])
            colors.append(PLANNER_COLORS.get(planner, 'gray'))

    if not planners:
        return

    bars = ax.bar(planners, rmses, color=colors, edgecolor='black', linewidth=1.5)

    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{rmse:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_curves(all_data: List[Dict], output_dir: Path):
    """Overlay info gain and travel cost curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for data in all_data:
        planner = data['planner']
        samples = data.get('samples')
        if samples is None:
            continue

        color = PLANNER_COLORS.get(planner, 'gray')
        label = PLANNER_NAMES.get(planner, planner)

        steps = samples['step'].values

        axes[0].plot(steps, samples['cumulative_info'].values, '-o', color=color,
                    label=label, linewidth=2, markersize=3)
        axes[1].plot(steps, samples['travel_cost'].values, '-o', color=color,
                    label=label, linewidth=2, markersize=3)

    axes[0].set_xlabel('Sample Number')
    axes[0].set_ylabel('Cumulative Info Gain (nats)')
    axes[0].set_title('Information Accumulation')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Sample Number')
    axes[1].set_ylabel('Cumulative Travel (m)')
    axes[1].set_title('Travel Cost')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_trajectories(all_data: List[Dict], output_dir: Path):
    """Side-by-side trajectories."""
    n = len(all_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, all_data):
        planner = data['planner']
        samples = data.get('samples')
        gt = data.get('ground_truth')
        label = PLANNER_NAMES.get(planner, planner)

        if gt is not None:
            ax.contourf(gt['X'], gt['Y'], gt['field'], levels=20, cmap='RdYlBu_r', alpha=0.6)

        if samples is not None:
            x, y = samples['x'].values, samples['y'].values
            ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)
            ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=30, edgecolors='black')

        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        ax.set_aspect('equal')

    fig.suptitle('Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_csv(all_data: List[Dict], output_dir: Path):
    """Comparison summary table."""
    rows = []

    for data in all_data:
        config = data.get('config', {})
        metrics = data.get('metrics', {})
        samples = data.get('samples')

        rows.append({
            'Planner': PLANNER_NAMES.get(data['planner'], data['planner']),
            'Field': data['field_type'],
            'Trial': data['trial'],
            'Samples': len(samples) if samples is not None else 'N/A',
            'RMSE': f"{metrics['rmse']:.4f}" if 'rmse' in metrics else 'N/A',
            'MAE': f"{metrics['mae']:.4f}" if 'mae' in metrics else 'N/A',
            'Travel': f"{samples['travel_cost'].iloc[-1]:.2f}" if samples is not None else 'N/A',
            'Info': f"{samples['cumulative_info'].iloc[-1]:.2f}" if samples is not None else 'N/A',
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'comparison_summary.csv', index=False)

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def generate_all_graphs():
    """Auto-detect trials and generate all graphs."""

    print("="*60)
    print("INFORMATIVE PATH PLANNING - GRAPH GENERATION")
    print("="*60)

    # Find all trials
    trials = find_all_trials()

    if not trials:
        print("\nNo trials found!")
        print(f"Looking in: {DATA_ROOT}")
        print("\nRun simulations first:")
        print("  ./container/info_gain/start_exact_sim.sh radial 1")
        print("  ./container/info_gain/start_pose_aware_sim.sh radial 1")
        return

    print(f"\nFound {len(trials)} trial(s):\n")
    for t in trials:
        print(f"  - {t['planner']}/{t['field_type']}/trial_{t['trial']:03d}")

    # Group by field_type and trial for comparison
    groups = {}
    for t in trials:
        key = (t['field_type'], t['trial'])
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    # Generate per-planner graphs
    print("\n" + "-"*60)
    print("Generating per-planner graphs...")
    print("-"*60)

    all_loaded_data = {}

    for trial_info in trials:
        planner = trial_info['planner']
        field = trial_info['field_type']
        trial = trial_info['trial']

        print(f"\n[{planner}/{field}/trial_{trial:03d}]")

        data = load_trial_data(trial_info)
        all_loaded_data[(planner, field, trial)] = data

        output_dir = OUTPUT_ROOT / field / planner / f'trial_{trial:03d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_cumulative_info_gain(data, output_dir)
        print("  - cumulative_info_gain.png")

        plot_travel_cost(data, output_dir)
        print("  - travel_cost.png")

        plot_position_uncertainty(data, output_dir)
        print("  - position_uncertainty.png")

        plot_trajectory(data, output_dir)
        print("  - trajectory.png")

        plot_reconstruction(data, output_dir)
        print("  - reconstruction.png")

        plot_info_per_step(data, output_dir)
        print("  - info_per_step.png")

        generate_summary_csv(data, output_dir)
        print("  - summary.csv")

    # Generate comparison graphs for groups with multiple planners
    print("\n" + "-"*60)
    print("Generating comparison graphs...")
    print("-"*60)

    for (field, trial), group in groups.items():
        if len(group) < 2:
            print(f"\n[{field}/trial_{trial:03d}] Skipping comparison (only 1 planner)")
            continue

        print(f"\n[{field}/trial_{trial:03d}] Comparing {len(group)} planners")

        group_data = [all_loaded_data[(t['planner'], t['field_type'], t['trial'])] for t in group]

        output_dir = OUTPUT_ROOT / field / 'comparison' / f'trial_{trial:03d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison_bar(group_data, output_dir)
        print("  - comparison_rmse.png")

        plot_comparison_curves(group_data, output_dir)
        print("  - comparison_curves.png")

        plot_comparison_trajectories(group_data, output_dir)
        print("  - comparison_trajectories.png")

        generate_comparison_csv(group_data, output_dir)
        print("  - comparison_summary.csv")

    print("\n" + "="*60)
    print(f"DONE! All graphs saved to:\n  {OUTPUT_ROOT}")
    print("="*60)


def list_trials():
    """Just list available trials."""
    trials = find_all_trials()

    if not trials:
        print("No trials found!")
        print(f"Looking in: {DATA_ROOT}")
        return

    print(f"Found {len(trials)} trial(s):\n")
    for t in trials:
        print(f"  {t['planner']}/{t['field_type']}/trial_{t['trial']:03d}")
        print(f"    Path: {t['path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graphs for all existing trials')
    parser.add_argument('--list', action='store_true', help='Just list available trials')
    args = parser.parse_args()

    if args.list:
        list_trials()
    else:
        generate_all_graphs()
