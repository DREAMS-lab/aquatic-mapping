#!/usr/bin/env python3
"""
Planner Analysis & Visualization Script

Generates comprehensive graphs comparing exact vs pose_aware planners.
Saves results to data/reconstruction/trial_N/planner/

Usage:
    source venv/bin/activate
    python3 analyze_planners.py --trials 1 2 3
    python3 analyze_planners.py --all
    python3 analyze_planners.py --trials 1 --fields radial x_compress

Output structure:
    data/reconstruction/trial_N/planner/
    ├── exact/{field}/
    │   ├── compute.png
    │   ├── trajectory.png
    │   ├── info_gain.png
    │   ├── position_uncertainty.png
    │   ├── timing.png
    │   └── reconstruction.png
    ├── pose_aware/{field}/
    │   └── (same as above)
    └── comparison/{field}/
        ├── trajectory_comparison.png
        ├── info_gain_comparison.png
        ├── compute_comparison.png
        ├── metrics_comparison.png
        ├── timing_comparison.png
        └── summary.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = SCRIPT_DIR.parent
TRIALS_DATA_DIR = WORKSPACE_ROOT / "data" / "trials"
RESULTS_DIR = WORKSPACE_ROOT / "data" / "reconstruction"

ALL_FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
PLANNERS = ["exact", "pose_aware"]

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'exact': '#2ecc71',       # Green
    'pose_aware': '#e74c3c',  # Red
    'cpu': '#3498db',         # Blue
    'gpu': '#9b59b6',         # Purple
    'ram': '#f39c12',         # Orange
    'temp': '#e74c3c',        # Red
}


def load_samples(trial_dir: Path) -> pd.DataFrame:
    """Load samples.csv"""
    csv_path = trial_dir / "samples.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_compute_metrics(trial_dir: Path) -> pd.DataFrame:
    """Load compute_metrics.csv"""
    csv_path = trial_dir / "compute_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns and len(df) > 0:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df


def load_summary(trial_dir: Path) -> dict:
    """Load summary.json"""
    json_path = trial_dir / "summary.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def load_ground_truth(trial_dir: Path):
    """Load ground_truth.npz"""
    npz_path = trial_dir / "ground_truth.npz"
    if not npz_path.exists():
        return None, None, None
    data = np.load(npz_path)
    return data['X'], data['Y'], data['field']


def load_reconstruction(trial_dir: Path):
    """Load gp_reconstruction.npz"""
    npz_path = trial_dir / "gp_reconstruction.npz"
    if not npz_path.exists():
        return None, None, None, None
    data = np.load(npz_path)
    return data['X'], data['Y'], data['mean'], data['variance']


# =============================================================================
# Per-Planner Graphs
# =============================================================================

def plot_compute_metrics(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate compute metrics graph (CPU, GPU, RAM, Temp)"""
    df = load_compute_metrics(trial_dir)
    if df is None or len(df) < 2:
        print(f"    Skipping compute graph - no data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Compute Metrics - {planner} / {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')

    # CPU Usage
    ax = axes[0, 0]
    ax.plot(df['time'], df['cpu_percent'], color=COLORS['cpu'], linewidth=1.5)
    ax.fill_between(df['time'], df['cpu_percent'], alpha=0.3, color=COLORS['cpu'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title(f"CPU Usage (avg: {df['cpu_percent'].mean():.1f}%, max: {df['cpu_percent'].max():.1f}%)")
    ax.set_ylim(0, max(100, df['cpu_percent'].max() * 1.1))

    # GPU Usage
    ax = axes[0, 1]
    ax.plot(df['time'], df['gpu_util'], color=COLORS['gpu'], linewidth=1.5)
    ax.fill_between(df['time'], df['gpu_util'], alpha=0.3, color=COLORS['gpu'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title(f"GPU Usage (avg: {df['gpu_util'].mean():.1f}%, max: {df['gpu_util'].max():.1f}%)")
    ax.set_ylim(0, max(100, df['gpu_util'].max() * 1.1))

    # RAM Usage
    ax = axes[1, 0]
    ram_gb = df['ram_used_mb'] / 1024
    ax.plot(df['time'], ram_gb, color=COLORS['ram'], linewidth=1.5)
    ax.fill_between(df['time'], ram_gb, alpha=0.3, color=COLORS['ram'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RAM Used (GB)')
    ax.set_title(f"RAM Usage (avg: {ram_gb.mean():.1f} GB, max: {ram_gb.max():.1f} GB)")
    ax.set_ylim(0, ram_gb.max() * 1.2)

    # GPU Temperature
    ax = axes[1, 1]
    ax.plot(df['time'], df['gpu_temp'], color=COLORS['temp'], linewidth=1.5)
    ax.fill_between(df['time'], df['gpu_temp'], alpha=0.3, color=COLORS['temp'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU Temperature (°C)')
    ax.set_title(f"GPU Temp (max: {df['gpu_temp'].max():.0f}°C)")
    ax.set_ylim(0, max(100, df['gpu_temp'].max() * 1.2))

    plt.tight_layout()
    out_path = out_dir / "compute.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_trajectory(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate trajectory plot with sample points on field"""
    samples = load_samples(trial_dir)
    gt_X, gt_Y, gt_field = load_ground_truth(trial_dir)

    if samples is None or gt_field is None:
        print(f"    Skipping trajectory graph - no data")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot ground truth field
    im = ax.pcolormesh(gt_X, gt_Y, gt_field, cmap='coolwarm', shading='auto', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Temperature (°C)')

    # Plot trajectory
    ax.plot(samples['x'], samples['y'], 'k-', linewidth=1.5, alpha=0.7, label='Path')

    # Plot sample points
    initial = samples[samples['phase'] == 'initial']
    adaptive = samples[samples['phase'] == 'adaptive']

    ax.scatter(initial['x'], initial['y'], c='blue', s=100, marker='s',
               edgecolors='white', linewidths=2, label='Initial', zorder=5)
    ax.scatter(adaptive['x'], adaptive['y'], c='black', s=50, marker='o',
               edgecolors='white', linewidths=1, label='Adaptive', zorder=5)

    # Number the points
    for i, row in samples.iterrows():
        ax.annotate(str(int(row['step'])), (row['x'], row['y']),
                   fontsize=6, ha='center', va='bottom',
                   xytext=(0, 3), textcoords='offset points')

    # Start and end markers
    ax.scatter(samples.iloc[0]['x'], samples.iloc[0]['y'],
               c='lime', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Start', zorder=10)
    ax.scatter(samples.iloc[-1]['x'], samples.iloc[-1]['y'],
               c='red', s=200, marker='X', edgecolors='black', linewidths=2,
               label='End', zorder=10)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Trajectory - {planner} / {field} / Trial {trial_num}\n'
                 f'({len(samples)} samples, {samples["travel_cost"].iloc[-1]:.1f}m traveled)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = out_dir / "trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_info_gain(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate info gain plots"""
    samples = load_samples(trial_dir)
    if samples is None:
        print(f"    Skipping info gain graph - no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step info gain
    ax = axes[0]
    adaptive = samples[samples['phase'] == 'adaptive']
    ax.bar(adaptive['step'], adaptive['info_gain'], color=COLORS[planner], alpha=0.7)
    ax.axhline(adaptive['info_gain'].mean(), color='black', linestyle='--',
               label=f"Mean: {adaptive['info_gain'].mean():.3f}")
    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Information Gain (nats)')
    ax.set_title('Information Gain per Step')
    ax.legend()

    # Cumulative info gain
    ax = axes[1]
    ax.plot(samples['step'], samples['cumulative_info'],
            color=COLORS[planner], linewidth=2)
    ax.fill_between(samples['step'], samples['cumulative_info'],
                    alpha=0.3, color=COLORS[planner])
    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Cumulative Information Gain (nats)')
    ax.set_title(f"Cumulative Info Gain (total: {samples['cumulative_info'].iloc[-1]:.2f})")

    fig.suptitle(f'Information Gain - {planner} / {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "info_gain.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_position_uncertainty(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate position uncertainty plot"""
    samples = load_samples(trial_dir)
    if samples is None or 'pos_std_x' not in samples.columns:
        print(f"    Skipping position uncertainty graph - no data")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(samples['step'], samples['pos_std_x'] * 100,
            label='σx', color='#3498db', linewidth=2)
    ax.plot(samples['step'], samples['pos_std_y'] * 100,
            label='σy', color='#e74c3c', linewidth=2)

    ax.fill_between(samples['step'], samples['pos_std_x'] * 100, alpha=0.3, color='#3498db')
    ax.fill_between(samples['step'], samples['pos_std_y'] * 100, alpha=0.3, color='#e74c3c')

    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Position Std Dev (cm)')
    ax.set_title(f'Position Uncertainty (PX4 EKF) - {planner} / {field} / Trial {trial_num}\n'
                 f'Mean: σx={samples["pos_std_x"].mean()*100:.2f}cm, σy={samples["pos_std_y"].mean()*100:.2f}cm',
                 fontsize=12, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    out_path = out_dir / "position_uncertainty.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_timing(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate timing analysis plot"""
    compute = load_compute_metrics(trial_dir)
    samples = load_samples(trial_dir)

    if compute is None or samples is None:
        print(f"    Skipping timing graph - no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    total_time = compute['time'].iloc[-1] if len(compute) > 0 else 0
    n_samples = len(samples)
    avg_time = total_time / n_samples if n_samples > 0 else 0

    # Travel cost per step (proxy for time)
    ax = axes[0]
    travel_diffs = samples['travel_cost'].diff().fillna(samples['travel_cost'].iloc[0])
    ax.bar(samples['step'], travel_diffs, color=COLORS[planner], alpha=0.7)
    ax.axhline(travel_diffs.mean(), color='black', linestyle='--',
               label=f'Avg: {travel_diffs.mean():.2f}m')
    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Distance per Step (m)')
    ax.set_title('Distance Traveled per Step')
    ax.legend()

    # Cumulative travel
    ax = axes[1]
    ax.plot(samples['step'], samples['travel_cost'], color=COLORS[planner], linewidth=2)
    ax.fill_between(samples['step'], samples['travel_cost'], alpha=0.3, color=COLORS[planner])
    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Cumulative Distance (m)')
    ax.set_title(f'Total Distance Traveled: {samples["travel_cost"].iloc[-1]:.1f}m\n'
                 f'Mission Time: {total_time:.0f}s ({total_time/60:.1f}min)')

    fig.suptitle(f'Travel & Timing - {planner} / {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "timing.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_reconstruction(trial_dir: Path, out_dir: Path, planner: str, field: str, trial_num: int):
    """Generate reconstruction comparison (GP vs Ground Truth)"""
    samples = load_samples(trial_dir)
    gt_X, gt_Y, gt_field = load_ground_truth(trial_dir)
    recon_X, recon_Y, recon_mean, recon_var = load_reconstruction(trial_dir)
    summary = load_summary(trial_dir)

    if gt_field is None or recon_mean is None:
        print(f"    Skipping reconstruction graph - no data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    samples_arr = None
    if samples is not None and len(samples) > 0:
        samples_arr = samples[['x', 'y']].values

    # Ground truth
    ax = axes[0, 0]
    im = ax.pcolormesh(gt_X, gt_Y, gt_field, cmap='coolwarm', shading='auto')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    if samples_arr is not None:
        ax.scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=20, marker='x')
    ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # GP Reconstruction
    ax = axes[0, 1]
    im = ax.pcolormesh(recon_X, recon_Y, recon_mean, cmap='coolwarm', shading='auto')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    if samples_arr is not None:
        ax.scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=20, marker='x')
    ax.set_title('GP Reconstruction', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # Absolute Error
    ax = axes[1, 0]
    error = np.abs(recon_mean - gt_field)
    im = ax.pcolormesh(gt_X, gt_Y, error, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax, label='|Error| (°C)')
    rmse = summary.get('reconstruction_rmse', np.sqrt(np.mean(error**2))) if summary else np.sqrt(np.mean(error**2))
    mae = summary.get('reconstruction_mae', np.mean(error)) if summary else np.mean(error)
    ax.set_title(f'Absolute Error\nRMSE: {rmse:.3f}°C, MAE: {mae:.3f}°C', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # GP Variance
    ax = axes[1, 1]
    im = ax.pcolormesh(recon_X, recon_Y, recon_var, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Variance')
    if samples_arr is not None:
        ax.scatter(samples_arr[:, 0], samples_arr[:, 1], c='white', s=15, edgecolors='black', linewidths=0.5)
    ax.set_title(f'GP Variance (mean: {np.mean(recon_var):.4f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    fig.suptitle(f'Reconstruction - {planner} / {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "reconstruction.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


# =============================================================================
# Comparison Graphs (exact vs pose_aware)
# =============================================================================

def plot_trajectory_comparison(exact_dir: Path, pose_dir: Path, out_dir: Path,
                                field: str, trial_num: int):
    """Compare trajectories of both planners"""
    exact_samples = load_samples(exact_dir)
    pose_samples = load_samples(pose_dir)
    gt_X, gt_Y, gt_field = load_ground_truth(exact_dir)

    if exact_samples is None or pose_samples is None or gt_field is None:
        print(f"    Skipping trajectory comparison - missing data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (samples, name, color) in zip(axes[:2],
            [(exact_samples, 'Exact', COLORS['exact']),
             (pose_samples, 'Pose-Aware', COLORS['pose_aware'])]):
        ax.pcolormesh(gt_X, gt_Y, gt_field, cmap='coolwarm', shading='auto', alpha=0.7)
        ax.plot(samples['x'], samples['y'], 'k-', linewidth=1.5, alpha=0.7)
        ax.scatter(samples['x'], samples['y'], c=color, s=30, edgecolors='white', linewidths=0.5, zorder=5)
        ax.set_title(f'{name}\n({len(samples)} samples, {samples["travel_cost"].iloc[-1]:.1f}m)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        ax.set_aspect('equal')

    # Overlay
    ax = axes[2]
    ax.pcolormesh(gt_X, gt_Y, gt_field, cmap='coolwarm', shading='auto', alpha=0.5)
    ax.plot(exact_samples['x'], exact_samples['y'], '-', color=COLORS['exact'],
            linewidth=2, alpha=0.8, label='Exact')
    ax.plot(pose_samples['x'], pose_samples['y'], '-', color=COLORS['pose_aware'],
            linewidth=2, alpha=0.8, label='Pose-Aware')
    ax.set_title('Overlay Comparison', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    fig.suptitle(f'Trajectory Comparison - {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "trajectory_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_info_gain_comparison(exact_dir: Path, pose_dir: Path, out_dir: Path,
                               field: str, trial_num: int):
    """Compare info gain between planners"""
    exact_samples = load_samples(exact_dir)
    pose_samples = load_samples(pose_dir)

    if exact_samples is None or pose_samples is None:
        print(f"    Skipping info gain comparison - missing data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative info gain
    ax = axes[0]
    ax.plot(exact_samples['step'], exact_samples['cumulative_info'],
            color=COLORS['exact'], linewidth=2, label='Exact')
    ax.plot(pose_samples['step'], pose_samples['cumulative_info'],
            color=COLORS['pose_aware'], linewidth=2, label='Pose-Aware')
    ax.fill_between(exact_samples['step'], exact_samples['cumulative_info'],
                    alpha=0.2, color=COLORS['exact'])
    ax.fill_between(pose_samples['step'], pose_samples['cumulative_info'],
                    alpha=0.2, color=COLORS['pose_aware'])
    ax.set_xlabel('Sample Step')
    ax.set_ylabel('Cumulative Information Gain (nats)')
    ax.set_title('Cumulative Information Gain')
    ax.legend()

    # Final comparison bar
    ax = axes[1]
    exact_total = exact_samples['cumulative_info'].iloc[-1]
    pose_total = pose_samples['cumulative_info'].iloc[-1]

    bars = ax.bar(['Exact', 'Pose-Aware'], [exact_total, pose_total],
                  color=[COLORS['exact'], COLORS['pose_aware']], alpha=0.8)
    ax.bar_label(bars, fmt='%.2f', fontsize=11)
    ax.set_ylabel('Total Information Gain (nats)')
    ax.set_title('Final Information Gain')

    diff = pose_total - exact_total
    diff_pct = (diff / exact_total) * 100 if exact_total > 0 else 0
    ax.text(0.5, 0.95, f'Difference: {diff:+.2f} ({diff_pct:+.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Information Gain Comparison - {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "info_gain_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_compute_comparison(exact_dir: Path, pose_dir: Path, out_dir: Path,
                            field: str, trial_num: int):
    """Compare compute metrics between planners"""
    exact_compute = load_compute_metrics(exact_dir)
    pose_compute = load_compute_metrics(pose_dir)

    if exact_compute is None or pose_compute is None or \
       'time' not in (exact_compute.columns if exact_compute is not None else []) or \
       'time' not in (pose_compute.columns if pose_compute is not None else []):
        print(f"    Skipping compute comparison - missing data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CPU
    ax = axes[0, 0]
    ax.plot(exact_compute['time'], exact_compute['cpu_percent'],
            color=COLORS['exact'], linewidth=1.5, alpha=0.8, label='Exact')
    ax.plot(pose_compute['time'], pose_compute['cpu_percent'],
            color=COLORS['pose_aware'], linewidth=1.5, alpha=0.8, label='Pose-Aware')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title(f"CPU (Exact: {exact_compute['cpu_percent'].mean():.1f}%, "
                 f"Pose: {pose_compute['cpu_percent'].mean():.1f}%)")
    ax.legend()

    # GPU
    ax = axes[0, 1]
    ax.plot(exact_compute['time'], exact_compute['gpu_util'],
            color=COLORS['exact'], linewidth=1.5, alpha=0.8, label='Exact')
    ax.plot(pose_compute['time'], pose_compute['gpu_util'],
            color=COLORS['pose_aware'], linewidth=1.5, alpha=0.8, label='Pose-Aware')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title(f"GPU (Exact: {exact_compute['gpu_util'].mean():.1f}%, "
                 f"Pose: {pose_compute['gpu_util'].mean():.1f}%)")
    ax.legend()

    # RAM
    ax = axes[1, 0]
    exact_ram = exact_compute['ram_used_mb'] / 1024
    pose_ram = pose_compute['ram_used_mb'] / 1024
    ax.plot(exact_compute['time'], exact_ram,
            color=COLORS['exact'], linewidth=1.5, alpha=0.8, label='Exact')
    ax.plot(pose_compute['time'], pose_ram,
            color=COLORS['pose_aware'], linewidth=1.5, alpha=0.8, label='Pose-Aware')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RAM Used (GB)')
    ax.set_title(f"RAM (Exact: {exact_ram.mean():.1f}GB, Pose: {pose_ram.mean():.1f}GB)")
    ax.legend()

    # Summary bar
    ax = axes[1, 1]
    metrics = ['CPU (%)', 'GPU (%)', 'RAM (GB)']
    exact_vals = [exact_compute['cpu_percent'].mean(), exact_compute['gpu_util'].mean(), exact_ram.mean()]
    pose_vals = [pose_compute['cpu_percent'].mean(), pose_compute['gpu_util'].mean(), pose_ram.mean()]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, exact_vals, width, label='Exact', color=COLORS['exact'], alpha=0.8)
    bars2 = ax.bar(x + width/2, pose_vals, width, label='Pose-Aware', color=COLORS['pose_aware'], alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Average Resource Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', fontsize=9)

    fig.suptitle(f'Compute Comparison - {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "compute_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_metrics_comparison(exact_dir: Path, pose_dir: Path, out_dir: Path,
                            field: str, trial_num: int):
    """Compare final metrics (RMSE, MAE, travel cost, info gain)"""
    exact_summary = load_summary(exact_dir)
    pose_summary = load_summary(pose_dir)

    if exact_summary is None or pose_summary is None:
        print(f"    Skipping metrics comparison - missing data")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    metrics = [
        ('reconstruction_rmse', 'RMSE (°C)', 'lower is better'),
        ('reconstruction_mae', 'MAE (°C)', 'lower is better'),
        ('total_travel_cost', 'Travel Cost (m)', 'lower is better'),
        ('cumulative_info_gain', 'Info Gain (nats)', 'higher is better'),
    ]

    for ax, (key, label, note) in zip(axes, metrics):
        exact_val = exact_summary.get(key, 0)
        pose_val = pose_summary.get(key, 0)

        bars = ax.bar(['Exact', 'Pose-Aware'], [exact_val, pose_val],
                      color=[COLORS['exact'], COLORS['pose_aware']], alpha=0.8)
        ax.bar_label(bars, fmt='%.2f', fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n({note})', fontsize=10)

        diff = pose_val - exact_val
        diff_pct = (diff / exact_val) * 100 if exact_val != 0 else 0
        ax.text(0.5, 0.02, f'Δ: {diff:+.2f} ({diff_pct:+.1f}%)',
                transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(f'Metrics Comparison - {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def plot_timing_comparison(exact_dir: Path, pose_dir: Path, out_dir: Path,
                           field: str, trial_num: int):
    """Compare timing between planners"""
    exact_compute = load_compute_metrics(exact_dir)
    pose_compute = load_compute_metrics(pose_dir)
    exact_samples = load_samples(exact_dir)
    pose_samples = load_samples(pose_dir)

    if exact_compute is None or pose_compute is None or \
       'time' not in (exact_compute.columns if exact_compute is not None else []) or \
       'time' not in (pose_compute.columns if pose_compute is not None else []):
        print(f"    Skipping timing comparison - missing data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    exact_time = exact_compute['time'].iloc[-1] if len(exact_compute) > 0 else 0
    pose_time = pose_compute['time'].iloc[-1] if len(pose_compute) > 0 else 0

    # Total mission time
    ax = axes[0]
    bars = ax.bar(['Exact', 'Pose-Aware'], [exact_time/60, pose_time/60],
                  color=[COLORS['exact'], COLORS['pose_aware']], alpha=0.8)
    ax.bar_label(bars, fmt='%.1f min', fontsize=11)
    ax.set_ylabel('Mission Time (minutes)')
    ax.set_title('Total Mission Duration')

    # Time per sample
    ax = axes[1]
    exact_n = len(exact_samples) if exact_samples is not None else 1
    pose_n = len(pose_samples) if pose_samples is not None else 1
    exact_per = exact_time / exact_n
    pose_per = pose_time / pose_n

    bars = ax.bar(['Exact', 'Pose-Aware'], [exact_per, pose_per],
                  color=[COLORS['exact'], COLORS['pose_aware']], alpha=0.8)
    ax.bar_label(bars, fmt='%.1f s', fontsize=11)
    ax.set_ylabel('Time per Sample (seconds)')
    ax.set_title('Average Time per Sample')

    fig.suptitle(f'Timing Comparison - {field} / Trial {trial_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / "timing_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


def save_summary_csv(exact_dir: Path, pose_dir: Path, out_dir: Path,
                     field: str, trial_num: int):
    """Save comparison summary as CSV"""
    exact_summary = load_summary(exact_dir)
    pose_summary = load_summary(pose_dir)
    exact_compute = load_compute_metrics(exact_dir)
    pose_compute = load_compute_metrics(pose_dir)

    if exact_summary is None or pose_summary is None:
        return

    exact_time = exact_compute['time'].iloc[-1] if exact_compute is not None and 'time' in exact_compute.columns and len(exact_compute) > 0 else 0
    pose_time = pose_compute['time'].iloc[-1] if pose_compute is not None and 'time' in pose_compute.columns and len(pose_compute) > 0 else 0

    data = {
        'Metric': ['RMSE (°C)', 'MAE (°C)', 'Max Error (°C)', 'Travel Cost (m)',
                   'Info Gain (nats)', 'Mean GP Variance', 'Mission Time (s)', 'Samples'],
        'Exact': [
            exact_summary.get('reconstruction_rmse', 0),
            exact_summary.get('reconstruction_mae', 0),
            exact_summary.get('reconstruction_max_error', 0),
            exact_summary.get('total_travel_cost', 0),
            exact_summary.get('cumulative_info_gain', 0),
            exact_summary.get('mean_gp_variance', 0),
            exact_time,
            exact_summary.get('total_samples', 0),
        ],
        'Pose-Aware': [
            pose_summary.get('reconstruction_rmse', 0),
            pose_summary.get('reconstruction_mae', 0),
            pose_summary.get('reconstruction_max_error', 0),
            pose_summary.get('total_travel_cost', 0),
            pose_summary.get('cumulative_info_gain', 0),
            pose_summary.get('mean_gp_variance', 0),
            pose_time,
            pose_summary.get('total_samples', 0),
        ],
    }

    df = pd.DataFrame(data)
    df['Difference'] = df['Pose-Aware'] - df['Exact']
    df['Diff %'] = (df['Difference'] / df['Exact'] * 100).round(2)

    out_path = out_dir / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"    Saved: {out_path.relative_to(SCRIPT_DIR)}")


# =============================================================================
# Main Processing
# =============================================================================

def process_trial(field: str, trial_num: int):
    """Process a single trial for both planners"""
    print(f"\n{'='*60}")
    print(f"Processing: {field} / Trial {trial_num}")
    print(f"{'='*60}")

    # Source data directories
    exact_data_dir = TRIALS_DATA_DIR / "exact" / field / f"trial_{trial_num:03d}"
    pose_data_dir = TRIALS_DATA_DIR / "pose_aware" / field / f"trial_{trial_num:03d}"

    # Output directories (in reconstruction/results/)
    trial_results_dir = RESULTS_DIR / f"trial_{trial_num}" / "planner"
    exact_out_dir = trial_results_dir / "exact" / field
    pose_out_dir = trial_results_dir / "pose_aware" / field
    comparison_out_dir = trial_results_dir / "comparison" / field

    # Check what data exists
    exact_exists = exact_data_dir.exists() and (exact_data_dir / "samples.csv").exists()
    pose_exists = pose_data_dir.exists() and (pose_data_dir / "samples.csv").exists()

    # Process exact planner
    if exact_exists:
        print(f"\n  [Exact Planner]")
        exact_out_dir.mkdir(parents=True, exist_ok=True)
        plot_compute_metrics(exact_data_dir, exact_out_dir, "exact", field, trial_num)
        plot_trajectory(exact_data_dir, exact_out_dir, "exact", field, trial_num)
        plot_info_gain(exact_data_dir, exact_out_dir, "exact", field, trial_num)
        plot_position_uncertainty(exact_data_dir, exact_out_dir, "exact", field, trial_num)
        plot_timing(exact_data_dir, exact_out_dir, "exact", field, trial_num)
        plot_reconstruction(exact_data_dir, exact_out_dir, "exact", field, trial_num)
    else:
        print(f"  [Exact] No data found")

    # Process pose_aware planner
    if pose_exists:
        print(f"\n  [Pose-Aware Planner]")
        pose_out_dir.mkdir(parents=True, exist_ok=True)
        plot_compute_metrics(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
        plot_trajectory(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
        plot_info_gain(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
        plot_position_uncertainty(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
        plot_timing(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
        plot_reconstruction(pose_data_dir, pose_out_dir, "pose_aware", field, trial_num)
    else:
        print(f"  [Pose-Aware] No data found")

    # Generate comparison graphs
    if exact_exists and pose_exists:
        print(f"\n  [Comparison]")
        comparison_out_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectory_comparison(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
        plot_info_gain_comparison(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
        plot_compute_comparison(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
        plot_metrics_comparison(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
        plot_timing_comparison(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
        save_summary_csv(exact_data_dir, pose_data_dir, comparison_out_dir, field, trial_num)
    else:
        print(f"  [Comparison] Skipped - need both planners")


def discover_trials() -> dict:
    """Discover all available trials"""
    trials = {}

    for planner in PLANNERS:
        planner_dir = TRIALS_DATA_DIR / planner
        if not planner_dir.exists():
            continue

        for field in ALL_FIELDS:
            field_dir = planner_dir / field
            if not field_dir.exists():
                continue

            for trial_dir in field_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                    try:
                        trial_num = int(trial_dir.name.split("_")[1])
                        key = (field, trial_num)
                        if key not in trials:
                            trials[key] = set()
                        trials[key].add(planner)
                    except:
                        pass

    return trials


def main():
    parser = argparse.ArgumentParser(description="Analyze planner trials")
    parser.add_argument("--trials", "-t", type=int, nargs="+",
                        help="Trial numbers to process (e.g., 1 2 3)")
    parser.add_argument("--fields", "-f", type=str, nargs="+",
                        help="Fields to process (default: all)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Process all available trials")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available trials")

    args = parser.parse_args()

    print(f"Planner Analysis Script")
    print(f"Data source: {TRIALS_DATA_DIR}")
    print(f"Output dir:  {RESULTS_DIR}")
    print()

    available = discover_trials()

    if args.list:
        print("Available trials:")
        for (field, trial_num), planners in sorted(available.items()):
            planners_str = ", ".join(sorted(planners))
            print(f"  {field}/trial_{trial_num:03d}: {planners_str}")
        return

    if not available:
        print("No trials found!")
        return

    fields_to_process = args.fields if args.fields else ALL_FIELDS

    if args.all:
        trials_to_process = [(f, t) for (f, t) in available.keys()
                             if f in fields_to_process]
    elif args.trials:
        trials_to_process = [(f, t) for f in fields_to_process
                             for t in args.trials
                             if (f, t) in available]
    else:
        trials_to_process = [(f, 1) for f in fields_to_process
                             if (f, 1) in available]

    if not trials_to_process:
        print("No matching trials found!")
        print("Use --list to see available trials")
        return

    print(f"Processing {len(trials_to_process)} trial(s)...")

    for field, trial_num in sorted(trials_to_process):
        process_trial(field, trial_num)

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
