#!/usr/bin/env python3
"""
Generate error maps comparing each reconstruction method to ground truth

Usage:
    python compare_methods.py <trial_number>
    python compare_methods.py all

Example:
    python compare_methods.py 1
    python compare_methods.py all
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.ground_truth import generate_ground_truth_field


FIELDS = ['radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt']
KERNELS = ['rbf', 'exponential', 'matern15', 'matern25']
METHODS = ['standard_gp', 'mchutchon_nigp', 'girard']

METHOD_DISPLAY = {
    'standard_gp': 'Standard GP',
    'mchutchon_nigp': 'McHutchon NIGP',
    'girard': 'Girard'
}


def generate_error_map(trial_number, field_type, kernel, results_dir):
    """generate error map for all available methods"""

    X, Y, gt_grid = generate_ground_truth_field(field_type, resolution=0.5)
    trial_dir = results_dir / f'trial_{trial_number}'

    # find available methods
    methods_found = []
    predictions = {}

    for method in METHODS:
        pred_path = trial_dir / method / field_type / kernel / f'{field_type}_{kernel}_predictions.npy'
        if pred_path.exists():
            predictions[method] = np.load(pred_path)
            methods_found.append(method)

    if not methods_found:
        return False

    # create figure: ground truth + error map for each method
    n_cols = 1 + len(methods_found)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))

    vmin, vmax = gt_grid.min(), gt_grid.max()

    # ground truth
    im0 = axes[0].pcolormesh(X, Y, gt_grid, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='Temp [°C]')

    # error maps for each method
    for i, method in enumerate(methods_found):
        pred_grid = predictions[method]
        error = pred_grid - gt_grid
        rmse = np.sqrt(np.mean(error**2))

        max_abs = max(abs(error.min()), abs(error.max()))
        if max_abs > 0:
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        else:
            norm = None

        im = axes[i+1].pcolormesh(X, Y, error, cmap='RdBu_r', norm=norm, shading='auto')
        axes[i+1].set_title(f'{METHOD_DISPLAY[method]}\nRMSE={rmse:.3f}', fontsize=11)
        axes[i+1].set_xlabel('X [m]')
        axes[i+1].set_ylabel('Y [m]')
        axes[i+1].set_aspect('equal')
        plt.colorbar(im, ax=axes[i+1], label='Error [°C]')

    plt.suptitle(f'{field_type} / {kernel}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # save
    output_dir = trial_dir / 'comparison' / field_type
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{kernel}_error_maps.png', dpi=150, bbox_inches='tight')
    plt.close()

    return True


def run_trial(trial_number, results_dir):
    """run error map generation for a single trial"""

    trial_dir = results_dir / f'trial_{trial_number}'
    if not trial_dir.exists():
        print(f"  trial {trial_number}: not found")
        return

    print(f"\nTrial {trial_number}:")

    count = 0
    for field_type in FIELDS:
        for kernel in KERNELS:
            if generate_error_map(trial_number, field_type, kernel, results_dir):
                count += 1

    print(f"  generated {count} error maps")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable trials:")
        results_dir = Path(__file__).parent / 'results'
        trials = sorted([d.name for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')])
        for t in trials:
            print(f"  {t}")
        sys.exit(1)

    results_dir = Path(__file__).parent / 'results'

    if sys.argv[1] == 'all':
        print("Generating error maps for all trials...")
        trials = sorted([int(d.name.split('_')[1]) for d in results_dir.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])
        for trial_number in trials:
            run_trial(trial_number, results_dir)
    else:
        trial_number = int(sys.argv[1])
        run_trial(trial_number, results_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
