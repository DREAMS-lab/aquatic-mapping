#!/usr/bin/env python3
"""
Main reconstruction script for comparing Standard GP, McHutchon NIGP, and Girard uncertain-input GP

Usage:
    python run_reconstruction.py <field_type> [trial_number] [method]

Examples:
    python run_reconstruction.py radial 1 all
    python run_reconstruction.py radial 1 standard
    python run_reconstruction.py radial 1 mchutchon
    python run_reconstruction.py radial 1 girard
    python run_reconstruction.py all 1 all
"""
import torch
import gpytorch
import numpy as np
import sys
from pathlib import Path

# add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loading import load_sample_data
from utils.ground_truth import generate_ground_truth_field
from utils.metrics import compute_metrics
from mchutchon_nigp import (
    McHutchonNIGPModel, train_mchutchon_nigp, predict_mchutchon_nigp,
    visualize_mchutchon_nigp, save_mchutchon_nigp_results
)
from girard_uncertain_input import (
    GirardUncertainInputModel, train_girard_gp, predict_girard_uncertain_input,
    visualize_girard_prediction, save_girard_results
)
from standard_gp import (
    StandardGPModel, train_standard_gp, predict_standard_gp,
    visualize_standard_gp, save_standard_gp_results
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")


def run_mchutchon_nigp(field_type, trial_number, kernel, train_x, train_y, train_cov,
                        test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                        output_dir):
    """run McHutchon NIGP reconstruction"""
    print(f"\n{'='*70}")
    print(f"METHOD: McHutchon NIGP | KERNEL: {kernel}")
    print(f"{'='*70}")

    # initialize model
    dummy_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = McHutchonNIGPModel(train_x, train_y, train_cov, dummy_likelihood, kernel).to(device)

    # train
    likelihood, input_var, hyperparams = train_mchutchon_nigp(model, train_x, train_y, n_iter=100, n_em_steps=3)

    # predict McHutchon
    print("predicting...")
    pred_mean, pred_std = predict_mchutchon_nigp(model, likelihood, test_x)
    pred_mean_grid = pred_mean.cpu().numpy().reshape(X.shape)
    pred_variance_grid = (pred_std ** 2).cpu().numpy().reshape(X.shape)

    # Read Standard GP predictions from saved results (computed separately in main loop)
    std_pred_mean_grid = None
    std_pred_variance_grid = None
    std_results_dir = Path('results') / f'trial_{trial_number}' / 'standard_gp' / field_type / kernel
    std_pred_path = std_results_dir / f'{field_type}_{kernel}_predictions.npy'
    std_var_path = std_results_dir / f'{field_type}_{kernel}_variance.npy'

    if std_pred_path.exists() and std_var_path.exists():
        import numpy as np
        std_pred_mean_grid = np.load(std_pred_path)
        std_pred_variance_grid = np.load(std_var_path)
        print(f"loaded Standard GP predictions from {std_results_dir}")

    # metrics
    metrics = compute_metrics(pred_mean, ground_truth)
    print(f"RMSE: {metrics['rmse']:.4f}, NRMSE: {metrics['nrmse']:.4f}")

    # visualize
    visualize_mchutchon_nigp(X, Y, temp_field_gt, pred_mean_grid,
                              x_samples, y_samples, kernel, field_type,
                              trial_number, output_dir, pred_variance=pred_variance_grid,
                              standard_gp_pred_mean=std_pred_mean_grid,
                              standard_gp_pred_variance=std_pred_variance_grid)

    # save results
    save_mchutchon_nigp_results(metrics, hyperparams, field_type, kernel,
                                 trial_number, output_dir)

    return metrics


def run_girard_uncertain_input(field_type, trial_number, kernel, train_x, train_y, train_cov,
                                 test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                                 output_dir):
    """run Girard uncertain-input GP reconstruction (RBF-only with analytic expected kernel)"""
    print(f"\n{'='*70}")
    print(f"METHOD: Girard Uncertain Input (RBF-only) | KERNEL: {kernel}")
    print(f"{'='*70}")

    # initialize model with training covariance
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GirardUncertainInputModel(train_x, train_y, train_cov, likelihood, kernel).to(device)

    # train (standard GP with expected RBF kernel)
    hyperparams = train_girard_gp(model, likelihood, train_x, train_y, n_iter=100)

    # predict with analytic expected kernel
    pred_mean, pred_variance = predict_girard_uncertain_input(model, likelihood, test_x)
    pred_mean_grid = pred_mean.cpu().numpy().reshape(X.shape)
    pred_variance_grid = pred_variance.cpu().numpy().reshape(X.shape)

    # Read Standard GP predictions from saved results (computed separately in main loop)
    std_pred_mean_grid = None
    std_pred_variance_grid = None
    std_results_dir = Path('results') / f'trial_{trial_number}' / 'standard_gp' / field_type / kernel
    std_pred_path = std_results_dir / f'{field_type}_{kernel}_predictions.npy'
    std_var_path = std_results_dir / f'{field_type}_{kernel}_variance.npy'

    if std_pred_path.exists() and std_var_path.exists():
        import numpy as np
        std_pred_mean_grid = np.load(std_pred_path)
        std_pred_variance_grid = np.load(std_var_path)
        print(f"loaded Standard GP predictions from {std_results_dir}")

    # metrics
    metrics = compute_metrics(pred_mean, ground_truth)
    print(f"RMSE: {metrics['rmse']:.4f}, NRMSE: {metrics['nrmse']:.4f}")

    # visualize
    visualize_girard_prediction(X, Y, temp_field_gt, pred_mean_grid, pred_variance_grid,
                                 x_samples, y_samples, kernel, field_type,
                                 trial_number, output_dir, standard_gp_pred_mean=std_pred_mean_grid,
                                 standard_gp_pred_variance=std_pred_variance_grid)

    # save results
    save_girard_results(metrics, hyperparams, field_type, kernel,
                        trial_number, output_dir)

    return metrics


def run_standard_gp(field_type, trial_number, kernel, train_x, train_y, train_cov,
                    test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                    output_dir):
    """run Standard GP reconstruction (deterministic inputs baseline)"""
    print(f"\n{'='*70}")
    print(f"METHOD: Standard GP (deterministic) | KERNEL: {kernel}")
    print(f"{'='*70}")

    # initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = StandardGPModel(train_x, train_y, likelihood, kernel).to(device)

    # train
    hyperparams = train_standard_gp(model, likelihood, train_x, train_y, n_iter=100)

    # predict
    pred_mean, pred_variance = predict_standard_gp(model, likelihood, test_x)
    pred_mean_grid = pred_mean.cpu().numpy().reshape(X.shape)
    pred_variance_grid = pred_variance.cpu().numpy().reshape(X.shape)

    # metrics
    metrics = compute_metrics(pred_mean, ground_truth)
    print(f"RMSE: {metrics['rmse']:.4f}, NRMSE: {metrics['nrmse']:.4f}")

    # visualize
    visualize_standard_gp(X, Y, temp_field_gt, pred_mean_grid, pred_variance_grid,
                          x_samples, y_samples, kernel, field_type,
                          trial_number, output_dir)

    # save results
    save_standard_gp_results(metrics, hyperparams, field_type, kernel,
                             trial_number, output_dir)

    # save predictions as numpy arrays for use by other methods
    import numpy as np
    np.save(output_dir / f'{field_type}_{kernel}_predictions.npy', pred_mean_grid)
    np.save(output_dir / f'{field_type}_{kernel}_variance.npy', pred_variance_grid)

    return metrics


def run_single_field(field_type, trial_number, method='all'):
    """run reconstruction for a single field type

    args:
        field_type: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt
        trial_number: trial number
        method: 'standard', 'mchutchon', 'girard', or 'all'
    """
    kernels = ['rbf', 'exponential', 'matern15', 'matern25']

    # data paths
    data_dir = Path(__file__).parent.parent / 'src' / 'sampling' / 'data' / 'missions'
    csv_path = data_dir / field_type / f'trial_{trial_number}' / f'{field_type}_samples.csv'

    if not csv_path.exists():
        print(f"ERROR: data not found at {csv_path}")
        return False

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION: {field_type} (trial {trial_number})")
    print(f"{'='*70}\n")

    # load data
    x_samples, y_samples, temp_samples, position_cov = load_sample_data(csv_path)

    # generate ground truth
    print("\ngenerating ground truth field...")
    X, Y, temp_field_gt = generate_ground_truth_field(field_type, resolution=0.5)
    print(f"ground truth shape: {temp_field_gt.shape}")
    print(f"temperature range: [{temp_field_gt.min():.2f}, {temp_field_gt.max():.2f}] °C\n")

    # convert to torch tensors
    train_x = torch.tensor(np.c_[x_samples, y_samples], dtype=torch.float32).to(device)
    train_y = torch.tensor(temp_samples, dtype=torch.float32).to(device)
    train_cov = torch.tensor(position_cov, dtype=torch.float32).to(device)

    test_x_np = np.c_[X.ravel(), Y.ravel()]
    test_x = torch.tensor(test_x_np, dtype=torch.float32).to(device)
    ground_truth = torch.tensor(temp_field_gt.ravel(), dtype=torch.float32).to(device)

    # run reconstructions
    for kernel in kernels:
        if method in ['standard', 'all']:
            # create output directory
            output_dir = Path('results') / f'trial_{trial_number}' / 'standard_gp' / field_type / kernel
            output_dir.mkdir(parents=True, exist_ok=True)

            run_standard_gp(field_type, trial_number, kernel, train_x, train_y, train_cov,
                            test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                            output_dir)

        if method in ['mchutchon', 'all']:
            # create output directory
            output_dir = Path('results') / f'trial_{trial_number}' / 'mchutchon_nigp' / field_type / kernel
            output_dir.mkdir(parents=True, exist_ok=True)

            run_mchutchon_nigp(field_type, trial_number, kernel, train_x, train_y, train_cov,
                                test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                                output_dir)

        if method in ['girard', 'all']:
            # Girard method RBF-only
            if kernel != 'rbf':
                print(f"\nskipping Girard for {kernel} (RBF-only)")
                continue

            # create output directory
            output_dir = Path('results') / f'trial_{trial_number}' / 'girard' / field_type / kernel
            output_dir.mkdir(parents=True, exist_ok=True)

            run_girard_uncertain_input(field_type, trial_number, kernel, train_x, train_y, train_cov,
                                        test_x, ground_truth, X, Y, temp_field_gt, x_samples, y_samples,
                                        output_dir)

    return True


def main():
    """main reconstruction pipeline"""
    if len(sys.argv) < 2:
        print("usage: python run_reconstruction.py <field_type> [trial_number] [method]")
        print("  field_type: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt, all")
        print("  trial_number: default=1")
        print("  method: standard, mchutchon, girard, all (default=all)")
        sys.exit(1)

    field_type = sys.argv[1]
    trial_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    method = sys.argv[3] if len(sys.argv) > 3 else 'all'

    valid_fields = ['radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt']
    valid_methods = ['standard', 'mchutchon', 'girard', 'all']

    if method not in valid_methods:
        print(f"ERROR: invalid method '{method}'")
        print(f"valid options: {', '.join(valid_methods)}")
        sys.exit(1)

    # run all fields or single field
    if field_type == 'all':
        print("\n" + "="*70)
        print(f"RECONSTRUCTION - ALL FIELDS (trial {trial_number}, method={method})")
        print("="*70)

        results = {}
        for field in valid_fields:
            try:
                success = run_single_field(field, trial_number, method)
                results[field] = success
            except Exception as e:
                print(f"\nERROR processing {field}: {e}")
                import traceback
                traceback.print_exc()
                results[field] = False

        # summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70 + "\n")

        for field, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {field}")

        all_success = all(results.values())
        if all_success:
            print("\n✓ all reconstructions completed successfully!")
            print(f"\nresults saved to: {Path('results').absolute()}")
        else:
            print("\n✗ some reconstructions failed")
            sys.exit(1)

    else:
        if field_type not in valid_fields:
            print(f"ERROR: invalid field_type '{field_type}'")
            print(f"valid options: {', '.join(valid_fields)}, all")
            sys.exit(1)

        run_single_field(field_type, trial_number, method)


if __name__ == '__main__':
    main()
