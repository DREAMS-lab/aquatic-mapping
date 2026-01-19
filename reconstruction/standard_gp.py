#!/usr/bin/env python3
"""
Standard Gaussian Process (deterministic inputs baseline)

Implements Standard GP with various kernels for comparison with
Girard uncertain-input GP and McHutchon NIGP.
"""
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardGPModel(gpytorch.models.ExactGP):
    """Standard GP with deterministic inputs

    Supports multiple kernels: RBF, Exponential, Matern1.5, Matern2.5
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        """
        args:
            train_x: (N, 2) training positions
            train_y: (N,) training outputs
            likelihood: Gaussian likelihood
            kernel_type: 'rbf', 'exponential', 'matern15', 'matern25'
        """
        super(StandardGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Map kernel type to GPyTorch kernel
        if kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'exponential':
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel_type == 'matern15':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == 'matern25':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_standard_gp(model, likelihood, train_x, train_y, n_iter=100):
    """train standard GP with deterministic inputs

    args:
        model: StandardGPModel instance
        likelihood: GaussianLikelihood
        train_x: (N, 2) training inputs
        train_y: (N,) training outputs
        n_iter: optimization iterations

    returns:
        hyperparams: dict of learned hyperparameters
    """
    print(f"\ntraining Standard GP ({n_iter} iterations)...")

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  iter {i+1}/{n_iter}, loss: {loss.item():.3f}")

    model.eval()
    likelihood.eval()

    # extract hyperparameters
    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'mean_constant': model.mean_module.constant.item(),
        'noise': likelihood.noise.item(),
    }

    return hyperparams


def predict_standard_gp(model, likelihood, test_x):
    """predict with standard GP

    args:
        model: trained StandardGPModel
        likelihood: trained likelihood
        test_x: (M, 2) test inputs

    returns:
        mean: (M,) predictive mean
        variance: (M,) predictive variance
    """
    model.eval()
    likelihood.eval()

    print("predicting (standard GP)...")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = model(test_x)
        pred = likelihood(dist)
        mean = pred.mean
        variance = pred.variance

    return mean, variance


def visualize_standard_gp(X, Y, ground_truth, pred_mean, pred_variance,
                          samples_x, samples_y, kernel, field_type,
                          trial_number, output_dir):
    """create visualization for standard GP prediction

    args:
        X, Y: meshgrid coordinates
        ground_truth: (H, W) ground truth field
        pred_mean: (H, W) predicted mean
        pred_variance: (H, W) predicted variance
        samples_x, samples_y: sample locations
        kernel: kernel type
        field_type: field type
        trial_number: trial number
        output_dir: output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    vmin, vmax = ground_truth.min(), ground_truth.max()

    # ground truth
    im0 = axes[0, 0].contourf(X, Y, ground_truth, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].scatter(samples_x, samples_y, c='cyan', s=10, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0, 0].set_title('ground truth')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    plt.colorbar(im0, ax=axes[0, 0], label='temperature [°C]')

    # predicted mean
    im1 = axes[0, 1].contourf(X, Y, pred_mean, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Standard GP predicted mean')
    axes[0, 1].set_xlabel('x [m]')
    axes[0, 1].set_ylabel('y [m]')
    plt.colorbar(im1, ax=axes[0, 1], label='temperature [°C]')

    # error
    error = pred_mean - ground_truth
    error_max = max(abs(error.min()), abs(error.max()))
    im2 = axes[1, 0].contourf(X, Y, error, levels=20, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    axes[1, 0].set_title('error (prediction - truth)')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('y [m]')
    plt.colorbar(im2, ax=axes[1, 0], label='error [°C]')

    # predicted variance
    im3 = axes[1, 1].contourf(X, Y, pred_variance, levels=20, cmap='viridis')
    axes[1, 1].set_title('Standard GP predictive variance')
    axes[1, 1].set_xlabel('x [m]')
    axes[1, 1].set_ylabel('y [m]')
    plt.colorbar(im3, ax=axes[1, 1], label='variance [°C²]')

    plt.tight_layout()
    output_path = output_dir / f'{field_type}_{kernel}_standard_gp.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {output_path}")


def save_standard_gp_results(metrics, hyperparams, field_type, kernel, trial_number, output_dir):
    """save standard GP metrics and hyperparameters

    args:
        metrics: dict with mse, rmse, mae, nrmse
        hyperparams: dict with learned hyperparameters
        field_type: field type
        kernel: kernel type
        trial_number: trial number
        output_dir: output directory
    """
    # save metrics CSV
    metrics_path = output_dir / f'{field_type}_{kernel}_metrics.csv'
    with open(metrics_path, 'w') as f:
        f.write('field_type,trial,kernel,method,mse,rmse,mae,nrmse\n')
        f.write(f'{field_type},{trial_number},{kernel},standard_gp,'
                f'{metrics["mse"]},{metrics["rmse"]},{metrics["mae"]},{metrics["nrmse"]}\n')

    # save hyperparameters JSON
    hyperparams_path = output_dir / f'{field_type}_{kernel}_hyperparams.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"saved: {metrics_path}")
    print(f"saved: {hyperparams_path}")
