#!/usr/bin/env python3
"""
Girard et al. (2003) Uncertain-Input Gaussian Process

Analytic uncertain-input GP using expected RBF kernel.
Handles positional uncertainty via closed-form expected kernel computation.

Reference: Girard et al., "Gaussian Process Priors with Uncertain Inputs", 2003
"""
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExpectedRBFKernel(gpytorch.kernels.Kernel):
    """Expected RBF kernel under input uncertainty

    Computes the analytic expected value of RBF kernel when inputs are uncertain:

        E[k(x, x')] where x ~ N(μ_x, Σ_x), x' ~ N(μ_x', Σ_x')

    Closed-form solution for Gaussian inputs and RBF kernel.

    Covariance handling: Uses explicit covariance attributes (Pattern A).
    Caller must set kernel.cov1 and kernel.cov2 before calling forward().
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(ExpectedRBFKernel, self).__init__(**kwargs)
        # Explicit covariance storage (Pattern A)
        # Caller sets these before forward() call
        self.cov1 = None  # (N, 3) covariance for x1 or None
        self.cov2 = None  # (M, 3) covariance for x2 or None

    def forward(self, x1, x2, diag=False, **kwargs):
        """
        args:
            x1: (N, 2) first set of means
            x2: (M, 2) second set of means
            diag: if True, return diagonal only
            cov1: set via self.cov1 before calling
            cov2: set via self.cov2 before calling

        returns:
            covariance matrix with expected kernel values
        """
        lengthscale = self.lengthscale

        if diag:
            # diagonal: E[k(x_i, x_i)]
            # Use cov1 as the covariance for x1
            return self._expected_diag(x1, lengthscale, self.cov1)
        else:
            # full matrix: E[k(x_i, x_j)] for all i, j
            return self._expected_kernel(x1, x2, lengthscale, self.cov1, self.cov2)

    def _expected_diag(self, x, lengthscale, cov):
        """compute diagonal E[k(x_i, x_i)]

        For uncertain inputs, diagonal may be < 1.
        For test grid points (no covariance), diagonal = 1.

        Girard et al. (2003) formula for diagonal:
            diag_i = (l²) / sqrt((l² + 2*σ²_x[i]) * (l² + 2*σ²_y[i]))

        args:
            x: (N, 2) input means
            lengthscale: kernel lengthscale
            cov: (N, 3) covariance or None
        """
        lengthscale_sq = lengthscale ** 2

        if cov is None:
            # No covariance: standard RBF diagonal = 1
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

        # Extract diagonal variances from covariance
        cov_xx = cov[:, 0]  # (N,)
        cov_yy = cov[:, 2]  # (N,)

        # For diagonal: sx = 2*cov_xx[i], sy = 2*cov_yy[i]
        sx = 2.0 * cov_xx
        sy = 2.0 * cov_yy

        # A matrices
        Ax = lengthscale_sq + sx
        Ay = lengthscale_sq + sy

        # Normalization factor: l² / sqrt(Ax * Ay)
        # When dx=dy=0 on diagonal, kernel = norm * exp(0) = norm
        diag = lengthscale_sq / torch.sqrt(Ax * Ay)

        return diag

    def _expected_kernel(self, x1, x2, lengthscale, cov1, cov2):
        """compute full expected kernel matrix using Girard et al. (2003) formula

        For Gaussian inputs x ~ N(μ_x, Σ_x) and x' ~ N(μ_x', Σ_x'):

        K_ij = (l² / sqrt(Ax * Ay)) * exp(-0.5 * (dx²/Ax + dy²/Ay))

        where:
            Ax = l² + σ²_x[i] + σ²_x[j]
            Ay = l² + σ²_y[i] + σ²_y[j]
            dx = μ_x[i] - μ_x'[j]
            dy = μ_y[i] - μ_y'[j]

        args:
            x1: (N, 2) first set of input means
            x2: (M, 2) second set of input means
            lengthscale: kernel lengthscale
            cov1: (N, 3) covariance for x1 or None (treat as zero)
            cov2: (M, 3) covariance for x2 or None (treat as zero)
        """
        lengthscale_sq = lengthscale ** 2

        # x1: (N, 2), x2: (M, 2)
        n1, n2 = x1.shape[0], x2.shape[0]

        # mean differences: (N, M, 2)
        mean_diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        dx = mean_diff[..., 0]  # (N, M)
        dy = mean_diff[..., 1]  # (N, M)

        # Extract diagonal variances from covariances (or use zeros if None or shape mismatch)
        if cov1 is None or cov1.shape[0] != n1:
            sigma_x2_1 = torch.zeros(n1, device=x1.device, dtype=x1.dtype)
            sigma_y2_1 = torch.zeros(n1, device=x1.device, dtype=x1.dtype)
        else:
            sigma_x2_1 = cov1[:, 0]  # (N,)
            sigma_y2_1 = cov1[:, 2]  # (N,)

        if cov2 is None or cov2.shape[0] != n2:
            sigma_x2_2 = torch.zeros(n2, device=x2.device, dtype=x2.dtype)
            sigma_y2_2 = torch.zeros(n2, device=x2.device, dtype=x2.dtype)
        else:
            sigma_x2_2 = cov2[:, 0]  # (M,)
            sigma_y2_2 = cov2[:, 2]  # (M,)

        # Pairwise sums: (N, M)
        sigma_x2_sum = sigma_x2_1.unsqueeze(1) + sigma_x2_2.unsqueeze(0)
        sigma_y2_sum = sigma_y2_1.unsqueeze(1) + sigma_y2_2.unsqueeze(0)

        # A matrices: l² + Σ_sum
        Ax = lengthscale_sq + sigma_x2_sum
        Ay = lengthscale_sq + sigma_y2_sum

        # Normalization factor: l² / sqrt(Ax * Ay)
        norm = lengthscale_sq / torch.sqrt(Ax * Ay)

        # Quadratic form: dx²/Ax + dy²/Ay
        quad_form = (dx ** 2 / Ax) + (dy ** 2 / Ay)

        # Girard kernel: norm * exp(-0.5 * quad_form)
        kernel_vals = norm * torch.exp(-0.5 * quad_form)

        return kernel_vals


class GirardUncertainInputModel(gpytorch.models.ExactGP):
    """Girard uncertain-input GP with expected RBF kernel

    RBF-only. Training uses deterministic inputs.
    Kernel analytically handles input uncertainty via explicit covariances.

    Covariance handling: Caller must set kernel.cov1 and kernel.cov2 before forward() (Pattern A).
    - During training: set cov1=train_cov, cov2=train_cov
    - During prediction: set cov1=None, cov2=train_cov
    """

    def __init__(self, train_x, train_y, train_cov, likelihood, kernel_type='rbf'):
        """
        args:
            train_x: (N, 2) training positions
            train_y: (N,) training outputs
            train_cov: (N, 3) position covariance [cov_xx, cov_xy, cov_yy]
            likelihood: Gaussian likelihood
            kernel_type: must be 'rbf' (other kernels not supported)
        """
        if kernel_type != 'rbf':
            raise ValueError(f"Girard uncertain-input GP only supports RBF kernel, got {kernel_type}")

        super(GirardUncertainInputModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # expected RBF kernel (covariances set explicitly by caller)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            ExpectedRBFKernel()
        )
        self.train_cov = train_cov

    def forward(self, x):
        """
        Forward pass. Caller must set kernel.cov1 and kernel.cov2 before calling.

        Example:
            model.covar_module.base_kernel.cov1 = train_cov
            model.covar_module.base_kernel.cov2 = train_cov
            output = model(train_x)  # Training

            model.covar_module.base_kernel.cov1 = None
            model.covar_module.base_kernel.cov2 = train_cov
            output = model(test_x)  # Prediction
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_girard_gp(model, likelihood, train_x, train_y, n_iter=100):
    """train Girard GP with explicit covariance setting (Pattern A)

    args:
        model: GirardUncertainInputModel instance
        likelihood: GaussianLikelihood
        train_x: (N, 2) training inputs
        train_y: (N,) training outputs
        n_iter: optimization iterations

    returns:
        hyperparams: dict of learned hyperparameters
    """
    print(f"\ntraining Girard GP (expected RBF kernel with explicit covariances, {n_iter} iterations)...")

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Set covariances for training (both train_x)
    model.covar_module.base_kernel.cov1 = model.train_cov
    model.covar_module.base_kernel.cov2 = model.train_cov

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

    # Clear covariances after training (hygiene measure to prevent state leakage)
    model.covar_module.base_kernel.cov1 = None
    model.covar_module.base_kernel.cov2 = None

    return hyperparams


def predict_girard_uncertain_input(model, likelihood, test_x):
    """predict with Girard uncertain-input GP (explicit covariances, Pattern A)

    Expected RBF kernel incorporates input uncertainty from training data.
    Prediction at deterministic test grid (no test-time covariance).

    args:
        model: trained GirardUncertainInputModel
        likelihood: trained likelihood
        test_x: (M, 2) test inputs (deterministic evaluation grid)

    returns:
        mean: (M,) predictive mean
        variance: (M,) predictive variance
    """
    model.eval()
    likelihood.eval()

    print("predicting (analytic expected RBF kernel with explicit covariances)...")

    # Set covariances for prediction: test has no covariance, key is training data
    model.covar_module.base_kernel.cov1 = None  # test points: no covariance
    model.covar_module.base_kernel.cov2 = model.train_cov  # key points: training covariance

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # get distribution
        dist = model(test_x)
        pred = likelihood(dist)

        # mean and variance
        mean = pred.mean
        variance = pred.variance

    # Clear covariances after prediction (hygiene measure to prevent state leakage)
    model.covar_module.base_kernel.cov1 = None
    model.covar_module.base_kernel.cov2 = None

    return mean, variance


def visualize_girard_prediction(X, Y, ground_truth, pred_mean, pred_variance,
                                 samples_x, samples_y, kernel, field_type,
                                 trial_number, output_dir, standard_gp_pred_mean=None,
                                 standard_gp_pred_variance=None):
    """create visualization for Girard uncertain-input prediction with Standard GP comparison

    args:
        X, Y: meshgrid coordinates
        ground_truth: (H, W) ground truth field
        pred_mean: (H, W) predicted mean (Girard)
        pred_variance: (H, W) predicted variance (Girard)
        samples_x, samples_y: sample locations
        kernel: kernel type
        field_type: field type
        trial_number: trial number
        output_dir: output directory
        standard_gp_pred_mean: (H, W) Standard GP predicted mean (optional)
        standard_gp_pred_variance: (H, W) Standard GP predicted variance (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    vmin, vmax = ground_truth.min(), ground_truth.max()

    # ground truth
    im0 = axes[0, 0].contourf(X, Y, ground_truth, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].scatter(samples_x, samples_y, c='cyan', s=10, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0, 0].set_title('ground truth')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    plt.colorbar(im0, ax=axes[0, 0], label='temperature [°C]')

    # Standard GP predicted mean (if available)
    if standard_gp_pred_mean is not None:
        im1 = axes[0, 1].contourf(X, Y, standard_gp_pred_mean, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Standard GP predicted mean')
        axes[0, 1].set_xlabel('x [m]')
        axes[0, 1].set_ylabel('y [m]')
        plt.colorbar(im1, ax=axes[0, 1], label='temperature [°C]')
    else:
        axes[0, 1].text(0.5, 0.5, 'Standard GP\n(not provided)',
                       ha='center', va='center', transform=axes[0, 1].transAxes,
                       fontsize=12, color='gray')
        axes[0, 1].set_title('Standard GP predicted mean')
        axes[0, 1].axis('off')

    # Girard predicted mean
    im2 = axes[0, 2].contourf(X, Y, pred_mean, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Girard predicted mean')
    axes[0, 2].set_xlabel('x [m]')
    axes[0, 2].set_ylabel('y [m]')
    plt.colorbar(im2, ax=axes[0, 2], label='temperature [°C]')

    # Girard error
    error = pred_mean - ground_truth
    error_max = max(abs(error.min()), abs(error.max()))
    im3 = axes[1, 0].contourf(X, Y, error, levels=20, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    axes[1, 0].set_title('Girard error (prediction - truth)')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('y [m]')
    plt.colorbar(im3, ax=axes[1, 0], label='error [°C]')

    # Girard predicted variance
    im4 = axes[1, 1].contourf(X, Y, pred_variance, levels=20, cmap='viridis')
    axes[1, 1].set_title('Girard predictive variance')
    axes[1, 1].set_xlabel('x [m]')
    axes[1, 1].set_ylabel('y [m]')
    plt.colorbar(im4, ax=axes[1, 1], label='variance [°C²]')

    # Right bottom panel (spacer or Standard GP variance if available)
    if standard_gp_pred_variance is not None:
        im5 = axes[1, 2].contourf(X, Y, standard_gp_pred_variance, levels=20, cmap='viridis')
        axes[1, 2].set_title('Standard GP predictive variance')
        axes[1, 2].set_xlabel('x [m]')
        axes[1, 2].set_ylabel('y [m]')
        plt.colorbar(im5, ax=axes[1, 2], label='variance [°C²]')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = output_dir / f'{field_type}_{kernel}_girard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {output_path}")


def save_girard_results(metrics, hyperparams, field_type, kernel, trial_number, output_dir):
    """save Girard method metrics and hyperparameters

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
        f.write(f'{field_type},{trial_number},{kernel},girard,'
                f'{metrics["mse"]},{metrics["rmse"]},{metrics["mae"]},{metrics["nrmse"]}\n')

    # save hyperparameters JSON
    hyperparams_path = output_dir / f'{field_type}_{kernel}_hyperparams.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"saved: {metrics_path}")
    print(f"saved: {hyperparams_path}")
