#!/usr/bin/env python3
"""
McHutchon & Rasmussen (2011) Noisy Input Gaussian Process

Implementation of gradient-based input uncertainty handling.
Treats positional uncertainty as heteroscedastic observation noise.

Reference: McHutchon & Rasmussen, "Gaussian Process Training with Input Noise", NIPS 2011
"""
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class McHutchonNIGPModel(gpytorch.models.ExactGP):
    """McHutchon & Rasmussen noisy-input GP model

    Computes input-induced variance via gradient-based approximation:
        σ²_input,i = ∇μ(x_i)ᵀ Σ_i ∇μ(x_i)

    where ∇μ(x_i) is the gradient of the GP posterior mean w.r.t. input position.
    """

    def __init__(self, train_x, train_y, train_cov, likelihood, kernel_type='rbf'):
        super(McHutchonNIGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'matern15':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == 'matern25':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel_type == 'exponential':
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.train_cov = train_cov  # Nx3: [cov_xx, cov_xy, cov_yy]

        # store input-induced variance (computed during training)
        self.register_buffer('input_induced_variance', torch.zeros(train_x.shape[0]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def compute_input_induced_variance(self):
        """compute σ²_input,i = ∇μ(x_i)ᵀ Σ_i ∇μ(x_i) for each training point

        Gradients are taken w.r.t. the GP posterior mean μ(x).

        returns:
            input_induced_variance: (N,) per-sample variance induced by input noise
        """
        # set to evaluation mode for gradient computation
        self.eval()

        # enable gradient computation for inputs
        x_copy = self.train_inputs[0].clone().detach().requires_grad_(True)

        # compute GP posterior mean
        with gpytorch.settings.fast_pred_var():
            output = self(x_copy)
            mean = output.mean

        input_vars = []

        # compute gradient of posterior mean at each training point
        for i in range(x_copy.shape[0]):
            # gradient of mean[i] w.r.t. all inputs
            grad_outputs = torch.zeros_like(mean)
            grad_outputs[i] = 1.0

            grads = torch.autograd.grad(
                outputs=mean,
                inputs=x_copy,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]

            # gradient vector at point i: ∇μ(x_i)
            g_i = grads[i]  # shape: [2]

            # reconstruct 2x2 covariance matrix Σ_i
            Sigma_i = torch.tensor([
                [self.train_cov[i, 0], self.train_cov[i, 1]],
                [self.train_cov[i, 1], self.train_cov[i, 2]]
            ], device=device)

            # σ²_input,i = ∇μ(x_i)ᵀ Σ_i ∇μ(x_i)
            input_var_i = g_i @ Sigma_i @ g_i
            input_vars.append(input_var_i.item())

        self.input_induced_variance = torch.tensor(input_vars, device=device)

        return self.input_induced_variance


def train_mchutchon_nigp(model, train_x, train_y, n_iter=100, n_em_steps=3):
    """train McHutchon NIGP with EM-style iterations

    E-step: compute input-induced variance via gradients
    M-step: optimize GP with heteroscedastic likelihood

    args:
        model: McHutchonNIGPModel instance
        train_x: (N, 2) training inputs
        train_y: (N,) training outputs
        n_iter: total optimization iterations
        n_em_steps: number of EM alternations

    returns:
        likelihood: trained FixedNoiseGaussianLikelihood
        input_induced_variance: final per-sample input-induced variance
        hyperparams: dict of learned hyperparameters
    """
    print(f"\ntraining McHutchon NIGP with {n_em_steps} EM steps...")

    # initialize with small base noise
    base_noise = 1e-4 * torch.ones(train_x.shape[0], device=device)

    for em_step in range(n_em_steps):
        print(f"\n  EM step {em_step+1}/{n_em_steps}")

        # E-step: compute input-induced variance
        input_var = model.compute_input_induced_variance()
        print(f"    input-induced variance: mean={input_var.mean():.6f}, max={input_var.max():.6f}")

        # effective noise: σ²_eff = σ²_y + σ²_input
        effective_noise = base_noise + input_var

        # create heteroscedastic likelihood with per-sample noise
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=effective_noise,
            learn_additional_noise=True
        ).to(device)

        # M-step: optimize with heteroscedastic noise
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(n_iter // n_em_steps):
            optimizer.zero_grad()

            # standard forward pass - likelihood handles noise
            output = model(train_x)
            loss = -mll(output, train_y)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"    iter {(em_step * n_iter // n_em_steps) + i + 1}/{n_iter}, loss: {loss.item():.3f}")

        # update base noise from learned additional noise
        if hasattr(likelihood, 'second_noise'):
            base_noise = likelihood.second_noise.detach()

    # set to evaluation mode
    model.eval()
    likelihood.eval()

    # final variance computation
    final_input_var = model.compute_input_induced_variance()
    print(f"\nfinal input-induced variance: mean={final_input_var.mean():.6f}, max={final_input_var.max():.6f}")

    # extract hyperparameters
    hyperparams = {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'mean_constant': model.mean_module.constant.item(),
        'learned_noise': likelihood.second_noise.item() if hasattr(likelihood, 'second_noise') else 0.0,
        'input_induced_variance_mean': final_input_var.mean().item(),
        'input_induced_variance_max': final_input_var.max().item(),
    }

    return likelihood, final_input_var, hyperparams


def predict_mchutchon_nigp(model, likelihood, test_x):
    """make predictions with McHutchon NIGP

    args:
        model: trained McHutchonNIGPModel
        likelihood: trained likelihood
        test_x: (M, 2) test inputs

    returns:
        mean: (M,) predicted means
        std: (M,) predicted standard deviations
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean
        std = pred.stddev

    return mean, std


def visualize_mchutchon_nigp(X, Y, ground_truth, prediction, samples_x, samples_y,
                              kernel, field_type, trial_number, output_dir,
                              pred_variance=None, standard_gp_pred_mean=None,
                              standard_gp_pred_variance=None):
    """create visualization for McHutchon NIGP reconstruction with Standard GP comparison

    args:
        X, Y: meshgrid coordinates
        ground_truth: (H, W) ground truth field
        prediction: (H, W) predicted field (McHutchon)
        samples_x, samples_y: sample locations
        kernel: kernel type
        field_type: field type
        trial_number: trial number
        output_dir: output directory
        pred_variance: (H, W) McHutchon predicted variance (optional)
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

    # McHutchon reconstruction
    im2 = axes[0, 2].contourf(X, Y, prediction, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('McHutchon NIGP reconstruction')
    axes[0, 2].set_xlabel('x [m]')
    axes[0, 2].set_ylabel('y [m]')
    plt.colorbar(im2, ax=axes[0, 2], label='temperature [°C]')

    # McHutchon error
    error = prediction - ground_truth
    error_max = max(abs(error.min()), abs(error.max()))
    im3 = axes[1, 0].contourf(X, Y, error, levels=20, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    axes[1, 0].set_title('McHutchon error (prediction - truth)')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('y [m]')
    plt.colorbar(im3, ax=axes[1, 0], label='error [°C]')

    # McHutchon predicted variance (if available)
    if pred_variance is not None:
        im4 = axes[1, 1].contourf(X, Y, pred_variance, levels=20, cmap='viridis')
        axes[1, 1].set_title('McHutchon predictive variance')
        axes[1, 1].set_xlabel('x [m]')
        axes[1, 1].set_ylabel('y [m]')
        plt.colorbar(im4, ax=axes[1, 1], label='variance [°C²]')
    else:
        axes[1, 1].axis('off')

    # Standard GP predicted variance (if available)
    if standard_gp_pred_variance is not None:
        im5 = axes[1, 2].contourf(X, Y, standard_gp_pred_variance, levels=20, cmap='viridis')
        axes[1, 2].set_title('Standard GP predictive variance')
        axes[1, 2].set_xlabel('x [m]')
        axes[1, 2].set_ylabel('y [m]')
        plt.colorbar(im5, ax=axes[1, 2], label='variance [°C²]')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = output_dir / f'{field_type}_{kernel}_mchutchon_nigp.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {output_path}")


def save_mchutchon_nigp_results(metrics, hyperparams, field_type, kernel, trial_number, output_dir):
    """save McHutchon NIGP metrics and hyperparameters

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
        f.write(f'{field_type},{trial_number},{kernel},mchutchon_nigp,'
                f'{metrics["mse"]},{metrics["rmse"]},{metrics["mae"]},{metrics["nrmse"]}\n')

    # save hyperparameters JSON
    hyperparams_path = output_dir / f'{field_type}_{kernel}_hyperparams.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"saved: {metrics_path}")
    print(f"saved: {hyperparams_path}")
