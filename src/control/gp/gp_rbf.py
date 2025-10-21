#!/usr/bin/env python3
"""GP Field Reconstruction with Optimized RBF Kernel"""

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_field_colormap(field_name):
    """Return matching colormap for each field"""
    
    if field_name == 'radial':
        # Blue -> Cyan -> Green -> Yellow -> Red (matches original jet)
        return 'jet'
    
    elif field_name == 'x_compress':
        # Purple/blue gradient
        colors = [[0.5 + i*0.5, i*0.5, 1.0 - i*0.5] for i in np.linspace(0, 1, 256)]
        return LinearSegmentedColormap.from_list('x_compress', colors)
    
    elif field_name == 'x_compress_tilt':
        # Yellow/green
        colors = [[i, 1.0, 0.0] for i in np.linspace(0, 1, 256)]
        return LinearSegmentedColormap.from_list('x_compress_tilt', colors)
    
    elif field_name == 'y_compress':
        # Cyan/blue
        colors = [[i*0.3, 0.5 + i*0.5, 1.0 - i] for i in np.linspace(0, 1, 256)]
        return LinearSegmentedColormap.from_list('y_compress', colors)
    
    elif field_name == 'y_compress_tilt':
        # Red/orange
        colors = [[1.0, 0.5 + i*0.5, 0.0] for i in np.linspace(0, 1, 256)]
        return LinearSegmentedColormap.from_list('y_compress_tilt', colors)
    
    return 'jet'


def generate_ground_truth(field_name, width=25.0, height=25.0, resolution=1.0):
    """Generate ground truth field"""
    center_x, center_y = 12.5, 12.5
    base_temp = 20.0
    amplitude = 10.0
    
    x = np.arange(0, width, resolution)
    y = np.arange(0, height, resolution)
    X, Y = np.meshgrid(x, y)
    
    if field_name == 'radial':
        sigma = 5.0
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        
    elif field_name == 'x_compress':
        sigma_x, sigma_y = 2.5, 7.0
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + 
                           (Y - center_y)**2 / (2 * sigma_y**2)))
        
    elif field_name == 'x_compress_tilt':
        sigma_x, sigma_y = 2.5, 7.0
        theta = np.pi / 4.0
        X_rot = (X - center_x) * np.cos(theta) + (Y - center_y) * np.sin(theta)
        Y_rot = -(X - center_x) * np.sin(theta) + (Y - center_y) * np.cos(theta)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        
    elif field_name == 'y_compress':
        sigma_x, sigma_y = 7.0, 2.5
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + 
                           (Y - center_y)**2 / (2 * sigma_y**2)))
        
    elif field_name == 'y_compress_tilt':
        sigma_x, sigma_y = 7.0, 2.5
        theta = np.pi / 4.0
        X_rot = (X - center_x) * np.cos(theta) + (Y - center_y) * np.sin(theta)
        Y_rot = -(X - center_x) * np.sin(theta) + (Y - center_y) * np.cos(theta)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
    
    temperature = base_temp + amplitude * gaussian
    return X, Y, temperature


def train_gp(train_x, train_y, train_temp, n_iter=200):
    """Train GP with optimized hyperparameters"""
    print(f"\nTraining GP with hyperparameter optimization...")
    
    train_xy = torch.tensor(np.column_stack([train_x, train_y]), 
                            dtype=torch.float32).to(device)
    train_t = torch.tensor(train_temp, dtype=torch.float32).to(device)
    
    # Initialize with reasonable values
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise = 0.01  # Low noise for clean synthetic data
    
    model = ExactGPModel(train_xy, train_t, likelihood).to(device)
    
    # Initialize lengthscale based on data spread
    data_range = np.sqrt((train_x.max() - train_x.min())**2 + 
                        (train_y.max() - train_y.min())**2)
    model.covar_module.base_kernel.lengthscale = data_range / 4.0
    
    model.train()
    likelihood.train()
    
    # Use LBFGS for better convergence
    optimizer = torch.optim.Adam([
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_xy)
        loss = -mll(output, train_t)
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at iteration {i+1}")
            break
        
        if (i + 1) % 40 == 0:
            print(f"Iter {i+1}/{n_iter} - Loss: {loss.item():.3f} - "
                  f"Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f} - "
                  f"Noise: {likelihood.noise.item():.5f}")
    
    return model, likelihood


def predict_gp(model, likelihood, X_grid, Y_grid):
    """Make predictions on grid"""
    test_xy = torch.tensor(np.column_stack([X_grid.flatten(), Y_grid.flatten()]), 
                          dtype=torch.float32).to(device)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_xy))
        mean = predictions.mean.cpu().numpy()
    
    return mean.reshape(X_grid.shape)


def get_kernel_matrix(model, train_xy):
    """Extract kernel matrix"""
    model.eval()
    with torch.no_grad():
        covar = model.covar_module(train_xy).evaluate()
        K = covar.cpu().numpy()
    
    print("\n" + "-"*60)
    print("KERNEL MATRIX (RBF)")
    print("-"*60)
    print(f"Shape: {K.shape}")
    print(f"Range: [{K.min():.6f}, {K.max():.6f}]")
    print(f"Mean: {K.mean():.6f}")
    print(f"Std: {K.std():.6f}")
    print(f"\nFirst 5x5 block:")
    print(K[:5, :5])
    print("-"*60)
    
    return K


def calculate_metrics(ground_truth, prediction):
    """Calculate MAE and RMSE"""
    mae = np.mean(np.abs(ground_truth - prediction))
    rmse = np.sqrt(np.mean((ground_truth - prediction)**2))
    return mae, rmse


def process_field(field_name, trial_number, workspace_root):
    """Process one field"""
    print("\n" + "-"*70)
    print(f"Processing: {field_name.upper()} - Trial {trial_number}")
    print("-"*70)
    
    csv_path = os.path.join(workspace_root, 'src', 'control', 'data',
                            field_name, f'trial_{trial_number}',
                            f'{field_name}_samples.csv')
    
    df = pd.read_csv(csv_path)
    train_x = df['x'].values
    train_y = df['y'].values
    train_temp = df['temperature'].values
    
    print(f"Samples: {len(df)}")
    print(f"X range: [{train_x.min():.1f}, {train_x.max():.1f}]")
    print(f"Y range: [{train_y.min():.1f}, {train_y.max():.1f}]")
    print(f"Temp range: [{train_temp.min():.2f}, {train_temp.max():.2f}]°C")
    
    X, Y, ground_truth = generate_ground_truth(field_name)
    
    model, likelihood = train_gp(train_x, train_y, train_temp, n_iter=200)
    
    train_xy = torch.tensor(np.column_stack([train_x, train_y]), 
                           dtype=torch.float32).to(device)
    K = get_kernel_matrix(model, train_xy)
    
    gp_prediction = predict_gp(model, likelihood, X, Y)
    
    mae, rmse = calculate_metrics(ground_truth, gp_prediction)
    
    print(f"\nMETRICS:")
    print(f"MAE:  {mae:.4f}°C")
    print(f"RMSE: {rmse:.4f}°C")
    
    # Learned hyperparameters
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    noise = likelihood.noise.item()
    
    print(f"\nLearned Hyperparameters:")
    print(f"Lengthscale: {lengthscale:.3f}")
    print(f"Output scale: {outputscale:.3f}")
    print(f"Noise: {noise:.5f}")
    
    return {
        'field_name': field_name,
        'X': X,
        'Y': Y,
        'ground_truth': ground_truth,
        'gp_prediction': gp_prediction,
        'train_x': train_x,
        'train_y': train_y,
        'mae': mae,
        'rmse': rmse,
        'kernel_matrix': K,
        'lengthscale': lengthscale
    }


def create_comparison_plot(results, trial_number, output_path):
    """Create comparison plot for all fields"""
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle(f'GP Reconstruction - Trial {trial_number} (RBF Kernel)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, result in enumerate(results):
        field_name = result['field_name']
        cmap = get_field_colormap(field_name)
        
        # Ground truth
        ax_gt = axes[idx, 0]
        im_gt = ax_gt.contourf(result['X'], result['Y'], result['ground_truth'], 
                               levels=20, cmap=cmap)
        ax_gt.scatter(result['train_x'], result['train_y'], 
                     c='white', s=5, alpha=0.6, edgecolors='black', linewidths=0.3)
        ax_gt.set_title(f'{field_name} - Ground Truth', fontweight='bold')
        ax_gt.set_xlabel('X (m)')
        ax_gt.set_ylabel('Y (m)')
        plt.colorbar(im_gt, ax=ax_gt, label='°C')
        
        # GP prediction
        ax_gp = axes[idx, 1]
        im_gp = ax_gp.contourf(result['X'], result['Y'], result['gp_prediction'], 
                               levels=20, cmap=cmap)
        ax_gp.scatter(result['train_x'], result['train_y'], 
                     c='white', s=5, alpha=0.6, edgecolors='black', linewidths=0.3)
        ax_gp.set_title(f'{field_name} - GP (l={result["lengthscale"]:.2f})\n'
                       f'MAE={result["mae"]:.3f}°C, RMSE={result["rmse"]:.3f}°C', 
                       fontweight='bold', fontsize=10)
        ax_gp.set_xlabel('X (m)')
        ax_gp.set_ylabel('Y (m)')
        plt.colorbar(im_gp, ax=ax_gp, label='°C')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python3 gp_rbf.py <trial_number>")
        print("Example: python3 gp_rbf.py 1\n")
        return
    
    trial_number = int(sys.argv[1])
    workspace_root = '/home/dreams-lab-u24/workspaces/aquatic-mapping'
    
    fields = ['radial', 'x_compress', 'x_compress_tilt', 'y_compress', 'y_compress_tilt']
    
    results = []
    for field_name in fields:
        result = process_field(field_name, trial_number, workspace_root)
        results.append(result)
    
    output_path = os.path.join(workspace_root, 'src', 'control', 'gp',
                               f'gp_comparison_trial_{trial_number}.png')
    create_comparison_plot(results, trial_number, output_path)
    
    print("\n" + "-"*70)
    print(f"SUMMARY - Trial {trial_number}")
    print("-"*70)
    for result in results:
        print(f"{result['field_name']:<20} MAE={result['mae']:.4f}°C  "
              f"RMSE={result['rmse']:.4f}°C  l={result['lengthscale']:.2f}")
    print("-"*70)


if __name__ == '__main__':
    main()