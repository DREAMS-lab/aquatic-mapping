"""data loading utilities for GP reconstruction"""
import numpy as np
import pandas as pd


def load_sample_data(csv_path):
    """load sampling data from CSV

    returns:
        x: (N,) x positions [m]
        y: (N,) y positions [m]
        temperature: (N,) temperature measurements [°C]
        position_cov: (N, 3) covariance [cov_xx, cov_xy, cov_yy]
    """
    df = pd.read_csv(csv_path)

    x = df['x'].values
    y = df['y'].values
    temperature = df['temperature'].values

    if 'cov_xx' in df.columns:
        cov_xx = df['cov_xx'].values
        cov_xy = df['cov_xy'].values
        cov_yy = df['cov_yy'].values
    else:
        cov_xx = df['pos_var_x'].values
        cov_xy = np.zeros_like(cov_xx)
        cov_yy = df['pos_var_y'].values

    position_cov = np.stack([cov_xx, cov_xy, cov_yy], axis=1)

    print(f"loaded {len(x)} samples")
    print(f"position covariance range: cov_xx=[{cov_xx.min():.6f}, {cov_xx.max():.6f}]")
    print(f"                          cov_yy=[{cov_yy.min():.6f}, {cov_yy.max():.6f}]")
    print(f"mean positional uncertainty: σ_x={np.sqrt(cov_xx.mean()):.4f}m, σ_y={np.sqrt(cov_yy.mean()):.4f}m")

    return x, y, temperature, position_cov
