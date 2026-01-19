
#!/usr/bin/env python3
"""
run_three_gp_recon.py

Build THREE Gaussian Process reconstructions from the SAME sampling run, matching your thesis math:

Case A: no_noise
    Σx = 0,  σy ≈ 0,  t = f(x_c)
Case B: measurement_only
    Σx = 0,  σy learned,  t = f(x_c) + εy
Case C: measurement_plus_position
    Σx > 0, σy learned,  x ~ N(x_c, Σx),  t = f(x) + εy
    (training-time uncertain inputs; reconstruction grid is treated as deterministic world coordinates)

Outputs (PNG + CSV + XLSX) are written under:
  <results_dir>/{original_fields,no_noise,measurement_only,measurement_plus_position}/<field_name>/

Author: ChatGPT
"""
import os
import math
import json
import argparse
import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Fields: data columns + ground-truth definitions (matches your ROS generators)
# -----------------------------
BASE_T = 20.0
AMP_T  = 10.0
CENTER_X = 12.5
CENTER_Y = 12.5
THETA = math.pi / 4.0

FIELD_SPECS = [
    # base_name, clean_col, noisy_col, (sigma_x, sigma_y, theta, radial_flag)
    ("radial",           "radial",           "radial_noisy",           (5.0, 5.0, 0.0, True)),
    ("x_compress",       "x_compress",       "x_compress_noisy",       (2.5, 7.0, 0.0, False)),
    ("y_compress",       "y_compress",       "y_compress_noisy",       (7.0, 2.5, 0.0, False)),
    ("x_compress_tilt",  "x_compress_tilt",  "x_compress_tilt_noisy",  (2.5, 7.0, THETA, False)),
    ("y_compress_tilt",  "y_compress_tilt",  "y_compress_tilt_noisy",  (7.0, 2.5, THETA, False)),
]

# -----------------------------
# Kernels
# -----------------------------
JITTER = 1e-6

def rbf_kernel(X: np.ndarray, Z: np.ndarray, ell: float, sigma_f: float) -> np.ndarray:
    """Standard isotropic RBF kernel k(x,z)=σ_f^2 exp(-||x-z||^2/(2ℓ^2))."""
    diff = X[:, None, :] - Z[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return (sigma_f**2) * np.exp(-0.5 * sqdist / (ell**2))

def expected_rbf_kernel(
    Xmu: np.ndarray,
    Zmu: np.ndarray,
    ell: float,
    sigma_f: float,
    Sigma_delta: np.ndarray,
) -> np.ndarray:
    """
    Expected RBF kernel for uncertain inputs:
      x ~ N(μx, Σx), z ~ N(μz, Σz)
      ΣΔ = Σx + Σz  (constant here)

    E[k(x,z)] = σ_f^2 |I + ΣΔ Λ^{-1}|^{-1/2} exp( -1/2 (μx-μz)^T (Λ+ΣΔ)^{-1} (μx-μz) )

    Λ = ℓ^2 I
    """
    Lambda = (ell**2) * np.eye(2)
    A = np.eye(2) + Sigma_delta @ np.linalg.inv(Lambda)
    scale = float(np.linalg.det(A) ** (-0.5))
    Qinv = np.linalg.inv(Lambda + Sigma_delta)

    diff = Xmu[:, None, :] - Zmu[None, :, :]
    quad = np.einsum("nmi,ij,nmj->nm", diff, Qinv, diff)
    return (sigma_f**2) * scale * np.exp(-0.5 * quad)

# -----------------------------
# GP utilities
# -----------------------------
def cholesky_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    Y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, Y)

def log_marginal_likelihood(K: np.ndarray, y: np.ndarray) -> float:
    N = y.shape[0]
    L = np.linalg.cholesky(K)
    alpha = cholesky_solve(L, y)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * y.T @ alpha - 0.5 * logdet - 0.5 * N * np.log(2.0 * np.pi))

def grid_search_hypers(
    K_base_fn,
    X_train: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    sigma_fs: np.ndarray,
    sigma_ys: np.ndarray,
):
    """
    Deterministic grid search for (ℓ, σ_f, σ_y) maximizing log marginal likelihood.
    K_base_fn(ell, sigma_f) must return the noise-free covariance matrix (N,N).
    """
    y_mean = float(np.mean(y))
    y0 = y - y_mean

    best = None
    for ell in lengths:
        for sigma_f in sigma_fs:
            K0 = K_base_fn(ell, sigma_f)
            for sigma_y in sigma_ys:
                K = K0 + (sigma_y**2 + JITTER) * np.eye(X_train.shape[0])
                try:
                    ll = log_marginal_likelihood(K, y0)
                except np.linalg.LinAlgError:
                    continue
                cand = (ll, float(ell), float(sigma_f), float(sigma_y), y_mean)
                if best is None or cand[0] > best[0]:
                    best = cand
    if best is None:
        raise RuntimeError("Hyperparameter search failed: Cholesky never succeeded.")
    return best  # (ll, ell, sigma_f, sigma_y, y_mean)

def gp_predict(
    K_train_fn,
    K_cross_fn,
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    ell: float,
    sigma_f: float,
    sigma_y: float,
):
    """
    Returns (mu_t, var_f, var_t) at test points.
    y is raw targets; we internally center by y_mean for stable LML and then add back.
    """
    y_mean = float(np.mean(y))
    y0 = y - y_mean

    K = K_train_fn(ell, sigma_f)
    C = K + (sigma_y**2 + JITTER) * np.eye(X_train.shape[0])

    L = np.linalg.cholesky(C)
    alpha = cholesky_solve(L, y0)

    Ks = K_cross_fn(ell, sigma_f)  # (M,N)
    mu_f = Ks @ alpha

    # Deterministic test points => k(x*,x*) = σ_f^2
    v = np.linalg.solve(L, Ks.T)  # (N,M)
    var_f = (sigma_f**2) - np.sum(v**2, axis=0)
    var_f = np.maximum(var_f, 0.0)
    var_t = var_f + sigma_y**2

    mu_t = mu_f + y_mean
    return mu_t, var_f, var_t

# -----------------------------
# Ground truth fields (analytic, matches generator params)
# -----------------------------
def truth_field(field_name: str, X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]

    for name, _, __, params in FIELD_SPECS:
        if name == field_name:
            sx, sy, theta, is_radial = params
            if is_radial:
                sigma = sx
                g = np.exp(-(((x - CENTER_X)**2 + (y - CENTER_Y)**2) / (2.0 * sigma**2)))
                return BASE_T + AMP_T * g

            # anisotropic, possibly rotated
            if abs(theta) > 1e-12:
                Xc = x - CENTER_X
                Yc = y - CENTER_Y
                Xr = Xc * math.cos(theta) + Yc * math.sin(theta)
                Yr = -Xc * math.sin(theta) + Yc * math.cos(theta)
                g = np.exp(-( (Xr**2)/(2.0*sx**2) + (Yr**2)/(2.0*sy**2) ))
                return BASE_T + AMP_T * g
            else:
                g = np.exp(-( ((x - CENTER_X)**2)/(2.0*sx**2) + ((y - CENTER_Y)**2)/(2.0*sy**2) ))
                return BASE_T + AMP_T * g

    raise KeyError(f"Unknown field_name: {field_name}")

# -----------------------------
# Metrics + plots
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    bias = float(np.mean(err))
    return {"MAE": mae, "RMSE": rmse, "MaxAbs": max_abs, "Bias": bias}

def save_heatmap_png(
    Xtest: np.ndarray,
    values: np.ndarray,
    title: str,
    out_png: str,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
):
    x_unique = np.unique(Xtest[:, 0])
    y_unique = np.unique(Xtest[:, 1])
    nx = len(x_unique)
    ny = len(y_unique)
    Z = values.reshape(ny, nx)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        Z,
        origin="lower",
        extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    plt.colorbar(label=title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def write_case_readme(case_dir: str, case_label: str, Sigma_x: np.ndarray, sigma_y_desc: str):
    txt = f"""Case: {case_label}

Math (as in your thesis):
  - Commanded input: x_c
  - Measurement model: t = f(x) + ε_y
  - Measurement noise: ε_y ~ N(0, σ_y^2)
  - Positional belief (if enabled): x ~ N(x_c, Σ_x)

This run uses:
  Σ_x =
{Sigma_x}
  σ_y: {sigma_y_desc}

Interpretation:
  The samples are the same across all cases.
  Only the GP assumptions (Σ_x and σ_y) change.
"""
    with open(os.path.join(case_dir, "ASSUMPTIONS.txt"), "w") as f:
        f.write(txt)

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Path to samples_clean.csv")
    ap.add_argument("--noisy", required=True, help="Path to samples_noisy.csv")
    ap.add_argument("--results", required=True, help="Results directory (will be created)")
    ap.add_argument("--sigma_input", default="1.0",
                    help="Input-position std in meters for Σx = diag([s^2,s^2]). Use 'auto' to estimate from (xhat-x_c,yhat-y_c).")
    ap.add_argument("--grid_step", type=float, default=0.5, help="Grid step in meters (default 0.5)")
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=25.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=25.0)
    args = ap.parse_args()

    df_clean = pd.read_csv(args.clean)
    df_noisy = pd.read_csv(args.noisy)

    # Inputs (use commanded locations consistently)
    X_train = df_noisy[["x_c", "y_c"]].to_numpy(float)

    # Σx for positional-uncertainty case
    if str(args.sigma_input).lower() == "auto":
        dx = (df_noisy["xhat"] - df_noisy["x_c"]).to_numpy(float)
        dy = (df_noisy["yhat"] - df_noisy["y_c"]).to_numpy(float)
        sx = float(np.std(dx, ddof=1)) if len(dx) > 1 else 1.0
        sy = float(np.std(dy, ddof=1)) if len(dy) > 1 else 1.0
        Sigma_x = np.diag([sx**2, sy**2])
        sigma_input_desc = f"auto (std dx={sx:.3f}m, dy={sy:.3f}m)"
    else:
        s = float(args.sigma_input)
        Sigma_x = np.diag([s**2, s**2])
        sigma_input_desc = f"{s:.3f}m isotropic"

    Sigma_0 = np.zeros((2, 2))

    # Grid
    xs = np.arange(args.x_min, args.x_max + 1e-9, args.grid_step)
    ys = np.arange(args.y_min, args.y_max + 1e-9, args.grid_step)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)

    os.makedirs(args.results, exist_ok=True)

    # Save grid
    pd.DataFrame({"x": X_test[:, 0], "y": X_test[:, 1]}).to_csv(os.path.join(args.results, "grid.csv"), index=False)

    # Hyperparameter grids
    LENGTHSCALES = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    SIGMA_FS     = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    SIGMA_YS     = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

    # Define the three cases
    CASES = [
        {
            "name": "no_noise",
            "use_df": df_clean,
            "col_key": "clean",
            "Sigma_train": Sigma_0,
            "sigma_y_mode": "fixed",
            "sigma_y_fixed": 1e-6,
        },
        {
            "name": "measurement_only",
            "use_df": df_noisy,
            "col_key": "noisy",
            "Sigma_train": Sigma_0,
            "sigma_y_mode": "learn",
            "sigma_y_fixed": None,
        },
        {
            "name": "measurement_plus_position",
            "use_df": df_noisy,
            "col_key": "noisy",
            "Sigma_train": Sigma_x,
            "sigma_y_mode": "learn",
            "sigma_y_fixed": None,
        },
    ]

    # Save ground-truth maps once (for comparison)
    gt_root = os.path.join(args.results, "original_fields")
    os.makedirs(gt_root, exist_ok=True)
    for field_name, clean_col, noisy_col, _ in FIELD_SPECS:
        gt_dir = os.path.join(gt_root, field_name)
        os.makedirs(gt_dir, exist_ok=True)
        y_true = truth_field(field_name, X_test)
        save_heatmap_png(X_test, y_true, f"{field_name} – ground truth", os.path.join(gt_dir, "truth.png"))
        pd.DataFrame({"x": X_test[:, 0], "y": X_test[:, 1], "truth": y_true}).to_csv(os.path.join(gt_dir, "truth.csv"), index=False)

    # Run all cases & collect metrics
    metrics_rows = []
    hyper_rows = []

    for case in CASES:
        case_root = os.path.join(args.results, case["name"])
        os.makedirs(case_root, exist_ok=True)

        Sigma_train = case["Sigma_train"]
        case_label = case["name"]
        sigma_y_desc = "≈0 (fixed 1e-6)" if case["sigma_y_mode"] == "fixed" else "learned by LML grid search"
        if case["name"] == "measurement_plus_position":
            sigma_y_desc += f"; Σx uses {sigma_input_desc}"

        write_case_readme(case_root, case_label, Sigma_train, sigma_y_desc)

        # Build kernel factory functions for this case
        def K_train_factory(ell, sigma_f):
            if np.allclose(Sigma_train, 0.0):
                return rbf_kernel(X_train, X_train, ell, sigma_f)
            # training-training: ΣΔ = Σx + Σx = 2Σx
            return expected_rbf_kernel(X_train, X_train, ell, sigma_f, Sigma_delta=(2.0 * Sigma_train))

        def K_cross_factory(X_test_local):
            # cross: test deterministic => ΣΔ = Σ_test + Σ_train = 0 + Σ_train
            def _K(ell, sigma_f):
                if np.allclose(Sigma_train, 0.0):
                    return rbf_kernel(X_test_local, X_train, ell, sigma_f)
                return expected_rbf_kernel(X_test_local, X_train, ell, sigma_f, Sigma_delta=Sigma_train)
            return _K

        for field_name, clean_col, noisy_col, _params in FIELD_SPECS:
            df_use = case["use_df"]
            y_col = clean_col if case["col_key"] == "clean" else noisy_col
            y = df_use[y_col].to_numpy(float)

            out_dir = os.path.join(case_root, field_name)
            os.makedirs(out_dir, exist_ok=True)

            # Hyperparams
            if case["sigma_y_mode"] == "fixed":
                # grid search only over ell and sigma_f, with sigma_y fixed
                sigma_y = float(case["sigma_y_fixed"])

                best = None
                y_mean = float(np.mean(y))
                y0 = y - y_mean
                for ell in LENGTHSCALES:
                    for sigma_f in SIGMA_FS:
                        K0 = K_train_factory(ell, sigma_f)
                        K = K0 + (sigma_y**2 + JITTER) * np.eye(X_train.shape[0])
                        try:
                            ll = log_marginal_likelihood(K, y0)
                        except np.linalg.LinAlgError:
                            continue
                        cand = (ll, float(ell), float(sigma_f), sigma_y, y_mean)
                        if best is None or cand[0] > best[0]:
                            best = cand
                if best is None:
                    raise RuntimeError(f"{case_label}/{field_name}: hyperparam search failed (noise-free).")
                ll, ell, sigma_f, sigma_y, y_mean = best
            else:
                ll, ell, sigma_f, sigma_y, y_mean = grid_search_hypers(
                    K_base_fn=lambda e, sf: K_train_factory(e, sf),
                    X_train=X_train,
                    y=y,
                    lengths=LENGTHSCALES,
                    sigma_fs=SIGMA_FS,
                    sigma_ys=SIGMA_YS,
                )

            # Predict
            mu_t, var_f, var_t = gp_predict(
                K_train_fn=lambda e, sf: K_train_factory(e, sf),
                K_cross_fn=lambda e, sf: K_cross_factory(X_test)(e, sf),
                X_train=X_train,
                y=y,
                X_test=X_test,
                ell=ell,
                sigma_f=sigma_f,
                sigma_y=sigma_y,
            )

            # Ground truth and metrics on the reconstruction grid
            y_true = truth_field(field_name, X_test)
            mets = compute_metrics(y_true, mu_t)

            # Save outputs
            pred_csv = os.path.join(out_dir, "pred.csv")
            pd.DataFrame({
                "x": X_test[:, 0],
                "y": X_test[:, 1],
                "truth": y_true,
                "mean": mu_t,
                "var_f": var_f,
                "var_t": var_t,
                "err": (mu_t - y_true),
                "abs_err": np.abs(mu_t - y_true),
            }).to_csv(pred_csv, index=False)

            # Plots
            save_heatmap_png(X_test, mu_t, f"{field_name} – GP mean ({case_label})", os.path.join(out_dir, "mean.png"))
            save_heatmap_png(X_test, var_f, f"{field_name} – GP var_f ({case_label})", os.path.join(out_dir, "var_f.png"), cmap="magma")
            save_heatmap_png(X_test, np.abs(mu_t - y_true), f"{field_name} – |error| ({case_label})", os.path.join(out_dir, "abs_error.png"), cmap="magma")

            # Save metrics + hyperparams
            row = {
                "case": case_label,
                "field": field_name,
                "MAE": mets["MAE"],
                "RMSE": mets["RMSE"],
                "MaxAbs": mets["MaxAbs"],
                "Bias": mets["Bias"],
                "ell": ell,
                "sigma_f": sigma_f,
                "sigma_y": sigma_y,
                "log_marginal_likelihood": ll,
                "Sigma_x_desc": sigma_input_desc if case_label == "measurement_plus_position" else "none",
            }
            metrics_rows.append(row)
            hyper_rows.append({k: row[k] for k in ["case","field","ell","sigma_f","sigma_y","log_marginal_likelihood","Sigma_x_desc"]})

            # Per-field metrics text file (easy to read in the folder)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(row, f, indent=2)

            print(f"[{case_label:24s}] {field_name:16s}  RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}  Max={row['MaxAbs']:.4f}  (ell={ell:.2f}, sf={sigma_f:.2f}, sy={sigma_y:.2f})")

    # Save summary tables
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["field","case"])
    metrics_csv = os.path.join(args.results, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Excel summary
    xlsx_path = os.path.join(args.results, "metrics_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        metrics_df.to_excel(xw, sheet_name="metrics", index=False)
        pd.DataFrame(hyper_rows).to_excel(xw, sheet_name="hyperparams", index=False)

    print("\nSaved:")
    print(f"  {metrics_csv}")
    print(f"  {xlsx_path}")
    print(f"  Grid: {os.path.join(args.results, 'grid.csv')}")
    print(f"  Ground truth: {gt_root}")

if __name__ == "__main__":
    main()
