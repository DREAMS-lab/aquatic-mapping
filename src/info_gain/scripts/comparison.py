#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Ground truth parameters (must match radial_field.py) ---
WIDTH, HEIGHT, RES = 25.0, 25.0, 1.0
CENTER_X, CENTER_Y = 12.5, 12.5
BASE_TEMP, AMPLITUDE, SIGMA = 20.0, 10.0, 5.0


def generate_true_field():
    xs = np.arange(0, WIDTH + 1e-9, RES)
    ys = np.arange(0, HEIGHT + 1e-9, RES)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    gaussian = np.exp(-((X - CENTER_X)**2 + (Y - CENTER_Y)**2) / (2 * SIGMA**2))
    field = BASE_TEMP + AMPLITUDE * gaussian
    return X, Y, field


def compare_fields(planner_num: str):
    base = Path.home() / "workspaces" / "aquatic-mapping" / "src" / "info_gain" / "scripts" / planner_num
    recon_path = base / "reconstructed_field.npz"

    if not recon_path.exists():
        print(f"❌ File not found: {recon_path}")
        return

    # --- Load reconstructed GP field ---
    data = np.load(recon_path)
    mu = data["mu"]

    # --- Generate ground truth field ---
    _, _, true_field = generate_true_field()

    if mu.shape != true_field.shape:
        raise ValueError(f"Shape mismatch: reconstructed {mu.shape}, true {true_field.shape}")

    # --- Metrics ---
    diff = mu - true_field
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    corr = np.corrcoef(mu.ravel(), true_field.ravel())[0, 1]

    print(f"\n📁 {recon_path}")
    print(f"RMSE: {rmse:.3f} °C")
    print(f"MAE : {mae:.3f} °C")
    print(f"Corr: {corr:.3f}")

    # --- Shared color limits for both plots ---
    vmin = min(true_field.min(), mu.min())
    vmax = max(true_field.max(), mu.max())

    # --- Dark style ---
    plt.style.use("dark_background")

    # --- Plot both fields with one shared colorbar ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax in axs:
        ax.set_facecolor("black")
        ax.grid(alpha=0.15, color="white")

    im = axs[0].imshow(true_field, origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
    axs[0].set_title("True Field", color='white', fontsize=11)
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")

    axs[1].imshow(mu, origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
    axs[1].set_title("Reconstructed GP Mean", color='white', fontsize=11)
    axs[1].set_xlabel("X [m]")
    axs[1].set_ylabel("Y [m]")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.035, pad=0.04)
    cbar.set_label("Temperature [°C]", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    fig.suptitle(f"Planner {planner_num}  |  RMSE={rmse:.3f}°C  |  MAE={mae:.3f}°C  |  Corr={corr:.3f}",
                 fontsize=12, color='orange')
    plt.show()

    # --- Scatter plot ---
    plt.figure(figsize=(5, 5), facecolor='black')
    plt.scatter(true_field.ravel(), mu.ravel(),
                s=12, alpha=0.6, color='cyan', edgecolors='none')
    plt.plot([true_field.min(), true_field.max()],
             [true_field.min(), true_field.max()],
             'r--', lw=1.3, label='Ideal y=x')
    plt.xlabel("True Temperature [°C]", color='white')
    plt.ylabel("Reconstructed Mean [°C]", color='white')
    plt.title(f"True vs Reconstructed (r = {corr:.3f})",
              fontsize=11, color='orange')
    plt.grid(True, alpha=0.3, color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()


def main():
    planner = input("planner: ").strip()
    if not planner.isdigit():
        print("Please enter a valid planner number (e.g., 1, 2, 3).")
        return
    compare_fields(planner)


if __name__ == "__main__":
    main()
