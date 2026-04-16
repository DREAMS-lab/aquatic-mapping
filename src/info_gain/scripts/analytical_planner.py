#!/usr/bin/env python3
"""
Analytical Closed-Form Position-Aware Planner

Mathematical Foundation (Girard 2003, Dallaire et al.):
========================================================

Standard GP prediction at certain x*:
    μ(x*) = k(x*, X)ᵀ K⁻¹ y
    σ²(x*) = k(x*, x*) - k(x*, X)ᵀ K⁻¹ k(X, x*)

Expected GP prediction at uncertain x̃ ~ N(μ, Σ):
    E[μ(x̃)] = E[k(x̃, X)]ᵀ K⁻¹ y
    E[σ²(x̃)] = E[k(x̃, x̃)] - E[k(x̃, X)]ᵀ K⁻¹ E[k(X, x̃)]

Expected RBF Kernel (Girard eq. 10-12, Dallaire eq. 7):
=======================================================
For x̃ ~ N(μ_x, Σ_x) and training point x_i:

    E[k(x̃, x_i)] = (σ_f² / √|I + Λ⁻¹Σ_x|) × exp(-½(μ_x - x_i)ᵀ(Λ + Σ_x)⁻¹(μ_x - x_i))

where:
    Λ = diag(ℓ², ℓ²)  (lengthscale matrix)
    σ_f² = signal variance (=1.0 for our normalized kernel)

Expected self-kernel:
    E[k(x̃, x̃)] = σ_f² = 1.0

Expected Information Gain (Analytical):
========================================
    U(x) ≈ ½ log(1 + E[σ²(x̃)]/σ_n²)

This is an APPROXIMATION of the MC pose-aware planner, not exact:

Approximation 1 — Jensen's inequality:
    ½ log(1 + E[σ²]/σ_n²)  ≥  E[½ log(1 + σ²/σ_n²)]
    The log is outside the expectation, always overestimating.

Approximation 2 — E[k]ᵀK⁻¹E[k] vs E[kᵀK⁻¹k]:
    Uses E[k(x̃,xᵢ)]·E[k(x̃,xⱼ)] instead of the correct
    E[k(x̃,xᵢ)·k(x̃,xⱼ)] (Girard 2003, Eq. 12 cross-kernel term).

Both approximations are small when σ_x/ℓ ≪ 1 (≈0.25 for our params).
Faster than MC (no sampling) and deterministic.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from std_msgs.msg import Float32
import numpy as np
import torch
import time
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull

import sys
script_dir = Path(__file__).parent
install_path = script_dir.parent / 'python3' / 'dist-packages'
if install_path.exists():
    sys.path.insert(0, str(install_path))
else:
    sys.path.insert(0, str(script_dir.parent))

from info_gain.gp_model import GPModel
from info_gain.peak_detection import detect_and_plot_peaks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_ground_truth_field(field_type, width=25.0, height=25.0, resolution=0.5):
    """Generate ground truth temperature field"""
    x_grid = np.arange(0, width + 1e-9, resolution)
    y_grid = np.arange(0, height + 1e-9, resolution)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

    center_x, center_y = width / 2, height / 2
    base_temperature = 20.0
    hotspot_amplitude = 10.0

    if field_type == 'radial':
        sigma = 5.0
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'x_compress':
        sigma_x, sigma_y = 2.5, 7.0
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'y_compress':
        sigma_x, sigma_y = 7.0, 2.5
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'x_compress_tilt':
        sigma_x, sigma_y = 2.5, 7.0
        angle = np.pi / 4
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    elif field_type == 'y_compress_tilt':
        sigma_x, sigma_y = 7.0, 2.5
        angle = np.pi / 4
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return X, Y, field


def expected_rbf_kernel_vectorized(mu_x, Sigma_x, train_x, lengthscale, signal_var=1.0):
    """
    Vectorized computation of expected RBF kernel E[k(x̃, X)] for uncertain input.

    Mathematical Formulation (Girard 2003, Equation 10-12):
    ========================================================
    For x̃ ~ N(μ, Σ) and training points X = {x_1, ..., x_N}:

        E[k(x̃, x_i)] = ∫ k(x̃, x_i) p(x̃) dx̃
                      = ∫ σ_f² exp(-½||x̃ - x_i||²/ℓ²) × (1/√|2πΣ|) exp(-½(x̃-μ)ᵀΣ⁻¹(x̃-μ)) dx̃

    Product of two Gaussians = another Gaussian:

        E[k(x̃, x_i)] = (σ_f² / √|I + Λ⁻¹Σ|) × exp(-½(μ - x_i)ᵀ(Λ + Σ)⁻¹(μ - x_i))

    where:
        Λ = diag(ℓ², ℓ²)  (squared lengthscale matrix)
        Σ = position covariance (2×2)

    Args:
        mu_x: Mean position (N_query, 2) or (2,)
        Sigma_x: Position covariance (2, 2) - assumed same for all queries
        train_x: Training locations (N_train, 2)
        lengthscale: GP lengthscale ℓ
        signal_var: σ_f² (default 1.0)

    Returns:
        E[k(x̃, X)]: Expected kernel matrix (N_query, N_train)
    """
    if mu_x.ndim == 1:
        mu_x = mu_x.reshape(1, -1)

    N_query = mu_x.shape[0]
    N_train = train_x.shape[0]

    # Λ = diag(ℓ², ℓ²)
    Lambda = np.eye(2) * (lengthscale ** 2)

    # A = Λ + Σ (common for all pairs)
    A = Lambda + Sigma_x
    A_inv = np.linalg.inv(A)

    # Normalization: 1 / √|I + Λ⁻¹Σ|
    Lambda_inv = np.eye(2) / (lengthscale ** 2)
    det_term = np.linalg.det(np.eye(2) + Lambda_inv @ Sigma_x)
    norm = 1.0 / np.sqrt(det_term)

    # Compute all pairwise expected kernels
    # E[k(x̃_i, x_j)] for all i,j
    expected_k = np.zeros((N_query, N_train))

    for i in range(N_query):
        for j in range(N_train):
            # Δ = μ - x_train
            delta = mu_x[i] - train_x[j]

            # Mahalanobis: (μ - x_i)ᵀ(Λ + Σ)⁻¹(μ - x_i)
            mahalanobis = delta @ A_inv @ delta

            # E[k(x̃, x_i)] = (σ_f² / √|I + Λ⁻¹Σ|) × exp(-½ Mahalanobis)
            expected_k[i, j] = signal_var * norm * np.exp(-0.5 * mahalanobis)

    return expected_k


def expected_posterior_variance_vectorized(mu_x, Sigma_x, gp, lengthscale, noise_var, signal_var=1.0):
    """
    Analytical expected posterior variance E[σ²(x̃)] for uncertain input.

    Mathematical Formulation (Girard 2003, Section 3):
    ==================================================
    Standard GP variance at certain x*:
        σ²(x*) = k(x*, x*) - k(x*, X)ᵀ (K + σ_n²I)⁻¹ k(X, x*)

    Expected GP variance at uncertain x̃ ~ N(μ, Σ):
        E[σ²(x̃)] = E[k(x̃, x̃)] - E[k(x̃, X)]ᵀ (K + σ_n²I)⁻¹ E[k(X, x̃)]

    where:
        E[k(x̃, x̃)] = σ_f² = 1.0  (expected self-kernel is just signal variance)
        E[k(x̃, X)] = computed via expected_rbf_kernel_vectorized()

    This is the KEY formula for analytical expected info gain!

    Args:
        mu_x: Mean positions (N, 2)
        Sigma_x: Position covariance (2, 2)
        gp: GPModel with current training data
        lengthscale: GP lengthscale
        noise_var: Observation noise σ_n²
        signal_var: Signal variance σ_f²

    Returns:
        E[σ²(x̃)]: Expected posterior variance (N,)
    """
    if mu_x.ndim == 1:
        mu_x = mu_x.reshape(1, -1)

    N = mu_x.shape[0]

    # Get training data
    train_x, _ = gp.get_training_data()

    if train_x is None or len(train_x) == 0:
        # Prior variance (no training data)
        return np.full(N, signal_var)

    N_train = train_x.shape[0]

    # 1. Expected self-kernel: E[k(x̃, x̃)] = σ_f²
    expected_k_self = signal_var

    # 2. Expected cross-kernel: E[k(x̃, X)]  (N, N_train)
    expected_k_cross = expected_rbf_kernel_vectorized(
        mu_x, Sigma_x, train_x, lengthscale, signal_var
    )

    # 3. Kernel matrix K (N_train, N_train)
    # Use standard RBF kernel for training points (no uncertainty)
    K = np.zeros((N_train, N_train))
    for i in range(N_train):
        for j in range(N_train):
            r2 = np.sum((train_x[i] - train_x[j])**2)
            K[i, j] = signal_var * np.exp(-r2 / (2 * lengthscale**2))

    # Add noise: K + σ_n² I
    K_noise = K + noise_var * np.eye(N_train)

    # 4. Inverse: (K + σ_n²I)⁻¹
    try:
        K_inv = np.linalg.inv(K_noise)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K_inv = np.linalg.pinv(K_noise)

    # 5. Expected posterior variance (vectorized over all queries)
    # E[σ²(x̃)] = E[k(x̃, x̃)] - E[k(x̃, X)]ᵀ K⁻¹ E[k(X, x̃)]
    #          = σ_f² - sum_ij E[k(x̃, x_i)] K⁻¹_ij E[k(x̃, x_j)]

    expected_var = np.zeros(N)
    for i in range(N):
        # E[k(x̃, X)]ᵀ K⁻¹ E[k(X, x̃)] = E[k(x̃, X)]ᵀ K⁻¹ E[k(x̃, X)]
        reduction = expected_k_cross[i] @ K_inv @ expected_k_cross[i]
        expected_var[i] = expected_k_self - reduction

        # Ensure non-negative (numerical stability)
        expected_var[i] = max(expected_var[i], 1e-10)

    return expected_var


def analytical_expected_information_gain(candidates, gp, noise_var, position_cov, lengthscale, signal_var=1.0):
    """
    ANALYTICAL expected information gain (CLOSED-FORM, NO MONTE CARLO).

    Mathematical Formulation:
    =========================
    U(x) = E[Δ(x̃)] where x̃ ~ N(x, Σ_x)
         = E[½ log(1 + σ²(x̃)/σ_n²)]
         ≈ ½ log(1 + E[σ²(x̃)]/σ_n²)   [Jensen's inequality approximation]

    This is an APPROXIMATION of the MC pose-aware planner:

    Approximation 1 — Jensen's inequality:
        ½ log(1 + E[σ²]/σ_n²)  ≥  E[½ log(1 + σ²/σ_n²)]
    Approximation 2 — E[k]ᵀK⁻¹E[k] vs E[kᵀK⁻¹k]:
        Missing Girard (2003) Eq. 12 cross-kernel term.

    Both approximations are small when σ_x/ℓ ≪ 1.
    Faster than MC (no sampling) and deterministic.

    Args:
        candidates: (N, 2) array of candidate positions
        gp: GPModel with current observations
        noise_var: σ_n²
        position_cov: Σ_x (2, 2) position uncertainty
        lengthscale: GP lengthscale ℓ
        signal_var: GP signal variance σ_f²

    Returns:
        (N,) array of expected information gains
    """
    # Compute expected posterior variance E[σ²(x̃)] analytically
    expected_var = expected_posterior_variance_vectorized(
        candidates, position_cov, gp, lengthscale, noise_var, signal_var
    )

    # Expected information gain: U(x) = ½ log(1 + E[σ²(x̃)]/σ_n²)
    expected_info = 0.5 * np.log(1 + expected_var / noise_var)

    return expected_info


def travel_cost(x1, x2):
    """Euclidean travel cost"""
    return np.linalg.norm(np.array(x1) - np.array(x2))


class LiveVisualizer:
    """Minimal visualizer for analytical planner"""

    def __init__(self, title="Analytical Planner", output_dir=None):
        self.fig = plt.figure(figsize=(16, 10))
        self.title = title
        self.output_dir = output_dir

        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        self.ax_mean = self.fig.add_subplot(gs[0, 0])
        self.ax_var = self.fig.add_subplot(gs[0, 1])
        self.ax_acq = self.fig.add_subplot(gs[0, 2])
        self.ax_traj = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_cost = self.fig.add_subplot(gs[1, 2])

        self.trajectory = []
        self.info_gains = []
        self.travel_costs = []
        self.steps = []

        self.grid_res = 0.5
        x = np.arange(0, 25 + self.grid_res, self.grid_res)
        y = np.arange(0, 25 + self.grid_res, self.grid_res)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.grid_points = np.column_stack([self.X_grid.ravel(), self.Y_grid.ravel()])

    def update(self, gp, candidates, scores, selected_idx, current_pos, target_pos,
               step, info_gain_val, cumulative_cost):
        """Update visualization"""
        self.trajectory.append(current_pos.copy())
        if info_gain_val > 0:
            self.info_gains.append(info_gain_val)
            self.travel_costs.append(cumulative_cost)
            self.steps.append(step)

        with torch.no_grad():
            grid_t = torch.tensor(self.grid_points, dtype=torch.float32).to(device)
            mean, var = gp.predict(grid_t)
            mean_grid = mean.cpu().numpy().reshape(self.X_grid.shape)
            var_grid = var.cpu().numpy().reshape(self.X_grid.shape)

        for ax in [self.ax_mean, self.ax_var, self.ax_acq, self.ax_traj, self.ax_info, self.ax_cost]:
            ax.clear()

        # GP Mean
        self.ax_mean.pcolormesh(self.X_grid, self.Y_grid, mean_grid, cmap='coolwarm', shading='auto')
        self.ax_mean.set_title('GP Mean')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_mean.plot(traj[:, 0], traj[:, 1], 'k.-', linewidth=1)
        self.ax_mean.scatter(current_pos[0], current_pos[1], c='lime', s=100, marker='o', edgecolors='black')

        # GP Variance
        self.ax_var.pcolormesh(self.X_grid, self.Y_grid, var_grid, cmap='viridis', shading='auto')
        self.ax_var.set_title('GP Variance')

        # Acquisition
        self.ax_acq.scatter(candidates[:, 0], candidates[:, 1], c=scores, cmap='hot', s=30, alpha=0.7)
        if selected_idx is not None:
            sel = candidates[selected_idx]
            self.ax_acq.scatter(sel[0], sel[1], c='cyan', s=200, marker='X', edgecolors='black', linewidths=2)
        self.ax_acq.set_title(f'Analytical Acquisition (Step {step})')
        self.ax_acq.set_xlim(0, 25)
        self.ax_acq.set_ylim(0, 25)

        # Trajectory
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
            for i, pt in enumerate(traj):
                self.ax_traj.scatter(pt[0], pt[1], c='blue', s=50)
                self.ax_traj.annotate(str(i+1), (pt[0], pt[1]), xytext=(5, 5), textcoords='offset points')
        self.ax_traj.scatter(target_pos[0], target_pos[1], c='red', s=150, marker='X', edgecolors='black')
        self.ax_traj.set_title('Trajectory')
        self.ax_traj.set_xlim(0, 25)
        self.ax_traj.set_ylim(0, 25)
        self.ax_traj.grid(True, alpha=0.3)

        # Info Gain
        if len(self.info_gains) > 0:
            self.ax_info.plot(self.steps, self.info_gains, 'g-o', linewidth=2)
            self.ax_info.fill_between(self.steps, self.info_gains, alpha=0.3, color='green')
        self.ax_info.set_title('Info Gain (Analytical)')
        self.ax_info.set_xlabel('Step')
        self.ax_info.grid(True)

        # Travel Cost
        if len(self.travel_costs) > 0:
            self.ax_cost.plot(self.steps, self.travel_costs, 'r-o', linewidth=2)
        self.ax_cost.set_title('Travel Cost')
        self.ax_cost.set_xlabel('Step')
        self.ax_cost.grid(True)

        self.fig.suptitle(f'{self.title} - Sample {step}/100', fontsize=14, fontweight='bold')

        if self.output_dir:
            progress_path = self.output_dir / 'figures' / 'progress.png'
            self.fig.savefig(progress_path, dpi=100, bbox_inches='tight')

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)


class AnalyticalPlanner(Node):
    """Analytical closed-form position-aware planner"""

    MAX_SAMPLES = 100

    def __init__(self):
        super().__init__('analytical_planner')

        # Parameters
        self.declare_parameter('field_type', 'radial')
        self.declare_parameter('trial', -1)
        self.declare_parameter('noise_var', 0.36)
        self.declare_parameter('lengthscale', 2.0)
        self.declare_parameter('lambda_cost', 0.1)
        self.declare_parameter('candidate_resolution', 1.0)
        self.declare_parameter('position_std', 0.5)
        self.declare_parameter('optimize_every', 10)      # Optimize hyperparams every N obs (0=disabled)
        self.declare_parameter('optimize_steps', 20)      # Gradient steps per optimization

        self.field_type = self.get_parameter('field_type').value
        self.trial_num = self.get_parameter('trial').value
        self.noise_var = self.get_parameter('noise_var').value
        self.lengthscale = self.get_parameter('lengthscale').value
        self.lambda_cost = self.get_parameter('lambda_cost').value
        self.candidate_res = self.get_parameter('candidate_resolution').value
        self.position_std = self.get_parameter('position_std').value
        self.optimize_every = self.get_parameter('optimize_every').value
        self.optimize_steps = self.get_parameter('optimize_steps').value

        self.position_cov = np.array([[self.position_std**2, 0],
                                       [0, self.position_std**2]])

        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.info_pub = self.create_publisher(Float32, '/info_gain/current', 10)
        self.cost_pub = self.create_publisher(Float32, '/info_gain/cumulative_cost', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, qos_profile)
        self.temp_sub = self.create_subscription(Float32, f'/gaussian_field/{self.field_type}/temperature_noisy', self.temp_callback, 10)

        # State
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_variance = np.array([self.position_std**2, self.position_std**2])
        self.current_temp = None
        self.counter = 0
        self.state = 'INIT'
        self.current_target = np.array([0.0, 0.0, 0.0])
        self.sample_count = 0
        self.waiting_for_observation = False
        self.last_command_time = None
        self.total_travel_cost = 0.0
        self.last_position = None
        self.cumulative_info_gain = 0.0
        self.stop_reason = None

        # Initial waypoints
        self.initial_waypoints = [
            np.array([5.0, 5.0, 0.0]),
            np.array([10.0, 5.0, 0.0]),
            np.array([10.0, 10.0, 0.0]),
        ]
        self.waypoint_idx = 0

        # GP model
        self.gp = GPModel(lengthscale=self.lengthscale, noise_var=self.noise_var,
                          optimize_every=self.optimize_every, optimize_steps=self.optimize_steps)

        # Candidates
        self.candidates = self._generate_candidate_grid(0.5, 24.5, 0.5, 24.5, self.candidate_res)

        # Output
        self.output_dir = self._create_trial_directory()
        self.gt_X, self.gt_Y, self.gt_field = generate_ground_truth_field(self.field_type)
        self._save_ground_truth()

        self.samples = []
        self.decisions = []
        self.samples_file = self.output_dir / 'samples.csv'
        self._init_samples_csv()

        # Viz
        try:
            self.viz = LiveVisualizer(
                title=f'Analytical - {self.field_type} (CLOSED-FORM)',
                output_dir=self.output_dir
            )
        except Exception as e:
            self.get_logger().error(f'Viz failed: {e}')
            self.viz = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self._save_config()

        self.get_logger().info('='*60)
        self.get_logger().info('ANALYTICAL CLOSED-FORM PLANNER')
        self.get_logger().info('  Method: Girard/Dallaire analytical expected variance')
        self.get_logger().info('  U(x) ≈ ½ log(1 + E[σ²(x̃)]/σ_n²) - Girard closed-form, no MC')
        self.get_logger().info(f'  Field: {self.field_type}')
        self.get_logger().info(f'  Trial: {self.trial_num}')
        self.get_logger().info('='*60)

    def _generate_candidate_grid(self, x_min, x_max, y_min, y_max, resolution):
        x = np.arange(x_min, x_max + 1e-9, resolution)
        y = np.arange(y_min, y_max + 1e-9, resolution)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _create_trial_directory(self):
        workspace_root = Path.cwd()
        base_dir = workspace_root / 'data' / 'trials' / 'analytical' / self.field_type
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.trial_num >= 0:
            trial_num = self.trial_num
        else:
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
            trial_num = max([int(d.name.split('_')[1]) for d in existing], default=0) + 1

        trial_dir = base_dir / f'trial_{trial_num:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / 'figures').mkdir(exist_ok=True)
        self.trial_num = trial_num
        return trial_dir

    def _save_config(self):
        config = {
            'method': 'analytical',
            'description': 'Closed-form analytical expected info (Girard/Dallaire)',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'noise_var': self.noise_var,
            'lengthscale': self.lengthscale,
            'lambda_cost': self.lambda_cost,
            'position_std_fallback': self.position_std,
            'use_px4_ekf_variance': True,
            'max_samples': self.MAX_SAMPLES,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def _save_ground_truth(self):
        gt_file = self.output_dir / 'ground_truth.npz'
        np.savez(gt_file, X=self.gt_X, Y=self.gt_Y, field=self.gt_field)

    def _init_samples_csv(self):
        self._csv_fieldnames = ['step', 'phase', 'x', 'y', 'temp', 'info_gain', 'cumulative_info',
                                'travel_cost', 'gp_n_obs', 'pos_var_x', 'pos_var_y', 'pos_std_x', 'pos_std_y',
                                'ls_optimized', 'learned_lengthscale', 'learned_signal_var', 'learned_mean']
        with open(self.samples_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()

    def _write_sample(self, sample_dict):
        try:
            with open(self.samples_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writerow(sample_dict)
        except Exception as e:
            self.get_logger().warn(f'CSV write failed: {e}')

    def _compute_reconstruction_metrics(self):
        grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
        grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

        with torch.no_grad():
            gp_mean, gp_var = self.gp.predict(grid_t)
            gp_mean = gp_mean.cpu().numpy().reshape(self.gt_X.shape)
            gp_var = gp_var.cpu().numpy().reshape(self.gt_X.shape)

        error = gp_mean - self.gt_field
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        max_error = float(np.max(np.abs(error)))

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'mean_variance': float(np.mean(gp_var)),
            'n_observations': self.gp.n_observations
        }

        with open(self.output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        np.savez(self.output_dir / 'gp_reconstruction.npz',
                 X=self.gt_X, Y=self.gt_Y,
                 mean=gp_mean, variance=gp_var, error=error)

        # Create comparison figure
        self._plot_reconstruction_comparison(gp_mean, gp_var, error)

        self.get_logger().info(f'Reconstruction metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, Max={max_error:.3f}')
        return metrics

    def _plot_reconstruction_comparison(self, gp_mean, gp_var, error):
        """Create 2x3 comparison plot matching other planners' layout."""
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        samples_arr = None
        if len(self.samples) > 0:
            samples_arr = np.array([[s['x'], s['y']] for s in self.samples])

        # [0,0] Ground truth
        im0 = axes[0, 0].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto')
        axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0], label='T [°C]')
        if samples_arr is not None:
            axes[0, 0].scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=30, marker='x', linewidths=2)

        # [0,1] GP Reconstruction
        im1 = axes[0, 1].pcolormesh(self.gt_X, self.gt_Y, gp_mean, cmap='coolwarm', shading='auto')
        axes[0, 1].set_title(f'GP Reconstruction (n={self.gp.n_observations})', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1], label='T [°C]')

        # [0,2] Absolute Error
        im2 = axes[0, 2].pcolormesh(self.gt_X, self.gt_Y, np.abs(error), cmap='hot', shading='auto')
        axes[0, 2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 2], label='|Error| [°C]')
        axes[0, 2].text(0.02, 0.98, f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C',
                       transform=axes[0, 2].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # [1,0] GP Variance
        im3 = axes[1, 0].pcolormesh(self.gt_X, self.gt_Y, gp_var, cmap='viridis', shading='auto')
        axes[1, 0].set_title(f'GP Variance', fontsize=12, fontweight='bold')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0], label='Variance')

        # [1,1] Sample trajectory
        if samples_arr is not None:
            axes[1, 1].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto', alpha=0.3)
            axes[1, 1].plot(samples_arr[:, 0], samples_arr[:, 1], 'b-', alpha=0.4, linewidth=1)
            sc = axes[1, 1].scatter(samples_arr[:, 0], samples_arr[:, 1],
                                    c=np.arange(len(samples_arr)), cmap='viridis',
                                    s=30, edgecolors='black', linewidths=0.5, zorder=5)
            plt.colorbar(sc, ax=axes[1, 1], label='Sample order')
        axes[1, 1].set_title('Sample Trajectory', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim(0, 25)
        axes[1, 1].set_ylim(0, 25)
        axes[1, 1].set_aspect('equal')

        # [1,2] Convex hull
        if samples_arr is not None and len(samples_arr) >= 3:
            try:
                hull = ConvexHull(samples_arr)
                hull_area = hull.volume
                coverage = (hull_area / 625.0) * 100

                axes[1, 2].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto', alpha=0.6)
                axes[1, 2].scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=50, marker='o', edgecolors='white', linewidths=2)
                for simplex in hull.simplices:
                    axes[1, 2].plot(samples_arr[simplex, 0], samples_arr[simplex, 1], 'r-', linewidth=2)
                hull_points = samples_arr[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                axes[1, 2].fill(hull_points[:, 0], hull_points[:, 1], color='yellow', alpha=0.3)
                axes[1, 2].set_title(f'Coverage {coverage:.1f}%', fontsize=12, fontweight='bold')
            except Exception:
                axes[1, 2].set_title('Coverage', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlim(0, 25)
        axes[1, 2].set_ylim(0, 25)
        axes[1, 2].set_aspect('equal')

        plt.suptitle(f'Analytical - {self.field_type} (Trial {self.trial_num}) - RMSE: {rmse:.3f}°C',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def odometry_callback(self, msg):
        # Position: NED frame used directly (no ENU conversion)
        self.current_position = np.array([msg.position[0], msg.position[1], msg.position[2]])

        # Position variance: NED frame [σ²_north, σ²_east] — same axis order as position
        self.position_variance = np.array([msg.position_variance[0], msg.position_variance[1]])

        # Build position covariance matrix for Girard expected kernel computation
        # Σ_x = diag(σ²_north, σ²_east) - PX4 EKF only provides diagonal variances
        self.position_cov = np.array([
            [self.position_variance[0], 0],
            [0, self.position_variance[1]]
        ])

    def temp_callback(self, msg):
        self.current_temp = msg.data

    def control_loop(self):
        self.publish_offboard_control()
        self.publish_setpoint()

        if self.state == 'INIT':
            self.counter += 1
            if self.counter >= 10:
                self.arm()
                self.engage_offboard()
                self.state = 'ARM'
                self.last_command_time = time.time()
                self._arm_retry_time = time.time()

        elif self.state == 'ARM':
            if time.time() - self._arm_retry_time >= 0.5:
                self.arm()
                self.engage_offboard()
                self._arm_retry_time = time.time()
            if time.time() - self.last_command_time >= 3.0:
                self.state = 'INITIAL_SAMPLING'
                self.current_target = self.initial_waypoints[0]
                self.last_position = self.current_position[:2].copy()
                self._stuck_check_time = time.time()

        elif self.state == 'INITIAL_SAMPLING':
            self._run_initial_sampling()

        elif self.state == 'ADAPTIVE_SAMPLING':
            self._run_adaptive_sampling()

    def _run_initial_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        if hasattr(self, '_stuck_check_time') and np.linalg.norm(self.current_position[:2]) < 1.0:
            if time.time() - self._stuck_check_time > 15.0:
                self.arm()
                self.engage_offboard()
                self._stuck_check_time = time.time()
        else:
            self._stuck_check_time = time.time()

        if dist < 0.5 and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            if self.last_position is not None:
                self.total_travel_cost += travel_cost(self.last_position, x)
            self.last_position = x.copy()

            self.gp.add_observation(x, y)
            self.sample_count += 1

            sample = {
                'step': self.sample_count,
                'phase': 'initial',
                'x': float(x[0]),
                'y': float(x[1]),
                'temp': float(y),
                'info_gain': 0.0,
                'cumulative_info': 0.0,
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1])
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.initial_waypoints):
                self.state = 'ADAPTIVE_SAMPLING'
                self.get_logger().info('Starting ANALYTICAL adaptive sampling')
                self._plan_next_sample()
            else:
                self.current_target = self.initial_waypoints[self.waypoint_idx]

    def _run_adaptive_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        if self.waiting_for_observation and dist < 0.5 and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            _, var_at_x = self.gp.predict(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
            realized_info = float(0.5 * np.log(1 + var_at_x.item() / self.noise_var))
            self.cumulative_info_gain += realized_info

            step_cost = travel_cost(self.last_position, x)
            self.total_travel_cost += step_cost
            self.last_position = x.copy()

            self.gp.add_observation(x, y)
            self.sample_count += 1

            # Optimize hyperparameters if scheduled
            optimized = False
            if self.gp.should_optimize():
                self.gp.optimize_hyperparameters(logger=self.get_logger())
                optimized = True

            sample = {
                'step': self.sample_count,
                'phase': 'adaptive',
                'x': float(x[0]),
                'y': float(x[1]),
                'temp': float(y),
                'info_gain': realized_info,
                'cumulative_info': float(self.cumulative_info_gain),
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1])),
                'ls_optimized': optimized,
                'learned_lengthscale': self.gp._learned_lengthscale,
                'learned_signal_var': self.gp._learned_signal_var,
                'learned_mean': self.gp._learned_mean
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.info_pub.publish(Float32(data=float(realized_info)))
            self.cost_pub.publish(Float32(data=float(self.total_travel_cost)))

            self.get_logger().info(
                f'Sample {self.sample_count}/{self.MAX_SAMPLES}: '
                f'({x[0]:.1f}, {x[1]:.1f}), ANALYTICAL info={realized_info:.4f}'
            )

            self.waiting_for_observation = False

            if self.sample_count >= self.MAX_SAMPLES:
                self.stop_reason = f'max_samples_reached ({self.MAX_SAMPLES})'
                self._finish_mission()
                return

            self._plan_next_sample()

    def _plan_next_sample(self):
        current_pos = self.current_position[:2].copy()

        best_idx, best_score, best_info, all_scores = self._greedy_single_step(current_pos)

        if best_idx is None:
            self.stop_reason = 'no_feasible_candidate'
            self._finish_mission()
            return

        x_next = self.candidates[best_idx]
        self.current_target = np.array([x_next[0], x_next[1], 0.0])
        self.waiting_for_observation = True

        # Log decision
        scores_np = np.array(all_scores) if not isinstance(all_scores, np.ndarray) else all_scores
        top5_idx = np.argsort(scores_np)[-5:][::-1]
        top5_pos = self.candidates[top5_idx]
        top5_scores = scores_np[top5_idx]
        decision = {
            'step': self.sample_count + 1,
            'current_x': float(current_pos[0]),
            'current_y': float(current_pos[1]),
            'selected_x': float(x_next[0]),
            'selected_y': float(x_next[1]),
            'selected_score': float(best_score),
            'selected_info': float(best_info),
            'travel_to_next': float(travel_cost(current_pos, x_next)),
            'lambda': self.lambda_cost,
            'n_candidates': len(self.candidates),
            'scores_min': float(scores_np.min()),
            'scores_max': float(scores_np.max()),
            'scores_mean': float(scores_np.mean()),
            'scores_std': float(scores_np.std()),
            'top5_x': top5_pos[:, 0].tolist(),
            'top5_y': top5_pos[:, 1].tolist(),
            'top5_scores': top5_scores.tolist(),
            'gp_n_obs': self.gp.n_observations
        }
        self.decisions.append(decision)

        # Update viz
        if self.viz is not None:
            try:
                self.viz.update(
                    gp=self.gp,
                    candidates=self.candidates,
                    scores=all_scores,
                    selected_idx=best_idx,
                    current_pos=current_pos,
                    target_pos=x_next,
                    step=self.sample_count + 1,
                    info_gain_val=best_info,
                    cumulative_cost=self.total_travel_cost
                )
            except Exception as e:
                self.get_logger().warn(f'Viz update failed: {e}')

        self.get_logger().info(
            f'ANALYTICAL planned: ({x_next[0]:.1f}, {x_next[1]:.1f}), score={best_score:.4f}'
        )

    def _run_final_hotspot_analysis(self):
        """Run full Kac-Rice peak detection on the final GP posterior."""
        try:
            grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

            with torch.no_grad():
                mu, var = self.gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(self.gt_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(self.gt_X.shape))

            samples_arr = None
            if self.samples:
                samples_arr = np.array([[s['x'], s['y']] for s in self.samples])

            peaks = detect_and_plot_peaks(
                gp_wrapper=self.gp,
                lat_mesh=self.gt_X, lon_mesh=self.gt_Y,
                mu_orig=mu_grid, std_grid=std_grid,
                X_scaled=samples_arr,
                variable_name='Temperature',
                out_dir=self.output_dir,
                kernel_type='rbf',
                y_mean=0.0,
                gp_label='Analytical GP',
            )

            n_sig = sum(1 for p in peaks if p.get('significant', False))
            self.get_logger().info(
                f'Final hotspot analysis: {len(peaks)} candidates, {n_sig} significant'
            )
        except Exception as e:
            self.get_logger().warn(f'Final hotspot analysis failed: {e}')

        self._run_ground_truth_hotspot_analysis()

    def _run_ground_truth_hotspot_analysis(self):
        """Fit GP to dense ground truth samples, run Kac-Rice for validation."""
        try:
            import shutil
            from info_gain.gp_model import GPModel as StationaryGPModel

            step = 3
            gt_sub_X = self.gt_X[::step, ::step]
            gt_sub_Y = self.gt_Y[::step, ::step]
            gt_sub_field = self.gt_field[::step, ::step]

            train_x = np.column_stack([gt_sub_X.ravel(), gt_sub_Y.ravel()])
            train_y = gt_sub_field.ravel()

            gt_gp = StationaryGPModel(noise_var=0.001, lengthscale=self.lengthscale)
            train_x_t = torch.tensor(train_x, dtype=torch.float32).to(device)
            train_y_t = torch.tensor(train_y, dtype=torch.float32).to(device)
            gt_gp.fit(train_x_t, train_y_t)

            grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, var = gt_gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(self.gt_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(self.gt_X.shape))

            gt_dir = self.output_dir / '_gt_hotspot_tmp'
            gt_dir.mkdir(exist_ok=True)

            peaks = detect_and_plot_peaks(
                gp_wrapper=gt_gp,
                lat_mesh=self.gt_X, lon_mesh=self.gt_Y,
                mu_orig=mu_grid, std_grid=std_grid,
                X_scaled=train_x,
                variable_name='Temperature (Ground Truth)',
                out_dir=gt_dir,
                kernel_type='rbf',
                y_mean=0.0,
                gp_label='Ground Truth GP',
            )

            for src_name, dst_name in [
                ('hotspots.png', 'ground_truth_hotspots.png'),
                ('peaks.csv', 'ground_truth_peaks.csv'),
            ]:
                src = gt_dir / src_name
                if src.exists():
                    shutil.move(str(src), str(self.output_dir / dst_name))

            shutil.rmtree(str(gt_dir), ignore_errors=True)

            n_sig = sum(1 for p in peaks if p.get('significant', False))
            self.get_logger().info(
                f'Ground truth hotspot analysis: {len(peaks)} candidates, {n_sig} significant'
            )
        except Exception as e:
            self.get_logger().warn(f'Ground truth hotspot analysis failed: {e}')

    def _greedy_single_step(self, current_pos):
        """
        Greedy single-step planning (H=1) with analytical expected variance.

        Mathematical Formulation:
        -------------------------
        Select next location to maximize:
            score(x) = E[Δ(x̃)] - λ * c(x_curr, x)

        Where:
            E[Δ(x̃)] = ½ log(1 + E[σ²(x̃)]/σ_n²)  [analytical expected info gain]
            E[σ²(x̃)] computed via Girard expected RBF kernel
            c(x_curr, x) = ||x_curr - x||₂  [travel cost]

        Args:
            current_pos: Current robot position (2,)

        Returns:
            best_idx: Index of best candidate
            best_score: Acquisition score
            best_info: Expected info gain
            scores: All candidate scores
        """
        # ANALYTICAL expected info gain (CLOSED-FORM via Girard!)
        # Use GP's current learned hyperparameters (updated by optimization)
        current_lengthscale = self.gp._learned_lengthscale
        current_signal_var = self.gp._learned_signal_var
        expected_info_gains = analytical_expected_information_gain(
            self.candidates, self.gp, self.noise_var,
            self.position_cov, current_lengthscale, current_signal_var
        )

        # Travel costs
        travel_costs = np.array([travel_cost(current_pos, x) for x in self.candidates])

        # Acquisition scores
        scores = expected_info_gains - self.lambda_cost * travel_costs

        # Best candidate
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_info = float(expected_info_gains[best_idx])

        return best_idx, best_score, best_info, scores

    def _finish_mission(self):
        self.state = 'DONE'

        self.get_logger().info('Computing reconstruction metrics...')
        reconstruction_metrics = self._compute_reconstruction_metrics()

        # Run final Kac-Rice hotspot analysis (post-mission, on final GP)
        self.get_logger().info('Running final hotspot analysis...')
        self._run_final_hotspot_analysis()

        # Save final visualization
        if self.viz is not None:
            self.viz.save(self.output_dir / 'figures' / 'final.png')

        # Save samples CSV
        with open(self.output_dir / 'samples.csv', 'w', newline='') as f:
            if self.samples:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.samples)

        # Save decisions CSV
        with open(self.output_dir / 'decisions.csv', 'w', newline='') as f:
            if self.decisions:
                flat_decisions = []
                for d in self.decisions:
                    flat = {k: v for k, v in d.items() if not isinstance(v, list)}
                    flat['top5_x'] = str(d.get('top5_x', []))
                    flat['top5_y'] = str(d.get('top5_y', []))
                    flat['top5_scores'] = str(d.get('top5_scores', []))
                    flat_decisions.append(flat)
                writer = csv.DictWriter(f, fieldnames=flat_decisions[0].keys())
                writer.writeheader()
                writer.writerows(flat_decisions)

        # Save decisions JSON
        with open(self.output_dir / 'decisions.json', 'w') as f:
            json.dump(self.decisions, f, indent=2)

        # Save summary
        summary = {
            'method': 'analytical',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'lambda_cost': self.lambda_cost,
            'total_samples': self.sample_count,
            'total_travel_cost': float(self.total_travel_cost),
            'cumulative_info_gain': float(self.cumulative_info_gain),
            'reconstruction_rmse': reconstruction_metrics['rmse'],
            'reconstruction_mae': reconstruction_metrics['mae'],
            'reconstruction_max_error': reconstruction_metrics['max_error'],
            'mean_gp_variance': reconstruction_metrics['mean_variance'],
            'stop_reason': self.stop_reason,
            'completed_at': datetime.now().isoformat()
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info('='*60)
        self.get_logger().info('ANALYTICAL PLANNER COMPLETE')
        self.get_logger().info(f'  Samples: {self.sample_count}')
        self.get_logger().info(f'  Travel: {self.total_travel_cost:.1f}m')
        self.get_logger().info(f'  Info gain: {self.cumulative_info_gain:.4f}')
        self.get_logger().info(f'  RMSE: {reconstruction_metrics["rmse"]:.3f}°C')
        self.get_logger().info(f'  MAE: {reconstruction_metrics["mae"]:.3f}°C')
        self.get_logger().info(f'  Data: {self.output_dir}')
        self.get_logger().info('='*60)

    def publish_offboard_control(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [float(self.current_target[0]), float(self.current_target[1]), float(self.current_target[2])]
        msg.velocity = [float('nan')] * 3
        msg.yaw = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def arm(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def engage_offboard(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)


def main(args=None):
    rclpy.init(args=args)
    node = AnalyticalPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        if node.viz is not None:
            node.viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
