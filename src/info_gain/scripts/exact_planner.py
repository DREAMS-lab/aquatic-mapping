#!/usr/bin/env python3
"""
Baseline Informative Path Planner (Code A) - Exact Position Case

Mathematical formulation (no positional uncertainty):

    psi* = argmax_psi  SUM_{x in S(psi)} (1/2) * log(1 + sigma^2_{S_<x}(x) / sigma_n^2)

At each step k, solve the greedy optimization:
    x*_{k+1} = argmax  Delta(x_{k+1} | S_k) - lambda * c(x_k, x_{k+1})

Where:
    - Delta(x | S) = (1/2) * log(1 + sigma^2_S(x) / sigma_n^2)  [information gain]
    - c(x_t, x_{t+1}) = ||x_t - x_{t+1}||_2  [Euclidean travel cost]
    - lambda: trade-off parameter

Execute x*_{k+1}, take measurement, update GP, repeat.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np
import torch
import time
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use Agg backend - file only, no display blocking
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull

# Add info_gain module to path
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
    """Generate ground truth temperature field (same as field generators)"""
    x_grid = np.arange(0, width + 1e-9, resolution)
    y_grid = np.arange(0, height + 1e-9, resolution)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

    center_x, center_y = width / 2, height / 2
    base_temperature = 20.0
    hotspot_amplitude = 10.0  # Matches ROS2 field generators

    if field_type == 'radial':
        sigma = 5.0
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        field = base_temperature + (hotspot_amplitude * gaussian)

    elif field_type == 'x_compress':
        sigma_x, sigma_y = 2.5, 7.0  # Matches ROS2 field generators
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)

    elif field_type == 'y_compress':
        sigma_x, sigma_y = 7.0, 2.5  # Matches ROS2 field generators
        gaussian = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)

    elif field_type == 'x_compress_tilt':
        sigma_x, sigma_y = 2.5, 7.0  # Matches ROS2 field generators
        angle = np.pi / 4  # 45 degrees, matches ROS2 field generators
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)

    elif field_type == 'y_compress_tilt':
        sigma_x, sigma_y = 7.0, 2.5  # Matches ROS2 field generators
        angle = np.pi / 4  # 45 degrees, matches ROS2 field generators
        X_rot = (X - center_x) * np.cos(angle) + (Y - center_y) * np.sin(angle)
        Y_rot = -(X - center_x) * np.sin(angle) + (Y - center_y) * np.cos(angle)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        field = base_temperature + (hotspot_amplitude * gaussian)

    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return X, Y, field


def information_gain(variance, noise_var):
    """
    Information gain (mutual information) from sampling at a location.

    Mathematical formulation:
    -------------------------
    Δ(x | S) = I(f(x); y | D_S)
             = H[f(x) | D_S] - H[f(x) | D_S, y]
             = (1/2) * log(1 + σ²_S(x) / σ_n²)

    where:
        - σ²_S(x) = GP posterior variance at x given data S
        - σ_n²    = observation noise variance
        - The formula comes from differential entropy of Gaussians

    Intuition: High variance → high info gain (we learn more by sampling there)

    Note: In the EXACT planner, we assume we sample exactly at commanded x.
          No averaging over position uncertainty.

    Args:
        variance: σ²_S(x), GP posterior variance at candidate location(s)
        noise_var: σ_n², observation noise variance

    Returns:
        Information gain in nats (natural log units)
    """
    if isinstance(variance, torch.Tensor):
        return 0.5 * torch.log(1 + variance / noise_var)
    else:
        return 0.5 * np.log(1 + variance / noise_var)


def travel_cost(x1, x2):
    """c(x1, x2) = ||x1 - x2||_2"""
    if isinstance(x1, torch.Tensor):
        return torch.norm(x1 - x2)
    else:
        return np.linalg.norm(np.array(x1) - np.array(x2))


class LiveVisualizer:
    """Visualization - saves to file immediately after each update (Agg backend)"""

    def __init__(self, title="Baseline Sampler", output_dir=None):
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

        # Data storage for plots
        self.trajectory = []
        self.info_gains = []
        self.travel_costs = []
        self.steps = []

        # Grid for GP visualization
        self.grid_res = 0.5
        x = np.arange(0, 25 + self.grid_res, self.grid_res)
        y = np.arange(0, 25 + self.grid_res, self.grid_res)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.grid_points = np.column_stack([self.X_grid.ravel(), self.Y_grid.ravel()])

    def update(self, gp, candidates, scores, selected_idx, current_pos, target_pos,
               step, info_gain_val, cumulative_cost):
        """Update all visualization panels"""

        # Store trajectory
        self.trajectory.append(current_pos.copy())
        if info_gain_val > 0:
            self.info_gains.append(info_gain_val)
            self.travel_costs.append(cumulative_cost)
            self.steps.append(step)

        # Get GP predictions
        with torch.no_grad():
            grid_t = torch.tensor(self.grid_points, dtype=torch.float32).to(device)
            mean, var = gp.predict(grid_t)
            mean_grid = mean.cpu().numpy().reshape(self.X_grid.shape)
            var_grid = var.cpu().numpy().reshape(self.X_grid.shape)

        # Reshape scores to grid if possible
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

        # Clear all axes
        for ax in [self.ax_mean, self.ax_var, self.ax_acq, self.ax_traj, self.ax_info, self.ax_cost]:
            ax.clear()

        # 1. GP Mean
        im1 = self.ax_mean.pcolormesh(self.X_grid, self.Y_grid, mean_grid, cmap='coolwarm', shading='auto')
        self.ax_mean.set_title('GP Mean (Reconstruction)')
        self.ax_mean.set_xlabel('X [m]')
        self.ax_mean.set_ylabel('Y [m]')
        self.ax_mean.set_aspect('equal')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_mean.plot(traj[:, 0], traj[:, 1], 'k.-', linewidth=1, markersize=4)
        self.ax_mean.scatter(current_pos[0], current_pos[1], c='lime', s=100, marker='o', edgecolors='black', zorder=10)

        # 2. GP Variance
        im2 = self.ax_var.pcolormesh(self.X_grid, self.Y_grid, var_grid, cmap='viridis', shading='auto')
        self.ax_var.set_title('GP Variance (Uncertainty)')
        self.ax_var.set_xlabel('X [m]')
        self.ax_var.set_ylabel('Y [m]')
        self.ax_var.set_aspect('equal')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_var.scatter(traj[:, 0], traj[:, 1], c='white', s=20, edgecolors='black', zorder=5)

        # 3. Acquisition Function
        self.ax_acq.scatter(candidates[:, 0], candidates[:, 1], c=scores_np, cmap='hot', s=30, alpha=0.7)
        if selected_idx is not None:
            sel = candidates[selected_idx]
            self.ax_acq.scatter(sel[0], sel[1], c='cyan', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)
            self.ax_acq.annotate(f'Next\n({sel[0]:.1f},{sel[1]:.1f})', (sel[0], sel[1]),
                                textcoords="offset points", xytext=(10, 10), fontsize=8,
                                bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
        self.ax_acq.scatter(current_pos[0], current_pos[1], c='lime', s=100, marker='o', edgecolors='black', zorder=10)
        self.ax_acq.set_title(f'Acquisition (Step {step})')
        self.ax_acq.set_xlabel('X [m]')
        self.ax_acq.set_ylabel('Y [m]')
        self.ax_acq.set_xlim(0, 25)
        self.ax_acq.set_ylim(0, 25)
        self.ax_acq.set_aspect('equal')

        # 4. Trajectory with numbers
        self.ax_traj.set_title('Trajectory')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
            for i, pt in enumerate(traj):
                self.ax_traj.scatter(pt[0], pt[1], c='blue', s=50, zorder=5)
                self.ax_traj.annotate(str(i+1), (pt[0], pt[1]), textcoords="offset points",
                                     xytext=(5, 5), fontsize=7)
        self.ax_traj.scatter(target_pos[0], target_pos[1], c='red', s=150, marker='X', edgecolors='black', zorder=10, label='Target')
        self.ax_traj.set_xlabel('X [m]')
        self.ax_traj.set_ylabel('Y [m]')
        self.ax_traj.set_xlim(0, 25)
        self.ax_traj.set_ylim(0, 25)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.legend(loc='upper right')

        # 5. Information Gain over time
        if len(self.info_gains) > 0:
            self.ax_info.plot(self.steps, self.info_gains, 'g-o', linewidth=2, markersize=4)
            self.ax_info.fill_between(self.steps, self.info_gains, alpha=0.3, color='green')
        self.ax_info.set_title('Information Gain per Step')
        self.ax_info.set_xlabel('Step')
        self.ax_info.set_ylabel('Info Gain (nats)')
        self.ax_info.grid(True, alpha=0.3)

        # 6. Cumulative Travel Cost
        if len(self.travel_costs) > 0:
            self.ax_cost.plot(self.steps, self.travel_costs, 'r-o', linewidth=2, markersize=4)
            self.ax_cost.fill_between(self.steps, self.travel_costs, alpha=0.3, color='red')
        self.ax_cost.set_title('Cumulative Travel Cost')
        self.ax_cost.set_xlabel('Step')
        self.ax_cost.set_ylabel('Distance (m)')
        self.ax_cost.grid(True, alpha=0.3)

        # Update title with current step
        self.fig.suptitle(f'{self.title} - Sample {step}/100', fontsize=14, fontweight='bold')

        # Save to file (Agg backend)
        if self.output_dir:
            progress_path = self.output_dir / 'figures' / 'progress.png'
            self.fig.savefig(progress_path, dpi=100, bbox_inches='tight')

    def save(self, path):
        """Save current figure"""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)


class BaselineSampler(Node):
    """Baseline informative path planner with visualization and detailed logging"""

    # === HARDCODED PARAMETERS ===
    MAX_SAMPLES = 100  # Fixed: always stop at 100 samples

    def __init__(self):
        super().__init__('baseline_sampler')

        # === ROS2 Parameters (configurable) ===
        self.declare_parameter('field_type', 'radial')
        self.declare_parameter('trial', -1)               # Trial number (-1 = auto-increment)
        self.declare_parameter('noise_var', 0.36)
        self.declare_parameter('lengthscale', 2.0)
        self.declare_parameter('lambda_cost', 0.1)        # Trade-off parameter
        self.declare_parameter('candidate_resolution', 1.0)
        self.declare_parameter('optimize_every', 10)      # Optimize hyperparams every N obs (0=disabled)
        self.declare_parameter('optimize_steps', 20)      # Gradient steps per optimization

        # Get parameters
        self.field_type = self.get_parameter('field_type').value
        self.trial_num = self.get_parameter('trial').value
        self.noise_var = self.get_parameter('noise_var').value
        self.lengthscale = self.get_parameter('lengthscale').value
        self.lambda_cost = self.get_parameter('lambda_cost').value
        self.candidate_res = self.get_parameter('candidate_resolution').value
        self.optimize_every = self.get_parameter('optimize_every').value
        self.optimize_steps = self.get_parameter('optimize_steps').value

        # QoS profile for PX4
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

        # RQT-friendly publishers
        self.info_pub = self.create_publisher(Float32, '/info_gain/current', 10)
        self.cost_pub = self.create_publisher(Float32, '/info_gain/cumulative_cost', 10)
        self.variance_pub = self.create_publisher(Float32, '/info_gain/mean_variance', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, qos_profile)
        self.temp_sub = self.create_subscription(Float32, f'/gaussian_field/{self.field_type}/temperature_noisy', self.temp_callback, 10)

        # State
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_variance = np.zeros(2)  # [σ_x², σ_y²] from PX4 EKF (ENU frame)
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

        # Initial waypoints: simple 3-point pattern (one short turn)
        self.initial_waypoints = [
            np.array([5.0, 5.0, 0.0]),
            np.array([10.0, 5.0, 0.0]),
            np.array([10.0, 10.0, 0.0]),
        ]
        self.waypoint_idx = 0

        # GP model
        self.gp = GPModel(lengthscale=self.lengthscale, noise_var=self.noise_var,
                          optimize_every=self.optimize_every, optimize_steps=self.optimize_steps)

        # CUDA warmup - trigger JIT compilation before main loop
        self.get_logger().info('Warming up CUDA/PyTorch...')
        dummy_X = torch.tensor([[12.5, 12.5], [10.0, 10.0], [15.0, 15.0]], dtype=torch.float32)
        dummy_y = torch.tensor([25.0, 24.0, 23.0], dtype=torch.float32)
        self.gp.fit(dummy_X, dummy_y)
        test_pts = torch.tensor([[12.0, 12.0], [13.0, 13.0]], dtype=torch.float32)
        _ = self.gp.predict(test_pts)  # Trigger CUDA compilation
        self.gp = GPModel(lengthscale=self.lengthscale, noise_var=self.noise_var,
                          optimize_every=self.optimize_every, optimize_steps=self.optimize_steps)  # Reset
        self.get_logger().info('CUDA warmup complete')

        # Candidate grid (0.5m buffer from edges)
        self.candidates = self._generate_candidate_grid(0.5, 24.5, 0.5, 24.5, self.candidate_res)

        # Output directory
        self.output_dir = self._create_trial_directory()

        # Ground truth field (for evaluation)
        self.get_logger().info('Generating ground truth field for evaluation...')
        self.gt_X, self.gt_Y, self.gt_field = generate_ground_truth_field(self.field_type)
        self._save_ground_truth()

        # Data logging
        self.samples = []
        self.decisions = []  # Detailed per-decision log
        self.samples_file = self.output_dir / 'samples.csv'
        self._init_samples_csv()

        # Visualization
        self.get_logger().info('Creating live visualizer...')
        try:
            self.viz = LiveVisualizer(
                title=f'Baseline Sampler - {self.field_type}',
                output_dir=self.output_dir
            )
            self.get_logger().info('Live visualizer created successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to create visualizer: {e}')
            self.viz = None

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # Save config
        self._save_config()

        self.get_logger().info(f'='*60)
        self.get_logger().info(f'Baseline Sampler initialized')
        self.get_logger().info(f'  Field: {self.field_type}')
        self.get_logger().info(f'  Trial: {self.trial_num}')
        self.get_logger().info(f'  Lambda: {self.lambda_cost}')
        self.get_logger().info(f'  Max samples: {self.MAX_SAMPLES}')
        self.get_logger().info(f'  Output: {self.output_dir}')
        self.get_logger().info(f'='*60)

    def _generate_candidate_grid(self, x_min, x_max, y_min, y_max, resolution):
        x = np.arange(x_min, x_max + 1e-9, resolution)
        y = np.arange(y_min, y_max + 1e-9, resolution)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _create_trial_directory(self):
        # Use workspace root (find it by going up from current working directory)
        # When run via ros2 run, cwd is typically the workspace root
        workspace_root = Path.cwd()
        base_dir = workspace_root / 'data' / 'trials' / 'exact' / self.field_type
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.trial_num >= 0:
            # Use specified trial number
            trial_num = self.trial_num
        else:
            # Auto-increment: find next available trial number
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
            trial_num = max([int(d.name.split('_')[1]) for d in existing], default=0) + 1

        trial_dir = base_dir / f'trial_{trial_num:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / 'figures').mkdir(exist_ok=True)
        self.trial_num = trial_num  # Store actual trial number used
        return trial_dir

    def _save_config(self):
        config = {
            'method': 'exact',
            'description': 'Exact planner: no positional uncertainty (but PX4 variance logged)',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'noise_var': self.noise_var,
            'lengthscale': self.lengthscale,
            'lambda_cost': self.lambda_cost,
            'candidate_resolution': self.candidate_res,
            'log_px4_ekf_variance': True,  # Logs PX4 variance but doesn't use it for planning
            'max_samples': self.MAX_SAMPLES,
            'n_initial': len(self.initial_waypoints),
            'n_candidates': len(self.candidates),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def _save_ground_truth(self):
        """Save ground truth field to file"""
        gt_file = self.output_dir / 'ground_truth.npz'
        np.savez(gt_file, X=self.gt_X, Y=self.gt_Y, field=self.gt_field)
        self.get_logger().info(f'Ground truth saved to {gt_file}')

    def _init_samples_csv(self):
        """Initialize samples.csv with header"""
        self._csv_fieldnames = ['step', 'phase', 'x', 'y', 'temp', 'info_gain', 'cumulative_info',
                                'travel_cost', 'gp_n_obs', 'pos_var_x', 'pos_var_y', 'pos_std_x', 'pos_std_y',
                                'ls_optimized', 'learned_lengthscale', 'learned_signal_var', 'learned_mean']
        with open(self.samples_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()

    def _write_sample(self, sample_dict):
        """Append a single sample to samples.csv"""
        try:
            with open(self.samples_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writerow(sample_dict)
        except Exception as e:
            self.get_logger().warn(f'Failed to write sample to CSV: {e}')

    def _compute_reconstruction_metrics(self):
        """Compute reconstruction error metrics vs ground truth"""
        # Get GP predictions over ground truth grid
        grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
        grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

        with torch.no_grad():
            gp_mean, gp_var = self.gp.predict(grid_t)
            gp_mean = gp_mean.cpu().numpy().reshape(self.gt_X.shape)
            gp_var = gp_var.cpu().numpy().reshape(self.gt_X.shape)

        # Compute errors
        error = gp_mean - self.gt_field
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        max_error = float(np.max(np.abs(error)))
        mean_variance = float(np.mean(gp_var))

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'mean_variance': mean_variance,
            'n_observations': self.gp.n_observations
        }

        # Save metrics
        with open(self.output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save GP predictions
        np.savez(self.output_dir / 'gp_reconstruction.npz',
                 X=self.gt_X, Y=self.gt_Y,
                 mean=gp_mean, variance=gp_var, error=error)

        # Create comparison figure
        self._plot_reconstruction_comparison(gp_mean, gp_var, error)

        self.get_logger().info(f'Reconstruction metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, Max={max_error:.3f}')

        return metrics

    def _plot_reconstruction_comparison(self, gp_mean, gp_var, error):
        """Create 2x3 comparison plot matching NS planner layout."""
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

        plt.suptitle(f'Exact - {self.field_type} (Trial {self.trial_num}) - RMSE: {rmse:.3f}°C',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def odometry_callback(self, msg):
        """
        Process PX4 odometry and extract position uncertainty from EKF.

        PX4 VehicleOdometry message contains:
        ------------------------------------
        - position[3]: [x_north, y_east, z_down] in NED frame
        - position_variance[3]: [σ²_north, σ²_east, σ²_down] in NED frame

        Planning frame:
        ---------------
        All planning uses NED directly (no ENU conversion).
        Position:  [x_north, y_east]
        Variance:  [σ²_north, σ²_east]  — same axis order as position

        Note (EXACT planner):
        --------------------
        This planner LOGS the PX4 variance but does NOT use it for planning.
        It assumes exact positions. The variance is logged to samples.csv
        for later analysis and comparison with pose-aware planner.
        """
        # Position: NED frame used directly (no ENU conversion)
        self.current_position = np.array([msg.position[0], msg.position[1], msg.position[2]])

        # Position variance: NED frame [σ²_north, σ²_east] — same axis order as position
        # Logged only, NOT used for planning (exact planner assumes no position uncertainty)
        self.position_variance = np.array([msg.position_variance[0], msg.position_variance[1]])

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
                self.get_logger().info('Armed, waiting 3s...')

        elif self.state == 'ARM':
            # Re-send arm + offboard every 0.5s in case PX4 missed the first commands
            if time.time() - self._arm_retry_time >= 0.5:
                self.arm()
                self.engage_offboard()
                self._arm_retry_time = time.time()
            if time.time() - self.last_command_time >= 3.0:
                self.state = 'INITIAL_SAMPLING'
                self.current_target = self.initial_waypoints[0]
                self.last_position = self.current_position[:2].copy()
                self._stuck_check_time = time.time()
                self.get_logger().info('Starting initial sampling (3 points)')

        elif self.state == 'INITIAL_SAMPLING':
            self._run_initial_sampling()

        elif self.state == 'ADAPTIVE_SAMPLING':
            self._run_adaptive_sampling()

    def _run_initial_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        # If rover hasn't moved after 15s, re-send arm + offboard commands
        if hasattr(self, '_stuck_check_time') and np.linalg.norm(self.current_position[:2]) < 1.0:
            if time.time() - self._stuck_check_time > 15.0:
                self.get_logger().warn('Rover stuck near origin - re-sending arm + offboard')
                self.arm()
                self.engage_offboard()
                self._stuck_check_time = time.time()
        else:
            # Rover is moving, clear stuck timer
            self._stuck_check_time = time.time()

        # Debug logging every 5 seconds
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 5.0:
            self.get_logger().info(
                f'Initial sampling: waypoint {self.waypoint_idx+1}/{len(self.initial_waypoints)}, '
                f'dist={dist:.2f}m, temp={self.current_temp}'
            )
            self._last_log_time = time.time()

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
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1]))
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.get_logger().info(f'Initial {self.waypoint_idx + 1}/3: ({x[0]:.1f}, {x[1]:.1f}), temp={y:.2f}')

            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.initial_waypoints):
                self.state = 'ADAPTIVE_SAMPLING'
                self.get_logger().info('='*60)
                self.get_logger().info('Starting adaptive sampling')
                self.get_logger().info('='*60)
                self._plan_next_sample()
            else:
                self.current_target = self.initial_waypoints[self.waypoint_idx]

    def _run_adaptive_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        if self.waiting_for_observation and dist < 0.5 and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            # Compute info gain BEFORE updating GP
            _, var_at_x = self.gp.predict(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
            realized_info = float(information_gain(var_at_x, self.noise_var).item())
            self.cumulative_info_gain += realized_info

            # Travel cost
            step_cost = travel_cost(self.last_position, x)
            self.total_travel_cost += step_cost
            self.last_position = x.copy()

            # Update GP
            self.gp.add_observation(x, y)
            self.sample_count += 1

            # Optimize hyperparameters if scheduled
            optimized = False
            if self.gp.should_optimize():
                self.gp.optimize_hyperparameters(logger=self.get_logger())
                optimized = True

            # Log sample
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

            # Publish to RQT topics (ensure Python float for ROS2)
            self.info_pub.publish(Float32(data=float(realized_info)))
            self.cost_pub.publish(Float32(data=float(self.total_travel_cost)))

            self.get_logger().info(
                f'Sample {self.sample_count}/{self.MAX_SAMPLES}: '
                f'({x[0]:.1f}, {x[1]:.1f}), info={realized_info:.4f}'
            )

            self.waiting_for_observation = False

            # Check stopping
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

        # Get top-5 candidates for logging
        scores_np = all_scores.cpu().numpy() if isinstance(all_scores, torch.Tensor) else all_scores
        top5_idx = np.argsort(scores_np)[-5:][::-1]
        top5_scores = scores_np[top5_idx]
        top5_pos = self.candidates[top5_idx]

        # Log decision details
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

        # Update visualization (with error handling to prevent crashes)
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
                self.get_logger().warn(f'Visualization update failed: {e}')

        self.get_logger().info(
            f'Planned: ({x_next[0]:.1f}, {x_next[1]:.1f}), score={best_score:.4f}'
        )

    def _greedy_single_step(self, current_pos):
        """
        Single-step greedy planning (H=1) - EXACT position assumption.

        Mathematical formulation:
        -------------------------
        Solve the myopic (one-step lookahead) optimization:

            x*_{k+1} = argmax_{x ∈ C}  Δ_k(x) - λ * c(x_k, x)

        where:
            - Δ_k(x) = (1/2) * log(1 + σ²_k(x) / σ_n²)  [info gain at EXACT location x]
            - c(x_k, x) = ||x_k - x||₂                   [Euclidean travel cost]
            - λ = trade-off parameter
            - C = candidate grid

        Key assumption (EXACT planner):
        -------------------------------
        The robot assumes it will sample EXACTLY at the commanded position x.
        No consideration of position uncertainty.

        This is the BASELINE for comparison with pose-aware planning.

        Note: PX4 EKF variance is logged in samples.csv but NOT used for planning.

        Args:
            current_pos: x_k, current robot position

        Returns:
            best_idx: index of best candidate in self.candidates
            best_score: Δ(x*) - λ * c(x_k, x*)
            best_info: Δ(x*), information gain at best candidate
            scores: tensor of scores for all candidates (for visualization)
        """
        # Vectorized GP prediction at all candidates
        candidates_t = torch.tensor(self.candidates, dtype=torch.float32).to(device)

        # Get posterior variance: σ²_k(x) for all x ∈ C
        _, variances = self.gp.predict(candidates_t)

        # Compute information gain: Δ_k(x) = (1/2) * log(1 + σ²_k(x) / σ_n²)
        # EXACT: evaluated at commanded positions, no pose uncertainty averaging
        info_gains = information_gain(variances, self.noise_var)

        # Compute travel costs: c(x_k, x) = ||x_k - x||₂
        current_t = torch.tensor(current_pos, dtype=torch.float32).to(device)
        travel_costs = torch.norm(candidates_t - current_t, dim=1)

        # Compute acquisition scores: score(x) = Δ_k(x) - λ * c(x_k, x)
        scores = info_gains - self.lambda_cost * travel_costs

        # Find best candidate: x* = argmax score(x)
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        best_info = float(info_gains[best_idx].item())

        return best_idx, best_score, best_info, scores

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
                gp_label='Exact GP',
            )

            n_sig = sum(1 for p in peaks if p.get('significant', False))
            self.get_logger().info(
                f'Final hotspot analysis: {len(peaks)} candidates, {n_sig} significant'
            )
        except Exception as e:
            self.get_logger().warn(f'Final hotspot analysis failed: {e}')

        # Also run on ground truth field for validation
        self._run_ground_truth_hotspot_analysis()

    def _run_ground_truth_hotspot_analysis(self):
        """Fit GP to dense ground truth samples, run Kac-Rice for validation."""
        try:
            import shutil
            from info_gain.gp_model import GPModel as StationaryGPModel

            # Subsample GT field (every 3rd point → ~17x17 = 289 training pts)
            step = 3
            gt_sub_X = self.gt_X[::step, ::step]
            gt_sub_Y = self.gt_Y[::step, ::step]
            gt_sub_field = self.gt_field[::step, ::step]

            train_x = np.column_stack([gt_sub_X.ravel(), gt_sub_Y.ravel()])
            train_y = gt_sub_field.ravel()

            # Fit fresh GP with very low noise (ground truth is noiseless)
            gt_gp = StationaryGPModel(noise_var=0.001, lengthscale=self.lengthscale)
            train_x_t = torch.tensor(train_x, dtype=torch.float32).to(device)
            train_y_t = torch.tensor(train_y, dtype=torch.float32).to(device)
            gt_gp.fit(train_x_t, train_y_t)

            # Predict on full grid
            grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, var = gt_gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(self.gt_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(self.gt_X.shape))

            # Run peak detection into temp subdir
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

            # Move outputs to trial directory
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

    def _finish_mission(self):
        self.state = 'DONE'

        self.get_logger().info('Computing reconstruction metrics...')
        reconstruction_metrics = self._compute_reconstruction_metrics()

        # Run final Kac-Rice hotspot analysis (post-mission, on final GP)
        self.get_logger().info('Running final hotspot analysis...')
        self._run_final_hotspot_analysis()

        # Save visualization
        if self.viz is not None:
            self.viz.save(self.output_dir / 'figures' / 'final.png')

        # Save samples CSV
        with open(self.output_dir / 'samples.csv', 'w', newline='') as f:
            if self.samples:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.samples)

        # Save decisions CSV (detailed)
        with open(self.output_dir / 'decisions.csv', 'w', newline='') as f:
            if self.decisions:
                # Flatten lists for CSV
                flat_decisions = []
                for d in self.decisions:
                    flat = {k: v for k, v in d.items() if not isinstance(v, list)}
                    flat['top5_x'] = str(d['top5_x'])
                    flat['top5_y'] = str(d['top5_y'])
                    flat['top5_scores'] = str(d['top5_scores'])
                    flat_decisions.append(flat)
                writer = csv.DictWriter(f, fieldnames=flat_decisions[0].keys())
                writer.writeheader()
                writer.writerows(flat_decisions)

        # Save decisions JSON (full)
        with open(self.output_dir / 'decisions.json', 'w') as f:
            json.dump(self.decisions, f, indent=2)

        # Save summary
        summary = {
            'method': 'exact',
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
        self.get_logger().info('MISSION COMPLETE')
        self.get_logger().info(f'  Samples: {self.sample_count}')
        self.get_logger().info(f'  Travel: {self.total_travel_cost:.1f}m')
        self.get_logger().info(f'  Info gain: {self.cumulative_info_gain:.4f}')
        self.get_logger().info(f'  Reconstruction RMSE: {reconstruction_metrics["rmse"]:.3f}°C')
        self.get_logger().info(f'  Reconstruction MAE: {reconstruction_metrics["mae"]:.3f}°C')
        self.get_logger().info(f'  Stop: {self.stop_reason}')
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
    node = BaselineSampler()

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
