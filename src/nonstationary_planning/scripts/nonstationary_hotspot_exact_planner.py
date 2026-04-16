#!/usr/bin/env python3
"""
Non-Stationary Hotspot-Guided Exact Planner - Gibbs Kernel + Kac-Rice Exploitation

Two-phase adaptive sampling strategy:
  Phase 1 (Exploration, samples 1-40): Pure information gain with Gibbs kernel.
  Phase 2 (Exploitation, samples 41-100): Kac-Rice hotspot detection followed by
    multiplicative hotspot-weighted acquisition function.

Acquisition function:
  Phase 1:  x* = argmax  Δ(x) - λ·c(x)
  Phase 2:  x* = argmax  Δ(x)·w(x) - λ·c(x)

Where w(x) = 1 + α·Σ_h exp(-‖x - x_h‖²/(2r²)) is a multiplicative weight
that amplifies information gain near detected hotspots.

Edge artifact mitigation (4 layers):
  1. Interior trim: evaluate Kac-Rice on 3m-cropped grid
  2. Height threshold: reject peaks below GP_mean + 1·GP_std
  3. Boundary guard: reject peaks within 3m of domain edge
  4. Persistence filter: peaks must appear in 2 consecutive runs

Key insight: multiplicative weighting is self-regulating — once the hotspot
region is well-sampled (Δ≈0), the weight has no effect and the planner
naturally returns to exploration.
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

# Add nonstationary_planning module to path
import sys
script_dir = Path(__file__).parent
install_path = script_dir.parent / 'python3' / 'dist-packages'
if install_path.exists():
    sys.path.insert(0, str(install_path))
else:
    sys.path.insert(0, str(script_dir.parent))

from nonstationary_planning.gibbs_gp_model import NonstationaryGPModel

# Add info_gain package path for peak_detection (separate ROS2 package)
# Try installed path first (container: install/info_gain/lib/python3/dist-packages/)
# Then source path (host: src/info_gain/)
_ws_root = Path(__file__).resolve().parent
for _ in range(6):  # Walk up to find workspace root
    if (_ws_root / 'src' / 'info_gain' / 'info_gain').exists():
        sys.path.insert(0, str(_ws_root / 'src' / 'info_gain'))
        break
    if (_ws_root / 'install' / 'info_gain' / 'lib' / 'python3' / 'dist-packages').exists():
        sys.path.insert(0, str(_ws_root / 'install' / 'info_gain' / 'lib' / 'python3' / 'dist-packages'))
        break
    _ws_root = _ws_root.parent
from info_gain.peak_detection import detect_and_plot_peaks, detect_peaks_only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_ground_truth_field(field_type, width=25.0, height=25.0, resolution=0.5):
    """Generate ground truth temperature field (same as field generators)"""
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


def information_gain(variance, noise_var):
    """Delta(x) = (1/2) * log(1 + sigma^2(x) / sigma_n^2)"""
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
    """Visualization with lengthscale field panel"""

    def __init__(self, title="NS-Exact Sampler", output_dir=None):
        self.fig = plt.figure(figsize=(20, 10))
        self.title = title
        self.output_dir = output_dir

        gs = gridspec.GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        self.ax_mean = self.fig.add_subplot(gs[0, 0])
        self.ax_var = self.fig.add_subplot(gs[0, 1])
        self.ax_acq = self.fig.add_subplot(gs[0, 2])
        self.ax_ls = self.fig.add_subplot(gs[0, 3])  # Lengthscale field
        self.ax_traj = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_cost = self.fig.add_subplot(gs[1, 2])
        self.ax_ls_hist = self.fig.add_subplot(gs[1, 3])  # Lengthscale history

        self.trajectory = []
        self.info_gains = []
        self.travel_costs = []
        self.steps = []
        self.l1_means = []
        self.l2_means = []
        self.aniso_means = []

        self.grid_res = 0.5
        x = np.arange(0, 25 + self.grid_res, self.grid_res)
        y = np.arange(0, 25 + self.grid_res, self.grid_res)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.grid_points = np.column_stack([self.X_grid.ravel(), self.Y_grid.ravel()])

    def update(self, gp, candidates, scores, selected_idx, current_pos, target_pos,
               step, info_gain_val, cumulative_cost):
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

        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

        # Get anisotropic lengthscale fields
        ls_X, ls_Y, ls_L1, ls_L2, ls_Theta = gp.get_lengthscale_field()
        self.l1_means.append(float(np.mean(ls_L1)))
        self.l2_means.append(float(np.mean(ls_L2)))
        self.aniso_means.append(float(np.mean(ls_L1 / ls_L2)))

        for ax in [self.ax_mean, self.ax_var, self.ax_acq, self.ax_ls,
                    self.ax_traj, self.ax_info, self.ax_cost, self.ax_ls_hist]:
            ax.clear()

        # 1. GP Mean
        self.ax_mean.pcolormesh(self.X_grid, self.Y_grid, mean_grid, cmap='coolwarm', shading='auto')
        self.ax_mean.set_title('GP Mean')
        self.ax_mean.set_aspect('equal')
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_mean.plot(traj[:, 0], traj[:, 1], 'k.-', linewidth=1, markersize=4)
        self.ax_mean.scatter(current_pos[0], current_pos[1], c='lime', s=100, marker='o', edgecolors='black', zorder=10)

        # 2. GP Variance
        self.ax_var.pcolormesh(self.X_grid, self.Y_grid, var_grid, cmap='viridis', shading='auto')
        self.ax_var.set_title('GP Variance')
        self.ax_var.set_aspect('equal')

        # 3. Acquisition Function
        self.ax_acq.scatter(candidates[:, 0], candidates[:, 1], c=scores_np, cmap='hot', s=30, alpha=0.7)
        if selected_idx is not None:
            sel = candidates[selected_idx]
            self.ax_acq.scatter(sel[0], sel[1], c='cyan', s=200, marker='X', edgecolors='black', linewidths=2, zorder=10)
        self.ax_acq.set_title(f'Acquisition (Step {step})')
        self.ax_acq.set_xlim(0, 25)
        self.ax_acq.set_ylim(0, 25)
        self.ax_acq.set_aspect('equal')

        # 4. Lengthscale field
        im_ls = self.ax_ls.pcolormesh(ls_X, ls_Y, ls_L1, cmap='plasma', shading='auto')
        self.ax_ls.set_title(f'l1(x) [{ls_L1.min():.2f}-{ls_L1.max():.2f}]')
        self.ax_ls.set_aspect('equal')
        plt.colorbar(im_ls, ax=self.ax_ls, label='l1(x) [m]')

        # 5. Trajectory
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax_traj.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
            for i, pt in enumerate(traj):
                self.ax_traj.scatter(pt[0], pt[1], c='blue', s=50, zorder=5)
        self.ax_traj.set_title('Trajectory')
        self.ax_traj.set_xlim(0, 25)
        self.ax_traj.set_ylim(0, 25)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True, alpha=0.3)

        # 6. Info gain
        if len(self.info_gains) > 0:
            self.ax_info.plot(self.steps, self.info_gains, 'g-o', linewidth=2, markersize=4)
        self.ax_info.set_title('Info Gain per Step')
        self.ax_info.set_xlabel('Step')
        self.ax_info.grid(True, alpha=0.3)

        # 7. Travel cost
        if len(self.travel_costs) > 0:
            self.ax_cost.plot(self.steps, self.travel_costs, 'r-o', linewidth=2, markersize=4)
        self.ax_cost.set_title('Cumulative Travel')
        self.ax_cost.set_xlabel('Step')
        self.ax_cost.grid(True, alpha=0.3)

        # 8. Lengthscale evolution
        ls_steps = list(range(1, len(self.l1_means) + 1))
        self.ax_ls_hist.plot(ls_steps, self.l1_means, 'purple', linewidth=2, label='l1')
        self.ax_ls_hist.plot(ls_steps, self.l2_means, 'orange', linewidth=2, label='l2')
        self.ax_ls_hist.plot(ls_steps, self.aniso_means, 'green', linewidth=1, linestyle='--', label='aniso')
        self.ax_ls_hist.set_title('Lengthscale Evolution')
        self.ax_ls_hist.set_xlabel('Step')
        self.ax_ls_hist.set_ylabel('l(x) [m]')
        self.ax_ls_hist.legend(fontsize=8)
        self.ax_ls_hist.grid(True, alpha=0.3)

        self.fig.suptitle(f'{self.title} - Sample {step}/100', fontsize=14, fontweight='bold')

        if self.output_dir:
            progress_path = self.output_dir / 'figures' / 'progress.png'
            self.fig.savefig(progress_path, dpi=100, bbox_inches='tight')

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)


class NonstationaryHotspotExactSampler(Node):
    """Non-stationary hotspot-guided exact planner with Gibbs kernel + Kac-Rice exploitation"""

    MAX_SAMPLES = 100

    def __init__(self):
        super().__init__('nonstationary_hotspot_exact_sampler')

        # ROS2 Parameters
        self.declare_parameter('field_type', 'radial')
        self.declare_parameter('trial', -1)
        self.declare_parameter('noise_var', 0.36)
        self.declare_parameter('lengthscale', 2.0)
        self.declare_parameter('lambda_cost', 0.1)
        self.declare_parameter('candidate_resolution', 1.0)
        self.declare_parameter('optimize_every', 10)
        self.declare_parameter('optimize_steps', 50)
        self.declare_parameter('grid_size', 5)
        self.declare_parameter('l_min', 0.5)
        self.declare_parameter('l_max', 5.0)
        # Hotspot-guided sampling parameters
        self.declare_parameter('exploration_samples', 40)      # Phase 1 duration
        self.declare_parameter('hotspot_recheck_interval', 10) # Re-run Kac-Rice every N samples
        self.declare_parameter('hotspot_alpha', 2.0)           # Multiplicative amplification at peak
        self.declare_parameter('hotspot_radius', 3.0)          # Gaussian bump radius (meters)

        self.field_type = self.get_parameter('field_type').value
        self.trial_num = self.get_parameter('trial').value
        self.noise_var = self.get_parameter('noise_var').value
        self.lengthscale = self.get_parameter('lengthscale').value
        self.lambda_cost = self.get_parameter('lambda_cost').value
        self.candidate_res = self.get_parameter('candidate_resolution').value
        self.optimize_every = self.get_parameter('optimize_every').value
        self.optimize_steps = self.get_parameter('optimize_steps').value
        self.grid_size = self.get_parameter('grid_size').value
        self.l_min = self.get_parameter('l_min').value
        self.l_max = self.get_parameter('l_max').value
        self.exploration_samples = self.get_parameter('exploration_samples').value
        self.hotspot_recheck_interval = self.get_parameter('hotspot_recheck_interval').value
        self.hotspot_alpha = self.get_parameter('hotspot_alpha').value
        self.hotspot_radius = self.get_parameter('hotspot_radius').value

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
        self.info_pub = self.create_publisher(Float32, '/info_gain/current', 10)
        self.cost_pub = self.create_publisher(Float32, '/info_gain/cumulative_cost', 10)
        self.variance_pub = self.create_publisher(Float32, '/info_gain/mean_variance', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, qos_profile)
        self.temp_sub = self.create_subscription(Float32, f'/gaussian_field/{self.field_type}/temperature_noisy', self.temp_callback, 10)

        # State
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_variance = np.zeros(2)
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

        # Hotspot-guided sampling state
        self.sampling_phase = 'EXPLORE'  # 'EXPLORE' or 'EXPLOIT'
        self.detected_hotspots = []       # List of significant peak dicts
        self.hotspot_weights = None       # (N_candidates,) multiplicative weights
        self.peak_history = {}            # {(rounded_x, rounded_y): count} for persistence filter
        self.phase_transition_sample = None  # Sample number when exploitation began

        # Initial waypoints
        self.initial_waypoints = [
            np.array([5.0, 5.0, 0.0]),
            np.array([10.0, 5.0, 0.0]),
            np.array([10.0, 10.0, 0.0]),
        ]
        self.waypoint_idx = 0

        # Non-stationary GP model (Gibbs kernel)
        self.gp = NonstationaryGPModel(
            noise_var=self.noise_var,
            signal_var=1.0,
            l_init=self.lengthscale,
            grid_size=self.grid_size,
            l_min=self.l_min,
            l_max=self.l_max,
            optimize_every=self.optimize_every,
            optimize_steps=self.optimize_steps,
        )

        # CUDA warmup
        self.get_logger().info('Warming up CUDA/PyTorch with Gibbs kernel...')
        dummy_X = np.array([[12.5, 12.5], [10.0, 10.0], [15.0, 15.0]])
        dummy_y = np.array([25.0, 24.0, 23.0])
        self.gp.fit(dummy_X, dummy_y)
        test_pts = torch.tensor([[12.0, 12.0], [13.0, 13.0]], dtype=torch.float32)
        _ = self.gp.predict(test_pts)
        self.gp = NonstationaryGPModel(
            noise_var=self.noise_var, signal_var=1.0, l_init=self.lengthscale,
            grid_size=self.grid_size, l_min=self.l_min, l_max=self.l_max,
            optimize_every=self.optimize_every, optimize_steps=self.optimize_steps,
        )
        self.get_logger().info('CUDA warmup complete')

        # Candidate grid
        self.candidates = self._generate_candidate_grid(0.5, 24.5, 0.5, 24.5, self.candidate_res)

        # Output directory
        self.output_dir = self._create_trial_directory()

        # Ground truth
        self.get_logger().info('Generating ground truth field...')
        self.gt_X, self.gt_Y, self.gt_field = generate_ground_truth_field(self.field_type)
        self._save_ground_truth()

        # Data logging
        self.samples = []
        self.decisions = []
        self.samples_file = self.output_dir / 'samples.csv'
        self._init_samples_csv()

        # Visualization
        self.get_logger().info('Creating live visualizer...')
        try:
            self.viz = LiveVisualizer(
                title=f'NS-Hotspot-Exact - {self.field_type}',
                output_dir=self.output_dir
            )
        except Exception as e:
            self.get_logger().error(f'Failed to create visualizer: {e}')
            self.viz = None

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self._save_config()

        self.get_logger().info(f'='*60)
        self.get_logger().info(f'Non-Stationary HOTSPOT-GUIDED EXACT Sampler initialized')
        self.get_logger().info(f'  Kernel: Gibbs (spatially varying lengthscale)')
        self.get_logger().info(f'  Strategy: Explore ({self.exploration_samples} samples) → Exploit (Kac-Rice hotspot)')
        self.get_logger().info(f'  Hotspot: alpha={self.hotspot_alpha}, radius={self.hotspot_radius}m, recheck={self.hotspot_recheck_interval}')
        self.get_logger().info(f'  Grid size: {self.grid_size}x{self.grid_size} ({self.grid_size**2} basis)')
        self.get_logger().info(f'  l range: [{self.l_min}, {self.l_max}], init={self.lengthscale}')
        self.get_logger().info(f'  Optimize every: {self.optimize_every} samples')
        self.get_logger().info(f'  Field: {self.field_type}')
        self.get_logger().info(f'  Trial: {self.trial_num}')
        self.get_logger().info(f'  Output: {self.output_dir}')
        self.get_logger().info(f'='*60)

    def _generate_candidate_grid(self, x_min, x_max, y_min, y_max, resolution):
        x = np.arange(x_min, x_max + 1e-9, resolution)
        y = np.arange(y_min, y_max + 1e-9, resolution)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _create_trial_directory(self):
        workspace_root = Path.cwd()
        base_dir = workspace_root / 'data' / 'trials' / 'nonstationary_hotspot_exact' / self.field_type
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.trial_num >= 0:
            trial_num = self.trial_num
        else:
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
            trial_num = max([int(d.name.split('_')[1]) for d in existing], default=0) + 1

        trial_dir = base_dir / f'trial_{trial_num:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / 'figures').mkdir(exist_ok=True)
        (trial_dir / 'lengthscale').mkdir(exist_ok=True)
        self.trial_num = trial_num
        return trial_dir

    def _save_config(self):
        config = {
            'method': 'nonstationary_hotspot_exact',
            'description': 'Non-stationary Gibbs kernel GP + Kac-Rice hotspot-guided sampling',
            'kernel': 'gibbs',
            'field_type': self.field_type,
            'trial': self.trial_num,
            'noise_var': self.noise_var,
            'lengthscale_init': self.lengthscale,
            'lambda_cost': self.lambda_cost,
            'candidate_resolution': self.candidate_res,
            'grid_size': self.grid_size,
            'l_min': self.l_min,
            'l_max': self.l_max,
            'optimize_every': self.optimize_every,
            'optimize_steps': self.optimize_steps,
            'exploration_samples': self.exploration_samples,
            'hotspot_recheck_interval': self.hotspot_recheck_interval,
            'hotspot_alpha': self.hotspot_alpha,
            'hotspot_radius': self.hotspot_radius,
            'max_samples': self.MAX_SAMPLES,
            'n_initial': len(self.initial_waypoints),
            'n_candidates': len(self.candidates),
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
                                'ls_optimized', 'l1_mean', 'l2_mean', 'theta_mean', 'aniso_ratio',
                                'sampling_phase', 'hotspot_weight']
        with open(self.samples_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()

    def _write_sample(self, sample_dict):
        try:
            with open(self.samples_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writerow(sample_dict)
        except Exception as e:
            self.get_logger().warn(f'Failed to write sample to CSV: {e}')

    def _maybe_optimize_lengthscale(self):
        """Optimize Gibbs kernel lengthscale if it's time."""
        if self.gp.should_optimize():
            self.get_logger().info(f'Optimizing lengthscale (n={self.gp.n_observations})...')
            t0 = time.time()
            self.gp.optimize_lengthscale(logger=self.get_logger())
            dt = time.time() - t0
            self.get_logger().info(f'Lengthscale optimization took {dt:.2f}s')

            # Save snapshot
            self.gp.save_lengthscale_snapshot(
                self.output_dir / 'lengthscale',
                self.sample_count
            )
            return True
        return False

    def _get_ls_stats(self):
        """Get current anisotropic lengthscale statistics."""
        _, _, L1, L2, Theta = self.gp.get_lengthscale_field()
        return (float(np.mean(L1)), float(np.mean(L2)),
                float(np.mean(Theta)), float(np.mean(L1 / L2)))

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
        mean_variance = float(np.mean(gp_var))

        metrics = {
            'rmse': rmse, 'mae': mae, 'max_error': max_error,
            'mean_variance': mean_variance, 'n_observations': self.gp.n_observations
        }

        with open(self.output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        np.savez(self.output_dir / 'gp_reconstruction.npz',
                 X=self.gt_X, Y=self.gt_Y, mean=gp_mean, variance=gp_var, error=error)

        self._plot_reconstruction_comparison(gp_mean, gp_var, error)
        self.get_logger().info(f'Reconstruction: RMSE={rmse:.3f}, MAE={mae:.3f}, Max={max_error:.3f}')
        return metrics

    def _plot_reconstruction_comparison(self, gp_mean, gp_var, error):
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        samples_arr = None
        if len(self.samples) > 0:
            samples_arr = np.array([[s['x'], s['y']] for s in self.samples])

        # Ground truth
        im0 = axes[0, 0].pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto')
        axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0], label='T [°C]')
        if samples_arr is not None:
            axes[0, 0].scatter(samples_arr[:, 0], samples_arr[:, 1], c='black', s=30, marker='x', linewidths=2)

        # GP Reconstruction
        im1 = axes[0, 1].pcolormesh(self.gt_X, self.gt_Y, gp_mean, cmap='coolwarm', shading='auto')
        axes[0, 1].set_title(f'GP Reconstruction (n={self.gp.n_observations})', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1], label='T [°C]')

        # Error
        im2 = axes[0, 2].pcolormesh(self.gt_X, self.gt_Y, np.abs(error), cmap='hot', shading='auto')
        axes[0, 2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 2], label='|Error| [°C]')
        axes[0, 2].text(0.02, 0.98, f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C',
                       transform=axes[0, 2].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Variance
        im3 = axes[1, 0].pcolormesh(self.gt_X, self.gt_Y, gp_var, cmap='viridis', shading='auto')
        axes[1, 0].set_title(f'GP Variance', fontsize=12, fontweight='bold')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0], label='Variance')

        # Lengthscale field
        ls_X, ls_Y, ls_L1, ls_L2, ls_Theta = self.gp.get_lengthscale_field()
        im4 = axes[1, 1].pcolormesh(ls_X, ls_Y, ls_L1, cmap='plasma', shading='auto')
        axes[1, 1].set_title(f'Learned l1(x) [aniso={np.mean(ls_L1/ls_L2):.2f}]', fontsize=12, fontweight='bold')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 1], label='l1(x) [m]')
        if samples_arr is not None:
            axes[1, 1].scatter(samples_arr[:, 0], samples_arr[:, 1], c='white', s=20, edgecolors='black')

        # Convex hull
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

        plt.suptitle(f'NS-Hotspot-Exact - {self.field_type} (Trial {self.trial_num}) - RMSE: {rmse:.3f}°C',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def odometry_callback(self, msg):
        # Position: NED frame used directly (no ENU conversion)
        self.current_position = np.array([msg.position[0], msg.position[1], msg.position[2]])
        # Position variance: NED frame [σ²_north, σ²_east] — same axis order as position
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

        if hasattr(self, '_stuck_check_time') and np.linalg.norm(self.current_position[:2]) < 1.0:
            if time.time() - self._stuck_check_time > 15.0:
                self.get_logger().warn('Rover stuck - re-sending arm + offboard')
                self.arm()
                self.engage_offboard()
                self._stuck_check_time = time.time()
        else:
            self._stuck_check_time = time.time()

        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0
        if time.time() - self._last_log_time > 5.0:
            self.get_logger().info(
                f'Initial sampling: wp {self.waypoint_idx+1}/{len(self.initial_waypoints)}, dist={dist:.2f}m'
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

            l1_mean, l2_mean, theta_mean, aniso_ratio = self._get_ls_stats()
            sample = {
                'step': self.sample_count, 'phase': 'initial',
                'x': float(x[0]), 'y': float(x[1]), 'temp': float(y),
                'info_gain': 0.0, 'cumulative_info': 0.0,
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1])),
                'ls_optimized': False, 'l1_mean': l1_mean, 'l2_mean': l2_mean,
                'theta_mean': theta_mean, 'aniso_ratio': aniso_ratio,
                'sampling_phase': 'initial', 'hotspot_weight': 1.0,
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.get_logger().info(f'Initial {self.waypoint_idx + 1}/3: ({x[0]:.1f}, {x[1]:.1f}), temp={y:.2f}')

            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.initial_waypoints):
                self.state = 'ADAPTIVE_SAMPLING'
                self.get_logger().info('='*60)
                self.get_logger().info('Starting adaptive sampling (non-stationary)')
                self.get_logger().info('='*60)
                self._plan_next_sample()
            else:
                self.current_target = self.initial_waypoints[self.waypoint_idx]

    def _run_adaptive_sampling(self):
        dist = np.linalg.norm(self.current_position[:2] - self.current_target[:2])

        if self.waiting_for_observation and dist < 0.5 and self.current_temp is not None:
            x = self.current_position[:2].copy()
            y = self.current_temp

            # Info gain BEFORE updating GP
            _, var_at_x = self.gp.predict(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
            realized_info = float(information_gain(var_at_x, self.noise_var).item())
            self.cumulative_info_gain += realized_info

            step_cost = travel_cost(self.last_position, x)
            self.total_travel_cost += step_cost
            self.last_position = x.copy()

            self.gp.add_observation(x, y)
            self.sample_count += 1

            # Optimize lengthscale if it's time
            optimized = self._maybe_optimize_lengthscale()

            # === Hotspot phase transition logic ===
            self._check_hotspot_phase_transition()

            # Compute hotspot weight at this sample location
            hw = 1.0
            if self.hotspot_weights is not None:
                # Find nearest candidate to get weight
                dists_to_cands = np.linalg.norm(self.candidates - x, axis=1)
                nearest_idx = np.argmin(dists_to_cands)
                hw = float(self.hotspot_weights[nearest_idx])

            l1_mean, l2_mean, theta_mean, aniso_ratio = self._get_ls_stats()
            sample = {
                'step': self.sample_count, 'phase': 'adaptive',
                'x': float(x[0]), 'y': float(x[1]), 'temp': float(y),
                'info_gain': realized_info,
                'cumulative_info': float(self.cumulative_info_gain),
                'travel_cost': float(self.total_travel_cost),
                'gp_n_obs': self.gp.n_observations,
                'pos_var_x': float(self.position_variance[0]),
                'pos_var_y': float(self.position_variance[1]),
                'pos_std_x': float(np.sqrt(self.position_variance[0])),
                'pos_std_y': float(np.sqrt(self.position_variance[1])),
                'ls_optimized': optimized, 'l1_mean': l1_mean, 'l2_mean': l2_mean,
                'theta_mean': theta_mean, 'aniso_ratio': aniso_ratio,
                'sampling_phase': self.sampling_phase.lower(),
                'hotspot_weight': hw,
            }
            self.samples.append(sample)
            self._write_sample(sample)

            self.info_pub.publish(Float32(data=float(realized_info)))
            self.cost_pub.publish(Float32(data=float(self.total_travel_cost)))

            phase_tag = f' [{self.sampling_phase}]' if self.sampling_phase == 'EXPLOIT' else ''
            self.get_logger().info(
                f'Sample {self.sample_count}/{self.MAX_SAMPLES}{phase_tag}: '
                f'({x[0]:.1f}, {x[1]:.1f}), info={realized_info:.4f}, '
                f'l1={l1_mean:.2f}, l2={l2_mean:.2f}, aniso={aniso_ratio:.2f}'
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

        scores_np = all_scores.cpu().numpy() if isinstance(all_scores, torch.Tensor) else all_scores
        top5_idx = np.argsort(scores_np)[-5:][::-1]
        top5_scores = scores_np[top5_idx]
        top5_pos = self.candidates[top5_idx]

        decision = {
            'step': self.sample_count + 1,
            'current_x': float(current_pos[0]), 'current_y': float(current_pos[1]),
            'selected_x': float(x_next[0]), 'selected_y': float(x_next[1]),
            'selected_score': float(best_score), 'selected_info': float(best_info),
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
            'gp_n_obs': self.gp.n_observations,
        }
        self.decisions.append(decision)

        if self.viz is not None:
            try:
                self.viz.update(
                    gp=self.gp, candidates=self.candidates, scores=all_scores,
                    selected_idx=best_idx, current_pos=current_pos, target_pos=x_next,
                    step=self.sample_count + 1, info_gain_val=best_info,
                    cumulative_cost=self.total_travel_cost
                )
            except Exception as e:
                self.get_logger().warn(f'Viz update failed: {e}')

        self.get_logger().info(f'Planned: ({x_next[0]:.1f}, {x_next[1]:.1f}), score={best_score:.4f}')

    def _check_hotspot_phase_transition(self):
        """Check if we should run Kac-Rice and transition to exploitation phase."""
        if self.sample_count < self.exploration_samples:
            return  # Still in exploration phase

        # Run Kac-Rice at the transition point or every recheck interval
        should_run = (
            self.sample_count == self.exploration_samples or
            (self.sample_count > self.exploration_samples and
             self.sample_count % self.hotspot_recheck_interval == 0)
        )
        if not should_run:
            return

        self.get_logger().info(f'Running online Kac-Rice hotspot detection (sample {self.sample_count})...')
        peaks = self._run_online_hotspot_detection()

        if peaks:
            self.detected_hotspots = peaks
            self._compute_hotspot_weights()
            if self.sampling_phase == 'EXPLORE':
                self.sampling_phase = 'EXPLOIT'
                self.phase_transition_sample = self.sample_count
                self.get_logger().info(
                    f'*** PHASE TRANSITION → EXPLOIT at sample {self.sample_count} ***'
                    f' ({len(peaks)} hotspot(s) detected)'
                )
            else:
                self.get_logger().info(f'Hotspot map updated: {len(peaks)} peak(s)')
        else:
            # No significant peaks — keep previous hotspot map if we had one (sticky)
            if self.sampling_phase == 'EXPLORE':
                self.get_logger().info('No significant peaks found, staying in EXPLORE')

    def _run_online_hotspot_detection(self):
        """Run Kac-Rice on trimmed interior grid with 4-layer edge artifact filtering."""
        try:
            grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)

            with torch.no_grad():
                mu, var = self.gp.predict(grid_t)
                mu_grid = mu.cpu().numpy().reshape(self.gt_X.shape)
                std_grid = np.sqrt(var.cpu().numpy().reshape(self.gt_X.shape))

            # Layer 1: Interior trim — crop 6 cells (3m) from each edge of 51×51 grid
            trim = 6
            mu_trimmed = mu_grid[trim:-trim, trim:-trim]
            lat_trimmed = self.gt_X[trim:-trim, trim:-trim]
            lon_trimmed = self.gt_Y[trim:-trim, trim:-trim]
            std_trimmed = std_grid[trim:-trim, trim:-trim]

            peaks = detect_peaks_only(
                gp_wrapper=self.gp,
                lat_mesh=lat_trimmed, lon_mesh=lon_trimmed,
                mu_orig=mu_trimmed, std_grid=std_trimmed,
                kernel_type='gibbs',
                y_mean=0.0,
                alpha=0.05,
                min_peak_distance=5,
                n_monte_carlo=50_000,
            )

            # Layer 2: Height threshold — reject peaks below GP_mean + 1*GP_std
            global_mean = float(mu_grid.mean())
            global_std = float(mu_grid.std())
            height_threshold = global_mean + 1.0 * global_std
            peaks = [p for p in peaks if p.get('significant', False) and p['height'] > height_threshold]

            # Layer 3: Boundary guard — reject peaks within 3m of domain edge
            peaks = [p for p in peaks if 3.0 < p['lat'] < 22.0 and 3.0 < p['lon'] < 22.0]

            # Layer 4: Persistence filter — peak must appear in 2+ consecutive runs
            new_history = {}
            for p in peaks:
                key = (round(p['lat']), round(p['lon']))
                prev_count = self.peak_history.get(key, 0)
                new_history[key] = prev_count + 1

            self.peak_history = new_history
            persistent_peaks = [
                p for p in peaks
                if self.peak_history.get((round(p['lat']), round(p['lon'])), 0) >= 2
            ]

            if persistent_peaks:
                for p in persistent_peaks:
                    self.get_logger().info(
                        f'  Persistent hotspot: ({p["lat"]:.1f}, {p["lon"]:.1f}), '
                        f'p={p["pvalue"]:.4f}, height={p["height"]:.1f}'
                    )
                return persistent_peaks

            # First detection — no persistent peaks yet, but store for next run
            if peaks:
                self.get_logger().info(f'  {len(peaks)} peak(s) found, waiting for persistence confirmation')
            return []

        except Exception as e:
            self.get_logger().warn(f'Online hotspot detection failed: {e}')
            return []

    def _compute_hotspot_weights(self):
        """Build multiplicative weight array w(x) over all candidates.

        w(x) = 1 + alpha * sum_h exp(-||x - x_h||^2 / (2 * r^2))

        Properties:
        - w(x) >= 1 everywhere (never reduces info gain)
        - w(x_h) = 1 + alpha at hotspot center
        - w(x) -> 1 far from hotspots
        """
        weights = np.ones(len(self.candidates))
        for hotspot in self.detected_hotspots:
            hx, hy = hotspot['lat'], hotspot['lon']
            dists_sq = (self.candidates[:, 0] - hx)**2 + (self.candidates[:, 1] - hy)**2
            weights += self.hotspot_alpha * np.exp(-dists_sq / (2 * self.hotspot_radius**2))
        self.hotspot_weights = weights

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
                kernel_type='gibbs',
                y_mean=0.0,
                gp_label='NS-Hotspot-Exact GP',
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

            gt_gp = StationaryGPModel(noise_var=0.001, lengthscale=2.0)
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
        """Greedy single-step with optional hotspot-weighted acquisition."""
        candidates_t = torch.tensor(self.candidates, dtype=torch.float32).to(device)
        _, variances = self.gp.predict(candidates_t)
        info_gains = information_gain(variances, self.noise_var)

        current_t = torch.tensor(current_pos, dtype=torch.float32).to(device)
        travel_costs = torch.norm(candidates_t - current_t, dim=1)

        if self.sampling_phase == 'EXPLOIT' and self.hotspot_weights is not None:
            # Multiplicative hotspot weighting: Δ(x)·w(x) - λ·c(x)
            weights_t = torch.tensor(self.hotspot_weights, dtype=torch.float32).to(device)
            scores = info_gains * weights_t - self.lambda_cost * travel_costs
        else:
            scores = info_gains - self.lambda_cost * travel_costs

        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        best_info = float(info_gains[best_idx].item())

        return best_idx, best_score, best_info, scores

    def _finish_mission(self):
        self.state = 'DONE'

        # Save final lengthscale snapshot
        self.gp.save_lengthscale_snapshot(self.output_dir / 'lengthscale', self.sample_count)

        self.get_logger().info('Computing reconstruction metrics...')
        reconstruction_metrics = self._compute_reconstruction_metrics()

        # Run final Kac-Rice hotspot analysis (post-mission, on final GP)
        self.get_logger().info('Running final hotspot analysis...')
        self._run_final_hotspot_analysis()

        # Generate hotspot sampling analysis figure
        self.get_logger().info('Generating hotspot sampling analysis...')
        self._plot_hotspot_sampling_analysis()

        if self.viz is not None:
            self.viz.save(self.output_dir / 'figures' / 'final.png')

        with open(self.output_dir / 'samples.csv', 'w', newline='') as f:
            if self.samples:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.samples)

        # Save decisions CSV (detailed)
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

        with open(self.output_dir / 'decisions.json', 'w') as f:
            json.dump(self.decisions, f, indent=2)

        # Compute hotspot-specific metrics
        hotspot_metrics = self._compute_hotspot_metrics()

        summary = {
            'method': 'nonstationary_hotspot_exact',
            'kernel': 'gibbs',
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
            'completed_at': datetime.now().isoformat(),
            'hotspot_metrics': hotspot_metrics,
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info('='*60)
        self.get_logger().info('MISSION COMPLETE (NS-Hotspot-Exact)')
        self.get_logger().info(f'  Samples: {self.sample_count}')
        self.get_logger().info(f'  Travel: {self.total_travel_cost:.1f}m')
        self.get_logger().info(f'  Info gain: {self.cumulative_info_gain:.4f}')
        self.get_logger().info(f'  RMSE: {reconstruction_metrics["rmse"]:.3f}°C')
        self.get_logger().info(f'  Phase: {self.sampling_phase} (transition at sample {self.phase_transition_sample})')
        self.get_logger().info(f'  Samples within 3m of peak: {hotspot_metrics["samples_within_3m"]}')
        self.get_logger().info(f'  Data: {self.output_dir}')
        self.get_logger().info('='*60)

    def _compute_hotspot_metrics(self):
        """Compute hotspot-specific metrics for summary.json."""
        TRUE_PEAK = np.array([12.5, 12.5])

        samples_arr = np.array([[s['x'], s['y']] for s in self.samples]) if self.samples else np.zeros((0, 2))
        dists_to_peak = np.linalg.norm(samples_arr - TRUE_PEAK, axis=1) if len(samples_arr) > 0 else np.array([])

        # Sample density near true peak
        within_3m = int(np.sum(dists_to_peak < 3.0))
        within_5m = int(np.sum(dists_to_peak < 5.0))

        # Hotspot region RMSE (grid points within 5m of true peak)
        grid_points = np.column_stack([self.gt_X.ravel(), self.gt_Y.ravel()])
        grid_dists = np.linalg.norm(grid_points - TRUE_PEAK, axis=1)
        hotspot_mask = grid_dists < 5.0

        hotspot_rmse = None
        hotspot_mae = None
        try:
            grid_t = torch.tensor(grid_points, dtype=torch.float32).to(device)
            with torch.no_grad():
                gp_mean, _ = self.gp.predict(grid_t)
                gp_mean = gp_mean.cpu().numpy()
            gt_flat = self.gt_field.ravel()
            error = gp_mean - gt_flat
            if hotspot_mask.any():
                hotspot_rmse = float(np.sqrt(np.mean(error[hotspot_mask]**2)))
                hotspot_mae = float(np.mean(np.abs(error[hotspot_mask])))
        except Exception:
            pass

        # Peak localization error
        peak_loc_error = None
        detected_peak_loc = None
        if self.detected_hotspots:
            best = min(self.detected_hotspots, key=lambda p: p['pvalue'])
            detected_peak_loc = [best['lat'], best['lon']]
            peak_loc_error = float(np.linalg.norm(np.array(detected_peak_loc) - TRUE_PEAK))

        return {
            'phase_transition_sample': self.phase_transition_sample,
            'final_phase': self.sampling_phase,
            'n_hotspots_detected': len(self.detected_hotspots),
            'detected_peak_location': detected_peak_loc,
            'peak_localization_error_m': peak_loc_error,
            'samples_within_3m': within_3m,
            'samples_within_5m': within_5m,
            'hotspot_rmse': hotspot_rmse,
            'hotspot_mae': hotspot_mae,
        }

    def _plot_hotspot_sampling_analysis(self):
        """Generate hotspot_sampling_analysis.png showing phase-colored trajectory and density."""
        try:
            TRUE_PEAK = np.array([12.5, 12.5])
            samples_arr = np.array([[s['x'], s['y']] for s in self.samples])
            phases = [s.get('sampling_phase', 'explore') for s in self.samples]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # (a) Trajectory with phase coloring
            ax = axes[0]
            ax.pcolormesh(self.gt_X, self.gt_Y, self.gt_field, cmap='coolwarm', shading='auto', alpha=0.5)
            colors = {'initial': 'green', 'explore': 'blue', 'exploit': 'red'}
            for i, (pt, ph) in enumerate(zip(samples_arr, phases)):
                c = colors.get(ph, 'grey')
                ax.scatter(pt[0], pt[1], c=c, s=40, edgecolors='black', linewidths=0.5, zorder=5)
                if i > 0:
                    ax.plot([samples_arr[i-1, 0], pt[0]], [samples_arr[i-1, 1], pt[1]],
                            c=c, linewidth=1, alpha=0.5)
            ax.scatter(*TRUE_PEAK, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2, zorder=10, label='True peak')
            if self.detected_hotspots:
                for h in self.detected_hotspots:
                    ax.scatter(h['lat'], h['lon'], c='magenta', s=150, marker='D',
                               edgecolors='black', linewidths=2, zorder=10)
            # Legend
            for label, color in colors.items():
                ax.scatter([], [], c=color, s=40, label=label.capitalize())
            ax.legend(fontsize=8, loc='upper right')
            ax.set_title('Trajectory (phase-colored)', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 25); ax.set_ylim(0, 25); ax.set_aspect('equal')

            # (b) Sample density heatmap
            ax = axes[1]
            ax.hist2d(samples_arr[:, 0], samples_arr[:, 1], bins=25, range=[[0, 25], [0, 25]], cmap='hot')
            ax.scatter(*TRUE_PEAK, c='cyan', s=200, marker='*', edgecolors='black', linewidths=2, zorder=10)
            ax.set_title('Sample density', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 25); ax.set_ylim(0, 25); ax.set_aspect('equal')

            # (c) Cumulative samples within 3m of true peak
            ax = axes[2]
            dists = np.linalg.norm(samples_arr - TRUE_PEAK, axis=1)
            cum_within_3m = np.cumsum(dists < 3.0)
            ax.plot(range(1, len(cum_within_3m) + 1), cum_within_3m, 'r-', linewidth=2, label='Within 3m')
            cum_within_5m = np.cumsum(dists < 5.0)
            ax.plot(range(1, len(cum_within_5m) + 1), cum_within_5m, 'b--', linewidth=2, label='Within 5m')
            if self.phase_transition_sample:
                ax.axvline(x=self.phase_transition_sample, color='green', linestyle=':', linewidth=2, label='Phase transition')
            ax.set_xlabel('Sample number')
            ax.set_ylabel('Cumulative samples near peak')
            ax.set_title('Hotspot sampling density', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.suptitle(f'NS-Hotspot-Exact - {self.field_type} (Trial {self.trial_num})',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'figures' / 'hotspot_sampling_analysis.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.get_logger().warn(f'Hotspot sampling analysis plot failed: {e}')

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
    node = NonstationaryHotspotExactSampler()
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
