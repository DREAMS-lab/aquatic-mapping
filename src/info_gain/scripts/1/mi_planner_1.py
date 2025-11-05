#!/usr/bin/env python3
import os, numpy as np, torch, gpytorch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import threading
import sys
from pathlib import Path
from collections import deque
_script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_script_dir, 'pg_viz.py')):
    _parent_dir = os.path.dirname(_script_dir)
    if os.path.exists(os.path.join(_parent_dir, 'pg_viz.py')):
        sys.path.insert(0, _parent_dir)
from pg_viz import PGLiveViz

def make_grid(width, height, res):
    xs = np.arange(0, width + 1e-9, res)
    ys = np.arange(0, height + 1e-9, res)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return xs, ys, X, Y, np.column_stack([X.ravel(), Y.ravel()])

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ell, sigma_f):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = float(ell)
        self.covar_module.outputscale = float(sigma_f)**2

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class TorchGP:
    def __init__(self, ell, sigma_f, sigma_n, device):
        self.device = device
        self.ell, self.sigma_f, self.sigma_n = float(ell), float(sigma_f), float(sigma_n)
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=self.sigma_n**2)

    def fit(self, X_np, y_np):
        if len(X_np) == 0:
            return
        X = torch.from_numpy(X_np).float().to(self.device)
        y = torch.from_numpy(y_np).float().to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=self.sigma_n**2).to(self.device)
        self.model = ExactGPModel(X, y, self.likelihood, self.ell, self.sigma_f).to(self.device)
        self.model.train(); self.likelihood.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(15):
            opt.zero_grad()
            loss = -mll(self.model(X), y)
            loss.backward(); opt.step()
        self.model.eval(); self.likelihood.eval()

    def predict(self, X_np):
        if self.model is None or len(X_np) == 0:
            return np.zeros(len(X_np)), np.ones(len(X_np)) * self.sigma_f
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X = torch.from_numpy(X_np).float().to(self.device)
            pred = self.likelihood(self.model(X))
            return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()

class MIPlanner(Node):
    def __init__(self):
        super().__init__('mi_planner_1')
        
        self.declare_parameter('width', 25.0)
        self.declare_parameter('height', 25.0)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('ell', 5.0)
        self.declare_parameter('sigma_f', 4.0)
        self.declare_parameter('sigma_n', 0.5)
        self.declare_parameter('max_samples', 50)
        self.declare_parameter('stop_variance_threshold', 0.05)
        
        self.W = self.get_parameter('width').value
        self.H = self.get_parameter('height').value
        self.res = self.get_parameter('resolution').value
        self.ell = self.get_parameter('ell').value
        self.sigma_f = self.get_parameter('sigma_f').value
        self.sigma_n = self.get_parameter('sigma_n').value
        # Override parameter with simple stopping value
        # self.max_samples = self.get_parameter('max_samples').value
        self.stop_variance_threshold = self.get_parameter('stop_variance_threshold').value
        
        self.xs, self.ys, self.X, self.Y, self.grid = make_grid(self.W, self.H, self.res)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.gp = TorchGP(self.ell, self.sigma_f, self.sigma_n, self.device)
        self.X_obs = np.empty((0, 2))
        self.y_obs = np.empty(0)
        self.pose_xy = None
        self.current_target = None
        self.first_wp_sent = False
        self.at_target = False
        self.done = False
        
        self.entropy_history = []
        self.mi_history = []
        
        # CSV logging
        self.csv_data = []
        
        self.out_dir = Path.home() / "workspaces" / "aquatic-mapping" / "src" / "info_gain" / "scripts" / "1"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple stopping: just use sample count
        self.max_samples = 40  # increased from 50
        
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.cb_odom, qos)
        self.sub_field = self.create_subscription(Float32, '/gaussian_field/radial/temperature', self.cb_measurement, 10)
        self.pub_wp = self.create_publisher(PoseStamped, '/mi/waypoint', 10)
        
        self.viz = PGLiveViz(W=self.W, H=self.H, xs=self.xs, ys=self.ys, title_suffix="Pure MI")
        self.viz.planner_id = 1
        self.viz.show()
        
        # ---- Keep color scales stable and meaningful ----
        # GP prior std ≤ sigma_f; MI max = 0.5 * ln(1 + sigma_f^2 / sigma_n^2)
        mi_max = 0.5 * np.log(1.0 + (self.sigma_f**2) / (self.sigma_n**2))
        self.viz.set_levels(
            mu=(20.0, 30.0),                # or whatever your expected temp range is
            std=(0.0, float(self.sigma_f)), # uncertainty in [0, sigma_f]
            mi=(0.0, float(mi_max)),        # MI upper bound from GP prior vs noise
            mi_title="MI"                   # label 3rd panel
        )
        
        self.get_logger().info(f"MI planner (Pure MI) | field [0,{self.W}]x[0,{self.H}] | max_samples={self.max_samples}")

    def cb_odom(self, msg):
        self.pose_xy = np.array([msg.position[0], msg.position[1]])
        
        if self.current_target is not None:
            dist = np.linalg.norm(self.pose_xy - self.current_target)
            if dist < 1.0:
                if not self.at_target:
                    self.at_target = True
                    self.get_logger().info(f"Arrived at ({self.current_target[0]:.1f},{self.current_target[1]:.1f})")
        
        if not self.first_wp_sent:
            self.first_wp_sent = True
            self.send_waypoint(0.0, 0.0)
            self.get_logger().info(f"Rover at ({self.pose_xy[0]:.2f},{self.pose_xy[1]:.2f}), sending to (0,0)")

    def cb_measurement(self, msg):
        if self.pose_xy is None or not self.at_target or self.done:
            return
        
        if len(self.X_obs) > 0 and np.linalg.norm(self.pose_xy - self.X_obs[-1]) < 0.5:
            return
        
        self.X_obs = np.vstack([self.X_obs, self.pose_xy])
        self.y_obs = np.append(self.y_obs, msg.data)
        self.get_logger().info(f"✓ Sample #{len(self.y_obs)}: pos ({self.pose_xy[0]:.2f},{self.pose_xy[1]:.2f}), T={msg.data:.1f}°C")
        
        self.at_target = False
        
        if len(self.y_obs) >= self.max_samples:
            self.get_logger().info(f"Reached max samples ({self.max_samples}). Stopping.")
            self.done = True
            self.save_csv()
            self.save_final()
            return
        
        self.plan_next()

    def plan_next(self):
        if len(self.y_obs) < 1:
            return
        
        self.gp.fit(self.X_obs, self.y_obs)
        mu, std = self.gp.predict(self.grid)
        var = std**2
        mean_var = float(np.mean(var))
        
        # Simple stopping: mean variance < threshold OR max samples
        if mean_var < 0.5 or len(self.y_obs) >= self.max_samples:
            reason = f"mean_var={mean_var:.3f} < 0.5" if mean_var < 0.5 else f"max_samples={self.max_samples}"
            self.get_logger().info(f"✓ Stopping: {reason}")
            self.done = True
            self.save_csv()
            self.save_final()
            return
        
        # Pure Krause MI (correlation-aware information gain)
        mi_gains = self.compute_krause_mi_gains(self.grid, n_subsample=150)
        
        valid = (self.grid[:, 0] >= 0) & (self.grid[:, 0] <= self.W) & (self.grid[:, 1] >= 0) & (self.grid[:, 1] <= self.H)
        mi_gains[~valid] = -np.inf
        
        best_idx = np.argmax(mi_gains)
        target = self.grid[best_idx]
        chosen_gain = float(mi_gains[best_idx])
        
        # Log to CSV
        self.csv_data.append({
            'sample': len(self.y_obs),
            'waypoint_x': target[0],
            'waypoint_y': target[1],
            'actual_x': self.pose_xy[0],
            'actual_y': self.pose_xy[1],
            'temperature': self.y_obs[-1],
            'krause_mi_gain': chosen_gain,
            'mean_variance': mean_var
        })
        
        self.send_waypoint(target[0], target[1])
        self.get_logger().info(f"→ Next: ({target[0]:.1f},{target[1]:.1f}), Krause_MI={chosen_gain:.3f}, mean_var={mean_var:.3f}")
        
        H_total = np.mean(0.5 * np.log(2 * np.pi * np.e * (var + self.sigma_n**2)))
        MI_mean = np.mean(mi_gains[np.isfinite(mi_gains)])
        self.entropy_history.append(H_total)
        self.mi_history.append(MI_mean)
        
        self.viz.push(mu, std, mi_gains, self.X_obs, self.current_target, self.y_obs, self.entropy_history, self.mi_history)

    def compute_krause_mi_gains(self, candidates, n_subsample=150):
        """
        Compute Krause MI gains using Cholesky decomposition on GPU.

        For each candidate x, computes:
            MI_gain(x) = H(Y_A ∪ {x}) - H(Y_A)
        where:
            H(Y) = 0.5 * [n*log(2πe) + log det(K_noisy)]
            log det(K) = 2 * sum(log(diag(L))) using Cholesky K = L L^T

        Args:
            candidates: Nx2 array of candidate positions
            n_subsample: Number of candidates to evaluate (random subset if more)

        Returns:
            Array of MI gains for each candidate
        """
        n_candidates = len(candidates)

        # Subsample candidates for computational efficiency
        if n_candidates > n_subsample:
            idx = np.random.choice(n_candidates, n_subsample, replace=False)
            candidates_sub = candidates[idx]
        else:
            idx = np.arange(n_candidates)
            candidates_sub = candidates

        n_eval = len(candidates_sub)
        mi_gains_sub = np.zeros(n_eval)

        # If no samples yet, use per-cell MI (equivalent for first point)
        if len(self.X_obs) == 0:
            _, std = self.gp.predict(candidates_sub)
            var = std**2
            mi_gains_sub = 0.5 * np.log(1 + var / self.sigma_n**2)

            if n_candidates > n_subsample:
                mi_gains = np.full(n_candidates, -np.inf)
                mi_gains[idx] = mi_gains_sub
                return mi_gains
            return mi_gains_sub

        # Move data to GPU
        X_obs_torch = torch.from_numpy(self.X_obs).float().to(self.device)
        candidates_torch = torch.from_numpy(candidates_sub).float().to(self.device)

        with torch.no_grad():
            # Compute H(A) for current observation set using Cholesky
            K_AA = self.gp.model.covar_module(X_obs_torch).evaluate()
            K_AA_noisy = K_AA + torch.eye(len(self.X_obs), device=self.device) * self.sigma_n**2

            try:
                L_A = torch.linalg.cholesky(K_AA_noisy)
                log_det_A = 2.0 * torch.sum(torch.log(torch.diag(L_A)))
                H_A = 0.5 * (len(self.X_obs) * np.log(2 * np.pi * np.e) + log_det_A.item())
            except RuntimeError as e:
                self.get_logger().warn(f"Cholesky failed for K_AA: {e}, adding jitter")
                K_AA_noisy += torch.eye(len(self.X_obs), device=self.device) * 1e-4
                L_A = torch.linalg.cholesky(K_AA_noisy)
                log_det_A = 2.0 * torch.sum(torch.log(torch.diag(L_A)))
                H_A = 0.5 * (len(self.X_obs) * np.log(2 * np.pi * np.e) + log_det_A.item())

            # Compute H(A ∪ {x}) for each candidate
            for i in range(n_eval):
                x_new = candidates_torch[i:i+1]
                X_aug = torch.cat([X_obs_torch, x_new], dim=0)

                K_aug = self.gp.model.covar_module(X_aug).evaluate()
                K_aug_noisy = K_aug + torch.eye(len(X_aug), device=self.device) * self.sigma_n**2

                try:
                    L_aug = torch.linalg.cholesky(K_aug_noisy)
                    log_det_aug = 2.0 * torch.sum(torch.log(torch.diag(L_aug)))
                    H_aug = 0.5 * (len(X_aug) * np.log(2 * np.pi * np.e) + log_det_aug.item())
                    mi_gains_sub[i] = H_aug - H_A
                except RuntimeError:
                    # Numerical issue - assign zero gain (no new information)
                    mi_gains_sub[i] = 0.0

        # Fill full array if subsampled
        if n_candidates > n_subsample:
            mi_gains = np.full(n_candidates, -np.inf)
            mi_gains[idx] = mi_gains_sub
            return mi_gains

        return mi_gains_sub

    def send_waypoint(self, x, y):
        wp = PoseStamped()
        wp.header.frame_id = 'world'
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.pose.position.x = float(x)
        wp.pose.position.y = float(y)
        yaw = float(np.arctan2(y - self.pose_xy[1], x - self.pose_xy[0])) if self.pose_xy is not None else 0.0
        wp.pose.orientation.z = np.sin(yaw / 2.0)
        wp.pose.orientation.w = np.cos(yaw / 2.0)
        self.pub_wp.publish(wp)
        self.current_target = np.array([x, y])
        self.at_target = False

    def save_csv(self):
        import csv
        csv_path = self.out_dir / "planner_1_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample', 'waypoint_x', 'waypoint_y', 'actual_x', 'actual_y', 'temperature', 'krause_mi_gain', 'mean_variance'])
            writer.writeheader()
            writer.writerows(self.csv_data)
        self.get_logger().info(f"CSV saved to {csv_path}")

    def save_final(self):
        # Always write the CSV once
        self.save_csv()

        # Recompute the final GP on the grid and save the exact field arrays
        if len(self.y_obs) > 0:
            self.gp.fit(self.X_obs, self.y_obs)
            mu, std = self.gp.predict(self.grid)
            
            # Save reconstructed field data
            out = self.out_dir / "reconstructed_field.npz"
            np.savez(
                out,
                mu=mu.reshape(self.Y.shape),
                std=std.reshape(self.Y.shape),
                xs=self.xs,
                ys=self.ys
            )
            self.get_logger().info(f"Saved reconstructed field to {out}")
            
            # Save final plots (separate files for fields and metrics)
            try:
                self.viz.save_figure(self.out_dir / "final_plots.png")
                self.get_logger().info(f"Saved final plots to {self.out_dir / 'final_fields.png'} and {self.out_dir / 'final_metrics.png'}")
            except Exception as e:
                self.get_logger().warn(f"Could not save plots: {e}")
        else:
            self.get_logger().warn("No samples collected; not saving reconstructed field.")

def main():
    rclpy.init()
    node = MIPlanner()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    try:
        node.viz.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
