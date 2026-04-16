"""
GP Model with Gibbs Non-Stationary Kernel

Wraps the anisotropic Paciorek kernel into a GPyTorch ExactGP model with the
same interface as info_gain.gp_model.GPModel, plus:
- Periodic hyperparameter optimization via marginal likelihood
- Lengthscale field snapshots for visualization

The key difference from the stationary GP:
- The kernel has spatially varying anisotropic covariance Sigma(x)
- Three fields: l1(x), l2(x) (lengthscales) and theta(x) (rotation)
- The parameters are learned online from data
- This causes the GP to allocate different effective resolution
  across the domain, concentrating samples near complex regions
"""

import math
import torch
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

from .gibbs_kernel import GibbsKernel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GibbsGPModel(ExactGP):
    """ExactGP with Gibbs non-stationary kernel."""

    def __init__(self, train_x, train_y, likelihood, gibbs_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = gibbs_kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class NonstationaryGPModel:
    """
    GP with anisotropic Paciorek non-stationary kernel.

    Features:
    - Spatially varying anisotropic covariance via Gibbs kernel
    - Three learned fields: l1(x), l2(x), theta(x)
    - Periodic online optimization of kernel parameters
    - Saves lengthscale field snapshots for thesis visualization
    - GPU acceleration

    Interface (compatible with info_gain.gp_model.GPModel):
    - fit(X, y)
    - predict(X_test)
    - add_observation(x_new, y_new)
    - get_training_data()
    - n_observations property

    Additional:
    - optimize_lengthscale(n_steps): Optimize basis weights via marginal likelihood
    - get_lengthscale_field(): Get current l1, l2, theta maps
    - save_lengthscale_snapshot(path, step): Save fields for visualization
    """

    def __init__(self, noise_var=0.36, signal_var=1.0, l_init=2.0,
                 grid_size=5, l_min=0.5, l_max=5.0,
                 optimize_every=10, optimize_steps=20):
        """
        Args:
            noise_var: Observation noise variance
            signal_var: Signal variance (outputscale)
            l_init: Initial lengthscale (uniform)
            grid_size: Basis grid size (grid_size^2 basis functions)
            l_min: Minimum lengthscale
            l_max: Maximum lengthscale
            optimize_every: Optimize lengthscale every N observations
            optimize_steps: Gradient steps per optimization
        """
        self.noise_var = noise_var
        self.signal_var = signal_var
        self.l_init = l_init
        self.grid_size = grid_size
        self.l_min = l_min
        self.l_max = l_max
        self.optimize_every = optimize_every
        self.optimize_steps = optimize_steps

        self.train_x = None
        self.train_y = None
        self.model = None
        self.likelihood = None

        # Create a persistent kernel that carries learned weights across refits
        self.gibbs_kernel = GibbsKernel(
            grid_size=grid_size, l_min=l_min, l_max=l_max,
            l_init=l_init, signal_var=signal_var
        ).to(device)

        # Persistent learned mean (survives model rebuilds)
        self._learned_mean = 0.0

        # Track optimization schedule
        self._obs_since_last_opt = 0
        self._total_opt_count = 0

    def fit(self, X, y):
        """
        Fit GP to training data.

        Args:
            X: (N, D) training inputs
            y: (N,) training targets
        """
        self.train_x = torch.tensor(X, dtype=torch.float32).to(device)
        self.train_y = torch.tensor(y, dtype=torch.float32).to(device)

        self._rebuild_model()

    def _rebuild_model(self):
        """Rebuild GP model with current training data and kernel weights."""
        self.likelihood = GaussianLikelihood().to(device)
        self.likelihood.noise = self.noise_var

        self.model = GibbsGPModel(
            self.train_x, self.train_y,
            self.likelihood, self.gibbs_kernel
        ).to(device)

        # Restore learned mean from previous optimization rounds
        self.model.mean_module.constant.data.fill_(self._learned_mean)

        self.model.eval()
        self.likelihood.eval()

    def predict(self, X_test):
        """
        Predict mean and variance at test points.

        Args:
            X_test: (M, D) test inputs

        Returns:
            mean: (M,) posterior mean
            var: (M,) posterior variance
        """
        if self.model is None:
            M = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
            return torch.zeros(M).to(device), torch.ones(M).to(device) * self.gibbs_kernel.signal_var

        X_test = torch.tensor(X_test, dtype=torch.float32).to(device) if not isinstance(X_test, torch.Tensor) else X_test.to(device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(X_test)
            mean = pred.mean
            var = pred.variance

        return mean, var

    def add_observation(self, x_new, y_new):
        """
        Add a new observation and refit GP.

        Args:
            x_new: (D,) new input location
            y_new: scalar observation
        """
        x_new = torch.tensor(x_new, dtype=torch.float32).reshape(1, -1).to(device)
        y_new = torch.tensor([y_new], dtype=torch.float32).to(device)

        if self.train_x is None:
            self.train_x = x_new
            self.train_y = y_new
        else:
            self.train_x = torch.cat([self.train_x, x_new], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)

        self._obs_since_last_opt += 1
        self._rebuild_model()

    def should_optimize(self):
        """Check if it's time to optimize lengthscale parameters."""
        return (self._obs_since_last_opt >= self.optimize_every and
                self.train_x is not None and
                len(self.train_x) >= 10)  # Need minimum data

    def optimize_lengthscale(self, n_steps=None, logger=None):
        """
        Optimize GP hyperparameters via marginal likelihood.

        Optimizes:
        - Gibbs kernel basis_weights_l1 (25 params) - lengthscale field 1
        - Gibbs kernel basis_weights_l2 (25 params) - lengthscale field 2
        - Gibbs kernel basis_weights_theta (25 params) - rotation angle field
        - Gibbs kernel signal variance (1 param) - overall kernel amplitude
        - GP mean constant (1 param) - data offset

        Noise variance is kept FIXED (known from sensor spec).

        Args:
            n_steps: Number of gradient steps (default: self.optimize_steps)
            logger: Optional ROS logger for progress messages

        Returns:
            final_loss: Final negative log marginal likelihood
        """
        if self.train_x is None or len(self.train_x) < 10:
            return None

        n_steps = n_steps or self.optimize_steps

        # Put model in training mode for optimization
        self.model.train()
        self.likelihood.train()

        # MAP optimization (Xu & Choi 2011): MLL + log-normal prior on lengthscales.
        # The prior prevents the "l → l_max" feedback loop by penalizing large
        # lengthscales. Without it, MLL always prefers smoother fits, which crushes
        # variance and kills the planner's ability to distinguish explored/unexplored.
        # Noise variance stays fixed (known sensor spec).
        optimizer = torch.optim.Adam([
            {'params': [self.gibbs_kernel.basis_weights_l1], 'lr': 0.05},
            {'params': [self.gibbs_kernel.basis_weights_l2], 'lr': 0.05},
            {'params': [self.gibbs_kernel.basis_weights_theta], 'lr': 0.02},
            {'params': [self.gibbs_kernel._log_signal_var], 'lr': 0.05},
            {'params': [self.model.mean_module.constant], 'lr': 0.1},
        ])
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Prior parameters: log-normal centered on l_init
        # tau_sq controls prior spread: smaller = tighter, larger = more freedom.
        # With K=25 basis points per field (50 terms total), penalty scales as
        # K * (log(l/l_init))^2 / (2*tau_sq). At tau_sq=1.5, doubling the
        # lengthscale costs ~4.0 total — enough to weakly regularize without
        # preventing the kernel from learning useful spatial variation.
        mu_prior = math.log(self.l_init)
        tau_sq = 1.5
        prior_points = self.gibbs_kernel.basis_centers  # (K, 2) fixed grid

        best_loss = float('inf')
        for i in range(n_steps):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)

            # MAP prior: penalize lengthscales deviating from l_init
            l1_p, l2_p = self.gibbs_kernel._lengthscales_at(prior_points)
            prior_penalty = (
                (torch.log(l1_p) - mu_prior).pow(2).sum() +
                (torch.log(l2_p) - mu_prior).pow(2).sum()
            ) / (2 * tau_sq)
            loss = loss + prior_penalty

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val

        # Back to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Persist learned mean for future model rebuilds
        self._learned_mean = self.model.mean_module.constant.item()

        # Reset counter
        self._obs_since_last_opt = 0
        self._total_opt_count += 1

        if logger:
            with torch.no_grad():
                l1, l2 = self.gibbs_kernel._lengthscales_at(self.train_x)
                theta = self.gibbs_kernel._theta_at(self.train_x)
                aniso = l1 / l2
            sig_var = self.gibbs_kernel.signal_var
            learned_mean = self._learned_mean
            logger.info(
                f'GP optimized (round {self._total_opt_count}): '
                f'loss={best_loss:.3f}, '
                f'l1=[{l1.min():.2f},{l1.max():.2f}], '
                f'l2=[{l2.min():.2f},{l2.max():.2f}], '
                f'theta=[{theta.min():.2f},{theta.max():.2f}], '
                f'aniso={aniso.mean():.2f}, '
                f'signal_var={sig_var:.2f}, mean={learned_mean:.2f}'
            )

        return best_loss

    def get_training_data(self):
        """Return current training data."""
        if self.train_x is None:
            return None, None
        return self.train_x.cpu().numpy(), self.train_y.cpu().numpy()

    @property
    def n_observations(self):
        """Number of observations in training set."""
        return 0 if self.train_x is None else len(self.train_x)

    def get_lengthscale_field(self, resolution=0.5):
        """Get current lengthscale field for visualization."""
        return self.gibbs_kernel.get_lengthscale_field(resolution=resolution)

    def save_lengthscale_snapshot(self, output_dir, step):
        """
        Save lengthscale field snapshot for thesis visualization.

        Args:
            output_dir: Path to save directory
            step: Current sample step number
        """
        X, Y, L1, L2, Theta = self.get_lengthscale_field()
        weights_l1 = self.gibbs_kernel.basis_weights_l1.detach().cpu().numpy()
        weights_l2 = self.gibbs_kernel.basis_weights_l2.detach().cpu().numpy()
        weights_theta = self.gibbs_kernel.basis_weights_theta.detach().cpu().numpy()

        np.savez(
            output_dir / f'lengthscale_step_{step:03d}.npz',
            X=X, Y=Y, L1=L1, L2=L2, Theta=Theta,
            basis_weights_l1=weights_l1,
            basis_weights_l2=weights_l2,
            basis_weights_theta=weights_theta,
            step=step,
            n_observations=self.n_observations
        )
