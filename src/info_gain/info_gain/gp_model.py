"""
GPyTorch-based Gaussian Process model for informative sampling

Standard GP with RBF kernel, GPU-accelerated batch predictions.
Supports optional online MAP hyperparameter optimization with
log-normal prior on lengthscale to prevent l→l_max collapse.
"""

import math
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExactGPModel(ExactGP):
    """Standard GP with RBF kernel"""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class GPModel:
    """
    Wrapper for GPyTorch GP with convenient interface for informative sampling

    Features:
    - GPU acceleration
    - Batch predictions
    - Incremental observation updates
    - Optional online MAP hyperparameter optimization with log-normal prior
    """

    def __init__(self, lengthscale=2.0, signal_var=1.0, noise_var=0.01,
                 optimize_every=10, optimize_steps=20):
        """
        Initialize GP with hyperparameters

        Args:
            lengthscale: RBF kernel lengthscale (in meters for 2D field)
            signal_var: signal variance (outputscale)
            noise_var: observation noise variance
            optimize_every: optimize hyperparameters every N observations (0=disabled)
            optimize_steps: gradient steps per optimization round
        """
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.optimize_every = optimize_every
        self.optimize_steps = optimize_steps

        # Initial values for prior
        self._l_init = lengthscale
        self._sv_init = signal_var

        # Persistent learned values (survive model rebuilds)
        self._learned_lengthscale = lengthscale
        self._learned_signal_var = signal_var
        self._learned_mean = 0.0

        # Optimization schedule
        self._obs_since_last_opt = 0
        self._total_opt_count = 0

        self.train_x = None
        self.train_y = None
        self.model = None
        self.likelihood = None

    def fit(self, X, y):
        """
        Fit GP to training data

        Args:
            X: (N, D) training inputs
            y: (N,) training targets
        """
        self.train_x = torch.tensor(X, dtype=torch.float32).to(device)
        self.train_y = torch.tensor(y, dtype=torch.float32).to(device)

        self._rebuild_model()

    def predict(self, X_test):
        """
        Predict mean and variance at test points (batched, GPU)

        Args:
            X_test: (M, D) test inputs

        Returns:
            mean: (M,) posterior mean
            var: (M,) posterior variance
        """
        if self.model is None:
            # No training data yet - return prior
            M = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
            return torch.zeros(M).to(device), torch.ones(M).to(device) * self.signal_var

        X_test = torch.tensor(X_test, dtype=torch.float32).to(device) if not isinstance(X_test, torch.Tensor) else X_test.to(device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(X_test)
            mean = pred.mean
            var = pred.variance

        return mean, var

    def _rebuild_model(self):
        """Rebuild GP model with current training data and learned hyperparameters."""
        self.likelihood = GaussianLikelihood().to(device)
        self.likelihood.noise = self.noise_var

        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device)
        self.model.covar_module.base_kernel.lengthscale = self._learned_lengthscale
        self.model.covar_module.outputscale = self._learned_signal_var
        self.model.mean_module.constant.data.fill_(self._learned_mean)

        self.model.eval()
        self.likelihood.eval()

    def add_observation(self, x_new, y_new):
        """
        Add a new observation and refit GP

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
        """Check if it's time to optimize hyperparameters."""
        return (self.optimize_every > 0 and
                self._obs_since_last_opt >= self.optimize_every and
                self.train_x is not None and
                len(self.train_x) >= 10)

    def optimize_hyperparameters(self, n_steps=None, logger=None):
        """
        Optimize GP hyperparameters via MAP (MLL + log-normal prior).

        Optimizes:
        - RBF lengthscale (1 param)
        - Signal variance / outputscale (1 param)
        - Mean constant (1 param)

        Noise variance is kept FIXED (known from sensor spec).

        The log-normal prior on lengthscale prevents the l→l_max feedback
        loop where MLL always prefers smoother fits, crushing variance
        contrast needed by the acquisition function.

        Args:
            n_steps: Number of gradient steps (default: self.optimize_steps)
            logger: Optional ROS logger for progress messages

        Returns:
            final_loss: Final MAP loss value
        """
        if self.train_x is None or len(self.train_x) < 10:
            return None

        n_steps = n_steps or self.optimize_steps

        # Training mode for optimization
        self.model.train()
        self.likelihood.train()

        # Optimize kernel params + mean, noise stays fixed
        optimizer = torch.optim.Adam([
            {'params': self.model.covar_module.base_kernel.raw_lengthscale, 'lr': 0.05},
            {'params': self.model.covar_module.raw_outputscale, 'lr': 0.05},
            {'params': [self.model.mean_module.constant], 'lr': 0.1},
        ])
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Log-normal prior on lengthscale centered at l_init
        # Prevents l→l_max collapse while allowing adaptation
        mu_prior = math.log(self._l_init)
        tau_sq = 1.5

        best_loss = float('inf')
        for i in range(n_steps):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)

            # MAP prior: penalize lengthscale deviating from l_init
            current_l = self.model.covar_module.base_kernel.lengthscale.squeeze()
            prior_penalty = (torch.log(current_l) - mu_prior).pow(2) / (2 * tau_sq)
            loss = loss + prior_penalty

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val

        # Back to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Persist learned values for future model rebuilds
        self._learned_lengthscale = self.model.covar_module.base_kernel.lengthscale.item()
        self._learned_signal_var = self.model.covar_module.outputscale.item()
        self._learned_mean = self.model.mean_module.constant.item()

        # Reset counter
        self._obs_since_last_opt = 0
        self._total_opt_count += 1

        if logger:
            logger.info(
                f'GP optimized (round {self._total_opt_count}): '
                f'loss={best_loss:.3f}, '
                f'l={self._learned_lengthscale:.3f}, '
                f'signal_var={self._learned_signal_var:.3f}, '
                f'mean={self._learned_mean:.2f}'
            )

        return best_loss

    def get_training_data(self):
        """Return current training data"""
        if self.train_x is None:
            return None, None
        return self.train_x.cpu().numpy(), self.train_y.cpu().numpy()

    @property
    def n_observations(self):
        """Number of observations in training set"""
        return 0 if self.train_x is None else len(self.train_x)
