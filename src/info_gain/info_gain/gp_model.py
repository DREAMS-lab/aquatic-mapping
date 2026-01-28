"""
GPyTorch-based Gaussian Process model for informative sampling

Standard GP with RBF kernel, GPU-accelerated batch predictions.
"""

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


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
    - Fixed hyperparameters (no re-optimization during sampling)
    """

    def __init__(self, lengthscale=2.0, signal_var=1.0, noise_var=0.01):
        """
        Initialize GP with fixed hyperparameters

        Args:
            lengthscale: RBF kernel lengthscale (in meters for 2D field)
            signal_var: signal variance (outputscale)
            noise_var: observation noise variance
        """
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var

        self.train_x = None
        self.train_y = None
        self.model = None
        self.likelihood = None

    def fit(self, X, y):
        """
        Fit GP to training data (no hyperparameter optimization)

        Args:
            X: (N, D) training inputs
            y: (N,) training targets
        """
        self.train_x = torch.tensor(X, dtype=torch.float32).to(device)
        self.train_y = torch.tensor(y, dtype=torch.float32).to(device)

        self.likelihood = GaussianLikelihood().to(device)
        self.likelihood.noise = self.noise_var

        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device)

        # Set fixed hyperparameters
        self.model.covar_module.base_kernel.lengthscale = self.lengthscale
        self.model.covar_module.outputscale = self.signal_var

        # Set to eval mode (no training)
        self.model.eval()
        self.likelihood.eval()

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

        # Refit with updated data
        self.likelihood = GaussianLikelihood().to(device)
        self.likelihood.noise = self.noise_var

        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device)
        self.model.covar_module.base_kernel.lengthscale = self.lengthscale
        self.model.covar_module.outputscale = self.signal_var

        self.model.eval()
        self.likelihood.eval()

    def get_training_data(self):
        """Return current training data"""
        if self.train_x is None:
            return None, None
        return self.train_x.cpu().numpy(), self.train_y.cpu().numpy()

    @property
    def n_observations(self):
        """Number of observations in training set"""
        return 0 if self.train_x is None else len(self.train_x)
