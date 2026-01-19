# GP Reconstruction with Uncertain Inputs

Comparison of two methods for handling positional uncertainty in Gaussian Process regression:

1. **McHutchon & Rasmussen (2011) NIGP**: Gradient-based approximation treating input noise as heteroscedastic observation noise
2. **Girard et al. (2003)**: Monte Carlo marginalization at prediction time

## Methods

### McHutchon NIGP

- **Training**: EM-style iteration computing input-induced variance via gradients
- **Formula**: σ²_input,i = ∇μ(x_i)ᵀ Σ_i ∇μ(x_i)
- **Noise handling**: Heteroscedastic likelihood with per-sample effective noise
- **Reference**: McHutchon & Rasmussen, "Gaussian Process Training with Input Noise", NIPS 2011

### Girard Uncertain Input

- **Training**: Standard GP with deterministic commanded positions
- **Prediction**: Monte Carlo sampling from N(x_c, Σ_x) at each test point
- **Noise handling**: None during training; marginalization at prediction time
- **Reference**: Girard et al., "Gaussian Process Priors with Uncertain Inputs", 2003

## Structure

```
reconstruction/
├── mchutchon_nigp.py           # McHutchon NIGP implementation
├── girard_uncertain_input.py   # Girard uncertain-input implementation
├── run_reconstruction.py       # Main runner script
├── utils/
│   ├── data_loading.py         # CSV data loading
│   ├── ground_truth.py         # Field generation
│   └── metrics.py              # Metrics computation
├── venv/                       # Python virtual environment (symlink)
└── requirements.txt            # Dependencies
```

## Setup

```bash
# activate virtual environment
cd reconstruction
source venv/bin/activate

# verify dependencies (already installed if using symlink)
pip install -r requirements.txt
```

## Usage

### Run single field with both methods

```bash
source venv/bin/activate
python run_reconstruction.py radial 1
```

### Run specific method only

```bash
# McHutchon NIGP only
python run_reconstruction.py radial 1 mchutchon

# Girard only
python run_reconstruction.py radial 1 girard
```

### Run all fields

```bash
python run_reconstruction.py all 1
```

## Arguments

```
python run_reconstruction.py <field_type> [trial_number] [method]

field_type:    radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt, all
trial_number:  default=1
method:        mchutchon, girard, both (default=both)
```

## Output Organization

Results are saved in method-specific directories:

```
results/
  trial_1/
    mchutchon_nigp/
      radial/
        rbf/
          radial_rbf_mchutchon_nigp.png
          radial_rbf_metrics.csv
          radial_rbf_hyperparams.json
        exponential/
          ...
        matern15/
          ...
        matern25/
          ...
      x_compress/
        ...
    girard/
      radial/
        rbf/
          radial_rbf_girard.png
          radial_rbf_metrics.csv
          radial_rbf_hyperparams.json
        ...
      ...
```

## Visualization

### McHutchon NIGP

3-panel plot:
- Ground truth with sample locations
- Predicted mean
- Error (prediction - truth)

### Girard

4-panel plot:
- Ground truth with sample locations
- Predicted mean
- Error (prediction - truth)
- Predicted variance

## Metrics

Both methods output CSV with:
- MSE: Mean squared error
- RMSE: Root mean squared error
- MAE: Mean absolute error
- NRMSE: Normalized RMSE

## Hyperparameters

JSON files contain learned parameters:

**McHutchon NIGP**:
- lengthscale
- outputscale
- mean_constant
- learned_noise
- input_induced_variance_mean
- input_induced_variance_max

**Girard**:
- lengthscale
- outputscale
- mean_constant
- noise

## Kernels

Both methods support:
1. **RBF** (Squared Exponential): Smooth, infinitely differentiable
2. **Exponential** (Matern-0.5): Rough, continuous but not differentiable
3. **Matern-1.5**: Once differentiable
4. **Matern-2.5**: Twice differentiable

## Hardware

- CUDA GPU recommended (tested on RTX 4070 Ti SUPER)
- CPU fallback available
- ~2GB GPU memory for typical field sizes
