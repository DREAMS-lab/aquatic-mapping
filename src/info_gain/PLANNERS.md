# Informative Path Planners

## Overview

Three different approaches to informative path planning under uncertainty:

| Planner | GP Model | Acquisition Function | Uncertainty Consideration |
|---------|----------|---------------------|---------------------------|
| `exact_planner.py` | Standard (exact inputs) | Δ(x) | None |
| `pose_aware_planner.py` | Standard (exact inputs) | U(x) = E[Δ(x̃)] | Position noise in planner |
| `model_aware_planner.py` | Uncertain inputs | Δ(x) | Position noise in GP [TODO] |
| `both_aware_planner.py` | Uncertain inputs | U(x) = E[Δ(x̃)] | Both GP and planner [TODO] |

---

## 1. Exact Planner (Baseline)

**File:** `scripts/exact_planner.py`
**Data path:** `data/trials/exact/`

### Mathematical Formulation:

```
psi* = argmax_psi  SUM_{x in S(psi)} Δ(x)  -  λ * SUM c(x_t, x_{t+1})

where:
  Δ(x) = (1/2) * log(1 + σ²(x) / σ_n²)    [information gain]
  σ²(x) = GP posterior variance at x
```

### Key Properties:
- **GP:** Standard GP with exact input locations
- **Planning:** Direct information gain (no uncertainty consideration)
- **Assumption:** Robot reaches commanded positions exactly

---

## 2. Pose-Aware Planner (Position-Uncertain Planning)

**File:** `scripts/pose_aware_planner.py`
**Data path:** `data/trials/pose_aware/`

### Mathematical Formulation:

**Step 1 — Commanded vs actual location:**
```
x̃_t = x_t + ξ_t,  ξ_t ~ N(0, Σ_x)
```
You command x_t, but actually sample at x̃_t

**Step 2 — Expected information gain:**
```
U(x) = E_{x̃ ~ N(x, Σ_x)} [Δ(x̃)]
     ≈ (1/M) Σ_{j=1}^M (1/2) * log(1 + σ²(x + ξ_j) / σ_n²)
```

**Step 3 — Path objective:**
```
psi* = argmax_psi  SUM U(x_t)  -  λ * SUM c(x_t, x_{t+1})
```

### Key Properties:
- **GP:** Standard GP (exact inputs) - same as exact planner
- **Planning:** Expected info gain under position noise (Monte Carlo integration)
- **Assumption:** Robot position has Gaussian noise when sampling
- **Parameters:**
  - `position_std`: Position uncertainty (meters), default 0.5m
  - `n_mc_samples`: Monte Carlo samples for expectation, default 30

### Implementation Details:

The key difference from exact planner is the acquisition function:

```python
# Exact planner:
def _greedy_single_step(self, current_pos):
    _, variances = self.gp.predict(candidates)
    info_gains = (1/2) * log(1 + variances / noise_var)
    scores = info_gains - lambda * travel_costs

# Pose-aware planner:
def _greedy_single_step(self, current_pos):
    expected_info_gains = []
    for x_candidate in candidates:
        # Monte Carlo integration
        info_sum = 0
        for _ in range(n_mc_samples):
            xi = sample_from_N(0, Σ_x)
            x_noisy = x_candidate + xi
            _, var = gp.predict(x_noisy)
            info_sum += (1/2) * log(1 + var / noise_var)
        expected_info_gains.append(info_sum / n_mc_samples)
    scores = expected_info_gains - lambda * travel_costs
```

---

## 3. Model-Aware Planner [TODO]

**File:** `scripts/model_aware_planner.py`
**Data path:** `data/trials/model_aware/`

Will use GP with uncertain inputs (heteroscedastic noise or uncertain input GP).

---

## 4. Both-Aware Planner [TODO]

**File:** `scripts/both_aware_planner.py`
**Data path:** `data/trials/both_aware/`

Will combine uncertain-input GP with expected information gain planning.

---

## Usage

### Exact Planner:
```bash
ros2 run info_gain exact_planner.py --ros-args \
    -p field_type:=radial \
    -p trial:=1 \
    -p horizon:=2
```

### Pose-Aware Planner:
```bash
ros2 run info_gain pose_aware_planner.py --ros-args \
    -p field_type:=radial \
    -p trial:=1 \
    -p horizon:=2 \
    -p position_std:=0.5 \
    -p n_mc_samples:=30
```

---

## Comparison

To compare performance:

1. Run both planners on same field with same trial number
2. Compare reconstruction metrics in `summary.json`:
   - `reconstruction_rmse`: How well field is reconstructed
   - `cumulative_info_gain`: Total information gathered
   - `total_travel_cost`: Distance traveled
3. Visualize: `figures/reconstruction_comparison.png`, `figures/convex_hull.png`

Expected result: Pose-aware planner should achieve better reconstruction under real position noise.
