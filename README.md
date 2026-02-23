# MICE: Multi-Iteration stochastiC Estimator

[![PyPI version](https://badge.fury.io/py/mice.svg)](https://badge.fury.io/py/mice)
[![Documentation](https://readthedocs.org/projects/mice/badge/?version=latest)](https://mice.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

MICE is a gradient estimator for stochastic optimization that uses successive control variates along the optimization path to reduce variance. By adaptively selecting which iterates to include in its index set and optimally distributing samples, MICE achieves accurate mean gradient estimation at minimal computational cost.

## Key Features

- **Adaptive variance reduction**: Controls relative L² error with user-specified tolerance ε
- **Efficient sample allocation**: Minimizes gradient sampling cost subject to error constraints
- **Index-set operators**: Add, Drop, Restart, and Clip operations for optimal hierarchy management
- **Flexible integration**: Non-intrusive design couples seamlessly with SGD, Adam, and other optimizers
- **Dual problem support**: Handles both expectation minimization and finite-sum problems
- **Robust stopping**: Resampling-based gradient norm estimation for stable termination criteria

## Theoretical Performance

For smooth, strongly convex problems, SGD-MICE achieves a gradient evaluation complexity of **O(tol⁻¹)** to reach tolerance `tol`, compared to **O(tol⁻¹ log(tol⁻¹))** for standard adaptive batch-size SGD.

## Installation

```bash
pip install mice
```

For development or to run experiments:

```bash
git clone https://github.com/agcarlon/mice.git
cd mice
pip install -e .
```

## Quick Start

```python
import numpy as np
from mice import MICE
from mice.policy import DropRestartClipPolicy

# Define gradient function: grad(x, thetas) -> gradients array
def gradient(x, thetas):
    """Compute gradients for batch of samples."""
    return x - thetas  # Simple example: minimize E[(x - θ)²]

# Define sampler: sampler(n) -> batch of n samples
def sampler(n):
    return np.random.randn(n, 1)

# Create MICE estimator
estimator = MICE(
    grad=gradient,
    sampler=sampler,
    eps=0.577,              # Relative error tolerance (1/√3)
    min_batch=10,
    policy=DropRestartClipPolicy(
        drop_param=0.5,
        restart_param=0.0,
        max_hierarchy_size=100
    ),
    max_cost=10000,         # Maximum gradient evaluations
    stop_crit_norm=1e-6,    # Stopping criterion
)

# Use in optimization loop
x = np.array([10.0])
for iteration in range(100):
    grad_estimate = estimator(x)
    x = x - 0.1 * grad_estimate  # Gradient descent step
    print(f"Iteration {iteration}: x = {x[0]:.6f}")
```

## Advanced Features

### Finite-Sum Problems

For finite datasets (empirical risk minimization):

```python
# Load your dataset
X_train = ...  # Training features
y_train = ...  # Training labels
data = np.column_stack([y_train, X_train])

# MICE automatically handles finite sampling
estimator = MICE(
    grad=your_gradient_function,
    sampler=data,  # Pass data directly
    eps=0.577,
    # ... other parameters
)
```

### Policy Configuration

Control index-set management with `DropRestartClipPolicy`:

```python
from mice.policy import DropRestartClipPolicy

policy = DropRestartClipPolicy(
    drop_param=0.5,           # Threshold for dropping last iterate
    restart_param=0.0,        # Threshold for restarting hierarchy
    max_hierarchy_size=100,   # Maximum |L_k|
    clip_type="full",         # Clipping strategy ("full", "all", or None)
    aggr_cost=0.1,           # Aggregation cost factor
)

estimator = MICE(grad=..., sampler=..., policy=policy)
```

### Resampling-Based Norm Estimation

Enable robust norm estimation for sizing and stopping:

```python
estimator = MICE(
    grad=gradient,
    sampler=sampler,
    use_resampling=True,
    re_part=5,              # Number of jackknife partitions
    re_quantile=0.05,       # Quantile for tolerance
    re_tot_cost=0.2,        # Resampling cost budget
    # ... other parameters
)
```

## API Reference

### `MICE`

Main estimator class.

**Parameters:**
- `grad` (callable): Gradient function with signature `grad(x: ndarray, thetas: Any) -> ndarray`
- `sampler` (callable or array): Sampler function `sampler(n: int) -> Any` or finite dataset
- `eps` (float): Relative error tolerance parameter (default: 0.577)
- `min_batch` (int): Minimum batch size (default: 10)
- `restart_factor` (int): Restart batch multiplier (default: 10)
- `max_cost` (float): Maximum gradient evaluations (default: inf)
- `stop_crit_norm` (float): Stopping criterion for gradient norm (default: 0.0)
- `stop_crit_prob` (float): Stopping criterion probability (default: 0.05)
- `convex` (bool): Whether problem is convex (default: False)
- `policy` (DropRestartClipPolicy): Index-set management policy
- `use_resampling` (bool): Enable resampling-based norm estimation (default: True)
- `recorder` (Recorder): Optional event recorder for diagnostics

**Methods:**
- `evaluate(x: ndarray) -> ndarray`: Evaluate MICE gradient estimate at point x
- `__call__(x: ndarray) -> ndarray`: Alias for evaluate
- `get_log() -> list`: Return recorded events (if recorder enabled)

## Reproducible Experiments

The repository includes all numerical experiments from the manuscript "Multi-Iteration Stochastic Optimizers". See [`experiments/README.md`](experiments/README.md) for detailed instructions on:

- Running operator ablations and sensitivity sweeps (quadratic benchmarks)
- Training logistic regression on mushrooms, gisette, and HIGGS datasets
- Generating all figures and tables from the paper

## Citation

If you use MICE in your research, please cite:

```bibtex
@article{carlon2025mice,
  title={Multi-Iteration Stochastic Optimizers},
  author={Carlon, Andr{\'e} and Espath, Luis and Holdorf, Rafael and Tempone, Ra{\'u}l},
  journal={Applied Mathematics \& Optimization},
  year={2025},
  note={Manuscript ID: AMOP-D-25-00161}
}
```

Preprint: [arXiv:2011.01718](https://arxiv.org/abs/2011.01718)

## Documentation

Full documentation available at [mice.readthedocs.io](https://mice.readthedocs.io)

Build docs locally:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e .
sphinx-build -b html docs docs/_build/html
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- André Carlon (RWTH Aachen University)
- Luis Espath (University of Nottingham)
- Rafael Holdorf (Federal University of Santa Catarina)
- Raúl Tempone (KAUST & RWTH Aachen University)
