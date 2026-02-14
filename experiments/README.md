# MICE Numerical Experiments

This directory contains all numerical experiments from the manuscript:
**"Multi-Iteration Stochastic Optimizers"** (AMOP-D-25-00161)

## Setup

1. Install the MICE package (from repository root):
   ```bash
   cd ..
   pip install -e .
   ```

2. Install experiment dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "from mice.core_impl import MICE; print('MICE installed successfully')"
   ```

## Quick Start: Generate All Paper Figures

```bash
# Generate all figures and tables (full manuscript settings)
python generate_all_figures.py

# Quick test mode (reduced budget/runs)
python generate_all_figures.py --quick

# Skip time-intensive logistic regression
python generate_all_figures.py --skip-logreg
```

## Quadratic Experiments

All experiments use a random quadratic function: `f(ξ,θ) = (1/2)ξ·H(θ)ξ - b·ξ`
with stochastic Hessian `H(θ)`.

### Table 2: Operator Ablation

Tests the contribution of each operator (Add, Drop, Restart, Clip):

```bash
python -m experiments.quadratic.run_ablations --budget 100000 --runs 50
```

**Output:** `quadratic_operator_ablation.csv`

**Configurations tested:**
- Add only (no Drop/Restart/Clip)
- Add+Drop
- Add+Drop+Restart
- All+Clip

### Table 3: Epsilon (ε) Sensitivity

Sweeps relative-error tolerance ε ∈ {0.3, 0.5, 0.577, 0.7, 1.0}:

```bash
python -m experiments.quadratic.run_epsilon_sweep --budget 100000 --runs 50
```

**Output:** `quadratic_epsilon_sensitivity.csv`

### Table 4: Delta (δ) Sensitivity

Sweeps drop and restart thresholds:

```bash
python -m experiments.quadratic.run_delta_sweep --budget 100000 --runs 50
```

**Outputs:**
- `quadratic_delta_drop_sensitivity.csv` (δ_drop sweep, δ_rest=0)
- `quadratic_delta_rest_sensitivity.csv` (δ_rest sweep, δ_drop=0.5)

### Table 5: Max Index-Set Size Sensitivity

Sweeps maximum hierarchy cardinality max|L_k| ∈ {100, 500, 1000}:

```bash
python -m experiments.quadratic.run_max_index_sweep --budget 100000 --runs 50
```

**Output:** `quadratic_max_index_sensitivity.csv`

### Figure 7: Runtime Benchmark

Breaks down MICE runtime into components (gradient eval, variance estimation, resampling, operators):

```bash
# Single dimension
python -m experiments.quadratic.run_benchmarks --dims 100 --runs 50

# Dimension sweep
python -m experiments.quadratic.run_benchmarks --dims "10,50,100,500,1000" --runs 50
```

**Output:** `overhead_pct_vs_dim.pdf`, `overhead_by_dim.csv`

### Figure 5: Condition Number (κ) Sweep

Tests scaling with condition number κ ∈ {10, 30, 100, 300, 1000}:

```bash
python -m experiments.quadratic.run_kappa_sweep --seed 0
```

**Output:** Figures showing convergence vs κ (varies by script implementation)

## Logistic Regression Experiments

Trains logistic regression on three LIBSVM datasets with different characteristics:

| Dataset    | Size       | Features | λ      | max\|L_k\| |
|------------|------------|----------|--------|------------|
| mushrooms  | 8,124      | 112      | 10⁻⁵   | 100        |
| gisette    | 6,000      | 5,000    | 10⁻⁴   | 100        |
| higgs      | 11,000,000 | 28       | 10⁻⁴   | **1000**   |

**Note:** HIGGS uses larger `max_hierarchy_size=1000` due to smaller dimensionality (manuscript line 2875).

### Step 1: Prepare Datasets

Datasets auto-download on first run, but you can pre-fetch:

```bash
python -m experiments.logistic_regression.datasets --dataset mushrooms
python -m experiments.logistic_regression.datasets --dataset gisette
python -m experiments.logistic_regression.datasets --dataset higgs
```

**Note:** HIGGS dataset is ~11M samples (~2.5 GB).

### Step 2: Compute Reference Optimum

Required before running comparisons:

```bash
python -m experiments.logistic_regression.compute_optimum --dataset mushrooms
python -m experiments.logistic_regression.compute_optimum --dataset gisette
python -m experiments.logistic_regression.compute_optimum --dataset higgs
```

**Output:** `<dataset>/Optimum.npy` (used to compute relative optimality gap)

### Step 3: Run SGD-MICE

Single seed:

```bash
python -m experiments.logistic_regression.run_logreg --dataset mushrooms --seed 0
```

The script automatically uses the correct `max_hierarchy_size` from `configs.py`:
- mushrooms: 100
- gisette: 100
- higgs: 1000

Multiple seeds (for statistics):

```bash
# Run 100 seeds (manuscript setting, ~hours for higgs)
for seed in {0..99}; do
    python -m experiments.logistic_regression.run_logreg --dataset mushrooms --seed $seed
done
```

### Step 4: Run Baseline Methods

Each baseline has its own script in `baselines/`:

```bash
python -m experiments.logistic_regression.baselines.sag --dataset mushrooms --seed 0
python -m experiments.logistic_regression.baselines.saga --dataset mushrooms --seed 0
python -m experiments.logistic_regression.baselines.svrg --dataset mushrooms --seed 0
python -m experiments.logistic_regression.baselines.sarah --dataset mushrooms --seed 0
```

### Step 5: Generate Statistical Plots (Figure 8)

After running all seeds for a dataset:

```bash
python -m experiments.logistic_regression.plots_losses --dataset mushrooms
```

**Output:** `relative_loss_gap_<dataset>_stat.pdf` with confidence bands

## Output Organization

```
experiments/
├── output/                          # Generated outputs
│   ├── quadratic_*.csv             # Sensitivity tables
│   ├── overhead_*.pdf              # Benchmark figures
│   └── figures/                     # Additional plots
├── logistic_regression/
│   ├── <dataset>/
│   │   ├── Optimum.npy            # Reference optimum
│   │   ├── <dataset>.npy          # Dataset (auto-downloaded)
│   │   ├── sgd_mice/<seed>/       # SGD-MICE results per seed
│   │   ├── sag/<seed>/            # SAG results per seed
│   │   ├── saga/<seed>/           # SAGA results per seed
│   │   ├── svrg/<seed>/           # SVRG results per seed
│   │   └── sarah/<seed>/          # SARAH results per seed
│   └── relative_loss_gap_<dataset>_stat.pdf
└── quadratic/
    └── <various pkl/csv outputs>
```

## Reproducibility

All experiments use fixed seeds for reproducibility:
- Quadratic: Default seed=0 (configurable via `--seed`)
- Logistic regression: Seeds 0-99 for statistical analysis (100 runs)

The configurations match the manuscript exactly:
- MICE parameters: ε=√(1/3), δ_drop=0.5, δ_rest=0.0, min_batch=5
- Resampling: n_part=5, p_re=0.05, δ_re=1.0
- Dataset-specific: max|L_k| from `configs.py`

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'mice'`:
```bash
# Make sure you installed the package
cd ..  # to mice/
pip install -e .
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Logistic Regression: Missing Optimum

If you see "Missing optimum file":
```bash
python -m experiments.logistic_regression.compute_optimum --dataset <name>
```

### HIGGS Dataset Download Issues

The HIGGS dataset is large (~2.5 GB). If download fails:
1. Manually download from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2
2. Place in `experiments/logistic_regression/higgs/`
3. Run `datasets.py` to convert to `.npy` format

## Computational Requirements

**Quadratic experiments:**
- Time: ~1-2 hours for all tables/sweeps (50 runs each)
- Memory: < 1 GB

**Logistic regression (100 seeds each):**
- mushrooms: ~30 minutes
- gisette: ~2-4 hours (large feature space)
- higgs: ~12-24 hours (large dataset)

**Quick mode** (`--quick`): ~10-15 minutes total

## Citation

See main [README.md](../README.md) for citation information.
