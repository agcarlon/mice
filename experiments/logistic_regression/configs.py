"""
Dataset-specific configurations for logistic regression experiments.

These settings match the manuscript (submission_4/main.tex):
- Table at line 2853: lambda, kappa, epochs, datasize, features
- Line 2432: max_hierarchy_size = 100 (general case)
- Line 2875: max_hierarchy_size = 1000 (HIGGS only, due to smaller dimensionality)
"""

DATASET_CONFIGS = {
    "mushrooms": {
        "lambda_reg": 1e-5,
        "kappa": 12316.30,
        "epochs": 100,
        "max_hierarchy_size": 100,
        "datasize": 8124,
        "n_features": 112,
    },
    "gisette": {
        "lambda_reg": 1e-4,
        "kappa": 1811.21,
        "epochs": 50,
        "max_hierarchy_size": 100,
        "datasize": 6000,
        "n_features": 5000,
    },
    "higgs": {
        "lambda_reg": 1e-4,
        "kappa": 765.76,
        "epochs": 10,
        "max_hierarchy_size": 1000,  # Larger for HIGGS (smaller dimensionality)
        "datasize": 11000000,
        "n_features": 28,
    },
}

# MICE parameters (manuscript line 2431-2432)
MICE_PARAMS = {
    "eps": 0.577,  # sqrt(1/3)
    "drop_param": 0.5,
    "restart_param":10.0,
    "min_batch": 5,
    "restart_factor": 100,  # M_min = 50 for restarts = 5 * 10
}

# Resampling parameters (manuscript line 2434)
RESAMPLING_PARAMS = {
    "re_part": 5,      # n_part
    "re_quantile": 0.05,  # p_re
    "re_tot_cost": 1.0,   # delta_re
    "re_min_n": 10,
}

# Baseline step-size formulas (manuscript Table at line 2878)
BASELINE_STEP_SIZES = {
    "sag": lambda L_hat, mu, N: 1.0 / (16 * (L_hat + mu * N)),
    "saga": lambda L_hat, mu, N: 1.0 / (2 * (L_hat + mu * N)),
    "sarah": lambda L_hat, mu, N: 1.0 / (2 * L_hat),
    "svrg": lambda L_hat, mu, N: 1.0 / (2 * L_hat),
}


def get_config(dataset_name: str) -> dict:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: One of "mushrooms", "gisette", "higgs"
    
    Returns:
        Dictionary with all configuration parameters
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    
    config = DATASET_CONFIGS[dataset_name].copy()
    config.update(MICE_PARAMS)
    config.update(RESAMPLING_PARAMS)
    config["baseline_step_sizes"] = BASELINE_STEP_SIZES
    
    return config
