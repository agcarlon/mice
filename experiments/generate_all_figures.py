#!/usr/bin/env python3
"""
Master script to generate all figures and tables from the manuscript:
"Multi-Iteration Stochastic Optimizers"

This script orchestrates all numerical experiments and produces the outputs
referenced in the manuscript.

Figures & Tables generated:
- Figure 5: κ-sweep (quadratic)
- Figure 7: Runtime benchmark breakdown
- Table 2: Operator ablation
- Table 3: ε sensitivity
- Table 4: δ_drop and δ_rest sensitivity
- Table 5: Max |L_k| sensitivity
- Figure 8: Logistic regression comparisons (3 datasets)

Usage:
    python experiments/generate_all_figures.py [--quick] [--skip-logreg]
    
Options:
    --quick: Run with reduced budget/runs for testing (not for paper)
    --skip-logreg: Skip time-intensive logistic regression experiments
    --output-dir: Output directory (default: experiments/output)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import time


def parse_seed_spec(spec: str) -> list[int]:
    """Parse seeds from comma list (0,1,2) or inclusive range (0:99)."""
    spec = spec.strip()
    if ":" in spec and "," not in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("seed range must be start <= end")
        return list(range(start, end + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and report progress."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    t_start = time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time() - t_start
    
    if result.returncode == 0:
        print(f"\n✓ Completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ Failed with exit code {result.returncode}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures and tables from the manuscript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (reduced budget/runs)")
    parser.add_argument("--skip-logreg", action="store_true",
                        help="Skip logistic regression experiments")
    default_output_dir = (Path(__file__).resolve().parent / "output")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir),
                        help="Output directory for results")
    parser.add_argument("--logreg-seeds", type=str, default=None,
                        help="Logistic seeds: comma list (0,1,2) or inclusive range (0:99)")
    args = parser.parse_args()
    py = sys.executable

    # Resolve relative output paths from the repo root so that passing
    # `--output-dir experiments/output` works as expected.
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    if args.quick:
        budget = 10000
        runs = 5
        seeds_logreg = "0:9"
        print("\n QUICK MODE: Using reduced budget/runs (not suitable for paper)")
    else:
        budget = 100000
        runs = 50
        seeds_logreg = "0:99"
        print("\n FULL MODE: Using manuscript settings (budget=100k, runs=50)")
    if args.logreg_seeds is not None:
        seeds_logreg = args.logreg_seeds
    seed_values = parse_seed_spec(seeds_logreg)
    
    failed_commands = []
    
    # ========== QUADRATIC EXPERIMENTS ==========
    print("\n" + "="*70)
    print("QUADRATIC EXPERIMENTS")
    print("="*70)
    
    # Table 2: Operator ablation
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_ablations",
         "--max-cost", str(budget), "--runs", str(runs), "--outdir", str(output_dir)],
        "Table 2: Operator ablation"
    )
    if ret != 0:
        failed_commands.append("Operator ablation")
    
    # Table 3: ε sensitivity
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_epsilon_sweep",
         "--max-cost", str(budget), "--runs", str(runs), "--outdir", str(output_dir)],
        "Table 3: Epsilon sensitivity"
    )
    if ret != 0:
        failed_commands.append("Epsilon sensitivity")
    
    # Table 4: δ sensitivity
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_delta_sweep",
         "--max-cost", str(budget), "--runs", str(runs), "--outdir", str(output_dir)],
        "Table 4: Delta sensitivity (drop & restart)"
    )
    if ret != 0:
        failed_commands.append("Delta sensitivity")
    
    # Table 5: Max |L_k| sensitivity
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_max_index_sweep",
         "--max-cost", str(budget), "--runs", str(runs), "--outdir", str(output_dir)],
        "Table 5: Max index set cardinality"
    )
    if ret != 0:
        failed_commands.append("Max index sensitivity")
    
    # Figure 7: Benchmark
    dims = "10,50,100" if args.quick else "10, 100, 1000, 10000"
    bench_runs = 5 if args.quick else 50
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_benchmarks",
         "--dims", dims, "--runs", str(bench_runs), "--outdir", str(output_dir)],
        "Figure 7: Runtime benchmark breakdown"
    )
    if ret != 0:
        failed_commands.append("Benchmark")
    
    # Single-run quadratic figures (manuscript Section 6.1)
    ret = run_command(
        [py, "-m", "experiments.quadratic.run_single_kappa",
         "--kappa", "100", "--seed", "1", "--resampling", "--outdir", str(output_dir)],
        "Quadratic single run: SGD_MICE_all_per_iter/grads"
    )
    if ret != 0:
        failed_commands.append("Quadratic single run (kappa_sweep)")

    ret = run_command(
        [py, "-m", "experiments.quadratic.run_sgd_a_comparison",
         "--outdir", str(output_dir)] + (["--quick"] if args.quick else []),
        "Quadratic: SGD-A comparison (sgd_a.pdf)"
    )
    if ret != 0:
        failed_commands.append("Quadratic SGD-A comparison")

    ret = run_command(
        [py, "-m", "experiments.quadratic.run_kappa_cost_scaling",
         "--outdir", str(output_dir)] + (["--quick"] if args.quick else []),
        "Quadratic: kappa scaling (kappa_test_w_sgd_a.pdf)"
    )
    if ret != 0:
        failed_commands.append("Quadratic kappa scaling")

    ret = run_command(
        [py, "-m", "experiments.quadratic.run_stopping_consistency",
         "--outdir", str(output_dir)] + (["--quick"] if args.quick else []),
        "Quadratic: stopping consistency violins"
    )
    if ret != 0:
        failed_commands.append("Quadratic stopping consistency")

    # Rosenbrock example (manuscript Section 6.2)
    ret = run_command(
        [py, "-m", "experiments.rosenbrock.run_rosenbrock_adam",
         "--outdir", str(output_dir)] + (["--quick"] if args.quick else []),
        "Rosenbrock: Adam vs Adam-MICE"
    )
    if ret != 0:
        failed_commands.append("Rosenbrock (Adam vs Adam-MICE)")
    
    # ========== LOGISTIC REGRESSION EXPERIMENTS ==========
    if not args.skip_logreg:
        print("\n" + "="*70)
        print("LOGISTIC REGRESSION EXPERIMENTS")
        print("="*70)
        
        for dataset in ["mushrooms", "gisette", "higgs"]:
            # Compute optimum first (if not exists)
            print(f"\n--- Checking optimum for {dataset} ---")
            ret = run_command(
                [py, "-m", "experiments.logistic_regression.compute_optimum",
                 "--dataset", dataset],
                f"Compute optimum for {dataset}"
            )
            
            methods = [
                ("SGD-MICE", "experiments.logistic_regression.run_logreg"),
                ("SAG", "experiments.logistic_regression.baselines.sag"),
                ("SAGA", "experiments.logistic_regression.baselines.saga"),
                ("SARAH", "experiments.logistic_regression.baselines.sarah"),
                ("SVRG", "experiments.logistic_regression.baselines.svrg"),
            ]
            print(f"\n--- Running {dataset} over seeds: {seeds_logreg} ({len(seed_values)} seeds) ---")
            for method_name, module_name in methods:
                for seed in seed_values:
                    ret = run_command(
                        [py, "-m", module_name, "--dataset", dataset, "--seed", str(seed)],
                        f"Figure 8: {method_name} on {dataset} (seed {seed})"
                    )
                    if ret != 0:
                        failed_commands.append(f"Logistic {dataset} {method_name} seed={seed}")

            # Generate confidence-interval plots from all seeds.
            ret = run_command(
                [py, "-m", "experiments.logistic_regression.plots_losses",
                 "--dataset", dataset, "--seeds", seeds_logreg, "--stats", "--outdir", str(output_dir)],
                f"Figure 8: Statistical plot for {dataset}"
            )
            if ret != 0:
                failed_commands.append(f"Logistic {dataset} stats plot")

            # Export chain-size plot for seed 0 with manuscript filename.
            ret = run_command(
                [py, "-m", "experiments.logistic_regression.export_chain_size",
                 "--dataset", dataset, "--outdir", str(output_dir)],
                f"Logistic regression: chain size export for {dataset}"
            )
            if ret != 0:
                failed_commands.append(f"Logistic {dataset} chain size export")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    
    print(f"\nOutput directory: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  - quadratic_gap_ablations_summary.csv (Table 2)")
    print(f"  - quadratic_epsilon_sensitivity.csv (Table 3)")
    print(f"  - quadratic_delta_drop_sensitivity.csv & quadratic_delta_rest_sensitivity.csv (Table 4)")
    print(f"  - quadratic_max_index_sensitivity.csv (Table 5)")
    print(f"  - overhead_by_dim.csv (Figure 7 data)")
    print(f"  - overhead_pct_vs_dim.pdf (Figure 7)")
    print(f"  - SGD_MICE_all_per_iter.pdf (Quadratic single-run)")
    print(f"  - SGD_MICE_all_per_grads.pdf (Quadratic single-run)")
    print(f"  - SGD_MICE_err_vkk_chain_size.pdf (Quadratic single-run)")
    print(f"  - sgd_a.pdf (Quadratic comparison)")
    print(f"  - kappa_test_w_sgd_a.pdf (Kappa scaling)")
    print(f"  - consistency_plot_violin.pdf (Stopping consistency, resampling)")
    print(f"  - consistency_plot_violinno_resamp.pdf (Stopping consistency, no resampling)")
    print(f"  - optimality_gap_Rosenbrock_1.pdf (Rosenbrock sigma=1e-4)")
    print(f"  - optimality_gap_Rosenbrock_2.pdf (Rosenbrock sigma=1e-1)")
    if not args.skip_logreg:
        print(f"  - experiments/logistic_regression/<dataset>/<method>/<seed>/ (Figure 8 data)")
        print(f"  - relative_loss_gap_<dataset>_stat.pdf (Figure 8 stats)")
        print(f"  - sgd_mice_chain_size_<dataset>.pdf (Figure 8 chain size)")
    
    if failed_commands:
        print(f"\n⚠ WARNING: {len(failed_commands)} command(s) failed:")
        for cmd in failed_commands:
            print(f"  - {cmd}")
        sys.exit(1)
    else:
        print(f"\n✓ All experiments completed successfully!")
        print(f"\nTo generate statistical plots from logistic regression data:")
        print(f"  python -m experiments.logistic_regression.plots_losses --dataset <name>")


if __name__ == "__main__":
    main()
