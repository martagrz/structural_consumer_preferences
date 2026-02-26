
"""
experiments/neural_demand/simulation/exp_penalty_sensitivity.py
===============================================================
Validation of regularity penalties in simulation.
Goal: Show tradeoff between fit, economic coherence (integrability), and welfare accuracy.

Grid over penalty weights: lambda_sym in {0, 0.1, 1, 10}

Reports:
1. Predictive fit (Test KL, RMSE)
2. Regularity diagnostics (Symmetry, Curvature, Homogeneity)
3. Welfare recovery (CV error)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.simulation import (
    CESConsumer,
    NeuralIRL,
    train_neural_irl,
)
from experiments.neural_demand.simulation.utils import (
    AVG_Y,
    predict_shares,
    compute_compensating_variation,
)
from src.evaluation.regularity import regularity_dashboard

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LAM_SLUT_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]
# LAM_SLUT_GRID = [0.0, 0.1] # For quick testing

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> list:
    N          = cfg["N_OBS"]
    DEVICE     = cfg["DEVICE"]
    EPOCHS     = cfg["EPOCHS"]
    HIDDEN     = cfg.get("hidden_dim", 128)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ── Data Generation (CES) ────────────────────────────────────────────────
    # Shared price / income draws
    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = np.clip(Z + np.random.normal(0, 0.1, (N, 3)), 1e-3, None)
    income = np.random.uniform(1200, 2000, N)
    
    consumer = CESConsumer()
    w_train  = consumer.solve_demand(p_pre, income)
    
    # Welfare shock (20% price increase on good 1)
    p_post  = p_pre.copy(); p_post[:, 1] *= 1.2
    w_post  = consumer.solve_demand(p_post, income)
    
    avg_p   = p_post.mean(0)
    p0_welf = avg_p / np.array([1.0, 1.2, 1.0])   # pre-shock
    p1_welf = avg_p                                  # post-shock
    
    # True Welfare
    KW_TRUE = dict(consumer=consumer, device=DEVICE)
    cv_true = compute_compensating_variation("truth", p0_welf, p1_welf, AVG_Y, **KW_TRUE)
    
    results = []
    
    # ── Loop over penalties ──────────────────────────────────────────────────
    for lam in LAM_SLUT_GRID:
        if verbose:
            print(f"  Training with lam_slut={lam}...")
            
        model = NeuralIRL(n_goods=3, hidden_dim=HIDDEN)
        model, hist = train_neural_irl(
            model, p_pre, income, w_train,
            epochs=EPOCHS, lr=5e-4, batch_size=256,
            lam_mono=0.3, lam_slut=lam, slut_start_frac=0.0, # Start penalty immediately for clear effect? Or 0.25
            device=DEVICE, verbose=False
        )
        
        # ── Evaluation ───────────────────────────────────────────────────────
        model.eval()
        
        # 1. Predictive Fit (on post-shock data, effectively a test set)
        w_pred = predict_shares("nd-static", p_post, income, nds=model, device=DEVICE)
        rmse = np.sqrt(np.mean((w_pred - w_post)**2))
        
        # 2. Regularity Diagnostics (on post-shock data)
        # We use a subset for expensive matrix checks if N is large
        n_eval = min(1000, N)
        idx_eval = np.random.choice(N, n_eval, replace=False)
        
        reg_metrics = regularity_dashboard(
            model, p_post[idx_eval], income[idx_eval], 
            device=DEVICE
        )
        
        # 3. Welfare Recovery
        cv_pred = compute_compensating_variation(
            "nd-static", p0_welf, p1_welf, AVG_Y, 
            nds=model, device=DEVICE
        )
        cv_err = abs(cv_pred - cv_true)
        cv_err_pct = 100 * cv_err / abs(cv_true)
        
        # Store results
        res = {
            "lam_slut": lam,
            "seed": seed,
            "RMSE": rmse,
            "CV_err_pct": cv_err_pct,
            "Sym_Mean": reg_metrics['symmetry']['mean_fro'],
            "Sym_Rel": reg_metrics['symmetry']['mean_rel'],
            "Pos_Eig_Share": reg_metrics['curvature']['share_positive'],
            "Homog_Err": reg_metrics['homogeneity'][1.2]['mean_diff'] # Check at c=1.2
        }
        results.append(res)
        
    return results

def aggregate_results(all_results):
    df = pd.DataFrame(all_results)
    # Group by lam_slut
    agg = df.groupby("lam_slut").agg(['mean', 'std', 'count'])
    return agg, df

def make_plots(df, out_dir):
    # Plot 1: Tradeoff - Symmetry vs CV Error
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    means = df.groupby("lam_slut").mean()
    
    color = 'tab:red'
    ax1.set_xlabel('Symmetry Violation (Mean Frobenius)')
    ax1.set_ylabel('CV Error (%)', color=color)
    ax1.plot(means['Sym_Mean'], means['CV_err_pct'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Annotate points with lambda values
    for lam in means.index:
        ax1.annotate(f"$\lambda={lam}$", (means.loc[lam, 'Sym_Mean'], means.loc[lam, 'CV_err_pct']))
        
    ax1.set_title("Regularity vs Welfare Accuracy")
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_penalty_tradeoff.png")
    plt.close(fig)
    
    # Plot 2: Metrics vs Lambda
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Symmetry
    axes[0].plot(means.index, means['Sym_Mean'], marker='o')
    axes[0].set_xscale('symlog', linthresh=0.01)
    axes[0].set_xlabel('$\lambda_{sym}$')
    axes[0].set_ylabel('Symmetry Violation')
    axes[0].set_title('Effect of Penalty on Symmetry')
    
    # CV Error
    axes[1].plot(means.index, means['CV_err_pct'], marker='o', color='orange')
    axes[1].set_xscale('symlog', linthresh=0.01)
    axes[1].set_xlabel('$\lambda_{sym}$')
    axes[1].set_ylabel('CV Error (%)')
    axes[1].set_title('Effect of Penalty on Welfare Error')
    
    # RMSE
    axes[2].plot(means.index, means['RMSE'], marker='o', color='green')
    axes[2].set_xscale('symlog', linthresh=0.01)
    axes[2].set_xlabel('$\lambda_{sym}$')
    axes[2].set_ylabel('RMSE (Fit)')
    axes[2].set_title('Effect of Penalty on Fit')
    
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_penalty_metrics.png")
    plt.close(fig)

def run(cfg):
    N_RUNS = cfg["N_RUNS"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    
    print("=" * 60)
    print("  Penalty Sensitivity Experiment")
    print("=" * 60)
    
    all_res = []
    for i in range(N_RUNS):
        seed = 42 + i
        print(f"Run {i+1}/{N_RUNS} (seed={seed})")
        res = run_one_seed(seed, cfg, verbose=True)
        all_res.extend(res)
        
    agg, df = aggregate_results(all_res)
    print("\nAggregated Results:")
    print(agg)
    
    df.to_csv(f"{cfg['out_dir']}/penalty_sensitivity.csv", index=False)
    make_plots(df, cfg['out_dir'])
    
    return df

if __name__ == "__main__":
    # Default config for standalone run
    CFG = {
        "N_OBS": 2000,
        "DEVICE": "cpu",
        "EPOCHS": 1000,
        "N_RUNS": 3,
        "out_dir": "results/neural_demand/simulation/penalty_sensitivity",
        "hidden_dim": 64
    }
    run(CFG)
