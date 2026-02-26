
"""
experiments/neural_demand/dominicks/exp_welfare_robustness.py
=============================================================
Welfare path robustness check for Dominick's data.

Goal:
1. Document degree of near-integrability (Regularity Dashboard).
2. Check robustness of welfare estimates to different integration paths.

Paths:
1. Linear in prices
2. Linear in log prices
3. Piecewise (one good at a time)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dominicks import NeuralIRL, _train
from experiments.dominicks.data import load

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def get_config():
    return {
        "N_OBS": None, # Use full data
        "DEVICE": "cpu",
        "EPOCHS": 500, # Enough for convergence
        "nirl_hidden": 128,
        "lr": 1e-3,
        "batch_size": 256,
        "lam_mono": 0.1,
        "lam_slut": 0.01, # Small penalty for Dominick's
        "slut_start_frac": 0.25,
        "out_dir": "results/neural_demand/dominicks/welfare_robustness",
        "fig_dir": "results/neural_demand/dominicks/welfare_robustness/figures",
        "verbose": False,
        # Data paths
        "weekly_path": "data/wana.csv",
        "upc_path": "data/upcana.csv",
        "min_store_wks": 50,
        "std_tablets": 100,
        "habit_decay": 0.5,
        "test_cutoff": 300, # Approximate split
        "shock_good": 0,
        "shock_pct": 0.2
    }

# ─────────────────────────────────────────────────────────────────────────────
#  PATH INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_cv_path(model, p0, p1, y, path_type='linear', steps=100, device='cpu'):
    """
    Computes Compensating Variation (CV) along a specified path.
    CV = Integral( h(p, u_new) . dp ) approx Integral( x(p(t), y) . p'(t) dt )
    
    Note: This uses Marshallian demand x(p, y) as approximation for Hicksian h(p, u).
    For small shocks, they are close. For exact CV with non-integrable demand, 
    path dependence is the key feature we are measuring.
    """
    model.eval()
    
    t = np.linspace(0, 1, steps)
    dt = 1.0 / (steps - 1)
    
    loss = 0.0
    
    # Pre-compute path and derivatives
    if path_type == 'linear':
        # p(t) = p0 + t * (p1 - p0)
        # p'(t) = p1 - p0
        diff = p1 - p0
        p_path = p0[None, :] + t[:, None] * diff[None, :]
        dp_dt = np.tile(diff, (steps, 1))
        
    elif path_type == 'log_linear':
        # ln p(t) = ln p0 + t * (ln p1 - ln p0)
        # p(t) = p0 * (p1/p0)^t
        # p'(t) = p(t) * ln(p1/p0)
        lp0 = np.log(np.maximum(p0, 1e-8))
        lp1 = np.log(np.maximum(p1, 1e-8))
        ldiff = lp1 - lp0
        
        lp_path = lp0[None, :] + t[:, None] * ldiff[None, :]
        p_path = np.exp(lp_path)
        dp_dt = p_path * ldiff[None, :]
        
    elif path_type == 'piecewise':
        # Move good 1, then good 2, then good 3...
        # We need to construct the path carefully
        # This is harder to vectorize simply with linspace, so we do it sequentially
        # But for comparison, let's just do 3 segments if G=3
        G = len(p0)
        # We will just do a simple implementation: 
        # Segment 1: p0 -> p0 with p1 changed
        # Segment 2: ...
        # Actually, let's stick to the steps format
        # We split steps into G chunks
        steps_per_g = steps // G
        p_curr = p0.copy()
        p_path_list = []
        dp_dt_list = []
        
        for g in range(G):
            p_next = p_curr.copy()
            p_next[g] = p1[g]
            
            # Linear sub-segment
            t_sub = np.linspace(0, 1, steps_per_g)
            diff = p_next - p_curr
            sub_path = p_curr[None, :] + t_sub[:, None] * diff[None, :]
            sub_dp = np.tile(diff, (steps_per_g, 1))
            
            p_path_list.append(sub_path)
            dp_dt_list.append(sub_dp)
            p_curr = p_next
            
        p_path = np.vstack(p_path_list)
        dp_dt = np.vstack(dp_dt_list)
        # Adjust steps to actual length
        steps = len(p_path)
        dt = 1.0 / steps # Approx
        
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    # Evaluate demand along path
    # We can batch this? Yes, treat time steps as batch
    # But y is scalar/fixed for this calculation (CV at new utility, usually approximated at old income)
    # Wait, CV is money needed at NEW prices to reach OLD utility.
    # EV is money needed at OLD prices to reach NEW utility.
    # The standard line integral formula -Integral x(p, y) dp calculates change in Consumer Surplus (CS).
    # CS is area to the left of demand curve.
    # Delta CS = Integral_{p0}^{p1} x(p, y) dp.
    # This is path dependent if curl != 0.
    # We will compute this integral.
    
    # Prepare inputs
    p_tensor = torch.tensor(p_path, dtype=torch.float32, device=device)
    y_tensor = torch.full((steps, 1), float(y), dtype=torch.float32, device=device)
    
    lp = torch.log(torch.clamp(p_tensor, min=1e-8))
    ly = torch.log(torch.clamp(y_tensor, min=1e-8))
    
    with torch.no_grad():
        w = model(lp, ly).cpu().numpy()
        
    # x = w * y / p
    x = w * y / p_path
    
    # Integral x . dp
    # Sum (x[t] * dp_dt[t]) * dt
    integrand = np.sum(x * dp_dt, axis=1)
    
    # Trapezoidal rule for better accuracy
    # integral = sum ( (y[i] + y[i+1])/2 * dt )
    # Here we have derivatives, so it's sum( integrand[t] ) * dt approx
    # Or simply:
    val = np.sum(integrand) * (1.0 / steps) # scaling is tricky with piecewise
    
    # Let's use simple Riemann sum: sum( x(p(t)) * p'(t) * dt )
    # In 'linear', dt = 1/(steps-1). p'(t) is constant.
    # In 'piecewise', we constructed it such that we iterate through segments.
    
    # Refined integration:
    # Delta P between steps
    # dp_actual = p_path[t+1] - p_path[t]
    # val = sum( x[t] * dp_actual )
    
    dp_actual = np.diff(p_path, axis=0)
    x_mid = (x[:-1] + x[1:]) / 2
    val = np.sum(x_mid * dp_actual)
    
    # Sign: Price increase -> Loss of CS -> Negative value.
    # CV is usually positive for price increase (amount to compensate).
    # CV approx -Delta CS.
    # So we return -val.
    
    return -val

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    cfg = get_config()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)
    
    print("Loading Dominick's Data...")
    # load returns (data, splits)
    _, splits = load(cfg)
    p_tr, y_tr, w_tr = splits["p_tr"], splits["y_tr"], splits["w_tr"]
    p_te, y_te, w_te = splits["p_te"], splits["y_te"], splits["w_te"]
    
    # Train Model
    print("Training NeuralIRL (Static)...")
    model = NeuralIRL(cfg["nirl_hidden"])
    model, _ = _train(
        model, p_tr, y_tr, w_tr, "nirl", cfg, 
        tag="NDS-Robustness"
    )
    
    # 1. Regularity Diagnostics
    print("\nRunning Regularity Diagnostics on Test Set...")
    # Subsample for speed
    idx = np.random.choice(len(p_te), min(1000, len(p_te)), replace=False)
    reg_res = regularity_dashboard(
        model, p_te[idx], y_te[idx], device=cfg["DEVICE"]
    )
    
    print("Regularity Dashboard:")
    print(f"  Symmetry (Mean Frobenius): {reg_res['symmetry']['mean_fro']:.4f}")
    print(f"  Symmetry (Mean Relative):  {reg_res['symmetry']['mean_rel']:.4f}")
    print(f"  Pos Eig Share:             {reg_res['curvature']['share_positive']:.4f}")
    print(f"  Homogeneity (c=1.2) Mean:  {reg_res['homogeneity'][1.2]['mean_diff']:.4f}")
    
    # Save diagnostics
    pd.DataFrame([reg_res['symmetry']]).to_csv(f"{cfg['out_dir']}/symmetry.csv", index=False)
    
    # 2. Welfare Path Robustness
    print("\nRunning Welfare Path Robustness...")
    
    # Define shock: 20% increase in Good 1
    # We evaluate this at the mean price/income of test set
    p0 = p_te.mean(0)
    y0 = y_te.mean()
    p1 = p0.copy()
    p1[0] *= 1.2 # Good 1
    
    # Also try a mixed shock: +20% G1, -10% G2
    p1_mixed = p0.copy()
    p1_mixed[0] *= 1.2
    p1_mixed[1] *= 0.9
    
    shocks = [
        ("Shock_G1_20", p1),
        ("Shock_Mixed", p1_mixed)
    ]
    
    paths = ['linear', 'log_linear', 'piecewise']
    
    results = []
    
    for shock_name, p_target in shocks:
        print(f"  Evaluating {shock_name}...")
        row = {"Shock": shock_name}
        vals = []
        
        for path in paths:
            cv = compute_cv_path(model, p0, p_target, y0, path_type=path, device=cfg["DEVICE"])
            row[path] = cv
            vals.append(cv)
            print(f"    Path {path:10s}: {cv:.4f}")
            
        # Compute dispersion
        vals = np.array(vals)
        row["Mean"] = vals.mean()
        row["Std"] = vals.std()
        row["Range"] = vals.max() - vals.min()
        row["Range_Pct"] = 100 * row["Range"] / abs(row["Mean"])
        
        results.append(row)
        
    df_res = pd.DataFrame(results)
    print("\nWelfare Robustness Results:")
    print(df_res)
    df_res.to_csv(f"{cfg['out_dir']}/welfare_robustness.csv", index=False)
    
    # Plot bars
    fig, ax = plt.subplots(figsize=(8, 5))
    df_res.plot(x="Shock", y=["linear", "log_linear", "piecewise"], kind="bar", ax=ax)
    ax.set_ylabel("Estimated CV")
    ax.set_title("Welfare Estimates across Integration Paths")
    plt.tight_layout()
    plt.savefig(f"{cfg['fig_dir']}/welfare_paths.png")
    
    return df_res

if __name__ == "__main__":
    run_experiment()
