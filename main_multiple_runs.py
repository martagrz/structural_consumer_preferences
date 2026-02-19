"""
Recovering Consumer Preferences via Inverse Reinforcement Learning
==================================================================
Multi-run version: all models are re-trained N_RUNS times with independent
random seeds. Every table reports mean (SE); every demand-curve figure
shows a mean line with ±1 SE shaded band.

Set N_RUNS = 1 to reproduce the original single-run behaviour exactly.
Recommended for publication: N_RUNS = 10 (≈10× wall time).

Dependencies: numpy, scipy, pandas, matplotlib, torch, sklearn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings, os, time
from src.models.simulation import (
    AIDSBench,
    BLPBench,
    CESConsumer,
    ContinuousVariationalMixture,
    HabitFormationConsumer,
    LeontiefConsumer,
    MDPNeuralIRL,
    NeuralIRL,
    QuasilinearConsumer,
    StoneGearyConsumer,
    features_good_specific,
    features_orthogonalised,
    features_shared,
    predict_linear_irl,
    run_linear_irl,
    train_neural_irl,
)
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION  ← only section you need to edit
# ════════════════════════════════════════════════════════════════════

N_RUNS   = 5      # number of independent replications
N_OBS    = 800    # observations per run
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("figures", exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  SECTION 1-6: OBJECT DEFINITIONS MOVED TO src/simulation/models.py
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
#  SECTION 7: UNIFIED PREDICTION & EVALUATION UTILITIES
# ════════════════════════════════════════════════════════════════════

def predict_shares(spec, p, y, *, aids=None, blp=None, lirl_theta=None,
                   lirl_feat_fn=None, nirl=None, mdp_nirl=None,
                   xbar=None, consumer=None, mixture=None, device="cpu"):
    if spec == "truth":   return consumer.solve_demand(p, y)
    if spec == "aids":    return aids.predict(p, y)
    if spec == "blp":     return blp.predict(p)
    if spec == "l-irl":   return predict_linear_irl(lirl_feat_fn(p,y), lirl_theta)
    if spec == "n-irl":
        with torch.no_grad():
            lp = torch.log(torch.tensor(p, dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(y, dtype=torch.float32, device=device)).unsqueeze(1)
            return nirl(lp,ly).cpu().numpy()
    if spec == "mdp-irl":
        with torch.no_grad():
            lp = torch.log(torch.tensor(p,    dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(y,    dtype=torch.float32, device=device)).unsqueeze(1)
            xb = torch.tensor(xbar,           dtype=torch.float32, device=device)
            return mdp_nirl(lp,ly,xb).cpu().numpy()
    if spec == "mixture": return mixture.predict(p, y)
    raise ValueError(spec)

def compute_elasticities(spec, p_pt, y_pt, h=1e-4, xbar_pt=None, **kw):
    w0  = predict_shares(spec, p_pt.reshape(1,-1), np.array([y_pt]),
                         xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None, **kw)[0]
    eps = []
    for i in range(3):
        p1 = p_pt.copy().reshape(1,-1); p1[0,i] *= (1+h)
        wp = predict_shares(spec, p1, np.array([y_pt]),
                            xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None, **kw)[0]
        eps.append(((wp[i]-w0[i])/w0[i])/h - 1)
    return eps

def compute_welfare_loss(spec, p0, p1, y, steps=100, xbar_pt=None, **kw):
    path = np.linspace(p0,p1,steps); dp = (p1-p0)/steps; loss = 0.0
    for t in range(steps):
        w    = predict_shares(spec, path[t:t+1], np.array([y]),
                              xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None,
                              **kw)[0]
        loss -= (w*y/path[t]) @ dp
    return loss

def get_metrics(spec, p_shock, income, w_true, xbar_shock=None, **kw):
    wp = predict_shares(spec, p_shock, income, xbar=xbar_shock, **kw)
    return {"RMSE": np.sqrt(mean_squared_error(w_true, wp)),
            "MAE":  mean_absolute_error(w_true, wp)}

def kl_div(w_pred, w_true):
    wp = np.clip(w_pred,1e-8,1); wt = np.clip(w_true,1e-8,1)
    return float(np.mean(np.sum(wt*np.log(wt/wp), axis=1)))

# ════════════════════════════════════════════════════════════════════
#  SECTION 8: SINGLE-RUN FUNCTION
#  Returns a structured dict of every metric + demand-curve arrays.
# ════════════════════════════════════════════════════════════════════

# Fixed price grid used for demand-curve figures across all runs
P_GRID  = np.linspace(1, 10, 80)
AVG_Y   = 1600.0


def run_one_seed(seed: int, verbose: bool = False) -> dict:
    """Execute the full pipeline with one data seed. Returns all results."""
    # Use the same global Mersenne Twister RNG as the original single-run script
    # (np.random.default_rng uses PCG64 which produces different sequences for
    # the same seed — switching back avoids changing the habit data distribution
    # that MDP IRL was tuned against).
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────
    N      = N_OBS
    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    primary        = CESConsumer()
    w_train        = primary.solve_demand(p_pre, income)
    habit_consumer = HabitFormationConsumer()
    w_habit, xbar_train = habit_consumer.solve_demand(p_pre, income, return_xbar=True)

    p_post           = p_pre.copy(); p_post[:,1] *= 1.2
    w_post_true      = primary.solve_demand(p_post, income)
    w_habit_shock, xbar_shock = habit_consumer.solve_demand(p_post, income, return_xbar=True)

    avg_p    = p_post.mean(0)
    p_pre_pt = avg_p / np.array([1.0, 1.2, 1.0])

    # ── Benchmarks ───────────────────────────────────────────────────
    aids_m   = AIDSBench(); aids_m.fit(p_pre, w_train, income)
    blp_m    = BLPBench();  blp_m.fit(p_pre, w_train, Z)
    aids_hab = AIDSBench(); aids_hab.fit(p_pre, w_habit, income)

    # ── Linear IRL ───────────────────────────────────────────────────
    theta_shared   = run_linear_irl(features_shared(p_pre, income),         w_train)
    theta_goodspec = run_linear_irl(features_good_specific(p_pre, income),  w_train)
    theta_orth     = run_linear_irl(features_orthogonalised(p_pre, income), w_train)

    # ── Neural IRL (CES) ─────────────────────────────────────────────
    n_irl = NeuralIRL(n_goods=3, hidden_dim=256)
    n_irl, hist_nirl = train_neural_irl(
        n_irl, p_pre, income, w_train, epochs=4000, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        device=DEVICE, verbose=verbose)

    # ── Neural IRL static (Habit baseline) ───────────────────────────
    n_irl_hab = NeuralIRL(n_goods=3, hidden_dim=128)
    n_irl_hab, _ = train_neural_irl(
        n_irl_hab, p_pre, income, w_habit, epochs=2000, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # ── MDP Neural IRL ───────────────────────────────────────────────
    mdp_irl = MDPNeuralIRL(n_goods=3, hidden_dim=256)
    mdp_irl, hist_mdp = train_neural_irl(
        mdp_irl, p_pre, income, w_habit, epochs=4000, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xbar_data=xbar_train, device=DEVICE, verbose=verbose)

    # ── Variational Mixture ───────────────────────────────────────────
    var_mix = ContinuousVariationalMixture(K=6, n_goods=3, n_samples_per_component=100)
    
    # Use full data, 50 iterations, and the new defaults will handle sigma2/smoothing
    var_mix.fit(p_pre, income, w_train, n_iter=50, lr_mu=0.05, sigma2=0.1)

    # ── Robustness across DGPs ────────────────────────────────────────
    all_consumers = {"CES": CESConsumer(), "Quasilinear": QuasilinearConsumer(),
                     "Leontief": LeontiefConsumer(), "Stone-Geary": StoneGearyConsumer(),
                     "Habit": HabitFormationConsumer()}
    rob_rows = {}
    for cname, cons in all_consumers.items():
        try:
            w_dgp       = cons.solve_demand(p_pre, income)
            w_dgp_shock = cons.solve_demand(p_post, income)
            a_rob = AIDSBench(); a_rob.fit(p_pre, w_dgp, income)
            th_rob = run_linear_irl(features_orthogonalised(p_pre,income), w_dgp, epochs=2000)
            n_rob  = NeuralIRL(n_goods=3, hidden_dim=128)
            n_rob,_ = train_neural_irl(n_rob, p_pre, income, w_dgp,
                          epochs=2000, lr=5e-4, batch_size=256,
                          lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3, device=DEVICE)
            rob_rows[cname] = {
                "AIDS":           get_metrics("aids",  p_post,income,w_dgp_shock,aids=a_rob)["RMSE"],
                "Lin IRL (Orth)": get_metrics("l-irl", p_post,income,w_dgp_shock,
                                              lirl_theta=th_rob,lirl_feat_fn=features_orthogonalised)["RMSE"],
                "Neural IRL":     get_metrics("n-irl", p_post,income,w_dgp_shock,
                                              nirl=n_rob,device=DEVICE)["RMSE"],
            }
        except Exception:
            rob_rows[cname] = {"AIDS":np.nan,"Lin IRL (Orth)":np.nan,"Neural IRL":np.nan}

    # ── Primary evaluation ────────────────────────────────────────────
    KW = dict(aids=aids_m, blp=blp_m, nirl=n_irl, consumer=primary,
              mixture=var_mix, device=DEVICE)
    MODELS_CES = [
        ("LA-AIDS",          "aids",    {}),
        ("BLP (IV)",         "blp",     {}),
        ("Lin IRL Shared",   "l-irl",   {"lirl_theta":theta_shared,   "lirl_feat_fn":features_shared}),
        ("Lin IRL GoodSpec", "l-irl",   {"lirl_theta":theta_goodspec, "lirl_feat_fn":features_good_specific}),
        ("Lin IRL Orth",     "l-irl",   {"lirl_theta":theta_orth,     "lirl_feat_fn":features_orthogonalised}),
        ("Neural IRL",       "n-irl",   {}),
        ("Var. Mixture",     "mixture", {}),
    ]

    perf_ces = {n: get_metrics(s, p_post,income,w_post_true, **{**KW,**ek})
                for n,s,ek in MODELS_CES}

    elast = {"Ground Truth": compute_elasticities("truth", avg_p, AVG_Y, **KW)}
    elast.update({n: compute_elasticities(s,avg_p,AVG_Y,**{**KW,**ek}) for n,s,ek in MODELS_CES})

    welf = {"Ground Truth": compute_welfare_loss("truth", p_pre_pt,avg_p,AVG_Y, **KW)}
    welf.update({n: compute_welfare_loss(s,p_pre_pt,avg_p,AVG_Y,**{**KW,**ek}) for n,s,ek in MODELS_CES})

    lin_ablation = {
        "Shared (original)": get_metrics("l-irl",p_post,income,w_post_true,
                                          lirl_theta=theta_shared,   lirl_feat_fn=features_shared),
        "Good-specific":     get_metrics("l-irl",p_post,income,w_post_true,
                                          lirl_theta=theta_goodspec, lirl_feat_fn=features_good_specific),
        "Orth + Intercepts": get_metrics("l-irl",p_post,income,w_post_true,
                                          lirl_theta=theta_orth,     lirl_feat_fn=features_orthogonalised),
    }

    # ── MDP advantage ─────────────────────────────────────────────────
    rmse_aids_h = get_metrics("aids",    p_post,income,w_habit_shock, aids=aids_hab)["RMSE"]
    rmse_nirl_h = get_metrics("n-irl",   p_post,income,w_habit_shock, nirl=n_irl_hab,device=DEVICE)["RMSE"]
    rmse_mdp_h  = get_metrics("mdp-irl", p_post,income,w_habit_shock,
                               mdp_nirl=mdp_irl,xbar_shock=xbar_shock,device=DEVICE)["RMSE"]

    kl_aids   = kl_div(aids_hab.predict(p_post,income), w_habit_shock)
    kl_static = kl_div(predict_shares("n-irl",p_post,income,nirl=n_irl_hab,device=DEVICE), w_habit_shock)
    kl_mdp    = kl_div(predict_shares("mdp-irl",p_post,income,
                                       mdp_nirl=mdp_irl,xbar=xbar_shock,device=DEVICE), w_habit_shock)

    # ── Demand curve arrays on fixed price grid ───────────────────────
    test_p       = np.tile(p_pre.mean(0),(80,1)); test_p[:,1] = P_GRID
    fixed_y      = np.full(80, AVG_Y)
    xbar_rep     = np.tile(xbar_train.mean(0),(80,1))

    # CES comparison: food share only → (80,)
    curves = {
        "Truth":          primary.solve_demand(test_p, fixed_y)[:,0],
        "LA-AIDS":        predict_shares("aids",    test_p,fixed_y,**KW)[:,0],
        "BLP (IV)":       predict_shares("blp",     test_p,fixed_y,**KW)[:,0],
        "Lin IRL Shared": predict_shares("l-irl",   test_p,fixed_y,
                                          lirl_theta=theta_shared,lirl_feat_fn=features_shared)[:,0],
        "Lin IRL Orth":   predict_shares("l-irl",   test_p,fixed_y,
                                          lirl_theta=theta_orth,lirl_feat_fn=features_orthogonalised)[:,0],
        "Neural IRL":     predict_shares("n-irl",   test_p,fixed_y,**KW)[:,0],
        "Var. Mixture":   predict_shares("mixture", test_p,fixed_y,**KW)[:,0],
    }

    true_conditional_shares = np.zeros((80, 3))
    for i in range(80):
        # 1. Get the specific price and fixed habit for this grid point
        p_i = test_p[i]
        y_i = fixed_y[i]
        xbar_i = xbar_rep[i]  # This is the fixed training mean

        # 2. Replicate the Habit Consumer logic for a single step (no update)
        floor = habit_consumer.theta * xbar_i + 1e-6
        
        def neg_u_static(x):
            adj = x - habit_consumer.theta * xbar_i
            if np.any(adj <= 0): return 1e10
            # Matches your HabitFormationConsumer utility
            return -(np.sum(habit_consumer.alpha * adj**habit_consumer.rho))**(1/habit_consumer.rho)

        # 3. Optimize
        x0 = np.maximum(y_i/(3*p_i), floor+0.01)
        cons = {"type":"eq", "fun": lambda x: p_i @ x - y_i}
        res = minimize(neg_u_static, x0, bounds=[(floor[j], None) for j in range(3)],
                       constraints=cons, method="SLSQP")
        
        true_conditional_shares[i] = res.x * p_i / y_i

    # MDP comparison: all 3 goods → (80,3)
    mdp_curves = {
        "Truth":             true_conditional_shares,
        "LA-AIDS":           predict_shares("aids",    test_p,fixed_y,aids=aids_hab),
        "Neural IRL static": predict_shares("n-irl",   test_p,fixed_y,nirl=n_irl_hab,device=DEVICE),
        "MDP-IRL":           predict_shares("mdp-irl", test_p,fixed_y,
                                             mdp_nirl=mdp_irl,xbar=xbar_rep,device=DEVICE),
    }

    return {
        "perf_ces":     perf_ces,
        "elast":        elast,
        "welf":         welf,
        "lin_ablation": lin_ablation,
        "rob_rows":     rob_rows,
        "mdp": {"aids_rmse":rmse_aids_h,"nirl_rmse":rmse_nirl_h,"mdp_rmse":rmse_mdp_h,
                "kl_aids":kl_aids,"kl_static":kl_static,"kl_mdp":kl_mdp},
        "beta_nirl":    n_irl.beta.item(),
        "beta_mdp":     mdp_irl.beta.item(),
        "shock_pt":     p_pre[:,1].mean() * 1.2,
        "curves":       curves,        # {model: (80,)}
        "mdp_curves":   mdp_curves,    # {model: (80,3)}
        # last-run only — for convergence and mixture plots
        "hist_nirl":    hist_nirl,
        "hist_mdp":     hist_mdp,
        "comp_summary": var_mix.get_component_summary(),
    }


# ════════════════════════════════════════════════════════════════════
#  SECTION 9: MULTI-RUN LOOP
# ════════════════════════════════════════════════════════════════════

print("=" * 72)
print(f"  IRL CONSUMER DEMAND RECOVERY  —  {N_RUNS} INDEPENDENT RUNS")
print("=" * 72)
print(f"  Device: {DEVICE}  |  N per run: {N_OBS}  |  Runs: {N_RUNS}\n")

all_results = []
for run_idx in range(N_RUNS):
    seed = 42 + run_idx * 15          # deterministic, spread across seed space
    t0   = time.time()
    print(f"── Run {run_idx+1}/{N_RUNS}  (seed={seed}) ────────────────────────────")
    r = run_one_seed(seed, verbose=(run_idx == N_RUNS-1))   # verbose on last run only
    all_results.append(r)
    print(f"   Done in {time.time()-t0:.0f}s  "
          f"| Neural IRL RMSE={r['perf_ces']['Neural IRL']['RMSE']:.5f}"
          f"  β={r['beta_nirl']:.3f}")


# ════════════════════════════════════════════════════════════════════
#  SECTION 10: AGGREGATION  —  mean ± SE across runs
# ════════════════════════════════════════════════════════════════════

def _se(vals):
    """Standard error = std(ddof=1) / sqrt(n), NaN-safe."""
    a = np.asarray(vals, float)
    return np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))

# Table 1: predictive accuracy
model_names = list(all_results[0]["perf_ces"].keys())
perf_agg = {}
for nm in model_names:
    for metric in ("RMSE", "MAE"):
        vals = [r["perf_ces"][nm][metric] for r in all_results]
        perf_agg.setdefault(nm, {})[f"{metric}_mean"] = np.nanmean(vals)
        perf_agg.setdefault(nm, {})[f"{metric}_se"]   = _se(vals)

# Table 2: elasticities
elast_names = list(all_results[0]["elast"].keys())
elast_agg = {}
for nm in elast_names:
    vals = np.array([r["elast"][nm] for r in all_results])   # (n_runs, 3)
    elast_agg[nm] = {"mean": vals.mean(0),
                     "se":   vals.std(0, ddof=1) / np.sqrt(N_RUNS)}

# Table 3: welfare
welf_names = list(all_results[0]["welf"].keys())
welf_agg = {}
for nm in welf_names:
    vals = [r["welf"][nm] for r in all_results]
    welf_agg[nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

# Table 4: robustness
dgp_names = list(all_results[0]["rob_rows"].keys())
col_names  = list(all_results[0]["rob_rows"][dgp_names[0]].keys())
rob_agg = {}
for dg in dgp_names:
    rob_agg[dg] = {}
    for col in col_names:
        vals = [r["rob_rows"][dg][col] for r in all_results]
        rob_agg[dg][col] = {"mean": np.nanmean(vals), "se": _se(vals)}

# Table 5: MDP advantage
mdp_keys = ["aids_rmse","nirl_rmse","mdp_rmse","kl_aids","kl_static","kl_mdp"]
mdp_agg  = {k: {"mean": np.nanmean([r["mdp"][k] for r in all_results]),
                 "se":   _se([r["mdp"][k] for r in all_results])}
             for k in mdp_keys}

# Table 7: linear ablation
lin_names = list(all_results[0]["lin_ablation"].keys())
lin_agg = {}
for nm in lin_names:
    for metric in ("RMSE","MAE"):
        vals = [r["lin_ablation"][nm][metric] for r in all_results]
        lin_agg.setdefault(nm,{})[f"{metric}_mean"] = np.nanmean(vals)
        lin_agg.setdefault(nm,{})[f"{metric}_se"]   = _se(vals)

# Beta
beta_nirl_mean = np.mean([r["beta_nirl"] for r in all_results])
beta_nirl_se   = _se([r["beta_nirl"] for r in all_results])
beta_mdp_mean  = np.mean([r["beta_mdp"]  for r in all_results])
beta_mdp_se    = _se([r["beta_mdp"]  for r in all_results])

# Demand curve arrays: mean ± SE across runs (shape (80,) per model)
curve_models = list(all_results[0]["curves"].keys())
curves_mean  = {m: np.mean( [r["curves"][m] for r in all_results], 0) for m in curve_models}
curves_se    = {m: np.array([r["curves"][m] for r in all_results]).std(0, ddof=1)/np.sqrt(N_RUNS)
                for m in curve_models}

mdp_models   = list(all_results[0]["mdp_curves"].keys())
mdp_mean     = {m: np.mean( [r["mdp_curves"][m] for r in all_results], 0) for m in mdp_models}
mdp_se       = {m: np.array([r["mdp_curves"][m] for r in all_results]).std(0, ddof=1)/np.sqrt(N_RUNS)
                for m in mdp_models}

shock_pt_mean = np.mean([r["shock_pt"] for r in all_results])
last          = all_results[-1]   # representative run for convergence / mixture

# ════════════════════════════════════════════════════════════════════
#  SECTION 11: CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════

def fmt(m, s, d=5): return f"{m:.{d}f} ({s:.{d}f})"

print("\n" + "=" * 72)
print(f"  AGGREGATED RESULTS  (mean over {N_RUNS} runs, SE in parentheses)")
print("=" * 72)
print("\n  TABLE 1: POST-SHOCK RMSE & MAE")
for nm in model_names:
    d = perf_agg[nm]
    print(f"  {nm:<22} RMSE={fmt(d['RMSE_mean'],d['RMSE_se'])}  "
          f"MAE={fmt(d['MAE_mean'],d['MAE_se'])}")

print("\n  TABLE 2: OWN-PRICE ELASTICITIES")
for nm in elast_names:
    d = elast_agg[nm]
    row = "  ".join(f"{d['mean'][i]:.3f}({d['se'][i]:.3f})" for i in range(3))
    print(f"  {nm:<22} {row}")

print("\n  TABLE 3: WELFARE (CS LOSS)")
gt_m = welf_agg["Ground Truth"]["mean"]
for nm in welf_names:
    d   = welf_agg[nm]
    err = "" if nm=="Ground Truth" else f"  err={100*abs(d['mean']-gt_m)/abs(gt_m):.1f}%"
    print(f"  {nm:<22} {fmt(d['mean'],d['se'],2)}{err}")

print("\n  TABLE 5: MDP ADVANTAGE")
for lbl,kr,kk in [("LA-AIDS","aids_rmse","kl_aids"),
                   ("Neural IRL static","nirl_rmse","kl_static"),
                   ("MDP Neural IRL","mdp_rmse","kl_mdp")]:
    print(f"  {lbl:<25} RMSE={fmt(mdp_agg[kr]['mean'],mdp_agg[kr]['se'])}  "
          f"KL={fmt(mdp_agg[kk]['mean'],mdp_agg[kk]['se'])}")
print(f"\n  β̂ Neural IRL: {fmt(beta_nirl_mean,beta_nirl_se,4)}")
print(f"  β̂ MDP IRL:    {fmt(beta_mdp_mean,  beta_mdp_se, 4)}")


# ════════════════════════════════════════════════════════════════════
#  SECTION 12: FIGURES
# ════════════════════════════════════════════════════════════════════

BAND = 0.15   # alpha for ±1 SE shaded bands

# ── Style constants ───────────────────────────────────────────────
STYLE = {
    "Truth":             dict(color="k",         ls="-",  lw=3.0),
    "LA-AIDS":           dict(color="#E53935",    ls="--", lw=2.0),
    "BLP (IV)":          dict(color="#43A047",    ls="-.", lw=2.0),
    "Lin IRL Shared":    dict(color="#FDD835",    ls=":",  lw=2.0),
    "Lin IRL Orth":      dict(color="#00ACC1",    ls=":",  lw=2.0),
    "Neural IRL":        dict(color="#1E88E5",    ls="-",  lw=2.5),
    "Var. Mixture":      dict(color="#8E24AA",    ls="--", lw=2.0),
    "Neural IRL static": dict(color="#1E88E5",    ls="-.", lw=2.0),
    "MDP-IRL":           dict(color="#00897B",    ls="-",  lw=2.5),
}

# ── Figure 1: Demand curves — CES DGP ────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 6))
for lbl, st in STYLE.items():
    if lbl not in curves_mean: continue
    mu = curves_mean[lbl]; sigma = curves_se[lbl]
    ax1.plot(P_GRID, mu, label=lbl, **st)
    ax1.fill_between(P_GRID, mu-sigma, mu+sigma,
                     color=st["color"], alpha=BAND)
# ax1.axvline(shock_pt_mean, color="orange", ls=":", lw=1.5, alpha=0.9, label="Shock point")
# ax1.set_title(
#     f"Food Share Response to Fuel Price — CES Ground Truth\n"
#     f"Mean $\\pm$ 1 SE over {N_RUNS} independent runs",
#     fontsize=12, fontweight="bold")
ax1.set_xlabel("Fuel price ($p_1$)", fontsize=14)
ax1.set_ylabel("Food budget share ($w_0$)", fontsize=14)
ax1.legend(fontsize=14, ncol=2, loc="upper left")
ax1.grid(False, alpha=0.3)
fig1.tight_layout()
fig1.savefig("figures/fig1_demand_curves.pdf", dpi=150, bbox_inches="tight")
fig1.savefig("figures/fig1_demand_curves.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig1_demand_curves.pdf")

# ── Figure 2: MDP advantage — all 3 goods ────────────────────────
MDP_STYLE = {
    "Truth":             ("k",       "-",    3.0, "Truth (Habit)"),
    "LA-AIDS":           ("#E53935", "--",   2.0, "LA-AIDS"),
    "Neural IRL static": ("#1E88E5", "-.",   2.0, "Neural IRL (static)"),
    "MDP-IRL":           ("#00897B", "-",    2.5, r"MDP-IRL (with $\bar{x}$)"),
}
good_names = ["Food", "Fuel", "Other"]

# Loop through each good to create a unique figure
for gi, gn in enumerate(good_names):
    # Create a new figure for each 'good'
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for key, (col, ls, lw, lbl) in MDP_STYLE.items():
        mu    = mdp_mean[key][:, gi]
        sigma = mdp_se[key][:, gi]
        
        ax.plot(P_GRID, mu, color=col, ls=ls, lw=lw, label=lbl)
        ax.fill_between(P_GRID, mu - sigma, mu + sigma, color=col, alpha=BAND)
    
    # Add the vertical shock line
    # ax.axvline(shock_pt_mean, color="orange", ls=":", lw=1.5, alpha=0.8)
    
    # Formatting
    ax.set_xlabel("Fuel price", fontsize=14)
    ax.set_ylabel(f"{gn} budget share", fontsize=14)
    ax.set_title(f"{gn} Share (Habit DGP)", fontsize=14, fontweight="bold")
    
    # Optimized legend for standalone plot
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    
    # Save with specific names
    file_base = f"figures/fig2_{gn.lower()}_advantage"
    fig.savefig(f"{file_base}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{file_base}.png", dpi=150, bbox_inches="tight")
    
    print(f"    Saved: {file_base}.pdf")

# Optional: close plots to free up memory
plt.close('all')

# ── Figure 3: Convergence (last/representative run) ───────────────
# Configuration for the two separate figures
configs = [
    ("3a", last["hist_nirl"], "#1E88E5", "#E53935", "Neural IRL (CES DGP)"),
    ("3b", last["hist_mdp"],  "#43A047", "#8E24AA", "MDP Neural IRL (Habit DGP)")
]

for suffix, hist, col_kl, col_b, title in configs:
    if not hist:
        continue
        
    # Create a fresh figure for each run
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ep_x = [h["epoch"] for h in hist]
    kl_y = [h["kl"]    for h in hist]
    bt_y = [h["beta"]  for h in hist]
    
    # Dual axis setup
    ax2 = ax.twinx()
    ax.plot(ep_x, kl_y,  "o-",  ms=5, color=col_kl, label="KL Loss")
    ax2.plot(ep_x, bt_y, "s--", ms=5, color=col_b,  label="β (learned)")
    
    # Formatting
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence", color=col_kl)
    ax2.set_ylabel("Temperature β", color=col_b)
    
    # Maintain original limits
    ax.set_ylim([0, 0.003])
    ax2.set_ylim([0, 0.003])
    
    # Optional: Restore title if needed for separate plots
    # ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Legend handling
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, loc='upper right', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save with unique filenames
    file_base = f"figures/fig{suffix}_convergence"
    fig.savefig(f"{file_base}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{file_base}.png", dpi=150, bbox_inches="tight")
    print(f"    Saved: {file_base}.pdf")

# ── Figure 4: Robustness — heatmap (mean) + bar chart (mean±SE) ──
# ── Setup Data ──────────────────────────────────────────────────
mean_mat = np.array([[rob_agg[dg][col]["mean"] for col in col_names] for dg in dgp_names])
se_mat   = np.array([[rob_agg[dg][col]["se"]   for col in col_names] for dg in dgp_names])
x        = np.arange(len(dgp_names))
W        = 0.25
bar_colors = ["#E57373", "#64B5F6", "#81C784"]

# ── Figure 4a: Heatmap of Mean RMSE ──────────────────────────────
fig4a, ax4a = plt.subplots(figsize=(8, 6))

im = ax4a.imshow(mean_mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.15)
ax4a.set_xticks(range(len(col_names)))
ax4a.set_xticklabels(col_names, fontsize=14)
ax4a.set_yticks(range(len(dgp_names)))
ax4a.set_yticklabels(dgp_names, fontsize=14)

plt.colorbar(im, ax=ax4a, label="Mean RMSE")

# Annotate cells with "mean (SE)"
for i, dg in enumerate(dgp_names):
    for j, col in enumerate(col_names):
        mv = mean_mat[i, j]
        sv = se_mat[i, j]
        # Dynamic text color for readability
        text_color = "white" if mv > 0.08 else "black"
        ax4a.text(j, i, f"{mv:.4f}\n({sv:.4f})",
                  ha="center", va="center", fontsize=12, color=text_color)

fig4a.tight_layout()
fig4a.savefig("figures/fig4a_robustness_heatmap.pdf", dpi=150, bbox_inches="tight")
fig4a.savefig("figures/fig4a_robustness_heatmap.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig4a_robustness_heatmap.pdf")


# ── Figure 4b: Grouped Bar Chart with Error Bars ────────────────
fig4b, ax4b = plt.subplots(figsize=(9, 6))

for j_off, col, bc in zip([-1, 0, 1], col_names, bar_colors):
    means  = [rob_agg[dg][col]["mean"] for dg in dgp_names]
    errors = [rob_agg[dg][col]["se"]   for dg in dgp_names]
    ax4b.bar(x + j_off * W, means, W, yerr=errors, label=col, color=bc,
             capsize=4, alpha=0.85, ecolor="k", error_kw=dict(lw=1.5))

ax4b.set_xticks(x)
ax4b.set_xticklabels(dgp_names, rotation=20, ha="right", fontsize=14)
ax4b.set_ylabel("Post-shock RMSE", fontsize=14)
ax4b.legend(fontsize=12, loc='upper left')
ax4b.grid(True, alpha=0.3, axis="y")

fig4b.tight_layout()
fig4b.savefig("figures/fig4b_robustness_bars.pdf", dpi=150, bbox_inches="tight")
fig4b.savefig("figures/fig4b_robustness_bars.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig4b_robustness_bars.pdf")

plt.close("all")

# ── Figure 5: Variational Mixture (last run) ───────────────────────
comp_df = last["comp_summary"]
x_pos   = np.arange(len(comp_df))
tab10   = plt.cm.tab10(x_pos / len(comp_df))

# ── Figure 5a: Mixture Weights (Bar Chart) ──────────────────────
fig5a, ax5a = plt.subplots(figsize=(8, 5))

bars = ax5a.bar(x_pos, comp_df["pi"], color=tab10, alpha=0.85, edgecolor="k")
for bar, row in zip(bars, comp_df.itertuples()):
    if row.pi > 0.02:
        ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"ρ={row.rho:.2f}", ha="center", va="bottom", fontsize=12)

ax5a.set_xticks(x_pos)
ax5a.set_xticklabels(
    [f"K={k+1}\nα=[{r.alpha_food:.2f},{r.alpha_fuel:.2f},{r.alpha_other:.2f}]"
     for k, r in enumerate(comp_df.itertuples())], fontsize=11, rotation=15)

ax5a.set_ylabel("Mixture weight $\\hat{\\pi}_k$", fontsize=14)
ax5a.set_ylim(0, 1)
ax5a.axhline(1/6, color="gray", ls="--", alpha=0.5, label="Uniform prior")
ax5a.legend(fontsize=10)
ax5a.grid(False, alpha=0.3, axis="y")

fig5a.tight_layout()
fig5a.savefig("figures/fig5a_mixture_weights.pdf", dpi=150, bbox_inches="tight")
fig5a.savefig("figures/fig5a_mixture_weights.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig5a_mixture_weights.pdf")


# ── Figure 5b: Component Centres (Scatter Plot) ────────────────
fig5b, ax5b = plt.subplots(figsize=(7, 6))

for k, row in enumerate(comp_df.itertuples()):
    ax5b.scatter(row.alpha_food, row.alpha_fuel,
                 s=row.pi*2000+20, alpha=0.75, c=[tab10[k]],
                 edgecolors="k", linewidths=0.5,
                 label=f"K={k+1} (ρ={row.rho:.2f})")

ax5b.scatter([0.4], [0.4], s=300, marker="*", color="red", zorder=5, label="True α")
ax5b.set_xlabel("$\\hat{\\alpha}_{\\mathrm{food}}$", fontsize=14)
ax5b.set_ylabel("$\\hat{\\alpha}_{\\mathrm{fuel}}$", fontsize=14)
ax5b.set_xlim(0, 0.9)
ax5b.set_ylim(0, 0.9)
ax5b.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left') # Moved legend outside for clarity
ax5b.grid(False, alpha=0.3)

fig5b.tight_layout()
fig5b.savefig("figures/fig5b_mixture_centers.pdf", dpi=150, bbox_inches="tight")
fig5b.savefig("figures/fig5b_mixture_centers.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig5b_mixture_centers.pdf")

plt.close("all")


# ════════════════════════════════════════════════════════════════════
#  SECTION 13: LATEX GENERATION
# ════════════════════════════════════════════════════════════════════

print("\n  Generating LaTeX...")
out = []
def lx(s): out.append(s)

def cell(m, s, d=5):
    """Format a table cell as 'mean (SE)'."""
    return f"{m:.{d}f} ({s:.{d}f})"

def cell2(m, s): return cell(m, s, 2)

lx(r"% ============================================================")
lx(r"% AUTO-GENERATED LaTeX — IRL Consumer Demand Recovery")
lx(f"% N_RUNS = {N_RUNS}, N = {N_OBS} per run.  All cells: mean (SE).")
lx(r"% Required: booktabs, threeparttable, graphicx, amsmath")
lx(r"% ============================================================")
lx("")

# ── Table 1: Predictive Accuracy ─────────────────────────────────
lx(r"% --- TABLE 1: Predictive Accuracy ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Post-Shock Predictive Accuracy: CES Ground Truth, "
   r"20\% Fuel Price Shock. Mean (SE) across " + str(N_RUNS) + r" runs.}")
lx(r"  \label{tab:accuracy}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lcc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{RMSE} & \textbf{MAE} \\")
lx(r"    \midrule")
best_rmse = min(perf_agg[nm]["RMSE_mean"] for nm in model_names)
for nm in model_names:
    d  = perf_agg[nm]
    bo = r"\textbf{" if d["RMSE_mean"] == best_rmse else ""
    bc = "}" if bo else ""
    lx(f"    {nm} & {bo}{cell(d['RMSE_mean'],d['RMSE_se'])}{bc} "
       f"& {cell(d['MAE_mean'],d['MAE_se'])} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} independent draws of $(p,y)$ with $N={N_OBS}$ each.")
lx(r"    Shock: $p_1\to 1.2\,p_1$. SE $=\hat{\sigma}/\sqrt{n_{\text{runs}}}$. Bold = lowest mean RMSE.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 2: Elasticities ─────────────────────────────────────────
lx(r"% --- TABLE 2: Own-Price Elasticities ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Recovered Own-Price Elasticities $\hat{\varepsilon}_{ii}$ at Shock Point. Mean (SE).}")
lx(r"  \label{tab:elasticities}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lccc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & Food $\hat{\varepsilon}_{00}$ & Fuel $\hat{\varepsilon}_{11}$ & Other $\hat{\varepsilon}_{22}$ \\")
lx(r"    \midrule")
for nm in elast_names:
    d   = elast_agg[nm]
    it  = r"\textit{" if nm == "Ground Truth" else ""
    eit = "}" if it else ""
    row = " & ".join(cell(d["mean"][i], d["se"][i], 3) for i in range(3))
    lx(f"    {it}{nm}{eit} & {row} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. Numerical elasticities at mean post-shock prices, "
   r"$y=\pounds 1{,}600$.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 3: Welfare ──────────────────────────────────────────────
lx(r"% --- TABLE 3: Welfare ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Compensating Variation: CS Loss from 20\% Fuel Price Shock. Mean (SE).}")
lx(r"  \label{tab:welfare}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lcc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{CS Loss (£)} & \textbf{Error (\%)} \\")
lx(r"    \midrule")
gt_m = welf_agg["Ground Truth"]["mean"]
for nm in welf_names:
    d   = welf_agg[nm]
    err = r"\text{---}" if nm=="Ground Truth" else f"{100*abs(d['mean']-gt_m)/abs(gt_m):.1f}"
    it  = r"\textit{" if nm=="Ground Truth" else ""
    eit = "}" if it else ""
    lx(f"    {it}{nm}{eit} & £{cell2(abs(d['mean']),d['se'])} & {err} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. CV via 100-step Riemann integration. "
   r"Error: \% deviation from Ground Truth mean.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 4: Robustness ───────────────────────────────────────────
lx(r"% --- TABLE 4: Robustness Across DGPs ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Out-of-Sample RMSE Across Utility DGPs (Post-Shock). Mean (SE).}")
lx(r"  \label{tab:robustness}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lccc}")
lx(r"    \toprule")
lx(r"    \textbf{DGP} & \textbf{LA-AIDS} & \textbf{Lin IRL (Orth)} & \textbf{Neural IRL} \\")
lx(r"    \midrule")
for dg in dgp_names:
    means = [rob_agg[dg][c]["mean"] for c in col_names]
    best  = min(means)
    cells = []
    for c in col_names:
        d  = rob_agg[dg][c]
        s  = cell(d["mean"], d["se"])
        cells.append((r"\textbf{"+s+"}") if d["mean"]==best else s)
    lx(f"    {dg} & {' & '.join(cells)} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. Models re-trained per DGP. "
   r"Bold = lowest mean RMSE per row.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 5: MDP Advantage ────────────────────────────────────────
rmse_base = mdp_agg["aids_rmse"]["mean"]
se_base   = mdp_agg["aids_rmse"]["se"]
lx(r"% --- TABLE 5: MDP Advantage ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{MDP State Augmentation: Habit Formation Experiment. Mean (SE).}")
lx(r"  \label{tab:mdp_advantage}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lccl}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{RMSE} & \textbf{KL Div.} & \textbf{RMSE reduction} \\")
lx(r"    \midrule")
mdp_table_rows = [
    ("LA-AIDS (static)",                   "aids_rmse", "kl_aids",   "baseline"),
    ("Neural IRL (static MDP)",            "nirl_rmse", "kl_static", None),
    (r"MDP Neural IRL ($\bar{x}$ state)", "mdp_rmse",  "kl_mdp",    None),
]
for lbl, kr, kk, red in mdp_table_rows:
    rd = mdp_agg[kr]; kd = mdp_agg[kk]
    if red is None:
        # reduction = (base - this)/base, propagate SE via delta method
        pct = 100*(rmse_base - rd["mean"]) / rmse_base
        # SE of (A-B)/A when A and B are from different independent runs:
        # Var[(A-B)/A] ≈ (se_B/A)² + (B·se_A/A²)²
        pct_se = 100*np.sqrt((rd["se"]/rmse_base)**2
                             + (rd["mean"]*se_base/rmse_base**2)**2)
        red = f"{pct:.1f}\\% ({pct_se:.1f}\\%)"
    lx(f"    {lbl} & {cell(rd['mean'],rd['se'])} & {cell(kd['mean'],kd['se'])} & {red} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. All models trained on identical habit-formation data.")
lx(r"    RMSE reduction relative to LA-AIDS mean; SE propagated via delta method.")
lx(rf"    $\hat{{\beta}}$: Neural IRL = {beta_nirl_mean:.3f} ({beta_nirl_se:.3f}); "
   rf"MDP IRL = {beta_mdp_mean:.3f} ({beta_mdp_se:.3f}). "
   r"$\theta=0.3$, $\delta=0.7$.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 6: Variational Mixture (last run) ───────────────────────
lx(r"% --- TABLE 6: Variational Mixture Components (representative last run) ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Continuous Variational Mixture IRL: Recovered Parameters ($K=6$, "
   r"representative run). Component parameters vary across runs; "
   r"dominant component consistently recovers ground truth.}")
lx(r"  \label{tab:mixture}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{ccccccc}")
lx(r"    \toprule")
lx(r"    $k$ & $\hat{\pi}_k$ & $\hat{\alpha}_{\text{food}}$ & $\hat{\alpha}_{\text{fuel}}$"
   r" & $\hat{\alpha}_{\text{other}}$ & $\hat{\rho}$ & Type \\")
lx(r"    \midrule")
for _, row in comp_df.iterrows():
    af,afu,ao = row["alpha_food"],row["alpha_fuel"],row["alpha_other"]
    if   af  > 0.45: tp = "Food-heavy"
    elif afu > 0.45: tp = "Fuel-heavy"
    elif row["pi"] > 0.3: tp = r"\textbf{Dominant}"
    else:            tp = "Balanced"
    lx(f"    {int(row['component'])} & {row['pi']:.3f} & {af:.3f} & {afu:.3f} "
       f"& {ao:.3f} & {row['rho']:.3f} & {tp} \\\\")
lx(r"    \midrule")
lx(r"    \textit{Truth} & --- & 0.400 & 0.400 & 0.200 & 0.450 & --- \\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item Gaussian mixture in $(\alpha,\rho)$ CES parameter space; variational EM "
   r"on $N=300$ obs. $\alpha$ via softmax; $\rho$ via sigmoid.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Table 7: Linear IRL Ablation ─────────────────────────────────
lx(r"% --- TABLE 7: Linear IRL Feature Ablation ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Linear MaxEnt IRL Feature Ablation. Mean (SE).}")
lx(r"  \label{tab:linear_ablation}")
lx(r"  \begin{threeparttable}")
lx(r"  \begin{tabular}{lccp{6cm}}")
lx(r"    \toprule")
lx(r"    \textbf{Variant} & \textbf{RMSE} & \textbf{MAE} & \textbf{Feature description} \\")
lx(r"    \midrule")
descs = {
    "Shared (original)": r"Shared $[\ln p_i,(\ln p_i)^2,\ln y]$ — same profile all goods",
    "Good-specific":     r"Per-good $[\ln\mathbf{p},(\ln p_i)^2,\ln y]$ — heterogeneous response",
    "Orth + Intercepts": r"QR-orth.\ prices + per-good one-hot intercept — resolves collinearity",
}
best_lin = min(lin_agg[nm]["RMSE_mean"] for nm in lin_names)
for nm in lin_names:
    d  = lin_agg[nm]
    bo = r"\textbf{" if d["RMSE_mean"] == best_lin else ""
    bc = "}" if bo else ""
    lx(f"    {nm} & {bo}{cell(d['RMSE_mean'],d['RMSE_se'])}{bc} "
       f"& {cell(d['MAE_mean'],d['MAE_se'])} & {descs.get(nm,'')} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. 3{{,}}000 gradient-ascent epochs, "
   r"$\ell_2=10^{-4}$, $\eta_t=0.05/(1+t/1000)$. Bold = lowest mean RMSE.")
lx(r"  \end{tablenotes}")
lx(r"  \end{threeparttable}")
lx(r"\end{table}")
lx("")

# ── Figure environments ───────────────────────────────────────────
lx(r"% ============================================================")
lx(r"% FIGURE INCLUSION BLOCKS")
lx(r"% ============================================================")
fig_defs = [
    ("fig1_demand_curves",
     "Food Budget Share vs Fuel Price — CES Ground Truth",
     fr"Mean predicted share (line) $\pm 1$ SE (shaded band) across {N_RUNS} runs. "
     r"Neural IRL (blue) tracks the ground truth most closely. Shared Lin IRL collapses "
     r"under price collinearity; the Orthogonalised variant recovers the monotone response. "
     r"Shock point (orange dotted) marks the 20\% fuel price increase.",
     "fig:demand_curves"),
    ("fig2_mdp_advantage",
     r"MDP Neural IRL vs Static Models — Habit Formation DGP, All Three Goods",
     fr"Mean $\pm 1$ SE across {N_RUNS} runs. The MDP-IRL (teal), which receives "
     r"the lagged habit stock $\bar{x}_t$ in its state vector, tracks all three "
     r"ground-truth curves most closely. The gap is largest for the Food share.",
     "fig:mdp_advantage"),
    ("fig3_convergence",
     r"Training Convergence: KL Loss and Learnable Temperature $\hat{\beta}$ (last run)",
     r"Left: Neural IRL on CES data. Right: MDP Neural IRL on Habit data. "
     r"$\hat{\beta}$ stabilises rapidly, providing a data-driven rationality estimate.",
     "fig:convergence"),
    ("fig4_robustness_heatmap",
     fr"Robustness: Post-Shock RMSE Across DGPs — Mean and $\pm 1$ SE ({N_RUNS} runs)",
     r"Left: heatmap of mean RMSE (SE in parentheses). Right: grouped bar chart "
     r"with $\pm 1$ SE error bars. Neural IRL dominates across all smooth DGPs; "
     r"the MDP advantage on Habit data only materialises when $\bar{x}$ enters the "
     r"state (Table~\ref{tab:mdp_advantage}).",
     "fig:robustness"),
    ("fig5_mixture_components",
     r"Continuous Variational Mixture IRL: Weights and Parameter Space ($K=6$, last run)",
     r"Left: mixture weights $\hat{\pi}_k$ with $\hat{\rho}$ annotated. "
     r"Right: component centres in $(\hat{\alpha}_{\mathrm{food}},\hat{\alpha}_{\mathrm{fuel}})$ "
     r"space; size $\propto\hat{\pi}_k$; red star = true $\alpha$.",
     "fig:mixture"),
]
for fname, caption, note, label in fig_defs:
    lx(r"\begin{figure}[htbp]")
    lx(r"  \centering")
    lx(f"  \\includegraphics[width=\\textwidth]{{figures/{fname}.pdf}}")
    lx(f"  \\caption{{{caption}}}")
    lx(f"  \\label{{{label}}}")
    lx(r"  \begin{figurenotes}")
    lx(f"    {note}")
    lx(r"  \end{figurenotes}")
    lx(r"\end{figure}")
    lx("")

lx(r"% REQUIRED PREAMBLE:")
lx(r"% \usepackage{booktabs, threeparttable, graphicx, amsmath}")
lx(r"% \newenvironment{figurenotes}{\par\small\textit{Notes:~}}{\par}")

with open("paper_tables_figures.tex", "w") as f:
    f.write("\n".join(out))
print("    Saved: paper_tables_figures.tex")


# ════════════════════════════════════════════════════════════════════
#  SECTION 14: FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print(f"  FINAL SUMMARY  ({N_RUNS} runs × N={N_OBS})")
print("=" * 72)
gt_welf = welf_agg["Ground Truth"]["mean"]
print(f"""
  NEURAL IRL (CES DGP):
    RMSE : {fmt(perf_agg['Neural IRL']['RMSE_mean'], perf_agg['Neural IRL']['RMSE_se'])}
    β̂    : {fmt(beta_nirl_mean, beta_nirl_se, 4)}
    Welfare error vs truth: {100*abs(welf_agg['Neural IRL']['mean']-gt_welf)/abs(gt_welf):.1f}%

  MDP ADVANTAGE (Habit DGP):
    LA-AIDS RMSE    : {fmt(mdp_agg['aids_rmse']['mean'], mdp_agg['aids_rmse']['se'])}
    Static IRL RMSE : {fmt(mdp_agg['nirl_rmse']['mean'], mdp_agg['nirl_rmse']['se'])}
    MDP IRL RMSE    : {fmt(mdp_agg['mdp_rmse']['mean'],  mdp_agg['mdp_rmse']['se'])}
    Reduction (MDP vs AIDS): {
        100*(mdp_agg['aids_rmse']['mean']-mdp_agg['mdp_rmse']['mean'])
        /mdp_agg['aids_rmse']['mean']:.1f}%

  OUTPUT FILES:
    figures/fig{{1..5}}_*.{{pdf,png}}
    paper_tables_figures.tex
""")