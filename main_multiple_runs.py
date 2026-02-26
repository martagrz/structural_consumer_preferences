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
    QUAIDS,
    SeriesDemand,
    CESConsumer,
    ContinuousVariationalMixture,
    HabitFormationConsumer,
    LeontiefConsumer,
    MDPNeuralIRL,
    MDPNeuralIRL_E2E,
    NeuralIRL,
    QuasilinearConsumer,
    StoneGearyConsumer,
    WindowIRL,
    build_window_features,
    cf_first_stage,
    compute_xbar_e2e,
    features_good_specific,
    features_orthogonalised,
    features_shared,
    predict_linear_irl,
    run_linear_irl,
    train_mdp_e2e,
    train_neural_irl,
    train_window_irl,
)
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION  ← only section you need to edit
# ════════════════════════════════════════════════════════════════════

N_RUNS   = 5      # number of independent replications
N_OBS    = 1000    # observations per run

EPOCHS = 1000

# Grid of candidate δ values used in the frozen-δ sweep.
# δ is NEVER jointly learned; for each candidate we train a separate frozen
# model, pick δ̂ = argmin validation-KL, and report the identified set as all
# δ values within 2 × SE of the minimum.
MDP_DELTA_GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("figures", exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  SECTION 1-6: OBJECT DEFINITIONS MOVED TO src/simulation/models.py
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
#  SECTION 7: UNIFIED PREDICTION & EVALUATION UTILITIES
# ════════════════════════════════════════════════════════════════════

def predict_shares(spec, p, y, *, aids=None, blp=None, quaids=None, series=None,
                   lirl_theta=None,
                   lirl_feat_fn=None, nirl=None, mdp_nirl=None,
                   xbar=None, q_prev=None, consumer=None, mixture=None,
                   mdp_e2e=None, xbar_e2e=None,
                   wirl=None, wirl_log_p_hist=None, wirl_log_q_hist=None,
                   wirl_window=4,
                   nirl_cf=None, mdp_nirl_cf=None,
                   device="cpu"):
    if spec == "truth":   return consumer.solve_demand(p, y)
    if spec == "aids":    return aids.predict(p, y)
    if spec == "blp":     return blp.predict(p)[:, :3]   # drop outside-option col
    if spec == "quaids":  return quaids.predict(p, y)
    if spec == "series":  return series.predict(p, y)
    if spec == "window-irl":
        # Build feature matrix using fixed mean history (structural interpretation)
        G      = p.shape[1]
        lp_cur = np.log(np.maximum(p, 1e-8))
        ly_cur = np.log(np.maximum(y, 1e-8))
        in_dim = G + 1 + wirl_window * 2 * G
        feats  = np.zeros((len(p), in_dim), dtype=np.float32)
        hist_f = np.concatenate(
            [wirl_log_p_hist, wirl_log_q_hist] * wirl_window)
        for i in range(len(p)):
            feats[i] = np.concatenate([lp_cur[i], [ly_cur[i]], hist_f])
        with torch.no_grad():
            xt = torch.tensor(feats, dtype=torch.float32, device=device)
            return wirl(xt).cpu().numpy()
    if spec == "l-irl":   return predict_linear_irl(lirl_feat_fn(p,y), lirl_theta)
    if spec == "n-irl":
        with torch.no_grad():
            lp = torch.log(torch.tensor(p, dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(y, dtype=torch.float32, device=device)).unsqueeze(1)
            return nirl(lp,ly).cpu().numpy()
    if spec == "mdp-irl":
        with torch.no_grad():
            lp  = torch.log(torch.tensor(p,         dtype=torch.float32, device=device))
            ly  = torch.log(torch.tensor(y,         dtype=torch.float32, device=device)).unsqueeze(1)
            xbp = torch.log(torch.clamp(torch.tensor(xbar,   dtype=torch.float32, device=device), min=1e-6))
            qp  = torch.log(torch.clamp(torch.tensor(q_prev, dtype=torch.float32, device=device), min=1e-6))
            return mdp_nirl(lp, ly, xbp, qp).cpu().numpy()
    if spec == "mdp-e2e":
        with torch.no_grad():
            lp = torch.log(torch.clamp(torch.tensor(p, dtype=torch.float32, device=device), min=1e-8))
            ly = torch.log(torch.clamp(torch.tensor(y, dtype=torch.float32, device=device), min=1e-8)).unsqueeze(1)
            xb = torch.tensor(xbar_e2e, dtype=torch.float32, device=device)
            return mdp_e2e(lp, ly, xb).cpu().numpy()
    if spec == "mixture": return mixture.predict(p, y)
    # ── Control-function (CF) model specs ──────────────────────────────────
    # Always evaluated with v_hat = 0 (structural / counterfactual mode so
    # the endogenous component is zeroed out).
    if spec == "n-irl-cf":
        with torch.no_grad():
            lp = torch.log(torch.clamp(torch.tensor(p, dtype=torch.float32, device=device), min=1e-8))
            ly = torch.log(torch.clamp(torch.tensor(y, dtype=torch.float32, device=device), min=1e-8)).unsqueeze(1)
            vh = torch.zeros(len(p), nirl_cf.n_cf, dtype=torch.float32, device=device)
            return nirl_cf(lp, ly, v_hat=vh).cpu().numpy()
    if spec == "mdp-irl-cf":
        with torch.no_grad():
            lp  = torch.log(torch.clamp(torch.tensor(p,      dtype=torch.float32, device=device), min=1e-8))
            ly  = torch.log(torch.clamp(torch.tensor(y,      dtype=torch.float32, device=device), min=1e-8)).unsqueeze(1)
            xbp = torch.log(torch.clamp(torch.tensor(xbar,   dtype=torch.float32, device=device), min=1e-6))
            qp  = torch.log(torch.clamp(torch.tensor(q_prev, dtype=torch.float32, device=device), min=1e-6))
            vh  = torch.zeros(len(p), mdp_nirl_cf.n_cf, dtype=torch.float32, device=device)
            return mdp_nirl_cf(lp, ly, xbp, qp, v_hat=vh).cpu().numpy()
    raise ValueError(spec)

def compute_elasticities(spec, p_pt, y_pt, h=1e-4, xbar_pt=None, q_prev_pt=None, **kw):
    w0  = predict_shares(spec, p_pt.reshape(1,-1), np.array([y_pt]),
                         xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None,
                         q_prev=q_prev_pt.reshape(1,-1) if q_prev_pt is not None else None,
                         **kw)[0]
    eps = []
    for i in range(3):
        p1 = p_pt.copy().reshape(1,-1); p1[0,i] *= (1+h)
        wp = predict_shares(spec, p1, np.array([y_pt]),
                            xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None,
                            q_prev=q_prev_pt.reshape(1,-1) if q_prev_pt is not None else None,
                            **kw)[0]
        eps.append(((wp[i]-w0[i])/w0[i])/h - 1)
    return eps


def compute_full_elasticity_matrix(spec, p_pt, y_pt, h=1e-4,
                                   xbar_pt=None, q_prev_pt=None, **kw):
    """Return (3, 3) demand elasticity matrix.

    eps[i, j] = d log x_j / d log p_i
              = (Δw_j / w_j) / (Δp_i / p_i) - δ_{ij}

    Diagonal  = own-price demand elasticity  (expected < 0).
    Off-diag  = cross-price demand elasticity (> 0 = substitutes,
                                               < 0 = complements).

    CES with ρ=0.45 → σ≈1.82: goods are substitutes, so off-diagonal
    entries should be positive.  Static benchmarks (LA-AIDS, QUAIDS, Series)
    recover elasticities from within-sample price variation; neural IRL models
    can recover negative cross-prices when warranted.
    """
    _xbkw = dict(
        xbar   = xbar_pt.reshape(1, -1)   if xbar_pt   is not None else None,
        q_prev = q_prev_pt.reshape(1, -1) if q_prev_pt is not None else None,
    )
    w0 = predict_shares(spec, p_pt.reshape(1, -1), np.array([y_pt]),
                        **_xbkw, **kw)[0]
    eps = np.zeros((3, 3))
    for i in range(3):
        p1 = p_pt.copy().reshape(1, -1); p1[0, i] *= (1 + h)
        w1 = predict_shares(spec, p1, np.array([y_pt]),
                            **_xbkw, **kw)[0]
        for j in range(3):
            eps[i, j] = ((w1[j] - w0[j]) / max(w0[j], 1e-9)) / h - (1 if i == j else 0)
    return eps

def compute_welfare_loss(spec, p0, p1, y, steps=100, xbar_pt=None, q_prev_pt=None, **kw):
    path = np.linspace(p0,p1,steps); dp = (p1-p0)/steps; loss = 0.0
    for t in range(steps):
        w    = predict_shares(spec, path[t:t+1], np.array([y]),
                              xbar=xbar_pt.reshape(1,-1) if xbar_pt is not None else None,
                              q_prev=q_prev_pt.reshape(1,-1) if q_prev_pt is not None else None,
                              **kw)[0]
        loss -= (w*y/path[t]) @ dp
    return loss

def get_metrics(spec, p_shock, income, w_true, xbar_shock=None, q_prev_shock=None, **kw):
    wp = predict_shares(spec, p_shock, income, xbar=xbar_shock, q_prev=q_prev_shock, **kw)
    return {"RMSE": np.sqrt(mean_squared_error(w_true, wp)),
            "MAE":  mean_absolute_error(w_true, wp)}

def kl_div(w_pred, w_true):
    wp = np.clip(w_pred,1e-8,1); wt = np.clip(w_true,1e-8,1)
    return float(np.mean(np.sum(wt*np.log(wt/wp), axis=1)))


# ════════════════════════════════════════════════════════════════════
#  δ GRID-SWEEP HELPER
#
#  Replaces end-to-end (joint-gradient) δ learning throughout.
#  For each δ in MDP_DELTA_GRID:
#    1. Build MDPNeuralIRL_E2E with log_delta frozen at that value.
#    2. Train the network weights normally.
#    3. Evaluate per-observation KL on the held-out validation set.
#  δ̂ = argmin KL_val.
#  Identified set = {δ : KL_val(δ) ≤ KL_val(δ̂) + se_multiplier × SE(δ̂)}.
# ════════════════════════════════════════════════════════════════════

def fit_mdp_delta_grid(
    p_tr,  y_tr,  w_tr,  lq_tr,
    p_val, y_val, w_val, lq_val,
    delta_grid     = None,
    epochs         = EPOCHS,
    device         = "cpu",
    store_ids_tr   = None,
    store_ids_val  = None,
    n_goods        = 3,
    hidden         = 256,
    lam_mono       = 0.3,
    lam_slut       = 0.1,
    batch          = 256,
    lr             = 5e-4,
    xbar_recompute = 10,
    se_multiplier  = 2.0,
    tag            = "delta-sweep",
    model_class    = None,
    extra_model_kw = None,
    pred_spec      = None,
    pred_model_key = None,
    pred_xbar_key  = None,
):
    """Train one frozen-δ model per grid point; pick δ̂ by minimum hold-out KL.

    Parameters
    ----------
    p_tr, y_tr, w_tr, lq_tr   : (N_tr, G), (N_tr,), (N_tr, G), (N_tr, G) — training data
    p_val, y_val, w_val, lq_val: analogous validation arrays
    delta_grid     : 1-D array of candidate δ; defaults to MDP_DELTA_GRID
    model_class    : class, default MDPNeuralIRL_E2E
    extra_model_kw : extra kwargs for model_class (e.g. n_stores, emb_dim for FE)
    pred_spec      : predict_shares spec, default "mdp-e2e" or "mdp-e2e-fb"
    pred_model_key : kwarg name for model in predict_shares, e.g. "mdp_e2e"
    pred_xbar_key  : kwarg name for xbar in predict_shares, e.g. "xbar_e2e"
    se_multiplier  : identified set threshold c (default 2 ≈ 95 % CI)

    Returns dict with:
        best_model : trained MDPNeuralIRL_E2E at δ̂
        delta_hat  : float — point estimate δ̂
        kl_grid    : (K,) — mean validation KL at each grid point
        se_grid    : (K,) — SE of per-obs KL at each grid point
        id_set     : (lo, hi) — identified-set interval
        id_mask    : (K,) bool
        all_models : {float(δ): model} for every grid point
    """
    if delta_grid is None:
        delta_grid = MDP_DELTA_GRID
    if model_class is None:
        model_class = MDPNeuralIRL_E2E
    if extra_model_kw is None:
        extra_model_kw = {}

    # resolve predict_shares spec / kwarg names
    _spec = pred_spec      or "mdp-e2e"
    _mk   = pred_model_key or "mdp_e2e"
    _xk   = pred_xbar_key  or "xbar_e2e"

    delta_grid  = np.asarray(delta_grid, dtype=float)
    K           = len(delta_grid)
    kl_grid     = np.zeros(K)
    se_grid     = np.zeros(K)
    all_models  = {}
    all_hists   = {}

    lq_val_t = torch.tensor(lq_val, dtype=torch.float32, device=device)

    for k, d in enumerate(delta_grid):
        # ── build model with δ frozen ────────────────────────────────────────
        mkw = dict(n_goods=n_goods, hidden_dim=hidden, delta_init=float(d))
        mkw.update(extra_model_kw)
        model = model_class(**mkw)
        model.log_delta.requires_grad_(False)   # freeze δ — no joint learning

        model, _hist = train_mdp_e2e(
            model, p_tr, y_tr, w_tr, lq_tr,
            store_ids=store_ids_tr,
            epochs=epochs, lr=lr, batch_size=batch,
            lam_mono=lam_mono, lam_slut=lam_slut,
            slut_start_frac=0.25,
            xbar_recompute_every=xbar_recompute,
            device=device, tag=f"{tag}-d{d:.2f}",
        )

        # ── validation KL ────────────────────────────────────────────────────
        d_t = torch.tensor(float(d), dtype=torch.float32, device=device)
        with torch.no_grad():
            xb_val = compute_xbar_e2e(
                d_t, lq_val_t, store_ids=store_ids_val).cpu().numpy()
        wp = predict_shares(_spec, p_val, y_val,
                            **{_mk: model, _xk: xb_val}, device=device)
        wt          = np.clip(w_val, 1e-8, 1.0)
        wp          = np.clip(wp,    1e-8, 1.0)
        kl_per_obs  = np.sum(wt * np.log(wt / wp), axis=1)   # (N_val,)
        kl_grid[k]  = float(kl_per_obs.mean())
        se_grid[k]  = float(kl_per_obs.std(ddof=min(1, len(kl_per_obs) - 1))
                            / np.sqrt(max(len(kl_per_obs), 1)))
        all_models[float(d)] = model
        all_hists[float(d)]  = _hist
        print(f"    {tag} δ={d:.2f}: val-KL={kl_grid[k]:.6f} ± {se_grid[k]:.6f}")

    # ── point estimate: δ̂ = argmin KL_val ───────────────────────────────────
    best_k     = int(np.argmin(kl_grid))
    delta_hat  = float(delta_grid[best_k])
    best_model = all_models[delta_hat]
    best_hist  = all_hists[delta_hat]

    # ── identified set: KL(δ) ≤ KL(δ̂) + se_multiplier × SE(δ̂) ─────────────
    threshold = kl_grid[best_k] + se_multiplier * se_grid[best_k]
    id_mask   = kl_grid <= threshold
    id_deltas = delta_grid[id_mask]
    id_set    = ((float(id_deltas.min()), float(id_deltas.max()))
                 if id_mask.any() else (delta_hat, delta_hat))

    print(f"  → {tag} δ̂={delta_hat:.2f}  IS=[{id_set[0]:.2f}, {id_set[1]:.2f}]  "
          f"(KL_min={kl_grid[best_k]:.6f})")
    return {
        "best_model": best_model,
        "best_hist":  best_hist,
        "delta_hat":  delta_hat,
        "kl_grid":    kl_grid,
        "se_grid":    se_grid,
        "id_set":     id_set,
        "id_mask":    id_mask,
        "all_models": all_models,
        "all_hists":  all_hists,
    }


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

    # ── Fresh validation set for δ grid sweep ────────────────────────────────
    # Independent draw from the same Habit DGP; used to select δ̂ without
    # contaminating the shock-period test evaluation.
    _val_rng = np.random.default_rng(seed + 99999)
    _N_val   = max(N // 5, 100)
    _Z_val   = _val_rng.uniform(1, 5, (_N_val, 3))
    _p_val   = np.clip(_Z_val + _val_rng.normal(0, 0.1, (_N_val, 3)), 1e-3, None)
    _y_val   = _val_rng.uniform(1200, 2000, _N_val)
    _hc_val  = HabitFormationConsumer()
    _w_val, _ = _hc_val.solve_demand(_p_val, _y_val, return_xbar=True)
    _q_val   = _w_val * _y_val[:, None] / np.maximum(_p_val, 1e-8)
    _lq_val  = np.log(np.maximum(_q_val, 1e-6))

    p_post           = p_pre.copy(); p_post[:,1] *= 1.2
    w_post_true      = primary.solve_demand(p_post, income)
    w_habit_shock, xbar_shock = habit_consumer.solve_demand(p_post, income, return_xbar=True)

    # ── Compute previous-period quantities for MDP model ──────────────────
    # q_train[i] ≈ quantity vector at observation i (from budget shares)
    q_train = w_habit * income[:, None] / np.maximum(p_pre, 1e-8)
    # q_prev_train[i] = quantities at period i-1 (first obs uses its own value)
    q_prev_train = np.zeros_like(q_train)
    q_prev_train[0]  = q_train[0]
    q_prev_train[1:] = q_train[:-1]

    q_shock = w_habit_shock * income[:, None] / np.maximum(p_post, 1e-8)
    q_prev_shock = np.zeros_like(q_shock)
    q_prev_shock[0]  = q_shock[0]
    q_prev_shock[1:] = q_shock[:-1]

    avg_p    = p_post.mean(0)
    p_pre_pt = avg_p / np.array([1.0, 1.2, 1.0])

    # ── Benchmarks ───────────────────────────────────────────────────
    aids_m   = AIDSBench();    aids_m.fit(p_pre, w_train, income)
    quaids_m = QUAIDS();       quaids_m.fit(p_pre, w_train, income)
    series_m = SeriesDemand(); series_m.fit(p_pre, w_train, income)

    # BLP-IV: requires (N, G+1) market shares including an outside option.
    # In simulation there is no genuine outside good; we add a 1 % constant
    # outside share and rescale the inside shares proportionally.
    _blp_out  = 0.01
    mw_train  = np.column_stack([w_train * (1 - _blp_out),
                                  np.full(N, _blp_out)])
    blp_m = BLPBench()
    blp_m.fit(p_pre, mw_train, Z)   # Z is the cost-draw instrument

    aids_hab   = AIDSBench();    aids_hab.fit(p_pre, w_habit, income)
    quaids_hab = QUAIDS();       quaids_hab.fit(p_pre, w_habit, income)
    series_hab = SeriesDemand(); series_hab.fit(p_pre, w_habit, income)

    mw_habit = np.column_stack([w_habit * (1 - _blp_out),
                                 np.full(N, _blp_out)])
    blp_hab = BLPBench()
    blp_hab.fit(p_pre, mw_habit, Z)

    # ── Linear IRL ───────────────────────────────────────────────────
    theta_shared   = run_linear_irl(features_shared(p_pre, income),         w_train)
    theta_goodspec = run_linear_irl(features_good_specific(p_pre, income),  w_train)
    theta_orth     = run_linear_irl(features_orthogonalised(p_pre, income), w_train)

    # ── Neural IRL (CES) ─────────────────────────────────────────────
    n_irl = NeuralIRL(n_goods=3, hidden_dim=256)
    n_irl, hist_nirl = train_neural_irl(
        n_irl, p_pre, income, w_train, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        device=DEVICE, verbose=verbose)

    # ── Neural IRL static (Habit baseline) ───────────────────────────
    n_irl_hab = NeuralIRL(n_goods=3, hidden_dim=128)
    n_irl_hab, hist_nirl_hab = train_neural_irl(
        n_irl_hab, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # ── MDP Neural IRL (original: pre-computed x̄ with fixed δ) ──────────
    mdp_irl = MDPNeuralIRL(n_goods=3, hidden_dim=256)
    mdp_irl, hist_mdp = train_neural_irl(
        mdp_irl, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xb_prev_data=xbar_train, q_prev_data=q_prev_train,
        device=DEVICE, verbose=verbose)

    # ── MDP Neural IRL — frozen-δ grid sweep ─────────────────────────────
    # δ is NOT jointly learned.  For each candidate in MDP_DELTA_GRID we train
    # a model with log_delta.requires_grad_(False), evaluate hold-out KL, and
    # select δ̂ = argmin.  The identified set is all δ within 2 SE of the min.
    log_q_seq = np.log(np.maximum(q_train, 1e-6))  # (N, G) — actual consumed qtys

    _sweep_e2e = fit_mdp_delta_grid(
        p_pre, income, w_habit, log_q_seq,
        _p_val, _y_val, _w_val, _lq_val,
        delta_grid=MDP_DELTA_GRID, epochs=EPOCHS, device=DEVICE,
        n_goods=3, hidden=256,
        lam_mono=0.3, lam_slut=0.1, batch=256, lr=5e-4,
        tag="E2E")
    mdp_e2e = _sweep_e2e["best_model"]

    # ── Control-Function (CF) endogeneity correction ──────────────────────
    # In the simulation prices = Z + noise, so Z (the pre-noise cost draw)
    # is a valid instrument for log(price).  We use log(Z) as the IV and
    # run per-good OLS first stages to get v̂.
    _log_p_pre = np.log(np.maximum(p_pre, 1e-8))
    _log_Z     = np.log(np.maximum(Z, 1e-8))   # Z defined at top of run_one_seed
    v_hat_train, _cf_rsq = cf_first_stage(_log_p_pre, _log_Z)
    if verbose:
        print(f"   CF first-stage R²: {_cf_rsq.round(3)}")

    # Neural IRL + CF (fit on CES DGP; uses w_train so endogeneity correction
    # demonstrates robustness rather than a real endogeneity problem here)
    n_irl_cf = NeuralIRL(n_goods=3, hidden_dim=256, n_cf=3)
    n_irl_cf, _ = train_neural_irl(
        n_irl_cf, p_pre, income, w_train, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        v_hat_data=v_hat_train, device=DEVICE)

    # MDP Neural IRL + CF (fit on habit DGP)
    mdp_irl_cf = MDPNeuralIRL(n_goods=3, hidden_dim=256, n_cf=3)
    mdp_irl_cf, _ = train_neural_irl(
        mdp_irl_cf, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xb_prev_data=xbar_train, q_prev_data=q_prev_train,
        v_hat_data=v_hat_train, device=DEVICE)

    # log sequences for E2E KL profile sweep and Window IRL
    log_p_seq = np.log(np.maximum(p_pre,  1e-8))
    log_y_seq = np.log(np.maximum(income, 1e-8))

    # ── Window IRL (CES DGP) ─────────────────────────────────────────
    # Uses the last 4 (log-price, log-quantity) lags plus current (log-price,
    # log-income) as the feature vector; no parametric habit assumption.
    _WIRL_W = 4
    q_ces = w_train * income[:, None] / np.maximum(p_pre, 1e-8)
    log_q_ces = np.log(np.maximum(q_ces, 1e-6))
    wf_ces_tr = build_window_features(log_p_seq, log_y_seq, log_q_ces,
                                      window=_WIRL_W, store_ids=None)
    wirl_ces = WindowIRL(n_goods=3, hidden_dim=256, window=_WIRL_W)
    wirl_ces, hist_wirl_ces = train_window_irl(
        wirl_ces, wf_ces_tr, w_train, epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        device=DEVICE, verbose=verbose, tag="Window-IRL-CES")

    # ── Window IRL (Habit DGP) ───────────────────────────────────────
    wf_hab_tr = build_window_features(log_p_seq, log_y_seq, log_q_seq,
                                      window=_WIRL_W, store_ids=None)
    wirl_hab = WindowIRL(n_goods=3, hidden_dim=256, window=_WIRL_W)
    wirl_hab, hist_wirl_hab = train_window_irl(
        wirl_hab, wf_hab_tr, w_habit, epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE, tag="Window-IRL-Habit")

    # Mean training log-price and log-quantity for structural (fixed-history)
    # Window IRL predictions (elasticities, welfare, demand curves).
    _wirl_lp_mean   = log_p_seq.mean(0)   # (G,) — used as fixed history
    _wirl_lq_mean   = log_q_ces.mean(0)
    _wirl_lq_h_mean = log_q_seq.mean(0)   # habit-DGP version

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
            th_rob = run_linear_irl(features_orthogonalised(p_pre,income), w_dgp, epochs=EPOCHS)
            n_rob  = NeuralIRL(n_goods=3, hidden_dim=128)
            n_rob,_ = train_neural_irl(n_rob, p_pre, income, w_dgp,
                          epochs=EPOCHS, lr=5e-4, batch_size=256,
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
    KW = dict(aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
              nirl=n_irl, consumer=primary,
              mixture=var_mix, device=DEVICE)
    MODELS_CES = [
        ("LA-AIDS",          "aids",    {}),
        ("BLP (IV)",         "blp",     {}),
        ("QUAIDS",           "quaids",  {}),
        ("Series Est.",      "series",  {}),
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

    # ── Full 3×3 cross-price elasticity matrices ──────────────────────
    # Only computed for the 4 most important comparators to keep wall time low.
    _cp_specs = [
        ("Ground Truth", "truth",   {}),
        ("LA-AIDS",      "aids",    {}),
        ("BLP (IV)",     "blp",     {}),
        ("QUAIDS",       "quaids",  {}),
        ("Neural IRL",   "n-irl",   {}),
    ]
    cross_elast = {
        nm: compute_full_elasticity_matrix(sp, avg_p, AVG_Y, **{**KW, **ek})
        for nm, sp, ek in _cp_specs
    }

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
                               mdp_nirl=mdp_irl, xbar_shock=xbar_shock,
                               q_prev_shock=q_prev_shock, device=DEVICE)["RMSE"]

    kl_aids   = kl_div(aids_hab.predict(p_post,income), w_habit_shock)
    kl_static = kl_div(predict_shares("n-irl",p_post,income,nirl=n_irl_hab,device=DEVICE), w_habit_shock)
    kl_mdp    = kl_div(predict_shares("mdp-irl",p_post,income,
                                       mdp_nirl=mdp_irl, xbar=xbar_shock,
                                       q_prev=q_prev_shock, device=DEVICE), w_habit_shock)

    # ── E2E MDP and Window IRL — habit advantage ──────────────────────
    # Compute x̄ for shock period using E2E model's learned δ
    log_q_shock_seq = np.log(np.maximum(q_shock, 1e-6))
    with torch.no_grad():
        lq_t = torch.tensor(log_q_shock_seq, dtype=torch.float32, device=DEVICE)
        xbar_shock_e2e_t = compute_xbar_e2e(
            mdp_e2e.delta.to(DEVICE), lq_t, store_ids=None)
    xbar_shock_e2e    = xbar_shock_e2e_t.cpu().numpy()

    rmse_mdp_e2e_h = get_metrics("mdp-e2e", p_post, income, w_habit_shock,
                                  mdp_e2e=mdp_e2e, xbar_e2e=xbar_shock_e2e,
                                  device=DEVICE)["RMSE"]
    kl_mdp_e2e = kl_div(predict_shares("mdp-e2e", p_post, income,
                                        mdp_e2e=mdp_e2e, xbar_e2e=xbar_shock_e2e,
                                        device=DEVICE), w_habit_shock)

    # ── BLP / QUAIDS / Series benchmarks on Habit DGP ────────────────
    rmse_blp_h    = get_metrics("blp",   p_post, income, w_habit_shock,
                                  blp=blp_hab)["RMSE"]
    rmse_quaids_h  = get_metrics("quaids", p_post, income, w_habit_shock,
                                  quaids=quaids_hab)["RMSE"]
    rmse_series_h  = get_metrics("series", p_post, income, w_habit_shock,
                                  series=series_hab)["RMSE"]
    kl_blp    = kl_div(predict_shares("blp",   p_post, income,
                                       blp=blp_hab), w_habit_shock)
    kl_quaids  = kl_div(predict_shares("quaids", p_post, income,
                                        quaids=quaids_hab), w_habit_shock)
    kl_series  = kl_div(predict_shares("series", p_post, income,
                                        series=series_hab), w_habit_shock)

    # ── Window IRL — habit advantage ──────────────────────────────────
    # Build shock-period window features for Window IRL (Habit DGP)
    log_p_shock_seq = np.log(np.maximum(p_post, 1e-8))
    log_y_shock_seq = np.log(np.maximum(income, 1e-8))
    log_q_sh_seq    = np.log(np.maximum(
        w_habit_shock * income[:, None] / np.maximum(p_post, 1e-8), 1e-6))
    wf_hab_sh = build_window_features(log_p_shock_seq, log_y_shock_seq,
                                       log_q_sh_seq, window=_WIRL_W, store_ids=None)
    with torch.no_grad():
        xt_sh = torch.tensor(wf_hab_sh, dtype=torch.float32, device=DEVICE)
        w_wirl_hab_sh = wirl_hab(xt_sh).cpu().numpy()
    rmse_wirl_h  = float(np.sqrt(np.mean((w_wirl_hab_sh - w_habit_shock)**2)))
    kl_wirl_h    = kl_div(w_wirl_hab_sh, w_habit_shock)

    # ── Demand curve arrays on fixed price grid ───────────────────────
    test_p       = np.tile(p_pre.mean(0),(80,1)); test_p[:,1] = P_GRID
    fixed_y      = np.full(80, AVG_Y)
    xbar_rep     = np.tile(xbar_train.mean(0),(80,1))
    q_prev_rep   = np.tile(q_prev_train.mean(0),(80,1))

    # Window IRL demand-curve predictions on CES DGP (fixed mean history)
    _wirl_ces_kw = dict(
        wirl=wirl_ces,
        wirl_log_p_hist=_wirl_lp_mean,
        wirl_log_q_hist=_wirl_lq_mean,
        wirl_window=_WIRL_W,
        device=DEVICE,
    )
    # CES comparison: food share only → (80,)
    curves = {
        "Truth":          primary.solve_demand(test_p, fixed_y)[:,0],
        "LA-AIDS":        predict_shares("aids",    test_p,fixed_y,**KW)[:,0],
        "BLP (IV)":       predict_shares("blp",     test_p,fixed_y,**KW)[:,0],
        "QUAIDS":         predict_shares("quaids",  test_p,fixed_y,**KW)[:,0],
        "Series Est.":    predict_shares("series",  test_p,fixed_y,**KW)[:,0],
        "Lin IRL Shared": predict_shares("l-irl",   test_p,fixed_y,
                                          lirl_theta=theta_shared,lirl_feat_fn=features_shared)[:,0],
        "Lin IRL Orth":   predict_shares("l-irl",   test_p,fixed_y,
                                          lirl_theta=theta_orth,lirl_feat_fn=features_orthogonalised)[:,0],
        "Neural IRL":     predict_shares("n-irl",   test_p,fixed_y,**KW)[:,0],
        "Window IRL":     predict_shares("window-irl", test_p,fixed_y,**_wirl_ces_kw)[:,0],
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

    # E2E xbar at mean training habit (for grid prediction)
    with torch.no_grad():
        lq_rep_t = torch.tensor(
            np.log(np.maximum(
                np.tile(q_train.mean(0), (80, 1)), 1e-6)),
            dtype=torch.float32, device=DEVICE)
        xbar_e2e_rep = compute_xbar_e2e(
            mdp_e2e.delta.to(DEVICE), lq_rep_t, store_ids=None
        ).cpu().numpy()

    # ── Habit DGP welfare (Priority 2): structural CV with fixed x̄_mean ──────
    # Integrate from mean pre-shock → mean post-shock prices while holding x̄
    # at the training mean throughout (ceteris paribus structural CV).
    # This is directly comparable across all models — habit dynamics held constant.
    _wh_p0      = avg_p / np.array([1.0, 1.2, 1.0])   # back-infer pre-shock mean
    _wh_p1      = avg_p                                  # post-shock mean
    _wh_y       = float(income.mean())
    _wh_xb      = xbar_train.mean(0)                    # (G,) fixed habit stock
    _wh_qp      = q_prev_train.mean(0)                  # (G,) fixed prev-period qty
    _wh_xb_e2e    = xbar_e2e_rep[0]                     # steady-state E2E x̄ (G,)
    _WH_S       = 60
    _wh_path    = np.linspace(_wh_p0, _wh_p1, _WH_S)   # (_WH_S, G)
    _wh_dp      = (_wh_p1 - _wh_p0) / _WH_S

    def _whl(spec_tag, **ekw):
        """Structural welfare integral for a single model."""
        _loss = 0.0
        for _tt in range(_WH_S):
            _pt = _wh_path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_wh_y]), **ekw)[0]
            _loss -= (_w * _wh_y / _wh_path[_tt]) @ _wh_dp
        return _loss

    # Ground truth: solve habit utility at each path step with fixed x̄_mean
    _wh_gt = 0.0
    for _tt in range(_WH_S):
        _p   = _wh_path[_tt]
        _flr = habit_consumer.theta * _wh_xb + 1e-6
        def _neg_u_wh(x, _p=_p):
            adj = x - habit_consumer.theta * _wh_xb
            return (1e10 if np.any(adj <= 0)
                    else -(np.sum(habit_consumer.alpha * adj**habit_consumer.rho)
                           ) ** (1 / habit_consumer.rho))
        _r_wh = minimize(
            _neg_u_wh, np.maximum(_wh_y / (3 * _p), _flr + 0.01),
            bounds=[(_flr[j], None) for j in range(3)],
            constraints=[{"type": "eq", "fun": lambda x, p=_p: p @ x - _wh_y}],
            method="SLSQP")
        _w_wh = _r_wh.x * _p / _wh_y if _r_wh.success else np.ones(3) / 3
        _wh_gt -= (_w_wh * _wh_y / _p) @ _wh_dp

    # Window IRL demand-curve predictions on Habit DGP (fixed mean history)
    _wirl_hab_kw = dict(
        wirl=wirl_hab,
        wirl_log_p_hist=_wirl_lp_mean,
        wirl_log_q_hist=_wirl_lq_h_mean,
        wirl_window=_WIRL_W,
        device=DEVICE,
    )

    welf_habit = {
        "Ground Truth":        _wh_gt,
        "LA-AIDS (static)":    _whl("aids",       aids=aids_hab),
        "BLP (IV) (static)":   _whl("blp",        blp=blp_hab),
        "QUAIDS (static)":     _whl("quaids",     quaids=quaids_hab),
        "Series Est. (static)":_whl("series",     series=series_hab),
        "Window IRL":          _whl("window-irl", **_wirl_hab_kw),
        "Neural IRL (static)": _whl("n-irl",      nirl=n_irl_hab, device=DEVICE),
        "MDP Neural IRL":      _whl("mdp-irl",    mdp_nirl=mdp_irl,
                                     xbar=_wh_xb.reshape(1, -1),
                                     q_prev=_wh_qp.reshape(1, -1), device=DEVICE),
        "MDP IRL (E2E δ)":     _whl("mdp-e2e",   mdp_e2e=mdp_e2e,
                                     xbar_e2e=_wh_xb_e2e.reshape(1, -1),
                                     device=DEVICE),
        "Neural IRL (CF)":     _whl("n-irl-cf",   nirl_cf=n_irl_cf, device=DEVICE),
        "MDP IRL (CF)":        _whl("mdp-irl-cf", mdp_nirl_cf=mdp_irl_cf,
                                     xbar=_wh_xb.reshape(1, -1),
                                     q_prev=_wh_qp.reshape(1, -1), device=DEVICE),
    }

    # ── Welfare across xbar distribution (Priority 3) ─────────────────────
    # CV at 10/25/50/75/90th percentiles of the empirical habit-stock dist.
    _xb_pcts  = np.percentile(xbar_train, [10, 25, 50, 75, 90], axis=0)  # (5, G)
    _qp_pcts  = np.percentile(q_prev_train, [10, 25, 50, 75, 90], axis=0)
    _pct_labels = [10, 25, 50, 75, 90]

    def _whl_pct(spec_tag, xb_pt, qp_pt, **ekw):
        """Welfare integral at a specific habit-stock point."""
        _loss = 0.0
        for _tt in range(_WH_S):
            _pt = _wh_path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_wh_y]),
                                 xbar=xb_pt.reshape(1, -1),
                                 q_prev=qp_pt.reshape(1, -1),
                                 **ekw)[0]
            _loss -= (_w * _wh_y / _wh_path[_tt]) @ _wh_dp
        return _loss

    welf_by_pct = {}   # {pct: {model: cv_value}}
    for _pi, _pct in enumerate(_pct_labels):
        _xb_pt = _xb_pcts[_pi]
        _qp_pt = _qp_pcts[_pi]
        # Compute E2E xbar at this percentile habit stock
        with torch.no_grad():
            _lq_pct = torch.tensor(
                np.log(np.maximum(np.tile(_xb_pt, (80, 1)), 1e-6)),
                dtype=torch.float32, device=DEVICE)
            _xb_e2e_pct = compute_xbar_e2e(
                mdp_e2e.delta.to(DEVICE), _lq_pct, store_ids=None
            ).cpu().numpy()[0]
        welf_by_pct[_pct] = {
            "Neural IRL (static)": _whl_pct("n-irl",      _xb_pt, _qp_pt,
                                             nirl=n_irl_hab, device=DEVICE),
            "MDP Neural IRL":      _whl_pct("mdp-irl",    _xb_pt, _qp_pt,
                                             mdp_nirl=mdp_irl, device=DEVICE),
            "MDP IRL (E2E δ)":     _whl("mdp-e2e",        mdp_e2e=mdp_e2e,
                                          xbar_e2e=_xb_e2e_pct.reshape(1, -1),
                                          device=DEVICE),
            "MDP IRL (CF)":        _whl_pct("mdp-irl-cf", _xb_pt, _qp_pt,
                                             mdp_nirl_cf=mdp_irl_cf, device=DEVICE),
        }

    # MDP comparison: all 3 goods → (80,3)
    mdp_curves = {
        "Truth":             true_conditional_shares,
        "LA-AIDS":           predict_shares("aids",       test_p,fixed_y,aids=aids_hab),
        "BLP (IV)":          predict_shares("blp",        test_p,fixed_y,blp=blp_hab),
        "QUAIDS":            predict_shares("quaids",     test_p,fixed_y,quaids=quaids_hab),
        "Series Est.":       predict_shares("series",     test_p,fixed_y,series=series_hab),
        "Window IRL":        predict_shares("window-irl", test_p,fixed_y,**_wirl_hab_kw),
        "Neural IRL static": predict_shares("n-irl",      test_p,fixed_y,nirl=n_irl_hab,device=DEVICE),
        "MDP-IRL":           predict_shares("mdp-irl",    test_p,fixed_y,
                                             mdp_nirl=mdp_irl, xbar=xbar_rep,
                                             q_prev=q_prev_rep, device=DEVICE),
        "MDP-IRL (E2E δ)":  predict_shares("mdp-e2e",   test_p, fixed_y,
                                             mdp_e2e=mdp_e2e, xbar_e2e=xbar_e2e_rep,
                                             device=DEVICE),
    }

    # ── δ identification results: directly from the training-time sweep ──────
    # The grid sweep above already evaluated hold-out KL at each δ; we reuse
    # those results here.  The "KL profile" is now the 7-point val-KL curve
    # (one per trained model) rather than a post-training analytical sweep.
    _kl_delta_grid    = MDP_DELTA_GRID.copy()            # (K,)
    _kl_e2e_arr       = _sweep_e2e["kl_grid"].copy()     # (K,) val-KL
    _se_e2e_arr       = _sweep_e2e["se_grid"].copy()     # (K,) SE

    _delta_hat_e2e    = _sweep_e2e["delta_hat"]          # float
    _id_set_e2e       = _sweep_e2e["id_set"]             # (lo, hi)

    # SE-based identified-set width (reported in place of the old c₀.₉₅)
    _best_k_e2e = int(np.argmin(_kl_e2e_arr))
    _c95_e2e    = float(2.0 * _se_e2e_arr[_best_k_e2e])  # 2 × SE threshold

    _delta_cs_lo_e2e = _id_set_e2e[0]
    _delta_cs_hi_e2e = _id_set_e2e[1]
    _true_delta_in_cs = bool(
        0.7 >= _delta_cs_lo_e2e and 0.7 <= _delta_cs_hi_e2e)

    return {
        "perf_ces":     perf_ces,
        "elast":        elast,
        "cross_elast":  cross_elast,
        "welf":         welf,
        "welf_habit":   welf_habit,    # structural CV on habit DGP (Priority 2)
        "welf_by_pct":  welf_by_pct,  # CV at xbar percentiles (Priority 3)
        "lin_ablation": lin_ablation,
        "rob_rows":     rob_rows,
        "mdp": {
            "aids_rmse":        rmse_aids_h,
            "blp_rmse":         rmse_blp_h,
            "quaids_rmse":      rmse_quaids_h,
            "series_rmse":      rmse_series_h,
            "wirl_rmse":        rmse_wirl_h,
            "nirl_rmse":        rmse_nirl_h,
            "mdp_rmse":         rmse_mdp_h,
            "mdp_e2e_rmse":     rmse_mdp_e2e_h,
            "kl_aids":          kl_aids,
            "kl_blp":           kl_blp,
            "kl_quaids":        kl_quaids,
            "kl_series":        kl_series,
            "kl_wirl":          kl_wirl_h,
            "kl_static":        kl_static,
            "kl_mdp":           kl_mdp,
            "kl_mdp_e2e":       kl_mdp_e2e,
        },
        "delta_mdp":         mdp_irl.delta.item(),
        "delta_mdp_e2e":     _sweep_e2e["delta_hat"],   # δ̂ from grid sweep
        "shock_pt":       p_pre[:,1].mean() * 1.2,
        "curves":         curves,        # {model: (80,)}
        "mdp_curves":     mdp_curves,    # {model: (80,3)}
        # KL profile over δ — validation KL at each grid point (K=7)
        "kl_delta_grid":   _kl_delta_grid,     # (K,) δ grid values
        "kl_profile_e2e":  _kl_e2e_arr,        # (K,) hold-out KL
        # SE-based identified set for δ  (replaces bootstrap CS)
        "delta_cs_e2e":   (_delta_cs_lo_e2e, _delta_cs_hi_e2e),
        "delta_hat_e2e":  float(_delta_hat_e2e),
        "c95_e2e":        _c95_e2e,    # 2 × SE at δ̂ (analogous to old bootstrap c₀.₉₅)
        "true_delta_in_cs": _true_delta_in_cs,   # True iff 0.7 ∈ CS_e2e
        # CF model scalars
        "cf_rsq":         _cf_rsq,
        # last-run only — for convergence and mixture plots
        "hist_nirl":       hist_nirl,
        "hist_nirl_hab":   hist_nirl_hab,
        "hist_mdp":        hist_mdp,
        "hist_mdp_e2e":    _sweep_e2e["best_hist"],
        "hist_wirl_ces":   hist_wirl_ces,
        "hist_wirl_hab":   hist_wirl_hab,
        "comp_summary":    var_mix.get_component_summary(),
    }


# ════════════════════════════════════════════════════════════════════
#  SECTION 8b: SWEEP HELPER (Priorities 1 & 3)
#  Trains all five habit models for arbitrary (delta, theta) values.
#  Uses smaller hidden dim (128) and fewer epochs to keep wall time low.
# ════════════════════════════════════════════════════════════════════

def run_habit_param_seed(seed: int, delta: float = 0.7, theta: float = 0.3,
                          epochs: int = EPOCHS) -> dict:
    """Train habit models for a given (delta, theta) pair.

    Used by the δ-identification sweep (Priority 1) and the θ-robustness
    sweep (Priority 3).  Returns recovered δ̂ values and post-shock RMSE
    for all five habit models.

    Parameters
    ----------
    seed   : RNG seed for reproducibility.
    delta  : True habit-decay parameter passed to HabitFormationConsumer.
    theta  : True habit-strength parameter passed to HabitFormationConsumer.
    epochs : Training epochs (default 10000).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    N     = N_OBS
    Z     = np.random.uniform(1, 5, (N, 3))
    p_pre = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    hc = HabitFormationConsumer(theta=theta, decay=delta)
    w_hab,  xbar_tr = hc.solve_demand(p_pre,  income, return_xbar=True)
    p_post  = p_pre.copy(); p_post[:, 1] *= 1.2
    w_shock, xbar_sh = hc.solve_demand(p_post, income, return_xbar=True)

    q_tr  = w_hab   * income[:, None] / np.maximum(p_pre,  1e-8)
    qp_tr = np.vstack([q_tr[0:1],  q_tr[:-1]])        # q_prev (train)
    q_sh  = w_shock * income[:, None] / np.maximum(p_post, 1e-8)
    qp_sh = np.vstack([q_sh[0:1],  q_sh[:-1]])        # q_prev (shock)

    log_q = np.log(np.maximum(q_tr,   1e-6))

    # ── LA-AIDS ────────────────────────────────────────────────────────────
    aids_h = AIDSBench(); aids_h.fit(p_pre, w_hab, income)

    # ── Static Neural IRL ─────────────────────────────────────────────────
    nirl_h = NeuralIRL(n_goods=3, hidden_dim=128)
    nirl_h, _ = train_neural_irl(
        nirl_h, p_pre, income, w_hab, epochs=epochs, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # ── MDP IRL (blend: fixed δ used only as blending weight, not recovered) ─
    mdp_b = MDPNeuralIRL(n_goods=3, hidden_dim=128)
    mdp_b, _ = train_neural_irl(
        mdp_b, p_pre, income, w_hab, epochs=epochs, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_tr, q_prev_data=qp_tr, device=DEVICE)

    # ── MDP IRL E2E — frozen-δ grid sweep ────────────────────────────────
    # Generate validation set for model selection.
    _vrng2 = np.random.default_rng(seed + 88888)
    _nv2   = max(len(p_pre) // 5, 80)
    _pv2   = np.clip(_vrng2.uniform(1, 5, (_nv2, 3)) +
                     _vrng2.normal(0, 0.1, (_nv2, 3)), 1e-3, None)
    _yv2   = _vrng2.uniform(1200, 2000, _nv2)
    _hcv2  = HabitFormationConsumer(theta=theta, decay=delta)
    _wv2, _ = _hcv2.solve_demand(_pv2, _yv2, return_xbar=True)
    _qv2   = _wv2 * _yv2[:, None] / np.maximum(_pv2, 1e-8)
    _lqv2  = np.log(np.maximum(_qv2, 1e-6))

    _sw_ee = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q,
        _pv2, _yv2, _wv2, _lqv2,
        delta_grid=MDP_DELTA_GRID, epochs=epochs, device=DEVICE,
        n_goods=3, hidden=128,
        lam_mono=0.2, lam_slut=0.05, batch=256, lr=5e-4,
        tag=f"Habit-E2E-d{delta:.1f}-t{theta:.1f}")
    mdp_ee = _sw_ee["best_model"]

    # ── Shock-period inputs for E2E ───────────────────────────────────────
    log_q_sh = np.log(np.maximum(q_sh,   1e-6))
    with torch.no_grad():
        _lq_sh_t = torch.tensor(log_q_sh, dtype=torch.float32, device=DEVICE)
        xbar_sh_ee    = compute_xbar_e2e(
            mdp_ee.delta.to(DEVICE), _lq_sh_t, store_ids=None).cpu().numpy()

    # ── Evaluation ────────────────────────────────────────────────────────
    def _rmse(spec, **kw):
        return get_metrics(spec, p_post, income, w_shock, **kw)["RMSE"]

    return {
        "true_delta":    delta,
        "true_theta":    theta,
        "delta_blend":   mdp_b.delta.item(),      # blend weight learned by MDP-IRL
        "delta_e2e":     _sw_ee["delta_hat"],      # δ̂ from grid sweep
        "rmse_aids":     _rmse("aids",        aids=aids_h),
        "rmse_nirl":     _rmse("n-irl",       nirl=nirl_h, device=DEVICE),
        "rmse_mdp":      _rmse("mdp-irl",     mdp_nirl=mdp_b,
                               xbar_shock=xbar_sh, q_prev_shock=qp_sh, device=DEVICE),
        "rmse_e2e":      _rmse("mdp-e2e",     mdp_e2e=mdp_ee,
                               xbar_e2e=xbar_sh_ee, device=DEVICE),
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
          f"  δ={r['delta_mdp']:.3f}")


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

# Habit welfare (Priority 2): structural CV on habit DGP
welf_hab_models = list(all_results[0]["welf_habit"].keys())
welf_hab_agg = {}
for nm in welf_hab_models:
    vals = [r["welf_habit"][nm] for r in all_results]
    welf_hab_agg[nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

# Welfare across xbar percentiles (Priority 3)
_pct_labels_agg = [10, 25, 50, 75, 90]
_pct_models     = list(all_results[0]["welf_by_pct"][10].keys())
welf_pct_agg = {}   # {pct: {model: {mean, se}}}
for _pct in _pct_labels_agg:
    welf_pct_agg[_pct] = {}
    for nm in _pct_models:
        vals = [r["welf_by_pct"][_pct][nm] for r in all_results]
        welf_pct_agg[_pct][nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

# Delta confidence sets (Priority 2): aggregate CS bounds across runs
delta_cs_e2e_lo = np.nanmean([r["delta_cs_e2e"][0] for r in all_results])
delta_cs_e2e_hi = np.nanmean([r["delta_cs_e2e"][1] for r in all_results])
delta_hat_e2e_mean = np.nanmean([r["delta_hat_e2e"] for r in all_results])
true_delta_in_cs_frac = np.mean([r["true_delta_in_cs"] for r in all_results])

# Cross-price elasticity matrices: mean ± SE across runs  (3×3 per model)
cross_elast_models = list(all_results[0]["cross_elast"].keys())
cross_elast_agg = {}
for _nm in cross_elast_models:
    _stack = np.stack([r["cross_elast"][_nm] for r in all_results], 0)  # (n_runs, 3, 3)
    cross_elast_agg[_nm] = {
        "mean": np.nanmean(_stack, 0),
        "se":   np.nanstd(_stack,  0, ddof=min(1, N_RUNS-1)) / np.sqrt(N_RUNS),
    }

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
mdp_keys = ["aids_rmse","blp_rmse","quaids_rmse","series_rmse","wirl_rmse",
            "nirl_rmse","mdp_rmse","mdp_e2e_rmse",
            "kl_aids","kl_blp","kl_quaids","kl_series","kl_wirl",
            "kl_static","kl_mdp","kl_mdp_e2e"]
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

# Delta
delta_mdp_mean     = np.mean([r["delta_mdp"]     for r in all_results])
delta_mdp_se       = _se([r["delta_mdp"]     for r in all_results])
delta_mdp_e2e_mean    = np.mean([r["delta_mdp_e2e"]    for r in all_results])
delta_mdp_e2e_se      = _se([r["delta_mdp_e2e"]    for r in all_results])
TRUE_DELTA = 0.7   # HabitFormationConsumer.decay

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
delta_e2e_arr    = np.array([r["delta_mdp_e2e"]    for r in all_results])
delta_mdp_arr = np.array([r["delta_mdp"]     for r in all_results])

# KL profile over δ: mean ± SE across runs (shape (80,) each)
kl_delta_grid   = last["kl_delta_grid"]   # same grid every run
kl_prof_e2e_arr = np.stack([r["kl_profile_e2e"] for r in all_results], 0)   # (n_runs, K)
kl_prof_e2e_mean = kl_prof_e2e_arr.mean(0)
kl_prof_e2e_se   = kl_prof_e2e_arr.std(0, ddof=1) / np.sqrt(N_RUNS)

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
for lbl,kr,kk in [("LA-AIDS",           "aids_rmse",       "kl_aids"),
                   ("BLP (IV)",          "blp_rmse",        "kl_blp"),
                   ("QUAIDS",            "quaids_rmse",     "kl_quaids"),
                   ("Series Est.",       "series_rmse",     "kl_series"),
                   ("Window IRL",        "wirl_rmse",       "kl_wirl"),
                   ("Neural IRL static", "nirl_rmse",       "kl_static"),
                   ("MDP Neural IRL",    "mdp_rmse",        "kl_mdp"),
                   ("MDP IRL (E2E δ)",   "mdp_e2e_rmse",    "kl_mdp_e2e")]:
    print(f"  {lbl:<25} RMSE={fmt(mdp_agg[kr]['mean'],mdp_agg[kr]['se'])}  "
          f"KL={fmt(mdp_agg[kk]['mean'],mdp_agg[kk]['se'])}")

# ── Table: Delta 95% Confidence Sets ──────────────────────────────────
print("\n  TABLE: 95% CONFIDENCE SETS FOR δ  (bootstrap-calibrated profile-KL)")
print(f"  True δ = 0.7  (HabitFormationConsumer.decay)")
print(f"  {'Model':<25} {'δ̂ (mean)':<12} {'CS [lo, hi]':<22} {'True δ in CS?'}")
print(f"  {'MDP IRL (E2E δ)':<25} {delta_hat_e2e_mean:.3f}        "
      f"[{delta_cs_e2e_lo:.3f}, {delta_cs_e2e_hi:.3f}]       "
      f"{'YES' if true_delta_in_cs_frac >= 0.5 else 'NO'} "
      f"({100*true_delta_in_cs_frac:.0f}% of runs)")

# ── Table: Welfare by Habit-Stock Percentile ──────────────────────────
print("\n  TABLE: CV WELFARE BY HABIT-STOCK PERCENTILE  (MDP advantage at each pct)")
_pct_hdr = "  ".join(f"p{p:<6}" for p in _pct_labels_agg)
print(f"  {'Model':<26} {_pct_hdr}")
for _nm in _pct_models:
    _row = "  ".join(
        f"{welf_pct_agg[p][_nm]['mean']:.4f}" for p in _pct_labels_agg)
    print(f"  {_nm:<26} {_row}")
print(f"  δ̂ MDP IRL (fixed): {fmt(delta_mdp_mean, delta_mdp_se, 4)}  (blend only)")
print(f"  δ̂ MDP E2E (recov):    {fmt(delta_mdp_e2e_mean,    delta_mdp_e2e_se,    4)}"
      f"  (true δ = {TRUE_DELTA})")


# ════════════════════════════════════════════════════════════════════
#  SECTION 12: FIGURES
# ════════════════════════════════════════════════════════════════════

BAND = 0.15   # alpha for ±1 SE shaded bands

# ── Style constants ───────────────────────────────────────────────
STYLE = {
    "Truth":             dict(color="k",         ls="-",  lw=3.0),
    "LA-AIDS":           dict(color="#E53935",    ls="--", lw=2.0),
    "BLP (IV)":          dict(color="#9C27B0",    ls="--", lw=2.0),
    "QUAIDS":            dict(color="#43A047",    ls="-.", lw=2.0),
    "Series Est.":       dict(color="#FB8C00",    ls=":",  lw=2.0),
    "Window IRL":        dict(color="#6D4C41",    ls="--", lw=2.0),
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
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("figures/fig1_demand_curves.pdf", dpi=150, bbox_inches="tight")
fig1.savefig("figures/fig1_demand_curves.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig1_demand_curves.pdf")

# ── Figure 2: MDP advantage — all 3 goods ────────────────────────
MDP_STYLE = {
    "Truth":             ("k",       "-",    3.0, "Truth (Habit)"),
    "LA-AIDS":           ("#E53935", "--",   2.0, "LA-AIDS"),
    "BLP (IV)":          ("#9C27B0", "--",   2.0, "BLP (IV)"),
    "QUAIDS":            ("#43A047", "-.",   2.0, "QUAIDS"),
    "Series Est.":       ("#FB8C00", ":",    2.0, "Series Estimator"),
    "Window IRL":        ("#6D4C41", "--",   2.0, "Window IRL"),
    "Neural IRL static": ("#1E88E5", "-.",   2.0, "Neural IRL (static)"),
    "MDP-IRL":           ("#00897B", "-",    2.5, r"MDP-IRL (with $\bar{x}$)"),
    "MDP-IRL (E2E δ)":  ("#FF6F00", "--",   2.0, r"MDP-IRL (E2E $\hat{\delta}$)"),
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
    # ax.set_title(f"{gn} Share (Habit DGP)", fontsize=14, fontweight="bold")
    
    # Optimized legend for standalone plot
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    
    # Save with specific names
    file_base = f"figures/fig2_{gn.lower()}_advantage"
    fig.savefig(f"{file_base}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{file_base}.png", dpi=150, bbox_inches="tight")
    
    print(f"    Saved: {file_base}.pdf")

# Optional: close plots to free up memory
plt.close('all')

# ── Figure 3: Convergence — all trained models (last/representative run) ──
# 5 panels in a 2×3 grid (bottom-right cell empty).
# For each non-E2E model: KL loss on left axis, β on right axis.
# For MDP E2E: KL loss on left axis, δ on right axis (with true δ=0.7 marker);
# β is shown as a secondary line on the δ axis is too cluttered, so we place δ
# on the right and annotate β's final value as text.
#
# Layout:
#   Row 0: Neural IRL (CES)  |  Neural IRL static (Habit)  |  MDP IRL (fixed δ)
#   Row 1: MDP E2E (learns δ)|  Window IRL                  |  [empty]

_conv_specs = [
    # (title,           hist_key,         col_kl,    show_delta)
    ("Neural IRL\n(CES DGP)",
     "hist_nirl",          "#1E88E5",  False),
    ("Neural IRL static\n(Habit DGP)",
     "hist_nirl_hab",      "#43A047",  False),
    (r"MDP Neural IRL" "\n" r"(fixed $\bar{x}$, δ=0.7)",
     "hist_mdp",           "#00897B",  False),
    (r"MDP IRL E2E" "\n" r"(learns $\hat{\delta}$)",
     "hist_mdp_e2e",       "#FF6F00",  True),
    ("Window IRL\n(Habit DGP)",
     "hist_wirl_hab",      "#6D4C41",  False),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(17, 9))
axes3_flat = axes3.flatten()

for idx, (title, hist_key, col_kl, show_delta) in enumerate(_conv_specs):
    ax = axes3_flat[idx]
    hist = last.get(hist_key, [])
    if not hist:
        ax.axis("off")
        continue

    ep_x = [h["epoch"] for h in hist]
    kl_y = [h["kl"]    for h in hist]

    ax2 = ax.twinx()
    col_r = "#E53935"

    line_kl, = ax.plot(ep_x, kl_y, "o-", ms=4, lw=1.8,
                       color=col_kl, label="KL Loss")
    lines = [line_kl]
    labels = ["KL Loss"]

    if show_delta:
        # Right axis shows learned δ converging toward true δ = 0.7
        dt_y = [h["delta"] for h in hist]
        line_d, = ax2.plot(ep_x, dt_y, "^-", ms=4, lw=1.8,
                           color=col_r, label=r"$\hat{\delta}$ (learned)")
        ax2.axhline(TRUE_DELTA, color=col_r, ls="--", lw=1.4, alpha=0.6,
                    label=f"True δ = {TRUE_DELTA}")
        ax2.set_ylabel(r"$\hat{\delta}$", color=col_r, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=col_r)
        ax2.set_ylim(0.4, 1.0)
        lines += [line_d]
        labels += [r"$\hat{\delta}$ (learned)", f"True δ = {TRUE_DELTA}"]
    else:
        ax2.axis("off")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("KL Divergence", color=col_kl, fontsize=12)
    ax.tick_params(axis="y", labelcolor=col_kl)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(lines, labels, loc="upper right", fontsize=9)

fig3.suptitle(
    "Training Convergence — All Models (representative last run)",
    fontsize=14, fontweight="bold", y=1.01)
fig3.tight_layout()
fig3.savefig("figures/fig3_convergence.pdf",  dpi=150, bbox_inches="tight")
fig3.savefig("figures/fig3_convergence.png",  dpi=150, bbox_inches="tight")
print("    Saved: figures/fig3_convergence.pdf")
plt.close(fig3)

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
# ── Setup Data ──────────────────────────────────────────────────
comp_df = last["comp_summary"]
x_pos   = np.arange(len(comp_df))
tab10   = plt.cm.tab10(x_pos / len(comp_df))

# Standardize figsize for both: width=12 provides ample space for X-labels
standard_figsize = (12, 6)

# ── Figure 5a: Mixture Weights (Bar Chart) ──────────────────────
fig5a, ax5a = plt.subplots(figsize=standard_figsize)

bars = ax5a.bar(x_pos, comp_df["pi"], color=tab10, alpha=0.85, edgecolor="k")
for bar, row in zip(bars, comp_df.itertuples()):
    if row.pi > 0.02:
        ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"ρ={row.rho:.2f}", ha="center", va="bottom", fontsize=12)

ax5a.set_xticks(x_pos)
ax5a.set_xticklabels(
    [f"K={k+1}\nα=[{r.alpha_food:.2f},{r.alpha_fuel:.2f},{r.alpha_other:.2f}]"
     for k, r in enumerate(comp_df.itertuples())], fontsize=14, rotation=15)

ax5a.set_ylabel("Mixture weight $\\hat{\\pi}_k$", fontsize=14)
ax5a.set_ylim(0, 1)
ax5a.axhline(1/6, color="gray", ls="--", alpha=0.5, label="Uniform prior")
ax5a.legend(fontsize=14)
ax5a.grid(True, alpha=0.3, axis="y")

fig5a.tight_layout()
fig5a.savefig("figures/fig5a_mixture_weights.pdf", dpi=150, bbox_inches="tight")
fig5a.savefig("figures/fig5a_mixture_weights.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig5a_mixture_weights.pdf")


# ── Figure 5b: Component Centres (Scatter Plot) ────────────────
fig5b, ax5b = plt.subplots(figsize=standard_figsize)

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

# Legend moved inside the grid in the top right corner
ax5b.legend(fontsize=14, loc='upper right') 
ax5b.grid(True, alpha=0.3)

fig5b.tight_layout()
fig5b.savefig("figures/fig5b_mixture_centers.pdf", dpi=150, bbox_inches="tight")
fig5b.savefig("figures/fig5b_mixture_centers.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig5b_mixture_centers.pdf")

plt.close("all")


# ── Figure 6: Delta recovery — violin plots across runs ──────────────────────
# Two models: MDP blend (fixed δ), E2E (learned δ)
if N_RUNS > 1:
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    data_violin = [delta_mdp_arr, delta_e2e_arr]
    parts = ax6.violinplot(data_violin, positions=[1, 2],
                           showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    parts["bodies"][0].set_facecolor("#00897B")
    parts["bodies"][1].set_facecolor("#FF6F00")
    ax6.axhline(TRUE_DELTA, color="k", ls="--", lw=2,
                label=f"True δ = {TRUE_DELTA}")
    ax6.set_xticks([1, 2])
    ax6.set_xticklabels([
        f"MDP-IRL\n(blend only)\nmean={delta_mdp_arr.mean():.3f}",
        f"MDP-IRL E2E\n(recovered)\nmean={delta_e2e_arr.mean():.3f}",
    ], fontsize=11)
    ax6.set_ylabel("Learned habit-decay δ̂", fontsize=13)
    ax6.legend(fontsize=12)
    ax6.grid(True, axis="y", alpha=0.35)
    fig6.tight_layout()
    fig6.savefig("figures/fig6_delta_recovery.pdf", dpi=150, bbox_inches="tight")
    fig6.savefig("figures/fig6_delta_recovery.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig6_delta_recovery.pdf")
    plt.close(fig6)
else:
    # Single-run: simple bar with annotated value
    fig6, ax6 = plt.subplots(figsize=(7, 4))
    vals  = [delta_mdp_mean, delta_mdp_e2e_mean]
    labs  = ["MDP-IRL\n(blend)", "MDP-IRL E2E\n(recovered)"]
    cols  = ["#00897B", "#FF6F00"]
    ax6.bar(labs, vals, color=cols, alpha=0.8, edgecolor="k")
    ax6.axhline(TRUE_DELTA, color="k", ls="--", lw=2, label=f"True δ = {TRUE_DELTA}")
    for i, v in enumerate(vals):
        ax6.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=12)
    ax6.set_ylabel("Learned habit-decay δ̂", fontsize=13)
    ax6.set_ylim(0, 1)
    ax6.legend(fontsize=12)
    ax6.grid(True, axis="y", alpha=0.35)
    fig6.tight_layout()
    fig6.savefig("figures/fig6_delta_recovery.pdf", dpi=150, bbox_inches="tight")
    fig6.savefig("figures/fig6_delta_recovery.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig6_delta_recovery.pdf")
    plt.close(fig6)


# ── Figure 7: Cross-price demand elasticity heatmaps ─────────────────────────
# eps[i, j] = d log x_j / d log p_i  (demand elasticities, not share-elasticities)
# Diagonal (<0): own-price.  CES off-diagonal (>0): substitutes (σ≈1.82 > 1).
# LA-AIDS and QUAIDS impose functional-form assumptions; Neural IRL learns
# cross-price effects directly from data without those constraints.
_good_labels = ["Food\n($w_0$)", "Fuel\n($w_1$)", "Other\n($w_2$)"]
_n_hm = len(cross_elast_models)
fig7, axes7 = plt.subplots(1, _n_hm, figsize=(4.5 * _n_hm, 4.8))
if _n_hm == 1:
    axes7 = [axes7]
_vabs = max(
    max(np.nanmax(np.abs(cross_elast_agg[nm]["mean"])) for nm in cross_elast_models),
    0.1)
for ax, nm in zip(axes7, cross_elast_models):
    mat = cross_elast_agg[nm]["mean"]
    se  = cross_elast_agg[nm]["se"]
    im  = ax.imshow(mat, cmap="RdBu_r", vmin=-_vabs, vmax=_vabs, aspect="auto")
    for i in range(3):
        for j in range(3):
            v, s = mat[i, j], se[i, j]
            txt  = f"{v:+.2f}\n({s:.2f})"
            ax.text(j, i, txt,
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.55 * _vabs else "black",
                    fontweight="bold")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(_good_labels, fontsize=9)
    ax.set_yticklabels(_good_labels, fontsize=9)
    ax.set_xlabel("Response quantity $x_j$", fontsize=9)
    ax.set_ylabel("Shock price $p_i$", fontsize=9)
    ax.set_title(nm, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$\varepsilon_{ij} = \partial\log x_j/\partial\log p_i$")
fig7.suptitle(
    "Cross-Price Demand Elasticity Heatmaps — CES Ground Truth\n"
    r"Diagonal = own-price ($<0$)  ·  Off-diagonal = cross-price  "
    r"(Blue $<0$: complements,  Red $>0$: substitutes)  ·  cells: mean (SE)",
    fontsize=11, fontweight="bold")
fig7.tight_layout()
fig7.savefig("figures/fig7_cross_elasticity.pdf", dpi=150, bbox_inches="tight")
fig7.savefig("figures/fig7_cross_elasticity.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig7_cross_elasticity.pdf")
plt.close(fig7)


# ── Figure 8: Habit DGP welfare — Priority 2 ─────────────────────────────────
# Bar chart: absolute error of each model's structural CV vs ground-truth CV.
# Structural means x̄ is held at training mean for all models (apples-to-apples).
# MDP models should show smaller welfare error than static baselines because
# they use the correct habit state; E2E error shows effect of δ misestimation.
_wh_gt_mean = welf_hab_agg["Ground Truth"]["mean"]
_wh_plot_nms = [nm for nm in welf_hab_models if nm != "Ground Truth"]
_wh_colors   = {"LA-AIDS (static)":    "#E53935",
                 "Neural IRL (static)": "#1E88E5",
                 "MDP Neural IRL":      "#00897B",
                 "MDP IRL (E2E δ)":    "#FF6F00"}
_wh_errs  = [abs(welf_hab_agg[nm]["mean"] - _wh_gt_mean) for nm in _wh_plot_nms]
_wh_se_b  = [welf_hab_agg[nm]["se"] for nm in _wh_plot_nms]
_wh_pct   = [100 * e / max(abs(_wh_gt_mean), 1e-9) for e in _wh_errs]
_x8 = np.arange(len(_wh_plot_nms))

fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: absolute CV error
bars8 = ax8a.bar(_x8, _wh_errs,
                  color=[_wh_colors.get(nm, "gray") for nm in _wh_plot_nms],
                  alpha=0.85, edgecolor="k", capsize=5,
                  yerr=_wh_se_b, error_kw=dict(lw=1.5, ecolor="k"))
for bar, val in zip(bars8, _wh_errs):
    ax8a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(_wh_errs)*0.01,
              f"{val:.2f}", ha="center", va="bottom", fontsize=11)
ax8a.set_xticks(_x8)
ax8a.set_xticklabels(_wh_plot_nms, rotation=15, ha="right", fontsize=12)
ax8a.set_ylabel("Absolute CV error  (£)", fontsize=12)
ax8a.set_title("Welfare Error — Habit DGP (absolute)", fontsize=12, fontweight="bold")
ax8a.grid(True, axis="y", alpha=0.3)

# Panel B: % error relative to Ground Truth CV magnitude
bars8b = ax8b.bar(_x8, _wh_pct,
                   color=[_wh_colors.get(nm, "gray") for nm in _wh_plot_nms],
                   alpha=0.85, edgecolor="k")
for bar, val in zip(bars8b, _wh_pct):
    ax8b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(_wh_pct)*0.01,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
ax8b.set_xticks(_x8)
ax8b.set_xticklabels(_wh_plot_nms, rotation=15, ha="right", fontsize=12)
ax8b.set_ylabel("CV error / |GT CV|  (%)", fontsize=12)
ax8b.set_title("Welfare Error — Habit DGP (% of truth)", fontsize=12, fontweight="bold")
ax8b.grid(True, axis="y", alpha=0.3)

fig8.suptitle(
    f"Structural Compensating Variation — Habit DGP  (mean ± SE, {N_RUNS} runs)\n"
    r"x̄ held at training mean; path: $p_{\text{pre}} \to p_{\text{post}}$, "
    r"GT CV = £" + f"{_wh_gt_mean:.2f}",
    fontsize=11, fontweight="bold")
fig8.tight_layout()
fig8.savefig("figures/fig8_habit_welfare.pdf", dpi=150, bbox_inches="tight")
fig8.savefig("figures/fig8_habit_welfare.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig8_habit_welfare.pdf")
plt.close(fig8)


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
    ("LA-AIDS (static)",                             "aids_rmse",       "kl_aids",       "baseline"),
    ("BLP (IV) (static)",                            "blp_rmse",        "kl_blp",        None),
    ("QUAIDS (static)",                              "quaids_rmse",     "kl_quaids",     None),
    ("Series Estimator (static)",                    "series_rmse",     "kl_series",     None),
    ("Window IRL",                                   "wirl_rmse",       "kl_wirl",       None),
    ("Neural IRL (static MDP)",                      "nirl_rmse",       "kl_static",     None),
    (r"MDP Neural IRL ($\bar{x}$ state)",            "mdp_rmse",        "kl_mdp",        None),
    (r"MDP IRL (E2E $\hat{\delta}$)",                "mdp_e2e_rmse",    "kl_mdp_e2e",   None),
]
for lbl, kr, kk, red in mdp_table_rows:
    rd = mdp_agg[kr]; kd = mdp_agg[kk]
    if red is None:
        pct = 100*(rmse_base - rd["mean"]) / rmse_base
        pct_se = 100*np.sqrt((rd["se"]/rmse_base)**2
                             + (rd["mean"]*se_base/rmse_base**2)**2)
        red = f"{pct:.1f}\\% ({pct_se:.1f}\\%)"
    lx(f"    {lbl} & {cell(rd['mean'],rd['se'])} & {cell(kd['mean'],kd['se'])} & {red} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(f"    \\item Mean (SE) across {N_RUNS} runs. All models trained on identical habit-formation data.")
lx(r"    RMSE reduction relative to LA-AIDS mean; SE propagated via delta method.")
lx(rf"    $\hat{{\delta}}$: MDP (blend) = {delta_mdp_mean:.3f} ({delta_mdp_se:.3f}); "
   rf"MDP E2E = {delta_mdp_e2e_mean:.3f} ({delta_mdp_e2e_se:.3f}) "
   rf"(true $\delta={TRUE_DELTA}$). $\theta=0.3$.")
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
lx(f"    \\item Mean (SE) across {N_RUNS} runs. 10{{,}}000 gradient-ascent epochs, "
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
    ("fig6_delta_recovery",
     fr"Habit-Decay Recovery: Learned $\hat{{\delta}}$ Across {N_RUNS} Runs (Habit DGP)",
     fr"True decay $\delta={TRUE_DELTA}$ (dashed line). "
     r"MDP-IRL (teal, left): blends a pre-built $\bar{x}$ with $q_{{t-1}}$ but "
     r"does not feed $\hat{{\delta}}$ back into the habit stock update. "
     r"MDP-IRL E2E (orange, right): recomputes $\bar{{x}}$ from scratch at every "
     r"training epoch using $\hat{{\delta}}$, so the recovered decay rate "
     r"converges toward the true $\delta$ as training progresses.",
     "fig:delta_recovery"),
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
    Welfare error vs truth: {100*abs(welf_agg['Neural IRL']['mean']-gt_welf)/abs(gt_welf):.1f}%

  MDP ADVANTAGE (Habit DGP):
    LA-AIDS RMSE       : {fmt(mdp_agg['aids_rmse']['mean'],    mdp_agg['aids_rmse']['se'])}
    QUAIDS RMSE        : {fmt(mdp_agg['quaids_rmse']['mean'],  mdp_agg['quaids_rmse']['se'])}
    Series Est. RMSE   : {fmt(mdp_agg['series_rmse']['mean'],  mdp_agg['series_rmse']['se'])}
    Window IRL RMSE    : {fmt(mdp_agg['wirl_rmse']['mean'],    mdp_agg['wirl_rmse']['se'])}
    Static IRL RMSE    : {fmt(mdp_agg['nirl_rmse']['mean'],    mdp_agg['nirl_rmse']['se'])}
    MDP IRL RMSE       : {fmt(mdp_agg['mdp_rmse']['mean'],     mdp_agg['mdp_rmse']['se'])}
    MDP IRL E2E RMSE   : {fmt(mdp_agg['mdp_e2e_rmse']['mean'],    mdp_agg['mdp_e2e_rmse']['se'])}
    Reduction (MDP vs AIDS)         : {
        100*(mdp_agg['aids_rmse']['mean']-mdp_agg['mdp_rmse']['mean'])
        /mdp_agg['aids_rmse']['mean']:.1f}%
    Reduction (E2E vs AIDS)         : {
        100*(mdp_agg['aids_rmse']['mean']-mdp_agg['mdp_e2e_rmse']['mean'])
        /mdp_agg['aids_rmse']['mean']:.1f}%

  DELTA RECOVERY (true δ = {TRUE_DELTA}):
    MDP IRL (blend only): δ̂ = {delta_mdp_mean:.4f} ± {delta_mdp_se:.4f}
    MDP IRL E2E:          δ̂ = {delta_mdp_e2e_mean:.4f} ± {delta_mdp_e2e_se:.4f}

  OUTPUT FILES:
    figures/fig{{1..10}}_*.{{pdf,png}}
    paper_tables_figures.tex
""")


# ════════════════════════════════════════════════════════════════════
#  SECTION 15: IDENTIFICATION & ROBUSTNESS CHECKS
#  Priority 1 — δ identification across true δ ∈ {0.3, 0.5, 0.7, 0.9}
#  Priority 3 — θ robustness: MDP advantage scales with habit strength
# ════════════════════════════════════════════════════════════════════

# Sweep configuration — raise N_SWEEP_SEEDS for publication-quality SEs.
DELTA_GRID    = [0.1, 0.3, 0.5, 0.7, 0.9]
THETA_GRID    = [0.0, 0.1, 0.2, 0.3, 0.5]
N_SWEEP_SEEDS = min(N_RUNS, 3)   # seeds per grid point (capped for speed)
SWEEP_EPOCHS  = EPOCHS             # training epochs for sweep (vs 4000 main)

# ── Priority 1: δ identification sweep ───────────────────────────────────────
print("\n" + "=" * 72)
print(f"  SECTION 15 — IDENTIFICATION & ROBUSTNESS CHECKS")
print("=" * 72)
print(f"\n  [P1] δ identification: {len(DELTA_GRID)} values × "
      f"{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs ...")

_delta_rows = []
for _td in DELTA_GRID:
    for _si in range(N_SWEEP_SEEDS):
        _seed_d = 500 + _si * 13
        t0d = time.time()
        print(f"    true_δ={_td:.1f}  seed={_seed_d}", end="", flush=True)
        _r = run_habit_param_seed(_seed_d, delta=_td, theta=0.3,
                                   epochs=SWEEP_EPOCHS)
        print(f"  →  δ̂_blend={_r['delta_blend']:.3f}"
              f"  δ̂_e2e={_r['delta_e2e']:.3f}"
              f"  ({time.time()-t0d:.0f}s)")
        _delta_rows.append(_r)

# ── Priority 3: θ robustness sweep ───────────────────────────────────────────
print(f"\n  [P3] θ robustness: {len(THETA_GRID)} values × "
      f"{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs ...")

_theta_rows = []
for _tt in THETA_GRID:
    for _si in range(N_SWEEP_SEEDS):
        _seed_t = 600 + _si * 13
        t0t = time.time()
        print(f"    true_θ={_tt:.1f}  seed={_seed_t}", end="", flush=True)
        _r = run_habit_param_seed(_seed_t, delta=0.7, theta=_tt,
                                   epochs=SWEEP_EPOCHS)
        print(f"  →  rmse_mdp={_r['rmse_mdp']:.5f}"
              f"  rmse_nirl={_r['rmse_nirl']:.5f}"
              f"  ({time.time()-t0t:.0f}s)")
        _theta_rows.append(_r)


# ── Figure 9: δ identification ────────────────────────────────────────────────
# Each point = one seed; lines connect the within-grid-point means.
# Perfect recovery lies on the 45° line.
# Systematic bias toward a fixed value regardless of truth = identification failure.
# Approximate unbiasedness + occasional outliers at one value = benign.
fig9, ax9 = plt.subplots(figsize=(8, 6))

_col_blend = "#00897B"
_col_e2e   = "#FF6F00"

for _td in DELTA_GRID:
    _rows_d = [r for r in _delta_rows if r["true_delta"] == _td]
    _db  = [r["delta_blend"] for r in _rows_d]
    _de  = [r["delta_e2e"]   for r in _rows_d]
    # Scatter individual seeds
    ax9.scatter([_td] * len(_db), _db, marker="o", color=_col_blend, s=60, alpha=0.55, zorder=4)
    ax9.scatter([_td] * len(_de), _de, marker="^", color=_col_e2e,   s=60, alpha=0.55, zorder=4)

# Mean line per grid point
_blend_means = [np.mean([r["delta_blend"] for r in _delta_rows if r["true_delta"] == _td])
                for _td in DELTA_GRID]
_e2e_means   = [np.mean([r["delta_e2e"]   for r in _delta_rows if r["true_delta"] == _td])
                for _td in DELTA_GRID]
_blend_se    = [np.std([r["delta_blend"]  for r in _delta_rows if r["true_delta"] == _td],
                        ddof=1) / np.sqrt(N_SWEEP_SEEDS) for _td in DELTA_GRID]
_e2e_se      = [np.std([r["delta_e2e"]   for r in _delta_rows if r["true_delta"] == _td],
                        ddof=1) / np.sqrt(N_SWEEP_SEEDS) for _td in DELTA_GRID]

ax9.errorbar(DELTA_GRID, _blend_means, yerr=_blend_se,
             fmt="o-", color=_col_blend, lw=2.2, ms=9, capsize=5,
             label="MDP-IRL blend  (mean ± SE)")
ax9.errorbar(DELTA_GRID, _e2e_means,   yerr=_e2e_se,
             fmt="^-", color=_col_e2e,   lw=2.2, ms=9, capsize=5,
             label=r"MDP-IRL E2E  (mean ± SE)")
ax9.plot([0.2, 1.0], [0.2, 1.0], "k--", lw=1.8, label="Perfect recovery")

ax9.set_xlabel(r"True $\delta$", fontsize=14)
ax9.set_ylabel(r"Recovered $\hat{\delta}$", fontsize=14)
ax9.set_xlim(0.2, 1.0); ax9.set_ylim(0.2, 1.0)
ax9.legend(fontsize=12, loc="upper left")
ax9.grid(True, alpha=0.3)
fig9.suptitle(
    r"$\delta$ Identification Across True $\delta \in \{0.3,0.5,0.7,0.9\}$"
    f"\n{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs  ·  θ fixed at 0.3",
    fontsize=12, fontweight="bold")
fig9.tight_layout()
fig9.savefig("figures/fig9_delta_identification.pdf", dpi=150, bbox_inches="tight")
fig9.savefig("figures/fig9_delta_identification.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig9_delta_identification.pdf")
plt.close(fig9)


# ── Figure 10: θ robustness (MDP advantage scales with habit strength) ────────
# Panel A: absolute RMSE for each model vs θ (shows ordering)
# Panel B: RMSE reduction (MDP − static baseline) vs θ (shows monotone scaling)
_theta_models = [
    ("LA-AIDS",                 "rmse_aids",   "#E53935", "--"),
    ("Neural IRL",              "rmse_nirl",   "#1E88E5", "-."),
    ("MDP Neural IRL",          "rmse_mdp",    "#00897B", "-"),
    ("MDP IRL (E2E δ)",         "rmse_e2e",    "#FF6F00", "-"),
]

# Aggregate across seeds for each θ
_th_agg = {}
for _tt in THETA_GRID:
    _rws = [r for r in _theta_rows if r["true_theta"] == _tt]
    _th_agg[_tt] = {
        key: {"mean": np.mean([r[key] for r in _rws]),
              "se":   np.std( [r[key] for r in _rws], ddof=1) / np.sqrt(len(_rws))}
        for key in ["rmse_aids", "rmse_nirl", "rmse_mdp", "rmse_e2e"]
    }

fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(14, 6))

for lbl, key, col, ls in _theta_models:
    _mu  = [_th_agg[_tt][key]["mean"] for _tt in THETA_GRID]
    _se_vals  = [_th_agg[_tt][key]["se"]   for _tt in THETA_GRID]
    ax10a.errorbar(THETA_GRID, _mu, yerr=_se_vals, fmt=f"{ls}", color=col,
                   lw=2.0, ms=7, capsize=4, marker="o", label=lbl)

ax10a.set_xlabel("Habit strength θ", fontsize=13)
ax10a.set_ylabel("Post-shock RMSE", fontsize=13)
ax10a.set_title("Absolute RMSE vs θ", fontsize=12, fontweight="bold")
ax10a.legend(fontsize=11)
ax10a.grid(True, alpha=0.3)

# Panel B: RMSE reduction of MDP models vs static Neural IRL baseline
_base_mu  = [_th_agg[_tt]["rmse_nirl"]["mean"] for _tt in THETA_GRID]
for lbl, key, col, ls in _theta_models[2:]:   # MDP models only
    _gain = [_base_mu[i] - _th_agg[_tt][key]["mean"]
             for i, _tt in enumerate(THETA_GRID)]
    _gain_se = [np.sqrt(_th_agg[_tt]["rmse_nirl"]["se"]**2 +
                         _th_agg[_tt][key]["se"]**2)
                for _tt in THETA_GRID]
    ax10b.errorbar(THETA_GRID, _gain, yerr=_gain_se,
                   fmt=f"{ls}", color=col, lw=2.0, ms=7, capsize=4, marker="o",
                   label=lbl)

ax10b.axhline(0, color="k", lw=1.2, ls="--")
ax10b.set_xlabel("Habit strength θ", fontsize=13)
ax10b.set_ylabel("RMSE reduction vs Neural IRL (static)", fontsize=13)
ax10b.set_title("MDP Advantage vs θ  (positive = MDP wins)", fontsize=12,
                fontweight="bold")
ax10b.legend(fontsize=11)
ax10b.grid(True, alpha=0.3)

fig10.suptitle(
    f"Habit-Strength Robustness: θ ∈ {THETA_GRID}  ·  δ fixed at 0.7\n"
    f"{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs per grid point",
    fontsize=12, fontweight="bold")
fig10.tight_layout()
fig10.savefig("figures/fig10_theta_robustness.pdf", dpi=150, bbox_inches="tight")
fig10.savefig("figures/fig10_theta_robustness.png", dpi=150, bbox_inches="tight")
print("    Saved: figures/fig10_theta_robustness.pdf")
plt.close(fig10)


# ── Figure 11: KL profile over δ ─────────────────────────────────────────────
# Neural-net weights frozen at convergence (mean over all runs).
# x-axis: δ swept from 0.2 → 0.99.  y-axis: KL(w_pred(δ), w_habit_train).
# Interpretation:
#   Sharp minimum at δ̂  → δ is well-identified from the data.
#   Flat bowl / monotone → δ is weakly identified; network compensates.
# Vertical lines mark the mean recovered δ̂ and the true δ = 0.7.
fig11, ax11 = plt.subplots(figsize=(9, 5))

_col_e2e = "#FF6F00"

# Mean ± 1 SE bands
ax11.plot(kl_delta_grid, kl_prof_e2e_mean, color=_col_e2e, lw=2.5,
          label=r"MDP-IRL E2E (learns $\hat{\delta}$)")
if N_RUNS > 1:
    ax11.fill_between(kl_delta_grid,
                      kl_prof_e2e_mean - kl_prof_e2e_se,
                      kl_prof_e2e_mean + kl_prof_e2e_se,
                      color=_col_e2e, alpha=0.18)

# Mark recovered δ̂ (mean across runs)
ax11.axvline(delta_mdp_e2e_mean, color=_col_e2e, ls=":", lw=1.8,
             label=rf"$\hat{{\delta}}_{{E2E}}$ = {delta_mdp_e2e_mean:.3f}")
# Mark true δ
ax11.axvline(TRUE_DELTA, color="k", ls="--", lw=2.0,
             label=f"True δ = {TRUE_DELTA}")

ax11.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=14)
ax11.set_ylabel("KL divergence  (training data)", fontsize=14)
ax11.legend(fontsize=11, loc="upper left")
ax11.grid(True, alpha=0.3)
se_note = f"  (shaded = ±1 SE, {N_RUNS} runs)" if N_RUNS > 1 else ""
fig11.suptitle(
    r"KL Loss Profile over $\delta$  — Neural-Net Weights Frozen at Convergence"
    f"\n{se_note}  Minimum identifies δ̂; flatness signals weak identification",
    fontsize=12, fontweight="bold")
fig11.tight_layout()
fig11.savefig("figures/fig11_kl_delta_profile.pdf", dpi=150, bbox_inches="tight")
fig11.savefig("figures/fig11_kl_delta_profile.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig11_kl_delta_profile.pdf")
plt.close(fig11)


# ── Console summary — sweep results ──────────────────────────────────────────
print("\n  δ IDENTIFICATION SUMMARY")
print(f"  {'true δ':>8}  {'blend mean':>12}  {'blend SE':>10}  "
      f"{'E2E mean':>10}  {'E2E SE':>8}")
for _td, _bm, _bs, _em, _es in zip(
        DELTA_GRID, _blend_means, _blend_se, _e2e_means, _e2e_se):
    print(f"  {_td:>8.2f}  {_bm:>12.4f}  {_bs:>10.4f}  "
          f"{_em:>10.4f}  {_es:>8.4f}")

print("\n  θ ROBUSTNESS SUMMARY  (RMSE)")
print(f"  {'θ':>5}  {'AIDS':>10}  {'NeuIRL':>10}  "
      f"{'MDP-bl':>10}  {'MDP-E2E':>10}")
for _tt in THETA_GRID:
    _d = _th_agg[_tt]
    print(f"  {_tt:>5.1f}  "
          f"{_d['rmse_aids']['mean']:>10.5f}  "
          f"{_d['rmse_nirl']['mean']:>10.5f}  "
          f"{_d['rmse_mdp']['mean']:>10.5f}  "
          f"{_d['rmse_e2e']['mean']:>10.5f}")

print("\n  Habit welfare (Priority 2) — structural CV error vs Ground Truth")
print(f"  GT CV = £{_wh_gt_mean:.2f}")
for nm in _wh_plot_nms:
    _abs_e = abs(welf_hab_agg[nm]["mean"] - _wh_gt_mean)
    _pct_e = 100 * _abs_e / max(abs(_wh_gt_mean), 1e-9)
    print(f"    {nm:<25}  abs_err={_abs_e:.4f}  ({_pct_e:.1f}%)")

print("\n  All figures saved to figures/")
print("  Done.")


# ════════════════════════════════════════════════════════════════════
#  SECTION 17: WELFARE BOUNDS OVER THE IDENTIFIED SET OF δ  (Priority 1)
#  Train MDP Neural IRL at each δ in the identified set [0.50, 0.90].
#  Compute structural CV for 10% ibuprofen shock at each δ.
#  Plot CV vs δ with E2E and blend attractors marked.
#  Converts the δ non-identification into a formal sensitivity analysis.
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 17 — WELFARE BOUNDS OVER IDENTIFIED SET OF δ")
print("=" * 72)

DELTA_WELF_GRID = [0.45, 0.60, 0.75, 0.90]
N_WELF_SEEDS    = min(N_RUNS, 3)   # seeds per grid point
WELF_EPOCHS     = EPOCHS

def run_welfare_delta_seed(seed: int, delta_fixed: float) -> dict:
    """Train MDP Neural IRL with a *fixed* δ and compute structural CV.

    The neural-network weights adapt to each δ value; only δ is pinned
    externally by pre-computing xbar with the given decay rate.  This
    traces out CV as a function of δ while keeping everything else
    identical to the main habit DGP.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = N_OBS
    Z = np.random.uniform(1, 5, (N, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    theta = 0.3                           # standard habit strength
    hc    = HabitFormationConsumer(theta=theta, decay=delta_fixed)
    w_hab, xbar_tr = hc.solve_demand(p_pre,  income, return_xbar=True)
    p_post = p_pre.copy(); p_post[:, 1] *= 1.1   # 10% ibuprofen shock
    w_shock, _ = hc.solve_demand(p_post, income, return_xbar=True)

    q_tr  = w_hab * income[:, None] / np.maximum(p_pre, 1e-8)
    qp_tr = np.vstack([q_tr[0:1], q_tr[:-1]])

    # Ground-truth CV with fixed δ
    xb_mn = xbar_tr.mean(0)
    qp_mn = qp_tr.mean(0)
    _p0   = p_pre.mean(0)
    _p1   = p_post.mean(0)
    _y    = float(income.mean())
    _WS   = 40
    _path = np.linspace(_p0, _p1, _WS)
    _dp   = (_p1 - _p0) / _WS

    _gt_cv = 0.0
    for _tt in range(_WS):
        _p   = _path[_tt]
        _flr = hc.theta * xb_mn + 1e-6
        def _neg_u(x, _p=_p):
            adj = x - hc.theta * xb_mn
            return (1e10 if np.any(adj <= 0)
                    else -(np.sum(hc.alpha * adj**hc.rho))**(1/hc.rho))
        from scipy.optimize import minimize as _minimize
        _r = _minimize(
            _neg_u, np.maximum(_y / (3 * _p), _flr + 0.01),
            bounds=[(_flr[j], None) for j in range(3)],
            constraints=[{"type": "eq", "fun": lambda x, p=_p: p @ x - _y}],
            method="SLSQP")
        _w_gt = _r.x * _p / _y if _r.success else np.ones(3) / 3
        _gt_cv -= (_w_gt * _y / _p) @ _dp

    # ── LA-AIDS ──────────────────────────────────────────────────────────
    aids_h = AIDSBench(); aids_h.fit(p_pre, w_hab, income)

    # ── Static Neural IRL ────────────────────────────────────────────────
    nirl_h = NeuralIRL(n_goods=3, hidden_dim=128)
    nirl_h, _ = train_neural_irl(
        nirl_h, p_pre, income, w_hab, epochs=WELF_EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # ── MDP Neural IRL with fixed δ (xbar pre-built) ─────────────────────
    mdp_h = MDPNeuralIRL(n_goods=3, hidden_dim=128)
    mdp_h, _ = train_neural_irl(
        mdp_h, p_pre, income, w_hab, epochs=WELF_EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_tr, q_prev_data=qp_tr, device=DEVICE)

    # ── MDP E2E — frozen-δ grid sweep ────────────────────────────────────
    log_q_tr = np.log(np.maximum(q_tr, 1e-6))
    _vrng_w = np.random.default_rng(seed + 77777)
    _nv_w   = max(len(p_pre) // 5, 80)
    _pv_w   = np.clip(_vrng_w.uniform(1, 5, (_nv_w, 3)) +
                      _vrng_w.normal(0, 0.1, (_nv_w, 3)), 1e-3, None)
    _yv_w   = _vrng_w.uniform(1200, 2000, _nv_w)
    _hcv_w  = HabitFormationConsumer(theta=0.3, decay=delta_fixed)
    _wv_w, _ = _hcv_w.solve_demand(_pv_w, _yv_w, return_xbar=True)
    _qv_w   = _wv_w * _yv_w[:, None] / np.maximum(_pv_w, 1e-8)
    _lqv_w  = np.log(np.maximum(_qv_w, 1e-6))

    _sw_welf = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q_tr,
        _pv_w, _yv_w, _wv_w, _lqv_w,
        delta_grid=MDP_DELTA_GRID, epochs=WELF_EPOCHS, device=DEVICE,
        n_goods=3, hidden=128,
        lam_mono=0.2, lam_slut=0.05, batch=256, lr=5e-4,
        tag=f"Welf-E2E-d{delta_fixed:.1f}")
    mdp_ee_h = _sw_welf["best_model"]

    with torch.no_grad():
        _lq_t = torch.tensor(log_q_tr, dtype=torch.float32, device=DEVICE)
        xbar_ee_h = compute_xbar_e2e(
            mdp_ee_h.delta.to(DEVICE), _lq_t, store_ids=None).cpu().numpy()

    # ── Structural CV for each model (x̄ fixed at training mean) ──────────
    xb_r = xb_mn.reshape(1, -1)
    qp_r = qp_mn.reshape(1, -1)
    xb_ee_r = xbar_ee_h.mean(0, keepdims=True)

    def _cv(spec_tag, **ekw):
        _loss = 0.0
        for _tt in range(_WS):
            _pt = _path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_y]), **ekw)[0]
            _loss -= (_w * _y / _path[_tt]) @ _dp
        return _loss

    return {
        "delta_fixed":   delta_fixed,
        "delta_ee_hat":  _sw_welf["delta_hat"],   # δ̂ from grid sweep
        "cv_gt":         _gt_cv,
        "cv_aids":       _cv("aids",    aids=aids_h),
        "cv_nirl":       _cv("n-irl",   nirl=nirl_h, device=DEVICE),
        "cv_mdp":        _cv("mdp-irl", mdp_nirl=mdp_h,
                              xbar=xb_r, q_prev=qp_r, device=DEVICE),
        "cv_e2e":        _cv("mdp-e2e", mdp_e2e=mdp_ee_h,
                              xbar_e2e=xb_ee_r, device=DEVICE),
    }


print(f"\n  [P1] Welfare sensitivity: {len(DELTA_WELF_GRID)} δ values × "
      f"{N_WELF_SEEDS} seeds × {WELF_EPOCHS} epochs ...")

_welf_delta_rows = []
for _wd in DELTA_WELF_GRID:
    for _si in range(N_WELF_SEEDS):
        _seed_w = 700 + _si * 17
        print(f"    δ={_wd:.2f}  seed={_seed_w}", end="", flush=True)
        import time as _time; _tw0 = _time.time()
        _wr = run_welfare_delta_seed(_seed_w, _wd)
        print(f"  →  CV_mdp={_wr['cv_mdp']:.3f}  CV_nirl={_wr['cv_nirl']:.3f}"
              f"  δ̂_E2E={_wr['delta_ee_hat']:.3f}"
              f"  ({_time.time()-_tw0:.0f}s)")
        _welf_delta_rows.append(_wr)

# ── Aggregate across seeds ────────────────────────────────────────────────
_welf_agg_by_delta = {}
for _wd in DELTA_WELF_GRID:
    _rows_w = [r for r in _welf_delta_rows if r["delta_fixed"] == _wd]
    _welf_agg_by_delta[_wd] = {
        key: {
            "mean": np.nanmean([r[key] for r in _rows_w]),
            "se":   (np.nanstd([r[key] for r in _rows_w], ddof=1)
                     / np.sqrt(len(_rows_w)) if len(_rows_w) > 1 else 0.0),
        }
        for key in ["cv_gt", "cv_aids", "cv_nirl", "cv_mdp", "cv_e2e",
                    "delta_ee_hat"]
    }

# ── Figure 12: CV vs δ with identified set ───────────────────────────────
_wc_models = [
    ("LA-AIDS (static)",  "cv_aids", "#E53935", "--"),
    ("Neural IRL (static)", "cv_nirl", "#1E88E5", "-."),
    ("MDP Neural IRL",    "cv_mdp",  "#00897B", "-"),
    ("MDP IRL (E2E)",     "cv_e2e",  "#FF6F00", "-"),
    ("Ground Truth",      "cv_gt",   "k",        ":"),
]

fig12, (ax12a, ax12b) = plt.subplots(1, 2, figsize=(15, 6))

for lbl, key, col, ls in _wc_models:
    _mu = [_welf_agg_by_delta[_wd][key]["mean"] for _wd in DELTA_WELF_GRID]
    _se_vals = [_welf_agg_by_delta[_wd][key]["se"]   for _wd in DELTA_WELF_GRID]
    ax12a.errorbar(DELTA_WELF_GRID, _mu, yerr=_se_vals,
                   fmt=ls, color=col, lw=2.2, ms=7, capsize=4, marker="o",
                   label=lbl)

# Mark E2E and blend attractors from main runs
_e2e_attr   = delta_mdp_e2e_mean    # from Section 10
_blend_attr = delta_mdp_mean
ax12a.axvline(_e2e_attr,   color="#FF6F00", ls=":", lw=1.8,
              label=rf"E2E attractor $\hat{{\delta}}$={_e2e_attr:.2f}")
ax12a.axvline(_blend_attr, color="#00897B", ls=":", lw=1.8,
              label=rf"Blend attractor $\hat{{\delta}}$={_blend_attr:.2f}")
# Shade the identified set
ax12a.axvspan(_e2e_attr, _blend_attr, color="grey", alpha=0.10,
              label="Identified set")
ax12a.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=13)
ax12a.set_ylabel("Structural CV  (£ equivalent variation)", fontsize=12)
ax12a.set_title("Panel A: CV vs δ  (10% ibuprofen shock)", fontsize=12,
                fontweight="bold")
ax12a.legend(fontsize=10, loc="best")
ax12a.grid(True, alpha=0.3)

# Panel B: absolute CV error relative to ground truth
_gt_curve = [_welf_agg_by_delta[_wd]["cv_gt"]["mean"] for _wd in DELTA_WELF_GRID]
for lbl, key, col, ls in _wc_models[:-1]:   # skip ground truth
    _mu = [_welf_agg_by_delta[_wd][key]["mean"] for _wd in DELTA_WELF_GRID]
    _err = [abs(_mu[i] - _gt_curve[i]) for i in range(len(DELTA_WELF_GRID))]
    ax12b.plot(DELTA_WELF_GRID, _err, ls, color=col, lw=2.2, ms=7,
               marker="o", label=lbl)
ax12b.axvline(_e2e_attr,   color="#FF6F00", ls=":", lw=1.8)
ax12b.axvline(_blend_attr, color="#00897B", ls=":", lw=1.8)
ax12b.axvspan(_e2e_attr, _blend_attr, color="grey", alpha=0.10)
ax12b.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=13)
ax12b.set_ylabel("|CV error| vs Ground Truth  (£)", fontsize=12)
ax12b.set_title("Panel B: |CV Error| vs δ  (lower = better)", fontsize=12,
                fontweight="bold")
ax12b.legend(fontsize=10)
ax12b.grid(True, alpha=0.3)

fig12.suptitle(
    r"Welfare Sensitivity to $\delta$ over the Identified Set  "
    r"$[\hat{\delta}_{E2E},\, \hat{\delta}_{blend}]$"
    f"\n{N_WELF_SEEDS} seeds × {WELF_EPOCHS} epochs · True δ = pinned to grid · "
    r"θ = 0.3 · Shock: 10% ↑ ibuprofen",
    fontsize=12, fontweight="bold")
fig12.tight_layout()
fig12.savefig("figures/fig12_welfare_delta_sensitivity.pdf", dpi=150,
              bbox_inches="tight")
fig12.savefig("figures/fig12_welfare_delta_sensitivity.png", dpi=150,
              bbox_inches="tight")
print("\n    Saved: figures/fig12_welfare_delta_sensitivity.pdf")
plt.close(fig12)

# Console summary
print("\n  WELFARE SENSITIVITY OVER IDENTIFIED SET OF δ")
print(f"  {'δ':>6}  {'GT CV':>8}  {'AIDS CV':>8}  {'NIRL CV':>8}  "
      f"{'MDP CV':>8}  {'E2E CV':>8}  {'δ̂ E2E':>8}")
for _wd in DELTA_WELF_GRID:
    _d = _welf_agg_by_delta[_wd]
    print(f"  {_wd:>6.2f}  "
          f"{_d['cv_gt']['mean']:>8.4f}  "
          f"{_d['cv_aids']['mean']:>8.4f}  "
          f"{_d['cv_nirl']['mean']:>8.4f}  "
          f"{_d['cv_mdp']['mean']:>8.4f}  "
          f"{_d['cv_e2e']['mean']:>8.4f}  "
          f"{_d['delta_ee_hat']['mean']:>8.4f}")
# Check monotonicity
_mdp_cv_vals = [_welf_agg_by_delta[_wd]["cv_mdp"]["mean"]
                for _wd in DELTA_WELF_GRID]
_is_mono = all(_mdp_cv_vals[i] <= _mdp_cv_vals[i+1]
               for i in range(len(_mdp_cv_vals)-1)) or \
           all(_mdp_cv_vals[i] >= _mdp_cv_vals[i+1]
               for i in range(len(_mdp_cv_vals)-1))
_min_cv = min(_mdp_cv_vals); _max_cv = max(_mdp_cv_vals)
print(f"\n  MDP CV range over identified set [{_e2e_attr:.2f}, {_blend_attr:.2f}]:")
print(f"    min={_min_cv:.4f}  max={_max_cv:.4f}  "
      f"{'MONOTONE ✓' if _is_mono else 'NON-MONOTONE ⚠'}")
if _is_mono:
    print("    → Report as bounds: CV ∈ "
          f"[{min(_mdp_cv_vals):.4f}, {max(_mdp_cv_vals):.4f}]")


# ════════════════════════════════════════════════════════════════════
#  SECTION 18: PLACEBO TEST — SORTING CONFOUND  (Priority 4)
#  DGP: static CES with no habit, but cross-sectional store heterogeneity.
#  By construction, aspirin-preferring "stores" face lower ibuprofen prices.
#  If MDP Neural IRL achieves an RMSE advantage on this purely static data,
#  it quantifies the maximum sorting contamination in the Dominick's results.
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 18 — PLACEBO TEST: SORTING CONFOUND")
print("=" * 72)

N_PLACEBO_SEEDS = N_RUNS
PLACEBO_EPOCHS  = EPOCHS

def run_placebo_seed(seed: int) -> dict:
    """Placebo DGP: static CES + store-level preference × price sorting.

    There is NO true habit formation.  However, "store types" differ in
    their preference weights (α), and the ibuprofen price is negatively
    correlated with the aspirin preference weight across stores.
    This mimics the sorting pattern observed in Dominick's.

    If MDP Neural IRL achieves lower RMSE than static Neural IRL on this
    data, the gain is purely spurious — a bound on sorting contamination.
    """
    rng = np.random.default_rng(seed)
    N   = N_OBS

    # ── Store-level types (S=10 stores, each repeated N//S times) ─────────
    S = 10
    n_per = N // S
    # Store preference: aspirin-heavy vs ibuprofen-heavy
    asp_prefs  = rng.uniform(0.20, 0.50, S)      # α₀ (aspirin) per store
    ibu_prices = 3.0 - 1.5 * (asp_prefs - asp_prefs.min()) / \
                   (asp_prefs.max() - asp_prefs.min() + 1e-8)  # negative correlation

    alpha_all  = np.zeros((N, 3))
    prices_all = np.zeros((N, 3))
    income_all = np.zeros(N)
    store_ids  = np.zeros(N, dtype=int)

    for s in range(S):
        sl = slice(s * n_per, (s + 1) * n_per if s < S-1 else N)
        n_s = len(range(N)[sl])

        a0 = asp_prefs[s]
        a2 = rng.uniform(0.15, 0.35)            # ibuprofen pref
        a1 = 1.0 - a0 - a2                       # acetaminophen
        a1 = max(a1, 0.05)
        _norm = a0 + a1 + a2
        alpha_s = np.array([a0, a1, a2]) / _norm

        prices_all[sl, 0] = rng.uniform(2.0, 5.0, n_s)   # aspirin price
        prices_all[sl, 1] = rng.uniform(2.0, 5.0, n_s)   # acetaminophen
        prices_all[sl, 2] = (ibu_prices[s]
                             + rng.normal(0, 0.2, n_s))    # ibuprofen (sorted)
        prices_all[sl, 2] = np.maximum(prices_all[sl, 2], 0.5)
        income_all[sl]    = rng.uniform(1200, 2000, n_s)
        alpha_all[sl]     = alpha_s
        store_ids[sl]     = s

    # ── True CES demand (rho = 0.45, no habit) ───────────────────────────
    rho = 0.45
    def _ces_demand(alpha, p, y):
        """CES budget shares for store-specific alpha."""
        N_loc = p.shape[0]
        w_out = np.zeros((N_loc, 3))
        for i in range(N_loc):
            numer = alpha[i] * p[i]**(rho - 1)
            denom = (alpha[i] * p[i]**(rho - 1)).sum()
            w_out[i] = numer / (denom + 1e-12)
        return w_out

    w_true = _ces_demand(alpha_all, prices_all, income_all)

    # Post-shock: 10% ibuprofen price increase
    p_shock = prices_all.copy(); p_shock[:, 2] *= 1.10
    w_shock = _ces_demand(alpha_all, p_shock, income_all)

    # ── Dummy xbar (static baseline: previous period quantity, no decay) ──
    q_prev = w_true * income_all[:, None] / np.maximum(prices_all, 1e-8)
    q_prev = np.vstack([q_prev[0:1], q_prev[:-1]])   # lag by one

    # For MDP E2E: we need log-quantity sequence
    log_q = np.log(np.maximum(w_true * income_all[:, None]
                               / np.maximum(prices_all, 1e-8), 1e-6))

    # ── LA-AIDS ──────────────────────────────────────────────────────────
    aids_pl = AIDSBench(); aids_pl.fit(prices_all, w_true, income_all)
    rmse_aids_pl = get_metrics("aids", p_shock, income_all, w_shock,
                               aids=aids_pl)["RMSE"]

    # ── Static Neural IRL ────────────────────────────────────────────────
    nirl_pl = NeuralIRL(n_goods=3, hidden_dim=256)
    nirl_pl, _ = train_neural_irl(
        nirl_pl, prices_all, income_all, w_true,
        epochs=PLACEBO_EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)
    rmse_nirl_pl = get_metrics("n-irl", p_shock, income_all, w_shock,
                               nirl=nirl_pl, device=DEVICE)["RMSE"]

    # ── MDP Neural IRL (xbar = lagged q, no true habit) ──────────────────
    xbar_pl = q_prev.copy()   # play the role of xbar
    mdp_pl  = MDPNeuralIRL(n_goods=3, hidden_dim=256)
    mdp_pl, _ = train_neural_irl(
        mdp_pl, prices_all, income_all, w_true,
        epochs=PLACEBO_EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_pl, q_prev_data=q_prev, device=DEVICE)
    xbar_sh_pl = np.vstack([xbar_pl[0:1], xbar_pl[:-1]])
    rmse_mdp_pl = get_metrics("mdp-irl", p_shock, income_all, w_shock,
                              mdp_nirl=mdp_pl,
                              xbar_shock=xbar_sh_pl,
                              q_prev_shock=q_prev, device=DEVICE)["RMSE"]

    # ── MDP IRL E2E — frozen-δ grid sweep (placebo: static DGP) ─────────
    # True δ is undefined here (static DGP); sweep still selects the
    # δ that minimises val-KL — expected to be arbitrary / near boundary.
    _vrng_pl = np.random.default_rng(seed + 66666)
    _nv_pl   = max(len(prices_all) // 5, 80)
    _pv_pl   = np.clip(_vrng_pl.uniform(1, 5, (_nv_pl, 3)) +
                       _vrng_pl.normal(0, 0.1, (_nv_pl, 3)), 1e-3, None)
    _yv_pl   = _vrng_pl.uniform(1200, 2000, _nv_pl)
    _hcv_pl  = CESConsumer()
    _wv_pl   = _hcv_pl.solve_demand(_pv_pl, _yv_pl)
    _qv_pl   = _wv_pl * _yv_pl[:, None] / np.maximum(_pv_pl, 1e-8)
    _lqv_pl  = np.log(np.maximum(_qv_pl, 1e-6))

    _sw_pl = fit_mdp_delta_grid(
        prices_all, income_all, w_true, log_q,
        _pv_pl, _yv_pl, _wv_pl, _lqv_pl,
        delta_grid=MDP_DELTA_GRID, epochs=PLACEBO_EPOCHS, device=DEVICE,
        n_goods=3, hidden=256,
        lam_mono=0.2, lam_slut=0.05, batch=256, lr=5e-4,
        tag="Placebo-E2E")
    mdp_ee_pl = _sw_pl["best_model"]

    with torch.no_grad():
        _lq_t = torch.tensor(log_q, dtype=torch.float32, device=DEVICE)
        xbar_ee_pl = compute_xbar_e2e(
            mdp_ee_pl.delta.to(DEVICE), _lq_t, store_ids=None).cpu().numpy()
    xbar_sh_ee_pl = np.vstack([xbar_ee_pl[0:1], xbar_ee_pl[:-1]])
    rmse_e2e_pl = get_metrics("mdp-e2e", p_shock, income_all, w_shock,
                              mdp_e2e=mdp_ee_pl,
                              xbar_e2e=xbar_sh_ee_pl, device=DEVICE)["RMSE"]

    return {
        "rmse_aids":    rmse_aids_pl,
        "rmse_nirl":    rmse_nirl_pl,
        "rmse_mdp":     rmse_mdp_pl,
        "rmse_e2e":     rmse_e2e_pl,
        "delta_e2e":    _sw_pl["delta_hat"],     # δ̂ on static data (spurious)
        "store_ids":    store_ids,
        "asp_prefs":    asp_prefs,
        "ibu_prices":   ibu_prices,
    }


# print(f"\n  [P4] Placebo test: {N_PLACEBO_SEEDS} seeds × {PLACEBO_EPOCHS} epochs ...")
# _placebo_rows = []
# for _si in range(N_PLACEBO_SEEDS):
#     _seed_pl = 800 + _si * 11
#     import time as _time2; _tpl0 = _time2.time()
#     print(f"    seed={_seed_pl}", end="", flush=True)
#     _pr = run_placebo_seed(_seed_pl)
#     print(f"  →  RMSE: AIDS={_pr['rmse_aids']:.5f}  NIRL={_pr['rmse_nirl']:.5f}"
#           f"  MDP={_pr['rmse_mdp']:.5f}  E2E={_pr['rmse_e2e']:.5f}"
#           f"  δ̂={_pr['delta_e2e']:.3f}"
#           f"  ({_time2.time()-_tpl0:.0f}s)")
#     _placebo_rows.append(_pr)

# Aggregate
_pl_keys = ["rmse_aids", "rmse_nirl", "rmse_mdp", "rmse_e2e", "delta_e2e"]
_pl_agg  = {k: {"mean": np.nanmean([r[k] for r in _placebo_rows]),
                 "se":   _se([r[k] for r in _placebo_rows])}
            for k in _pl_keys}

# Compare with main habit runs
_hab_aids_mu  = mdp_agg["aids_rmse"]["mean"]
_hab_nirl_mu  = mdp_agg["nirl_rmse"]["mean"]
_hab_mdp_mu   = mdp_agg["mdp_rmse"]["mean"]
_hab_e2e_mu   = mdp_agg["mdp_e2e_rmse"]["mean"]

# ── Figure 13: Placebo vs habit-DGP bar chart ─────────────────────────────
fig13, (ax13a, ax13b) = plt.subplots(1, 2, figsize=(14, 6))

_pl_labels  = ["LA-AIDS", "Neural IRL\n(static)", "MDP Neural\nIRL", "MDP IRL\n(E2E)"]
_pl_rmse_pl = [_pl_agg[k]["mean"]  for k in _pl_keys[:4]]
_pl_se_pl   = [_pl_agg[k]["se"]    for k in _pl_keys[:4]]
_pl_rmse_hab= [_hab_aids_mu, _hab_nirl_mu, _hab_mdp_mu, _hab_e2e_mu]

_x = np.arange(len(_pl_labels))
_w = 0.35
_col_pl  = ["#EF9A9A", "#90CAF9", "#80CBC4", "#FFCC80"]
_col_hab = ["#E53935", "#1E88E5", "#00897B", "#FF6F00"]

bars_pl  = ax13a.bar(_x - _w/2, _pl_rmse_pl,  _w, color=_col_pl,
                     label="Placebo (static + sorting)", yerr=_pl_se_pl,
                     capsize=5, edgecolor="k", lw=0.8)
bars_hab = ax13a.bar(_x + _w/2, _pl_rmse_hab, _w, color=_col_hab,
                     label="True habit DGP", edgecolor="k", lw=0.8)
ax13a.set_xticks(_x); ax13a.set_xticklabels(_pl_labels, fontsize=11)
ax13a.set_ylabel("Post-shock RMSE", fontsize=12)
ax13a.set_title("Panel A: Absolute RMSE — Placebo vs Habit DGP",
                fontsize=12, fontweight="bold")
ax13a.legend(fontsize=11)
ax13a.grid(axis="y", alpha=0.3)

# Panel B: MDP RMSE reduction (vs static Neural IRL)
_pl_gain_mdp  = _pl_agg["rmse_nirl"]["mean"]  - _pl_agg["rmse_mdp"]["mean"]
_pl_gain_e2e  = _pl_agg["rmse_nirl"]["mean"]  - _pl_agg["rmse_e2e"]["mean"]
_hab_gain_mdp = _hab_nirl_mu  - _hab_mdp_mu
_hab_gain_e2e = _hab_nirl_mu  - _hab_e2e_mu

_gain_labels = ["MDP Neural IRL\nvs static", "MDP IRL (E2E)\nvs static"]
_gain_pl  = [_pl_gain_mdp,  _pl_gain_e2e]
_gain_hab = [_hab_gain_mdp, _hab_gain_e2e]
_gain_se_pl = [np.sqrt(_pl_agg["rmse_nirl"]["se"]**2 + _pl_agg["rmse_mdp"]["se"]**2),
               np.sqrt(_pl_agg["rmse_nirl"]["se"]**2 + _pl_agg["rmse_e2e"]["se"]**2)]

_x2 = np.arange(len(_gain_labels))
ax13b.bar(_x2 - _w/2, _gain_pl,  _w, color=["#80CBC4", "#FFCC80"],
          label="Placebo (static + sorting)",
          yerr=_gain_se_pl, capsize=5, edgecolor="k", lw=0.8)
ax13b.bar(_x2 + _w/2, _gain_hab, _w, color=["#00897B", "#FF6F00"],
          label="True habit DGP", edgecolor="k", lw=0.8)
ax13b.axhline(0, color="k", lw=1.2, ls="--")
ax13b.set_xticks(_x2); ax13b.set_xticklabels(_gain_labels, fontsize=11)
ax13b.set_ylabel("RMSE reduction vs static Neural IRL", fontsize=12)
ax13b.set_title("Panel B: MDP Advantage — Placebo vs Habit DGP\n"
                "(positive = MDP wins; negative = sorting artefact)",
                fontsize=12, fontweight="bold")
ax13b.legend(fontsize=11)
ax13b.grid(axis="y", alpha=0.3)

fig13.suptitle(
    "Placebo Test: Static CES + Store Sorting vs True Habit DGP\n"
    f"{N_PLACEBO_SEEDS} seeds × {PLACEBO_EPOCHS} epochs  ·  "
    "S=10 store types  ·  Aspirin pref ↔ Ibuprofen price  (negative correlation)",
    fontsize=12, fontweight="bold")
fig13.tight_layout()
fig13.savefig("figures/fig13_placebo_sorting.pdf", dpi=150, bbox_inches="tight")
fig13.savefig("figures/fig13_placebo_sorting.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig13_placebo_sorting.pdf")
plt.close(fig13)

# Console summary
print("\n  PLACEBO TEST RESULTS")
print(f"  {'Model':<22}  {'Placebo RMSE':>14}  {'Habit RMSE':>12}  "
      f"{'RMSE diff':>11}  {'Sorting %':>10}")
for lbl, pl_key, hab_mu in [
    ("LA-AIDS",         "rmse_aids", _hab_aids_mu),
    ("Neural IRL",      "rmse_nirl", _hab_nirl_mu),
    ("MDP Neural IRL",  "rmse_mdp",  _hab_mdp_mu),
    ("MDP IRL (E2E)",   "rmse_e2e",  _hab_e2e_mu),
]:
    _pl_m  = _pl_agg[pl_key]["mean"]
    _diff  = _pl_m - hab_mu
    _sort_pct = 100 * (_pl_agg["rmse_nirl"]["mean"] - _pl_m) / \
                max(_hab_nirl_mu - hab_mu, 1e-9)
    print(f"  {lbl:<22}  {_pl_m:>14.5f}  {hab_mu:>12.5f}  "
          f"{_diff:>+11.5f}  {_sort_pct:>9.1f}%")
_pl_mdp_adv  = _pl_agg["rmse_nirl"]["mean"] - _pl_agg["rmse_mdp"]["mean"]
_hab_mdp_adv = _hab_nirl_mu - _hab_mdp_mu
print(f"\n  MDP advantage (RMSE reduction vs static):")
print(f"    Placebo DGP: {_pl_mdp_adv:.5f}  "
      f"({'positive → sorting artefact ⚠' if _pl_mdp_adv > 0 else 'zero/negative → no spurious gain ✓'})")
print(f"    Habit DGP:   {_hab_mdp_adv:.5f}")
if _hab_mdp_adv > 0:
    _spurious_share = max(0, _pl_mdp_adv) / _hab_mdp_adv
    print(f"    Spurious share: {100*_spurious_share:.1f}% of Dominick's-style gain"
          f" could be sorting")

print("\n  All figures saved to figures/")
print("  Done (Sections 17-18).")


# ════════════════════════════════════════════════════════════════════
#  SECTION 19: LARGE-N KL PROFILE SWEEP  (N = 5 000 and N = 10 000)
#  Objective: verify that the KL-profile minimum at δ̂ sharpens as N
#  grows, confirming that δ is identified in the population limit.
#  If the bowl is still flat at N=10 000 the parameter is structurally
#  non-identified; if it sharpens the earlier flat profile was purely
#  a small-sample phenomenon.
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 19 — LARGE-N KL PROFILE SWEEP  (N=5000 & N=10000)")
print("=" * 72)

LARGE_N_GRID   = [5_000, 8_000]
N_LARGE_SEEDS  = 5          # seeds per N; increase for publication
LARGE_EPOCHS   = 8_000
_KL_DELTA_GRID = np.linspace(0.2, 0.99, 80)   # same grid as Section 10


def run_large_n_kl_sweep(seed: int, n_obs: int, epochs: int = LARGE_EPOCHS) -> dict:
    """Train MDP-E2E models via frozen-δ grid sweep on n_obs observations.

    δ is NOT jointly learned.  For each candidate in MDP_DELTA_GRID we train
    a model with frozen δ, evaluate hold-out KL, and select δ̂ = argmin.
    The identified set is all δ within 2 SE of the minimum hold-out KL.

    Returns
    -------
    dict with keys:
        kl_profile_e2e  : (K,) hold-out KL at each grid point (MDP-IRL E2E)
        delta_e2e_hat   : scalar δ̂ (grid sweep)
        id_set_e2e      : (lo, hi) identified-set interval
        kl_delta_grid   : (K,) δ grid used
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Training data (Habit DGP, same process as run_one_seed) ────────────
    Z      = np.random.uniform(1, 5, (n_obs, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (n_obs, 3))
    income = np.random.uniform(1200, 2000, n_obs)

    hc = HabitFormationConsumer()          # true δ = 0.7, θ = 0.3
    w_hab, _ = hc.solve_demand(p_pre, income, return_xbar=True)

    q_tr      = w_hab * income[:, None] / np.maximum(p_pre, 1e-8)
    log_q_seq = np.log(np.maximum(q_tr, 1e-6))

    # ── Fresh validation set for δ selection ──────────────────────────────
    _vrng_lg = np.random.default_rng(seed + 55555)
    _nv_lg   = max(n_obs // 5, 200)
    _pv_lg   = np.clip(_vrng_lg.uniform(1, 5, (_nv_lg, 3)) +
                       _vrng_lg.normal(0, 0.1, (_nv_lg, 3)), 1e-3, None)
    _yv_lg   = _vrng_lg.uniform(1200, 2000, _nv_lg)
    _hcv_lg  = HabitFormationConsumer()
    _wv_lg, _ = _hcv_lg.solve_demand(_pv_lg, _yv_lg, return_xbar=True)
    _qv_lg   = _wv_lg * _yv_lg[:, None] / np.maximum(_pv_lg, 1e-8)
    _lqv_lg  = np.log(np.maximum(_qv_lg, 1e-6))

    # ── Grid sweep — MDP IRL E2E ──────────────────────────────────────────
    _sw_lg = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q_seq,
        _pv_lg, _yv_lg, _wv_lg, _lqv_lg,
        delta_grid=MDP_DELTA_GRID, epochs=epochs, device=DEVICE,
        n_goods=3, hidden=256,
        lam_mono=0.3, lam_slut=0.1, batch=512, lr=5e-4,
        tag=f"E2E-N{n_obs}-s{seed}")

    return {
        "n_obs":           n_obs,
        "seed":            seed,
        "kl_delta_grid":   MDP_DELTA_GRID.copy(),
        "kl_profile_e2e":  _sw_lg["kl_grid"],
        "delta_e2e_hat":   _sw_lg["delta_hat"],
        "id_set_e2e":      _sw_lg["id_set"],
    }


# ── Run the sweep ─────────────────────────────────────────────────────────
_large_n_rows = {}   # {n_obs: [row, ...]}
for _ln in LARGE_N_GRID:
    _large_n_rows[_ln] = []
    print(f"\n  N = {_ln:,d}  ({N_LARGE_SEEDS} seeds × {LARGE_EPOCHS} epochs)")
    for _si in range(N_LARGE_SEEDS):
        _seed_ln = 900 + _si * 23
        import time as _tln_mod; _tln0 = _tln_mod.time()
        print(f"    seed={_seed_ln}", end="", flush=True)
        _lr = run_large_n_kl_sweep(_seed_ln, n_obs=_ln)
        print(f"  →  δ̂_E2E={_lr['delta_e2e_hat']:.3f}"
              f"  ({_tln_mod.time()-_tln0:.0f}s)")
        _large_n_rows[_ln].append(_lr)


# ── Aggregate KL profiles ─────────────────────────────────────────────────
# For each N, compute mean ± SE across seeds
_large_n_kl_agg = {}   # {n_obs: {"e2e": {"mean", "se"}}}
for _ln, _rows in _large_n_rows.items():
    _e2e_stack = np.stack([r["kl_profile_e2e"] for r in _rows], 0)  # (seeds, K)
    _n_s = len(_rows)
    _large_n_kl_agg[_ln] = {
        "e2e": {
            "mean": _e2e_stack.mean(0),
            "se":   _e2e_stack.std(0, ddof=1) / np.sqrt(_n_s) if _n_s > 1 else np.zeros(_e2e_stack.shape[1]),
        },
        "delta_e2e_mean": np.mean([r["delta_e2e_hat"] for r in _rows]),
    }


# ── Figure 14: Large-N KL profile — comparison across N ──────────────────
# Layout: one subplot per N (columns), plus the original N=N_OBS as reference.
# Each subplot shows E2E (orange) plus the mean recovered δ̂ and the true δ=0.7.

_all_n = [N_OBS] + LARGE_N_GRID       # [800, 5000, 10000]
_n_panels = len(_all_n)
fig14, axes14 = plt.subplots(1, _n_panels, figsize=(6 * _n_panels, 5),
                              sharey=False)

_col_e2e_lg = "#FF6F00"

for _ax, _n in zip(axes14, _all_n):
    if _n == N_OBS:
        # Reuse the main-run aggregate (already computed in Section 10)
        _mu_e2e = kl_prof_e2e_mean
        _se_e2e = kl_prof_e2e_se
        _d_e2e  = delta_mdp_e2e_mean
        _dg     = kl_delta_grid
    else:
        _agg    = _large_n_kl_agg[_n]
        _mu_e2e = _agg["e2e"]["mean"]
        _se_e2e = _agg["e2e"]["se"]
        _d_e2e  = _agg["delta_e2e_mean"]
        _dg     = _KL_DELTA_GRID

    _ax.plot(_dg, _mu_e2e, color=_col_e2e_lg, lw=2.5,
             label=r"MDP-IRL E2E (learns $\hat{\delta}$)")
    if N_LARGE_SEEDS > 1 or _n == N_OBS:
        _ax.fill_between(_dg, _mu_e2e - _se_e2e, _mu_e2e + _se_e2e,
                         color=_col_e2e_lg, alpha=0.18)

    _ax.axvline(_d_e2e, color=_col_e2e_lg, ls=":", lw=1.8,
                label=rf"$\hat{{\delta}}_{{E2E}}$={_d_e2e:.3f}")
    _ax.axvline(TRUE_DELTA, color="k", ls="--", lw=2.0,
                label=f"True δ={TRUE_DELTA}")

    _ax.set_xlabel(r"$\delta$", fontsize=13)
    _ax.set_ylabel("KL divergence  (training data)", fontsize=11)
    _ax.set_title(f"N = {_n:,d}", fontsize=13, fontweight="bold")
    _ax.legend(fontsize=9, loc="upper left")
    _ax.grid(True, alpha=0.3)

fig14.suptitle(
    r"KL Profile over $\delta$ — Does identification sharpen with more data?"
    f"\nTrue δ = {TRUE_DELTA} · Habit DGP · Frozen net weights after convergence",
    fontsize=13, fontweight="bold")
fig14.tight_layout()
fig14.savefig("figures/fig14_large_n_kl_profile.pdf", dpi=150, bbox_inches="tight")
fig14.savefig("figures/fig14_large_n_kl_profile.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig14_large_n_kl_profile.pdf")
plt.close(fig14)


# ── Console summary ───────────────────────────────────────────────────────
print("\n  LARGE-N KL PROFILE SUMMARY")
print(f"  True δ = {TRUE_DELTA}")
print(f"  {'N':>8}  {'δ̂_E2E (mean)':>15}"
      f"  {'KL min E2E':>12}"
      f"  {'KL at true δ (E2E)':>20}")
for _n in _all_n:
    if _n == N_OBS:
        _mu_e2e = kl_prof_e2e_mean
        _d_e2e  = delta_mdp_e2e_mean
        _dg     = kl_delta_grid
    else:
        _agg    = _large_n_kl_agg[_n]
        _mu_e2e = _agg["e2e"]["mean"]
        _d_e2e  = _agg["delta_e2e_mean"]
        _dg     = _KL_DELTA_GRID
    _kl_min_e2e = _mu_e2e.min()
    # KL at the δ grid point closest to the true value
    _true_idx   = int(np.argmin(np.abs(_dg - TRUE_DELTA)))
    _kl_true_e2e = _mu_e2e[_true_idx]
    print(f"  {_n:>8,d}  {_d_e2e:>15.4f}"
          f"  {_kl_min_e2e:>12.6f}"
          f"  {_kl_true_e2e:>20.6f}")

# Identification verdict
print()
for _n in LARGE_N_GRID:
    _agg = _large_n_kl_agg[_n]
    _mu  = _agg["e2e"]["mean"]
    _dg  = _KL_DELTA_GRID
    _argmin = int(np.argmin(_mu))
    _delta_at_min = _dg[_argmin]
    _bias = abs(_delta_at_min - TRUE_DELTA)
    print(f"  N={_n:,d}: KL minimised at δ̂={_delta_at_min:.3f} "
          f"(bias vs truth = {_bias:.3f}) | "
          f"{'✓ well-identified' if _bias < 0.10 else '⚠ weakly identified'}")


# ════════════════════════════════════════════════════════════════════
#  SECTION 20: WELFARE ROBUSTNESS AT δ = 0.90
#  Run the Habit DGP welfare evaluation with delta FIXED at 0.90
#  (the upper end of the identified attractor range [0.51, 0.90]).
#  If the MDP CV error is near zero at δ=0.90 the welfare conclusion
#  is robust across the FULL attractor range — the strongest possible
#  robustness claim.  If it degrades, welfare claims must be bounded
#  to the E2E attractor range.
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 20 — WELFARE ROBUSTNESS CHECK AT δ = 0.90")
print("=" * 72)

DELTA_090      = 0.90
N_090_SEEDS    = min(N_RUNS, 3)   # seeds; 3 gives tight SEs without much cost
EPOCHS_090     = WELF_EPOCHS      # same as Section 17 for fair comparison

print(f"\n  Running {N_090_SEEDS} seeds × {EPOCHS_090} epochs at δ = {DELTA_090} ...")

_rows_090 = []
for _si in range(N_090_SEEDS):
    _seed_090 = 950 + _si * 31
    import time as _t090_mod; _t090_0 = _t090_mod.time()
    print(f"    seed={_seed_090}", end="", flush=True)
    _wr090 = run_welfare_delta_seed(_seed_090, DELTA_090)
    print(f"  →  CV_gt={_wr090['cv_gt']:.4f}"
          f"  CV_mdp={_wr090['cv_mdp']:.4f}"
          f"  CV_e2e={_wr090['cv_e2e']:.4f}"
          f"  δ̂_E2E={_wr090['delta_ee_hat']:.3f}"
          f"  ({_t090_mod.time()-_t090_0:.0f}s)")
    _rows_090.append(_wr090)

# Aggregate across seeds
_keys_090 = ["cv_gt", "cv_aids", "cv_nirl", "cv_mdp", "cv_e2e", "delta_ee_hat"]
_agg_090  = {k: {
    "mean": np.nanmean([r[k] for r in _rows_090]),
    "se":   (_se([r[k] for r in _rows_090]) if N_090_SEEDS > 1 else 0.0),
} for k in _keys_090}

_gt090   = _agg_090["cv_gt"]["mean"]
_mdp090  = _agg_090["cv_mdp"]["mean"]
_e2e090  = _agg_090["cv_e2e"]["mean"]
_nirl090 = _agg_090["cv_nirl"]["mean"]
_aids090 = _agg_090["cv_aids"]["mean"]

_err_mdp090  = abs(_mdp090  - _gt090)
_err_e2e090  = abs(_e2e090  - _gt090)
_err_nirl090 = abs(_nirl090 - _gt090)
_err_aids090 = abs(_aids090 - _gt090)

_pct_mdp090  = 100 * _err_mdp090  / max(abs(_gt090), 1e-9)
_pct_e2e090  = 100 * _err_e2e090  / max(abs(_gt090), 1e-9)
_pct_nirl090 = 100 * _err_nirl090 / max(abs(_gt090), 1e-9)
_pct_aids090 = 100 * _err_aids090 / max(abs(_gt090), 1e-9)

# ── Compare with the existing δ identified-set results (Section 17) ───────
# Pull results from the two attractor endpoints that were already computed.
# E2E attractor  ≈ delta_mdp_e2e_mean  (from Section 10 main runs)
# Blend attractor ≈ delta_mdp_mean     (from Section 10 main runs)
# The full grid in Section 17 covers [0.50, 0.90]; retrieve 0.90 from there.
if DELTA_090 in _welf_agg_by_delta:
    _sec17_mdp_err090 = abs(
        _welf_agg_by_delta[DELTA_090]["cv_mdp"]["mean"]
        - _welf_agg_by_delta[DELTA_090]["cv_gt"]["mean"])
    _sec17_note = (f"Section 17 (same δ=0.90): |CV error|={_sec17_mdp_err090:.4f}")
else:
    _sec17_note = "Section 17 did not include δ=0.90 in its grid."

# ── Console report ────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print(f"  WELFARE AT δ = {DELTA_090}  (Habit DGP, 10% ibuprofen shock)")
print("-" * 72)
print(f"  {'Model':<25}  {'CV (£)':>10}  {'|error| (£)':>12}  {'error %':>9}")
for _lbl, _cv, _err, _pct in [
    ("Ground Truth",         _gt090,   0.0,         0.0),
    ("LA-AIDS (static)",     _aids090, _err_aids090, _pct_aids090),
    ("Neural IRL (static)",  _nirl090, _err_nirl090, _pct_nirl090),
    ("MDP Neural IRL",       _mdp090,  _err_mdp090,  _pct_mdp090),
    ("MDP IRL (E2E δ)",      _e2e090,  _err_e2e090,  _pct_e2e090),
]:
    print(f"  {_lbl:<25}  {_cv:>10.4f}  {_err:>12.4f}  {_pct:>8.1f}%")

print(f"\n  ({_sec17_note})")

# ── Verdict ───────────────────────────────────────────────────────────────
_NEAR_ZERO_THRESH = 2.0   # % — anything below 2% counts as "near zero"
print("\n  ROBUSTNESS VERDICT:")
if _pct_mdp090 < _NEAR_ZERO_THRESH:
    print(f"  ✓  MDP CV error at δ=0.90 is {_pct_mdp090:.2f}% < {_NEAR_ZERO_THRESH}% threshold.")
    print(f"     Welfare conclusion is ROBUST across the full attractor range")
    print(f"     [{min(delta_mdp_e2e_mean, delta_mdp_mean):.2f}, {DELTA_090:.2f}].")
    print(f"     This is the strongest possible result — δ non-identification")
    print(f"     does NOT materially affect welfare inference.")
else:
    print(f"  ⚠  MDP CV error at δ=0.90 is {_pct_mdp090:.2f}% ≥ {_NEAR_ZERO_THRESH}% threshold.")
    print(f"     Welfare claims should be BOUNDED to the E2E attractor range")
    print(f"     [δ̂_E2E ≈ {delta_mdp_e2e_mean:.2f}, δ̂_blend ≈ {delta_mdp_mean:.2f}].")
    print(f"     Report Section 17 CV bounds rather than a point estimate.")

if _pct_e2e090 < _NEAR_ZERO_THRESH:
    print(f"\n  ✓  E2E CV error at δ=0.90 is also {_pct_e2e090:.2f}% — robustness confirmed")
    print(f"     for the end-to-end model as well.")

# ── Figure 15: bar chart comparing δ=0.90 welfare errors to the
#    attractor endpoints from Section 17 ──────────────────────────────────
# Select the two attractor δ values closest to what the main runs converged to.
_att_e2e_key   = min(_welf_agg_by_delta.keys(),
                     key=lambda d: abs(d - delta_mdp_e2e_mean))
_att_blend_key = min(_welf_agg_by_delta.keys(),
                     key=lambda d: abs(d - delta_mdp_mean))

def _pct_err(d_key, model_key):
    _cv  = _welf_agg_by_delta[d_key][model_key]["mean"]
    _gt  = _welf_agg_by_delta[d_key]["cv_gt"]["mean"]
    return 100 * abs(_cv - _gt) / max(abs(_gt), 1e-9)

_fig15_deltas  = [_att_e2e_key, _att_blend_key, DELTA_090]
_fig15_labels  = [
    rf"$\delta_{{E2E}}$={_att_e2e_key:.2f}",
    rf"$\delta_{{blend}}$={_att_blend_key:.2f}",
    rf"$\delta$={DELTA_090:.2f}  (upper bound)",
]
_fig15_colors  = ["#FF6F00", "#00897B", "#C62828"]

_fig15_models  = [
    ("LA-AIDS",           "cv_aids",  "#EF9A9A"),
    ("Neural IRL",        "cv_nirl",  "#90CAF9"),
    ("MDP Neural IRL",    "cv_mdp",   "#80CBC4"),
    ("MDP IRL (E2E)",     "cv_e2e",   "#FFCC80"),
]

fig15, ax15 = plt.subplots(figsize=(11, 6))
_x15   = np.arange(len(_fig15_models))
_w15   = 0.22
_offsets = np.linspace(-(len(_fig15_deltas)-1)/2 * _w15,
                        (len(_fig15_deltas)-1)/2 * _w15,
                        len(_fig15_deltas))

for _oi, (_dk, _dlbl, _dcol) in enumerate(
        zip(_fig15_deltas, _fig15_labels, _fig15_colors)):

    _errs = []
    for _modnm, _modkey, _ in _fig15_models:
        if _dk == DELTA_090:
            # Use the freshly computed Section 20 results
            _cvs_090 = [r[_modkey] for r in _rows_090]
            _gt_vals  = [r["cv_gt"] for r in _rows_090]
            _errs.append(100 * np.mean([abs(c - g) for c, g in
                                        zip(_cvs_090, _gt_vals)])
                         / max(abs(np.mean(_gt_vals)), 1e-9))
        else:
            _errs.append(_pct_err(_dk, _modkey))

    ax15.bar(_x15 + _offsets[_oi], _errs, _w15,
             color=_dcol, label=_dlbl, edgecolor="k", lw=0.8, alpha=0.88)

ax15.axhline(_NEAR_ZERO_THRESH, color="k", ls="--", lw=1.5,
             label=f"{_NEAR_ZERO_THRESH}% robustness threshold")
ax15.set_xticks(_x15)
ax15.set_xticklabels([m[0] for m in _fig15_models], fontsize=12)
ax15.set_ylabel("CV error  (% deviation from ground truth)", fontsize=12)
ax15.set_title(
    r"Welfare Robustness at $\delta = 0.90$ vs Attractor Endpoints"
    "\nHabit DGP · 10% ibuprofen shock · structural CV",
    fontsize=12, fontweight="bold")
ax15.legend(fontsize=10, loc="upper left")
ax15.grid(axis="y", alpha=0.3)
fig15.tight_layout()
fig15.savefig("figures/fig15_welfare_delta090.pdf", dpi=150, bbox_inches="tight")
fig15.savefig("figures/fig15_welfare_delta090.png", dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig15_welfare_delta090.pdf")
plt.close(fig15)

print("\n  All figures saved to figures/")
print("  Done (Sections 19-20).")


# ════════════════════════════════════════════════════════════════════
#  SECTION 21: FROZEN-δ GRID IDENTIFICATION TEST
#  Stage 1 — Train with frozen δ
#    For each δ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, build an
#    MDP-IRL E2E model, *freeze* log_delta so only the network
#    weights ψ and temperature β are updated, then train to
#    convergence on the training set.
#
#  Stage 2 — Evaluate on held-out test data
#    Score each frozen network on an independent test set generated
#    with the *true* δ = 0.7.  Because the network was never exposed
#    to this data it cannot re-absorb δ-misspecification, so
#    KL_test(δ) has a sharper minimum than Fig 11's KL_train curve.
#
#  Output: Figure 16 — overlays KL_test(δ) and KL_train(δ) on the
#    continuous Fig 11 background for direct identification comparison.
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 21 — FROZEN-δ GRID IDENTIFICATION TEST (STAGE 1 + 2)")
print("=" * 72)

FROZEN_DELTA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_FROZEN_SEEDS    = min(N_RUNS, 3)   # seeds per grid point; 3 gives tight SEs
FROZEN_EPOCHS     = 10_000

print(f"\n  δ grid : {FROZEN_DELTA_GRID}")
print(f"  Seeds  : {N_FROZEN_SEEDS}  |  Epochs : {FROZEN_EPOCHS}")
print(f"  True δ : {TRUE_DELTA}")


def run_frozen_delta_seed(seed: int, delta_fixed: float,
                          epochs: int = 10_000) -> dict:
    """Train MDP-E2E with δ *frozen* at delta_fixed; score on held-out test.

    Stage 1
    -------
    Generate N training observations with the true DGP (HabitFormationConsumer,
    δ=0.7).  Build MDPNeuralIRL_E2E with delta_init=delta_fixed, then call
    ``model.log_delta.requires_grad_(False)`` so the optimiser only updates
    the network weights ψ and the temperature β.

    Stage 2
    -------
    Draw an *independent* test set of the same size (different RNG state).
    Compute xbar on the test set using the same frozen δ, then evaluate
    KL(w_pred, w_true) on the test data.  Also record KL on the training
    data for a direct Fig-11 comparison.

    Parameters
    ----------
    seed        : RNG seed.
    delta_fixed : δ value at which the model is frozen.
    epochs      : training epochs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Training data (true DGP: δ = 0.7) ─────────────────────────────────
    N = N_OBS
    Z_tr     = np.random.uniform(1, 5, (N, 3))
    p_tr     = Z_tr + np.random.normal(0, 0.1, (N, 3))
    inc_tr   = np.random.uniform(1200, 2000, N)
    hc_true  = HabitFormationConsumer()            # decay=0.7, theta=0.3
    w_tr, _  = hc_true.solve_demand(p_tr, inc_tr, return_xbar=True)
    q_tr     = w_tr * inc_tr[:, None] / np.maximum(p_tr, 1e-8)
    lq_tr    = np.log(np.maximum(q_tr, 1e-6))     # (N, G) log-quantities

    # ── Held-out test data (independent draw, same DGP) ───────────────────
    Z_te     = np.random.uniform(1, 5, (N, 3))
    p_te     = Z_te + np.random.normal(0, 0.1, (N, 3))
    inc_te   = np.random.uniform(1200, 2000, N)
    w_te, _  = hc_true.solve_demand(p_te, inc_te, return_xbar=True)
    q_te     = w_te * inc_te[:, None] / np.maximum(p_te, 1e-8)
    lq_te    = np.log(np.maximum(q_te, 1e-6))     # (N, G) log-quantities

    # ── Stage 1: build model and freeze δ ─────────────────────────────────
    model = MDPNeuralIRL_E2E(n_goods=3, hidden_dim=256, delta_init=delta_fixed)
    # Freeze the habit-decay parameter — only ψ (network weights) and β update
    model.log_delta.requires_grad_(False)

    model, _ = train_mdp_e2e(
        model, p_tr, inc_tr, w_tr, lq_tr,
        store_ids=None, epochs=epochs, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xbar_recompute_every=10, device=DEVICE,
        tag=f"frozen-d{delta_fixed:.1f}-s{seed}")

    # ── Stage 2: evaluate KL on train and test ─────────────────────────────
    _dt = torch.tensor(float(delta_fixed), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # — test KL —
        lq_te_t  = torch.tensor(lq_te, dtype=torch.float32, device=DEVICE)
        xb_te    = compute_xbar_e2e(_dt, lq_te_t, store_ids=None).cpu().numpy()
        w_pred_te = predict_shares("mdp-e2e", p_te, inc_te,
                                   mdp_e2e=model, xbar_e2e=xb_te, device=DEVICE)
        kl_test  = kl_div(w_pred_te, w_te)

        # — train KL (for Fig-11 comparison) —
        lq_tr_t  = torch.tensor(lq_tr, dtype=torch.float32, device=DEVICE)
        xb_tr    = compute_xbar_e2e(_dt, lq_tr_t, store_ids=None).cpu().numpy()
        w_pred_tr = predict_shares("mdp-e2e", p_tr, inc_tr,
                                   mdp_e2e=model, xbar_e2e=xb_tr, device=DEVICE)
        kl_train = kl_div(w_pred_tr, w_tr)

    return {
        "delta_fixed": delta_fixed,
        "seed":        seed,
        "kl_test":     float(kl_test),
        "kl_train":    float(kl_train),
    }


# ── Run Stage 1 + 2 over the grid ─────────────────────────────────────────
_frozen_rows = {dv: [] for dv in FROZEN_DELTA_GRID}
for _dv in FROZEN_DELTA_GRID:
    print(f"\n  δ = {_dv:.1f}", end="", flush=True)
    for _si in range(N_FROZEN_SEEDS):
        _seed_fz = 700 + _si * 41
        import time as _tfz_mod; _tfz0 = _tfz_mod.time()
        _rfz = run_frozen_delta_seed(_seed_fz, _dv, epochs=FROZEN_EPOCHS)
        print(f"  [seed={_seed_fz} "
              f"KL_test={_rfz['kl_test']:.5f} "
              f"KL_train={_rfz['kl_train']:.5f} "
              f"({_tfz_mod.time()-_tfz0:.0f}s)]",
              end="", flush=True)
        _frozen_rows[_dv].append(_rfz)
    print()

# ── Aggregate: mean ± SE across seeds ─────────────────────────────────────
_frozen_agg = {}   # {delta: {"kl_test": {"mean", "se"}, "kl_train": {"mean", "se"}}}
for _dv, _rows in _frozen_rows.items():
    _n_s = len(_rows)
    _kl_test_vals  = [r["kl_test"]  for r in _rows]
    _kl_train_vals = [r["kl_train"] for r in _rows]
    _frozen_agg[_dv] = {
        "kl_test": {
            "mean": float(np.mean(_kl_test_vals)),
            "se":   (_se(_kl_test_vals)  if _n_s > 1 else 0.0),
        },
        "kl_train": {
            "mean": float(np.mean(_kl_train_vals)),
            "se":   (_se(_kl_train_vals) if _n_s > 1 else 0.0),
        },
    }

_fdg_arr    = np.array(FROZEN_DELTA_GRID)
_kl_te_mean = np.array([_frozen_agg[d]["kl_test"]["mean"]  for d in FROZEN_DELTA_GRID])
_kl_te_se   = np.array([_frozen_agg[d]["kl_test"]["se"]    for d in FROZEN_DELTA_GRID])
_kl_tr_mean = np.array([_frozen_agg[d]["kl_train"]["mean"] for d in FROZEN_DELTA_GRID])
_kl_tr_se   = np.array([_frozen_agg[d]["kl_train"]["se"]   for d in FROZEN_DELTA_GRID])

# ── Figure 16: Frozen-δ KL profile with Fig-11 background ─────────────────
# Layout: single axes.
# Background (thin, grey):  continuous Fig-11 E2E KL_train curve (joint optimisation).
# Foreground:
#   Orange circles + error bars : KL_train for frozen-δ models at each grid point.
#   Blue  squares + error bars  : KL_test  for frozen-δ models at each grid point.
# Vertical lines: true δ=0.7 (black dashed) and argmin KL_test (blue dotted).

_col_test  = "#1565C0"   # deep blue  — held-out test KL
_col_train = "#E65100"   # deep orange — training KL (frozen δ)
_col_bg    = "#BDBDBD"   # grey        — Fig-11 continuous background

fig16, ax16 = plt.subplots(figsize=(10, 5))

# Background: Fig-11 continuous E2E KL_train curve (joint optimisation)
ax16.plot(kl_delta_grid, kl_prof_e2e_mean,
          color=_col_bg, lw=1.5, ls="-", alpha=0.7,
          label=r"Fig 11: KL$_{\rm train}$ (joint $\hat{\delta}$+$\hat{\beta}$, continuous sweep)")
if N_RUNS > 1:
    ax16.fill_between(kl_delta_grid,
                      kl_prof_e2e_mean - kl_prof_e2e_se,
                      kl_prof_e2e_mean + kl_prof_e2e_se,
                      color=_col_bg, alpha=0.20)

# Frozen-δ training KL (grid points)
ax16.errorbar(_fdg_arr, _kl_tr_mean, yerr=_kl_tr_se,
              fmt="o-", color=_col_train, lw=2.0, ms=8, capsize=5,
              label=r"KL$_{\rm train}$ — frozen $\delta$, ψ+β adapted"
                    + f"  ({N_FROZEN_SEEDS} seeds)")

# Frozen-δ test KL (grid points) — the key identification curve
ax16.errorbar(_fdg_arr, _kl_te_mean, yerr=_kl_te_se,
              fmt="s-", color=_col_test, lw=2.5, ms=9, capsize=5,
              label=r"KL$_{\rm test}$ — frozen $\delta$, held-out data"
                    + f"  ({N_FROZEN_SEEDS} seeds)")

# True δ
ax16.axvline(TRUE_DELTA, color="k", ls="--", lw=2.0,
             label=f"True δ = {TRUE_DELTA}")

# Argmin of test KL
_argmin_te   = int(np.argmin(_kl_te_mean))
_delta_te_min = FROZEN_DELTA_GRID[_argmin_te]
ax16.axvline(_delta_te_min, color=_col_test, ls=":", lw=2.0,
             label=rf"argmin KL$_{{test}}$ = {_delta_te_min:.1f}")

ax16.set_xlabel(r"Frozen habit-decay parameter $\delta$", fontsize=14)
ax16.set_ylabel("KL divergence", fontsize=14)
ax16.legend(fontsize=10, loc="upper left")
ax16.grid(True, alpha=0.3)
ax16.set_xticks(_fdg_arr)

_se_note = f"  (error bars = ±1 SE, {N_FROZEN_SEEDS} seeds)" if N_FROZEN_SEEDS > 1 else ""
fig16.suptitle(
    r"Frozen-$\delta$ Identification Test — KL on Held-Out vs Training Data"
    "\nStage 1: train ψ+β with δ frozen  ·  "
    "Stage 2: evaluate on independent test set"
    f"\nTrue δ = {TRUE_DELTA}  ·  Habit DGP  ·  N={N_OBS}{_se_note}",
    fontsize=12, fontweight="bold")
fig16.tight_layout()
fig16.savefig("figures/fig16_frozen_delta_kl_profile.pdf", dpi=150,
              bbox_inches="tight")
fig16.savefig("figures/fig16_frozen_delta_kl_profile.png", dpi=150,
              bbox_inches="tight")
print("\n    Saved: figures/fig16_frozen_delta_kl_profile.pdf")
plt.close(fig16)

# ── Console summary ────────────────────────────────────────────────────────
print("\n  SECTION 21 — FROZEN-δ IDENTIFICATION SUMMARY")
print(f"  True δ = {TRUE_DELTA}  |  {N_FROZEN_SEEDS} seeds × {FROZEN_EPOCHS} epochs")
print(f"  {'δ_fixed':>8}  {'KL_train (mean)':>17}  {'KL_train SE':>13}"
      f"  {'KL_test (mean)':>16}  {'KL_test SE':>12}")
for _dv in FROZEN_DELTA_GRID:
    _a = _frozen_agg[_dv]
    print(f"  {_dv:>8.1f}  "
          f"{_a['kl_train']['mean']:>17.6f}  "
          f"{_a['kl_train']['se']:>13.6f}  "
          f"{_a['kl_test']['mean']:>16.6f}  "
          f"{_a['kl_test']['se']:>12.6f}")

# Identification verdict
_bias_te = abs(_delta_te_min - TRUE_DELTA)
print(f"\n  KL_test minimised at δ̂ = {_delta_te_min:.1f}  "
      f"(bias vs truth = {_bias_te:.1f})")
if _bias_te < 0.15:
    print("  ✓ Held-out KL identifies δ within one grid step of the truth.")
    print("    The frozen-δ test profile is SHARPER than the joint-optimisation")
    print("    curve in Fig 11 — confirming that joint training absorbs")
    print("    δ-misspecification through the network weights.")
else:
    print("  ⚠ KL_test minimum is not at the true δ on this grid.")
    print("    Consider narrowing the grid or increasing N / epochs.")

# Compare sharpness: coefficient of variation of KL_test vs Fig-11 KL_train
_cv_test   = _kl_te_mean.std() / max(_kl_te_mean.mean(), 1e-9)
# restrict Fig-11 curve to the same δ range as the frozen grid
_fig11_sub = kl_prof_e2e_mean[
    (kl_delta_grid >= min(FROZEN_DELTA_GRID)) &
    (kl_delta_grid <= max(FROZEN_DELTA_GRID))]
_cv_fig11  = _fig11_sub.std() / max(_fig11_sub.mean(), 1e-9)
print(f"\n  Sharpness (CV of KL curve):")
print(f"    KL_test  (frozen δ, held-out)  CV = {_cv_test:.4f}")
print(f"    KL_train (Fig 11, joint opt.)  CV = {_cv_fig11:.4f}")
if _cv_test > _cv_fig11:
    print("  ✓ KL_test is SHARPER than Fig-11 KL_train "
          f"(ratio = {_cv_test / max(_cv_fig11, 1e-9):.2f}×).")
else:
    print("  ⚠ KL_test is not sharper than Fig-11 on this N — "
          "try larger N or more seeds.")

print("\n  All figures saved to figures/")
print("  Done (Sections 19-21).")


# ════════════════════════════════════════════════════════════════════
#  SECTION 22: LINEAR-IN-x̄ REWARD — δ IDENTIFICATION SIMULATION
#
#  Restricts the reward to R_ψ(p, y, x̄) = f_ψ(p, y) + θ · x̄, which
#  makes the identification argument for δ fully transparent:
#
#    For matched pairs (t, t') sharing the same (p, y) but drawn from
#    different positions in the consumption sequence (hence different x̄):
#
#      log w_{j,t} − log w_{j,t'} = θ_j · (x̄_{j,t}(δ) − x̄_{j,t'}(δ))
#
#    because f_ψ(p, y) cancels exactly.  Profiling out θ̂_j(δ) by
#    per-good OLS leaves a normalised residual M(δ) that depends only
#    on δ and achieves its global minimum at the true δ₀.
#    Analogous to a difference-in-differences estimator where the shared
#    price-income environment acts as the "control" stripping out f_ψ.
# ════════════════════════════════════════════════════════════════════


class LinearXbarConsumer:
    """Consumer whose per-good reward is affine in the log habit stock.

    Reward for good j:
        R_j = α_j · log(p_j) + γ · log(y) + θ_j · log x̄_j

    Shares are obtained from the softmax of (R_1, …, R_G).  The habit
    stock evolves in log-quantity space (matching the MDP-E2E convention):

        log x̄_t = δ · log x̄_{t-1} + (1−δ) · log q_{t-1}

    Parameters
    ----------
    alpha  : (G,) price-sensitivity vector (entries typically negative).
    gamma  : common income sensitivity (scalar).
    theta  : (G,) habit-weight vector.
    delta  : habit-decay ∈ (0, 1).
    """
    name = "Linear-Xbar"

    def __init__(self, alpha=None, gamma=-0.5, theta=None, delta=0.7):
        self.alpha = (np.array(alpha) if alpha is not None
                      else np.array([-0.50, -0.60, -0.40]))
        self.gamma = gamma
        self.theta = (np.array(theta) if theta is not None
                      else np.array([0.30, 0.25, 0.35]))
        self.delta = delta

    def solve_demand(self, prices, income, return_xbar=False):
        N, G = prices.shape
        lp  = np.log(np.maximum(prices, 1e-8))   # (N, G)
        ly  = np.log(np.maximum(income, 1e-8))   # (N,)
        lxbar  = lp.mean(0)                       # (G,) initial log habit
        shares = np.zeros((N, G))
        xbars  = np.zeros((N, G))

        for i in range(N):
            xbars[i] = lxbar
            R   = self.alpha * lp[i] + self.gamma * ly[i] + self.theta * lxbar
            ew  = np.exp(R - R.max())
            shares[i] = ew / ew.sum()
            q_i   = shares[i] * income[i] / np.maximum(prices[i], 1e-8)
            lxbar = (self.delta * lxbar
                     + (1.0 - self.delta) * np.log(np.maximum(q_i, 1e-6)))

        if return_xbar:
            return np.clip(shares, 1e-6, 1.0), xbars
        return np.clip(shares, 1e-6, 1.0)


def _xbar_sweep_np(lq: np.ndarray, delta: float,
                   store_ids=None) -> np.ndarray:
    """Compute (N, G) log-habit-stock for a given scalar delta (no gradient).

    Matches compute_xbar_e2e:  x̄_t = δ · x̄_{t-1} + (1−δ) · q_{t-1}.

    Parameters
    ----------
    lq        : (N, G) log-quantities (or log-shares) in sequential order.
    delta     : habit-decay scalar ∈ (0, 1).
    store_ids : (N,) int array or None — resets x̄ at store boundaries.
    """
    N, G = lq.shape
    xb   = np.zeros((N, G))
    prev = lq.mean(0)
    for i in range(N):
        if store_ids is not None and i > 0 and store_ids[i] != store_ids[i - 1]:
            prev = lq.mean(0)
        xb[i] = prev
        prev  = delta * prev + (1.0 - delta) * lq[i]
    return xb


def run_linear_xbar_id_seed(seed: int,
                             true_delta: float = 0.7,
                             n_grid: int = 80,
                             max_pairs: int = 3000,
                             n_bins: int = 4) -> dict:
    """Linear-in-x̄ identification simulation for one random seed.

    DGP
    ---
    Shares are generated by LinearXbarConsumer(delta=true_delta).
    The per-good reward is
        R_j(p, y, x̄) = f_j(p, y) + θ_j · log x̄_j
    so for any two observations (t, t') with the same (p, y):
        Δlog w_j ≡ log w_{j,t} − log w_{j,t'} = θ_j · Δx̄_j(δ)
    The f_j term cancels exactly.

    Identification curve
    --------------------
    For each δ in a fine grid:
      1. Compute x̄(δ) for all N observations.
      2. Form ΔlogW (P×G) and Δx̄(δ) (P×G) for matched pairs.
      3. Per-good OLS: θ̂_j(δ) = (ΔlogW_j · Δx̄_j(δ)) / ‖Δx̄_j(δ)‖²
      4. Normalised residual: M(δ) = Σ_j ‖ΔlogW_j − θ̂_j Δx̄_j(δ)‖² / ‖ΔlogW_j‖²

    M(δ) has its global minimum at δ = true_delta (identification result).

    Returns
    -------
    dict : delta_grid, residuals, corr_moments, n_pairs, true_delta,
           argmin_delta, argmin_corr, DW, DXbar_true, DXbar_wrong.
    """
    np.random.seed(seed)
    N = N_OBS
    Z = np.random.uniform(1, 5, (N, 3))
    p = Z + np.random.normal(0, 0.1, (N, 3))
    y = np.random.uniform(1200, 2000, N)

    consumer = LinearXbarConsumer(delta=true_delta)
    w, _     = consumer.solve_demand(p, y, return_xbar=True)

    lw = np.log(np.maximum(w, 1e-8))          # (N, G) log shares
    lp = np.log(np.maximum(p, 1e-8))          # (N, G) log prices
    ly = np.log(np.maximum(y, 1e-8))          # (N,)   log income
    q  = w * y[:, None] / np.maximum(p, 1e-8)
    lq = np.log(np.maximum(q, 1e-6))          # (N, G) log quantities → xbar input

    # ── Matched pairs: quantile-bin (lp, ly) space ───────────────────────
    def _qbin(arr, B):
        """Rank-based quantile discretisation into B bins (0…B-1)."""
        pcts = np.percentile(arr, np.linspace(0, 100, B + 1))
        return np.searchsorted(pcts[1:-1], arr)

    cid = (_qbin(lp[:, 0], n_bins) * n_bins ** 3
           + _qbin(lp[:, 1], n_bins) * n_bins ** 2
           + _qbin(lp[:, 2], n_bins) * n_bins
           + _qbin(ly,       n_bins))

    pairs = []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        if len(idx) >= 2:
            for ii in range(len(idx)):
                for jj in range(ii + 1, len(idx)):
                    pairs.append((idx[ii], idx[jj]))

    rng = np.random.default_rng(seed + 999)
    if len(pairs) > max_pairs:
        sel   = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]

    pairs = np.array(pairs, dtype=int)   # (P, 2)
    P     = len(pairs)

    _nan_dict = {
        "error": "too few matched pairs", "n_pairs": P,
        "delta_grid": np.linspace(0.05, 0.98, n_grid),
        "residuals":  np.full(n_grid, np.nan),
        "corr_moments": np.full(n_grid, np.nan),
        "true_delta": true_delta,
        "argmin_delta": np.nan, "argmin_corr": np.nan,
        "DW": None, "DXbar_true": None, "DXbar_wrong": None,
    }
    if P < 20:
        return _nan_dict

    DW = lw[pairs[:, 0]] - lw[pairs[:, 1]]   # (P, G) observed log-share diffs

    # ── Identification sweep over δ ──────────────────────────────────────
    delta_grid   = np.linspace(0.05, 0.98, n_grid)
    _step        = delta_grid[1] - delta_grid[0]
    residuals    = np.zeros(n_grid)
    corr_moments = np.zeros(n_grid)

    DXbar_true  = None   # saved at grid point closest to true_delta
    DXbar_wrong = None   # saved at δ ≈ 0.30 (clearly wrong value)

    for d_idx, d_test in enumerate(delta_grid):
        lxb   = _xbar_sweep_np(lq, d_test)            # (N, G)
        DXbar = lxb[pairs[:, 0]] - lxb[pairs[:, 1]]   # (P, G)

        res_tot = 0.0
        cor_tot = 0.0
        for g in range(3):
            dw = DW[:, g];  dx = DXbar[:, g]
            ss = np.dot(dx, dx)
            if ss < 1e-12:
                res_tot += 1.0;  cor_tot += 1.0
                continue
            th    = np.dot(dw, dx) / ss
            resid = dw - th * dx
            ss_dw = max(np.dot(dw, dw), 1e-12)
            res_tot += np.dot(resid, resid) / ss_dw
            if dw.std() > 1e-8 and dx.std() > 1e-8:
                cor_tot += 1.0 - abs(float(np.corrcoef(dw, dx)[0, 1]))
            else:
                cor_tot += 1.0

        residuals[d_idx]    = res_tot / 3.0
        corr_moments[d_idx] = cor_tot / 3.0

        # Save scatter data near the true δ and at a clearly wrong δ
        if DXbar_true is None and d_test >= true_delta - _step:
            DXbar_true = DXbar.copy()
        if DXbar_wrong is None and d_test >= 0.29:
            DXbar_wrong = DXbar.copy()

    return {
        "delta_grid":   delta_grid,
        "residuals":    residuals,
        "corr_moments": corr_moments,
        "n_pairs":      P,
        "true_delta":   true_delta,
        "argmin_delta": float(delta_grid[np.argmin(residuals)]),
        "argmin_corr":  float(delta_grid[np.argmin(corr_moments)]),
        "DW":           DW,
        "DXbar_true":   DXbar_true,
        "DXbar_wrong":  DXbar_wrong,
    }


# ════════════════════════════════════════════════════════════════════
#  SECTION 22 EXECUTION
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 22 — LINEAR-IN-x̄ REWARD: δ IDENTIFICATION SIMULATION")
print("=" * 72)
print(f"\n  DGP: R_j(p, y, x̄) = f_j(p, y) + θ_j · x̄_j  (linear in x̄)")
print(f"  Identification: matched pairs with same (p, y) cancel f_j exactly")
print(f"  Profiled OLS residual M(δ) has unique minimum at true δ = {TRUE_DELTA}")

_LX_N_SEEDS = min(N_RUNS, 3)
_LX_SEEDS   = [42 + i * 23 for i in range(_LX_N_SEEDS)]
print(f"\n  Seeds: {_LX_SEEDS}  |  N_obs = {N_OBS}")

_lx_rows = []
for _s22 in _LX_SEEDS:
    _t22 = time.time()
    _r22 = run_linear_xbar_id_seed(_s22, true_delta=TRUE_DELTA)
    _lx_rows.append(_r22)
    print(f"  seed={_s22}  n_pairs={_r22['n_pairs']:4d}  "
          f"argmin δ̂={_r22['argmin_delta']:.3f}  "
          f"argmin |r|={_r22['argmin_corr']:.3f}  "
          f"({time.time()-_t22:.1f}s)")

# Aggregate across seeds
_lx_grid     = _lx_rows[0]["delta_grid"]
_lx_res_arr  = np.stack([r["residuals"]    for r in _lx_rows], 0)  # (S, n_grid)
_lx_corr_arr = np.stack([r["corr_moments"] for r in _lx_rows], 0)

_lx_res_mean  = np.nanmean(_lx_res_arr,  0)
_lx_res_se    = (np.nanstd(_lx_res_arr,  0, ddof=min(1, _LX_N_SEEDS - 1))
                 / np.sqrt(_LX_N_SEEDS))
_lx_corr_mean = np.nanmean(_lx_corr_arr, 0)
_lx_corr_se   = (np.nanstd(_lx_corr_arr, 0, ddof=min(1, _LX_N_SEEDS - 1))
                 / np.sqrt(_LX_N_SEEDS))

_lx_argmin_vals = [r["argmin_delta"] for r in _lx_rows]
_lx_argmin_mean = float(np.nanmean(_lx_argmin_vals))
_lx_argmin_se   = _se(_lx_argmin_vals)

print(f"\n  Pooled argmin δ̂ = {_lx_argmin_mean:.3f} ± {_lx_argmin_se:.3f}  "
      f"(true δ = {TRUE_DELTA})")
_lx_bias = abs(_lx_argmin_mean - TRUE_DELTA)
if _lx_bias < 0.10:
    print(f"  ✓ Identification successful: bias = {_lx_bias:.3f} < 0.10")
else:
    print(f"  ⚠ Bias = {_lx_bias:.3f} — consider more observations or finer grid")

# ── Figure 17: Identification residual curve + scatter panels ────────────
_r22_last    = _lx_rows[-1]   # representative seed for scatter visualisation
_GOOD_COLS17 = ["#2196F3", "#4CAF50", "#FF5722"]
_GOOD_LBLS17 = ["Aspirin", "Acetaminophen", "Ibuprofen"]

fig17, axes17 = plt.subplots(1, 3, figsize=(17, 5))

# ── Panel A: M(δ) identification residual curve ──────────────────────────
ax17a = axes17[0]
ax17a.plot(_lx_grid, _lx_res_mean, color="#1565C0", lw=2.5,
           label=rf"OLS residual $M(\delta)$  ({_LX_N_SEEDS} seeds)")
if _LX_N_SEEDS > 1:
    ax17a.fill_between(_lx_grid,
                       _lx_res_mean - _lx_res_se,
                       _lx_res_mean + _lx_res_se,
                       color="#1565C0", alpha=0.18)
ax17a.plot(_lx_grid, _lx_corr_mean, color="#E65100", lw=1.8, ls="--",
           label=r"$1 - |\mathrm{corr}|$ moment")
if _LX_N_SEEDS > 1:
    ax17a.fill_between(_lx_grid,
                       _lx_corr_mean - _lx_corr_se,
                       _lx_corr_mean + _lx_corr_se,
                       color="#E65100", alpha=0.12)
ax17a.axvline(TRUE_DELTA, color="k", ls="--", lw=2.0,
              label=rf"True $\delta = {TRUE_DELTA}$")
ax17a.axvline(_lx_argmin_mean, color="#1565C0", ls=":", lw=2.0,
              label=rf"$\hat{{\delta}} = {_lx_argmin_mean:.3f}$")
ax17a.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=12)
ax17a.set_ylabel(r"Normalised residual $M(\delta)$", fontsize=12)
ax17a.set_title(
    "Panel A: δ Identification Curve\n"
    r"(profiled OLS residual over matched pairs)",
    fontsize=11, fontweight="bold")
ax17a.legend(fontsize=9)
ax17a.grid(True, alpha=0.3)

# ── Panels B & C: scatter Δlog w vs Δx̄ at true δ and wrong δ ────────────
for ax17_, DXbar_, panel_lbl, delta_lbl in [
    (axes17[1], _r22_last.get("DXbar_true"),
     "Panel B: True δ",
     f"δ = {TRUE_DELTA} (true)"),
    (axes17[2], _r22_last.get("DXbar_wrong"),
     "Panel C: Wrong δ",
     "δ = 0.30 (wrong)"),
]:
    if DXbar_ is None or _r22_last.get("DW") is None:
        ax17_.set_visible(False)
        continue
    for g in range(3):
        dw_g = _r22_last["DW"][:, g]
        dx_g = DXbar_[:, g]
        ax17_.scatter(dx_g, dw_g, s=6, alpha=0.30,
                      color=_GOOD_COLS17[g], label=_GOOD_LBLS17[g])
        # OLS fit line
        ss = np.dot(dx_g, dx_g)
        if ss > 1e-12:
            th_g = np.dot(dw_g, dx_g) / ss
            xr   = np.array([dx_g.min(), dx_g.max()])
            ax17_.plot(xr, th_g * xr, color=_GOOD_COLS17[g], lw=1.8, alpha=0.85)
    ax17_.set_xlabel(r"$\Delta\bar{x}_j(\delta)$  (habit-stock difference)",
                     fontsize=11)
    ax17_.set_ylabel(r"$\Delta\log w_j$  (log-share difference)", fontsize=11)
    ax17_.set_title(
        f"{panel_lbl}: Δlog w  vs  Δx̄\n({delta_lbl})",
        fontsize=11, fontweight="bold")
    ax17_.legend(fontsize=9, markerscale=3)
    ax17_.grid(True, alpha=0.3)

fig17.suptitle(
    r"Section 22 — Linear-in-$\bar{x}$ Reward: $\delta$ Identification"
    "\n"
    r"$R_\psi(p, y, \bar{x}) = f_\psi(p, y) + \theta \cdot \bar{x}$"
    r" — matched pairs (same $p, y$) cancel $f_\psi$, "
    r"leaving $\delta$ as the only free parameter"
    f"\nTrue δ = {TRUE_DELTA}  ·  N = {N_OBS}  ·  "
    f"{_lx_rows[-1]['n_pairs']} matched pairs  ·  {_LX_N_SEEDS} seeds",
    fontsize=12, fontweight="bold")
fig17.tight_layout()
fig17.savefig("figures/fig17_linear_xbar_identification.pdf",
              dpi=150, bbox_inches="tight")
fig17.savefig("figures/fig17_linear_xbar_identification.png",
              dpi=150, bbox_inches="tight")
print("\n    Saved: figures/fig17_linear_xbar_identification.pdf")
plt.close(fig17)

print(f"\n  Done (Section 22).  argmin δ̂ = {_lx_argmin_mean:.3f} ± {_lx_argmin_se:.3f}"
      f"  (true δ = {TRUE_DELTA},  bias = {_lx_bias:.3f})")
