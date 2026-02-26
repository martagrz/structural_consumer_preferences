"""
experiments/simulation/utils.py
================================
Shared prediction, evaluation utilities, and the frozen-δ grid sweep helper
used across all simulation experiment modules.

Extracted from main_multiple_runs.py  (Sections 7 & δ-grid helper).
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.simulation import (
    MDPNeuralIRL_E2E,
    compute_xbar_e2e,
    train_mdp_e2e,
)

# ─────────────────────────────────────────────────────────────────────────────
#  FIXED PRICE GRID / DEFAULT INCOME
# ─────────────────────────────────────────────────────────────────────────────

P_GRID = np.linspace(1, 10, 80)
AVG_Y  = 1600.0

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BAND = 0.15   # alpha for ±1 SE shaded bands

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
    "MDP E2E (β=1)":     dict(color="#E53935",    ls=":",  lw=1.8),
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7: UNIFIED PREDICTION & EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def predict_shares(spec, p, y, *, aids=None, blp=None, quaids=None, series=None,
                   lirl_theta=None,
                   lirl_feat_fn=None, nirl=None, mdp_nirl=None,
                   xbar=None, q_prev=None, consumer=None, mixture=None,
                   mdp_e2e=None, xbar_e2e=None,
                   mdp_e2e_fb=None, xbar_e2e_fb=None,
                   wirl=None, wirl_log_p_hist=None, wirl_log_q_hist=None,
                   wirl_window=4,
                   nirl_cf=None, mdp_nirl_cf=None,
                   device="cpu"):
    from src.models.simulation import predict_linear_irl
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
    if spec == "l-irl":   return predict_linear_irl(lirl_feat_fn(p, y), lirl_theta)
    if spec == "n-irl":
        with torch.no_grad():
            lp = torch.log(torch.tensor(p, dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(y, dtype=torch.float32, device=device)).unsqueeze(1)
            return nirl(lp, ly).cpu().numpy()
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
    if spec == "mdp-e2e-fb":   # fixed-β ablation
        with torch.no_grad():
            lp = torch.log(torch.clamp(torch.tensor(p, dtype=torch.float32, device=device), min=1e-8))
            ly = torch.log(torch.clamp(torch.tensor(y, dtype=torch.float32, device=device), min=1e-8)).unsqueeze(1)
            xb = torch.tensor(xbar_e2e_fb, dtype=torch.float32, device=device)
            return mdp_e2e_fb(lp, ly, xb).cpu().numpy()
    if spec == "mixture": return mixture.predict(p, y)
    # ── Control-function (CF) model specs ──────────────────────────────────
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
    w0  = predict_shares(spec, p_pt.reshape(1, -1), np.array([y_pt]),
                         xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None,
                         q_prev=q_prev_pt.reshape(1, -1) if q_prev_pt is not None else None,
                         **kw)[0]
    eps = []
    for i in range(3):
        p1 = p_pt.copy().reshape(1, -1); p1[0, i] *= (1 + h)
        wp = predict_shares(spec, p1, np.array([y_pt]),
                            xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None,
                            q_prev=q_prev_pt.reshape(1, -1) if q_prev_pt is not None else None,
                            **kw)[0]
        eps.append(((wp[i] - w0[i]) / w0[i]) / h - 1)
    return eps


def compute_full_elasticity_matrix(spec, p_pt, y_pt, h=1e-4,
                                   xbar_pt=None, q_prev_pt=None, **kw):
    """Return (3, 3) demand elasticity matrix.

    eps[i, j] = d log x_j / d log p_i
    Diagonal  = own-price elasticity  (expected < 0).
    Off-diag  = cross-price elasticity (> 0 = substitutes).
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
    path = np.linspace(p0, p1, steps); dp = (p1 - p0) / steps; loss = 0.0
    for t in range(steps):
        w    = predict_shares(spec, path[t:t+1], np.array([y]),
                              xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None,
                              q_prev=q_prev_pt.reshape(1, -1) if q_prev_pt is not None else None,
                              **kw)[0]
        loss -= (w * y / path[t]) @ dp
    return loss


def get_metrics(spec, p_shock, income, w_true, xbar_shock=None, q_prev_shock=None, **kw):
    wp = predict_shares(spec, p_shock, income, xbar=xbar_shock, q_prev=q_prev_shock, **kw)
    return {"RMSE": np.sqrt(mean_squared_error(w_true, wp)),
            "MAE":  mean_absolute_error(w_true, wp)}


def kl_div(w_pred, w_true):
    wp = np.clip(w_pred, 1e-8, 1); wt = np.clip(w_true, 1e-8, 1)
    return float(np.mean(np.sum(wt * np.log(wt / wp), axis=1)))


# ─────────────────────────────────────────────────────────────────────────────
#  δ GRID-SWEEP HELPER
#
#  For each δ in MDP_DELTA_GRID:
#    1. Build MDPNeuralIRL_E2E with log_delta frozen at that value.
#    2. Train the network weights normally.
#    3. Evaluate per-observation KL on the held-out validation set.
#  δ̂ = argmin KL_val.
#  Identified set = {δ : KL_val(δ) ≤ KL_val(δ̂) + se_multiplier × SE(δ̂)}.
# ─────────────────────────────────────────────────────────────────────────────

def fit_mdp_delta_grid(
    p_tr,  y_tr,  w_tr,  lq_tr,
    p_val, y_val, w_val, lq_val,
    delta_grid     = None,
    epochs         = 333,
    device         = "cpu",
    store_ids_tr   = None,
    store_ids_val  = None,
    n_goods        = 3,
    hidden         = 256,
    fixed_beta     = None,   # kept for backward compat, no longer used
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
    mdp_delta_grid = None,   # fallback default grid
):
    """Train one frozen-δ model per grid point; pick δ̂ by minimum hold-out KL.

    Parameters
    ----------
    delta_grid     : 1-D array of candidate δ; if None uses mdp_delta_grid.
    model_class    : class, default MDPNeuralIRL_E2E
    extra_model_kw : extra kwargs for model_class
    pred_spec      : predict_shares spec, default "mdp-e2e"
    pred_model_key : kwarg name for model in predict_shares
    pred_xbar_key  : kwarg name for xbar in predict_shares
    se_multiplier  : identified set threshold c (default 2 ≈ 95% CI)
    mdp_delta_grid : fallback default grid when delta_grid is None

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
        if mdp_delta_grid is None:
            raise ValueError("Either delta_grid or mdp_delta_grid must be provided.")
        delta_grid = mdp_delta_grid
    if model_class is None:
        model_class = MDPNeuralIRL_E2E
    if extra_model_kw is None:
        extra_model_kw = {}

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
        # build model with δ frozen
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

        # validation KL
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

    # point estimate: δ̂ = argmin KL_val
    best_k     = int(np.argmin(kl_grid))
    delta_hat  = float(delta_grid[best_k])
    best_model = all_models[delta_hat]
    best_hist  = all_hists[delta_hat]

    # identified set: KL(δ) ≤ KL(δ̂) + se_multiplier × SE(δ̂)
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
