"""
experiments/dominicks/utils.py
================================
Section 7 of dominicks_multiple_runs.py.

Prediction, evaluation, and welfare utilities for the Dominick's experiments.
All functions accept a config dict (cfg) rather than using globals.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from experiments.dominicks.data import G


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def pred(spec: str, p, y,
         cfg: dict,
         xb_prev=None, q_prev=None,
         store_idx=None,
         s_te_mode_idx: int = 0,
         **kw):
    """Unified prediction helper for all Dominick's model specs.

    For MDP models pass *both* xb_prev (log-normalised habit stock) and
    q_prev (log-normalised previous quantities).
    For store-FE models pass *store_idx* as an integer array.
    When store_idx is None for an FE model the modal test-store index is used.
    """
    dev = cfg['device']

    if spec == 'aids':   return kw['aids'].predict(p, y)
    if spec == 'blp':
        return kw['blp'].predict(p)[:, :G]        # inside market shares (N, G); drop outside-option col
    if spec == 'quaids': return kw['quaids'].predict(p, y)
    if spec == 'series': return kw['series'].predict(p, y)

    if spec == 'window-irl':
        _wm   = kw['wirl']
        _lp_h = kw.get('wirl_log_p_hist', np.zeros(G))
        _lq_h = kw.get('wirl_log_q_hist', np.zeros(G))
        _ww   = kw.get('wirl_window', 4)
        lp_cur = np.log(np.maximum(p, 1e-8))
        ly_cur = np.log(np.maximum(y, 1e-8))
        in_dim = G + 1 + _ww * 2 * G
        feats  = np.zeros((len(p), in_dim), dtype=np.float32)
        hist_f = np.concatenate([_lp_h, _lq_h] * _ww)
        for _i in range(len(p)):
            feats[_i] = np.concatenate([lp_cur[_i], [ly_cur[_i]], hist_f])
        with torch.no_grad():
            xt = torch.tensor(feats, dtype=torch.float32, device=dev)
            return _wm(xt).cpu().numpy()

    if spec == 'lirl':
        from src.models.dominicks import pred_lirl
        return pred_lirl(kw['ff'], kw['theta'], p, y)

    if spec == 'nirl':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            return kw['nirl'](lp, ly).cpu().numpy()

    if spec == 'mdp':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            return kw['mdp'](lp, ly, xbp, qp).cpu().numpy()

    if spec == 'mdp-e2e':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            xb = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            return kw['mdp_e2e'](lp, ly, xb).cpu().numpy()

    # ── Control-function (CF) variants ────────────────────────────────────────
    if spec == 'nirl-cf':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            _m = kw['nirl_cf']
            vh = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, v_hat=vh).cpu().numpy()

    if spec == 'mdp-cf':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _m  = kw['mdp_cf']
            vh  = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, xbp, qp, v_hat=vh).cpu().numpy()

    if spec == 'mdp-fe-cf':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si  = torch.tensor(_si, dtype=torch.long).to(dev)
            _m  = kw['mdp_fe_cf']
            vh  = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, xbp, qp, si, v_hat=vh).cpu().numpy()

    # ── Store-FE variants ─────────────────────────────────────────────────────
    if spec == 'nirl-fe':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['nirl_fe'](lp, ly, si).cpu().numpy()

    if spec == 'mdp-fe':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['mdp_fe'](lp, ly, xbp, qp, si).cpu().numpy()

    if spec == 'mdp-e2e-fe':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            xb = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['mdp_e2e_fe'](lp, ly, xb, si).cpu().numpy()

    raise ValueError(f"Unknown spec: {spec!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  ELASTICITIES
# ─────────────────────────────────────────────────────────────────────────────

def own_elasticity(spec, p0, y0, cfg, xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    w0  = pred(spec, p0[None], np.array([y0]), cfg,
               xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
               q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
    eps = []
    for i in range(G):
        p1 = p0.copy()[None]; p1[0, i] *= 1 + h
        w1 = pred(spec, p1, np.array([y0]), cfg,
                  xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                  q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        eps.append(((w1[i] - w0[i]) / w0[i]) / h - 1)
    return np.array(eps)


def full_elasticity_matrix(spec, p0, y0, cfg, xb_prev0=None, q_prev0=None,
                            h=1e-4, **kw):
    """Return (G, G) matrix of price elasticities.

    eps[i, j] = d log(w_j) / d log(p_i)
    """
    _kw_xb = dict(
        xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
        q_prev =q_prev0[None]  if q_prev0  is not None else None,
    )
    w0  = pred(spec, p0[None], np.array([y0]), cfg, **_kw_xb, **kw)[0]
    eps = np.zeros((G, G))
    for i in range(G):
        p1 = p0.copy()[None]; p1[0, i] *= (1 + h)
        w1 = pred(spec, p1, np.array([y0]), cfg, **_kw_xb, **kw)[0]
        for j in range(G):
            eps[i, j] = ((w1[j] - w0[j]) / max(w0[j], 1e-9)) / h
            if i == j:
                eps[i, j] -= 1.0
    return eps


# ─────────────────────────────────────────────────────────────────────────────
#  WELFARE (COMPENSATING VARIATION)
# ─────────────────────────────────────────────────────────────────────────────

def comp_var(spec, p0, p1, y, cfg, xb_prev0=None, q_prev0=None, **kw):
    cv_steps = cfg['cv_steps']
    path = np.linspace(p0, p1, cv_steps)
    dp   = (p1 - p0) / cv_steps
    cv   = 0.0
    for t in range(cv_steps):
        w   = pred(spec, path[t:t+1], np.array([y]), cfg,
                   xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                   q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        cv -= (w * y / path[t]) @ dp
    return cv


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics(spec, p, y, w_true, cfg, xb_prev=None, q_prev=None, **kw):
    wp = pred(spec, p, y, cfg, xb_prev=xb_prev, q_prev=q_prev, **kw)
    return {'RMSE': np.sqrt(mean_squared_error(w_true, wp)),
            'MAE':  mean_absolute_error(w_true, wp)}


def kl_div(spec, p, y, w_true, cfg, xb_prev=None, q_prev=None, **kw):
    wp = np.clip(pred(spec, p, y, cfg, xb_prev=xb_prev,
                      q_prev=q_prev, **kw), 1e-8, 1.0)
    wt = np.clip(w_true, 1e-8, 1.0)
    return float(np.mean(np.sum(wt * np.log(wt / wp), 1)))


# ─────────────────────────────────────────────────────────────────────────────
#  HABIT-STOCK CONDITIONAL ON PRICE GRID (for smooth demand curves)
# ─────────────────────────────────────────────────────────────────────────────

def mdp_price_cond_habit(p_grid, shock_g: int, p_te, xb_te, qp_te,
                          bandwidth=None):
    """Gaussian-kernel-weighted average habit stock conditional on price grid.

    Returns (xbr, qpr) each of shape (N_grid, G), suitable for _pred kwargs.
    """
    if bandwidth is None:
        bandwidth = 0.45 * np.std(p_te[:, shock_g])
        bandwidth = max(bandwidth, 1e-3)

    xbr = np.zeros((len(p_grid), G))
    qpr = np.zeros((len(p_grid), G))
    for k, pg in enumerate(p_grid):
        dists   = np.abs(p_te[:, shock_g] - pg)
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        weights = weights / (weights.sum() + 1e-12)
        xbr[k]  = (xb_te * weights[:, None]).sum(0)
        qpr[k]  = (qp_te * weights[:, None]).sum(0)
    return xbr, qpr


# ─────────────────────────────────────────────────────────────────────────────
#  FROZEN-δ GRID SWEEP (Dominick's version)
# ─────────────────────────────────────────────────────────────────────────────

MDP_DELTA_GRID_DOM = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def fit_mdp_delta_grid_dom(
    p_tr, y_tr, w_tr, ls_tr,
    p_val, y_val, w_val, ls_val,
    cfg: dict,
    delta_grid=None,
    store_ids_tr=None,
    store_ids_val=None,
    store_idx_tr=None,
    store_idx_val=None,
    hidden=256,
    model_class=None,
    pred_spec=None,
    pred_model_key=None,
    extra_model_kw=None,
    se_multiplier=2.0,
    tag="dom-delta-sweep",
):
    """Train one frozen-δ model per grid point on Dominick's data.

    δ̂ = argmin test-KL; identified set uses ``se_multiplier × SE`` rule.
    """
    from src.models.dominicks import (
        MDPNeuralIRL_E2E,
        compute_xbar_e2e,
        train_mdp_e2e,
    )

    if delta_grid is None:
        delta_grid = MDP_DELTA_GRID_DOM
    if model_class is None:
        model_class = MDPNeuralIRL_E2E
    if extra_model_kw is None:
        extra_model_kw = {}

    if pred_spec is None:
        pred_spec = 'mdp-e2e'
    if pred_model_key is None:
        pred_model_key = ('mdp_e2e_fe' if 'n_stores' in extra_model_kw else 'mdp_e2e')

    delta_grid = np.asarray(delta_grid, dtype=float)
    K          = len(delta_grid)
    kl_grid    = np.zeros(K)
    se_grid    = np.zeros(K)
    all_models = {}
    all_hists  = {}
    dev        = cfg['device']

    ls_val_t = torch.tensor(ls_val, dtype=torch.float32, device=dev)

    for k, d in enumerate(delta_grid):
        mkw = dict(n_goods=G, hidden_dim=hidden, delta_init=float(d))
        mkw.update(extra_model_kw)
        model = model_class(**mkw)
        model.log_delta.requires_grad_(False)

        model, _hist = train_mdp_e2e(
            model, p_tr, y_tr, w_tr, ls_tr,
            store_ids=store_ids_tr,
            store_idx=store_idx_tr,
            epochs=cfg['mdp_e2e_epochs'], lr=cfg['mdp_e2e_lr'],
            batch_size=cfg['mdp_e2e_batch'],
            lam_mono=cfg['mdp_e2e_lam_mono'], lam_slut=cfg['mdp_e2e_lam_slut'],
            slut_start_frac=cfg['mdp_e2e_slut_start'],
            xbar_recompute_every=10,
            device=dev, tag=f"{tag}-d{d:.2f}",
        )

        d_t = torch.tensor(float(d), dtype=torch.float32, device=dev)
        with torch.no_grad():
            xb_val = compute_xbar_e2e(
                d_t, ls_val_t, store_ids=store_ids_val).cpu().numpy()

        pred_kw = {pred_model_key: model, 'xb_prev': xb_val}
        if store_idx_val is not None:
            pred_kw['store_idx'] = store_idx_val
        wp = np.clip(pred(pred_spec, p_val, y_val, cfg, **pred_kw), 1e-8, 1.0)
        wt = np.clip(w_val, 1e-8, 1.0)
        kl_per_obs = np.sum(wt * np.log(wt / wp), axis=1)
        kl_grid[k] = float(kl_per_obs.mean())
        se_grid[k] = float(kl_per_obs.std(ddof=min(1, len(kl_per_obs) - 1))
                           / np.sqrt(max(len(kl_per_obs), 1)))
        all_models[float(d)] = model
        all_hists[float(d)]  = _hist
        print(f"    {tag} δ={d:.2f}: test-KL={kl_grid[k]:.6f} ± {se_grid[k]:.6f}")

    best_k     = int(np.argmin(kl_grid))
    delta_hat  = float(delta_grid[best_k])
    best_model = all_models[delta_hat]
    best_hist  = all_hists[delta_hat]

    threshold = kl_grid[best_k] + se_multiplier * se_grid[best_k]
    id_mask   = kl_grid <= threshold
    id_deltas = delta_grid[id_mask]
    id_set    = ((float(id_deltas.min()), float(id_deltas.max()))
                 if id_mask.any() else (delta_hat, delta_hat))

    print(f"  → {tag} δ̂={delta_hat:.2f}  IS=[{id_set[0]:.2f}, {id_set[1]:.2f}]  "
          f"(KL_min={kl_grid[best_k]:.6f})")
    return {
        'best_model': best_model,
        'best_hist':  best_hist,
        'delta_hat':  delta_hat,
        'kl_grid':    kl_grid,
        'se_grid':    se_grid,
        'id_set':     id_set,
        'id_mask':    id_mask,
        'all_models': all_models,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def dm_test_by_store(resid1, resid2, store_ids):
    """Diebold-Mariano type test blocking by store.

    1. Compute loss difference d_it = e1_it^2 - e2_it^2 for each observation.
       (Sum of squared errors across goods if multivariate).
    2. Compute mean difference d_i for each store i.
    3. Perform paired t-test on d_i values (H0: mean difference is 0).

    Returns
    -------
    t_stat : float
    p_val  : float
    mean_diff : float (positive means model 2 is better if d = e1^2 - e2^2)
    """
    if resid1.ndim > 1:
        loss1 = (resid1**2).sum(axis=1)
        loss2 = (resid2**2).sum(axis=1)
    else:
        loss1 = resid1**2
        loss2 = resid2**2

    diff = loss1 - loss2

    # Aggregate by store
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame({'diff': diff, 'store': store_ids})
    store_means = df.groupby('store')['diff'].mean()

    n = len(store_means)
    mean_diff = store_means.mean()
    std_diff  = store_means.std(ddof=1)
    se_diff   = std_diff / np.sqrt(n)

    if se_diff < 1e-12:
        return 0.0, 1.0, 0.0

    t_stat = mean_diff / se_diff
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    return t_stat, p_val, mean_diff
