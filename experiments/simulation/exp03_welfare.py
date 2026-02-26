"""
experiments/simulation/exp03_welfare.py
========================================
Section 17 of main_multiple_runs.py.

Welfare Bounds Over the Identified Set of δ:
  - Train MDP Neural IRL at each δ in the identified set.
  - Compute structural CV for a 10% ibuprofen shock at each δ.
  - Plot CV vs δ (Figure 12).

run_welfare_delta_seed is also imported by exp06_frozen_delta.py (Section 20).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import time
from scipy.optimize import minimize as _minimize

from src.models.simulation import (
    AIDSBench,
    HabitFormationConsumer,
    MDPNeuralIRL,
    NeuralIRL,
    compute_xbar_e2e,
    train_neural_irl,
    train_mdp_e2e,
)

from experiments.simulation.utils import (
    predict_shares,
    fit_mdp_delta_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-SEED WELFARE FUNCTION (also used by exp06)
# ─────────────────────────────────────────────────────────────────────────────

def run_welfare_delta_seed(seed: int, delta_fixed: float, cfg: dict) -> dict:
    """Train MDP Neural IRL with a *fixed* δ and compute structural CV.

    The neural-network weights adapt to each δ value; only δ is pinned
    externally by pre-computing xbar with the given decay rate.

    Parameters
    ----------
    seed        : RNG seed.
    delta_fixed : δ value at which the model is trained/evaluated.
    cfg         : experiment config dict.
    """
    N              = cfg["N_OBS"]
    DEVICE         = cfg["DEVICE"]
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]
    WELF_EPOCHS    = cfg.get("WELF_EPOCHS", 333)

    np.random.seed(seed)
    torch.manual_seed(seed)

    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    theta = 0.3
    hc    = HabitFormationConsumer(theta=theta, decay=delta_fixed)
    w_hab, xbar_tr = hc.solve_demand(p_pre, income, return_xbar=True)
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
        _r = _minimize(
            _neg_u, np.maximum(_y / (3 * _p), _flr + 0.01),
            bounds=[(_flr[j], None) for j in range(3)],
            constraints=[{"type": "eq", "fun": lambda x, p=_p: p @ x - _y}],
            method="SLSQP")
        _w_gt = _r.x * _p / _y if _r.success else np.ones(3) / 3
        _gt_cv -= (_w_gt * _y / _p) @ _dp

    # LA-AIDS
    aids_h = AIDSBench(); aids_h.fit(p_pre, w_hab, income)

    # Static Neural IRL
    nirl_h = NeuralIRL(n_goods=3, hidden_dim=128)
    nirl_h, _ = train_neural_irl(
        nirl_h, p_pre, income, w_hab, epochs=WELF_EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # MDP Neural IRL with fixed δ
    mdp_h = MDPNeuralIRL(n_goods=3, hidden_dim=128)
    mdp_h, _ = train_neural_irl(
        mdp_h, p_pre, income, w_hab, epochs=WELF_EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_tr, q_prev_data=qp_tr, device=DEVICE)

    # MDP E2E — frozen-δ grid sweep
    log_q_tr = np.log(np.maximum(q_tr, 1e-6))
    _vrng_w  = np.random.default_rng(seed + 77777)
    _nv_w    = max(len(p_pre) // 5, 80)
    _pv_w    = np.clip(_vrng_w.uniform(1, 5, (_nv_w, 3)) +
                       _vrng_w.normal(0, 0.1, (_nv_w, 3)), 1e-3, None)
    _yv_w    = _vrng_w.uniform(1200, 2000, _nv_w)
    _hcv_w   = HabitFormationConsumer(theta=0.3, decay=delta_fixed)
    _wv_w, _ = _hcv_w.solve_demand(_pv_w, _yv_w, return_xbar=True)
    _qv_w    = _wv_w * _yv_w[:, None] / np.maximum(_pv_w, 1e-8)
    _lqv_w   = np.log(np.maximum(_qv_w, 1e-6))

    _sw_welf = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q_tr,
        _pv_w, _yv_w, _wv_w, _lqv_w,
        delta_grid=MDP_DELTA_GRID, epochs=WELF_EPOCHS, device=DEVICE,
        n_goods=3, hidden=128, fixed_beta=None,
        lam_mono=0.2, lam_slut=0.05, batch=256, lr=5e-4,
        tag=f"Welf-E2E-d{delta_fixed:.1f}")
    mdp_ee_h = _sw_welf["best_model"]

    with torch.no_grad():
        _lq_t = torch.tensor(log_q_tr, dtype=torch.float32, device=DEVICE)
        xbar_ee_h = compute_xbar_e2e(
            mdp_ee_h.delta.to(DEVICE), _lq_t, store_ids=None).cpu().numpy()

    # Structural CV for each model
    xb_r    = xb_mn.reshape(1, -1)
    qp_r    = qp_mn.reshape(1, -1)
    xb_ee_r = xbar_ee_h.mean(0, keepdims=True)

    def _cv(spec_tag, **ekw):
        _loss = 0.0
        for _tt in range(_WS):
            _pt = _path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_y]), **ekw)[0]
            _loss -= (_w * _y / _path[_tt]) @ _dp
        return _loss

    return {
        "delta_fixed":  delta_fixed,
        "delta_ee_hat": _sw_welf["delta_hat"],
        "cv_gt":        _gt_cv,
        "cv_aids":      _cv("aids",    aids=aids_h),
        "cv_nirl":      _cv("n-irl",   nirl=nirl_h, device=DEVICE),
        "cv_mdp":       _cv("mdp-irl", mdp_nirl=mdp_h,
                             xbar=xb_r, q_prev=qp_r, device=DEVICE),
        "cv_e2e":       _cv("mdp-e2e", mdp_e2e=mdp_ee_h,
                             xbar_e2e=xb_ee_r, device=DEVICE),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, agg_main: dict | None = None) -> dict:
    """Sweep δ over the identified set, compute welfare bounds, plot Fig 12.

    Parameters
    ----------
    cfg       : experiment config dict.
    agg_main  : aggregated results from exp01 (used to mark E2E / blend
                attractors on the figure); if None attractors are skipped.
    """
    N_RUNS           = cfg["N_RUNS"]
    fig_dir          = cfg.get("fig_dir", "figures")
    DELTA_WELF_GRID  = cfg.get("DELTA_WELF_GRID",
                                [0.50, 0.55, 0.60, 0.65, 0.70,
                                 0.75, 0.80, 0.85, 0.90])
    N_WELF_SEEDS     = cfg.get("N_WELF_SEEDS", min(N_RUNS, 3))
    cfg.setdefault("WELF_EPOCHS", 333)

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SECTION 17 — WELFARE BOUNDS OVER IDENTIFIED SET OF δ")
    print("=" * 72)
    print(f"\n  [P1] Welfare sensitivity: {len(DELTA_WELF_GRID)} δ values × "
          f"{N_WELF_SEEDS} seeds × {cfg['WELF_EPOCHS']} epochs ...")

    _welf_delta_rows = []
    for _wd in DELTA_WELF_GRID:
        for _si in range(N_WELF_SEEDS):
            _seed_w = 700 + _si * 17
            print(f"    δ={_wd:.2f}  seed={_seed_w}", end="", flush=True)
            _tw0 = time.time()
            _wr = run_welfare_delta_seed(_seed_w, _wd, cfg)
            print(f"  →  CV_mdp={_wr['cv_mdp']:.3f}  CV_nirl={_wr['cv_nirl']:.3f}"
                  f"  δ̂_E2E={_wr['delta_ee_hat']:.3f}"
                  f"  ({time.time()-_tw0:.0f}s)")
            _welf_delta_rows.append(_wr)

    # Aggregate across seeds
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

    # Figure 12: CV vs δ
    _wc_models = [
        ("LA-AIDS (static)",    "cv_aids", "#E53935", "--"),
        ("Neural IRL (static)", "cv_nirl", "#1E88E5", "-."),
        ("MDP Neural IRL",      "cv_mdp",  "#00897B", "-"),
        ("MDP IRL (E2E)",       "cv_e2e",  "#FF6F00", "-"),
        ("Ground Truth",        "cv_gt",   "k",        ":"),
    ]

    fig12, (ax12a, ax12b) = plt.subplots(1, 2, figsize=(15, 6))

    for lbl, key, col, ls in _wc_models:
        _mu      = [_welf_agg_by_delta[_wd][key]["mean"] for _wd in DELTA_WELF_GRID]
        _se_vals = [_welf_agg_by_delta[_wd][key]["se"]   for _wd in DELTA_WELF_GRID]
        ax12a.errorbar(DELTA_WELF_GRID, _mu, yerr=_se_vals,
                       fmt=ls, color=col, lw=2.2, ms=7, capsize=4, marker="o",
                       label=lbl)

    if agg_main is not None:
        _e2e_attr   = agg_main["delta_mdp_e2e_mean"]
        _blend_attr = agg_main["delta_mdp_mean"]
        ax12a.axvline(_e2e_attr,   color="#FF6F00", ls=":", lw=1.8,
                      label=rf"E2E attractor $\hat{{\delta}}$={_e2e_attr:.2f}")
        ax12a.axvline(_blend_attr, color="#00897B", ls=":", lw=1.8,
                      label=rf"Blend attractor $\hat{{\delta}}$={_blend_attr:.2f}")
        ax12a.axvspan(_e2e_attr, _blend_attr, color="grey", alpha=0.10,
                      label="Identified set")

    ax12a.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=13)
    ax12a.set_ylabel("Structural CV  (£ equivalent variation)", fontsize=12)
    ax12a.set_title("Panel A: CV vs δ  (10% ibuprofen shock)", fontsize=12,
                    fontweight="bold")
    ax12a.legend(fontsize=10, loc="best"); ax12a.grid(True, alpha=0.3)

    _gt_curve = [_welf_agg_by_delta[_wd]["cv_gt"]["mean"] for _wd in DELTA_WELF_GRID]
    for lbl, key, col, ls in _wc_models[:-1]:
        _mu  = [_welf_agg_by_delta[_wd][key]["mean"] for _wd in DELTA_WELF_GRID]
        _err = [abs(_mu[i] - _gt_curve[i]) for i in range(len(DELTA_WELF_GRID))]
        ax12b.plot(DELTA_WELF_GRID, _err, ls, color=col, lw=2.2, ms=7,
                   marker="o", label=lbl)
    if agg_main is not None:
        ax12b.axvline(_e2e_attr,   color="#FF6F00", ls=":", lw=1.8)
        ax12b.axvline(_blend_attr, color="#00897B", ls=":", lw=1.8)
        ax12b.axvspan(_e2e_attr, _blend_attr, color="grey", alpha=0.10)
    ax12b.set_xlabel(r"Habit-decay parameter $\delta$", fontsize=13)
    ax12b.set_ylabel("|CV error| vs Ground Truth  (£)", fontsize=12)
    ax12b.set_title("Panel B: |CV Error| vs δ  (lower = better)", fontsize=12,
                    fontweight="bold")
    ax12b.legend(fontsize=10); ax12b.grid(True, alpha=0.3)

    fig12.suptitle(
        r"Welfare Sensitivity to $\delta$ over the Identified Set"
        f"\n{N_WELF_SEEDS} seeds × {cfg['WELF_EPOCHS']} epochs · True δ = pinned to grid · "
        r"θ = 0.3 · Shock: 10% ↑ ibuprofen",
        fontsize=12, fontweight="bold")
    fig12.tight_layout()
    fig12.savefig(f"{fig_dir}/fig12_welfare_delta_sensitivity.pdf", dpi=150,
                  bbox_inches="tight")
    fig12.savefig(f"{fig_dir}/fig12_welfare_delta_sensitivity.png", dpi=150,
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
              f"{_d['cv_gt']['mean']:>8.4f}  {_d['cv_aids']['mean']:>8.4f}  "
              f"{_d['cv_nirl']['mean']:>8.4f}  {_d['cv_mdp']['mean']:>8.4f}  "
              f"{_d['cv_e2e']['mean']:>8.4f}  {_d['delta_ee_hat']['mean']:>8.4f}")

    _mdp_cv_vals = [_welf_agg_by_delta[_wd]["cv_mdp"]["mean"] for _wd in DELTA_WELF_GRID]
    _is_mono = (all(_mdp_cv_vals[i] <= _mdp_cv_vals[i+1] for i in range(len(_mdp_cv_vals)-1)) or
                all(_mdp_cv_vals[i] >= _mdp_cv_vals[i+1] for i in range(len(_mdp_cv_vals)-1)))
    print(f"\n  MDP CV range: min={min(_mdp_cv_vals):.4f}  max={max(_mdp_cv_vals):.4f}  "
          f"{'MONOTONE ✓' if _is_mono else 'NON-MONOTONE ⚠'}")

    return {"welf_delta_rows": _welf_delta_rows,
            "welf_agg_by_delta": _welf_agg_by_delta}
