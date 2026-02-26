"""
experiments/simulation/exp04_placebo.py
========================================
Section 18 of main_multiple_runs.py.

Placebo Test — Sorting Confound:
  - DGP: static CES with no habit, but cross-sectional store heterogeneity.
  - Aspirin-preferring stores face lower ibuprofen prices (sorting confound).
  - Measures maximum spurious MDP advantage from sorting alone.
  - Generates Figure 13.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import time

from src.models.simulation import (
    AIDSBench,
    CESConsumer,
    MDPNeuralIRL,
    NeuralIRL,
    compute_xbar_e2e,
    train_neural_irl,
)

from experiments.simulation.utils import (
    predict_shares,
    get_metrics,
    fit_mdp_delta_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-SEED PLACEBO FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_placebo_seed(seed: int, cfg: dict) -> dict:
    """Placebo DGP: static CES + store-level preference × price sorting.

    There is NO true habit formation.  However, "store types" differ in
    their preference weights (α), and the ibuprofen price is negatively
    correlated with the aspirin preference weight across stores.

    If MDP Neural IRL achieves lower RMSE than static Neural IRL on this
    data, the gain is purely spurious — a bound on sorting contamination.
    """
    N              = cfg["N_OBS"]
    DEVICE         = cfg["DEVICE"]
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]
    PLACEBO_EPOCHS = cfg.get("PLACEBO_EPOCHS", 333)

    rng = np.random.default_rng(seed)

    # Store-level types (S=10 stores, each repeated N//S times)
    S = 10; n_per = N // S
    asp_prefs  = rng.uniform(0.20, 0.50, S)
    ibu_prices = 3.0 - 1.5 * (asp_prefs - asp_prefs.min()) / \
                   (asp_prefs.max() - asp_prefs.min() + 1e-8)

    alpha_all  = np.zeros((N, 3))
    prices_all = np.zeros((N, 3))
    income_all = np.zeros(N)
    store_ids  = np.zeros(N, dtype=int)

    for s in range(S):
        sl  = slice(s * n_per, (s + 1) * n_per if s < S - 1 else N)
        n_s = len(range(N)[sl])
        a0  = asp_prefs[s]
        a2  = rng.uniform(0.15, 0.35)
        a1  = max(1.0 - a0 - a2, 0.05)
        _n  = a0 + a1 + a2
        alpha_s = np.array([a0, a1, a2]) / _n
        prices_all[sl, 0] = rng.uniform(2.0, 5.0, n_s)
        prices_all[sl, 1] = rng.uniform(2.0, 5.0, n_s)
        prices_all[sl, 2] = np.maximum(ibu_prices[s] + rng.normal(0, 0.2, n_s), 0.5)
        income_all[sl]    = rng.uniform(1200, 2000, n_s)
        alpha_all[sl]     = alpha_s
        store_ids[sl]     = s

    rho = 0.45
    def _ces_demand(alpha, p, y):
        N_loc = p.shape[0]; w_out = np.zeros((N_loc, 3))
        for i in range(N_loc):
            numer = alpha[i] * p[i]**(rho - 1)
            w_out[i] = numer / (numer.sum() + 1e-12)
        return w_out

    w_true  = _ces_demand(alpha_all, prices_all, income_all)
    p_shock = prices_all.copy(); p_shock[:, 2] *= 1.10
    w_shock = _ces_demand(alpha_all, p_shock, income_all)

    q_prev = w_true * income_all[:, None] / np.maximum(prices_all, 1e-8)
    q_prev = np.vstack([q_prev[0:1], q_prev[:-1]])
    log_q  = np.log(np.maximum(
        w_true * income_all[:, None] / np.maximum(prices_all, 1e-8), 1e-6))

    # LA-AIDS
    aids_pl = AIDSBench(); aids_pl.fit(prices_all, w_true, income_all)
    rmse_aids_pl = get_metrics("aids", p_shock, income_all, w_shock,
                               aids=aids_pl)["RMSE"]

    # Static Neural IRL
    nirl_pl = NeuralIRL(n_goods=3, hidden_dim=256)
    nirl_pl, _ = train_neural_irl(
        nirl_pl, prices_all, income_all, w_true,
        epochs=PLACEBO_EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3, device=DEVICE)
    rmse_nirl_pl = get_metrics("n-irl", p_shock, income_all, w_shock,
                               nirl=nirl_pl, device=DEVICE)["RMSE"]

    # MDP Neural IRL (xbar = lagged q, no true habit)
    xbar_pl = q_prev.copy()
    mdp_pl  = MDPNeuralIRL(n_goods=3, hidden_dim=256)
    mdp_pl, _ = train_neural_irl(
        mdp_pl, prices_all, income_all, w_true,
        epochs=PLACEBO_EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_pl, q_prev_data=q_prev, device=DEVICE)
    xbar_sh_pl  = np.vstack([xbar_pl[0:1], xbar_pl[:-1]])
    rmse_mdp_pl = get_metrics("mdp-irl", p_shock, income_all, w_shock,
                              mdp_nirl=mdp_pl,
                              xbar_shock=xbar_sh_pl,
                              q_prev_shock=q_prev, device=DEVICE)["RMSE"]

    # MDP IRL E2E — frozen-δ grid sweep
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
        n_goods=3, hidden=256, fixed_beta=None,
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
        "rmse_aids":  rmse_aids_pl,
        "rmse_nirl":  rmse_nirl_pl,
        "rmse_mdp":   rmse_mdp_pl,
        "rmse_e2e":   rmse_e2e_pl,
        "delta_e2e":  _sw_pl["delta_hat"],
        "store_ids":  store_ids,
        "asp_prefs":  asp_prefs,
        "ibu_prices": ibu_prices,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def _se(vals):
    a = np.asarray(vals, float)
    return np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))


def run(cfg: dict, agg_main: dict | None = None) -> dict:
    """Run placebo test and generate Figure 13.

    Parameters
    ----------
    cfg      : experiment config dict.
    agg_main : aggregated results from exp01 (used for habit-DGP comparison bars).
    """
    N_RUNS         = cfg["N_RUNS"]
    fig_dir        = cfg.get("fig_dir", "figures")
    N_PLACEBO      = cfg.get("N_PLACEBO_SEEDS", N_RUNS)
    cfg.setdefault("PLACEBO_EPOCHS", 333)

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SECTION 18 — PLACEBO TEST: SORTING CONFOUND")
    print("=" * 72)
    print(f"\n  [P4] Placebo test: {N_PLACEBO} seeds × {cfg['PLACEBO_EPOCHS']} epochs ...")

    _placebo_rows = []
    for _si in range(N_PLACEBO):
        _seed_pl = 800 + _si * 11
        _tpl0 = time.time()
        print(f"    seed={_seed_pl}", end="", flush=True)
        _pr = run_placebo_seed(_seed_pl, cfg)
        print(f"  →  RMSE: AIDS={_pr['rmse_aids']:.5f}  NIRL={_pr['rmse_nirl']:.5f}"
              f"  MDP={_pr['rmse_mdp']:.5f}  E2E={_pr['rmse_e2e']:.5f}"
              f"  δ̂={_pr['delta_e2e']:.3f}"
              f"  ({time.time()-_tpl0:.0f}s)")
        _placebo_rows.append(_pr)

    _pl_keys = ["rmse_aids", "rmse_nirl", "rmse_mdp", "rmse_e2e", "delta_e2e"]
    _pl_agg  = {k: {"mean": np.nanmean([r[k] for r in _placebo_rows]),
                    "se":   _se([r[k] for r in _placebo_rows])}
                for k in _pl_keys}

    # Habit-DGP comparison values (from exp01 agg or defaults)
    if agg_main is not None:
        mdp_agg = agg_main["mdp_agg"]
        _hab_aids_mu = mdp_agg["aids_rmse"]["mean"]
        _hab_nirl_mu = mdp_agg["nirl_rmse"]["mean"]
        _hab_mdp_mu  = mdp_agg["mdp_rmse"]["mean"]
        _hab_e2e_mu  = mdp_agg["mdp_e2e_rmse"]["mean"]
    else:
        _hab_aids_mu = _hab_nirl_mu = _hab_mdp_mu = _hab_e2e_mu = None

    # Figure 13
    fig13, (ax13a, ax13b) = plt.subplots(1, 2, figsize=(14, 6))
    _pl_labels = ["LA-AIDS", "Neural IRL\n(static)", "MDP Neural\nIRL", "MDP IRL\n(E2E)"]
    _pl_rmse   = [_pl_agg[k]["mean"] for k in _pl_keys[:4]]
    _pl_se_arr = [_pl_agg[k]["se"]   for k in _pl_keys[:4]]
    _x = np.arange(len(_pl_labels)); _w = 0.35
    _col_pl  = ["#EF9A9A", "#90CAF9", "#80CBC4", "#FFCC80"]
    _col_hab = ["#E53935", "#1E88E5", "#00897B", "#FF6F00"]

    ax13a.bar(_x - _w/2 if _hab_aids_mu is not None else _x,
              _pl_rmse, _w, color=_col_pl,
              label="Placebo (static + sorting)", yerr=_pl_se_arr,
              capsize=5, edgecolor="k", lw=0.8)
    if _hab_aids_mu is not None:
        _pl_rmse_hab = [_hab_aids_mu, _hab_nirl_mu, _hab_mdp_mu, _hab_e2e_mu]
        ax13a.bar(_x + _w/2, _pl_rmse_hab, _w, color=_col_hab,
                  label="True habit DGP", edgecolor="k", lw=0.8)
    ax13a.set_xticks(_x); ax13a.set_xticklabels(_pl_labels, fontsize=11)
    ax13a.set_ylabel("Post-shock RMSE", fontsize=12)
    ax13a.set_title("Panel A: Absolute RMSE — Placebo vs Habit DGP",
                    fontsize=12, fontweight="bold")
    ax13a.legend(fontsize=11); ax13a.grid(axis="y", alpha=0.3)

    # Panel B: MDP advantage comparison
    _pl_gain_mdp = _pl_agg["rmse_nirl"]["mean"] - _pl_agg["rmse_mdp"]["mean"]
    _pl_gain_e2e = _pl_agg["rmse_nirl"]["mean"] - _pl_agg["rmse_e2e"]["mean"]
    _gain_labels = ["MDP Neural IRL\nvs static", "MDP IRL (E2E)\nvs static"]
    _gain_pl     = [_pl_gain_mdp, _pl_gain_e2e]
    _gain_se_pl  = [np.sqrt(_pl_agg["rmse_nirl"]["se"]**2 + _pl_agg["rmse_mdp"]["se"]**2),
                    np.sqrt(_pl_agg["rmse_nirl"]["se"]**2 + _pl_agg["rmse_e2e"]["se"]**2)]
    _x2 = np.arange(len(_gain_labels))
    ax13b.bar(_x2 - _w/2 if _hab_aids_mu is not None else _x2,
              _gain_pl, _w, color=["#80CBC4", "#FFCC80"],
              label="Placebo (static + sorting)",
              yerr=_gain_se_pl, capsize=5, edgecolor="k", lw=0.8)
    if _hab_aids_mu is not None:
        _hab_gain_mdp = _hab_nirl_mu - _hab_mdp_mu
        _hab_gain_e2e = _hab_nirl_mu - _hab_e2e_mu
        ax13b.bar(_x2 + _w/2, [_hab_gain_mdp, _hab_gain_e2e], _w,
                  color=["#00897B", "#FF6F00"],
                  label="True habit DGP", edgecolor="k", lw=0.8)
    ax13b.axhline(0, color="k", lw=1.2, ls="--")
    ax13b.set_xticks(_x2); ax13b.set_xticklabels(_gain_labels, fontsize=11)
    ax13b.set_ylabel("RMSE reduction vs static Neural IRL", fontsize=12)
    ax13b.set_title("Panel B: MDP Advantage — Placebo vs Habit DGP", fontsize=12, fontweight="bold")
    ax13b.legend(fontsize=11); ax13b.grid(axis="y", alpha=0.3)

    fig13.suptitle(
        "Placebo Test: Static CES + Store Sorting vs True Habit DGP\n"
        f"{N_PLACEBO} seeds × {cfg['PLACEBO_EPOCHS']} epochs  ·  "
        "S=10 store types  ·  Aspirin pref ↔ Ibuprofen price  (negative correlation)",
        fontsize=12, fontweight="bold")
    fig13.tight_layout()
    fig13.savefig(f"{fig_dir}/fig13_placebo_sorting.pdf", dpi=150, bbox_inches="tight")
    fig13.savefig(f"{fig_dir}/fig13_placebo_sorting.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig13_placebo_sorting.pdf")
    plt.close(fig13)

    # Console summary
    print("\n  PLACEBO TEST RESULTS")
    print(f"  {'Model':<22}  {'Placebo RMSE':>14}  {'Habit RMSE':>12}")
    for lbl, pl_key, hab_mu in [
        ("LA-AIDS",         "rmse_aids", _hab_aids_mu),
        ("Neural IRL",      "rmse_nirl", _hab_nirl_mu),
        ("MDP Neural IRL",  "rmse_mdp",  _hab_mdp_mu),
        ("MDP IRL (E2E)",   "rmse_e2e",  _hab_e2e_mu),
    ]:
        _pl_m = _pl_agg[pl_key]["mean"]
        hab_str = f"{hab_mu:>12.5f}" if hab_mu is not None else "    (n/a)"
        print(f"  {lbl:<22}  {_pl_m:>14.5f}  {hab_str}")

    _pl_mdp_adv = _pl_agg["rmse_nirl"]["mean"] - _pl_agg["rmse_mdp"]["mean"]
    print(f"\n  MDP advantage on placebo DGP: {_pl_mdp_adv:.5f}  "
          f"({'⚠ positive → spurious sorting gain' if _pl_mdp_adv > 0 else '✓ zero/negative'})")

    return {"placebo_rows": _placebo_rows, "pl_agg": _pl_agg}
