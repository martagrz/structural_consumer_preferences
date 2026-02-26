"""
experiments/simulation/exp01_main_runs.py
==========================================
Sections 8 – 14 of the original main_multiple_runs.py.

Defines
-------
run_one_seed(seed, cfg)
    Full pipeline for one random seed → returns all metrics.

run_habit_param_seed(seed, cfg, delta, theta, epochs)
    Sweep helper: train habit models for arbitrary (δ, θ).

run(cfg)
    Orchestrate N_RUNS calls to run_one_seed, aggregate results,
    print console summary, generate figures 1-8, write LaTeX tables,
    and print the final summary.  Returns (all_results, aggregated).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warnings
import os
import time
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error, mean_absolute_error

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

from experiments.simulation.utils import (
    P_GRID, AVG_Y, BAND, STYLE,
    predict_shares,
    compute_elasticities,
    compute_full_elasticity_matrix,
    compute_welfare_loss,
    get_metrics,
    kl_div,
    fit_mdp_delta_grid,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8: SINGLE-RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = 10 

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> dict:
    """Execute the full pipeline with one data seed.

    Parameters
    ----------
    seed    : integer RNG seed.
    cfg     : experiment config dict (keys: N_OBS, DEVICE, MDP_DELTA_GRID, …).
    verbose : if True, print extra diagnostic info.

    Returns
    -------
    dict    : all metrics, demand curves, training histories, etc.
    """
    N              = cfg["N_OBS"]
    DEVICE         = cfg["DEVICE"]
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    primary        = CESConsumer()
    w_train        = primary.solve_demand(p_pre, income)
    habit_consumer = HabitFormationConsumer()
    w_habit, xbar_train = habit_consumer.solve_demand(p_pre, income, return_xbar=True)

    # Fresh validation set for δ grid sweep
    _val_rng = np.random.default_rng(seed + 99999)
    _N_val   = max(N // 5, 100)
    _Z_val   = _val_rng.uniform(1, 5, (_N_val, 3))
    _p_val   = np.clip(_Z_val + _val_rng.normal(0, 0.1, (_N_val, 3)), 1e-3, None)
    _y_val   = _val_rng.uniform(1200, 2000, _N_val)
    _hc_val  = HabitFormationConsumer()
    _w_val, _ = _hc_val.solve_demand(_p_val, _y_val, return_xbar=True)
    _q_val   = _w_val * _y_val[:, None] / np.maximum(_p_val, 1e-8)
    _lq_val  = np.log(np.maximum(_q_val, 1e-6))

    p_post           = p_pre.copy(); p_post[:, 1] *= 1.2
    w_post_true      = primary.solve_demand(p_post, income)
    w_habit_shock, xbar_shock = habit_consumer.solve_demand(p_post, income, return_xbar=True)

    # Previous-period quantities for MDP model
    q_train      = w_habit * income[:, None] / np.maximum(p_pre, 1e-8)
    q_prev_train = np.zeros_like(q_train)
    q_prev_train[0]  = q_train[0]
    q_prev_train[1:] = q_train[:-1]

    q_shock = w_habit_shock * income[:, None] / np.maximum(p_post, 1e-8)
    q_prev_shock = np.zeros_like(q_shock)
    q_prev_shock[0]  = q_shock[0]
    q_prev_shock[1:] = q_shock[:-1]

    avg_p    = p_post.mean(0)
    p_pre_pt = avg_p / np.array([1.0, 1.2, 1.0])

    # ── Benchmarks ───────────────────────────────────────────────────────────
    aids_m   = AIDSBench();    aids_m.fit(p_pre, w_train, income)
    quaids_m = QUAIDS();       quaids_m.fit(p_pre, w_train, income)
    series_m = SeriesDemand(); series_m.fit(p_pre, w_train, income)

    _blp_out  = 0.01
    mw_train  = np.column_stack([w_train * (1 - _blp_out), np.full(N, _blp_out)])
    blp_m = BLPBench(); blp_m.fit(p_pre, mw_train, Z)

    aids_hab   = AIDSBench();    aids_hab.fit(p_pre, w_habit, income)
    quaids_hab = QUAIDS();       quaids_hab.fit(p_pre, w_habit, income)
    series_hab = SeriesDemand(); series_hab.fit(p_pre, w_habit, income)

    mw_habit = np.column_stack([w_habit * (1 - _blp_out), np.full(N, _blp_out)])
    blp_hab = BLPBench(); blp_hab.fit(p_pre, mw_habit, Z)

    # ── Linear IRL ───────────────────────────────────────────────────────────
    theta_shared   = run_linear_irl(features_shared(p_pre, income),         w_train)
    theta_goodspec = run_linear_irl(features_good_specific(p_pre, income),  w_train)
    theta_orth     = run_linear_irl(features_orthogonalised(p_pre, income), w_train)

    # ── Neural IRL (CES) ─────────────────────────────────────────────────────
    n_irl = NeuralIRL(n_goods=3, hidden_dim=256)
    n_irl, hist_nirl = train_neural_irl(
        n_irl, p_pre, income, w_train, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        device=DEVICE, verbose=verbose)

    # ── Neural IRL static (Habit baseline) ───────────────────────────────────
    n_irl_hab = NeuralIRL(n_goods=3, hidden_dim=128)
    n_irl_hab, hist_nirl_hab = train_neural_irl(
        n_irl_hab, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    # ── MDP Neural IRL (pre-computed x̄ with fixed δ) ─────────────────────────
    mdp_irl = MDPNeuralIRL(n_goods=3, hidden_dim=256)
    mdp_irl, hist_mdp = train_neural_irl(
        mdp_irl, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xb_prev_data=xbar_train, q_prev_data=q_prev_train,
        device=DEVICE, verbose=verbose)

    # ── MDP Neural IRL — frozen-δ grid sweep ─────────────────────────────────
    log_q_seq = np.log(np.maximum(q_train, 1e-6))

    _sweep_e2e = fit_mdp_delta_grid(
        p_pre, income, w_habit, log_q_seq,
        _p_val, _y_val, _w_val, _lq_val,
        delta_grid=MDP_DELTA_GRID, epochs=EPOCHS, device=DEVICE,
        n_goods=3, hidden=256,
        lam_mono=0.3, lam_slut=0.1, batch=256, lr=5e-4,
        tag="E2E")
    mdp_e2e = _sweep_e2e["best_model"]

    # ── Control-Function (CF) endogeneity correction ──────────────────────────
    _log_p_pre = np.log(np.maximum(p_pre, 1e-8))
    _log_Z     = np.log(np.maximum(Z, 1e-8))
    v_hat_train, _cf_rsq = cf_first_stage(_log_p_pre, _log_Z)
    if verbose:
        print(f"   CF first-stage R²: {_cf_rsq.round(3)}")

    n_irl_cf = NeuralIRL(n_goods=3, hidden_dim=256, n_cf=3)
    n_irl_cf, _ = train_neural_irl(
        n_irl_cf, p_pre, income, w_train, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        v_hat_data=v_hat_train, device=DEVICE)

    mdp_irl_cf = MDPNeuralIRL(n_goods=3, hidden_dim=256, n_cf=3)
    mdp_irl_cf, _ = train_neural_irl(
        mdp_irl_cf, p_pre, income, w_habit, epochs=EPOCHS, lr=5e-4,
        batch_size=256, lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xb_prev_data=xbar_train, q_prev_data=q_prev_train,
        v_hat_data=v_hat_train, device=DEVICE)

    log_p_seq = np.log(np.maximum(p_pre,  1e-8))
    log_y_seq = np.log(np.maximum(income, 1e-8))

    # ── Window IRL (CES DGP) ─────────────────────────────────────────────────
    _WIRL_W = 4
    q_ces     = w_train * income[:, None] / np.maximum(p_pre, 1e-8)
    log_q_ces = np.log(np.maximum(q_ces, 1e-6))
    wf_ces_tr = build_window_features(log_p_seq, log_y_seq, log_q_ces,
                                      window=_WIRL_W, store_ids=None)
    wirl_ces = WindowIRL(n_goods=3, hidden_dim=256, window=_WIRL_W)
    wirl_ces, hist_wirl_ces = train_window_irl(
        wirl_ces, wf_ces_tr, w_train, epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        device=DEVICE, verbose=verbose, tag="Window-IRL-CES")

    # ── Window IRL (Habit DGP) ────────────────────────────────────────────────
    wf_hab_tr = build_window_features(log_p_seq, log_y_seq, log_q_seq,
                                      window=_WIRL_W, store_ids=None)
    wirl_hab = WindowIRL(n_goods=3, hidden_dim=256, window=_WIRL_W)
    wirl_hab, hist_wirl_hab = train_window_irl(
        wirl_hab, wf_hab_tr, w_habit, epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE, tag="Window-IRL-Habit")

    _wirl_lp_mean   = log_p_seq.mean(0)
    _wirl_lq_mean   = log_q_ces.mean(0)
    _wirl_lq_h_mean = log_q_seq.mean(0)

    # ── Variational Mixture ───────────────────────────────────────────────────
    var_mix = ContinuousVariationalMixture(K=6, n_goods=3, n_samples_per_component=100)
    var_mix.fit(p_pre, income, w_train, n_iter=50, lr_mu=0.05, sigma2=0.1)

    # ── Robustness across DGPs ────────────────────────────────────────────────
    all_consumers = {"CES": CESConsumer(), "Quasilinear": QuasilinearConsumer(),
                     "Leontief": LeontiefConsumer(), "Stone-Geary": StoneGearyConsumer(),
                     "Habit": HabitFormationConsumer()}
    rob_rows = {}
    for cname, cons in all_consumers.items():
        try:
            w_dgp       = cons.solve_demand(p_pre, income)
            w_dgp_shock = cons.solve_demand(p_post, income)
            a_rob = AIDSBench(); a_rob.fit(p_pre, w_dgp, income)
            th_rob = run_linear_irl(features_orthogonalised(p_pre, income), w_dgp, epochs=EPOCHS)
            n_rob  = NeuralIRL(n_goods=3, hidden_dim=128)
            n_rob, _ = train_neural_irl(n_rob, p_pre, income, w_dgp,
                          epochs=EPOCHS, lr=5e-4, batch_size=256,
                          lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3, device=DEVICE)
            rob_rows[cname] = {
                "AIDS":           get_metrics("aids",  p_post, income, w_dgp_shock, aids=a_rob)["RMSE"],
                "Lin IRL (Orth)": get_metrics("l-irl", p_post, income, w_dgp_shock,
                                              lirl_theta=th_rob, lirl_feat_fn=features_orthogonalised)["RMSE"],
                "Neural IRL":     get_metrics("n-irl", p_post, income, w_dgp_shock,
                                              nirl=n_rob, device=DEVICE)["RMSE"],
            }
        except Exception:
            rob_rows[cname] = {"AIDS": np.nan, "Lin IRL (Orth)": np.nan, "Neural IRL": np.nan}

    # ── Primary evaluation ────────────────────────────────────────────────────
    KW = dict(aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
              nirl=n_irl, consumer=primary, mixture=var_mix, device=DEVICE)
    MODELS_CES = [
        ("LA-AIDS",          "aids",    {}),
        ("BLP (IV)",         "blp",     {}),
        ("QUAIDS",           "quaids",  {}),
        ("Series Est.",      "series",  {}),
        ("Lin IRL Shared",   "l-irl",   {"lirl_theta": theta_shared,   "lirl_feat_fn": features_shared}),
        ("Lin IRL GoodSpec", "l-irl",   {"lirl_theta": theta_goodspec, "lirl_feat_fn": features_good_specific}),
        ("Lin IRL Orth",     "l-irl",   {"lirl_theta": theta_orth,     "lirl_feat_fn": features_orthogonalised}),
        ("Neural IRL",       "n-irl",   {}),
        ("Var. Mixture",     "mixture", {}),
    ]

    perf_ces = {n: get_metrics(s, p_post, income, w_post_true, **{**KW, **ek})
                for n, s, ek in MODELS_CES}

    elast = {"Ground Truth": compute_elasticities("truth", avg_p, AVG_Y, **KW)}
    elast.update({n: compute_elasticities(s, avg_p, AVG_Y, **{**KW, **ek})
                  for n, s, ek in MODELS_CES})

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

    welf = {"Ground Truth": compute_welfare_loss("truth", p_pre_pt, avg_p, AVG_Y, **KW)}
    welf.update({n: compute_welfare_loss(s, p_pre_pt, avg_p, AVG_Y, **{**KW, **ek})
                 for n, s, ek in MODELS_CES})

    lin_ablation = {
        "Shared (original)": get_metrics("l-irl", p_post, income, w_post_true,
                                          lirl_theta=theta_shared,   lirl_feat_fn=features_shared),
        "Good-specific":     get_metrics("l-irl", p_post, income, w_post_true,
                                          lirl_theta=theta_goodspec, lirl_feat_fn=features_good_specific),
        "Orth + Intercepts": get_metrics("l-irl", p_post, income, w_post_true,
                                          lirl_theta=theta_orth,     lirl_feat_fn=features_orthogonalised),
    }

    # ── MDP advantage ─────────────────────────────────────────────────────────
    rmse_aids_h = get_metrics("aids",    p_post, income, w_habit_shock, aids=aids_hab)["RMSE"]
    rmse_nirl_h = get_metrics("n-irl",   p_post, income, w_habit_shock, nirl=n_irl_hab, device=DEVICE)["RMSE"]
    rmse_mdp_h  = get_metrics("mdp-irl", p_post, income, w_habit_shock,
                               mdp_nirl=mdp_irl, xbar_shock=xbar_shock,
                               q_prev_shock=q_prev_shock, device=DEVICE)["RMSE"]

    kl_aids   = kl_div(aids_hab.predict(p_post, income), w_habit_shock)
    kl_static = kl_div(predict_shares("n-irl", p_post, income, nirl=n_irl_hab, device=DEVICE), w_habit_shock)
    kl_mdp    = kl_div(predict_shares("mdp-irl", p_post, income,
                                       mdp_nirl=mdp_irl, xbar=xbar_shock,
                                       q_prev=q_prev_shock, device=DEVICE), w_habit_shock)

    # E2E MDP and Window IRL — habit advantage
    log_q_shock_seq = np.log(np.maximum(q_shock, 1e-6))
    with torch.no_grad():
        lq_t = torch.tensor(log_q_shock_seq, dtype=torch.float32, device=DEVICE)
        xbar_shock_e2e_t = compute_xbar_e2e(mdp_e2e.delta.to(DEVICE), lq_t, store_ids=None)
    xbar_shock_e2e = xbar_shock_e2e_t.cpu().numpy()

    rmse_mdp_e2e_h = get_metrics("mdp-e2e", p_post, income, w_habit_shock,
                                  mdp_e2e=mdp_e2e, xbar_e2e=xbar_shock_e2e,
                                  device=DEVICE)["RMSE"]
    kl_mdp_e2e = kl_div(predict_shares("mdp-e2e", p_post, income,
                                        mdp_e2e=mdp_e2e, xbar_e2e=xbar_shock_e2e,
                                        device=DEVICE), w_habit_shock)

    rmse_blp_h    = get_metrics("blp",    p_post, income, w_habit_shock, blp=blp_hab)["RMSE"]
    rmse_quaids_h = get_metrics("quaids", p_post, income, w_habit_shock, quaids=quaids_hab)["RMSE"]
    rmse_series_h = get_metrics("series", p_post, income, w_habit_shock, series=series_hab)["RMSE"]
    kl_blp    = kl_div(predict_shares("blp",    p_post, income, blp=blp_hab), w_habit_shock)
    kl_quaids = kl_div(predict_shares("quaids", p_post, income, quaids=quaids_hab), w_habit_shock)
    kl_series = kl_div(predict_shares("series", p_post, income, series=series_hab), w_habit_shock)

    # Window IRL — habit advantage
    log_p_shock_seq = np.log(np.maximum(p_post, 1e-8))
    log_y_shock_seq = np.log(np.maximum(income, 1e-8))
    log_q_sh_seq    = np.log(np.maximum(
        w_habit_shock * income[:, None] / np.maximum(p_post, 1e-8), 1e-6))
    wf_hab_sh = build_window_features(log_p_shock_seq, log_y_shock_seq,
                                       log_q_sh_seq, window=_WIRL_W, store_ids=None)
    with torch.no_grad():
        xt_sh = torch.tensor(wf_hab_sh, dtype=torch.float32, device=DEVICE)
        w_wirl_hab_sh = wirl_hab(xt_sh).cpu().numpy()
    rmse_wirl_h = float(np.sqrt(np.mean((w_wirl_hab_sh - w_habit_shock)**2)))
    kl_wirl_h   = kl_div(w_wirl_hab_sh, w_habit_shock)

    # ── Demand curve arrays on fixed price grid ───────────────────────────────
    test_p     = np.tile(p_pre.mean(0), (80, 1)); test_p[:, 1] = P_GRID
    fixed_y    = np.full(80, AVG_Y)
    xbar_rep   = np.tile(xbar_train.mean(0), (80, 1))
    q_prev_rep = np.tile(q_prev_train.mean(0), (80, 1))

    _wirl_ces_kw = dict(wirl=wirl_ces, wirl_log_p_hist=_wirl_lp_mean,
                        wirl_log_q_hist=_wirl_lq_mean, wirl_window=_WIRL_W, device=DEVICE)

    curves = {
        "Truth":          primary.solve_demand(test_p, fixed_y)[:, 0],
        "LA-AIDS":        predict_shares("aids",      test_p, fixed_y, **KW)[:, 0],
        "BLP (IV)":       predict_shares("blp",       test_p, fixed_y, **KW)[:, 0],
        "QUAIDS":         predict_shares("quaids",    test_p, fixed_y, **KW)[:, 0],
        "Series Est.":    predict_shares("series",    test_p, fixed_y, **KW)[:, 0],
        "Lin IRL Shared": predict_shares("l-irl",     test_p, fixed_y,
                                          lirl_theta=theta_shared, lirl_feat_fn=features_shared)[:, 0],
        "Lin IRL Orth":   predict_shares("l-irl",     test_p, fixed_y,
                                          lirl_theta=theta_orth, lirl_feat_fn=features_orthogonalised)[:, 0],
        "Neural IRL":     predict_shares("n-irl",     test_p, fixed_y, **KW)[:, 0],
        "Window IRL":     predict_shares("window-irl", test_p, fixed_y, **_wirl_ces_kw)[:, 0],
        "Var. Mixture":   predict_shares("mixture",   test_p, fixed_y, **KW)[:, 0],
    }

    # True conditional shares (habit consumer with fixed x̄_mean)
    true_conditional_shares = np.zeros((80, 3))
    for i in range(80):
        p_i = test_p[i]; y_i = fixed_y[i]; xbar_i = xbar_rep[i]
        floor = habit_consumer.theta * xbar_i + 1e-6
        def neg_u_static(x, _p=p_i, _xb=xbar_i):
            adj = x - habit_consumer.theta * _xb
            if np.any(adj <= 0): return 1e10
            return -(np.sum(habit_consumer.alpha * adj**habit_consumer.rho))**(1/habit_consumer.rho)
        x0   = np.maximum(y_i / (3 * p_i), floor + 0.01)
        cons = {"type": "eq", "fun": lambda x, p=p_i: p @ x - y_i}
        res  = minimize(neg_u_static, x0, bounds=[(floor[j], None) for j in range(3)],
                        constraints=cons, method="SLSQP")
        true_conditional_shares[i] = res.x * p_i / y_i

    # E2E xbar at mean training habit
    with torch.no_grad():
        lq_rep_t = torch.tensor(
            np.log(np.maximum(np.tile(q_train.mean(0), (80, 1)), 1e-6)),
            dtype=torch.float32, device=DEVICE)
        xbar_e2e_rep    = compute_xbar_e2e(mdp_e2e.delta.to(DEVICE), lq_rep_t, store_ids=None).cpu().numpy()

    # ── Habit DGP welfare (structural CV with fixed x̄_mean) ──────────────────
    _wh_p0  = avg_p / np.array([1.0, 1.2, 1.0])
    _wh_p1  = avg_p
    _wh_y   = float(income.mean())
    _wh_xb  = xbar_train.mean(0)
    _wh_qp  = q_prev_train.mean(0)
    _wh_xb_e2e    = xbar_e2e_rep[0]
    _WH_S   = 60
    _wh_path = np.linspace(_wh_p0, _wh_p1, _WH_S)
    _wh_dp   = (_wh_p1 - _wh_p0) / _WH_S

    def _whl(spec_tag, **ekw):
        _loss = 0.0
        for _tt in range(_WH_S):
            _pt = _wh_path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_wh_y]), **ekw)[0]
            _loss -= (_w * _wh_y / _wh_path[_tt]) @ _wh_dp
        return _loss

    _wirl_hab_kw = dict(wirl=wirl_hab, wirl_log_p_hist=_wirl_lp_mean,
                        wirl_log_q_hist=_wirl_lq_h_mean, wirl_window=_WIRL_W, device=DEVICE)

    # Ground truth CV for habit
    _wh_gt = 0.0
    for _tt in range(_WH_S):
        _p   = _wh_path[_tt]
        _flr = habit_consumer.theta * _wh_xb + 1e-6
        def _neg_u_wh(x, _p=_p):
            adj = x - habit_consumer.theta * _wh_xb
            return (1e10 if np.any(adj <= 0)
                    else -(np.sum(habit_consumer.alpha * adj**habit_consumer.rho)
                           )**(1/habit_consumer.rho))
        _r_wh = minimize(
            _neg_u_wh, np.maximum(_wh_y / (3 * _p), _flr + 0.01),
            bounds=[(_flr[j], None) for j in range(3)],
            constraints=[{"type": "eq", "fun": lambda x, p=_p: p @ x - _wh_y}],
            method="SLSQP")
        _w_wh = _r_wh.x * _p / _wh_y if _r_wh.success else np.ones(3) / 3
        _wh_gt -= (_w_wh * _wh_y / _p) @ _wh_dp

    welf_habit = {
        "Ground Truth":         _wh_gt,
        "LA-AIDS (static)":     _whl("aids",       aids=aids_hab),
        "BLP (IV) (static)":    _whl("blp",        blp=blp_hab),
        "QUAIDS (static)":      _whl("quaids",     quaids=quaids_hab),
        "Series Est. (static)": _whl("series",     series=series_hab),
        "Window IRL":           _whl("window-irl", **_wirl_hab_kw),
        "Neural IRL (static)":  _whl("n-irl",      nirl=n_irl_hab, device=DEVICE),
        "MDP Neural IRL":       _whl("mdp-irl",    mdp_nirl=mdp_irl,
                                     xbar=_wh_xb.reshape(1, -1),
                                     q_prev=_wh_qp.reshape(1, -1), device=DEVICE),
        "MDP IRL (E2E δ)":      _whl("mdp-e2e",   mdp_e2e=mdp_e2e,
                                     xbar_e2e=_wh_xb_e2e.reshape(1, -1), device=DEVICE),
        "Neural IRL (CF)":      _whl("n-irl-cf",   nirl_cf=n_irl_cf, device=DEVICE),
        "MDP IRL (CF)":         _whl("mdp-irl-cf", mdp_nirl_cf=mdp_irl_cf,
                                     xbar=_wh_xb.reshape(1, -1),
                                     q_prev=_wh_qp.reshape(1, -1), device=DEVICE),
    }

    # ── Welfare across xbar distribution ─────────────────────────────────────
    _xb_pcts    = np.percentile(xbar_train, [10, 25, 50, 75, 90], axis=0)
    _qp_pcts    = np.percentile(q_prev_train, [10, 25, 50, 75, 90], axis=0)
    _pct_labels = [10, 25, 50, 75, 90]

    def _whl_pct(spec_tag, xb_pt, qp_pt, **ekw):
        _loss = 0.0
        for _tt in range(_WH_S):
            _pt = _wh_path[_tt:_tt+1]
            _w  = predict_shares(spec_tag, _pt, np.array([_wh_y]),
                                 xbar=xb_pt.reshape(1, -1),
                                 q_prev=qp_pt.reshape(1, -1), **ekw)[0]
            _loss -= (_w * _wh_y / _wh_path[_tt]) @ _wh_dp
        return _loss

    welf_by_pct = {}
    for _pi, _pct in enumerate(_pct_labels):
        _xb_pt = _xb_pcts[_pi]; _qp_pt = _qp_pcts[_pi]
        with torch.no_grad():
            _lq_pct = torch.tensor(
                np.log(np.maximum(np.tile(_xb_pt, (80, 1)), 1e-6)),
                dtype=torch.float32, device=DEVICE)
            _xb_e2e_pct    = compute_xbar_e2e(mdp_e2e.delta.to(DEVICE), _lq_pct, store_ids=None).cpu().numpy()[0]
        welf_by_pct[_pct] = {
            "Neural IRL (static)": _whl_pct("n-irl",      _xb_pt, _qp_pt, nirl=n_irl_hab, device=DEVICE),
            "MDP Neural IRL":      _whl_pct("mdp-irl",    _xb_pt, _qp_pt, mdp_nirl=mdp_irl, device=DEVICE),
            "MDP IRL (E2E δ)":     _whl("mdp-e2e",        mdp_e2e=mdp_e2e,
                                          xbar_e2e=_xb_e2e_pct.reshape(1, -1), device=DEVICE),
            "MDP IRL (CF)":        _whl_pct("mdp-irl-cf", _xb_pt, _qp_pt, mdp_nirl_cf=mdp_irl_cf, device=DEVICE),
        }

    # MDP demand curves
    mdp_curves = {
        "Truth":             true_conditional_shares,
        "LA-AIDS":           predict_shares("aids",       test_p, fixed_y, aids=aids_hab),
        "BLP (IV)":          predict_shares("blp",        test_p, fixed_y, blp=blp_hab),
        "QUAIDS":            predict_shares("quaids",     test_p, fixed_y, quaids=quaids_hab),
        "Series Est.":       predict_shares("series",     test_p, fixed_y, series=series_hab),
        "Window IRL":        predict_shares("window-irl", test_p, fixed_y, **_wirl_hab_kw),
        "Neural IRL static": predict_shares("n-irl",      test_p, fixed_y, nirl=n_irl_hab, device=DEVICE),
        "MDP-IRL":           predict_shares("mdp-irl",    test_p, fixed_y,
                                             mdp_nirl=mdp_irl, xbar=xbar_rep,
                                             q_prev=q_prev_rep, device=DEVICE),
        "MDP-IRL (E2E δ)":  predict_shares("mdp-e2e",   test_p, fixed_y,
                                             mdp_e2e=mdp_e2e, xbar_e2e=xbar_e2e_rep, device=DEVICE),
    }

    # ── δ identification results from training-time sweep ────────────────────
    _kl_delta_grid    = MDP_DELTA_GRID.copy()
    _kl_e2e_arr       = _sweep_e2e["kl_grid"].copy()
    _se_e2e_arr       = _sweep_e2e["se_grid"].copy()
    _delta_hat_e2e    = _sweep_e2e["delta_hat"]
    _id_set_e2e       = _sweep_e2e["id_set"]
    _best_k_e2e = int(np.argmin(_kl_e2e_arr))
    _c95_e2e    = float(2.0 * _se_e2e_arr[_best_k_e2e])
    _delta_cs_lo_e2e = _id_set_e2e[0]; _delta_cs_hi_e2e = _id_set_e2e[1]
    _true_delta_in_cs = bool(0.7 >= _delta_cs_lo_e2e and 0.7 <= _delta_cs_hi_e2e)

    return {
        "perf_ces":     perf_ces,
        "elast":        elast,
        "cross_elast":  cross_elast,
        "welf":         welf,
        "welf_habit":   welf_habit,
        "welf_by_pct":  welf_by_pct,
        "lin_ablation": lin_ablation,
        "rob_rows":     rob_rows,
        "mdp": {
            "aids_rmse":        rmse_aids_h,     "blp_rmse":         rmse_blp_h,
            "quaids_rmse":      rmse_quaids_h,   "series_rmse":      rmse_series_h,
            "wirl_rmse":        rmse_wirl_h,      "nirl_rmse":        rmse_nirl_h,
            "mdp_rmse":         rmse_mdp_h,       "mdp_e2e_rmse":     rmse_mdp_e2e_h,
            "kl_aids":          kl_aids,           "kl_blp":           kl_blp,
            "kl_quaids":        kl_quaids,         "kl_series":        kl_series,
            "kl_wirl":          kl_wirl_h,         "kl_static":        kl_static,
            "kl_mdp":           kl_mdp,            "kl_mdp_e2e":       kl_mdp_e2e,
        },
        "delta_mdp":         mdp_irl.delta.item(),
        "delta_mdp_e2e":     _sweep_e2e["delta_hat"],
        "shock_pt":          p_pre[:, 1].mean() * 1.2,
        "curves":            curves,
        "mdp_curves":        mdp_curves,
        "kl_delta_grid":     _kl_delta_grid,
        "kl_profile_e2e":    _kl_e2e_arr,
        "delta_cs_e2e":      (_delta_cs_lo_e2e, _delta_cs_hi_e2e),
        "delta_hat_e2e":     float(_delta_hat_e2e),
        "c95_e2e":           _c95_e2e,
        "true_delta_in_cs":  _true_delta_in_cs,
        "cf_rsq":            _cf_rsq,
        "hist_nirl":         hist_nirl,
        "hist_nirl_hab":     hist_nirl_hab,
        "hist_mdp":          hist_mdp,
        "hist_mdp_e2e":      _sweep_e2e["best_hist"],
        "hist_wirl_ces":     hist_wirl_ces,
        "hist_wirl_hab":     hist_wirl_hab,
        "comp_summary":      var_mix.get_component_summary(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8b: SWEEP HELPER — (δ, θ) robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_habit_param_seed(seed: int, cfg: dict, delta: float = 0.7,
                         theta: float = 0.3, epochs: int = EPOCHS) -> dict:
    """Train habit models for a given (delta, theta) pair.

    Used by exp02_identification.py for δ-identification and θ-robustness sweeps.
    """
    N              = cfg["N_OBS"]
    DEVICE         = cfg["DEVICE"]
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
    income = np.random.uniform(1200, 2000, N)

    hc = HabitFormationConsumer(theta=theta, decay=delta)
    w_hab, xbar_tr = hc.solve_demand(p_pre, income, return_xbar=True)
    p_post = p_pre.copy(); p_post[:, 1] *= 1.2
    w_shock, xbar_sh = hc.solve_demand(p_post, income, return_xbar=True)

    q_tr  = w_hab   * income[:, None] / np.maximum(p_pre,  1e-8)
    qp_tr = np.vstack([q_tr[0:1],  q_tr[:-1]])
    q_sh  = w_shock * income[:, None] / np.maximum(p_post, 1e-8)
    qp_sh = np.vstack([q_sh[0:1],  q_sh[:-1]])
    log_q = np.log(np.maximum(q_tr, 1e-6))

    aids_h = AIDSBench(); aids_h.fit(p_pre, w_hab, income)

    nirl_h = NeuralIRL(n_goods=3, hidden_dim=128)
    nirl_h, _ = train_neural_irl(
        nirl_h, p_pre, income, w_hab, epochs=epochs, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE)

    mdp_b = MDPNeuralIRL(n_goods=3, hidden_dim=128)
    mdp_b, _ = train_neural_irl(
        mdp_b, p_pre, income, w_hab, epochs=epochs, lr=5e-4,
        batch_size=256, lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        xb_prev_data=xbar_tr, q_prev_data=qp_tr, device=DEVICE)

    _vrng2 = np.random.default_rng(seed + 88888)
    _nv2   = max(len(p_pre) // 5, 80)
    _pv2   = np.clip(_vrng2.uniform(1, 5, (_nv2, 3)) + _vrng2.normal(0, 0.1, (_nv2, 3)), 1e-3, None)
    _yv2   = _vrng2.uniform(1200, 2000, _nv2)
    _hcv2  = HabitFormationConsumer(theta=theta, decay=delta)
    _wv2, _ = _hcv2.solve_demand(_pv2, _yv2, return_xbar=True)
    _qv2   = _wv2 * _yv2[:, None] / np.maximum(_pv2, 1e-8)
    _lqv2  = np.log(np.maximum(_qv2, 1e-6))

    _sw_ee = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q, _pv2, _yv2, _wv2, _lqv2,
        delta_grid=MDP_DELTA_GRID, epochs=epochs, device=DEVICE,
        n_goods=3, hidden=128,
        lam_mono=0.2, lam_slut=0.05, batch=256, lr=5e-4,
        tag=f"Habit-E2E-d{delta:.1f}-t{theta:.1f}")
    mdp_ee = _sw_ee["best_model"]

    log_q_sh = np.log(np.maximum(q_sh, 1e-6))
    with torch.no_grad():
        _lq_sh_t = torch.tensor(log_q_sh, dtype=torch.float32, device=DEVICE)
        xbar_sh_ee = compute_xbar_e2e(mdp_ee.delta.to(DEVICE), _lq_sh_t, store_ids=None).cpu().numpy()

    def _rmse(spec, **kw):
        return get_metrics(spec, p_post, income, w_shock, **kw)["RMSE"]

    return {
        "true_delta":   delta,
        "true_theta":   theta,
        "delta_blend":  mdp_b.delta.item(),
        "delta_e2e":    _sw_ee["delta_hat"],
        "rmse_aids":    _rmse("aids",       aids=aids_h),
        "rmse_nirl":    _rmse("n-irl",      nirl=nirl_h, device=DEVICE),
        "rmse_mdp":     _rmse("mdp-irl",    mdp_nirl=mdp_b, xbar_shock=xbar_sh,
                               q_prev_shock=qp_sh, device=DEVICE),
        "rmse_e2e":     _rmse("mdp-e2e",    mdp_e2e=mdp_ee, xbar_e2e=xbar_sh_ee, device=DEVICE),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _se(vals):
    """Standard error = std(ddof=1) / sqrt(n), NaN-safe."""
    a = np.asarray(vals, float)
    return np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))


def _aggregate(all_results: list, N_RUNS: int) -> dict:
    """Aggregate scalar and array metrics across runs."""
    model_names = list(all_results[0]["perf_ces"].keys())
    perf_agg = {}
    for nm in model_names:
        for metric in ("RMSE", "MAE"):
            vals = [r["perf_ces"][nm][metric] for r in all_results]
            perf_agg.setdefault(nm, {})[f"{metric}_mean"] = np.nanmean(vals)
            perf_agg.setdefault(nm, {})[f"{metric}_se"]   = _se(vals)

    elast_names = list(all_results[0]["elast"].keys())
    elast_agg = {}
    for nm in elast_names:
        vals = np.array([r["elast"][nm] for r in all_results])
        elast_agg[nm] = {"mean": vals.mean(0),
                         "se":   vals.std(0, ddof=1) / np.sqrt(N_RUNS)}

    welf_names = list(all_results[0]["welf"].keys())
    welf_agg = {}
    for nm in welf_names:
        vals = [r["welf"][nm] for r in all_results]
        welf_agg[nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

    welf_hab_models = list(all_results[0]["welf_habit"].keys())
    welf_hab_agg = {}
    for nm in welf_hab_models:
        vals = [r["welf_habit"][nm] for r in all_results]
        welf_hab_agg[nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

    _pct_labels_agg = [10, 25, 50, 75, 90]
    _pct_models     = list(all_results[0]["welf_by_pct"][10].keys())
    welf_pct_agg = {}
    for _pct in _pct_labels_agg:
        welf_pct_agg[_pct] = {}
        for nm in _pct_models:
            vals = [r["welf_by_pct"][_pct][nm] for r in all_results]
            welf_pct_agg[_pct][nm] = {"mean": np.nanmean(vals), "se": _se(vals)}

    delta_cs_e2e_lo = np.nanmean([r["delta_cs_e2e"][0] for r in all_results])
    delta_cs_e2e_hi = np.nanmean([r["delta_cs_e2e"][1] for r in all_results])
    delta_hat_e2e_mean = np.nanmean([r["delta_hat_e2e"] for r in all_results])
    true_delta_in_cs_frac = np.mean([r["true_delta_in_cs"] for r in all_results])

    cross_elast_models = list(all_results[0]["cross_elast"].keys())
    cross_elast_agg = {}
    for _nm in cross_elast_models:
        _stack = np.stack([r["cross_elast"][_nm] for r in all_results], 0)
        cross_elast_agg[_nm] = {
            "mean": np.nanmean(_stack, 0),
            "se":   np.nanstd(_stack, 0, ddof=min(1, N_RUNS - 1)) / np.sqrt(N_RUNS),
        }

    dgp_names = list(all_results[0]["rob_rows"].keys())
    col_names  = list(all_results[0]["rob_rows"][dgp_names[0]].keys())
    rob_agg = {}
    for dg in dgp_names:
        rob_agg[dg] = {}
        for col in col_names:
            vals = [r["rob_rows"][dg][col] for r in all_results]
            rob_agg[dg][col] = {"mean": np.nanmean(vals), "se": _se(vals)}

    mdp_keys = ["aids_rmse", "blp_rmse", "quaids_rmse", "series_rmse", "wirl_rmse",
                "nirl_rmse", "mdp_rmse", "mdp_e2e_rmse",
                "kl_aids", "kl_blp", "kl_quaids", "kl_series", "kl_wirl",
                "kl_static", "kl_mdp", "kl_mdp_e2e"]
    mdp_agg = {k: {"mean": np.nanmean([r["mdp"][k] for r in all_results]),
                   "se":   _se([r["mdp"][k] for r in all_results])}
               for k in mdp_keys}

    lin_names = list(all_results[0]["lin_ablation"].keys())
    lin_agg = {}
    for nm in lin_names:
        for metric in ("RMSE", "MAE"):
            vals = [r["lin_ablation"][nm][metric] for r in all_results]
            lin_agg.setdefault(nm, {})[f"{metric}_mean"] = np.nanmean(vals)
            lin_agg.setdefault(nm, {})[f"{metric}_se"]   = _se(vals)

    delta_mdp_mean  = np.mean([r["delta_mdp"] for r in all_results])
    delta_mdp_se    = _se([r["delta_mdp"]     for r in all_results])
    delta_mdp_e2e_mean    = np.mean([r["delta_mdp_e2e"]    for r in all_results])
    delta_mdp_e2e_se      = _se([r["delta_mdp_e2e"]        for r in all_results])

    curve_models = list(all_results[0]["curves"].keys())
    curves_mean  = {m: np.mean( [r["curves"][m] for r in all_results], 0) for m in curve_models}
    curves_se    = {m: np.array([r["curves"][m] for r in all_results]).std(0, ddof=1) / np.sqrt(N_RUNS)
                    for m in curve_models}
    mdp_models   = list(all_results[0]["mdp_curves"].keys())
    mdp_mean     = {m: np.mean( [r["mdp_curves"][m] for r in all_results], 0) for m in mdp_models}
    mdp_se       = {m: np.array([r["mdp_curves"][m] for r in all_results]).std(0, ddof=1) / np.sqrt(N_RUNS)
                    for m in mdp_models}

    shock_pt_mean = np.mean([r["shock_pt"] for r in all_results])
    last = all_results[-1]
    delta_e2e_arr = np.array([r["delta_mdp_e2e"] for r in all_results])
    delta_mdp_arr = np.array([r["delta_mdp"]     for r in all_results])

    kl_delta_grid    = last["kl_delta_grid"]
    kl_prof_e2e_arr  = np.stack([r["kl_profile_e2e"] for r in all_results], 0)
    kl_prof_e2e_mean = kl_prof_e2e_arr.mean(0)
    kl_prof_e2e_se   = kl_prof_e2e_arr.std(0, ddof=1) / np.sqrt(N_RUNS)

    return dict(
        model_names=model_names,       perf_agg=perf_agg,
        elast_names=elast_names,       elast_agg=elast_agg,
        welf_names=welf_names,         welf_agg=welf_agg,
        welf_hab_agg=welf_hab_agg,     welf_pct_agg=welf_pct_agg,
        _pct_labels_agg=_pct_labels_agg,
        _pct_models=_pct_models,
        delta_cs_e2e_lo=delta_cs_e2e_lo, delta_cs_e2e_hi=delta_cs_e2e_hi,
        delta_hat_e2e_mean=delta_hat_e2e_mean,
        true_delta_in_cs_frac=true_delta_in_cs_frac,
        cross_elast_agg=cross_elast_agg,
        rob_agg=rob_agg,               mdp_agg=mdp_agg,
        lin_agg=lin_agg,
        delta_mdp_mean=delta_mdp_mean, delta_mdp_se=delta_mdp_se,
        delta_mdp_e2e_mean=delta_mdp_e2e_mean, delta_mdp_e2e_se=delta_mdp_e2e_se,
        curves_mean=curves_mean,       curves_se=curves_se,
        mdp_mean=mdp_mean,             mdp_se=mdp_se,
        shock_pt_mean=shock_pt_mean,
        last=last,
        delta_e2e_arr=delta_e2e_arr,
        delta_mdp_arr=delta_mdp_arr,
        kl_delta_grid=kl_delta_grid,
        kl_prof_e2e_mean=kl_prof_e2e_mean, kl_prof_e2e_se=kl_prof_e2e_se,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(agg: dict, N_RUNS: int, N_OBS: int):
    """Print aggregated results tables to stdout."""
    def fmt(m, s, d=5): return f"{m:.{d}f} ({s:.{d}f})"

    print("\n" + "=" * 72)
    print(f"  AGGREGATED RESULTS  (mean over {N_RUNS} runs, SE in parentheses)")
    print("=" * 72)

    print("\n  TABLE 1: POST-SHOCK RMSE & MAE")
    for nm in agg["model_names"]:
        d = agg["perf_agg"][nm]
        print(f"  {nm:<22} RMSE={fmt(d['RMSE_mean'],d['RMSE_se'])}  "
              f"MAE={fmt(d['MAE_mean'],d['MAE_se'])}")

    print("\n  TABLE 2: OWN-PRICE ELASTICITIES")
    for nm in agg["elast_names"]:
        d = agg["elast_agg"][nm]
        row = "  ".join(f"{d['mean'][i]:.3f}({d['se'][i]:.3f})" for i in range(3))
        print(f"  {nm:<22} {row}")

    print("\n  TABLE 3: WELFARE (CS LOSS)")
    gt_m = agg["welf_agg"]["Ground Truth"]["mean"]
    for nm in agg["welf_names"]:
        d   = agg["welf_agg"][nm]
        err = "" if nm == "Ground Truth" else f"  err={100*abs(d['mean']-gt_m)/abs(gt_m):.1f}%"
        print(f"  {nm:<22} {fmt(d['mean'],d['se'],2)}{err}")

    print("\n  TABLE 5: MDP ADVANTAGE")
    for lbl, kr, kk in [
        ("LA-AIDS",           "aids_rmse",       "kl_aids"),
        ("BLP (IV)",          "blp_rmse",        "kl_blp"),
        ("QUAIDS",            "quaids_rmse",     "kl_quaids"),
        ("Series Est.",       "series_rmse",     "kl_series"),
        ("Window IRL",        "wirl_rmse",       "kl_wirl"),
        ("Neural IRL static", "nirl_rmse",       "kl_static"),
        ("MDP Neural IRL",    "mdp_rmse",        "kl_mdp"),
        ("MDP IRL (E2E δ)",   "mdp_e2e_rmse",    "kl_mdp_e2e"),
    ]:
        print(f"  {lbl:<25} RMSE={fmt(agg['mdp_agg'][kr]['mean'],agg['mdp_agg'][kr]['se'])}  "
              f"KL={fmt(agg['mdp_agg'][kk]['mean'],agg['mdp_agg'][kk]['se'])}")

    print("\n  TABLE: 95% CONFIDENCE SETS FOR δ")
    print(f"  True δ = 0.7  (HabitFormationConsumer.decay)")
    print(f"  {'MDP IRL (E2E δ)':<25} {agg['delta_hat_e2e_mean']:.3f}        "
          f"[{agg['delta_cs_e2e_lo']:.3f}, {agg['delta_cs_e2e_hi']:.3f}]       "
          f"{'YES' if agg['true_delta_in_cs_frac'] >= 0.5 else 'NO'} "
          f"({100*agg['true_delta_in_cs_frac']:.0f}% of runs)")

    _pct_hdr = "  ".join(f"p{p:<6}" for p in agg["_pct_labels_agg"])
    print(f"\n  TABLE: CV WELFARE BY HABIT-STOCK PERCENTILE")
    print(f"  {'Model':<26} {_pct_hdr}")
    for _nm in agg["_pct_models"]:
        _row = "  ".join(f"{agg['welf_pct_agg'][p][_nm]['mean']:.4f}"
                         for p in agg["_pct_labels_agg"])
        print(f"  {_nm:<26} {_row}")

    dm = agg["delta_mdp_mean"]; ds = agg["delta_mdp_se"]
    em = agg["delta_mdp_e2e_mean"]; es = agg["delta_mdp_e2e_se"]
    print(f"  δ̂ MDP IRL (fixed): {fmt(dm, ds, 4)}  (blend only)")
    print(f"  δ̂ MDP E2E (recov): {fmt(em, es, 4)}  (true δ = 0.7)")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES 1-8
# ─────────────────────────────────────────────────────────────────────────────

def _make_figures(agg: dict, N_RUNS: int, fig_dir: str = "figures",
                  true_delta: float = 0.7):
    """Generate and save Figures 1-8 and 11 (KL delta profile)."""
    os.makedirs(fig_dir, exist_ok=True)

    curves_mean  = agg["curves_mean"];  curves_se  = agg["curves_se"]
    mdp_mean     = agg["mdp_mean"];     mdp_se     = agg["mdp_se"]
    last         = agg["last"]
    kl_delta_grid    = agg["kl_delta_grid"]
    kl_prof_e2e_mean = agg["kl_prof_e2e_mean"];  kl_prof_e2e_se = agg["kl_prof_e2e_se"]

    # ── Figure 1: Demand curves — CES DGP ────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for lbl, st in STYLE.items():
        if lbl not in curves_mean: continue
        mu = curves_mean[lbl]; sigma = curves_se[lbl]
        ax1.plot(P_GRID, mu, label=lbl, **st)
        ax1.fill_between(P_GRID, mu - sigma, mu + sigma, color=st["color"], alpha=BAND)
    ax1.set_xlabel("Fuel price ($p_1$)", fontsize=14)
    ax1.set_ylabel("Food budget share ($w_0$)", fontsize=14)
    ax1.legend(fontsize=14, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{fig_dir}/fig1_demand_curves.pdf", dpi=150, bbox_inches="tight")
    fig1.savefig(f"{fig_dir}/fig1_demand_curves.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig1_demand_curves.pdf")
    plt.close(fig1)

    # ── Figure 2: MDP advantage — all 3 goods ────────────────────────────────
    MDP_STYLE = {
        "Truth":             ("k",       "-",   3.0, "Truth (Habit)"),
        "LA-AIDS":           ("#E53935", "--",  2.0, "LA-AIDS"),
        "BLP (IV)":          ("#9C27B0", "--",  2.0, "BLP (IV)"),
        "QUAIDS":            ("#43A047", "-.",  2.0, "QUAIDS"),
        "Series Est.":       ("#FB8C00", ":",   2.0, "Series Estimator"),
        "Window IRL":        ("#6D4C41", "--",  2.0, "Window IRL"),
        "Neural IRL static": ("#1E88E5", "-.",  2.0, "Neural IRL (static)"),
        "MDP-IRL":           ("#00897B", "-",   2.5, r"MDP-IRL (with $\bar{x}$)"),
        "MDP-IRL (E2E δ)":  ("#FF6F00", "--",  2.0, r"MDP-IRL (E2E $\hat{\delta}$)"),
    }
    good_names = ["Food", "Fuel", "Other"]
    for gi, gn in enumerate(good_names):
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, (col, ls, lw, lbl) in MDP_STYLE.items():
            mu    = mdp_mean[key][:, gi]
            sigma = mdp_se[key][:, gi]
            ax.plot(P_GRID, mu, color=col, ls=ls, lw=lw, label=lbl)
            ax.fill_between(P_GRID, mu - sigma, mu + sigma, color=col, alpha=BAND)
        ax.set_xlabel("Fuel price", fontsize=14)
        ax.set_ylabel(f"{gn} budget share", fontsize=14)
        ax.legend(fontsize=14, loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        base = f"{fig_dir}/fig2_{gn.lower()}_advantage"
        fig.savefig(f"{base}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
        print(f"    Saved: {base}.pdf")
        plt.close(fig)

    # ── Figure 3: Convergence ─────────────────────────────────────────────────
    conv_data = [
        ("Neural IRL\n(CES)",      last["hist_nirl"],       None,    None,  "#1E88E5"),
        ("Neural IRL\n(Habit)",    last["hist_nirl_hab"],   None,    None,  "#FB8C00"),
        ("MDP IRL\n(pre-comp x̄)", last["hist_mdp"],        None,    None,  "#00897B"),
        ("MDP E2E\n(frozen-δ)",    last["hist_mdp_e2e"],   "delta", 0.70,  "#FF6F00"),
        ("Window IRL\n(CES)",      last["hist_wirl_ces"],   None,    None,  "#6D4C41"),
        ("Window IRL\n(Habit)",    last["hist_wirl_hab"],   None,    None,  "#8D6E63"),
    ]
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    axes3 = axes3.flatten()
    for idx, (ttl, hist, param_key, true_val, col) in enumerate(conv_data):
        ax = axes3[idx]
        # hist is a list of dicts: [{"epoch": ..., "kl": ..., "beta": ..., "delta": ...}, ...]
        kl_vals = [h["kl"] for h in hist if "kl" in h] if hist else []
        if kl_vals:
            ax.plot(kl_vals, color=col, lw=1.5, label="KL loss")
        if param_key and hist and param_key in hist[0]:
            param_vals = [h[param_key] for h in hist if param_key in h]
            ax2 = ax.twinx()
            ax2.plot(param_vals, color="gray", lw=1.2, ls="--", label=f"learned {param_key}")
            if true_val is not None:
                ax2.axhline(true_val, color="k", ls=":", lw=1.0, label=f"true {param_key}={true_val}")
            ax2.set_ylabel(param_key, fontsize=9)
            ax2.legend(fontsize=8, loc="upper right")
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("KL loss", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    fig3.suptitle(f"Training Convergence — Representative Run  (N={N_RUNS} total runs)",
                  fontsize=12, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(f"{fig_dir}/fig3_convergence.pdf", dpi=150, bbox_inches="tight")
    fig3.savefig(f"{fig_dir}/fig3_convergence.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig3_convergence.pdf")
    plt.close(fig3)

    # ── Figure 4: DGP robustness ──────────────────────────────────────────────
    rob_agg  = agg["rob_agg"]
    dgp_names = list(rob_agg.keys())
    col_names = list(rob_agg[dgp_names[0]].keys())
    rob_col   = {"AIDS": "#E53935", "Lin IRL (Orth)": "#00ACC1", "Neural IRL": "#1E88E5"}
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    x        = np.arange(len(dgp_names))
    width    = 0.22
    for ci, col in enumerate(col_names):
        means = [rob_agg[dg][col]["mean"] for dg in dgp_names]
        ses   = [rob_agg[dg][col]["se"]   for dg in dgp_names]
        ax4.bar(x + ci * width, means, width, yerr=ses, capsize=4,
                color=rob_col.get(col, "#888"), label=col, alpha=0.85)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(dgp_names, fontsize=10)
    ax4.set_ylabel("Post-shock RMSE", fontsize=12)
    ax4.set_title("DGP Robustness", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f"{fig_dir}/fig4_robustness.pdf", dpi=150, bbox_inches="tight")
    fig4.savefig(f"{fig_dir}/fig4_robustness.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig4_robustness.pdf")
    plt.close(fig4)

    # ── Figure 5: Mixture components ─────────────────────────────────────────
    comp_summary = last["comp_summary"]
    if comp_summary is not None and len(comp_summary) > 0:
        n_comp = len(comp_summary)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        # comp_summary is a DataFrame; convert to list-of-dicts for .get() access
        _comp_recs = (comp_summary.to_dict("records")
                      if hasattr(comp_summary, "to_dict") else comp_summary)
        mus   = [c.get("mu_mean", c.get("rho", 0))        for c in _comp_recs]
        wts   = [c.get("weight",  c.get("pi", 1 / n_comp)) for c in _comp_recs]
        ax5.bar(range(n_comp), wts, color="#8E24AA", alpha=0.7)
        ax5.set_xlabel("Component", fontsize=12)
        ax5.set_ylabel("Mixture weight", fontsize=12)
        ax5.set_title("Variational Mixture Components", fontsize=12, fontweight="bold")
        ax5.grid(axis="y", alpha=0.3)
        fig5.tight_layout()
        fig5.savefig(f"{fig_dir}/fig5_mixture.pdf", dpi=150, bbox_inches="tight")
        fig5.savefig(f"{fig_dir}/fig5_mixture.png", dpi=150, bbox_inches="tight")
        print("    Saved: figures/fig5_mixture.pdf")
        plt.close(fig5)

    # ── Figure 6: δ recovery (KL profile over δ grid) ────────────────────────
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.plot(kl_delta_grid, kl_prof_e2e_mean, color="#FF6F00", lw=2.2,
             label=r"MDP-IRL E2E ($\hat{\delta}$)")
    ax6.fill_between(kl_delta_grid,
                     kl_prof_e2e_mean - kl_prof_e2e_se,
                     kl_prof_e2e_mean + kl_prof_e2e_se,
                     color="#FF6F00", alpha=0.2)
    ax6.axvline(0.7, color="k", ls="--", lw=1.8, label=r"True $\delta=0.7$")
    ax6.axvline(agg["delta_hat_e2e_mean"], color="#FF6F00", ls=":", lw=1.6,
                label=rf"$\hat{{\delta}}_{{E2E}}={agg['delta_hat_e2e_mean']:.2f}$ (mean)")
    ax6.set_xlabel(r"Candidate $\delta$", fontsize=13)
    ax6.set_ylabel("Validation KL divergence", fontsize=13)
    ax6.set_title(rf"$\delta$ Recovery: KL Profile over $\delta$ Grid  (mean ± SE, {N_RUNS} runs)",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(f"{fig_dir}/fig6_delta_recovery.pdf", dpi=150, bbox_inches="tight")
    fig6.savefig(f"{fig_dir}/fig6_delta_recovery.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig6_delta_recovery.pdf")
    plt.close(fig6)

    # ── Figure 7: Cross-price elasticity matrices ─────────────────────────────
    cross_elast_agg = agg["cross_elast_agg"]
    models_7 = list(cross_elast_agg.keys())
    n_models  = len(models_7)
    fig7, axes7 = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1: axes7 = [axes7]
    for ax, nm in zip(axes7, models_7):
        mat = cross_elast_agg[nm]["mean"]
        im  = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_title(nm, fontsize=9, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["Food", "Fuel", "Other"], fontsize=8)
        ax.set_yticklabels(["Food", "Fuel", "Other"], fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig7.suptitle("Cross-Price Elasticity Matrices (mean)", fontsize=11, fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(f"{fig_dir}/fig7_cross_elasticities.pdf", dpi=150, bbox_inches="tight")
    fig7.savefig(f"{fig_dir}/fig7_cross_elasticities.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig7_cross_elasticities.pdf")
    plt.close(fig7)

    # ── Figure 8: Habit welfare ───────────────────────────────────────────────
    welf_hab_agg = agg["welf_hab_agg"]
    _wh_models   = list(welf_hab_agg.keys())
    _wh_means    = [welf_hab_agg[m]["mean"] for m in _wh_models]
    _wh_ses      = [welf_hab_agg[m]["se"]   for m in _wh_models]
    gt_wh = welf_hab_agg["Ground Truth"]["mean"]

    fig8, ax8 = plt.subplots(figsize=(12, 5))
    col_map = {"Ground Truth": "#000000", "MDP Neural IRL": "#00897B",
               "MDP IRL (E2E δ)": "#FF6F00",
               "Neural IRL (static)": "#1E88E5"}
    colors_8 = [col_map.get(m, "#888888") for m in _wh_models]
    bars8 = ax8.barh(range(len(_wh_models)), _wh_means, xerr=_wh_ses,
                     capsize=4, color=colors_8, alpha=0.8)
    ax8.axvline(gt_wh, color="k", ls="--", lw=1.5, label=f"Ground truth CV={gt_wh:.4f}")
    ax8.set_yticks(range(len(_wh_models)))
    ax8.set_yticklabels(_wh_models, fontsize=9)
    ax8.set_xlabel("Structural CV (compensating variation)", fontsize=11)
    ax8.set_title("Welfare Analysis: Habit DGP (10% shock, fixed x̄_mean)", fontsize=11)
    ax8.legend(fontsize=9)
    ax8.grid(axis="x", alpha=0.3)
    fig8.tight_layout()
    fig8.savefig(f"{fig_dir}/fig8_habit_welfare.pdf", dpi=150, bbox_inches="tight")
    fig8.savefig(f"{fig_dir}/fig8_habit_welfare.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig8_habit_welfare.pdf")
    plt.close(fig8)

    # ── Figure 11: KL profile over δ (frozen weights) ────────────────────────
    kl_delta_grid    = agg.get("kl_delta_grid")
    kl_prof_e2e_mean = agg.get("kl_prof_e2e_mean")
    kl_prof_e2e_se   = agg.get("kl_prof_e2e_se")
    delta_hat_e2e    = agg.get("delta_hat_e2e_mean", true_delta)

    if kl_delta_grid is not None and kl_prof_e2e_mean is not None:
        _col_e2e = "#FF6F00"
        fig11, ax11 = plt.subplots(figsize=(9, 5))
        ax11.plot(kl_delta_grid, kl_prof_e2e_mean, color=_col_e2e, lw=2.5,
                  label=r"MDP-IRL E2E (learns $\hat{\delta}$)")
        if N_RUNS > 1 and kl_prof_e2e_se is not None:
            ax11.fill_between(kl_delta_grid,
                              kl_prof_e2e_mean - kl_prof_e2e_se,
                              kl_prof_e2e_mean + kl_prof_e2e_se,
                              color=_col_e2e, alpha=0.18)
        ax11.axvline(delta_hat_e2e, color=_col_e2e, ls=":", lw=1.8,
                     label=rf"$\hat{{\delta}}_{{E2E}}$ = {delta_hat_e2e:.3f}")
        ax11.axvline(true_delta, color="k", ls="--", lw=2.0,
                     label=f"True δ = {true_delta}")
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
        fig11.savefig(f"{fig_dir}/fig11_kl_delta_profile.pdf", dpi=150, bbox_inches="tight")
        fig11.savefig(f"{fig_dir}/fig11_kl_delta_profile.png", dpi=150, bbox_inches="tight")
        print("    Saved: figures/fig11_kl_delta_profile.pdf")
        plt.close(fig11)


# ─────────────────────────────────────────────────────────────────────────────
#  LATEX GENERATION (Sections 13)
# ─────────────────────────────────────────────────────────────────────────────

def _write_latex(agg: dict, N_RUNS: int, N_OBS: int, out_file: str = "paper_tables_figures.tex"):
    """Write LaTeX tables and figure inclusion blocks to out_file."""

    def cell(m, s, d=5):
        return f"{m:.{d}f} ({s:.{d}f})"

    def cell2(m, s): return cell(m, s, 2)

    out = []
    def lx(s): out.append(s)

    lx(r"% ============================================================")
    lx(r"% AUTO-GENERATED LaTeX — IRL Consumer Demand Recovery")
    lx(f"% N_RUNS = {N_RUNS}, N = {N_OBS} per run.  All cells: mean (SE).")
    lx(r"% Required: booktabs, threeparttable, graphicx, amsmath")
    lx(r"% ============================================================")
    lx("")

    model_names = agg["model_names"]
    perf_agg    = agg["perf_agg"]

    # Table 1: Predictive Accuracy
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
        bc = "}"        if bo              else ""
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

    # Table 5: MDP Advantage
    mdp_agg = agg["mdp_agg"]
    lx(r"% --- TABLE 5: MDP Advantage ---")
    lx(r"\begin{table}[htbp]")
    lx(r"  \centering")
    lx(r"  \caption{Habit DGP: Post-Shock RMSE and KL Divergence. "
       r"Mean (SE) across " + str(N_RUNS) + r" runs.}")
    lx(r"  \label{tab:mdp_advantage}")
    lx(r"  \begin{tabular}{lcc}")
    lx(r"    \toprule")
    lx(r"    \textbf{Model} & \textbf{RMSE} & \textbf{KL} \\")
    lx(r"    \midrule")
    for lbl, kr, kk in [
        ("LA-AIDS",           "aids_rmse",       "kl_aids"),
        ("BLP (IV)",          "blp_rmse",        "kl_blp"),
        ("QUAIDS",            "quaids_rmse",     "kl_quaids"),
        ("Series Est.",       "series_rmse",     "kl_series"),
        ("Window IRL",        "wirl_rmse",       "kl_wirl"),
        ("Neural IRL (static)", "nirl_rmse",     "kl_static"),
        ("MDP Neural IRL",    "mdp_rmse",        "kl_mdp"),
        ("MDP IRL (E2E)",     "mdp_e2e_rmse",    "kl_mdp_e2e"),
    ]:
        lx(f"    {lbl} & {cell(mdp_agg[kr]['mean'],mdp_agg[kr]['se'])} "
           f"& {cell(mdp_agg[kk]['mean'],mdp_agg[kk]['se'])} \\\\")
    lx(r"    \bottomrule")
    lx(r"  \end{tabular}")
    lx(r"\end{table}")
    lx("")

    # Figures inclusion
    for fig_num, caption, label in [
        (1, "Demand Curves — CES DGP", "fig:demand_ces"),
        (6, r"$\delta$ Recovery: KL Profile over $\delta$ Grid", "fig:delta_recovery"),
        (8, "Welfare Analysis: Habit DGP", "fig:habit_welfare"),
    ]:
        lx(r"\begin{figure}[htbp]")
        lx(r"  \centering")
        lx(f"  \\includegraphics[width=0.85\\linewidth]{{figures/fig{fig_num}_")
        lx(r"  \caption{" + caption + r"}")
        lx(f"  \\label{{{label}}}")
        lx(r"\end{figure}")
        lx("")

    latex_str = "\n".join(out)
    with open(out_file, "w") as f:
        f.write(latex_str)
    print(f"\n  LaTeX written to {out_file}  ({len(out)} lines)")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY (Section 14)
# ─────────────────────────────────────────────────────────────────────────────

def _print_final_summary(agg: dict, N_RUNS: int, N_OBS: int):
    def fmt(m, s, d=5): return f"{m:.{d}f} ({s:.{d}f})"

    welf_agg = agg["welf_agg"]
    mdp_agg  = agg["mdp_agg"]
    gt_welf  = welf_agg["Ground Truth"]["mean"]

    print("\n" + "=" * 72)
    print(f"  FINAL SUMMARY  ({N_RUNS} runs × N={N_OBS})")
    print("=" * 72)
    print(f"""
  NEURAL IRL (CES DGP):
    RMSE : {fmt(agg['perf_agg']['Neural IRL']['RMSE_mean'], agg['perf_agg']['Neural IRL']['RMSE_se'])}
    Welfare error vs truth: {100*abs(welf_agg['Neural IRL']['mean']-gt_welf)/abs(gt_welf):.1f}%

  MDP ADVANTAGE (Habit DGP):
    LA-AIDS RMSE       : {fmt(mdp_agg['aids_rmse']['mean'],       mdp_agg['aids_rmse']['se'])}
    QUAIDS RMSE        : {fmt(mdp_agg['quaids_rmse']['mean'],     mdp_agg['quaids_rmse']['se'])}
    Series Est. RMSE   : {fmt(mdp_agg['series_rmse']['mean'],     mdp_agg['series_rmse']['se'])}
    Window IRL RMSE    : {fmt(mdp_agg['wirl_rmse']['mean'],       mdp_agg['wirl_rmse']['se'])}
    Static IRL RMSE    : {fmt(mdp_agg['nirl_rmse']['mean'],       mdp_agg['nirl_rmse']['se'])}
    MDP IRL RMSE       : {fmt(mdp_agg['mdp_rmse']['mean'],        mdp_agg['mdp_rmse']['se'])}
    MDP IRL E2E RMSE   : {fmt(mdp_agg['mdp_e2e_rmse']['mean'],    mdp_agg['mdp_e2e_rmse']['se'])}

  DELTA RECOVERY (true δ = 0.7):
    MDP IRL (blend only): δ̂ = {agg['delta_mdp_mean']:.4f} ± {agg['delta_mdp_se']:.4f}
    MDP IRL E2E:          δ̂ = {agg['delta_mdp_e2e_mean']:.4f} ± {agg['delta_mdp_e2e_se']:.4f}

  OUTPUT FILES:
    figures/fig{{1..8}}_*.{{pdf,png}}
    paper_tables_figures.tex
""")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> tuple[list, dict]:
    """Run all N_RUNS seeds, aggregate, print, plot, write LaTeX.

    Parameters
    ----------
    cfg : dict with keys N_RUNS, N_OBS, DEVICE, MDP_DELTA_GRID,
          and optional fig_dir (default "figures"), latex_file.

    Returns
    -------
    (all_results, aggregated) : list of per-seed dicts + aggregated dict.
    """
    N_RUNS         = cfg["N_RUNS"]
    N_OBS          = cfg["N_OBS"]
    fig_dir        = cfg.get("fig_dir", "figures")
    latex_file     = cfg.get("latex_file", "paper_tables_figures.tex")

    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 72)
    print(f"  IRL CONSUMER DEMAND RECOVERY  —  {N_RUNS} INDEPENDENT RUNS")
    print("=" * 72)
    print(f"  Device: {cfg['DEVICE']}  |  N per run: {N_OBS}  |  Runs: {N_RUNS}\n")

    all_results = []
    for run_idx in range(N_RUNS):
        seed = 42 + run_idx * 15
        t0   = time.time()
        print(f"── Run {run_idx+1}/{N_RUNS}  (seed={seed}) ────────────────────────────")
        r = run_one_seed(seed, cfg, verbose=(run_idx == N_RUNS - 1))
        all_results.append(r)
        print(f"   Done in {time.time()-t0:.0f}s  "
              f"| Neural IRL RMSE={r['perf_ces']['Neural IRL']['RMSE']:.5f}"
              f"  δ={r['delta_mdp']:.3f}")

    agg = _aggregate(all_results, N_RUNS)
    _print_summary(agg, N_RUNS, N_OBS)
    _make_figures(agg, N_RUNS, fig_dir=fig_dir,
                  true_delta=cfg.get("TRUE_DELTA", 0.7))
    _write_latex(agg, N_RUNS, N_OBS, out_file=latex_file)
    _print_final_summary(agg, N_RUNS, N_OBS)

    return all_results, agg
