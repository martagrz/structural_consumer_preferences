"""
experiments/neural_demand/dominicks/exp01_predictive_accuracy.py
=================================================================
Section 5.1 — Predictive Accuracy on Dominick's Analgesics.

Trains all Neural Demand paper models on the Dominick's training split and
evaluates out-of-sample RMSE, MAE, and KL divergence on the held-out test
weeks.

Models evaluated
----------------
  LA-AIDS, BLP (IV), QUAIDS, Series Est.,
  LDS (Shared), LDS (GoodSpec), LDS (Orth),
  Neural Demand (static), Neural Demand (habit),
  Neural Demand (FE), Neural Demand (habit, FE)

Produces
--------
results/neural_demand/dominicks/
  table_dom_predictive_accuracy.csv / .tex
  fig_dom_predictive_accuracy.{pdf,png}
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dominicks import (
    LAAIDS, BLPLogitIV, QUAIDS, SeriesDemand,
    NeuralIRL, NeuralIRL_FE,
    MDPNeuralIRL,
    MDPNeuralIRL_E2E, MDPNeuralIRL_E2E_FE,
    _train,
    cf_first_stage,
    compute_xbar_e2e,
    feat_good_specific, feat_orth, feat_shared,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.data import G as _G
from experiments.neural_demand.dominicks.utils import (
    predict,
    metrics,
    kl_divergence,
    fit_nd_delta_grid_dom,
    ALL_MODEL_NAMES,
    STYLE,
    bar_chart,
    make_performance_table,
    aggregate_runs,
    BAND,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Train all models for one seed and return performance dicts."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    mw_tr = splits["mw_tr"]
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    xb_tr = splits["xb_tr"]; xb_te = splits["xb_te"]
    qp_tr = splits["qp_tr"]; qp_te = splits["qp_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]
    Z_tr  = splits["Z_tr"]

    N_STORES      = splits["N_STORES"]
    STORE_EMB_DIM = splits["STORE_EMB_DIM"]
    s_tr_idx      = splits["s_tr_idx"]
    s_te_idx      = splits["s_te_idx"]
    s_te_mode_idx = splits["s_te_mode_idx"]
    dev           = cfg["device"]

    # ── Static benchmarks ─────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)

    th_sh   = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs   = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_orth = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    # ── Neural Demand (static) ─────────────────────────────────────────────────
    nirl_m, _ = _train(
        NeuralIRL(cfg["nirl_hidden"]),
        p_tr, y_tr, w_tr, "nirl", cfg,
        tag=f"NDS s={seed}")

    # ── Neural Demand (habit) ─────────────────────────────────────────────────
    sw_hab = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit s={seed}")
    nd_hab_m  = sw_hab["best_model"]
    delta_hat = sw_hab["delta_hat"]

    # Habit stock for test set
    d_t = torch.tensor(float(delta_hat), dtype=torch.float32, device=dev)
    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_e2e_te = compute_xbar_e2e(d_t, ls_te_t, store_ids=s_te).cpu().numpy()

    # ── Store-FE variants ─────────────────────────────────────────────────────
    nirl_fe_m, _ = _train(
        NeuralIRL_FE(cfg["nirl_hidden"], n_stores=N_STORES, emb_dim=STORE_EMB_DIM),
        p_tr, y_tr, w_tr, "nirl", cfg,
        store_idx_tr=s_tr_idx,
        tag=f"NDS-FE s={seed}")

    sw_fe = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=True,
        n_stores=N_STORES, store_emb_dim=STORE_EMB_DIM,
        store_ids_tr=s_tr, store_ids_val=s_te,
        store_idx_tr=s_tr_idx, store_idx_val=s_te_idx,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit-FE s={seed}")
    nd_fe_m  = sw_fe["best_model"]
    dfe_hat  = sw_fe["delta_hat"]

    d_fe_t = torch.tensor(float(dfe_hat), dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_fe_te = compute_xbar_e2e(d_fe_t, ls_te_t, store_ids=s_te).cpu().numpy()

    # ── CF first-stage residuals (Hausman instruments) ────────────────────────
    v_hat_tr, cf_rsq = cf_first_stage(
        np.log(np.maximum(p_tr, 1e-8)), Z_tr)
    if cfg.get("verbose"):
        print(f"    CF first-stage R²: {cf_rsq.round(3)}")

    # ── Neural Demand (CF) ─────────────────────────────────────────────────────
    nirl_cf_m, _ = _train(
        NeuralIRL(cfg["nirl_hidden"], n_cf=_G),
        p_tr, y_tr, w_tr, "nirl", cfg,
        v_hat_tr=v_hat_tr,
        tag=f"NDS-CF s={seed}")

    # ── Neural Demand (habit, CF) — MDPNeuralIRL with precomputed xbar ────────
    mdp_hidden = cfg.get("mdp_hidden", cfg["nirl_hidden"])
    mdp_cf_m, _ = _train(
        MDPNeuralIRL(mdp_hidden, n_cf=_G),
        p_tr, y_tr, w_tr, "mdp", cfg,
        xb_prev_tr=xb_tr, q_prev_tr=qp_tr,
        v_hat_tr=v_hat_tr,
        tag=f"ND-Habit-CF s={seed}")

    # ── Keyword-argument bundles for predict() ────────────────────────────────
    KW = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        ff=feat_shared, theta=th_sh,          # LDS (Shared)
        nirl=nirl_m,
        nirl_cf=nirl_cf_m,
        mdp_cf=mdp_cf_m,
        mdp_e2e=nd_hab_m,
        nirl_fe=nirl_fe_m,
        mdp_e2e_fe=nd_fe_m,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # helper: get xb for habit models
    def _xb(paper_name):
        if "FE" in paper_name:
            return xb_fe_te
        if "CF" in paper_name and "habit" in paper_name:
            return xb_te   # precomputed EWMA for MDPNeuralIRL-CF
        if "habit" in paper_name:
            return xb_e2e_te
        return xb_te   # for MDPNeuralIRL (non-E2E)

    def _si(paper_name):
        if "FE" in paper_name:
            return s_te_idx
        return None

    # LDS variants need different ff/theta — merged into KW to avoid duplicate-kwarg error
    def _kw_extra(paper_name):
        if paper_name == "LDS (GoodSpec)":
            return {"ff": feat_good_specific, "theta": th_gs}
        if paper_name == "LDS (Orth)":
            return {"ff": feat_orth, "theta": th_orth}
        return {}

    perf = {}
    kl   = {}
    for nm in ALL_MODEL_NAMES:
        try:
            _kw = {**KW, **_kw_extra(nm)}
            m = metrics(nm, p_te, y_te, w_te, cfg,
                        xb_prev=_xb(nm), q_prev=qp_te,
                        store_idx=_si(nm),
                        s_te_mode_idx=s_te_mode_idx,
                        **_kw)
            kl_v = kl_divergence(nm, p_te, y_te, w_te, cfg,
                                  xb_prev=_xb(nm), q_prev=qp_te,
                                  store_idx=_si(nm),
                                  s_te_mode_idx=s_te_mode_idx,
                                  **_kw)
            perf[nm] = {**m, "KL": kl_v}
        except Exception as exc:
            print(f"    Warning: {nm} evaluation failed — {exc}")
            perf[nm] = {"RMSE": np.nan, "MAE": np.nan, "KL": np.nan}

    return dict(
        perf=perf,
        delta_hat=delta_hat,
        delta_hat_fe=dfe_hat,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list) -> dict:
    return aggregate_runs(all_results, ALL_MODEL_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(perf_agg: dict, cfg: dict, n_runs: int = 1) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""

    model_names = list(perf_agg.keys())
    rmse_means  = {nm: perf_agg[nm]["RMSE_mean"] for nm in model_names}
    rmse_ses    = {nm: perf_agg[nm]["RMSE_se"]   for nm in model_names}
    kl_means    = {nm: perf_agg[nm]["KL_mean"]   for nm in model_names}
    kl_ses      = {nm: perf_agg[nm]["KL_se"]     for nm in model_names}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bar_chart(rmse_means, rmse_ses, "Out-of-Sample RMSE",
              f"RMSE — Dominick's Analgesics{se_note}",
              ax=axes[0], n_runs=n_runs)
    bar_chart(kl_means, kl_ses, "KL Divergence KL(truth‖pred)",
              f"KL Divergence — Dominick's Analgesics{se_note}",
              ax=axes[1], n_runs=n_runs)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_dom_predictive_accuracy.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_dom_predictive_accuracy.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    """Run predictive accuracy experiment over multiple seeds.

    Parameters
    ----------
    splits : from experiments.dominicks.data.load()
    cfg    : config dict (must include 'N_RUNS', 'out_dir', 'fig_dir')
    """
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 01: Predictive Accuracy")
    print("=" * 68)

    all_results = []
    for ri in range(N_RUNS):
        seed = 500 + ri * 11
        t0   = time.time()
        print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
        r = run_once(seed, splits, cfg)
        all_results.append(r)
        nd_rmse = r["perf"].get("Neural Demand (static)", {}).get("RMSE", np.nan)
        nh_rmse = r["perf"].get("Neural Demand (habit)",  {}).get("RMSE", np.nan)
        print(f"    Done in {time.time()-t0:.0f}s  "
              f"NDS_RMSE={nd_rmse:.5f}  ND+Habit_RMSE={nh_rmse:.5f}")

    perf_agg = aggregate(all_results)
    make_figures(perf_agg, cfg, n_runs=N_RUNS)
    make_performance_table(
        perf_agg,
        out_dir=cfg["out_dir"],
        label="table_dom_predictive_accuracy",
        caption=(r"Predictive Accuracy --- Dominick's Analgesics. "
                 r"Out-of-sample RMSE, MAE, and KL divergence on held-out test weeks. "
                 r"Best result per column in \textbf{bold}."),
        n_runs=N_RUNS,
    )

    print("\n── Predictive Accuracy Summary ─────────────────────────────────────")
    for nm, d in perf_agg.items():
        print(f"  {nm:40s}  RMSE={d['RMSE_mean']:.5f}±{d['RMSE_se']:.5f}"
              f"  KL={d['KL_mean']:.5f}±{d['KL_se']:.5f}")

    return all_results, perf_agg
