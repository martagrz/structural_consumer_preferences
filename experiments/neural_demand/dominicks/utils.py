"""
experiments/neural_demand/dominicks/utils.py
=============================================
Prediction, evaluation, and welfare utilities for the Neural Demand paper's
Dominick's analgesics experiments.

This module re-exports the low-level `pred`, `own_elasticity`,
`full_elasticity_matrix`, `comp_var`, `get_metrics`, `kl_div`, and
`mdp_price_cond_habit` helpers from `experiments.dominicks.utils` and
augments them with:

  - paper-aligned model name constants
  - a `predict` wrapper that uses Neural Demand paper naming conventions
    ("neural-demand-static", "neural-demand-habit")
  - `fit_nd_delta_grid_dom` — frozen-δ profile sweep for the Neural Demand
    paper (thin wrapper around `fit_mdp_delta_grid_dom`)
  - figure and table helpers shared across all Dominick's experiments
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from experiments.dominicks.data import G, GOODS
from experiments.dominicks.utils import (
    pred as _pred_dom,
    own_elasticity as _own_elast_dom,
    full_elasticity_matrix as _full_elast_dom,
    comp_var as _comp_var_dom,
    get_metrics as _get_metrics_dom,
    kl_div as _kl_div_dom,
    mdp_price_cond_habit,
    fit_mdp_delta_grid_dom,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAPER MODEL NAMES  (Neural Demand paper ordering)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAMES_STATIC = [
    "LA-AIDS", "BLP (IV)", "QUAIDS", "Series Est.",
    "LDS (Shared)", "LDS (GoodSpec)", "LDS (Orth)",
    "Neural Demand (static)",
]
MODEL_NAMES_HABIT = MODEL_NAMES_STATIC + ["Neural Demand (habit)"]
MODEL_NAMES_FE    = MODEL_NAMES_HABIT  + [
    "Neural Demand (FE)", "Neural Demand (habit, FE)",
]
MODEL_NAMES_CF = MODEL_NAMES_FE + [
    "Neural Demand (CF)", "Neural Demand (habit, CF)",
]
ALL_MODEL_NAMES = MODEL_NAMES_CF

# Map: paper name → internal spec used by pred()
_SPEC_MAP = {
    "LA-AIDS":                       "aids",
    "BLP (IV)":                      "blp",
    "QUAIDS":                        "quaids",
    "Series Est.":                   "series",
    "LDS (Shared)":                  "lirl",
    "LDS (GoodSpec)":                "lirl",
    "LDS (Orth)":                    "lirl",
    "Neural Demand (static)":        "nirl",
    "Neural Demand (habit)":         "mdp-e2e",
    "Neural Demand (FE)":            "nirl-fe",
    "Neural Demand (habit, FE)":     "mdp-e2e-fe",
    "Neural Demand (CF)":            "nirl-cf",
    "Neural Demand (habit, CF)":     "mdp-cf",
}

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR / STYLE MAP
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "LA-AIDS":                    dict(color="#E53935", ls="--", lw=1.8),
    "BLP (IV)":                   dict(color="#8E24AA", ls="--", lw=1.8),
    "QUAIDS":                     dict(color="#43A047", ls="-.", lw=1.8),
    "Series Est.":                dict(color="#FB8C00", ls=":",  lw=1.8),
    "LDS (Shared)":               dict(color="#039BE5", ls=":",  lw=1.5),
    "LDS (GoodSpec)":             dict(color="#00ACC1", ls=":",  lw=1.5),
    "LDS (Orth)":                 dict(color="#0277BD", ls=":",  lw=1.5),
    "Neural Demand (static)":     dict(color="#1E88E5", ls="-",  lw=2.5),
    "Neural Demand (habit)":      dict(color="#00897B", ls="-",  lw=2.5),
    "Neural Demand (FE)":         dict(color="#1565C0", ls="-",  lw=2.0),
    "Neural Demand (habit, FE)":  dict(color="#004D40", ls="-",  lw=2.0),
    "Neural Demand (CF)":         dict(color="#283593", ls="--", lw=2.0),
    "Neural Demand (habit, CF)":  dict(color="#1B5E20", ls="--", lw=2.0),
}

# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION (paper names → internal specs)
# ─────────────────────────────────────────────────────────────────────────────

def predict(paper_name: str, p, y, cfg: dict,
            xb_prev=None, q_prev=None,
            store_idx=None, s_te_mode_idx: int = 0,
            **kw):
    """Predict budget shares using a paper-facing model name.

    Wraps `experiments.dominicks.utils.pred` and maps Neural Demand paper
    names to internal spec strings.

    Parameters
    ----------
    paper_name : str  — one of the keys in MODEL_NAMES_* lists
    p          : (N, G) price array
    y          : (N,) income array
    cfg        : experiment config dict (passed through to pred)
    xb_prev    : (N, G) log-share habit stock (for habit models)
    q_prev     : (N, G) log-quantity of previous period (for MDP models)
    store_idx  : (N,) int store indices (for FE models)
    s_te_mode_idx : int  — modal test-store index (fallback for FE models)
    **kw       : additional keyword args forwarded to pred
                 (aids=, blp=, quaids=, series=, nirl=, mdp_e2e=, …)
    """
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _pred_dom(spec, p, y, cfg,
                     xb_prev=xb_prev, q_prev=q_prev,
                     store_idx=store_idx,
                     s_te_mode_idx=s_te_mode_idx,
                     **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC WRAPPERS  (paper names)
# ─────────────────────────────────────────────────────────────────────────────

def metrics(paper_name: str, p, y, w_true, cfg: dict,
            xb_prev=None, q_prev=None, **kw) -> dict:
    """Return {'RMSE': …, 'MAE': …} for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _get_metrics_dom(spec, p, y, w_true, cfg,
                            xb_prev=xb_prev, q_prev=q_prev, **kw)


def kl_divergence(paper_name: str, p, y, w_true, cfg: dict,
                  xb_prev=None, q_prev=None, **kw) -> float:
    """KL(truth‖pred) for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _kl_div_dom(spec, p, y, w_true, cfg,
                       xb_prev=xb_prev, q_prev=q_prev, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  ELASTICITY / WELFARE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def own_elasticity(paper_name: str, p0, y0, cfg: dict,
                   xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    """Own-price elasticities vector (G,) for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _own_elast_dom(spec, p0, y0, cfg,
                          xb_prev0=xb_prev0, q_prev0=q_prev0, h=h, **kw)


def elasticity_matrix(paper_name: str, p0, y0, cfg: dict,
                      xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    """(G, G) price elasticity matrix for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _full_elast_dom(spec, p0, y0, cfg,
                           xb_prev0=xb_prev0, q_prev0=q_prev0, h=h, **kw)


def compensating_variation(paper_name: str, p0, p1, y, cfg: dict,
                            xb_prev0=None, q_prev0=None, **kw) -> float:
    """Compensating variation for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _comp_var_dom(spec, p0, p1, y, cfg,
                         xb_prev0=xb_prev0, q_prev0=q_prev0, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  δ SWEEP (Neural Demand paper alias)
# ─────────────────────────────────────────────────────────────────────────────

def fit_nd_delta_grid_dom(
    p_tr, y_tr, w_tr, ls_tr,
    p_val, y_val, w_val, ls_val,
    cfg: dict,
    delta_grid=None,
    store_ids_tr=None,
    store_ids_val=None,
    store_idx_tr=None,
    store_idx_val=None,
    hidden=256,
    with_fe: bool = False,
    n_stores=None,
    store_emb_dim=None,
    tag: str = "nd-habit-dom",
):
    """Frozen-δ profile sweep for Neural Demand (habit) on Dominick's data.

    This is a thin wrapper around `fit_mdp_delta_grid_dom` that chooses
    the right model class (E2E vs E2E-FE) and spec string automatically.

    Returns the same dict as `fit_mdp_delta_grid_dom`.
    """
    from src.models.dominicks import MDPNeuralIRL_E2E, MDPNeuralIRL_E2E_FE

    if with_fe:
        n_stores = n_stores if n_stores is not None else cfg.get("n_stores", 1)
        emb_dim  = store_emb_dim if store_emb_dim is not None else cfg.get("store_emb_dim", 8)
        return fit_mdp_delta_grid_dom(
            p_tr, y_tr, w_tr, ls_tr,
            p_val, y_val, w_val, ls_val,
            cfg,
            delta_grid=delta_grid,
            store_ids_tr=store_ids_tr, store_ids_val=store_ids_val,
            store_idx_tr=store_idx_tr, store_idx_val=store_idx_val,
            hidden=hidden,
            model_class=MDPNeuralIRL_E2E_FE,
            extra_model_kw={"n_stores": n_stores, "emb_dim": emb_dim},
            pred_spec="mdp-e2e-fe",
            pred_model_key="mdp_e2e_fe",
            tag=tag,
        )
    return fit_mdp_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_val, y_val, w_val, ls_val,
        cfg,
        delta_grid=delta_grid,
        store_ids_tr=store_ids_tr, store_ids_val=store_ids_val,
        hidden=hidden,
        pred_spec="mdp-e2e",
        pred_model_key="mdp_e2e",
        tag=tag,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr
                    if x is not None and not np.isnan(float(x))], float)
    if len(a) < 2:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def aggregate_runs(all_results: list, model_names: list) -> dict:
    """Aggregate metrics over N simulation/bootstrap runs.

    Parameters
    ----------
    all_results : list of dicts from run_once (each with 'perf' sub-dict)
    model_names : list of paper-facing model names to aggregate

    Returns
    -------
    dict with {model_name: {'RMSE_mean', 'RMSE_se', 'MAE_mean', 'MAE_se',
                             'KL_mean', 'KL_se'}}
    """
    n = len(all_results)

    def _mean_se(key, nm):
        vals = []
        for r in all_results:
            v = (r.get("perf", {}).get(nm, {}).get(key)
                 or r.get(key, {}).get(nm))
            if v is not None:
                vals.append(float(v))
        if not vals:
            return np.nan, np.nan
        return float(np.nanmean(vals)), _se(vals)

    out = {}
    for nm in model_names:
        rm, rs = _mean_se("RMSE", nm)
        mm, ms = _mean_se("MAE",  nm)
        km, ks = _mean_se("KL",   nm)
        out[nm] = {"RMSE_mean": rm, "RMSE_se": rs,
                   "MAE_mean":  mm, "MAE_se":  ms,
                   "KL_mean":   km, "KL_se":   ks}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED FIGURE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

BAND = 0.15   # alpha for ±1 SE bands


def bar_chart(means: dict, ses: dict, ylabel: str, title: str,
              ax=None, n_runs: int = 1) -> plt.Axes:
    """Generic horizontal bar chart for model comparison."""
    names  = list(means.keys())
    vals   = np.asarray([means[n] for n in names], float)
    errs   = np.asarray([ses.get(n, 0.0) for n in names], float)
    colors = [STYLE.get(nm, {}).get("color", "#888") for nm in names]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(names))
    ax.bar(x, vals, yerr=errs if n_runs > 1 else None,
           capsize=5, color=colors, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    # ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    return ax


def demand_curve_plot(curves: dict, pgr, shock_good_idx: int,
                      ax=None, n_runs: int = 1,
                      ses: dict | None = None,
                      title: str = "Demand Curves") -> plt.Axes:
    """Plot budget shares vs. price for each model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for nm, mu in curves.items():
        sty = STYLE.get(nm, {})
        ax.plot(pgr, mu, label=nm, **sty)
        if n_runs > 1 and ses is not None and nm in ses:
            sig = ses[nm]
            ax.fill_between(pgr, mu - sig, mu + sig,
                            color=sty.get("color", "#888"), alpha=BAND)

    ax.set_xlabel(f"Good-{shock_good_idx} price", fontsize=13)
    ax.set_ylabel(f"Budget share $w_{{0}}$", fontsize=13)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED TABLE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_performance_table(perf_agg: dict, out_dir: str, label: str,
                            caption: str, n_runs: int = 1) -> None:
    """Write CSV + LaTeX performance table.

    Parameters
    ----------
    perf_agg : dict from aggregate_runs with {model: {RMSE_mean, …}}
    out_dir  : output directory
    label    : LaTeX label stem (e.g. 'dom_main_perf')
    caption  : LaTeX caption string
    n_runs   : number of runs (for SE annotation)
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for nm, d in perf_agg.items():
        rows.append({
            "Model":     nm,
            "RMSE_mean": d.get("RMSE_mean", np.nan),
            "RMSE_se":   d.get("RMSE_se",   0.0),
            "MAE_mean":  d.get("MAE_mean",  np.nan),
            "MAE_se":    d.get("MAE_se",    0.0),
            "KL_mean":   d.get("KL_mean",   np.nan),
            "KL_se":     d.get("KL_se",     0.0),
        })
    df = pd.DataFrame(rows)
    csv_path = f"{out_dir}/{label}.csv"
    df.round(6).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    def _c(m, s, d=5):
        if np.isnan(float(m)):
            return "{---}"
        if n_runs > 1 and s > 0:
            return f"${float(m):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(m):.{d}f}$"

    lines = [
        r"% ============================================================",
        f"% Neural Demand — {label} (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label}}}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{KL Div.} \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['Model']} & "
            f"{_c(row['RMSE_mean'], row['RMSE_se'])} & "
            f"{_c(row['MAE_mean'],  row['MAE_se'])} & "
            f"{_c(row['KL_mean'],   row['KL_se'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        rf"    \item Mean $\pm$ SE over {n_runs} run(s). "
        r"RMSE and MAE computed on held-out test weeks.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}", "",
    ]
    tex_path = f"{out_dir}/{label}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


def make_elasticity_table(elast_rows: list, out_dir: str, label: str,
                           caption: str, n_runs: int = 1) -> None:
    """Write CSV + LaTeX table of own-price elasticities.

    Parameters
    ----------
    elast_rows : list of dicts with keys 'Model', 'eps_0', 'eps_1', 'eps_2'
                 (and optional '_se' variants)
    """
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(elast_rows).round(4).to_csv(
        f"{out_dir}/{label}.csv", index=False)
    print(f"  Saved: {out_dir}/{label}.csv")

    def _c(v, s=None, d=3):
        if v is None or np.isnan(float(v)):
            return "{---}"
        if n_runs > 1 and s is not None and not np.isnan(float(s)):
            return f"${float(v):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(v):.{d}f}$"

    lines = [
        r"% ============================================================",
        f"% Neural Demand — {label} (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label}}}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Model} & "
        r"\textbf{$\epsilon_{00}$ (ASP)} & "
        r"\textbf{$\epsilon_{11}$ (ACET)} & "
        r"\textbf{$\epsilon_{22}$ (IBU)} \\",
        r"    \midrule",
    ]
    for row in elast_rows:
        nm = row["Model"]
        e0 = _c(row.get("eps_0"), row.get("eps_0_se"))
        e1 = _c(row.get("eps_1"), row.get("eps_1_se"))
        e2 = _c(row.get("eps_2"), row.get("eps_2_se"))
        lines.append(f"    {nm} & {e0} & {e1} & {e2} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    tex_path = f"{out_dir}/{label}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")
