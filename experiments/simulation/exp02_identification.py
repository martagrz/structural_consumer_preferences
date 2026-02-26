"""
experiments/simulation/exp02_identification.py
===============================================
Section 15 of main_multiple_runs.py.

Identification & Robustness Checks:
  - Priority 1: δ identification sweep across true δ ∈ {0.3, 0.5, 0.7, 0.9}
  - Priority 3: θ robustness — MDP advantage scales with habit strength

Generates Figures 9 (δ identification) and 10 (θ robustness).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from experiments.simulation.exp01_main_runs import run_habit_param_seed


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict:
    """Run δ-identification and θ-robustness sweeps; produce Figs 9 & 10.

    Parameters
    ----------
    cfg : dict with keys N_RUNS, N_OBS, DEVICE, MDP_DELTA_GRID,
          optional DELTA_GRID, THETA_GRID, N_SWEEP_SEEDS, SWEEP_EPOCHS,
          fig_dir.
    """
    N_RUNS        = cfg["N_RUNS"]
    fig_dir       = cfg.get("fig_dir", "figures")

    DELTA_GRID    = cfg.get("DELTA_GRID",    [0.3, 0.5, 0.7, 0.9])
    THETA_GRID    = cfg.get("THETA_GRID",    [0.0, 0.1, 0.2, 0.3, 0.5])
    N_SWEEP_SEEDS = cfg.get("N_SWEEP_SEEDS", min(N_RUNS, 3))
    SWEEP_EPOCHS  = cfg.get("SWEEP_EPOCHS",  333)

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"  SECTION 15 — IDENTIFICATION & ROBUSTNESS CHECKS")
    print("=" * 72)

    # ── δ identification sweep ────────────────────────────────────────────────
    print(f"\n  [P1] δ identification: {len(DELTA_GRID)} values × "
          f"{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs ...")

    _delta_rows = []
    for _td in DELTA_GRID:
        for _si in range(N_SWEEP_SEEDS):
            _seed_d = 500 + _si * 13
            t0d = time.time()
            print(f"    true_δ={_td:.1f}  seed={_seed_d}", end="", flush=True)
            _r = run_habit_param_seed(_seed_d, cfg, delta=_td, theta=0.3,
                                      epochs=SWEEP_EPOCHS)
            print(f"  →  δ̂_blend={_r['delta_blend']:.3f}"
                  f"  δ̂_e2e={_r['delta_e2e']:.3f}"
                  f"  ({time.time()-t0d:.0f}s)")
            _delta_rows.append(_r)

    # ── θ robustness sweep ────────────────────────────────────────────────────
    print(f"\n  [P3] θ robustness: {len(THETA_GRID)} values × "
          f"{N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs ...")

    _theta_rows = []
    for _tt in THETA_GRID:
        for _si in range(N_SWEEP_SEEDS):
            _seed_t = 600 + _si * 13
            t0t = time.time()
            print(f"    true_θ={_tt:.1f}  seed={_seed_t}", end="", flush=True)
            _r = run_habit_param_seed(_seed_t, cfg, delta=0.7, theta=_tt,
                                      epochs=SWEEP_EPOCHS)
            print(f"  →  rmse_mdp={_r['rmse_mdp']:.5f}"
                  f"  rmse_nirl={_r['rmse_nirl']:.5f}"
                  f"  ({time.time()-t0t:.0f}s)")
            _theta_rows.append(_r)

    # ── Figure 9: δ identification ────────────────────────────────────────────
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    _col_blend = "#00897B"; _col_e2e = "#FF6F00"

    for _td in DELTA_GRID:
        _rows_d = [r for r in _delta_rows if r["true_delta"] == _td]
        _db  = [r["delta_blend"] for r in _rows_d]
        _de  = [r["delta_e2e"]   for r in _rows_d]
        ax9.scatter([_td] * len(_db), _db, marker="o", color=_col_blend, s=60, alpha=0.55, zorder=4)
        ax9.scatter([_td] * len(_de), _de, marker="^", color=_col_e2e,   s=60, alpha=0.55, zorder=4)

    _blend_means = [np.mean([r["delta_blend"] for r in _delta_rows if r["true_delta"] == _td]) for _td in DELTA_GRID]
    _e2e_means   = [np.mean([r["delta_e2e"]   for r in _delta_rows if r["true_delta"] == _td]) for _td in DELTA_GRID]
    _blend_se    = [np.std([r["delta_blend"]  for r in _delta_rows if r["true_delta"] == _td],
                            ddof=1) / np.sqrt(N_SWEEP_SEEDS) for _td in DELTA_GRID]
    _e2e_se      = [np.std([r["delta_e2e"]   for r in _delta_rows if r["true_delta"] == _td],
                            ddof=1) / np.sqrt(N_SWEEP_SEEDS) for _td in DELTA_GRID]

    ax9.errorbar(DELTA_GRID, _blend_means, yerr=_blend_se, fmt="o-", color=_col_blend, lw=2.2,
                 ms=9, capsize=5, label="MDP-IRL blend  (mean ± SE)")
    ax9.errorbar(DELTA_GRID, _e2e_means,   yerr=_e2e_se, fmt="^-", color=_col_e2e, lw=2.2,
                 ms=9, capsize=5, label=r"MDP-IRL E2E  (mean ± SE)")
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
    fig9.savefig(f"{fig_dir}/fig9_delta_identification.pdf", dpi=150, bbox_inches="tight")
    fig9.savefig(f"{fig_dir}/fig9_delta_identification.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig9_delta_identification.pdf")
    plt.close(fig9)

    # ── Figure 10: θ robustness ───────────────────────────────────────────────
    _theta_models = [
        ("LA-AIDS",                 "rmse_aids",   "#E53935", "--"),
        ("Neural IRL",              "rmse_nirl",   "#1E88E5", "-."),
        ("MDP Neural IRL",          "rmse_mdp",    "#00897B", "-"),
        ("MDP IRL (E2E δ)",         "rmse_e2e",    "#FF6F00", "-"),
    ]
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
        _mu      = [_th_agg[_tt][key]["mean"] for _tt in THETA_GRID]
        _se_vals = [_th_agg[_tt][key]["se"]   for _tt in THETA_GRID]
        ax10a.errorbar(THETA_GRID, _mu, yerr=_se_vals, fmt=ls, color=col,
                       lw=2.0, ms=7, capsize=4, marker="o", label=lbl)
    ax10a.set_xlabel("Habit strength θ", fontsize=13)
    ax10a.set_ylabel("Post-shock RMSE", fontsize=13)
    ax10a.set_title("Absolute RMSE vs θ", fontsize=12, fontweight="bold")
    ax10a.legend(fontsize=11); ax10a.grid(True, alpha=0.3)

    # Panel B: RMSE reduction vs θ
    _base_mdp  = [_th_agg[_tt]["rmse_aids"]["mean"] for _tt in THETA_GRID]
    for lbl, key, col, ls in _theta_models[1:]:   # skip AIDS (baseline)
        _delta_rmse = [_th_agg[_tt]["rmse_aids"]["mean"] - _th_agg[_tt][key]["mean"]
                       for _tt in THETA_GRID]
        ax10b.plot(THETA_GRID, _delta_rmse, color=col, ls=ls, lw=2.0, marker="o", ms=6, label=lbl)
    ax10b.axhline(0, color="k", ls="--", lw=1.0)
    ax10b.set_xlabel("Habit strength θ", fontsize=13)
    ax10b.set_ylabel("RMSE reduction vs AIDS", fontsize=13)
    ax10b.set_title("RMSE Reduction over AIDS vs θ", fontsize=12, fontweight="bold")
    ax10b.legend(fontsize=11); ax10b.grid(True, alpha=0.3)

    fig10.suptitle(
        f"θ Robustness  ({N_SWEEP_SEEDS} seeds × {SWEEP_EPOCHS} epochs per point)",
        fontsize=12, fontweight="bold")
    fig10.tight_layout()
    fig10.savefig(f"{fig_dir}/fig10_theta_robustness.pdf", dpi=150, bbox_inches="tight")
    fig10.savefig(f"{fig_dir}/fig10_theta_robustness.png", dpi=150, bbox_inches="tight")
    print("    Saved: figures/fig10_theta_robustness.pdf")
    plt.close(fig10)

    return {"delta_rows": _delta_rows, "theta_rows": _theta_rows,
            "th_agg": _th_agg}
