"""
experiments/simulation/exp05_large_n.py
=========================================
Section 19 of main_multiple_runs.py.

Large-N KL Profile Sweep (N = 5 000 and N = 8 000):
  - Verifies that the KL-profile minimum at δ̂ sharpens as N grows.
  - If the bowl sharpens, δ is identified in the population limit.
  - Generates Figure 14.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from src.models.simulation import HabitFormationConsumer

from experiments.simulation.utils import fit_mdp_delta_grid


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-SEED LARGE-N SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_large_n_kl_sweep(seed: int, n_obs: int, cfg: dict,
                         epochs: int | None = None) -> dict:
    """Train MDP-E2E models via frozen-δ grid sweep on n_obs observations.

    δ is NOT jointly learned.  For each candidate in MDP_DELTA_GRID we train
    a model with frozen δ, evaluate hold-out KL, and select δ̂ = argmin.

    Parameters
    ----------
    seed   : RNG seed.
    n_obs  : number of training observations.
    cfg    : experiment config dict.
    epochs : training epochs; defaults to cfg["LARGE_EPOCHS"].
    """
    import numpy as np

    DEVICE         = cfg["DEVICE"]
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]
    if epochs is None:
        epochs = cfg.get("LARGE_EPOCHS", 8000)

    np.random.seed(seed)

    Z      = np.random.uniform(1, 5, (n_obs, 3))
    p_pre  = Z + np.random.normal(0, 0.1, (n_obs, 3))
    income = np.random.uniform(1200, 2000, n_obs)

    hc = HabitFormationConsumer()
    w_hab, _ = hc.solve_demand(p_pre, income, return_xbar=True)
    q_tr      = w_hab * income[:, None] / np.maximum(p_pre, 1e-8)
    log_q_seq = np.log(np.maximum(q_tr, 1e-6))

    # Validation set
    _vrng_lg = np.random.default_rng(seed + 55555)
    _nv_lg   = max(n_obs // 5, 200)
    _pv_lg   = np.clip(_vrng_lg.uniform(1, 5, (_nv_lg, 3)) +
                       _vrng_lg.normal(0, 0.1, (_nv_lg, 3)), 1e-3, None)
    _yv_lg   = _vrng_lg.uniform(1200, 2000, _nv_lg)
    _hcv_lg  = HabitFormationConsumer()
    _wv_lg, _ = _hcv_lg.solve_demand(_pv_lg, _yv_lg, return_xbar=True)
    _qv_lg   = _wv_lg * _yv_lg[:, None] / np.maximum(_pv_lg, 1e-8)
    _lqv_lg  = np.log(np.maximum(_qv_lg, 1e-6))

    _sw_lg = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q_seq,
        _pv_lg, _yv_lg, _wv_lg, _lqv_lg,
        delta_grid=MDP_DELTA_GRID, epochs=epochs, device=DEVICE,
        n_goods=3, hidden=256, fixed_beta=None,
        lam_mono=0.3, lam_slut=0.1, batch=512, lr=5e-4,
        tag=f"E2E-N{n_obs}-s{seed}")

    _sw_lg_fb = fit_mdp_delta_grid(
        p_pre, income, w_hab, log_q_seq,
        _pv_lg, _yv_lg, _wv_lg, _lqv_lg,
        delta_grid=MDP_DELTA_GRID, epochs=epochs, device=DEVICE,
        n_goods=3, hidden=256, fixed_beta=1.0,
        lam_mono=0.3, lam_slut=0.1, batch=512, lr=5e-4,
        tag=f"E2E-fb-N{n_obs}-s{seed}")

    return {
        "n_obs":          n_obs,
        "seed":           seed,
        "kl_delta_grid":  MDP_DELTA_GRID.copy(),
        "kl_profile_e2e": _sw_lg["kl_grid"],
        "kl_profile_fb":  _sw_lg_fb["kl_grid"],
        "delta_e2e_hat":  _sw_lg["delta_hat"],
        "delta_fb_hat":   _sw_lg_fb["delta_hat"],
        "id_set_e2e":     _sw_lg["id_set"],
        "id_set_fb":      _sw_lg_fb["id_set"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict:
    """Run large-N KL profile sweeps and generate Figure 14.

    Parameters
    ----------
    cfg : dict with keys DEVICE, MDP_DELTA_GRID,
          optional LARGE_N_GRID, N_LARGE_SEEDS, LARGE_EPOCHS, fig_dir.
    """
    fig_dir        = cfg.get("fig_dir", "figures")
    LARGE_N_GRID   = cfg.get("LARGE_N_GRID",   [5_000, 8_000])
    N_LARGE_SEEDS  = cfg.get("N_LARGE_SEEDS",  5)
    LARGE_EPOCHS   = cfg.get("LARGE_EPOCHS",   8_000)
    cfg.setdefault("LARGE_EPOCHS", LARGE_EPOCHS)
    MDP_DELTA_GRID = cfg["MDP_DELTA_GRID"]

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SECTION 19 — LARGE-N KL PROFILE SWEEP")
    print("=" * 72)

    _large_n_rows: dict[int, list] = {}
    for _ln in LARGE_N_GRID:
        _large_n_rows[_ln] = []
        print(f"\n  N = {_ln:,d}  ({N_LARGE_SEEDS} seeds × {LARGE_EPOCHS} epochs)")
        for _si in range(N_LARGE_SEEDS):
            _seed_ln = 900 + _si * 23
            _tln0 = time.time()
            print(f"    seed={_seed_ln}", end="", flush=True)
            _lr = run_large_n_kl_sweep(_seed_ln, n_obs=_ln, cfg=cfg,
                                        epochs=LARGE_EPOCHS)
            print(f"  →  δ̂_E2E={_lr['delta_e2e_hat']:.3f}"
                  f"  δ̂_β=1={_lr['delta_fb_hat']:.3f}"
                  f"  ({time.time()-_tln0:.0f}s)")
            _large_n_rows[_ln].append(_lr)

    # Aggregate KL profiles
    _large_n_kl_agg: dict = {}
    for _ln, _rows in _large_n_rows.items():
        _e2e_stack = np.stack([r["kl_profile_e2e"] for r in _rows], 0)
        _fb_stack  = np.stack([r["kl_profile_fb"]  for r in _rows], 0)
        _n_s = len(_rows)
        _large_n_kl_agg[_ln] = {
            "e2e": {
                "mean": _e2e_stack.mean(0),
                "se":   (_e2e_stack.std(0, ddof=1) / np.sqrt(_n_s)
                         if _n_s > 1 else np.zeros_like(_e2e_stack[0])),
            },
            "fb": {
                "mean": _fb_stack.mean(0),
                "se":   (_fb_stack.std(0, ddof=1) / np.sqrt(_n_s)
                         if _n_s > 1 else np.zeros_like(_fb_stack[0])),
            },
        }

    kl_delta_grid = _rows[0]["kl_delta_grid"]   # same for all rows

    # ── Figure 14: KL profile sharpening with N ───────────────────────────────
    _ln_colors = ["#1565C0", "#0D47A1", "#006064", "#004D40", "#1B5E20",
                  "#827717", "#E65100", "#B71C1C", "#4A148C", "#311B92"]
    fig14, (ax14a, ax14b) = plt.subplots(1, 2, figsize=(14, 6))

    for i, _ln in enumerate(LARGE_N_GRID):
        col = _ln_colors[i % len(_ln_colors)]
        _e = _large_n_kl_agg[_ln]["e2e"]
        _f = _large_n_kl_agg[_ln]["fb"]
        ax14a.plot(kl_delta_grid, _e["mean"], color=col, lw=2.2,
                   label=f"N={_ln:,d}  (mean)")
        ax14a.fill_between(kl_delta_grid,
                            _e["mean"] - _e["se"],
                            _e["mean"] + _e["se"],
                            color=col, alpha=0.15)
        ax14b.plot(kl_delta_grid, _f["mean"], color=col, lw=2.2, ls="--",
                   label=f"N={_ln:,d}")
        ax14b.fill_between(kl_delta_grid,
                            _f["mean"] - _f["se"],
                            _f["mean"] + _f["se"],
                            color=col, alpha=0.12)

    for ax in (ax14a, ax14b):
        ax.axvline(0.7, color="k", ls="--", lw=1.8, label="True δ=0.7")
        ax.set_xlabel(r"Candidate $\delta$", fontsize=12)
        ax.set_ylabel("Validation KL divergence", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
    ax14a.set_title("Panel A: MDP-IRL E2E  (β learned)", fontsize=12, fontweight="bold")
    ax14b.set_title(r"Panel B: MDP E2E ($\beta$=1, ablation)", fontsize=12, fontweight="bold")

    fig14.suptitle(
        f"Large-N KL Profile Sharpening  ({N_LARGE_SEEDS} seeds each)\n"
        "Sharper bowl with larger N → δ identified in population limit",
        fontsize=12, fontweight="bold")
    fig14.tight_layout()
    fig14.savefig(f"{fig_dir}/fig14_large_n_kl.pdf", dpi=150, bbox_inches="tight")
    fig14.savefig(f"{fig_dir}/fig14_large_n_kl.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig14_large_n_kl.pdf")
    plt.close(fig14)

    return {"large_n_rows": _large_n_rows,
            "large_n_kl_agg": _large_n_kl_agg}
