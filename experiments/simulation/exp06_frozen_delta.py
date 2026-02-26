"""
experiments/simulation/exp06_frozen_delta.py
=============================================
Sections 20 and 21 of main_multiple_runs.py.

Section 20 — Welfare Robustness at δ = 0.90
    Re-runs the welfare evaluation (from exp03) with δ fixed at 0.90,
    the upper bound of the identified attractor range.  If MDP CV error
    remains near zero the welfare conclusion is robust across the FULL
    attractor range.  Generates Figure 15.

Section 21 — Frozen-δ Grid Identification Test (Stage 1 + 2)
    For each δ ∈ {0.3, …, 0.9}:
      Stage 1: train ψ+β with log_delta frozen (not jointly learned).
      Stage 2: evaluate KL on an independent held-out test set.
    The held-out curve has a sharper minimum than the joint-training
    curve from Section 11 → confirms δ is identified.  Generates Figure 16.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from src.models.simulation import (
    HabitFormationConsumer,
    MDPNeuralIRL_E2E,
    train_mdp_e2e,
    compute_xbar_e2e,
)

from experiments.simulation.utils import predict_shares, kl_div
from experiments.simulation.exp03_welfare import run_welfare_delta_seed


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 20 — WELFARE ROBUSTNESS AT δ = 0.90
# ─────────────────────────────────────────────────────────────────────────────

def run_welfare_090(cfg: dict, welf_agg_by_delta: dict,
                    delta_mdp_e2e_mean: float,
                    delta_mdp_mean: float) -> dict:
    """Run welfare evaluation at δ = 0.90 and generate Figure 15.

    Parameters
    ----------
    cfg                 : experiment config dict.
    welf_agg_by_delta   : aggregated welfare results from exp03 keyed by delta.
    delta_mdp_e2e_mean  : mean E2E delta estimate from exp01.
    delta_mdp_mean      : mean MDP blend delta estimate from exp01.
    """
    fig_dir    = cfg.get("fig_dir", "figures")
    N_RUNS     = cfg.get("N_RUNS", 1)
    WELF_EPOCHS = cfg.get("WELF_EPOCHS", 8_000)
    DEVICE     = cfg["DEVICE"]

    DELTA_090   = 0.90
    N_090_SEEDS = min(N_RUNS, 3)
    EPOCHS_090  = WELF_EPOCHS

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SECTION 20 — WELFARE ROBUSTNESS CHECK AT δ = 0.90")
    print("=" * 72)
    print(f"\n  Running {N_090_SEEDS} seeds × {EPOCHS_090} epochs at δ = {DELTA_090} ...")

    def _se(vals):
        return (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1 else 0.0)

    rows_090 = []
    for _si in range(N_090_SEEDS):
        _seed_090 = 950 + _si * 31
        _t090_0 = time.time()
        print(f"    seed={_seed_090}", end="", flush=True)
        _wr090 = run_welfare_delta_seed(_seed_090, DELTA_090, cfg=cfg)
        print(f"  →  CV_gt={_wr090['cv_gt']:.4f}"
              f"  CV_mdp={_wr090['cv_mdp']:.4f}"
              f"  CV_e2e={_wr090['cv_e2e']:.4f}"
              f"  δ̂_E2E={_wr090['delta_ee_hat']:.3f}"
              f"  ({time.time()-_t090_0:.0f}s)")
        rows_090.append(_wr090)

    # Aggregate across seeds
    _keys_090 = ["cv_gt", "cv_aids", "cv_nirl", "cv_mdp", "cv_e2e", "delta_ee_hat"]
    agg_090 = {k: {
        "mean": np.nanmean([r[k] for r in rows_090]),
        "se":   (_se([r[k] for r in rows_090]) if N_090_SEEDS > 1 else 0.0),
    } for k in _keys_090}

    _gt090   = agg_090["cv_gt"]["mean"]
    _mdp090  = agg_090["cv_mdp"]["mean"]
    _e2e090  = agg_090["cv_e2e"]["mean"]
    _nirl090 = agg_090["cv_nirl"]["mean"]
    _aids090 = agg_090["cv_aids"]["mean"]

    _err_mdp090  = abs(_mdp090  - _gt090)
    _err_e2e090  = abs(_e2e090  - _gt090)
    _err_nirl090 = abs(_nirl090 - _gt090)
    _err_aids090 = abs(_aids090 - _gt090)

    _pct_mdp090  = 100 * _err_mdp090  / max(abs(_gt090), 1e-9)
    _pct_e2e090  = 100 * _err_e2e090  / max(abs(_gt090), 1e-9)
    _pct_nirl090 = 100 * _err_nirl090 / max(abs(_gt090), 1e-9)
    _pct_aids090 = 100 * _err_aids090 / max(abs(_gt090), 1e-9)

    # Compare with Section 17 results
    if DELTA_090 in welf_agg_by_delta:
        _sec17_mdp_err090 = abs(
            welf_agg_by_delta[DELTA_090]["cv_mdp"]["mean"]
            - welf_agg_by_delta[DELTA_090]["cv_gt"]["mean"])
        _sec17_note = f"Section 17 (same δ=0.90): |CV error|={_sec17_mdp_err090:.4f}"
    else:
        _sec17_note = "Section 17 did not include δ=0.90 in its grid."

    # Console report
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

    _NEAR_ZERO_THRESH = 2.0
    print("\n  ROBUSTNESS VERDICT:")
    if _pct_mdp090 < _NEAR_ZERO_THRESH:
        print(f"  ✓  MDP CV error at δ=0.90 is {_pct_mdp090:.2f}% < {_NEAR_ZERO_THRESH}% threshold.")
        print(f"     Welfare conclusion is ROBUST across the full attractor range"
              f" [{min(delta_mdp_e2e_mean, delta_mdp_mean):.2f}, {DELTA_090:.2f}].")
    else:
        print(f"  ⚠  MDP CV error at δ=0.90 is {_pct_mdp090:.2f}% ≥ {_NEAR_ZERO_THRESH}% threshold.")
        print(f"     Welfare claims should be BOUNDED to the E2E attractor range"
              f" [δ̂_E2E ≈ {delta_mdp_e2e_mean:.2f}, δ̂_blend ≈ {delta_mdp_mean:.2f}].")
    if _pct_e2e090 < _NEAR_ZERO_THRESH:
        print(f"\n  ✓  E2E CV error at δ=0.90 is also {_pct_e2e090:.2f}% — robustness confirmed.")

    # ── Figure 15: bar chart ──────────────────────────────────────────────────
    _att_e2e_key   = min(welf_agg_by_delta.keys(),
                         key=lambda d: abs(d - delta_mdp_e2e_mean))
    _att_blend_key = min(welf_agg_by_delta.keys(),
                         key=lambda d: abs(d - delta_mdp_mean))

    def _pct_err(d_key, model_key):
        _cv = welf_agg_by_delta[d_key][model_key]["mean"]
        _gt = welf_agg_by_delta[d_key]["cv_gt"]["mean"]
        return 100 * abs(_cv - _gt) / max(abs(_gt), 1e-9)

    _fig15_deltas = [_att_e2e_key, _att_blend_key, DELTA_090]
    _fig15_labels = [
        rf"$\delta_{{E2E}}$={_att_e2e_key:.2f}",
        rf"$\delta_{{blend}}$={_att_blend_key:.2f}",
        rf"$\delta$={DELTA_090:.2f}  (upper bound)",
    ]
    _fig15_colors = ["#FF6F00", "#00897B", "#C62828"]
    _fig15_models = [
        ("LA-AIDS",        "cv_aids", "#EF9A9A"),
        ("Neural IRL",     "cv_nirl", "#90CAF9"),
        ("MDP Neural IRL", "cv_mdp",  "#80CBC4"),
        ("MDP IRL (E2E)",  "cv_e2e",  "#FFCC80"),
    ]

    fig15, ax15 = plt.subplots(figsize=(11, 6))
    _x15    = np.arange(len(_fig15_models))
    _w15    = 0.22
    _offsets = np.linspace(-(len(_fig15_deltas) - 1) / 2 * _w15,
                            (len(_fig15_deltas) - 1) / 2 * _w15,
                            len(_fig15_deltas))

    for _oi, (_dk, _dlbl, _dcol) in enumerate(
            zip(_fig15_deltas, _fig15_labels, _fig15_colors)):
        _errs = []
        for _modnm, _modkey, _ in _fig15_models:
            if _dk == DELTA_090:
                _cvs_090 = [r[_modkey] for r in rows_090]
                _gt_vals  = [r["cv_gt"] for r in rows_090]
                _errs.append(100 * np.mean([abs(c - g)
                                            for c, g in zip(_cvs_090, _gt_vals)])
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
    fig15.savefig(f"{fig_dir}/fig15_welfare_delta090.pdf", dpi=150, bbox_inches="tight")
    fig15.savefig(f"{fig_dir}/fig15_welfare_delta090.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig15_welfare_delta090.pdf")
    plt.close(fig15)

    return {
        "rows_090":  rows_090,
        "agg_090":   agg_090,
        "pct_mdp090": _pct_mdp090,
        "pct_e2e090": _pct_e2e090,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 21 — FROZEN-δ GRID IDENTIFICATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_frozen_delta_seed(seed: int, delta_fixed: float,
                          cfg: dict,
                          epochs: int = 10_000) -> dict:
    """Train MDP-E2E with δ *frozen* at delta_fixed; score on held-out test.

    Stage 1
    -------
    Generate N training observations (HabitFormationConsumer, δ=0.7).
    Build MDPNeuralIRL_E2E with delta_init=delta_fixed, then freeze log_delta
    so only the network weights ψ and temperature β are updated.

    Stage 2
    -------
    Draw an *independent* test set; evaluate KL on test and training data.

    Parameters
    ----------
    seed        : RNG seed.
    delta_fixed : δ value at which the model is frozen.
    cfg         : experiment config dict (needs N_OBS, DEVICE).
    epochs      : training epochs.
    """
    N_OBS  = cfg["N_OBS"]
    DEVICE = cfg["DEVICE"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Training data (true DGP: δ = 0.7)
    Z_tr    = np.random.uniform(1, 5, (N_OBS, 3))
    p_tr    = Z_tr + np.random.normal(0, 0.1, (N_OBS, 3))
    inc_tr  = np.random.uniform(1200, 2000, N_OBS)
    hc_true = HabitFormationConsumer()
    w_tr, _ = hc_true.solve_demand(p_tr, inc_tr, return_xbar=True)
    q_tr    = w_tr * inc_tr[:, None] / np.maximum(p_tr, 1e-8)
    lq_tr   = np.log(np.maximum(q_tr, 1e-6))

    # Held-out test data (independent draw, same DGP)
    Z_te    = np.random.uniform(1, 5, (N_OBS, 3))
    p_te    = Z_te + np.random.normal(0, 0.1, (N_OBS, 3))
    inc_te  = np.random.uniform(1200, 2000, N_OBS)
    w_te, _ = hc_true.solve_demand(p_te, inc_te, return_xbar=True)
    q_te    = w_te * inc_te[:, None] / np.maximum(p_te, 1e-8)
    lq_te   = np.log(np.maximum(q_te, 1e-6))

    # Stage 1: build model and freeze δ
    model = MDPNeuralIRL_E2E(n_goods=3, hidden_dim=256, delta_init=delta_fixed)
    model.log_delta.requires_grad_(False)   # freeze habit-decay parameter

    model, _ = train_mdp_e2e(
        model, p_tr, inc_tr, w_tr, lq_tr,
        store_ids=None, epochs=epochs, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
        xbar_recompute_every=10, device=DEVICE,
        tag=f"frozen-d{delta_fixed:.1f}-s{seed}")

    # Stage 2: evaluate KL on train and test
    _dt = torch.tensor(float(delta_fixed), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # test KL
        lq_te_t  = torch.tensor(lq_te,  dtype=torch.float32, device=DEVICE)
        xb_te    = compute_xbar_e2e(_dt, lq_te_t, store_ids=None).cpu().numpy()
        w_pred_te = predict_shares("mdp-e2e", p_te, inc_te,
                                   mdp_e2e=model, xbar_e2e=xb_te, device=DEVICE)
        kl_test  = kl_div(w_pred_te, w_te)

        # train KL (for Fig-11 comparison)
        lq_tr_t  = torch.tensor(lq_tr,  dtype=torch.float32, device=DEVICE)
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


def run_frozen_id_test(cfg: dict,
                       kl_delta_grid: np.ndarray,
                       kl_prof_e2e_mean: np.ndarray,
                       kl_prof_e2e_se: np.ndarray) -> dict:
    """Run Stage 1+2 frozen-δ identification test and generate Figure 16.

    Parameters
    ----------
    cfg               : experiment config dict (N_OBS, DEVICE, N_RUNS, fig_dir, …).
    kl_delta_grid     : δ grid from exp02 KL sweep (for Fig 16 background).
    kl_prof_e2e_mean  : mean KL profile from exp02 (for Fig 16 background).
    kl_prof_e2e_se    : SE of KL profile from exp02 (for Fig 16 background).
    """
    fig_dir        = cfg.get("fig_dir", "figures")
    N_RUNS         = cfg.get("N_RUNS", 1)
    TRUE_DELTA     = cfg.get("TRUE_DELTA", 0.7)

    FROZEN_DELTA_GRID = cfg.get("FROZEN_DELTA_GRID", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    N_FROZEN_SEEDS    = min(N_RUNS, 3)
    FROZEN_EPOCHS     = cfg.get("FROZEN_EPOCHS", 10_000)

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SECTION 21 — FROZEN-δ GRID IDENTIFICATION TEST (STAGE 1 + 2)")
    print("=" * 72)
    print(f"\n  δ grid : {FROZEN_DELTA_GRID}")
    print(f"  Seeds  : {N_FROZEN_SEEDS}  |  Epochs : {FROZEN_EPOCHS}")
    print(f"  True δ : {TRUE_DELTA}")

    def _se(vals):
        return (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1 else 0.0)

    frozen_rows: dict[float, list] = {dv: [] for dv in FROZEN_DELTA_GRID}
    for _dv in FROZEN_DELTA_GRID:
        print(f"\n  δ = {_dv:.1f}", end="", flush=True)
        for _si in range(N_FROZEN_SEEDS):
            _seed_fz = 700 + _si * 41
            _tfz0 = time.time()
            _rfz = run_frozen_delta_seed(_seed_fz, _dv, cfg=cfg,
                                          epochs=FROZEN_EPOCHS)
            print(f"  [seed={_seed_fz} "
                  f"KL_test={_rfz['kl_test']:.5f} "
                  f"KL_train={_rfz['kl_train']:.5f} "
                  f"({time.time()-_tfz0:.0f}s)]",
                  end="", flush=True)
            frozen_rows[_dv].append(_rfz)
        print()

    # Aggregate
    frozen_agg: dict = {}
    for _dv, _rows in frozen_rows.items():
        _n_s = len(_rows)
        _kl_test_vals  = [r["kl_test"]  for r in _rows]
        _kl_train_vals = [r["kl_train"] for r in _rows]
        frozen_agg[_dv] = {
            "kl_test":  {"mean": float(np.mean(_kl_test_vals)),
                         "se":   _se(_kl_test_vals)  if _n_s > 1 else 0.0},
            "kl_train": {"mean": float(np.mean(_kl_train_vals)),
                         "se":   _se(_kl_train_vals) if _n_s > 1 else 0.0},
        }

    _fdg_arr    = np.array(FROZEN_DELTA_GRID)
    _kl_te_mean = np.array([frozen_agg[d]["kl_test"]["mean"]  for d in FROZEN_DELTA_GRID])
    _kl_te_se   = np.array([frozen_agg[d]["kl_test"]["se"]    for d in FROZEN_DELTA_GRID])
    _kl_tr_mean = np.array([frozen_agg[d]["kl_train"]["mean"] for d in FROZEN_DELTA_GRID])
    _kl_tr_se   = np.array([frozen_agg[d]["kl_train"]["se"]   for d in FROZEN_DELTA_GRID])

    # ── Figure 16: Frozen-δ KL profile with Fig-11 background ────────────────
    _col_test  = "#1565C0"
    _col_train = "#E65100"
    _col_bg    = "#BDBDBD"

    fig16, ax16 = plt.subplots(figsize=(10, 5))

    # Background: Fig-11 continuous E2E KL_train curve
    ax16.plot(kl_delta_grid, kl_prof_e2e_mean,
              color=_col_bg, lw=1.5, ls="-", alpha=0.7,
              label=r"Fig 11: KL$_{\rm train}$ (joint $\hat{\delta}$+$\hat{\beta}$, continuous sweep)")
    if N_RUNS > 1:
        ax16.fill_between(kl_delta_grid,
                          kl_prof_e2e_mean - kl_prof_e2e_se,
                          kl_prof_e2e_mean + kl_prof_e2e_se,
                          color=_col_bg, alpha=0.20)

    # Frozen-δ training KL
    ax16.errorbar(_fdg_arr, _kl_tr_mean, yerr=_kl_tr_se,
                  fmt="o-", color=_col_train, lw=2.0, ms=8, capsize=5,
                  label=r"KL$_{\rm train}$ — frozen $\delta$, ψ+β adapted"
                        + f"  ({N_FROZEN_SEEDS} seeds)")

    # Frozen-δ test KL (key identification curve)
    ax16.errorbar(_fdg_arr, _kl_te_mean, yerr=_kl_te_se,
                  fmt="s-", color=_col_test, lw=2.5, ms=9, capsize=5,
                  label=r"KL$_{\rm test}$ — frozen $\delta$, held-out data"
                        + f"  ({N_FROZEN_SEEDS} seeds)")

    _argmin_te    = int(np.argmin(_kl_te_mean))
    _delta_te_min = FROZEN_DELTA_GRID[_argmin_te]
    ax16.axvline(TRUE_DELTA, color="k",        ls="--", lw=2.0,
                 label=f"True δ = {TRUE_DELTA}")
    ax16.axvline(_delta_te_min, color=_col_test, ls=":", lw=2.0,
                 label=rf"argmin KL$_{{test}}$ = {_delta_te_min:.1f}")

    ax16.set_xlabel(r"Frozen habit-decay parameter $\delta$", fontsize=14)
    ax16.set_ylabel("KL divergence", fontsize=14)
    ax16.legend(fontsize=10, loc="upper left")
    ax16.grid(True, alpha=0.3)
    ax16.set_xticks(_fdg_arr)

    _se_note = (f"  (error bars = ±1 SE, {N_FROZEN_SEEDS} seeds)"
                if N_FROZEN_SEEDS > 1 else "")
    fig16.suptitle(
        r"Frozen-$\delta$ Identification Test — KL on Held-Out vs Training Data"
        "\nStage 1: train ψ+β with δ frozen  ·  "
        "Stage 2: evaluate on independent test set"
        f"\nTrue δ = {TRUE_DELTA}  ·  Habit DGP  ·  N={cfg['N_OBS']}{_se_note}",
        fontsize=12, fontweight="bold")
    fig16.tight_layout()
    fig16.savefig(f"{fig_dir}/fig16_frozen_delta_kl_profile.pdf", dpi=150, bbox_inches="tight")
    fig16.savefig(f"{fig_dir}/fig16_frozen_delta_kl_profile.png", dpi=150, bbox_inches="tight")
    print("\n    Saved: figures/fig16_frozen_delta_kl_profile.pdf")
    plt.close(fig16)

    # Console summary
    print("\n  SECTION 21 — FROZEN-δ IDENTIFICATION SUMMARY")
    print(f"  True δ = {TRUE_DELTA}  |  {N_FROZEN_SEEDS} seeds × {FROZEN_EPOCHS} epochs")
    print(f"  {'δ_fixed':>8}  {'KL_train (mean)':>17}  {'KL_train SE':>13}"
          f"  {'KL_test (mean)':>16}  {'KL_test SE':>12}")
    for _dv in FROZEN_DELTA_GRID:
        _a = frozen_agg[_dv]
        print(f"  {_dv:>8.1f}  "
              f"{_a['kl_train']['mean']:>17.6f}  "
              f"{_a['kl_train']['se']:>13.6f}  "
              f"{_a['kl_test']['mean']:>16.6f}  "
              f"{_a['kl_test']['se']:>12.6f}")

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

    # Sharpness comparison
    _cv_test  = _kl_te_mean.std() / max(_kl_te_mean.mean(), 1e-9)
    _fig11_sub = kl_prof_e2e_mean[
        (kl_delta_grid >= min(FROZEN_DELTA_GRID)) &
        (kl_delta_grid <= max(FROZEN_DELTA_GRID))]
    _cv_fig11 = _fig11_sub.std() / max(_fig11_sub.mean(), 1e-9)
    print(f"\n  Sharpness (CV of KL curve):")
    print(f"    KL_test  (frozen δ, held-out)  CV = {_cv_test:.4f}")
    print(f"    KL_train (Fig 11, joint opt.)  CV = {_cv_fig11:.4f}")
    if _cv_test > _cv_fig11:
        print(f"  ✓ KL_test is SHARPER than Fig-11 KL_train "
              f"(ratio = {_cv_test / max(_cv_fig11, 1e-9):.2f}×).")
    else:
        print("  ⚠ KL_test is not sharper than Fig-11 on this N — "
              "try larger N or more seeds.")

    print("\n  All figures saved to figures/")
    print("  Done (Sections 20-21).")

    return {
        "frozen_rows":      frozen_rows,
        "frozen_agg":       frozen_agg,
        "delta_te_min":     _delta_te_min,
        "bias_te":          _bias_te,
        "kl_te_mean":       _kl_te_mean,
        "kl_te_se":         _kl_te_se,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict,
        welf_agg_by_delta: dict,
        delta_mdp_e2e_mean: float,
        delta_mdp_mean: float,
        kl_delta_grid: np.ndarray,
        kl_prof_e2e_mean: np.ndarray,
        kl_prof_e2e_se: np.ndarray) -> dict:
    """Run Sections 20 and 21: welfare robustness + frozen-δ identification.

    Parameters
    ----------
    cfg                : full experiment config dict.
    welf_agg_by_delta  : results from exp03.run() keyed by delta.
    delta_mdp_e2e_mean : mean E2E delta from exp01.
    delta_mdp_mean     : mean blend delta from exp01.
    kl_delta_grid      : δ grid from exp02 KL sweep.
    kl_prof_e2e_mean   : mean KL profile from exp02.
    kl_prof_e2e_se     : SE   of KL profile from exp02.
    """
    results_s20 = run_welfare_090(cfg, welf_agg_by_delta,
                                   delta_mdp_e2e_mean, delta_mdp_mean)
    results_s21 = run_frozen_id_test(cfg, kl_delta_grid,
                                      kl_prof_e2e_mean, kl_prof_e2e_se)
    return {"section20": results_s20, "section21": results_s21}
