"""
run_simulation_experiments.py
==============================
Main entry-point that orchestrates all simulation experiments.

Sections
--------
  01  Main multi-seed runs (Sections 1–10 of main_multiple_runs.py)
  02  Delta identification / KL profile sweep (Sections 11–14)
  03  Welfare bounds over the identified set of delta (Sections 15–17)
  04  Placebo test for sorting confound (Sections 18)
  05  Large-N KL profile sharpening (Section 19)
  06  Welfare robustness at delta=0.90 + frozen-delta ID test (Sections 20–21)
  07  Linear-in-x̄ reward delta identification (Section 22)

Usage
-----
    python run_simulation_experiments.py [--n-runs N] [--n-obs N]
                                         [--device DEVICE] [--fig-dir DIR]
                                         [--skip EXP [EXP ...]]
                                         [--only EXP [EXP ...]]

Arguments
---------
  --n-runs   : number of independent random seeds for main runs (default 5).
  --n-obs    : observations per run (default 1000).
  --device   : torch device, e.g. "cpu" or "cuda" (auto-detected by default).
  --fig-dir  : output directory for figures (default "figures").
  --skip     : space-separated list of experiment IDs to skip, e.g. 05 06 07.
  --only     : run only the specified experiment IDs (overrides --skip).
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

# ── Simulation experiment modules ────────────────────────────────────────────
import experiments.simulation.exp01_main_runs       as exp01
import experiments.simulation.exp02_identification  as exp02
import experiments.simulation.exp03_welfare         as exp03
import experiments.simulation.exp04_placebo         as exp04
import experiments.simulation.exp05_large_n         as exp05
import experiments.simulation.exp06_frozen_delta    as exp06
import experiments.simulation.exp07_linear_xbar     as exp07


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run all simulation experiments for the IRL consumer paper.")
    p.add_argument("--n-runs",  type=int,   default=5,
                   help="Number of independent random seeds for main runs.")
    p.add_argument("--n-obs",   type=int,   default=1000,
                   help="Observations per run.")
    p.add_argument("--device",  type=str,   default=None,
                   help="Torch device ('cpu' or 'cuda'). Auto-detected if omitted.")
    p.add_argument("--fig-dir", type=str,   default="figures",
                   help="Output directory for figures.")
    p.add_argument("--skip",    nargs="*",  default=[],
                   help="Experiment IDs to skip (e.g. 05 06).")
    p.add_argument("--only",    nargs="*",  default=[],
                   help="Run only these experiment IDs (overrides --skip).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _should_run(exp_id: str, only: list, skip: list) -> bool:
    if only:
        return exp_id in only
    return exp_id not in skip


def _banner(title: str) -> None:
    width = 72
    print("\n" + "█" * width)
    print(f"  {title}")
    print("█" * width)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = 5000

def main():
    args = parse_args()

    device = (args.device if args.device is not None
              else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Shared configuration dict passed to every experiment ─────────────────
    cfg = dict(
        # Core
        N_RUNS          = args.n_runs,
        N_OBS           = args.n_obs,
        DEVICE          = device,
        fig_dir         = args.fig_dir,

        # Grid of candidate δ values for the frozen-δ / E2E sweep
        MDP_DELTA_GRID  = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

        # Price grid and average income for welfare / elasticity plots
        P_GRID          = np.linspace(1, 10, 80),
        AVG_Y           = 1600.0,

        # True habit-decay parameter in the simulation DGP
        TRUE_DELTA      = 0.7,

        # Training epochs for welfare experiments (exp03 / exp06)
        WELF_EPOCHS     = EPOCHS,

        # Frozen-delta identification experiment (exp06 section 21)
        FROZEN_DELTA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        FROZEN_EPOCHS   = EPOCHS,

        # Large-N sharpening experiment (exp05)
        LARGE_N_GRID    = [5_000, 8_000],
        N_LARGE_SEEDS   = 5,
        LARGE_EPOCHS    = EPOCHS,
    )

    os.makedirs(cfg["fig_dir"], exist_ok=True)

    only = [s.lstrip("0") or "0" for s in args.only]  # normalise "05" → "5"
    skip = [s.lstrip("0") or "0" for s in args.skip]
    # Also support zero-padded variants
    only += args.only
    skip += args.skip

    t0_total = time.time()

    # ── Accumulated results (passed forward between experiments) ─────────────
    results: dict = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 01 — Main multi-seed runs
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("01", only, skip):
        _banner("EXP 01 — MAIN MULTI-SEED RUNS  (Sections 1–10)")
        t0 = time.time()
        _all_results_e01, _agg_e01 = exp01.run(cfg)
        results["exp01"] = _agg_e01
        print(f"\n  ✓ EXP 01 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 01")
        results["exp01"] = {}

    # Convenience unpacking for downstream experiments
    _e01 = results["exp01"]
    delta_mdp_e2e_mean = _e01.get("delta_mdp_e2e_mean", cfg["TRUE_DELTA"])
    delta_mdp_mean     = _e01.get("delta_mdp_mean",     cfg["TRUE_DELTA"])

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 02 — Delta identification / KL profile sweep
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("02", only, skip):
        _banner("EXP 02 — DELTA IDENTIFICATION / KL PROFILE  (Sections 11–14)")
        t0 = time.time()
        results["exp02"] = exp02.run(cfg)
        print(f"\n  ✓ EXP 02 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 02")
        results["exp02"] = {}

    _e02 = results["exp02"]
    kl_delta_grid    = _e02.get("kl_delta_grid",    cfg["MDP_DELTA_GRID"])
    kl_prof_e2e_mean = _e02.get("kl_prof_e2e_mean", np.zeros_like(kl_delta_grid))
    kl_prof_e2e_se   = _e02.get("kl_prof_e2e_se",   np.zeros_like(kl_delta_grid))

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 03 — Welfare bounds over the identified set
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("03", only, skip):
        _banner("EXP 03 — WELFARE BOUNDS OVER IDENTIFIED SET  (Sections 15–17)")
        t0 = time.time()
        results["exp03"] = exp03.run(cfg)
        print(f"\n  ✓ EXP 03 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 03")
        results["exp03"] = {}

    welf_agg_by_delta = results["exp03"].get("welf_agg_by_delta", {})

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 04 — Placebo test for sorting confound
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("04", only, skip):
        _banner("EXP 04 — PLACEBO TEST FOR SORTING CONFOUND  (Section 18)")
        t0 = time.time()
        results["exp04"] = exp04.run(cfg)
        print(f"\n  ✓ EXP 04 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 04")
        results["exp04"] = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 05 — Large-N KL profile sharpening
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("05", only, skip):
        _banner("EXP 05 — LARGE-N KL PROFILE SHARPENING  (Section 19)")
        t0 = time.time()
        results["exp05"] = exp05.run(cfg)
        print(f"\n  ✓ EXP 05 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 05")
        results["exp05"] = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 06 — Welfare robustness at delta=0.90 + Frozen-δ ID test
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("06", only, skip):
        _banner("EXP 06 — WELFARE ROBUSTNESS + FROZEN-δ ID TEST  (Sections 20–21)")
        t0 = time.time()
        results["exp06"] = exp06.run(
            cfg,
            welf_agg_by_delta  = welf_agg_by_delta,
            delta_mdp_e2e_mean = delta_mdp_e2e_mean,
            delta_mdp_mean     = delta_mdp_mean,
            kl_delta_grid      = kl_delta_grid,
            kl_prof_e2e_mean   = kl_prof_e2e_mean,
            kl_prof_e2e_se     = kl_prof_e2e_se,
        )
        print(f"\n  ✓ EXP 06 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 06")
        results["exp06"] = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  EXP 07 — Linear-in-x̄ reward delta identification
    # ══════════════════════════════════════════════════════════════════════════
    if _should_run("07", only, skip):
        _banner("EXP 07 — LINEAR-IN-x̄ REWARD: δ IDENTIFICATION  (Section 22)")
        t0 = time.time()
        results["exp07"] = exp07.run(cfg)
        print(f"\n  ✓ EXP 07 done in {time.time()-t0:.0f}s")
    else:
        print("\n  [skip] EXP 07")
        results["exp07"] = {}

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("ALL SIMULATION EXPERIMENTS COMPLETE")
    print(f"  Total time : {(time.time()-t0_total)/60:.1f} min")
    print(f"  Figures saved to: {os.path.abspath(cfg['fig_dir'])}/")
    print()

    return results


if __name__ == "__main__":
    main()
