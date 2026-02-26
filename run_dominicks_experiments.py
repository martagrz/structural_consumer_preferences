"""
run_dominicks_experiments.py
=============================
Main orchestration script for all Dominick's analgesics experiments.

Mirrors the top-level execution flow of ``dominicks_multiple_runs.py``
but delegates each logical section to its own module under
``experiments/dominicks/``.

Usage
-----
    python run_dominicks_experiments.py

Sections
--------
  Section  8 : Data loading and preparation  (data.load)
  Section  9 : n_runs training loop          (exp01_main_runs.run_once)
  Section 10 : Aggregation                   (exp01_main_runs.aggregate)
  Section 11 : Evaluation tables             (printed inline)
  Section 12 : Figures                       (printed inline)
  Section 13 : CSV tables                    (printed inline)
  Section 14 : LaTeX                         (printed inline)
  Section D16: KL profile over δ             (exp02_kl_profile.run_kl_profile)
  Section D17: ICC decomposition             (exp03_icc.run_icc)
  Section D18: Welfare sensitivity           (exp04_welfare_sensitivity.run_welfare_sensitivity)
  Section D19: Linear-in-x̄ identification   (exp05_linear_xbar.run_linear_xbar_id)
"""

import os
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

# ── Experiment modules ────────────────────────────────────────────────────────
from experiments.dominicks.data import load
from experiments.dominicks.exp01_main_runs import run_once, aggregate, MODEL_NAMES, _make_figures, _make_tables
from experiments.dominicks.exp02_kl_profile import run_kl_profile
from experiments.dominicks.exp03_icc import run_icc
from experiments.dominicks.exp04_welfare_sensitivity import run_welfare_sensitivity
from experiments.dominicks.exp05_linear_xbar import run_linear_xbar_id

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = 10

CFG = dict(
    weekly_path   = './data/wana.csv',
    upc_path      = './data/upcana.csv',
    std_tablets   = 100,
    min_store_wks = 20,
    test_cutoff   = 351,
    habit_decay   = 0.70,
    n_runs        = 5,
    n_jobs        = 1,
    # Linear IRL
    lirl_lr=0.05, lirl_epochs=EPOCHS, lirl_l2=1e-4,
    # Neural IRL
    nirl_hidden=128, nirl_epochs=EPOCHS, nirl_lr=5e-4,
    nirl_batch=512, nirl_lam_mono=0.20, nirl_lam_slut=0.10,
    nirl_slut_start=0.25,
    # MDP Neural IRL
    mdp_hidden=128, mdp_epochs=EPOCHS, mdp_lr=5e-4,
    mdp_batch=512, mdp_lam_mono=0.20, mdp_lam_slut=0.10,
    mdp_slut_start=0.25,
    # MDP IRL E2E
    mdp_e2e_hidden=128, mdp_e2e_epochs=EPOCHS, mdp_e2e_lr=5e-4,
    mdp_e2e_batch=512, mdp_e2e_lam_mono=0.20, mdp_e2e_lam_slut=0.10,
    mdp_e2e_slut_start=0.25,
    # Variational Mixture
    mix_K=6, mix_n_spc=100, mix_n_iter=200,
    mix_lr_mu=0.05, mix_sigma2=0.1, mix_subsamp=300,
    # Welfare
    shock_good=2, shock_pct=0.10, cv_steps=100,
    # Output directories
    fig_dir='figures/dominicks',
    out_dir='results/dominicks',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

N_RUNS = CFG['n_runs']

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0_total = time.time()

    print('=' * 72)
    print("  IRL DEMAND RECOVERY — DOMINICK'S ANALGESICS")
    print('=' * 72)
    print(f"  Device: {CFG['device']}")
    print(f"  n_runs: {N_RUNS}  (standard errors from repeated re-estimation)")

    os.makedirs(CFG['fig_dir'], exist_ok=True)
    os.makedirs(CFG['out_dir'], exist_ok=True)

    # ── Section 8: Data loading and preparation ───────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION 8 — DATA LOADING AND PREPARATION')
    print('=' * 72)
    t0 = time.time()
    _raw_data, splits = load(CFG)
    # `splits` already contains 'shares', 'stores', 'tr', 'te', etc. from prepare_splits
    print(f'  Done ({time.time() - t0:.1f}s).')

    # ── Section 9-10: n_runs training loop ────────────────────────────────
    print('\n' + '=' * 72)
    print(f'  SECTION 9-10 — TRAINING LOOP  ({N_RUNS} runs)')
    print('=' * 72)
    seeds = [42 + i * 15 for i in range(N_RUNS)]
    _n_jobs = (CFG.get('n_jobs', 1)
               if CFG['device'] == 'cpu' else 1)
    print(f'  n_jobs={_n_jobs}  seeds={seeds}')

    t0 = time.time()
    if _n_jobs > 1:
        with ProcessPoolExecutor(max_workers=_n_jobs) as pool:
            all_runs = list(pool.map(
                lambda s: run_once(s, splits, CFG), seeds))
    else:
        all_runs = []
        for run_idx, seed in enumerate(seeds):
            print(f'\n  ── Run {run_idx + 1}/{N_RUNS}  (seed={seed}) ──')
            all_runs.append(run_once(seed, splits, CFG))
    print(f'\n  Training complete ({time.time() - t0:.0f}s).')

    # ── Section 10: Aggregate ─────────────────────────────────────────────
    agg = aggregate(all_runs)
    last = agg['last']

    # ── Section 11: Evaluation — print tables ─────────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION 11 — EVALUATION')
    print('=' * 72)

    print('\n  Table 1 — Out-of-sample accuracy  (mean ± std)')
    print(f'  {"Model":<22}  {"RMSE (mean)":>12}  {"RMSE (std)":>12}  '
          f'{"MAE (mean)":>12}  {"MAE (std)":>12}')
    print('  ' + '-' * 74)
    for nm in MODEL_NAMES:
        if nm in agg['rmse_mean']:
            print(f'  {nm:<22}  '
                  f'{agg["rmse_mean"][nm]:>12.5f}  '
                  f'{agg["rmse_std"][nm]:>12.5f}  '
                  f'{agg["mae_mean"][nm]:>12.5f}  '
                  f'{agg["mae_std"][nm]:>12.5f}')

    print('\n  Table 2 — Own-price elasticities  (mean ± std)')
    for nm in MODEL_NAMES[:10]:   # first 10 models for brevity
        if nm in agg['elast_mean']:
            e_m = agg['elast_mean'][nm]
            e_s = agg['elast_std'][nm]
            print(f'  {nm:<22}  ' +
                  '  '.join(f'{m:+.3f}±{s:.3f}' for m, s in zip(e_m, e_s)))

    print('\n  Table 3 — Welfare CV  (mean ± std)')
    for nm in MODEL_NAMES:
        if nm in agg['welf_mean']:
            print(f'  {nm:<22}  '
                  f'{agg["welf_mean"][nm]:+.5f} ± {agg["welf_std"][nm]:.5f}')

    print('\n  Delta CS:',
          f'δ̂ = {agg["dom_delta_hat"]:.3f}',
          f'IS = [{agg["dom_delta_cs_lo"]:.3f}, {agg["dom_delta_cs_hi"]:.3f}]')

    # ── Section 12: Figures ───────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION 12 — FIGURES')
    print('=' * 72)
    t0_fig = time.time()
    _make_figures(agg, splits, CFG)
    print(f'  Figures done ({time.time() - t0_fig:.1f}s).')

    # ── Sections 13–14: CSV Tables + LaTeX ───────────────────────────────
    print('\n' + '=' * 72)
    print('  SECTIONS 13–14 — CSV TABLES + LATEX')
    print('=' * 72)
    t0_tab = time.time()
    _make_tables(agg, splits, CFG)
    print(f'  Tables done ({time.time() - t0_tab:.1f}s).')

    # ── Section D16: KL profile over δ ───────────────────────────────────
    print('\n' + '=' * 72)
    print("  SECTION D16 — KL PROFILE OVER δ  (Dominick's, frozen weights)")
    print('=' * 72)
    t0 = time.time()
    kl_result = run_kl_profile(last, splits, CFG)
    print(f'  Done ({time.time() - t0:.1f}s).')

    # ── Section D17: ICC decomposition ────────────────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION D17 — WITHIN/BETWEEN STORE VARIANCE DECOMPOSITION  (ICC)')
    print('=' * 72)
    t0 = time.time()
    icc_result = run_icc(splits, CFG)
    print(f'  Done ({time.time() - t0:.1f}s).')

    # ── Section D18: Welfare sensitivity ──────────────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION D18 — δ SENSITIVITY: WELFARE ROBUSTNESS')
    print('=' * 72)
    t0 = time.time()
    sens_result = run_welfare_sensitivity(agg['welf_mean'], splits, CFG)
    print(f'  Done ({time.time() - t0:.1f}s).')

    # ── Section D19: Linear-in-x̄ identification ──────────────────────────
    print('\n' + '=' * 72)
    print('  SECTION D19 — LINEAR-IN-x̄ δ IDENTIFICATION (EMPIRICAL)')
    print('=' * 72)
    t0 = time.time()
    lx_result = run_linear_xbar_id(all_runs, splits, CFG)
    print(f'  Done ({time.time() - t0:.1f}s).')

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    print('\n' + '=' * 72)
    print(f"  ALL DOMINICK'S EXPERIMENTS COMPLETED  ({elapsed:.0f}s total)")
    print('=' * 72)

    return dict(
        all_runs=all_runs,
        agg=agg,
        kl_result=kl_result,
        icc_result=icc_result,
        sens_result=sens_result,
        lx_result=lx_result,
        splits=splits,
    )


if __name__ == '__main__':
    main()
