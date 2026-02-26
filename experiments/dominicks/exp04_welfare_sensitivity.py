"""
experiments/dominicks/exp04_welfare_sensitivity.py
====================================================
Section D18 of dominicks_multiple_runs.py.

δ sensitivity — welfare robustness check.

Re-trains MDP Neural IRL with δ fixed (not learned) at each value in
{0.50, 0.60, 0.70, 0.80, 0.90}.  Reports out-of-sample RMSE and
CV welfare for each value, confirming that the welfare conclusion
(MDP implies a substantially larger welfare loss than static Neural IRL)
is robust across the entire identified δ set.
"""

from __future__ import annotations

import numpy as np
import torch
import pandas as pd

from src.models.dominicks import MDPNeuralIRL, _train
from experiments.dominicks.utils import pred, comp_var
from experiments.dominicks.data import G


def _xbar_fixed_delta(shares: np.ndarray, stores: np.ndarray, delta: float) -> np.ndarray:
    """Recompute (N, G) log-habit stock for a fixed scalar delta.

    Mirrors ``build_arrays`` in data.py: full panel sorted by (STORE, WEEK),
    warm-started at the global mean log-share, then returned in the original
    row order.

    Parameters
    ----------
    shares : (N, G) share array for the *full* panel (train + test).
    stores : (N,)   store IDs in the same row order as shares.
    delta  : habit-decay scalar ∈ (0, 1).
    """
    log_w = np.log(np.maximum(shares, 1e-6))
    gm    = log_w.mean(0)
    xb    = np.zeros_like(log_w)
    prev  = gm.copy()
    for i in range(len(log_w)):
        if i > 0 and stores[i] != stores[i - 1]:
            prev = gm.copy()
        xb[i] = prev
        prev   = delta * prev + (1.0 - delta) * log_w[i]
    return xb


def run_welfare_sensitivity(
    welf_mean: dict,
    splits: dict,
    cfg: dict,
    delta_grid: list[float] | None = None,
    n_seeds: int = 3,
) -> dict:
    """Train MDP at each fixed δ; collect RMSE and CV welfare.

    Parameters
    ----------
    welf_mean   : aggregated welfare means from ``exp01_main_runs.aggregate()``
                  (used as the Neural IRL baseline CV).
    splits      : data splits dict from ``experiments.dominicks.data.load()``.
    cfg         : configuration dict.
    delta_grid  : list of δ values to sweep.
                  Defaults to ``[0.50, 0.60, 0.70, 0.80, 0.90]``.
    n_seeds     : number of random seeds per δ.

    Returns
    -------
    dict with keys:
        dsens_res   — {delta → {cv_mean, cv_std, rmse_mean, rmse_std}}
        nirl_cv_ref — Neural IRL baseline CV (float)
        pct_range   — span (pp) of % vs Neural IRL across δ grid
        dsens_df    — pd.DataFrame with the sensitivity table
    """
    if delta_grid is None:
        delta_grid = [0.50, 0.60, 0.70, 0.80, 0.90]

    p_tr   = splits['p_tr'];  p_te   = splits['p_te']
    w_tr   = splits['w_tr'];  w_te   = splits['w_te']
    y_tr   = splits['y_tr'];  y_te   = splits['y_te']
    qp_tr  = splits['qp_tr']; qp_te  = splits['qp_te']
    shares = splits['shares']
    stores = splits['stores']
    tr     = splits.get('tr')
    te     = splits.get('te')
    p0w    = splits['p0w']
    p1w    = splits['p1w']
    y_mn   = splits['y_mn']
    qp_mn  = splits['qp_mn']

    nirl_cv_ref = welf_mean.get('Neural IRL', float('nan'))
    seeds = [42 + i * 7 for i in range(n_seeds)]

    print(f'  Grid : {delta_grid}')
    print(f'  Seeds: {seeds}  ({n_seeds} per δ)')
    print(f'  Neural IRL CV baseline: {nirl_cv_ref:+.6f}')

    dsens_res: dict = {}

    for dval in delta_grid:
        # Recompute xbar with this fixed δ
        xb_full = _xbar_fixed_delta(shares, stores, dval)
        if tr is not None and te is not None:
            xb_d_tr, xb_d_te = xb_full[tr], xb_full[te]
        else:
            # fallback: use the split arrays directly (should not happen)
            xb_d_tr = xb_full[:len(p_tr)]
            xb_d_te = xb_full[len(p_tr):]
        xb_d_mn = xb_d_te.mean(0)

        cv_seeds   = []
        rmse_seeds = []
        print(f'\n  δ={dval:.2f} ', end='', flush=True)
        for s in seeds:
            np.random.seed(s); torch.manual_seed(s)
            mdp_d, _ = _train(
                MDPNeuralIRL(cfg['mdp_hidden']),
                p_tr, y_tr, w_tr, 'mdp', cfg,
                xb_prev_tr=xb_d_tr,
                q_prev_tr=qp_tr,
                tag=f'MDP δ={dval:.2f} s={s}')
            _KW_d = {'mdp': mdp_d}
            cv_seeds.append(
                comp_var('mdp', p0w, p1w, y_mn, cfg,
                         xb_prev0=xb_d_mn, q_prev0=qp_mn, **_KW_d))
            wp_d = pred('mdp', p_te, y_te, cfg,
                        xb_prev=xb_d_te, q_prev=qp_te, **_KW_d)
            rmse_seeds.append(float(np.sqrt(np.mean((w_te - wp_d) ** 2))))
            print('.', end='', flush=True)

        dsens_res[dval] = {
            'cv_mean':   float(np.nanmean(cv_seeds)),
            'cv_std':    float(np.nanstd(cv_seeds,   ddof=min(1, n_seeds - 1))),
            'rmse_mean': float(np.nanmean(rmse_seeds)),
            'rmse_std':  float(np.nanstd(rmse_seeds, ddof=min(1, n_seeds - 1))),
        }
        pct_d = (100 * (dsens_res[dval]['cv_mean'] - nirl_cv_ref) / abs(nirl_cv_ref)
                 if not np.isnan(nirl_cv_ref) else float('nan'))
        print(f'  CV={dsens_res[dval]["cv_mean"]:+.5f} ± {dsens_res[dval]["cv_std"]:.5f}'
              f'  ({pct_d:+.1f}% vs Neural IRL)')

    # ── Summary ───────────────────────────────────────────────────────────
    cv_vals  = [dsens_res[d]['cv_mean'] for d in delta_grid]
    pct_vals = [100 * (v - nirl_cv_ref) / abs(nirl_cv_ref)
                for v in cv_vals
                if not np.isnan(nirl_cv_ref)]
    cv_range  = float(np.nanmax(cv_vals) - np.nanmin(cv_vals))
    pct_lo    = float(min(pct_vals)) if pct_vals else float('nan')
    pct_hi    = float(max(pct_vals)) if pct_vals else float('nan')
    pct_range = pct_hi - pct_lo if pct_vals else float('nan')

    print('\n')
    print('  ' + '─' * 70)
    print(f'  {"δ":>5}  {"RMSE (mean ± std)":>22}  '
          f'{"CV loss (mean ± std)":>24}  {"% vs Neural IRL":>16}')
    print('  ' + '─' * 70)
    for dval in delta_grid:
        r   = dsens_res[dval]
        pct = (100 * (r['cv_mean'] - nirl_cv_ref) / abs(nirl_cv_ref)
               if not np.isnan(nirl_cv_ref) else float('nan'))
        print(f'  {dval:.2f}  '
              f'{r["rmse_mean"]:.5f} ± {r["rmse_std"]:.5f}  '
              f'{r["cv_mean"]:+.5f} ± {r["cv_std"]:.5f}   '
              f'{pct:+.1f}%')
    print('  ' + '─' * 70)
    if pct_range < 6.0:
        print(f'  ✓  Span of {pct_range:.1f} pp — NARROW.  '
              f'Welfare conclusion is robust across the full identified δ set.')
    else:
        print(f'  ⚠  Span of {pct_range:.1f} pp — non-trivial sensitivity to δ.')

    # ── CSV ───────────────────────────────────────────────────────────────
    dsens_df = pd.DataFrame([
        {'delta':       d,
         'rmse_mean':   dsens_res[d]['rmse_mean'],
         'rmse_std':    dsens_res[d]['rmse_std'],
         'cv_mean':     dsens_res[d]['cv_mean'],
         'cv_std':      dsens_res[d]['cv_std'],
         'pct_vs_nirl': (100 * (dsens_res[d]['cv_mean'] - nirl_cv_ref) / abs(nirl_cv_ref)
                         if not np.isnan(nirl_cv_ref) else float('nan'))}
        for d in delta_grid
    ]).round(6)
    dsens_df.to_csv(f"{cfg['out_dir']}/table_delta_sensitivity.csv", index=False)
    print(f"  Saved: {cfg['out_dir']}/table_delta_sensitivity.csv")

    return dict(
        dsens_res=dsens_res,
        nirl_cv_ref=nirl_cv_ref,
        cv_range=cv_range,
        pct_lo=pct_lo,
        pct_hi=pct_hi,
        pct_range=pct_range,
        dsens_df=dsens_df,
    )
