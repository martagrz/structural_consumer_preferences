"""
experiments/dominicks/exp02_kl_profile.py
==========================================
Section D16 of dominicks_multiple_runs.py.

KL profile over δ with frozen network weights (post-hoc δ sweep).
Holds the MDP Neural IRL and MDP E2E weights frozen at convergence
(from the last training run) and sweeps δ over a grid, recomputing
xbar at each value.

Interpretation
--------------
  Flat / very shallow profile → δ weakly identified; network compensates.
  Sharp minimum at δ̂          → δ well-identified from budget-share data.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.dominicks import compute_xbar_e2e
from experiments.dominicks.utils import kl_div


def run_kl_profile(
    last_run: dict,
    splits: dict,
    cfg: dict,
    delta_grid: np.ndarray | None = None,
) -> dict:
    """Sweep δ with frozen weights; return KL profile arrays and figure.

    Parameters
    ----------
    last_run    : result dict from ``run_once`` (last seed).
    splits      : data splits dict from ``experiments.dominicks.data.load()``.
    cfg         : configuration dict.
    delta_grid  : 1-D array of δ values to sweep.  Defaults to
                  ``np.arange(0.10, 1.00, 0.1)``.

    Returns
    -------
    dict with keys:
        delta_grid, kl_mdp_arr, kl_e2e_arr,
        argmin_delta_e2e, range_e2e, range_mdp, flat_note, fig
    """
    if delta_grid is None:
        delta_grid = np.arange(0.10, 1.00, 0.1)

    p_te  = splits['p_te']
    w_te  = splits['w_te']
    y_te  = splits['y_te']
    xb_te = splits['xb_te']
    qp_te = splits['qp_te']
    ls_te = splits.get('ls_te')  # raw log-shares for E2E xbar computation
    s_te  = splits.get('s_te')
    dev   = cfg['device']

    KW = last_run['KW']

    _ls_te_tensor = torch.tensor(ls_te, dtype=torch.float32).to(dev)

    kl_mdp_profile: list[float] = []
    kl_e2e_profile: list[float] = []

    print(f'  Sweeping δ ∈ [{delta_grid[0]:.2f}, {delta_grid[-1]:.2f}]  '
          f'({len(delta_grid)} points) on test set ({len(p_te):,} obs) ...')

    with torch.no_grad():
        for _dkl in delta_grid:
            _dt = torch.tensor(_dkl, dtype=torch.float32, device=dev)
            _xb_kl = compute_xbar_e2e(
                _dt, _ls_te_tensor,
                store_ids=None).cpu().numpy()

            try:
                kl_mdp_profile.append(
                    kl_div('mdp', p_te, y_te, w_te, cfg,
                           xb_prev=_xb_kl, q_prev=qp_te, **KW))
            except Exception:
                kl_mdp_profile.append(float('nan'))

            try:
                kl_e2e_profile.append(
                    kl_div('mdp-e2e', p_te, y_te, w_te, cfg,
                           xb_prev=_xb_kl, **KW))
            except Exception:
                kl_e2e_profile.append(float('nan'))

    kl_mdp_arr = np.array(kl_mdp_profile)
    kl_e2e_arr = np.array(kl_e2e_profile)

    _range_e2e = float(np.nanmax(kl_e2e_arr) - np.nanmin(kl_e2e_arr))
    _range_mdp = float(np.nanmax(kl_mdp_arr) - np.nanmin(kl_mdp_arr))
    flat_note  = (
        "Profile is FLAT (range < 5×min) → δ weakly identified → "
        "observational equivalence"
        if _range_e2e < 5 * max(np.nanmin(kl_e2e_arr), 1e-9)
        else "Profile shows curvature → δ partially identified"
    )

    argmin_delta = float(delta_grid[np.nanargmin(kl_e2e_arr)])
    print(f'  E2E KL minimum: δ={argmin_delta:.3f}  '
          f'KL={kl_e2e_arr[np.nanargmin(kl_e2e_arr)]:.5f}')
    print(f'  E2E KL range : {_range_e2e:.5f}  ({flat_note})')
    print(f'  MDP blend KL range : {_range_mdp:.5f}')

    # ── Figure ────────────────────────────────────────────────────────────
    TEAL = '#009688'
    fig, ax = plt.subplots(figsize=(10, 5))

    delta_m_mu  = float(np.mean([r['delta_mdp']  for r in [last_run]]))
    delta_e2e_mu = float(np.mean([r['delta_e2e']  for r in [last_run]]))

    ax.plot(delta_grid, kl_mdp_arr,
            color=TEAL,      lw=2.5, label=r'MDP Neural IRL (blend $\bar{x}$)')
    ax.plot(delta_grid, kl_e2e_arr,
            color='#FF6F00', lw=2.5, label=r'MDP IRL (E2E $\hat{\delta}$)')
    ax.axvline(delta_m_mu,   color=TEAL,      ls=':',  lw=1.8,
               label=rf'Blend $\hat{{\delta}}$ = {delta_m_mu:.3f}')
    ax.axvline(delta_e2e_mu, color='#FF6F00', ls='-.', lw=1.8,
               label=rf'E2E $\hat{{\delta}}$ = {delta_e2e_mu:.3f}')

    ax.set_xlabel(r'Habit-decay parameter $\delta$', fontsize=13)
    ax.set_ylabel('KL divergence (test set)', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        "KL Loss Profile over δ — Dominick's Analgesics  (network weights frozen)\n"
        r"x-axis: δ swept  ·  y-axis: KL(predicted || observed) on test set"
        f"\n{flat_note}",
        fontsize=11, fontweight='bold')
    fig.tight_layout()

    for _ext in ('pdf', 'png'):
        fig.savefig(f"{cfg['fig_dir']}/fig_kl_delta_profile.{_ext}",
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig_kl_delta_profile')

    return dict(
        delta_grid=delta_grid,
        kl_mdp_arr=kl_mdp_arr,
        kl_e2e_arr=kl_e2e_arr,
        argmin_delta_e2e=argmin_delta,
        range_e2e=_range_e2e,
        range_mdp=_range_mdp,
        flat_note=flat_note,
    )
