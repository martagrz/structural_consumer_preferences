"""
experiments/dominicks/exp05_linear_xbar.py
===========================================
Section D19 of dominicks_multiple_runs.py.

Linear-in-x̄ δ identification — empirical validation on Dominick's analgesics.

Theory (mirrors simulation Section 22 in main_multiple_runs.py)
----------------------------------------------------------------
If R_ψ(p, y, x̄) = f_ψ(p, y) + θ · x̄, then for any two observations
(t, t') sharing the same (p, y) but different consumption histories:

    Δlog w_j ≡ log w_{j,t} − log w_{j,t'} = θ_j · Δx̄_j(δ)

because f_ψ(p, y) cancels exactly.  Profiling out θ̂_j(δ) by per-good
OLS over matched pairs yields a normalised residual M(δ) that achieves
its minimum at the true habit-decay.

Empirical identification
------------------------
• Within-store pairs: observations from the same store in different
  weeks with similar (p, y) — the shared store environment controls
  for f_ψ while the EWMA habit stock varies across weeks.
• Quantile-bin (log p₁, log p₂, log p₃, log y) within each store;
  pair observations that fall in the same bin across different weeks.
• Sweep δ ∈ [0.05, 0.98]; compare argmin M(δ) with the MDP E2E δ̂.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.dominicks.data import G, GOODS


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _d19_xbar_sweep(lw_panel: np.ndarray,
                    store_ids: np.ndarray,
                    delta: float) -> np.ndarray:
    """Store-aware EWMA log-habit-stock for a given delta (NumPy, no grad)."""
    N, Gd = lw_panel.shape
    xb   = np.zeros((N, Gd))
    prev = lw_panel.mean(0)
    for i in range(N):
        if i > 0 and store_ids[i] != store_ids[i - 1]:
            prev = lw_panel.mean(0)
        xb[i] = prev
        prev  = delta * prev + (1.0 - delta) * lw_panel[i]
    return xb


def _qbin(arr: np.ndarray, B: int) -> np.ndarray:
    """Rank-based quantile discretisation into B bins (0 … B-1)."""
    pcts = np.percentile(arr, np.linspace(0, 100, B + 1))
    return np.searchsorted(pcts[1:-1], arr)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_xbar_id(
    all_runs: list,
    splits: dict,
    cfg: dict,
    n_bins: int = 4,
    max_pairs: int = 5000,
    n_grid: int = 80,
) -> dict:
    """Linear-in-x̄ δ identification on Dominick's training set.

    Parameters
    ----------
    all_runs  : list of ``run_once`` result dicts (for MDP E2E δ̂ estimate).
    splits    : data splits dict from ``experiments.dominicks.data.load()``.
    cfg       : configuration dict.
    n_bins    : quantile bins per price/income dimension (n_bins^4 cells per store).
    max_pairs : cap on the number of matched pairs.
    n_grid    : δ sweep grid resolution.

    Returns
    -------
    dict with keys:
        n_pairs, delta_grid, residuals, corr_moments,
        argmin_delta, argmin_corr, delta_fit,
        DW, DXbar_fit, DXbar_wrong
    """
    p_tr   = splits['p_tr']
    w_tr   = splits['w_tr']
    y_tr   = splits['y_tr']
    s_tr   = splits['s_tr']
    shares = splits['shares']
    stores = splits['stores']
    _store_map = splits['_store_map']

    # Full-panel log-shares (matching build_arrays convention for xbar updates)
    lw_all = np.log(np.maximum(shares, 1e-8))   # (N_all, G) full panel
    st_all = stores                               # (N_all,) store IDs

    lp_tr = np.log(np.maximum(p_tr, 1e-8))       # (N_tr, G)
    ly_tr = np.log(np.maximum(y_tr, 1e-8))        # (N_tr,)
    lw_tr = np.log(np.maximum(w_tr, 1e-8))        # (N_tr, G) observed log shares

    # ── MDP E2E estimated δ̂ ────────────────────────────────────────────
    try:
        delta_fit = float(np.nanmean([r['delta_e2e'] for r in all_runs]))
    except Exception:
        delta_fit = 0.70
    print(f'\n  Within-store matched pairs build (n_bins={n_bins}, '
          f'max_pairs={max_pairs})')
    print(f'  MDP E2E estimated δ̂: {delta_fit:.4f}')

    # ── Build matched pairs within store × price-income bin ───────────
    store_int = np.array([_store_map.get(int(s), 0) for s in s_tr])
    cell = (
        store_int                            * n_bins ** 4
        + _qbin(lp_tr[:, 0], n_bins)        * n_bins ** 3
        + _qbin(lp_tr[:, 1], n_bins)        * n_bins ** 2
        + _qbin(lp_tr[:, 2], n_bins)        * n_bins
        + _qbin(ly_tr,        n_bins)
    )

    pairs = []
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        if len(idx) >= 2:
            for ii in range(len(idx)):
                for jj in range(ii + 1, len(idx)):
                    pairs.append((idx[ii], idx[jj]))

    rng = np.random.default_rng(19)
    if len(pairs) > max_pairs:
        sel   = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]

    pairs   = np.array(pairs, dtype=int)
    n_pairs = len(pairs)
    print(f'  Within-store matched pairs (same price-income bin): {n_pairs:,}')

    if n_pairs < 20:
        print('  ⚠  Too few matched pairs — skipping Section D19.')
        return {
            'n_pairs': n_pairs, 'skipped': True,
            'delta_grid': np.linspace(0.05, 0.98, n_grid),
            'residuals':  np.full(n_grid, float('nan')),
            'corr_moments': np.full(n_grid, float('nan')),
            'argmin_delta': float('nan'), 'argmin_corr': float('nan'),
            'delta_fit': delta_fit,
            'DW': None, 'DXbar_fit': None, 'DXbar_wrong': None,
        }

    DW = lw_tr[pairs[:, 0]] - lw_tr[pairs[:, 1]]   # (P, G)

    # ── Identification sweep ──────────────────────────────────────────
    delta_grid   = np.linspace(0.05, 0.98, n_grid)
    _step        = delta_grid[1] - delta_grid[0]
    residuals    = np.zeros(n_grid)
    corr_moments = np.zeros(n_grid)
    DXbar_fit    = None
    DXbar_wrong  = None

    # Need train-row indices (tr) to slice the full-panel xbar
    tr = splits.get('tr')

    for di, d in enumerate(delta_grid):
        xb_full = _d19_xbar_sweep(lw_all, st_all, d)
        if tr is not None:
            xb_tr_d = xb_full[tr]
        else:
            xb_tr_d = xb_full[:len(p_tr)]
        DXbar = xb_tr_d[pairs[:, 0]] - xb_tr_d[pairs[:, 1]]   # (P, G)

        r_tot = 0.0; c_tot = 0.0
        for g in range(G):
            dw = DW[:, g];  dx = DXbar[:, g]
            ss = np.dot(dx, dx)
            if ss < 1e-12:
                r_tot += 1.0; c_tot += 1.0; continue
            th    = np.dot(dw, dx) / ss
            resid = dw - th * dx
            ss_dw = max(np.dot(dw, dw), 1e-12)
            r_tot += np.dot(resid, resid) / ss_dw
            if dw.std() > 1e-8 and dx.std() > 1e-8:
                c_tot += 1.0 - abs(float(np.corrcoef(dw, dx)[0, 1]))
            else:
                c_tot += 1.0

        residuals[di]    = r_tot / G
        corr_moments[di] = c_tot / G

        if DXbar_fit is None and d >= delta_fit - _step:
            DXbar_fit = DXbar.copy()
        if DXbar_wrong is None and d >= 0.29:
            DXbar_wrong = DXbar.copy()

    argmin_res  = float(delta_grid[np.argmin(residuals)])
    argmin_corr = float(delta_grid[np.argmin(corr_moments)])

    print(f'  argmin M(δ)        = {argmin_res:.4f}')
    print(f'  argmin |corr|(δ)   = {argmin_corr:.4f}')
    print(f'  MDP E2E δ̂          = {delta_fit:.4f}')
    close = abs(argmin_res - delta_fit) < 0.15
    if close:
        print(f'  ✓ argmin M(δ) is within 0.15 of MDP E2E δ̂ '
              f'(gap = {abs(argmin_res - delta_fit):.3f})')
    else:
        print(f'  ⚠ argmin M(δ) differs from MDP E2E δ̂ by '
              f'{abs(argmin_res - delta_fit):.3f}')

    # ── Figure ────────────────────────────────────────────────────────
    GCOLS = ['#2196F3', '#4CAF50', '#FF5722']
    GLBLS = GOODS

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: M(δ) curve
    ax_a = axes[0]
    ax_a.plot(delta_grid, residuals,    color='#1565C0', lw=2.5,
              label=r'OLS residual $M(\delta)$')
    ax_a.plot(delta_grid, corr_moments, color='#E65100', lw=1.8, ls='--',
              label=r'$1 - |\mathrm{corr}|$ moment')
    ax_a.axvline(argmin_res,  color='#1565C0', ls=':', lw=2.0,
                 label=rf'argmin $M(\delta)$ = {argmin_res:.3f}')
    ax_a.axvline(delta_fit,   color='#9C27B0', ls='-.', lw=2.0,
                 label=rf'MDP E2E $\hat{{\delta}}$ = {delta_fit:.3f}')
    ax_a.set_xlabel(r'Habit-decay parameter $\delta$', fontsize=12)
    ax_a.set_ylabel(r'Normalised residual $M(\delta)$', fontsize=12)
    ax_a.set_title(
        "Panel A: δ Identification Curve\n"
        r"(Dominick's analgesics, linear-in-$\bar{x}$ restriction)",
        fontsize=11, fontweight='bold')
    ax_a.legend(fontsize=9); ax_a.grid(True, alpha=0.3)

    # Panels B & C
    for ax_, DXb_, ptitle, dlbl in [
        (axes[1], DXbar_fit,
         'Panel B: Fitted δ', f'δ = {delta_fit:.3f} (MDP E2E est.)'),
        (axes[2], DXbar_wrong,
         'Panel C: Wrong δ',  'δ = 0.30 (wrong)'),
    ]:
        if DXb_ is None:
            ax_.set_visible(False)
            continue
        for g in range(G):
            dw = DW[:, g]; dx = DXb_[:, g]
            ax_.scatter(dx, dw, s=5, alpha=0.25, color=GCOLS[g], label=GLBLS[g])
            ss = np.dot(dx, dx)
            if ss > 1e-12:
                th = np.dot(dw, dx) / ss
                xr = np.array([dx.min(), dx.max()])
                ax_.plot(xr, th * xr, color=GCOLS[g], lw=1.8, alpha=0.85)
        ax_.set_xlabel(r'$\Delta\bar{x}_j(\delta)$  (habit-stock diff.)', fontsize=11)
        ax_.set_ylabel(r'$\Delta\log w_j$  (log-share diff.)', fontsize=11)
        ax_.set_title(f'{ptitle}: Δlog w vs Δx̄\n({dlbl})',
                      fontsize=11, fontweight='bold')
        ax_.legend(fontsize=9, markerscale=3); ax_.grid(True, alpha=0.3)

    fig.suptitle(
        r"Section D19 — Linear-in-$\bar{x}$ Reward: $\delta$ Identification"
        r" (Dominick's Analgesics)"
        '\n'
        r'Within-store matched pairs (same price-income bin, different weeks) '
        r'cancel $f_\psi(p, y)$; '
        r'$M(\delta) = \sum_j \|\Delta\log w_j - \hat{\theta}_j(\delta)\,'
        r'\Delta\bar{x}_j(\delta)\|^2 / \|\Delta\log w_j\|^2$'
        f'\n{n_pairs:,} matched pairs  ·  {len(p_tr):,} training obs  ·  '
        f'MDP E2E δ̂ = {delta_fit:.3f}  ·  '
        f'argmin M = {argmin_res:.3f}',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    for _ext in ('pdf', 'png'):
        fig.savefig(f"{cfg['fig_dir']}/fig_d19_linear_xbar_id.{_ext}",
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {cfg['fig_dir']}/fig_d19_linear_xbar_id.pdf")

    return dict(
        n_pairs=n_pairs,
        delta_grid=delta_grid,
        residuals=residuals,
        corr_moments=corr_moments,
        argmin_delta=argmin_res,
        argmin_corr=argmin_corr,
        delta_fit=delta_fit,
        DW=DW,
        DXbar_fit=DXbar_fit,
        DXbar_wrong=DXbar_wrong,
    )
