"""
experiments/dominicks/exp03_icc.py
====================================
Section D17 of dominicks_multiple_runs.py.

Within-store vs between-store variance decomposition of the MDP habit stock
x̄_t.  Reports the intraclass correlation coefficient (ICC).

ICC = σ²_between / (σ²_between + σ²_within)

ICC < 0.3  → most variation is within-store over time
             → supports dynamic interpretation of habit persistence.
ICC > 0.6  → most variation is cross-sectional
             → sorting concern dominates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def run_icc(splits: dict, cfg: dict) -> dict:
    """Compute ICC decomposition for each good and produce the figure.

    Parameters
    ----------
    splits : data splits dict from ``experiments.dominicks.data.load()``.
    cfg    : configuration dict.

    Returns
    -------
    dict with keys:
        icc_results  — dict {good_name → {between_var, within_var, total_var,
                                           icc, interpretation}}
        icc_df       — pd.DataFrame table
    """
    from experiments.dominicks.data import GOODS, G

    xb_tr  = splits['xb_tr']
    xb_te  = splits['xb_te']
    s_tr   = splits['s_tr']
    s_te   = splits['s_te']
    N_STORES  = splits['N_STORES']
    _store_map = splits['_store_map']

    xb_all = np.concatenate([xb_tr, xb_te], axis=0)
    st_all = np.concatenate([s_tr,  s_te],  axis=0)

    print(f'  N_all={len(xb_all):,}  N_stores={N_STORES}')
    print(f'  Goods: {GOODS}')

    _store_ids_all = np.array([_store_map[int(s)] for s in st_all])

    icc_results: dict = {}
    for gi, gname in enumerate(GOODS):
        xb_g     = xb_all[:, gi]
        grand_m  = xb_g.mean()
        grand_v  = xb_g.var()

        store_means = np.array([
            xb_g[_store_ids_all == si].mean()
            if (_store_ids_all == si).sum() > 0 else grand_m
            for si in range(N_STORES)
        ])
        n_per = np.array([(_store_ids_all == si).sum()
                          for si in range(N_STORES)], dtype=float)
        bw_var = float(np.average((store_means - grand_m) ** 2, weights=n_per))

        wi_vars = [xb_g[_store_ids_all == si].var()
                   for si in range(N_STORES)
                   if (_store_ids_all == si).sum() > 1]
        wi_var = float(np.mean(wi_vars)) if wi_vars else 0.0

        icc = bw_var / (bw_var + wi_var + 1e-12)
        icc_results[gname] = {
            "between_var":    bw_var,
            "within_var":     wi_var,
            "total_var":      grand_v,
            "icc":            icc,
            "interpretation": (
                "within-store dominant → supports dynamic habit"
                if icc < 0.3 else
                "moderate between-store → mixed evidence"
                if icc < 0.6 else
                "between-store dominant → sorting concern ⚠"
            ),
        }

    # ── Print table ───────────────────────────────────────────────────────
    print(f'\n  {"Good":<14}  {"Between σ²":>12}  {"Within σ²":>12}  '
          f'{"Total σ²":>12}  {"ICC":>8}  Interpretation')
    for gname, d in icc_results.items():
        print(f'  {gname:<14}  '
              f'{d["between_var"]:>12.5f}  '
              f'{d["within_var"]:>12.5f}  '
              f'{d["total_var"]:>12.5f}  '
              f'{d["icc"]:>8.4f}  {d["interpretation"]}')

    # ── Save CSV ──────────────────────────────────────────────────────────
    icc_df = pd.DataFrame([
        {"Good": k, **{kk: vv for kk, vv in v.items()}}
        for k, v in icc_results.items()
    ])
    icc_df.round(5).to_csv(f"{cfg['out_dir']}/table_icc.csv", index=False)
    print(f"  Saved: {cfg['out_dir']}/table_icc.csv")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    goods_lbl = [g[:8] for g in GOODS]
    bw_vars   = [icc_results[g]["between_var"] for g in GOODS]
    wi_vars   = [icc_results[g]["within_var"]  for g in GOODS]
    iccs      = [icc_results[g]["icc"]         for g in GOODS]
    x_icc = np.arange(G)

    bw_col = '#E53935'; wi_col = '#1E88E5'
    ax_a.bar(x_icc - 0.2, bw_vars, 0.35, color=bw_col,
             label='Between-store σ²', edgecolor='k', lw=0.8)
    ax_a.bar(x_icc + 0.2, wi_vars, 0.35, color=wi_col,
             label='Within-store σ²',  edgecolor='k', lw=0.8)
    ax_a.set_xticks(x_icc); ax_a.set_xticklabels(goods_lbl, fontsize=12)
    ax_a.set_ylabel('Variance of log-habit stock $\\bar{x}$', fontsize=11)
    ax_a.set_title('Panel A: Between vs Within-Store Variance',
                   fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=11); ax_a.grid(axis='y', alpha=0.3)

    icc_cols = [('#43A047' if v < 0.3 else '#FFA726' if v < 0.6 else '#E53935')
                for v in iccs]
    ax_b.bar(x_icc, iccs, 0.5, color=icc_cols, edgecolor='k', lw=0.8)
    ax_b.axhline(0.3, color='green',  ls='--', lw=1.5,
                 label='ICC=0.3 (within-store threshold)')
    ax_b.axhline(0.6, color='orange', ls='--', lw=1.5,
                 label='ICC=0.6 (between-store threshold)')
    for xi, v in zip(x_icc, iccs):
        ax_b.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
                  fontsize=12, fontweight='bold')
    ax_b.set_xticks(x_icc); ax_b.set_xticklabels(goods_lbl, fontsize=12)
    ax_b.set_ylim(0, max(max(iccs) + 0.15, 0.75))
    ax_b.set_ylabel('Intraclass Correlation Coefficient (ICC)', fontsize=11)
    ax_b.set_title('Panel B: ICC — Between-Store Fraction of Total Variance',
                   fontsize=12, fontweight='bold')
    legend_els = [
        Patch(color='#43A047', label='ICC < 0.3: within-store dominant'),
        Patch(color='#FFA726', label='0.3 ≤ ICC < 0.6: mixed'),
        Patch(color='#E53935', label='ICC ≥ 0.6: between-store dominant ⚠'),
    ]
    ax_b.legend(handles=legend_els + ax_b.get_legend_handles_labels()[0],
                fontsize=9, loc='upper right')
    ax_b.grid(axis='y', alpha=0.3)

    fig.suptitle(
        "Habit-Stock Variance Decomposition — Dominick's Analgesics\n"
        r"$\bar{x}_{it}$ = log-normalised habit stock (train + test sets combined)"
        f"\nN={len(xb_all):,} obs  ·  {N_STORES} stores  ·  "
        "ICC = between-store fraction of total variance",
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    for _ext in ('pdf', 'png'):
        fig.savefig(f"{cfg['fig_dir']}/fig_habit_icc.{_ext}",
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig_habit_icc')

    return dict(
        icc_results=icc_results,
        icc_df=icc_df,
    )
