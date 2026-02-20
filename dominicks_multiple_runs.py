"""
dominicks_irl.py  —  IRL Consumer Demand Recovery, Dominick's Analgesics
=========================================================================
Applies the full simulation model suite (LA-AIDS, BLP-IV, three Linear IRL
variants, Neural IRL, MDP Neural IRL, Variational Mixture IRL) to real
Dominick's Finer Foods scanner data, producing all tables and figures for
direct inclusion in the paper.

Input files
-----------
  wana_copy.csv  (or full wana.csv)
      STORE, UPC, WEEK, MOVE (units sold), QTY (pack-size multiplier),
      PRICE (shelf price, $), SALE, PROFIT, OK
  upcana.csv
      COM_CODE, UPC, DESCRIP, SIZE, CASE, NITEM

Three-good demand system
------------------------
  Good 0  Aspirin       : Bayer, Bufferin, Anacin, Ascriptin, Ecotrin,
                          Empirin, BC Powder, Vanquish, Cope, generics
  Good 1  Acetaminophen : Tylenol, Excedrin (all), Anacin-3/A/F, Panadol,
                          Datril, Pamprin, Midol (non-IB), DOM non-aspirin,
                          store-brand acetaminophen
  Good 2  Ibuprofen     : Advil, Motrin IB, Nuprin, Aleve, Actron, Orudis KT,
                          Haltran, Medipren, Midol 200/IB, Menadol,
                          DOM ibuprofen, Dominick's Tab-Profen, Trendar,
                          Vitalan, Excedrin IB, TC Iburofen

Price construction (unit-value method, Deaton 1988)
----------------------------------------------------
  unit_price = shelf_PRICE × 100 / tablet_count
  Standardised to $/100-tablet basis.
  Aggregated to (store, week, good) as revenue-weighted mean unit price.
  Zero-sales weeks: price forward-/backward-filled within store×good.

Outputs
-------
  figures/dominicks/fig_{demand_curves, mdp_advantage, convergence,
                         scatter, mixture}.{pdf,png}
  results/dominicks/table{0_desc, 1_accuracy, 2_elasticities,
                          3_welfare, 4_mdp, 5_mixture}.csv
  results/dominicks/dominicks_latex.tex

Standard-error estimation
-------------------------
  All stochastic models (Lin IRL, Neural IRL, MDP Neural IRL, Var. Mixture)
  are re-estimated n_runs times with different random seeds.  Deterministic
  models (LA-AIDS, BLP-IV) produce the same result each run, so their se=0
  is exact.  Tables report mean ± std; figures show ±1-std shaded bands on
  demand curves and error bars on bar charts.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.dominicks import (
    LAAIDS,
    BLPLogitIV,
    NeuralIRL,
    MDPNeuralIRL,
    VarMixture,
    _train,
    feat_good_specific,
    feat_orth,
    feat_shared,
    hausman_iv,
    pred_lirl,
    run_lirl,
)
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
CFG = dict(
    weekly_path   = './data/wana.csv',   # swap for full wana.csv
    upc_path      = './data/upcana.csv',
    std_tablets   = 100,               # normalise to $/100-tablet basis
    min_store_wks = 20,                # drop stores below this threshold
    test_cutoff   = 351,               # weeks ≥ 351 held out; fallback: 75/25
    # Number of independent re-estimation runs for standard errors
    n_runs        = 5,
    # Linear IRL
    lirl_lr=0.05, lirl_epochs=3000, lirl_l2=1e-4,
    # Neural IRL
    nirl_hidden=256, nirl_epochs=5000, nirl_lr=5e-4,
    nirl_batch=512, nirl_lam_mono=0.20, nirl_lam_slut=0.10,
    nirl_slut_start=0.25,

    # MDP Neural IRL
    mdp_hidden=256, mdp_epochs=5000, mdp_lr=5e-4,
    mdp_batch=512, mdp_lam_mono=0.15, mdp_lam_slut=0.08,
    mdp_slut_start=0.25,
    habit_decay=0.70,
    # Variational Mixture
    mix_K=6, 
    mix_n_spc=100,       # INCREASED from 5 -> 20 (Stable gradients)
    mix_n_iter=100,     # Enough iterations for convergence
    mix_lr_mu=0.05, 
    mix_sigma2=0.1,     # INCREASED from 0.003 -> 0.1 (Prevents mode collapse)
    mix_subsamp=300,    # Keep at 300 to manage speed
    # Welfare
    shock_good=2, shock_pct=0.10, cv_steps=100,
    # Output
    fig_dir='figures/dominicks', out_dir='results/dominicks',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

GOODS = ['Aspirin', 'Acetaminophen', 'Ibuprofen']
G = 3
os.makedirs(CFG['fig_dir'], exist_ok=True)
os.makedirs(CFG['out_dir'], exist_ok=True)

print('=' * 72)
print('  IRL DEMAND RECOVERY — DOMINICK\'S ANALGESICS')
print('=' * 72)
print(f"  Device: {CFG['device']}")
print(f"  n_runs: {CFG['n_runs']}  (standard errors from repeated re-estimation)")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1  DATA LOADING AND AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Product classification ────────────────────────────────────────────────────
# Validated against the complete 641-product Dominick's catalogue.
# Priority order: IBU first (Excedrin IB must override plain Excedrin),
# then ACET, then ASP.  Any UPC not matched → OTHER (excluded).

_IBU_KEYS = [
    'ADVIL', '#ADVIL', 'MOTRIN', '#MOTRIN', '~MOTRIN',
    'IBUPROFEN', 'NUPRIN', '~NUPRIN', 'ALEVE', 'IBUPRIN',
    'ACTRON', 'ORUDIS', 'HALTRAN', 'MEDIPREN',
    'MIDOL 200', 'MIDOL IB', '~MIDOL IB', 'MYDOL 200', 'MENADOL',
    'TRENDAR', 'VITALAN', 'DOM IBUPROFEN', 'DOMINICKS TAB-PROFEN',
    'TC IBUROFEN', 'TOPCARE IBUPROFEN',
    'EXCEDRIN IB', '~EXCEDRIN IB', '~ARTH FNDT IB',
    'F/S ADVIL', 'TRL-SZ ADVIL', 'TRL-SZ MOTRIN',
]
_ACET_KEYS = [
    'TYLENOL', 'TYLNL', 'ACETAMINOPHEN',
    'EXCEDRIN ASPIRIN-FREE', 'EXCEDRIN A/F', 'A/F EXCEDRIN',
    'A/F 20 CT EXCEDRIN', 'A/F 80 CT EXCEDRIN',
    '~A/F 40 CT EXCEDRIN', '*EXCEDRIN',
    'EXCEDRIN 175', 'EXCEDRIN CAPS', 'EXCEDRIN CAP ',
    'EXCEDRIN DUAL', 'EXCEDRIN GELTAB', 'EXCEDRIN TAB',
    'EXCEDRIN TABS', 'EXCEDRIN X', 'EXCEDRIN PM', 'EXCEDRIN P M',
    'EXCEDRIN P.M', 'EXCERIN PM', 'TRL-SZ EXCEDRIN', '~EXCEDRIN P.M',
    'PANADOL', 'DATRIL',
    'PAMPRIN', '~PAMPRIN',
    'ANACIN-3', 'ANACIN 3 ', 'M/S ANACIN 3', 'JHO-ANACIN 3',
    'ANACIN A/F', '~ANACIN A/F', '~ASP FREE ANANCIN', '~ASPIRIN FREE ANACIN',
    'DOM NON-ASP', 'DOM NON ASP', 'DOM X/S NON', 'DOM X-STR NON',
    'DOM REG STR NON', 'DOM CHILD CHEW', 'DOM CHILDS', 'DOM E/S NON',
    'DOM ADDED STRENGTH A', 'DOM X/S PAIN RELIEVE',
    'ST JOSEPH A/F', 'TC NON-ASP', 'TC X/STR N/A',
    'TEMPRA', 'FEVERALL', 'DORCOL', 'FEVERNOL', 'LIQUIPRIN',
    'PEDIA CARE INFANT', '$TC PAIN REL INFANT',
    'CHILD CHEW FRT TYLE', 'CHILD CHEW GRAP TYLE',
    'TS TYLENOL CHEW', 'CHLDRN PANADOL',
    'VALUE TIME ACETA',
    'MIDOL CAPLET', 'MIDOL CAPLETS MAX', 'MIDOL M/S', 'MIDOL MAXIMUM',
    'MIDOL PM', 'MIDOL PMS', 'MIDOLPMS', 'T/S MIDOL PM', '~MIDOL PM',
    'TEEN MIDOL', '~TEEN MIDOL',
    'PREMSYN PMS', 'PMS BALANCE', '~TC PMS',
    'X/S CAPLETS', 'XS LIQUID', 'X/S TYLENOL',
    'YL PHARMACIST', 'E.S TYLENOL', 'ES TYLENOL', 'EX STR TYLENOL',
    'EX TYLENOL', 'TRL-SZ EXCEDRIN',
    '~TYLENOL', '~ANACIN A/F',
]
_ASP_KEYS = [
    'ASPIRIN', 'BAYER', '~BAYER', 'BUFFERIN', '~BUFFERIN',
    'ASCRIPTIN', '~ACSRIPTIN', 'ECOTRIN', 'EMPIRIN', 'HALFRIN',
    'NORWICH', 'VALUE TIME ASP', 'DOM COATED ASP', 'DOM ENTERIC COATED',
    'DOM TRI0BUFFERED', 'RUGBY COATED ASPIRIN', 'GENUINE BAYER',
    'EXTRA STRENGTH BAYER', 'CAMA ARTHRITIS', 'VANQUISH',
    'COPE TABS', 'MOMENTUM CAP', 'MOMENTUN CAP', '~MOMENTUM',
    'BACKAID', 'B.C ', 'BC HEADACHE', 'CONGESPIRIN',
    'DOMINICKS CHILD ASPI',
    'ANACIN TABS', 'ANACIN CAP', 'ANACIN CPLT', 'ANACIN MAX',
    'ANACIN ARTH', 'ANACIN TABLETS',
    'JHO-ANACIN PEGABLE', 'T/S F/S ANACIN PM',
    'ARTH FNDTN ASP',
    '~ANACIN 125', '~ANACIN EASY', '~ANACIN MAX TAB', '~ANANCIN PAIN FORM',
    'MAX BAYER', 'EX BAYER', 'F/S BAYER', 'T/S BAYER', 'F/S NORWICH',
]


def _classify(descrip: str) -> str:
    """Classify one DESCRIP string. IBU checked first to handle Excedrin IB."""
    d = descrip.upper()
    for k in _IBU_KEYS:
        if k.upper() in d:
            return 'IBU'
    for k in _ACET_KEYS:
        if k.upper() in d:
            return 'ACET'
    for k in _ASP_KEYS:
        if k.upper() in d:
            return 'ASP'
    return 'OTHER'


def _parse_tablets(size: str) -> float:
    """
    Extract tablet count from SIZE string.
    '100 CT' → 100  |  '2/50 C' → 100  |  '2-100' → 200  |  '4 OZ' → NaN
    """
    if not isinstance(size, str):
        return np.nan
    s = size.strip().upper()
    m = re.match(r'(\d+)[/\-](\d+)', s)       # multi-pack
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.match(r'(\d+\.?\d*)\s*CT', s)       # single count
    if m:
        return float(m.group(1))
    return np.nan


def load_panel() -> pd.DataFrame:
    """
    Load raw files → balanced (store, week, good) panel.

    Steps
    -----
    1. Classify each UPC into ASP / ACET / IBU / OTHER.
    2. unit_price = PRICE × STD_TABLETS / tablets  ($/100-tablet basis).
    3. Aggregate per (store, week, category):
         price   = revenue-weighted mean unit price
         units   = Σ MOVE × QTY
         revenue = Σ MOVE × QTY × PRICE
    4. Forward-/backward-fill prices within store×good for zero-sales weeks.
    5. Pivot to wide format; drop rows with any missing price.
    6. Drop stores below min_store_wks.
    """
    print('\n[1/7] Loading data...')
    wdf = pd.read_csv(CFG['weekly_path'])
    udf = pd.read_csv(CFG['upc_path'])
    print(f'  Weekly panel : {len(wdf):,} rows | '
          f'{wdf.STORE.nunique()} stores | {wdf.UPC.nunique()} UPCs | '
          f'weeks {wdf.WEEK.min()}–{wdf.WEEK.max()}')
    print(f'  UPC catalogue: {len(udf):,} products')

    # Step 1
    udf = udf.copy()
    udf['CAT']     = udf['DESCRIP'].apply(_classify)
    udf['TABLETS'] = udf['SIZE'].apply(_parse_tablets)
    cc = udf.CAT.value_counts()
    print(f'  UPC assignment — ASP: {cc.get("ASP",0)}, '
          f'ACET: {cc.get("ACET",0)}, IBU: {cc.get("IBU",0)}, '
          f'OTHER (excluded): {cc.get("OTHER",0)}')

    # Step 2
    merged = wdf.merge(udf[['UPC', 'CAT', 'TABLETS']], on='UPC', how='left')
    merged = merged[merged['CAT'].isin(['ASP', 'ACET', 'IBU'])].copy()
    std = CFG['std_tablets']
    merged['UNIT_PX'] = np.where(
        (merged['TABLETS'] > 0) & (merged['PRICE'] > 0),
        merged['PRICE'] * std / merged['TABLETS'], np.nan)
    merged['UNITS']   = merged['MOVE'] * merged['QTY']
    merged['REVENUE'] = merged['UNITS'] * merged['PRICE']

    # Step 3
    def _rev_wtd(grp):
        pos = grp[(grp['UNITS'] > 0)].dropna(subset=['UNIT_PX'])
        val = grp.dropna(subset=['UNIT_PX'])
        if len(pos):
            return np.average(pos['UNIT_PX'], weights=pos['UNITS'])
        return val['UNIT_PX'].mean() if len(val) else np.nan

    agg = (merged.groupby(['STORE', 'WEEK', 'CAT'])
                 .apply(lambda g: pd.Series({
                     'PX':  _rev_wtd(g),
                     'QQ':  g['UNITS'].sum(),
                     'RR':  g['REVENUE'].sum()}))
                 .reset_index())

    # Step 4
    agg = agg.sort_values(['STORE', 'CAT', 'WEEK'])
    agg['PX'] = agg.groupby(['STORE', 'CAT'])['PX'].transform(
        lambda s: s.ffill().bfill())

    # Step 5
    cats = ['ASP', 'ACET', 'IBU']
    def _piv(col):
        p = agg.pivot_table(index=['STORE','WEEK'], columns='CAT',
                            values=col, aggfunc='first')
        p.columns.name = None
        return p

    px = _piv('PX'); qq = _piv('QQ'); rr = _piv('RR')
    panel = px.copy()
    for c in cats:
        panel[f'U_{c}'] = qq[c] if c in qq.columns else 0.0
        panel[f'R_{c}'] = rr[c] if c in rr.columns else 0.0
    panel = panel.dropna(subset=cats).reset_index()

    # Step 6
    sw = panel.groupby('STORE')['WEEK'].count()
    panel = panel[panel['STORE'].isin(
        sw[sw >= CFG['min_store_wks']].index)].reset_index(drop=True)

    n_pos = (panel[[f'U_{c}' for c in cats]].sum(axis=1) > 0).sum()
    print(f'  Panel: {len(panel):,} store×week obs | '
          f'{panel.STORE.nunique()} stores | {n_pos:,} with positive sales')
    return panel


def build_arrays(panel: pd.DataFrame) -> dict:
    """
    Convert wide panel to numpy arrays.
    
    Adjustments for Neural Network stability:
    - Income scaled to $100s (reduces log-variance).
    - Habit stock tracks QUANTITIES (not shares) to match CES utility theory.
    """
    cats  = ['ASP', 'ACET', 'IBU']
    panel = panel.sort_values(['STORE', 'WEEK']).reset_index(drop=True)

    prices = panel[cats].values.astype(float)
    rev    = np.stack([panel[f'R_{c}'].fillna(0).values for c in cats], 1).astype(float)
    
    # 1. Total Revenue (Income Proxy)
    tot = rev.sum(1, keepdims=True)
    
    # 2. Shares (Budget Allocation)
    shares = np.where(tot > 0, rev / tot, 1.0 / G)
    shares = np.clip(shares, 1e-6, 1.0)
    shares /= shares.sum(1, keepdims=True)
    
    # 3. Income Scaling: Convert to "Hundreds of Dollars"
    #    Real revenue ~ $1000 -> 10.0. Log(10) ~ 2.3.
    #    This aligns log_y with log_p (which is ~1.5)
    income = np.maximum(tot.squeeze(), 1.0) / 100.0  
    
    # 4. Habit Stock: Track QUANTITIES (Units), not Shares
    #    Matches simulation structure: U(q - theta*xbar)
    #    q = (Share * Income) / Price
    q_approx = (shares * income[:,None]) / np.maximum(prices, 0.01)
    
    delta   = CFG['habit_decay']
    xb       = np.zeros_like(q_approx)   # habit stock at time i  (= xb_prev for model)
    q_prev   = np.zeros_like(q_approx)   # quantities at time i-1 (= q_prev for model)
    stv = panel['STORE'].values

    # Initialize with mean quantities, not mean shares
    gm      = q_approx.mean(0)
    prev    = gm.copy()   # habit stock before first obs
    prev_q  = gm.copy()   # previous-period quantities (initialised at global mean)

    for i in range(len(shares)):
        if i > 0 and stv[i] != stv[i-1]:
            prev   = gm.copy()   # reset at store boundary
            prev_q = gm.copy()
        xb[i]     = prev    # habit stock entering period i
        q_prev[i] = prev_q  # quantities purchased in period i-1
        # Save current quantities for next iteration, then update habit
        prev_q = q_approx[i]
        prev   = delta * prev + (1 - delta) * q_approx[i]

    return dict(prices=prices, shares=shares, income=income,
                xbar=xb, q_prev=q_prev, week=panel['WEEK'].values, store=stv)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2-6  MODEL OBJECTS (moved to src/dominicks/models.py)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7  PREDICTION & EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _pred(spec, p, y, xb_prev=None, q_prev=None, **kw):
    """Unified prediction helper.

    For the MDP model pass *both* xb_prev (log-normalised previous habit
    stock) and q_prev (log-normalised previous quantities) as numpy arrays.
    """
    dev = CFG['device']
    if spec == 'aids':  return kw['aids'].predict(p, y)
    if spec == 'blp':   return kw['blp'].predict(p)
    if spec == 'lirl':  return pred_lirl(kw['ff'], kw['theta'], p, y)
    if spec == 'nirl':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p,1e-8), dtype=torch.float32)).to(dev)
            ly = torch.log(torch.tensor(np.maximum(y,1e-8), dtype=torch.float32)).unsqueeze(1).to(dev)
            return kw['nirl'](lp, ly).cpu().numpy()
    if spec == 'mdp':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p,1e-8), dtype=torch.float32)).to(dev)
            ly  = torch.log(torch.tensor(np.maximum(y,1e-8), dtype=torch.float32)).unsqueeze(1).to(dev)
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            return kw['mdp'](lp, ly, xbp, qp).cpu().numpy()
    if spec == 'mix':   return kw['mix'].predict(p, y)
    raise ValueError(spec)


def own_elasticity(spec, p0, y0, xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    w0  = _pred(spec, p0[None], np.array([y0]),
                xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
    eps = []
    for i in range(G):
        p1 = p0.copy()[None]; p1[0,i] *= 1+h
        w1 = _pred(spec, p1, np.array([y0]),
                   xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                   q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        eps.append(((w1[i]-w0[i])/w0[i])/h - 1)
    return np.array(eps)


def comp_var(spec, p0, p1, y, xb_prev0=None, q_prev0=None, **kw):
    path = np.linspace(p0, p1, CFG['cv_steps'])
    dp   = (p1-p0) / CFG['cv_steps']
    cv   = 0.0
    for t in range(CFG['cv_steps']):
        w   = _pred(spec, path[t:t+1], np.array([y]),
                    xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                    q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        cv -= (w * y / path[t]) @ dp
    return cv


def _xbt_kw(xbt):
    """Unpack an xbt entry (None or (xb_prev, q_prev) tuple) into kwargs."""
    if xbt is None:
        return {}
    xbp, qp = xbt
    return {'xb_prev': xbp, 'q_prev': qp}


def metrics(spec, p, y, w_true, xb_prev=None, q_prev=None, **kw):
    wp = _pred(spec, p, y, xb_prev=xb_prev, q_prev=q_prev, **kw)
    return {'RMSE': np.sqrt(mean_squared_error(w_true, wp)),
            'MAE':  mean_absolute_error(w_true, wp)}

def kl_div(spec, p, y, w_true, xb_prev=None, q_prev=None, **kw):
    wp = np.clip(_pred(spec, p, y, xb_prev=xb_prev, q_prev=q_prev, **kw), 1e-8, 1.0)
    wt = np.clip(w_true, 1e-8, 1.0)
    return float(np.mean(np.sum(wt * np.log(wt/wp), 1)))


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8  DATA PREPARATION (run once before the n_runs loop)
# ─────────────────────────────────────────────────────────────────────────────

panel  = load_panel()
data   = build_arrays(panel)
prices = data['prices']; shares = data['shares']
income = data['income']; xbar   = data['xbar']
q_prev_raw = data['q_prev']          # quantities at t-1 (raw, same units as xbar)
weeks  = data['week'];   stores = data['store']

# Descriptive stats
print('\n[2/7] Descriptive statistics:')
desc = pd.DataFrame({
    'Good':           GOODS,
    'Mean price':     prices.mean(0).round(3),
    'Std price':      prices.std(0).round(3),
    'Mean share':     shares.mean(0).round(4),
    'Std share':      shares.std(0).round(4),
})
print(desc.to_string(index=False))

# Train / test split
print('\n[3/7] Train / test split...')
tr = np.where(weeks < CFG['test_cutoff'])[0]
te = np.where(weeks >= CFG['test_cutoff'])[0]
if len(te) < 30 or len(tr) < 50:
    print(f'  Week split insufficient (tr={len(tr)}, te={len(te)}). '
          'Using random 75/25.')
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(prices))
    tr  = idx[:int(0.75*len(prices))]; te = idx[int(0.75*len(prices)):]

p_tr, p_te = prices[tr], prices[te]
w_tr, w_te = shares[tr], shares[te]
y_tr, y_te = income[tr], income[te]
xb_tr, xb_te         = xbar[tr],      xbar[te]        # habit stock at t (= xb_prev)
qp_tr, qp_te         = q_prev_raw[tr], q_prev_raw[te]  # quantities at t-1 (= q_prev)
s_tr, s_te   = stores[tr], stores[te]
wk_tr, wk_te = weeks[tr],  weeks[te]
print(f'  Train: {len(tr):,}  |  Test: {len(te):,}')

# ── Store-level demeaning (same means applied to xb and q_prev) ───────────────
# Network sees deviations from store average, not levels.
store_xb_means = {}
for s in np.unique(s_tr):
    mask = s_tr == s
    store_xb_means[s] = xb_tr[mask].mean(0)

global_mean = np.mean(list(store_xb_means.values()), axis=0)

def _demean(arr, idxs, store_ids):
    out = arr.copy()
    for i, s in enumerate(store_ids):
        out[i] -= store_xb_means.get(s, global_mean)
    return out

def _demean_train(arr, store_ids):
    out = arr.copy()
    for s in np.unique(s_tr):
        mask = store_ids == s
        out[mask] -= store_xb_means[s]
    return out

xb_tr = _demean_train(xb_tr, s_tr)
xb_te = _demean(xb_te, range(len(s_te)), s_te)
qp_tr = _demean_train(qp_tr, s_tr)
qp_te = _demean(qp_te, range(len(s_te)), s_te)

# ── Normalise both log_xb_prev and log_q_prev to match log_p scale ───────────
# Use xb_prev training statistics for *both* so the weighted average δ·xb + (1-δ)·q
# is in a consistent log-space.
log_xb_tr_raw = np.log(np.maximum(xb_tr, 1e-6))
log_xb_te_raw = np.log(np.maximum(xb_te, 1e-6))
log_qp_tr_raw = np.log(np.maximum(qp_tr, 1e-6))
log_qp_te_raw = np.log(np.maximum(qp_te, 1e-6))

_xb_mean = log_xb_tr_raw.mean(0, keepdims=True)
_xb_std  = log_xb_tr_raw.std(0,  keepdims=True)
_lp_std  = np.log(np.maximum(p_tr, 1e-8)).std()

def _norm(x): return (x - _xb_mean) / (_xb_std + 1e-8) * _lp_std

xb_tr = _norm(log_xb_tr_raw)
xb_te = _norm(log_xb_te_raw)
qp_tr = _norm(log_qp_tr_raw)
qp_te = _norm(log_qp_te_raw)

print(f'  log_xb_prev normalised: mean={xb_tr.mean():.3f} std={xb_tr.std():.3f} '
      f'(target log_p std={_lp_std:.3f})')
print(f'  log_q_prev  normalised: mean={qp_tr.mean():.3f} std={qp_tr.std():.3f}')

# Instruments (deterministic — computed once)
print('\n[4/7] Building Hausman instruments...')
Z_tr = hausman_iv(p_tr, s_tr, wk_tr)

# Grid for demand-curve figures
sg   = CFG['shock_good']
ss   = CFG['shock_pct']
plo  = max(p_te[:,sg].min()*0.8, 0.5)
phi  = p_te[:,sg].max()*1.2
pgr  = np.linspace(plo, phi, 80)
tpx  = np.tile(p_te.mean(0), (80,1)); tpx[:,sg] = pgr
fy   = np.full(80, float(y_te.mean()))
xbr  = np.tile(xb_te.mean(0), (80,1))   # grid xb_prev (log-normalised)
qpr  = np.tile(qp_te.mean(0), (80,1))   # grid q_prev  (log-normalised)

p_mn    = p_te.mean(0); y_mn = float(y_te.mean())
xb_mn   = xb_te.mean(0)    # mean log-normalised xb_prev for test
qp_mn   = qp_te.mean(0)    # mean log-normalised q_prev  for test
p0w     = p_mn.copy(); p1w = p_mn.copy(); p1w[sg] *= 1+ss

TEAL  = '#009688'


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9  run_once() — one full training + evaluation pass
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int) -> dict:
    """
    Re-estimate all stochastic models with the given random seed and
    return a dict of every scalar/array result needed for tables and figures.

    Deterministic models (LA-AIDS, BLP-IV) are identical across seeds
    but are re-run for completeness.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Train models ──────────────────────────────────────────────────────
    aids_m = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m  = BLPLogitIV().fit(p_tr, w_tr, Z_tr)

    th_sh = run_lirl(feat_shared,        p_tr, y_tr, w_tr, CFG)
    th_gs = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, CFG)
    th_or = run_lirl(feat_orth,          p_tr, y_tr, w_tr, CFG)

    nirl_m, hist_n = _train(NeuralIRL(CFG['nirl_hidden']),
                             p_tr, y_tr, w_tr, 'nirl', CFG,
                             tag=f'Neural IRL s={seed}')

    mdp_m, hist_m = _train(MDPNeuralIRL(CFG['mdp_hidden']),
                            p_tr, y_tr, w_tr, 'mdp', CFG,
                            xb_prev_tr=xb_tr,
                            q_prev_tr=qp_tr,
                            tag=f'MDP-IRL s={seed}')

    ns   = min(CFG['mix_subsamp'], len(tr))
    mi   = np.random.choice(len(tr), ns, replace=False)
    vmix = VarMixture(CFG, CFG['mix_K'], seed=seed)
    vmix.fit(p_tr[mi], y_tr[mi], w_tr[mi])
    cdf  = vmix.summary()

    KW = dict(aids=aids_m, blp=blp_m, nirl=nirl_m, mdp=mdp_m, mix=vmix,
              ff=feat_shared, theta=th_sh)

    # xbt entries: None → non-MDP model; (xb_prev, q_prev) tuple → MDP model
    _mdp_te = (xb_te, qp_te)
    SPECS = [
        ('LA-AIDS',          'aids', {},                                            None),
        ('BLP (IV)',         'blp',  {},                                            None),
        ('Lin IRL Shared',   'lirl', {'ff': feat_shared,        'theta': th_sh},   None),
        ('Lin IRL GoodSpec', 'lirl', {'ff': feat_good_specific, 'theta': th_gs},   None),
        ('Lin IRL Orth',     'lirl', {'ff': feat_orth,          'theta': th_or},   None),
        ('Neural IRL',       'nirl', {},                                            None),
        ('MDP Neural IRL',   'mdp',  {},                                            _mdp_te),
        ('Var. Mixture',     'mix',  {},                                            None),
    ]

    # ── Table 1: accuracy ─────────────────────────────────────────────────
    perf = {}
    for nm, sp, ek, xbt in SPECS:
        perf[nm] = metrics(sp, p_te, y_te, w_te, **_xbt_kw(xbt), **{**KW, **ek})

    # ── Table 2: elasticities ─────────────────────────────────────────────
    elast = {}
    for nm, sp, ek, xbt in SPECS:
        try:
            mdp_kw = ({'xb_prev0': xb_mn, 'q_prev0': qp_mn}
                      if xbt is not None else {})
            elast[nm] = own_elasticity(sp, p_mn, y_mn,
                                        **mdp_kw, **{**KW, **ek})
        except Exception as e:
            elast[nm] = np.full(G, np.nan)

    # ── Table 3: welfare ──────────────────────────────────────────────────
    welf = {}
    for nm, sp, ek, xbt in SPECS:
        try:
            mdp_kw = ({'xb_prev0': xb_mn, 'q_prev0': qp_mn}
                      if xbt is not None else {})
            welf[nm] = comp_var(sp, p0w, p1w, y_mn,
                                **mdp_kw, **{**KW, **ek})
        except:
            welf[nm] = np.nan

    # ── Table 4: MDP advantage ────────────────────────────────────────────
    r_a  = perf['LA-AIDS']['RMSE']
    r_n  = perf['Neural IRL']['RMSE']
    r_m  = perf['MDP Neural IRL']['RMSE']
    kl_a = kl_div('aids', p_te, y_te, w_te, **KW)
    kl_n = kl_div('nirl', p_te, y_te, w_te, **KW)
    kl_m = kl_div('mdp',  p_te, y_te, w_te,
                  xb_prev=xb_te, q_prev=qp_te, **KW)

    # ── Demand curves (for figures) ───────────────────────────────────────
    curves = {}
    curve_specs = [
        ('aids', {},                                    None,              'LA-AIDS'),
        ('blp',  {},                                    None,              'BLP (IV)'),
        ('lirl', {'ff':feat_shared,  'theta':th_sh},   None,              'Lin IRL (Shared)'),
        ('lirl', {'ff':feat_orth,    'theta':th_or},   None,              'Lin IRL (Orth)'),
        ('nirl', {},                                    None,              'Neural IRL'),
        ('mdp',  {},                                    (xbr, qpr),        'MDP Neural IRL'),
        ('mix',  {},                                    None,              'Var. Mixture'),
    ]
    for sp, ek, xbt, lbl in curve_specs:
        try:
            curves[lbl] = _pred(sp, tpx, fy, **_xbt_kw(xbt), **{**KW, **ek})
        except Exception as e:
            curves[lbl] = np.full((len(pgr), G), np.nan)

    # ── β and δ values ────────────────────────────────────────────────────
    beta_nirl  = nirl_m.beta.item()
    beta_mdp   = mdp_m.beta.item()
    delta_mdp  = mdp_m.delta.item()

    # ── Mixture summary ───────────────────────────────────────────────────
    dk = cdf.loc[cdf.pi.idxmax()]

    return dict(
        perf=perf, elast=elast, welf=welf,
        r_a=r_a, r_n=r_n, r_m=r_m,
        kl_a=kl_a, kl_n=kl_n, kl_m=kl_m,
        curves=curves,
        beta_nirl=beta_nirl, beta_mdp=beta_mdp,
        delta_mdp=delta_mdp,
        cdf=cdf, dk=dk,
        hist_n=hist_n, hist_m=hist_m,
        nirl_m=nirl_m, mdp_m=mdp_m,
        vmix=vmix,
        # keep raw model objects from last run for scatter figure
        aids_m=aids_m, blp_m=blp_m,
        th_sh=th_sh, th_gs=th_gs, th_or=th_or,
        KW=KW, SPECS=SPECS,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10  n_runs LOOP — collect results across seeds
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS = CFG['n_runs']
print(f'\n[5/7] Training models ({N_RUNS} independent run(s))...')

all_runs = []
for run_idx in range(N_RUNS):
    seed = 42 + run_idx * 15          # deterministic, spread seeds
    print(f'\n  ── Run {run_idx+1}/{N_RUNS}  (seed={seed}) ──')
    all_runs.append(run_once(seed))

# Use the last run's model objects for the scatter / convergence figures
last = all_runs[-1]

# ── Aggregate helpers ─────────────────────────────────────────────────────────

MODEL_NAMES = [
    'LA-AIDS', 'BLP (IV)',
    'Lin IRL Shared', 'Lin IRL GoodSpec', 'Lin IRL Orth',
    'Neural IRL', 'MDP Neural IRL', 'Var. Mixture',
]

def _agg_scalar(key, model_names=MODEL_NAMES):
    """
    Aggregate a per-model scalar across runs.
    Returns (mean_dict, std_dict) keyed by model name.
    """
    raw = {nm: [] for nm in model_names}
    for r in all_runs:
        for nm in model_names:
            raw[nm].append(r[key][nm] if isinstance(r[key], dict) else np.nan)
    means = {nm: np.nanmean(raw[nm]) for nm in model_names}
    stds  = {nm: np.nanstd(raw[nm], ddof=min(1, N_RUNS-1)) for nm in model_names}
    return means, stds

def _agg_metric(metric):
    """Aggregate RMSE or MAE across runs → (mean_dict, std_dict)."""
    raw = {nm: [] for nm in MODEL_NAMES}
    for r in all_runs:
        for nm in MODEL_NAMES:
            raw[nm].append(r['perf'][nm][metric])
    means = {nm: np.nanmean(raw[nm]) for nm in MODEL_NAMES}
    stds  = {nm: np.nanstd(raw[nm], ddof=min(1, N_RUNS-1)) for nm in MODEL_NAMES}
    return means, stds

def _agg_elast():
    """Aggregate elasticities → mean and std arrays per model."""
    raw = {nm: [] for nm in MODEL_NAMES}
    for r in all_runs:
        for nm in MODEL_NAMES:
            raw[nm].append(r['elast'][nm])
    means = {nm: np.nanmean(raw[nm], axis=0) for nm in MODEL_NAMES}
    stds  = {nm: np.nanstd(raw[nm],  axis=0, ddof=min(1, N_RUNS-1))
             for nm in MODEL_NAMES}
    return means, stds

def _agg_welf():
    """Aggregate welfare CV across runs."""
    raw = {nm: [] for nm in MODEL_NAMES}
    for r in all_runs:
        for nm in MODEL_NAMES:
            raw[nm].append(r['welf'][nm])
    means = {nm: np.nanmean(raw[nm]) for nm in MODEL_NAMES}
    stds  = {nm: np.nanstd(raw[nm], ddof=min(1, N_RUNS-1)) for nm in MODEL_NAMES}
    return means, stds

def _agg_curves():
    """
    Aggregate demand curves across runs.
    Returns (mean_curves, std_curves) each a dict label→array(80,G).
    """
    labels = list(all_runs[0]['curves'].keys())
    means, stds = {}, {}
    for lbl in labels:
        stack = np.stack([r['curves'][lbl] for r in all_runs], 0)  # (n_runs,80,G)
        means[lbl] = np.nanmean(stack, 0)
        stds[lbl]  = np.nanstd(stack,  0, ddof=min(1, N_RUNS-1))
    return means, stds

# ── Compute aggregated results ────────────────────────────────────────────────

rmse_mean, rmse_std = _agg_metric('RMSE')
mae_mean,  mae_std  = _agg_metric('MAE')
elast_mean, elast_std = _agg_elast()
welf_mean, welf_std   = _agg_welf()
curve_mean, curve_std = _agg_curves()

# MDP advantage scalars
r_a_arr  = np.array([r['r_a']  for r in all_runs])
r_n_arr  = np.array([r['r_n']  for r in all_runs])
r_m_arr  = np.array([r['r_m']  for r in all_runs])
kl_a_arr = np.array([r['kl_a'] for r in all_runs])
kl_n_arr = np.array([r['kl_n'] for r in all_runs])
kl_m_arr = np.array([r['kl_m'] for r in all_runs])

r_a_mu  = r_a_arr.mean();  r_a_se  = r_a_arr.std(ddof=min(1,N_RUNS-1))
r_n_mu  = r_n_arr.mean();  r_n_se  = r_n_arr.std(ddof=min(1,N_RUNS-1))
r_m_mu  = r_m_arr.mean();  r_m_se  = r_m_arr.std(ddof=min(1,N_RUNS-1))
kl_a_mu = kl_a_arr.mean(); kl_a_se = kl_a_arr.std(ddof=min(1,N_RUNS-1))
kl_n_mu = kl_n_arr.mean(); kl_n_se = kl_n_arr.std(ddof=min(1,N_RUNS-1))
kl_m_mu = kl_m_arr.mean(); kl_m_se = kl_m_arr.std(ddof=min(1,N_RUNS-1))

# Beta and delta stats
beta_n_arr  = np.array([r['beta_nirl']  for r in all_runs])
beta_m_arr  = np.array([r['beta_mdp']   for r in all_runs])
delta_m_arr = np.array([r['delta_mdp']  for r in all_runs])
beta_n_mu   = beta_n_arr.mean();  beta_n_se  = beta_n_arr.std(ddof=min(1,N_RUNS-1))
beta_m_mu   = beta_m_arr.mean();  beta_m_se  = beta_m_arr.std(ddof=min(1,N_RUNS-1))
delta_m_mu  = delta_m_arr.mean(); delta_m_se = delta_m_arr.std(ddof=min(1,N_RUNS-1))

# Use last run for convenience access (backward-compat with original code)
perf  = last['perf']
elast = last['elast']
welf  = last['welf']
cdf   = last['cdf']
dk    = last['dk']
hist_n = last['hist_n']
hist_m = last['hist_m']
nirl_m = last['nirl_m']
mdp_m  = last['mdp_m']
vmix   = last['vmix']

r_a = r_a_mu; r_n = r_n_mu; r_m = r_m_mu
kl_a = kl_a_mu; kl_n = kl_n_mu; kl_m = kl_m_mu
KW   = last['KW']
SPECS = last['SPECS']


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 11  EVALUATION — print tables using aggregated results
# ─────────────────────────────────────────────────────────────────────────────

print('\n[6/7] Evaluating...')

# Table 1
print('\n' + '='*72)
print(f'  TABLE 1: OUT-OF-SAMPLE ACCURACY  (mean ± std, n_runs={N_RUNS})')
print('='*72)
t1_rows = []
for nm in MODEL_NAMES:
    t1_rows.append({'Model': nm,
                    'RMSE_mean': rmse_mean[nm], 'RMSE_std': rmse_std[nm],
                    'MAE_mean':  mae_mean[nm],  'MAE_std':  mae_std[nm]})
t1_df = pd.DataFrame(t1_rows).set_index('Model')
print(t1_df.round(5).to_string())

# Table 2
elast_df = pd.DataFrame({n: elast_mean[n] for n in MODEL_NAMES},
                         index=[f'{g} ε' for g in GOODS]).T
print(f'\n  TABLE 2: OWN-PRICE ELASTICITIES  (mean over {N_RUNS} run(s))')
print(elast_df.round(3).to_string())

# Table 3
nw_mu = welf_mean.get('Neural IRL', np.nan)
print(f'\n  TABLE 3: WELFARE  ({int(ss*100)}% ibuprofen shock, mean ± std)')
for k in MODEL_NAMES:
    v = welf_mean[k]; se = welf_std[k]
    tag = (f'  ({100*(v-nw_mu)/abs(nw_mu):+.1f}% vs Neural IRL)'
           if k != 'Neural IRL' and not np.isnan(nw_mu) else '')
    print(f'  {k:<22}: ${v:+.5f} ± {se:.5f}{tag}')

# Table 4
mdp_rows = [
    ('LA-AIDS',                    r_a_mu, r_a_se,  kl_a_mu, kl_a_se, 'baseline'),
    ('Neural IRL (static)',         r_n_mu, r_n_se,  kl_n_mu, kl_n_se,
     f'{100*(r_a_mu-r_n_mu)/r_a_mu:.1f}%'),
    ('MDP Neural IRL (x̄ state)', r_m_mu, r_m_se,  kl_m_mu, kl_m_se,
     f'{100*(r_a_mu-r_m_mu)/r_a_mu:.1f}%'),
]
print(f'\n  TABLE 4: MDP ADVANTAGE  (n_runs={N_RUNS})')
print(f'  {"Model":<30} {"RMSE mean":>10}  {"±std":>7}  {"KL mean":>8}  {"±std":>7}  Reduction')
for mn, rm, rs, km, ks, rd in mdp_rows:
    print(f'  {mn:<30} {rm:>10.5f}  {rs:>7.5f}  {km:>8.5f}  {ks:>7.5f}  {rd}')


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 12  FIGURES  (with ±1-std bands / error bars)
# ─────────────────────────────────────────────────────────────────────────────

print('\n[7/7] Generating figures...')

# ── Fig 1: Demand curves — all models, with ±1-std shaded bands ───────────────
fig1, ax1 = plt.subplots(figsize=(11, 6))
curve_defs = [
    ('r--',  2.0,  None,  'LA-AIDS'),
    ('g-.',  2.0,  None,  'BLP (IV)'),
    # ('y:',   1.8,  None,  'Lin IRL (Shared)'),
    ('c:',   1.8,  None,  'Lin IRL (Orth)'),
    ('b-',   2.5,  None,  'Neural IRL'),
    ('-',    2.0,  TEAL,  'MDP Neural IRL'),
    # ('m--',  2.0,  None,  'Var. Mixture'),
]
for sty, lw, col, lbl in curve_defs:
    mu  = curve_mean.get(lbl)
    std = curve_std.get(lbl)
    if mu is None:
        continue
    kw_plot = dict(lw=lw, label=lbl, alpha=0.9)
    if col: kw_plot['color'] = col
    line, = ax1.plot(pgr, mu[:,0], sty, **kw_plot)
    if N_RUNS > 1:
        fc = line.get_color()
        ax1.fill_between(pgr,
                         (mu[:,0] - std[:,0]).clip(0),
                          mu[:,0] + std[:,0],
                         color=fc, alpha=0.12)
# ax1.axvline(p_mn[sg], color='orange', ls=':', lw=1.5, alpha=0.9,
            # label='Mean ibuprofen price')
se_note = f'  (shaded bands = ±1 SD, n={N_RUNS})' if N_RUNS > 1 else ''
# ax1.set_title(f"Aspirin Budget Share vs Ibuprofen Unit Price\n"
#               f"Dominick's Analgesics — All Models{se_note}",
#               fontsize=12, fontweight='bold')
ax1.set_xlabel('Ibuprofen Unit Price ($/100 tablets)', fontsize=14)
ax1.set_ylabel('Aspirin Budget Share $w_0$', fontsize=14)
ax1.legend(fontsize=9, ncol=2, framealpha=0.93)
ax1.grid(True, alpha=0.3); fig1.tight_layout()
for ext in ('pdf','png'):
    fig1.savefig(f"{CFG['fig_dir']}/fig_demand_curves.{ext}",
                 dpi=150, bbox_inches='tight')
plt.close(fig1)
print('  Saved: fig_demand_curves')

# ── Fig 2: MDP advantage — three goods, with ±1-std bands ────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
mdp_curve_defs = [
    ('r--', 2.0, None, 'LA-AIDS',             'LA-AIDS'),
    ('b-.', 2.0, None, 'Neural IRL (static)',  'Neural IRL'),
    ('-',   2.5, TEAL, 'MDP-IRL ($\\bar{x}$)', 'MDP Neural IRL'),
]
for gi, (gn, ax) in enumerate(zip(GOODS, axes2)):
    for sty, lw, col, lbl_disp, lbl_key in mdp_curve_defs:
        mu  = curve_mean.get(lbl_key)
        std = curve_std.get(lbl_key)
        if mu is None: continue
        kw_p = dict(lw=lw, label=lbl_disp, alpha=0.9)
        if col: kw_p['color'] = col
        line, = ax.plot(pgr, mu[:,gi], sty, **kw_p)
        if N_RUNS > 1:
            fc = line.get_color()
            ax.fill_between(pgr,
                            (mu[:,gi] - std[:,gi]).clip(0),
                             mu[:,gi] + std[:,gi],
                            color=fc, alpha=0.14)
    ax.axvline(p_mn[sg], color='orange', ls=':', lw=1.3, alpha=0.8)
    # ax.set_title(f'{gn} Budget Share', fontsize=11, fontweight='bold')
    ax.set_xlabel('Ibuprofen Price ($/100 tab)', fontsize=14)
    ax.set_ylabel(f'$w_{gi}$', fontsize=14)
    ax.set_ylim(0.15, 0.55)
    ax.legend(fontsize=14); ax.grid(True, alpha=0.3)
se_note2 = f'  (bands = ±1 SD, n={N_RUNS})' if N_RUNS > 1 else ''
# fig2.suptitle(f"MDP Neural IRL vs Static Models — Dominick's Analgesics\n"
#               f"All Three Budget Shares vs Ibuprofen Price{se_note2}",
#               fontsize=12, fontweight='bold')
fig2.tight_layout()
for ext in ('pdf','png'):
    fig2.savefig(f"{CFG['fig_dir']}/fig_mdp_advantage.{ext}",
                 dpi=150, bbox_inches='tight')
plt.close(fig2)
print('  Saved: fig_mdp_advantage')

# ── Fig 3: Training convergence (last run) ────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
for ax, hist, ck, cb, title in [
    (axes3[0], hist_n, 'b',  'r', 'Neural IRL (Static)'),
    (axes3[1], hist_m, TEAL, 'm', 'MDP Neural IRL'),
]:
    if hist:
        ex = [h['epoch'] for h in hist]
        ky = [h['kl']    for h in hist]
        by = [h['beta']  for h in hist]
        ax2 = ax.twinx()
        ax.plot(ex, ky, 'o-', ms=5, lw=1.8, color=ck, label='KL Loss')
        ax2.plot(ex, by, 's--', ms=5, lw=1.8, color=cb, label='β (learned)')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('KL Divergence', color=ck, fontsize=10)
        ax2.set_ylabel('Temperature β', color=cb, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        l1,lb1 = ax.get_legend_handles_labels()
        l2,lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=9); ax.grid(True, alpha=0.3)
fig3.suptitle(f"Training Convergence — Dominick's Analgesics  (last run, seed={42+(N_RUNS-1)*17})",
              fontsize=12, fontweight='bold')
fig3.tight_layout()
for ext in ('pdf','png'):
    fig3.savefig(f"{CFG['fig_dir']}/fig_convergence.{ext}",
                 dpi=150, bbox_inches='tight')
plt.close(fig3)
print('  Saved: fig_convergence')

# ── Fig 3b: β convergence across runs — violin / box plot ────────────────────
if N_RUNS > 1:
    fig3b, (ax_bn, ax_bm) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, arr, title, col in [
        (ax_bn, beta_n_arr, 'Neural IRL — β̂ across runs',     'steelblue'),
        (ax_bm, beta_m_arr, 'MDP Neural IRL — β̂ across runs', TEAL),
    ]:
        ax.violinplot(arr, showmeans=True, showmedians=True)
        ax.set_xticks([1]); ax.set_xticklabels(['β̂'])
        ax.set_title(f'{title}\nmean={arr.mean():.3f} ± {arr.std(ddof=1):.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Learned temperature β', fontsize=10)
        ax.grid(True, axis='y', alpha=0.4)
    fig3b.suptitle(f'Learned Temperature β̂ Across {N_RUNS} Runs',
                   fontsize=12, fontweight='bold')
    fig3b.tight_layout()
    for ext in ('pdf','png'):
        fig3b.savefig(f"{CFG['fig_dir']}/fig_beta_runs.{ext}",
                      dpi=150, bbox_inches='tight')
    plt.close(fig3b)
    print('  Saved: fig_beta_runs')

# ── Fig 4: Observed vs predicted scatter (last run) ───────────────────────────
fig4, axes4 = plt.subplots(3, 3, figsize=(14, 12))
scat_defs = [
    ('LA-AIDS',       'aids', {},  None,               '#E53935'),
    ('Neural IRL',    'nirl', {},  None,               '#1E88E5'),
    ('MDP Neural IRL','mdp',  {},  (xb_te, qp_te),    TEAL),
]
for row, (mn, sp, ek, xbt, col) in enumerate(scat_defs):
    wp = _pred(sp, p_te, y_te, **(_xbt_kw(xbt) if xbt else {}), **{**KW, **ek})
    for gi, gn in enumerate(GOODS):
        ax = axes4[row, gi]
        ax.scatter(w_te[:,gi], wp[:,gi], alpha=0.35, s=8, color=col)
        lim = [0, max(w_te[:,gi].max(), wp[:,gi].max())*1.05]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ri = np.sqrt(mean_squared_error(w_te[:,gi], wp[:,gi]))
        ax.set_title(f'{mn} — {gn}\nRMSE = {ri:.4f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Observed', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3)
fig4.suptitle(f"Observed vs Predicted Budget Shares — Dominick's Analgesics  (last run)",
              fontsize=13, fontweight='bold')
fig4.tight_layout()
for ext in ('pdf','png'):
    fig4.savefig(f"{CFG['fig_dir']}/fig_scatter.{ext}",
                 dpi=150, bbox_inches='tight')
plt.close(fig4)
print('  Saved: fig_scatter')

# ── Fig 5: Variational mixture components (last run) ─────────────────────────
fig5, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
cols = plt.cm.tab10(np.linspace(0, 0.6, CFG['mix_K']))
xpos = np.arange(CFG['mix_K'])
bars = axL.bar(xpos, cdf['pi'], color=cols, edgecolor='k', alpha=0.85)
for bar, (_, row) in zip(bars, cdf.iterrows()):
    if row['pi'] > 0.01:
        axL.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f"ρ={row['rho']:.2f}", ha='center', fontsize=8)
axL.axhline(1/CFG['mix_K'], color='grey', ls='--', alpha=0.6, label='Uniform prior')
axL.set_xticks(xpos)
axL.set_xticklabels(
    [f"K={int(r['K'])}\n[{r['alpha_asp']:.2f},{r['alpha_acet']:.2f},"
     f"{r['alpha_ibu']:.2f}]" for _, r in cdf.iterrows()], fontsize=7)
axL.set_ylabel(r'Mixture Weight $\hat{\pi}_k$')
axL.set_ylim(0, 1.05)
axL.set_title(r'Component Weights $\hat{\pi}_k$',
              fontsize=11, fontweight='bold')
axL.legend(fontsize=9); axL.grid(True, axis='y', alpha=0.3)

for ki, (_, row) in enumerate(cdf.iterrows()):
    axR.scatter(row['alpha_asp'], row['alpha_acet'],
                s=row['pi']*2500+30, c=[cols[ki]], alpha=0.8,
                label=f"K={int(row['K'])} (ρ={row['rho']:.2f})",
                edgecolors='k', linewidths=0.5)
axR.set_xlabel(r'$\hat{\alpha}_{\mathrm{Aspirin}}$', fontsize=11)
axR.set_ylabel(r'$\hat{\alpha}_{\mathrm{Acetaminophen}}$', fontsize=11)
axR.set_xlim(-0.05, 1.0); axR.set_ylim(-0.05, 1.0)
axR.set_title(r'Component Centres $(\hat{\alpha}_{\mathrm{Asp}},'
              r'\hat{\alpha}_{\mathrm{Acet}})$'
              '\n(size $\\propto \\hat{\\pi}_k$)',
              fontsize=11, fontweight='bold')
axR.legend(fontsize=8, loc='upper right'); axR.grid(True, alpha=0.3)
fig5.suptitle(r'Continuous Variational Mixture IRL ($K=6$) — '
              f"Dominick's Analgesics  (last run)",
              fontsize=12, fontweight='bold')
fig5.tight_layout()
for ext in ('pdf','png'):
    fig5.savefig(f"{CFG['fig_dir']}/fig_mixture.{ext}",
                 dpi=150, bbox_inches='tight')
plt.close(fig5)
print('  Saved: fig_mixture')

# ── Fig 6 (new): RMSE bar chart with error bars across all models ─────────────
if N_RUNS > 1:
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    xp   = np.arange(len(MODEL_NAMES))
    rmu  = np.array([rmse_mean[nm] for nm in MODEL_NAMES])
    rse  = np.array([rmse_std[nm]  for nm in MODEL_NAMES])
    clrs = plt.cm.tab10(np.linspace(0, 0.8, len(MODEL_NAMES)))
    bars6 = ax6.bar(xp, rmu, yerr=rse, capsize=5, color=clrs,
                    edgecolor='k', alpha=0.85, error_kw=dict(lw=1.5, ecolor='#333'))
    ax6.set_xticks(xp)
    ax6.set_xticklabels(MODEL_NAMES, rotation=25, ha='right', fontsize=9)
    ax6.set_ylabel('Out-of-Sample RMSE', fontsize=11)
    ax6.set_title(f'Out-of-Sample RMSE — Mean ± 1 SD  ({N_RUNS} runs)\n'
                  f"Dominick's Analgesics",
                  fontsize=12, fontweight='bold')
    ax6.grid(True, axis='y', alpha=0.35)
    fig6.tight_layout()
    for ext in ('pdf','png'):
        fig6.savefig(f"{CFG['fig_dir']}/fig_rmse_bars.{ext}",
                     dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print('  Saved: fig_rmse_bars')


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 13  CSV TABLES  (mean ± std columns added)
# ─────────────────────────────────────────────────────────────────────────────

# Table 0: descriptive (unchanged — deterministic)
desc.to_csv(f"{CFG['out_dir']}/table0_desc.csv", index=False)

# Table 1: accuracy with SE
t1_out = pd.DataFrame({
    'Model':     MODEL_NAMES,
    'RMSE_mean': [rmse_mean[nm] for nm in MODEL_NAMES],
    'RMSE_std':  [rmse_std[nm]  for nm in MODEL_NAMES],
    'MAE_mean':  [mae_mean[nm]  for nm in MODEL_NAMES],
    'MAE_std':   [mae_std[nm]   for nm in MODEL_NAMES],
    'n_runs':    N_RUNS,
}).round(6)
t1_out.to_csv(f"{CFG['out_dir']}/table1_accuracy.csv", index=False)

# Table 2: elasticities with SE
t2_rows = []
for nm in MODEL_NAMES:
    row = {'Model': nm}
    for gi, gn in enumerate(GOODS):
        row[f'{gn}_mean'] = round(float(elast_mean[nm][gi]), 4)
        row[f'{gn}_std']  = round(float(elast_std[nm][gi]),  4)
    row['n_runs'] = N_RUNS
    t2_rows.append(row)
pd.DataFrame(t2_rows).to_csv(f"{CFG['out_dir']}/table2_elasticities.csv", index=False)

# Table 3: welfare with SE
# Scale CV back to Dollars ($) for reporting
t3_rows = pd.DataFrame({
    'Model':     MODEL_NAMES,
    'CV_Loss_mean': [welf_mean[nm] * 100.0 for nm in MODEL_NAMES], # <--- Multiply by 100
    'CV_Loss_std':  [welf_std[nm]  * 100.0 for nm in MODEL_NAMES], # <--- Multiply by 100
    'n_runs': N_RUNS,
}).round(6)
t3_rows.to_csv(f"{CFG['out_dir']}/table3_welfare.csv", index=False)

# Table 4: MDP advantage with SE
t4_rows = pd.DataFrame([
    {'Model': 'LA-AIDS',                   'RMSE_mean': r_a_mu, 'RMSE_std': r_a_se,
     'KL_mean': kl_a_mu, 'KL_std': kl_a_se, 'Reduction': 'baseline', 'n_runs': N_RUNS},
    {'Model': 'Neural IRL (static)',        'RMSE_mean': r_n_mu, 'RMSE_std': r_n_se,
     'KL_mean': kl_n_mu, 'KL_std': kl_n_se,
     'Reduction': f'{100*(r_a_mu-r_n_mu)/r_a_mu:.1f}%', 'n_runs': N_RUNS},
    {'Model': 'MDP Neural IRL (x̄ state)', 'RMSE_mean': r_m_mu, 'RMSE_std': r_m_se,
     'KL_mean': kl_m_mu, 'KL_std': kl_m_se,
     'Reduction': f'{100*(r_a_mu-r_m_mu)/r_a_mu:.1f}%', 'n_runs': N_RUNS},
]).round(6)
t4_rows.to_csv(f"{CFG['out_dir']}/table4_mdp.csv", index=False)

# Table 5: mixture (last run — stochastic variability visible in π values)
cdf.round(4).assign(n_runs=N_RUNS).to_csv(
    f"{CFG['out_dir']}/table5_mixture.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 14  LATEX  (tables updated with mean ± std, figure envs updated)
# ─────────────────────────────────────────────────────────────────────────────

def L(*lines): return list(lines)

tex = []
tex += L(r'% ================================================================',
         r"% EMPIRICAL APPLICATION — Dominick's Analgesics",
         r'% Auto-generated by dominicks_irl.py',
         rf'% n\_runs = {N_RUNS}  (mean $\pm$ std across independent re-estimations)',
         r'% Packages: booktabs, threeparttable, graphicx, siunitx',
         r'% ================================================================', '')

# ─ Table D0: Descriptive statistics ──────────────────────────────────────────
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
r"  \caption{Descriptive Statistics: Dominick's Analgesics Scanner Panel}",
r'  \label{tab:dom_desc}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{lS[table-format=2.3]S[table-format=1.3]S[table-format=1.4]S[table-format=1.4]}',
r'      \toprule',
r'      \textbf{Good} & {\textbf{Mean Price}} & {\textbf{Std Price}} & {\textbf{Mean Share}} & {\textbf{Std Share}} \\',
r'      & {(\$/100 tablets)} & & & \\',
r'      \midrule',
)
for _, row in desc.iterrows():
    tex.append(f"      {row['Good']} & {row['Mean price']:.3f} & "
               f"{row['Std price']:.3f} & {row['Mean share']:.4f} & "
               f"{row['Std share']:.4f} \\\\")
tex += L(r'      \midrule',
         f"      \\multicolumn{{5}}{{l}}{{\\textit{{Train: {len(tr):,} "
         f"obs\\quad Test: {len(te):,} obs\\quad Stores: "
         f"{panel.STORE.nunique()}\\quad Weeks: "
         f"{int(weeks.min())}--{int(weeks.max())}}}}} \\\\",
         r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         r'      \item Unit prices standardised to per-100-tablet equivalent: '
         r'$\hat{p}_{ij}=\text{PRICE}_{ij}\times100/\text{TABLETS}_{ij}$.',
         r'      Revenue-weighted mean across UPCs within each store$\times$week$\times$category cell.',
         r'      Aspirin: Bayer, Bufferin, Anacin, Ascriptin, Ecotrin, generics.',
         r'      Acetaminophen: Tylenol, Excedrin, Anacin-3, Panadol, Datril, Pamprin, store brands.',
         r'      Ibuprofen: Advil, Motrin~IB, Nuprin, Aleve, Actron, Orudis~KT, Haltran, Medipren.',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Table D1: Accuracy with mean ± std ─────────────────────────────────────────
best_rmse_nm = min(MODEL_NAMES, key=lambda nm: rmse_mean[nm])
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
rf"  \caption{{Out-of-Sample Predictive Accuracy --- Dominick's Analgesics "
rf"({N_RUNS} independent re-estimations; mean $\pm$ std)}}",
r'  \label{tab:dom_acc}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{lcc}',
r'      \toprule',
r'      \textbf{Model} & \textbf{RMSE} & \textbf{MAE} \\',
r'      \midrule',
)
for nm in MODEL_NAMES:
    b  = r'\textbf{' if nm == best_rmse_nm else ''
    bc = '}' if b else ''
    r_str = (f'${rmse_mean[nm]:.5f} \\pm {rmse_std[nm]:.5f}$'
             if N_RUNS > 1 else f'${rmse_mean[nm]:.5f}$')
    m_str = (f'${mae_mean[nm]:.5f} \\pm {mae_std[nm]:.5f}$'
             if N_RUNS > 1 else f'${mae_mean[nm]:.5f}$')
    tex.append(f'      {b}{nm}{bc} & {b}{r_str}{bc} & {m_str} \\\\')
tex += L(r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         rf'      \item RMSE and MAE on held-out test observations, mean $\pm$ std over {N_RUNS} run(s).',
        r'      MDP Neural IRL augments the state with lagged quantities',
        r'      $\bar{x}_t=\hat{\delta}\bar{x}_{t-1}+(1-\hat{\delta})q_{t-1}$ ($\hat{\delta}$ learned)',
        r'      capturing brand-loyalty inertia. Bold: best mean RMSE.',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Table D2: Elasticities with mean ± std ────────────────────────────────────
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
rf"  \caption{{Own-Price Quantity Elasticities --- Dominick's Analgesics "
rf"({N_RUNS} run(s); mean $\pm$ std)}}",
r'  \label{tab:dom_elast}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{lccc}',
r'      \toprule',
r'      \textbf{Model} & {$\hat{\varepsilon}_{00}$ (Aspirin)} & {$\hat{\varepsilon}_{11}$ (Acetaminophen)} & {$\hat{\varepsilon}_{22}$ (Ibuprofen)} \\',
r'      \midrule',
)
for nm in MODEL_NAMES:
    def _fmt_e(mu, se):
        if np.isnan(mu): return '{---}'
        if N_RUNS > 1:   return f'${mu:.3f} \\pm {se:.3f}$'
        return f'${mu:.3f}$'
    row = ' & '.join(_fmt_e(elast_mean[nm][j], elast_std[nm][j]) for j in range(G))
    tex.append(f'      {nm} & {row} \\\\')
tex += L(r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         r'      \item Numerical own-price quantity elasticities at mean test prices and expenditure.',
         rf'      Mean $\pm$ std over {N_RUNS} independent re-estimation(s).',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Table D3: Welfare with mean ± std ──────────────────────────────────────────
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
rf'  \caption{{Consumer Surplus Loss from {int(ss*100)}\% Ibuprofen Price Increase --- '
rf"Dominick\'s Analgesics ({N_RUNS} run(s); mean $\pm$ std)}}",
r'  \label{tab:dom_welfare}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{lcr}',
r'      \toprule',
r'      \textbf{Model} & \textbf{CV Loss (\$)} & \textbf{vs Neural IRL} \\',
r'      \midrule',
)
for nm in MODEL_NAMES:
    v = welf_mean[nm]; se = welf_std[nm]
    diff = ('' if nm == 'Neural IRL' or np.isnan(nw_mu)
            else f'{100*(v-nw_mu)/abs(nw_mu):+.1f}\\%')
    if N_RUNS > 1:
        cv_str = f'${v:+.4f} \\pm {se:.4f}$'
    else:
        cv_str = f'${v:+.4f}$'
    tex.append(f'      {nm} & {cv_str} & {diff} \\\\')
tex += L(r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         rf'      \item Compensating variation via 100-step Riemann sum, '
         rf'$p_{{\mathrm{{Ibu}}}}\to(1+{ss})\,p_{{\mathrm{{Ibu}}}}$.',
         rf'      Mean $\pm$ std over {N_RUNS} run(s).',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Table D4: MDP advantage with mean ± std ───────────────────────────────────
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
rf"  \caption{{MDP State Augmentation: Brand Loyalty in Analgesic Demand "
rf"({N_RUNS} run(s); mean $\pm$ std)}}",
r'  \label{tab:dom_mdp}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{lcccc}',
r'      \toprule',
r'      \textbf{Model} & \textbf{RMSE} & \textbf{KL Div.} & \textbf{Reduction} \\',
r'      \midrule',
)
for (mn, rm, rs, km, ks, rd) in mdp_rows:
    b  = r'\textbf{' if 'MDP' in mn else ''
    bc = '}' if b else ''
    if N_RUNS > 1:
        r_str = f'${rm:.5f} \\pm {rs:.5f}$'
        k_str = f'${km:.5f} \\pm {ks:.5f}$'
    else:
        r_str = f'${rm:.5f}$'
        k_str = f'${km:.5f}$'
    tex.append(f'      {b}{mn}{bc} & {b}{r_str}{bc} & {k_str} & {rd} \\\\')
tex += L(r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         r'      \item $\bar{x}_t=\hat{\delta}\bar{x}_{t-1}+(1-\hat{\delta})q_{t-1}$; $\hat{\delta}$ is learned end-to-end via sigmoid re-parameterisation.',
         r'      Captures repeat-purchase inertia: aspirin loyalists (Bayer, Bufferin)',
         r'      rarely switch to ibuprofen products even under promotional pricing.',
         rf'      RMSE and KL: mean $\pm$ std over {N_RUNS} independent re-estimation(s).',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Table D5: Mixture components (last run) ───────────────────────────────────
tex += L(
r'\begin{table}[htbp]',
r'  \centering',
rf"  \caption{{Variational Mixture IRL: Consumer Segments --- Dominick\'s Analgesics "
rf"($K=6$; last of {N_RUNS} run(s))}}",
r'  \label{tab:dom_mix}',
r'  \begin{threeparttable}',
r'    \begin{tabular}{cS[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]l}',
r'      \toprule',
r'      $k$ & {$\hat{\pi}_k$} & {$\hat{\alpha}_{\text{Asp}}$} & {$\hat{\alpha}_{\text{Acet}}$} & {$\hat{\alpha}_{\text{Ibu}}$} & {$\hat{\rho}$} & Segment \\',
r'      \midrule',
)
for _, row in cdf.iterrows():
    if   row['alpha_asp']  > 0.45: seg = 'Aspirin-loyal (Bayer/Bufferin)'
    elif row['alpha_acet'] > 0.45: seg = 'Tylenol-loyal'
    elif row['alpha_ibu']  > 0.45: seg = 'Advil/Motrin-loyal'
    elif row['pi']         > 0.25: seg = r'\textbf{Dominant / mixed}'
    elif row['rho']        > 0.55: seg = 'Price-sensitive switchers'
    else:                           seg = 'Balanced'
    tex.append(f"      {int(row['K'])} & {row['pi']:.3f} & {row['alpha_asp']:.3f} & "
               f"{row['alpha_acet']:.3f} & {row['alpha_ibu']:.3f} & {row['rho']:.3f} & {seg} \\\\")
tex += L(r'      \bottomrule',
         r'    \end{tabular}',
         r'    \begin{tablenotes}\small',
         r'      \item Gaussian mixture in $(\bm{\alpha},\rho)$ CES parameter space.',
         r'      Brand-loyal segments reflect documented repeat-purchase inertia in',
         r'      OTC analgesics: aspirin users (Bayer/Bufferin loyalists), Tylenol',
         r'      users, and Advil/Motrin users rarely switch across active-ingredient',
         r'      categories. Price-sensitive switchers respond to promotional pricing.',
         rf'      Results shown for the last of {N_RUNS} independent run(s).',
         r'    \end{tablenotes}',
         r'  \end{threeparttable}',
         r'\end{table}', '')

# ─ Figure environments ────────────────────────────────────────────────────────
_se_band_note = (r' Shaded bands indicate $\pm 1$ standard deviation across '
                 rf'{N_RUNS} independent re-estimations.' if N_RUNS > 1 else '')

FDEFS = [
    ('fig_demand_curves',
     f"Aspirin Budget Share as a Function of Ibuprofen Unit Price --- "
     f"Dominick's Analgesics. All models trained on pre-cutoff "
     r"store$\times$week observations. The orange dotted line marks the "
     r"mean ibuprofen unit price in the test sample. Neural IRL (blue) and "
     r"MDP-IRL (teal) trace nearly identical curves, lying above LA-AIDS "
     r"at high prices --- consistent with stronger own-price sensitivity "
     r"in the data. Lin IRL Shared (yellow) is flattened by promotional "
     r"price collinearity, replicating the feature-collinearity finding "
     f"from the simulation study.{_se_band_note}",
     'fig:dom_demand'),
    ('fig_mdp_advantage',
     f"MDP-Aware Neural IRL vs.\\ Static Models --- Dominick's Analgesics "
     r"(all three budget shares). The MDP Neural IRL (teal) conditions on "
     r"the lagged budget share $\bar{x}_t$, capturing brand-loyalty "
     r"persistence absent from static models. Differences are largest for "
     r"aspirin (left panel), where Bayer and Bufferin repeat-purchase "
     f"inertia is strongest.{_se_band_note}",
     'fig:dom_mdp'),
    ('fig_convergence',
     f"Training Convergence --- Dominick's Analgesics  (last run). "
     r"Left: Neural IRL (static). Right: MDP Neural IRL. "
     r"The learnable temperature $\hat{\beta}$ stabilises rapidly, "
     r"providing a data-driven estimate of consumer rationality in the "
     r"analgesics category. The MDP model's KL trajectory reflects "
     r"the additional information extracted from the lagged-share state.",
     'fig:dom_conv'),
    ('fig_scatter',
     f"Observed vs.\\ Predicted Budget Shares --- Dominick's Analgesics "
     r"(3 models $\times$ 3 goods; last run). Points on the 45-degree line indicate "
     r"perfect prediction. Neural IRL and MDP-IRL cluster tighter around "
     r"the diagonal than LA-AIDS, consistent with the RMSE comparisons "
     r"in Table~\ref{tab:dom_acc}.",
     'fig:dom_scatter'),
    ('fig_mixture',
     f"Continuous Variational Mixture IRL ($K=6$) --- Dominick's Analgesics  (last run). "
     r"Left: recovered weights $\hat{\pi}_k$ with $\hat{\rho}$ annotated. "
     r"Right: component centres in "
     r"$(\hat{\alpha}_{\mathrm{Asp}},\hat{\alpha}_{\mathrm{Acet}})$ "
     r"space; marker size $\propto\hat{\pi}_k$. Identified segments "
     r"correspond to brand-loyal aspirin, acetaminophen, and ibuprofen "
     r"users, plus a price-sensitive switching segment with high $\hat{\rho}$.",
     'fig:dom_mix'),
]
if N_RUNS > 1:
    FDEFS += [
        ('fig_rmse_bars',
         f'Out-of-Sample RMSE across all models --- mean $\\pm 1$ SD over {N_RUNS} '
         f"independent re-estimations.  Error bars quantify sensitivity to random "
         f"initialisation.  Deterministic models (LA-AIDS, BLP-IV) show zero variance "
         f"by construction.",
         'fig:dom_rmse_bars'),
        ('fig_beta_runs',
         f'Learned temperature parameter $\\hat{{\\beta}}$ across {N_RUNS} '
         f'independent re-estimations.  Violin plots show the full sampling '
         f'distribution; means and medians are marked.  The narrow spread '
         f'confirms that the rationaliy level identified by MaxEnt IRL '
         f'is robust to initialisation.',
         'fig:dom_beta_runs'),
    ]

for fn, cap, lbl in FDEFS:
    tex += L(r'\begin{figure}[htbp]',
             r'  \centering',
             f"  \\includegraphics[width=\\textwidth]{{figures/dominicks/{fn}.pdf}}",
             f'  \\caption{{{cap}}}',
             f'  \\label{{{lbl}}}',
             r'\end{figure}', '')

tp = f"{CFG['out_dir']}/dominicks_latex.tex"
with open(tp, 'w') as f:
    f.write('\n'.join(tex))
print(f'  Saved: {tp}')


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
bm = min(MODEL_NAMES, key=lambda nm: rmse_mean[nm])
print('\n' + '='*72)
print("  RESULTS SUMMARY — DOMINICK'S ANALGESICS")
print('='*72)
print(f"""
  n_runs = {N_RUNS}  (mean ± std across {N_RUNS} independent re-estimation(s))

  ACCURACY (test RMSE, mean ± std):
    LA-AIDS:        {r_a_mu:.5f} ± {r_a_se:.5f}
    Neural IRL:     {r_n_mu:.5f} ± {r_n_se:.5f}  (β̂={beta_n_mu:.4f} ± {beta_n_se:.4f})
    MDP Neural IRL: {r_m_mu:.5f} ± {r_m_se:.5f}  (β̂={beta_m_mu:.4f} ± {beta_m_se:.4f}  δ̂={delta_m_mu:.4f} ± {delta_m_se:.4f})
    Best model (by mean RMSE): {bm}

  MDP GAIN (using means):
    Neural vs AIDS: {100*(r_a_mu-r_n_mu)/r_a_mu:.1f}% RMSE reduction
    MDP vs AIDS:    {100*(r_a_mu-r_m_mu)/r_a_mu:.1f}% RMSE reduction
    MDP vs static:  {100*(r_n_mu-r_m_mu)/max(r_n_mu,1e-9):.1f}% RMSE reduction

  WELFARE ({int(ss*100)}% ibuprofen shock, mean ± std):
    Neural IRL CV:  ${abs(welf_mean.get('Neural IRL', float('nan'))):.4f} ± {welf_std.get('Neural IRL', 0.0):.4f}
    MDP IRL CV:     ${abs(welf_mean.get('MDP Neural IRL', float('nan'))):.4f} ± {welf_std.get('MDP Neural IRL', 0.0):.4f}

  MIXTURE (dominant component, last run):
    π={dk['pi']:.3f}  α=[{dk['alpha_asp']:.3f},{dk['alpha_acet']:.3f},{dk['alpha_ibu']:.3f}]  ρ={dk['rho']:.3f}

  FILES:
    {CFG['fig_dir']}/fig_{{demand_curves,mdp_advantage,convergence,scatter,mixture}}.{{pdf,png}}
    {CFG['fig_dir']}/fig_{{rmse_bars,beta_runs}}.{{pdf,png}}  (new, n_runs>1 only)
    {CFG['out_dir']}/table{{0,1,2,3,4,5}}_*.csv
    {CFG['out_dir']}/dominicks_latex.tex
""")