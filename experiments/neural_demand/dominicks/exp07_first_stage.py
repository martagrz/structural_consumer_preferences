"""
experiments/neural_demand/dominicks/exp07_first_stage.py
========================================================
Calculates first-stage F-statistics, R2, and partial R2 for the Hausman instruments.
Also investigates Ibuprofen specific stats.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from experiments.dominicks.data import load, GOODS, G

def run_first_stage_diagnostics(cfg):
    print("Loading data...")
    data, splits = load(cfg)
    
    p_tr = splits['p_tr']
    s_tr = splits['s_tr']
    wk_tr = splits['wk_tr']
    Z_tr = splits['Z_tr']
    
    # Log prices
    log_p = np.log(np.maximum(p_tr, 1e-8))
    
    results = []
    
    print("\nFirst-stage diagnostics (Regression: log(p_j) ~ Z_j + const)")
    print("-" * 80)
    print(f"{'Good':<15} {'F-stat':<10} {'p-val':<10} {'R2':<10} {'Adj R2':<10} {'Coeff':<10} {'t-stat':<10}")
    print("-" * 80)
    
    for j in range(G):
        y = log_p[:, j]
        X = sm.add_constant(Z_tr[:, j])
        
        model = sm.OLS(y, X)
        res = model.fit()
        
        f_stat = res.fvalue
        f_pvalue = res.f_pvalue
        r2 = res.rsquared
        adj_r2 = res.rsquared_adj
        coeff = res.params[1]
        t_stat = res.tvalues[1]
        
        print(f"{GOODS[j]:<15} {f_stat:<10.2f} {f_pvalue:<10.2e} {r2:<10.4f} {adj_r2:<10.4f} {coeff:<10.4f} {t_stat:<10.2f}")
        
        results.append({
            'Good': GOODS[j],
            'F_stat': f_stat,
            'R2': r2,
            'Partial_R2': r2 # Same as R2 in simple regression
        })
        
    # With Store Fixed Effects
    print("\nFirst-stage diagnostics with Store Fixed Effects (Regression: log(p_j) ~ Z_j + StoreFE)")
    print("-" * 80)
    print(f"{'Good':<15} {'F-stat (Z)':<12} {'R2':<10} {'Partial R2':<12}")
    print("-" * 80)
    
    # Create dummy variables for stores
    # This might be memory intensive if many stores. 
    # Dominicks has ~90 stores. 
    # We can use the within transformation (demeaning) to calculate partial R2 without full dummies.
    
    # Unique stores in train
    unique_stores = np.unique(s_tr)
    n_stores = len(unique_stores)
    print(f"Number of stores in train: {n_stores}")
    
    # Create dummies
    # s_tr is the store id.
    # We can use pandas get_dummies
    store_dummies = pd.get_dummies(s_tr, prefix='store', drop_first=True).astype(float)
    
    for j in range(G):
        y = log_p[:, j]
        z = Z_tr[:, j]
        
        # Full model: log(p) ~ Z + StoreFE
        X_full = pd.DataFrame({'Z': z})
        X_full = pd.concat([X_full, store_dummies.reset_index(drop=True)], axis=1)
        X_full = sm.add_constant(X_full)
        
        model_full = sm.OLS(y, X_full)
        res_full = model_full.fit()
        
        # Restricted model: log(p) ~ StoreFE
        X_rest = sm.add_constant(store_dummies.reset_index(drop=True))
        model_rest = sm.OLS(y, X_rest)
        res_rest = model_rest.fit()
        
        # Partial R2 = (RSS_rest - RSS_full) / RSS_rest
        rss_full = res_full.ssr
        rss_rest = res_rest.ssr
        partial_r2 = (rss_rest - rss_full) / rss_rest
        
        # F-stat for Z
        # We can get this from the t-test of Z in the full model (F = t^2 for single variable)
        # Or use the F-test comparison
        f_stat = res_full.tvalues['Z'] ** 2
        
        print(f"{GOODS[j]:<15} {f_stat:<12.2f} {res_full.rsquared:<10.4f} {partial_r2:<12.4f}")
        
        # Update results
        results[j]['F_stat_FE'] = f_stat
        results[j]['R2_FE'] = res_full.rsquared
        results[j]['Partial_R2_FE'] = partial_r2

    # Save results
    df_res = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df_res)
    
    # Investigate Ibuprofen (Index 2)
    print("\nInvestigating Ibuprofen (IBU) correlations...")
    ibu_idx = 2
    ibu_p = p_tr[:, ibu_idx]
    ibu_z = Z_tr[:, ibu_idx]
    
    # Correlation with other prices
    corr_p = np.corrcoef(p_tr.T)
    print("Price Correlation Matrix:")
    print(corr_p)
    
    # Correlation with total revenue (proxy for category demand?)
    # We don't have total category demand shock directly.
    # But we can look at correlation with total quantity or revenue.
    w_tr = splits['w_tr']
    y_tr = splits['y_tr'] # Income proxy (total revenue / 100)
    
    # Total quantity Q = sum(q_j)
    # q_j = w_j * Y / p_j
    # But Y is scaled. Real Y = y_tr * 100.
    q_tr = w_tr * (y_tr[:, None] * 100) / np.maximum(p_tr, 1e-8)
    total_q = q_tr.sum(axis=1)
    
    corr_ibu_p_q = np.corrcoef(ibu_p, total_q)[0, 1]
    print(f"Correlation between IBU Price and Total Category Quantity: {corr_ibu_p_q:.4f}")
    
    corr_ibu_p_y = np.corrcoef(ibu_p, y_tr)[0, 1]
    print(f"Correlation between IBU Price and Total Category Revenue: {corr_ibu_p_y:.4f}")

if __name__ == "__main__":
    # Minimal config
    CFG = {
        'weekly_path': 'data/wana.csv',
        'upc_path':    'data/upcana.csv',
        'min_store_wks': 100,
        'test_cutoff':   350,
        'std_tablets':   100,
        'habit_decay':   0.9,
        'shock_good':    2,  # IBU
        'shock_pct':     0.10,
        'fig_dir':       'figures_diagnostics',
        'out_dir':       'output_diagnostics',
        'device':        'cpu',
    }
    
    # Check if data exists, otherwise mock or warn
    import os
    if os.path.exists(CFG['weekly_path']):
        run_first_stage_diagnostics(CFG)
    else:
        print(f"Data file {CFG['weekly_path']} not found. Please ensure data is available.")
