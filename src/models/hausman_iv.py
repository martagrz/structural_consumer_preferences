"""Hausman IV construction."""

import numpy as np


def hausman_iv(prices, stores, weeks):
    """Mean price of same good across other stores in same week."""

    G = prices.shape[1]
    Z = np.zeros_like(prices)
    for j in range(G):
        for i, (s, wk) in enumerate(zip(stores, weeks)):
            mask = (stores != s) & (weeks == wk)
            Z[i, j] = prices[mask, j].mean() if mask.sum() else prices[:, j].mean()
    return Z

