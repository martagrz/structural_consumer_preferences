"""BLP logit-IV model implementation."""

import numpy as np


class BLPLogitIV:
    """BLP logit with Hausman IV. Last good is outside option."""

    name = "BLP (IV)"

    def fit(self, p, w, z):
        y = np.log(np.maximum(w[:, :-1], 1e-8) / np.maximum(w[:, -1:], 1e-8))
        z_reduced = z[:, :-1]
        p_hat = z_reduced @ np.linalg.lstsq(z_reduced, p[:, :-1], rcond=None)[0]
        self.beta_ = np.linalg.lstsq(p_hat, y, rcond=None)[0]
        return self

    def predict(self, p):
        lgt = np.clip(p[:, :-1] @ self.beta_, -30, 30)
        eu = np.exp(lgt)
        denom = 1.0 + eu.sum(1, keepdims=True)
        return np.c_[eu / denom, 1.0 / denom]


class BLPBench(BLPLogitIV):
    """Simulation benchmark alias keeping original non-chaining fit API."""

    name = "BLP (IV)"

    def fit(self, p, w, z):
        super().fit(p, w, z)

