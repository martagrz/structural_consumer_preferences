"""LA-AIDS model implementation."""

import numpy as np


class LAAIDS:
    """Linear Approximate AIDS with Stone price index, estimated by OLS."""

    name = "LA-AIDS"

    def fit(self, p, w, y):
        lp = np.log(np.maximum(p, 1e-8))
        lPs = (w * lp).sum(1, keepdims=True)
        ly = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lPs
        X = np.c_[np.ones(len(y)), lp, ly.squeeze()]
        self.coef_ = np.linalg.lstsq(X, w, rcond=None)[0]
        return self

    def predict(self, p, y):
        g = p.shape[1]
        lp = np.log(np.maximum(p, 1e-8))
        lP = (np.full((len(p), g), 1.0 / g) * lp).sum(1, keepdims=True)
        ly = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lP
        w = np.clip(np.c_[np.ones(len(p)), lp, ly.squeeze()] @ self.coef_, 1e-6, 1.0)
        return w / w.sum(1, keepdims=True)


class AIDSBench(LAAIDS):
    """Simulation benchmark alias keeping original non-chaining fit API."""

    name = "LA-AIDS"

    def fit(self, p, w, y):
        super().fit(p, w, y)

