"""Unified mixture-model module with both Dominicks and simulation classes."""

import numpy as np

from .ces_consumer import CESConsumer


class VarMixture:
    """Dominicks variational mixture IRL."""

    name = "Var. Mixture IRL"

    def __init__(self, cfg, K=6, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.K = K
        self.G = 3
        a0 = np.ones((K, self.G)) / self.G
        noise = rng.uniform(-0.1, 0.1, (K, self.G))
        a0 = np.abs(a0 + noise)
        a0 /= a0.sum(1, keepdims=True)
        r0 = np.linspace(0.3, 0.6, K)
        self.mu_a = np.log(a0 + 1e-6)
        self.mu_r = np.log(r0 / (1 - r0))
        self.sa = 0.5 * np.ones((K, self.G))
        self.sr = 0.3 * np.ones(K)
        self.pi = np.ones(K) / K

    def _decode(self, la, lr):
        a = np.exp(la - la.max())
        a /= a.sum()
        r = float(np.clip(1 / (1 + np.exp(-lr)), 0.05, 0.95))
        return a, r

    def _ces(self, p, a, r):
        s = 1.0 / (1.0 - r)
        num = a[None, :] ** s * np.maximum(p, 1e-8) ** (1 - s)
        return num / num.sum(1, keepdims=True)

    def _comp(self, k, p, _y):
        rng = np.random.default_rng(k * 99)
        return np.stack(
            [
                self._ces(
                    p,
                    *self._decode(rng.normal(self.mu_a[k], self.sa[k]), rng.normal(self.mu_r[k], self.sr[k])),
                )
                for _ in range(self.cfg["mix_n_spc"])
            ]
        ).mean(0)

    def fit(self, p, y, w):
        lr, sig2 = self.cfg["mix_lr_mu"], self.cfg["mix_sigma2"]
        n_grad = min(100, len(p))

        for it in range(self.cfg["mix_n_iter"]):
            wk = np.stack([self._comp(k, p, y) for k in range(self.K)])
            log_r = np.array(
                [
                    -np.sum((wk[k] - w) ** 2, 1) / (2 * sig2) + np.log(self.pi[k] + 1e-10)
                    for k in range(self.K)
                ]
            )
            log_r -= log_r.max(0)
            resp = np.exp(log_r)
            resp /= resp.sum(0, keepdims=True)

            self.pi = resp.mean(1) + 0.01
            self.pi /= self.pi.sum()

            for k in range(self.K):
                sig = np.mean(resp[k, :, None] * (w - wk[k]), 0)
                for j in range(self.G):
                    h = 0.01
                    self.mu_a[k, j] += h
                    d = (self._comp(k, p[:n_grad], y[:n_grad]) - wk[k][:n_grad]).mean(0)
                    self.mu_a[k, j] -= h
                    self.mu_a[k, j] += lr * np.dot(sig, d) / (h + 1e-8)

                h = 0.01
                self.mu_r[k] += h
                d = (self._comp(k, p[:n_grad], y[:n_grad]) - wk[k][:n_grad]).mean(0)
                self.mu_r[k] -= h
                self.mu_r[k] += lr * np.dot(sig, d) / (h + 1e-8)

            if (it + 1) % 10 == 0:
                mse = np.mean((np.einsum("k,kng->ng", self.pi, wk) - w) ** 2)
                print(f"    iter {it + 1:2d} | MSE={mse:.5f} | pi={np.round(self.pi, 3)}")
        return self

    def predict(self, p, y):
        wk = np.stack([self._comp(k, p, y) for k in range(self.K)])
        return np.einsum("k,kng->ng", self.pi, wk)

    def summary(self):
        import pandas as pd

        rows = []
        for k in range(self.K):
            a, r = self._decode(self.mu_a[k], self.mu_r[k])
            rows.append(
                {
                    "K": k + 1,
                    "pi": self.pi[k],
                    "alpha_asp": a[0],
                    "alpha_acet": a[1],
                    "alpha_ibu": a[2],
                    "rho": r,
                }
            )
        return pd.DataFrame(rows)


class ContinuousVariationalMixture:
    """Simulation continuous variational mixture IRL."""

    def __init__(self, K=6, n_goods=3, n_samples_per_component=5):
        self.K = K
        self.G = n_goods
        self.n_spc = n_samples_per_component
        rng = np.random.default_rng(0)
        alpha_means = np.ones((K, n_goods)) / n_goods
        noise = rng.uniform(-0.1, 0.1, (K, n_goods))
        alpha_means = np.abs(alpha_means + noise)
        alpha_means /= alpha_means.sum(1, keepdims=True)
        rho_means = np.linspace(0.3, 0.6, K)
        self.mu_alpha = np.log(alpha_means + 1e-6)
        self.mu_rho = np.log(rho_means / (1 - rho_means))
        self.sigma_alpha = 0.5 * np.ones((K, n_goods))
        self.sigma_rho = 0.3 * np.ones(K)
        self.pi = np.ones(K) / K

    def _decode_alpha(self, la):
        a = np.exp(la - la.max())
        return a / a.sum()

    def _decode_rho(self, lr):
        return 1.0 / (1.0 + np.exp(-lr))

    def _sample_consumers(self, k, n_samples):
        rng = np.random.default_rng(k * 100)
        out = []
        for _ in range(n_samples):
            la = rng.normal(self.mu_alpha[k], self.sigma_alpha[k])
            lr = rng.normal(self.mu_rho[k], self.sigma_rho[k])
            out.append(
                CESConsumer(alpha=self._decode_alpha(la), rho=float(np.clip(self._decode_rho(lr), 0.05, 0.95)))
            )
        return out

    def _predict_component(self, k, prices, income):
        cs = self._sample_consumers(k, self.n_spc)
        return np.stack([c.solve_demand(prices, income) for c in cs]).mean(0)

    def fit(self, prices, income, w_obs, n_iter=50, lr_mu=0.05, sigma2=0.1):
        n_grad = min(100, len(prices))
        for _ in range(n_iter):
            w_k = np.stack([self._predict_component(k, prices, income) for k in range(self.K)])
            log_resp = np.array(
                [
                    -np.sum((w_k[k] - w_obs) ** 2, axis=1) / (2 * sigma2) + np.log(self.pi[k] + 1e-10)
                    for k in range(self.K)
                ]
            )
            log_resp -= log_resp.max(0)
            resp = np.exp(log_resp)
            resp /= resp.sum(0, keepdims=True)

            self.pi = resp.mean(1) + 0.01
            self.pi /= self.pi.sum()

            for k in range(self.K):
                rk = resp[k]
                err_k = w_obs - w_k[k]
                signal = np.mean(rk[:, None] * err_k, 0)

                for j in range(self.G):
                    h = 0.01
                    self.mu_alpha[k, j] += h
                    wp = self._predict_component(k, prices[:n_grad], income[:n_grad])
                    self.mu_alpha[k, j] -= h
                    d = (wp - w_k[k][:n_grad]).mean(0)
                    self.mu_alpha[k, j] += lr_mu * np.dot(signal, d) / (h + 1e-8)

                h = 0.01
                self.mu_rho[k] += h
                wp = self._predict_component(k, prices[:n_grad], income[:n_grad])
                self.mu_rho[k] -= h
                d = (wp - w_k[k][:n_grad]).mean(0)
                self.mu_rho[k] += lr_mu * np.dot(signal, d) / (h + 1e-8)
        return self

    def predict(self, prices, income):
        w_k = np.stack([self._predict_component(k, prices, income) for k in range(self.K)])
        return np.einsum("k,kng->ng", self.pi, w_k)

    def get_component_summary(self):
        import pandas as pd

        rows = []
        for k in range(self.K):
            alpha = self._decode_alpha(self.mu_alpha[k])
            rho = float(np.clip(self._decode_rho(self.mu_rho[k]), 0.05, 0.95))
            rows.append(
                {
                    "component": k + 1,
                    "pi": self.pi[k],
                    "alpha_food": alpha[0],
                    "alpha_fuel": alpha[1],
                    "alpha_other": alpha[2],
                    "rho": rho,
                }
            )
        return pd.DataFrame(rows)

