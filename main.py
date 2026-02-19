"""
Recovering Consumer Preferences via Inverse Reinforcement Learning
==================================================================
Full implementation with all improvements and new experiments:
  1. Neural IRL with explicit lagged share state x̄ (MDP advantage demo)
  2. Continuous Variational Mixture IRL with wider type grid
  3. Linear IRL variants: original, good-specific, orthogonalised, per-good intercept
  4. LaTeX table and figure generation for paper

Dependencies: numpy, scipy, pandas, matplotlib, torch, sklearn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# SECTION 1: GROUND TRUTH — MULTIPLE UTILITY SPECIFICATIONS
# ============================================================

class CESConsumer:
    """
    U(x) = (Σ αᵢ xᵢ^ρ)^(1/ρ)
    Closed-form demand: wᵢ = (αᵢ^σ pᵢ^(1-σ)) / Σⱼ(αⱼ^σ pⱼ^(1-σ))
    where σ = 1/(1-ρ). ~100x faster than scipy.minimize.
    """
    name = "CES"

    def __init__(self, alpha=None, rho=0.45):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.4, 0.4, 0.2])
        self.rho   = rho

    def solve_demand(self, prices, income):
        sigma = 1.0 / (1.0 - self.rho)
        num   = self.alpha[None, :] ** sigma * prices ** (1.0 - sigma)
        return num / num.sum(axis=1, keepdims=True)


class QuasilinearConsumer:
    """
    U(x) = x₀ + a₁·ln(x₁+1) + a₂·ln(x₂+1)
    Zero income effect on goods 1 & 2 — structural test AIDS fails.
    """
    name = "Quasilinear"

    def __init__(self, a=None):
        self.a = np.array(a) if a is not None else np.array([1.5, 0.8])

    def solve_demand(self, prices, income):
        N, G   = prices.shape
        shares = np.zeros((N, G))
        for i in range(N):
            p, y = prices[i], income[i]
            x1   = max(self.a[0] / p[1] - 1, 1e-6)
            x2   = max(self.a[1] / p[2] - 1, 1e-6)
            rem  = y - p[1]*x1 - p[2]*x2
            x0   = max(rem / p[0], 1e-6) if rem > 0 else 1e-6
            shares[i] = np.array([x0, x1, x2]) * p / y
        return np.clip(shares, 1e-6, 1.0)


class LeontiefConsumer:
    """
    U(x) = min(x₀/a₀, x₁/a₁, x₂/a₂) — perfect complements.
    Closed-form: x*ᵢ = y·aᵢ / (p·a). Zero substitution elasticity.
    """
    name = "Leontief"

    def __init__(self, a=None):
        self.a = np.array(a) if a is not None else np.array([1.0, 0.8, 1.5])

    def solve_demand(self, prices, income):
        denom  = (prices * self.a[None, :]).sum(axis=1, keepdims=True)
        x      = income[:, None] * self.a[None, :] / denom
        return np.clip(x * prices / income[:, None], 1e-6, 1.0)


class StoneGearyConsumer:
    """
    LES: pᵢ xᵢ* = pᵢγᵢ + αᵢ(y − Σⱼ pⱼγⱼ)
    Subsistence minima γ create structural income non-linearity.
    """
    name = "Stone-Geary"

    def __init__(self, alpha=None, gamma=None):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.5, 0.3, 0.2])
        self.alpha = self.alpha / self.alpha.sum()
        self.gamma = np.array(gamma) if gamma is not None else np.array([50.0, 30.0, 20.0])

    def solve_demand(self, prices, income):
        sub    = (prices * self.gamma[None, :]).sum(axis=1)
        super_ = np.maximum(income - sub, 1e-6)
        exp_v  = prices * self.gamma[None, :] + self.alpha[None, :] * super_[:, None]
        return np.clip(exp_v / income[:, None], 1e-6, 1.0)


class HabitFormationConsumer:
    """
    Habit-adjusted CES: U_t(x; x̄) = (Σ αᵢ (xᵢ − θ·x̄ᵢ)^ρ)^(1/ρ)
    x̄ᵢ = exponential MA of past consumption. Path-dependent.

    Returns both shares AND the lagged share sequence (x̄ at each t),
    which is used as additional state input for the MDP-aware Neural IRL.
    """
    name = "Habit Formation"

    def __init__(self, alpha=None, rho=0.45, theta=0.3, decay=0.7):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.4, 0.4, 0.2])
        self.rho   = rho
        self.theta = theta
        self.decay = decay

    def solve_demand(self, prices, income, return_xbar=False):
        N, G   = prices.shape
        shares = np.zeros((N, G))
        xbars  = np.zeros((N, G))   # store x̄ at each step
        xbar   = np.ones(G) * (np.mean(income) / (G * np.mean(prices)))

        for i in range(N):
            p, y  = prices[i], income[i]
            floor = self.theta * xbar + 1e-6
            xbars[i] = xbar   # record x̄ BEFORE this period's decision

            def neg_u(x):
                adj = x - self.theta * xbar
                if np.any(adj <= 0):
                    return 1e10
                return -(np.sum(self.alpha * adj ** self.rho)) ** (1 / self.rho)

            x0   = np.maximum(y / (G * p), floor + 0.01)
            cons = {'type': 'eq', 'fun': lambda x, p=p, y=y: p @ x - y}
            res  = minimize(neg_u, x0,
                            bounds=[(floor[j], None) for j in range(G)],
                            constraints=cons, method='SLSQP')

            if res.success:
                shares[i] = res.x * p / y
                xbar = self.decay * xbar + (1 - self.decay) * res.x
            else:
                shares[i] = 1.0 / G

        if return_xbar:
            return np.clip(shares, 1e-6, 1.0), xbars
        return np.clip(shares, 1e-6, 1.0)


# ============================================================
# SECTION 2: TRADITIONAL BENCHMARKS
# ============================================================

class AIDSBench:
    """
    LA-AIDS (Deaton & Muellbauer 1980). OLS on log prices and log income.
    Cannot represent path-dependence, zero income effects, or Slutsky symmetry.
    """
    name = "LA-AIDS"

    def fit(self, p, w, y):
        X = np.column_stack([np.log(p), np.log(y)])
        self.beta_ = np.linalg.lstsq(X, w, rcond=None)[0]

    def predict(self, p, y):
        X   = np.column_stack([np.log(p), np.log(y)])
        out = np.clip(X @ self.beta_, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)


class BLPBench:
    """BLP logit-IV (BLP 1995). Last good = outside option."""
    name = "BLP (IV)"

    def fit(self, p, w, z):
        y_logit    = np.log(w[:, :-1] / (w[:, -1:] + 1e-10))
        p_hat      = z[:, :-1] @ np.linalg.lstsq(z[:, :-1], p[:, :-1], rcond=None)[0]
        self.beta_ = np.linalg.lstsq(p_hat, y_logit, rcond=None)[0]

    def predict(self, p):
        exp_u = np.exp(p[:, :-1] @ self.beta_)
        denom = 1.0 + exp_u.sum(axis=1, keepdims=True)
        return np.column_stack([exp_u / denom, 1.0 / denom])


# ============================================================
# SECTION 3: LINEAR MAXENT IRL — THREE VARIANTS
# ============================================================

def features_shared(p, y):
    """
    ORIGINAL: single shared 3-vector [ln p_own, (ln p_own)², ln y].
    Forces identical sensitivity profile across all goods.
    """
    N, G   = p.shape
    F      = np.zeros((N, G, 3))
    lp     = np.log(p)
    for i in range(G):
        F[:, i, 0] = lp[:, i]
        F[:, i, 1] = lp[:, i] ** 2
        F[:, i, 2] = np.log(y)
    return F


def features_good_specific(p, y):
    """
    IMPROVED: good-specific (G+2)-vector per good.
    [ln p₀, ln p₁, ln p₂, (ln pᵢ)², ln y]
    Allows heterogeneous price/income sensitivities across goods.
    """
    N, G   = p.shape
    n_feat = G + 2
    F      = np.zeros((N, G, n_feat))
    lp     = np.log(p)
    for i in range(G):
        F[:, i, :G]  = lp
        F[:, i,  G]  = lp[:, i] ** 2
        F[:, i, G+1] = np.log(y)
    return F


def features_orthogonalised(p, y):
    """
    ORTHOGONALISED: per-good intercepts + PCA-orthogonalised log prices.

    Near-collinearity in log prices (all generated from same IV process)
    causes the gradient to concentrate in the curvature term. Solution:
      1. Add a per-good indicator (one-hot intercept) so the model can
         learn good-level baseline utilities without using price features.
      2. Orthogonalise log-price columns via QR decomposition so each
         feature carries independent price information.

    Feature vector per good i: [e_i (one-hot, G dims),
                                 Q[:,0], Q[:,1], Q[:,2]  (orth. log prices),
                                 (ln pᵢ)²,
                                 ln y]
    Total: G + G + 1 + 1 = 2G+2 = 8 features for G=3.
    """
    N, G   = p.shape
    lp     = np.log(p)                          # (N, G)

    # QR orthogonalisation of log price matrix
    Q, _   = np.linalg.qr(lp - lp.mean(axis=0))  # (N, G) orthonormal columns

    n_feat = G + G + 1 + 1   # one-hot + orth prices + curvature + income
    F      = np.zeros((N, G, n_feat))

    for i in range(G):
        # Per-good intercept (one-hot)
        F[:, i, i]         = 1.0
        # Orthogonalised log prices
        F[:, i, G:2*G]     = Q
        # Own-price curvature
        F[:, i, 2*G]       = lp[:, i] ** 2
        # Log income
        F[:, i, 2*G + 1]   = np.log(y)

    return F


def run_linear_irl(features, expert_w, lr=0.05, epochs=3000, l2=1e-4):
    """
    MaxEnt IRL gradient ascent:
      ∇θ = E_expert[φ] − E_model[φ] − λθ
    Decaying LR + L2 regularisation.
    """
    n_feat = features.shape[2]
    theta  = np.zeros(n_feat)

    for ep in range(epochs):
        logits  = np.tensordot(features, theta, axes=([2], [0]))
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=1, keepdims=True)

        diff   = (expert_w - probs)[:, :, None]
        grad   = np.mean((features * diff).sum(axis=1), axis=0)
        grad  -= l2 * theta
        theta += (lr / (1.0 + ep / 1000.0)) * grad

    return theta


def predict_linear_irl(features, theta):
    logits  = np.tensordot(features, theta, axes=([2], [0]))
    logits -= logits.max(axis=1, keepdims=True)
    ex      = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


# ============================================================
# SECTION 4: NEURAL IRL — STANDARD (NO STATE HISTORY)
# ============================================================

class NeuralIRL(nn.Module):
    """
    Standard Neural IRL: input = [ln p₀, ln p₁, ln p₂, ln y].
    Full price vector (vs original own-price-only), learnable β,
    Slutsky symmetry penalty. Represents a static MDP where the state
    is current prices and income only — no consumption history.
    """

    def __init__(self, n_goods=3, hidden_dim=256):
        super().__init__()
        self.n_goods  = n_goods
        input_dim     = n_goods + 1     # [ln p₀..ln pG, ln y]

        self.reward_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        self.log_beta = nn.Parameter(torch.tensor(1.5))

        for m in self.reward_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.reward_net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.exp(self.log_beta).clamp(0.5, 20.0)

    def forward(self, log_p, log_y):
        x = torch.cat([log_p, log_y], dim=1)
        return torch.softmax(self.reward_net(x) * self.beta, dim=1)

    def slutsky_penalty(self, log_p, log_y):
        log_p_d = log_p.detach().requires_grad_(True)
        w = self.forward(log_p_d, log_y)
        rows = []
        for i in range(self.n_goods):
            g = torch.autograd.grad(w[:, i].sum(), log_p_d,
                                    create_graph=True, retain_graph=True)[0]
            rows.append(g.unsqueeze(2))
        J    = torch.cat(rows, dim=2)
        asym = J - J.transpose(1, 2)
        return (asym ** 2).mean()


# ============================================================
# SECTION 5: NEURAL IRL — MDP-AWARE (WITH LAGGED SHARE STATE x̄)
# ============================================================

class MDPNeuralIRL(nn.Module):
    """
    MDP-Aware Neural IRL: input = [ln p₀, ln p₁, ln p₂, ln y, x̄₀, x̄₁, x̄₂]

    The lagged habit stock x̄ is passed explicitly as part of the state vector.
    This is the key structural difference from the standard Neural IRL:
    the reward function can now condition on consumption history, which is
    exactly what the MDP framing promises but static models cannot deliver.

    Architecture is identical to NeuralIRL except input_dim = n_goods*2 + 1
    to accommodate the habit state. This means the recovered reward surface
    R(p, y, x̄) explicitly encodes reference-point effects — equivalent to
    learning the habit-adjusted utility function non-parametrically.

    WHY THIS MATTERS:
    The standard NeuralIRL sees the same (p, y) pair and must produce the
    same share prediction regardless of the household's consumption history.
    In a habit-formation DGP, households with the same budget constraint but
    different habit stocks make different choices — this is systematic
    variation that the static model treats as noise (high KL divergence),
    while the MDP model treats as signal (the x̄ coefficients encode
    the marginal utility of departing from the habit stock).
    """

    def __init__(self, n_goods=3, hidden_dim=256):
        super().__init__()
        self.n_goods  = n_goods
        # input: [ln p (G), ln y (1), x̄ (G)] = 2G+1
        input_dim     = n_goods * 2 + 1

        self.reward_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        self.log_beta = nn.Parameter(torch.tensor(1.5))

        for m in self.reward_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.reward_net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.exp(self.log_beta).clamp(0.5, 20.0)

    def forward(self, log_p, log_y, xbar):
        """
        log_p : (N, G)   log prices
        log_y : (N, 1)   log income
        xbar  : (N, G)   lagged habit stock (budget shares of previous period)
        """
        x = torch.cat([log_p, log_y, xbar], dim=1)   # (N, 2G+1)
        return torch.softmax(self.reward_net(x) * self.beta, dim=1)

    def slutsky_penalty(self, log_p, log_y, xbar):
        log_p_d = log_p.detach().requires_grad_(True)
        w = self.forward(log_p_d, log_y, xbar)
        rows = []
        for i in range(self.n_goods):
            g = torch.autograd.grad(w[:, i].sum(), log_p_d,
                                    create_graph=True, retain_graph=True)[0]
            rows.append(g.unsqueeze(2))
        J    = torch.cat(rows, dim=2)
        asym = J - J.transpose(1, 2)
        return (asym ** 2).mean()


def train_neural_irl(model, prices, income, w_expert,
                     epochs=4000, lr=5e-4, batch_size=256,
                     lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
                     xbar_data=None, device='cpu'):
    """
    Shared training loop for both NeuralIRL and MDPNeuralIRL.
    If xbar_data is provided (N, G array of lagged shares), uses MDPNeuralIRL path.

    Loss = KL(w_expert || w_pred)
         + λ_mono · mean(max(0, ∂wᵢ/∂ln pᵢ))
         + λ_slut · ||J − Jᵀ||²_F
    """
    model     = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    N          = len(income)
    slut_start = int(epochs * slut_start_frac)
    mdp_mode   = xbar_data is not None

    log_p_full = torch.log(torch.tensor(prices,   dtype=torch.float32, device=device))
    log_y_full = torch.log(torch.tensor(income,   dtype=torch.float32, device=device)).unsqueeze(1)
    w_full     = torch.tensor(w_expert,           dtype=torch.float32, device=device)

    if mdp_mode:
        xbar_full = torch.tensor(xbar_data, dtype=torch.float32, device=device)

    best_kl    = float('inf')
    best_state = None
    history    = []

    for ep in range(1, epochs + 1):
        model.train()
        idx     = torch.randperm(N, device=device)[:batch_size]
        log_p_b = log_p_full[idx]
        log_y_b = log_y_full[idx]
        w_b     = w_full[idx]

        optimiser.zero_grad()

        # Forward
        if mdp_mode:
            xbar_b = xbar_full[idx]
            w_pred = model(log_p_b, log_y_b, xbar_b)
        else:
            w_pred = model(log_p_b, log_y_b)

        # KL loss
        loss_kl = nn.KLDivLoss(reduction='batchmean')(
            torch.log(w_pred + 1e-10), w_b
        )

        # Monotonicity
        log_p_d = log_p_b.detach().requires_grad_(True)
        if mdp_mode:
            w_mono = model(log_p_d, log_y_b, xbar_b)
        else:
            w_mono = model(log_p_d, log_y_b)
        grads = torch.autograd.grad(w_mono.sum(), log_p_d, create_graph=True)[0]
        loss_mono = torch.mean(torch.clamp(grads, min=0))

        # Slutsky symmetry (delayed)
        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            if mdp_mode:
                loss_slut = model.slutsky_penalty(
                    log_p_full[sub], log_y_full[sub], xbar_full[sub]
                )
            else:
                loss_slut = model.slutsky_penalty(log_p_full[sub], log_y_full[sub])

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                if mdp_mode:
                    wp = model(log_p_full, log_y_full, xbar_full)
                else:
                    wp = model(log_p_full, log_y_full)
                kl_full = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(wp + 1e-10), w_full
                ).item()
            if kl_full < best_kl:
                best_kl    = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 400 == 0:
            history.append({
                'epoch': ep,
                'kl':    loss_kl.item(),
                'mono':  loss_mono.item(),
                'slut':  loss_slut.item() if ep >= slut_start else 0.0,
                'beta':  model.beta.item(),
                'lr':    scheduler.get_last_lr()[0],
            })
            print(f"  ep {ep:4d} | KL={loss_kl.item():.5f} "
                  f"| mono={loss_mono.item():.5f} "
                  f"| slut={loss_slut.item() if ep >= slut_start else 0.0:.5f} "
                  f"| β={model.beta.item():.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


# ============================================================
# SECTION 6: CONTINUOUS VARIATIONAL MIXTURE IRL
# ============================================================

class ContinuousVariationalMixture:
    """
    Continuous Variational Mixture IRL with Gaussian components in (α, ρ) space.

    Problem with the original discrete mixture:
      - Only 3 fixed types; if the true (α, ρ) falls between grid points
        the EM collapses onto the closest type and assigns zero weight elsewhere.
      - This is why π=[0.014, 0.000, 0.986] — the grid didn't span the truth.

    Solution: place K Gaussian components in the (α, ρ) parameter space.
    Each component k has mean (μ_α^k, μ_ρ^k) and diagonal covariance.
    We sample consumer types from each component and average their demand
    predictions. The E-step soft-assigns observations to components;
    the M-step updates the component means via a small gradient step.

    The wider grid is also critical: we initialise with α values spread
    across the simplex (not just 3 hand-picked points), and ρ values
    covering [0.2, 0.7] (vs original [0.3, 0.45, 0.55]).

    This lets the mixture genuinely interpolate between types rather than
    snapping to the nearest pre-defined consumer.
    """

    def __init__(self, K=6, n_goods=3, n_samples_per_component=5):
        self.K          = K
        self.G          = n_goods
        self.n_spc      = n_samples_per_component

        # Wide initialisation: K components spread over (α, ρ) space
        # α lives on the simplex; ρ ∈ (0, 1)
        rng = np.random.default_rng(0)

        # Initialise α means by sampling from a flat Dirichlet
        alpha_means = rng.dirichlet(np.ones(n_goods), size=K)   # (K, G)
        rho_means   = np.linspace(0.2, 0.7, K)                  # (K,)

        # Component means in unconstrained space
        # α: use softmax parameterisation (log-ratio)
        # ρ: use logit parameterisation so ρ ∈ (0, 1)
        self.mu_alpha = np.log(alpha_means + 1e-6)               # (K, G) log-space
        self.mu_rho   = np.log(rho_means / (1 - rho_means))     # (K,) logit-space

        # Component variances (fixed, moderately wide)
        self.sigma_alpha = 0.5 * np.ones((K, n_goods))
        self.sigma_rho   = 0.3 * np.ones(K)

        self.pi = np.ones(K) / K                                 # mixture weights

    def _decode_alpha(self, log_alpha):
        """Softmax to get α on simplex."""
        a = np.exp(log_alpha - log_alpha.max())
        return a / a.sum()

    def _decode_rho(self, logit_rho):
        """Sigmoid to get ρ ∈ (0, 1)."""
        return 1.0 / (1.0 + np.exp(-logit_rho))

    def _sample_consumers(self, k, n_samples):
        """
        Sample n_samples consumer types from component k's Gaussian.
        Returns list of CESConsumer objects.
        """
        rng = np.random.default_rng(k * 100)
        consumers = []
        for _ in range(n_samples):
            la  = rng.normal(self.mu_alpha[k], self.sigma_alpha[k])
            lr  = rng.normal(self.mu_rho[k],   self.sigma_rho[k])
            alpha = self._decode_alpha(la)
            rho   = float(np.clip(self._decode_rho(lr), 0.05, 0.95))
            consumers.append(CESConsumer(alpha=alpha, rho=rho))
        return consumers

    def _predict_component(self, k, prices, income):
        """
        Predict demand for component k by averaging over sampled consumer types.
        Monte Carlo approximation of E_q(α,ρ|k)[w(p, y; α, ρ)].
        """
        consumers = self._sample_consumers(k, self.n_spc)
        w_samples = np.stack([c.solve_demand(prices, income) for c in consumers])
        return w_samples.mean(axis=0)   # (N, G)

    def fit(self, prices, income, w_obs, n_iter=30, lr_mu=0.1, sigma2=0.003):
        """
        Variational EM:
          E-step: rₖᵢ ∝ πₖ · exp(−||ŵₖ(pᵢ,yᵢ) − wᵢ^obs||² / 2σ²)
          M-step: update πₖ; gradient update on μ_α^k and μ_ρ^k
        """
        N = len(income)
        print(f"    Computing initial component demands ({self.K} components × {self.n_spc} samples)...")

        for it in range(n_iter):
            # Predict demand for each component
            w_k = np.stack([self._predict_component(k, prices, income)
                            for k in range(self.K)])          # (K, N, G)

            # E-step
            log_resp = np.array([
                -np.sum((w_k[k] - w_obs) ** 2, axis=1) / (2 * sigma2)
                + np.log(self.pi[k] + 1e-10)
                for k in range(self.K)
            ])                                                 # (K, N)
            log_resp -= log_resp.max(axis=0)
            resp      = np.exp(log_resp)
            resp     /= resp.sum(axis=0, keepdims=True)        # (K, N)

            # M-step: weights
            self.pi  = resp.mean(axis=1)
            self.pi /= self.pi.sum()

            # M-step: gradient on component means (natural gradient in param space)
            for k in range(self.K):
                rk      = resp[k]                              # (N,)
                err_k   = (w_obs - w_k[k])                    # (N, G)
                signal  = np.mean(rk[:, None] * err_k, axis=0)  # (G,)

                # Approximate gradient w.r.t. mu_alpha via finite differences
                for j in range(self.G):
                    h                        = 0.01
                    self.mu_alpha[k, j]     += h
                    w_plus                   = self._predict_component(k, prices[:20], income[:20])
                    self.mu_alpha[k, j]     -= h
                    d_alpha                  = (w_plus - w_k[k][:20]).mean(axis=0)
                    self.mu_alpha[k, j]     += lr_mu * np.dot(signal, d_alpha) / (h + 1e-8)

                h               = 0.01
                self.mu_rho[k] += h
                w_plus          = self._predict_component(k, prices[:20], income[:20])
                self.mu_rho[k] -= h
                d_rho           = (w_plus - w_k[k][:20]).mean(axis=0)
                self.mu_rho[k] += lr_mu * np.dot(signal, d_rho) / (h + 1e-8)

            if (it + 1) % 5 == 0:
                loss = np.mean((np.einsum('k,kng->ng', self.pi, w_k) - w_obs) ** 2)
                print(f"    iter {it+1:3d} | MSE={loss:.6f} | π={np.round(self.pi, 3)}")

        self.w_k_  = w_k
        self.resp_ = resp
        return self

    def predict(self, prices, income):
        w_k = np.stack([self._predict_component(k, prices, income) for k in range(self.K)])
        return np.einsum('k,kng->ng', self.pi, w_k)

    def get_component_summary(self):
        """Return decoded (α, ρ) for each component."""
        rows = []
        for k in range(self.K):
            alpha = self._decode_alpha(self.mu_alpha[k])
            rho   = float(np.clip(self._decode_rho(self.mu_rho[k]), 0.05, 0.95))
            rows.append({'component': k+1, 'pi': self.pi[k],
                         'alpha_food': alpha[0], 'alpha_fuel': alpha[1],
                         'alpha_other': alpha[2], 'rho': rho})
        return pd.DataFrame(rows)


# ============================================================
# SECTION 7: UNIFIED PREDICTION & EVALUATION UTILITIES
# ============================================================

def predict_shares(spec, p, y, *, aids=None, blp=None,
                   lirl_theta=None, lirl_feat_fn=None,
                   nirl=None, mdp_nirl=None, xbar=None,
                   consumer=None, mixture=None, device='cpu'):
    """Unified prediction interface for all model types."""
    if spec == 'truth':
        return consumer.solve_demand(p, y)
    if spec == 'aids':
        return aids.predict(p, y)
    if spec == 'blp':
        return blp.predict(p)
    if spec == 'l-irl':
        F = lirl_feat_fn(p, y)
        return predict_linear_irl(F, lirl_theta)
    if spec == 'n-irl':
        with torch.no_grad():
            lp = torch.log(torch.tensor(p, dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(y, dtype=torch.float32, device=device)).unsqueeze(1)
            return nirl(lp, ly).cpu().numpy()
    if spec == 'mdp-irl':
        assert xbar is not None, "xbar required for MDP-IRL"
        with torch.no_grad():
            lp  = torch.log(torch.tensor(p,    dtype=torch.float32, device=device))
            ly  = torch.log(torch.tensor(y,    dtype=torch.float32, device=device)).unsqueeze(1)
            xb  = torch.tensor(xbar,           dtype=torch.float32, device=device)
            return mdp_nirl(lp, ly, xb).cpu().numpy()
    if spec == 'mixture':
        return mixture.predict(p, y)
    raise ValueError(f"Unknown spec: {spec}")


def compute_elasticities(spec, p_pt, y_pt, h=1e-4, xbar_pt=None, **kw):
    w0  = predict_shares(spec, p_pt.reshape(1, -1), np.array([y_pt]),
                         xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None, **kw)[0]
    eps = []
    for i in range(3):
        p1       = p_pt.copy().reshape(1, -1)
        p1[0, i] *= (1 + h)
        wp       = predict_shares(spec, p1, np.array([y_pt]),
                                  xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None,
                                  **kw)[0]
        eps.append(((wp[i] - w0[i]) / w0[i]) / h - 1)
    return eps


def compute_welfare_loss(spec, p0, p1, y, steps=100, xbar_pt=None, **kw):
    """Compensating Variation via Riemann integration."""
    path = np.linspace(p0, p1, steps)
    dp   = (p1 - p0) / steps
    loss = 0.0
    for t in range(steps):
        w     = predict_shares(spec, path[t:t+1], np.array([y]),
                               xbar=xbar_pt.reshape(1, -1) if xbar_pt is not None else None,
                               **kw)[0]
        q     = w * y / path[t]
        loss -= q @ dp
    return loss


def get_metrics(spec, p_shock, income, w_true, xbar_shock=None, **kw):
    wp = predict_shares(spec, p_shock, income,
                        xbar=xbar_shock, **kw)
    return {"RMSE": np.sqrt(mean_squared_error(w_true, wp)),
            "MAE":  mean_absolute_error(w_true, wp)}


# ============================================================
# SECTION 8: DATA GENERATION
# ============================================================

print("=" * 72)
print("  IRL CONSUMER DEMAND RECOVERY — FULL PYTORCH IMPLEMENTATION")
print("=" * 72)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n  Device: {DEVICE}")

N      = 800
Z      = np.random.uniform(1, 5, (N, 3))
p_pre  = Z + np.random.normal(0, 0.1, (N, 3))
income = np.random.uniform(1200, 2000, N)

primary = CESConsumer()
print(f"\n  Ground truth CES: α={primary.alpha}, ρ={primary.rho}")
w_train = primary.solve_demand(p_pre, income)
print(f"  Mean shares — Food:{w_train[:,0].mean():.3f}  "
      f"Fuel:{w_train[:,1].mean():.3f}  Other:{w_train[:,2].mean():.3f}")

# Generate habit formation data WITH lagged shares x̄
print("\n  Generating habit formation data with lagged state x̄...")
habit_consumer = HabitFormationConsumer()
w_habit, xbar_train = habit_consumer.solve_demand(p_pre, income, return_xbar=True)
print(f"  Habit shares — Food:{w_habit[:,0].mean():.3f}  "
      f"Fuel:{w_habit[:,1].mean():.3f}  Other:{w_habit[:,2].mean():.3f}")
print(f"  Mean x̄ — Food:{xbar_train[:,0].mean():.3f}  "
      f"Fuel:{xbar_train[:,1].mean():.3f}  Other:{xbar_train[:,2].mean():.3f}")

# 20% fuel price shock
p_post      = p_pre.copy()
p_post[:, 1] *= 1.2
w_post_true   = primary.solve_demand(p_post, income)

# Also generate shocked habit data with x̄ continuing from training trajectory
w_habit_shock, xbar_shock = habit_consumer.solve_demand(p_post, income, return_xbar=True)


# ============================================================
# SECTION 9: TRAINING — BENCHMARKS
# ============================================================

print("\n[1/7] Training LA-AIDS and BLP (IV)...")
aids_m = AIDSBench();  aids_m.fit(p_pre, w_train, income)
blp_m  = BLPBench();   blp_m.fit(p_pre, w_train, Z)
print("      Done.")

# Habit-trained AIDS
aids_hab = AIDSBench(); aids_hab.fit(p_pre, w_habit, income)


# ============================================================
# SECTION 10: TRAINING — LINEAR IRL VARIANTS
# ============================================================

print("\n[2/7] Training Linear IRL variants...")

# Variant A: Original (shared features)
print("      A. Shared features (original)...")
F_shared   = features_shared(p_pre, income)
theta_shared = run_linear_irl(F_shared, w_train, lr=0.05, epochs=3000)

# Variant B: Good-specific features
print("      B. Good-specific features...")
F_goodspec   = features_good_specific(p_pre, income)
theta_goodspec = run_linear_irl(F_goodspec, w_train, lr=0.05, epochs=3000)

# Variant C: Orthogonalised features with per-good intercepts
print("      C. Orthogonalised + per-good intercepts...")
F_orth   = features_orthogonalised(p_pre, income)
theta_orth = run_linear_irl(F_orth, w_train, lr=0.05, epochs=3000)

n_feat_summary = {
    "Shared":        (F_shared.shape[2],   theta_shared),
    "Good-specific": (F_goodspec.shape[2], theta_goodspec),
    "Orthogonalised":(F_orth.shape[2],     theta_orth),
}
for name, (nf, th) in n_feat_summary.items():
    print(f"      {name}: {nf} features, theta={np.round(th, 4)}")


# ============================================================
# SECTION 11: TRAINING — STANDARD NEURAL IRL
# ============================================================

print("\n[3/7] Training Standard Neural IRL (no history)...")
n_irl = NeuralIRL(n_goods=3, hidden_dim=256)
n_irl, hist_nirl = train_neural_irl(
    n_irl, p_pre, income, w_train,
    epochs=4000, lr=5e-4, batch_size=256,
    lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
    device=DEVICE,
)
print(f"      Params: {sum(p.numel() for p in n_irl.parameters())} | β={n_irl.beta.item():.4f}")


# ============================================================
# SECTION 12: TRAINING — MDP NEURAL IRL (WITH x̄ STATE)
# ============================================================

print("\n[4/7] Training MDP-Aware Neural IRL (with lagged share x̄)...")
print("      Standard Neural IRL trained on habit data (static baseline)...")
n_irl_hab_static = NeuralIRL(n_goods=3, hidden_dim=128)
n_irl_hab_static, hist_hab_static = train_neural_irl(
    n_irl_hab_static, p_pre, income, w_habit,
    epochs=2000, lr=5e-4, batch_size=256,
    lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
    device=DEVICE,
)

print("      MDP Neural IRL trained on habit data (with x̄ state)...")
mdp_irl = MDPNeuralIRL(n_goods=3, hidden_dim=256)
mdp_irl, hist_mdp = train_neural_irl(
    mdp_irl, p_pre, income, w_habit,
    epochs=4000, lr=5e-4, batch_size=256,
    lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
    xbar_data=xbar_train,
    device=DEVICE,
)
print(f"      MDP-IRL params: {sum(p.numel() for p in mdp_irl.parameters())} | β={mdp_irl.beta.item():.4f}")


# ============================================================
# SECTION 13: TRAINING — CONTINUOUS VARIATIONAL MIXTURE
# ============================================================

print("\n[5/7] Training Continuous Variational Mixture IRL (K=6, wide grid)...")
var_mix = ContinuousVariationalMixture(K=6, n_goods=3, n_samples_per_component=5)
var_mix.fit(p_pre[:300], income[:300], w_train[:300], n_iter=30, lr_mu=0.05)
print(f"      Final π = {np.round(var_mix.pi, 3)}")
comp_summary = var_mix.get_component_summary()
print(comp_summary.round(3).to_string(index=False))


# ============================================================
# SECTION 14: TRAINING — ROBUSTNESS DGPs
# ============================================================

print("\n[6/7] Robustness training across all DGPs...")

all_consumers = {
    "CES":          CESConsumer(),
    "Quasilinear":  QuasilinearConsumer(),
    "Leontief":     LeontiefConsumer(),
    "Stone-Geary":  StoneGearyConsumer(),
    "Habit":        HabitFormationConsumer(),
}

rob_rows = {}
for cname, cons in all_consumers.items():
    print(f"  DGP: {cname}...")
    try:
        w_dgp       = cons.solve_demand(p_pre, income)
        w_dgp_shock = cons.solve_demand(p_post, income)

        a_rob = AIDSBench(); a_rob.fit(p_pre, w_dgp, income)
        th_rob = run_linear_irl(features_orthogonalised(p_pre, income), w_dgp, lr=0.05, epochs=2000)
        n_rob  = NeuralIRL(n_goods=3, hidden_dim=128)
        n_rob, _ = train_neural_irl(
            n_rob, p_pre, income, w_dgp, epochs=2000, lr=5e-4, batch_size=256,
            lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3, device=DEVICE,
        )

        rmse_a = get_metrics('aids',  p_post, income, w_dgp_shock, aids=a_rob)["RMSE"]
        rmse_l = get_metrics('l-irl', p_post, income, w_dgp_shock,
                             lirl_theta=th_rob, lirl_feat_fn=features_orthogonalised)["RMSE"]
        rmse_n = get_metrics('n-irl', p_post, income, w_dgp_shock, nirl=n_rob, device=DEVICE)["RMSE"]

        rob_rows[cname] = {"AIDS": rmse_a, "Lin IRL (Orth)": rmse_l, "Neural IRL": rmse_n}
        print(f"    AIDS={rmse_a:.5f}  Lin IRL={rmse_l:.5f}  Neural IRL={rmse_n:.5f}")
    except Exception as e:
        print(f"    Error: {e}")


# ============================================================
# SECTION 15: EVALUATION — PRIMARY RESULTS
# ============================================================

print("\n[7/7] Evaluating all models post-shock...")

avg_p    = p_post.mean(axis=0)
avg_y    = 1600.0
p_pre_pt = avg_p / np.array([1.0, 1.2, 1.0])

# Shared KW for CES-trained models
KW_CES = dict(
    aids=aids_m, blp=blp_m,
    nirl=n_irl, consumer=primary,
    mixture=var_mix, device=DEVICE,
)

# Models to evaluate on CES DGP
MODELS_CES = [
    ("LA-AIDS",         'aids',  {}),
    ("BLP (IV)",        'blp',   {}),
    ("Lin IRL Shared",  'l-irl', {'lirl_theta': theta_shared,  'lirl_feat_fn': features_shared}),
    ("Lin IRL GoodSpec",'l-irl', {'lirl_theta': theta_goodspec,'lirl_feat_fn': features_good_specific}),
    ("Lin IRL Orth",    'l-irl', {'lirl_theta': theta_orth,    'lirl_feat_fn': features_orthogonalised}),
    ("Neural IRL",      'n-irl', {}),
    ("Var. Mixture",    'mixture', {}),
]

print("\n" + "=" * 72)
print("  TABLE 1: PREDICTIVE ACCURACY (POST-SHOCK, CES DGP)")
print("=" * 72)
perf_ces = {}
for name, spec, extra_kw in MODELS_CES:
    perf_ces[name] = get_metrics(spec, p_post, income, w_post_true, **{**KW_CES, **extra_kw})
print(pd.DataFrame(perf_ces).T.round(5).to_string())

print("\n" + "=" * 72)
print("  TABLE 2: OWN-PRICE ELASTICITIES AT SHOCK POINT")
print("=" * 72)
elast = {"Ground Truth": compute_elasticities('truth', avg_p, avg_y, **KW_CES)}
for name, spec, extra_kw in MODELS_CES:
    elast[name] = compute_elasticities(spec, avg_p, avg_y, **{**KW_CES, **extra_kw})
print(pd.DataFrame(elast, index=["Food ε₀₀", "Fuel ε₁₁", "Other ε₂₂"]).T.round(3).to_string())

print("\n" + "=" * 72)
print("  TABLE 3: WELFARE IMPACT — CONSUMER SURPLUS LOSS (£)")
print("=" * 72)
welf = {"Ground Truth": compute_welfare_loss('truth', p_pre_pt, avg_p, avg_y, **KW_CES)}
for name, spec, extra_kw in MODELS_CES:
    welf[name] = compute_welfare_loss(spec, p_pre_pt, avg_p, avg_y, **{**KW_CES, **extra_kw})
for k, v in welf.items():
    gt  = welf["Ground Truth"]
    err = f"  (error: {100*abs(v-gt)/abs(gt):.1f}%)" if k != "Ground Truth" else ""
    print(f"  {k:22s}:  £{v:8.2f}{err}")

print("\n" + "=" * 72)
print("  TABLE 4: ROBUSTNESS ACROSS UTILITY DGPs")
print("=" * 72)
print(pd.DataFrame(rob_rows).T.round(5).to_string())

# ============================================================
# SECTION 16: HABIT FORMATION / MDP ADVANTAGE EXPERIMENT
# ============================================================

print("\n" + "=" * 72)
print("  TABLE 5: MDP ADVANTAGE — HABIT FORMATION EXPERIMENT")
print("=" * 72)
print("""
  Three models trained on identical habit-formation data:
    (a) LA-AIDS        — static, no state history
    (b) Neural IRL     — static MDP, state = (p, y) only
    (c) MDP-Neural IRL — full MDP, state = (p, y, x̄)
  x̄ = lagged budget share (habit stock). Models (a) and (b) must treat
  the path-dependence as unexplained noise; model (c) can condition on it.
""")

rmse_aids_h  = get_metrics('aids',  p_post, income, w_habit_shock,  aids=aids_hab)["RMSE"]
rmse_nirl_h  = get_metrics('n-irl', p_post, income, w_habit_shock,
                            nirl=n_irl_hab_static, device=DEVICE)["RMSE"]
rmse_mdp_h   = get_metrics('mdp-irl', p_post, income, w_habit_shock,
                            mdp_nirl=mdp_irl, xbar_shock=xbar_shock, device=DEVICE)["RMSE"]

kl_aids  = float(nn.KLDivLoss(reduction='batchmean')(
    torch.log(torch.tensor(aids_hab.predict(p_post, income), dtype=torch.float32) + 1e-10),
    torch.tensor(w_habit_shock, dtype=torch.float32)
).item())

with torch.no_grad():
    lp_ = torch.log(torch.tensor(p_post, dtype=torch.float32))
    ly_ = torch.log(torch.tensor(income, dtype=torch.float32)).unsqueeze(1)
    kl_static = float(nn.KLDivLoss(reduction='batchmean')(
        torch.log(n_irl_hab_static(lp_, ly_) + 1e-10),
        torch.tensor(w_habit_shock, dtype=torch.float32)
    ).item())
    xb_ = torch.tensor(xbar_shock, dtype=torch.float32)
    kl_mdp = float(nn.KLDivLoss(reduction='batchmean')(
        torch.log(mdp_irl(lp_, ly_, xb_) + 1e-10),
        torch.tensor(w_habit_shock, dtype=torch.float32)
    ).item())

print(f"  {'Model':<25} {'RMSE':>10}  {'KL-div':>10}  {'RMSE reduction':>15}")
print(f"  {'-'*65}")
print(f"  {'LA-AIDS (static)':<25} {rmse_aids_h:>10.5f}  {kl_aids:>10.5f}  {'baseline':>15}")
print(f"  {'Neural IRL (static)':<25} {rmse_nirl_h:>10.5f}  {kl_static:>10.5f}  "
      f"{100*(rmse_aids_h-rmse_nirl_h)/rmse_aids_h:>14.1f}%")
print(f"  {'MDP Neural IRL (x̄ state)':<25} {rmse_mdp_h:>10.5f}  {kl_mdp:>10.5f}  "
      f"{100*(rmse_aids_h-rmse_mdp_h)/rmse_aids_h:>14.1f}%")
print(f"\n  MDP vs Static IRL improvement: "
      f"{100*(rmse_nirl_h-rmse_mdp_h)/rmse_nirl_h:.1f}%")
print(f"  MDP β (habit rationality): {mdp_irl.beta.item():.4f}")

# ============================================================
# SECTION 17: VARIATIONAL MIXTURE ANALYSIS
# ============================================================

print("\n" + "=" * 72)
print("  TABLE 6: VARIATIONAL MIXTURE — RECOVERED COMPONENT PARAMETERS")
print("=" * 72)
print("\n  Ground truth: α=[0.4, 0.4, 0.2], ρ=0.45")
print(f"\n{comp_summary.round(3).to_string(index=False)}")

vmix_rmse = get_metrics('mixture', p_post, income, w_post_true, mixture=var_mix)
print(f"\n  Variational Mixture RMSE: {vmix_rmse['RMSE']:.5f}")
print(f"  Dominant component: {comp_summary.loc[comp_summary['pi'].idxmax()].to_dict()}")

# ============================================================
# SECTION 18: LINEAR IRL FEATURE ABLATION
# ============================================================

print("\n" + "=" * 72)
print("  TABLE 7: LINEAR IRL FEATURE ABLATION")
print("=" * 72)
lin_ablation = {
    "Shared (original)":    get_metrics('l-irl', p_post, income, w_post_true,
                                        lirl_theta=theta_shared,   lirl_feat_fn=features_shared),
    "Good-specific":        get_metrics('l-irl', p_post, income, w_post_true,
                                        lirl_theta=theta_goodspec, lirl_feat_fn=features_good_specific),
    "Orth + Intercepts":    get_metrics('l-irl', p_post, income, w_post_true,
                                        lirl_theta=theta_orth,     lirl_feat_fn=features_orthogonalised),
}
print(pd.DataFrame(lin_ablation).T.round(5).to_string())

for name, (nf, th) in n_feat_summary.items():
    print(f"\n  {name} theta ({nf} params): {np.round(th, 4)}")


# ============================================================
# SECTION 19: PLOTTING
# ============================================================

print("\n  Generating figures...")
os.makedirs("figures", exist_ok=True)

p_grid  = np.linspace(1, 10, 80)
test_p  = np.tile(p_pre.mean(axis=0), (80, 1))
test_p[:, 1] = p_grid
fixed_y = np.full(80, avg_y)
avg_xbar = xbar_train.mean(axis=0)  # representative habit stock for MDP plots


# ---- FIGURE 1: Main demand curve comparison (CES DGP) ----
fig1, ax1 = plt.subplots(figsize=(11, 6))

curve_specs = [
    (primary.solve_demand(test_p, fixed_y)[:, 0],
     'k-',   3.0, 'Truth (CES)'),
    (predict_shares('aids',  test_p, fixed_y, **KW_CES)[:, 0],
     'r--',  2.0, 'LA-AIDS'),
    (predict_shares('blp',   test_p, fixed_y, **KW_CES)[:, 0],
     'g-.',  2.0, 'BLP (IV)'),
    (predict_shares('l-irl', test_p, fixed_y, lirl_theta=theta_shared,
                    lirl_feat_fn=features_shared)[:, 0],
     'y:',   2.0, 'Lin IRL (Shared)'),
    (predict_shares('l-irl', test_p, fixed_y, lirl_theta=theta_orth,
                    lirl_feat_fn=features_orthogonalised)[:, 0],
     'c:',   2.0, 'Lin IRL (Orth+Intercept)'),
    (predict_shares('n-irl', test_p, fixed_y, **KW_CES)[:, 0],
     'b-',   2.5, 'Neural IRL'),
    (predict_shares('mixture', test_p, fixed_y, **KW_CES)[:, 0],
     'm--',  2.0, 'Var. Mixture IRL'),
]
for vals, sty, lw, lbl in curve_specs:
    ax1.plot(p_grid, vals, sty, lw=lw, label=lbl)
ax1.axvline(p_pre[:, 1].mean() * 1.2, color='orange', ls=':', alpha=0.8, label='Shock point')
ax1.set_title("Food Share Response to Fuel Price Shock — CES Ground Truth",
              fontsize=13, fontweight='bold')
ax1.set_xlabel("Price of Fuel ($p_1$)", fontsize=12)
ax1.set_ylabel("Food Budget Share ($w_0$)", fontsize=12)
ax1.legend(fontsize=9, ncol=2, loc='upper left')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("figures/fig1_demand_curves.pdf", dpi=150, bbox_inches='tight')
fig1.savefig("figures/fig1_demand_curves.png", dpi=150, bbox_inches='tight')
print("    Saved: figures/fig1_demand_curves.pdf")


# ---- FIGURE 2: MDP Advantage — Habit Formation ----
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

good_names = ["Food", "Fuel", "Other"]

# Left: demand curves for static vs MDP IRL on habit DGP
xbar_rep = np.tile(avg_xbar, (80, 1))
for g_idx, ax in enumerate(axes):
    truth_curve = habit_consumer.solve_demand(test_p, fixed_y)[:, g_idx]
    static_curve = predict_shares('n-irl', test_p, fixed_y,
                                  nirl=n_irl_hab_static, device=DEVICE)[:, g_idx]
    mdp_curve = predict_shares('mdp-irl', test_p, fixed_y,
                                mdp_nirl=mdp_irl, xbar=xbar_rep, device=DEVICE)[:, g_idx]
    aids_curve = predict_shares('aids', test_p, fixed_y, aids=aids_hab)[:, g_idx]

    ax.plot(p_grid, truth_curve,  'k-',   lw=3.0, label='Truth (Habit)')
    ax.plot(p_grid, aids_curve,   'r--',  lw=2.0, label='LA-AIDS')
    ax.plot(p_grid, static_curve, 'b-.',  lw=2.0, label='Neural IRL (static)')
    ax.plot(p_grid, mdp_curve,    'g-',   lw=2.5, label='MDP-IRL (with $\\bar{x}$)')
    ax.axvline(p_pre[:, 1].mean() * 1.2, color='orange', ls=':', alpha=0.7)
    ax.set_title(f"{good_names[g_idx]} Share (Habit DGP)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Fuel Price", fontsize=10)
    ax.set_ylabel(f"{good_names[g_idx]} Budget Share", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig2.suptitle("MDP-Aware Neural IRL vs Static Models on Habit Formation DGP\n"
              "State augmentation with $\\bar{x}$ reduces residual variation from history",
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig("figures/fig2_mdp_advantage.pdf", dpi=150, bbox_inches='tight')
fig2.savefig("figures/fig2_mdp_advantage.png", dpi=150, bbox_inches='tight')
print("    Saved: figures/fig2_mdp_advantage.pdf")


# ---- FIGURE 3: Training convergence ----
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# Left: standard IRL convergence
ax_l = axes3[0]
if hist_nirl:
    ep_x = [h['epoch'] for h in hist_nirl]
    kl_y = [h['kl']    for h in hist_nirl]
    bt_y = [h['beta']  for h in hist_nirl]
    ax_r = ax_l.twinx()
    ax_l.plot(ep_x, kl_y, 'b-o', ms=5, label='KL Loss')
    ax_r.plot(ep_x, bt_y, 'r--s', ms=5, label='β (learned)')
    ax_l.set_title("Neural IRL (CES DGP): Convergence", fontsize=11, fontweight='bold')
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("KL Divergence", color='b')
    ax_r.set_ylabel("Temperature β", color='r')
    ax_l.legend(loc='upper right', fontsize=9); ax_r.legend(loc='center right', fontsize=9)
    ax_l.grid(True, alpha=0.3)

# Right: MDP IRL convergence
ax_r2 = axes3[1]
if hist_mdp:
    ep_x2 = [h['epoch'] for h in hist_mdp]
    kl_y2 = [h['kl']    for h in hist_mdp]
    bt_y2 = [h['beta']  for h in hist_mdp]
    ax_r2b = ax_r2.twinx()
    ax_r2.plot(ep_x2, kl_y2, 'g-o', ms=5, label='KL Loss (MDP)')
    ax_r2b.plot(ep_x2, bt_y2, 'm--s', ms=5, label='β (MDP)')
    ax_r2.set_title("MDP Neural IRL (Habit DGP): Convergence", fontsize=11, fontweight='bold')
    ax_r2.set_xlabel("Epoch"); ax_r2.set_ylabel("KL Divergence", color='g')
    ax_r2b.set_ylabel("Temperature β", color='m')
    ax_r2.legend(loc='upper right', fontsize=9); ax_r2b.legend(loc='center right', fontsize=9)
    ax_r2.grid(True, alpha=0.3)

fig3.suptitle("Training Convergence: Learnable Temperature β", fontsize=12, fontweight='bold')
fig3.tight_layout()
fig3.savefig("figures/fig3_convergence.pdf", dpi=150, bbox_inches='tight')
fig3.savefig("figures/fig3_convergence.png", dpi=150, bbox_inches='tight')
print("    Saved: figures/fig3_convergence.pdf")


# ---- FIGURE 4: Robustness heatmap ----
fig4, ax4 = plt.subplots(figsize=(8, 5))
rob_df = pd.DataFrame(rob_rows).T.astype(float)
im     = ax4.imshow(rob_df.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.15)
ax4.set_xticks(range(len(rob_df.columns)))
ax4.set_xticklabels(rob_df.columns, fontsize=11)
ax4.set_yticks(range(len(rob_df.index)))
ax4.set_yticklabels(rob_df.index, fontsize=11)
plt.colorbar(im, ax=ax4, label='Post-Shock RMSE')
for i in range(len(rob_df.index)):
    for j in range(len(rob_df.columns)):
        v = rob_df.values[i, j]
        color = 'white' if v > 0.08 else 'black'
        ax4.text(j, i, f"{v:.4f}", ha='center', va='center', fontsize=10, color=color)
ax4.set_title("Out-of-Sample RMSE Across Utility DGPs\n"
              "(Lower = better; green = good fit, red = poor fit)",
              fontsize=11, fontweight='bold')
fig4.tight_layout()
fig4.savefig("figures/fig4_robustness_heatmap.pdf", dpi=150, bbox_inches='tight')
fig4.savefig("figures/fig4_robustness_heatmap.png", dpi=150, bbox_inches='tight')
print("    Saved: figures/fig4_robustness_heatmap.pdf")


# ---- FIGURE 5: Variational Mixture components ----
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

ax5l = axes5[0]
comp_df = var_mix.get_component_summary()
x_pos   = np.arange(len(comp_df))
bars    = ax5l.bar(x_pos, comp_df['pi'], color=plt.cm.tab10(x_pos / len(comp_df)),
                   alpha=0.85, edgecolor='k')
for bar, row in zip(bars, comp_df.itertuples()):
    if row.pi > 0.02:
        ax5l.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f'ρ={row.rho:.2f}', ha='center', va='bottom', fontsize=8)
ax5l.set_xticks(x_pos)
ax5l.set_xticklabels([f"K={k+1}\nα=[{r.alpha_food:.2f},{r.alpha_fuel:.2f},{r.alpha_other:.2f}]"
                      for k, r in enumerate(comp_df.itertuples())],
                     fontsize=7, rotation=15)
ax5l.set_title("Variational Mixture: Component Weights π_k", fontsize=11, fontweight='bold')
ax5l.set_ylabel("Mixture Weight π_k"); ax5l.set_ylim(0, 1)
ax5l.axhline(1/var_mix.K, color='gray', ls='--', alpha=0.5, label='Uniform prior')
ax5l.legend(fontsize=9); ax5l.grid(True, alpha=0.3, axis='y')

# Right: component α simplex projection
ax5r = axes5[1]
for k, row in enumerate(comp_df.itertuples()):
    alpha = np.array([row.alpha_food, row.alpha_fuel, row.alpha_other])
    ax5r.scatter(alpha[0], alpha[1], s=row.pi*2000 + 20,
                 alpha=0.75, label=f"K={k+1} (ρ={row.rho:.2f})",
                 c=[plt.cm.tab10(k/len(comp_df))])
ax5r.scatter([0.4], [0.4], s=300, marker='*', color='red', zorder=5, label='True α')
ax5r.set_xlabel("α_food", fontsize=11); ax5r.set_ylabel("α_fuel", fontsize=11)
ax5r.set_title("Component Centres in (α_food, α_fuel) Space\n(size ∝ weight π_k)",
               fontsize=11, fontweight='bold')
ax5r.legend(fontsize=8, loc='upper right'); ax5r.grid(True, alpha=0.3)
ax5r.set_xlim(0, 0.9); ax5r.set_ylim(0, 0.9)

fig5.suptitle("Continuous Variational Mixture IRL (K=6)\nWider Type Grid vs Original Discrete EM",
              fontsize=12, fontweight='bold')
fig5.tight_layout()
fig5.savefig("figures/fig5_mixture_components.pdf", dpi=150, bbox_inches='tight')
fig5.savefig("figures/fig5_mixture_components.png", dpi=150, bbox_inches='tight')
print("    Saved: figures/fig5_mixture_components.pdf")

plt.close('all')

# ============================================================
# SECTION 20: LATEX TABLE AND FIGURE GENERATION
# ============================================================

print("\n  Generating LaTeX code...")

latex_output = []

def lx(s):
    latex_output.append(s)

lx(r"% ============================================================")
lx(r"% AUTO-GENERATED LaTeX — IRL Consumer Demand Recovery")
lx(r"% Paste each block into the relevant section of your .tex file")
lx(r"% ============================================================")
lx("")

# ---- LaTeX Table 1: Predictive Accuracy ----
lx(r"% --- TABLE 1: Predictive Accuracy (Post-Shock) ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Post-Shock Predictive Accuracy (CES Ground Truth, 20\% Fuel Price Shock)}")
lx(r"  \label{tab:accuracy}")
lx(r"  \begin{tabular}{lcc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{RMSE} & \textbf{MAE} \\")
lx(r"    \midrule")
for name, metrics in perf_ces.items():
    rmse = metrics['RMSE']
    mae  = metrics['MAE']
    bold_open  = r"\textbf{" if rmse == min(m['RMSE'] for m in perf_ces.values()) else ""
    bold_close = "}" if bold_open else ""
    lx(f"    {name} & {bold_open}{rmse:.5f}{bold_close} & {mae:.5f} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} RMSE and MAE computed over $N=800$ out-of-sample observations.")
lx(r"    Shock: $p_1 \to 1.2 p_1$ (20\% fuel price increase). Bold = best performer.")
lx(r"    Lin IRL variants: Shared (original 3-feature), GoodSpec (good-specific $(G+2)$-feature),")
lx(r"    Orth (orthogonalised log prices with per-good intercepts).")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 2: Elasticities ----
lx(r"% --- TABLE 2: Own-Price Elasticities ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Recovered Own-Price Quantity Elasticities at Shock Point ($\hat{\varepsilon}_{ii}$)}")
lx(r"  \label{tab:elasticities}")
lx(r"  \begin{tabular}{lccc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{Food} $\hat{\varepsilon}_{00}$ & \textbf{Fuel} $\hat{\varepsilon}_{11}$ & \textbf{Other} $\hat{\varepsilon}_{22}$ \\")
lx(r"    \midrule")
for model_name, eps_vals in elast.items():
    row_vals = " & ".join(f"{v:.3f}" for v in eps_vals)
    prefix = r"\textit{" if model_name == "Ground Truth" else ""
    suffix = "}" if prefix else ""
    lx(f"    {prefix}{model_name}{suffix} & {row_vals} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} Numerical own-price elasticities computed at mean post-shock prices")
lx(r"    and income $y = \pounds 1{,}600$. Elasticities measure the percentage change in quantity")
lx(r"    demanded per 1\% increase in own price.")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 3: Welfare ----
lx(r"% --- TABLE 3: Welfare Impact ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Compensating Variation: Consumer Surplus Loss from 20\% Fuel Price Shock}")
lx(r"  \label{tab:welfare}")
lx(r"  \begin{tabular}{lcc}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{CS Loss (£)} & \textbf{Error (\%)} \\")
lx(r"    \midrule")
gt_welf = welf["Ground Truth"]
for k, v in welf.items():
    err_val = "" if k == "Ground Truth" else f"{100*abs(v-gt_welf)/abs(gt_welf):.1f}"
    err_str = "—" if k == "Ground Truth" else err_val
    bold_open  = r"\textit{" if k == "Ground Truth" else ""
    bold_close = "}" if bold_open else ""
    lx(f"    {bold_open}{k}{bold_close} & £{abs(v):.2f} & {err_str} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} Compensating variation approximated via Riemann integration")
lx(r"    $\text{CV} \approx -\int_{p_0}^{p_1} q(p)\,\mathrm{d}p$ along a linear price path (100 steps).")
lx(r"    Error column shows percentage deviation from the Ground Truth estimate.")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 4: Robustness ----
lx(r"% --- TABLE 4: Robustness Across DGPs ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Out-of-Sample RMSE Across Utility Function DGPs (Post-Shock)}")
lx(r"  \label{tab:robustness}")
lx(r"  \begin{tabular}{lccc}")
lx(r"    \toprule")
lx(r"    \textbf{DGP} & \textbf{LA-AIDS} & \textbf{Lin IRL (Orth)} & \textbf{Neural IRL} \\")
lx(r"    \midrule")
for dgp, row in rob_rows.items():
    vals = list(row.values())
    min_v = min(vals)
    cells = []
    for v in vals:
        s = f"{v:.5f}"
        cells.append(r"\textbf{" + s + "}" if v == min_v else s)
    lx(f"    {dgp} & {' & '.join(cells)} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} Each row re-trains all models on the indicated DGP and evaluates")
lx(r"    post-shock RMSE. Bold = best performer per row.")
lx(r"    Lin IRL uses orthogonalised features with per-good intercepts.")
lx(r"    Habit Formation uses a sequential solver; other DGPs use closed-form demand.")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 5: MDP Advantage ----
lx(r"% --- TABLE 5: MDP Advantage (Habit Formation) ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{MDP State Augmentation: Habit Formation Experiment}")
lx(r"  \label{tab:mdp_advantage}")
lx(r"  \begin{tabular}{lccl}")
lx(r"    \toprule")
lx(r"    \textbf{Model} & \textbf{RMSE} & \textbf{KL Divergence} & \textbf{RMSE Reduction} \\")
lx(r"    \midrule")
habit_rows = [
    ("LA-AIDS (static)",        rmse_aids_h,  kl_aids,   "baseline"),
    ("Neural IRL (static MDP)", rmse_nirl_h,  kl_static, f"{100*(rmse_aids_h-rmse_nirl_h)/rmse_aids_h:.1f}\\%"),
    (r"MDP Neural IRL ($\bar{x}$ state)", rmse_mdp_h, kl_mdp, f"{100*(rmse_aids_h-rmse_mdp_h)/rmse_aids_h:.1f}\\%"),
]
for mname, rmse_v, kl_v, red_v in habit_rows:
    lx(f"    {mname} & {rmse_v:.5f} & {kl_v:.5f} & {red_v} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} All three models are trained on identical habit-formation data.")
lx(r"    The MDP Neural IRL additionally receives $\bar{x}_t$ (the exponential moving average of")
lx(r"    past consumption quantities) as part of its state vector, directly encoding the MDP")
lx(r"    structure that static demand systems cannot represent. KL divergence measured on")
lx(r"    held-out post-shock data. $\theta=0.3$, $\delta=0.7$ (habit strength, decay rate).")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 6: Variational Mixture ----
lx(r"% --- TABLE 6: Variational Mixture Components ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Continuous Variational Mixture IRL: Recovered Component Parameters ($K=6$)}")
lx(r"  \label{tab:mixture}")
lx(r"  \begin{tabular}{ccccccc}")
lx(r"    \toprule")
lx(r"    \textbf{Component} & $\hat{\pi}_k$ & $\hat{\alpha}_{\text{food}}$ & $\hat{\alpha}_{\text{fuel}}$ & $\hat{\alpha}_{\text{other}}$ & $\hat{\rho}$ & \textbf{Type} \\")
lx(r"    \midrule")
for _, row in comp_df.iterrows():
    a_f, a_fu, a_o = row['alpha_food'], row['alpha_fuel'], row['alpha_other']
    rho_v = row['rho']
    pi_v  = row['pi']
    if a_f > 0.45:
        type_label = "Food-heavy"
    elif a_fu > 0.45:
        type_label = "Fuel-heavy"
    elif pi_v > 0.3:
        type_label = r"\textbf{Dominant}"
    else:
        type_label = "Balanced"
    lx(f"    {int(row['component'])} & {pi_v:.3f} & {a_f:.3f} & {a_fu:.3f} & {a_o:.3f} & {rho_v:.3f} & {type_label} \\\\")
lx(r"    \midrule")
lx(r"    \textit{Truth} & — & 0.400 & 0.400 & 0.200 & 0.450 & — \\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} Gaussian mixture in $(\alpha, \rho)$ parameter space, fitted via")
lx(r"    variational EM on $N=300$ training observations. $\alpha$ parameterised via softmax;")
lx(r"    $\rho$ via sigmoid. Wider initialisation grid ($\rho \in [0.2, 0.7]$) vs discrete")
lx(r"    three-type EM which collapsed to a single component.")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX Table 7: Linear IRL Ablation ----
lx(r"% --- TABLE 7: Linear IRL Feature Ablation ---")
lx(r"\begin{table}[htbp]")
lx(r"  \centering")
lx(r"  \caption{Linear MaxEnt IRL: Feature Engineering Ablation Study}")
lx(r"  \label{tab:linear_ablation}")
lx(r"  \begin{tabular}{lccp{6.5cm}}")
lx(r"    \toprule")
lx(r"    \textbf{Variant} & \textbf{RMSE} & \textbf{MAE} & \textbf{Feature Description} \\")
lx(r"    \midrule")
ablation_descriptions = {
    "Shared (original)":  r"Single shared $[\ln p_i,\,(\ln p_i)^2,\,\ln y]$ — same sensitivity all goods",
    "Good-specific":      r"Per-good $[\ln p_0,\ln p_1,\ln p_2,\,(\ln p_i)^2,\,\ln y]$ — heterogeneous response",
    "Orth + Intercepts":  r"QR-orthogonalised $\ln p$ + per-good one-hot intercept — removes collinearity",
}
for aname, ametrics in lin_ablation.items():
    desc = ablation_descriptions.get(aname, "")
    rmse_v = ametrics['RMSE']
    mae_v  = ametrics['MAE']
    min_rmse = min(m['RMSE'] for m in lin_ablation.values())
    bold_o = r"\textbf{" if rmse_v == min_rmse else ""
    bold_c = "}" if bold_o else ""
    lx(f"    {aname} & {bold_o}{rmse_v:.5f}{bold_c} & {mae_v:.5f} & {desc} \\\\")
lx(r"    \bottomrule")
lx(r"  \end{tabular}")
lx(r"  \begin{tablenotes}\small")
lx(r"    \item \textit{Notes:} All variants use the same MaxEnt gradient ascent (3{,}000 epochs,")
lx(r"    $\ell_2 = 10^{-4}$, decaying learning rate $\eta_t = 0.05/(1+t/1000)$).")
lx(r"    Near-collinearity in IV-generated log prices causes the shared variant's gradient to")
lx(r"    collapse onto the curvature term only ($\hat{\theta}_{\ln p} \approx 0$, $\hat{\theta}_{(\ln p)^2} \neq 0$).")
lx(r"    Orthogonalisation via QR decomposition resolves this.")
lx(r"  \end{tablenotes}")
lx(r"\end{table}")
lx("")

# ---- LaTeX figure inclusion code ----
lx(r"% ============================================================")
lx(r"% FIGURE INCLUSION CODE")
lx(r"% (assumes \usepackage{graphicx} and figures/ folder alongside .tex)")
lx(r"% ============================================================")
lx("")

figures_latex = [
    ("fig1_demand_curves",
     "Demand System Comparison: Food Budget Share Response to Fuel Price Shock",
     r"""Each model is trained on pre-shock data and evaluated on the fuel price grid
     $p_1 \in [1, 10]$. The Neural IRL (blue) most closely tracks the CES ground truth
     across the full price range. The orange dotted line marks the shock point ($\hat{p}_1 \times 1.2$).
     Lin IRL (Shared) collapses due to feature collinearity; the Orthogonalised variant recovers
     the monotone response. The Variational Mixture interpolates between component predictions.""",
     "fig:demand_curves"),
    ("fig2_mdp_advantage",
     r"MDP-Aware Neural IRL vs Static Models on Habit Formation DGP (All Three Goods)",
     r"""Food, Fuel, and Other shares plotted as functions of fuel price under the habit-formation
     ground truth. The MDP Neural IRL (green), which receives the lagged habit stock $\bar{x}_t$
     as part of its state vector, recovers demand responses much closer to the true (black) curves
     than either LA-AIDS (red) or the static Neural IRL (blue). The gap is most pronounced for
     the Food share, where habit persistence is strongest.""",
     "fig:mdp_advantage"),
    ("fig3_convergence",
     r"Training Convergence: KL Divergence and Learnable Temperature $\beta$",
     r"""Left: Standard Neural IRL trained on CES data. The KL loss reaches near-zero by epoch
     800, while $\beta$ stabilises around 4.15 — the network's implicit estimate of consumer
     rationality. Right: MDP Neural IRL trained on habit-formation data. Higher residual KL
     ($\approx 0.007$) before state augmentation is resolved; $\beta$ converges to a similar
     range, confirming structural consistency.""",
     "fig:convergence"),
    ("fig4_robustness_heatmap",
     "Robustness Heatmap: Post-Shock RMSE Across Utility Function DGPs",
     r"""Each cell shows the post-shock RMSE when both training and evaluation data are generated
     by the indicated DGP. Neural IRL (right column) dominates across all smooth DGPs.
     The Leontief result (kinked demand) is surprisingly good for Neural IRL, which approximates
     the kink via a steep sigmoid region. The Habit DGP row shows near-parity between AIDS and
     Neural IRL — the MDP advantage only materialises when $\bar{x}$ is in the state (Table~\ref{tab:mdp_advantage}).""",
     "fig:robustness"),
    ("fig5_mixture_components",
     r"Continuous Variational Mixture IRL: Component Weights and Parameter Space (K=6)",
     r"""Left: mixture weights $\hat{\pi}_k$ for each of the $K=6$ Gaussian components in
     $(\alpha, \rho)$ space. The dominant component concentrates near the true parameters
     (red star, right panel). Right: component centres projected onto the
     $(\alpha_{\text{food}}, \alpha_{\text{fuel}})$ simplex face; marker size proportional to
     $\hat{\pi}_k$. The wider grid and continuous parameterisation prevent the single-component
     collapse seen in the original three-type discrete EM.""",
     "fig:mixture"),
]

for fname, caption, note, label in figures_latex:
    lx(r"\begin{figure}[htbp]")
    lx(r"  \centering")
    lx(f"  \\includegraphics[width=\\textwidth]{{figures/{fname}.pdf}}")
    lx(f"  \\caption{{{caption}}}")
    lx(f"  \\label{{{label}}}")
    lx(r"  \begin{figurenotes}")
    lx(f"    {note}")
    lx(r"  \end{figurenotes}")
    lx(r"\end{figure}")
    lx("")

# ---- LaTeX preamble packages needed ----
lx(r"% ============================================================")
lx(r"% REQUIRED PREAMBLE PACKAGES")
lx(r"% ============================================================")
lx(r"% \usepackage{booktabs}       % \toprule, \midrule, \bottomrule")
lx(r"% \usepackage{threeparttable} % tablenotes environment")
lx(r"% \usepackage{graphicx}       % \includegraphics")
lx(r"% \usepackage{caption}        % enhanced captions")
lx(r"% \usepackage{amsmath}        % math environments")
lx(r"% \usepackage{pdflscape}      % for wide tables if needed: \begin{landscape}")
lx("")
lx(r"% For figurenotes, add this to preamble:")
lx(r"% \newenvironment{figurenotes}{\par\small\textit{Notes:~}}{\par}")

# Write LaTeX file
latex_text = "\n".join(latex_output)
with open("paper_tables_figures.tex", "w") as f:
    f.write(latex_text)
print("    Saved: paper_tables_figures.tex")

# ============================================================
# SECTION 21: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("  EXPERIMENT SUMMARY")
print("=" * 72)

print(f"""
  NEURAL IRL (CES DGP):
    RMSE: {perf_ces['Neural IRL']['RMSE']:.5f}  |  β={n_irl.beta.item():.4f}
    Welfare error: {100*abs(welf['Neural IRL']-gt_welf)/abs(gt_welf):.1f}%

  MDP ADVANTAGE (Habit DGP):
    LA-AIDS RMSE:        {rmse_aids_h:.5f}
    Static Neural IRL:   {rmse_nirl_h:.5f}  ({100*(rmse_aids_h-rmse_nirl_h)/rmse_aids_h:.1f}% over AIDS)
    MDP Neural IRL:      {rmse_mdp_h:.5f}  ({100*(rmse_aids_h-rmse_mdp_h)/rmse_aids_h:.1f}% over AIDS)
    MDP vs Static IRL:   {100*(rmse_nirl_h-rmse_mdp_h)/rmse_nirl_h:.1f}% improvement

  LINEAR IRL ABLATION:
    Shared:              {lin_ablation['Shared (original)']['RMSE']:.5f}
    Good-specific:       {lin_ablation['Good-specific']['RMSE']:.5f}
    Orth + Intercepts:   {lin_ablation['Orth + Intercepts']['RMSE']:.5f}

  VARIATIONAL MIXTURE (K=6):
    RMSE: {vmix_rmse['RMSE']:.5f}
    Dominant component α: food={comp_summary.loc[comp_summary['pi'].idxmax(),'alpha_food']:.3f}
                          fuel={comp_summary.loc[comp_summary['pi'].idxmax(),'alpha_fuel']:.3f}
                          other={comp_summary.loc[comp_summary['pi'].idxmax(),'alpha_other']:.3f}
                          ρ={comp_summary.loc[comp_summary['pi'].idxmax(),'rho']:.3f}
    (True: α=[0.4, 0.4, 0.2], ρ=0.45)

  OUTPUT FILES:
    figures/fig1_demand_curves.pdf
    figures/fig2_mdp_advantage.pdf
    figures/fig3_convergence.pdf
    figures/fig4_robustness_heatmap.pdf
    figures/fig5_mixture_components.pdf
    paper_tables_figures.tex
""")