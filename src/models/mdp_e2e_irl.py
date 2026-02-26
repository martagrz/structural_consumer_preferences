"""
End-to-end MDP Neural IRL and Window IRL models.

Two new sequential demand models:

1. MDPNeuralIRL_E2E
   Like MDPNeuralIRL but the habit stock x̄ is *computed on-the-fly* each
   training epoch using the model's own learned δ, so δ is recovered from
   the data rather than being a tuning constant.

2. WindowIRL
   Free-form lookback model: conditions on the last L (log-price, log-quantity)
   pairs plus the current (log-price, log-income).  No parametric assumption on
   how history enters — the network learns any persistent pattern in the data.

Training helpers
----------------
compute_xbar_e2e   — sequential EWMA with gradient through δ
build_window_features — (N, in_dim) feature matrix for WindowIRL
train_mdp_e2e      — epoch loop for MDPNeuralIRL_E2E
train_window_irl   — epoch loop for WindowIRL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: sequential x̄ computation with differentiable δ
# ─────────────────────────────────────────────────────────────────────────────

def compute_xbar_e2e(delta, log_q_tensor, store_ids=None, init_val=None):
    """Compute the habit stock x̄ for all N observations, keeping the
    computation graph live so that gradients flow back to *delta*.

    The update rule is
        x̄_0  = init_val
        x̄_t  = δ · x̄_{t-1} + (1−δ) · q_{t-1}
    which matches the HabitFormationConsumer DGP (decay applied to the
    running average, not to individual observations).

    Parameters
    ----------
    delta        : scalar Tensor — learned decay ∈ (0, 1), from sigmoid.
    log_q_tensor : Tensor (N, G) — log-quantities (or log-shares) in
                   *sequential / temporal order*.
    store_ids    : array-like (N,) of int — resets x̄ at store boundaries.
                   Pass None for a single continuous sequence (simulation).
    init_val     : Tensor (G,) — initial habit; defaults to mean of log_q.

    Returns
    -------
    xbar : Tensor (N, G) with computation graph through delta.
    """
    N, G = log_q_tensor.shape
    if init_val is None:
        init_val = log_q_tensor.detach().mean(0)   # (G,) — no grad needed for init

    xb = init_val.clone()
    xbar_list: list = []

    for i in range(N):
        # ── Reset at store boundary ──────────────────────────────────────────
        if store_ids is not None and i > 0 and store_ids[i] != store_ids[i - 1]:
            xb = init_val.clone()

        xbar_list.append(xb)                                    # record x̄ entering t
        xb = delta * xb + (1.0 - delta) * log_q_tensor[i]      # EWMA update

    return torch.stack(xbar_list)  # (N, G)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: window feature matrix for WindowIRL
# ─────────────────────────────────────────────────────────────────────────────

def build_window_features(log_p_all, log_y_all, log_q_all,
                          window: int, store_ids=None):
    """Build the (N, in_dim) feature matrix for WindowIRL.

    For each observation i the feature vector is:
        [log_p_i, log_y_i,
         log_p_{i-1}, log_q_{i-1},
         log_p_{i-2}, log_q_{i-2},
         ...
         log_p_{i-L}, log_q_{i-L}]

    Missing history (store boundaries or beginning of sequence) is filled
    with the column-wise global mean of log_p / log_q.

    Parameters
    ----------
    log_p_all  : (N, G) ndarray — log prices in sequential order.
    log_y_all  : (N,)   ndarray — log incomes.
    log_q_all  : (N, G) ndarray — log quantities (or log-shares).
    window     : int — number of lag periods L.
    store_ids  : (N,) array or None — resets history at boundaries.

    Returns
    -------
    features : (N, G + 1 + window * 2 * G) ndarray
    """
    N, G = log_p_all.shape
    init_p = log_p_all.mean(0)    # (G,) fallback for missing history
    init_q = log_q_all.mean(0)    # (G,)
    out_dim = G + 1 + window * 2 * G
    features = np.zeros((N, out_dim), dtype=np.float32)

    for i in range(N):
        parts = [log_p_all[i], np.array([log_y_all[i]])]
        for lag in range(1, window + 1):
            j = i - lag
            if (j >= 0
                    and (store_ids is None or store_ids[j] == store_ids[i])):
                parts.extend([log_p_all[j], log_q_all[j]])
            else:
                parts.extend([init_p, init_q])
        features[i] = np.concatenate(parts)

    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Model 1: MDPNeuralIRL_E2E
# ─────────────────────────────────────────────────────────────────────────────

class MDPNeuralIRL_E2E(nn.Module):
    """MDP Neural IRL with fully end-to-end learned habit decay.

    Unlike MDPNeuralIRL (which receives a habit stock pre-built with a *fixed*
    δ), this model computes x̄ at each training epoch by calling
    ``compute_xbar_e2e`` with its own learned δ.  The gradient of the training
    loss w.r.t. δ therefore flows through the x̄ computation, allowing the
    model to *recover the true decay rate* from sequential consumption data.

    Forward interface: (log_p, log_y, log_xbar) → shares
    where log_xbar is supplied by the training loop (computed with delta).
    """
    name = "MDP IRL (E2E δ)"

    def __init__(self, n_goods: int = 3, hidden_dim: int = 256,
                 delta_init: float = 0.5, fixed_beta: float = None):
        super().__init__()
        self.n_goods = n_goods
        # input: log_p (G) + log_y (1) + log_xbar (G)
        self.net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        delta_init = float(np.clip(delta_init, 1e-3, 1.0 - 1e-3))
        self.log_delta = nn.Parameter(
            torch.tensor(np.log(delta_init / (1.0 - delta_init)),
                         dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.tensor(1.0)

    @property
    def delta(self):
        """Learned habit-decay δ ∈ (0, 1)."""
        return torch.sigmoid(self.log_delta)

    def forward(self, log_p, log_y, log_xbar):
        """
        log_p    : (B, G)
        log_y    : (B, 1)
        log_xbar : (B, G) — habit stock computed via compute_xbar_e2e
        """
        x = torch.cat([log_p, log_y, log_xbar], dim=1)
        return torch.softmax(self.net(x), dim=1)

    def _jacobian_symmetry_penalty(self, log_p, log_y, log_xbar):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, log_xbar)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    # Both API aliases
    def slutsky_penalty(self, log_p, log_y, log_xbar):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xbar)

    def slutsky(self, log_p, log_y, log_xbar):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xbar)


# ─────────────────────────────────────────────────────────────────────────────
#  Model 1b: MDPNeuralIRL_E2E with store fixed effects
# ─────────────────────────────────────────────────────────────────────────────

class MDPNeuralIRL_E2E_FE(nn.Module):
    """End-to-end MDP Neural IRL with store fixed effects.

    Identical to MDPNeuralIRL_E2E but with an additional store embedding
    concatenated to the state vector.  The habit stock x̄ is still computed
    on-the-fly with the learnable δ; the store embedding absorbs persistent
    between-store heterogeneity that is not captured by habit dynamics.

    Parameters
    ----------
    n_stores   : int   — number of unique stores.
    emb_dim    : int   — embedding dimension (default 8).
    """
    name = "MDP E2E (FE)"

    def __init__(self, n_goods: int = 3, hidden_dim: int = 256,
                 delta_init: float = 0.5, fixed_beta: float = None,
                 n_stores: int = 100, emb_dim: int = 8):
        super().__init__()
        self.n_goods = n_goods
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        # input: log_p (G) + log_y (1) + log_xbar (G) + store_emb (emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1 + emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        delta_init = float(np.clip(delta_init, 1e-3, 1.0 - 1e-3))
        self.log_delta = nn.Parameter(
            torch.tensor(np.log(delta_init / (1.0 - delta_init)),
                         dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.normal_(self.store_emb.weight, std=0.01)

    @property
    def beta(self):
        return torch.tensor(1.0)

    @property
    def delta(self):
        return torch.sigmoid(self.log_delta)

    def forward(self, log_p, log_y, log_xbar, store_idx):
        """
        log_p     : (B, G)
        log_y     : (B, 1)
        log_xbar  : (B, G) — habit stock from compute_xbar_e2e
        store_idx : (B,)   — integer store indices
        """
        emb = self.store_emb(store_idx)   # (B, emb_dim)
        x = torch.cat([log_p, log_y, log_xbar, emb], dim=1)
        return torch.softmax(self.net(x), dim=1)

    def _jacobian_symmetry_penalty(self, log_p, log_y, log_xbar, store_idx):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, log_xbar, store_idx)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    def slutsky_penalty(self, log_p, log_y, log_xbar, store_idx):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xbar, store_idx)

    def slutsky(self, log_p, log_y, log_xbar, store_idx):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xbar, store_idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Model 2: WindowIRL
# ─────────────────────────────────────────────────────────────────────────────

class WindowIRL(nn.Module):
    """Free-form sequential IRL over a fixed lookback window.

    Conditions on the last *window* periods of (log price, log quantity)
    together with the current (log price, log income).  No parametric
    assumption is imposed on how history enters — the MLP learns whatever
    dynamic pattern (habit, brand loyalty, seasonal) exists in the data.

    Input layout (per observation):
        [log_p_t (G), log_y_t (1),
         log_p_{t-1} (G), log_q_{t-1} (G),
         ...
         log_p_{t-L} (G), log_q_{t-L} (G)]   total dim = G+1+2GL
    """
    name = "Window IRL"

    def __init__(self, n_goods: int = 3, hidden_dim: int = 256, window: int = 4):
        super().__init__()
        self.n_goods = n_goods
        self.window = window
        in_dim = n_goods + 1 + window * 2 * n_goods
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.tensor(1.0)

    def forward(self, x):
        """x : (B, in_dim) — pre-built window feature vector."""
        return torch.softmax(self.net(x), dim=1)

    def _jacobian_symmetry_penalty(self, x):
        """Slutsky symmetry w.r.t. the *current* log prices (first G dims)."""
        G = self.n_goods
        x_d = x.detach().clone().requires_grad_(True)
        w = self.forward(x_d)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), x_d, create_graph=True, retain_graph=True
            )[0][:, :G].unsqueeze(2)
            for i in range(G)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    def slutsky_penalty(self, x):
        return self._jacobian_symmetry_penalty(x)

    def slutsky(self, x):
        return self._jacobian_symmetry_penalty(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Training: MDPNeuralIRL_E2E
# ─────────────────────────────────────────────────────────────────────────────

def train_mdp_e2e(model, prices, income, w_expert, log_q_seq,
                  store_ids=None, epochs=3000, lr=5e-4, batch_size=256,
                  lam_mono=0.2, lam_slut=0.1, slut_start_frac=0.25,
                  xbar_recompute_every=1, delta_lr_scale=0.1, device="cpu",
                  verbose=False, tag="", store_idx=None):
    """Train MDPNeuralIRL_E2E.

    At the start of every *xbar_recompute_every* epochs the habit stock x̄ is
    recomputed for all N observations using the model's current δ so that the
    gradient of the KL loss w.r.t. δ flows through the x̄ computation.

    Parameters
    ----------
    log_q_seq : (N, G) ndarray — log-quantities (or log-shares) in the SAME
                sequential order as prices / income / w_expert.
    store_ids : (N,) int array or None — resets x̄ at store boundaries.
    xbar_recompute_every : int — recompute x̄ every this many epochs.
                Smaller values give more accurate gradients through δ at the
                cost of O(N) sequential ops per recompute step.  Default = 1.
    """
    model = model.to(device)
    # Detect whether this is a store-FE model
    _fe = (store_idx is not None) and hasattr(model, 'store_emb')
    SI  = torch.tensor(store_idx, dtype=torch.long).to(device) if _fe else None
    # Two-timescale optimizer: network weights use full lr; log_delta uses a
    # much slower rate (delta_lr_scale × lr).  Adam is used for both groups so
    # the scalar log_delta parameter gets adaptive scaling regardless of its
    # gradient magnitude relative to the large MLP weight matrices.
    net_params   = [p for n, p in model.named_parameters() if n != 'log_delta']
    delta_params = [model.log_delta]
    opt = optim.Adam([
        {'params': net_params,   'lr': lr,                       'weight_decay': 1e-5},
        {'params': delta_params, 'lr': lr * delta_lr_scale,      'weight_decay': 0.0},
    ])
    N = len(income)
    slut_start = int(epochs * slut_start_frac)

    LP = torch.log(torch.clamp(
        torch.tensor(prices, dtype=torch.float32), min=1e-8)).to(device)
    LY = torch.log(torch.clamp(
        torch.tensor(income, dtype=torch.float32), min=1e-8)).unsqueeze(1).to(device)
    W  = torch.tensor(w_expert, dtype=torch.float32).to(device)
    LQ = torch.log(torch.clamp(
        torch.tensor(log_q_seq if log_q_seq.ndim == 2 else log_q_seq,
                     dtype=torch.float32), min=1e-8)).to(device)
    # log_q_seq is already in log-space for Dominick's; exponentiate-then-log
    # is identity when min-clamped, so this is safe either way.
    # Correct: pass raw log-quantities so we don't double-log.
    LQ_raw = torch.tensor(log_q_seq, dtype=torch.float32).to(device)

    best_kl, best_state, history = float("inf"), None, []
    XB_ALL = None   # will be computed below
    _last_full_kl = float("inf")   # cache full-data KL for history logging

    for ep in range(1, epochs + 1):
        model.train()

        # ── Recompute x̄ for all N observations ──────────────────────────────
        if (ep == 1) or (ep % xbar_recompute_every == 0):
            XB_ALL = compute_xbar_e2e(model.delta, LQ_raw,
                                      store_ids=store_ids)   # (N,G) with grad

        # ── Random mini-batch ────────────────────────────────────────────────
        idx = torch.randperm(N, device=device)[:batch_size]
        lp_b, ly_b, xb_b, w_b = LP[idx], LY[idx], XB_ALL[idx], W[idx]

        opt.zero_grad()
        if _fe:
            wp = model(lp_b, ly_b, xb_b, SI[idx])
        else:
            wp = model(lp_b, ly_b, xb_b)
        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), w_b)

        # Monotonicity: ∂w/∂log_p ≤ 0  (shares fall when own price rises)
        lp_d = lp_b.detach().requires_grad_(True)
        if _fe:
            wm = model(lp_d, ly_b, xb_b.detach(), SI[idx])
        else:
            wm = model(lp_d, ly_b, xb_b.detach())
        grads = torch.autograd.grad(wm.sum(), lp_d, create_graph=True)[0]
        loss_mono = torch.mean(torch.clamp(grads, min=0))

        # Slutsky symmetry (delayed)
        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            if _fe:
                loss_slut = model.slutsky_penalty(LP[sub], LY[sub], XB_ALL[sub].detach(), SI[sub])
            else:
                loss_slut = model.slutsky_penalty(LP[sub], LY[sub], XB_ALL[sub].detach())

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ── Free the computation graph through δ ─────────────────────────────
        # backward() above frees XB_ALL's graph nodes.  Detach so that
        # non-recompute epochs treat XB_ALL as a plain constant tensor
        # (no freed-graph error).  The next recompute epoch builds a fresh
        # graph, giving δ its gradient, then detaches again.
        XB_ALL = XB_ALL.detach()

        # ── Checkpoint (eval on full data) ───────────────────────────────────
        # Aligned with xbar_recompute_every so we reuse XB_ALL (already fresh)
        # instead of triggering an extra O(N) sequential pass every 50 epochs.
        if ep % max(50, xbar_recompute_every) == 0:
            model.eval()
            with torch.no_grad():
                if _fe:
                    wp_full = model(LP, LY, XB_ALL, SI)
                else:
                    wp_full = model(LP, LY, XB_ALL)
                kl_full = nn.KLDivLoss(reduction="batchmean")(
                    torch.log(wp_full + 1e-10), W).item()
            _last_full_kl = kl_full   # cache for history logging
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        # ── History: log full-data KL (not noisy mini-batch KL) ─────────────
        if ep % 100 == 0:
            d_val = model.delta.item()
            history.append({"epoch": ep, "kl": _last_full_kl, "delta": d_val})
            if verbose:
                print(f"    [{tag}] ep {ep:4d} | KL={_last_full_kl:.5f}"
                      f" | delta={d_val:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
#  Training: WindowIRL
# ─────────────────────────────────────────────────────────────────────────────

def train_window_irl(model, window_feats, w_expert,
                     epochs=3000, lr=5e-4, batch_size=256,
                     lam_mono=0.2, lam_slut=0.1, slut_start_frac=0.25,
                     device="cpu", verbose=False, tag=""):
    """Train WindowIRL on pre-built window feature matrix.

    Parameters
    ----------
    window_feats : (N, in_dim) ndarray — output of build_window_features().
    w_expert     : (N, G)     ndarray — observed budget shares (targets).
    """
    model = model.to(device)
    # Adam (not SGD): the scalar log_beta parameter benefits from adaptive
    # per-parameter step sizes, preventing the β oscillations seen with SGD.
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    G = model.n_goods
    N = len(w_expert)
    slut_start = int(epochs * slut_start_frac)

    X = torch.tensor(window_feats, dtype=torch.float32).to(device)
    W = torch.tensor(w_expert,     dtype=torch.float32).to(device)

    best_kl, best_state, history = float("inf"), None, []
    _last_full_kl = float("inf")   # cache full-data KL for history logging

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)[:batch_size]
        x_b, w_b = X[idx], W[idx]

        opt.zero_grad()
        wp = model(x_b)
        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), w_b)

        # Monotonicity w.r.t. current log prices (first G dims)
        x_d = x_b.detach().clone()
        x_d.requires_grad_(True)
        wm = model(x_d)
        grads = torch.autograd.grad(wm.sum(), x_d, create_graph=True)[0][:, :G]
        loss_mono = torch.mean(torch.clamp(grads, min=0))

        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            loss_slut = model.slutsky_penalty(X[sub])

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                kl_full = nn.KLDivLoss(reduction="batchmean")(
                    torch.log(model(X) + 1e-10), W).item()
            _last_full_kl = kl_full   # cache for history logging
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        # Log full-data KL (not noisy mini-batch KL) for clean convergence plots
        if ep % 100 == 0:
            history.append({"epoch": ep, "kl": _last_full_kl, "delta": float("nan")})
            if verbose:
                print(f"    [{tag}] ep {ep:4d} | KL={_last_full_kl:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
