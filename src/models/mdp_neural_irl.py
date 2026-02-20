"""Shared MDP-aware Neural IRL model used by both pipelines."""

import numpy as np
import torch
import torch.nn as nn


class MDPNeuralIRL(nn.Module):
    name = "MDP Neural IRL"

    def __init__(self, h=256, n_goods=3, hidden_dim=None, delta_init=0.7):
        """
        Parameters
        ----------
        h / hidden_dim : int
            Width of the hidden layers.
        n_goods : int
            Number of goods in the demand system.
        delta_init : float in (0, 1)
            Starting value for the habit-decay parameter δ.  The model
            parameterises δ = sigmoid(log_delta) so that it is always in
            (0, 1) and is learned end-to-end during training.
        """
        super().__init__()
        self.n_goods = n_goods
        hdim = hidden_dim if hidden_dim is not None else h
        self.net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        self.log_beta = nn.Parameter(torch.tensor(1.5))

        # Learnable habit-decay δ = sigmoid(log_delta), initialised at delta_init
        delta_init = float(np.clip(delta_init, 1e-3, 1.0 - 1e-3))
        self.log_delta = nn.Parameter(
            torch.tensor(np.log(delta_init / (1.0 - delta_init)), dtype=torch.float32)
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    # ------------------------------------------------------------------ #
    #  Learned parameters                                                  #
    # ------------------------------------------------------------------ #

    @property
    def beta(self):
        return torch.exp(self.log_beta).clamp(0.5, 20.0)

    @property
    def delta(self):
        """Habit-decay rate δ ∈ (0, 1), learned via sigmoid re-parameterisation."""
        return torch.sigmoid(self.log_delta)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, log_p, log_y, log_xb_prev, log_q_prev):
        """
        Compute predicted budget shares.

        Parameters
        ----------
        log_p      : Tensor (B, G)  – log prices (both pipelines pass this raw)
        log_y      : Tensor (B, 1)  – log income
        log_xb_prev : Tensor (B, G) – pre-processed log of the previous-period
                      habit stock  x̄_{t-1}.  The Dominicks pipeline normalises
                      this to match the log-price scale; the simulation pipeline
                      passes plain log(x̄_{t-1}).
        log_q_prev  : Tensor (B, G) – pre-processed log of the previous-period
                      quantities q_{t-1}, using the *same* transformation as
                      log_xb_prev.

        The model forms its working habit measure as

            log_x̄_t^eff = δ · log_xb_prev + (1 − δ) · log_q_prev

        i.e. a convex combination in log-space with learnable δ.  When
        δ = δ_ref (the initialisation value) and both inputs are in the same
        normalised log-space, this closely approximates what a fixed-δ model
        would see.
        """
        delta = self.delta
        xb_input = delta * log_xb_prev + (1.0 - delta) * log_q_prev
        return torch.softmax(
            self.net(torch.cat([log_p, log_y, xb_input], dim=1)) * self.beta, dim=1
        )

    # ------------------------------------------------------------------ #
    #  Slutsky symmetry penalty                                            #
    # ------------------------------------------------------------------ #

    def _jacobian_symmetry_penalty(self, log_p, log_y, log_xb_prev, log_q_prev):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, log_xb_prev, log_q_prev)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    # Dominicks API
    def slutsky(self, lp, ly, log_xb_prev, log_q_prev):
        return self._jacobian_symmetry_penalty(lp, ly, log_xb_prev, log_q_prev)

    # Simulation API
    def slutsky_penalty(self, log_p, log_y, log_xb_prev, log_q_prev):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xb_prev, log_q_prev)
