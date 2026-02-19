"""Shared MDP-aware Neural IRL model used by both pipelines."""

import torch
import torch.nn as nn


class MDPNeuralIRL(nn.Module):
    name = "MDP Neural IRL"

    def __init__(self, h=256, n_goods=3, hidden_dim=None):
        super().__init__()
        self.n_goods = n_goods
        hdim = hidden_dim if hidden_dim is not None else h
        self.reward_net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        self.log_beta = nn.Parameter(torch.tensor(1.5))
        for m in self.reward_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.reward_net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.exp(self.log_beta).clamp(0.5, 20.0)

    def forward(self, log_p, log_y, xbar):
        log_xbar = torch.log(torch.clamp(xbar, min=1e-6))
        return torch.softmax(self.reward_net(torch.cat([log_p, log_y, log_xbar], 1)) * self.beta, 1)

    def _jacobian_symmetry_penalty(self, log_p, log_y, xbar):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, xbar)
        rows = [
            torch.autograd.grad(w[:, i].sum(), lp_d, create_graph=True, retain_graph=True)[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, 2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    # Dominicks API
    def slutsky(self, lp, ly, xb):
        return self._jacobian_symmetry_penalty(lp, ly, xb)

    # Simulation API
    def slutsky_penalty(self, log_p, log_y, xbar):
        return self._jacobian_symmetry_penalty(log_p, log_y, xbar)

