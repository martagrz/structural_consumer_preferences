"""Dominicks neural training loop."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_dominicks(model, p_tr, y_tr, w_tr, pfx, cfg, xb_tr=None, tag=""):
    """Shared training loop for NeuralIRL / MDPNeuralIRL in Dominicks pipeline."""

    dev = cfg["device"]
    model = model.to(dev)
    opt = optim.Adam(model.parameters(), lr=cfg[f"{pfx}_lr"], weight_decay=1e-5)
    ep_tot = cfg[f"{pfx}_epochs"]
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=40, min_lr=1e-6
    )
    N, bs = len(y_tr), cfg[f"{pfx}_batch"]
    slut0 = int(ep_tot * cfg[f"{pfx}_slut_start"])
    mdp = xb_tr is not None

    LP = torch.log(torch.tensor(np.maximum(p_tr, 1e-8), dtype=torch.float32)).to(dev)
    LY = torch.log(torch.tensor(np.maximum(y_tr, 1e-8), dtype=torch.float32)).unsqueeze(1).to(dev)
    W = torch.tensor(w_tr, dtype=torch.float32).to(dev)
    XB = torch.tensor(xb_tr, dtype=torch.float32).to(dev) if mdp else None

    best_kl, best_sd, hist = float("inf"), None, []

    for ep in range(1, ep_tot + 1):
        model.train()
        idx = torch.randperm(N, device=dev)[:bs]
        lp_b, ly_b, w_b = LP[idx], LY[idx], W[idx]
        xb_b = XB[idx] if mdp else None

        opt.zero_grad()
        wp = model(lp_b, ly_b, xb_b) if mdp else model(lp_b, ly_b)
        lkl = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), w_b)

        lp_d = lp_b.detach().requires_grad_(True)
        wm = model(lp_d, ly_b, xb_b) if mdp else model(lp_d, ly_b)
        g = torch.autograd.grad(wm.sum(), lp_d, create_graph=True)[0]
        lmono = torch.mean(torch.clamp(g, min=0))

        lslut = torch.tensor(0.0, device=dev)
        if ep >= slut0:
            sub = torch.randperm(N, device=dev)[:64]
            lslut = model.slutsky(LP[sub], LY[sub], XB[sub]) if mdp else model.slutsky(LP[sub], LY[sub])

        loss = lkl + cfg[f"{pfx}_lam_mono"] * lmono + cfg[f"{pfx}_lam_slut"] * lslut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                wa = model(LP, LY, XB) if mdp else model(LP, LY)
                kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wa + 1e-10), W).item()
            sched.step(kl)
            if kl < best_kl:
                best_kl = kl
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 20 == 0:
            hist.append({"epoch": ep, "kl": lkl.item(), "beta": model.beta.item()})
            print(f"    [{tag}] ep {ep:4d} | KL={lkl.item():.5f} | beta={model.beta.item():.3f}")

    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()
    return model, hist

