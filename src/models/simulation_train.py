"""Simulation neural training loop."""

import torch
import torch.nn as nn
import torch.optim as optim


def train_neural_irl(
    model,
    prices,
    income,
    w_expert,
    epochs=4000,
    lr=5e-4,
    batch_size=256,
    lam_mono=0.3,
    lam_slut=0.1,
    slut_start_frac=0.25,
    xbar_data=None,
    device="cpu",
    verbose=False,
):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    N = len(income)
    slut_start = int(epochs * slut_start_frac)
    mdp_mode = xbar_data is not None

    LP = torch.log(torch.tensor(prices, dtype=torch.float32, device=device))
    LY = torch.log(torch.tensor(income, dtype=torch.float32, device=device)).unsqueeze(1)
    W = torch.tensor(w_expert, dtype=torch.float32, device=device)
    XB = torch.tensor(xbar_data, dtype=torch.float32, device=device) if mdp_mode else None

    best_kl = float("inf")
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)[:batch_size]
        lp_b, ly_b, w_b = LP[idx], LY[idx], W[idx]
        xb_b = XB[idx] if mdp_mode else None
        optimiser.zero_grad()

        w_pred = model(lp_b, ly_b, xb_b) if mdp_mode else model(lp_b, ly_b)
        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(w_pred + 1e-10), w_b)

        lp_d = lp_b.detach().requires_grad_(True)
        w_mn = model(lp_d, ly_b, xb_b) if mdp_mode else model(lp_d, ly_b)
        grads = torch.autograd.grad(w_mn.sum(), lp_d, create_graph=True)[0]
        loss_mono = torch.mean(torch.clamp(grads, min=0))

        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            loss_slut = (
                model.slutsky_penalty(LP[sub], LY[sub], XB[sub])
                if mdp_mode
                else model.slutsky_penalty(LP[sub], LY[sub])
            )

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                wp = model(LP, LY, XB) if mdp_mode else model(LP, LY)
                kl_full = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), W).item()
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 400 == 0:
            history.append({"epoch": ep, "kl": loss_kl.item(), "beta": model.beta.item()})
            if verbose:
                print(f"    ep {ep:4d} | KL={loss_kl.item():.5f} | beta={model.beta.item():.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history

