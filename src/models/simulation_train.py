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
    xb_prev_data=None,
    q_prev_data=None,
    # Legacy alias kept for backward compatibility: xbar_data → xb_prev_data
    xbar_data=None,
    device="cpu",
    verbose=False,
):
    """Train a NeuralIRL or MDPNeuralIRL model.

    For MDPNeuralIRL pass *both* xb_prev_data and q_prev_data (raw
    quantities in physical units).  The loop converts them to log-space
    before handing them to the model, matching the model's expected input
    convention.

    Legacy callers that still pass ``xbar_data`` (single array) are
    redirected: xb_prev_data=xbar_data, q_prev_data derived from shares.
    That path is deprecated; prefer the two-array interface.
    """
    # ── Backward-compat shim ──────────────────────────────────────────────
    if xbar_data is not None and xb_prev_data is None:
        # Old single-array interface: treat xbar_data as xb_prev, derive
        # q_prev as a one-step lag of itself (approximation).
        import numpy as np
        xb_prev_data = xbar_data
        # q_prev ≈ xbar itself when we don't have explicit quantity data
        # (conservative fall-back: q_prev = xb_prev, so delta doesn't change the input)
        q_prev_data  = xbar_data

    mdp_mode = (xb_prev_data is not None) and (q_prev_data is not None)

    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    N = len(income)
    slut_start = int(epochs * slut_start_frac)

    import numpy as np
    LP = torch.log(torch.tensor(prices,  dtype=torch.float32, device=device))
    LY = torch.log(torch.tensor(income,  dtype=torch.float32, device=device)).unsqueeze(1)
    W  = torch.tensor(w_expert, dtype=torch.float32, device=device)

    # Convert raw quantities to log-space (model always expects log-space inputs)
    if mdp_mode:
        XB_PREV = torch.log(torch.clamp(
            torch.tensor(xb_prev_data, dtype=torch.float32, device=device), min=1e-6))
        Q_PREV  = torch.log(torch.clamp(
            torch.tensor(q_prev_data,  dtype=torch.float32, device=device), min=1e-6))
    else:
        XB_PREV = Q_PREV = None

    best_kl = float("inf")
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)[:batch_size]
        lp_b, ly_b, w_b = LP[idx], LY[idx], W[idx]
        optimiser.zero_grad()

        if mdp_mode:
            xbp_b = XB_PREV[idx]
            qp_b  = Q_PREV[idx]
            w_pred = model(lp_b, ly_b, xbp_b, qp_b)
        else:
            w_pred = model(lp_b, ly_b)

        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(w_pred + 1e-10), w_b)

        lp_d = lp_b.detach().requires_grad_(True)
        if mdp_mode:
            w_mn = model(lp_d, ly_b, xbp_b, qp_b)
        else:
            w_mn = model(lp_d, ly_b)
        grads = torch.autograd.grad(w_mn.sum(), lp_d, create_graph=True)[0]
        loss_mono = torch.mean(torch.clamp(grads, min=0))

        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            if mdp_mode:
                loss_slut = model.slutsky_penalty(
                    LP[sub], LY[sub], XB_PREV[sub], Q_PREV[sub])
            else:
                loss_slut = model.slutsky_penalty(LP[sub], LY[sub])

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                if mdp_mode:
                    wp = model(LP, LY, XB_PREV, Q_PREV)
                else:
                    wp = model(LP, LY)
                kl_full = nn.KLDivLoss(reduction="batchmean")(
                    torch.log(wp + 1e-10), W).item()
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 400 == 0:
            delta_val = model.delta.item() if mdp_mode else float("nan")
            history.append({
                "epoch": ep,
                "kl":    loss_kl.item(),
                "beta":  model.beta.item(),
                "delta": delta_val,
            })
            if verbose:
                delta_str = f" | delta={delta_val:.3f}" if mdp_mode else ""
                print(f"    ep {ep:4d} | KL={loss_kl.item():.5f} "
                      f"| beta={model.beta.item():.3f}{delta_str}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
