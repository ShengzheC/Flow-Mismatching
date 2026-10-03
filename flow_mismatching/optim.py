"""StableAdamW: AdamW with per-parameter update-RMS clipping and optional AMSGrad."""

import math
from typing import Iterable, Optional, Tuple

import torch


class StableAdamW(torch.optim.Optimizer):
    """AdamW whose update is rescaled per parameter tensor so that its RMS is at most ``update_clip``.

    Clipping the optimizer update (rather than the raw gradient) keeps training stable
    without an additional gradient-norm clip.

    Args:
        params: parameters to optimize.
        lr: learning rate.
        betas: coefficients of the running averages of the gradient and its square.
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled weight decay, as in AdamW.
        amsgrad: use the running maximum of the second moment (AMSGrad).
        update_clip: RMS threshold of the update; None or <= 0 disables clipping.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-4,
        amsgrad: bool = True,
        update_clip: Optional[float] = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, update_clip=update_clip
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            update_clip = group["update_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("StableAdamW does not support sparse gradients")

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay.
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_sq = max_exp_avg_sq
                else:
                    denom_sq = exp_avg_sq

                denom = denom_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(bias_correction1).div(denom)

                # Rescale the update so that its RMS does not exceed update_clip.
                if update_clip is not None and update_clip > 0:
                    update_rms = update.pow(2).mean().sqrt()
                    update = update / torch.clamp(update_rms / update_clip, min=1.0)

                p.add_(update, alpha=-lr)

        return loss
