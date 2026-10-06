import torch
import math

from collections.abc import Callable
from typing import Optional, Iterable

class AdamW(torch.optim.Optimizer):

    def __init__(
        self, 
        params, 
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m, v = state.get("m", torch.zeros_like(p)), state.get("v", torch.zeros_like(p))

                # adjusted learning rate
                lr_t = lr * (math.sqrt(1.0 - math.pow(b2, t)) / (1.0 - math.pow(b1, t)))

                # apply weight decay
                p.data = p.data - (lr * weight_decay * p.data)

                # first moment
                m = b1 * m + (1 - b1) * p.grad.data

                # second moment
                v = b2 * v + (1 - b2) * p.grad.data ** 2

                # weight update
                p.data = p.data - lr_t * m / (v ** 0.5 + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_squared_sum = None

    for p in parameters:
        if p.grad is None:
            continue

        if total_squared_sum is None:
            total_squared_sum = torch.sum(p.grad ** 2)
        else:
            total_squared_sum += torch.sum(p.grad ** 2)

    if total_squared_sum is None:
        return

    l2_norm = torch.sqrt(total_squared_sum)

    if l2_norm.item() > max_l2_norm:
        scale = max_l2_norm / (l2_norm + 1e-6)

        for p in parameters:
            if p.grad is None:
                continue

            p.grad.mul_(scale)
