import torch
from torch import nn
from cs336_basics.utils import softmax

def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits shape: [..., vocab_size]
    # targets shape: [...]
    
    #  Subtract the largest element for numerical stability.
    max_values = torch.max(logits, dim=-1, keepdim=True).values # [..., 1]
    stable_logits = logits - max_values # [..., vocab_size]

    probabilities = softmax(stable_logits, i=-1) # [..., vocab_size]
    targets = torch.unsqueeze(targets, dim=-1) # [..., 1]
    target_prob = torch.gather(
        input=probabilities,
        dim=-1,
        index=targets
    ) # [..., 1]
    target_prob = target_prob.squeeze(-1) # [...]

    return torch.mean(-torch.log(target_prob)) # scalar tensor
