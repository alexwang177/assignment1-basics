import torch
import math

def lr_cosine_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
) -> float:
    
    if it < warmup_iters:
        return float(it) / float(warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        return (
            min_learning_rate 
            + 0.5 
            * (1 + math.cos(float(it - warmup_iters) / float(cosine_cycle_iters - warmup_iters) * math.pi)) 
            * (max_learning_rate - min_learning_rate)
        )

    return min_learning_rate
