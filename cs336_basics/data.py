import numpy as np
import numpy.typing as npt
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    
    start_indices = [
        np.random.randint(low=0, high=len(dataset) - context_length)
        for _ in range(batch_size)
    ]
    
    inputs = np.stack([dataset[i: i + context_length] for i in start_indices])
    targets = np.stack([dataset[i+1: i + 1 + context_length] for i in start_indices])
    
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
   