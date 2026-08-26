import torch
from torch import nn

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):

        super().__init__()

        sigma = (2.0 / (in_features + out_features)) ** 0.5

        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0.0,
                std=sigma,
                a=-3.0 * sigma,
                b=3.0 * sigma
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):

        super().__init__()

        self.table = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0.0,
                std=1.0,
                a=-3.0,
                b=3.0
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids has shape (batch, sequence_length)
        return self.table[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gain


class SiLU(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
