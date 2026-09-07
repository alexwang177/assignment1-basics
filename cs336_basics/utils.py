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


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0

        token_positions = torch.arange(start=0, end=max_seq_len, step=1, device=device, dtype=torch.float).reshape((max_seq_len, 1))
        pair_indices = torch.arange(start=0, end=d_k, step=2, device=device, dtype=torch.float).reshape((1, d_k // 2))

        freqs = theta ** (-pair_indices / d_k)
        angles = token_positions * freqs

        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_rows = self.cos_table[token_positions]
        sin_rows = self.sin_table[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        even_rotated = x_even * cos_rows + x_odd * -sin_rows
        odd_rotated = x_even * sin_rows + x_odd * cos_rows

        output = torch.empty_like(x)
        output[..., 0::2] = even_rotated
        output[..., 1::2] = odd_rotated
        return output


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    stable = x - torch.max(x, dim=i, keepdim=True).values
    exp_values = torch.exp(stable)
    output = exp_values / torch.sum(exp_values, dim=i, keepdim=True)
    return output


def scaled_dpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor:
    d_k = Q.shape[-1]
    scaled_scores = (Q @ torch.transpose(K, -2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, -torch.inf)

    attn_probs = softmax(scaled_scores, i=-1)
    return attn_probs @ V
