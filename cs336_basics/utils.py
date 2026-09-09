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


class CausalMHA(nn.Module):

    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None, use_rope=False) -> None:
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_proj = Linear(in_features=d_model, out_features=3 * d_model)
        self.out_proj = Linear(in_features=d_model, out_features=d_model)

        self.rope = None
        if use_rope:
            assert theta is not None and max_seq_len is not None
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    
    def _reshape(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        t = torch.reshape(t, (*t.shape[:-2], seq_len, self.num_heads, self.d_k))
        return torch.transpose(t, -3, -2)

    
    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        seq_len = x.shape[-2]

        QKV = self.qkv_proj(x) # (..., seq_len, 3 * d_model)
        Q, K, V = torch.split(QKV, self.d_model, dim=-1) # each of them are (..., seq_len, d_model)
        Q, K, V = self._reshape(Q), self._reshape(K), self._reshape(V) # (..., num_heads, seq_len, d_k)

        if self.rope is not None:
            assert token_positions is not None

            token_positions = torch.unsqueeze(token_positions, -2) # (..., 1, seq_len)
            token_positions = token_positions.expand((*token_positions.shape[:-2], self.num_heads, seq_len)) # (..., num_heads, seq_len)

            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)) # (seq_len, seq_len)
        attn_out = scaled_dpa(Q, K, V, mask) # (..., num_heads, seq_len, d_k)

        attn_out = torch.transpose(attn_out, -3, -2) # (..., seq_len, num_heads, d_k)
        attn_out = torch.reshape(attn_out, (*attn_out.shape[:-2], self.num_heads * self.d_k))

        return self.out_proj(attn_out)
