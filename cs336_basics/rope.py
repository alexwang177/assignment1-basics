import torch

from typing import Optional

class RotaryPositionalEmbeddings(torch.nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        ) # [dim // 2]
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        ) # [max_seq_len]

        # Create a matrix where each row is the sequence index multiplied by the theta values.
        idx_theta = torch.outer(seq_idx, self.theta)  # [max_seq_len, dim // 2]

        cache_cos = torch.cos(idx_theta)  # [max_seq_len, dim // 2]
        cache_sin = torch.sin(idx_theta)  # [max_seq_len, dim // 2]
        self.register_buffer("cache_cos", cache_cos, persistent=False)
        self.register_buffer("cache_sin", cache_sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,               # (..., seq_len, d_k)
        token_positions: torch.Tensor  # (..., seq_len)
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., seq_len, d_k)
            token_positions: (..., seq_len) int tensor, indices into positions

        Returns:
            Same shape as x
        """
        # Get input shape
        *batch_shape, seq_len, d_k = x.shape

        assert d_k == self.dim, f"Expected input last dim {self.dim}, got {d_k}"

        # Slice cos/sin caches with token positions
        cos = self.cache_cos[token_positions]  # (seq_len, dim // 2)
        sin = self.cache_sin[token_positions]  # (seq_len, dim // 2)

        # Reshape x to (..., seq_len, d_k//2, 2)
        # For each token embedding of size d_k, we group adjacent elements into pairs
        x_ = x.float().reshape(*batch_shape, seq_len, d_k // 2, 2)
        x_real = x_[..., 0]  # (..., seq_len, d_k // 2)
        x_imag = x_[..., 1]  # (..., seq_len, d_k // 2)

        # Now match cos/sin's shape to x's batch dims for broadcasting
        for _ in range(len(x_.shape) - len(cos.shape)):
            cos = cos.unsqueeze(-3)  # expand batch dims if needed
            sin = sin.unsqueeze(-3)

        # Apply RoPE rotation
        out_real = x_real * cos - x_imag * sin # (..., seq_len, d_k // 2)
        out_imag = x_imag * cos + x_real * sin # (..., seq_len, d_k // 2)

        out = torch.stack((out_real, out_imag), dim=-1)  # (..., seq_len, d_k//2, 2)
        out = out.reshape(*batch_shape, seq_len, d_k)    # (..., seq_len, d_k)
        return out.type_as(x)
