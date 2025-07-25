import torch

from typing import Optional

from cs336_basics.linear import Linear
from cs336_basics.attention import ScaledDotProductAttention
from cs336_basics.rope import RotaryPositionalEmbeddings

class CausalMHSA(torch.nn.Module):

    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Causal Multi-Head Self-Attention (MHSA) layer.

        Args:
            d_model (int): Dimension of the Transformer block input.
            n_heads (int): Number of attention heads.
            device (torch.device, optional): Device to place the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.dk = d_model // n_heads
        self.dv = self.dk

        self.q_proj = Linear(d_model, n_heads * self.dk, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_heads * self.dk, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_heads * self.dv, device=device, dtype=dtype)
        self.output_proj = Linear(n_heads * self.dv, d_model, device=device, dtype=dtype)

        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.rope = RotaryPositionalEmbeddings(self.dk) if use_rope else None
 
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Causal Multi-Head Self-Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., seq_len, d_model).
        """
        batch_size, *other_dims, seq_len, d_model = x.shape

        Q = self.q_proj(x) # Shape: (batch_size, ..., seq_len, n_heads * d_k) n_heads * d_k == d_model
        K = self.k_proj(x) # Shape: (batch_size, ..., seq_len, n_heads * d_k)
        V = self.v_proj(x) # Shape: (batch_size, ..., seq_len, n_heads * d_v)

        # Reshape to (batch, ..., seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, *other_dims, seq_len, self.n_heads, self.dk)
        K = K.view(batch_size, *other_dims, seq_len, self.n_heads, self.dk)
        V = V.view(batch_size, *other_dims, seq_len, self.n_heads, self.dk)

        # Permute to (batch_size, ..., n_heads, seq_len, head_dim)
        Q = Q.transpose(-3, -2)  # (batch_size, ..., n_heads, seq_len, head_dim)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        # Apply RoPE
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)).bool() # Shape: (seq_len, seq_len)

        attention_scores = self.scaled_dot_product_attention(Q, K, V, mask=mask) # Shape: (batch_size, ..., n_heads, seq_len, head_dim)
        attention_scores = attention_scores.transpose(-3, -2)  # (batch_size, ..., seq_len, n_heads, head_dim)
        attention_scores = attention_scores.reshape(batch_size, *other_dims, seq_len, self.n_heads * self.dv) # Shape: (batch_size, ..., seq_len, n_heads * d_v)

        out = self.output_proj(attention_scores)
        return out  # Shape: (batch_size, ..., seq_len, d_model)
