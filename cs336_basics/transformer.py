import torch

from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.causal_mhsa import CausalMHSA

class TransformerBlock(torch.nn.Module):
    """
    A Transformer block that applies multi-head self-attention with RoPE and a SwiGLU feed-forward network.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Transformer block.

        Args:
            d_model (int): Dimension of the input tensor.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            device (torch.device, optional): Device to place the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMHSA(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Transformer block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., seq_len, d_model).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x  # Shape: (batch_size, ..., seq_len, d_model)
