import torch

from typing import Optional

from cs336_basics.softmax import SoftMax

class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Computes attention scores and applies them to the value vectors.
    """

    def __init__(self):
        """
        Initializes the Scaled Dot-Product Attention layer.
        """
        super().__init__()
        

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies scaled dot-product attention to the input tensors.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len_q, d_k).
            K (torch.Tensor): Key tensor of shape (batch_size,  ..., seq_len_k, d_k).
            V (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len_v, d_v).
            mask (Optional[torch.Tensor]): Optional mask tensor of shape (batch_size, ... , seq_len, seq_len) to apply attention masking.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., seq_len, d_v).
        """

        d_k = Q.size(-1)
        scaling = d_k ** -0.5
        softmax = SoftMax(dim=-1)

        scores = Q @ K.transpose(-2, -1) * scaling
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return softmax(scores) @ V
