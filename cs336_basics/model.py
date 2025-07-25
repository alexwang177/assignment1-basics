import torch

from cs336_basics.embedding import Embedding
from cs336_basics.softmax import SoftMax
from cs336_basics.causal_mhsa import CausalMHSA
from cs336_basics.transformer import TransformerBlock
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.linear import Linear


class TransformerLM(torch.nn.Module):

    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        """
        Initializes the Transformer language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Length of the context.
            d_model (int): Dimension of the model.
            num_layers (int): Number of layers in the Transformer.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
        """
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_ff, rope_theta=rope_theta, rope_max_seq_length=context_length) for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x # Shape: (batch_size, sequence_length, vocab_size)
