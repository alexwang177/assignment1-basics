import torch

class Embedding(torch.nn.Module):
    """
    A simple embedding layer that maps tokens to dense vectors.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Embedding layer.

        Args:
            num_embeddings (int): Number of unique tokens (vocabulary size).
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device, optional): Device to place the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0.0,
                std=1.0,
                a=-3.0,
                b=3.0,
            )
        ) # shape (num_embeddings, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps input token ids to their corresponding embedding vectors.

        Args:
            token_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        return self.weight[token_ids] # shape (batch_size, sequence_length, embedding_dim)
