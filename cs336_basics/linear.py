import torch

class Linear(torch.nn.Module):
    """
    A simple linear layer that applies a linear transformation to the input.
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (torch.device, optional): Device to place the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()

        std = (2.0 / (in_features + out_features)) ** 0.5

        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return x @ self.W.T
