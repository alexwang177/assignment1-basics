import torch

from cs336_basics.linear import Linear

class SiLU(torch.nn.Module):
    """
    Sigmoid Linear Unit (SiLU) activation function.
    Combines the sigmoid activation with a linear transformation.
    """

    def __init__(self):
        """
        Initializes the SiLU layer.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SiLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    """
    Swish-Gated Linear Unit (SwigLU) activation function.
    Combines the Swish activation with a linear transformation.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the SwigLU layer.

        Args:
            d_model (int): Dimension of the input tensor.
            d_ff (int): Dimension of the feed-forward layer.
            device (torch.device, optional): Device to place the layer on.
            dtype (torch.dtype, optional): Data type of the layer parameters.
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwigLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
