import torch

class SoftMax(torch.nn.Module):
    """
    A simple softmax layer that applies the softmax function to the input.
    """

    def __init__(self, dim: int = -1):
        """
        Initializes the SoftMax layer.

        Args:
            dim (int): Dimension along which to apply softmax.
        """
        super().__init__()
        self.dim = dim
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the softmax function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        max_vals = torch.max(x, dim=self.dim, keepdim=True).values
        e_x = torch.exp(x - max_vals)
        sum_e_x = torch.sum(e_x, dim=self.dim, keepdim=True)
        return e_x / sum_e_x
        
# softmax = SoftMax(dim=-1)
# x = torch.randn(2, 3, 4)
# output = softmax(x)
