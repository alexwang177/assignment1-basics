import torch

class RMSNorm(torch.nn.Module):
    """
    RMS Normalization layer that normalizes the input tensor along the last dimension.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): Dimension to normalize.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype) # Initialize with ones
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Convert to float32 for numerical stability

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight

        return x.to(in_dtype)  # Convert back to original dtype
        