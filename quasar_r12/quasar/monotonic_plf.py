import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotonicPLF(nn.Module):
    """
    Piecewise Linear Function (PLF) with monotonically increasing slopes.

    The function is defined as:
    f(x) = base + slope_0 * x + sum_{i=1}^N delta_slope_i * max(0, x - knot_i)

    where slope_0 = exp(log_base_slope) and delta_slope_i = exp(log_delta_slopes_i) >= 0.
    This ensures that the slope is non-decreasing across knots.

    Args:
        knots (torch.Tensor or list/tuple): A 1D tensor or list containing the knot locations (k_1, ..., k_N).
                                           Must be sorted in ascending order.
    """
    def __init__(self, knots):
        super().__init__()
        if not isinstance(knots, torch.Tensor):
            # Ensure knots are stored as a float tensor
            knots = torch.tensor(knots, dtype=torch.float32)
        if knots.dim() != 1:
            raise ValueError("Knots must be a 1D tensor.")
        if torch.any(knots[1:] < knots[:-1]):
            # Check for strict inequality as duplicate knots are problematic
            raise ValueError("Knots must be sorted in strictly ascending order.")

        # Register knots as buffer (non-trainable state but part of the module's state)
        self.register_buffer('knots', knots)
        num_knots = len(knots)

        # Learnable parameters
        self.base = nn.Parameter(torch.randn(1)) # Base value offset
        self.log_base_slope = nn.Parameter(torch.randn(1)) # Log of the base slope (for x < k_1)
        self.log_delta_slopes = nn.Parameter(torch.randn(num_knots)) # Log of slope increments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the PLF value for the input tensor x.

        Args:
            x (torch.Tensor): Input tensor. Can be scalar or batched (e.g., shape (batch_size,)).

        Returns:
            torch.Tensor: Output tensor with the same shape as x.
        """
        # Ensure x is float
        x = x.to(dtype=self.knots.dtype, device=self.knots.device)

        # Handle scalar input consistently
        original_shape = x.shape
        if x.dim() == 0:
            x = x.unsqueeze(0)

        # Calculate positive slopes
        base_slope = torch.exp(self.log_base_slope)
        delta_slopes = torch.exp(self.log_delta_slopes) # Shape: (num_knots,)

        # Calculate ReLU terms: max(0, x - knot_i)
        # x shape: (batch_size,), knots shape: (num_knots,)
        # Unsqueeze x to (batch_size, 1) for broadcasting with knots (1, num_knots)
        relu_terms = F.relu(x.unsqueeze(-1) - self.knots.unsqueeze(0)) # Shape: (batch_size, num_knots)

        # Compute the PLF output
        # base: scalar -> broadcasts
        # base_slope * x: (batch_size,)
        # delta_slopes * relu_terms: (batch_size, num_knots) element-wise product
        # torch.sum(..., dim=-1): (batch_size,)
        output = self.base + base_slope * x + torch.sum(delta_slopes * relu_terms, dim=-1)

        # Reshape output to match original input shape
        return output.reshape(original_shape)

    def get_slopes(self) -> torch.Tensor:
        """
        Returns the actual slopes for each segment.
        The segments are (-inf, k_1], (k_1, k_2], ..., (k_{N-1}, k_N], (k_N, +inf).
        There are N+1 segments for N knots.
        """
        base_slope = torch.exp(self.log_base_slope)
        delta_slopes = torch.exp(self.log_delta_slopes)
        # Cumulative sum of delta slopes represents the *increase* from the base slope
        cumulative_delta_slopes = torch.cumsum(delta_slopes, dim=0)
        # Prepend the base slope
        segment_slopes = torch.cat(
            (base_slope, base_slope + cumulative_delta_slopes),
            dim=0
        )
        return segment_slopes