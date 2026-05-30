import torch


class SimpleEncoder:
    """Encoder for ARC grids to environment state. Small grids are centered in the buffer."""

    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.steps = steps
        self.width = width    # buffer width
        self.height = height  # buffer height
        self.channels = 16

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode a 2D (or 3D with leading batch dim) integer grid to a 4D state tensor.
        The grid is centered in the (width, height) buffer; padding cells get channel 0 (BG).
        """
        if grid.dim() == 3:
            grid = grid.squeeze(0)

        grid_h, grid_w = grid.shape

        if grid_w > self.width or grid_h > self.height:
            raise ValueError(
                f"Grid ({grid_h}x{grid_w}) is larger than the buffer ({self.height}x{self.width})."
            )

        state = torch.zeros((self.steps, self.channels, self.width, self.height))

        # Compute top-left offset to centre the grid in the buffer
        offset_x = (self.width - grid_w) // 2
        offset_y = (self.height - grid_h) // 2

        # Map each colour 0-9 to its channel inside the centred region
        for color in range(10):
            mask = (grid == color).float()
            state[:, color, offset_x:offset_x + grid_w, offset_y:offset_y + grid_h] = mask

        # Fill padding cells with BG (channel 0)
        # Build a mask of cells that are outside the centred grid
        padding_mask = torch.ones((self.width, self.height), dtype=torch.bool)
        padding_mask[offset_x:offset_x + grid_w, offset_y:offset_y + grid_h] = False
        state[:, 0, :, :] = torch.where(padding_mask, torch.ones_like(state[0, 0]), state[:, 0, :, :])

        return state

    def decode(self, state: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Decode state back to a 2D grid at a specific step.
        Returns the full buffer grid (width x height) with integer colour labels.
        """
        return torch.argmax(state[step, :10, :, :], dim=0)

    def decode_centered(self, state: torch.Tensor, grid_w: int, grid_h: int, step: int = 0) -> torch.Tensor:
        """
        Decode and crop back to the original grid size, removing the padding.
        """
        full = self.decode(state, step)
        offset_x = (self.width - grid_w) // 2
        offset_y = (self.height - grid_h) // 2
        return full[offset_x:offset_x + grid_w, offset_y:offset_y + grid_h]