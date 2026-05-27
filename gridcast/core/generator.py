# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Generator & Flow Operators
Pure matrix manipulations for fluid routing, toroidal shifting, and displacement.
"""
import logging
import torch

logger = logging.getLogger("gridcast.core.generator")


def shift_toroidal(block: torch.Tensor, t: int, channel_idx: int, dx: int, dy: int) -> torch.Tensor:
    """
    Shifts a specific fluid channel within a closed toroidal circuit (donut flow).
    Fluid passing through the boundary seamlessly rolls over to the opposite side.
    
    Args:
        block (torch.Tensor): The 4D space-time tensor of shape (T, C, X, Y).
        t (int): The target entropy step index.
        channel_idx (int): The specific channel layer to shift (e.g., a chemical dye).
        dx (int): Horizontal shift offset.
        dy (int): Vertical shift offset.
        
    Returns:
        torch.Tensor: The modified block tensor.
    """
    logger.debug("Applying toroidal shift on channel %d at step %d (dx=%d, dy=%d)", channel_idx, t, dx, dy)
    
    # Extract the 2D slice, apply circular shift, and re-inject
    grid_slice = block[t, channel_idx]
    shifted_slice = torch.roll(grid_slice, shifts=(dx, dy), dims=(0, 1))
    block[t, channel_idx] = shifted_slice
    
    return block


def apply_hydraulic_displacement(
    block: torch.Tensor, 
    t: int, 
    dx: int, 
    dy: int, 
    shove_solvent: bool = True
) -> torch.Tensor:
    """
    Displaces all Chemical Dyes (Z=1..9) simultaneously across a vector,
    respecting immutable physical barriers from the Structural Pipes (Z=10).
    
    Args:
        block (torch.Tensor): The 4D space-time tensor of shape (T, C, X, Y).
        t (int): The target entropy step index.
        dx (int): Horizontal translation velocity.
        dy (int): Vertical translation velocity.
        shove_solvent (bool): If True, Base Solvent (Z=0) is dynamically rearranged 
                              to absorb the volume changes and preserve total pressure.
                              
    Returns:
        torch.Tensor: The modified block tensor.
    """
    logger.debug("Executing translation flow at step %d via vector (%d, %d)", t, dx, dy)
    
    # 1.0 means solid wall, 0.0 means free pipe volume
    pipe_mask = block[t, 10] > 0.5
    
    # Identify cells that are blocked from expanding or moving out because their
    # target destination contains a structural pipe barrier.
    # We shift the pipe mask backward to map the barrier onto the source cells.
    source_blocked_mask = torch.roll(pipe_mask, shifts=(-dx, -dy), dims=(0, 1))
    
    # Process each chemical dye layer (Z=1..9)
    for c in range(1, 10):
        dye_layer = block[t, c]
        
        # Calculate shifted state under toroidal mapping
        shifted_dye = torch.roll(dye_layer, shifts=(dx, dy), dims=(0, 1))
        
        # Combine rules:
        # 1. Destination cells that ARE walls (pipe_mask == True) must force 0.0 dye content.
        # 2. Source cells whose paths are blocked (source_blocked_mask == True) must retain their original dye.
        # 3. All other clear channels accept the shifted dye fluid.
        
        # First, allow fluid to shift, but immediately zero out anything that hit a wall
        free_flow = torch.where(pipe_mask, torch.tensor(0.0, device=block.device), shifted_dye)
        
        # Second, for cells that couldn't move because of a roadblock, keep their original fluid intact
        block[t, c] = torch.where(source_blocked_mask, dye_layer, free_flow)
        
    if shove_solvent:
        # Re-equalize the base carrier solvent (Z=0) to pick up the slack.
        dyes_sum = block[t, 1:10].sum(dim=0)
        block[t, 0] = torch.clamp(1.0 - dyes_sum, min=0.0)
        
    return block

def inject_pressure_slope(
    block: torch.Tensor, 
    t: int, 
    direction: str = "down"
) -> torch.Tensor:
    """
    Populates the Pressure Gradient channel (Z=14) with a continuous linear slope.
    This informs the downstream convolutional layers which way the current should flow.
    
    Args:
        block (torch.Tensor): The 4D space-time tensor of shape (T, C, X, Y).
        t (int): The target entropy step index.
        direction (str): Cardinal direction of the gradient ('down', 'up', 'left', 'right').
        
    Returns:
        torch.Tensor: The modified block tensor with updated Z=14 layer.
    """
    device = block.device
    X, Y = block.shape[2], block.shape[3]
    
    # Generate linear continuous float fields normalized between 0.0 and 1.0
    if direction in ("down", "up"):
        lin = torch.linspace(0.0, 1.0, steps=X, device=device)
        gradient_2d = lin.unsqueeze(1).expand(-1, Y)
        if direction == "up":
            gradient_2d = 1.0 - gradient_2d
    elif direction in ("right", "left"):
        lin = torch.linspace(0.0, 1.0, steps=Y, device=device)
        gradient_2d = lin.unsqueeze(0).expand(X, -1)
        if direction == "left":
            gradient_2d = 1.0 - gradient_2d
    else:
        logger.error("Unknown gradient direction parameter: %s", direction)
        return block

    block[t, 14] = gradient_2d
    return block