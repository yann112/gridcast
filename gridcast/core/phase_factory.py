# -*- coding: utf-8 -*-
"""
GridCast - Pressure Gradient & Phase Factory
Generates continuous vector fields and pressure slopes (Z=14) 
to route fluids dynamically across the grid.
"""
import logging
import math
import torch

logger = logging.getLogger("gridcast.core.phase_factory")


def generate_diagonal_slope(
    block: torch.Tensor, 
    t: int, 
    angle_degrees: float
) -> torch.Tensor:
    """
    Injects a directional linear pressure slope at step t into Z=14.
    Useful for guiding fluid along precise angular vectors (e.g., 45-degree draws).
    
    Args:
        block (torch.Tensor): The 4D space-time tensor (T, C, X, Y).
        t (int): The target entropy step index.
        angle_degrees (float): Angle of the pressure descent (0 = Down, 90 = Right).
        
    Returns:
        torch.Tensor: The modified block tensor.
    """
    logger.debug("Generating linear pressure slope at %d° for step %d", angle_degrees, t)
    
    device = block.device
    X, Y = block.shape[2], block.shape[3]
    
    # Convert angle to radians
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    # Generate coordinate grids
    rows = torch.linspace(-1.0, 1.0, steps=X, device=device).unsqueeze(1).expand(-1, Y)
    cols = torch.linspace(-1.0, 1.0, steps=Y, device=device).unsqueeze(0).expand(X, -1)
    
    # Project coordinates onto the angular vector
    slope = rows * cos_a + cols * sin_a
    
    # Normalize field strictly between 0.0 and 1.0
    slope_min, slope_max = slope.min(), slope.max()
    if slope_max != slope_min:
        slope = (slope - slope_min) / (slope_max - slope_min)
    else:
        slope = torch.zeros_like(slope)
        
    block[t, 14] = slope
    return block


def generate_hydraulic_vortex(
    block: torch.Tensor, 
    t: int, 
    center_x: int, 
    center_y: int, 
    force: float = 1.0
) -> torch.Tensor:
    """
    Injects a circular pressure vortex field at step t into Z=14.
    Forces the convolutional layers to route fluids in a rotational loop 
    around a structural pivot (perfect for ARC rotation puzzles).
    
    Args:
        block (torch.Tensor): The 4D space-time tensor (T, C, X, Y).
        t (int): The target entropy step index.
        center_x (int): Row coordinate of the vortex core.
        center_y (int): Column coordinate of the vortex core.
        force (float): Intensity / steepness of the pressure basin.
        
    Returns:
        torch.Tensor: The modified block tensor.
    """
    logger.debug("Injecting hydraulic vortex at (%d, %d) with force %f", center_x, center_y, force)
    
    device = block.device
    X, Y = block.shape[2], block.shape[3]
    
    # Generate distance coordinates relative to the center
    r_grid = torch.arange(X, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, Y) - center_x
    c_grid = torch.arange(Y, dtype=torch.float32, device=device).unsqueeze(0).expand(X, -1) - center_y
    
    # Calculate angular field (theta) using arctan2
    angles = torch.atan2(r_grid, c_grid)
    
    # Normalize angles from [-pi, pi] to [0.0, 1.0] continuous slope
    vortex_field = (angles + math.pi) / (2.0 * math.pi)
    
    # Apply force scaling
    vortex_field = torch.clamp(vortex_field * force, min=0.0, max=1.0)
    
    block[t, 14] = vortex_field
    return block