# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Channel Encoder
Translates discrete ARC puzzle matrices into the 16-channel continuous fluid tensor.
"""
import logging
import torch

logger = logging.getLogger("gridcast.core.encoder")


class HydraulicEncoder:
    """
    Encodes standard 2D ARC integer grids into 4D space-time fluid blocks,
    and decodes fluid dye densities back into discrete output matrices.
    """
    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.T = steps
        self.C = 16
        self.X = width
        self.Y = height

    def encode_grid(self, arc_matrix: torch.Tensor, static_obstacles: torch.Tensor = None) -> torch.Tensor:
        """
        Translates a 2D ARC grid (values 0-9) into a 4D fluid tensor (T, C, X, Y).
        
        - ARC Values 1..9 map directly to Chemical Dyes Z=1..9 with a density of 1.0.
        - ARC Value 0 (background) maps to the Base Solvent Z=0.
        - Any node containing a dye will have its Base Solvent Z=0 set to 0.0 
          to respect strict hydraulic pressure limits (Sum Z_0..9 = 1.0).
          
        Args:
            arc_matrix (torch.Tensor): 2D integer tensor of shape (X_in, Y_in) with values 0-9.
            static_obstacles (torch.Tensor): Optional 2D binary tensor marking fixed walls (Z=10).
            
        Returns:
            torch.Tensor: Normalized 4D block tensor of shape (T, C, X, Y).
        """
        logger.info("Encoding 2D ARC matrix into 16-channel hydraulic block.")
        
        # Initialize an empty block universe (Time, Channels, X, Y)
        block = torch.zeros((self.T, self.C, self.X, self.Y), dtype=torch.float32)
        
        # Get actual input dimensions to handle padding safely
        in_x, in_y = arc_matrix.shape
        X_lim = min(in_x, self.X)
        Y_lim = min(in_y, self.Y)
        
        # Slice input data into the initial time step (t=0)
        source_grid = arc_matrix[:X_lim, :Y_lim]
        
        # 1. Map Chemical Dyes (Z=1..9)
        for dye_val in range(1, 10):
            dye_mask = (source_grid == dye_val).float()
            block[0, dye_val, :X_lim, :Y_lim] = dye_mask
            
        # 2. Establish Hydraulic Equilibrium for the Base Solvent (Z=0)
        # Solvent volume is 1.0 minus the sum of all active dyes at that node
        active_dyes_sum = block[0, 1:10].sum(dim=0)
        block[0, 0] = torch.clamp(1.0 - active_dyes_sum, min=0.0)
        
        # 3. Inject Structural Pipes (Z=10) if static boundaries are provided
        if static_obstacles is not None:
            pipe_mask = (static_obstacles[:X_lim, :Y_lim] > 0).float()
            block[:, 10, :X_lim, :Y_lim] = pipe_mask.unsqueeze(0).expand(self.T, -1, -1)
            
        # Propagate the initial state (t=0) across the entire temporal timeline (t=1..T-1)
        # to serve as a steady-state baseline before the model applies pressure gradients.
        for t in range(1, self.T):
            block[t, 0:11] = block[0, 0:11].clone()
            
        return block

    def decode_grid(self, block: torch.Tensor, t: int = -1) -> torch.Tensor:
        """
        Decodes a specific time step of the fluid tensor back into a discrete ARC grid.
        Uses an argmax selection over the chemical dyes (Z=1..9). If the highest dye 
        density is below the base solvent concentration or zero, it decodes as background (0).
        
        Args:
            block (torch.Tensor): 4D fluid tensor of shape (T, C, X, Y).
            t (int): The target time step to decode (defaults to the final step -1).
            
        Returns:
            torch.Tensor: 2D integer tensor of shape (X, Y) with values 0-9.
        """
        logger.info("Decoding hydraulic block at step %d back to discrete ARC grid.", t)
        
        # Extract the fluid channels (Z=0 to Z=9) at the target step
        fluid_slice = block[t, 0:10]  # Shape: (10, X, Y)
        
        # Select the dominant channel via argmax
        decoded_grid = torch.argmax(fluid_slice, dim=0)  # Shape: (X, Y), values 0-9
        
        return decoded_grid