# -*- coding: utf-8 -*-
"""
GridCast - Synthetic Trajectory Generator
Creates fully simulated 4D block-universes to train the Conv3D network 
on predictive fluid routing and collision avoidance.
"""
import logging
import random
import torch
from gridcast.core.environment import GridCastEnvironment
from gridcast.core.generator import apply_hydraulic_displacement
from gridcast.core.phase_factory import generate_diagonal_slope

logger = logging.getLogger("gridcast.core.synthetic_generator")


class SyntheticHydraulicDataset:
    """
    Generates procedural fluid dynamics datasets containing complete 
    spatio-temporal histories (T steps) for model training.
    """
    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.env = GridCastEnvironment(steps=steps, width=width, height=height)
        
    def generate_single_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a single complete 4D block universe trajectory.
        
        Returns:
            input_block (torch.Tensor): Block with initial state at t=0, 
                                         and the permanent structural layers.
            target_block (torch.Tensor): The fully simulated physics block 
                                          serving as the ground truth.
        """
        # 1. Initialize an empty block universe completely filled with Carrier Solvent
        target_block = torch.zeros((self.env.T, self.env.C, self.env.X, self.env.Y), dtype=torch.float32)
        target_block[:, 0] = 1.0  
        
        # 2. Generate random Structural Pipes (Z=10) - e.g., a random wall row or column
        if random.random() > 0.3:
            wall_idx = random.randint(5, 25)
            if random.random() > 0.5:
                target_block[:, 10, wall_idx, :] = 1.0  # Horizontal pipe wall
            else:
                target_block[:, 10, :, wall_idx] = 1.0  # Vertical pipe wall
                
        # 3. Inject a random chemical dye shape at t=0
        dye_channel = random.randint(1, 9)
        start_x, start_y = random.randint(2, 20), random.randint(2, 20)
        # Place a small 3x3 block of dye, making sure it doesn't overwrite a wall completely
        target_block[0, dye_channel, start_x:start_x+3, start_y:start_y+3] = 1.0
        
        # Clear base solvent where dye was placed to maintain pressure
        dyes_sum = target_block[0, 1:10].sum(dim=0)
        target_block[0, 0] = torch.clamp(1.0 - dyes_sum, min=0.0)
        
        # 4. Define a constant drift velocity vector for this trajectory
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            dx = 1  # Force at least some movement
            
        # 5. Inject the corresponding Pressure Gradient field (Z=14) across all steps
        # Map velocity vector to a rough angle for the phase factory
        angle = 0.0
        if dx == 1 and dy == 0: angle = 0.0    # Down
        elif dx == 0 and dy == 1: angle = 90.0  # Right
        elif dx == -1 and dy == 0: angle = 180.0 # Up
        elif dx == 0 and dy == -1: angle = 270.0 # Left
        
        for t in range(self.env.T):
            target_block = generate_diagonal_slope(target_block, t, angle)
            
        # 6. Simulate the fluid mechanics step-by-step to populate intermediate states
        for t in range(1, self.env.T):
            # Copy previous state as baseline
            target_block[t, 0:11] = target_block[t-1, 0:11].clone()
            # Apply displacement from t-1 to t
            target_block = apply_hydraulic_displacement(target_block, t=t, dx=dx, dy=dy, shove_solvent=True)
            # Guarantee numerical sealing
            target_block = self.env.enforce_containment(target_block)
            
        # 7. Construct the Input Block
        # The network only gets the initial state (t=0) and the guidance tracks (Z=10, Z=14)
        input_block = torch.zeros_like(target_block)
        input_block[0, 0:10] = target_block[0, 0:10].clone()  # Initial fluid state
        input_block[:, 10] = target_block[:, 10].clone()      # Permanent walls
        input_block[:, 14] = target_block[:, 14].clone()      # Permanent pressure gradients
        
        return input_block, target_block