# -*- coding: utf-8 -*-
"""
GridCast - Neural Routing Model
Frugal Conv3D network stabilized with GroupNorm for single-trajectory learning.
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("gridcast.models.model")


class HydraulicRoutingNetwork(nn.Module):
    """
    A lightweight, localized Conv3D network that acts as the routing brain.
    Uses GroupNorm instead of BatchNorm to robustly handle standard batch sizes of 1.
    """
    def __init__(self, in_channels: int = 16, hidden_dim: int = 32):
        super().__init__()
        logger.info("Initializing Frugal Hydraulic Routing Network (Conv3D + GroupNorm).")
        
        # Using GroupNorm with 4 groups for 32 hidden dimensions (8 channels per group).
        # This provides spatial-temporal normalization without cross-batch pollution.
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.ReLU()
        )
        
        # Refinement layer to map hidden dimensions back into our 16-channel system
        self.decoder = nn.Conv3d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
    def forward(self, block: torch.Tensor) -> torch.Tensor:
        """
        Processes the 4D space-time block.
        Expects shape: (Batch, Channels, Time, X, Y)
        """
        if block.dim() == 4:
            block = block.unsqueeze(0)
            
        features = self.encoder(block)
        output = self.decoder(features)
        
        return output