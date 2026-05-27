# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Flow Visualizer
Generates spatial-temporal grid plots to monitor fluid routing and model predictions.
"""
import logging
import os
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger("gridcast.utils.visualizer")


def plot_hydraulic_trajectory(block: torch.Tensor, save_path: str = "trajectory.png"):
    """
    Plots the temporal evolution of the fluid system.
    Extracts the dominant fluid states (Z=0..9) across time steps using argmax
    and saves a clean, chronological sequence map.
    
    Args:
        block (torch.Tensor): A 4D fluid tensor of shape (T, C, X, Y) 
                              or a 5D model tensor of shape (1, C, T, X, Y).
        save_path (str): Filepath where the final diagnostic plot will be stored.
    """
    # Normalize shape to standard 4D internal circuit format (T, C, X, Y)
    if block.dim() == 5:
        # If it comes straight from the Conv3D output, remove batch and swap Time/Channels
        block = block.squeeze(0).permute(1, 0, 2, 3)
        
    T, C, X, Y = block.shape
    logger.info("Generating diagnostic visualization for a %dx%dx%dx%d tensor.", T, C, X, Y)
    
    # Extract only the fluid channels (Base Solvent + Dyes: Z=0..9)
    fluid_slice = block[:, 0:10]
    
    # Decode the dense continuous floats into discrete integer values via argmax
    decoded_timeline = torch.argmax(fluid_slice, dim=1).cpu().numpy()
    
    # Set up a grid of subplots (1 row, T columns)
    fig, axes = plt.subplots(1, T, figsize=(2 * T, 2), dpi=150)
    fig.suptitle("GridCast Spatio-Temporal Fluid Routing Timeline", fontsize=12, y=1.1)
    
    # Standard ARC color palette representation mapping (0-9 values)
    # Using 'tab10' or 'accent' as a discrete colormap for clarity
    cmap = plt.cm.get_cmap("tab10", 10)
    
    for t in range(T):
        ax = axes[t] if T > 1 else axes
        ax.imshow(decoded_timeline[t], cmap=cmap, vmin=0, vmax=9, origin="upper")
        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")
        
    plt.tight_layout()
    
    # Ensure directory structure exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    logger.info("Hydraulic trajectory visualization successfully saved to: %s", save_path)