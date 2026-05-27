# -*- coding: utf-8 -*-
"""
GridCast - Visualizer Verification Suite
"""
import os
import pytest
import torch
from gridcast.utils.flow_visualizer import plot_hydraulic_trajectory


def test_visualizer_file_generation(tmp_path):
    """Ensures the visualizer successfully processes tensors and exports the file."""
    # Create a mock 4D block universe (T=6, C=16, X=30, Y=30)
    mock_block = torch.zeros((6, 16, 30, 30), dtype=torch.float32)
    mock_block[:, 0] = 1.0  # Filled with base solvent
    
    # Place a random dye signature that changes position over time
    for t in range(6):
        mock_block[t, 3, t + 5, 10] = 1.0
        mock_block[t, 0, t + 5, 10] = 0.0  # Clear pressure backfill
        
    output_file = os.path.join(tmp_path, "outputs", "test_flow.png")
    
    # Execute the visualizer
    plot_hydraulic_trajectory(mock_block, save_path=output_file)
    
    # Confirm file generation
    assert os.path.exists(output_file) is True
    assert os.path.getsize(output_file) > 0