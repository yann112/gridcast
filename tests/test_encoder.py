# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Encoder Unit Tests
Validates channel assignment, initial system pressure, and argmax decoding.
"""
import pytest
import torch
from gridcast.core.encoder import HydraulicEncoder
from gridcast.core.environment import GridCastEnvironment


@pytest.fixture
def encoder():
    """Initializes an encoder with the standard pipeline architecture."""
    return HydraulicEncoder(steps=12, width=30, height=30)


@pytest.fixture
def env():
    """Initializes the environment validator to check for physical leaks."""
    return GridCastEnvironment(steps=12, width=30, height=30)


def test_encode_and_pressure_balance(encoder, env):
    """Ensures that encoding an ARC grid creates correct pressures without leaks."""
    # Create a mock 5x5 ARC grid with a background (0) and two distinct colors (3 and 7)
    arc_grid = torch.zeros((5, 5), dtype=torch.int64)
    arc_grid[1, 1] = 3
    arc_grid[2, 3] = 7
    
    # Encode the discrete grid into the 4D fluid block
    block = encoder.encode_grid(arc_grid)
    
    # 1. Structural check
    assert block.shape == (encoder.T, encoder.C, encoder.X, encoder.Y)
    
    # 2. Check channel positioning at t=0
    assert block[0, 3, 1, 1] == 1.0  # Color 3 mapped to Dye Z=3
    assert block[0, 0, 1, 1] == 0.0  # Base Solvent displaced at (1,1)
    
    assert block[0, 7, 2, 3] == 1.0  # Color 7 mapped to Dye Z=7
    assert block[0, 0, 2, 3] == 0.0  # Base Solvent displaced at (2,3)
    
    assert block[0, 0, 0, 0] == 1.0  # Empty background is 1.0 Base Solvent
    
    # 3. Validate that the entire block universe satisfies containment laws
    assert env.check_hydraulic_leak(block) is True


def test_static_obstacles_propagation(encoder):
    """Verifies that structural pipes (Z=10) map correctly and persist over time."""
    arc_grid = torch.zeros((4, 4), dtype=torch.int64)
    obstacles = torch.zeros((4, 4), dtype=torch.int64)
    obstacles[0, :] = 1  # Top row defined as a solid barrier
    
    block = encoder.encode_grid(arc_grid, static_obstacles=obstacles)
    
    # Check that the wall is written on Z=10 across ALL time steps within input boundaries
    for t in range(encoder.T):
        # The 4 elements from the input obstacle slice must be active
        assert (block[t, 10, 0, :4] == 1.0).all()
        # The remainder of the padded 30-wide grid row must be inactive
        assert (block[t, 10, 0, 4:] == 0.0).all()
        # Other rows remain empty
        assert (block[t, 10, 1:, :] == 0.0).all()


def test_roundtrip_encode_decode(encoder):
    """Validates that a grid can be encoded and successfully decoded back via argmax."""
    original_grid = torch.zeros((6, 6), dtype=torch.int64)
    original_grid[0, 0] = 1
    original_grid[2, 2] = 5
    original_grid[5, 4] = 9
    
    # Encode to continuous fluid and immediately decode the final state
    block = encoder.encode_grid(original_grid)
    decoded = encoder.decode_grid(block, t=-1)
    
    # Check that the dimensions match our environment constraints (30x30 padding)
    assert decoded.shape == (encoder.X, encoder.Y)
    
    # Verify the values match the original subgrid coordinates
    assert decoded[0, 0] == 1
    assert decoded[2, 2] == 5
    assert decoded[5, 4] == 9
    assert decoded[1, 1] == 0  # Background preserved