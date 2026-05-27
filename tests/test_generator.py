# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Generator Unit Tests
Validates toroidal shifts, collision handling against pipes, and gradient layouts.
"""
import pytest
import torch
from gridcast.core.environment import GridCastEnvironment
from gridcast.core.generator import shift_toroidal, apply_hydraulic_displacement, inject_pressure_slope


@pytest.fixture
def env():
    """Initializes standard environment dimensions."""
    return GridCastEnvironment(steps=12, width=30, height=30)


@pytest.fixture
def clean_block(env):
    """Generates a raw block filled entirely with Base Solvent (Z=0)."""
    block = torch.zeros((env.T, env.C, env.X, env.Y), dtype=torch.float32)
    block[:, 0] = 1.0  # System completely filled with base solvent
    return block


def test_toroidal_shift_wrapping(env, clean_block):
    """Verifies that fluid rolling past boundaries wraps around seamlessly."""
    # Place a concentrated point of Chemical Dye 1 at coordinate (0, 0)
    clean_block[0, 1, 0, 0] = 1.0
    clean_block[0, 0, 0, 0] = 0.0  # Displace solvent
    
    # Shift left by 1 (-1) and up by 1 (-1) -> should wrap to the bottom-right corner
    shifted_block = shift_toroidal(clean_block, t=0, channel_idx=1, dx=-1, dy=-1)
    
    # Assert old position is empty and new position contains the wrapped dye
    assert shifted_block[0, 1, 0, 0] == 0.0
    assert shifted_block[0, 1, env.X - 1, env.Y - 1] == 1.0


def test_hydraulic_displacement_collision(env, clean_block):
    """Ensures Structural Pipes (Z=10) halt fluid flow and preserve shape boundaries."""
    # Set up a solid wall (Z=10) at column index 5
    clean_block[0, 10, 5, :] = 1.0
    
    # Place a block of Chemical Dye 2 at column index 4 (right next to the wall)
    clean_block[0, 2, 4, 10:15] = 1.0
    clean_block[0, 0, 4, 10:15] = 0.0  # Clear carrier solvent
    
    # Attempt to pump the fluid to the right by dx=1 (straight into the wall)
    processed_block = apply_hydraulic_displacement(clean_block, t=0, dx=1, dy=0, shove_solvent=True)
    
    # Invariant checks:
    # 1. The fluid must NOT have penetrated column 5 because of the wall mask
    assert (processed_block[0, 2, 5, 10:15] == 0.0).all()
    # 2. The fluid must have stayed in place at column 4 due to the collision bounce
    assert (processed_block[0, 2, 4, 10:15] == 1.0).all()
    
    # Verify the environment still passes hydraulic containment validation
    assert env.check_hydraulic_leak(processed_block) is True


def test_pressure_slope_injection(env, clean_block):
    """Validates that continuous linear gradient fields are generated smoothly."""
    # Inject a downward-pointing pressure field into the first step
    block_with_gradient = inject_pressure_slope(clean_block, t=0, direction="down")
    
    gradient_layer = block_with_gradient[0, 14]
    
    # Check that gradient values increase down the rows (X dimension)
    assert gradient_layer[0, 0] == 0.0
    assert gradient_layer[-1, 0] == 1.0
    # Check that it remains uniform across columns (Y dimension)
    assert torch.equal(gradient_layer[5, :], gradient_layer[5, :].clone().fill_(gradient_layer[5, 0].item()))