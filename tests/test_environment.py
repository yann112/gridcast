# -*- coding: utf-8 -*-
"""
GridCast - Hydraulic Environment Unit Tests
Validates mass containment, pressure stabilization, and drain vent mechanics.
"""
import pytest
import torch
from gridcast.core.environment import GridCastEnvironment


@pytest.fixture
def env():
    """Initializes a standard 12-step, 30x30 hydraulic environment."""
    return GridCastEnvironment(steps=12, width=30, height=30)


@pytest.fixture
def clean_block(env):
    """Generates a raw block filled entirely with Carrier Solvent (Z=0)."""
    block = torch.zeros((env.T, env.C, env.X, env.Y), dtype=torch.float32)
    block[:, 0] = 1.0  # System completely filled with base solvent
    return block


def test_initial_state_integrity(env, clean_block):
    """Ensures a freshly filled solvent block passes the leakage test."""
    assert env.check_hydraulic_leak(clean_block) is True


def test_pressure_drift_stabilization(env, clean_block):
    """Verifies that L1 normalization corrects floating-point pressure drifts."""
    # Simulate a drift where solvent evaporates slightly (0.95 instead of 1.0)
    # and a chemical dye leaks in weakly (0.1) -> Total = 1.05 (Overpressure)
    clean_block[0, 0, 5, 5] = 0.95
    clean_block[0, 1, 5, 5] = 0.1  # Dye index 1
    
    # The leak check must catch this pressure imbalance
    assert env.check_hydraulic_leak(clean_block) is False
    
    # Enforce containment to seal the drift
    sealed_block = env.enforce_containment(clean_block)
    
    # Total fluid volume must be exactly 1.0 again
    assert env.check_hydraulic_leak(sealed_block) is True
    assert torch.allclose(sealed_block[0, 0:10, 5, 5].sum(), torch.tensor(1.0))


def test_drain_vent_annihilation(env, clean_block):
    """Verifies that active Drain Vents (Z=15) flush dyes and trigger solvent backfill."""
    # Open a drain vent at node (10, 10) for the first time step
    clean_block[0, 15, 10, 10] = 1.0
    # Force an illegal chemical dye presence into that same node
    clean_block[0, 3, 10, 10] = 1.0  # Dye index 3
    clean_block[0, 0, 10, 10] = 0.0  # Remove solvent to accommodate dye
    
    # The leak detector must immediately flag the dye in the drain
    assert env.check_hydraulic_leak(clean_block) is False
    
    # Process the block through the containment systems
    processed_block = env.enforce_containment(clean_block)
    
    # Invariant checks post-containment
    assert env.check_hydraulic_leak(processed_block) is True
    assert processed_block[0, 3, 10, 10] == 0.0  # Dye must be completely flushed
    assert processed_block[0, 0, 10, 10] == 1.0  # Solvent must have backfilled the space


def test_invalid_tensor_shape_assertion(env):
    """Ensures the environment rejects incorrectly shaped structural tensors."""
    invalid_block = torch.zeros((12, 16, 10, 10))  # Wrong width/height
    with pytest.raises(AssertionError):
        env.check_hydraulic_leak(invalid_block)