import torch
import pytest
from gridcast.core.environment import GridCastEnvironment


GRID_3X3 = torch.tensor([
    [1, 2, 1],
    [0, 1, 2],
    [2, 0, 1],
])

GRID_4X4 = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [2, 2, 3, 3],
    [2, 2, 3, 3],
])


@pytest.fixture
def env3():
    e = GridCastEnvironment(steps=4, width=3, height=3)
    e.reset(GRID_3X3)
    return e

@pytest.fixture
def env4():
    e = GridCastEnvironment(steps=6, width=4, height=4)
    e.reset(GRID_4X4)
    return e


# --- Reset ---

def test_state_shape(env3):
    assert env3.state.shape == (4, 16, 3, 3)

def test_color_channels_match_input(env3):
    for color in range(10):
        expected = (GRID_3X3 == color).float()
        assert torch.allclose(env3.state[0, color], expected)

def test_non_color_channels_zero_after_reset(env3):
    assert torch.all(env3.state[0, 10:] == 0.0)

def test_original_input_preserved_after_drain(env3):
    env3.drain(1, step=0)
    assert torch.equal(env3.get_original_input(), GRID_3X3)


# --- Drain + sequence ---

def test_drain_changes_state(env3):
    state_before = env3.state.clone()
    env3.drain(1, step=0)
    assert not torch.equal(env3.state, state_before)

def test_drain_output_differs_from_input(env3):
    env3.drain(1, step=0)
    output = env3.get_output_at_step(env3.T - 1)
    assert not torch.equal(env3.get_original_input(), output)

def test_get_output_shape(env3):
    env3.drain(1, step=0)
    output = env3.get_output_at_step(0)
    assert output.shape == (3, 3)


# --- Edge cases ---

def test_state_values_bounded_after_drain(env3):
    env3.drain(1, step=0)
    assert torch.all(env3.state >= 0.0) and torch.all(env3.state <= 1.0)

def test_wall_does_not_break_physics(env3):
    env3.state[0, 10, 1, 1] = 1.0  # wall at centre cell
    env3.drain(1, step=0)
    assert env3.state.shape == (4, 16, 3, 3)
    assert torch.all(env3.state >= 0.0) and torch.all(env3.state <= 1.0)

def test_drain_empty_layer_does_not_corrupt_state(env3):
    # Layer 9 has no cells in GRID_3X3 — should not corrupt other channels
    env3.drain(9, step=0)
    for color in [1, 2]:  # these exist in GRID_3X3
        assert env3.state[:, color].sum() > 0, f"Channel {color} was wiped unexpectedly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])