import torch
import pytest
from gridcast.core.environment import GridCastEnvironment
from gridcast.utils.visualizer import GridCastVisualizer

INPUT_GRID = torch.tensor([
    [2, 2, 0, 0],
    [2, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
], dtype=torch.long)


@pytest.fixture
def drained_env():
    env = GridCastEnvironment(steps=12, width=4, height=4)
    env.reset(INPUT_GRID)
    env.execute('drain', layer_idx=2) 
    return env


@pytest.fixture
def moved_env():
    """Isolates the translation phase within the transient layer."""
    env = GridCastEnvironment(steps=12, width=4, height=4)
    env.reset(INPUT_GRID)
    env.execute('drain', layer_idx=2)
    env.execute('move', layer_idx=0, direction='right', steps=2)
    env.execute('paint', layer_idx=3)
    return env


@pytest.fixture
def painted_env():
    env = GridCastEnvironment(steps=12, width=4, height=4)
    env.reset(INPUT_GRID)
    env.execute('drain', layer_idx=2)
    env.execute('paint', layer_idx=3)
    return env


def test_full_drain_sequence(drained_env):
    viz = GridCastVisualizer(drained_env)
    viz.show(drained_env.state, steps=[0, 1, 6, 11], mode="channels", title="Channels")
    viz.show(
        drained_env.state,
        steps=[0, 3, 6, 9, 11],
        mode="sequence",
        title="Drain sequence",
        save_path="full_analysis.html",
    )

    original = drained_env.get_original_input()
    output_final = drained_env.get_output_at_step(11)

    assert not torch.equal(original, output_final)
    assert not torch.equal(original[0:2, 0:2], output_final[0:2, 0:2])


def test_full_move_sequence(moved_env):
    viz = GridCastVisualizer(moved_env)
    viz.show(moved_env.state, steps=[0, 3, 6, 9, 11], mode="sequence", title="Move sequence", save_path="full_move_sequence.html")
    
    output_final = moved_env.get_output_at_step(11)
    assert torch.all(output_final[0:2, 0:2] == 0)
    assert torch.all(output_final[0:2, 2:4] == 3)


def test_full_paint_sequence(painted_env):
    viz = GridCastVisualizer(painted_env)
    viz.show(painted_env.state, steps=[0, 1, 6, 11], mode="channels", title="Channels")
    viz.show(
        painted_env.state,
        steps=[0, 3, 6, 9, 11],
        mode="sequence",
        title="Paint sequence",
        save_path="full_paint_analysis.html",
    )
    original = painted_env.get_original_input()
    output_final = painted_env.get_output_at_step(11)

    assert not torch.equal(original, output_final)
    assert torch.all(output_final[0:2, 0:2] == 0)
    
    expected_destination = torch.tensor([
        [3, 3],
        [3, 3]
    ], dtype=torch.long)
    torch.testing.assert_close(output_final[0:2, 2:4], expected_destination)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])