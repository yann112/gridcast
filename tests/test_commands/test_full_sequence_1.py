import torch
import pytest
from gridcast.core.environment import GridCastEnvironment
from gridcast.utils.visualizer import GridCastVisualizer


INPUT_GRID = torch.tensor([
    [2, 2, 0, 0],
    [2, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])


@pytest.fixture
def drained_env():
    env = GridCastEnvironment(steps=12, width=4, height=4)
    env.reset(INPUT_GRID)
    env.drain(2, step=0)
    return env


def test_full_drain_sequence(drained_env):
    viz = GridCastVisualizer(drained_env)

    # Channel view — one 4×4 block per step, auto-displays
    viz.show(drained_env.state, steps=[0, 1, 6, 11], mode="channels", title="Channels")

    # Sequence view — input vs decoded output across steps
    viz.show(
        drained_env.state,
        steps=[0, 3, 6, 9, 11],
        mode="sequence",
        title="Drain sequence",
        save_path="full_analysis.html",
    )

    original = drained_env.get_original_input()
    output_final = drained_env.get_output_at_step(11)

    assert not torch.equal(original, output_final), \
        "Final output should differ from input after drain"

    assert not torch.equal(original[0:2, 0:2], output_final[0:2, 0:2]), \
        "Drained region should change from original"


if __name__ == "__main__":
    pytest.main([__file__])