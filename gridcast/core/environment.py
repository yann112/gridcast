import torch
from typing import Tuple, List, NamedTuple, Dict
from gridcast.core.encoder import SimpleEncoder

class CommandRecord(NamedTuple):
    operation_type: str
    layer_idx: int
    direction: str = 'right'
    steps: int = 0

class BaseCommand:
    def execute(self, block: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DrainCommand(BaseCommand):
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def execute(self, block: torch.Tensor) -> torch.Tensor:
        block = block.clone()
        target_layer_idx = self.layer_idx

        src_layer = block[target_layer_idx:target_layer_idx+1, :, :]
        trans_layer = block[13:14, :, :]
        bg_layer = block[0:1, :, :]

        is_trigger = (src_layer == 1.0)

        src_layer = torch.where(is_trigger, torch.zeros_like(src_layer), src_layer)
        bg_layer = torch.where(is_trigger, torch.ones_like(bg_layer), bg_layer)
        trans_layer = torch.where(is_trigger, torch.ones_like(trans_layer), trans_layer)

        block[0:1, :, :] = bg_layer
        block[target_layer_idx:target_layer_idx+1, :, :] = src_layer
        block[13:14, :, :] = trans_layer

        return block

class PaintCommand(BaseCommand):
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def execute(self, block: torch.Tensor) -> torch.Tensor:
        block = block.clone()
        target_layer_idx = self.layer_idx

        color_channels = block[0:10, :, :]
        trans_layer = block[13:14, :, :]
        src_layer = color_channels[target_layer_idx:target_layer_idx+1, :, :]

        is_trigger = (src_layer == 0.0) & (trans_layer == 1.0)

        if not is_trigger.any():
            return block

        color_channels = torch.where(is_trigger, torch.zeros_like(color_channels), color_channels)
        
        color_channels[target_layer_idx:target_layer_idx+1, :, :] = torch.where(
            is_trigger,
            torch.ones_like(src_layer),
            color_channels[target_layer_idx:target_layer_idx+1, :, :]
        )

        trans_layer = torch.where(is_trigger, torch.zeros_like(trans_layer), trans_layer)

        block[0:10, :, :] = color_channels
        block[13:14, :, :] = trans_layer

        return block

class MoveCommand(BaseCommand):
    def __init__(self, layer_idx: int, direction: str = 'right', steps: int = 1):
        self.layer_idx = layer_idx
        self.direction = direction
        self.steps = steps

    def execute(self, block: torch.Tensor) -> torch.Tensor:
        block = block.clone()
        trans_layer = block[13:14, :, :]
        shifted_trans = torch.zeros_like(trans_layer)
        _, X, Y = trans_layer.shape

        dx, dy = 0, 0
        if self.direction == 'down':    dx = self.steps
        elif self.direction == 'up':    dx = -self.steps
        elif self.direction == 'right': dy = self.steps
        elif self.direction == 'left':  dy = -self.steps

        st_x, en_x = max(0, dx), min(X, X + dx)
        st_y, en_y = max(0, dy), min(Y, Y + dy)
        src_st_x, src_en_x = max(0, -dx), min(X, X - dx)
        src_st_y, src_en_y = max(0, -dy), min(Y, Y - dy)

        if src_en_x > src_st_x and src_en_y > src_st_y:
            shifted_trans[:, st_x:en_x, st_y:en_y] = trans_layer[:, src_st_x:src_en_x, src_st_y:src_en_y]

        block[13:14, :, :] = shifted_trans
        return block


def run_operation_sequence(
    initial_state: torch.Tensor,
    history: List[CommandRecord],
    registry: Dict[str, type]
) -> List[Tuple[int, torch.Tensor]]:
    """
    Generates discrete keyframes. Each sequential command automatically 
    increments the timeline step uniformly by 1.
    """
    current_state = initial_state.clone()
    keyframes = [(0, current_state.clone())]
    cumulative_time = 0

    for record in history:
        cmd_class = registry[record.operation_type]
        
        if record.operation_type == 'move':
            command = cmd_class(record.layer_idx, direction=record.direction, steps=record.steps)
        else:
            command = cmd_class(record.layer_idx)
            
        current_state = command.execute(current_state)

        walls = current_state[10:11, :, :]
        current_state = torch.clamp(current_state, 0.0, 1.0)
        current_state[13:14] = current_state[13:14] * (1.0 - walls)

        # Every action takes exactly 1 uniform relational time slot
        cumulative_time += 1
        keyframes.append((cumulative_time, current_state.clone()))

    return keyframes


def interpolate_to_buffer(
    keyframes: List[Tuple[int, torch.Tensor]],
    T: int
) -> torch.Tensor:
    C, X, Y = keyframes[0][1].shape
    state = torch.zeros((T, C, X, Y), dtype=torch.float32)

    if len(keyframes) == 1:
        state[:] = keyframes[0][1].clone()
        return state

    actual_duration = keyframes[-1][0] if keyframes[-1][0] > 0 else 1
    buffer_positions = [
        int(t * (T - 1) / actual_duration) for t, _ in keyframes
    ]

    for idx in range(1, len(buffer_positions)):
        if buffer_positions[idx] <= buffer_positions[idx - 1]:
            buffer_positions[idx] = min(buffer_positions[idx - 1] + 1, T - 1)

    for buf_pos, (_, frame) in zip(buffer_positions, keyframes):
        state[buf_pos] = frame.clone()

    for i in range(len(buffer_positions) - 1):
        start = buffer_positions[i]
        end = buffer_positions[i + 1]
        for t in range(start + 1, end):
            alpha = (t - start) / (end - start)
            state[t] = (1 - alpha) * state[start] + alpha * state[end]

    return state


def set_global_progress_timer(state: torch.Tensor) -> torch.Tensor:
    T = state.shape[0]
    for step in range(T):
        progress = step / (T - 1) if T > 1 else 0.0
        state[step, 15, :, :] = progress
    return state


class GridCastEnvironment:
    _command_classes: Dict[str, type] = {
        'drain': DrainCommand,
        'move': MoveCommand,
        'paint': PaintCommand,
    }

    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.T = steps
        self.C = 16
        self.X = width
        self.Y = height
        self.state: torch.Tensor | None = None
        self.original_input: torch.Tensor | None = None
        self.command_history: List[CommandRecord] = []
        self.encoder = SimpleEncoder(steps, width, height)
        self.initial_frame: torch.Tensor | None = None

    def reset(self, color_grid: torch.Tensor, encoder=None) -> torch.Tensor:
        color_grid = color_grid.long()
        if color_grid.dim() == 2:
            color_grid = color_grid.unsqueeze(0)

        encoder = encoder or self.encoder
        encoded_full = encoder.encode(color_grid)
        self.initial_frame = encoded_full[0].clone() 
        self.original_input = color_grid.squeeze(0).clone()
        self.command_history = []
        
        self.state = torch.zeros((self.T, self.C, self.X, self.Y), dtype=torch.float32)
        self.state[:] = self.initial_frame.clone()
        self.state = set_global_progress_timer(self.state)
        
        return encoded_full

    def _register_command(self, name: str, layer_idx: int, direction: str = 'right', steps: int = 0):
        if name not in self._command_classes:
            raise ValueError(f"Unknown command: {name}")
        self.command_history.append(CommandRecord(name, layer_idx, direction, steps))

    def execute(self, name: str, layer_idx: int, direction: str = 'right', steps: int = 0):
        if self.initial_frame is None:
            raise RuntimeError("Call reset() first.")
        self._register_command(name, layer_idx, direction, steps)
        self._run_pipeline()

    def _run_pipeline(self):
        keyframes = run_operation_sequence(self.initial_frame, self.command_history, self._command_classes)
        self.state = interpolate_to_buffer(keyframes, self.T)
        self.state = set_global_progress_timer(self.state)

    def get_original_input(self) -> torch.Tensor:
        if self.original_input is None:
            raise RuntimeError("Call reset() first.")
        return self.original_input

    def get_output_at_step(self, step: int) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        return torch.argmax(self.state[step, :10, :, :], dim=0)