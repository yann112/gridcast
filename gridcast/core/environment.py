import torch
from typing import Tuple, List, NamedTuple, Dict, Any
from gridcast.core.encoder import SimpleEncoder

class CommandHistory(NamedTuple):
    step: int
    layer_idx: int
    operation_type: str  # 'drain', 'copy', etc.

class BaseCommand:
    """Base class for all commands."""
    def execute(self, env: 'GridCastEnvironment', step: int = 0):
        """Execute the command without completion."""
        raise NotImplementedError

    def execute_single_step_scaled(self, block: torch.Tensor, time_scale: float = 1.0) -> torch.Tensor:
        """Execute command-specific physics logic with time scaling."""
        raise NotImplementedError

class DrainCommand(BaseCommand):
    """Drain command implementation."""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
    
    def execute(self, env: 'GridCastEnvironment', step: int = 0):
        """Execute the drain command without completion."""
        # Record the command in the history for supervision
        env.command_history.append(CommandHistory(step, self.layer_idx, 'drain'))

    def execute_single_step_scaled(self, block: torch.Tensor, time_scale: float = 1.0) -> torch.Tensor:
        """Execute drain logic with time scaling."""
        block = block.clone()
        
        # Use the target layer from the command, not from the command layer
        target_layer_idx = self.layer_idx
        target_duration = float(target_layer_idx)
        
        cmd_layer = block[14:15, :, :]  # Keep command layer unchanged
        time_layer = block[15:16, :, :]
        src_layer = block[target_layer_idx:target_layer_idx+1, :, :]
        trans_layer = block[13:14, :, :]
        
        # 1. TRIGGER: check if this operation should run based on history
        # For now, assume it should run if source is full and timer is 0
        is_trigger = (src_layer == 1.0) & (time_layer == 0.0)
        
        # 2. INIT HALF-STATE
        src_layer = torch.where(is_trigger, torch.full_like(src_layer, 0.5), src_layer)
        trans_layer = torch.where(is_trigger, torch.full_like(trans_layer, 0.5), trans_layer)
        
        # 3. ACTIVE MASK & 4. ADVANCE TIMER (scaled)
        active_mask = (trans_layer == 0.5)
        time_layer = torch.where(active_mask, time_layer + time_scale, time_layer)
        
        # 5. COMPLETION CHECK (physics time vs duration)
        is_done = active_mask & (time_layer >= target_duration)
        
        # 6. FINALIZE TRANSFER
        src_layer = torch.where(is_done, torch.full_like(src_layer, 0.0), src_layer)
        trans_layer = torch.where(is_done, torch.full_like(trans_layer, 1.0), trans_layer)
        
        # 7. RESET TIMER (but keep command layer unchanged for supervision)
        time_layer = torch.where(is_done, torch.full_like(time_layer, 0.0), time_layer)
        
        # Write back
        block[target_layer_idx:target_layer_idx+1, :, :] = src_layer
        block[13:14, :, :] = trans_layer
        # block[14:15, :, :] = cmd_layer  # Keep original commands for supervision
        block[15:16, :, :] = time_layer  # Individual operation timers
        
        return block

class GridCastEnvironment:
    # Command registry
    _command_classes = {
        'drain': DrainCommand,
    }

    def __init__(self, steps: int = 12, width: int = 30, height: int = 30):
        self.T = steps  # Anthropomorphic steps (fixed temporal buffer)
        self.C = 16     # Channels
        self.X = width  # Width
        self.Y = height # Height
        self.state: torch.Tensor | None = None
        self.original_input: torch.Tensor | None = None
        self.command_history: List[CommandHistory] = []
        self.encoder = SimpleEncoder(steps, width, height)

    def reset(self, color_grid: torch.Tensor, encoder=None):
        """Initialize environment with a (X, Y) integer color grid (values 0..9)."""
        color_grid = color_grid.long()
        if color_grid.dim() == 2:
            color_grid = color_grid.unsqueeze(0)  # (1, X, Y)

        self.state = torch.zeros((self.T, self.C, self.X, self.Y), dtype=torch.float32)
        encoder = encoder or self.encoder

        # Use encoder to properly format the input
        encoded_state = encoder.encode(color_grid)
        self.state = encoded_state
                    
        self.original_input = color_grid.squeeze(0).clone()
        self.command_history = []  # Reset command history
        return self.state

    def execute_commands(self, commands: List[Dict[str, Any]]):
        """Execute a list of commands with parameters."""
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        
        for cmd_dict in commands:
            cmd_name = cmd_dict['name']
            step = cmd_dict.get('step', 0)
            params = {k: v for k, v in cmd_dict.items() if k not in ['name', 'step']}
            
            if cmd_name not in self._command_classes:
                raise ValueError(f"Unknown command: {cmd_name}")
            
            command_class = self._command_classes[cmd_name]
            command = command_class(**params)
            command.execute(self, step)
        
        # Complete the sequence after all commands are injected
        self.execute_all_commands_and_complete_sequence()

    def drain(self, layer_idx: int, step: int = 0):
        """User-friendly method: drain layer."""
        self.execute_commands([{'name': 'drain', 'layer_idx': layer_idx, 'step': step}])

    def execute_all_commands_and_complete_sequence(self):
        """Complete the sequence: run operations, interpolate, apply laws."""
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        
        if not self.command_history:
            return  # No commands to execute
        
        # Run operations to completion
        actual_duration = self.run_operation_sequence()
        
        # Interpolate to buffer
        self.interpolate_to_buffer(actual_duration)
        
        # Apply physics laws
        self.apply_physics_laws()
        
        # Set global progress timer
        self.set_global_progress_timer()

    def set_global_progress_timer(self):
        """Set timer layer to represent global progress through sequence."""
        for step in range(self.T):
            # Normalize step to [0, 1] range
            progress = step / (self.T - 1) if self.T > 1 else 0.0
            # Set entire timer layer to progress value
            self.state[step, 15, :, :] = torch.full_like(self.state[step, 15, :, :], progress)

    def run_operation_sequence(self) -> int:
        """
        Run the actual operation sequence and return the actual duration.
        Returns: actual number of steps taken for all operations
        """
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        
        # Run operations until all are completed
        max_steps = self.T  # Allow up to full buffer length
        actual_steps = 1  # Start from step 1
        
        for actual_steps in range(1, max_steps):
            # Copy from previous state
            self.state[actual_steps] = self.state[actual_steps-1].clone()
            
            # Apply physics for all commands
            for hist in self.command_history:
                if hist.operation_type in self._command_classes:
                    command_class = self._command_classes[hist.operation_type]
                    command = command_class(hist.layer_idx)
                    self.state[actual_steps] = command.execute_single_step_scaled(
                        self.state[actual_steps], time_scale=1.0
                    )
            
            self.state[actual_steps] = self._enforce_containment_single_step(self.state[actual_steps])
            
            # Check if all operations have completed
            # Check based on the expected completion time for each command
            all_completed = True
            for hist in self.command_history:
                target_duration = float(hist.layer_idx)
                timer_layer = self.state[actual_steps, 15:16, :, :]
                # Check if timer has reached the target duration for this command's target layer
                src_layer = self.state[actual_steps, hist.layer_idx:hist.layer_idx+1, :, :]
                if (src_layer == 0.5).any():  # If any cell is still in half-state
                    all_completed = False
                    break
            
            if all_completed:
                # All operations completed
                break
        
        return actual_steps

    def interpolate_to_buffer(self, actual_duration: int):
        """
        Interpolate the actual operation sequence to fit the full temporal buffer.
        Maps actual steps to buffer steps with interpolation.
        """
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        
        # Calculate mapping: actual steps -> buffer steps
        actual_keyframes = list(range(actual_duration + 1))  # [0, 1, 2, 3] for duration=3
        buffer_positions = [int(t * (self.T - 1) / actual_duration) for t in actual_keyframes]
        
        # Create interpolated sequence
        interpolated_state = self.state.clone()
        
        # Fill in the keyframe states
        for i, buf_pos in enumerate(buffer_positions):
            if i < actual_duration + 1:
                # Use the actual computed state for this keyframe
                actual_pos = min(i, self.T - 1)
                interpolated_state[buf_pos] = self.state[actual_pos].clone()
        
        # Linear interpolation between keyframes
        for i in range(len(buffer_positions) - 1):
            start_pos = buffer_positions[i]
            end_pos = buffer_positions[i + 1]
            
            for t in range(start_pos + 1, end_pos):
                alpha = (t - start_pos) / (end_pos - start_pos)
                interpolated_state[t] = (
                    (1 - alpha) * interpolated_state[start_pos] + 
                    alpha * interpolated_state[end_pos]
                )
        
        self.state = interpolated_state

    def apply_physics_laws(self):
        """Apply post-operation physics laws to all steps."""
        for step in range(self.T):
            block = self.state[step].clone()
            
            # Apply background filling (channel 0 fills empty spaces)
            color_sum = block[:10, :, :].sum(dim=0)  # Sum all color channels
            empty_mask = (color_sum == 0)  # Where no colors exist
            block[0, :, :] = torch.where(empty_mask, torch.ones_like(block[0, :, :]), block[0, :, :])
            
            # Apply mutual exclusion (only one color per cell)
            color_channels = block[:10, :, :]
            max_values, max_indices = torch.max(color_channels, dim=0)
            new_colors = torch.zeros_like(color_channels)
            for c in range(10):
                mask = (max_indices == c)
                new_colors[c, :, :] = torch.where(mask, color_channels[c, :, :], torch.zeros_like(color_channels[c, :, :]))
            
            block[:10, :, :] = new_colors
            
            # Clamp all values to [0, 1]
            block = torch.clamp(block, 0.0, 1.0)
            
            self.state[step] = block

    def _enforce_containment_single_step(self, block: torch.Tensor) -> torch.Tensor:
        """Apply containment to a single time slice."""
        walls = block[10:11, :, :]
        block = torch.clamp(block, 0.0, 1.0)
        block[13:14] = block[13:14] * (1.0 - walls)
        return block

    def get_original_input(self) -> torch.Tensor:
        """Return the preserved original input grid."""
        if self.original_input is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.original_input

    def get_output_at_step(self, step: int) -> torch.Tensor:
        """Get the decoded output at a specific anthropic step."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Decode the color channels (0-9) for this step
        color_channels = self.state[step, :10, :, :]
        return torch.argmax(color_channels, dim=0)