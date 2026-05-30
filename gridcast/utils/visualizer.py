import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class GridCastVisualizer:
    """Visualizer for GridCast environment tensors."""

    CHANNEL_NAMES = {
        0: "BG", 1: "L1", 2: "L2", 3: "L3", 4: "L4",
        5: "L5", 6: "L6", 7: "L7", 8: "L8", 9: "L9",
        10: "WALL", 11: "U11", 12: "U12", 13: "TRANS",
        14: "CMD", 15: "TIMER",
    }

    _GRAYSCALE = [[0, "black"], [0.5, "gray"], [1, "white"]]

    def __init__(self, env):
        self.env = env

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(
        self,
        state,
        steps: list[int] | None = None,
        mode: str = "channels",
        title: str = "",
        save_path: str | None = None,
    ) -> go.Figure:
        """
        Unified display method. Always auto-renders to screen.

        Args:
            state:      The environment state tensor (steps, channels, H, W).
            steps:      Which time steps to include. Defaults to all.
            mode:       "channels" — 4×4 grayscale channel grid (default)
                        "sequence" — decoded input vs output across steps
            title:      Optional figure title.
            save_path:  If given, also saves an HTML file at this path.

        Returns:
            The Plotly Figure (useful for further customisation or pytest checks).
        """
        steps = steps if steps is not None else list(range(state.shape[0]))

        if mode == "channels":
            fig = self._channels_figure(state, steps, title)
        elif mode == "sequence":
            fig = self._sequence_figure(steps, title)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'channels' or 'sequence'.")

        fig.show()
        if save_path:
            fig.write_html(save_path)

        return fig

    def decode(self, state, step: int = 0) -> torch.Tensor:
        """Decode a state tensor at one step to a 2D colour-index grid."""
        return torch.argmax(state[step, :10, :, :], dim=0)

    # ------------------------------------------------------------------
    # Private figure builders
    # ------------------------------------------------------------------

    def _channels_figure(self, state, steps: list[int], title: str) -> go.Figure:
        """4×4 grayscale channel grid, one block per requested step."""
        n_steps = len(steps)
        # 4 data rows per step + 1 spacer row between steps
        total_rows = 4 * n_steps + max(n_steps - 1, 0)

        subplot_titles = []
        for step in steps:
            subplot_titles += [f"Step {step} — {self.CHANNEL_NAMES.get(i, f'C{i}')}" for i in range(16)]
            if step != steps[-1]:
                subplot_titles += [""] * 4  # spacer row

        fig = make_subplots(
            rows=total_rows,
            cols=4,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03,
            horizontal_spacing=0.06,
        )

        for step_idx, step in enumerate(steps):
            base_row = step_idx * 5  # 4 data rows + 1 spacer
            for ch in range(16):
                channel_data = state[step, ch, :, :]
                h, w = channel_data.shape
                fig.add_trace(
                    go.Heatmap(
                        z=channel_data.numpy(),
                        colorscale=self._GRAYSCALE,
                        zmin=0, zmax=1,
                        showscale=False,
                        hovertemplate=(
                            f"Channel: {self.CHANNEL_NAMES.get(ch, f'C{ch}')}"
                            "<br>Value: %{z}<extra></extra>"
                        ),
                    ),
                    row=base_row + (ch // 4) + 1,
                    col=(ch % 4) + 1,
                )

        for r in range(1, total_rows + 1):
            for c in range(1, 5):
                fig.update_yaxes(autorange="reversed", row=r, col=c)

        fig.update_layout(
            title_text=title or "Channel analysis",
            height=220 * total_rows,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="white",
        )
        return fig

    # RGB colour for each channel index 0-9 (ARC palette).
    # Channel 0 (background) composited last so it fills empty cells.
    _ARC_COLORS = {
        0: (220, 220, 220),   # background -> light grey
        1: ( 30, 100, 200),   # blue
        2: (220,  50,  50),   # red
        3: ( 50, 180,  80),   # green
        4: (230, 200,  20),   # yellow
        5: (150, 150, 150),   # grey
        6: (220,  80, 180),   # magenta
        7: (230, 130,  20),   # orange
        8: ( 60, 190, 210),   # cyan
        9: (140,  40, 140),   # purple
    }

    _CHANNEL_NAMES = {
        0: "BG", 1: "blue", 2: "red", 3: "green", 4: "yellow",
        5: "grey", 6: "magenta", 7: "orange", 8: "cyan", 9: "purple",
    }

    def _composite_cell_data(self, state, step: int):
        """
        Returns:
            colorscale  : list of [norm, "rgb(r,g,b)"] stops (one per unique composited colour)
            z           : (H, W) float array — index into colorscale (row * W + col, normalised)
            customdata  : (H, W, 10) float array — raw alpha per channel per cell
            colorscale_rgb : (H*W, 3) uint8 array of composited RGB per cell
        """
        import numpy as np

        h, w = state.shape[2], state.shape[3]
        canvas = np.ones((h, w, 3), dtype=np.float32)  # white

        composite_order = list(range(1, 10)) + [0]
        for ch in composite_order:
            r, g, b = [v / 255.0 for v in self._ARC_COLORS[ch]]
            alpha = state[step, ch, :, :].numpy().astype(np.float32)
            a = alpha[..., None]
            canvas = a * np.array([r, g, b]) + (1 - a) * canvas

        rgb_uint8 = (canvas * 255).clip(0, 255).astype(np.uint8)  # (H, W, 3)

        # Build a per-cell colorscale for go.Heatmap
        # z is just the flat cell index; each cell gets its own colour stop
        n_cells = h * w
        z = np.arange(n_cells, dtype=np.float32).reshape(h, w)
        z_norm = z / max(n_cells - 1, 1)

        colorscale = []
        rgb_flat = rgb_uint8.reshape(-1, 3)
        for i, (r, g, b) in enumerate(rgb_flat):
            colorscale.append([float(i) / max(n_cells - 1, 1), f"rgb({r},{g},{b})"])

        # customdata: (H, W, 10) — channel alphas
        alphas = state[step, :10, :, :].numpy()          # (10, H, W)
        customdata = alphas.transpose(1, 2, 0)            # (H, W, 10)

        return colorscale, z, customdata

    def _hover_template(self, step: int) -> str:
        """Build hovertemplate showing position, step, and all 10 channel alphas."""
        lines = [f"pos: (%{{x}}, %{{y}})  step: {step}"]
        for ch in range(10):
            name = self._CHANNEL_NAMES[ch]
            lines.append(f"{name}: %{{customdata[{ch}]:.3f}}")
        return "<br>".join(lines) + "<extra></extra>"

    def _sequence_figure(self, steps: list[int], title: str) -> go.Figure:
        """One column per step, single row. Hover shows all 10 channel alphas per cell."""
        n = len(steps)

        fig = make_subplots(
            rows=1,
            cols=n,
            subplot_titles=[f"Step {s}" for s in steps],
            horizontal_spacing=max(0.04, 0.12 / n),
        )

        for idx, step in enumerate(steps):
            colorscale, z, customdata = self._composite_cell_data(self.env.state, step)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    colorscale=colorscale,
                    customdata=customdata,
                    hovertemplate=self._hover_template(step),
                    showscale=False,
                    zmin=0,
                    zmax=max(z.size - 1, 1),
                ),
                row=1,
                col=idx + 1,
            )
            fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=idx + 1)
            fig.update_xaxes(showticklabels=False, row=1, col=idx + 1)

        cell_px = 64
        h_cells, w_cells = self.env.state.shape[2], self.env.state.shape[3]
        fig.update_layout(
            title_text=title or "Drain sequence",
            height=cell_px * h_cells + 120,
            width=max(600, cell_px * w_cells * n + 80),
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
        )
        return fig