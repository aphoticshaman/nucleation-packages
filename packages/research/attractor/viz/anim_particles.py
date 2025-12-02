"""
anim_particles.py

Particle swarm animation for Great Attractor visualization.

Provides:
- Real-time animation of particle dynamics
- Potential field background rendering
- Geodesic flow overlay (optional)
- Video export capabilities (MP4, GIF)

Requires matplotlib with animation support.
For video export, ffmpeg must be installed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


Array = NDArray[np.float64]


@dataclass
class AnimationConfig:
    """Configuration for particle animation."""

    # Figure
    figsize: Tuple[int, int] = (10, 8)
    xlim: Tuple[float, float] = (-6.0, 6.0)
    ylim: Tuple[float, float] = (-6.0, 6.0)
    dpi: int = 100

    # Animation
    interval: int = 50  # milliseconds between frames
    blit: bool = True
    repeat: bool = True

    # Particles
    particle_size: int = 20
    particle_color: str = "cyan"
    particle_alpha: float = 0.7
    trail_length: int = 10
    show_trails: bool = True
    trail_alpha: float = 0.3

    # Potential background
    show_potential: bool = True
    potential_cmap: str = "magma"
    potential_alpha: float = 0.6
    potential_resolution: int = 50
    potential_levels: int = 30

    # Attractor marker
    show_attractor: bool = True
    attractor_color: str = "yellow"
    attractor_size: int = 200

    # Centroid
    show_centroid: bool = True
    centroid_color: str = "white"
    centroid_size: int = 100

    # Output
    save_path: Optional[str] = None
    fps: int = 20
    writer: str = "ffmpeg"  # or 'pillow' for GIF


class ParticleAnimator:
    """
    Animate particle swarm dynamics.

    Can operate in two modes:
    1. From position history: animate pre-computed positions
    2. Live simulation: step simulator each frame (not implemented here)

    Parameters
    ----------
    position_history : list of arrays
        Each array has shape (n_particles, 2).
    config : AnimationConfig
        Animation settings.
    potential_fn : callable, optional
        U(x) -> float for background visualization.
    attractor_positions : list of arrays, optional
        Attractor position at each frame (for moving attractors).
    """

    def __init__(
        self,
        position_history: List[Array],
        config: AnimationConfig = AnimationConfig(),
        potential_fn: Optional[Callable[[Array], float]] = None,
        attractor_positions: Optional[List[Array]] = None,
    ):
        self.positions = position_history
        self.cfg = config
        self.potential_fn = potential_fn
        self.attractor_positions = attractor_positions

        self.n_frames = len(position_history)
        if self.n_frames == 0:
            raise ValueError("position_history cannot be empty")

        self.n_particles = position_history[0].shape[0]

        # Trail buffer
        self.trails: List[List[Array]] = [[] for _ in range(self.n_particles)]

        # Figure and artists (initialized in animate())
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.scatter: Optional[Any] = None
        self.trail_lines: List[Any] = []
        self.attractor_marker: Optional[Any] = None
        self.centroid_marker: Optional[Any] = None
        self.time_text: Optional[Any] = None

    def _compute_potential_background(self) -> Tuple[Array, Array, Array]:
        """Compute potential field on grid."""
        cfg = self.cfg
        x = np.linspace(cfg.xlim[0], cfg.xlim[1], cfg.potential_resolution)
        y = np.linspace(cfg.ylim[0], cfg.ylim[1], cfg.potential_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        if self.potential_fn is not None:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.potential_fn(np.array([X[i, j], Y[i, j]]))

        return X, Y, Z

    def _init_figure(self) -> None:
        """Initialize figure and artists."""
        cfg = self.cfg

        self.fig, self.ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
        self.ax.set_xlim(cfg.xlim)
        self.ax.set_ylim(cfg.ylim)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("black")

        # Potential background
        if cfg.show_potential and self.potential_fn is not None:
            X, Y, Z = self._compute_potential_background()
            self.ax.contourf(
                X, Y, Z,
                levels=cfg.potential_levels,
                cmap=cfg.potential_cmap,
                alpha=cfg.potential_alpha,
            )

        # Initialize particle scatter
        pos = self.positions[0]
        self.scatter = self.ax.scatter(
            pos[:, 0], pos[:, 1],
            s=cfg.particle_size,
            c=cfg.particle_color,
            alpha=cfg.particle_alpha,
            zorder=10,
        )

        # Initialize trails
        if cfg.show_trails:
            for _ in range(self.n_particles):
                line, = self.ax.plot(
                    [], [],
                    color=cfg.particle_color,
                    alpha=cfg.trail_alpha,
                    linewidth=1,
                    zorder=5,
                )
                self.trail_lines.append(line)

        # Attractor marker
        if cfg.show_attractor and self.attractor_positions is not None:
            att_pos = self.attractor_positions[0]
            self.attractor_marker = self.ax.scatter(
                [att_pos[0]], [att_pos[1]],
                s=cfg.attractor_size,
                c=cfg.attractor_color,
                marker="*",
                zorder=15,
            )

        # Centroid marker
        if cfg.show_centroid:
            centroid = pos.mean(axis=0)
            self.centroid_marker = self.ax.scatter(
                [centroid[0]], [centroid[1]],
                s=cfg.centroid_size,
                c=cfg.centroid_color,
                marker="x",
                linewidths=3,
                zorder=12,
            )

        # Time text
        self.time_text = self.ax.text(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            ha="left", va="top",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
        )

        self.ax.set_xlabel("x₁", color="white")
        self.ax.set_ylabel("x₂", color="white")
        self.ax.tick_params(colors="white")

    def _init_animation(self) -> List[Any]:
        """Initialize function for FuncAnimation."""
        artists = [self.scatter]
        if self.cfg.show_trails:
            artists.extend(self.trail_lines)
        if self.attractor_marker is not None:
            artists.append(self.attractor_marker)
        if self.centroid_marker is not None:
            artists.append(self.centroid_marker)
        if self.time_text is not None:
            artists.append(self.time_text)
        return artists

    def _update_frame(self, frame: int) -> List[Any]:
        """Update function for each animation frame."""
        cfg = self.cfg
        pos = self.positions[frame]

        # Update particle positions
        self.scatter.set_offsets(pos)

        # Update trails
        if cfg.show_trails:
            for i in range(self.n_particles):
                self.trails[i].append(pos[i].copy())
                if len(self.trails[i]) > cfg.trail_length:
                    self.trails[i].pop(0)

                if len(self.trails[i]) >= 2:
                    trail = np.array(self.trails[i])
                    self.trail_lines[i].set_data(trail[:, 0], trail[:, 1])

        # Update attractor
        if self.attractor_marker is not None and self.attractor_positions is not None:
            att_idx = min(frame, len(self.attractor_positions) - 1)
            att_pos = self.attractor_positions[att_idx]
            self.attractor_marker.set_offsets([[att_pos[0], att_pos[1]]])

        # Update centroid
        if self.centroid_marker is not None:
            centroid = pos.mean(axis=0)
            self.centroid_marker.set_offsets([[centroid[0], centroid[1]]])

        # Update time text
        if self.time_text is not None:
            variance = np.mean(np.sum((pos - pos.mean(axis=0)) ** 2, axis=1))
            self.time_text.set_text(f"Frame: {frame}/{self.n_frames}\nVariance: {variance:.3f}")

        # Collect artists
        artists = [self.scatter]
        if cfg.show_trails:
            artists.extend(self.trail_lines)
        if self.attractor_marker is not None:
            artists.append(self.attractor_marker)
        if self.centroid_marker is not None:
            artists.append(self.centroid_marker)
        if self.time_text is not None:
            artists.append(self.time_text)

        return artists

    def animate(self, show: bool = True) -> FuncAnimation:
        """
        Create and display/save animation.

        Parameters
        ----------
        show : bool
            If True, display animation. If False, only create it.

        Returns
        -------
        FuncAnimation
            The animation object.
        """
        cfg = self.cfg

        self._init_figure()

        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=self._init_animation,
            frames=self.n_frames,
            interval=cfg.interval,
            blit=cfg.blit,
            repeat=cfg.repeat,
        )

        # Save if path provided
        if cfg.save_path is not None:
            print(f"[ParticleAnimator] Saving animation to {cfg.save_path}...")
            if cfg.save_path.endswith(".gif"):
                anim.save(cfg.save_path, writer="pillow", fps=cfg.fps)
            else:
                anim.save(cfg.save_path, writer=cfg.writer, fps=cfg.fps)
            print("[ParticleAnimator] Done.")

        if show:
            plt.show()

        return anim


def animate_from_history(
    position_history: List[Array],
    potential_fn: Optional[Callable[[Array], float]] = None,
    attractor_positions: Optional[List[Array]] = None,
    save_path: Optional[str] = None,
    **kwargs: Any,
) -> FuncAnimation:
    """
    Convenience function to animate particle history.

    Parameters
    ----------
    position_history : list of arrays
        Particle positions at each time step.
    potential_fn : callable, optional
        Potential function for background.
    attractor_positions : list of arrays, optional
        Attractor position at each time step.
    save_path : str, optional
        Path to save animation (MP4 or GIF).
    **kwargs
        Additional arguments for AnimationConfig.

    Returns
    -------
    FuncAnimation
    """
    config = AnimationConfig(save_path=save_path, **kwargs)
    animator = ParticleAnimator(
        position_history=position_history,
        config=config,
        potential_fn=potential_fn,
        attractor_positions=attractor_positions,
    )
    return animator.animate()
