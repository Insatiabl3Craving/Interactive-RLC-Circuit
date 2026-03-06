from __future__ import annotations

import numpy as np

from utils import get_definition


# ---------------------------------------------------------------------------
# Persistent-artist Bode and pole-zero views
# ---------------------------------------------------------------------------

class BodePlot:
    """Manages Bode magnitude and phase axes with reusable line artists."""

    def __init__(self, ax_mag, ax_phase) -> None:
        self.ax_mag = ax_mag
        self.ax_phase = ax_phase

        # Magnitude axis — create artists once
        self.mag_line, = ax_mag.semilogx([], [], color="tab:blue", linewidth=2.0)
        self.mag_marker = ax_mag.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.3, label="Drive freq")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.legend(loc="upper right")

        # Phase axis — create artists once
        self.phase_line, = ax_phase.semilogx([], [], color="tab:purple", linewidth=2.0)
        self.phase_marker = ax_phase.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.3)
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (degrees)")
        ax_phase.set_title("Bode Phase")

    def update(self, state: dict[str, object]) -> None:
        """Refresh Bode lines and marker from cached sweep arrays in *state*."""
        definition = get_definition(str(state["mode_key"]))
        freq_hz = state["bode_freq_hz"]
        mag_db = state["bode_mag_db"]
        phase_deg = state["bode_phase_deg"]
        drive_hz = float(state["frequency_hz"])

        # Update line data
        self.mag_line.set_data(freq_hz, mag_db)
        self.phase_line.set_data(freq_hz, phase_deg)

        # Move frequency markers
        self.mag_marker.set_xdata([drive_hz, drive_hz])
        self.phase_marker.set_xdata([drive_hz, drive_hz])

        # Rescale axes to fit data
        f_min, f_max = float(freq_hz[0]), float(freq_hz[-1])
        self.ax_mag.set_xlim(f_min, f_max)
        self.ax_phase.set_xlim(f_min, f_max)

        mag_min, mag_max = float(np.min(mag_db)), float(np.max(mag_db))
        margin = max((mag_max - mag_min) * 0.1, 3.0)
        self.ax_mag.set_ylim(mag_min - margin, mag_max + margin)

        phase_min, phase_max = float(np.min(phase_deg)), float(np.max(phase_deg))
        p_margin = max((phase_max - phase_min) * 0.1, 5.0)
        self.ax_phase.set_ylim(phase_min - p_margin, phase_max + p_margin)

        self.ax_mag.set_title(f"Bode Magnitude — {definition.display_name}")
        self.ax_mag.set_ylabel(definition.bode_label)

    def update_marker_only(self, drive_hz: float) -> None:
        """Move the drive-frequency marker without recomputing the sweep."""
        self.mag_marker.set_xdata([drive_hz, drive_hz])
        self.phase_marker.set_xdata([drive_hz, drive_hz])


class PoleZeroPlot:
    """Manages the pole-zero scatter plot with reusable artists."""

    def __init__(self, ax) -> None:
        self.ax = ax

        ax.axhline(0.0, color="0.35", linewidth=1.0)
        ax.axvline(0.0, color="0.35", linewidth=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")

        # Create scatter artists (initially empty data)
        self.zeros_scatter = ax.scatter([], [], marker="o", s=80, facecolors="none",
                                        edgecolors="tab:blue", linewidths=2.0, label="Zeros")
        self.poles_scatter = ax.scatter([], [], marker="x", s=90, color="tab:red",
                                        linewidths=2.0, label="Poles")
        ax.legend(loc="upper right")

        self.annotation = ax.text(
            0.03, 0.97, "",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
        )

    def update(self, state: dict[str, object]) -> None:
        definition = get_definition(str(state["mode_key"]))
        poles = state["poles"]
        zeros = state["zeros"]

        # Update scatter data
        if zeros.size:
            self.zeros_scatter.set_offsets(np.column_stack([zeros.real, zeros.imag]))
        else:
            self.zeros_scatter.set_offsets(np.empty((0, 2)))

        if poles.size:
            self.poles_scatter.set_offsets(np.column_stack([poles.real, poles.imag]))
        else:
            self.poles_scatter.set_offsets(np.empty((0, 2)))

        # Compute symmetric limits
        limit_seed = [1.0]
        for arr in (zeros, poles):
            if arr.size:
                limit_seed.append(float(np.max(np.abs(arr.real))))
                limit_seed.append(float(np.max(np.abs(arr.imag))))
        limit = max(limit_seed) * 1.25

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(f"Pole-Zero Plot — {definition.display_name}")

        self.annotation.set_text(
            f"f₀ = {state['f0_hz']:.2f} Hz\n"
            f"Q = {state['q_factor']:.3f}\n"
            f"ζ = {state['damping_ratio']:.3f}"
        )
