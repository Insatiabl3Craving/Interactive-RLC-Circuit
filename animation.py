from __future__ import annotations

import time

import numpy as np
from matplotlib.animation import FuncAnimation

from utils import get_definition, hz_to_rad_s


# Number of sample points per frame (lower = faster)
_NUM_POINTS = 600

# How many drive periods to show on screen
_PERIODS_VISIBLE = 4.0

# Fraction by which the data peak must exceed current ylim before rescaling
_YLIM_HYSTERESIS = 0.20

# Target frame interval in ms (~20 FPS — smooth enough, much lighter than 30)
_FRAME_INTERVAL_MS = 50


def _safe_peak(arr: np.ndarray | None, fallback: float) -> float:
    if arr is None or arr.size == 0:
        return fallback
    peak = float(np.max(np.abs(arr)))
    return peak if peak > 0 else fallback


def _needs_rescale(current_lim: float, data_peak: float) -> bool:
    """Return True if ylim should change, using hysteresis to avoid jitter."""
    if current_lim <= 0:
        return True
    ratio = data_peak / current_lim
    # Expand if data exceeds current limits; shrink if data is much smaller
    return ratio > (1.0 - _YLIM_HYSTERESIS) or ratio < 0.4


def create_animation(fig, ax_voltage, ax_current, get_state):
    start_time = time.perf_counter()

    # Pre-allocate a reusable normalised-phase array [0, 1)
    _phase_buf = np.linspace(0.0, 1.0, _NUM_POINTS, endpoint=False)

    # Create persistent line artists
    input_line, = ax_voltage.plot([], [], color="tab:blue", linewidth=1.8, label="Input voltage")
    output_line, = ax_voltage.plot([], [], color="tab:orange", linewidth=1.8, label="Output voltage")
    current_line, = ax_current.plot([], [], color="tab:green", linewidth=1.8, label="Source current")

    status_text = ax_voltage.text(
        0.02, 0.98, "",
        transform=ax_voltage.transAxes, va="top", ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )

    # One-time axis setup (never repeated per frame)
    ax_voltage.set_xlabel("Time relative to present (ms)")
    ax_voltage.set_ylabel("Voltage (V)")
    ax_current.set_ylabel("Current (A)")
    ax_voltage.grid(True, alpha=0.25)

    # Tracking for mode changes and ylim hysteresis
    _cache = {
        "mode_key": None,
        "output_visible": None,
        "v_lim": 1.0,
        "i_lim": 1e-3,
    }

    def animate(_frame_index):
        state = get_state()
        if not state:
            return input_line, output_line, current_line, status_text

        frequency_hz = float(state["frequency_hz"])
        amplitude = float(state["amplitude"])
        omega = float(hz_to_rad_s(frequency_hz))

        # Build rolling time window from cached phase buffer
        window_duration = _PERIODS_VISIBLE / frequency_hz
        t_now = time.perf_counter() - start_time
        t_start = t_now - window_duration
        time_values = t_start + _phase_buf * window_duration
        time_ms = (time_values - t_now) * 1000.0

        # Compute waveforms from cached phasors — NO freqresp call
        sin_base = np.sin(omega * time_values)
        input_voltage = amplitude * sin_base

        current_amp = float(state["current_amplitude"])
        current_phase = float(state["current_phase"])
        current = current_amp * np.sin(omega * time_values + current_phase)

        output_amp = state["output_amplitude"]
        output_phase = state["output_phase"]
        has_output = output_amp is not None
        output_voltage = None
        if has_output:
            output_voltage = float(output_amp) * np.sin(omega * time_values + float(output_phase))

        # Update line data
        input_line.set_data(time_ms, input_voltage)
        current_line.set_data(time_ms, current)

        if has_output:
            output_line.set_data(time_ms, output_voltage)
            output_line.set_visible(True)
        else:
            output_line.set_data([], [])
            output_line.set_visible(False)

        # X-axis always matches window
        ax_voltage.set_xlim(time_ms[0], time_ms[-1])

        # Y-axis with hysteresis to prevent jitter
        v_peak = max(_safe_peak(input_voltage, 1.0), _safe_peak(output_voltage, 0.0), 0.5)
        i_peak = max(_safe_peak(current, 1e-3), 1e-3)

        if _needs_rescale(_cache["v_lim"], v_peak):
            _cache["v_lim"] = v_peak * 1.15
            ax_voltage.set_ylim(-_cache["v_lim"], _cache["v_lim"])
        if _needs_rescale(_cache["i_lim"], i_peak):
            _cache["i_lim"] = i_peak * 1.15
            ax_current.set_ylim(-_cache["i_lim"], _cache["i_lim"])

        # Mode-change housekeeping (runs rarely)
        mode_key = str(state["mode_key"])
        if _cache["mode_key"] != mode_key or _cache["output_visible"] != has_output:
            definition = get_definition(mode_key)
            input_line.set_label("Input voltage")
            current_line.set_label("Source current")
            output_line.set_label(definition.output_label or "")
            title = definition.display_name
            if definition.output_label:
                title += f" — Output: {definition.output_label}"
            ax_voltage.set_title(title)

            # Rebuild legend only on mode/visibility change
            handles = [input_line]
            if has_output:
                handles.append(output_line)
            handles.append(current_line)
            ax_voltage.legend(handles, [h.get_label() for h in handles], loc="upper right")

            _cache["mode_key"] = mode_key
            _cache["output_visible"] = has_output

        # Lightweight status text
        resp_text = ""
        if output_amp is not None and output_phase is not None:
            resp_text = f"|H| = {float(output_amp)/amplitude:.3f}  φ = {np.degrees(float(output_phase)):.1f}°\n"

        status_text.set_text(
            f"{state['definition'].display_name}  "
            f"f={frequency_hz:.1f} Hz  f₀={state['f0_hz']:.1f} Hz\n"
            f"V₀={amplitude:.1f} V  Ipk={i_peak:.4f} A\n"
            f"{resp_text}"
        )

        return input_line, output_line, current_line, status_text

    return FuncAnimation(fig, animate, interval=_FRAME_INTERVAL_MS, blit=False, cache_frame_data=False)
