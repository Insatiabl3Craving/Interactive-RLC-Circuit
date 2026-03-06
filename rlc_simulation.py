from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from animation import create_animation
from bode_plot import BodePlot, PoleZeroPlot
from utils import DISPLAY_NAME_TO_KEY, build_state, get_mode_names, update_state_light, validate_parameters

# Minimum seconds between heavy redraws triggered by slider dragging
_THROTTLE_SECONDS = 0.08


@dataclass
class SliderBinding:
    slider: Slider
    value_getter: Callable[[], float]
    display_updater: Callable[[], None]
    key: str  # "R", "L", "C", "V0", or "f"


class RlcSimulatorApp:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}
        self.mode_names = get_mode_names()
        self.current_mode_name = self.mode_names[0]

        # Throttle tracking
        self._last_heavy_update: float = 0.0
        self._pending_timer: object | None = None

        self.figure = plt.figure(figsize=(15, 9))
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Live RLC Simulator")

        self._create_axes()
        self._create_controls()
        self._setup_plots()

        self.animation = create_animation(self.figure, self.ax_time, self.ax_current, self.get_state)
        self._do_heavy_update()

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    def _create_axes(self) -> None:
        self.ax_time = self.figure.add_axes([0.31, 0.56, 0.29, 0.34])
        self.ax_current = self.ax_time.twinx()
        self.ax_mag = self.figure.add_axes([0.66, 0.71, 0.30, 0.19])
        self.ax_phase = self.figure.add_axes([0.66, 0.46, 0.30, 0.19])
        self.ax_pz = self.figure.add_axes([0.66, 0.11, 0.30, 0.26])
        self.ax_info = self.figure.add_axes([0.03, 0.58, 0.23, 0.32])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=10)

    def _setup_plots(self) -> None:
        """Create persistent Bode and pole-zero artist managers."""
        self.bode_plot = BodePlot(self.ax_mag, self.ax_phase)
        self.pz_plot = PoleZeroPlot(self.ax_pz)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _create_controls(self) -> None:
        self.slider_bindings: dict[str, SliderBinding] = {}
        self.control_title_ax = self.figure.add_axes([0.03, 0.92, 0.23, 0.05])
        self.control_title_ax.axis("off")
        self.control_title_ax.text(0.0, 0.5, "Controls", fontsize=13, weight="bold", va="center")

        self.slider_bindings["R"] = self._create_log_slider(
            key="R",
            position=[0.05, 0.47, 0.18, 0.035],
            label="R (Ω)",
            min_value=1.0,
            max_value=1000.0,
            initial_value=100.0,
            formatter=lambda value: f"{value:,.2f} Ω",
        )
        self.slider_bindings["L"] = self._create_log_slider(
            key="L",
            position=[0.05, 0.41, 0.18, 0.035],
            label="L (mH)",
            min_value=0.1,
            max_value=100.0,
            initial_value=10.0,
            formatter=lambda value: f"{value:.3f} mH",
            unit_scale=1e-3,
        )
        self.slider_bindings["C"] = self._create_log_slider(
            key="C",
            position=[0.05, 0.35, 0.18, 0.035],
            label="C (µF)",
            min_value=0.01,
            max_value=100.0,
            initial_value=10.0,
            formatter=lambda value: f"{value:.4f} µF",
            unit_scale=1e-6,
        )
        self.slider_bindings["V0"] = self._create_linear_slider(
            key="V0",
            position=[0.05, 0.29, 0.18, 0.035],
            label="V₀ (V)",
            min_value=0.1,
            max_value=20.0,
            initial_value=5.0,
            formatter=lambda value: f"{value:.2f} V",
        )
        self.slider_bindings["f"] = self._create_log_slider(
            key="f",
            position=[0.05, 0.23, 0.18, 0.035],
            label="f (Hz)",
            min_value=1.0,
            max_value=10000.0,
            initial_value=100.0,
            formatter=lambda value: f"{value:,.2f} Hz",
        )

        for binding in self.slider_bindings.values():
            binding.slider.on_changed(lambda _val, b=binding: self._on_slider_change(b))
            binding.display_updater()

        radio_ax = self.figure.add_axes([0.05, 0.05, 0.18, 0.14], facecolor="0.97")
        self.mode_selector = RadioButtons(radio_ax, self.mode_names)
        self.mode_selector.on_clicked(self._on_mode_change)
        radio_ax.set_title("Circuit mode")

        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _create_log_slider(
        self,
        key: str,
        position: list[float],
        label: str,
        min_value: float,
        max_value: float,
        initial_value: float,
        formatter: Callable[[float], str],
        unit_scale: float = 1.0,
    ) -> SliderBinding:
        axis = self.figure.add_axes(position, facecolor="0.97")
        slider = Slider(
            ax=axis,
            label=label,
            valmin=float(math.log10(min_value)),
            valmax=float(math.log10(max_value)),
            valinit=float(math.log10(initial_value)),
        )

        def value_getter() -> float:
            return (10.0 ** slider.val) * unit_scale

        def display_updater() -> None:
            slider.valtext.set_text(formatter(10.0 ** slider.val))

        return SliderBinding(slider=slider, value_getter=value_getter, display_updater=display_updater, key=key)

    def _create_linear_slider(
        self,
        key: str,
        position: list[float],
        label: str,
        min_value: float,
        max_value: float,
        initial_value: float,
        formatter: Callable[[float], str],
    ) -> SliderBinding:
        axis = self.figure.add_axes(position, facecolor="0.97")
        slider = Slider(axis, label, min_value, max_value, valinit=initial_value)

        def value_getter() -> float:
            return float(slider.val)

        def display_updater() -> None:
            slider.valtext.set_text(formatter(float(slider.val)))

        return SliderBinding(slider=slider, value_getter=value_getter, display_updater=display_updater, key=key)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_slider_change(self, binding: SliderBinding) -> None:
        """Route slider changes to light or heavy update paths."""
        binding.display_updater()

        if binding.key in ("V0", "f"):
            # Light update — only phasors and marker change
            self._do_light_update()
        else:
            # Heavy update — R, L, or C changed; throttle
            self._schedule_heavy_update()

    def _on_mode_change(self, label: str) -> None:
        self.current_mode_name = label
        self._do_heavy_update()

    def _on_scroll(self, event) -> None:
        if event.button not in {"up", "down"}:
            return
        current_index = self.mode_names.index(self.current_mode_name)
        step = 1 if event.button == "up" else -1
        next_index = (current_index + step) % len(self.mode_names)
        self.current_mode_name = self.mode_names[next_index]
        self.mode_selector.set_active(next_index)

    # ------------------------------------------------------------------
    # Throttle helpers
    # ------------------------------------------------------------------

    def _schedule_heavy_update(self) -> None:
        """Coalesce rapid slider drags into one heavy update."""
        now = time.perf_counter()
        elapsed = now - self._last_heavy_update
        if elapsed >= _THROTTLE_SECONDS:
            self._do_heavy_update()
        else:
            # Schedule a deferred update if not already pending
            if self._pending_timer is None:
                delay_ms = int((_THROTTLE_SECONDS - elapsed) * 1000) + 1
                self._pending_timer = self.figure.canvas.new_timer(interval=delay_ms)
                self._pending_timer.add_callback(self._deferred_heavy_update)
                self._pending_timer.single_shot = True
                self._pending_timer.start()

    def _deferred_heavy_update(self) -> None:
        self._pending_timer = None
        self._do_heavy_update()

    # ------------------------------------------------------------------
    # Update paths
    # ------------------------------------------------------------------

    def _read_parameters(self) -> tuple[float, float, float, float, float]:
        resistance = self.slider_bindings["R"].value_getter()
        inductance = self.slider_bindings["L"].value_getter()
        capacitance = self.slider_bindings["C"].value_getter()
        amplitude = self.slider_bindings["V0"].value_getter()
        frequency_hz = self.slider_bindings["f"].value_getter()
        validate_parameters(resistance, inductance, capacitance, amplitude, frequency_hz)
        return resistance, inductance, capacitance, amplitude, frequency_hz

    def _do_heavy_update(self) -> None:
        """Full rebuild — transfer function, Bode sweep, poles/zeros."""
        self._last_heavy_update = time.perf_counter()
        resistance, inductance, capacitance, amplitude, frequency_hz = self._read_parameters()
        mode_key = DISPLAY_NAME_TO_KEY[self.current_mode_name]
        self.state = build_state(mode_key, resistance, inductance, capacitance, amplitude, frequency_hz)

        self.bode_plot.update(self.state)
        self.pz_plot.update(self.state)
        self._refresh_text()
        self.figure.canvas.draw_idle()

    def _do_light_update(self) -> None:
        """Phasor + marker refresh — skips Bode sweep and pole-zero recalc."""
        if not self.state:
            return self._do_heavy_update()

        amplitude = self.slider_bindings["V0"].value_getter()
        frequency_hz = self.slider_bindings["f"].value_getter()
        self.state = update_state_light(self.state, amplitude, frequency_hz)

        # Only move the drive-frequency marker on the existing Bode curves
        self.bode_plot.update_marker_only(frequency_hz)
        self._refresh_text()
        self.figure.canvas.draw_idle()

    def _refresh_text(self) -> None:
        definition = self.state["definition"]
        resistance = float(self.state["resistance"])
        inductance = float(self.state["inductance"])
        capacitance = float(self.state["capacitance"])
        amplitude = float(self.state["amplitude"])
        frequency_hz = float(self.state["frequency_hz"])

        self.figure.suptitle(
            f"Live RLC Simulator — {definition.display_name} | "
            f"f₀ = {self.state['f0_hz']:.2f} Hz | Q = {self.state['q_factor']:.3f}",
            fontsize=15, weight="bold", y=0.98,
        )

        self.info_text.set_text(
            f"Mode: {definition.display_name}\n\n"
            f"R = {resistance:.3f} Ω\n"
            f"L = {inductance * 1e3:.4f} mH\n"
            f"C = {capacitance * 1e6:.4f} µF\n"
            f"V₀ = {amplitude:.3f} V\n"
            f"f = {frequency_hz:.3f} Hz\n\n"
            f"Output: {definition.output_label or 'Source current emphasis'}\n"
            f"Bode metric: {definition.bode_label}\n\n"
            f"Scroll wheel to cycle circuit modes."
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, object]:
        return self.state

    def run(self) -> None:
        plt.show()


def main() -> None:
    simulator = RlcSimulatorApp()
    simulator.run()


if __name__ == "__main__":
    main()
