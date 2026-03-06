import matplotlib.pyplot as plt

from animation import create_animation
from bode_plot import BodePlot, PoleZeroPlot
from utils import build_state


def test_bode_and_pole_zero_plots_update_without_error():
    state = build_state("series", 100.0, 0.01, 10e-6, 5.0, 100.0)
    figure = plt.figure(figsize=(8, 6))
    ax_mag = figure.add_subplot(221)
    ax_phase = figure.add_subplot(223)
    ax_pz = figure.add_subplot(122)

    bode_plot = BodePlot(ax_mag, ax_phase)
    pole_zero_plot = PoleZeroPlot(ax_pz)

    bode_plot.update(state)
    pole_zero_plot.update(state)
    bode_plot.update_marker_only(250.0)

    assert list(bode_plot.mag_marker.get_xdata()) == [250.0, 250.0]
    assert list(bode_plot.phase_marker.get_xdata()) == [250.0, 250.0]
    assert "Bode Magnitude" in ax_mag.get_title()
    assert "Pole-Zero Plot" in ax_pz.get_title()


def test_create_animation_smoke_runs_one_frame():
    state = build_state("series", 100.0, 0.01, 10e-6, 5.0, 100.0)
    figure, ax_voltage = plt.subplots()
    ax_current = ax_voltage.twinx()

    animation = create_animation(figure, ax_voltage, ax_current, lambda: state)
    artists = animation._func(0)
    animation._draw_was_started = True

    assert len(artists) == 4
    assert ax_voltage.get_xlabel() == "Time relative to present (ms)"
    assert ax_voltage.get_ylabel() == "Voltage (V)"
    assert ax_current.get_ylabel() == "Current (A)"