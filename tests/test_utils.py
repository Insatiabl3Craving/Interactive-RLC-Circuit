import math

import numpy as np
import pytest

from utils import (
    build_state,
    damping_ratio,
    get_mode_keys,
    quality_factor,
    resonant_frequency,
    update_state_light,
    validate_parameters,
)


@pytest.mark.parametrize("field_name", ["resistance", "inductance", "capacitance", "amplitude", "frequency_hz"])
def test_validate_parameters_rejects_non_positive_values(field_name):
    params = {
        "resistance": 100.0,
        "inductance": 0.01,
        "capacitance": 10e-6,
        "amplitude": 5.0,
        "frequency_hz": 100.0,
    }
    params[field_name] = 0.0

    with pytest.raises(ValueError, match="greater than zero"):
        validate_parameters(**params)


def test_resonance_and_q_factor_formulas_match_expected_values():
    resistance = 100.0
    inductance = 0.01
    capacitance = 10e-6

    expected_f0 = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))
    expected_series_q = (1.0 / resistance) * math.sqrt(inductance / capacitance)
    expected_parallel_q = resistance * math.sqrt(capacitance / inductance)

    assert resonant_frequency(inductance, capacitance) == pytest.approx(expected_f0)
    assert quality_factor("series", resistance, inductance, capacitance) == pytest.approx(expected_series_q)
    assert quality_factor("parallel", resistance, inductance, capacitance) == pytest.approx(expected_parallel_q)
    assert damping_ratio("series", resistance, inductance, capacitance) == pytest.approx(1.0 / (2.0 * expected_series_q))


@pytest.mark.parametrize("mode_key", get_mode_keys())
def test_build_state_populates_cached_arrays_and_phasors(mode_key):
    state = build_state(mode_key, 100.0, 0.01, 10e-6, 5.0, 100.0)

    assert state["mode_key"] == mode_key
    assert state["definition"].key == mode_key
    assert state["bode_freq_hz"].shape == (500,)
    assert state["bode_mag_db"].shape == (500,)
    assert state["bode_phase_deg"].shape == (500,)
    assert np.all(np.diff(state["bode_freq_hz"]) > 0)
    assert np.isfinite(state["bode_mag_db"]).all()
    assert np.isfinite(state["bode_phase_deg"]).all()
    assert np.isfinite(state["poles"]).all()
    assert np.isfinite(state["zeros"]).all()
    assert state["current_amplitude"] > 0.0
    assert math.isfinite(state["current_phase"])

    if state["definition"].output_label is None:
        assert state["output_amplitude"] is None
        assert state["output_phase"] is None
    else:
        assert state["output_amplitude"] > 0.0
        assert math.isfinite(state["output_phase"])


def test_update_state_light_reuses_cached_heavy_data_and_updates_drive_state():
    state = build_state("series", 100.0, 0.01, 10e-6, 5.0, 100.0)
    updated = update_state_light(state, amplitude=10.0, frequency_hz=200.0)

    assert updated is not state
    assert updated["tf"] is state["tf"]
    assert updated["bode_freq_hz"] is state["bode_freq_hz"]
    assert updated["bode_mag_db"] is state["bode_mag_db"]
    assert updated["bode_phase_deg"] is state["bode_phase_deg"]
    assert updated["poles"] is state["poles"]
    assert updated["zeros"] is state["zeros"]
    assert updated["amplitude"] == pytest.approx(10.0)
    assert updated["frequency_hz"] == pytest.approx(200.0)
    assert state["amplitude"] == pytest.approx(5.0)
    assert state["frequency_hz"] == pytest.approx(100.0)
    assert updated["current_amplitude"] != pytest.approx(state["current_amplitude"])