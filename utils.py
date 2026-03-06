from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import TransferFunction, bode, freqresp, tf2zpk


@dataclass(frozen=True)
class CircuitDefinition:
    key: str
    display_name: str
    output_label: Optional[str]
    bode_label: str
    topology: str
    notes: str


CIRCUIT_DEFINITIONS: Dict[str, CircuitDefinition] = {
    "series": CircuitDefinition(
        key="series",
        display_name="Series RLC",
        output_label="Capacitor voltage",
        bode_label="Voltage gain |Vc / Vin| (dB)",
        topology="series",
        notes="Series RLC with the output measured across the capacitor.",
    ),
    "parallel": CircuitDefinition(
        key="parallel",
        display_name="Parallel RLC",
        output_label=None,
        bode_label="Normalized impedance response (dB)",
        topology="parallel",
        notes="Parallel RLC shows source voltage and source current.\nBode plots a normalized impedance-like response.",
    ),
    "bandpass": CircuitDefinition(
        key="bandpass",
        display_name="RLC Bandpass",
        output_label="Resistor voltage",
        bode_label="Voltage gain |Vr / Vin| (dB)",
        topology="series",
        notes="Series RLC with output measured across the resistor.",
    ),
    "bandstop": CircuitDefinition(
        key="bandstop",
        display_name="RLC Bandstop",
        output_label="LC branch voltage",
        bode_label="Voltage gain |Vlc / Vin| (dB)",
        topology="series",
        notes="Series RLC with output measured across the L-C branch.",
    ),
}


DISPLAY_NAME_TO_KEY = {
    definition.display_name: key for key, definition in CIRCUIT_DEFINITIONS.items()
}


def get_mode_keys() -> list[str]:
    return list(CIRCUIT_DEFINITIONS.keys())


def get_mode_names() -> list[str]:
    return [CIRCUIT_DEFINITIONS[key].display_name for key in get_mode_keys()]


def get_definition(mode_key: str) -> CircuitDefinition:
    if mode_key not in CIRCUIT_DEFINITIONS:
        raise KeyError(f"Unsupported mode: {mode_key}")
    return CIRCUIT_DEFINITIONS[mode_key]


def validate_parameters(
    resistance: float,
    inductance: float,
    capacitance: float,
    amplitude: float,
    frequency_hz: float,
) -> None:
    values = {
        "Resistance": resistance,
        "Inductance": inductance,
        "Capacitance": capacitance,
        "Voltage amplitude": amplitude,
        "Frequency": frequency_hz,
    }
    for label, value in values.items():
        if value <= 0:
            raise ValueError(f"{label} must be greater than zero.")


def hz_to_rad_s(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    return 2.0 * np.pi * frequency_hz


def resonant_frequency(inductance: float, capacitance: float) -> float:
    return 1.0 / (2.0 * np.pi * np.sqrt(inductance * capacitance))


def quality_factor(mode_key: str, resistance: float, inductance: float, capacitance: float) -> float:
    if get_definition(mode_key).topology == "parallel":
        return resistance * np.sqrt(capacitance / inductance)
    return (1.0 / resistance) * np.sqrt(inductance / capacitance)


def damping_ratio(mode_key: str, resistance: float, inductance: float, capacitance: float) -> float:
    q_value = quality_factor(mode_key, resistance, inductance, capacitance)
    if q_value == 0:
        return np.inf
    return 1.0 / (2.0 * q_value)


def build_transfer_function(
    mode_key: str,
    resistance: float,
    inductance: float,
    capacitance: float,
) -> TransferFunction:
    if mode_key == "series":
        numerator = [1.0 / (inductance * capacitance)]
        denominator = [1.0, resistance / inductance, 1.0 / (inductance * capacitance)]
    elif mode_key == "parallel":
        numerator = [1.0 / (resistance * capacitance), 0.0]
        denominator = [1.0, 1.0 / (resistance * capacitance), 1.0 / (inductance * capacitance)]
    elif mode_key == "bandpass":
        numerator = [resistance / inductance, 0.0]
        denominator = [1.0, resistance / inductance, 1.0 / (inductance * capacitance)]
    elif mode_key == "bandstop":
        numerator = [1.0, 0.0, 1.0 / (inductance * capacitance)]
        denominator = [1.0, resistance / inductance, 1.0 / (inductance * capacitance)]
    else:
        raise KeyError(f"Unsupported mode: {mode_key}")

    return TransferFunction(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))


def complex_response(tf: TransferFunction, frequency_hz: float) -> complex:
    omega = float(hz_to_rad_s(frequency_hz))
    _, response = freqresp(tf, [omega])
    return complex(np.squeeze(response))


def compute_steady_state(
    tf: TransferFunction,
    input_amplitude: float,
    frequency_hz: float,
) -> tuple[float, float, complex]:
    response = complex_response(tf, frequency_hz)
    return input_amplitude * abs(response), float(np.angle(response)), response


def get_poles_zeros(tf: TransferFunction) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.atleast_1d(np.squeeze(tf.num))
    denominator = np.atleast_1d(np.squeeze(tf.den))
    zeros, poles, _ = tf2zpk(numerator, denominator)
    return poles, zeros


def series_impedance(
    resistance: float,
    inductance: float,
    capacitance: float,
    omega: float,
) -> complex:
    return resistance + 1j * omega * inductance + 1.0 / (1j * omega * capacitance)


def parallel_impedance(
    resistance: float,
    inductance: float,
    capacitance: float,
    omega: float,
) -> complex:
    admittance = (1.0 / resistance) + (1.0 / (1j * omega * inductance)) + (1j * omega * capacitance)
    return 1.0 / admittance


def source_current_phasor(
    mode_key: str,
    resistance: float,
    inductance: float,
    capacitance: float,
    input_amplitude: float,
    frequency_hz: float,
) -> complex:
    omega = float(hz_to_rad_s(frequency_hz))
    if get_definition(mode_key).topology == "parallel":
        impedance = parallel_impedance(resistance, inductance, capacitance, omega)
    else:
        impedance = series_impedance(resistance, inductance, capacitance, omega)
    return complex(input_amplitude) / impedance


# ---------------------------------------------------------------------------
# Bode sweep helper (computed once per heavy update, cached in state)
# ---------------------------------------------------------------------------

def _build_frequency_sweep(resonance_hz: float, drive_frequency_hz: float) -> np.ndarray:
    min_anchor = max(min(resonance_hz, drive_frequency_hz, 1.0), 1e-1)
    max_anchor = max(resonance_hz, drive_frequency_hz, 1_000.0)
    f_min = max(1e-1, min_anchor / 100.0)
    f_max = min(1e6, max_anchor * 100.0)
    if f_max <= f_min:
        f_max = f_min * 1000.0
    return np.logspace(np.log10(f_min), np.log10(f_max), 500)


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------

def build_state(
    mode_key: str,
    resistance: float,
    inductance: float,
    capacitance: float,
    amplitude: float,
    frequency_hz: float,
) -> dict[str, object]:
    """Full heavy state rebuild — call only when R, L, C, or mode changes."""
    validate_parameters(resistance, inductance, capacitance, amplitude, frequency_hz)
    tf = build_transfer_function(mode_key, resistance, inductance, capacitance)
    poles, zeros = get_poles_zeros(tf)
    f0 = resonant_frequency(inductance, capacitance)

    # Pre-compute Bode sweep so it is never recomputed during animation
    sweep_hz = _build_frequency_sweep(f0, frequency_hz)
    sweep_w = hz_to_rad_s(sweep_hz)
    _, mag_db, phase_deg = bode(tf, sweep_w)

    # Pre-compute phasors at the current drive frequency
    output_amp, output_phase, _ = compute_steady_state(tf, amplitude, frequency_hz) \
        if get_definition(mode_key).output_label is not None else (None, None, None)
    current_phasor = source_current_phasor(
        mode_key, resistance, inductance, capacitance, amplitude, frequency_hz,
    )

    return {
        "mode_key": mode_key,
        "definition": get_definition(mode_key),
        "resistance": resistance,
        "inductance": inductance,
        "capacitance": capacitance,
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "tf": tf,
        "poles": poles,
        "zeros": zeros,
        "f0_hz": f0,
        "q_factor": quality_factor(mode_key, resistance, inductance, capacitance),
        "damping_ratio": damping_ratio(mode_key, resistance, inductance, capacitance),
        # Cached Bode arrays
        "bode_freq_hz": np.asarray(sweep_hz, dtype=float),
        "bode_mag_db": np.asarray(mag_db, dtype=float),
        "bode_phase_deg": np.asarray(phase_deg, dtype=float),
        # Cached phasors for the animation
        "output_amplitude": output_amp,
        "output_phase": output_phase,
        "current_amplitude": float(abs(current_phasor)),
        "current_phase": float(np.angle(current_phasor)),
    }


def update_state_light(
    state: dict[str, object],
    amplitude: float,
    frequency_hz: float,
) -> dict[str, object]:
    """Light state update — recompute only phasors and drive-frequency marker.

    Reuses the existing transfer function and Bode sweep arrays.
    Call when only V₀ or f changes (R, L, C, mode unchanged).
    """
    tf = state["tf"]
    mode_key = str(state["mode_key"])
    resistance = float(state["resistance"])
    inductance = float(state["inductance"])
    capacitance = float(state["capacitance"])
    definition = get_definition(mode_key)

    output_amp, output_phase, _ = compute_steady_state(tf, amplitude, frequency_hz) \
        if definition.output_label is not None else (None, None, None)
    current_phasor = source_current_phasor(
        mode_key, resistance, inductance, capacitance, amplitude, frequency_hz,
    )

    # Shallow copy then overwrite only the changed fields
    new_state = dict(state)
    new_state["amplitude"] = amplitude
    new_state["frequency_hz"] = frequency_hz
    new_state["output_amplitude"] = output_amp
    new_state["output_phase"] = output_phase
    new_state["current_amplitude"] = float(abs(current_phasor))
    new_state["current_phase"] = float(np.angle(current_phasor))
    return new_state
