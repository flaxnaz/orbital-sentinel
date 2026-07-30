from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbital_sentinel.config import MU_EARTH_KM3_S2, EARTH_RADIUS_KM
from orbital_sentinel.propagator import (
    HighFidelityPropagator,
    SGP4Propagator,
    StateVector,
    TLEObject,
    two_body_derivative,
)

# ISS TLE (representative, low-eccentricity LEO)
ISS_TLE = TLEObject(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)


def test_sgp4_propagate_altitude_is_leo() -> None:
    prop = SGP4Propagator(ISS_TLE)
    epoch = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    state = prop.propagate(epoch)
    r_mag = float(np.linalg.norm(state.r_km))
    altitude_km = r_mag - EARTH_RADIUS_KM
    # ISS orbits at roughly 400-420 km altitude.
    assert 300.0 < altitude_km < 500.0


def test_sgp4_sample_window_returns_expected_count() -> None:
    prop = SGP4Propagator(ISS_TLE)
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)
    states = prop.sample_window(start, end, step_s=60.0)
    assert len(states) == 11  # 0,60,...,600 seconds inclusive


def test_two_body_derivative_circular_orbit_magnitude() -> None:
    # Circular orbit at LEO altitude: v = sqrt(mu/r)
    r_mag = EARTH_RADIUS_KM + 400.0
    v_mag = np.sqrt(MU_EARTH_KM3_S2 / r_mag)
    y0 = np.array([r_mag, 0.0, 0.0, 0.0, v_mag, 0.0])
    dydt = two_body_derivative(0.0, y0)
    accel = dydt[3:]
    expected_accel_mag = MU_EARTH_KM3_S2 / r_mag**2
    assert np.linalg.norm(accel) == pytest.approx(expected_accel_mag, rel=1e-6)


def test_high_fidelity_propagator_conserves_circular_radius() -> None:
    r_mag = EARTH_RADIUS_KM + 400.0
    v_mag = np.sqrt(MU_EARTH_KM3_S2 / r_mag)
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state0 = StateVector(
        epoch=epoch,
        r_km=np.array([r_mag, 0.0, 0.0]),
        v_km_s=np.array([0.0, v_mag, 0.0]),
    )
    prop = HighFidelityPropagator(rtol=1e-9, atol=1e-12)
    target = epoch + timedelta(minutes=30)
    state1 = prop.propagate(state0, target)
    r1_mag = float(np.linalg.norm(state1.r_km))
    # A circular orbit should preserve orbital radius closely.
    assert r1_mag == pytest.approx(r_mag, rel=1e-6)


def test_high_fidelity_propagate_dense_length() -> None:
    r_mag = EARTH_RADIUS_KM + 400.0
    v_mag = np.sqrt(MU_EARTH_KM3_S2 / r_mag)
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state0 = StateVector(
        epoch=epoch,
        r_km=np.array([r_mag, 0.0, 0.0]),
        v_km_s=np.array([0.0, v_mag, 0.0]),
    )
    prop = HighFidelityPropagator()
    states = prop.propagate_dense(state0, duration_s=600.0, num_points=11)
    assert len(states) == 11
    assert states[0].epoch == epoch
