from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orbital_sentinel.config import ScreeningConfig
from orbital_sentinel.propagator import TLEObject
from orbital_sentinel.screener import _is_colocated, screen_catalog, screen_pair

ISS_TLE = TLEObject(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)

# A geostationary-regime object: wildly different orbital altitude, should
# never trigger a conjunction against the ISS.
GEO_TLE = TLEObject(
    norad_id=99999,
    name="GEO-TEST-OBJECT",
    line1="1 99999U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 99999  00.0500 247.4627 0006703 130.5360 325.0288 01.00273790563537",
)

# A near-duplicate of the ISS TLE, same orbital regime, should pass the
# coarse distance gate and be evaluated by the refinement step.
ISS_CLOSE_TLE = TLEObject(
    norad_id=88888,
    name="ISS-CLOSE-TEST-OBJECT",
    line1="1 88888U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 88888  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)


def test_screen_pair_distant_regimes_returns_none() -> None:
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    cfg = ScreeningConfig(miss_distance_threshold_km=5.0)
    result = screen_pair(ISS_TLE, GEO_TLE, start, end, cfg)
    assert result is None


def test_screen_pair_same_regime_runs_without_error() -> None:
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    cfg = ScreeningConfig(miss_distance_threshold_km=50.0)
    # Same orbital plane/regime objects: screening should complete and
    # either return a conjunction event or None, without raising.
    result = screen_pair(ISS_TLE, ISS_CLOSE_TLE, start, end, cfg)
    assert result is None or result.miss_distance_km <= 50.0


def test_screen_catalog_sorts_by_miss_distance() -> None:
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    cfg = ScreeningConfig(miss_distance_threshold_km=10000.0)
    events = screen_catalog(ISS_TLE, [GEO_TLE, ISS_CLOSE_TLE], start, end, cfg)
    if len(events) > 1:
        distances = [e.miss_distance_km for e in events]
        assert distances == sorted(distances)


def test_screen_catalog_skips_self() -> None:
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    cfg = ScreeningConfig(miss_distance_threshold_km=10000.0)
    events = screen_catalog(ISS_TLE, [ISS_TLE], start, end, cfg)
    assert all(e.secondary_norad_id != ISS_TLE.norad_id for e in events)


def test_is_colocated_flags_constant_tiny_range() -> None:
    ranges = [0.001, 0.0012, 0.0009, 0.0011, 0.001]
    assert _is_colocated(ranges, std_threshold_km=0.1) is True


def test_is_colocated_does_not_flag_v_shaped_approach() -> None:
    # A genuine conjunction: range decreases toward TCA then increases —
    # far more variance than an attached module, even though the minimum
    # is also small.
    ranges = [3.0, 1.5, 0.4, 0.05, 0.4, 1.5, 3.0]
    assert _is_colocated(ranges, std_threshold_km=0.1) is False


def test_is_colocated_ignores_pairs_that_were_never_close() -> None:
    ranges = [500.0, 500.1, 499.9, 500.0]
    assert _is_colocated(ranges, std_threshold_km=0.1) is False


def test_screen_pair_treats_identical_orbit_as_colocated_not_conjunction() -> None:
    # ISS_CLOSE_TLE shares ISS_TLE's exact orbital elements — a stand-in
    # for a catalogued ISS module (e.g. NAUKA, ZVEZDA) rather than a
    # separate object on a genuine collision course.
    start = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    cfg = ScreeningConfig(miss_distance_threshold_km=50.0, colocation_std_threshold_km=0.1)
    result = screen_pair(ISS_TLE, ISS_CLOSE_TLE, start, end, cfg)
    assert result is None
