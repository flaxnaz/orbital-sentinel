from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orbital_sentinel.cam_planner import (
    cw_along_track_delta_v,
    cw_secular_drift_rate,
    mean_motion_rad_s,
    plan_all,
    plan_cam,
)
from orbital_sentinel.classifier import ClassifiedConjunction
from orbital_sentinel.config import EARTH_RADIUS_KM, CamConfig
from orbital_sentinel.screener import ConjunctionEvent

# Reference scenario from SMC interview prep / mock rendezvous:
# n = 0.00111 rad/s corresponds to a semi-major axis near LEO altitude.
REF_N_RAD_S = 0.00111


def _classified(miss_km: float, tier: str) -> ClassifiedConjunction:
    event = ConjunctionEvent(
        primary_norad_id=25544,
        secondary_norad_id=42,
        secondary_name="TEST-DEBRIS",
        tca=datetime(2024, 1, 1, tzinfo=timezone.utc),
        miss_distance_km=miss_km,
        relative_velocity_km_s=8.0,
    )
    return ClassifiedConjunction(
        event=event,
        probability_of_collision=2e-4,
        risk_tier=tier,
        object_type="DEBRIS",
        pc_method="hard_body_radius_model",
    )


def test_mean_motion_matches_reference_scenario() -> None:
    # Solve for the semi-major axis implied by n = 0.00111 rad/s and check
    # mean_motion_rad_s recovers it.
    from orbital_sentinel.config import MU_EARTH_KM3_S2

    a = (MU_EARTH_KM3_S2 / REF_N_RAD_S**2) ** (1.0 / 3.0)
    n = mean_motion_rad_s(a)
    assert n == pytest.approx(REF_N_RAD_S, rel=1e-6)


def test_cw_along_track_delta_v_matches_reference_relationship() -> None:
    n = REF_N_RAD_S
    x0_km = 1.0
    delta_v = cw_along_track_delta_v(n, x0_km)
    assert delta_v == pytest.approx(n * x0_km / 2.0, rel=1e-9)


def test_cw_secular_drift_rate_matches_reference_relationship() -> None:
    n = REF_N_RAD_S
    x0_km = 1.0
    drift = cw_secular_drift_rate(n, x0_km)
    assert drift == pytest.approx(6.0 * n * x0_km, rel=1e-9)


def test_plan_cam_returns_none_for_non_red_tier() -> None:
    classified = _classified(0.05, "YELLOW")
    cfg = CamConfig()
    result = plan_cam(classified, semi_major_axis_km=EARTH_RADIUS_KM + 400.0, cfg=cfg)
    assert result is None


def test_plan_cam_returns_none_if_already_safe() -> None:
    classified = _classified(10.0, "RED")
    cfg = CamConfig(safe_miss_distance_km=5.0)
    result = plan_cam(classified, semi_major_axis_km=EARTH_RADIUS_KM + 400.0, cfg=cfg)
    assert result is None


def test_plan_cam_produces_positive_delta_v_for_red_tier() -> None:
    classified = _classified(0.5, "RED")
    cfg = CamConfig(safe_miss_distance_km=5.0)
    result = plan_cam(classified, semi_major_axis_km=EARTH_RADIUS_KM + 400.0, cfg=cfg)
    assert result is not None
    assert result.delta_v_m_s > 0.0
    assert result.burn_direction == "ALONG_TRACK"
    assert result.predicted_new_miss_distance_km == pytest.approx(5.0, rel=1e-6)
    assert result.propellant_cost_kg > 0.0


def test_plan_cam_propellant_cost_scales_with_delta_v() -> None:
    cfg = CamConfig(safe_miss_distance_km=5.0)
    small_gap = _classified(4.9, "RED")
    large_gap = _classified(0.1, "RED")
    small_result = plan_cam(small_gap, EARTH_RADIUS_KM + 400.0, cfg)
    large_result = plan_cam(large_gap, EARTH_RADIUS_KM + 400.0, cfg)
    assert small_result is not None and large_result is not None
    assert large_result.propellant_cost_kg > small_result.propellant_cost_kg


def test_plan_all_filters_to_red_tier_only() -> None:
    cfg = CamConfig(safe_miss_distance_km=5.0)
    batch = [_classified(0.5, "RED"), _classified(0.5, "GREEN"), _classified(0.5, "YELLOW")]
    results = plan_all(batch, EARTH_RADIUS_KM + 400.0, cfg)
    assert len(results) == 1
