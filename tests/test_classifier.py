from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orbital_sentinel.classifier import (
    CdmCovariance,
    classify_all,
    classify_conjunction,
    classify_object_type,
    pc_from_covariance,
    pc_hard_body_fallback,
)
from orbital_sentinel.config import ClassifierConfig, classify_tier
from orbital_sentinel.screener import ConjunctionEvent


def _event(miss_km: float, norad_id: int = 1) -> ConjunctionEvent:
    return ConjunctionEvent(
        primary_norad_id=25544,
        secondary_norad_id=norad_id,
        secondary_name="TEST-OBJ",
        tca=datetime(2024, 1, 1, tzinfo=timezone.utc),
        miss_distance_km=miss_km,
        relative_velocity_km_s=8.0,
    )


def test_classify_tier_boundaries() -> None:
    cfg = ClassifierConfig()
    assert classify_tier(1e-6, cfg) == "GREEN"
    assert classify_tier(9.99e-6, cfg) == "GREEN"
    assert classify_tier(1e-5, cfg) == "YELLOW"
    assert classify_tier(9.99e-5, cfg) == "YELLOW"
    assert classify_tier(1e-4, cfg) == "RED"
    assert classify_tier(1.0, cfg) == "RED"


def test_pc_from_covariance_decreases_with_miss_distance() -> None:
    cov = CdmCovariance(sigma_r_km=0.1, sigma_t_km=0.5, sigma_n_km=0.3)
    pc_close = pc_from_covariance(0.01, cov, hard_body_radius_km=0.02)
    pc_far = pc_from_covariance(5.0, cov, hard_body_radius_km=0.02)
    assert pc_close > pc_far
    assert 0.0 <= pc_far <= 1.0
    assert 0.0 <= pc_close <= 1.0


def test_pc_from_covariance_zero_sigma_returns_zero() -> None:
    cov = CdmCovariance(sigma_r_km=0.0, sigma_t_km=0.0, sigma_n_km=0.0)
    assert pc_from_covariance(0.01, cov, hard_body_radius_km=0.02) == 0.0


def test_pc_hard_body_fallback_bounded() -> None:
    pc = pc_hard_body_fallback(0.001, 8.0, hard_body_radius_km=0.02)
    assert 0.0 <= pc <= 1.0


def test_classify_object_type() -> None:
    assert classify_object_type(10.0, 0.0, is_known_debris=True) == "DEBRIS"
    assert classify_object_type(10.0, 0.0, is_known_debris=False) == "ACTIVE_SATELLITE"
    assert classify_object_type(10.0, 2.0, is_known_debris=True) == "MANOEUVRING"
    assert classify_object_type(0.01, 0.0, is_known_debris=False) == "MANOEUVRING"


def test_classify_conjunction_uses_covariance_when_available() -> None:
    event = _event(0.05)
    cfg = ClassifierConfig()
    cov = CdmCovariance(sigma_r_km=0.05, sigma_t_km=0.05, sigma_n_km=0.05)
    result = classify_conjunction(event, hard_body_radius_km=0.02, cfg=cfg, covariance=cov)
    assert result.pc_method == "cdm_covariance"
    assert result.risk_tier in {"GREEN", "YELLOW", "RED"}


def test_classify_conjunction_falls_back_without_covariance() -> None:
    event = _event(0.05)
    cfg = ClassifierConfig()
    result = classify_conjunction(event, hard_body_radius_km=0.02, cfg=cfg, covariance=None)
    assert result.pc_method == "hard_body_radius_model"


def test_classify_conjunction_prefers_reported_pc_over_covariance() -> None:
    event = _event(0.05)
    cfg = ClassifierConfig()
    cov = CdmCovariance(sigma_r_km=0.05, sigma_t_km=0.05, sigma_n_km=0.05)
    result = classify_conjunction(
        event, hard_body_radius_km=0.02, cfg=cfg, covariance=cov, reported_pc=2e-4
    )
    assert result.pc_method == "cdm_reported"
    assert result.probability_of_collision == pytest.approx(2e-4)
    assert result.risk_tier == "RED"


def test_classify_all_applies_reported_pcs_by_norad_id() -> None:
    events = [_event(5.0, norad_id=1), _event(0.001, norad_id=2)]
    cfg = ClassifierConfig()
    results = classify_all(
        events, hard_body_radius_km=0.02, cfg=cfg, reported_pcs={1: 5e-4}
    )
    by_id = {r.event.secondary_norad_id: r for r in results}
    assert by_id[1].pc_method == "cdm_reported"
    assert by_id[1].probability_of_collision == pytest.approx(5e-4)
    assert by_id[2].pc_method == "hard_body_radius_model"


def test_classify_all_sorts_by_pc_descending() -> None:
    events = [_event(5.0, norad_id=1), _event(0.001, norad_id=2), _event(1.0, norad_id=3)]
    cfg = ClassifierConfig()
    results = classify_all(events, hard_body_radius_km=0.02, cfg=cfg)
    pcs = [r.probability_of_collision for r in results]
    assert pcs == sorted(pcs, reverse=True)
