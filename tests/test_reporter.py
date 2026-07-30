from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from orbital_sentinel.cam_planner import CamRecommendation
from orbital_sentinel.classifier import ClassifiedConjunction
from orbital_sentinel.reporter import (
    render_dashboard,
    render_pc_trend_svg,
    write_csv,
    write_dashboard,
)
from orbital_sentinel.screener import ConjunctionEvent


def _classified(tier: str, norad_id: int) -> ClassifiedConjunction:
    event = ConjunctionEvent(
        primary_norad_id=25544,
        secondary_norad_id=norad_id,
        secondary_name="TEST-OBJ",
        tca=datetime(2024, 1, 1, tzinfo=timezone.utc),
        miss_distance_km=1.2,
        relative_velocity_km_s=8.0,
    )
    return ClassifiedConjunction(
        event=event,
        probability_of_collision=2e-4,
        risk_tier=tier,
        object_type="DEBRIS",
        pc_method="hard_body_radius_model",
    )


def test_render_dashboard_contains_tier_counts() -> None:
    classified = [_classified("RED", 1), _classified("GREEN", 2)]
    html_out = render_dashboard(classified, cams=[])
    assert "RED" in html_out
    assert "GREEN: 1" in html_out
    assert "RED: 1" in html_out


def test_write_dashboard_creates_file(tmp_path: Path) -> None:
    classified = [_classified("YELLOW", 1)]
    out_path = tmp_path / "dashboard" / "index.html"
    write_dashboard(classified, cams=[], path=out_path)
    assert out_path.exists()
    assert "YELLOW" in out_path.read_text()


def test_write_csv_creates_rows(tmp_path: Path) -> None:
    classified = [_classified("RED", 1), _classified("GREEN", 2)]
    cam = CamRecommendation(
        secondary_norad_id=1,
        burn_time_s_before_tca=1000.0,
        burn_direction="ALONG_TRACK",
        delta_v_m_s=0.5,
        predicted_new_miss_distance_km=5.0,
        propellant_cost_kg=0.01,
    )
    out_path = tmp_path / "report.csv"
    write_csv(classified, cams={1: cam}, path=out_path)
    with out_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["risk_tier"] == "RED"
    assert rows[0]["cam_delta_v_m_s"] == "0.5"
    assert rows[1]["cam_delta_v_m_s"] == ""


def test_render_pc_trend_svg_returns_new_label_for_single_point() -> None:
    assert render_pc_trend_svg([2e-4]) == '<span class="trend-new">new</span>'


def test_render_pc_trend_svg_returns_new_label_for_empty() -> None:
    assert render_pc_trend_svg([]) == '<span class="trend-new">new</span>'


def test_render_pc_trend_svg_renders_polyline_for_multiple_points() -> None:
    svg = render_pc_trend_svg([1e-6, 5e-5, 2e-4])
    assert "<svg" in svg
    assert "polyline" in svg
    assert "circle" in svg


def test_render_pc_trend_svg_handles_flat_series_without_error() -> None:
    # Identical values would divide by zero if not guarded against.
    svg = render_pc_trend_svg([2e-4, 2e-4, 2e-4])
    assert "<svg" in svg


def test_render_pc_trend_svg_handles_zero_pc_without_math_error() -> None:
    # log10(0) would raise; the floor value must prevent that.
    svg = render_pc_trend_svg([0.0, 1e-8, 0.0])
    assert "<svg" in svg


def test_render_pc_trend_svg_colours_rising_trend_differently() -> None:
    rising = render_pc_trend_svg([1e-8, 1e-6, 1e-4])
    falling = render_pc_trend_svg([1e-4, 1e-6, 1e-8])
    assert rising != falling


def test_render_dashboard_shows_trend_for_object_with_history() -> None:
    classified = [_classified("RED", 1)]
    html_out = render_dashboard(classified, cams=[], pc_histories={1: [1e-6, 2e-4]})
    assert "<svg" in html_out
    assert "Pc trend" in html_out


def test_render_dashboard_shows_new_label_without_history() -> None:
    classified = [_classified("RED", 1)]
    html_out = render_dashboard(classified, cams=[], pc_histories={})
    assert 'class="trend-new"' in html_out


def test_write_dashboard_passes_through_pc_histories(tmp_path: Path) -> None:
    classified = [_classified("YELLOW", 1)]
    out_path = tmp_path / "dashboard" / "index.html"
    write_dashboard(classified, cams=[], path=out_path, pc_histories={1: [1e-5, 3e-5, 9e-5]})
    content = out_path.read_text()
    assert "<svg" in content
