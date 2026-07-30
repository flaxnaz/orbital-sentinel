from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orbital_sentinel.cam_planner import CamRecommendation
from orbital_sentinel.classifier import ClassifiedConjunction
from orbital_sentinel.config import ScreeningConfig
from orbital_sentinel.database import (
    get_connection,
    get_conjunctions_for_run,
    get_pc_history,
    get_recent_runs,
    record_cam_recommendations,
    record_classified_conjunctions,
    record_screening_run,
)
from orbital_sentinel.propagator import TLEObject
from orbital_sentinel.screener import ConjunctionEvent

PRIMARY = TLEObject(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)


def _classified(tier: str, norad_id: int, pc: float = 2e-4) -> ClassifiedConjunction:
    event = ConjunctionEvent(
        primary_norad_id=25544,
        secondary_norad_id=norad_id,
        secondary_name=f"TEST-{norad_id}",
        tca=datetime(2024, 1, 1, tzinfo=timezone.utc),
        miss_distance_km=1.2,
        relative_velocity_km_s=8.0,
    )
    return ClassifiedConjunction(
        event=event,
        probability_of_collision=pc,
        risk_tier=tier,
        object_type="DEBRIS",
        pc_method="hard_body_radius_model",
    )


def test_get_connection_creates_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "sub" / "test.db"
    conn = get_connection(db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"screening_runs", "conjunctions", "cam_recommendations"} <= tables
        assert db_path.exists()
    finally:
        conn.close()


def test_record_screening_run_returns_incrementing_ids(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        run_id_1 = record_screening_run(conn, PRIMARY, catalog_size=10, cfg=cfg)
        run_id_2 = record_screening_run(conn, PRIMARY, catalog_size=20, cfg=cfg)
        assert run_id_2 == run_id_1 + 1
    finally:
        conn.close()


def test_record_and_retrieve_classified_conjunctions(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        run_id = record_screening_run(conn, PRIMARY, catalog_size=5, cfg=cfg)
        classified = [_classified("RED", 111), _classified("GREEN", 222)]
        id_map = record_classified_conjunctions(conn, run_id, classified)

        assert set(id_map.keys()) == {111, 222}

        rows = get_conjunctions_for_run(conn, run_id)
        assert len(rows) == 2
        # sorted by probability_of_collision descending; both have same pc here
        secondary_ids = {row["secondary_norad_id"] for row in rows}
        assert secondary_ids == {111, 222}
    finally:
        conn.close()


def test_record_cam_recommendations_links_to_conjunction(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        run_id = record_screening_run(conn, PRIMARY, catalog_size=5, cfg=cfg)
        classified = [_classified("RED", 111)]
        id_map = record_classified_conjunctions(conn, run_id, classified)

        cam = CamRecommendation(
            secondary_norad_id=111,
            burn_time_s_before_tca=1000.0,
            burn_direction="ALONG_TRACK",
            delta_v_m_s=0.5,
            predicted_new_miss_distance_km=5.0,
            propellant_cost_kg=0.01,
        )
        record_cam_recommendations(conn, id_map, [cam])

        row = conn.execute("SELECT * FROM cam_recommendations").fetchone()
        assert row["conjunction_id"] == id_map[111]
        assert row["delta_v_m_s"] == 0.5
    finally:
        conn.close()


def test_record_cam_recommendations_skips_unknown_secondary(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        run_id = record_screening_run(conn, PRIMARY, catalog_size=5, cfg=cfg)
        classified = [_classified("RED", 111)]
        id_map = record_classified_conjunctions(conn, run_id, classified)

        # CAM references a secondary that wasn't in this run's classified batch.
        stray_cam = CamRecommendation(
            secondary_norad_id=999,
            burn_time_s_before_tca=1000.0,
            burn_direction="ALONG_TRACK",
            delta_v_m_s=0.5,
            predicted_new_miss_distance_km=5.0,
            propellant_cost_kg=0.01,
        )
        record_cam_recommendations(conn, id_map, [stray_cam])

        count = conn.execute("SELECT COUNT(*) AS n FROM cam_recommendations").fetchone()["n"]
        assert count == 0
    finally:
        conn.close()


def test_get_pc_history_orders_by_time_across_runs(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        run_1 = record_screening_run(
            conn, PRIMARY, 5, cfg, run_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        record_classified_conjunctions(conn, run_1, [_classified("YELLOW", 111, pc=1e-5)])

        run_2 = record_screening_run(
            conn, PRIMARY, 5, cfg, run_timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        record_classified_conjunctions(conn, run_2, [_classified("RED", 111, pc=2e-4)])

        history = get_pc_history(conn, 111)
        assert len(history) == 2
        assert history[0]["probability_of_collision"] == 1e-5
        assert history[1]["probability_of_collision"] == 2e-4
    finally:
        conn.close()


def test_get_recent_runs_returns_newest_first(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "test.db")
    try:
        cfg = ScreeningConfig()
        record_screening_run(
            conn, PRIMARY, 5, cfg, run_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        record_screening_run(
            conn, PRIMARY, 5, cfg, run_timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc)
        )
        record_screening_run(
            conn, PRIMARY, 5, cfg, run_timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        runs = get_recent_runs(conn, limit=10)
        timestamps = [row["run_timestamp"] for row in runs]
        assert timestamps == sorted(timestamps, reverse=True)
    finally:
        conn.close()


def test_get_connection_is_idempotent_on_existing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn1 = get_connection(db_path)
    cfg = ScreeningConfig()
    record_screening_run(conn1, PRIMARY, 5, cfg)
    conn1.close()

    # Reopening shouldn't wipe existing data or fail on CREATE TABLE.
    conn2 = get_connection(db_path)
    try:
        runs = get_recent_runs(conn2)
        assert len(runs) == 1
    finally:
        conn2.close()
