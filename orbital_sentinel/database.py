"""Persistence: stores each screening run, its classified conjunctions,
and any CAM recommendations in a SQLite database, so history accumulates
across runs (e.g. the 6-hourly GitHub Actions dashboard refresh) and Pc
trends over time become queryable — one of the "more than a screener"
goals from the project brief.

SQLite (not a hosted server) is deliberate: this pipeline runs on
ephemeral GitHub Actions runners, so the .db file is committed to the
repo alongside the dashboard on each scheduled run, exactly like
conjunction-screener already does for its dashboard HTML. No external
database service or extra credentials required.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from orbital_sentinel.cam_planner import CamRecommendation
from orbital_sentinel.classifier import ClassifiedConjunction
from orbital_sentinel.config import ScreeningConfig
from orbital_sentinel.propagator import TLEObject

SCHEMA = """
CREATE TABLE IF NOT EXISTS screening_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    primary_norad_id INTEGER NOT NULL,
    primary_name TEXT NOT NULL,
    catalog_size INTEGER NOT NULL,
    window_hours REAL NOT NULL,
    miss_distance_threshold_km REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conjunctions (
    conjunction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES screening_runs(run_id),
    secondary_norad_id INTEGER NOT NULL,
    secondary_name TEXT,
    tca TEXT NOT NULL,
    miss_distance_km REAL NOT NULL,
    relative_velocity_km_s REAL NOT NULL,
    probability_of_collision REAL NOT NULL,
    pc_method TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    object_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cam_recommendations (
    cam_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conjunction_id INTEGER NOT NULL REFERENCES conjunctions(conjunction_id),
    burn_time_s_before_tca REAL NOT NULL,
    burn_direction TEXT NOT NULL,
    delta_v_m_s REAL NOT NULL,
    predicted_new_miss_distance_km REAL NOT NULL,
    propellant_cost_kg REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conjunctions_secondary_tca
    ON conjunctions (secondary_norad_id, tca);

CREATE INDEX IF NOT EXISTS idx_conjunctions_run
    ON conjunctions (run_id);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection, creating the parent directory and schema if
    they don't exist yet. Row factory returns dict-like rows.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def record_screening_run(
    conn: sqlite3.Connection,
    primary: TLEObject,
    catalog_size: int,
    cfg: ScreeningConfig,
    run_timestamp: datetime | None = None,
) -> int:
    """Insert a row for this screening run and return its run_id."""
    ts = (run_timestamp or datetime.now(timezone.utc)).isoformat()
    cur = conn.execute(
        """
        INSERT INTO screening_runs
            (run_timestamp, primary_norad_id, primary_name, catalog_size,
             window_hours, miss_distance_threshold_km)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            primary.norad_id,
            primary.name,
            catalog_size,
            cfg.window_hours,
            cfg.miss_distance_threshold_km,
        ),
    )
    conn.commit()
    run_id = cur.lastrowid
    assert run_id is not None
    return run_id


def record_classified_conjunctions(
    conn: sqlite3.Connection,
    run_id: int,
    classified: list[ClassifiedConjunction],
) -> dict[int, int]:
    """Insert one row per classified conjunction for this run. Returns a
    map of secondary_norad_id -> conjunction_id so CAM recommendations
    for the same run can be linked to the right row.
    """
    conjunction_id_by_secondary: dict[int, int] = {}
    for c in classified:
        cur = conn.execute(
            """
            INSERT INTO conjunctions
                (run_id, secondary_norad_id, secondary_name, tca,
                 miss_distance_km, relative_velocity_km_s,
                 probability_of_collision, pc_method, risk_tier, object_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                c.event.secondary_norad_id,
                c.event.secondary_name,
                c.event.tca.isoformat(),
                c.event.miss_distance_km,
                c.event.relative_velocity_km_s,
                c.probability_of_collision,
                c.pc_method,
                c.risk_tier,
                c.object_type,
            ),
        )
        conjunction_id = cur.lastrowid
        assert conjunction_id is not None
        conjunction_id_by_secondary[c.event.secondary_norad_id] = conjunction_id
    conn.commit()
    return conjunction_id_by_secondary


def record_cam_recommendations(
    conn: sqlite3.Connection,
    conjunction_id_by_secondary: dict[int, int],
    cams: list[CamRecommendation],
) -> None:
    """Insert one row per CAM recommendation, linked to its conjunction
    via the map returned by record_classified_conjunctions for the same
    run. CAMs whose secondary isn't in the map are skipped (shouldn't
    happen in normal use, but avoids a hard failure on a mismatch).
    """
    for cam in cams:
        conjunction_id = conjunction_id_by_secondary.get(cam.secondary_norad_id)
        if conjunction_id is None:
            continue
        conn.execute(
            """
            INSERT INTO cam_recommendations
                (conjunction_id, burn_time_s_before_tca, burn_direction,
                 delta_v_m_s, predicted_new_miss_distance_km, propellant_cost_kg)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conjunction_id,
                cam.burn_time_s_before_tca,
                cam.burn_direction,
                cam.delta_v_m_s,
                cam.predicted_new_miss_distance_km,
                cam.propellant_cost_kg,
            ),
        )
    conn.commit()


def get_recent_runs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    """Most recent screening runs, newest first."""
    cur = conn.execute(
        "SELECT * FROM screening_runs ORDER BY run_timestamp DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def get_pc_history(conn: sqlite3.Connection, secondary_norad_id: int) -> list[sqlite3.Row]:
    """Every recorded conjunction for a given secondary object, oldest
    first — the raw material for a Pc-over-time trend plot as TCA
    approaches, joined against the run timestamp it was observed at.
    """
    cur = conn.execute(
        """
        SELECT c.*, r.run_timestamp
        FROM conjunctions c
        JOIN screening_runs r ON r.run_id = c.run_id
        WHERE c.secondary_norad_id = ?
        ORDER BY r.run_timestamp ASC
        """,
        (secondary_norad_id,),
    )
    return cur.fetchall()


def get_conjunctions_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM conjunctions WHERE run_id = ? ORDER BY probability_of_collision DESC",
        (run_id,),
    )
    return cur.fetchall()
