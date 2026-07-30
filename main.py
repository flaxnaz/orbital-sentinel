"""orbital-sentinel pipeline entrypoint.

Orchestrates: ingest -> screen -> classify -> respond (CAM) -> present.

Run with live Space-Track credentials (see .env.example):
    python main.py

Or against a local fixture catalog for offline testing:
    python main.py --offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from orbital_sentinel.cache import CDM_CACHE_TTL_S, GP_CACHE_TTL_S, load_cached, save_cache
from orbital_sentinel.cam_planner import plan_all
from orbital_sentinel.classifier import classify_all
from orbital_sentinel.config import EARTH_RADIUS_KM, Config
from orbital_sentinel.database import (
    get_connection,
    get_pc_history,
    record_cam_recommendations,
    record_classified_conjunctions,
    record_screening_run,
)
from orbital_sentinel.ingestor import (
    SpaceTrackAuthError,
    SpaceTrackClient,
    SpaceTrackRequestError,
    parse_cdm_records,
    parse_gp_catalog,
)
from orbital_sentinel.propagator import TLEObject
from orbital_sentinel.reporter import write_csv, write_dashboard
from orbital_sentinel.screener import screen_catalog

logger = logging.getLogger(__name__)

# Small offline fixture catalog so the pipeline is runnable and testable
# without live Space-Track credentials.
_OFFLINE_PRIMARY = TLEObject(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
_OFFLINE_CATALOG = [
    TLEObject(
        norad_id=88888,
        name="DEBRIS-TEST-1",
        line1="1 88888U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        line2="2 88888  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    ),
]
_OFFLINE_SEMI_MAJOR_AXIS_KM = EARTH_RADIUS_KM + 400.0


def _map_object_type(spacetrack_type: str | None) -> str:
    """Map Space-Track's SAT2_OBJECT_TYPE (PAYLOAD/ROCKET BODY/DEBRIS/
    UNKNOWN/TBA) onto our coarser ObjectType. CDMs don't signal
    manoeuvring status, so that tier is only ever set by the TLE-based
    heuristic in classifier.classify_object_type, not from CDM data.
    """
    if spacetrack_type is None:
        return "DEBRIS"
    normalized = spacetrack_type.strip().upper()
    if normalized == "PAYLOAD":
        return "ACTIVE_SATELLITE"
    return "DEBRIS"


def _ingest_live(
    cfg: Config, cache_dir: str = "data/cache"
) -> tuple[TLEObject, list[TLEObject], float, list]:
    """Fetch the primary object and an altitude-band-filtered conjunctor
    catalog from Space-Track, plus any recent public CDMs for the primary.
    Returns (primary, catalog, primary_semi_major_axis_km, cdm_records).

    Enforces Space-Track's documented query-frequency limits (GP: 1/hour,
    CDM: 1/8 hours for all-constellation pulls) via a local TTL cache —
    repeated runs within the window are served from disk instead of
    hitting Space-Track again, regardless of how often main.py itself is
    invoked (e.g. during manual testing between scheduled runs).
    """
    primary_key = f"gp_primary_{cfg.screening.primary_norad_id}"
    catalog_key = f"gp_catalog_{cfg.screening.primary_norad_id}"
    cdm_key = "cdm_public_all"

    primary_raw = load_cached(cache_dir, primary_key, GP_CACHE_TTL_S)
    catalog_raw = load_cached(cache_dir, catalog_key, GP_CACHE_TTL_S)
    cdm_raw = load_cached(cache_dir, cdm_key, CDM_CACHE_TTL_S)

    if primary_raw is None or catalog_raw is None or cdm_raw is None:
        with SpaceTrackClient() as client:
            if primary_raw is None:
                primary_raw = client.fetch_primary_gp(cfg.screening.primary_norad_id)
                if primary_raw is None:
                    raise RuntimeError(
                        f"No GP record found for NORAD ID {cfg.screening.primary_norad_id}"
                    )
                save_cache(cache_dir, primary_key, primary_raw)
            else:
                logger.info("Using cached GP primary record (Space-Track limit: 1 query/hour)")

            # NOTE: Space-Track's actual field names are PERIAPSIS/APOAPSIS,
            # not PERIGEE/APOGEE (confirmed against a live gp record).
            perigee_km = float(primary_raw.get("PERIAPSIS", 0.0) or 0.0)
            apogee_km = float(primary_raw.get("APOAPSIS", 0.0) or 0.0)

            if catalog_raw is None:
                catalog_raw = client.fetch_catalog_by_altitude_band(
                    perigee_km=perigee_km,
                    apogee_km=apogee_km,
                    exclude_norad_id=cfg.screening.primary_norad_id,
                )
                save_cache(cache_dir, catalog_key, catalog_raw)
            else:
                logger.info(
                    "Using cached GP altitude-band catalog (Space-Track limit: 1 query/hour)"
                )

            if cdm_raw is None:
                cdm_raw = client.fetch_cdms(days=2)
                save_cache(cache_dir, cdm_key, cdm_raw)
            else:
                logger.info(
                    "Using cached CDM data (Space-Track limit: 1 query/8 hours for all-constellation pulls)"
                )
    else:
        logger.info("All Space-Track data served from local cache — no live queries made")

    primary_objects = parse_gp_catalog([primary_raw])
    if not primary_objects:
        raise RuntimeError(
            f"Primary NORAD ID {cfg.screening.primary_norad_id} has no valid TLE lines"
        )
    primary = primary_objects[0]
    perigee_km = float(primary_raw.get("PERIAPSIS", 0.0) or 0.0)
    apogee_km = float(primary_raw.get("APOAPSIS", 0.0) or 0.0)
    semi_major_axis_km = EARTH_RADIUS_KM + (perigee_km + apogee_km) / 2.0

    catalog = parse_gp_catalog(catalog_raw)
    cdm_records = [
        r
        for r in parse_cdm_records(cdm_raw)
        if primary.norad_id in (r.primary_norad_id, r.secondary_norad_id)
    ]

    logger.info(
        "Ingested primary %s (%d), %d conjunctor candidates in altitude band, %d relevant CDMs",
        primary.name,
        primary.norad_id,
        len(catalog),
        len(cdm_records),
    )
    return primary, catalog, semi_major_axis_km, cdm_records


def run(offline: bool, config_path: str, output_dir: str, db_path: str, cache_dir: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()  # reads .env in the current directory into os.environ, if present
    cfg = Config.from_yaml(config_path)

    cdm_records: list = []
    if offline:
        primary = _OFFLINE_PRIMARY
        catalog = _OFFLINE_CATALOG
        semi_major_axis_km = _OFFLINE_SEMI_MAJOR_AXIS_KM
    else:
        try:
            primary, catalog, semi_major_axis_km, cdm_records = _ingest_live(cfg, cache_dir)
        except SpaceTrackAuthError as exc:
            print(f"Space-Track auth failed ({exc}); falling back to --offline mode.", file=sys.stderr)
            primary, catalog = _OFFLINE_PRIMARY, _OFFLINE_CATALOG
            semi_major_axis_km = _OFFLINE_SEMI_MAJOR_AXIS_KM
        except SpaceTrackRequestError as exc:
            print(f"Space-Track query failed ({exc}); falling back to --offline mode.", file=sys.stderr)
            primary, catalog = _OFFLINE_PRIMARY, _OFFLINE_CATALOG
            semi_major_axis_km = _OFFLINE_SEMI_MAJOR_AXIS_KM

    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=cfg.screening.window_hours)

    events = screen_catalog(primary, catalog, start, end, cfg.screening)

    reported_pcs = {
        r.secondary_norad_id: r.collision_probability
        for r in cdm_records
        if r.collision_probability is not None
    }
    object_types = {
        r.secondary_norad_id: _map_object_type(r.secondary_object_type) for r in cdm_records
    }
    classified = classify_all(
        events,
        cfg.screening.hard_body_radius_km,
        cfg.classifier,
        object_types=object_types,
        reported_pcs=reported_pcs,
    )

    cams = plan_all(classified, semi_major_axis_km, cfg.cam)
    cam_by_id = {c.secondary_norad_id: c for c in cams}

    conn = get_connection(db_path)
    try:
        run_id = record_screening_run(conn, primary, len(catalog), cfg.screening)
        conjunction_id_by_secondary = record_classified_conjunctions(conn, run_id, classified)
        record_cam_recommendations(conn, conjunction_id_by_secondary, cams)

        pc_histories = {
            c.event.secondary_norad_id: [
                row["probability_of_collision"]
                for row in get_pc_history(conn, c.event.secondary_norad_id)
            ]
            for c in classified
        }
    finally:
        conn.close()

    write_dashboard(
        classified,
        cams,
        f"{output_dir}/index.html",
        pc_histories=pc_histories,
        slider_min_km=1.0,
        slider_max_km=cfg.screening.miss_distance_threshold_km,
        slider_default_km=cfg.dashboard.default_alert_km,
    )
    write_csv(classified, cam_by_id, f"{output_dir}/report.csv")

    red_count = sum(1 for c in classified if c.risk_tier == "RED")
    print(
        f"Screened {len(catalog)} objects, {len(events)} conjunctions found, "
        f"{red_count} RED tier, {len(cams)} CAM(s) recommended, "
        f"{len(cdm_records)} matching public CDM(s). "
        f"Saved to {db_path} (run_id={run_id})."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="orbital-sentinel pipeline")
    parser.add_argument("--offline", action="store_true", help="run against local fixture catalog")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--output-dir", default="dashboard", help="output directory for reports")
    parser.add_argument(
        "--db-path", default="data/orbital_sentinel.db", help="path to the SQLite database file"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="directory for the Space-Track query cache (enforces documented rate limits)",
    )
    args = parser.parse_args()
    return run(
        offline=args.offline,
        config_path=args.config,
        output_dir=args.output_dir,
        db_path=args.db_path,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
