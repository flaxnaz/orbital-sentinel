from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orbital_sentinel.cache import (
    CDM_CACHE_TTL_S,
    GP_CACHE_TTL_S,
    cache_age_s,
    load_cached,
    save_cache,
)


def test_save_and_load_cache_roundtrip(tmp_path: Path) -> None:
    save_cache(tmp_path, "gp_primary_25544", {"NORAD_CAT_ID": "25544"})
    result = load_cached(tmp_path, "gp_primary_25544", max_age_s=3600.0)
    assert result == {"NORAD_CAT_ID": "25544"}


def test_load_cached_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_cached(tmp_path, "does_not_exist", max_age_s=3600.0) is None


def test_load_cached_returns_none_when_expired(tmp_path: Path) -> None:
    path = tmp_path / "stale_key.json"
    stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
    path.write_text(json.dumps({"cached_at": stale_time.isoformat(), "data": [1, 2, 3]}))
    assert load_cached(tmp_path, "stale_key", max_age_s=GP_CACHE_TTL_S) is None


def test_load_cached_returns_data_when_within_ttl(tmp_path: Path) -> None:
    path = tmp_path / "fresh_key.json"
    fresh_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    path.write_text(json.dumps({"cached_at": fresh_time.isoformat(), "data": [1, 2, 3]}))
    assert load_cached(tmp_path, "fresh_key", max_age_s=GP_CACHE_TTL_S) == [1, 2, 3]


def test_load_cached_handles_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("not valid json{{{")
    assert load_cached(tmp_path, "corrupt", max_age_s=3600.0) is None


def test_cache_age_s_reports_none_when_missing(tmp_path: Path) -> None:
    assert cache_age_s(tmp_path, "nope") is None


def test_cache_age_s_reports_approximate_age(tmp_path: Path) -> None:
    save_cache(tmp_path, "aged_key", {"x": 1})
    age = cache_age_s(tmp_path, "aged_key")
    assert age is not None
    assert 0 <= age < 5  # just saved, should be near-zero


def test_cdm_ttl_is_eight_hours() -> None:
    assert CDM_CACHE_TTL_S == 8 * 3600.0


def test_gp_ttl_is_one_hour() -> None:
    assert GP_CACHE_TTL_S == 3600.0
