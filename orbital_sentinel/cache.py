"""Local disk cache enforcing Space-Track's documented per-class query
frequency limits (see https://www.space-track.org/documentation —
"API Use Guidelines"):

    GP (aka TLEs)  — at most 1 query / hour
    CDM            — at most 1 query / 8 hours for all-constellation pulls

These aren't just courtesy limits — Space-Track states accounts may be
suspended for exceeding them. main.py previously called both classes on
every single run, which is fine for the scheduled 6-hourly GitHub Actions
job in isolation, but breaks the moment anyone (rightly) runs the
pipeline more often during development or manual checks. This module
makes the TTL structural: repeated runs within the window transparently
reuse the last fetch instead of hitting Space-Track again.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GP_CACHE_TTL_S = 3600.0  # 1 hour, per Space-Track's GP class limit
CDM_CACHE_TTL_S = 8 * 3600.0  # 8 hours, per Space-Track's all-constellation CDM limit


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def load_cached(cache_dir: str | Path, key: str, max_age_s: float) -> Any | None:
    """Return the cached payload for `key` if it exists and is younger
    than max_age_s, otherwise None (meaning: caller should fetch fresh).
    """
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(wrapper["cached_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None  # corrupt cache file — treat as a miss, will be overwritten
    age_s = (datetime.now(timezone.utc) - cached_at).total_seconds()
    if age_s > max_age_s:
        return None
    return wrapper["data"]


def save_cache(cache_dir: str | Path, key: str, data: Any) -> None:
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {"cached_at": datetime.now(timezone.utc).isoformat(), "data": data}
    path.write_text(json.dumps(wrapper), encoding="utf-8")


def cache_age_s(cache_dir: str | Path, key: str) -> float | None:
    """Age of the cached entry in seconds, or None if there isn't one."""
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        wrapper = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(wrapper["cached_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    return (datetime.now(timezone.utc) - cached_at).total_seconds()
