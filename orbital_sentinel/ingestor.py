"""Space-Track ingestion: live GP (TLE-equivalent) catalog pull and public
CDM (Conjunction Data Message) pull.

This client is deliberately conservative about how it talks to Space-Track:

- Credentials come from environment variables (SPACETRACK_USER,
  SPACETRACK_PASS), matching the .env pattern used in conjunction-screener.
- Every request is rate-limited client-side (Space-Track's fair-use policy
  asks API consumers not to hammer the service). We space requests at
  least MIN_REQUEST_INTERVAL_S apart and cap ourselves well under the
  documented ~30 requests/minute ceiling.
- Login failures are detected properly: Space-Track can return HTTP 200
  with an error message in the body for bad credentials, not just a
  non-200 status, so we check the response body too.
- Transient failures (429 rate-limited, 5xx) are retried with exponential
  backoff; auth failures and other 4xx client errors are not retried.
- The GP catalog is never pulled in bulk for screening. Space-Track's `gp`
  class exposes PERIAPSIS/APOAPSIS as query-able fields, so we filter
  server-side to an altitude band around the primary object instead of
  downloading the ~25,000+ object catalog and filtering in Python. This
  keeps both the network payload and the downstream SGP4 screening pass
  tractable.
- Network calls are isolated behind SpaceTrackClient so the rest of the
  pipeline can be tested against fixtures without live credentials.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import requests

from orbital_sentinel.propagator import TLEObject

logger = logging.getLogger(__name__)

SPACETRACK_BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE_URL}/ajaxauth/login"
LOGOUT_URL = f"{SPACETRACK_BASE_URL}/ajaxauth/logout"
QUERY_BASE = f"{SPACETRACK_BASE_URL}/basicspacedata/query"

MIN_REQUEST_INTERVAL_S = 3.0  # keep comfortably under ~30 req/min including login
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 3.0
RATE_LIMIT_COOLDOWN_S = 65.0  # Space-Track's per-minute window plus margin

DEFAULT_ALTITUDE_BUFFER_KM = 50.0
DEFAULT_CATALOG_LIMIT = 2000


class SpaceTrackAuthError(RuntimeError):
    """Raised when login fails (missing/invalid credentials, or Space-Track
    rejects the login attempt)."""


class SpaceTrackRequestError(RuntimeError):
    """Raised when a query fails after retries are exhausted."""


class _RateLimiter:
    """Enforces a minimum interval between successive requests."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


@dataclass
class CdmRecord:
    """A record from Space-Track's `cdm_public` class.

    Confirmed against the class's own modeldef (GET
    /basicspacedata/modeldef/class/cdm_public/format/json): this public
    class is a lightweight summary, NOT the full CCSDS 508.0-B-1 CDM —
    it has no covariance data, no relative position/velocity, and no
    COLLISION_PROBABILITY_METHOD. It provides CDM_ID, CREATED,
    EMERGENCY_REPORTABLE, TCA, MIN_RNG, PC, SAT_1_ID/NAME, SAT_2_ID/NAME,
    SAT{1,2}_OBJECT_TYPE, and SAT{1,2}_RCS only.
    """

    cdm_id: str
    primary_norad_id: int
    secondary_norad_id: int
    secondary_name: str | None
    tca_iso: str
    miss_distance_km: float
    collision_probability: float | None
    secondary_object_type: str | None
    emergency_reportable: bool | None


class SpaceTrackClient:
    """Authenticated, rate-limited client for Space-Track's REST API.

    Use as a context manager so the session is always logged out cleanly:

        with SpaceTrackClient() as client:
            primary = client.fetch_primary_gp(25544)
            catalog = client.fetch_catalog_by_altitude_band(...)
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        min_request_interval_s: float = MIN_REQUEST_INTERVAL_S,
    ) -> None:
        self.username = username or os.environ.get("SPACETRACK_USER")
        self.password = password or os.environ.get("SPACETRACK_PASS")
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "orbital-sentinel/0.1 (+https://github.com/flaxnaz/orbital-sentinel)"}
        )
        self._authenticated = False
        self._rate_limiter = _RateLimiter(min_request_interval_s)

    # -- session lifecycle -------------------------------------------------

    def __enter__(self) -> "SpaceTrackClient":
        self.login()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.logout()

    def login(self) -> None:
        if not self.username or not self.password:
            raise SpaceTrackAuthError(
                "SPACETRACK_USER / SPACETRACK_PASS not set (see .env.example)"
            )
        self._rate_limiter.wait()
        resp = self._session.post(
            LOGIN_URL,
            data={"identity": self.username, "password": self.password},
            timeout=30,
        )
        if resp.status_code != 200:
            raise SpaceTrackAuthError(
                f"Space-Track login failed: HTTP {resp.status_code}"
            )
        body_lower = resp.text.lower()
        if "fail" in body_lower or ("invalid" in body_lower and "identity" in body_lower):
            raise SpaceTrackAuthError(f"Space-Track login rejected: {resp.text[:200]}")
        self._authenticated = True
        logger.info("Space-Track: authenticated as %s", self.username)

    def logout(self) -> None:
        if not self._authenticated:
            return
        try:
            self._rate_limiter.wait()
            self._session.get(LOGOUT_URL, timeout=10)
        except requests.RequestException:
            pass
        finally:
            self._authenticated = False

    def _ensure_auth(self) -> None:
        if not self._authenticated:
            self.login()

    def _get_json(self, url: str) -> list[dict[str, Any]]:
        self._ensure_auth()
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limiter.wait()
            try:
                resp = self._session.get(url, timeout=60)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Space-Track request error (attempt %d): %s", attempt, exc)
                time.sleep(RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)))
                continue

            if resp.status_code == 200:
                result: list[dict[str, Any]] = resp.json()
                return result

            # Space-Track's per-minute query rate limit surfaces as HTTP
            # 500 (not 429), and the exact wording of the error body has
            # varied in practice (sometimes a JSON message, sometimes a
            # full HTML error page). Rather than pattern-match text, treat
            # any 500 as a likely rate-limit hit and wait out the full
            # window — a well-formed query (confirmed via diagnose.py
            # against live Space-Track) should not otherwise 500.
            if resp.status_code == 500:
                logger.warning(
                    "Space-Track HTTP 500 (attempt %d) — likely per-minute rate limit; "
                    "waiting %.0fs before retry",
                    attempt,
                    RATE_LIMIT_COOLDOWN_S,
                )
                last_exc = SpaceTrackRequestError(f"HTTP 500: {resp.text[:200]}")
                time.sleep(RATE_LIMIT_COOLDOWN_S)
                continue

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                logger.warning(
                    "Space-Track returned HTTP %d (attempt %d)", resp.status_code, attempt
                )
                last_exc = SpaceTrackRequestError(f"HTTP {resp.status_code}")
                time.sleep(RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)))
                continue

            # Any other 4xx: not retryable (bad query syntax, expired auth, etc.)
            raise SpaceTrackRequestError(
                f"Space-Track query failed: HTTP {resp.status_code} for {url}"
            )
        raise SpaceTrackRequestError(
            f"Space-Track query failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    def fetch_primary_gp(self, norad_id: int) -> dict[str, Any] | None:
        """Fetch the latest GP record for a single object by NORAD ID.
        Includes Space-Track's documented recommended filter
        (decay_date/null-val + epoch/now-10) to retrieve only propagable,
        current ephemerides — see the GP row of the API Use Guidelines.
        """
        url = (
            f"{QUERY_BASE}/class/gp/norad_cat_id/{norad_id}/"
            "decay_date/null-val/epoch/%3Enow-10/"
            "orderby/epoch desc/limit/1/format/json"
        )
        records = self._get_json(url)
        return records[0] if records else None

    def fetch_catalog_by_altitude_band(
        self,
        perigee_km: float,
        apogee_km: float,
        exclude_norad_id: int | None = None,
        buffer_km: float = DEFAULT_ALTITUDE_BUFFER_KM,
        max_epoch_age_days: int = 10,
        limit: int = DEFAULT_CATALOG_LIMIT,
    ) -> list[dict[str, Any]]:
        """Fetch non-decayed GP records whose perigee/apogee fall within a
        buffered band around the given altitudes (server-side filtering,
        equivalent to conjunction-screener's altitude-band auto-discovery).

        NOTE: Space-Track's actual query-able field names are PERIAPSIS
        and APOAPSIS, not PERIGEE/APOGEE — confirmed against a live gp
        record and by successfully running this exact query pattern.
        """
        perigee_min = max(perigee_km - buffer_km, 0.0)
        perigee_max = perigee_km + buffer_km
        apogee_min = max(apogee_km - buffer_km, 0.0)
        apogee_max = apogee_km + buffer_km
        url = (
            f"{QUERY_BASE}/class/gp/decay_date/null-val/"
            f"epoch/%3Enow-{max_epoch_age_days}/"
            f"PERIAPSIS/{perigee_min:.0f}--{perigee_max:.0f}/"
            f"APOAPSIS/{apogee_min:.0f}--{apogee_max:.0f}/"
            f"orderby/norad_cat_id/limit/{limit}/format/json"
        )
        records = self._get_json(url)
        if exclude_norad_id is not None:
            records = [
                r for r in records if int(r.get("NORAD_CAT_ID", -1)) != exclude_norad_id
            ]
        return records

    def fetch_cdms(self, days: int = 2) -> list[dict[str, Any]]:
        """Fetch recent public CDMs. `cdm_public` is a genuinely public
        Space-Track class (no special operator permissions required), but
        it only contains conjunctions 18 SDS has chosen to publish, so an
        empty result is expected behaviour, not an error. Note this class
        is a lightweight summary (CDM_ID, CREATED, TCA, MIN_RNG, PC,
        SAT_1/2_ID, SAT_1/2_NAME, SAT1/2_OBJECT_TYPE, SAT1/2_RCS,
        EMERGENCY_REPORTABLE) — it does not include covariance data.
        """
        url = (
            f"{QUERY_BASE}/class/cdm_public/"
            f"CREATED/%3Enow-{days}/orderby/CREATED desc/format/json"
        )
        return self._get_json(url)


def parse_gp_catalog(raw_records: list[dict[str, Any]]) -> list[TLEObject]:
    """Convert Space-Track GP JSON records into TLEObject instances."""
    objects: list[TLEObject] = []
    for rec in raw_records:
        line1 = rec.get("TLE_LINE1")
        line2 = rec.get("TLE_LINE2")
        if not line1 or not line2:
            continue
        objects.append(
            TLEObject(
                norad_id=int(rec.get("NORAD_CAT_ID", 0)),
                name=str(rec.get("OBJECT_NAME", "UNKNOWN")),
                line1=line1,
                line2=line2,
            )
        )
    return objects


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_cdm_records(raw_records: list[dict[str, Any]]) -> list[CdmRecord]:
    """Convert Space-Track public CDM JSON records into CdmRecord instances.

    Field names below are confirmed against the class's own modeldef
    (GET /basicspacedata/modeldef/class/cdm_public/format/json), not the
    full CCSDS 508.0-B-1 standard — `cdm_public` is a public summary
    class with a much smaller field set: CDM_ID, CREATED,
    EMERGENCY_REPORTABLE, TCA, MIN_RNG, PC, SAT_1_ID, SAT_1_NAME,
    SAT1_OBJECT_TYPE, SAT1_RCS, SAT_2_ID, SAT_2_NAME, SAT2_OBJECT_TYPE,
    SAT2_RCS. No covariance or relative-state data is exposed here.

    Unit note: MIN_RNG has no unit suffix in the modeldef (unlike the
    full CDM's explicit "MISS_DISTANCE [m]"). We assume km, consistent
    with Space-Track's convention elsewhere in the `gp` class
    (PERIAPSIS/APOAPSIS are km) — verify this against a few real records
    before trusting it operationally; if values look like they're in the
    hundreds-of-thousands range, they're metres and this needs a /1000.
    """
    records: list[CdmRecord] = []
    for rec in raw_records:
        try:
            primary_id = int(rec.get("SAT_1_ID", 0))
            secondary_id = int(rec.get("SAT_2_ID", 0))
        except (TypeError, ValueError):
            continue

        miss_distance_km = _to_float(rec.get("MIN_RNG"))
        if miss_distance_km is None:
            continue

        emergency_raw = rec.get("EMERGENCY_REPORTABLE")
        emergency: bool | None
        if emergency_raw in (None, ""):
            emergency = None
        else:
            emergency = str(emergency_raw).strip().upper() == "Y"

        records.append(
            CdmRecord(
                cdm_id=str(rec.get("CDM_ID", "")),
                primary_norad_id=primary_id,
                secondary_norad_id=secondary_id,
                secondary_name=rec.get("SAT_2_NAME") or None,
                tca_iso=str(rec.get("TCA", "")),
                miss_distance_km=miss_distance_km,
                collision_probability=_to_float(rec.get("PC")),
                secondary_object_type=rec.get("SAT2_OBJECT_TYPE") or None,
                emergency_reportable=emergency,
            )
        )
    return records
