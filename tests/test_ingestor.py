from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from orbital_sentinel.ingestor import (
    SpaceTrackAuthError,
    SpaceTrackClient,
    SpaceTrackRequestError,
    _RateLimiter,
    parse_cdm_records,
    parse_gp_catalog,
)


def test_parse_gp_catalog_skips_missing_lines() -> None:
    raw = [
        {
            "NORAD_CAT_ID": "25544",
            "OBJECT_NAME": "ISS (ZARYA)",
            "TLE_LINE1": "1 25544U ...",
            "TLE_LINE2": "2 25544 ...",
        },
        {"NORAD_CAT_ID": "99999", "OBJECT_NAME": "NO TLE"},
    ]
    objects = parse_gp_catalog(raw)
    assert len(objects) == 1
    assert objects[0].norad_id == 25544
    assert objects[0].name == "ISS (ZARYA)"


def test_parse_cdm_records_uses_correct_field_names() -> None:
    # Field names confirmed against Space-Track's own modeldef for
    # cdm_public: SAT_1_ID, SAT_2_ID, MIN_RNG, PC, SAT2_OBJECT_TYPE,
    # EMERGENCY_REPORTABLE — not the full CCSDS CDM field names.
    raw = [
        {
            "CDM_ID": "1",
            "SAT_1_ID": "25544",
            "SAT_2_ID": "12345",
            "SAT_2_NAME": "COSMOS 1234 DEB",
            "TCA": "2024-01-01T00:00:00",
            "MIN_RNG": "1.5",
            "PC": "0.0002",
            "SAT2_OBJECT_TYPE": "DEBRIS",
            "EMERGENCY_REPORTABLE": "Y",
        },
    ]
    records = parse_cdm_records(raw)
    assert len(records) == 1
    r = records[0]
    assert r.primary_norad_id == 25544
    assert r.secondary_norad_id == 12345
    assert r.secondary_name == "COSMOS 1234 DEB"
    assert r.miss_distance_km == pytest.approx(1.5)
    assert r.collision_probability == pytest.approx(0.0002)
    assert r.secondary_object_type == "DEBRIS"
    assert r.emergency_reportable is True


def test_parse_cdm_records_handles_missing_probability() -> None:
    raw = [
        {
            "CDM_ID": "2",
            "SAT_1_ID": "25544",
            "SAT_2_ID": "54321",
            "TCA": "2024-01-01T00:00:00",
            "MIN_RNG": "2.0",
            "PC": "",
        }
    ]
    records = parse_cdm_records(raw)
    assert len(records) == 1
    assert records[0].collision_probability is None


def test_parse_cdm_records_skips_malformed() -> None:
    raw = [{"CDM_ID": "3", "SAT_1_ID": "not-a-number"}]
    records = parse_cdm_records(raw)
    assert records == []


def test_parse_cdm_records_skips_missing_miss_distance() -> None:
    raw = [
        {
            "CDM_ID": "4",
            "SAT_1_ID": "25544",
            "SAT_2_ID": "12345",
        }
    ]
    assert parse_cdm_records(raw) == []



def test_client_requires_credentials() -> None:
    client = SpaceTrackClient(username=None, password=None)
    with pytest.raises(SpaceTrackAuthError):
        client.login()


def test_rate_limiter_enforces_minimum_interval() -> None:
    limiter = _RateLimiter(min_interval_s=0.05)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05


@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_login_detects_200_with_failure_body(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"Login": "Failed"}'
    mock_post.return_value = mock_resp

    client = SpaceTrackClient(username="user", password="wrong", min_request_interval_s=0.0)
    with pytest.raises(SpaceTrackAuthError):
        client.login()


@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_login_success_sets_authenticated(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "[]"
    mock_post.return_value = mock_resp

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    client.login()
    assert client._authenticated is True


@patch("orbital_sentinel.ingestor.requests.Session.get")
@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_get_json_retries_on_429_then_succeeds(
    mock_post: MagicMock, mock_get: MagicMock
) -> None:
    mock_login_resp = MagicMock(status_code=200, text="[]")
    mock_post.return_value = mock_login_resp

    rate_limited_resp = MagicMock(status_code=429, text="")
    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = [{"NORAD_CAT_ID": "1"}]
    mock_get.side_effect = [rate_limited_resp, ok_resp]

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    with patch("orbital_sentinel.ingestor.time.sleep"):
        result = client._get_json("https://www.space-track.org/fake")
    assert result == [{"NORAD_CAT_ID": "1"}]
    assert mock_get.call_count == 2


@patch("orbital_sentinel.ingestor.requests.Session.get")
@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_get_json_waits_out_spacetrack_per_minute_rate_limit(
    mock_post: MagicMock, mock_get: MagicMock
) -> None:
    """Space-Track's per-minute rate limit surfaces as HTTP 500. Confirm
    we wait the full cooldown rather than the generic short backoff,
    regardless of the exact error body wording.
    """
    mock_post.return_value = MagicMock(status_code=200, text="[]")

    limited_resp = MagicMock(status_code=500, text="You have violated your query rate limit")
    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = [{"NORAD_CAT_ID": "1"}]
    mock_get.side_effect = [limited_resp, ok_resp]

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    with patch("orbital_sentinel.ingestor.time.sleep") as mock_sleep:
        result = client._get_json("https://www.space-track.org/fake")
    assert result == [{"NORAD_CAT_ID": "1"}]
    assert mock_get.call_count == 2
    from orbital_sentinel.ingestor import RATE_LIMIT_COOLDOWN_S

    mock_sleep.assert_any_call(RATE_LIMIT_COOLDOWN_S)


@patch("orbital_sentinel.ingestor.requests.Session.get")
@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_get_json_waits_out_500_even_with_html_error_body(
    mock_post: MagicMock, mock_get: MagicMock
) -> None:
    """Observed in production: Space-Track sometimes returns a 500 with a
    full HTML error page rather than the JSON rate-limit phrase. We must
    still treat it as a rate-limit cooldown, not give up after the
    generic short backoff — this is what caused a real outage before the
    fix (three quick retries all still inside the same rate-limit window).
    """
    mock_post.return_value = MagicMock(status_code=200, text="[]")

    html_error_resp = MagicMock(
        status_code=500,
        text="<!DOCTYPE html><html><head><title>Space-Track.Org</title></head></html>",
    )
    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = [{"NORAD_CAT_ID": "1"}]
    mock_get.side_effect = [html_error_resp, ok_resp]

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    with patch("orbital_sentinel.ingestor.time.sleep") as mock_sleep:
        result = client._get_json("https://www.space-track.org/fake")
    assert result == [{"NORAD_CAT_ID": "1"}]
    from orbital_sentinel.ingestor import RATE_LIMIT_COOLDOWN_S

    mock_sleep.assert_any_call(RATE_LIMIT_COOLDOWN_S)


@patch("orbital_sentinel.ingestor.requests.Session.get")
@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_get_json_raises_on_non_retryable_4xx(
    mock_post: MagicMock, mock_get: MagicMock
) -> None:
    mock_post.return_value = MagicMock(status_code=200, text="[]")
    mock_get.return_value = MagicMock(status_code=400)

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    with pytest.raises(SpaceTrackRequestError):
        client._get_json("https://www.space-track.org/fake")
    assert mock_get.call_count == 1  # no retry on a plain 4xx


@patch("orbital_sentinel.ingestor.requests.Session.get")
@patch("orbital_sentinel.ingestor.requests.Session.post")
def test_fetch_catalog_by_altitude_band_excludes_primary(
    mock_post: MagicMock, mock_get: MagicMock
) -> None:
    mock_post.return_value = MagicMock(status_code=200, text="[]")
    catalog_resp = MagicMock(status_code=200)
    catalog_resp.json.return_value = [
        {"NORAD_CAT_ID": "25544"},
        {"NORAD_CAT_ID": "88888"},
    ]
    mock_get.return_value = catalog_resp

    client = SpaceTrackClient(username="user", password="pass", min_request_interval_s=0.0)
    records = client.fetch_catalog_by_altitude_band(
        perigee_km=400.0, apogee_km=420.0, exclude_norad_id=25544
    )
    assert len(records) == 1
    assert records[0]["NORAD_CAT_ID"] == "88888"
