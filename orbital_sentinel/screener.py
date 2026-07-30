"""Conjunction screening: detect close approaches between a primary object
and a catalog of conjunctor objects across a time window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from orbital_sentinel.config import ScreeningConfig
from orbital_sentinel.propagator import SGP4Propagator, StateVector, TLEObject


@dataclass
class ConjunctionEvent:
    primary_norad_id: int
    secondary_norad_id: int
    secondary_name: str
    tca: datetime  # time of closest approach
    miss_distance_km: float
    relative_velocity_km_s: float


def _relative_state(a: StateVector, b: StateVector) -> tuple[float, float]:
    """Return (range_km, relative_speed_km_s) between two states at the same epoch."""
    dr = a.r_km - b.r_km
    dv = a.v_km_s - b.v_km_s
    return float(np.linalg.norm(dr)), float(np.linalg.norm(dv))


def _is_colocated(ranges: list[float], std_threshold_km: float) -> bool:
    """Detect catalog duplicates riding along with the primary (e.g. ISS
    modules like UNITY/ZVEZDA/DESTINY/NAUKA catalogued as separate NORAD
    objects). A real conjunction shows a V-shaped range profile — the
    separation decreases toward TCA and increases afterward. Physically
    attached or co-orbiting hardware instead sits at a near-constant tiny
    separation for the entire window. We flag low-variance-and-tiny-range
    pairs as co-located rather than genuine close approaches.
    """
    if len(ranges) < 3:
        return False
    if min(ranges) > std_threshold_km:
        return False  # not tiny to begin with — not a co-location candidate
    return bool(np.std(ranges) < std_threshold_km)


def screen_pair(
    primary: TLEObject,
    secondary: TLEObject,
    start: datetime,
    end: datetime,
    cfg: ScreeningConfig,
    coarse_step_s: float = 60.0,
    refine_step_s: float = 1.0,
) -> ConjunctionEvent | None:
    """Coarse SGP4 sweep across the window, then refine near the local
    minimum to locate TCA and miss distance more precisely.
    Returns None if no approach under the configured threshold is found,
    or if the pair looks like a co-located catalog duplicate rather than
    a genuine conjunction (see `_is_colocated`).
    """
    prop_p = SGP4Propagator(primary)
    prop_s = SGP4Propagator(secondary)

    states_p = prop_p.sample_window(start, end, coarse_step_s)
    states_s = prop_s.sample_window(start, end, coarse_step_s)

    ranges = [
        _relative_state(sp, ss)[0] for sp, ss in zip(states_p, states_s, strict=True)
    ]
    if not ranges:
        return None

    if _is_colocated(ranges, cfg.colocation_std_threshold_km):
        return None

    min_idx = int(np.argmin(ranges))
    if ranges[min_idx] > cfg.miss_distance_threshold_km * 5:
        # Not even close on the coarse grid — skip expensive refinement.
        return None

    # Refine in a window around the coarse minimum.
    refine_start = states_p[max(min_idx - 1, 0)].epoch
    refine_end = states_p[min(min_idx + 1, len(states_p) - 1)].epoch
    if refine_end <= refine_start:
        refine_end = refine_start

    fine_p = prop_p.sample_window(refine_start, refine_end, refine_step_s)
    fine_s = prop_s.sample_window(refine_start, refine_end, refine_step_s)

    if not fine_p:
        best_p, best_s = states_p[min_idx], states_s[min_idx]
    else:
        fine_ranges = [
            _relative_state(sp, ss)[0] for sp, ss in zip(fine_p, fine_s, strict=True)
        ]
        fine_idx = int(np.argmin(fine_ranges))
        best_p, best_s = fine_p[fine_idx], fine_s[fine_idx]

    miss_km, rel_v = _relative_state(best_p, best_s)
    if miss_km > cfg.miss_distance_threshold_km:
        return None

    return ConjunctionEvent(
        primary_norad_id=primary.norad_id,
        secondary_norad_id=secondary.norad_id,
        secondary_name=secondary.name,
        tca=best_p.epoch,
        miss_distance_km=miss_km,
        relative_velocity_km_s=rel_v,
    )


def screen_catalog(
    primary: TLEObject,
    catalog: list[TLEObject],
    start: datetime,
    end: datetime,
    cfg: ScreeningConfig,
) -> list[ConjunctionEvent]:
    """Screen the primary object against an entire conjunctor catalog."""
    events: list[ConjunctionEvent] = []
    for secondary in catalog:
        if secondary.norad_id == primary.norad_id:
            continue
        try:
            event = screen_pair(primary, secondary, start, end, cfg)
        except RuntimeError:
            continue  # skip objects with bad/expired elements
        if event is not None:
            events.append(event)
    events.sort(key=lambda e: e.miss_distance_km)
    return events
