"""Collision avoidance manoeuvre (CAM) planning using the Clohessy-Wiltshire
(CW) equations for short-timescale relative motion near TCA.

Reference relationship (mock rendezvous scenario used for validation):
    n = 0.00111 rad/s
    secular drift rate ~= 6 n x0
    delta-v per burn ~= n x0 / 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbital_sentinel.classifier import ClassifiedConjunction
from orbital_sentinel.config import CamConfig, MU_EARTH_KM3_S2

BurnDirection = str  # "RADIAL" | "ALONG_TRACK" | "CROSS_TRACK"


@dataclass
class CamRecommendation:
    secondary_norad_id: int
    burn_time_s_before_tca: float
    burn_direction: BurnDirection
    delta_v_m_s: float
    predicted_new_miss_distance_km: float
    propellant_cost_kg: float


def mean_motion_rad_s(semi_major_axis_km: float, mu_km3_s2: float = MU_EARTH_KM3_S2) -> float:
    """Mean motion n = sqrt(mu / a^3), rad/s."""
    return math.sqrt(mu_km3_s2 / semi_major_axis_km**3)


def cw_along_track_delta_v(
    n_rad_s: float, x0_km: float
) -> float:
    """Delta-v (km/s) for a single along-track impulsive burn that arrests
    the CW secular drift induced by a radial offset x0, using the
    validated relationship delta_v = n * x0 / 2.
    """
    return n_rad_s * x0_km / 2.0


def cw_secular_drift_rate(n_rad_s: float, x0_km: float) -> float:
    """Along-track secular drift rate (km per orbit-radian-time) induced
    by a radial offset x0: drift_rate = 6 * n * x0.
    """
    return 6.0 * n_rad_s * x0_km


def _propellant_mass_kg(
    delta_v_m_s: float, spacecraft_mass_kg: float, isp_s: float, g0_m_s2: float
) -> float:
    """Rocket equation propellant mass for a given delta-v."""
    mass_ratio = math.exp(delta_v_m_s / (isp_s * g0_m_s2))
    return spacecraft_mass_kg * (1.0 - 1.0 / mass_ratio)


def plan_cam(
    classified: ClassifiedConjunction,
    semi_major_axis_km: float,
    cfg: CamConfig,
) -> CamRecommendation | None:
    """Compute a collision avoidance manoeuvre for a Red-tier conjunction.
    Returns None for non-Red tiers (no manoeuvre recommended).

    Approach: treat the current miss distance as the CW radial offset x0
    that needs to grow to at least the configured safe threshold. Solve
    for the along-track impulsive delta-v (applied one orbit-quarter,
    i.e. pi/(2n) seconds, before TCA) that grows the along-track secular
    separation to the safe miss distance by TCA.
    """
    if classified.risk_tier != "RED":
        return None

    event = classified.event
    n = mean_motion_rad_s(semi_major_axis_km)

    current_miss_km = event.miss_distance_km
    required_growth_km = max(cfg.safe_miss_distance_km - current_miss_km, 0.0)

    if required_growth_km <= 0.0:
        # Already outside the safe threshold; no manoeuvre needed.
        return None

    # Burn applied a quarter-orbit before TCA gives the drift time to
    # accumulate the required along-track separation by TCA.
    burn_time_s_before_tca = math.pi / (2.0 * n)

    # Solve delta_v from the along-track secular drift relationship:
    # separation_at_tca ~= drift_rate * burn_time = 6 n x0 * burn_time,
    # and delta_v = n x0 / 2  =>  x0 = 2 delta_v / n
    # => separation = 6 n * (2 delta_v / n) * burn_time = 12 delta_v * burn_time
    # => delta_v = separation / (12 * burn_time)
    delta_v_km_s = required_growth_km / (12.0 * burn_time_s_before_tca)
    delta_v_m_s = delta_v_km_s * 1000.0

    predicted_new_miss_km = current_miss_km + required_growth_km

    propellant_kg = _propellant_mass_kg(
        delta_v_m_s, cfg.spacecraft_mass_kg, cfg.isp_s, cfg.g0_m_s2
    )

    return CamRecommendation(
        secondary_norad_id=event.secondary_norad_id,
        burn_time_s_before_tca=burn_time_s_before_tca,
        burn_direction="ALONG_TRACK",
        delta_v_m_s=delta_v_m_s,
        predicted_new_miss_distance_km=predicted_new_miss_km,
        propellant_cost_kg=propellant_kg,
    )


def plan_all(
    classified_events: list[ClassifiedConjunction],
    semi_major_axis_km: float,
    cfg: CamConfig,
) -> list[CamRecommendation]:
    """Plan CAMs for every Red-tier conjunction in a classified batch."""
    recommendations = []
    for c in classified_events:
        rec = plan_cam(c, semi_major_axis_km, cfg)
        if rec is not None:
            recommendations.append(rec)
    return recommendations
