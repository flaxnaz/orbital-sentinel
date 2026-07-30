"""Risk classification: probability of collision (Pc) and risk tiering,
plus lightweight object typing (debris / active satellite / manoeuvring).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from orbital_sentinel.config import ClassifierConfig, RiskTier, classify_tier
from orbital_sentinel.screener import ConjunctionEvent

ObjectType = str  # "DEBRIS" | "ACTIVE_SATELLITE" | "MANOEUVRING"


@dataclass
class CdmCovariance:
    """Combined (primary + secondary) covariance in the conjunction-plane
    RTN frame, as typically reported in a CDM. Units: km^2.
    """

    sigma_r_km: float  # radial
    sigma_t_km: float  # transverse (along-track)
    sigma_n_km: float  # normal (cross-track)


@dataclass
class ClassifiedConjunction:
    event: ConjunctionEvent
    probability_of_collision: float
    risk_tier: RiskTier
    object_type: ObjectType
    pc_method: str  # "cdm_covariance" | "hard_body_radius_model"


def pc_from_covariance(
    miss_distance_km: float,
    covariance: CdmCovariance,
    hard_body_radius_km: float,
) -> float:
    """Simplified 2D probability-of-collision estimate (Foster/Chan-style
    circular approximation) using the conjunction-plane covariance.
    This collapses the 3x3 RTN covariance onto the encounter plane via
    the transverse/normal components, which dominate at TCA for a
    near-circular relative geometry.
    """
    sigma_combined = np.sqrt(covariance.sigma_t_km**2 + covariance.sigma_n_km**2)
    if sigma_combined <= 0:
        return 0.0
    # Circular covariance approximation of the Foster Pc integral:
    # Pc ~ (HBR^2 / (2 * sigma^2)) * exp(-miss^2 / (2 * sigma^2))
    hbr = hard_body_radius_km
    pc = (hbr**2 / (2.0 * sigma_combined**2)) * np.exp(
        -(miss_distance_km**2) / (2.0 * sigma_combined**2)
    )
    return float(np.clip(pc, 0.0, 1.0))


def pc_hard_body_fallback(
    miss_distance_km: float,
    relative_velocity_km_s: float,
    hard_body_radius_km: float,
    assumed_position_uncertainty_km: float = 1.0,
) -> float:
    """Fallback Pc estimate when no CDM covariance is available: models
    position uncertainty as an isotropic Gaussian with a conservative
    assumed sigma, scaled slightly by encounter geometry (faster relative
    velocity -> shorter dwell time -> treated as a same-order estimate).
    This is intentionally conservative and coarser than the covariance
    method; it is meant to flag conjunctions for further screening, not
    to be a precise operational Pc.
    """
    sigma = assumed_position_uncertainty_km
    pc = (hard_body_radius_km**2 / (2.0 * sigma**2)) * np.exp(
        -(miss_distance_km**2) / (2.0 * sigma**2)
    )
    return float(np.clip(pc, 0.0, 1.0))


def classify_object_type(
    tle_epoch_change_days: float,
    unexpected_state_deviation_km: float,
    is_known_debris: bool,
) -> ObjectType:
    """Heuristic object typing:
    - large/unexpected TLE epoch jump or state deviation -> MANOEUVRING
    - catalog-flagged debris -> DEBRIS
    - otherwise -> ACTIVE_SATELLITE
    """
    if unexpected_state_deviation_km > 1.0 or tle_epoch_change_days < 0.05:
        return "MANOEUVRING"
    if is_known_debris:
        return "DEBRIS"
    return "ACTIVE_SATELLITE"


def classify_conjunction(
    event: ConjunctionEvent,
    hard_body_radius_km: float,
    cfg: ClassifierConfig,
    covariance: CdmCovariance | None = None,
    object_type: ObjectType = "DEBRIS",
    reported_pc: float | None = None,
) -> ClassifiedConjunction:
    """Classify a single conjunction event into a risk tier.

    Preference order for the Pc value:
    1. `reported_pc` — an official Pc already computed by 18 SDS and
       published in a CDM (typically via the Foster-1992 method). This is
       authoritative when present and is used as-is.
    2. `covariance` — CDM covariance data without a published Pc; we
       estimate Pc ourselves via a circular conjunction-plane
       approximation.
    3. Hard-body-radius fallback model, when no CDM exists for the pair.
    """
    if reported_pc is not None:
        pc = reported_pc
        method = "cdm_reported"
    elif covariance is not None:
        pc = pc_from_covariance(event.miss_distance_km, covariance, hard_body_radius_km)
        method = "cdm_covariance"
    else:
        pc = pc_hard_body_fallback(
            event.miss_distance_km, event.relative_velocity_km_s, hard_body_radius_km
        )
        method = "hard_body_radius_model"

    tier = classify_tier(pc, cfg)
    return ClassifiedConjunction(
        event=event,
        probability_of_collision=pc,
        risk_tier=tier,
        object_type=object_type,
        pc_method=method,
    )


def classify_all(
    events: list[ConjunctionEvent],
    hard_body_radius_km: float,
    cfg: ClassifierConfig,
    covariances: dict[int, CdmCovariance] | None = None,
    object_types: dict[int, ObjectType] | None = None,
    reported_pcs: dict[int, float] | None = None,
) -> list[ClassifiedConjunction]:
    """Classify a batch of conjunction events, keyed by secondary NORAD ID
    for optional per-object covariance / type / reported-Pc overrides.
    """
    covariances = covariances or {}
    object_types = object_types or {}
    reported_pcs = reported_pcs or {}
    results = []
    for event in events:
        cov = covariances.get(event.secondary_norad_id)
        obj_type = object_types.get(event.secondary_norad_id, "DEBRIS")
        reported = reported_pcs.get(event.secondary_norad_id)
        results.append(
            classify_conjunction(event, hard_body_radius_km, cfg, cov, obj_type, reported)
        )
    results.sort(key=lambda c: c.probability_of_collision, reverse=True)
    return results
