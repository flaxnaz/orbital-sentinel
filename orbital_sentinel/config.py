"""Configuration: thresholds, constants, and YAML-backed settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --- Physical / astrodynamics constants ---
MU_EARTH_KM3_S2: float = 398600.4418  # Earth gravitational parameter, km^3/s^2
EARTH_RADIUS_KM: float = 6378.137

# --- Risk classification thresholds (probability of collision) ---
PC_GREEN_MAX: float = 1e-5   # Pc < 1e-5  -> Green, no action
PC_YELLOW_MAX: float = 1e-4  # 1e-5 <= Pc < 1e-4 -> Yellow, monitor
# Pc >= 1e-4 -> Red, manoeuvre recommended

# --- Screening defaults ---
DEFAULT_SCREENING_WINDOW_HOURS: float = 24.0
DEFAULT_MISS_DISTANCE_THRESHOLD_KM: float = 5.0
DEFAULT_HARD_BODY_RADIUS_KM: float = 0.02  # combined hard-body radius fallback

# --- CAM planner defaults ---
DEFAULT_SAFE_MISS_DISTANCE_KM: float = 5.0
DEFAULT_ISP_S: float = 220.0  # propellant Isp, seconds (small-sat cold-gas/mono-prop)
DEFAULT_G0_M_S2: float = 9.80665

RiskTier = str  # "GREEN" | "YELLOW" | "RED"


@dataclass
class ScreeningConfig:
    window_hours: float = DEFAULT_SCREENING_WINDOW_HOURS
    miss_distance_threshold_km: float = DEFAULT_MISS_DISTANCE_THRESHOLD_KM
    hard_body_radius_km: float = DEFAULT_HARD_BODY_RADIUS_KM
    primary_norad_id: int = 25544  # ISS default
    colocation_std_threshold_km: float = 0.1  # flag near-constant-range pairs as catalog duplicates


@dataclass
class ClassifierConfig:
    pc_green_max: float = PC_GREEN_MAX
    pc_yellow_max: float = PC_YELLOW_MAX


@dataclass
class CamConfig:
    safe_miss_distance_km: float = DEFAULT_SAFE_MISS_DISTANCE_KM
    isp_s: float = DEFAULT_ISP_S
    g0_m_s2: float = DEFAULT_G0_M_S2
    spacecraft_mass_kg: float = 150.0


@dataclass
class DashboardConfig:
    refresh_hours: float = 6.0
    output_path: str = "dashboard/index.html"
    default_alert_km: float = 5.0  # slider's initial position — the "real" operational threshold

@dataclass
class Config:
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    cam: CamConfig = field(default_factory=CamConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        return cls(
            screening=ScreeningConfig(**raw.get("screening", {})),
            classifier=ClassifierConfig(**raw.get("classifier", {})),
            cam=CamConfig(**raw.get("cam", {})),
            dashboard=DashboardConfig(**raw.get("dashboard", {})),
        )


def classify_tier(pc: float, cfg: ClassifierConfig = ClassifierConfig()) -> RiskTier:
    """Map a probability-of-collision value to a risk tier."""
    if pc < cfg.pc_green_max:
        return "GREEN"
    if pc < cfg.pc_yellow_max:
        return "YELLOW"
    return "RED"
