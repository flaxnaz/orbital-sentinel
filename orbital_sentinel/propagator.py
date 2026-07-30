"""Propagation: SGP4 for catalog-level screening, DOP853 numerical
integration for high-fidelity propagation of confirmed conjunctions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sgp4.api import Satrec, jday

from orbital_sentinel.config import MU_EARTH_KM3_S2


@dataclass
class StateVector:
    """Position (km) and velocity (km/s) in a common inertial frame, at a UTC epoch."""

    epoch: datetime
    r_km: NDArray[np.float64]  # shape (3,)
    v_km_s: NDArray[np.float64]  # shape (3,)


@dataclass
class TLEObject:
    norad_id: int
    name: str
    line1: str
    line2: str


class SGP4Propagator:
    """Thin wrapper around python-sgp4 for catalog-level screening."""

    def __init__(self, tle: TLEObject) -> None:
        self.tle = tle
        self._sat = Satrec.twoline2rv(tle.line1, tle.line2)

    def propagate(self, epoch: datetime) -> StateVector:
        """Propagate to a given UTC datetime. Returns TEME-frame state (km, km/s)."""
        epoch_utc = epoch.astimezone(timezone.utc)
        jd, fr = jday(
            epoch_utc.year,
            epoch_utc.month,
            epoch_utc.day,
            epoch_utc.hour,
            epoch_utc.minute,
            epoch_utc.second + epoch_utc.microsecond / 1e6,
        )
        error_code, r, v = self._sat.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(
                f"SGP4 propagation error {error_code} for NORAD {self.tle.norad_id}"
            )
        return StateVector(
            epoch=epoch_utc,
            r_km=np.array(r, dtype=np.float64),
            v_km_s=np.array(v, dtype=np.float64),
        )

    def sample_window(
        self, start: datetime, end: datetime, step_s: float = 60.0
    ) -> list[StateVector]:
        """Sample states across a time window at a fixed step, for catalog screening."""
        states: list[StateVector] = []
        t = start
        step = timedelta(seconds=step_s)
        while t <= end:
            states.append(self.propagate(t))
            t += step
        return states


def two_body_derivative(
    _t: float, y: NDArray[np.float64], mu: float = MU_EARTH_KM3_S2
) -> NDArray[np.float64]:
    """Two-body EOM: y = [rx, ry, rz, vx, vy, vz]."""
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.concatenate([v, a])


class HighFidelityPropagator:
    """DOP853 numerical propagation for precise miss-distance calculation
    of confirmed high-risk conjunctions. Two-body dynamics by default;
    same integrator/tolerance settings used in nrho-visibility.
    """

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-12) -> None:
        self.rtol = rtol
        self.atol = atol

    def propagate(
        self, state0: StateVector, target_epoch: datetime
    ) -> StateVector:
        dt = (target_epoch.astimezone(timezone.utc) - state0.epoch).total_seconds()
        y0 = np.concatenate([state0.r_km, state0.v_km_s])
        sol = solve_ivp(
            two_body_derivative,
            t_span=(0.0, dt),
            y0=y0,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"DOP853 propagation failed: {sol.message}")
        y_final = sol.y[:, -1]
        return StateVector(
            epoch=target_epoch.astimezone(timezone.utc),
            r_km=y_final[:3],
            v_km_s=y_final[3:],
        )

    def propagate_dense(
        self, state0: StateVector, duration_s: float, num_points: int = 200
    ) -> list[StateVector]:
        """Return a dense set of states over [0, duration_s] for fine-grained
        miss-distance search (used to refine time of closest approach).
        """
        y0 = np.concatenate([state0.r_km, state0.v_km_s])
        t_eval = np.linspace(0.0, duration_s, num_points)
        sol = solve_ivp(
            two_body_derivative,
            t_span=(0.0, duration_s),
            y0=y0,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            t_eval=t_eval,
        )
        if not sol.success:
            raise RuntimeError(f"DOP853 propagation failed: {sol.message}")
        out = []
        for i, t in enumerate(t_eval):
            out.append(
                StateVector(
                    epoch=state0.epoch + timedelta(seconds=float(t)),
                    r_km=sol.y[:3, i],
                    v_km_s=sol.y[3:, i],
                )
            )
        return out
