# orbital-sentinel

An open-source space domain awareness and response planning platform — a
Solstice OS analogue. It ingests live orbital data, screens for conjunctions,
classifies collision risk by probability of collision (Pc), and generates
collision-avoidance manoeuvre (CAM) recommendations with delta-v cost,
presented on a self-updating operator dashboard.

Built as a portfolio project extending [conjunction-screener](https://github.com/flaxnaz/conjunction-screener),
directly demonstrating the astrodynamics-to-C2 pipeline: scenario
identification → solution design → manoeuvre optimisation → V&V → systems
integration.

## Pipeline

```
ingest  → orbital_sentinel/ingestor.py    Space-Track TLE + CDM pull
propagate → orbital_sentinel/propagator.py  SGP4 (screening) + DOP853 (precision)
screen  → orbital_sentinel/screener.py    conjunction detection, TCA, miss distance
classify → orbital_sentinel/classifier.py  Pc, risk tier (Green/Yellow/Red), object type
respond → orbital_sentinel/cam_planner.py CW-based avoidance manoeuvre + delta-v
present → orbital_sentinel/reporter.py    operator dashboard (HTML) + CSV export
```

### Risk tiers

| Tier | Probability of collision (Pc) | Action |
|------|-------------------------------|--------|
| Green | < 1e-5 | none |
| Yellow | 1e-5 – 1e-4 | monitor |
| Red | ≥ 1e-4 | manoeuvre recommended |

Pc is computed from CDM covariance data where available (circular
conjunction-plane approximation), falling back to a conservative hard-body
radius model when no CDM exists for the pair.

### CAM planning

For Red-tier conjunctions, `cam_planner.py` applies the Clohessy-Wiltshire
(CW) equations for short-timescale relative motion to size a single
along-track impulsive burn that grows the miss distance to a safe threshold
(5 km default) by TCA, then converts that delta-v to a propellant mass via
the rocket equation.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline (fixture catalog, no credentials needed):
python main.py --offline

# Live (requires Space-Track credentials, see .env.example):
cp .env.example .env  # fill in SPACETRACK_USER / SPACETRACK_PASS
python main.py
```

Outputs land in `dashboard/index.html` (operator dashboard) and
`dashboard/report.csv` (flat export).

## Testing

```bash
pytest -v
mypy orbital_sentinel
```

30+ tests across propagator, screener, classifier, CAM planner, ingestor,
reporter, and database modules; mypy strict mode clean.

## History database

Every run appends to a SQLite database at `data/orbital_sentinel.db`
(override with `--db-path`): the run itself, every classified conjunction,
and any CAM recommendations. This is what makes Pc-over-time trend
tracking possible as TCA approaches, and gives the dashboard a queryable
history rather than only ever showing the latest snapshot.

```python
from orbital_sentinel.database import get_connection, get_pc_history

conn = get_connection("data/orbital_sentinel.db")
history = get_pc_history(conn, secondary_norad_id=12345)
for row in history:
    print(row["run_timestamp"], row["probability_of_collision"])
```

SQLite (rather than a hosted server) is deliberate: this pipeline runs on
ephemeral GitHub Actions runners, so the `.db` file is committed to the
repo alongside the dashboard on each scheduled refresh — no external
database service or extra credentials needed.

## Dashboard

The dashboard is rebuilt automatically every 6 hours via GitHub Actions
(`.github/workflows/dashboard.yml`) and republished to GitHub Pages, matching
the refresh cadence used in `conjunction-screener`. Target: an operator can
read the situation and decide within 60 seconds.

## Background

This project reuses and extends engineering from earlier portfolio work:

- **conjunction-screener** — live Space-Track TLE ingestion, SGP4 screening, CDM pull
- **nrho-visibility** — DOP853 numerical propagation at rtol=1e-9
- **CW equations** — derived and validated against a mock rendezvous scenario
  (n = 0.00111 rad/s, drift rate = 6n·x0, delta-v per burn = n·x0/2)
