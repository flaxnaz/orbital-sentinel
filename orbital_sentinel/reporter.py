"""Report generation: operator dashboard (self-contained HTML) and CSV
exports of classified conjunctions and CAM recommendations.
"""

from __future__ import annotations

import csv
import html
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from orbital_sentinel.cam_planner import CamRecommendation
from orbital_sentinel.classifier import ClassifiedConjunction

TIER_COLOURS = {
    "GREEN": "#2e7d32",
    "YELLOW": "#f9a825",
    "RED": "#c62828",
}

_TREND_PC_FLOOR = 1e-30  # avoids log10(0) for a Pc that underflows to exactly zero


def write_csv(
    classified: list[ClassifiedConjunction],
    cams: dict[int, CamRecommendation],
    path: str | Path,
) -> None:
    """Write classified conjunctions (with any matching CAM recommendation)
    to a flat CSV file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "secondary_norad_id",
        "secondary_name",
        "tca",
        "miss_distance_km",
        "relative_velocity_km_s",
        "probability_of_collision",
        "pc_method",
        "risk_tier",
        "object_type",
        "cam_delta_v_m_s",
        "cam_predicted_miss_km",
        "cam_propellant_kg",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in classified:
            cam = cams.get(c.event.secondary_norad_id)
            writer.writerow(
                {
                    "secondary_norad_id": c.event.secondary_norad_id,
                    "secondary_name": c.event.secondary_name,
                    "tca": c.event.tca.isoformat(),
                    "miss_distance_km": round(c.event.miss_distance_km, 4),
                    "relative_velocity_km_s": round(c.event.relative_velocity_km_s, 4),
                    "probability_of_collision": f"{c.probability_of_collision:.3e}",
                    "pc_method": c.pc_method,
                    "risk_tier": c.risk_tier,
                    "object_type": c.object_type,
                    "cam_delta_v_m_s": round(cam.delta_v_m_s, 4) if cam else "",
                    "cam_predicted_miss_km": (
                        round(cam.predicted_new_miss_distance_km, 4) if cam else ""
                    ),
                    "cam_propellant_kg": round(cam.propellant_cost_kg, 5) if cam else "",
                }
            )


def render_pc_trend_svg(pc_history: list[float], width: int = 130, height: int = 30) -> str:
    """Render a small inline SVG sparkline of Pc across past runs, log10
    scaled since Pc spans many orders of magnitude. Fewer than 2 points
    isn't a trend yet, so callers should show a "new" label instead —
    this function returns that placeholder itself for convenience.
    """
    if len(pc_history) < 2:
        return '<span class="trend-new">new</span>'

    logs = [math.log10(max(v, _TREND_PC_FLOOR)) for v in pc_history]
    lo, hi = min(logs), max(logs)
    if hi == lo:
        hi = lo + 1.0  # flat series — render as a level line rather than divide by zero

    n = len(logs)
    pad = 3.0
    points: list[tuple[float, float]] = []
    for i, lv in enumerate(logs):
        x = pad + (width - 2 * pad) * i / (n - 1)
        y = height - pad - (height - 2 * pad) * (lv - lo) / (hi - lo)
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    last_x, last_y = points[-1]
    rising = logs[-1] > logs[0]
    stroke = "#c62828" if rising else "#58a6ff"

    return (
        f'<svg class="trend" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Pc trend">'
        f'<polyline points="{polyline}" fill="none" stroke="{stroke}" stroke-width="1.5"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" fill="{stroke}"/>'
        f"</svg>"
    )


def _row_html(
    c: ClassifiedConjunction,
    cam: CamRecommendation | None,
    pc_history: list[float] | None = None,
) -> str:
    colour = TIER_COLOURS.get(c.risk_tier, "#666")
    name = html.escape(c.event.secondary_name)
    cam_cell = (
        f"{cam.delta_v_m_s:.3f} m/s ({cam.burn_direction}), "
        f"new miss {cam.predicted_new_miss_distance_km:.2f} km, "
        f"{cam.propellant_cost_kg:.4f} kg prop"
        if cam
        else "&mdash;"
    )
    trend_cell = render_pc_trend_svg(pc_history) if pc_history else '<span class="trend-new">new</span>'
    return f"""
    <tr data-miss="{c.event.miss_distance_km:.4f}" data-tier="{c.risk_tier}" data-cam="{1 if cam else 0}">
      <td>{c.event.secondary_norad_id}</td>
      <td>{name}</td>
      <td>{c.event.tca.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
      <td>{c.event.miss_distance_km:.3f}</td>
      <td>{c.probability_of_collision:.2e}</td>
      <td>{trend_cell}</td>
      <td><span class="tier" style="background:{colour}">{c.risk_tier}</span></td>
      <td>{c.object_type}</td>
      <td>{cam_cell}</td>
    </tr>"""


def render_dashboard(
    classified: list[ClassifiedConjunction],
    cams: list[CamRecommendation],
    generated_at: datetime | None = None,
    pc_histories: dict[int, list[float]] | None = None,
    slider_min_km: float = 5.0,
    slider_max_km: float = 500.0,
    slider_default_km: float | None = None,
) -> str:
    """Render a self-contained operator dashboard HTML page.

    pc_histories maps secondary_norad_id -> chronological list of past
    Pc values (oldest first, current run's value included as the last
    entry) — the raw material for the per-row trend sparkline. Objects
    missing from the map, or with fewer than 2 entries, show a "new"
    label instead of a chart, since a single point isn't a trend.

    The dashboard includes a live miss-distance slider (slider_min_km to
    slider_max_km) that filters the already-rendered rows client-side —
    no re-screening or server round-trip, since every row already carries
    its own miss distance as a data attribute. This only reveals rows
    that were actually screened in: if the pipeline itself ran with a
    tight miss_distance_threshold_km (e.g. 5 km), sliding up to 500 km
    won't show anything beyond what was found at screening time. Run the
    pipeline with a wider screening.miss_distance_threshold_km if you
    want the slider's full range to have real data to reveal.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    cam_by_id = {c.secondary_norad_id: c for c in cams}
    pc_histories = pc_histories or {}
    slider_default_km = slider_default_km if slider_default_km is not None else slider_max_km

    counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for c in classified:
        counts[c.risk_tier] = counts.get(c.risk_tier, 0) + 1

    rows = "\n".join(
        _row_html(
            c,
            cam_by_id.get(c.event.secondary_norad_id),
            pc_histories.get(c.event.secondary_norad_id),
        )
        for c in classified
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>orbital-sentinel — operator dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:2rem; }}
  h1 {{ font-weight:600; }}
  .meta {{ color:#8b949e; margin-bottom:1.5rem; }}
  .summary {{ display:flex; gap:1rem; margin-bottom:1.5rem; }}
  .summary div {{ padding:0.75rem 1.25rem; border-radius:8px; background:#161b22; }}
  .filter-bar {{ display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem; padding:0.9rem 1.25rem; background:#161b22; border-radius:8px; }}
  .filter-bar label {{ white-space:nowrap; font-size:0.9rem; color:#c9d1d9; }}
  .filter-bar input[type="range"] {{ flex:1; accent-color:#58a6ff; }}
  .filter-count {{ color:#8b949e; font-size:0.85rem; white-space:nowrap; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:0.6rem 0.9rem; text-align:left; border-bottom:1px solid #21262d; font-size:0.9rem; }}
  th {{ background:#21262d; text-transform:uppercase; font-size:0.75rem; letter-spacing:0.05em; }}
  .tier {{ padding:0.15rem 0.6rem; border-radius:12px; color:white; font-weight:600; font-size:0.75rem; }}
  .trend-new {{ color:#8b949e; font-size:0.8rem; font-style:italic; }}
  svg.trend {{ vertical-align:middle; }}
  .no-rows {{ padding:1.5rem; text-align:center; color:#8b949e; font-style:italic; }}
</style>
</head>
<body>
  <h1>orbital-sentinel &mdash; operator dashboard</h1>
  <div class="meta">Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} &middot; refreshes every 6 hours</div>
  <div class="summary">
    <div>GREEN: {counts.get('GREEN', 0)}</div>
    <div>YELLOW: {counts.get('YELLOW', 0)}</div>
    <div>RED: {counts.get('RED', 0)}</div>
    <div>CAMs recommended: {len(cams)}</div>
  </div>
  <div class="filter-bar">
    <label for="missSlider">Miss distance &le; <span id="missValue">{slider_default_km:.0f}</span> km</label>
    <input type="range" id="missSlider" min="{slider_min_km:.0f}" max="{slider_max_km:.0f}" step="1" value="{slider_default_km:.0f}">
    <span class="filter-count" id="filterCount"></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>NORAD ID</th><th>Object</th><th>TCA</th><th>Miss (km)</th>
        <th>Pc</th><th>Pc trend</th><th>Tier</th><th>Type</th><th>CAM recommendation</th>
      </tr>
    </thead>
    <tbody id="conjunctionRows">
      {rows}
    </tbody>
  </table>
  <div class="no-rows" id="noRowsMsg" style="display:none;">No conjunctions within the selected miss-distance range.</div>
<script>
(function() {{
  var slider = document.getElementById('missSlider');
  var valueLabel = document.getElementById('missValue');
  var countLabel = document.getElementById('filterCount');
  var noRowsMsg = document.getElementById('noRowsMsg');
  var rows = Array.prototype.slice.call(document.querySelectorAll('#conjunctionRows tr'));

  function update() {{
    var threshold = parseFloat(slider.value);
    valueLabel.textContent = threshold.toFixed(0);
    var visible = {{GREEN: 0, YELLOW: 0, RED: 0}};
    var camsVisible = 0;
    var anyVisible = false;
    rows.forEach(function(row) {{
      var miss = parseFloat(row.getAttribute('data-miss'));
      var show = miss <= threshold;
      row.style.display = show ? '' : 'none';
      if (show) {{
        anyVisible = true;
        var tier = row.getAttribute('data-tier');
        if (visible[tier] !== undefined) visible[tier]++;
        if (row.getAttribute('data-cam') === '1') camsVisible++;
      }}
    }});
    countLabel.textContent = 'Showing ' + visible.GREEN + ' green, ' + visible.YELLOW +
      ' yellow, ' + visible.RED + ' red \\u00b7 ' + camsVisible + ' CAM(s)';
    noRowsMsg.style.display = (anyVisible || rows.length === 0) ? 'none' : 'block';
  }}

  slider.addEventListener('input', update);
  update();
}})();
</script>
</body>
</html>"""


def write_dashboard(
    classified: list[ClassifiedConjunction],
    cams: list[CamRecommendation],
    path: str | Path,
    generated_at: datetime | None = None,
    pc_histories: dict[int, list[float]] | None = None,
    slider_min_km: float = 5.0,
    slider_max_km: float = 500.0,
    slider_default_km: float | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        render_dashboard(
            classified,
            cams,
            generated_at,
            pc_histories,
            slider_min_km,
            slider_max_km,
            slider_default_km,
        ),
        encoding="utf-8",
    )