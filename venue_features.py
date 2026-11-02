"""Leakage-safe venue feature engineering for pre-toss match models.

Input rows need match_date, venue, format, team1, team2, winner,
first_innings_runs. Each row uses only earlier completed matches.
"""
from collections import defaultdict
import math
import pandas as pd

REQUIRED_COLUMNS = {"match_date", "venue", "format", "team1", "team2", "winner", "first_innings_runs"}

def _normal(value):
    return " ".join(str(value or "").strip().casefold().split())

def build_venue_features(matches, smoothing_matches=5, global_average=None):
    """Add prior-only venue stats; format-specific averages; smoothed team rates."""
    missing = REQUIRED_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError("Missing required historical columns: " + ", ".join(sorted(missing)))
    if smoothing_matches < 0:
        raise ValueError("smoothing_matches must be non-negative")
    frame = matches.copy().reset_index(drop=True)
    frame["_parsed_date"] = pd.to_datetime(frame["match_date"], utc=True, errors="coerce")
    if frame["_parsed_date"].isna().any():
        raise ValueError("match_date contains invalid or missing dates")
    frame["_original_order"] = range(len(frame))
    frame = frame.sort_values(["_parsed_date", "_original_order"], kind="stable")

    venue_stats = defaultdict(lambda: {"count": 0, "runs_sum": 0.0})
    team_stats = defaultdict(lambda: {"wins": 0, "matches": 0})
    runs_all = pd.to_numeric(frame["first_innings_runs"], errors="coerce").dropna()
    prior_global = float(global_average) if global_average is not None else (
        float(runs_all.mean()) if len(runs_all) else 150.0
    )
    if not math.isfinite(prior_global):
        prior_global = 150.0

    output = []
    for _, row in frame.iterrows():
        venue, fmt = _normal(row["venue"]), _normal(row["format"])
        team1, team2 = _normal(row["team1"]), _normal(row["team2"])
        key = (venue, fmt)
        v = venue_stats[key]
        t1 = team_stats[(venue, fmt, team1)] if venue and team1 else {"wins": 0, "matches": 0}
        t2 = team_stats[(venue, fmt, team2)] if venue and team2 else {"wins": 0, "matches": 0}
        d1, d2 = t1["matches"] + smoothing_matches, t2["matches"] + smoothing_matches
        output.append({
            "_original_order": int(row["_original_order"]),
            "venue_id": venue or None,
            "venue_format_prior_matches": int(v["count"]) if venue else 0,
            "venue_format_prior_avg_first_innings_runs": (
                v["runs_sum"] / v["count"] if venue and v["count"] else prior_global
            ),
            "team1_venue_prior_win_rate": (
                (t1["wins"] + 0.5 * smoothing_matches) / d1 if venue and d1 else 0.5
            ),
            "team2_venue_prior_win_rate": (
                (t2["wins"] + 0.5 * smoothing_matches) / d2 if venue and d2 else 0.5
            ),
        })
        winner = _normal(row["winner"])
        runs = pd.to_numeric(pd.Series([row["first_innings_runs"]]), errors="coerce").iloc[0]
        if venue:
            if pd.notna(runs):
                v["runs_sum"] += float(runs)
                v["count"] += 1
            for team in (team1, team2):
                if team:
                    team_stats[(venue, fmt, team)]["matches"] += 1
                    if winner == team:
                        team_stats[(venue, fmt, team)]["wins"] += 1

    features = pd.DataFrame(output).set_index("_original_order").sort_index()
    return pd.concat([matches.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
