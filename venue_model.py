"""Optional pre-toss venue-aware candidate inference.

The candidate is deliberately opt-in: the legacy production model remains the
fallback until the candidate artifacts are installed and validated.
"""
import json
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "venue_aware_candidate.joblib")
STATE_PATH = os.path.join("data", "venue_model_state.json")


def load_venue_candidate(model_path=MODEL_PATH, state_path=STATE_PATH):
    if not (os.path.isfile(model_path) and os.path.isfile(state_path)):
        return None, None
    try:
        artifact = joblib.load(model_path)
        with open(state_path, "r", encoding="utf-8") as stream:
            state = json.load(stream)
        required = {"model", "preprocessor", "feature_columns", "threshold"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            return None, None
        if not all(k in state for k in ("team_form10", "h2h", "venue_format", "venue_team")):
            return None, None
        return artifact, state
    except Exception as exc:
        print(f"Venue candidate unavailable: {exc}")
        return None, None


def _key(*parts):
    return "||".join(str(part or "").strip() for part in parts)


def build_candidate_row(team1, team2, match_format, gender, league, venue, state):
    team1, team2 = str(team1 or "").strip(), str(team2 or "").strip()
    fmt, gender = str(match_format or "").strip().upper(), str(gender or "male").lower()
    venue, league = str(venue or "").strip(), str(league or "").strip()
    form = state["team_form10"]
    h2h = state["h2h"]
    vf = state["venue_format"].get(_key(venue, fmt), {})
    t1v = state["venue_team"].get(_key(venue, fmt, team1), {})
    t2v = state["venue_team"].get(_key(venue, fmt, team2), {})
    h = h2h.get(_key(team1, team2))
    reversed_h2h = False
    if h is None:
        h = h2h.get(_key(team2, team1), {})
        reversed_h2h = True
    h2h_matches = int(h.get("matches", 0))
    h2h_wins = int(h.get("team1_wins", 0))
    if reversed_h2h:
        h2h_wins = h2h_matches - h2h_wins
    h2h_rate = (h2h_wins + 0.5) / (h2h_matches + 1)
    t1_form = form.get(team1, [])
    t2_form = form.get(team2, [])
    avg = float(vf.get("first_innings_runs_sum", 0)) / int(vf.get("matches", 1)) if vf.get("matches") else 150.0
    def rate(item):
        n = int(item.get("matches", 0))
        return (float(item.get("wins", 0)) + 2.5) / (n + 5.0)
    return {
        "team1": team1, "team2": team2, "venue": venue or "Unknown",
        "format": fmt, "gender": gender, "event": league or "Unknown",
        "team1_form10": sum(t1_form) / len(t1_form) if t1_form else 0.5,
        "team2_form10": sum(t2_form) / len(t2_form) if t2_form else 0.5,
        "team1_prior_matches": len(t1_form), "team2_prior_matches": len(t2_form),
        "h2h_team1_win_rate": h2h_rate, "h2h_prior_matches": h2h_matches,
        "venue_format_prior_matches": int(vf.get("matches", 0)),
        "venue_prior_avg_first_innings_runs": avg,
        "team1_venue_win_rate": rate(t1v), "team2_venue_win_rate": rate(t2v),
        "team1_venue_matches": int(t1v.get("matches", 0)),
        "team2_venue_matches": int(t2v.get("matches", 0)),
    }


def predict_venue_candidate(artifact, state, team1, team2, match_format, gender, league, venue):
    if not artifact or not state or not venue:
        return None
    try:
        row = build_candidate_row(team1, team2, match_format, gender, league, venue, state)
        frame = pd.DataFrame([row], columns=artifact["feature_columns"])
        transformed = artifact["preprocessor"].transform(frame)
        probability = artifact["model"].predict_proba(transformed)[0]
        classes = list(artifact["model"].classes_)
        if 1 not in classes:
            return None
        p1 = float(probability[classes.index(1)])
        p2 = 1.0 - p1
        winner = team1 if p1 >= p2 else team2
        return (
            f"AI Pick: {winner} | "
            f"{team1}: {round(p1 * 100, 1)}% | "
            f"{team2}: {round(p2 * 100, 1)}%"
        )
    except Exception as exc:
        print(f"Venue candidate prediction error: {exc}")
        return None
