import json
import joblib
import pandas as pd
from venue_model import build_candidate_row, load_venue_candidate, predict_venue_candidate


def sample_state():
    return {
        "team_form10": {"Alpha": [1, 1, 0], "Beta": [0, 1, 0]},
        "h2h": {"Alpha||Beta": {"team1_wins": 3, "matches": 5}},
        "venue_format": {"Ground A||T20": {"matches": 4, "first_innings_runs_sum": 680}},
        "venue_team": {
            "Ground A||T20||Alpha": {"wins": 3, "matches": 4},
            "Ground A||T20||Beta": {"wins": 1, "matches": 4},
        },
    }


def test_candidate_row_uses_venue_and_prior_state():
    row = build_candidate_row("Alpha", "Beta", "T20", "male", "League", "Ground A", sample_state())
    assert row["venue_format_prior_matches"] == 4
    assert row["venue_prior_avg_first_innings_runs"] == 170
    assert row["team1_venue_matches"] == 4
    assert row["team2_venue_matches"] == 4
    assert row["team1_form10"] == 2 / 3


def test_unseen_venue_uses_neutral_prior():
    row = build_candidate_row("Alpha", "Beta", "ODI", "male", "League", "New Ground", sample_state())
    assert row["venue_format_prior_matches"] == 0
    assert row["venue_prior_avg_first_innings_runs"] == 150
    assert row["team1_venue_win_rate"] == 0.5


def test_missing_venue_candidate_artifacts_fail_closed(tmp_path):
    artifact, state = load_venue_candidate(tmp_path / "missing.joblib", tmp_path / "missing.json")
    assert artifact is None
    assert state is None


def test_prediction_fails_closed_without_venue():
    assert predict_venue_candidate({}, sample_state(), "Alpha", "Beta", "T20", "male", "League", None) is None


def test_reversed_h2h_is_inverted_for_team1():
    state = sample_state()
    row = build_candidate_row("Beta", "Alpha", "T20", "male", "League", "Ground A", state)
    assert row["h2h_team1_win_rate"] == (2 + 0.5) / (5 + 1)
