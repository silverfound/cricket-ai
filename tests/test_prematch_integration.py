import json
from pathlib import Path

import fetch_data as app


def test_history_file_has_both_gender_partitions():
    path = Path("data/team_form_h2h_history.json")
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["valid_matches"] > 0
    assert payload["teams"]["male"]
    assert payload["teams"]["female"]
    assert payload["h2h"]["male"]
    assert payload["h2h"]["female"]


def test_history_lookup_returns_valid_values_for_known_teams():
    assert 0.0 <= app.get_historical_form("India", "male") <= 1.0
    assert 0.0 <= app.get_historical_form("Australia", "female") <= 1.0
    assert 0.0 <= app.get_historical_h2h("England", "India", "male") <= 1.0


def test_gender_inference_handles_explicit_and_label_metadata():
    assert app.infer_match_gender({"gender": "female"}, "League", "A", "B") == "female"
    assert app.infer_match_gender({}, "Women's Premier League", "A", "B") == "female"
    assert app.infer_match_gender({"gender": "male"}, "WPL", "A", "B") == "male"


def test_final_artifact_and_encoders_load():
    assert app.pre_match_model is not None, "Pre-match model failed to load"
    assert app.team_league_encoders is not None, "Team/league encoders failed to load"
    assert hasattr(app.pre_match_model, "predict_proba")
    assert 0.0 <= app.pre_match_threshold <= 1.0


def test_prediction_runs_with_known_encoder_classes():
    assert app.pre_match_model is not None
    team_encoder = app.team_league_encoders["team"]
    league_encoder = app.team_league_encoders["league"]
    teams = list(team_encoder.classes_)
    leagues = list(league_encoder.classes_)
    assert len(teams) >= 2 and len(leagues) >= 1
    # Search for an encoder-supported pair with history; the first pair is
    # sufficient to exercise feature construction and estimator inference.
    prediction = app.get_pre_match_prediction(
        str(teams[0]), str(teams[1]), "T20", str(leagues[0]), "male"
    )
    assert prediction is not None, "Prediction returned None for encoder-known inputs"
    assert "%" in prediction
