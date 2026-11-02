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


def test_duplicate_gender_helper_is_removed():
    import inspect
    assert inspect.getsource(app).count("def infer_match_gender(") == 1


def test_live_highlightly_prediction_passes_inferred_gender(monkeypatch):
    captured = {}
    def fake_prediction(home_info, away_info, match_format, league_name, home_team, away_team, gender=None):
        captured["gender"] = gender
        return 61.0, None
    monkeypatch.setattr(app, "get_ai_predictions", fake_prediction)
    payload = {"data": [{
        "id": "women-live-1",
        "state": {"description": "In Play", "teams": {
            "home": {"score": "45/2", "info": "6.0 overs"},
            "away": {"score": "-", "info": ""}}},
        "format": "T20", "league": {"name": "Domestic League"},
        "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"},
        "gender": "female"}]}
    matches = app.standardize_data(payload, "Highlightly")
    assert len(matches) == 1
    assert captured["gender"] == "female"
    assert matches[0]["win_probability"] == 61.0



def test_final_artifact_label_mapping_and_feature_schema():
    artifact = app.pre_match_artifact
    assert artifact["label_mapping"] == {1: "Team 1 wins", 0: "Team 2 wins"}
    assert list(app.pre_match_model.classes_) == [0, 1]
    expected = ["league_id", "team1_id", "team2_id", "t1_form", "t2_form", "h2h_win_rate"]
    assert artifact["features"] == expected
    assert list(app.pre_match_model.feature_names_in_) == expected



def test_completed_new_zealand_india_match_retrospective(capsys):
    """Run the production pre-match pipeline on the Nov 1, 2026 T20I fixture.

    This is a pipeline/retrospective check, not a model-accuracy gate:
    a single match may legitimately be predicted incorrectly.
    """
    import pytest

    assert app.pre_match_model is not None
    encoders = app.team_league_encoders
    assert encoders is not None
    team_encoder = encoders["team"]
    league_encoder = encoders["league"]

    def find_team(wanted):
        for cls in team_encoder.classes_:
            if app.normalize_history_team(str(cls)) == app.normalize_history_team(wanted):
                return str(cls)
        return None

    nz = find_team("New Zealand")
    india = find_team("India")
    league = next(
        (str(cls) for cls in league_encoder.classes_
         if "india in new zealand" in str(cls).lower()),
        None,
    )
    if not (nz and india and league):
        pytest.skip(
            "Retrospective fixture unsupported by saved encoders: "
            f"New Zealand={bool(nz)}, India={bool(india)}, "
            f"India in New Zealand league={bool(league)}"
        )

    # Log the exact pre-match inputs and raw class probabilities so a
    # retrospective miss can be audited without changing model behavior.
    import pandas as pd
    nz_id = app.safe_encode_team(nz, team_encoder, "male")
    india_id = app.safe_encode_team(india, team_encoder, "male")
    league_id = app.safe_encode_league(league, league_encoder)
    nz_form = app.get_historical_form(nz, "male")
    india_form = app.get_historical_form(india, "male")
    nz_h2h = app.get_historical_h2h(nz, india, "male")
    india_h2h = app.get_historical_h2h(india, nz, "male")
    assert nz_id >= 0 and india_id >= 0 and league_id >= 0
    assert abs((nz_h2h + india_h2h) - 1.0) < 1e-9, "Directional H2H rates must be complementary"
    audit_input = pd.DataFrame([{
        "league_id": league_id,
        "team1_id": nz_id,
        "team2_id": india_id,
        "t1_form": nz_form,
        "t2_form": india_form,
        "h2h_win_rate": nz_h2h,
    }], columns=app.pre_match_artifact["features"])
    raw_probabilities = app.pre_match_model.predict_proba(audit_input)[0]
    print(
        "RETROSPECTIVE_FEATURE_AUDIT "
        f"team1=New Zealand team1_id={nz_id} form={nz_form:.4f} "
        f"team2=India team2_id={india_id} form={india_form:.4f} "
        f"league_id={league_id} h2h_team1_rate={nz_h2h:.6f} "
        f"h2h_team2_rate={india_h2h:.6f} "
        f"class0_team2_probability={float(raw_probabilities[list(app.pre_match_model.classes_).index(0)]):.6f} "
        f"class1_team1_probability={float(raw_probabilities[list(app.pre_match_model.classes_).index(1)]):.6f}"
    )

    prediction = app.get_pre_match_prediction(nz, india, "T20", league, "male")
    assert prediction is not None, "Production prediction pipeline returned None"
    predicted_team = prediction.rsplit(" (", 1)[0]
    actual_winner = "New Zealand"  # Won the 5th T20I on 2026-11-01.
    print(
        "RETROSPECTIVE_MATCH_CHECK "
        f"fixture=New Zealand-vs-India-2026-11-01 "
        f"prediction={prediction} actual_winner={actual_winner} "
        f"matched={predicted_team.casefold() == actual_winner.casefold()}"
    )
