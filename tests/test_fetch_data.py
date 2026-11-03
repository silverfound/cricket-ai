import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import LabelEncoder

import fetch_data


def test_womens_team_encoding_never_falls_back_to_mens_class():
    encoder = LabelEncoder().fit(["Delhi Capitals", "Delhi Capitals Women"])
    male_id = fetch_data.safe_encode_team("Delhi Capitals", encoder, "male")
    female_id = fetch_data.safe_encode_team("Delhi Capitals", encoder, "female")
    assert male_id != -1
    assert female_id != -1
    assert male_id != female_id
    assert encoder.inverse_transform([male_id])[0] == "Delhi Capitals"
    assert encoder.inverse_transform([female_id])[0] == "Delhi Capitals Women"


def test_unknown_womens_team_does_not_encode_as_mens_team():
    encoder = LabelEncoder().fit(["Gujarat Giants"])
    assert fetch_data.safe_encode_team("Gujarat Giants", encoder, "female") == -1


def test_hundred_progress_is_reported_in_balls():
    assert fetch_data.extract_overs("82 balls") == "82 balls"


def test_deduplication_uses_source_id_and_preserves_distinct_double_headers():
    matches = [
        {"source_match_id": "abc", "home": "A", "away": "B"},
        {"source_match_id": "abc", "home": "A", "away": "B"},
        {"home": "A", "away": "B", "league": "League", "format": "T20",
         "start_time_iso": "2026-01-01T10:00:00Z"},
        {"home": "A", "away": "B", "league": "League", "format": "T20",
         "start_time_iso": "2026-01-01T14:00:00Z"},
        {"home": "Unknown", "away": "TBD"},
    ]
    result = fetch_data.deduplicate_matches(matches)
    assert len(result) == 4


def test_womens_competition_detection():
    assert fetch_data.is_womens_competition("Women's Premier League")
    assert fetch_data.is_womens_competition("WBBL")
    assert not fetch_data.is_womens_competition("Indian Premier League")

def test_unknown_odi_tour_name_uses_only_an_existing_team_odi_series_category():
    league_encoder = LabelEncoder().fit(["New Zealand ODI Series", "India ODI Series"])
    encoded = fetch_data.safe_encode_league(
        "India in New Zealand", league_encoder,
        ["New Zealand ODI Series", "India ODI Series"]
    )
    assert encoded == int(league_encoder.transform(["New Zealand ODI Series"])[0])


def test_unknown_league_without_known_alias_stays_unavailable():
    league_encoder = LabelEncoder().fit(["New Zealand ODI Series"])
    assert fetch_data.safe_encode_league("Unseen Competition", league_encoder, []) == -1


def test_odi_tour_prediction_can_use_trained_neutral_league_bucket():
    league_encoder = LabelEncoder().fit(["Unknown League", "New Zealand ODI Series"])
    encoded = fetch_data.safe_encode_league(
        "India in New Zealand", league_encoder,
        ["New Zealand ODI Series", "India ODI Series", "Unknown League"]
    )
    assert encoded == int(league_encoder.transform(["New Zealand ODI Series"])[0])


def test_neutral_league_bucket_is_available_when_series_alias_is_not():
    league_encoder = LabelEncoder().fit(["Unknown League", "Other ODI Series"])
    encoded = fetch_data.safe_encode_league(
        "India in New Zealand", league_encoder,
        ["New Zealand ODI Series", "India ODI Series", "Unknown League"]
    )
    assert encoded == int(league_encoder.transform(["Unknown League"])[0])
def test_international_odi_tour_is_supported_independent_of_series_name_and_team_order():
    assert fetch_data.is_ai_supported({
        "format": "ODI", "league": "New Zealand tour of India",
        "home": "India", "away": "New Zealand"
    })
    assert fetch_data.is_ai_supported({
        "format": "ODI", "league": "India in New Zealand",
        "home": "New Zealand", "away": "India"
    })


def test_international_t20_tour_is_supported_without_icc_tournament_name():
    assert fetch_data.is_ai_supported({
        "format": "T20I", "league": "India tour of New Zealand",
        "home": "New Zealand", "away": "India"
    })
    assert fetch_data.is_ai_supported({
        "format": "T20", "league": "Bilateral Series",
        "home": "India", "away": "New Zealand"
    })


def test_unknown_domestic_t20_competition_remains_filtered():
    assert not fetch_data.is_ai_supported({
        "format": "T20", "league": "Unlisted Local League",
        "home": "City XI", "away": "County XI"
    })


def test_tour_host_detection_does_not_depend_on_home_away_order():
    assert fetch_data.extract_tour_host("India in New Zealand", "New Zealand", "India") == "New Zealand"
    assert fetch_data.extract_tour_host("India tour of New Zealand", "India", "New Zealand") == "New Zealand"


def test_t20_international_prediction_can_use_neutral_league_bucket():
    league_encoder = LabelEncoder().fit(["Unknown League", "New Zealand ODI Series"])
    encoded = fetch_data.safe_encode_league(
        "India tour of New Zealand", league_encoder,
        ["New Zealand T20 Series", "Unknown League"]
    )
    assert encoded == int(league_encoder.transform(["Unknown League"])[0])



def test_post_toss_logistic_candidate_returns_probability_in_range():
    features = [
        0.60, 0.40, 0.55, 0.50, 0.65, 0.45, 0.0, 0.0, 1.0,
    ]
    probability = fetch_data.post_toss_model_probability(features)
    assert probability is not None
    assert 0.0 < probability < 1.0


def test_post_toss_prediction_builds_batting_first_and_chase_percentages():
    prediction = fetch_data.get_post_toss_prediction(
        "Team Alpha", "Team Beta", "T20", "Example League",
        "male", "Unknown Ground", "bat",
    )
    assert prediction is not None
    assert prediction.startswith("AI Pick: ")
    assert "Team Alpha:" in prediction
    assert "Team Beta:" in prediction
    percentages = [
        float(part.rsplit(":", 1)[1].strip().rstrip("%"))
        for part in prediction.split("|")[1:]
    ]
    assert len(percentages) == 2
    assert abs(sum(percentages) - 100.0) <= 0.2


def test_cricketdata_catalog_matches_fixture_by_team_pair_and_start_time():
    fixture = {
        "home": "India", "away": "New Zealand",
        "start_time_iso": "2026-11-04T02:00:00Z",
    }
    catalog = [{
        "id": "cricketdata-123",
        "teams": ["New Zealand", "India"],
        "dateTimeGMT": "2026-11-04T02:00:00",
    }]
    found = fetch_data._catalog_match_for_fixture(fixture, catalog)
    assert found is not None
    assert found["id"] == "cricketdata-123"


