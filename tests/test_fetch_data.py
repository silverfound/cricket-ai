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
