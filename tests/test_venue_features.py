import pandas as pd
import pytest
from venue_features import build_venue_features

def sample_matches():
    return pd.DataFrame([
        {"match_date":"2024-01-01","venue":"Ground A","format":"T20","team1":"Alpha","team2":"Beta","winner":"Alpha","first_innings_runs":180},
        {"match_date":"2024-01-02","venue":"Ground A","format":"T20","team1":"Beta","team2":"Alpha","winner":"Beta","first_innings_runs":160},
        {"match_date":"2024-01-03","venue":"Ground A","format":"ODI","team1":"Alpha","team2":"Beta","winner":"Beta","first_innings_runs":250},
        {"match_date":"2024-01-04","venue":"Ground B","format":"T20","team1":"Alpha","team2":"Beta","winner":"Beta","first_innings_runs":140},
    ])

def test_venue_features_use_only_prior_matches_and_format_partition():
    result = build_venue_features(sample_matches())
    assert result.loc[0,"venue_format_prior_matches"] == 0
    assert result.loc[1,"venue_format_prior_matches"] == 1
    assert result.loc[1,"venue_format_prior_avg_first_innings_runs"] == 180
    assert result.loc[2,"venue_format_prior_matches"] == 0
    assert result.loc[3,"venue_format_prior_matches"] == 0

def test_team_venue_rates_are_smoothed_and_prior_only():
    result = build_venue_features(sample_matches(), smoothing_matches=2)
    assert result.loc[0,"team1_venue_prior_win_rate"] == 0.5
    assert result.loc[1,"team2_venue_prior_win_rate"] == pytest.approx(2/3)

def test_invalid_schema_and_date_fail_closed():
    with pytest.raises(ValueError, match="Missing required"):
        build_venue_features(pd.DataFrame([{"venue":"Ground"}]))
    frame = sample_matches()
    frame.loc[0,"match_date"] = "not-a-date"
    with pytest.raises(ValueError, match="invalid"):
        build_venue_features(frame)

def test_shuffling_rows_does_not_change_chronological_features():
    frame = sample_matches()
    first = build_venue_features(frame).sort_values("match_date").reset_index(drop=True)
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    second = build_venue_features(shuffled).sort_values("match_date").reset_index(drop=True)
    cols = ["venue_format_prior_matches","venue_format_prior_avg_first_innings_runs","team1_venue_prior_win_rate","team2_venue_prior_win_rate"]
    pd.testing.assert_frame_equal(first[cols], second[cols])

def test_same_day_matches_do_not_leak_into_each_other():
    frame = pd.DataFrame([
        {"match_date":"2024-02-01","venue":"Ground C","format":"T20","team1":"Alpha","team2":"Beta","winner":"Alpha","first_innings_runs":190},
        {"match_date":"2024-02-01","venue":"Ground C","format":"T20","team1":"Gamma","team2":"Delta","winner":"Gamma","first_innings_runs":170},
        {"match_date":"2024-02-02","venue":"Ground C","format":"T20","team1":"Alpha","team2":"Gamma","winner":"Alpha","first_innings_runs":180},
    ])
    result = build_venue_features(frame)
    assert result.loc[0,"venue_format_prior_matches"] == 0
    assert result.loc[1,"venue_format_prior_matches"] == 0
    assert result.loc[2,"venue_format_prior_matches"] == 2
    assert result.loc[2,"venue_format_prior_avg_first_innings_runs"] == 180
