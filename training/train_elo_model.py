#!/usr/bin/env python3
"""Leakage-aware chronological LightGBM training from matches_with_elo_features.csv.

Run:
  pip install lightgbm scikit-learn pandas numpy joblib
  python train_elo_model.py --input matches_with_elo_features.csv --output-dir artifacts

This script expects one row per completed match and prior-only rolling/Elo features.
The final chronological test set is never used for early stopping or tuning.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

CATEGORICAL = ["format", "gender", "team1", "team2", "venue", "competition"]
FEATURES = [
    "format", "gender", "team1", "team2", "venue", "competition",
    "team1_strength_prior20", "team2_strength_prior20",
    "team1_form_prior10", "team2_form_prior10",
    "h2h_team1_win_prior10", "venue_first_innings_avg_prior30",
    "team1_prior_matches", "team2_prior_matches", "h2h_prior_matches",
    "venue_prior_matches", "toss_won_by_first_batting", "toss_bat_first",
    "team1_elo", "team2_elo", "elo_diff",
    "team1_days_since_last_match", "team2_days_since_last_match",
    "days_since_last_match_diff",
]
TRAIN_CUTOFF = pd.Timestamp("2021-10-01")
TEST_CUTOFF = pd.Timestamp("2023-10-19")


def add_toss_features(df):
    df = df.copy()
    decision = df["toss_decision"].fillna("").astype(str).str.lower()
    winner = df["toss_winner"].fillna("").astype(str)
    team1, team2 = df["team1"].astype(str), df["team2"].astype(str)
    # True when team1 is the side batting first.
    df["toss_won_by_first_batting"] = (
        ((winner == team1) & (decision == "bat")) |
        ((winner == team2) & (decision == "field"))
    ).astype("int8")
    df["toss_bat_first"] = (decision == "bat").astype("int8")
    return df


def category_maps(frame):
    return {c: sorted(frame[c].fillna("Unknown").astype(str).unique().tolist())
            for c in CATEGORICAL}


def make_matrix(frame, categories):
    x = frame[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c].fillna("Unknown").astype(str),
                              categories=categories[c]).codes.astype("int32")
    return x.replace([np.inf, -np.inf], np.nan)


def score(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    required = [c for c in FEATURES if c not in {
        "toss_won_by_first_batting", "toss_bat_first"
    }] + ["date", "match_id", "team1_win", "toss_winner", "toss_decision"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team1_win"] = pd.to_numeric(df["team1_win"], errors="coerce")
    df = df.dropna(subset=["date", "match_id", "team1_win"])
    df = df[df.team1_win.isin([0, 1])].copy()
    if df.match_id.duplicated().any():
        raise ValueError("Expected one row per match; duplicate match_id found.")
    df = add_toss_features(df).sort_values(["date", "match_id"]).reset_index(drop=True)

    train = df[df.date < TRAIN_CUTOFF].copy()
    valid = df[(df.date >= TRAIN_CUTOFF) & (df.date < TEST_CUTOFF)].copy()
    test = df[df.date >= TEST_CUTOFF].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    cats = category_maps(train)
    candidate = LGBMClassifier(
        n_estimators=2000, learning_rate=0.025, num_leaves=15,
        min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=8.0, reg_alpha=1.0, random_state=42, verbosity=-1,
    )
    candidate.fit(
        make_matrix(train, cats), train.team1_win,
        eval_set=[(make_matrix(valid, cats), valid.team1_win)],
        callbacks=[early_stopping(100, verbose=False)],
    )
    best_iteration = int(candidate.best_iteration_ or 200)

    train_valid = pd.concat([train, valid], ignore_index=True)
    final_categories = category_maps(train_valid)
    final_model = LGBMClassifier(
        n_estimators=best_iteration, learning_rate=0.025, num_leaves=15,
        min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=8.0, reg_alpha=1.0, random_state=42, verbosity=-1,
    )
    final_model.fit(make_matrix(train_valid, final_categories), train_valid.team1_win)
    probabilities = final_model.predict_proba(make_matrix(test, final_categories))[:, 1]

    metrics = score(test.team1_win, probabilities)
    metrics.update({
        "model": "LightGBM Elo + prior-form v1",
        "best_iteration": best_iteration,
        "train_matches": int(len(train)),
        "validation_matches": int(len(valid)),
        "test_matches": int(len(test)),
        "train_end": str(train.date.max().date()),
        "validation_end": str(valid.date.max().date()),
        "test_start": str(test.date.min().date()),
        "test_end": str(test.date.max().date()),
        "prediction_timing": "After toss; toss decision is an input feature.",
        "features": FEATURES,
        "warning": "Compare only with models scored on this same untouched chronological test window.",
    })
    joblib.dump({
        "model": final_model, "features": FEATURES,
        "categorical": CATEGORICAL, "categories": final_categories,
        "version": "elo-v1", "prediction_timing": "after_toss",
    }, out / "cricket_match_winner_elo_v1.joblib")
    (out / "elo_v1_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
