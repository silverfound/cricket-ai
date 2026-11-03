# Cricket prediction model stage contract

This repository has distinct prediction stages. Features must be gated by the information available at the moment each prediction is made. Pre-toss and post-toss winner predictions are both retained; the later prediction must never overwrite the earlier one.

## Stage 1 — Pre-toss match winner prediction

Runtime: `get_pre_match_prediction()` in `fetch_data.py`.

Current production input schema:
- `league_id`
- `team1_id`
- `team2_id`
- `t1_form`
- `t2_form`
- `h2h_win_rate`

Current runtime receives `match_format` but does not use it in the six-column model input. Toss winner/decision must not be included in a prediction made before the toss.

Venue is an eligible pre-toss feature because the scheduled ground is normally known before the toss. The current production artifact does not yet use venue: it has no venue columns and must not be given extra columns at inference. The fixture adapter now captures an explicitly provider-supplied venue name in match output and the prediction audit record. Retraining requires historical match-level rows with venue, match date, teams, result, format and gender. Derive venue rolling averages and venue-team performance strictly from matches completed before each fixture timestamp, with minimum-sample smoothing and chronological validation. Do not infer missing venues or use future matches.

Recommended next-artifact venue features: stable venue ID; prior format-aware average first-innings runs; smoothed prior venue win rates for each team; and pace/spin or boundary indicators only where reliable historical sources exist. Ground identity and historical ground behaviour are distinct from today’s pitch condition, which belongs only in a separate post-toss model.

Persist this prediction as its own immutable record, including match identifier, prediction timestamp, model version, both team probabilities, and stage=`pre_toss`.

## Stage 2 — Post-toss, pre-match winner prediction

This is a separate prediction made after the toss but before the first ball. It uses the same eligible pre-match information plus:
- `toss_winner`
- `toss_decision` (bat/field)
- today’s observed pitch condition (batting-friendly, pace assistance, spin assistance, two-paced), only from a reliable match-day source or verified input
- optionally weather/dew observations available at prediction time.

Do not substitute a generic venue profile for today’s pitch report. Store pitch source and observation time; if unknown, keep it unknown rather than guessing.

Train and validate a dedicated post-toss artifact with this exact stage-appropriate schema. Do not silently add toss columns to the existing pre-toss artifact or reuse it with an incompatible input schema.

Persist this as a second immutable record for the same match, with stage=`post_toss`, prediction timestamp, model version, and both team probabilities. Keep the pre-toss record unchanged so the two predictions can be compared later.

## Measuring the incremental value of the toss

Compare pre-toss and post-toss predictions on the same held-out matches:
- Report accuracy, ROC-AUC, log loss, Brier score and calibration for each stage, with sample counts.
- Report the paired change in probability for each team's winning chance from pre-toss to post-toss.
- Also compare the post-toss model against a post-toss ablation trained without toss fields. This isolates the incremental predictive value of toss information from other pipeline differences.
- Break results down by format, venue, toss decision and toss winner where sample sizes support it.
- Use chronological splits and ensure every feature is available at its prediction timestamp.

A probability shift after the toss shows how the model updated its view; by itself it does not prove the toss caused that change or improved predictive accuracy. The paired held-out evaluation and toss-feature ablation are needed to assess predictive importance.

## Stage 3 — First-innings live score projection

Runtime: the non-chase branch of `get_ai_predictions()`.

Current production input schema:
- `league_id`
- `batting_team_id`
- `bowling_team_id`
- `format`
- `current_score`
- `balls_remaining`
- `wickets_lost`
- `current_run_rate`
- `last_3_overs_runs`

This branch only runs after an innings has started and a score/ball count is available. It is therefore an in-innings projection, not a prediction of the first-innings total before the first ball.

Audit finding: `last_3_overs_runs` is currently hard-coded to 10 in runtime feature construction. This is not a live observation and should be replaced with a real recent-runs feature only after its source and training-time definition are confirmed. Until then, do not claim the model uses actual recent-over momentum.

Toss context (winner and decision) can be useful for a separate pre-innings score model or as historical context, but must not be inserted into this existing artifact without matching training schema and retraining.

## Stage 4 — Match winner after the first innings / during chase

Runtime: the chase branch of `get_ai_predictions()`.

Current production input schema:
- `league_id`
- `chasing_team_id`
- `defending_team_id`
- `runs_needed`
- `balls_remaining`
- `wickets_left`
- `current_run_rate`
- `required_run_rate`

The target and chase state already encode much of the first-innings outcome. Toss is not currently an explicit model input. Do not append toss fields to this artifact without retraining and checking whether they add signal after target and match state are known.

## Stage 5 — Chase-innings / live chase state

The current runtime does not expose a separate chase-score projection artifact. It exposes chase win probability via the Stage 4 branch. If a distinct chase-innings score prediction is intended, it needs its own target, training dataset, artifact, and inference function; do not conflate it with win probability.

## Toss feature policy

- Before toss: toss fields are unavailable and must not be used.
- After toss but before first ball: toss winner and decision are valid features for the dedicated post-toss winner prediction.
- During the first innings: toss is known, but the existing score artifact has no toss columns. Add only with matching training data and retraining.
- During the chase: toss is known, but target, score, wickets, and balls remaining are the immediate state. Test toss as an incremental feature only with a stage-correct dataset and temporal validation.

## First-innings total experiment (historical dataset)

A separate experiment was run for a **pre-first-ball innings-total estimate**, not the existing live score projection artifact. Target: `first_innings_runs`. Training/test rows used the dataset's supplied chronological split (10,992 train; 5,591 test). Predictors included format, gender, teams, competition, venue, pre-match rolling/Elo/strength/rest features and venue historical average. The toss variant added `toss_winner` and `toss_decision`.

| Model inputs | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Without toss | 31.517 | 40.426 | 0.4679 |
| With toss | 31.559 | 40.434 | 0.4677 |

Toss fields did not improve this overall test result in this feature setup. This does **not** prove toss is useless: effect can vary by format, venue, team and decision; the current experiment uses one aggregate target and existing feature dataset. Format sample sizes: T20 5,102; ODI 488; IT20 1 (not enough for a meaningful IT20 conclusion). This is a pre-innings total experiment and must not be presented as a result for the live score projection or chase-win models.

## Audit findings requiring attention

1. The existing live first-innings score inference currently hard-codes `last_3_overs_runs=10`. Replace only when real recent-over runs are available and the model is trained with the same definition.
2. Existing chase inference returns chase win probability, not a separate chase score projection.
3. Existing pre-toss model accepts a format argument but its artifact has no format feature.
4. Pre-toss and post-toss winner predictions must be stored as separate records; the post-toss record must not overwrite the pre-toss record.
5. Historical rolling features still require source-level verification that each value uses only matches completed before the prediction timestamp.

## Audit constraints before model replacement

1. Verify every historical rolling feature is calculated strictly from matches completed before the prediction timestamp.
2. Keep separate chronological validation for pre-toss, post-toss, first-innings live, and chase predictions.
3. Report AUC/log loss/Brier and calibration for win probabilities; MAE/RMSE and error by format for score projections.
4. Report sample sizes separately by format and gender. A one-match subgroup is not evaluable.
5. Keep current production artifacts unchanged until a replacement is trained with an exactly matching inference schema and passes integration tests.


## Venue feature engineering implementation

The utility in `venue_features.py` builds leakage-safe venue history columns from a completed-match table. Required columns are `match_date`, `venue`, `format`, `team1`, `team2`, `winner`, and `first_innings_runs`. It emits:
- `venue_id`
- `venue_format_prior_matches`
- `venue_format_prior_avg_first_innings_runs`
- `team1_venue_prior_win_rate`
- `team2_venue_prior_win_rate`

Venue averages are format-specific. Team rates are smoothed toward 0.5. Matches on the same calendar date are featurized against the state before that date, preventing same-day result leakage when exact completion timestamps are unavailable. Unseen venue averages use a fixed neutral prior (150 runs) unless a training-only prior is explicitly supplied. Do not calculate that prior from the full backtest dataset.

**Current data blocker:** this repository currently contains the aggregate team-form/H2H JSON but no match-level historical CSV with venue, date, winner and innings totals. Therefore no honest venue-aware model can be trained or evaluated from the repository data yet. Obtain a licensed/provider historical match dataset, normalize venue aliases, then run chronological training/evaluation. Keep the current production artifact active until the new artifact passes those checks.

Post-toss pitch modelling also needs timestamped match-day pitch observations joined to match outcomes. The current fixture feed extraction only records venue identity; it does not claim to know the pitch condition.
