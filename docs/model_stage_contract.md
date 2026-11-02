# Cricket prediction model stage contract

This repository has distinct prediction stages. Features must be gated by the information available at the moment each prediction is made.

## Stage 1 — Pre-match winner (before toss)

Runtime: `get_pre_match_prediction()` in `fetch_data.py`.

Current production input schema:
- `league_id`
- `team1_id`
- `team2_id`
- `t1_form`
- `t2_form`
- `h2h_win_rate`

Current runtime receives `match_format` but does not use it in the six-column model input. Toss winner/decision must not be included in a prediction made before the toss.

Potential future extension: a separate post-toss pre-match prediction, with toss winner and bat/field decision. Do not silently add those fields to the existing pre-toss artifact.

## Stage 2 — First-innings live score projection

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

## Stage 3 — Match winner after the first innings / during chase

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

## Stage 4 — Chase-innings / live chase state

The current runtime does not expose a separate chase-score projection artifact. It exposes chase win probability via the Stage 3 branch. If a distinct chase-innings score prediction is intended, it needs its own target, training dataset, artifact, and inference function; do not conflate it with win probability.

## Toss feature policy

- Before toss: toss fields are unavailable and must not be used.
- After toss but before first ball: toss winner and decision are valid features for a dedicated post-toss prediction.
- During the first innings: toss is known, but the existing score artifact has no toss columns. Add only with matching training data and retraining.
- During the chase: toss is known, but target, score, wickets, and balls remaining are the immediate state. Test toss as an incremental feature only with a stage-correct dataset and temporal validation.

## Audit constraints before model replacement

1. Verify every historical rolling feature is calculated strictly from matches completed before the prediction timestamp.
2. Keep separate chronological validation for pre-toss, post-toss, first-innings live, and chase predictions.
3. Report AUC/log loss/Brier and calibration for win probabilities; MAE/RMSE and error by format for score projections.
4. Report sample sizes separately by format and gender. A one-match subgroup is not evaluable.
5. Keep current production artifacts unchanged until a replacement is trained with an exactly matching inference schema and passes integration tests.
