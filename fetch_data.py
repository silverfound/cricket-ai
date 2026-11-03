import os
import re
import json
import requests
import joblib
import pandas as pd

from datetime import datetime, timedelta, timezone
from prediction_audit import log_pre_match_prediction
from venue_model import load_venue_candidate, predict_venue_candidate

venue_candidate_artifact, venue_candidate_state = load_venue_candidate()
VENUE_MODEL_ACTIVE = venue_candidate_artifact is not None and venue_candidate_state is not None
ACTIVE_PRE_MATCH_MODEL = "venue_aware_candidate" if VENUE_MODEL_ACTIVE else "final_pre_match_model_v3"
print(f"PRE-MATCH ENGINE: {ACTIVE_PRE_MATCH_MODEL}")

# =========================================================
# CONFIGURATION
# =========================================================

KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY")

today = datetime.now().strftime("%Y-%m-%d")
tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

# 🔥 STRICT ALLOWLIST (Reviewer's Safe Version)
ALLOWED_KEYWORDS = [
    "IPL", "INDIAN PREMIER LEAGUE",
    "BBL", "BIG BASH LEAGUE",
    "PSL", "PAKISTAN SUPER LEAGUE",
    "CPL", "CARIBBEAN PREMIER LEAGUE",
    "SA20", "ILT20", "T20 BLAST", "LPL", "BPL", "SUPER SMASH",
    "CSA T20 CHALLENGE", "CSA T20", "CSA PRO20",
    "GT20", "GLOBAL T20", "GLOBAL T20 CANADA",
    "WBBL", "WOMEN'S BIG BASH LEAGUE",
    "WPL", "WOMEN'S PREMIER LEAGUE",
    "ICC T20 WORLD CUP", "ICC CRICKET WORLD CUP",
    "ICC MEN'S CRICKET WORLD CUP LEAGUE 2",
    "INDIA IN NEW ZEALAND",
    "THE HUNDRED"
]

# =========================================================
# LOAD MODELS
# =========================================================

try:
    # Only the final pre-match winner model is active.
    # Legacy live score/chase model files remain in the repository as backups.
    score_model = None
    win_model = None
    team_league_encoders = joblib.load("team_league_encoders.pkl")
    final_pre_match_path = os.path.join("models", "final_pre_match_model_v3.pkl")
    if not os.path.exists(final_pre_match_path):
        raise FileNotFoundError("Final pre-match model artifact is required; legacy fallback is disabled.")

    pre_match_artifact = joblib.load(final_pre_match_path)
    if not isinstance(pre_match_artifact, dict) or "model" not in pre_match_artifact:
        raise ValueError("Final pre-match artifact has an unexpected format.")
    pre_match_model = pre_match_artifact["model"]
    pre_match_threshold = float(pre_match_artifact.get("threshold", 0.5))
    required_features = ["league_id", "team1_id", "team2_id", "t1_form", "t2_form", "h2h_win_rate"]
    expected_features = pre_match_artifact.get("features")
    label_mapping = pre_match_artifact.get("label_mapping", {})
    if list(expected_features or []) != required_features:
        raise ValueError("Final pre-match model feature order does not match production.")
    if label_mapping != {1: "Team 1 wins", 0: "Team 2 wins"}:
        raise ValueError("Final pre-match model label mapping is missing or incompatible.")
    model_classes = list(getattr(pre_match_model, "classes_", []))
    if model_classes != [0, 1]:
        raise ValueError(f"Unexpected final pre-match model classes: {model_classes}")
    model_features = list(getattr(pre_match_model, "feature_names_in_", []))
    if model_features and model_features != required_features:
        raise ValueError("Final pre-match estimator feature names do not match production.")
    if not 0.0 <= pre_match_threshold <= 1.0:
        raise ValueError("Final pre-match model threshold must be between 0 and 1.")
    print(f"Final pre-match model loaded (threshold={pre_match_threshold:.2f}).")
    print("Legacy live score/chase models are disabled.")
except Exception as e:
    print(f"Model loading error: {e}")
    score_model = win_model = team_league_encoders = pre_match_model = None
    pre_match_threshold = 0.5

# =========================================================
# COMMON HELPERS
# =========================================================

def safe_dict(value):
    return value if isinstance(value, dict) else {}

def safe_list(value):
    return value if isinstance(value, list) else []

def safe_text(value, default=""):
    if value is None: return default
    return str(value).strip()

HISTORY_PATH = os.path.join("data", "team_form_h2h_history.json")
historical_features = {}
try:
    with open(HISTORY_PATH, "r", encoding="utf-8") as history_file:
        historical_features = json.load(history_file)
    history_meta = safe_dict(historical_features.get("metadata"))
    print(f"Historical form/H2H loaded: {history_meta.get('valid_matches', 0)} matches.")
except FileNotFoundError:
    print("Historical form/H2H file not found; neutral defaults will be used.")
except Exception as history_error:
    print(f"Historical form/H2H load error: {history_error}")
    historical_features = {}

def normalize_history_team(team_name):
    return re.sub(r"\s+", " ", safe_text(team_name)).strip().casefold()

def get_historical_form(team_name, gender):
    gender_key = "female" if str(gender).lower() == "female" else "male"
    teams = safe_dict(safe_dict(historical_features.get("teams")).get(gender_key))
    wanted = normalize_history_team(team_name)
    for name, values in teams.items():
        if normalize_history_team(name) == wanted:
            try:
                return float(safe_dict(values).get("form", 0.5))
            except (TypeError, ValueError):
                return 0.5
    return 0.5

def get_historical_h2h(team1, team2, gender):
    gender_key = "female" if str(gender).lower() == "female" else "male"
    h2h = safe_dict(safe_dict(historical_features.get("h2h")).get(gender_key))
    wanted1, wanted2 = normalize_history_team(team1), normalize_history_team(team2)
    for name1, opponents in h2h.items():
        if normalize_history_team(name1) == wanted1:
            for name2, record in safe_dict(opponents).items():
                if normalize_history_team(name2) == wanted2:
                    try:
                        return float(safe_dict(record).get("win_rate", 0.5))
                    except (TypeError, ValueError):
                        return 0.5
    return 0.5

def extract_venue(match_data):
    """Return provider-supplied stadium/ground name without guessing."""
    match_data = safe_dict(match_data)
    for key in ("venue", "ground", "stadium"):
        value = match_data.get(key)
        if isinstance(value, dict):
            name = safe_text(value.get("name") or value.get("title") or value.get("groundName"))
            if name:
                return name
        elif isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("venueName", "groundName", "stadiumName"):
        value = safe_text(match_data.get(key))
        if value:
            return value
    return ""

def infer_match_gender(match_data, league_name, home_team, away_team):
    match_data = safe_dict(match_data)
    for key in ("gender", "sex"):
        value = safe_text(match_data.get(key)).lower()
        if value in ("female", "women", "woman", "womens"):
            return "female"
        if value in ("male", "men", "mens"):
            return "male"
    labels = " ".join([safe_text(league_name), safe_text(home_team), safe_text(away_team)]).lower()
    if re.search(r"women|woman|wbb|wpl|\(w\)", labels):
        return "female"
    return "male"


def extract_overs(info_str):
    if not info_str: return ""
    text = str(info_str).strip()
    # The Hundred is reported in balls, not six-ball overs.
    balls_match = re.search(r"\b(\d+)\s*balls?\b", text, re.I)
    if balls_match:
        return f"{balls_match.group(1)} balls"
    match = re.search(r"(\d+\.\d+|\d+)\s*ov(?:er)?s?\b", text, re.I)
    if match: return f"{match.group(1)} ov"
    match = re.search(r"^\s*(\d+\.\d+|\d+)\s*/", text)
    if match: return f"{match.group(1)} ov"
    return ""

def clean_team_name(name):
    name = safe_text(name)
    name = re.sub(r'\s*\(NPL\)\s*', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\[NPL\]\s*', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.casefold()

def is_ai_supported(match):
    fmt = str(match.get("format", "")).upper()
    league = str(match.get("league", "")).upper()
    
    if "THE FORD TROPHY" in league:
        return False

    if not any(x in fmt for x in ["T20", "ODI", "HUNDRED"]):
        return False

    # ODI is a supported model format; explicitly exclude The Ford Trophy above.
    if "ODI" in fmt:
        return True

    return any(keyword in league for keyword in ALLOWED_KEYWORDS)

# =========================================================
# SMART ENCODERS
# =========================================================

def safe_encode_league(league_name, encoder, fallback_names=None):
    """Encode the exact competition, then only explicitly supplied known aliases."""
    if encoder is None: return -1
    try:
        available = {str(cls).strip().casefold(): cls for cls in encoder.classes_}
        candidates = [league_name] + list(fallback_names or [])
        for candidate in candidates:
            key = str(candidate or "").strip().casefold()
            if key and key in available:
                return int(encoder.transform([available[key]])[0])
        return -1
    except Exception as exc:
        print(f"League encoding failed for {league_name!r}: {exc}")
        return -1

def is_womens_competition(league_name):
    league = str(league_name or "").strip().lower()
    return "women" in league or "wpl" in league or "wbbl" in league


def safe_encode_team(team_name, encoder, gender=None):
    """Encode a team without crossing the men's/women's class boundary."""
    if encoder is None or not team_name:
        return -1
    try:
        requested = str(team_name).strip()
        requested_lower = requested.lower()

        def is_women(name):
            return bool(re.search(r"\bwomen\b|\bw\b|\(w\)", str(name).strip().lower()))

        def normalize(name):
            value = re.sub(r"\(.*?\)", "", str(name).strip().lower())
            value = re.sub(r"\bwomen\b|\(w\)|\bw\b", "", value)
            return re.sub(r"\s+", " ", value).strip()

        wanted_women = (gender == "female") if gender in ("female", "male") else is_women(requested)
        candidates = [str(cls) for cls in encoder.classes_
                      if is_women(cls) == wanted_women and normalize(cls) == normalize(requested)]
        if len(candidates) == 1:
            return int(encoder.transform([candidates[0]])[0])
        exact = [str(cls) for cls in encoder.classes_
                 if str(cls).strip().lower() == requested_lower and is_women(cls) == wanted_women]
        if len(exact) == 1:
            return int(encoder.transform([exact[0]])[0])
        return -1
    except Exception as exc:
        print(f"Team encoding failed for {team_name!r}: {exc}")
        return -1

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def get_pre_match_prediction(home_team, away_team, match_format, league_name, gender="male", venue=None):
    # Prefer the venue-aware candidate only when its separately versioned artifacts
    # are installed. Otherwise retain the validated production baseline unchanged.
    venue_prediction = predict_venue_candidate(
        venue_candidate_artifact, venue_candidate_state, home_team, away_team,
        match_format, gender, league_name, venue
    )
    if venue_prediction:
        return venue_prediction
    if pre_match_model is None or team_league_encoders is None:
        return None
    try:
        team_encoder = team_league_encoders.get("team")
        league_encoder = team_league_encoders.get("league")
        if team_encoder is None or league_encoder is None:
            return None

        gender = "female" if str(gender).lower() == "female" else "male"
        t1 = safe_encode_team(home_team, team_encoder, gender)
        t2 = safe_encode_team(away_team, team_encoder, gender)
        # Some providers publish a tour label (e.g. "India in New Zealand")
        # while the training encoder stores the host country's ODI-series label.
        # Use this fallback only when that exact category exists in the fitted encoder.
        league_fallbacks = []
        if "ODI" in str(match_format or "").upper():
            for team in (home_team, away_team):
                league_fallbacks.append(f"{team} ODI Series")
            # The trained encoder includes a neutral "Unknown League" bucket.
            # Use it only when the real competition and team-series aliases are absent.
            league_fallbacks.append("Unknown League")
        l_id = safe_encode_league(league_name, league_encoder, league_fallbacks)
        if t1 == -1 or t2 == -1 or l_id == -1:
            print(
                "Pre-match inputs unavailable: "
                f"home_team_id={t1}, away_team_id={t2}, league_id={l_id}, "
                f"league={league_name!r}, format={match_format!r}"
            )
            return None

        input_data = pd.DataFrame({
            "league_id": [l_id], "team1_id": [t1], "team2_id": [t2],
            "t1_form": [get_historical_form(home_team, gender)],
            "t2_form": [get_historical_form(away_team, gender)],
            "h2h_win_rate": [get_historical_h2h(home_team, away_team, gender)]
        })
        probabilities = pre_match_model.predict_proba(input_data)
        if probabilities is None or len(probabilities) == 0 or len(probabilities[0]) < 2:
            return None
        classes = list(getattr(pre_match_model, "classes_", [0, 1]))
        if 0 not in classes or 1 not in classes:
            return None
        home_probability = float(probabilities[0][classes.index(1)])
        away_probability = float(probabilities[0][classes.index(0)])
        threshold = float(globals().get("pre_match_threshold", 0.5))
        if home_probability >= threshold:
            return f"{home_team} ({round(home_probability * 100, 1)}%)"
        return f"{away_team} ({round(away_probability * 100, 1)}%)"
    except Exception as prediction_error:
        print(f"Pre-match prediction error: {prediction_error}")
        return None

def get_ai_predictions(home_info, away_info, match_format, league_name, home_team, away_team, gender=None):
    if (score_model is None or team_league_encoders is None or win_model is None): return None, None
    try:
        home_info = safe_dict(home_info)
        away_info = safe_dict(away_info)
        fmt = str(match_format or "").upper()

        if "HUNDRED" in fmt: total_balls, fmt_code = 100, 2
        elif "T20" in fmt: total_balls, fmt_code = 120, 0
        elif "ODI" in fmt: total_balls, fmt_code = 300, 1
        else: total_balls, fmt_code = 600, 2

        h_score_str = safe_text(home_info.get("score"), "-")
        h_info = safe_text(home_info.get("info"))
        a_score_str = safe_text(away_info.get("score"), "-")
        a_info = safe_text(away_info.get("info"))

        def parse_score(score_string):
            if not score_string: return None, None
            text = str(score_string).strip().lower()
            if text in ("-", "null", "none", ""): return None, None
            match = re.search(r"^\s*(\d+)\s*(?:/|-|\bfor\b)\s*(\d+)\b", text)
            if match:
                runs, wickets = int(match.group(1)), int(match.group(2))
                return (runs, wickets) if 0 <= wickets <= 10 else (runs, None)
            match = re.search(r"^\s*(\d+)\b", text)
            return (int(match.group(1)), None) if match else (None, None)

        def safe_wickets(wickets, default=0):
            try:
                if wickets is None: return default
                wickets = int(wickets)
                return wickets if 0 <= wickets <= 10 else default
            except: return default

        def parse_overs(info_string):
            if not info_string: return 0
            text = str(info_string).lower().strip()
            if "hundred" in fmt:
                match = re.search(r"\b(\d+)\s*balls?\b", text)
                if match: return int(match.group(1)) if 0 <= int(match.group(1)) <= 100 else 0
                match = re.search(r"^\s*(\d+)\s*/\s*100\b", text)
                if match: return int(match.group(1)) if 0 <= int(match.group(1)) <= 100 else 0
            match = re.search(r"\b(\d+)\.(\d+)\s*(?:ov(?:er)?s?)?\b", text)
            if match:
                completed_overs, extra_balls = int(match.group(1)), int(match.group(2))
                return 0 if extra_balls > 5 else (completed_overs * 6 + extra_balls)
            match = re.search(r"\b(\d+)\s*ov(?:er)?s?\b", text)
            return int(match.group(1)) * 6 if match else 0

        h_runs, h_wkts = parse_score(h_score_str)
        a_runs, a_wkts = parse_score(a_score_str)
        h_balls_bowled = parse_overs(h_info)
        a_balls_bowled = parse_overs(a_info)

        if h_runs is None and a_runs is None: return None, None
        if "break" in h_info.lower() or "break" in a_info.lower(): return None, None

        target = None
        chasing_score, chasing_wickets, chasing_balls_bowled = 0, None, 0
        chasing_team, defending_team = "", ""

        h_target_match = re.search(r"T\s*:\s*(\d+)", h_info, re.IGNORECASE)
        a_target_match = re.search(r"T\s*:\s*(\d+)", a_info, re.IGNORECASE)

        if h_target_match and a_runs is not None:
            target, chasing_score, chasing_wickets, chasing_balls_bowled = int(h_target_match.group(1)), a_runs, a_wkts, a_balls_bowled
            chasing_team, defending_team = away_team, home_team
        elif a_target_match and h_runs is not None:
            target, chasing_score, chasing_wickets, chasing_balls_bowled = int(a_target_match.group(1)), h_runs, h_wkts, h_balls_bowled
            chasing_team, defending_team = home_team, away_team
        elif h_runs is not None and a_runs is not None:
            home_completed = (h_balls_bowled >= total_balls or safe_wickets(h_wkts, default=-1) == 10)
            away_completed = (a_balls_bowled >= total_balls or safe_wickets(a_wkts, default=-1) == 10)
            if home_completed and not away_completed:
                target, chasing_score, chasing_wickets, chasing_balls_bowled = h_runs + 1, a_runs, a_wkts, a_balls_bowled
                chasing_team, defending_team = away_team, home_team
            elif away_completed and not home_completed:
                target, chasing_score, chasing_wickets, chasing_balls_bowled = a_runs + 1, h_runs, h_wkts, h_balls_bowled
                chasing_team, defending_team = home_team, away_team
            else: return None, None

        league_encoder = team_league_encoders.get("league")
        team_encoder = team_league_encoders.get("team")
        if league_encoder is None or team_encoder is None: return None, None

        l_id = safe_encode_league(league_name, league_encoder)
        if l_id == -1: return None, None

        if target is not None:
            runs_needed = target - chasing_score
            chasing_wickets_model = safe_wickets(chasing_wickets, default=0)
            wickets_left = 10 - chasing_wickets_model
            balls_remaining = total_balls - chasing_balls_bowled

            if balls_remaining <= 0 or runs_needed <= 0 or wickets_left <= 0 or chasing_balls_bowled < 0: return None, None

            overs_bowled = chasing_balls_bowled / 6
            crr = chasing_score / overs_bowled if overs_bowled > 0 else 0
            rrr = runs_needed / (balls_remaining / 6) if balls_remaining > 0 else 0

            match_gender = gender or ("female" if is_womens_competition(league_name) else "male")
            c_id = safe_encode_team(chasing_team, team_encoder, match_gender)
            d_id = safe_encode_team(defending_team, team_encoder, match_gender)
            if c_id == -1 or d_id == -1: return None, None

            input_data = pd.DataFrame({
                "league_id": [l_id], "chasing_team_id": [c_id], "defending_team_id": [d_id],
                "runs_needed": [runs_needed], "balls_remaining": [balls_remaining],
                "wickets_left": [wickets_left], "current_run_rate": [crr], "required_run_rate": [rrr]
            })
            probabilities = win_model.predict_proba(input_data)
            if probabilities is None or len(probabilities) == 0 or len(probabilities[0]) < 2: return None, None
            classes = getattr(win_model, "classes_", [0, 1])
            if 1 not in classes: return None, None
            return round(float(probabilities[0][list(classes).index(1)]) * 100, 1), None
        else:
            batting_score, batting_wickets, batting_balls_bowled = None, None, 0
            batting_team, bowling_team = "", ""
            if h_runs is not None and a_runs is None:
                batting_score, batting_wickets, batting_balls_bowled, batting_team, bowling_team = h_runs, h_wkts, h_balls_bowled, home_team, away_team
            elif a_runs is not None and h_runs is None:
                batting_score, batting_wickets, batting_balls_bowled, batting_team, bowling_team = a_runs, a_wkts, a_balls_bowled, away_team, home_team
            else: return None, None

            batting_is_all_out = (safe_wickets(batting_wickets, default=-1) == 10)
            balls_remaining = total_balls - batting_balls_bowled
            if batting_is_all_out or balls_remaining <= 0 or batting_balls_bowled <= 0: return None, None

            batting_wickets_model = safe_wickets(batting_wickets, default=0)
            overs_bowled = batting_balls_bowled / 6
            crr = batting_score / overs_bowled if overs_bowled > 0 else 0

            match_gender = gender or ("female" if is_womens_competition(league_name) else "male")
            b_id = safe_encode_team(batting_team, team_encoder, match_gender)
            bw_id = safe_encode_team(bowling_team, team_encoder, match_gender)
            if b_id == -1 or bw_id == -1: return None, None

            input_data = pd.DataFrame({
                "league_id": [l_id], "batting_team_id": [b_id], "bowling_team_id": [bw_id],
                "format": [fmt_code], "current_score": [batting_score], "balls_remaining": [balls_remaining],
                "wickets_lost": [batting_wickets_model], "current_run_rate": [crr], "last_3_overs_runs": [10]
            })
            predictions = score_model.predict(input_data)
            if predictions is None or len(predictions) == 0: return None, None

            pred_score = float(predictions[0])
            if pred_score < batting_score: pred_score = batting_score
            return None, f"{int(pred_score) - 5} - {int(pred_score) + 5}"
    except: return None, None

# =========================================================
# API FETCH FUNCTIONS
# =========================================================

def fetch_highlightly():
    if not KEY_HIGHLIGHTLY: return None, None
    all_data = []
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    for match_date in [today, tomorrow]:
        url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={match_date}"
        try:
            response = requests.get(
                "https://cricket-highlights-api.p.rapidapi.com/matches",
                headers=headers,
                params={
                    "date": match_date,
                    "limit": 100,
                    "offset": 0
                },
                timeout=20
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    all_data.extend(safe_list(data.get("data")))
                elif isinstance(data, list):
                    all_data.extend(data)
            else:
                print(f"Highlightly API error for {match_date}: HTTP {response.status_code} - {response.text[:300]}")
        except Exception as e:
            print(f"Highlightly request failed for {match_date}: {e}")
    return {"data": all_data}, "Highlightly"

def fetch_cricketdata():
    if not KEY_CRICKETDATA: return None, None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200: return response.json(), "CricketData"
    except: pass
    return None, None

def fetch_matchstat():
    if not KEY_MATCHSTAT: return None, None
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live"
    headers = {"x-rapidapi-key": KEY_MATCHSTAT, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200: return response.json(), "Matchstat"
    except: pass
    return None, None

def fetch_sportscore():
    url = "https://sportscore.com/api/widget/matches/?sport=cricket&limit=50"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200: return response.json(), "Sportscore"
    except: pass
    return None, None

# =========================================================
# DATA STANDARDIZATION
# =========================================================

def standardize_data(data, source):
    matches = []
    if not isinstance(data, dict): return matches

    if source == "Highlightly":
        for m in safe_list(data.get("data")):
            m = safe_dict(m)
            state = safe_dict(m.get("state"))
            status_raw = safe_text(state.get("description"), "Unknown")
            status_lower = status_raw.lower()

            if "no live coverage" in status_lower: continue
            if "scheduled" in status_lower or "not started" in status_lower: match_state = "upcoming"
            elif any(word in status_lower for word in ["finished", "won by", "abandoned", "result"]): match_state = "finished"
            else: match_state = "live"

            state_teams = safe_dict(state.get("teams"))
            home_info = safe_dict(state_teams.get("home"))
            away_info = safe_dict(state_teams.get("away"))
            fmt = safe_text(m.get("format"), "")
            
            league_data = safe_dict(m.get("league"))
            home_data = safe_dict(m.get("homeTeam"))
            away_data = safe_dict(m.get("awayTeam"))

            league = safe_text(league_data.get("name"), "League")
            home_team = safe_text(home_data.get("name"), "Home")
            away_team = safe_text(away_data.get("name"), "Away")

            win_prob, proj_score, pre_match_winner = None, None, None
            # The Hundred fixtures may be displayed, but prediction models are
            # not enabled until Hundred-specific ball-rate features are validated.
            prediction_supported_format = any(x in fmt.upper() for x in ["T20", "ODI"])

            match_gender = infer_match_gender(m, league, home_team, away_team)
            venue_name = extract_venue(m)
            if match_state == "upcoming" and prediction_supported_format:
                pre_match_winner = get_pre_match_prediction(home_team, away_team, fmt, league, match_gender, venue_name)
                prediction_model = "venue_aware_candidate" if VENUE_MODEL_ACTIVE and venue_name else "final_pre_match_model_v3"
                log_pre_match_prediction({
                    "source": "Highlightly",
                    "source_match_id": safe_text(m.get("id") or m.get("matchId")),
                    "league": league,
                    "format": fmt,
                    "venue": venue_name,
                    "start_time_iso": safe_text(m.get("startTime")),
                    "team1": home_team,
                    "team2": away_team,
                    "gender": match_gender,
                    "prediction": pre_match_winner,
                    "model_version": prediction_model,
                    "actual_winner": None,
                })

            matches.append({
                "source_match_id": safe_text(m.get("id") or m.get("matchId")),
                "league": league, "format": fmt, "venue": venue_name, "start_time_iso": safe_text(m.get("startTime")), "match_state": match_state,
                "home": home_team, "away": away_team, "score_home": safe_text(home_info.get("score"), "-"),
                "overs_home": extract_overs(home_info.get("info", "")), "score_away": safe_text(away_info.get("score"), "-"),
                "overs_away": extract_overs(away_info.get("info", "")), "status": status_raw, "api_source": source,
                "win_probability": win_prob, "projected_score": proj_score, "pre_match_winner": pre_match_winner
            })

    elif source == "CricketData":
        for m in safe_list(data.get("data")):
            m = safe_dict(m)
            status_raw = safe_text(m.get("status"), "Unknown")
            if m.get("matchEnded"): match_state = "finished"
            elif m.get("matchStarted"): match_state = "live"
            else: match_state = "upcoming"

            home_score, home_overs, away_score, away_overs = "-", "", "-", ""
            teams = safe_list(m.get("teams"))
            if len(teams) < 2: continue

            home_team, away_team = safe_text(teams[0], "Home"), safe_text(teams[1], "Away")

            for score in safe_list(m.get("score")):
                score = safe_dict(score)
                inning = safe_text(score.get("inning"))
                runs, wickets, overs = score.get("r", 0), score.get("w", 0), score.get("o", 0)
                if inning.startswith(home_team): home_score, home_overs = f"{runs}/{wickets}", f"{overs} ov"
                elif inning.startswith(away_team): away_score, away_overs = f"{runs}/{wickets}", f"{overs} ov"

            fmt = safe_text(m.get("matchType"), "")
            league = safe_text(m.get("name"), "League")
            win_prob, proj_score = None, None

            prediction_supported_format = any(x in fmt.upper() for x in ("T20", "ODI"))

            matches.append({
                "league": league, "format": fmt, "start_time_iso": safe_text(m.get("dateTimeGMT")), "match_state": match_state,
                "home": home_team, "away": away_team, "score_home": home_score, "overs_home": home_overs,
                "score_away": away_score, "overs_away": away_overs, "status": status_raw, "api_source": source,
                "win_probability": win_prob, "projected_score": proj_score, "pre_match_winner": None
            })

    elif source == "Matchstat":
        for m in safe_list(data.get("results")):
            m = safe_dict(m)
            tournament = safe_dict(m.get("tournament"))
            home_data, away_data = safe_dict(m.get("homeTeam")), safe_dict(m.get("awayTeam"))
            home_score_data, away_score_data = safe_dict(m.get("homeScore")), safe_dict(m.get("awayScore"))
            status_data = safe_dict(m.get("status"))

            matches.append({
                "league": safe_text(tournament.get("name"), "League"), 
                "format": safe_text(m.get("format"), ""), 
                "start_time_iso": "", "match_state": "live",
                "home": safe_text(home_data.get("name"), "Home"), "away": safe_text(away_data.get("name"), "Away"),
                "score_home": safe_text(home_score_data.get("display"), "-"), "overs_home": "",
                "score_away": safe_text(away_score_data.get("display"), "-"), "overs_away": "",
                "status": safe_text(status_data.get("description"), "In Play"), "api_source": source,
                "win_probability": None, "projected_score": None, "pre_match_winner": None
            })

    elif source == "Sportscore":
        for m in safe_list(data.get("matches")):
            m = safe_dict(m)
            status_raw = safe_text(m.get("status_text"))
            status_lower = status_raw.lower()
            if "upcoming" in status_lower or "abnormal" in status_lower: match_state = "upcoming"
            elif "finished" in status_lower: match_state = "finished"
            else: match_state = "live"

            matches.append({
                "league": safe_text(m.get("competition"), "League"), 
                "format": safe_text(m.get("format"), ""), 
                "start_time_iso": safe_text(m.get("time")), "match_state": match_state,
                "home": safe_text(m.get("home"), "Home"), "away": safe_text(m.get("away"), "Away"),
                "score_home": safe_text(m.get("home_score"), "-"), "overs_home": "",
                "score_away": safe_text(m.get("away_score"), "-"), "overs_away": "",
                "status": status_raw, "api_source": source,
                "win_probability": None, "projected_score": None, "pre_match_winner": None
            })

    return matches

# =========================================================
# MATCH DEDUPLICATION
# =========================================================

def deduplicate_matches(matches):
    """Deduplicate only with a reliable ID or exact fixture timestamp."""
    seen = set()
    unique_matches = []
    for match in matches:
        source_id = safe_text(match.get("source_match_id"))
        home = clean_team_name(match.get("home"))
        away = clean_team_name(match.get("away"))
        start_time = safe_text(match.get("start_time_iso"))
        league = safe_text(match.get("league")).casefold()
        fmt = safe_text(match.get("format")).casefold()
        if source_id:
            key = ("id", source_id)
        elif home and away and start_time:
            key = ("fixture", tuple(sorted((home, away))), league, fmt, start_time)
        else:
            unique_matches.append(match)
            continue
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    return unique_matches

# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    all_matches = []
    successful_sources = []

    # 1. PRIMARY: Highlightly
    print("Trying Primary API: Highlightly...")
    data, source = fetch_highlightly()
    if data:
        primary_matches = standardize_data(data, source)
        supported_primary = [m for m in primary_matches if is_ai_supported(m)]
        if supported_primary:
            all_matches = supported_primary
            successful_sources.append("Highlightly")
            print(f"Received {len(supported_primary)} supported matches from Highlightly.")

    # 2. FALLBACK: Only if no supported matches found from Primary
    if len(all_matches) == 0:
        print("No supported Highlightly matches. Trying fallbacks...")
        fallbacks = [
            ("CricketData", fetch_cricketdata),
            ("Matchstat", fetch_matchstat),
            ("Sportscore", fetch_sportscore)
        ]
        for api_name, fetch_func in fallbacks:
            print(f"Trying fallback: {api_name}...")
            f_data, f_source = fetch_func()
            if f_data:
                raw_matches = standardize_data(f_data, f_source)
                supported_fallback = [m for m in raw_matches if is_ai_supported(m)]
                if len(supported_fallback) > 0:
                    all_matches = supported_fallback
                    successful_sources.append(f_source)
                    print(f"Fallback success: {api_name} ({len(supported_fallback)} supported matches).")
                    break

    # 3. DEDUPLICATION & SORTING
    final_matches = deduplicate_matches(all_matches)
    final_matches.sort(key=lambda x: x.get("start_time_iso") or "")

    used_api = ", ".join(successful_sources) if successful_sources else "None"
    
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(final_matches),
        "data_source_api": used_api,
        "prediction_engine": ACTIVE_PRE_MATCH_MODEL,
        "venue_model_active": VENUE_MODEL_ACTIVE,
        "matches": final_matches
    }

    # 4. SAVE SAFELY
    temp_file = "live_matches.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        os.replace(temp_file, "live_matches.json")
        print(f"Saved {len(final_matches)} matches. Sources: {used_api}")
    except Exception as e:
        print(f"JSON save error: {e}")
        if os.path.exists(temp_file): os.remove(temp_file)

if __name__ == "__main__":
    main()
