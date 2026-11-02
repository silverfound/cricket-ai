import os
import re
import json
import requests
import joblib
import pandas as pd

from datetime import datetime, timedelta

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
    "WBBL", "WOMEN'S BIG BASH LEAGUE",
    "WPL", "WOMEN'S PREMIER LEAGUE",
    "ICC T20 WORLD CUP", "ICC CRICKET WORLD CUP",
    "THE HUNDRED"
]

# =========================================================
# LOAD MODELS
# =========================================================

try:
    score_model = joblib.load("score_pred_model_v3.pkl")
    team_league_encoders = joblib.load("team_league_encoders.pkl")
    win_model = joblib.load("win_prob_model_v3.pkl")

    # Prefer the newly trained artifact. It stores the estimator and its
    # validation-selected decision threshold together in a dictionary.
    final_pre_match_path = os.path.join("models", "final_pre_match_model_v3.pkl")
    if os.path.exists(final_pre_match_path):
        pre_match_artifact = joblib.load(final_pre_match_path)
        if not isinstance(pre_match_artifact, dict) or "model" not in pre_match_artifact:
            raise ValueError("Final pre-match artifact has an unexpected format.")
        pre_match_model = pre_match_artifact["model"]
        pre_match_threshold = float(pre_match_artifact.get("threshold", 0.5))
        expected_features = pre_match_artifact.get("features")
        required_features = [
            "league_id", "team1_id", "team2_id",
            "t1_form", "t2_form", "h2h_win_rate"
        ]
        if expected_features and list(expected_features) != required_features:
            raise ValueError("Final pre-match model feature order does not match production.")
        print(f"Final pre-match model loaded (threshold={pre_match_threshold:.2f}).")
    else:
        # Backward-compatible legacy estimator; used only if the new artifact
        # has not been deployed to this runtime.
        pre_match_model = joblib.load("pre_match_model_v3.pkl")
        pre_match_threshold = 0.5
        print("Legacy pre-match model loaded; final artifact not found.")

    print("Models successfully loaded.")
except Exception as e:
    print(f"Model loading error: {e}")
    score_model = team_league_encoders = win_model = pre_match_model = None
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
    
    if not any(x in fmt for x in ["T20", "ODI", "HUNDRED"]):
        return False
        
    for keyword in ALLOWED_KEYWORDS:
        if keyword in league:
            return True
            
    return False

# =========================================================
# SMART ENCODERS
# =========================================================

def safe_encode_league(league_name, encoder):
    if encoder is None or not league_name: return -1
    try:
        league_clean = str(league_name).strip().lower()
        for cls in encoder.classes_:
            if str(cls).strip().lower() == league_clean:
                return int(encoder.transform([cls])[0])
        return -1
    except: return -1

def safe_encode_team(team_name, encoder):
    if encoder is None or not team_name: return -1
    try:
        team_name_clean = str(team_name).strip().lower()
        for cls in encoder.classes_:
            if str(cls).strip().lower() == team_name_clean:
                return int(encoder.transform([cls])[0])

        def is_women_team(name):
            name = str(name).strip().lower()
            return bool(re.search(r"\bwomen\b", name) or re.search(r"\bw\b", name) or re.search(r"\(w\)", name))

        def normalize(name):
            name = str(name).strip().lower()
            name = re.sub(r"\(.*?\)", "", name)
            name = re.sub(r"\bwomen\b", "", name)
            name = re.sub(r"\(w\)", "", name)
            name = re.sub(r"\bw\b", "", name)
            name = re.sub(r"\s+", " ", name)
            return name.strip()

        api_is_women = is_women_team(team_name_clean)
        normalized_api = normalize(team_name_clean)
        matches = []

        for cls in encoder.classes_:
            cls_clean = str(cls).strip().lower()
            encoder_is_women = is_women_team(cls_clean)
            if api_is_women != encoder_is_women: continue
            if normalize(cls_clean) == normalized_api:
                matches.append(cls)

        if len(matches) == 1:
            return int(encoder.transform([matches[0]])[0])
        return -1
    except: return -1

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def get_pre_match_prediction(home_team, away_team, match_format, league_name, gender="male"):
    if pre_match_model is None or team_league_encoders is None: return None
    try:
        team_encoder = team_league_encoders.get("team")
        league_encoder = team_league_encoders.get("league")
        if team_encoder is None or league_encoder is None: return None

        t1 = safe_encode_team(home_team, team_encoder)
        t2 = safe_encode_team(away_team, team_encoder)
        l_id = safe_encode_league(league_name, league_encoder)
        if t1 == -1 or t2 == -1 or l_id == -1: return None

        t1_form = get_historical_form(home_team, gender)
        t2_form = get_historical_form(away_team, gender)
        h2h_win_rate = get_historical_h2h(home_team, away_team, gender)
        input_data = pd.DataFrame({
            "league_id": [l_id], "team1_id": [t1], "team2_id": [t2],
            "t1_form": [t1_form], "t2_form": [t2_form],
            "h2h_win_rate": [h2h_win_rate]
        })
        probabilities = pre_match_model.predict_proba(input_data)
        if probabilities is None or len(probabilities) == 0 or len(probabilities[0]) < 2: return None

        classes = list(getattr(pre_match_model, "classes_", [0, 1]))
        if 0 not in classes or 1 not in classes: return None
        home_index = classes.index(1)
        away_index = classes.index(0)
        home_probability = float(probabilities[0][home_index])
        away_probability = float(probabilities[0][away_index])
        home_prob = home_probability * 100
        away_prob = away_probability * 100

        if home_probability >= pre_match_threshold:
            return f"{home_team} ({round(home_prob, 1)}%)"
        return f"{away_team} ({round(away_prob, 1)}%)"
    except Exception as prediction_error:
        print(f"Pre-match prediction error: {prediction_error}")
        return None

def get_ai_predictions(home_info, away_info, match_format, league_name, home_team, away_team):
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

            c_id = safe_encode_team(chasing_team, team_encoder)
            d_id = safe_encode_team(defending_team, team_encoder)
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

            b_id = safe_encode_team(batting_team, team_encoder)
            bw_id = safe_encode_team(bowling_team, team_encoder)
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
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict): all_data.extend(safe_list(data.get("data")))
        except: pass
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
            supported_format = any(x in fmt.upper() for x in ["T20", "ODI", "HUNDRED"])

            if match_state == "live" and supported_format:
                win_prob, proj_score = get_ai_predictions(home_info, away_info, fmt, league, home_team, away_team)
            elif match_state == "upcoming" and supported_format:
                match_gender = infer_match_gender(m, league, home_team, away_team)
                pre_match_winner = get_pre_match_prediction(home_team, away_team, fmt, league, match_gender)

            matches.append({
                "league": league, "format": fmt, "start_time_iso": safe_text(m.get("startTime")), "match_state": match_state,
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

            if match_state == "live":
                win_prob, proj_score = get_ai_predictions(
                    {"score": home_score, "info": home_overs}, {"score": away_score, "info": away_overs},
                    fmt, league, home_team, away_team
                )

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
    seen = set()
    unique_matches = []

    for match in matches:
        home = clean_team_name(match.get("home"))
        away = clean_team_name(match.get("away"))

        if not home or not away:
            unique_matches.append(match)
            continue

        start_time = safe_text(match.get("start_time_iso"))
        date_key = start_time[:10] if start_time else ""

        # Date missing ho to match preserve karein
        if not date_key:
            unique_matches.append(match)
            continue

        teams_key = tuple(sorted([home, away]))
        key = (teams_key, date_key)

        if key in seen:
            continue

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
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_matches": len(final_matches),
        "data_source_api": used_api,
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
