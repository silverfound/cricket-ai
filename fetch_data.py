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


# =========================================================
# LOAD MODELS
# =========================================================

try:
    score_model = joblib.load("score_pred_model_v3.pkl")
    team_league_encoders = joblib.load("team_league_encoders.pkl")
    win_model = joblib.load("win_prob_model_v3.pkl")
    pre_match_model = joblib.load("pre_match_model_v3.pkl")

    print("Models successfully loaded.")

except Exception as e:
    print(f"Model loading error: {e}")

    score_model = None
    team_league_encoders = None
    win_model = None
    pre_match_model = None


# =========================================================
# COMMON HELPERS
# =========================================================

def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def safe_text(value, default=""):
    if value is None:
        return default

    return str(value).strip()


def extract_overs(info_str):
    """
    Extract overs for display in the UI.
    Examples:
        12.3 ov -> 12.3 ov
        15 overs -> 15 ov
        12.3/20 -> 12.3 ov
    """

    if not info_str:
        return ""

    text = str(info_str).strip()

    match = re.search(r"(\d+\.\d+|\d+)\s*ov(?:er)?s?\b", text, re.I)

    if match:
        return f"{match.group(1)} ov"

    match = re.search(r"^\s*(\d+\.\d+|\d+)\s*/", text)

    if match:
        return f"{match.group(1)} ov"

    return ""


# =========================================================
# SMART ENCODERS
# =========================================================

def safe_encode_league(league_name, encoder):
    if encoder is None or not league_name:
        return -1

    try:
        league_clean = str(league_name).strip().lower()

        for cls in encoder.classes_:
            if str(cls).strip().lower() == league_clean:
                return int(encoder.transform([cls])[0])

        return -1

    except Exception as e:
        print(f"League encoding error: {e}")
        return -1


def safe_encode_team(team_name, encoder):
    """
    Exact matching first.
    Normalized matching second.
    Men and Women teams are kept separate.
    Ambiguous matches are rejected.
    """

    if encoder is None or not team_name:
        return -1

    try:
        team_name_clean = str(team_name).strip().lower()

        # 1. Exact match
        for cls in encoder.classes_:
            if str(cls).strip().lower() == team_name_clean:
                return int(encoder.transform([cls])[0])

        def is_women_team(name):
            name = str(name).strip().lower()

            return bool(
                re.search(r"\bwomen\b", name)
                or re.search(r"\bw\b", name)
                or re.search(r"\(w\)", name)
            )

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

            # Never mix men's and women's teams
            if api_is_women != encoder_is_women:
                continue

            if normalize(cls_clean) == normalized_api:
                matches.append(cls)

        # Accept only one unambiguous result
        if len(matches) == 1:
            return int(encoder.transform([matches[0]])[0])

        return -1

    except Exception as e:
        print(f"Team encoding error: {e}")
        return -1


# =========================================================
# PRE-MATCH PREDICTION
# =========================================================

def get_pre_match_prediction(home_team, away_team, match_format, league_name):

    if pre_match_model is None or team_league_encoders is None:
        return None

    try:
        team_encoder = team_league_encoders.get("team")
        league_encoder = team_league_encoders.get("league")

        if team_encoder is None or league_encoder is None:
            return None

        t1 = safe_encode_team(home_team, team_encoder)
        t2 = safe_encode_team(away_team, team_encoder)

        l_id = safe_encode_league(league_name, league_encoder)

        # Do not silently assign unknown teams or leagues to ID 0
        if t1 == -1 or t2 == -1 or l_id == -1:
            return None

        input_data = pd.DataFrame({
            "league_id": [l_id],
            "team1_id": [t1],
            "team2_id": [t2],
            "t1_form": [0.5],
            "t2_form": [0.5],
            "h2h_win_rate": [0.5]
        })

        probabilities = pre_match_model.predict_proba(input_data)

        if probabilities is None or len(probabilities) == 0:
            return None

        if len(probabilities[0]) < 2:
            return None

        classes = getattr(pre_match_model, "classes_", [0, 1])

        if 0 not in classes or 1 not in classes:
            return None

        home_index = list(classes).index(1)
        away_index = list(classes).index(0)

        home_prob = float(probabilities[0][home_index]) * 100
        away_prob = float(probabilities[0][away_index]) * 100

        if home_prob > away_prob:
            return f"{home_team} ({round(home_prob, 1)}%)"

        return f"{away_team} ({round(away_prob, 1)}%)"

    except (IndexError, TypeError, ValueError, KeyError, AttributeError) as e:
        print(f"Pre-match prediction error: {e}")
        return None

    except Exception as e:
        print(f"Pre-match model error: {e}")
        return None


# =========================================================
# LIVE MATCH PREDICTION
# =========================================================

def get_ai_predictions(
    home_info,
    away_info,
    match_format,
    league_name,
    home_team,
    away_team
):

    if (
        score_model is None
        or team_league_encoders is None
        or win_model is None
    ):
        return None, None

    try:
        home_info = safe_dict(home_info)
        away_info = safe_dict(away_info)

        fmt = str(match_format or "").upper()

        # The Hundred has 100 BALLS, not 100 overs
        if "HUNDRED" in fmt:
            total_balls = 100
            fmt_code = 2

        elif "T20" in fmt:
            total_balls = 120
            fmt_code = 0

        elif "ODI" in fmt:
            total_balls = 300
            fmt_code = 1

        else:
            total_balls = 600
            fmt_code = 2

        h_score_str = safe_text(home_info.get("score"), "-")
        h_info = safe_text(home_info.get("info"))
        a_score_str = safe_text(away_info.get("score"), "-")
        a_info = safe_text(away_info.get("info"))

        # -------------------------------------------------
        # SCORE PARSER
        # -------------------------------------------------

        def parse_score(score_string):

            if not score_string:
                return None, None

            text = str(score_string).strip().lower()

            if text in ("-", "null", "none", ""):
                return None, None

            # Formats: 182/5, 182-5, 182 for 5
            match = re.search(
                r"^\s*(\d+)\s*(?:/|-|\bfor\b)\s*(\d+)\b",
                text
            )

            if match:
                runs = int(match.group(1))
                wickets = int(match.group(2))

                if 0 <= wickets <= 10:
                    return runs, wickets

                return runs, None

            # Format: 182 (wickets unknown)
            match = re.search(r"^\s*(\d+)\b", text)

            if match:
                return int(match.group(1)), None

            return None, None

        # -------------------------------------------------
        # SAFE WICKETS
        # -------------------------------------------------

        def safe_wickets(wickets, default=0):

            try:
                if wickets is None:
                    return default

                wickets = int(wickets)

                if 0 <= wickets <= 10:
                    return wickets

                return default

            except (TypeError, ValueError):
                return default

        # -------------------------------------------------
        # OVERS TO BALLS
        # -------------------------------------------------

        def parse_overs(info_string):

            if not info_string:
                return 0

            text = str(info_string).lower().strip()

            # The Hundred: e.g. 42 balls
            if "hundred" in fmt:
                match = re.search(r"\b(\d+)\s*balls?\b", text)

                if match:
                    balls = int(match.group(1))
                    return balls if 0 <= balls <= 100 else 0

                # Sometimes displayed as 42/100
                match = re.search(r"^\s*(\d+)\s*/\s*100\b", text)

                if match:
                    balls = int(match.group(1))
                    return balls if 0 <= balls <= 100 else 0

            # Cricket overs: 19.5 = 119 balls
            match = re.search(
                r"\b(\d+)\.(\d+)\s*(?:ov(?:er)?s?)?\b",
                text
            )

            if match:
                completed_overs = int(match.group(1))
                extra_balls = int(match.group(2))

                if extra_balls > 5:
                    return 0

                balls = completed_overs * 6 + extra_balls

                return balls if balls <= total_balls else 0

            # Whole overs: 15 overs = 90 balls
            match = re.search(
                r"\b(\d+)\s*ov(?:er)?s?\b",
                text
            )

            if match:
                completed_overs = int(match.group(1))
                balls = completed_overs * 6

                return balls if balls <= total_balls else 0

            return 0

        # -------------------------------------------------
        # PARSE MATCH DATA
        # -------------------------------------------------

        h_runs, h_wkts = parse_score(h_score_str)
        a_runs, a_wkts = parse_score(a_score_str)

        h_balls_bowled = parse_overs(h_info)
        a_balls_bowled = parse_overs(a_info)

        if h_runs is None and a_runs is None:
            return None, None

        if "break" in h_info.lower() or "break" in a_info.lower():
            return None, None

        # -------------------------------------------------
        # DETERMINE INNINGS AND CHASING TEAM
        # -------------------------------------------------

        target = None

        chasing_score = 0
        chasing_wickets = None
        chasing_balls_bowled = 0

        chasing_team = ""
        defending_team = ""

        h_target_match = re.search(
            r"T\s*:\s*(\d+)",
            h_info,
            re.IGNORECASE
        )

        a_target_match = re.search(
            r"T\s*:\s*(\d+)",
            a_info,
            re.IGNORECASE
        )

        # Explicit target in home info means away team is chasing
        if h_target_match and a_runs is not None:

            target = int(h_target_match.group(1))

            chasing_score = a_runs
            chasing_wickets = a_wkts
            chasing_balls_bowled = a_balls_bowled

            chasing_team = away_team
            defending_team = home_team

        # Explicit target in away info means home team is chasing
        elif a_target_match and h_runs is not None:

            target = int(a_target_match.group(1))

            chasing_score = h_runs
            chasing_wickets = h_wkts
            chasing_balls_bowled = h_balls_bowled

            chasing_team = home_team
            defending_team = away_team

        # Both scores are present: require evidence that one innings ended
        elif h_runs is not None and a_runs is not None:

            home_completed = (
                h_balls_bowled >= total_balls
                or safe_wickets(h_wkts, default=-1) == 10
            )

            away_completed = (
                a_balls_bowled >= total_balls
                or safe_wickets(a_wkts, default=-1) == 10
            )

            if home_completed and not away_completed:

                target = h_runs + 1

                chasing_score = a_runs
                chasing_wickets = a_wkts
                chasing_balls_bowled = a_balls_bowled

                chasing_team = away_team
                defending_team = home_team

            elif away_completed and not home_completed:

                target = a_runs + 1

                chasing_score = h_runs
                chasing_wickets = h_wkts
                chasing_balls_bowled = h_balls_bowled

                chasing_team = home_team
                defending_team = away_team

            else:
                # Do not guess the chasing team from score comparison
                return None, None

        # -------------------------------------------------
        # LEAGUE ENCODING
        # -------------------------------------------------

        league_encoder = team_league_encoders.get("league")
        team_encoder = team_league_encoders.get("team")

        if league_encoder is None or team_encoder is None:
            return None, None

        l_id = safe_encode_league(league_name, league_encoder)

        if l_id == -1:
            return None, None

        # =================================================
        # SECOND INNINGS / CHASE PREDICTION
        # =================================================

        if target is not None:

            runs_needed = target - chasing_score

            chasing_wickets_model = safe_wickets(
                chasing_wickets,
                default=0
            )

            wickets_left = 10 - chasing_wickets_model

            balls_remaining = total_balls - chasing_balls_bowled

            if (
                balls_remaining <= 0
                or runs_needed <= 0
                or wickets_left <= 0
                or chasing_balls_bowled < 0
            ):
                return None, None

            overs_bowled = chasing_balls_bowled / 6

            crr = (
                chasing_score / overs_bowled
                if overs_bowled > 0
                else 0
            )

            rrr = (
                runs_needed / (balls_remaining / 6)
                if balls_remaining > 0
                else 0
            )

            c_id = safe_encode_team(chasing_team, team_encoder)
            d_id = safe_encode_team(defending_team, team_encoder)

            # Unknown teams must not be mapped to another team's ID
            if c_id == -1 or d_id == -1:
                return None, None

            input_data = pd.DataFrame({
                "league_id": [l_id],
                "chasing_team_id": [c_id],
                "defending_team_id": [d_id],
                "runs_needed": [runs_needed],
                "balls_remaining": [balls_remaining],
                "wickets_left": [wickets_left],
                "current_run_rate": [crr],
                "required_run_rate": [rrr]
            })

            probabilities = win_model.predict_proba(input_data)

            if probabilities is None or len(probabilities) == 0:
                return None, None

            if len(probabilities[0]) < 2:
                return None, None

            classes = getattr(win_model, "classes_", [0, 1])

            if 1 not in classes:
                return None, None

            positive_index = list(classes).index(1)

            prob = float(probabilities[0][positive_index]) * 100

            return round(prob, 1), None

        # =================================================
        # FIRST INNINGS / PROJECTED SCORE
        # =================================================

        else:

            batting_score = None
            batting_wickets = None
            batting_balls_bowled = 0

            batting_team = ""
            bowling_team = ""

            if h_runs is not None and a_runs is None:

                batting_score = h_runs
                batting_wickets = h_wkts
                batting_balls_bowled = h_balls_bowled

                batting_team = home_team
                bowling_team = away_team

            elif a_runs is not None and h_runs is None:

                batting_score = a_runs
                batting_wickets = a_wkts
                batting_balls_bowled = a_balls_bowled

                batting_team = away_team
                bowling_team = home_team

            else:
                return None, None

            batting_is_all_out = (
                safe_wickets(batting_wickets, default=-1) == 10
            )

            balls_remaining = total_balls - batting_balls_bowled

            if (
                batting_is_all_out
                or balls_remaining <= 0
                or batting_balls_bowled <= 0
            ):
                return None, None

            batting_wickets_model = safe_wickets(
                batting_wickets,
                default=0
            )

            overs_bowled = batting_balls_bowled / 6

            crr = (
                batting_score / overs_bowled
                if overs_bowled > 0
                else 0
            )

            b_id = safe_encode_team(batting_team, team_encoder)
            bw_id = safe_encode_team(bowling_team, team_encoder)

            if b_id == -1 or bw_id == -1:
                return None, None

            input_data = pd.DataFrame({
                "league_id": [l_id],
                "batting_team_id": [b_id],
                "bowling_team_id": [bw_id],
                "format": [fmt_code],
                "current_score": [batting_score],
                "balls_remaining": [balls_remaining],
                "wickets_lost": [batting_wickets_model],
                "current_run_rate": [crr],
                "last_3_overs_runs": [10]
            })

            predictions = score_model.predict(input_data)

            if predictions is None or len(predictions) == 0:
                return None, None

            pred_score = float(predictions[0])

            if pred_score < batting_score:
                pred_score = batting_score

            return None, (
                f"{int(pred_score) - 5} - {int(pred_score) + 5}"
            )

    except IndexError:
        print("AI prediction error: model output index missing.")
        return None, None

    except (TypeError, ValueError, KeyError, AttributeError) as e:
        print(f"AI prediction data error: {e}")
        return None, None

    except Exception as e:
        print(f"AI prediction error: {e}")
        return None, None


# =========================================================
# API FETCH FUNCTIONS
# =========================================================

def fetch_highlightly():

    if not KEY_HIGHLIGHTLY:
        return None, None

    all_data = []

    headers = {
        "x-rapidapi-key": KEY_HIGHLIGHTLY,
        "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"
    }

    for match_date in [today, tomorrow]:

        url = (
            "https://cricket-highlights-api.p.rapidapi.com/"
            f"matches?date={match_date}"
        )

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            if response.status_code != 200:
                print(f"Highlightly HTTP error: {response.status_code}")
                continue

            data = response.json()

            if isinstance(data, dict):
                all_data.extend(safe_list(data.get("data")))

        except (requests.RequestException, ValueError) as e:
            print(f"Highlightly fetch error: {e}")

    return {"data": all_data}, "Highlightly"


def fetch_cricketdata():

    if not KEY_CRICKETDATA:
        return None, None

    url = (
        "https://api.cricapi.com/v1/currentMatches"
        f"?apikey={KEY_CRICKETDATA}&offset=0"
    )

    try:
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"CricketData HTTP error: {response.status_code}")
            return None, None

        data = response.json()

        if isinstance(data, dict):
            return data, "CricketData"

    except (requests.RequestException, ValueError) as e:
        print(f"CricketData fetch error: {e}")

    return None, None


def fetch_matchstat():

    if not KEY_MATCHSTAT:
        return None, None

    url = (
        "https://cricket-api-data-live-scores-stats-odds-predictions."
        "p.rapidapi.com/v1/live"
    )

    headers = {
        "x-rapidapi-key": KEY_MATCHSTAT,
        "x-rapidapi-host": (
            "cricket-api-data-live-scores-stats-odds-predictions."
            "p.rapidapi.com"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            print(f"Matchstat HTTP error: {response.status_code}")
            return None, None

        data = response.json()

        if isinstance(data, dict):
            return data, "Matchstat"

    except (requests.RequestException, ValueError) as e:
        print(f"Matchstat fetch error: {e}")

    return None, None


def fetch_sportscore():

    url = "https://sportscore.com/api/widget/matches/?sport=cricket&limit=50"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"Sportscore HTTP error: {response.status_code}")
            return None, None

        data = response.json()

        if isinstance(data, dict):
            return data, "Sportscore"

    except (requests.RequestException, ValueError) as e:
        print(f"Sportscore fetch error: {e}")

    return None, None


# =========================================================
# DATA STANDARDIZATION
# =========================================================

def standardize_data(data, source):

    matches = []

    if not isinstance(data, dict):
        return matches

    # -----------------------------------------------------
    # HIGHLIGHTLY
    # -----------------------------------------------------

    if source == "Highlightly":

        for m in safe_list(data.get("data")):

            m = safe_dict(m)

            state = safe_dict(m.get("state"))

            status_raw = safe_text(
                state.get("description"),
                "Unknown"
            )

            status_lower = status_raw.lower()

            if "no live coverage" in status_lower:
                continue

            if (
                "scheduled" in status_lower
                or "not started" in status_lower
            ):
                match_state = "upcoming"

            elif any(
                word in status_lower
                for word in [
                    "finished",
                    "won by",
                    "abandoned",
                    "result"
                ]
            ):
                match_state = "finished"

            else:
                match_state = "live"

            state_teams = safe_dict(state.get("teams"))

            home_info = safe_dict(state_teams.get("home"))
            away_info = safe_dict(state_teams.get("away"))

            fmt = safe_text(m.get("format"), "T20")

            league_data = safe_dict(m.get("league"))
            home_data = safe_dict(m.get("homeTeam"))
            away_data = safe_dict(m.get("awayTeam"))

            league = safe_text(league_data.get("name"), "League")
            home_team = safe_text(home_data.get("name"), "Home")
            away_team = safe_text(away_data.get("name"), "Away")

            win_prob = None
            proj_score = None
            pre_match_winner = None

            supported_format = any(
                x in fmt.upper()
                for x in ["T20", "ODI", "HUNDRED"]
            )

            if match_state == "live" and supported_format:

                win_prob, proj_score = get_ai_predictions(
                    home_info,
                    away_info,
                    fmt,
                    league,
                    home_team,
                    away_team
                )

            elif match_state == "upcoming" and supported_format:

                pre_match_winner = get_pre_match_prediction(
                    home_team,
                    away_team,
                    fmt,
                    league
                )

            matches.append({
                "league": league,
                "format": fmt,
                "start_time_iso": safe_text(m.get("startTime")),
                "match_state": match_state,
                "home": home_team,
                "away": away_team,
                "score_home": safe_text(
                    home_info.get("score"),
                    "-"
                ),
                "overs_home": extract_overs(
                    home_info.get("info", "")
                ),
                "score_away": safe_text(
                    away_info.get("score"),
                    "-"
                ),
                "overs_away": extract_overs(
                    away_info.get("info", "")
                ),
                "status": status_raw,
                "api_source": source,
                "win_probability": win_prob,
                "projected_score": proj_score,
                "pre_match_winner": pre_match_winner
            })

    # -----------------------------------------------------
    # CRICKETDATA
    # -----------------------------------------------------

    elif source == "CricketData":

        for m in safe_list(data.get("data")):

            m = safe_dict(m)

            status_raw = safe_text(
                m.get("status"),
                "Unknown"
            )

            if m.get("matchEnded"):
                match_state = "finished"

            elif m.get("matchStarted"):
                match_state = "live"

            else:
                match_state = "upcoming"

            home_score = "-"
            home_overs = ""

            away_score = "-"
            away_overs = ""

            teams = safe_list(m.get("teams"))

            if len(teams) < 2:
                continue

            home_team = safe_text(teams[0], "Home")
            away_team = safe_text(teams[1], "Away")

            for score in safe_list(m.get("score")):

                score = safe_dict(score)

                inning = safe_text(score.get("inning"))

                runs = score.get("r", 0)
                wickets = score.get("w", 0)
                overs = score.get("o", 0)

                score_text = f"{runs}/{wickets}"
                overs_text = f"{overs} ov"

                if inning.startswith(home_team):
                    home_score = score_text
                    home_overs = overs_text

                elif inning.startswith(away_team):
                    away_score = score_text
                    away_overs = overs_text

            fmt = safe_text(m.get("matchType"), "T20")
            league = safe_text(m.get("name"), "League")

            win_prob = None
            proj_score = None

            if match_state == "live":

                home_info = {
                    "score": home_score,
                    "info": home_overs
                }

                away_info = {
                    "score": away_score,
                    "info": away_overs
                }

                win_prob, proj_score = get_ai_predictions(
                    home_info,
                    away_info,
                    fmt,
                    league,
                    home_team,
                    away_team
                )

            matches.append({
                "league": league,
                "format": fmt,
                "start_time_iso": safe_text(m.get("dateTimeGMT")),
                "match_state": match_state,
                "home": home_team,
                "away": away_team,
                "score_home": home_score,
                "overs_home": home_overs,
                "score_away": away_score,
                "overs_away": away_overs,
                "status": status_raw,
                "api_source": source,
                "win_probability": win_prob,
                "projected_score": proj_score,
                "pre_match_winner": None
            })

    # -----------------------------------------------------
    # MATCHSTAT
    # -----------------------------------------------------

    elif source == "Matchstat":

        for m in safe_list(data.get("results")):

            m = safe_dict(m)

            tournament = safe_dict(m.get("tournament"))
            home_data = safe_dict(m.get("homeTeam"))
            away_data = safe_dict(m.get("awayTeam"))
            home_score_data = safe_dict(m.get("homeScore"))
            away_score_data = safe_dict(m.get("awayScore"))
            status_data = safe_dict(m.get("status"))

            matches.append({
                "league": safe_text(
                    tournament.get("name"),
                    "League"
                ),
                "format": "T20",
                "start_time_iso": "",
                "match_state": "live",
                "home": safe_text(
                    home_data.get("name"),
                    "Home"
                ),
                "away": safe_text(
                    away_data.get("name"),
                    "Away"
                ),
                "score_home": safe_text(
                    home_score_data.get("display"),
                    "-"
                ),
                "overs_home": "",
                "score_away": safe_text(
                    away_score_data.get("display"),
                    "-"
                ),
                "overs_away": "",
                "status": safe_text(
                    status_data.get("description"),
                    "In Play"
                ),
                "api_source": source,
                "win_probability": None,
                "projected_score": None,
                "pre_match_winner": None
            })

    # -----------------------------------------------------
    # SPORTscore
    # -----------------------------------------------------

    elif source == "Sportscore":

        for m in safe_list(data.get("matches")):

            m = safe_dict(m)

            status_raw = safe_text(m.get("status_text"))
            status_lower = status_raw.lower()

            if (
                "upcoming" in status_lower
                or "abnormal" in status_lower
            ):
                match_state = "upcoming"

            elif "finished" in status_lower:
                match_state = "finished"

            else:
                match_state = "live"

            matches.append({
                "league": safe_text(
                    m.get("competition"),
                    "League"
                ),
                "format": "T20",
                "start_time_iso": safe_text(m.get("time")),
                "match_state": match_state,
                "home": safe_text(m.get("home"), "Home"),
                "away": safe_text(m.get("away"), "Away"),
                "score_home": safe_text(
                    m.get("home_score"),
                    "-"
                ),
                "overs_home": "",
                "score_away": safe_text(
                    m.get("away_score"),
                    "-"
                ),
                "overs_away": "",
                "status": status_raw,
                "api_source": source,
                "win_probability": None,
                "projected_score": None,
                "pre_match_winner": None
            })

    return matches


# =========================================================
# MATCH DEDUPLICATION
# =========================================================

def deduplicate_matches(matches):

    seen = set()
    unique_matches = []

    for match in matches:

        home = safe_text(match.get("home")).lower()
        away = safe_text(match.get("away")).lower()

        if not home or not away:
            continue

        # Normalize team order for duplicate detection
        teams_key = tuple(sorted([home, away]))

        start_time = safe_text(
            match.get("start_time_iso")
        )

        # Use date if full timestamp is unavailable
        date_key = start_time[:10] if start_time else ""

        key = (teams_key, date_key)

        if key in seen:
            continue

        seen.add(key)
        unique_matches.append(match)

    return unique_matches


# =========================================================
# MAIN
# =========================================================

def main():

    apis = [
        ("Highlightly", fetch_highlightly),
        ("CricketData", fetch_cricketdata),
        ("Matchstat", fetch_matchstat),
        ("Sportscore", fetch_sportscore)
    ]

    all_matches = []
    successful_sources = []

    # Fetch from all available APIs instead of stopping
    # after the first API returns matches.
    for api_name, fetch_func in apis:

        print(f"Trying {api_name}...")

        try:
            data, source = fetch_func()

            if not data or not source:
                print(f"No data from {api_name}.")
                continue

            matches = standardize_data(data, source)

            if matches:
                all_matches.extend(matches)

                if source not in successful_sources:
                    successful_sources.append(source)

                print(
                    f"{len(matches)} matches received from {api_name}."
                )

            else:
                print(f"No matches found from {api_name}.")

        except Exception as e:
            print(f"{api_name} processing error: {e}")

    # Remove duplicates
    final_matches = deduplicate_matches(all_matches)

    # Sort by start time
    final_matches.sort(
        key=lambda x: x.get("start_time_iso") or ""
    )

    used_api = (
        ", ".join(successful_sources)
        if successful_sources
        else "None"
    )

    output = {
        "last_updated": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "total_matches": len(final_matches),
        "data_source_api": used_api,
        "matches": final_matches
    }

    # Write safely to avoid leaving a partially written JSON file
    temp_file = "live_matches.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                output,
                f,
                indent=4,
                ensure_ascii=False
            )

        os.replace(temp_file, "live_matches.json")

        print(
            f"Saved {len(final_matches)} matches. "
            f"Sources: {used_api}"
        )

    except Exception as e:
        print(f"JSON save error: {e}")

        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()
