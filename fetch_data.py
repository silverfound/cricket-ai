import os
import requests
import json
from datetime import datetime
import re
import joblib
import pandas as pd

# API Keys
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_BIGBALLS = os.environ.get("TERTIARY_API_KEY")
KEY_SPORTSCORE = os.environ.get("SPORTSCORE_API_KEY") 
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")

# 🧠 DONO AI MODELS AUR ENCODERS LOAD KARNA
try:
    win_model = joblib.load('win_prob_model.pkl')
    score_model = joblib.load('score_pred_model_v3.pkl')
    encoders = joblib.load('team_league_encoders.pkl') # Naya Encoders load
    print("✅ V3 AI Models and Encoders loaded successfully!")
except Exception as e:
    print(f"⚠️ Model load error: {e}")
    win_model, score_model, encoders = None, None, None

# 🔥 OVERS NIKALNE KA FUNCTION
def extract_overs(info_str):
    if not info_str: return ""
    if '/' in info_str:
        part = info_str.split('/')[0].strip()
        if part.replace('.', '', 1).isdigit():
            return f"{part} ov"
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

# 🤖 AI PREDICTION LOGIC (V3 - Context Aware)
def get_ai_predictions(home_info, away_info, match_format, league_name, home_team, away_team):
    if not win_model or not score_model or not encoders: return None, None
    
    try:
        total_overs = 20 if match_format.upper() == "T20" else (50 if match_format.upper() == "ODI" else 100)
        h_score = home_info.get("score", "")
        h_info = home_info.get("info", "")
        a_score = away_info.get("score", "")
        a_info = away_info.get("info", "")

        target = None
        chasing_score = 0
        chasing_wickets = 0
        chasing_overs = 0.0

        # Target check karna
        if "T:" in h_info:
            target = int(re.search(r'T:(\d+)', h_info).group(1))
            if a_score and "/" in a_score:
                chasing_score = int(a_score.split("/")[0])
                chasing_wickets = int(a_score.split("/")[1])
            if "ov" in a_info:
                chasing_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', a_info).group(1))
        elif "T:" in a_info:
            target = int(re.search(r'T:(\d+)', a_info).group(1))
            if h_score and "/" in h_score:
                chasing_score = int(h_score.split("/")[0])
                chasing_wickets = int(h_score.split("/")[1])
            if "ov" in h_info:
                chasing_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', h_info).group(1))

        # ==========================================
        # 2nd Innings: Win Probability (Purana Model)
        # ==========================================
        if target:
            runs_needed = target - chasing_score
            wickets_left = 10 - chasing_wickets
            balls_bowled = int(chasing_overs) * 6 + int((chasing_overs % 1) * 10)
            balls_remaining = (total_overs * 6) - balls_bowled
            
            if balls_remaining > 0 and runs_needed > 0:
                crr = (chasing_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
                rrr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
                
                input_data = pd.DataFrame({
                    'runs_needed': [runs_needed], 'balls_remaining': [balls_remaining],
                    'wickets_left': [wickets_left], 'current_run_rate': [crr],
                    'required_run_rate': [rrr]
                })
                prob = win_model.predict_proba(input_data)[0][1] * 100
                return round(prob, 1), None
        
        # ==========================================
        # 1st Innings: Projected Score (Naya V3 Context Model)
        # ==========================================
        else:
            batting_score, batting_wickets, batting_overs = 0, 0, 0.0
            batting_team, bowling_team = "", ""
            
            # Kaun batting kar raha hai pata karna
            if h_score and not a_score:
                batting_score = int(h_score.split("/")[0]) if "/" in h_score else 0
                batting_wickets = int(h_score.split("/")[1]) if "/" in h_score else 0
                batting_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', h_info).group(1)) if "ov" in h_info else 0
                batting_team, bowling_team = home_team, away_team
            elif a_score and not h_score:
                batting_score = int(a_score.split("/")[0]) if "/" in a_score else 0
                batting_wickets = int(a_score.split("/")[1]) if "/" in a_score else 0
                batting_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', a_info).group(1)) if "ov" in a_info else 0
                batting_team, bowling_team = away_team, home_team
            else:
                return None, None
            
            if batting_overs > 0 and batting_score > 0:
                balls_bowled = int(batting_overs) * 6 + int((batting_overs % 1) * 10)
                balls_remaining = (total_overs * 6) - balls_bowled
                crr = batting_score / (balls_bowled / 6) if balls_bowled > 0 else 0
                
                # Encode League and Teams (Agar nayi team/league ho toh error handle karein)
                try:
                    league_id = encoders['league'].transform([league_name])[0]
                except:
                    league_id = -1 # Unknown League
                
                try:
                    bat_team_id = encoders['bat_team'].transform([batting_team])[0]
                except:
                    bat_team_id = -1
                    
                try:
                    bowl_team_id = encoders['bowl_team'].transform([bowling_team])[0]
                except:
                    bowl_team_id = -1

                fmt_code = 0 if "T20" in match_format.upper() else (1 if "ODI" in match_format.upper() else 2)

                # V3 Model Input
                input_data = pd.DataFrame({
                    'league_id': [league_id], 'batting_team_id': [bat_team_id], 
                    'bowling_team_id': [bowl_team_id], 'format': [fmt_code],
                    'current_score': [batting_score], 'balls_remaining': [balls_remaining],
                    'wickets_lost': [batting_wickets], 'current_run_rate': [crr],
                    'last_3_overs_runs': [10] # Approx momentum (live API mein nahi milta, default daal rahe hain)
                })
                
                pred_score = score_model.predict(input_data)[0]
                
                # Range dikhana
                lower_bound = int(pred_score) - 5
                upper_bound = int(pred_score) + 5
                return None, f"{lower_bound} - {upper_bound}"
                
        return None, None
    except Exception as e:
        print(f"AI Error: {e}")
        return None, None

# --- APIs (Wahi purane functions) ---
def fetch_highlightly():
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Highlightly"
    except: pass
    return None, None

def fetch_matchstat():
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live"
    headers = {"x-rapidapi-key": KEY_MATCHSTAT, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Matchstat"
    except: pass
    return None, None

def fetch_cricketdata():
    if not KEY_CRICKETDATA: return None, None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0" 
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "CricketData"
    except: pass
    return None, None

def fetch_bigballs():
    if not KEY_BIGBALLS: return None, None
    url = "https://api.bigballsdata.com/v1/cricket/matches" 
    headers = {"Authorization": f"Bearer {KEY_BIGBALLS}"} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "BigBalls"
    except: pass
    return None, None

def fetch_sportscore():
    if not KEY_SPORTSCORE: return None, None
    url = "https://api.sportscore.com/v1/matches/live" 
    headers = {"X-API-Key": KEY_SPORTSCORE} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Sportscore"
    except: pass
    return None, None

# --- Formatter ---
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("state", {}).get("description", "")
            status_lower = status_raw.lower()
            
            if "no live coverage" in status_lower: continue

            if "scheduled" in status_lower or "not started" in status_lower:
                match_state = "upcoming"
            elif "finished" in status_lower or "won by" in status_lower or "abandoned" in status_lower or "result" in status_lower:
                match_state = "finished"
            else:
                match_state = "live"

            home_info = m.get("state", {}).get("teams", {}).get("home", {})
            away_info = m.get("state", {}).get("teams", {}).get("away", {})
            fmt = m.get("format", "T20")
            league = m.get("league", {}).get("name", "League")
            home_team = m.get("homeTeam", {}).get("name", "Home")
            away_team = m.get("awayTeam", {}).get("name", "Away")

            win_prob, proj_score = None, None
            if match_state == "live":
                win_prob, proj_score = get_ai_predictions(home_info, away_info, fmt, league, home_team, away_team)

            matches.append({
                "league": league, "format": fmt,
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": home_team, "away": away_team,
                "score_home": home_info.get("score", "-"), "overs_home": extract_overs(home_info.get("info", "")),
                "score_away": away_info.get("score", "-"), "overs_away": extract_overs(away_info.get("info", "")),
                "status": status_raw, "api_source": source, 
                "win_probability": win_prob, "projected_score": proj_score
            })
            
    # (Baaki APIs ke liye purana logic chalega, filhal maine sirf Highlightly update kiya hai taaki test ho sake)
    return matches

def main():
    apis = [
        ("Highlightly", fetch_highlightly), ("CricketData", fetch_cricketdata), 
        ("Matchstat", fetch_matchstat), ("BigBalls", fetch_bigballs), ("Sportscore", fetch_sportscore)
    ]
    final_matches, used_api = [], "None"
    
    for api_name, fetch_func in apis:
        print(f"Trying {api_name}...")
        data, source = fetch_func()
        if data:
            matches = standardize_data(data, source)
            if len(matches) > 0:
                final_matches, used_api = matches, source
                print(f"✅ Success with {api_name}!")
                break 

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "total_matches": len(final_matches), "data_source_api": used_api, "matches": final_matches
    }
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"💾 Saved! Source: {used_api}")

if __name__ == "__main__":
    main()
