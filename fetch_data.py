import os
import requests
import json
from datetime import datetime
import re
import joblib
import pandas as pd

# Saari API Keys
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_BIGBALLS = os.environ.get("TERTIARY_API_KEY")
KEY_SPORTSCORE = os.environ.get("SPORTSCORE_API_KEY") 
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")

# =====================================================================
# 🔒 PURANA SYSTEM (OVERS NIKALNE KA FUNCTION) - ISME KOI CHANGE NAHI HAI
# =====================================================================
def extract_overs(info_str):
    if not info_str: return ""
    if '/' in info_str:
        part = info_str.split('/')[0].strip()
        if part.replace('.', '', 1).isdigit():
            return f"{part} ov"
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

# =====================================================================
# 🔒 PURANA SYSTEM (SAARI 5 APIs) - ISME KOI CHANGE NAHI HAI
# =====================================================================
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

# =====================================================================
# 🤖 NAYA SYSTEM: AI WIN PROBABILITY (ADD-ON)
# =====================================================================
try:
    model = joblib.load('win_prob_model.pkl')
    print("✅ AI Model loaded successfully!")
except Exception as e:
    print(f"⚠️ AI Model load failed (Live scores will still work): {e}")
    model = None

def calculate_win_prob(home_info, away_info, match_format):
    if not model: return None
    try:
        target = None
        chasing_score = 0
        chasing_wickets = 0
        chasing_overs = 0.0
        total_overs = 20 if match_format.upper() == "T20" else (50 if match_format.upper() == "ODI" else 100)

        h_score = home_info.get("score", "")
        h_info = home_info.get("info", "")
        a_score = away_info.get("score", "")
        a_info = away_info.get("info", "")

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

        if not target: return None # Agar 1st innings chal rahi hai

        runs_needed = target - chasing_score
        wickets_left = 10 - chasing_wickets
        balls_bowled = int(chasing_overs) * 6 + int((chasing_overs % 1) * 10)
        balls_remaining = (total_overs * 6) - balls_bowled
        
        if balls_remaining <= 0 or runs_needed <= 0: return None
            
        current_run_rate = (chasing_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
        required_run_rate = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0

        input_data = pd.DataFrame({
            'runs_needed': [runs_needed], 'balls_remaining': [balls_remaining],
            'wickets_left': [wickets_left], 'current_run_rate': [current_run_rate],
            'required_run_rate': [required_run_rate]
        })
        
        prob = model.predict_proba(input_data)[0]
        return round(prob[1] * 100, 1)
    except Exception:
        return None # Agar AI fail ho, toh None return karega aur app crash nahi hogi

# =====================================================================
# 🔒 PURANA SYSTEM (DATA FORMATTER) - ISME BAS AI KA 1 LINE ADD HUA HAI
# =====================================================================
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("state", {}).get("description", "")
            status_lower = status_raw.lower()
            
            if "scheduled" in status_lower or "not started" in status_lower:
                match_state = "upcoming"
            elif "finished" in status_lower or "won by" in status_lower or "abandoned" in status_lower or "result" in status_lower:
                match_state = "finished"
            else:
                match_state = "live"

            home_info = m.get("state", {}).get("teams", {}).get("home", {})
            away_info = m.get("state", {}).get("teams", {}).get("away", {})
            fmt = m.get("format", "T20")

            # 🤖 AI ADD-ON: Sirf Live match mein AI calculate hoga
            win_prob = calculate_win_prob(home_info, away_info, fmt) if match_state == "live" else None

            matches.append({
                "league": m.get("league", {}).get("name", "League"), "format": fmt,
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": home_info.get("score", "-"), "overs_home": extract_overs(home_info.get("info", "")),
                "score_away": away_info.get("score", "-"), "overs_away": extract_overs(away_info.get("info", "")),
                "status": status_raw, "api_source": source, 
                "win_probability": win_prob # Naya field
            })
            
    elif source == "Matchstat" and "results" in data:
        for m in data["results"]:
            matches.append({
                "league": m.get("tournament", {}).get("name", "League"), "format": "T20",
                "start_time_iso": "", "match_state": "live",
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("homeScore", {}).get("display", "-"), "overs_home": "",
                "score_away": m.get("awayScore", {}).get("display", "-"), "overs_away": "",
                "status": m.get("status", {}).get("description", "In Play"), "api_source": source, "win_probability": None
            })
            
    elif source == "CricketData" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("status", "Unknown")
            if m.get("matchEnded"): match_state = "finished"
            elif m.get("matchStarted"): match_state = "live"
            else: match_state = "upcoming"
            
            matches.append({
                "league": m.get("name", "League"), "format": m.get("matchType", "T20"),
                "start_time_iso": m.get("dateTimeGMT", ""), "match_state": match_state,
                "home": m.get("teams", ["Home", "Away"])[0], "away": m.get("teams", ["Home", "Away"])[1],
                "score_home": "-", "overs_home": "", "score_away": "-", "overs_away": "",
                "status": status_raw, "api_source": source, "win_probability": None
            })
    return matches

# =====================================================================
# 🔒 PURANA SYSTEM (MAIN EXECUTION) - ISME KOI CHANGE NAHI HAI
# =====================================================================
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
