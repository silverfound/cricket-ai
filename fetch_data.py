import os
import requests
import json
from datetime import datetime
import re

# GitHub Secrets se saari API Keys uthana
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_BIGBALLS = os.environ.get("TERTIARY_API_KEY")
KEY_SPORTSCORE = os.environ.get("SPORTSCORE_API_KEY") 
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")

# --- 1. Matchstat API ---
def fetch_matchstat():
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live"
    headers = {"x-rapidapi-key": KEY_MATCHSTAT, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Matchstat"
    except: pass
    return None, None

# --- 2. Highlightly API ---
def fetch_highlightly():
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Highlightly"
    except: pass
    return None, None

# --- 3. CricketData API (Official) ---
def fetch_cricketdata():
    if not KEY_CRICKETDATA: return None, None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0" 
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "CricketData"
    except: pass
    return None, None

# --- 4. Big Balls API (Official) ---
def fetch_bigballs():
    if not KEY_BIGBALLS: return None, None
    # BigBalls ka exact endpoint unki docs par depend karta hai, yeh standard hai
    url = "https://api.bigballsdata.com/v1/cricket/matches" 
    headers = {"Authorization": f"Bearer {KEY_BIGBALLS}"} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "BigBalls"
    except: pass
    return None, None

# --- 5. Sportscore API (Official) ---
def fetch_sportscore():
    if not KEY_SPORTSCORE: return None, None
    # ⚠️ Sportscore ka exact URL yahan daalein, filhal dummy URL hai
    url = "https://api.sportscore.com/v1/matches/live" 
    headers = {"X-API-Key": KEY_SPORTSCORE} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Sportscore"
    except: pass
    return None, None

# --- Helper: Overs nikalne ke liye ---
def extract_overs(info_str):
    if not info_str: return ""
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

# --- Data Formatting (Sabko ek jaisa banana) ---
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_desc = m.get("state", {}).get("description", "Unknown")
            match_state = "live" if "inplay" in status_desc.lower() else ("upcoming" if "scheduled" in status_desc.lower() else "finished")
            h_info = m.get("state", {}).get("teams", {}).get("home", {})
            a_info = m.get("state", {}).get("teams", {}).get("away", {})
            matches.append({
                "league": m.get("league", {}).get("name", "League"), "format": m.get("format", "T20"),
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": h_info.get("score", "-"), "overs_home": extract_overs(h_info.get("info", "")),
                "score_away": a_info.get("score", "-"), "overs_away": extract_overs(a_info.get("info", "")),
                "status": status_desc, "api_source": source
            })
            
    elif source == "CricketData" and "data" in data:
        for m in data["data"]:
            status_desc = m.get("status", "Unknown")
            match_state = "live" if m.get("matchStarted") and not m.get("matchEnded") else ("finished" if m.get("matchEnded") else "upcoming")
            matches.append({
                "league": m.get("name", "League"), "format": m.get("matchType", "T20"),
                "start_time_iso": m.get("dateTimeGMT", ""), "match_state": match_state,
                "home": m.get("teams", ["Home", "Away"])[0], "away": m.get("teams", ["Home", "Away"])[1],
                "score_home": "-", "overs_home": "", "score_away": "-", "overs_away": "",
                "status": status_desc, "api_source": source
            })
            
    elif source == "Matchstat" and "results" in data:
        for m in data["results"]:
            matches.append({
                "league": m.get("tournament", {}).get("name", "League"), "format": "T20",
                "start_time_iso": "", "match_state": "live",
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("homeScore", {}).get("display", "-"), "overs_home": "",
                "score_away": m.get("awayScore", {}).get("display", "-"), "overs_away": "",
                "status": m.get("status", {}).get("description", "In Play"), "api_source": source
            })
            
    return matches

def main():
    # APIs ki priority (Pehle sabse best, phir backup)
    apis = [
        ("Highlightly", fetch_highlightly),
        ("CricketData", fetch_cricketdata),
        ("Matchstat", fetch_matchstat),
        ("BigBalls", fetch_bigballs),
        ("Sportscore", fetch_sportscore)
    ]
    
    final_matches = []
    used_api = "None"
    
    for api_name, fetch_func in apis:
        print(f"Trying {api_name}...")
        data, source = fetch_func()
        if data:
            matches = standardize_data(data, source)
            if len(matches) > 0:
                final_matches = matches
                used_api = source
                print(f"✅ Success with {api_name}! Found {len(matches)} matches.")
                break 
            else:
                print(f"⚠️ {api_name} returned empty data.")
        else:
            print(f"❌ {api_name} failed, trying next...")

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "total_matches": len(final_matches), 
        "data_source_api": used_api, 
        "matches": final_matches
    }
    
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"💾 Saved! Data came from: {used_api}")

if __name__ == "__main__":
    main()
