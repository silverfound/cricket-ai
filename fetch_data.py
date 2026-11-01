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

# ==========================================
# 🔥 OVERS NIKALNE KA SMART FUNCTION (FIXED)
# ==========================================
def extract_overs(info_str):
    if not info_str: return ""
    # "1.2/20 ov, T:138" -> Hum "1.2 ov" nikalenge
    if '/' in info_str:
        part = info_str.split('/')[0].strip()
        if part.replace('.', '', 1).isdigit():
            return f"{part} ov"
    # Agar format alag ho
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

# ==========================================
# 🔌 1. HIGHLIGHTLY API
# ==========================================
def fetch_highlightly():
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Highlightly"
    except: pass
    return None, None

# ==========================================
# 🔌 2. MATCHSTAT API
# ==========================================
def fetch_matchstat():
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live"
    headers = {"x-rapidapi-key": KEY_MATCHSTAT, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Matchstat"
    except: pass
    return None, None

# ==========================================
# 🔌 3. CRICKET DATA API (Official)
# ==========================================
def fetch_cricketdata():
    if not KEY_CRICKETDATA: return None, None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0" 
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "CricketData"
    except: pass
    return None, None

# ==========================================
# 🔌 4. BIG BALLS API (Official)
# ==========================================
def fetch_bigballs():
    if not KEY_BIGBALLS: return None, None
    url = "https://api.bigballsdata.com/v1/cricket/matches" 
    headers = {"Authorization": f"Bearer {KEY_BIGBALLS}"} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "BigBalls"
    except: pass
    return None, None

# ==========================================
# 🔌 5. SPORTSCORE API (Official)
# ==========================================
def fetch_sportscore():
    if not KEY_SPORTSCORE: return None, None
    url = "https://api.sportscore.com/v1/matches/live" 
    headers = {"X-API-Key": KEY_SPORTSCORE} 
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Sportscore"
    except: pass
    return None, None

# ==========================================
# 🧠 DATA FORMATTER (Sabhi APIs ka data ek format mein)
# ==========================================
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("state", {}).get("description", "")
            status_lower = status_raw.lower()
            
            # Smart State Logic
            if "scheduled" in status_lower or "not started" in status_lower:
                match_state = "upcoming"
            elif "finished" in status_lower or "won by" in status_lower or "abandoned" in status_lower or "result" in status_lower:
                match_state = "finished"
            else:
                match_state = "live"

            home_info = m.get("state", {}).get("teams", {}).get("home", {})
            away_info = m.get("state", {}).get("teams", {}).get("away", {})

            matches.append({
                "league": m.get("league", {}).get("name", "League"), "format": m.get("format", "T20"),
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": home_info.get("score", "-"), "overs_home": extract_overs(home_info.get("info", "")),
                "score_away": away_info.get("score", "-"), "overs_away": extract_overs(away_info.get("info", "")),
                "status": status_raw, "api_source": source
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
                "status": status_raw, "api_source": source
            })
            
    return matches

# ==========================================
# 🚀 MAIN EXECUTION (Smart Fallback)
# ==========================================
def main():
    # APIs ki priority (Agar pehli fail hogi toh dusri try hogi)
    apis = [
        ("Highlightly", fetch_highlightly),
        ("CricketData", fetch_cricketdata),
        ("Matchstat", fetch_matchstat),
        ("BigBalls", fetch_bigballs),
        ("Sportscore", fetch_sportscore)
    ]
    
    final_matches, used_api = [], "None"
    
    for api_name, fetch_func in apis:
        print(f"Trying {api_name}...")
        data, source = fetch_func()
        if data:
            matches = standardize_data(data, source)
            if len(matches) > 0:
                final_matches, used_api = matches, source
                print(f"✅ Success with {api_name}! Found {len(matches)} matches.")
                break 
            else:
                print(f"⚠️ {api_name} returned empty data.")
        else:
            print(f"❌ {api_name} failed or limit reached, trying next...")

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
