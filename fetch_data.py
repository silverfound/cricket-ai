import os
import requests
import json
from datetime import datetime

# Saari API Keys (GitHub Secrets se)
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_BIGBALLS = os.environ.get("TERTIARY_API_KEY")
KEY_SPORTSCORE = os.environ.get("SPORTSCORE_API_KEY") # Aapke GitHub secret ka naam
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") # Aapke GitHub secret ka naam

today = datetime.now().strftime("%Y-%m-%d")

# --- 1. Matchstat (RapidAPI) ---
def fetch_matchstat():
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live"
    headers = {"x-rapidapi-key": KEY_MATCHSTAT, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Matchstat"
    except: pass
    return None, None

# --- 2. Highlightly (RapidAPI) ---
def fetch_highlightly():
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Highlightly"
    except: pass
    return None, None

# --- 3. Big Balls (Official API) ---
def fetch_bigballs():
    # Big Balls ka endpoint (Aapko Docs se confirm karna hoga, abhi standard use kar rahe hain)
    url = "https://api.bigballsdata.com/v1/cricket/matches" 
    headers = {"Authorization": f"Bearer {KEY_BIGBALLS}"} # Official API usually Bearer token leti hai
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Big Balls"
    except: pass
    return None, None

# --- 4. Sportscore (Official API) ---
def fetch_sportscore():
    # ⚠️ Yahan Sportscore ka sahi URL aur Headers daalein
    url = "https://api.sportscore.com/v1/matches/live" # Dummy URL
    headers = {"X-API-Key": KEY_SPORTSCORE} # Ya jo bhi unka header format ho
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Sportscore"
    except: pass
    return None, None

# --- 5. CricketData (Official API) ---
def fetch_cricketdata():
    # ⚠️ Yahan CricketData ka sahi URL aur Headers daalein
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0" 
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "CricketData"
    except: pass
    return None, None

# --- Data Formatting ---
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_desc = m.get("state", {}).get("description", "Unknown")
            match_state = "live" if "inplay" in status_desc.lower() else ("upcoming" if "scheduled" in status_desc.lower() else "finished")
            matches.append({
                "league": m.get("league", {}).get("name", "League"), "format": m.get("format", "T20"),
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("state", {}).get("teams", {}).get("home", {}).get("score", "-"),
                "score_away": m.get("state", {}).get("teams", {}).get("away", {}).get("score", "-"),
                "status": status_desc, "api_source": source
            })
    elif source == "Matchstat" and "results" in data:
        for m in data["results"]:
            matches.append({
                "league": m.get("tournament", {}).get("name", "League"), "format": "T20",
                "start_time_iso": "", "match_state": "live",
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("homeScore", {}).get("display", "-"), "score_away": m.get("awayScore", {}).get("display", "-"),
                "status": m.get("status", {}).get("description", "In Play"), "api_source": source
            })
    elif source == "CricketData" and "data" in data:
        # CricketData.org ka format
        for m in data["data"]:
            matches.append({
                "league": m.get("name", "League"), "format": m.get("matchType", "T20"),
                "start_time_iso": m.get("dateTimeGMT", ""), 
                "match_state": "live" if m.get("matchStarted") and not m.get("matchEnded") else ("finished" if m.get("matchEnded") else "upcoming"),
                "home": m.get("teams", ["Home", "Away"])[0], "away": m.get("teams", ["Home", "Away"])[1],
                "score_home": "-", "score_away": "-", # Score nikalne ka alag logic hota hai
                "status": m.get("status", "Unknown"), "api_source": source
            })
            
    # (Baaki APIs ke formatting yahan add honge)
    return matches

def main():
    # APIs ki priority (Agar pehli fail hogi toh dusri try hogi)
    apis = [
        ("Matchstat", fetch_matchstat),
        ("Highlightly", fetch_highlightly),
        ("CricketData", fetch_cricketdata),
        ("Big Balls", fetch_bigballs),
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
