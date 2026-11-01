import os
import requests
import json
from datetime import datetime
import re

# API Keys
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_BIGBALLS = os.environ.get("TERTIARY_API_KEY")
KEY_SPORTSCORE = os.environ.get("SPORTSCORE_API_KEY") 
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")

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

def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("state", {}).get("description", "")
            status_lower = status_raw.lower()
            
            # 🎯 SMART LOGIC: Koi confusion nahi
            if "scheduled" in status_lower or "not started" in status_lower:
                match_state = "upcoming"
            elif "finished" in status_lower or "won by" in status_lower or "abandoned" in status_lower or "result" in status_lower:
                match_state = "finished"
            else:
                match_state = "live" # Baaki sab (In play, 1st inning) Live hain

            matches.append({
                "league": m.get("league", {}).get("name", "League"), "format": m.get("format", "T20"),
                "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("state", {}).get("teams", {}).get("home", {}).get("score", "-"),
                "score_away": m.get("state", {}).get("teams", {}).get("away", {}).get("score", "-"),
                "status": status_raw, "api_source": source
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
                "score_home": "-", "score_away": "-", "status": status_raw, "api_source": source
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
            
    return matches

def main():
    apis = [("Highlightly", fetch_highlightly), ("CricketData", fetch_cricketdata), ("Matchstat", fetch_matchstat)]
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
        print(f"❌ {api_name} failed.")

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "total_matches": len(final_matches), 
        "data_source_api": used_api, 
        "matches": final_matches
    }
    
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print("💾 File Saved.")

if __name__ == "__main__":
    main()
