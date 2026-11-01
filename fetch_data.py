import os
import requests
import json
from datetime import datetime

# GitHub Secrets se teeno keys uthana
PRIMARY_KEY = os.environ.get("RAPIDAPI_KEY")
SECONDARY_KEY = os.environ.get("SECONDARY_API_KEY")
TERTIARY_KEY = os.environ.get("TERTIARY_API_KEY")

def fetch_primary():
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live" # Primary URL
    headers = {"x-rapidapi-key": PRIMARY_KEY, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "primary"
    except: pass
    return None, None

def fetch_secondary():
    url = "https://cricket-highlights-api.p.rapidapi.com/matches" # Highlightly URL
    headers = {"x-rapidapi-key": SECONDARY_KEY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "secondary"
    except: pass
    return None, None

def fetch_tertiary():
    # Big Balls API (Aapko unki docs se exact URL lena hoga, abhi ke liye maine standard daala hai)
    url = "https://api.bigballsdata.com/v1/cricket/matches" 
    headers = {"Authorization": f"Bearer {TERTIARY_KEY}"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "tertiary"
    except: pass
    return None, None

def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    # Primary API (Matchstat)
    if source == "primary" and "results" in data:
        for m in data["results"]:
            matches.append({"league": m.get("tournament", {}).get("name", ""), "home": m.get("homeTeam", {}).get("name", ""), "away": m.get("awayTeam", {}).get("name", ""), "score_home": m.get("homeScore", {}).get("display", ""), "score_away": m.get("awayScore", {}).get("display", ""), "status": m.get("status", {}).get("description", "")})
            
    # Secondary API (Highlightly)
    elif source == "secondary" and "data" in data:
        for m in data["data"]:
            matches.append({"league": m.get("league", {}).get("name", ""), "home": m.get("homeTeam", {}).get("name", ""), "away": m.get("awayTeam", {}).get("name", ""), "score_home": m.get("state", {}).get("teams", {}).get("home", {}).get("score", ""), "score_away": m.get("state", {}).get("teams", {}).get("away", {}).get("score", ""), "status": m.get("state", {}).get("description", "")})
            
    return matches

def main():
    # 1. Primary try karein
    data, source = fetch_primary()
    
    # 2. Agar Primary fail ho, toh Secondary try karein
    if not data:
        print("Primary fail, trying Secondary...")
        data, source = fetch_secondary()
        
    # 3. Agar Secondary bhi fail ho, toh Tertiary try karein
    if not data:
        print("Secondary fail, trying Tertiary...")
        data, source = fetch_tertiary()

    matches = standardize_data(data, source)
    
    output = {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "total_matches": len(matches), "matches": matches}
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"✅ Success! {len(matches)} matches saved.")

if __name__ == "__main__":
    main()
