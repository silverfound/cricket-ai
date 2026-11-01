import os
import requests
import json
from datetime import datetime

# GitHub Secrets se keys uthana
PRIMARY_KEY = os.environ.get("RAPIDAPI_KEY")
SECONDARY_KEY = os.environ.get("SECONDARY_API_KEY")
TERTIARY_KEY = os.environ.get("TERTIARY_API_KEY")

def fetch_highlightly(): # Abhi ke liye isko primary bana rahe hain
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": SECONDARY_KEY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "highlightly"
    except Exception as e: print(f"Highlightly error: {e}")
    return None, None

def fetch_matchstat():
    # Matchstat ka sahi URL abhi confirm nahi hai, par fallback ke liye rakh rahe hain
    url = "https://cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com/v1/live" 
    headers = {"x-rapidapi-key": PRIMARY_KEY, "x-rapidapi-host": "cricket-api-data-live-scores-stats-odds-predictions.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "matchstat"
    except: pass
    return None, None

def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "highlightly" and "data" in data:
        for m in data["data"]:
            # Sirf woh matches dikhayein jinka score available hai
            matches.append({
                "league": m.get("league", {}).get("name", "Unknown League"),
                "home": m.get("homeTeam", {}).get("name", "Home Team"),
                "away": m.get("awayTeam", {}).get("name", "Away Team"),
                "score_home": m.get("state", {}).get("teams", {}).get("home", {}).get("score", "-"),
                "score_away": m.get("state", {}).get("teams", {}).get("away", {}).get("score", "-"),
                "status": m.get("state", {}).get("description", "Match Status")
            })
            
    elif source == "matchstat" and "results" in data:
        for m in data["results"]:
            matches.append({
                "league": m.get("tournament", {}).get("name", "Unknown League"),
                "home": m.get("homeTeam", {}).get("name", "Home Team"),
                "away": m.get("awayTeam", {}).get("name", "Away Team"),
                "score_home": m.get("homeScore", {}).get("display", "-"),
                "score_away": m.get("awayScore", {}).get("display", "-"),
                "status": m.get("status", {}).get("description", "Match Status")
            })
    return matches

def main():
    # 1. Pehle Highlightly try karein (Aaj ki date ke sath)
    data, source = fetch_highlightly()
    
    # 2. Agar wahan nahi mila, toh Matchstat try karein
    if not data:
        print("Highlightly fail, trying Matchstat...")
        data, source = fetch_matchstat()

    matches = standardize_data(data, source)
    
    output = {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "total_matches": len(matches), "matches": matches}
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"✅ Success! {len(matches)} matches saved.")

if __name__ == "__main__":
    main()
