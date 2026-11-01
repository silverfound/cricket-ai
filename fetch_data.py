import os
import requests
import json
from datetime import datetime

PRIMARY_KEY = os.environ.get("RAPIDAPI_KEY")
SECONDARY_KEY = os.environ.get("SECONDARY_API_KEY")
TERTIARY_KEY = os.environ.get("TERTIARY_API_KEY")

def fetch_highlightly():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": SECONDARY_KEY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200: return res.json(), "highlightly"
    except Exception as e: print(f"Highlightly error: {e}")
    return None, None

def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "highlightly" and "data" in data:
        for m in data["data"]:
            # Date ko format karna (jaise 2024-01-13T08:30:00 -> 13 Jan, 08:30 AM)
            raw_date = m.get("startTime", "")
            match_date = raw_date.split("T")[0] if raw_date else "Date TBD"
            
            matches.append({
                "league": m.get("league", {}).get("name", "Unknown League"),
                "format": m.get("format", "N/A"),
                "date": match_date,
                "home": m.get("homeTeam", {}).get("name", "Home Team"),
                "away": m.get("awayTeam", {}).get("name", "Away Team"),
                "score_home": m.get("state", {}).get("teams", {}).get("home", {}).get("score", "-"),
                "score_away": m.get("state", {}).get("teams", {}).get("away", {}).get("score", "-"),
                "status": m.get("state", {}).get("description", "Match Status")
            })
    return matches

def main():
    data, source = fetch_highlightly()
    matches = standardize_data(data, source)
    
    output = {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "total_matches": len(matches), "matches": matches}
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"✅ Success! {len(matches)} matches saved.")

if __name__ == "__main__":
    main()
