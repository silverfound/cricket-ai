import os
import requests
import json
from datetime import datetime

PRIMARY_KEY = os.environ.get("RAPIDAPI_KEY")
SECONDARY_KEY = os.environ.get("SECONDARY_API_KEY")

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
            raw_datetime = m.get("startTime", "")
            
            # Date aur Time alag karna
            if "T" in raw_datetime:
                match_date = raw_datetime.split("T")[0]
                # Time nikalna (jaise 08:30)
                match_time = raw_datetime.split("T")[1][:5] + " UTC" 
            else:
                match_date = "Date TBD"
                match_time = "Time TBD"
            
            matches.append({
                "league": m.get("league", {}).get("name", "Unknown League"),
                "format": m.get("format", "N/A"),
                "date": match_date,
                "time": match_time, # Naya field add kiya
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
    print(f"✅ Success! {len(matches)} matches saved with Date and Time.")

if __name__ == "__main__":
    main()
