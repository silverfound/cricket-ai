import os
import requests
import json
from datetime import datetime
import re

KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY")
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")

# Overs nikalne ka helper function (jaise "17.2/20 ov" se "17.2 ov" nikalna)
def extract_overs(info_str):
    if not info_str: return ""
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

def fetch_highlightly():
    url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={today}"
    headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json(), "Highlightly"
    except: pass
    return None, None

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

            # Teams ka info nikalna (jahan overs likhe hote hain)
            home_info = m.get("state", {}).get("teams", {}).get("home", {})
            away_info = m.get("state", {}).get("teams", {}).get("away", {})

            matches.append({
                "league": m.get("league", {}).get("name", "League"), 
                "format": m.get("format", "T20"),
                "start_time_iso": m.get("startTime", ""), 
                "match_state": match_state,
                "home": m.get("homeTeam", {}).get("name", "Home"), 
                "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": home_info.get("score", "-"), 
                "overs_home": extract_overs(home_info.get("info", "")), # Overs yahan se aa rahe hain
                "score_away": away_info.get("score", "-"), 
                "overs_away": extract_overs(away_info.get("info", "")), # Overs yahan se aa rahe hain
                "status": status_raw, 
                "api_source": source
            })
            
    return matches

def main():
    apis = [("Highlightly", fetch_highlightly)]
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
        "total_matches": len(final_matches), 
        "data_source_api": used_api, 
        "matches": final_matches
    }
    
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print("💾 File Saved.")

if __name__ == "__main__":
    main()
