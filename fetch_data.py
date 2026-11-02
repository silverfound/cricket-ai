import os
import requests
import json
from datetime import datetime, timedelta
import re
import joblib
import pandas as pd

# API Keys
KEY_MATCHSTAT = os.environ.get("RAPIDAPI_KEY")
KEY_HIGHLIGHTLY = os.environ.get("SECONDARY_API_KEY") 
KEY_CRICKETDATA = os.environ.get("CRICKETDATA_API_KEY") 

today = datetime.now().strftime("%Y-%m-%d")
tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

# 🧠 SAARE MODELS LOAD KARNA
try:
    score_model = joblib.load('score_pred_model_v3.pkl')
    team_league_encoders = joblib.load('team_league_encoders.pkl')
    win_model = joblib.load('win_prob_model_v3.pkl')
    chase_encoders = joblib.load('chase_encoders.pkl')
    pre_match_model = joblib.load('pre_match_model_v3.pkl')
    print("✅ Saare Models load ho gaye!")
except Exception as e:
    print(f"⚠️ Model load error: {e}")
    score_model, team_league_encoders, win_model, chase_encoders, pre_match_model = None, None, None, None, None

# 🔥 OVERS NIKALNE KA FUNCTION
def extract_overs(info_str):
    if not info_str: return ""
    if '/' in info_str:
        part = info_str.split('/')[0].strip()
        if part.replace('.', '', 1).isdigit():
            return f"{part} ov"
    ov_match = re.search(r'(\d+\.\d+|\d+)\s*ov', info_str)
    return ov_match.group(0) if ov_match else ""

# 🛡️ SMART ENCODERS
def safe_encode_league(league_name, encoder):
    if not encoder: return -1
    try:
        league_clean = str(league_name).strip().lower()
        for cls in encoder.classes_:
            if str(cls).strip().lower() == league_clean:
                return encoder.transform([cls])[0]
        return -1
    except: return -1

def safe_encode_team(team_name, encoder):
    if not encoder: return -1
    try:
        team_name_clean = str(team_name).strip().lower()
        # Exact match
        for cls in encoder.classes_:
            if str(cls).strip().lower() == team_name_clean:
                return encoder.transform([cls])[0]
        # Bracket aur Women hata kar match
        clean_api = re.sub(r'\(.*?\)', '', team_name_clean).replace("women", "").strip()
        is_women_api = "women" in team_name_clean
        
        for cls in encoder.classes_:
            cls_clean = str(cls).strip().lower()
            clean_enc = re.sub(r'\(.*?\)', '', cls_clean).replace("women", "").strip()
            is_women_encoder = "women" in cls_clean
            
            if is_women_api == is_women_encoder and (clean_api == clean_enc or clean_api in clean_enc or clean_enc in clean_api):
                return encoder.transform([cls])[0]
        return -1
    except: return -1

# 🤖 PRE-MATCH AI PREDICTION
def get_pre_match_prediction(home_team, away_team, match_format, league_name):
    if not pre_match_model or not team_league_encoders: return None
    try:
        t1 = safe_encode_team(home_team, team_league_encoders['team'])
        t2 = safe_encode_team(away_team, team_league_encoders['team'])
        l_id = safe_encode_league(league_name, team_league_encoders['league'])

        if t1 == -1 or t2 == -1: return None

        # 🔥 FIX: Model ko 6 features chahiye (Form aur H2H 50% default de rahe hain)
input_data = pd.DataFrame({
    'league_id': [l_id], 'team1_id': [t1], 'team2_id': [t2],
    't1_form': [0.5], 't2_form': [0.5], 'h2h_win_rate': [0.5]
})
        prob = pre_match_model.predict_proba(input_data)[0]
        
        home_prob = prob[1] * 100
        away_prob = prob[0] * 100
        
        if home_prob > away_prob: return f"{home_team} ({home_prob:.1f}%)"
        else: return f"{away_team} ({away_prob:.1f}%)"
    except: return None

# 🤖 LIVE MATCH AI PREDICTION (V3)
def get_ai_predictions(home_info, away_info, match_format, league_name, home_team, away_team):
    if not all([score_model, team_league_encoders, win_model, chase_encoders]): return None, None
    try:
        total_overs = 20 if match_format.upper() == "T20" else (50 if match_format.upper() == "ODI" else 100)
        h_score = home_info.get("score", "")
        h_info = home_info.get("info", "")
        a_score = away_info.get("score", "")
        a_info = away_info.get("info", "")

        target = None
        chasing_score, chasing_wickets, chasing_overs = 0, 0, 0.0
        chasing_team, defending_team = "", ""

        if "T:" in h_info:
            target = int(re.search(r'T:(\d+)', h_info).group(1))
            chasing_team, defending_team = away_team, home_team
            if a_score and "/" in a_score:
                chasing_score = int(a_score.split("/")[0])
                chasing_wickets = int(a_score.split("/")[1])
            if "ov" in a_info:
                chasing_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', a_info).group(1))
        elif "T:" in a_info:
            target = int(re.search(r'T:(\d+)', a_info).group(1))
            chasing_team, defending_team = home_team, away_team
            if h_score and "/" in h_score:
                chasing_score = int(h_score.split("/")[0])
                chasing_wickets = int(h_score.split("/")[1])
            if "ov" in h_info:
                chasing_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', h_info).group(1))

        if target:
            runs_needed = target - chasing_score
            wickets_left = 10 - chasing_wickets
            balls_bowled = int(chasing_overs) * 6 + int((chasing_overs % 1) * 10)
            balls_remaining = (total_overs * 6) - balls_bowled
            if balls_remaining > 0 and runs_needed > 0:
                crr = (chasing_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
                rrr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
                
                l_id = safe_encode_league(league_name, chase_encoders['league'])
                c_id = safe_encode_team(chasing_team, chase_encoders['chasing_team'])
                d_id = safe_encode_team(defending_team, chase_encoders['defending_team'])

                input_data = pd.DataFrame({
                    'league_id': [l_id], 'chasing_team_id': [c_id], 'defending_team_id': [d_id],
                    'runs_needed': [runs_needed], 'balls_remaining': [balls_remaining],
                    'wickets_left': [wickets_left], 'current_run_rate': [crr],
                    'required_run_rate': [rrr]
                })
                prob = win_model.predict_proba(input_data)[0][1] * 100
                return round(prob, 1), None
        else:
            batting_score, batting_wickets, batting_overs = 0, 0, 0.0
            batting_team, bowling_team = "", ""
            if h_score and not a_score:
                batting_score = int(h_score.split("/")[0]) if "/" in h_score else 0
                batting_wickets = int(h_score.split("/")[1]) if "/" in h_score else 0
                batting_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', h_info).group(1)) if "ov" in h_info else 0
                batting_team, bowling_team = home_team, away_team
            elif a_score and not h_score:
                batting_score = int(a_score.split("/")[0]) if "/" in a_score else 0
                batting_wickets = int(a_score.split("/")[1]) if "/" in a_score else 0
                batting_overs = float(re.search(r'(\d+\.\d+|\d+)\s*ov', a_info).group(1)) if "ov" in a_info else 0
                batting_team, bowling_team = away_team, home_team
            else: return None, None
            
            if batting_overs > 0 and batting_score > 0:
                balls_bowled = int(batting_overs) * 6 + int((batting_overs % 1) * 10)
                balls_remaining = (total_overs * 6) - balls_bowled
                crr = batting_score / (balls_bowled / 6) if balls_bowled > 0 else 0
                
                l_id = safe_encode_league(league_name, team_league_encoders['league'])
                b_id = safe_encode_team(batting_team, team_league_encoders['team'])
                bw_id = safe_encode_team(bowling_team, team_league_encoders['team'])
                fmt_code = 0 if "T20" in match_format.upper() else (1 if "ODI" in match_format.upper() else 2)

                input_data = pd.DataFrame({
                    'league_id': [l_id], 'batting_team_id': [b_id], 'bowling_team_id': [bw_id], 
                    'format': [fmt_code], 'current_score': [batting_score], 
                    'balls_remaining': [balls_remaining], 'wickets_lost': [batting_wickets], 
                    'current_run_rate': [crr], 'last_3_overs_runs': [10]
                })
                pred_score = score_model.predict(input_data)[0]
                return None, f"{int(pred_score)-5} - {int(pred_score)+5}"
        return None, None
    except: return None, None

# --- APIs FUNCTIONS ---
def fetch_highlightly():
    all_data = []
    for d in [today, tomorrow]:
        url = f"https://cricket-highlights-api.p.rapidapi.com/matches?date={d}"
        headers = {"x-rapidapi-key": KEY_HIGHLIGHTLY, "x-rapidapi-host": "cricket-highlights-api.p.rapidapi.com"}
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "data" in data: all_data.extend(data["data"])
        except: pass
    return {"data": all_data}, "Highlightly"

def fetch_cricketdata():
    if not KEY_CRICKETDATA: return None, None
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={KEY_CRICKETDATA}&offset=0" 
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "CricketData"
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

def fetch_sportscore():
    url = "https://sportscore.com/api/widget/matches/?sport=cricket&limit=50"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json(), "Sportscore"
    except: pass
    return None, None

# --- Formatter ---
def standardize_data(data, source):
    matches = []
    if not data: return matches
    
    if source == "Highlightly" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("state", {}).get("description", "")
            status_lower = status_raw.lower()
            if "no live coverage" in status_lower: continue
            if "scheduled" in status_lower or "not started" in status_lower: match_state = "upcoming"
            elif "finished" in status_lower or "won by" in status_lower or "abandoned" in status_lower or "result" in status_lower: match_state = "finished"
            else: match_state = "live"

            home_info = m.get("state", {}).get("teams", {}).get("home", {})
            away_info = m.get("state", {}).get("teams", {}).get("away", {})
            fmt = m.get("format", "T20")
            league = m.get("league", {}).get("name", "League")
            home_team = m.get("homeTeam", {}).get("name", "Home")
            away_team = m.get("awayTeam", {}).get("name", "Away")

            win_prob, proj_score, pre_match_winner = None, None, None
            if match_state == "live" and any(x in fmt.upper() for x in ['T20', 'ODI', 'HUNDRED']):
                win_prob, proj_score = get_ai_predictions(home_info, away_info, fmt, league, home_team, away_team)
            elif match_state == "upcoming" and any(x in fmt.upper() for x in ['T20', 'ODI', 'HUNDRED']):
                pre_match_winner = get_pre_match_prediction(home_team, away_team, fmt, league)

            matches.append({
                "league": league, "format": fmt, "start_time_iso": m.get("startTime", ""), "match_state": match_state,
                "home": home_team, "away": away_team,
                "score_home": home_info.get("score", "-"), "overs_home": extract_overs(home_info.get("info", "")),
                "score_away": away_info.get("score", "-"), "overs_away": extract_overs(away_info.get("info", "")),
                "status": status_raw, "api_source": source, 
                "win_probability": win_prob, "projected_score": proj_score, "pre_match_winner": pre_match_winner
            })
            
    elif source == "CricketData" and "data" in data:
        for m in data["data"]:
            status_raw = m.get("status", "Unknown")
            if m.get("matchEnded"): match_state = "finished"
            elif m.get("matchStarted"): match_state = "live"
            else: match_state = "upcoming"
            
            home_score, home_overs, away_score, away_overs = "-", "", "-", ""
            teams = m.get("teams", ["Home", "Away"])
            
            for score in m.get("score", []):
                if score.get("inning", "").startswith(teams[0]):
                    home_score = f"{score.get('r', 0)}/{score.get('w', 0)}"
                    home_overs = f"{score.get('o', 0.0)} ov"
                elif score.get("inning", "").startswith(teams[1]):
                    away_score = f"{score.get('r', 0)}/{score.get('w', 0)}"
                    away_overs = f"{score.get('o', 0.0)} ov"

            win_prob, proj_score = None, None
            if match_state == "live":
                c_home_info = {"score": home_score, "info": home_overs}
                c_away_info = {"score": away_score, "info": away_overs}
                win_prob, proj_score = get_ai_predictions(c_home_info, c_away_info, m.get("matchType", "T20"), m.get("name", "League"), teams[0], teams[1])

            matches.append({
                "league": m.get("name", "League"), "format": m.get("matchType", "T20"),
                "start_time_iso": m.get("dateTimeGMT", ""), "match_state": match_state,
                "home": teams[0], "away": teams[1],
                "score_home": home_score, "overs_home": home_overs, 
                "score_away": away_score, "overs_away": away_overs,
                "status": status_raw, "api_source": source, 
                "win_probability": win_prob, "projected_score": proj_score, "pre_match_winner": None
            })

    elif source == "Matchstat" and "results" in data:
        for m in data["results"]:
            matches.append({
                "league": m.get("tournament", {}).get("name", "League"), "format": "T20",
                "start_time_iso": "", "match_state": "live",
                "home": m.get("homeTeam", {}).get("name", "Home"), "away": m.get("awayTeam", {}).get("name", "Away"),
                "score_home": m.get("homeScore", {}).get("display", "-"), "overs_home": "",
                "score_away": m.get("awayScore", {}).get("display", "-"), "overs_away": "",
                "status": m.get("status", {}).get("description", "In Play"), "api_source": source, 
                "win_probability": None, "projected_score": None, "pre_match_winner": None
            })

    elif source == "Sportscore" and "matches" in data:
        for m in data["matches"]:
            status_raw = m.get("status_text", "")
            status_lower = status_raw.lower()
            if "upcoming" in status_lower or "abnormal" in status_lower: match_state = "upcoming"
            elif "finished" in status_lower: match_state = "finished"
            else: match_state = "live"

            matches.append({
                "league": m.get("competition", "League"), "format": "T20",
                "start_time_iso": m.get("time", ""), "match_state": match_state,
                "home": m.get("home", "Home"), "away": m.get("away", "Away"),
                "score_home": m.get("home_score", "-") if m.get("home_score") else "-", "overs_home": "",
                "score_away": m.get("away_score", "-") if m.get("away_score") else "-", "overs_away": "",
                "status": status_raw, "api_source": source, 
                "win_probability": None, "projected_score": None, "pre_match_winner": None
            })
            
    return matches

def main():
    apis = [
        ("Highlightly", fetch_highlightly),    
        ("CricketData", fetch_cricketdata),    
        ("Matchstat", fetch_matchstat),        
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
                print(f"✅ Success with {api_name}!")
                break 

    final_matches.sort(key=lambda x: x.get("start_time_iso") or "")

    # 🔥 DUPLICATE MATCHES HATAEN (Yeh naya code hai)
    seen = set()
    unique_matches = []
    for m in final_matches:
        # Match ki unique pehchan (Home + Away + Time)
        key = (m.get("home", "").lower(), m.get("away", "").lower(), m.get("start_time_iso", ""))
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
    final_matches = unique_matches
    # 🔥 DUPLICATE LOGIC END

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "total_matches": len(final_matches), "data_source_api": used_api, "matches": final_matches
    }
    with open("live_matches.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"💾 Saved! Source: {used_api}")

if __name__ == "__main__":
    main()
