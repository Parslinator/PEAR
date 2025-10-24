from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import io
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests
from io import BytesIO
import matplotlib.colors as mcolors
from datetime import datetime
import pytz

app = FastAPI(title="PEAR Ratings API")

# CORS middleware to allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global constants
GLOBAL_HFA = 3
BASE_PATH = "./PEAR/PEAR Football"

# Calculate current week
central = pytz.timezone("US/Central")
now_ct = datetime.now(central)
start_dt = central.localize(datetime(2025, 9, 2, 9, 0, 0))

if now_ct < start_dt:
    CURRENT_WEEK = 1
else:
    CURRENT_WEEK = 2
    first_sunday = start_dt + pd.Timedelta(days=(6 - start_dt.weekday()))
    first_sunday = first_sunday.replace(hour=12, minute=0, second=0, microsecond=0)
    if first_sunday <= start_dt:
        first_sunday += pd.Timedelta(weeks=1)
    if now_ct >= first_sunday:
        weeks_since = ((now_ct - first_sunday).days // 7) + 1
        CURRENT_WEEK += weeks_since

CURRENT_YEAR = 2025

# Load team logos
team_logos = {}
logo_folder = f"{BASE_PATH}/logos/"
if os.path.exists(logo_folder):
    for filename in os.listdir(logo_folder):
        if filename.endswith(".png"):
            team_name = filename[:-4].replace("_", " ")
            try:
                img = Image.open(os.path.join(logo_folder, filename)).convert("RGBA")
                team_logos[team_name] = img
            except Exception as e:
                print(f"Error loading {filename}: {e}")

class SpreadRequest(BaseModel):
    away_team: str
    home_team: str
    neutral: bool = False

class MatchupRequest(BaseModel):
    away_team: str
    home_team: str
    neutral: bool = False
    year: int = CURRENT_YEAR
    week: int = CURRENT_WEEK

def load_data(year: int, week: int):
    """Load team data and ratings for a given year and week"""
    try:
        ratings_path = f"{BASE_PATH}/y{year}/Ratings/PEAR_week{week}.csv"
        data_path = f"{BASE_PATH}/y{year}/Data/team_data_week{week}.csv"
        
        ratings = pd.read_csv(ratings_path)
        if 'Unnamed: 0' in ratings.columns:
            ratings = ratings.drop(columns=['Unnamed: 0'])
        
        all_data = pd.read_csv(data_path)
        
        return ratings, all_data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Data not found for year {year}, week {week}")

def PEAR_Win_Prob(home_power_rating, away_power_rating, neutral):
    if not neutral:
        home_power_rating = home_power_rating + 1.5
    return round((1 / (1 + 10 ** ((away_power_rating - home_power_rating) / 20.5))) * 100, 2)

def calculate_gq(home_pr, away_pr, min_pr, max_pr):
    tq = (home_pr + away_pr) / 2
    tq_norm = np.clip((tq - min_pr) / (max_pr - min_pr), 0, 1)
    
    spread_cap = 30
    beta = 8.5
    spread = home_pr - away_pr
    sc = np.clip(1 - (abs(spread) / spread_cap), 0, 1)
    
    x = (0.65 * tq_norm + 0.35 * sc)
    gq_raw = 1 / (1 + np.exp(-beta * (x - 0.5)))
    
    gq = np.clip((1 + 9 * gq_raw) + 0.1, None, 10)
    return round(gq, 1)

@app.get("/")
def read_root():
    return {"message": "PEAR Ratings API", "version": "1.0", "current_week": CURRENT_WEEK}

@app.get("/api/current-season")
def get_current_season():
    return {"year": CURRENT_YEAR, "week": CURRENT_WEEK}

@app.get("/api/ratings/{year}/{week}")
def get_ratings(year: int, week: int):
    """Get power ratings for a specific year and week"""
    ratings, all_data = load_data(year, week)
    
    # Prepare response data
    all_data['OFF'] = all_data['offensive_rank']
    all_data['DEF'] = all_data['defensive_rank']
    all_data['MD'] = all_data.get('most_deserving', '')
    all_data['Rating'] = all_data['power_rating']
    all_data['Team'] = all_data['team']
    all_data['CONF'] = all_data.get('conference', '')
    all_data['ST'] = all_data.get('STM_rank', '')
    all_data['PBR'] = all_data.get('PBR_rank', '')
    all_data['DCE'] = all_data.get('DCE_rank', '')
    all_data['DDE'] = all_data.get('DDE_rank', '')
    
    result = all_data[['Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'PBR', 'DCE', 'DDE', 'CONF']].to_dict('records')
    
    return {"data": result, "year": year, "week": week}

@app.get("/api/spreads/{year}/{week}")
def get_spreads(year: int, week: int):
    """Get weekly spreads"""
    try:
        spreads_path = f"{BASE_PATH}/y{year}/Spreads/spreads_tracker_week{week}.xlsx"
        spreads = pd.read_excel(spreads_path)
        
        spreads['Vegas'] = spreads.get('formattedSpread', spreads.get('formatted_spread', ''))
        result = spreads[['home_team', 'away_team', 'PEAR', 'Vegas', 'difference', 'GQI', 'pr_spread']].dropna().to_dict('records')
        
        return {"data": result, "year": year, "week": week}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Spreads not found: {str(e)}")

@app.post("/api/calculate-spread")
def calculate_spread(request: SpreadRequest):
    """Calculate spread between two teams"""
    ratings, all_data = load_data(CURRENT_YEAR, CURRENT_WEEK)
    
    home_team_data = all_data[all_data['team'] == request.home_team]
    away_team_data = all_data[all_data['team'] == request.away_team]
    
    if home_team_data.empty or away_team_data.empty:
        raise HTTPException(status_code=404, detail="Team not found")
    
    home_pr = home_team_data['power_rating'].values[0]
    away_pr = away_team_data['power_rating'].values[0]
    
    home_offense = home_team_data['offensive_total'].values[0]
    away_offense = away_team_data['offensive_total'].values[0]
    home_defense = home_team_data['defensive_total'].values[0]
    away_defense = away_team_data['defensive_total'].values[0]
    
    home_elo = home_team_data.get('elo', pd.Series([1500])).values[0]
    away_elo = away_team_data.get('elo', pd.Series([1500])).values[0]
    
    home_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100, 2)
    adjustment = ((home_win_prob - 50) / 50) * 1
    
    raw_spread = GLOBAL_HFA + home_pr + adjustment - away_pr
    if request.neutral:
        raw_spread -= GLOBAL_HFA
    
    spread = round(raw_spread, 1)
    PEAR_win_prob = PEAR_Win_Prob(home_pr, away_pr, request.neutral)
    
    home_score = round(((home_offense - away_defense) + (GLOBAL_HFA/2 if not request.neutral else 0)), 1)
    away_score = round(((away_offense - home_defense) - (GLOBAL_HFA/2 if not request.neutral else 0)), 1)
    predicted_total = round(home_score + away_score, 1)
    
    gq = calculate_gq(home_pr, away_pr, all_data['power_rating'].min(), all_data['power_rating'].max())
    
    if spread >= 0:
        formatted_spread = f"{request.home_team} -{spread}"
    else:
        formatted_spread = f"{request.away_team} {spread}"
    
    return {
        "spread": spread,
        "formatted_spread": formatted_spread,
        "home_win_prob": PEAR_win_prob,
        "away_win_prob": round(100 - PEAR_win_prob, 1),
        "home_score": home_score,
        "away_score": away_score,
        "predicted_total": predicted_total,
        "game_quality": gq,
        "home_pr": round(home_pr, 2),
        "away_pr": round(away_pr, 2)
    }

@app.get("/api/teams")
def get_teams():
    """Get list of all teams"""
    ratings, all_data = load_data(CURRENT_YEAR, CURRENT_WEEK)
    teams = sorted(all_data['team'].unique().tolist())
    return {"teams": teams}

@app.get("/api/team/{team_name}")
def get_team_stats(team_name: str):
    """Get stats for a specific team"""
    ratings, all_data = load_data(CURRENT_YEAR, CURRENT_WEEK)
    
    team_data = all_data[all_data['team'] == team_name]
    if team_data.empty:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return {"data": team_data.to_dict('records')[0]}

@app.get("/api/historical-ratings")
def get_historical_ratings():
    """Get normalized ratings across all years"""
    try:
        hist_path = f"{BASE_PATH}/normalized_power_rating_across_years.csv"
        hist_data = pd.read_csv(hist_path)
        hist_data['Team'] = hist_data['team']
        hist_data['Season'] = hist_data['season'].astype(str)
        hist_data['Normalized Rating'] = hist_data['norm_pr']
        
        result = hist_data[['Team', 'Normalized Rating', 'Season']].to_dict('records')
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Historical data not found: {str(e)}")

@app.get("/api/team-history/{team_name}")
def get_team_history(team_name: str):
    """Get historical stats for a specific team"""
    try:
        hist_path = f"{BASE_PATH}/normalized_power_rating_across_years.csv"
        hist_data = pd.read_csv(hist_path)
        
        team_hist = hist_data[hist_data['team'] == team_name]
        if team_hist.empty:
            raise HTTPException(status_code=404, detail="Team not found")
        
        team_hist['Season'] = team_hist['season'].astype(str)
        team_hist['Normalized Rating'] = team_hist['norm_pr']
        
        result = team_hist[['Season', 'Normalized Rating', 'most_deserving', 'SOS', 'SOR', 
                           'offensive_rank', 'defensive_rank', 'STM_rank', 'PBR_rank', 
                           'DCE_rank', 'DDE_rank']].to_dict('records')
        return {"data": result, "team": team_name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {str(e)}")

@app.get("/api/game-images/{year}/{week}")
def get_game_images(year: int, week: int):
    """Get list of game images for a week"""
    try:
        folder_path = f"{BASE_PATH}/y{year}/Visuals/week_{week}/Games"
        if not os.path.exists(folder_path):
            return {"images": []}
        
        images = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        return {"images": images, "year": year, "week": week}
    except Exception as e:
        return {"images": []}

@app.get("/api/stat-profiles/{year}/{week}")
def get_stat_profiles(year: int, week: int):
    """Get list of stat profile images for a week"""
    try:
        folder_path = f"{BASE_PATH}/y{year}/Visuals/week_{week}/Stat Profiles"
        if not os.path.exists(folder_path):
            return {"images": []}
        
        images = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        return {"images": images, "year": year, "week": week}
    except Exception as e:
        return {"images": []}

@app.get("/api/image/{year}/{week}/{image_type}/{filename}")
def get_image(year: int, week: int, image_type: str, filename: str):
    """Serve an image file"""
    folder_map = {
        "games": "Games",
        "profiles": "Stat Profiles"
    }
    
    folder_name = folder_map.get(image_type)
    if not folder_name:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    file_path = f"{BASE_PATH}/y{year}/Visuals/week_{week}/{folder_name}/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)