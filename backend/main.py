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
from bs4 import BeautifulSoup
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import glob
import re

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

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the parent directory, then into PEAR
BASE_PATH = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Football")

print(f"Backend directory: {BACKEND_DIR}")
print(f"Base path for data: {BASE_PATH}")
print(f"Base path exists: {os.path.exists(BASE_PATH)}")

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

# Calculate current date for baseball
cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year

# Load team logos
team_logos = {}
logo_folder = os.path.join(BASE_PATH, "logos")
print(f"Logo folder path: {logo_folder}")
print(f"Logo folder exists: {os.path.exists(logo_folder)}")

if os.path.exists(logo_folder):
    for filename in os.listdir(logo_folder):
        if filename.endswith(".png"):
            team_name = filename[:-4].replace("_", " ")
            try:
                img = Image.open(os.path.join(logo_folder, filename)).convert("RGBA")
                team_logos[team_name] = img
            except Exception as e:
                print(f"Error loading {filename}: {e}")

@app.get("/api/football-logo/{team_name}")
def get_football_logo(team_name: str):
    """Serve team logo"""
    # Replace spaces with underscores for the filename
    logo_filename = f"{team_name.replace(' ', '_')}.png"
    logo_path = os.path.join(logo_folder, logo_filename)
    
    print(f"Looking for logo at: {logo_path}")
    print(f"Logo folder: {logo_folder}")
    print(f"File exists: {os.path.exists(logo_path)}")
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail=f"Logo not found at: {logo_path}")
    
    return FileResponse(logo_path, media_type="image/png")

@app.get("/api/game-preview/{year}/{week}/{filename}")
def get_game_preview(year: int, week: int, filename: str):
    """Serve game preview image"""
    # The filename comes as "home_team vs away_team" (without .png)
    image_filename = f"{filename}.png"
    image_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", "Games", image_filename)
    
    print(f"Looking for game preview at: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Game preview not found at: {image_path}")
    
    return FileResponse(image_path, media_type="image/png")

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
        ratings_path = os.path.join(BASE_PATH, f"y{year}", "Ratings", f"PEAR_week{week}.csv")
        data_path = os.path.join(BASE_PATH, f"y{year}", "Data", f"team_data_week{week}.csv")
        
        print(f"Attempting to load ratings from: {ratings_path}")
        print(f"Attempting to load data from: {data_path}")
        print(f"Ratings file exists: {os.path.exists(ratings_path)}")
        print(f"Data file exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(ratings_path):
            raise HTTPException(status_code=404, detail=f"Ratings file not found at: {ratings_path}")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found at: {data_path}")
        
        ratings = pd.read_csv(ratings_path)
        if 'Unnamed: 0' in ratings.columns:
            ratings = ratings.drop(columns=['Unnamed: 0'])
        
        all_data = pd.read_csv(data_path)
        
        return ratings, all_data
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise HTTPException(status_code=404, detail=f"Data files not found for year {year}, week {week}: {str(e)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

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

def plot_matchup_new(all_data, team_logos, away_team, home_team, neutrality, current_year, current_week):
    """Generate matchup visualization using complete original function"""
    import math
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.font_manager as fm
    
    # Helper functions
    def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                        facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
        ax.add_patch(Rectangle(
            (x - width/2, y - height/2), width, height,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=1
        ))
        ax.text(x, y, text, ha="center", va="center", zorder=2, **kwargs)

    def rank_to_color(rank, vmin=1, vmax=136):
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rank_cmap", ["#00008B", "#D3D3D3", "#8B0000"]
        )
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(rank))
        return mcolors.to_hex(rgba)

    def rank_text_color(rank):
        if rank != "" and (1 <= rank <= 55 or 81 <= rank <= 136):
            return 'white'
        return 'black'

    def get_value_and_rank(df, team, column, higher_is_better=True):
        ascending = not higher_is_better
        ranks = df[column].rank(ascending=ascending, method="first").astype(int)
        value = df.loc[df['team'] == team, column].values[0]
        rank = ranks.loc[df['team'] == team].values[0]
        return value, rank

    def get_column_value(df, team, column):
        return df.loc[df['team'] == team, column].values[0]
    
    def adjust_home_pr(home_win_prob):
        if home_win_prob is None or (isinstance(home_win_prob, float) and math.isnan(home_win_prob)):
            return 0
        return ((home_win_prob - 50) / 50) * 1

    def plot_logo(ax, img, xy, zoom=0.2):
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, frameon=False)
        ax.add_artist(ab)

    # Get all stats for both teams
    away_pr, away_rank = get_value_and_rank(all_data, away_team, 'power_rating')
    away_elo = get_column_value(all_data, away_team, 'elo')
    away_offense, away_offense_rank = get_value_and_rank(all_data, away_team, 'offensive_rating')
    away_defense, away_defense_rank = get_value_and_rank(all_data, away_team, 'defensive_rating', False)
    away_off_3, away_off_3_rank = get_value_and_rank(all_data, away_team, 'Offense_thirdDown_adj')
    away_off_sr, away_off_sr_rank = get_value_and_rank(all_data, away_team, 'Offense_successRate_adj')
    away_off_rus, away_off_rus_rank = get_value_and_rank(all_data, away_team, 'Offense_rushing_adj')
    away_off_pas, away_off_pas_rank = get_value_and_rank(all_data, away_team, 'Offense_passing_adj')
    away_off_ppo, away_off_ppo_rank = get_value_and_rank(all_data, away_team, 'adj_offense_ppo')
    away_off_ppa, away_off_ppa_rank = get_value_and_rank(all_data, away_team, 'Offense_ppa_adj')
    away_def_3, away_def_3_rank = get_value_and_rank(all_data, away_team, 'Defense_thirdDown_adj', False)
    away_def_sr, away_def_sr_rank = get_value_and_rank(all_data, away_team, 'Defense_successRate_adj', False)
    away_def_rus, away_def_rus_rank = get_value_and_rank(all_data, away_team, 'Defense_rushing_adj', False)
    away_def_pas, away_def_pas_rank = get_value_and_rank(all_data, away_team, 'Defense_passing_adj', False)
    away_def_ppo, away_def_ppo_rank = get_value_and_rank(all_data, away_team, 'adj_defense_ppo', False)
    away_def_ppa, away_def_ppa_rank = get_value_and_rank(all_data, away_team, 'Defense_ppa_adj', False)

    home_pr, home_rank = get_value_and_rank(all_data, home_team, 'power_rating')
    home_elo = get_column_value(all_data, home_team, 'elo')
    home_offense, home_offense_rank = get_value_and_rank(all_data, home_team, 'offensive_rating')
    home_defense, home_defense_rank = get_value_and_rank(all_data, home_team, 'defensive_rating', False)
    home_off_3, home_off_3_rank = get_value_and_rank(all_data, home_team, 'Offense_thirdDown_adj')
    home_off_sr, home_off_sr_rank = get_value_and_rank(all_data, home_team, 'Offense_successRate_adj')
    home_off_rus, home_off_rus_rank = get_value_and_rank(all_data, home_team, 'Offense_rushing_adj')
    home_off_pas, home_off_pas_rank = get_value_and_rank(all_data, home_team, 'Offense_passing_adj')
    home_off_ppo, home_off_ppo_rank = get_value_and_rank(all_data, home_team, 'adj_offense_ppo')
    home_off_ppa, home_off_ppa_rank = get_value_and_rank(all_data, home_team, 'Offense_ppa_adj')
    home_def_3, home_def_3_rank = get_value_and_rank(all_data, home_team, 'Defense_thirdDown_adj', False)
    home_def_sr, home_def_sr_rank = get_value_and_rank(all_data, home_team, 'Defense_successRate_adj', False)
    home_def_rus, home_def_rus_rank = get_value_and_rank(all_data, home_team, 'Defense_rushing_adj', False)
    home_def_pas, home_def_pas_rank = get_value_and_rank(all_data, home_team, 'Defense_passing_adj', False)
    home_def_ppo, home_def_ppo_rank = get_value_and_rank(all_data, home_team, 'adj_defense_ppo', False)
    home_def_ppa, home_def_ppa_rank = get_value_and_rank(all_data, home_team, 'Defense_ppa_adj', False)

    home_wins = get_column_value(all_data, home_team, 'wins')
    home_losses = get_column_value(all_data, home_team, 'losses')
    home_conf_wins = get_column_value(all_data, home_team, 'conference_wins')
    home_conf_losses = get_column_value(all_data, home_team, 'conference_losses')
    away_wins = get_column_value(all_data, away_team, 'wins')
    away_losses = get_column_value(all_data, away_team, 'losses')
    away_conf_wins = get_column_value(all_data, away_team, 'conference_wins')
    away_conf_losses = get_column_value(all_data, away_team, 'conference_losses')
    
    away_off_dq, away_off_dq_rank = get_value_and_rank(all_data, away_team, 'adj_offense_drive_quality')
    away_def_dq, away_def_dq_rank = get_value_and_rank(all_data, away_team, 'adj_defense_drive_quality', False)
    home_off_dq, home_off_dq_rank = get_value_and_rank(all_data, home_team, 'adj_offense_drive_quality')
    home_def_dq, home_def_dq_rank = get_value_and_rank(all_data, home_team, 'adj_defense_drive_quality', False)
    
    away_off_fp, away_off_fp_rank = get_value_and_rank(all_data, away_team, 'Offense_fieldPosition_averageStart', False)
    away_def_fp, away_def_fp_rank = get_value_and_rank(all_data, away_team, 'Defense_fieldPosition_averageStart')
    home_off_fp, home_off_fp_rank = get_value_and_rank(all_data, home_team, 'Offense_fieldPosition_averageStart', False)
    home_def_fp, home_def_fp_rank = get_value_and_rank(all_data, home_team, 'Defense_fieldPosition_averageStart')
    
    away_md, away_md_rank = get_value_and_rank(all_data, away_team, 'most_deserving_wins')
    home_md, home_md_rank = get_value_and_rank(all_data, home_team, 'most_deserving_wins')
    away_sos, away_sos_rank = get_value_and_rank(all_data, away_team, 'avg_expected_wins', False)
    home_sos, home_sos_rank = get_value_and_rank(all_data, home_team, 'avg_expected_wins', False)
    away_mov, away_mov_rank = get_value_and_rank(all_data, away_team, 'RTP')
    home_mov, home_mov_rank = get_value_and_rank(all_data, home_team, 'RTP')

    # Calculate predictions
    home_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)
    PEAR_home_prob = PEAR_Win_Prob(home_pr, away_pr, neutrality)
    spread = (3+home_pr+adjust_home_pr(home_win_prob)-away_pr).round(1)
    if neutrality:
        spread = (spread-3).round(1)
    
    if spread <= 0:
        formatted_spread = f'{away_team} {spread}'
    else:
        formatted_spread = f'{home_team} -{spread}'

    home_offense_contrib = (home_offense + away_defense) / 2
    away_offense_contrib = (away_offense + home_defense) / 2
    predicted_total = round(home_offense_contrib + away_offense_contrib, 1)
    home_score = round((predicted_total + spread) / 2, 1)
    away_score = round((predicted_total - spread) / 2, 1)
    gq_value = calculate_gq(home_pr, away_pr, all_data['power_rating'].min(), all_data['power_rating'].max())

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Plot logos
    if away_team in team_logos and team_logos[away_team] is not None:
        plot_logo(ax, team_logos[away_team], (0.15, 0.75), zoom=0.5)
    if home_team in team_logos and team_logos[home_team] is not None:
        plot_logo(ax, team_logos[home_team], (0.85, 0.75), zoom=0.5)

    # Title
    if neutrality:
        ax.text(0.5, 0.96, f"{away_team} (N) {home_team}", ha='center', fontsize=32, fontweight='bold')
    else:
        ax.text(0.5, 0.96, f"{away_team} at {home_team}", ha='center', fontsize=32, fontweight='bold')
    
    # Main predictions
    ax.text(0.5, 0.57, f"{formatted_spread}", ha='center', fontsize=28, fontweight='bold')
    ax.text(0.5, 0.625, f"O/U: {predicted_total}", ha='center', fontsize=28, fontweight='bold')
    ax.text(0.5, 0.68, f"GQI: {gq_value}", ha='center', fontsize=28, fontweight='bold')

    # Win probabilities
    ax.text(0.4, 0.89, f"WIN PROB (%)", ha='center', fontsize=11, fontweight='bold')
    ax.text(0.4, 0.84, f"{round(100-PEAR_home_prob,1)}", ha='center', fontsize=36, fontweight='bold')
    ax.text(0.4, 0.785, f"PROJ. POINTS", ha='center', fontsize=11, fontweight='bold')
    ax.text(0.4, 0.735, f"{away_score}", ha='center', fontsize=36, fontweight='bold')
    ax.text(0.5, 0.74, f"—", ha='center', fontsize=36, fontweight='bold')
    ax.text(0.6, 0.89, f"WIN PROB (%)", ha='center', fontsize=11, fontweight='bold')
    ax.text(0.6, 0.84, f"{round(PEAR_home_prob,1)}", ha='center', fontsize=36, fontweight='bold')
    ax.text(0.6, 0.785, f"PROJ. POINTS", ha='center', fontsize=11, fontweight='bold')
    ax.text(0.6, 0.735, f"{home_score}", ha='center', fontsize=36, fontweight='bold')

    # Helper for stat rows
    def add_row(x_vals, y, away_val, away_rank, metric_name, home_rank, home_val):
        alpha_val = 0.9
        if away_val != "":
            ax.text(x_vals[0], y, f"{away_val:.2f}", ha='center', fontsize=16, fontweight='bold')
        if away_rank != "":
            fixed_width_text(
                ax, x_vals[1], y+0.007, f"{away_rank}", width=0.06, height=0.04,
                facecolor=rank_to_color(away_rank), alpha=alpha_val,
                fontsize=16, fontweight='bold', color=rank_text_color(away_rank)
            )
        if metric_name != "":
            ax.text(x_vals[2], y, metric_name, ha='center', fontsize=16, fontweight='bold')
        if home_rank != "":
            fixed_width_text(
                ax, x_vals[3], y+0.007, f"{home_rank}", width=0.06, height=0.04,
                facecolor=rank_to_color(home_rank), alpha=alpha_val,
                fontsize=16, fontweight='bold', color=rank_text_color(home_rank)
            )
        if home_val != "":
            ax.text(x_vals[4], y, f"{home_val:.2f}", ha='center', fontsize=16, fontweight='bold')

    alpha_val = 0.9
    x_cols = [0.31, 0.378, 0.5, 0.622, 0.69]

    # Matchup stats sections
    ax.text(0.5, 0.528, f"{away_team} OFF vs {home_team} DEF", ha='center', fontsize=16, fontweight='bold')
    ax.hlines(y=0.518, xmin=0.29, xmax=0.71, colors='black', linewidth=1)
    add_row(x_cols, 0.49, away_off_sr, away_off_sr_rank, "SUCCESS RATE", home_def_sr_rank, home_def_sr)
    add_row(x_cols, 0.45, away_off_3, away_off_3_rank, "3RD DOWN PPA", home_def_3_rank, home_def_3)
    add_row(x_cols, 0.41, away_off_rus, away_off_rus_rank, "RUSHING PPA", home_def_rus_rank, home_def_rus)
    add_row(x_cols, 0.37, away_off_pas, away_off_pas_rank, "PASSING PPA", home_def_pas_rank, home_def_pas)
    add_row(x_cols, 0.33, away_off_ppa, away_off_ppa_rank, "TOTAL PPA", home_def_ppa_rank, home_def_ppa)
    add_row(x_cols, 0.29, away_off_ppo, away_off_ppo_rank, "POINTS PER OPP", home_def_ppo_rank, home_def_ppo)

    ax.text(0.5, 0.248, f"{away_team} DEF vs {home_team} OFF", ha='center', fontsize=16, fontweight='bold')
    ax.hlines(y=0.238, xmin=0.29, xmax=0.71, colors='black', linewidth=1)
    add_row(x_cols, 0.21, away_def_sr, away_def_sr_rank, "SUCCESS RATE", home_off_sr_rank, home_off_sr)
    add_row(x_cols, 0.17, away_def_3, away_def_3_rank, "3RD DOWN PPA", home_off_3_rank, home_off_3)
    add_row(x_cols, 0.13, away_def_rus, away_def_rus_rank, "RUSHING PPA", home_off_rus_rank, home_off_rus)
    add_row(x_cols, 0.09, away_def_pas, away_def_pas_rank, "PASSING PPA", home_off_pas_rank, home_off_pas)
    add_row(x_cols, 0.05, away_def_ppa, away_def_ppa_rank, "TOTAL PPA", home_off_ppa_rank, home_off_ppa)
    add_row(x_cols, 0.01, away_def_ppo, away_def_ppo_rank, "POINTS PER OPP", home_off_ppo_rank, home_off_ppo)
    add_row(x_cols, -0.03, "", "", "@PEARatings", "", "")

    # Left side team stats
    ax.text(0.01, 0.53, f"{away_wins}-{away_losses} ({away_conf_wins}-{away_conf_losses})", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.01, 0.49, f"RATING", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.478, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.19, 0.49, f"{away_pr:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.49+0.007, f"{away_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_rank))
    
    ax.text(0.08, 0.45, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.45, f"{away_offense:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.45+0.007, f"{away_offense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_offense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_offense_rank))

    ax.text(0.08, 0.41, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.41, f"{away_defense:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.41+0.007, f"{away_defense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_defense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_defense_rank))

    ax.text(0.01, 0.37, f"DRIVE QUALITY", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.08, 0.33, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.33, f"{away_off_dq:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.33+0.007, f"{away_off_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_off_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_off_dq_rank))
    
    ax.text(0.08, 0.29, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.29, f"{away_def_dq:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.29+0.007, f"{away_def_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_def_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_def_dq_rank))

    ax.text(0.01, 0.25, f"FIELD POSITION", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.238, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.08, 0.21, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.21, f"{75-away_off_fp:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.21+0.007, f"{away_off_fp_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_off_fp_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_off_fp_rank))
    
    ax.text(0.08, 0.17, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.17, f"{away_def_fp-75:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.17+0.007, f"{away_def_fp_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_def_fp_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_def_fp_rank))
    
    ax.text(0.01, 0.13, f"RESUME", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.118, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.08, 0.09, f"MD", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.09, f"{away_md:.0f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.09+0.007, f"{away_md_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_md_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_md_rank))

    ax.text(0.08, 0.05, f"SOS", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.05, f"{away_sos:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.05+0.007, f"{away_sos_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_sos_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_sos_rank))

    ax.text(0.08, 0.01, f"MOV", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.01, f"{away_mov:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.01+0.007, f"{away_mov_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_mov_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_mov_rank))

    # Right side team stats
    ax.text(0.99, 0.53, f"{home_wins}-{home_losses} ({home_conf_wins}-{home_conf_losses})", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.99, 0.49, f"RATING", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.478, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.81, 0.49, f"{home_pr:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.49+0.007, f"{home_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_rank))
    
    ax.text(0.92, 0.45, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.45, f"{home_offense:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.45+0.007, f"{home_offense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_offense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_offense_rank))

    ax.text(0.92, 0.41, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.41, f"{home_defense:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.41+0.007, f"{home_defense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_defense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_defense_rank))
    
    ax.text(0.99, 0.37, f"DRIVE QUALITY", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.92, 0.33, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.33, f"{home_off_dq:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.33+0.007, f"{home_off_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_off_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_off_dq_rank))
    
    ax.text(0.92, 0.29, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.29, f"{home_def_dq:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.29+0.007, f"{home_def_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_def_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_def_dq_rank))

    ax.text(0.99, 0.25, f"FIELD POSITION", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.238, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.92, 0.21, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.21, f"{75-home_off_fp:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.21+0.007, f"{home_off_fp_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_off_fp_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_off_fp_rank))

    ax.text(0.92, 0.17, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.17, f"{home_def_fp-75:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.17+0.007, f"{home_def_fp_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_def_fp_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_def_fp_rank))
    
    ax.text(0.99, 0.13, f"RESUME", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.118, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.92, 0.09, f"MD", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.09, f"{home_md:.0f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.09+0.007, f"{home_md_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_md_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_md_rank))

    ax.text(0.92, 0.05, f"SOS", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.05, f"{home_sos:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.05+0.007, f"{home_sos_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_sos_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_sos_rank))

    ax.text(0.92, 0.01, f"MOV", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.01, f"{home_mov:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.01+0.007, f"{home_mov_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_mov_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_mov_rank))
    
    return fig

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
    
    all_data['Rating'] = all_data['power_rating']
    all_data['Team'] = all_data['team']
    all_data['CONF'] = all_data.get('conference', '')
    all_data['MD'] = all_data.get('most_deserving_wins', 0)
    all_data['SOS'] = all_data.get('avg_expected_wins', 0)
    
    result = all_data[['Team', 'Rating', 'offensive_rating', 'defensive_rating', 'MD', 'SOS', 'CONF']].to_dict('records')
    
    return {"data": result, "year": year, "week": week}

@app.get("/api/team-stats/{year}/{week}")
def get_team_stats(year: int, week: int):
    """Get detailed team statistics for offense and defense"""
    ratings, all_data = load_data(year, week)
    
    # Add conference field if it exists
    all_data['conference'] = all_data.get('conference', '')
    
    # Select required columns
    stats_columns = [
        'team',
        'power_rating',
        'offensive_rating',
        'Offense_successRate_adj',
        'Offense_ppa_adj',
        'Offense_rushing_adj',
        'Offense_passing_adj',
        'adj_offense_ppo',
        'adj_offense_drive_quality',
        'defensive_rating',
        'Defense_successRate_adj',
        'Defense_ppa_adj',
        'Defense_rushing_adj',
        'Defense_passing_adj',
        'adj_defense_ppo',
        'adj_defense_drive_quality',
        'conference'
    ]
    
    result = all_data[stats_columns].to_dict('records')
    
    return {"data": result, "year": year, "week": week}

@app.get("/api/spreads/{year}/{week}")
def get_spreads(year: int, week: int):
    """Get weekly spreads"""
    try:
        spreads_path = os.path.join(BASE_PATH, f"y{year}", "Spreads", f"spreads_tracker_week{week}.xlsx")
        print(f"Attempting to load spreads from: {spreads_path}")
        print(f"Spreads file exists: {os.path.exists(spreads_path)}")
        
        if not os.path.exists(spreads_path):
            raise HTTPException(status_code=404, detail=f"Spreads file not found at: {spreads_path}")
        
        spreads = pd.read_excel(spreads_path)
        spreads['start_date'] = pd.to_datetime(spreads['start_date']).dt.strftime('%Y-%m-%d')
        
        vegas_col = 'formattedSpread' if 'formattedSpread' in spreads.columns else 'formatted_spread'
        spreads['Vegas'] = spreads.get(vegas_col, '')
        
        required_cols = ['start_date', 'start_time', 'home_team', 'away_team', 'home_score', 'away_score', 'PEAR_win_prob', 'PEAR', 'pr_spread', 'difference', 'GQI']
        missing_cols = [col for col in required_cols if col not in spreads.columns]
        
        if missing_cols:
            print(f"Available columns: {spreads.columns.tolist()}")
            print(f"Missing columns: {missing_cols}")
            raise HTTPException(status_code=500, detail=f"Missing columns in spreads file: {missing_cols}")

        result = spreads[['start_date', 'start_time', 'home_team', 'away_team', 'home_score', 'away_score', 'PEAR_win_prob', 'PEAR', 'Vegas', 'difference', 'GQI', 'pr_spread']].dropna().to_dict('records')

        return {"data": result, "year": year, "week": week}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error loading spreads: {e}")
        raise HTTPException(status_code=500, detail=f"Spreads error: {str(e)}")

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

@app.post("/api/generate-matchup-image")
async def generate_matchup_image(request: MatchupRequest):
    """Generate matchup visualization image"""
    try:
        ratings, all_data = load_data(request.year, request.week)
        
        fig = plot_matchup_new(
            all_data, 
            team_logos, 
            request.away_team, 
            request.home_team, 
            request.neutral,
            request.year,
            request.week
        )
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Error generating matchup image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.get("/api/teams")
def get_teams():
    """Get list of all teams"""
    try:
        ratings, all_data = load_data(CURRENT_YEAR, CURRENT_WEEK)
        teams = sorted(all_data['team'].unique().tolist())
        return {"teams": teams}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")

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
        hist_path = os.path.join(BASE_PATH, "normalized_power_rating_across_years.csv")
        print(f"Attempting to load historical data from: {hist_path}")
        print(f"File exists: {os.path.exists(hist_path)}")
        
        if not os.path.exists(hist_path):
            raise HTTPException(status_code=404, detail=f"Historical data file not found at: {hist_path}")
        
        hist_data = pd.read_csv(hist_path)
        hist_data['Team'] = hist_data['team']
        hist_data['Season'] = hist_data['season'].astype(str)
        hist_data['Normalized Rating'] = hist_data['norm_pr']
        
        result = hist_data[['Team', 'Normalized Rating', 'Season']].to_dict('records')
        return {"data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_historical_ratings: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading historical data: {str(e)}")

@app.get("/api/team-history/{team_name}")
def get_team_history(team_name: str):
    """Get historical stats for a specific team"""
    try:
        hist_path = os.path.join(BASE_PATH, "normalized_power_rating_across_years.csv")
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
        folder_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", "Games")
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
        folder_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", "Stat Profiles")
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
    
    file_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", folder_name, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)

@app.get("/api/logo")
def get_logo():
    """Serve the PEAR logo"""
    # Path to the logo on your local machine
    logo_path = r"C:\Users\wpars\OneDrive\Documents\Post-School\Family Feud Data\PEAR\pear_logo.jpg"
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail="Logo not found")
    
    return FileResponse(logo_path, media_type="image/jpeg")

# ========================================
# BASEBALL (CBASE) ENDPOINTS
# ========================================

BASEBALL_BASE_PATH = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Baseball")
BASEBALL_CURRENT_SEASON = 2025
BASEBALL_HFA = 0.3  # Home field advantage in runs for baseball

print(f"Baseball base path: {BASEBALL_BASE_PATH}")
print(f"Baseball base path exists: {os.path.exists(BASEBALL_BASE_PATH)}")

class BaseballSpreadRequest(BaseModel):
    away_team: str
    home_team: str
    neutral: bool = False

class RegionalRequest(BaseModel):
    team_1: str
    team_2: str
    team_3: str
    team_4: str
    simulations: int = 1000

def load_baseball_data():
    """Load the most recent baseball data file"""
    try:
        folder_path = os.path.join(BASEBALL_BASE_PATH, f"y{BASEBALL_CURRENT_SEASON}", "Data")
        
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Baseball data folder not found: {folder_path}")
        
        # Find all baseball CSV files
        csv_files = [f for f in os.listdir(folder_path) 
                    if f.startswith("baseball_") and f.endswith(".csv")]
        
        if not csv_files:
            raise HTTPException(status_code=404, detail="No baseball data files found")
        
        # Extract dates and find most recent
        def extract_date(filename):
            try:
                return datetime.strptime(filename.replace("baseball_", "").replace(".csv", ""), "%m_%d_%Y")
            except ValueError:
                return None
        
        date_files = {extract_date(f): f for f in csv_files if extract_date(f) is not None}
        
        if not date_files:
            raise HTTPException(status_code=404, detail="No valid date files found")
        
        sorted_dates = sorted(date_files.keys(), reverse=True)
        latest_date = sorted_dates[0]
        latest_file = date_files[latest_date]
        
        file_path = os.path.join(folder_path, latest_file)
        modeling_stats = pd.read_csv(file_path)
        
        # If file doesn't have expected number of teams, try previous day
        if len(modeling_stats) < 290 and len(sorted_dates) > 1:
            previous_date = sorted_dates[1]
            previous_file = date_files[previous_date]
            file_path = os.path.join(folder_path, previous_file)
            modeling_stats = pd.read_csv(file_path)
            latest_date = previous_date
        
        formatted_date = latest_date.strftime("%B %d, %Y")
        
        return modeling_stats, formatted_date
    
    except Exception as e:
        print(f"Error loading baseball data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading baseball data: {str(e)}")
    
def load_elo():
    try:    
        # URL of the page to scrape
        url = 'https://www.warrennolan.com/baseball/2025/elo'

        # Fetch the webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table with the specified class
        table = soup.find('table', class_='normal-grid alternating-rows stats-table')

        if table:
            # Extract table headers
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]
            headers.insert(1, "Team Link")  # Adding extra column for team link

            # Extract table rows
            data = []
            for row in table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                row_data = []
                for i, cell in enumerate(cells):
                    # If it's the first cell, extract team name and link from 'name-subcontainer'
                    if i == 0:
                        name_container = cell.find('div', class_='name-subcontainer')
                        if name_container:
                            team_name = name_container.text.strip()
                            team_link_tag = name_container.find('a')
                            team_link = team_link_tag['href'] if team_link_tag else ''
                        else:
                            team_name = cell.text.strip()
                            team_link = ''
                        row_data.append(team_name)
                        row_data.append(team_link)  # Add team link separately
                    else:
                        row_data.append(cell.text.strip())
                data.append(row_data)


            elo_data = pd.DataFrame(data, columns=[headers])
            elo_data.columns = elo_data.columns.get_level_values(0)
            elo_data = elo_data.drop_duplicates(subset='Team', keep='first')
            elo_data = elo_data.astype({col: 'str' for col in elo_data.columns if col not in ['ELO', 'Rank']})
            elo_data['ELO'] = elo_data['ELO'].astype(float, errors='ignore')
            elo_data['Rank'] = elo_data['Rank'].astype(int, errors='ignore')

        else:
            print("Table not found on the page.")

        # Define mapping for team name replacements
        team_replacements = {
            'North Carolina St.': 'NC State',
            'Southern Miss': 'Southern Miss.',
            'USC': 'Southern California',
            'Dallas Baptist': 'DBU',
            'Charleston': 'Col. of Charleston',
            'Georgia Southern': 'Ga. Southern',
            'UNCG': 'UNC Greensboro',
            'East Tennessee St.': 'ETSU',
            'Lamar': 'Lamar University',
            "Saint Mary's College": "Saint Mary's (CA)",
            'Western Kentucky': 'Western Ky.',
            'FAU': 'Fla. Atlantic',
            'Connecticut': 'UConn',
            'Southeast Missouri': 'Southeast Mo. St.',
            'Alcorn St.': 'Alcorn',
            'Appalachian St.': 'App State',
            'Arkansas-Pine Bluff': 'Ark.-Pine Bluff',
            'Army': 'Army West Point',
            'Cal St. Bakersfield': 'CSU Bakersfield',
            'Cal St. Northridge': 'CSUN',
            'Central Arkansas': 'Central Ark.',
            'Central Michigan': 'Central Mich.',
            'Charleston Southern': 'Charleston So.',
            'Eastern Illinois': 'Eastern Ill.',
            'Eastern Kentucky': 'Eastern Ky.',
            'Eastern Michigan': 'Eastern Mich.',
            'Fairleigh Dickinson': 'FDU',
            'Grambling St.': 'Grambling',
            'Incarnate Word': 'UIW',
            'Long Island': 'LIU',
            'Maryland Eastern Shore': 'UMES',
            'Middle Tennessee': 'Middle Tenn.',
            'Mississippi Valley St.': 'Mississippi Val.',
            "Mount Saint Mary's": "Mount St. Mary's",
            'North Alabama': 'North Ala.',
            'North Carolina A&T': 'N.C. A&T',
            'Northern Colorado': 'Northern Colo.',
            'Northern Kentucky': 'Northern Ky.',
            'Prairie View A&M': 'Prairie View',
            'Presbyterian College': 'Presbyterian',
            'Saint Bonaventure': 'St. Bonaventure',
            "Saint John's": "St. John's (NY)",
            'Sam Houston St.': 'Sam Houston',
            'Seattle University': 'Seattle U',
            'South Carolina Upstate': 'USC Upstate',
            'South Florida': 'South Fla.',
            'Southeastern Louisiana': 'Southeastern La.',
            'Southern': 'Southern U.',
            'Southern Illinois': 'Southern Ill.',
            'Stephen F. Austin': 'SFA',
            'Tennessee-Martin': 'UT Martin',
            'Texas A&M-Corpus Christi': 'A&M-Corpus Christi',
            'UMass-Lowell': 'UMass Lowell',
            'UTA': 'UT Arlington',
            'Western Carolina': 'Western Caro.',
            'Western Illinois': 'Western Ill.',
            'Western Michigan': 'Western Mich.',
            'Albany': 'UAlbany',
            'Southern Indiana': 'Southern Ind.',
            'Queens': 'Queens (NC)',
            'Central Connecticut': 'Central Conn. St.',
            'Saint Thomas': 'St. Thomas (MN)',
            'Northern Illinois': 'NIU',
            'UMass':'Massachusetts',
            'Loyola-Marymount':'LMU (CA)'
        }

        elo_data['Team'] = elo_data['Team'].str.replace('State', 'St.', regex=False)
        elo_data['Team'] = elo_data['Team'].replace(team_replacements)
        return elo_data
    except Exception as e:
        print(f"Error loading elo: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading elo: {str(e)}")

def load_baseball_schedule():
    """Load the current season schedule"""
    try:
        schedule_path = os.path.join(BASEBALL_BASE_PATH, f"y{BASEBALL_CURRENT_SEASON}", 
                                     f"schedule_{BASEBALL_CURRENT_SEASON}.csv")
        
        if not os.path.exists(schedule_path):
            raise HTTPException(status_code=404, detail="Schedule file not found")
        
        schedule_df = pd.read_csv(schedule_path)
        schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
        
        return schedule_df
    
    except Exception as e:
        print(f"Error loading schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading schedule: {str(e)}")

def baseball_win_prob(home_pr, away_pr, location="Neutral"):
    """Calculate win probability for baseball"""
    if location != "Neutral":
        home_pr += BASEBALL_HFA
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

def adjust_home_pr_baseball(home_win_prob):
    """Adjust home power rating based on ELO win probability"""
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_baseball_spread(home_pr, away_pr, home_elo, away_elo, location):
    """Calculate spread for baseball matchup"""
    if location != "Neutral":
        home_pr += BASEBALL_HFA
    
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / 
                        ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    
    spread = round(adjust_home_pr_baseball(elo_win_prob) + home_pr - away_pr, 2)
    
    return spread, elo_win_prob

def calculate_gqi_baseball(home_pr, away_pr, min_pr, max_pr):
    """Calculate Game Quality Index for baseball"""
    # Team quality
    tq = (home_pr + away_pr) / 2
    tq_norm = np.clip((tq - min_pr) / (max_pr - min_pr), 0, 1)
    
    # Spread competitiveness
    spread_cap = 8  # Adjusted for baseball
    beta = 8.5
    spread = abs(home_pr - away_pr)
    sc = np.clip(1 - (spread / spread_cap), 0, 1)
    
    # Combine factors
    x = (0.65 * tq_norm + 0.35 * sc)
    gqi_raw = 1 / (1 + np.exp(-beta * (x - 0.5)))
    
    gqi = np.clip((1 + 9 * gqi_raw) + 0.1, None, 10)
    return round(gqi, 1)

@app.get("/api/cbase/ratings")
def get_baseball_ratings():
    """Get current baseball team ratings"""
    try:
        modeling_stats, data_date = load_baseball_data()
        
        # Prepare data for frontend
        teams = modeling_stats[[
            'Team', 'Rating', 'NET', 'NET_Score', 'SOS', 'SOR', 
            'Conference', 'ELO', 'RPI', 'PRR', 'RQI'
        ]].copy()
        
        teams = teams.rename(columns={
            'Rating': 'power_rating',
            'NET_Score': 'net_score'
        })
        
        teams = teams.sort_values('power_rating', ascending=False).reset_index(drop=True)
        
        return {
            "teams": teams.to_dict('records'),
            "date": data_date,
            "count": len(teams)
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_baseball_ratings: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/stats")
def get_baseball_stats():
    """Get comprehensive baseball team statistics"""
    try:
        modeling_stats, data_date = load_baseball_data()
        
        # All the stats columns
        stats_columns = [
            'Team', 'Conference', 'Rating', 'NET', 'NET_Score', 'RPI', 'ELO', 'ELO_Rank', 'PRR', 'RQI', 
            'resume_quality', 'avg_expected_wins', 'SOS', 'SOR', 'Q1', 'Q2', 'Q3', 'Q4',
            'fWAR', 'oWAR_z', 'pWAR_z', 'WPOE', 'PYTHAG',
            'ERA', 'WHIP', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS'
        ]
        
        available_columns = [col for col in stats_columns if col in modeling_stats.columns]
        stats = modeling_stats[available_columns].copy()
        
        # If ELO_Rank doesn't exist, use ELO column as the rank
        if 'ELO_Rank' not in stats.columns and 'ELO' in stats.columns:
            stats['ELO_Rank'] = stats['ELO']
        
        stats = stats.sort_values('Rating', ascending=False).reset_index(drop=True)
        
        return {
            "stats": stats.to_dict('records'),
            "date": data_date
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_baseball_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/teams")
def get_baseball_teams():
    """Get list of all baseball teams"""
    try:
        modeling_stats, _ = load_baseball_data()
        teams = sorted(modeling_stats['Team'].unique().tolist())
        return {"teams": teams}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_baseball_teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/cbase/calculate-spread")
def calculate_baseball_matchup_spread(request: BaseballSpreadRequest):
    """Calculate spread for baseball matchup"""
    try:
        modeling_stats, _ = load_baseball_data()
        
        home_team_data = modeling_stats[modeling_stats['Team'] == request.home_team]
        away_team_data = modeling_stats[modeling_stats['Team'] == request.away_team]
        
        if home_team_data.empty or away_team_data.empty:
            raise HTTPException(status_code=404, detail="Team not found")
        
        home_pr = home_team_data['Rating'].values[0]
        away_pr = away_team_data['Rating'].values[0]
        home_elo = home_team_data['ELO'].values[0]
        away_elo = away_team_data['ELO'].values[0]
        
        location = "Neutral" if request.neutral else "Home"
        spread, elo_win_prob = calculate_baseball_spread(home_pr, away_pr, home_elo, away_elo, location)
        win_prob = baseball_win_prob(home_pr, away_pr, location)
        
        gqi = calculate_gqi_baseball(
            home_pr, away_pr, 
            modeling_stats['Rating'].min(), 
            modeling_stats['Rating'].max()
        )
        
        if spread >= 0:
            formatted_spread = f"{request.home_team} -{spread}"
        else:
            formatted_spread = f"{request.away_team} {abs(spread)}"
        
        return {
            "spread": spread,
            "formatted_spread": formatted_spread,
            "home_win_prob": win_prob,
            "away_win_prob": round(100 - win_prob, 2),
            "elo_win_prob": elo_win_prob,
            "game_quality": gqi,
            "home_pr": round(home_pr, 2),
            "away_pr": round(away_pr, 2),
            "home_elo": round(home_elo, 0),
            "away_elo": round(away_elo, 0)
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error calculating baseball spread: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/schedule/today")
def get_todays_baseball_games():
    """Get today's baseball games"""
    try:
        schedule_df = load_baseball_schedule()
        
        cst = pytz.timezone('America/Chicago')
        today = datetime.now(cst).date()
        
        today_games = schedule_df[schedule_df['Date'].dt.date == today].copy()
        
        if len(today_games) == 0:
            return {"games": [], "date": today.strftime("%B %d, %Y")}
        
        # Process results
        today_games = today_games[[
            'home_team', 'away_team', 'PEAR', 'GQI', 'Date', 'Result'
        ]].copy()
        
        today_games = today_games.sort_values('GQI', ascending=False).reset_index(drop=True)
        
        return {
            "games": today_games.to_dict('records'),
            "date": today.strftime("%B %d, %Y"),
            "count": len(today_games)
        }
    
    except Exception as e:
        print(f"Error getting today's games: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/team/{team_name}")
def get_baseball_team_info(team_name: str):
    """Get detailed information for a specific baseball team"""
    try:
        modeling_stats, data_date = load_baseball_data()
        schedule_df = load_baseball_schedule()
        
        team_data = modeling_stats[modeling_stats['Team'] == team_name]
        
        if team_data.empty:
            raise HTTPException(status_code=404, detail="Team not found")
        
        # Get team schedule
        team_schedule = schedule_df[
            (schedule_df['home_team'] == team_name) | 
            (schedule_df['away_team'] == team_name)
        ].copy()
        
        cst = pytz.timezone('America/Chicago')
        today = datetime.now(cst).date()
        
        # Upcoming games
        upcoming = team_schedule[team_schedule['Date'].dt.date >= today].copy()
        upcoming = upcoming.sort_values('Date').head(10)
        
        # Recent results
        completed = team_schedule[
            (team_schedule['Date'].dt.date < today) & 
            (team_schedule['Result'].notna())
        ].copy()
        completed = completed.sort_values('Date', ascending=False).head(10)
        
        return {
            "team": team_data.to_dict('records')[0],
            "upcoming_games": upcoming.to_dict('records') if len(upcoming) > 0 else [],
            "recent_games": completed.to_dict('records') if len(completed) > 0 else [],
            "date": data_date
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error getting team info: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/conferences")
def get_baseball_conferences():
    """Get list of all conferences"""
    try:
        modeling_stats, _ = load_baseball_data()
        conferences = sorted(modeling_stats['Conference'].unique().tolist())
        # Remove "Independent" if present
        conferences = [c for c in conferences if c != "Independent"]
        return {"conferences": conferences}
    
    except Exception as e:
        print(f"Error getting conferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/conference/{conference_name}")
def get_conference_standings(conference_name: str):
    """Get standings for a specific conference"""
    try:
        modeling_stats, data_date = load_baseball_data()
        
        conf_teams = modeling_stats[modeling_stats['Conference'] == conference_name].copy()
        
        if conf_teams.empty:
            raise HTTPException(status_code=404, detail="Conference not found")
        
        conf_teams = conf_teams.sort_values('Rating', ascending=False).reset_index(drop=True)
        
        return {
            "teams": conf_teams.to_dict('records'),
            "conference": conference_name,
            "date": data_date
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error getting conference standings: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def calculate_series_probabilities(win_prob):
    """Calculate series win probabilities for 3-game series"""
    # Team A win probabilities
    P_A_0 = (1 - win_prob) ** 3
    P_A_1 = 3 * win_prob * (1 - win_prob) ** 2
    P_A_2 = 3 * win_prob ** 2 * (1 - win_prob)
    P_A_3 = win_prob ** 3

    # Team B win probabilities
    lose_prob = 1 - win_prob
    P_B_0 = win_prob ** 3
    P_B_1 = 3 * lose_prob * win_prob ** 2
    P_B_2 = 3 * lose_prob ** 2 * win_prob
    P_B_3 = lose_prob ** 3

    # Summing for at least conditions
    P_A_at_least_1 = 1 - P_A_0
    P_A_at_least_2 = P_A_2 + P_A_3
    P_B_at_least_1 = 1 - P_B_0
    P_B_at_least_2 = P_B_2 + P_B_3

    return [P_A_at_least_1, P_A_at_least_2, P_A_3], [P_B_at_least_1, P_B_at_least_2, P_B_3]

def get_total_record(row):
    """Get total record from quadrant records"""
    try:
        wins = sum(int(str(row[col]).split("-")[0]) for col in ["Q1", "Q2", "Q3", "Q4"])
        losses = sum(int(str(row[col]).split("-")[1]) for col in ["Q1", "Q2", "Q3", "Q4"])
        return f"{wins}-{losses}"
    except:
        return "0-0"

def get_text_color_for_bubble(color_rgba):
    """Determine if text should be black or white based on background color luminance"""
    # Extract RGB values (color_rgba is a tuple of (r, g, b, a) with values 0-1)
    r, g, b = color_rgba[0], color_rgba[1], color_rgba[2]
    
    # Calculate relative luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Return 'white' for dark backgrounds, 'black' for light backgrounds
    return 'white' if luminance < 0.5 else 'black'

def darken_color(color, factor=0.3):
    """Darken a color for edge effects"""
    color = mcolors.to_rgba(color)
    darkened_color = [max(c - factor, 0) for c in color[:3]]
    return mcolors.rgb2hex(darkened_color)

@app.post("/api/cbase/matchup-image")
def generate_matchup_image(request: BaseballSpreadRequest):
    """Generate matchup comparison image"""
    try:
        modeling_stats, _ = load_baseball_data()
        
        away_team = request.away_team
        home_team = request.home_team
        location = "Neutral" if request.neutral else "Home"
        
        # Get team data
        team1_data = modeling_stats[modeling_stats['Team'] == home_team]
        team2_data = modeling_stats[modeling_stats['Team'] == away_team]
        
        if team1_data.empty or team2_data.empty:
            raise HTTPException(status_code=404, detail="Team not found")
        
        # Load team logos
        logo_folder = os.path.join(BASEBALL_BASE_PATH, "logos")
        team1_logo = None
        team2_logo = None
        
        if os.path.exists(logo_folder):
            # Try to find logos for both teams (keep spaces, don't replace with underscores)
            team1_logo_path = os.path.join(logo_folder, f"{home_team}.png")
            team2_logo_path = os.path.join(logo_folder, f"{away_team}.png")
            
            if os.path.exists(team1_logo_path):
                team1_logo = Image.open(team1_logo_path).convert("RGBA")
            if os.path.exists(team2_logo_path):
                team2_logo = Image.open(team2_logo_path).convert("RGBA")
        
        # Percentile columns
        percentile_columns = ['pNET_Score', 'pRating', 'pResume_Quality', 'pPYTHAG', 'pfWAR', 
                             'pwOBA', 'pOPS', 'pISO', 'pBB%', 'pFIP', 'pWHIP', 'pLOB%', 'pK/BB']
        custom_labels = ['NET', 'TSR', 'RQI', 'PWP', 'WAR', 'wOBA', 'OPS', 'ISO', 'BB%', 'FIP', 'WHIP', 'LOB%', 'K/BB']
        
        # Extract team 1 (home) data
        team1_record = get_total_record(team1_data.iloc[0])
        team1_proj_record = team1_data['Projected_Record'].values[0] if 'Projected_Record' in team1_data.columns else "N/A"
        team1_rating = team1_data['Rating'].values[0]
        team1_net = team1_data['NET'].values[0]
        team1_Q1 = team1_data['Q1'].values[0]
        team1_Q2 = team1_data['Q2'].values[0]
        team1_Q3 = team1_data['Q3'].values[0]
        team1_Q4 = team1_data['Q4'].values[0]
        
        # Extract team 2 (away) data
        team2_record = get_total_record(team2_data.iloc[0])
        team2_proj_record = team2_data['Projected_Record'].values[0] if 'Projected_Record' in team2_data.columns else "N/A"
        team2_rating = team2_data['Rating'].values[0]
        team2_net = team2_data['NET'].values[0]
        team2_Q1 = team2_data['Q1'].values[0]
        team2_Q2 = team2_data['Q2'].values[0]
        team2_Q3 = team2_data['Q3'].values[0]
        team2_Q4 = team2_data['Q4'].values[0]
        
        # Get available percentile columns
        available_percentile_cols = [col for col in percentile_columns if col in modeling_stats.columns]
        
        team1_percentiles = team1_data[available_percentile_cols].values[0] if available_percentile_cols else [50] * len(percentile_columns)
        team2_percentiles = team2_data[available_percentile_cols].values[0] if available_percentile_cols else [50] * len(percentile_columns)
        
        # Calculate win probabilities
        home_pr = team1_rating
        away_pr = team2_rating
        home_elo = team1_data['ELO'].values[0] if 'ELO' in team1_data.columns else 1200
        away_elo = team2_data['ELO'].values[0] if 'ELO' in team2_data.columns else 1200
        
        spread, elo_win_prob = calculate_baseball_spread(home_pr, away_pr, home_elo, away_elo, location)
        win_prob = baseball_win_prob(home_pr, away_pr, location)
        
        # Format spread
        if spread >= 0:
            spread_text = f"{home_team} -{abs(spread)}"
        else:
            spread_text = f"{away_team} -{abs(spread)}"
        
        # Calculate GQI
        max_net = 299
        w_tq = 0.70
        w_wp = 0.20
        w_ned = 0.10
        avg_net = (team1_net + team2_net) / 2
        tq = (max_net - avg_net) / (max_net - 1)
        wp_calc = 1 - 2 * np.abs((win_prob / 100) - 0.5)
        ned = 1 - (np.abs(team2_net - team1_net) / (max_net - 1))
        gqi = round(10 * (w_tq * tq + w_wp * wp_calc + w_ned * ned), 1)
        
        # Calculate series probabilities
        team1_win_prob = win_prob / 100
        team2_win_prob = 1 - team1_win_prob
        team1_probs, team2_probs = calculate_series_probabilities(team1_win_prob)
        
        # Calculate win quality
        bubble_team_rating = modeling_stats['Rating'].quantile(0.90)
        team1_quality = 1 - baseball_win_prob(team2_rating, bubble_team_rating, location) / 100
        team1_win_quality = 1 - team1_quality
        team1_loss_quality = -team1_quality
        
        team2_quality = baseball_win_prob(bubble_team_rating, team1_rating, location) / 100
        team2_win_quality = 1 - team2_quality
        team2_loss_quality = -team2_quality
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(8, 10))
        fig.patch.set_facecolor('#CECEB2')
        ax.set_facecolor('#CECEB2')
        
        # Calculate differences for center bars
        percentile_diffs = [team1_percentiles[i] - team2_percentiles[i] for i in range(len(team1_percentiles))]
        
        # Create colormap
        cmap = plt.get_cmap('seismic')
        colors = [cmap(abs(p) / 100) for p in percentile_diffs]
        colors1 = [cmap(p / 100) for p in team1_percentiles]
        colors2 = [cmap(p / 100) for p in team2_percentiles]
        
        # Create darkened colors for edges
        darkened_colors = [darken_color(c) for c in colors]
        darkened_colors1 = [darken_color(c) for c in colors1]
        darkened_colors2 = [darken_color(c) for c in colors2]
        
        # Draw background gray bars
        ax.barh(range(len(percentile_diffs)), 99, color='gray', height=0.1, left=0)
        ax.barh(range(len(percentile_diffs)), -99, color='gray', height=0.1, left=0)
        
        # Draw main bars
        bars = ax.barh(range(len(percentile_diffs)), percentile_diffs, color=colors, height=0.3, 
                       edgecolor=darkened_colors, linewidth=3)
        bars1 = ax.barh(range(len(team1_percentiles)), team1_percentiles, color=colors1, height=0.3, 
                        edgecolor=darkened_colors1, linewidth=3)
        bars2 = ax.barh(range(len(team2_percentiles)), [-p for p in team2_percentiles], color=colors2, height=0.3, 
                        edgecolor=darkened_colors2, linewidth=3)
        
        # Add labels for team1 (home)
        for i, (bar, percentile) in enumerate(zip(bars1, team1_percentiles)):
            text_color = get_text_color_for_bubble(colors1[i])
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
                   str(int(percentile)), ha='center', va='center',
                   fontsize=16, fontweight='bold', color=text_color, zorder=2,
                   bbox=dict(facecolor=colors1[i], edgecolor=darkened_colors1[i], 
                            boxstyle='circle,pad=0.4', linewidth=3))
            ax.text(0, bar.get_y() - 0.35, custom_labels[i], fontsize=12, 
                   fontweight='bold', ha='center', va='center')
        
        # Add labels for team2 (away)
        for i, (bar, percentile) in enumerate(zip(bars2, team2_percentiles)):
            text_color = get_text_color_for_bubble(colors2[i])
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
                   str(int(percentile)), ha='center', va='center',
                   fontsize=16, fontweight='bold', color=text_color, zorder=2,
                   bbox=dict(facecolor=colors2[i], edgecolor=darkened_colors2[i], 
                            boxstyle='circle,pad=0.4', linewidth=3))
        
        # Add labels for difference bars
        for i, (bar, diff) in enumerate(zip(bars, percentile_diffs)):
            text_color = get_text_color_for_bubble(colors[i])
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
                   str(int(abs(diff))), ha='center', va='center',
                   fontsize=14, fontweight='bold', color=text_color, zorder=2,
                   bbox=dict(facecolor=colors[i], edgecolor=darkened_colors[i], 
                            boxstyle='circle,pad=0.3', linewidth=3))
        
        ax.set_xlim(-104, 104)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add title and info
        plt.text(0, -1.7, f"#{team2_net} {away_team} vs. #{team1_net} {home_team}", 
                ha='center', fontsize=20, fontweight='bold')
        plt.text(0, -1.25, f"Game Quality: {gqi}", ha='center', fontsize=16, fontweight='bold')
        plt.text(0, -0.8, spread_text, ha='center', fontsize=16, fontweight='bold')
        plt.text(0, 12.8, "@PEARatings", ha='center', fontsize=16, fontweight='bold')
        
        # Team 2 (away) info - left side
        plt.text(-135, 0.5, away_team, ha='center', fontsize=16, fontweight='bold')
        plt.text(-135, 1.0, team2_record, ha='center', fontsize=16)
        plt.text(-135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
        plt.text(-135, 2.5, f"{round(team2_win_prob * 100)}%", ha='center', fontsize=16)
        plt.text(-135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
        plt.text(-135, 4.0, f"Win 1: {round(team2_probs[0] * 100)}%", ha='center', fontsize=16)
        plt.text(-135, 4.5, f"Win 2: {round(team2_probs[1] * 100)}%", ha='center', fontsize=16)
        plt.text(-135, 5.0, f"Win 3: {round(team2_probs[2] * 100)}%", ha='center', fontsize=16)
        plt.text(-135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
        plt.text(-150, 6.5, f"Q1: {team2_Q1}", ha='left', fontsize=16)
        plt.text(-150, 7.0, f"Q2: {team2_Q2}", ha='left', fontsize=16)
        plt.text(-150, 7.5, f"Q3: {team2_Q3}", ha='left', fontsize=16)
        plt.text(-150, 8.0, f"Q4: {team2_Q4}", ha='left', fontsize=16)
        plt.text(-135, 9.0, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
        plt.text(-135, 9.5, team2_proj_record, ha='center', fontsize=16)
        plt.text(-135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
        plt.text(-160, 11.0, f"{team2_win_quality:.2f}", ha='left', fontsize=16, 
                color='green', fontweight='bold')
        plt.text(-110, 11.0, f"{team2_loss_quality:.2f}", ha='right', fontsize=16, 
                color='red', fontweight='bold')
        
        # Team 1 (home) info - right side
        plt.text(135, 0.5, home_team, ha='center', fontsize=16, fontweight='bold')
        plt.text(135, 1.0, team1_record, ha='center', fontsize=16)
        plt.text(135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
        plt.text(135, 2.5, f"{round(team1_win_prob * 100)}%", ha='center', fontsize=16)
        plt.text(135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
        plt.text(135, 4.0, f"Win 1: {round(team1_probs[0] * 100)}%", ha='center', fontsize=16)
        plt.text(135, 4.5, f"Win 2: {round(team1_probs[1] * 100)}%", ha='center', fontsize=16)
        plt.text(135, 5.0, f"Win 3: {round(team1_probs[2] * 100)}%", ha='center', fontsize=16)
        plt.text(135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
        plt.text(121, 6.5, f"Q1: {team1_Q1}", ha='left', fontsize=16)
        plt.text(121, 7.0, f"Q2: {team1_Q2}", ha='left', fontsize=16)
        plt.text(121, 7.5, f"Q3: {team1_Q3}", ha='left', fontsize=16)
        plt.text(121, 8.0, f"Q4: {team1_Q4}", ha='left', fontsize=16)
        plt.text(135, 9.0, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
        plt.text(135, 9.5, team1_proj_record, ha='center', fontsize=16)
        plt.text(135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
        plt.text(110, 11.0, f"{team1_win_quality:.2f}", ha='left', fontsize=16, 
                color='green', fontweight='bold')
        plt.text(160, 11.0, f"{team1_loss_quality:.2f}", ha='right', fontsize=16, 
                color='red', fontweight='bold')
        
        # Add explanation text
        plt.text(-155, 13.2, "Middle Bubble is Difference Between Team Percentiles", 
                ha='left', fontsize=12)
        plt.text(155, 13.2, "Series Percentages are the Chance to Win __ Games", 
                ha='right', fontsize=12)
        plt.text(-155, 13.6, "NET - PEAR's Ranking System, Combining TSR and RQI", 
                ha='left', fontsize=12)
        plt.text(155, 13.6, "TSR - Team Strength Rating, How Good Your Team Is", 
                ha='right', fontsize=12)
        plt.text(-155, 14.0, "RQI - Resume Quality Index, How Good Your Wins Are", 
                ha='left', fontsize=12)
        plt.text(155, 14.0, "PWP - Pythagorean Win Percent, Expected Win Rate", 
                ha='right', fontsize=12)
        
        # Add team logos if available
        if team1_logo:
            ax_img1 = fig.add_axes([0.94, 0.83, 0.15, 0.15])
            ax_img1.imshow(team1_logo)
            ax_img1.axis("off")
        
        if team2_logo:
            ax_img2 = fig.add_axes([-0.065, 0.83, 0.15, 0.15])
            ax_img2.imshow(team2_logo)
            ax_img2.axis("off")
        
        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error generating matchup image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

from collections import Counter, defaultdict

def get_conference(team, stats_df):
    """Get conference for a team"""
    return stats_df.loc[stats_df["Team"] == team, "Conference"].values[0]

def count_conflict_conferences(teams, stats_df):
    """Count conference conflicts in a regional"""
    conferences = [get_conference(team, stats_df) for team in teams]
    return sum(count - 1 for count in Counter(conferences).values() if count > 1)

def resolve_conflicts(formatted_df, stats_df):
    """Resolve conference conflicts by swapping seeds between regionals"""
    seed_cols = ["seed_2", "seed_3", "seed_4"]

    for seed_col in seed_cols:
        num_regionals = len(formatted_df)

        for i in range(num_regionals):
            row = formatted_df.iloc[i]
            teams_i = [row["seed_1"], row["seed_2"], row["seed_3"], row["seed_4"]]
            conflict_i = count_conflict_conferences(teams_i, stats_df)

            if conflict_i == 0:
                continue

            current_team = row[seed_col]

            for j in range(num_regionals):
                if i == j:
                    continue

                alt_team = formatted_df.at[j, seed_col]
                if alt_team == current_team:
                    continue

                row_j = formatted_df.iloc[j]
                teams_j = [row_j["seed_1"], row_j["seed_2"], row_j["seed_3"], row_j["seed_4"]]

                temp_i = teams_i.copy()
                temp_j = teams_j.copy()
                temp_i[seed_cols.index(seed_col) + 1] = alt_team
                temp_j[seed_cols.index(seed_col) + 1] = current_team

                new_conflict_i = count_conflict_conferences(temp_i, stats_df)
                new_conflict_j = count_conflict_conferences(temp_j, stats_df)

                if (new_conflict_i + new_conflict_j) < (conflict_i + count_conflict_conferences(teams_j, stats_df)):
                    formatted_df.at[i, seed_col] = alt_team
                    formatted_df.at[j, seed_col] = current_team
                    break

    return formatted_df

@app.get("/api/cbase/tournament-outlook")
def get_tournament_outlook():
    """Generate projected NCAA Tournament bracket"""
    try:
        modeling_stats, _ = load_baseball_data()
        
        # Hardcoded lists from the streamlit app
        aq_list = ["Binghamton", "East Carolina", "Stetson", "Rhode Island", "North Carolina", "Arizona",
                "Creighton", "USC Upstate", "Nebraska", "Cal Poly", "Northeastern", "Western Ky.", "Wright St.",
                "Columbia", "Fairfield", "Miami (OH)", "Murray St.", "Fresno St.",
                "Central Conn. St.", "Little Rock", "Holy Cross", "Vanderbilt", "Houston Christian",
                "ETSU", "Bethune-Cookman", "North Dakota St.", "Coastal Carolina", "Saint Mary's (CA)", "Utah Valley", "Oregon St."]
        
        host_seeds_list = ["Georgia", "Auburn", "Texas", "LSU", "North Carolina", "Clemson", "Coastal Carolina", "Oregon St.",
                    "Oregon", "Arkansas", "Southern Miss.", "Tennessee", "UCLA", "Vanderbilt", "Ole Miss", "Florida St."]
        
        # Get automatic qualifiers and host seeds
        automatic_qualifiers = (
            modeling_stats[modeling_stats["Team"].isin(aq_list)]
            .sort_values("NET")
        )
        
        host_seeds = (
            modeling_stats[modeling_stats["Team"].isin(host_seeds_list)]
            .sort_values("NET")
        )
        
        # Calculate at-large bids
        amount_of_at_large = 64 - len(set(automatic_qualifiers["Team"]) - set(host_seeds["Team"])) - len(host_seeds)
        
        at_large = modeling_stats.drop(automatic_qualifiers.index)
        at_large = at_large[~at_large["Team"].isin(host_seeds_list)]
        automatic_qualifiers = automatic_qualifiers[~automatic_qualifiers["Team"].isin(host_seeds_list)]
        at_large = at_large.nsmallest(amount_of_at_large, "NET")
        
        # Get bubble teams
        last_four_in = at_large[-4:].reset_index()
        next_8 = modeling_stats.drop(automatic_qualifiers.index)
        next_8 = next_8[~next_8["Team"].isin(host_seeds_list)]
        next_8_teams = next_8.nsmallest(amount_of_at_large + 8, "NET").iloc[amount_of_at_large:].reset_index(drop=True)
        
        # Build tournament bracket
        remaining_teams = pd.concat([automatic_qualifiers, at_large]).sort_values("NET").reset_index(drop=True)
        
        seed_1_df = host_seeds.sort_values("NET").reset_index(drop=True)
        seed_2_df = remaining_teams.iloc[0:16].sort_values("NET", ascending=False).copy()
        seed_3_df = remaining_teams.iloc[16:32].copy()
        seed_4_df = remaining_teams.iloc[32:48].sort_values("NET", ascending=False).copy()
        
        # Create formatted dataframe
        formatted_df = pd.DataFrame({
            'host': seed_1_df['Team'].values,
            'seed_1': seed_1_df['Team'].values,
            'seed_2': seed_2_df['Team'].values,
            'seed_3': seed_3_df['Team'].values,
            'seed_4': seed_4_df['Team'].values
        })
        
        # Resolve conflicts
        formatted_df = resolve_conflicts(formatted_df, modeling_stats)
        
        # Get multi-bid conferences
        all_teams = pd.unique(formatted_df[["seed_1", "seed_2", "seed_3", "seed_4"]].values.ravel())
        bracket_teams = modeling_stats[modeling_stats["Team"].isin(all_teams)]
        conference_counts = bracket_teams["Conference"].value_counts()
        multibid = conference_counts[conference_counts > 1]
        
        # Format response
        regionals = []
        for i, row in formatted_df.iterrows():
            regionals.append({
                "regional_number": i + 1,
                "host": row["host"],
                "seed_1": row["seed_1"],
                "seed_2": row["seed_2"],
                "seed_3": row["seed_3"],
                "seed_4": row["seed_4"]
            })
        
        return {
            "regionals": regionals,
            "last_four_in": last_four_in['Team'].tolist(),
            "first_four_out": next_8_teams.iloc[0:4]['Team'].tolist(),
            "next_four_out": next_8_teams.iloc[4:8]['Team'].tolist(),
            "multibid_conferences": multibid.to_dict(),
            "automatic_qualifiers": aq_list
        }
    
    except Exception as e:
        print(f"Error generating tournament outlook: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def simulate_tournament_home_field(teams, ratings):
    """Simulate a single double-elimination regional tournament"""
    import random
    
    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)

    team_a, team_b, team_c, team_d = teams
    r = ratings

    def adjusted(team):
        return r[team] + 0.3 if team == team_a else r[team]

    # Game 1: #1 vs #4
    w1, l1 = (team_a, team_d) if random.random() < PEAR_Win_Prob(adjusted(team_a), adjusted(team_d)) else (team_d, team_a)
    # Game 2: #2 vs #3
    w2, l2 = (team_b, team_c) if random.random() < PEAR_Win_Prob(adjusted(team_b), adjusted(team_c)) else (team_c, team_b)
    # Game 3: Loser's bracket
    w3 = l2 if random.random() < PEAR_Win_Prob(adjusted(l2), adjusted(l1)) else l1
    # Game 4: Winner's bracket
    w4, l4 = (w1, w2) if random.random() < PEAR_Win_Prob(adjusted(w1), adjusted(w2)) else (w2, w1)
    # Game 5: Loser's bracket final
    w5 = l4 if random.random() < PEAR_Win_Prob(adjusted(l4), adjusted(w3)) else w3
    # Game 6: Championship
    game6_prob = PEAR_Win_Prob(adjusted(w4), adjusted(w5))
    w6 = w4 if random.random() < game6_prob else w5

    # If from loser's bracket, need to win twice
    return w6 if w6 == w4 else (w4 if random.random() < game6_prob else w5)

def run_simulation_home_field(team_a, team_b, team_c, team_d, stats_and_metrics, num_simulations=5000):
    """Run multiple simulations of a regional tournament"""
    teams = [team_a, team_b, team_c, team_d]
    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}
    results = defaultdict(int)

    for _ in range(num_simulations):
        winner = simulate_tournament_home_field(teams, ratings)
        results[winner] += 1

    total = num_simulations
    return defaultdict(float, {team: round(count / total, 3) for team, count in results.items()})

class RegionalSimulationRequest(BaseModel):
    seed_1: str
    seed_2: str
    seed_3: str
    seed_4: str

@app.post("/api/cbase/simulate-regional")
def simulate_regional(request: RegionalSimulationRequest):
    """Simulate a regional tournament and return visualization"""
    try:
        modeling_stats, _ = load_baseball_data()
        
        team_a = request.seed_1  # Host
        team_b = request.seed_2
        team_c = request.seed_3
        team_d = request.seed_4
        
        # Verify all teams exist
        for team in [team_a, team_b, team_c, team_d]:
            if team not in modeling_stats['Team'].values:
                raise HTTPException(status_code=404, detail=f"Team not found: {team}")
        
        # Run simulation
        output = run_simulation_home_field(team_a, team_b, team_c, team_d, modeling_stats)
        
        # Format results
        regional_prob = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
        seed_map = {
            team_a: f"#1 {team_a}",
            team_b: f"#2 {team_b}",
            team_c: f"#3 {team_c}",
            team_d: f"#4 {team_d}",
        }
        regional_prob["Team"] = regional_prob["Team"].map(seed_map)
        regional_prob['Win Regional'] = regional_prob['Win Regional'] * 100
        seed_order = [seed_map[team_a], seed_map[team_b], seed_map[team_c], seed_map[team_d]]
        regional_prob["SeedOrder"] = regional_prob["Team"].apply(lambda x: seed_order.index(x))
        regional_prob = regional_prob.sort_values("SeedOrder").drop(columns="SeedOrder")

        # Normalize values for color gradient
        min_value = regional_prob.iloc[:, 1:].replace(0, np.nan).min().min()
        max_value = regional_prob.iloc[:, 1:].max().max()

        def normalize(value, min_val, max_val):
            if pd.isna(value) or value == 0:
                return 0
            return (value - min_val) / (max_val - min_val)

        # Create visualization
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

        fig, ax = plt.subplots(figsize=(4, 4), dpi=125)
        fig.patch.set_facecolor('#CECEB2')

        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=regional_prob.values,
            colLabels=regional_prob.columns,
            cellLoc='center',
            loc='center',
            colColours=['#CECEB2'] * len(regional_prob.columns)
        )
        
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(1.2)
            if i == 0:
                cell.set_facecolor('#CECEB2')
                cell.set_text_props(fontsize=14, weight='bold', color='black')
            elif j == 0:
                cell.set_facecolor('#CECEB2')
                cell.set_text_props(fontsize=14, weight='bold', color='black')
            else:
                value = regional_prob.iloc[i-1, j]
                normalized_value = normalize(value, min_value, max_value)
                color = cmap(normalized_value)
                cell.set_facecolor(color)
                cell.set_text_props(fontsize=14, weight='bold', color='black')
                if value == 0:
                    cell.get_text().set_text("<1%")
                else:
                    cell.get_text().set_text(f"{value:.1f}%")
            cell.set_height(0.2)

        plt.text(0, 0.07, f'{team_a} Regional', fontsize=16, fontweight='bold', ha='center')
        plt.text(0, 0.06, f"@PEARatings", fontsize=12, fontweight='bold', ha='center')

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error simulating regional: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =====================================================================
# NEW: Team Profile and Historical Performance Endpoints
# =====================================================================

class TeamProfileRequest(BaseModel):
    team_name: str

class HistoricalPerformanceRequest(BaseModel):
    team_name: str

def load_schedule_data():
    """Load schedule data for baseball"""
    try:
        baseball_path = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Baseball")
        schedule_path = os.path.join(baseball_path, f"y{current_season}", f"schedule_{current_season}.csv")
        schedule_df = pd.read_csv(schedule_path)
        schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
        return schedule_df
    except Exception as e:
        print(f"Error loading schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading schedule: {str(e)}")

def load_historical_data():
    """Load all historical baseball data"""
    try:
        baseball_path = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Baseball")
        seasons = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                   2018, 2019, 2021, 2022, 2023, 2024, 2025]
        
        all_data_list = []
        date_pattern = re.compile(r"baseball_(\d{2}_\d{2}_\d{4})\.csv")
        
        for year in seasons:
            folder = os.path.join(baseball_path, f"y{year}", "Data")
            pattern = os.path.join(folder, "baseball_*.csv")
            files = glob.glob(pattern)
            
            valid_files = []
            for f in files:
                filename = os.path.basename(f)
                match = date_pattern.match(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        file_date = datetime.strptime(date_str, "%m_%d_%Y")
                        if file_date.month >= 6:
                            valid_files.append((f, file_date))
                    except ValueError:
                        continue
            
            if not valid_files:
                continue
            
            latest_file = max(valid_files, key=lambda x: x[1])[0]
            
            try:
                df = pd.read_csv(latest_file)
                df["Season"] = year
                all_data_list.append(df)
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
                continue
        

        all_data = pd.concat(all_data_list, ignore_index=True)
        all_data['Normalized_Rating'] = all_data.groupby('Season')['Rating'].transform(
            lambda x: (x - x.mean()) / x.std()
        )

        all_data['Normalized_Rating'] = all_data['Normalized_Rating'] * 2.5
        all_data['Normalized_Rating'] = round(all_data['Normalized_Rating'] - all_data['Normalized_Rating'].mean(),2)
        all_data['Normalized_Rating'] = round(all_data['Normalized_Rating'], 2)
        all_data = all_data.sort_values('Normalized_Rating', ascending=False).reset_index(drop=True)
        all_data['Season'] = all_data['Season'].astype(int)
        return all_data
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading historical data: {str(e)}")

def PEAR_Win_Prob_Baseball(home_pr, away_pr, location="Neutral"):
    """Baseball-specific win probability calculation"""
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

def get_total_record(row):
    """Calculate total win-loss record from quad records"""
    wins = sum(int(str(row[col]).split("-")[0]) for col in ["Q1", "Q2", "Q3", "Q4"])
    losses = sum(int(str(row[col]).split("-")[1]) for col in ["Q1", "Q2", "Q3", "Q4"])
    return f"{wins}-{losses}"

def get_location_records(team, schedule_df):
    """Get home, away, and neutral records for a team"""
    df = schedule_df[
        ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
        (schedule_df['Result'].str.startswith(('W', 'L')))
    ].copy()
    
    def get_loc(row):
        if row['Location'] == 'Neutral':
            return 'Neutral'
        elif row['home_team'] == team:
            return 'Home'
        else:
            return 'Away'
    
    df['loc'] = df.apply(get_loc, axis=1)
    df['is_win'] = df['Result'].str.startswith('W')
    
    records = {}
    for loc in ['Home', 'Away', 'Neutral']:
        group = df[df['loc'] == loc]
        wins = group['is_win'].sum()
        losses = len(group) - wins
        records[loc] = f"{int(wins)}-{int(losses)}"
    
    return records

def get_conference_record(team, schedule_df, stats_and_metrics):
    """Calculate conference record for a team"""
    team_to_conf = stats_and_metrics.set_index('Team')['Conference'].to_dict()
    team_conf = team_to_conf.get(team)
    schedule_df['home_conf'] = schedule_df['home_team'].map(team_to_conf)
    schedule_df['away_conf'] = schedule_df['away_team'].map(team_to_conf)
    schedule_df["matchup"] = schedule_df["home_team"] + " vs " + schedule_df["away_team"]
    matchup = schedule_df["matchup"].values
    home_conf = schedule_df["home_conf"].values
    away_conf = schedule_df["away_conf"].values
    
    match0 = matchup[:-2]
    match1 = matchup[1:-1]
    match2 = matchup[2:]
    conf_check_0 = home_conf[:-2] == away_conf[:-2]
    conf_check_1 = home_conf[1:-1] == away_conf[1:-1]
    conf_check_2 = home_conf[2:] == away_conf[2:]
    valid_series = (
        (match0 == match1) & (match1 == match2) &
        conf_check_0 & conf_check_1 & conf_check_2
    )
    base_indices = np.where(valid_series)[0]
    valid_indices = np.unique(np.concatenate([base_indices, base_indices + 1, base_indices + 2]))
    df = schedule_df.iloc[valid_indices].reset_index(drop=True)
    
    conf_games = df[
        (df['home_conf'] == team_conf) &
        (df['away_conf'] == team_conf) &
        (df['Result'].str.startswith(('W', 'L')))
    ]
    wins = conf_games['Result'].str.startswith('W')
    wins_count = int(wins.sum())
    games_count = int(len(conf_games))
    
    return f"{wins_count}-{games_count - wins_count}"

def get_metric_values(teams, column):
    values = []
    for team in teams:
        try:
            val = team[column].values[0]
            values.append(int(val))
        except:
            values.append("N/A")
    return values

def simulate_team_win_distribution(schedule_df, comparison_date, team_name, num_simulations=1000):
    # Ensure "Date" is datetime
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])

    # --- Step 1: Filter to games involving the specified team ---
    team_games = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True).copy()

    # --- Step 2: Split into completed and remaining games ---
    completed_games = team_games[
        (team_games["Date"] <= comparison_date) & (team_games["home_score"].notnull()) & (team_games["away_score"].notnull())
    ].copy()

    remaining_games = team_games[
        (team_games["Date"] >= comparison_date) & (team_games["home_win_prob"].notnull() & (team_games['home_score'] == team_games['away_score']))
    ].copy()

    # --- Step 3: Calculate current win total ---
    completed_games["winner"] = np.where(
        completed_games["home_score"] > completed_games["away_score"],
        completed_games["home_team"],
        completed_games["away_team"]
    )
    current_wins = (completed_games["winner"] == team_name).sum()

    # --- Step 4: Simulate outcomes of remaining games ---
    home_teams = remaining_games["home_team"].values
    away_teams = remaining_games["away_team"].values
    home_win_probs = remaining_games["home_win_prob"].values

    simulations = []
    for _ in range(num_simulations):
        random_vals = np.random.rand(len(remaining_games))
        home_wins = random_vals < home_win_probs
        winners = np.where(home_wins, home_teams, away_teams)

        sim_wins = (winners == team_name).sum()
        total_wins = current_wins + sim_wins
        simulations.append(total_wins)

    simulations = np.array(simulations)

    # Output: Series with counts of each win total
    win_distribution = pd.Series(simulations).value_counts().sort_index()

    return win_distribution

@app.post("/api/cbase/team-profile")
def team_profile(request: TeamProfileRequest):
    """Generate team profile visualization"""
    try:
        stats_and_metrics, comparison_date = load_baseball_data()
        schedule_df = load_schedule_data()
        team_name = request.team_name
        
        BASE_URL = "https://www.warrennolan.com"
        completed_schedule = schedule_df[
            (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
        ].reset_index(drop=True)
        team_schedule = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True)
        team_data = stats_and_metrics[stats_and_metrics['Team'] == team_name]
        team_net = team_data['NET'].values[0]
        team_conference = team_data['Conference'].values[0]
        team_record = get_total_record(team_data.iloc[0])
        Conf_Record = get_conference_record(team_name, team_schedule, stats_and_metrics)
        team_Q1 = team_data['Q1'].values[0]
        team_Q2 = team_data['Q2'].values[0]
        team_Q3 = team_data['Q3'].values[0]
        team_Q4 = team_data['Q4'].values[0]
        team_rpi = team_data['RPI'].values[0]
        team_elo = int(team_data['ELO_Rank'].values[0])
        team_rqi = team_data['RQI'].values[0]
        team_tsr = team_data['PRR'].values[0]
        team_sos = team_data['SOS'].values[0]
        record = get_location_records(team_name, team_schedule)
        home_record = record['Home']
        away_record = record['Away']
        neutral_record = record['Neutral']
        
        fig, ax = plt.subplots(figsize=(8, 10), dpi=500) # , dpi=500
        fig.patch.set_facecolor('#CECEB2')
        ax.set_facecolor('#CECEB2')

        # percentile sliders code
        percentile_columns = ['pNET_Score', 'pRating', 'pResume_Quality', 'pPYTHAG', 'pfWAR', 'pwOBA', 'pOPS', 'pISO', 'pBB%', 'pFIP', 'pWHIP', 'pLOB%', 'pK/BB']
        team_data = team_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
        cmap = plt.get_cmap('seismic')
        colors = [cmap(p / 100) for p in team_data['Percentile']]
        def darken_color(color, factor=0.3):
            color = mcolors.hex2color(color)
            darkened_color = [max(c - factor, 0) for c in color]
            return mcolors.rgb2hex(darkened_color)
        darkened_colors = [darken_color(c) for c in colors]
        ax.barh(team_data['Metric'], 99, color='gray', height=0.1, left=0)
        bars = ax.barh(team_data['Metric'], team_data['Percentile'], color=colors, height=0.6, edgecolor=darkened_colors, linewidth=3)
        i = 0
        for idx, (bar, percentile) in enumerate(zip(bars, team_data['Percentile'])):
            text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        str(percentile), ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='white', zorder=2,
                        bbox=dict(facecolor=colors[i], edgecolor=darkened_colors[i], boxstyle='circle,pad=0.4', linewidth=3))
            text.set_path_effects([
                pe.withStroke(linewidth=2, foreground='black')
            ])
            if idx == 4 or idx == 8:  # Check if the index is 5th or 9th bar (0-based index)
                y_position = bar.get_y() + bar.get_height() + 0.185
                ax.hlines(y_position, 0, 99,
                        colors='black', linestyles='dashed', linewidth=2, zorder=1)
                    
            i = i + 1
        ax.set_xlim(0, 102)
        ax.set_xticks([])
        custom_labels = ['NET', 'TSR', 'RQI', 'PWP', 'fWAR', 'wOBA', 'OPS', 'ISO', 'BB%', 'FIP', 'WHIP', 'LOB%', 'K/BB']
        ax.set_yticks(range(len(custom_labels)))
        ax.set_yticklabels(custom_labels, fontweight='bold', fontsize=16)
        ax.tick_params(axis='y', which='both', length=0, pad=14)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # team logo - must stay above all text calls
        logo_folder = os.path.join(BASEBALL_BASE_PATH, "logos")
        logo_path = os.path.join(logo_folder, f"{team_name}.png")
        img = Image.open(logo_path).convert("RGBA")
        ax_img1 = fig.add_axes([0.04, 0.85, 0.2, 0.2])
        ax_img1.imshow(img)
        ax_img1.axis("off")

        ### PLOT TITLE
        plt.text(0.18, 1.16, f'#{team_net} {team_name}', fontsize=34, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        plt.text(0.18, 1.11, f"{team_record} ({Conf_Record})", fontsize=24, ha='left', va='center', transform=ax.transAxes)
        plt.text(0.18, 1.06, f'@PEARatings', fontsize=24, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        plt.text(0.18, 1.01, f'Team Profile', fontsize=24, ha='left', va='center', transform=ax.transAxes)

        ### TEAM SCHEDULE
        def get_opponent_net(row, team):
            if row['home_team'] == team:
                return row['away_net']
            elif row['away_team'] == team:
                return row['home_net']
            else:
                return np.nan

        team_schedule['opponent_net'] = team_schedule.apply(lambda row: get_opponent_net(row, team_name), axis=1)

        conditions = [
            ((team_schedule["Location"] == "Home") & (team_schedule["opponent_net"] <= 25)) |
            ((team_schedule["Location"] == "Neutral") & (team_schedule["opponent_net"] <= 40)) |
            ((team_schedule["Location"] == "Away") & (team_schedule["home_net"] <= 60)),

            ((team_schedule["Location"] == "Home") & (team_schedule["opponent_net"] <= 50)) |
            ((team_schedule["Location"] == "Neutral") & (team_schedule["opponent_net"] <= 80)) |
            ((team_schedule["Location"] == "Away") & (team_schedule["opponent_net"] <= 120)),

            ((team_schedule["Location"] == "Home") & (team_schedule["opponent_net"] <= 100)) |
            ((team_schedule["Location"] == "Neutral") & (team_schedule["opponent_net"] <= 160)) |
            ((team_schedule["Location"] == "Away") & (team_schedule["opponent_net"] <= 240))
        ]

        # Define corresponding quadrant labels
        quadrants = ["Q1", "Q2", "Q3"]

        # Assign Quadrant values
        team_schedule["Quad"] = np.select(conditions, quadrants, default="Q4")
        num_items = len(team_schedule)
        schedule_x = 0.9
        schedule_y = 0.95
        schedule_size = 15
        counter = 0
        columns = 0
        best_rq_row = None
        worst_rq_row = None
        max_rq = float('-inf')
        min_rq = float('inf')
        for idx, (_, row) in enumerate(team_schedule.iterrows()):
            if row['resume_quality'] > max_rq and row['Result'].startswith("W"):
                max_rq = row['resume_quality']
                best_rq_row = row
            if row['resume_quality'] < min_rq and row['Result'].startswith("L"):
                min_rq = row['resume_quality']
                worst_rq_row = row
            if counter % 15 == 0:
                schedule_x +=0.35
                schedule_y = 0.95
                columns += 1
            if row['home_team'] == team_name:
                opponent = row['away_team']
                net = row['away_net']
                win_prob = row['home_win_prob']
                symbol = ""
            else:
                opponent = row['home_team']
                net = row['home_net']
                win_prob = 1 - row['home_win_prob']
                symbol = "@"
            if row['Location'] == "Neutral":
                symbol = "vs"
            if "Non Div I" in opponent:
                opponent = "Non Div I"
            if pd.notna(net):
                net = int(net)
            if row['resume_quality'] < 0:
                color = '#8B0000' #red
            else:
                color = '#2C5E00' #green
            # # ax.text(0.5, 0.8, opponent, ha='center', va='center', fontsize=40, fontweight='bold', color=color)
            # # ax.text(0.1, 0.3, f'#{net}', ha='left', va='center', fontsize=32)
            # # ax.text(0.5, 0.5, row['Quad'], ha='right', va='center', fontsize=32, fontweight='bold')
            result_first_letter = row['Result'][0].upper() if row['Result'][0].upper() in ['W', 'L'] else ''

            if result_first_letter:
                if (row['home_team'] == team_name) & (row['Location'] == "Home"):
                    plt.text(schedule_x, schedule_y, f'{opponent}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color=color, transform=ax.transAxes)
                else:
                    plt.text(schedule_x, schedule_y, f'{symbol} {opponent}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color=color, transform=ax.transAxes)
                ax.text(schedule_x, schedule_y-0.026, f'{row["Quad"]} | {round(win_prob*100)}% | {row["resume_quality"]:.2f}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color=color, transform=ax.transAxes)
            else:
                if (row['home_team'] == team_name) & (row['Location'] == "Home"):
                    ax.text(schedule_x, schedule_y, f'{opponent}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color='#555555', transform=ax.transAxes)
                else:
                    ax.text(schedule_x, schedule_y, f'{symbol} {opponent}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color='#555555', transform=ax.transAxes)
                ax.text(schedule_x, schedule_y-0.026, f'{row["Quad"]} | {round(win_prob*100)}% | {1 - abs(row["resume_quality"]):.2f}', ha='center', va='center', fontsize=schedule_size, fontweight='bold', color='#555555', transform=ax.transAxes)
            schedule_y = schedule_y - 0.062
            counter += 1

        ### TOP TEXT

        team_completed = completed_schedule[completed_schedule['Team'] == team_name].reset_index(drop=True)
        num_rows = len(team_completed)
        last_n_games = team_completed['Result'].iloc[-10 if num_rows >= 10 else -num_rows:]
        wins = last_n_games.str.count('W').sum()
        losses = (10 if num_rows >= 10 else num_rows) - wins
        last_ten = f'{wins}-{losses}'
        team_completed['is_home'] = team_completed['home_team'] == team_name
        team_completed['runs_scored'] = team_completed.apply(
            lambda row: row['home_score'] if row['is_home'] else row['away_score'], axis=1
        )
        team_completed['runs_allowed'] = team_completed.apply(
            lambda row: row['away_score'] if row['is_home'] else row['home_score'], axis=1
        )
        run_diff = team_completed['runs_scored'].sub(team_completed['runs_allowed']).mean()

        if columns == 4:
            plt.text(1.20, 0.00, f"Best: {best_rq_row['Quad']} {best_rq_row['Opponent']} {best_rq_row['resume_quality']:.2f}", ha='left', va='center', fontsize=15, fontweight='bold', color='#2C5E00', transform=ax.transAxes)
            plt.text(2.35, 0.00, f"Worst: {worst_rq_row['Quad']} {worst_rq_row['Opponent']} {worst_rq_row['resume_quality']:.2f}", ha='right', va='center', fontsize=15, fontweight='bold', color='#8B0000', transform=ax.transAxes)
            plt.text(1.25, 1.16, f"RPI: {team_rpi}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.60, 1.16, f"ELO: {team_elo}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.95, 1.16, f"RQI: {team_rqi}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.30, 1.16, f"TSR: {team_tsr}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.25, 1.11, f"H: {home_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.60, 1.11, f"A: {away_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.95, 1.11, f"N: {neutral_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.30, 1.11, f"SOS: {team_sos}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.25, 1.06, f"Q1: {team_Q1}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.60, 1.06, f"Q2: {team_Q2}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.95, 1.06, f"Q3: {team_Q3}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.30, 1.06, f"Q4: {team_Q4}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.25, 1.01, f"L10: {last_ten}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.30, 1.01, f"MOV: {run_diff:.1f}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.775, 1.005, "Quad | Win Prob | Resume Points", fontsize=16, ha='center', va='center', transform=ax.transAxes)
        elif columns == 5:
            plt.text(1.20, 0.00, f"Best: {best_rq_row['Quad']} {best_rq_row['Opponent']} {best_rq_row['resume_quality']:.2f}", ha='left', va='center', fontsize=15, fontweight='bold', color='#2C5E00', transform=ax.transAxes)
            plt.text(2.70, 0.00, f"Worst: {worst_rq_row['Quad']} {worst_rq_row['Opponent']} {worst_rq_row['resume_quality']:.2f}", ha='right', va='center', fontsize=15, fontweight='bold', color='#8B0000', transform=ax.transAxes)
            plt.text(1.425, 1.16, f"RPI: {team_rpi}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.775, 1.16, f"ELO: {team_elo}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.125, 1.16, f"RQI: {team_rqi}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.475, 1.16, f"TSR: {team_tsr}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.425, 1.11, f"H: {home_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.775, 1.11, f"A: {away_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.125, 1.11, f"N: {neutral_record}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.475, 1.11, f"SOS: {team_sos}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.425, 1.06, f"Q1: {team_Q1}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.775, 1.06, f"Q2: {team_Q2}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.125, 1.06, f"Q3: {team_Q3}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.475, 1.06, f"Q4: {team_Q4}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.425, 1.01, f"L10: {last_ten}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.475, 1.01, f"MOV: {run_diff:.1f}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.95, 1.005, "Quad | Win Prob | Resume Points", fontsize=16, ha='center', va='center', transform=ax.transAxes)


        automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
        at_large = stats_and_metrics.drop(automatic_qualifiers.index)
        at_large = at_large.nsmallest(34, "NET")
        last_four_in = at_large[-8:].reset_index()
        next_4_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(38, "NET").iloc[34:].reset_index(drop=True)
        projected = ""
        if team_net <= 16:
            projected = "Host"
        elif team_name in last_four_in['Team'].values:
            projected = "Last Four In"
        elif team_name in at_large['Team'].values:
            projected = "At-Large"
        elif team_name in automatic_qualifiers['Team'].values:
            projected = "Autobid"
        elif team_name in next_4_teams['Team'].values:
            projected = "First Four Out"
        else:
            projected = "Miss"
        if columns == 4:
            plt.text(0.82, 1.06, f"Projection:", fontsize=24, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
            plt.text(0.82, 1.01, f"{projected}", fontsize=24, ha='center', va='center', transform=ax.transAxes)
        elif columns == 5:
            plt.text(0.92, 1.06, f"Projection:", fontsize=24, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
            plt.text(0.92, 1.01, f"{projected}", fontsize=24, ha='center', va='center', transform=ax.transAxes)

        ### PREVIOUS YEARS DATA
        data_2022 = pd.read_csv(f"{BASEBALL_BASE_PATH}/y2022/Data/baseball_06_26_2022.csv")
        team_2022 = data_2022[data_2022['Team'] == team_name]
        data_2023 = pd.read_csv(f"{BASEBALL_BASE_PATH}/y2023/Data/baseball_06_26_2023.csv")
        team_2023 = data_2023[data_2023['Team'] == team_name]
        data_2024 = pd.read_csv(f"{BASEBALL_BASE_PATH}/y2024/Data/baseball_06_25_2024.csv")
        team_2024 = data_2024[data_2024['Team'] == team_name]
        # Column labels
        plt.text(-0.11, -0.06, "2024", fontsize=20, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        plt.text(-0.11, -0.12, "2023", fontsize=20, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        plt.text(-0.11, -0.18, "2022", fontsize=20, fontweight='bold', ha='left', va='center', transform=ax.transAxes)

        def draw_metric_column(x, label, values, y_start=0.00, y_step=-0.06):
            plt.text(x, y_start, label, fontsize=20, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

            # Find the lowest numeric value
            numeric_values = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float, float))]
            bold_index = min(numeric_values, key=lambda t: t[1])[0] if numeric_values else -1

            for i, val in enumerate(values):
                y = y_start + y_step * (i + 1)
                if val == "N/A":
                    display_val = "N/A"
                else:
                    display_val = f"{int(val)}"
                fontweight = 'bold' if i == bold_index else 'normal'
                color = '#9932CC' if i == bold_index else 'black'
                plt.text(x, y, display_val, fontsize=20, fontweight=fontweight, ha='center', va='center', color=color, transform=ax.transAxes)

        teams = [team_2024, team_2023, team_2022]
        draw_metric_column(0.1, "NET", get_metric_values(teams, "NET"))
        draw_metric_column(0.3, "RPI", get_metric_values(teams, "RPI"))
        draw_metric_column(0.5, "ELO", get_metric_values(teams, "ELO_Rank"))
        draw_metric_column(0.7, "RQI", get_metric_values(teams, "RQI"))
        draw_metric_column(0.9, "TSR", get_metric_values(teams, "PRR"))

        ### PROJECTED WINS
        projected_wins = simulate_team_win_distribution(schedule_df, comparison_date, team_name)
        peak = projected_wins.idxmax()
        start = max(0, peak - 4)
        end = peak + 5
        while (end - start + 1) < 10:
            end += 1
        full_range = range(start, end + 1)
        filled_distribution = projected_wins.reindex(full_range, fill_value=0)

        stat_rankings = stats_and_metrics.copy()
        higher = ["TB", "SLG", "KP9", "BB", "RS", "H", "BA", "PCT", "HBP", "OBP", "OPS", 
                "PYTHAG", "wOBA", "wRAA", "ISO", "BB%", "LOB%", "K/BB"]
        lower = ["WP9", "ERA", "E", "RA9", "FIP", "WHIP"]
        all_ranked_stats = higher + lower
        stat_rankings[higher] = stat_rankings[higher].rank(ascending=False, method="min")
        stat_rankings[lower] = stat_rankings[lower].rank(ascending=True, method="min")
        team_stats = stat_rankings[stat_rankings['Team'] == team_name].squeeze()
        team_stats = team_stats[all_ranked_stats]
        team_stats = pd.to_numeric(team_stats, errors='coerce')
        best_stats = team_stats.nsmallest(3)
        worst_stats = team_stats.nlargest(3)
        plt.text(1.4,-0.06, "Best Stats", fontsize=20, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
        plt.text(1.2,-0.12, f'{best_stats.index[0]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        plt.text(1.2,-0.18, f'{int(best_stats.values[0])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        plt.text(1.4,-0.12, f'{best_stats.index[1]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        plt.text(1.4,-0.18, f'{int(best_stats.values[1])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        plt.text(1.6,-0.12, f'{best_stats.index[2]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        plt.text(1.6,-0.18, f'{int(best_stats.values[2])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        if columns == 4:
            plt.text(2.1,-0.06, "Worst Stats", fontsize=20, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
            plt.text(1.9,-0.12, f'{worst_stats.index[2]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(1.9,-0.18, f'{int(worst_stats.values[2])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.1,-0.12, f'{worst_stats.index[1]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.1,-0.18, f'{int(worst_stats.values[1])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.3,-0.12, f'{worst_stats.index[0]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.3,-0.18, f'{int(worst_stats.values[0])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        elif columns == 5:
            plt.text(2.5,-0.06, "Worst Stats", fontsize=20, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
            plt.text(2.3,-0.12, f'{worst_stats.index[2]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.3,-0.18, f'{int(worst_stats.values[2])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.5,-0.12, f'{worst_stats.index[1]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.5,-0.18, f'{int(worst_stats.values[1])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.7,-0.12, f'{worst_stats.index[0]}', fontsize=20, ha='center', va='center', transform=ax.transAxes)
            plt.text(2.7,-0.18, f'{int(worst_stats.values[0])}', fontsize=20, ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
            
        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error generating team profile: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/cbase/historical-performance")
def historical_performance(request: HistoricalPerformanceRequest):
    """Generate historical team performance visualization"""
    try:
        all_data = load_historical_data()
        team_name = request.team_name
        
        team_data = all_data[all_data['Team'] == team_name].copy()
        
        if team_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for team: {team_name}")
        
        team_data['Season'] = team_data['Season'].astype(int)
        team_data = team_data.sort_values('Season')
        team_avg_rating = team_data['Normalized_Rating'].mean()
        team_avg_net = team_data['NET_Score'].mean()
        
        seasons = sorted(team_data['Season'].unique())
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        fig.patch.set_facecolor('#CECEB2')
        ax.set_facecolor('#CECEB2')
        
        # Add elite box
        x_start = 4
        x_end = 5
        y_start = 0.85
        y_end = 1
        box_width = x_end - x_start
        box_height = y_end - y_start
        
        elite_box = FancyBboxPatch((x_start, y_start),
                                width=box_width,
                                height=box_height,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                edgecolor="#D51F1F",
                                facecolor="#C8416E",
                                linewidth=2,
                                alpha=0.35,
                                mutation_scale=0.05,
                                zorder=1)
        ax.add_patch(elite_box)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Highlight national champions
        champions = [
            ("Fresno St.", 2008), ("LSU", 2009), ("South Carolina", 2010), ("South Carolina", 2011),
            ('Arizona', 2012), ('UCLA', 2013), ('Vanderbilt', 2014), ('Virginia', 2015),
            ('Coastal Carolina', 2016), ('Florida', 2017), ('Oregon St.', 2018),
            ('Vanderbilt', 2019), ("Mississippi St.", 2021), ('Ole Miss', 2022), 
            ('LSU', 2023), ('Tennessee', 2024), ('LSU', 2025)
        ]
        champ_df = all_data[all_data.apply(lambda row: (row['Team'], row['Season']) in champions, axis=1)]
        
        if not champ_df.empty:
            ax.scatter(champ_df['Normalized_Rating'], champ_df['NET_Score'],
                      color='gold', edgecolor='black', s=200, zorder=5, alpha=0.3)
            for _, row in champ_df.iterrows():
                ax.text(row['Normalized_Rating'], row['NET_Score'], str(row['Season'])[2:],
                       fontsize=7, ha='center', va='center', color='black', fontweight='bold')
        
        # Plot team seasons
        for season in seasons:
            season_data = team_data[team_data['Season'] == season]
            ax.scatter(season_data['Normalized_Rating'], season_data['NET_Score'],
                      color='darkgreen', edgecolor='black', s=400, label=str(season), alpha=0.7)
        
        for _, row in team_data.iterrows():
            ax.text(row['Normalized_Rating'], row['NET_Score'], str(row['Season'])[2:],
                   fontsize=9, ha='center', va='center', color='black', fontweight='bold')
        
        # Add labels and styling
        plt.title(f"Team Strength vs NET for {team_name} Since 2008 (excl. '20)", 
                 fontsize=18, color='black', pad=30, fontweight='bold')
        plt.suptitle("@PEARatings", y=0.865, fontsize=16, ha='center', va='center')
        ax.set_xlabel('Team Strength', color='black', fontsize=14)
        ax.set_ylabel('NET Score', color='black', fontsize=14)
        ax.tick_params(colors='black')
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        
        # Add average lines
        x_mean = all_data['Normalized_Rating'].mean()
        y_mean = all_data['NET_Score'].mean()
        
        if team_data['Normalized_Rating'].min() < x_mean:
            ax.axvline(x_mean, linestyle='--', linewidth=1, color='red')
            ax.text(x_mean-0.02, ax.get_ylim()[1]-0.02, f'Avg Rating', 
                   color='red', fontsize=10, ha='right', va='top', rotation=90)
        
        if team_data['NET_Score'].min() < y_mean:
            ax.axhline(y_mean, linestyle='--', linewidth=1, color='red')
            ax.text(ax.get_xlim()[1]-0.02, y_mean-0.005, f'Avg NET', 
                   color='red', fontsize=10, ha='right', va='top')
        
        # Add legend
        avg_rating_patch = mpatches.Patch(color='none', label=f'Average Rating is 0', linewidth=0)
        team_rating_patch = mpatches.Patch(color='none', label=f"{team_name} avg: {team_avg_rating:.2f}")
        avg_net_patch = mpatches.Patch(color='none', label=f'Average NET is 0.56', linewidth=0)
        team_net_patch = mpatches.Patch(color='none', label=f"{team_name} avg: {team_avg_net:.2f}")
        
        leg = ax.legend(handles=[avg_rating_patch, team_rating_patch, avg_net_patch, team_net_patch],
                       fontsize=11,
                       title_fontsize=12,
                       loc='best',
                       frameon=True,
                       facecolor='#4B5320',
                       edgecolor='black',
                       handlelength=0,
                       handletextpad=0,
                       borderpad=1,
                       labelspacing=0.6)
        
        for text in leg.get_texts():
            text.set_ha('center')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error generating historical performance: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/cbase/teams")
def get_teams():
    """Get list of all teams"""
    try:
        modeling_stats, _ = load_baseball_data()
        teams = sorted(modeling_stats['Team'].unique().tolist())
        return {"teams": teams}
    except Exception as e:
        print(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)