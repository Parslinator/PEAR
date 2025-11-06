from fastapi import FastAPI, HTTPException, Response
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
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

app = FastAPI(title="PEAR Ratings API")

# CORS middleware to allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://129.212.189.128",
        "http://pearatings.com",      # Add these
        "https://pearatings.com",     # Add these
        "http://www.pearatings.com",  # Add these
        "https://www.pearatings.com", # Add these
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import gc

def cleanup_figure_memory(fig):
    """Aggressively clean up matplotlib figure and force garbage collection"""
    try:
        plt.close(fig)
        fig.clf()  # Clear the figure
        del fig
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Error cleaning up figure: {e}")

# Global constants
GLOBAL_HFA = 3

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the parent directory, then into PEAR
BASE_PATH = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Football")

# print(f"Backend directory: {BACKEND_DIR}")
# print(f"Base path for data: {BASE_PATH}")
# print(f"Base path exists: {os.path.exists(BASE_PATH)}")

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
# print(f"Logo folder path: {logo_folder}")
# print(f"Logo folder exists: {os.path.exists(logo_folder)}")

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
    
    # print(f"Looking for logo at: {logo_path}")
    # print(f"Logo folder: {logo_folder}")
    # print(f"File exists: {os.path.exists(logo_path)}")
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail=f"Logo not found at: {logo_path}")
    
    return FileResponse(logo_path, media_type="image/png")

@app.get("/api/game-preview/{year}/{week}/{filename}")
def get_game_preview(year: int, week: int, filename: str):
    """Serve game preview image"""
    # The filename comes as "home_team vs away_team" (without .png)
    image_filename = f"{filename}.png"
    image_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", "Games", image_filename)
    
    # print(f"Looking for game preview at: {image_path}")
    # print(f"File exists: {os.path.exists(image_path)}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Game preview not found at: {image_path}")
    
    return FileResponse(image_path, media_type="image/png")

@app.get("/api/team-profile/{year}/{week}/{team_name}")
def get_team_profile(year: int, week: int, team_name: str):
    image_path = os.path.join(BASE_PATH, f"y{year}", "Visuals", f"week_{week}", "Stat Profiles", f"{team_name}.png")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    img = None
    try:
        # Optimize the image
        img = Image.open(image_path)
        
        # Resize if too large (e.g., max width 1200px)
        max_width = 1200
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Save as optimized format
        buffer = io.BytesIO()
        img.save(buffer, format='WEBP', quality=85, method=6)
        buffer.seek(0)
        
        # Get the content before closing
        content = buffer.getvalue()
        
        return Response(
            content=content, 
            media_type="image/webp"
        )
    finally:
        # Always close the image to free memory
        if img:
            img.close()

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
        
        # print(f"Attempting to load ratings from: {ratings_path}")
        # print(f"Attempting to load data from: {data_path}")
        # print(f"Ratings file exists: {os.path.exists(ratings_path)}")
        # print(f"Data file exists: {os.path.exists(data_path)}")
        
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
    ax.text(0.19, 0.49, f"{away_pr:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.49+0.007, f"{away_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_rank))
    
    ax.text(0.08, 0.45, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.45, f"{away_offense:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.45+0.007, f"{away_offense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_offense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_offense_rank))

    ax.text(0.08, 0.41, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.41, f"{away_defense:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.41+0.007, f"{away_defense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_defense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_defense_rank))

    ax.text(0.01, 0.37, f"DRIVE QUALITY", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.08, 0.33, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.33, f"{away_off_dq:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.33+0.007, f"{away_off_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_off_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_off_dq_rank))
    
    ax.text(0.08, 0.29, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.29, f"{away_def_dq:.1f}", ha='right', fontsize=16, fontweight='bold')
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
    ax.text(0.19, 0.09, f"{away_md:.3f}", ha='right', fontsize=16, fontweight='bold')
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
    ax.text(0.81, 0.49, f"{home_pr:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.49+0.007, f"{home_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_rank))
    
    ax.text(0.92, 0.45, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.45, f"{home_offense:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.45+0.007, f"{home_offense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_offense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_offense_rank))

    ax.text(0.92, 0.41, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.41, f"{home_defense:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.41+0.007, f"{home_defense_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_defense_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_defense_rank))
    
    ax.text(0.99, 0.37, f"DRIVE QUALITY", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.92, 0.33, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.33, f"{home_off_dq:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.33+0.007, f"{home_off_dq_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_off_dq_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_off_dq_rank))
    
    ax.text(0.92, 0.29, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.29, f"{home_def_dq:.1f}", ha='left', fontsize=16, fontweight='bold')
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
    ax.text(0.81, 0.09, f"{home_md:.3f}", ha='left', fontsize=16, fontweight='bold')
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
        # print(f"Attempting to load spreads from: {spreads_path}")
        # print(f"Spreads file exists: {os.path.exists(spreads_path)}")
        
        if not os.path.exists(spreads_path):
            raise HTTPException(status_code=404, detail=f"Spreads file not found at: {spreads_path}")
        
        spreads = pd.read_excel(spreads_path)
        spreads['start_date'] = pd.to_datetime(spreads['start_date']).dt.strftime('%Y-%m-%d')
        
        vegas_col = 'formattedSpread' if 'formattedSpread' in spreads.columns else 'formatted_spread'
        spreads['Vegas'] = spreads.get(vegas_col, '')
        
        required_cols = ['start_date', 'start_time', 'outlet', 'home_team', 'away_team', 'home_score', 'away_score', 'PEAR_win_prob', 'PEAR', 'pr_spread', 'difference', 'GQI']
        missing_cols = [col for col in required_cols if col not in spreads.columns]
        
        if missing_cols:
            print(f"Available columns: {spreads.columns.tolist()}")
            print(f"Missing columns: {missing_cols}")
            raise HTTPException(status_code=500, detail=f"Missing columns in spreads file: {missing_cols}")

        result = spreads[['start_date', 'start_time', 'outlet', 'home_team', 'away_team', 'home_score', 'away_score', 'PEAR_win_prob', 'PEAR', 'Vegas', 'difference', 'GQI', 'pr_spread']].dropna().to_dict('records')

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
        # print(f"Attempting to load historical data from: {hist_path}")
        # print(f"File exists: {os.path.exists(hist_path)}")
        
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
    logo_path = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "pear_logo.jpg")
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail="Logo not found")
    
    return FileResponse(logo_path, media_type="image/jpeg")

# ========================================
# BASEBALL (CBASE) ENDPOINTS
# ========================================

BASEBALL_BASE_PATH = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Baseball")
BASEBALL_CURRENT_SEASON = 2025
BASEBALL_HFA = 0.3  # Home field advantage in runs for baseball

# print(f"Baseball base path: {BASEBALL_BASE_PATH}")
# print(f"Baseball base path exists: {os.path.exists(BASEBALL_BASE_PATH)}")

@app.get("/api/baseball-logo/{team_name}")
def get_baseball_logo(team_name: str):
    """Serve team logo"""
    # Replace spaces with underscores for the filename
    logo_filename = f"{team_name}.png"
    logo_path = os.path.join(BASEBALL_BASE_PATH, "logos", logo_filename)

    # print(f"Looking for logo at: {logo_path}")
    # print(f"Logo folder: {logo_folder}")
    # print(f"File exists: {os.path.exists(logo_path)}")
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail=f"Logo not found at: {logo_path}")
    
    return FileResponse(logo_path, media_type="image/png")

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
            'ERA', 'WHIP', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS', 'PCT'
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

@app.get("/api/cbase/team-conferences")
def get_team_conferences():
    """Get mapping of teams to their conferences"""
    try:
        modeling_stats, _ = load_baseball_data()
        
        # Create a dictionary mapping team names to conferences
        team_conference_map = {}
        for _, row in modeling_stats[['Team', 'Conference']].drop_duplicates().iterrows():
            team_conference_map[row['Team']] = row['Conference']
        
        return {"team_conferences": team_conference_map}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_team_conferences: {e}")
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
            # If we're past July and there are no games for today,
            # use the most recent date in the dataframe instead.
            if today.month > 7:
                last_date = schedule_df['Date'].max().date()
                today_games = schedule_df[schedule_df['Date'].dt.date == last_date].copy()
                today_games = today_games[['home_team', 'away_team', 'Location', 'PEAR', 'GQI', 'Date', 'home_win_prob', 'home_net', 'away_net']].drop_duplicates()
                return {"games": today_games.to_dict(orient="records"), "date": last_date.strftime("%B %d, %Y")}
            else:
                return {"games": [], "date": today.strftime("%B %d, %Y")}
        
        # Process results
        today_games = today_games[[
            'home_team', 'away_team', 'Location', 'PEAR', 'GQI', 'Date', 'home_win_prob', 'home_net', 'away_net'
        ]].copy().drop_duplicates()
        
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

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location):
    if location != "Neutral":
        home_pr += 0.3
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    spread = round(adjust_home_pr(elo_win_prob) + home_pr - away_pr, 2)
    return spread, elo_win_prob
    
def calculate_series_probabilities(win_prob):
    # Team A win probabilities
    P_A_0 = (1 - win_prob) ** 3
    P_A_1 = 3 * win_prob * (1 - win_prob) ** 2
    P_A_2 = 3 * win_prob ** 2 * (1 - win_prob)
    P_A_3 = win_prob ** 3

    # Team B win probabilities (q = 1 - p)
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

    return [P_A_at_least_1,P_A_at_least_2,P_A_3], [P_B_at_least_1,P_B_at_least_2,P_B_3]

@app.post("/api/cbase/matchup-image")
def generate_matchup_image(request: BaseballSpreadRequest):
    """Generate matchup comparison image"""
    home_logo = None
    away_logo = None
    
    try:

        stats_and_metrics, _ = load_baseball_data()
        
        away_team = request.away_team
        home_team = request.home_team
        neutrality = "Neutral" if request.neutral else "Home"
        
            
        # Load team logos
        logo_folder = os.path.join(BASEBALL_BASE_PATH, "logos")
        home_logo = None
        away_logo = None

        def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
            if location != "Neutral":
                home_pr += 0.3
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

        def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                            facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
            # Draw rectangle behind text
            ax.add_patch(Rectangle(
                (x - width/2, y - height/2), width, height,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=1
            ))

            # Draw text centered on top
            ax.text(x, y, text,
                    ha="center", va="center", zorder=2, **kwargs)
            
        def get_text_color(bg_color: str) -> str:
            """Determine if text should be black or white based on background color luminance"""
            import re
            
            # Handle hex colors
            if bg_color.startswith('#'):
                # Convert hex to RGB
                bg_color = bg_color.lstrip('#')
                r = int(bg_color[0:2], 16)
                g = int(bg_color[2:4], 16)
                b = int(bg_color[4:6], 16)
            else:
                # Handle rgb() format
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if not match:
                    return 'white'
                
                r = int(match.group(1))
                g = int(match.group(2))
                b = int(match.group(3))
            
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return 'black' if luminance > 0.5 else 'white'

        def rank_text_color(rank, vmin=1, vmax=300):
            """Get appropriate text color (black or white) based on rank background color"""
            if rank == "":
                return 'black'
            
            bg_color = rank_to_color(rank, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def percent_text_color(win_pct, vmin=0.0, vmax=1.0):
            """Get appropriate text color (black or white) based on win percentage background color"""
            if win_pct == "":
                return 'black'
            
            bg_color = rank_to_color(win_pct, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def plot_logo(ax, img, xy, zoom=0.2):
            """Helper to plot a logo at given xy coords."""
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy, frameon=False)
            ax.add_artist(ab)

        def rank_to_color(rank, vmin=1, vmax=300):
            """
            Map a rank (1–300) to a hex color.
            Dark blue = best (1), grey = middle, dark red = worst (300).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from blue → grey → red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rank_cmap", ["#00008B", "#00FFFF", "#D3D3D3", "#FFA500", "#8B0000"]  # dark blue, cyan, light gray, orange, dark red
            )
            
            # Normalize rank to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(rank))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def percent_to_color(win_pct, vmin=0.0, vmax=1.0):
            """
            Map a win percentage (0.0–1.0) to a hex color.
            Dark blue = best (1.0), grey = middle (0.5), dark red = worst (0.0).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from red → grey → blue
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "percent_cmap", ["#8B0000", "#FFA500", "#D3D3D3", "#00FFFF", "#00008B"]  # dark red, orange, light gray, cyan, dark blue
            )
            
            # Normalize percentage to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(win_pct))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def get_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (value, rank) for a given team and column.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and stat columns.
                team (str): Team name to look up.
                column (str): Column name to extract.
                higher_is_better (bool): If True, high values rank better (1 = highest).
                                        If False, low values rank better (1 = lowest).
            """
            ascending = not higher_is_better
            ranks = df[column].rank(ascending=ascending, method="first").astype(int)

            value = df.loc[df['Team'] == team, column].values[0]
            rank = ranks.loc[df['Team'] == team].values[0]

            return value, rank
        
        def get_record_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (record_string, win_percentage, rank) for a given team and column containing W-L records.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and record columns.
                team (str): Team name to look up.
                column (str): Column name containing records in "W-L" format.
                higher_is_better (bool): If True, high win% ranks better (1 = highest).
                                        If False, low win% ranks better (1 = lowest).
            
            Returns:
                tuple: (record_string, win_percentage as float, rank as int)
            """
            def calculate_win_pct(record):
                """Convert 'W-L' string to win percentage."""
                if pd.isna(record) or record == '':
                    return 0.0
                parts = str(record).split('-')
                wins = int(parts[0])
                losses = int(parts[1])
                total = wins + losses
                return wins / total if total > 0 else 0.0
            
            # Calculate win percentages for all teams
            win_pcts = df[column].apply(calculate_win_pct)
            
            # Calculate ranks
            ascending = not higher_is_better
            ranks = win_pcts.rank(ascending=ascending, method="first").astype(int)
            
            # Get values for specified team
            team_idx = df['Team'] == team
            record_string = df.loc[team_idx, column].values[0]
            win_pct = win_pcts.loc[team_idx].values[0]
            rank = ranks.loc[team_idx].values[0]
            
            return record_string, win_pct
        
        def add_row(x_vals, y, away_val, away_rank, away_name, home_name, home_rank, home_val, away_digits, home_digits):
            # Helper to choose text color based on rank

            # Away value
            if away_val != "":
                ax.text(x_vals[0], y, f"{away_val:.{away_digits}f}", ha='center', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            # Away rank box
            if away_rank != "":
                fixed_width_text(
                    ax, x_vals[1], y+0.007, f"{away_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_rank)
                )

            # Metric name
            if away_name != "":
                ax.text(x_vals[2], y, away_name, ha='left', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            if home_name != "":
                ax.text(x_vals[3], y, home_name, ha='right', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            # Home rank box
            if home_rank != "":
                fixed_width_text(
                    ax, x_vals[4], y+0.007, f"{home_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_rank)
                )

            # Home value
            if home_val != "":
                ax.text(x_vals[5], y, f"{home_val:.{home_digits}f}", ha='center', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))
        
        if os.path.exists(logo_folder):
            home_logo_path = os.path.join(logo_folder, f"{home_team}.png")
            away_logo_path = os.path.join(logo_folder, f"{away_team}.png")
            
            if os.path.exists(home_logo_path):
                home_logo = Image.open(home_logo_path).convert("RGBA")
            if os.path.exists(away_logo_path):
                away_logo = Image.open(away_logo_path).convert("RGBA")

        fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
        fig.patch.set_facecolor('#CECEB2')
        ax.set_facecolor('#CECEB2')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ----------------
        # logos, score, win prob, spread, O/U
        # ----------------
        plot_logo(ax, away_logo, (0.15, 0.75), zoom=0.3)
        plot_logo(ax, home_logo, (0.85, 0.75), zoom=0.3)

        if neutrality == "Neutral":
            ax.text(0.5, 0.96, f"{away_team} (N) {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))
        else:
            ax.text(0.5, 0.96, f"{away_team} at {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))

        alpha_val = 0.9

        away_pr, away_rank = get_value_and_rank(stats_and_metrics, away_team, 'Rating')
        home_pr, home_rank = get_value_and_rank(stats_and_metrics, home_team, 'Rating')
        away_elo, away_elo_rank = get_value_and_rank(stats_and_metrics, away_team, 'ELO')
        home_elo, home_elo_rank = get_value_and_rank(stats_and_metrics, home_team, 'ELO')
        home_net_score, home_net_rank = get_value_and_rank(stats_and_metrics, home_team, 'NET_Score')
        away_net_score, away_net_rank = get_value_and_rank(stats_and_metrics, away_team, 'NET_Score')
        home_rq, home_rq_rank = get_value_and_rank(stats_and_metrics, home_team, 'resume_quality')
        away_rq, away_rq_rank = get_value_and_rank(stats_and_metrics, away_team, 'resume_quality')
        home_sos, home_sos_rank = get_value_and_rank(stats_and_metrics, home_team, 'avg_expected_wins', False)
        away_sos, away_sos_rank = get_value_and_rank(stats_and_metrics, away_team, 'avg_expected_wins', False)
        home_pythag, home_pythag_rank = get_value_and_rank(stats_and_metrics, home_team, 'PYTHAG')
        away_pythag, away_pythag_rank = get_value_and_rank(stats_and_metrics, away_team, 'PYTHAG')
        home_war, home_war_rank = get_value_and_rank(stats_and_metrics, home_team, 'fWAR')
        away_war, away_war_rank = get_value_and_rank(stats_and_metrics, away_team, 'fWAR')
        home_wpoe, home_wpoe_rank = get_value_and_rank(stats_and_metrics, home_team, 'wpoe_pct')
        away_wpoe, away_wpoe_rank = get_value_and_rank(stats_and_metrics, away_team, 'wpoe_pct')
        home_q1, home_q1_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q1')
        home_q2, home_q2_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q2')
        home_q3, home_q3_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q3')
        home_q4, home_q4_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q4')
        away_q1, away_q1_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q1')
        away_q2, away_q2_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q2')
        away_q3, away_q3_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q3')
        away_q4, away_q4_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q4')

        home_rpg, home_rpg_rank = get_value_and_rank(stats_and_metrics, home_team, 'RPG')
        away_rpg, away_rpg_rank = get_value_and_rank(stats_and_metrics, away_team, 'RPG')
        home_ba, home_ba_rank = get_value_and_rank(stats_and_metrics, home_team, 'BA')
        away_ba, away_ba_rank = get_value_and_rank(stats_and_metrics, away_team, 'BA')
        home_obp, home_obp_rank = get_value_and_rank(stats_and_metrics, home_team, 'OBP')
        away_obp, away_obp_rank = get_value_and_rank(stats_and_metrics, away_team, 'OBP')
        home_slg, home_slg_rank = get_value_and_rank(stats_and_metrics, home_team, 'SLG')
        away_slg, away_slg_rank = get_value_and_rank(stats_and_metrics, away_team, 'SLG')
        home_ops, home_ops_rank = get_value_and_rank(stats_and_metrics, home_team, 'OPS')
        away_ops, away_ops_rank = get_value_and_rank(stats_and_metrics, away_team, 'OPS')
        home_iso, home_iso_rank = get_value_and_rank(stats_and_metrics, home_team, 'ISO')
        away_iso, away_iso_rank = get_value_and_rank(stats_and_metrics, away_team, 'ISO')
        home_era, home_era_rank = get_value_and_rank(stats_and_metrics, home_team, 'ERA', False)
        away_era, away_era_rank = get_value_and_rank(stats_and_metrics, away_team, 'ERA', False)
        home_whip, home_whip_rank = get_value_and_rank(stats_and_metrics, home_team, 'WHIP', False)
        away_whip, away_whip_rank = get_value_and_rank(stats_and_metrics, away_team, 'WHIP', False)
        home_k9, home_k9_rank = get_value_and_rank(stats_and_metrics, home_team, 'KP9')
        away_k9, away_k9_rank = get_value_and_rank(stats_and_metrics, away_team, 'KP9')
        home_lob, home_lob_rank = get_value_and_rank(stats_and_metrics, home_team, 'LOB%')
        away_lob, away_lob_rank = get_value_and_rank(stats_and_metrics, away_team, 'LOB%')
        home_kbb, home_kbb_rank = get_value_and_rank(stats_and_metrics, home_team, 'K/BB')
        away_kbb, away_kbb_rank = get_value_and_rank(stats_and_metrics, away_team, 'K/BB')
        home_pct, home_pct_rank = get_value_and_rank(stats_and_metrics, home_team, 'PCT')
        away_pct, away_pct_rank = get_value_and_rank(stats_and_metrics, away_team, 'PCT')

        home_win_prob = PEAR_Win_Prob(home_pr, away_pr, neutrality)
        home_series, away_series = calculate_series_probabilities(home_win_prob/100)
        spread, elo_win_prob = calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, neutrality)
        if spread < 0:
            formatted_spread = f"{away_team} -{abs(spread):.2f}"
        else:
            formatted_spread = f"{home_team} -{spread:.2f}"

        max_net = 299
        w_tq = 0.70   # NET AVG
        w_wp = 0.20   # Win Probability
        w_ned = 0.10  # NET Differential
        avg_net = (home_net_rank + away_net_rank) / 2
        tq = (max_net - avg_net) / (max_net - 1)
        wp = 1 - 2 * np.abs((home_win_prob/100) - 0.5)
        ned = 1 - (np.abs(away_net_rank - home_net_rank) / (max_net - 1))
        gqi = round(10*(w_tq * tq + w_wp * wp + w_ned * ned), 1)

        bubble_team_rating = stats_and_metrics['Rating'].quantile(0.90)
        home_quality = PEAR_Win_Prob(bubble_team_rating, away_pr, neutrality) / 100
        home_win_quality, home_loss_quality = (1 - home_quality), -home_quality
        away_quality = 1-PEAR_Win_Prob(home_pr, bubble_team_rating, neutrality) / 100
        away_win_quality, away_loss_quality = (1 - away_quality), -away_quality

        ax.text(0.5, 0.57, f"{formatted_spread}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))
        ax.text(0.5, 0.625, f"GQI: {gqi}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))
        ax.text(0.6, 0.89, f"ONE GAME (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.84, f"{round(home_win_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.78, f"SERIES (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.75, f"≥1: {round(home_series[0]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.72, f"≥2: {round(home_series[1]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.69, f"SWEEP: {round(home_series[2]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        
        ax.text(0.4, 0.89, f"ONE GAME (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.84, f"{round(100-home_win_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.78, f"SERIES (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.75, f"≥1: {round(away_series[0]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.72, f"≥2: {round(away_series[1]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.69, f"SWEEP: {round(away_series[2]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))

        away_record = stats_and_metrics.loc[stats_and_metrics['Team'] == away_team, 'Record'].values[0]
        ax.text(0.01, 0.53, f"{away_record}", ha='left', fontsize=16, fontweight='bold')

        home_record = stats_and_metrics.loc[stats_and_metrics['Team'] == home_team, 'Record'].values[0]
        ax.text(0.99, 0.53, f"{home_record}", ha='right', fontsize=16, fontweight='bold')

        # X positions for the 5 columns
        x_cols = [0.31, 0.378, 0.42, 0.58, 0.622, 0.69]

        ax.text(0.5, 0.528, f"{away_team} OFF vs {home_team} PCH",
                ha='center', fontsize=16, fontweight='bold',
                bbox=dict(facecolor='green', alpha=0))
        ax.hlines(y=0.518, xmin=0.29, xmax=0.71, colors='black', linewidth=1)

        # Away OFF vs Home DEF
        add_row(x_cols, 0.49, away_rpg, away_rpg_rank, "RPG", "ERA", home_era_rank, home_era, 2, 2)
        add_row(x_cols, 0.45, away_ba, away_ba_rank, "BA", "WHIP", home_whip_rank, home_whip, 3, 2)
        add_row(x_cols, 0.41, away_obp, away_obp_rank, "OBP", "K/9", home_k9_rank, home_k9, 3, 1)
        add_row(x_cols, 0.37, away_slg, away_slg_rank, "SLG", "LOB%", home_lob_rank, home_lob, 3, 2)
        add_row(x_cols, 0.33, away_ops, away_ops_rank, "OPS", "K/BB", home_kbb_rank, home_kbb, 3, 2)
        add_row(x_cols, 0.29, away_iso, away_iso_rank, "ISO", "PCT", home_pct_rank, home_pct, 3, 3)

        # Header for Away DEF vs Home OFF
        ax.text(0.5, 0.248, f"{away_team} PCH vs {home_team} OFF",
                ha='center', fontsize=16, fontweight='bold', bbox=dict(facecolor='green', alpha=0))
        ax.hlines(y=0.238, xmin=0.29, xmax=0.71, colors='black', linewidth=1)
        add_row(x_cols, 0.21, away_era, away_era_rank, "ERA", "RPG", home_rpg_rank, home_rpg, 2, 2)
        add_row(x_cols, 0.17, away_whip, away_whip_rank, "WHIP", "BA", home_ba_rank, home_ba, 2, 3)
        add_row(x_cols, 0.13, away_k9, away_k9_rank, "K/9", "OBP", home_obp_rank, home_obp, 1, 3)
        add_row(x_cols, 0.09, away_lob, away_lob_rank, "LOB%", "SLG", home_slg_rank, home_slg, 2, 3)
        add_row(x_cols, 0.05, away_kbb, away_kbb_rank, "K/BB", "OPS", home_ops_rank, home_ops, 2, 3)
        add_row(x_cols, 0.01, away_pct, away_pct_rank, "PCT", "ISO", home_iso_rank, home_iso, 2, 3)
        ax.text(0.5, -0.03, "@PEARatings", ha='center', fontsize=16, fontweight='bold',bbox=dict(facecolor='green', alpha=0))

        ### AWAY SIDE

        ax.text(0.01, 0.49, f"NET", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.478, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
        ax.text(0.19, 0.49, f"{away_net_score:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.49+0.007, f"{away_net_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_net_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_net_rank))
        
        ax.text(0.04, 0.45, f"RATING", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.45, f"{away_pr}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.45+0.007, f"{away_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_rank))

        ax.text(0.04, 0.41, f"RQI", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.41, f"{away_rq:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.41+0.007, f"{away_rq_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_rq_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_rq_rank))
        
        ax.text(0.04, 0.37, f"SOS", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.37, f"{away_sos:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.37+0.007, f"{away_sos_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_sos_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_sos_rank))

        ax.text(0.04, 0.33, f"PYTHAG", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.33, f"{away_pythag:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.33+0.007, f"{away_pythag_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_pythag_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_pythag_rank))

        ax.text(0.04, 0.29, f"WAR", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.29, f"{away_war:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.29+0.007, f"{away_war_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_war_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_war_rank))

        ax.text(0.04, 0.25, f"WPOE", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.25, f"{away_wpoe:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.25+0.007, f"{away_wpoe_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_wpoe_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_wpoe_rank))

        ax.text(0.01, 0.21, f"NET QUADS", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.198, xmin=0.01, xmax=0.26, colors='black', linewidth=1)

        ax.text(0.04, 0.17, f"Q1", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.17, f"{away_q1}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.17+0.007, f"{away_q1_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q1_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q1_rank))

        ax.text(0.04, 0.13, f"Q2", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.13, f"{away_q2}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.13+0.007, f"{away_q2_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q2_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q2_rank))
        
        ax.text(0.04, 0.09, f"Q3", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.09, f"{away_q3}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.09+0.007, f"{away_q3_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q3_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q3_rank))

        ax.text(0.04, 0.05, f"Q4", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.05, f"{away_q4}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.05+0.007, f"{away_q4_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q4_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q4_rank))
        
        ax.text(0.01, 0.01, f"WIN QUALITY", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.0, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
        ax.text(0.04, -0.03, f"{away_win_quality:.2f}", ha='left', fontsize=16, fontweight='bold', color='green')
        ax.text(0.19, -0.03, f"{away_loss_quality:.2f}", ha='right', fontsize=16, fontweight='bold', color='red')

        #### HOME SIDE

        ax.text(0.99, 0.49, f"NET", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.478, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.81, 0.49, f"{home_net_score:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.49+0.007, f"{home_net_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_net_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_net_rank))

        ax.text(0.96, 0.45, f"RATING", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.45, f"{home_pr}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.45+0.007, f"{home_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_rank))

        ax.text(0.96, 0.41, f"RQI", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.41, f"{home_rq:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.41+0.007, f"{home_rq_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_rq_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_rq_rank))

        ax.text(0.96, 0.37, f"SOS", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.37, f"{home_sos:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.37+0.007, f"{home_sos_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_sos_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_sos_rank))

        ax.text(0.96, 0.33, f"PYTHAG", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.33, f"{home_pythag:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.33+0.007, f"{home_pythag_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_pythag_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_pythag_rank))

        ax.text(0.96, 0.29, f"WAR", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.29, f"{home_war:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.29+0.007, f"{home_war_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_war_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_war_rank))
        
        ax.text(0.96, 0.25, f"WPOE", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.25, f"{home_wpoe:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.25+0.007, f"{home_wpoe_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_wpoe_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_wpoe_rank))
        
        ax.text(0.99, 0.21, f"NET QUADS", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.198, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.96, 0.17, f"Q1", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.17, f"{home_q1}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.17+0.007, f"{home_q1_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q1_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q1_rank))

        ax.text(0.96, 0.13, f"Q2", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.13, f"{home_q2}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.13+0.007, f"{home_q2_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q2_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q2_rank))

        ax.text(0.96, 0.09, f"Q3", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.09, f"{home_q3}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.09+0.007, f"{home_q3_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q3_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q3_rank))

        ax.text(0.96, 0.05, f"Q4", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.05, f"{home_q4}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.05+0.007, f"{home_q4_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q4_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q4_rank))

        ax.text(0.99, 0.01, f"WIN QUALITY", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.0, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.81, -0.03, f"{home_win_quality:.2f}", ha='left', fontsize=16, fontweight='bold', color='green')
        ax.text(0.96, -0.03, f"{home_loss_quality:.2f}", ha='right', fontsize=16, fontweight='bold', color='red')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        
        # Get image data
        image_data = buf.getvalue()
        
        # Aggressive cleanup
        plt.close(fig)
        fig.clf()
        del fig
        del buf
        gc.collect()
        
        return Response(content=image_data, media_type="image/png")
    
    except Exception as e:
        print(f"Error generating matchup image: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        # Always close logo images
        if home_logo is not None:
            home_logo.close()
            del home_logo
        if away_logo is not None:
            away_logo.close()
            del away_logo
        gc.collect()

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

class TeamScheduleRequest(BaseModel):
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

@app.post("/api/cbase/profile-page")
def baseball_team_schedule(request: TeamScheduleRequest):
    stats_and_metrics, comparison_date = load_baseball_data()
    schedule_df = load_schedule_data()
    team_name = request.team_name

    team_schedule = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True)
    metrics = stats_and_metrics[stats_and_metrics['Team'] == team_name].reset_index(drop=True)
    
    # Process schedule data
    schedule_data = []
    for idx, row in team_schedule.iterrows():
        game_data = {
            'date': row['Date'],
            'opponent': row['Opponent'] if pd.notna(row['Opponent']) else 'Non D-I',
            'location': row['Location'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
            'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
            'result': row['Result'] if pd.notna(row['Result']) else None,
            'home_win_prob': float(row['home_win_prob']) if pd.notna(row['home_win_prob']) else None,
            'resume_quality': float(row['resume_quality']) if pd.notna(row['resume_quality']) else None,
            'home_net': int(row['home_net']) if pd.notna(row['home_net']) else None,
            'away_net': int(row['away_net']) if pd.notna(row['away_net']) else None,
            'gqi': float(row['GQI']) if pd.notna(row['GQI']) else None,
            'pear': row['PEAR'] if pd.notna(row['PEAR']) else None,
        }
        
        # Determine opponent NET ranking
        if row['Team'] == row['home_team']:
            game_data['opponent_net'] = game_data['away_net']
            game_data['team_win_prob'] = game_data['home_win_prob']
        else:
            game_data['opponent_net'] = game_data['home_net']
            game_data['team_win_prob'] = 1 - game_data['home_win_prob'] if game_data['home_win_prob'] is not None else None
        
        schedule_data.append(game_data)
    
    # Get team metrics if available
    team_metrics = {}
    if len(metrics) > 0:
        team_metrics = {
            'conference': metrics.iloc[0]['Conference'] if 'Conference' in metrics.columns else None,
            'rating': float(metrics.iloc[0]['Rating']) if 'Rating' in metrics.columns and pd.notna(metrics.iloc[0]['Rating']) else None,
            'tsr': int(metrics.iloc[0]['PRR']) if 'PRR' in metrics.columns else None,
            'net': int(metrics.iloc[0]['NET']) if 'NET' in metrics.columns and pd.notna(metrics.iloc[0]['NET']) else None,
            'net_score': float(metrics.iloc[0]['NET_Score']) if 'NET_Score' in metrics.columns and pd.notna(metrics.iloc[0]['NET_Score']) else None,
            'rpi': int(metrics.iloc[0]['RPI']) if 'RPI' in metrics.columns and pd.notna(metrics.iloc[0]['RPI']) else None,
            'elo': float(metrics.iloc[0]['ELO']) if 'ELO' in metrics.columns and pd.notna(metrics.iloc[0]['ELO']) else None,
            'elo_rank': int(metrics.iloc[0]['ELO_Rank']) if 'ELO_Rank' in metrics.columns and pd.notna(metrics.iloc[0]['ELO_Rank']) else None,
            'resume_quality': float(metrics.iloc[0]['resume_quality']) if 'resume_quality' in metrics.columns and pd.notna(metrics.iloc[0]['resume_quality']) else None,
            'record': metrics.iloc[0]['Record'] if 'Record' in metrics.columns else None,
            'q1': metrics.iloc[0]['Q1'] if 'Q1' in metrics.columns else None,
            'q2': metrics.iloc[0]['Q2'] if 'Q2' in metrics.columns else None,
            'q3': metrics.iloc[0]['Q3'] if 'Q3' in metrics.columns else None,
            'q4': metrics.iloc[0]['Q4'] if 'Q4' in metrics.columns else None
        }
    
    return {
        'team_name': team_name,
        'schedule': schedule_data,
        'metrics': team_metrics,
        'data_date': comparison_date
    }

@app.post("/api/cbase/team-profile")
def baseball_team_profile(request: TeamProfileRequest):
    logo = None
    try:
        stats_and_metrics, comparison_date = load_baseball_data()
        schedule_df = load_schedule_data()
        team_name = request.team_name

        logo_folder = os.path.join(BASEBALL_BASE_PATH, "logos")

        def plot_box(x, y, width, height, color='black', fill=False, linewidth=2, ax=None):
            if ax is None:
                fig, ax = plt.subplots()

            rect = Rectangle((x, y), width, height,
                                    linewidth=linewidth,
                                    edgecolor="black",
                                    facecolor=color if fill else 'none')
            ax.add_patch(rect)

            return ax

        def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
            if location != "Neutral":
                home_pr += 0.3
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

        def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                            facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
            # Draw rectangle behind text
            ax.add_patch(Rectangle(
                (x - width/2, y - height/2), width, height,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=1
            ))

            # Draw text centered on top
            ax.text(x, y, text,
                    ha="center", va="center", zorder=2, **kwargs)
            
        def get_text_color(bg_color: str) -> str:
            """Determine if text should be black or white based on background color luminance"""
            import re
            
            # Handle hex colors
            if bg_color.startswith('#'):
                # Convert hex to RGB
                bg_color = bg_color.lstrip('#')
                r = int(bg_color[0:2], 16)
                g = int(bg_color[2:4], 16)
                b = int(bg_color[4:6], 16)
            else:
                # Handle rgb() format
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if not match:
                    return 'white'
                
                r = int(match.group(1))
                g = int(match.group(2))
                b = int(match.group(3))
            
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return 'black' if luminance > 0.5 else 'white'

        def rank_text_color(rank, vmin=1, vmax=300):
            """Get appropriate text color (black or white) based on rank background color"""
            if rank == "":
                return 'black'
            
            bg_color = rank_to_color(rank, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def percent_text_color(win_pct, vmin=0.0, vmax=1.0):
            """Get appropriate text color (black or white) based on win percentage background color"""
            if win_pct == "":
                return 'black'
            
            bg_color = rank_to_color(win_pct, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def plot_logo(ax, img, xy, zoom=0.2, zorder=3):
            """Helper to plot a logo at given xy coords."""
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy, frameon=False, zorder=zorder,
                            xycoords='axes fraction', box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

        def rank_to_color(rank, vmin=1, vmax=300):
            """
            Map a rank (1–300) to a hex color.
            Dark blue = best (1), grey = middle, dark red = worst (300).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from blue → grey → red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rank_cmap", ["#00008B", "#00FFFF", "#D3D3D3", "#FFA500", "#8B0000"]  # dark blue, cyan, light gray, orange, dark red
            )
            
            # Normalize rank to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(rank))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def percent_to_color(win_pct, vmin=0.0, vmax=1.0):
            """
            Map a win percentage (0.0–1.0) to a hex color.
            Dark blue = best (1.0), grey = middle (0.5), dark red = worst (0.0).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from red → grey → blue
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "percent_cmap", ["#8B0000", "#FFA500", "#D3D3D3", "#00FFFF", "#00008B"]  # dark red, orange, light gray, cyan, dark blue
            )
            
            # Normalize percentage to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(win_pct))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def get_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (value, rank) for a given team and column.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and stat columns.
                team (str): Team name to look up.
                column (str): Column name to extract.
                higher_is_better (bool): If True, high values rank better (1 = highest).
                                        If False, low values rank better (1 = lowest).
            """
            ascending = not higher_is_better
            ranks = df[column].rank(ascending=ascending, method="first").astype(int)

            value = df.loc[df['Team'] == team, column].values[0]
            rank = ranks.loc[df['Team'] == team].values[0]

            return value, rank
        
        def get_record_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (record_string, win_percentage, rank) for a given team and column containing W-L records.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and record columns.
                team (str): Team name to look up.
                column (str): Column name containing records in "W-L" format.
                higher_is_better (bool): If True, high win% ranks better (1 = highest).
                                        If False, low win% ranks better (1 = lowest).
            
            Returns:
                tuple: (record_string, win_percentage as float, rank as int)
            """
            def calculate_win_pct(record):
                """Convert 'W-L' string to win percentage."""
                if pd.isna(record) or record == '':
                    return 0.0
                parts = str(record).split('-')
                wins = int(parts[0])
                losses = int(parts[1])
                total = wins + losses
                return wins / total if total > 0 else 0.0
            
            # Calculate win percentages for all teams
            win_pcts = df[column].apply(calculate_win_pct)
            
            # Calculate ranks
            ascending = not higher_is_better
            ranks = win_pcts.rank(ascending=ascending, method="first").astype(int)
            
            # Get values for specified team
            team_idx = df['Team'] == team
            record_string = df.loc[team_idx, column].values[0]
            win_pct = win_pcts.loc[team_idx].values[0]
            rank = ranks.loc[team_idx].values[0]
            
            return record_string, win_pct
        
        def get_game_color(result):
            if "W" in result:
                return "palegreen"
            elif "L" in result:
                return "lightcoral"
            else:
                return "whitesmoke"

        def get_quadrant(opponent_net, location):
            thresholds = {
                "Home": [25, 50, 100, 307],
                "Neutral": [40, 80, 160, 307],
                "Away": [60, 120, 240, 307]
            }
            
            # Get the thresholds for the given location
            location_thresholds = thresholds.get(location, thresholds["Neutral"])
            
            # Determine quadrant based on opponent NET ranking
            if opponent_net <= location_thresholds[0]:
                return "Q1"
            elif opponent_net <= location_thresholds[1]:
                return "Q2"
            elif opponent_net <= location_thresholds[2]:
                return "Q3"
            else:
                return "Q4"

        def get_location_records(team, team_schedule):
            """Get home, away, and neutral records for a team"""
            
            team_schedule['is_win'] = team_schedule['Result'].str.startswith('W')
            
            records = {}
            for loc in ['Home', 'Away', 'Neutral']:
                group = team_schedule[team_schedule['Location'] == loc]
                wins = group['is_win'].sum()
                losses = len(group) - wins
                records[loc] = f"{int(wins)}-{int(losses)}"
            
            return records

        def display_stat_row(stats_ax, x1, y1, x2, y2, x3, y3, label, value, value_format, 
                            rank, ha, fontsize=16, alpha_val=0.9):
            """
            Display a stat row with label, value, and rank badge.
            
            Parameters:
            -----------
            stats_ax : matplotlib axis
                The axis to draw on
            x1, y1 : float
                Position for the label text
            x2, y2 : float
                Position for the value text
            x3, y3 : float
                Position for the rank badge (center)
            label : str
                The stat label (e.g., "RPG", "ERA")
            value : float
                The stat value
            value_format : str
                Format string for the value (e.g., ".2f", ".3f", ".1f")
            rank : str
                The rank badge text
            ha : str
                Horizontal alignment for label and value ("left" or "right")
            fontsize : int, optional
                Font size for all text (default=16)
            alpha_val : float, optional
                Alpha transparency for rank badge (default=0.3)
            """
            # Determine alignment based on ha parameter
            if ha == "left":
                label_ha = "left"
                value_ha = "right"
            else:  # ha == "right"
                label_ha = "right"
                value_ha = "left"
            
            # Display label
            stats_ax.text(x1, y1, label, fontsize=fontsize, fontweight='bold', ha=label_ha)
            
            # Display value with proper formatting
            stats_ax.text(x2, y2, f"{value:{value_format}}", fontsize=fontsize, fontweight='bold', ha=value_ha)
            
            # Display rank badge
            fixed_width_text(stats_ax, x3, y3, f"{rank}", width=0.15, height=0.05,
                            facecolor=rank_to_color(rank), alpha=alpha_val,
                            fontsize=fontsize, fontweight='bold', color=rank_text_color(rank))

        if os.path.exists(logo_folder):
            # Try to find logo for the team (keep spaces, don't replace with underscores)
            logo_path = os.path.join(logo_folder, f"{team_name}.png")

            if os.path.exists(logo_path):
                logo = Image.open(logo_path).convert("RGBA")

        team_schedule = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True)
        
        # Calculate layout
        num_games = len(team_schedule)
        games_per_col = 10
        num_cols = (num_games + games_per_col - 1) // games_per_col

        # ----------------
        # Load opponent logos
        # ----------------
        opponent_logos = {}
        opponents_set = set(team_schedule['Opponent'].dropna())
        for opponent in opponents_set:
            logo_path = os.path.join(logo_folder, f"{opponent}.png")
            if os.path.exists(logo_path):
                opponent_logos[opponent] = Image.open(logo_path).convert("RGBA")
            else:
                opponent_logos[opponent] = None

        # ----------------
        # Create figure with GridSpec
        # ----------------
        # Calculate figure width based on number of columns
        col_width = 1  # Width per column in inches
        stats_width = 5   # Width for stats section
        total_width = (num_cols * col_width) + stats_width
        
        fig = plt.figure(figsize=(total_width, 12), dpi=200)
        fig.patch.set_facecolor('#CECEB2')
        
        # Create main grid: schedule area (left) and stats area (right)
        main_gs = gridspec.GridSpec(1, 2, figure=fig, 
                                width_ratios=[num_cols * col_width, stats_width], 
                                left=0.02, right=0.98, wspace=0.02)
        
        # Create nested grid for schedule columns
        schedule_gs = gridspec.GridSpecFromSubplotSpec(games_per_col, num_cols, 
                                                    subplot_spec=main_gs[0],
                                                    hspace=0.01, wspace=0.01)
        
        # Create stats area
        alpha_val = 0.9
        stats_ax = fig.add_subplot(main_gs[1])
        stats_ax.set_xlim(0, 1)
        stats_ax.set_ylim(0, 1)
        stats_ax.axis('off')
        stats_ax.set_facecolor('#CECEB2')
        net_score, net_rank = get_value_and_rank(stats_and_metrics, team_name, "NET_Score")
        rpg, rpg_rank = get_value_and_rank(stats_and_metrics, team_name, "RPG")
        ba, ba_rank = get_value_and_rank(stats_and_metrics, team_name, "BA")
        obp, obp_rank = get_value_and_rank(stats_and_metrics, team_name, "OBP")
        slg, slg_rank = get_value_and_rank(stats_and_metrics, team_name, "SLG")
        ops, ops_rank = get_value_and_rank(stats_and_metrics, team_name, "OPS")
        iso, iso_rank = get_value_and_rank(stats_and_metrics, team_name, "ISO")
        wOBA, wOBA_rank = get_value_and_rank(stats_and_metrics, team_name, "wOBA")
        era, era_rank = get_value_and_rank(stats_and_metrics, team_name, "ERA", False)
        whip, whip_rank = get_value_and_rank(stats_and_metrics, team_name, "WHIP", False)
        kp9, kp9_rank = get_value_and_rank(stats_and_metrics, team_name, "KP9")
        lob, lob_rank = get_value_and_rank(stats_and_metrics, team_name, "LOB%")
        kbb, kbb_rank = get_value_and_rank(stats_and_metrics, team_name, "K/BB")
        fip, fip_rank = get_value_and_rank(stats_and_metrics, team_name, "FIP", False)
        pct, pct_rank = get_value_and_rank(stats_and_metrics, team_name, "PCT")
        rating, rating_rank = get_value_and_rank(stats_and_metrics, team_name, "Rating")
        rqi, rqi_rank = get_value_and_rank(stats_and_metrics, team_name, "resume_quality")
        sos, sos_rank = get_value_and_rank(stats_and_metrics, team_name, "avg_expected_wins", False)
        war, war_rank = get_value_and_rank(stats_and_metrics, team_name, "fWAR")
        wpoe, wpoe_rank = get_value_and_rank(stats_and_metrics, team_name, "wpoe_pct")
        pythag, pythag_rank = get_value_and_rank(stats_and_metrics, team_name, "PYTHAG")
        record, record_rank = get_record_value_and_rank(stats_and_metrics, team_name, "Record")
        q1, q1_rank = get_value_and_rank(stats_and_metrics, team_name, "Q1")
        q2, q2_rank = get_value_and_rank(stats_and_metrics, team_name, "Q2")
        q3, q3_rank = get_value_and_rank(stats_and_metrics, team_name, "Q3")
        q4, q4_rank = get_value_and_rank(stats_and_metrics, team_name, "Q4")
        location_records = get_location_records(team_name, team_schedule)
        home_record = location_records.get("Home", "0-0")
        away_record = location_records.get("Away", "0-0")
        neutral_record = location_records.get("Neutral", "0-0")
        
        # Add team stats
        stats_ax.text(0.5, 0.973, f"#{net_rank} {team_name}", fontsize=24, fontweight='bold', 
                    ha='center', va='center')

        stats_ax.text(0.23, 0.923, f"OFFENSE", fontsize=16, fontweight='bold', 
                    ha='center', va='center')
        
        stats_ax.hlines(y=0.9, xmin=0.0, xmax=1, colors='black', linewidth=1)
        stats_ax.vlines(x=0.5, ymin=0.55, ymax=0.9, colors='black', linewidth=1)

        # axis, stat name coords, stat value coords, stat rank coords, ..., alignment of stat name
        display_stat_row(stats_ax, 0.0, 0.868, 0.48, 0.868, 0.23, 0.873+0.002, "RPG", rpg, ".2f", rpg_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.818, 0.48, 0.818, 0.23, 0.823+0.002, "BA", ba, ".3f", ba_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.768, 0.48, 0.768, 0.23, 0.773+0.002, "OBP", obp, ".3f", obp_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.718, 0.48, 0.718, 0.23, 0.723+0.002, "SLG", slg, ".3f", slg_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.668, 0.48, 0.668, 0.23, 0.673+0.002, "OPS", ops, ".3f", ops_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.618, 0.48, 0.618, 0.23, 0.623+0.002, "ISO", iso, ".3f", iso_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.568, 0.48, 0.568, 0.23, 0.573+0.002, "wOBA", wOBA, ".3f", wOBA_rank, "left")

        # PITCHING section header
        stats_ax.text(0.77, 0.923, f"PITCHING", fontsize=16, fontweight='bold', 
                    ha='center', va='center')

        # Pitching stats
        display_stat_row(stats_ax, 1.0, 0.868, 0.52, 0.868, 0.77, 0.873+0.002, "ERA", era, ".2f", era_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.818, 0.52, 0.818, 0.77, 0.823+0.002, "WHIP", whip, ".2f", whip_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.768, 0.52, 0.768, 0.77, 0.773+0.002, "K/9", kp9, ".1f", kp9_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.718, 0.52, 0.718, 0.77, 0.723+0.002, "LOB%", lob, ".3f", lob_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.668, 0.52, 0.668, 0.77, 0.673+0.002, "K/BB", kbb, ".2f", kbb_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.618, 0.52, 0.618, 0.77, 0.623+0.002, "FIP", fip, ".2f", fip_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.568, 0.52, 0.568, 0.77, 0.573+0.002, "PCT", pct, ".3f", pct_rank, "right")

        # TEAM METRICS section header
        stats_ax.text(0.5, 0.523, f"TEAM METRICS", fontsize=16, fontweight='bold', 
                    ha='center', va='center')
        stats_ax.hlines(y=0.5, xmin=0.0, xmax=1, colors='black', linewidth=1)
        stats_ax.vlines(x=0.5, ymin=0.25, ymax=0.5, colors='black', linewidth=1)

        # Team metrics - left column
        display_stat_row(stats_ax, 0.0, 0.468, 0.48, 0.468, 0.23, 0.473+0.002, 
                        "NET", net_score, ".3f", net_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.418, 0.48, 0.418, 0.23, 0.423+0.002, 
                        "RAT", rating, ".2f", rating_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.368, 0.48, 0.368, 0.23, 0.373+0.002, 
                        "RQI", rqi, ".2f", rqi_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.318, 0.48, 0.318, 0.23, 0.323+0.002, 
                        "SOS", sos, ".3f", sos_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.268, 0.48, 0.268, 0.23, 0.273+0.002, 
                        "WAR", war, ".2f", war_rank, "left")

        # Team metrics - right column
        display_stat_row(stats_ax, 1.0, 0.468, 0.52, 0.468, 0.77, 0.473+0.002, 
                        "WPOE", wpoe, ".2f", wpoe_rank, "right")

        display_stat_row(stats_ax, 1.0, 0.418, 0.52, 0.418, 0.77, 0.423+0.002, 
                        "PYT", pythag, ".3f", pythag_rank, "right")
        
        stats_ax.text(0.0, 0.218, "REC", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.218, f"{record}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.168, "Q1", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.168, f"{q1}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.118, "Q2", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.118, f"{q2}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.068, "Q3", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.068, f"{q3}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.018, "Q4", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.018, f"{q4}", fontsize=16, fontweight='bold', ha='center')
        
        stats_ax.text(1.0, 0.368, "HOME", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.368, f"{home_record}", fontsize=16, fontweight='bold', ha='left')

        stats_ax.text(1.0, 0.318, "AWAY", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.318, f"{away_record}", fontsize=16, fontweight='bold', ha='left')

        stats_ax.text(1.0, 0.268, "NEUTRAL", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.268, f"{neutral_record}", fontsize=16, fontweight='bold', ha='left')

        # Add team logo
        if logo is not None:
            plot_logo(stats_ax, logo, (0.685, 0.125), zoom=0.2)

        # Fixed positions within each game axis (0-1 scale)
        logo_x_offset = 0.5       # Logo position
        rank_x_offset = 0.05       # NET ranking position
        loc_x_offset = 0.05       # Team name position (center-left)
        quad_x_offset = 0.95       # Quadrant position
        prob_x_offset = 0.95       # Win probability position

        # ----------------
        # Create axes for each game
        # ----------------
        for idx, (_, game_row) in enumerate(team_schedule.iterrows()):
            row = idx % games_per_col
            col = idx // games_per_col
            
            # Create axis for this game
            game_ax = fig.add_subplot(schedule_gs[row, col])
            game_ax.set_xlim(0, 1)
            game_ax.set_ylim(0, 1)
            game_ax.axis('off')
            
            if pd.notna(game_row.get('home_net')) and pd.notna(game_row.get('away_net')):
                if game_row['home_team'] == team_name:
                    opponent_net = int(game_row['away_net'])
                    if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                    else:
                        bottom_right_text = round(100 * game_row['home_win_prob'])
                        percent = "%"
                else:
                    opponent_net = int(game_row['home_net'])
                    if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                    else:
                        bottom_right_text = round(100 * (1 - game_row['home_win_prob']))
                        percent = "%"
                
                quadrant = get_quadrant(opponent_net, game_row['Location'])
                has_net_data = True
            else:
                opponent_net = ""
                if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                else:
                    bottom_right_text = round(100 * game_row['home_win_prob'])
                    percent = "%"
                quadrant = ""
                has_net_data = False

            game_color = get_game_color(game_row["Result"])
            if percent == "":
                max_val = round(team_schedule['resume_quality'].max(), 2)
                min_val = round(team_schedule['resume_quality'].min(), 2)

                if bottom_right_text == max_val:
                    game_color = 'mediumseagreen'
                elif bottom_right_text == min_val:
                    game_color = 'indianred'
            bg_rect = Rectangle((0, 0), 1, 1, transform=game_ax.transAxes,
                        facecolor=game_color, edgecolor='black', 
                        linewidth=1, zorder=0)
            game_ax.add_patch(bg_rect)
            
            # Add opponent logo
            opponent = game_row["Opponent"]
            if pd.isna(opponent):
                # handle missing opponent (e.g., Non D-I or bye week)
                game_ax.text(0.5, 0.5, "Non D-I", fontsize=10, ha='center', va='center', color='black')
            else:
                if opponent in opponent_logos and opponent_logos[opponent] is not None:
                    plot_logo(game_ax, opponent_logos[opponent], (logo_x_offset, 0.5), zoom=0.04)
                else:
                    # fallback if opponent not found or logo is None
                    game_ax.text(0.5, 0.5, opponent, fontsize=10, ha='center', va='center', color='black')
            
            # Add NET ranking
            game_ax.text(rank_x_offset, 0.88, f"#{opponent_net}", 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='left', va='center')
            
            # Add opponent name
            if game_row['Location'] == 'Home':
                loc = ""
            elif game_row['Location'] == 'Away':
                loc = "@"
            else:
                loc = "vs"
            game_ax.text(loc_x_offset, 0.1, loc, 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='left', va='center')
            
            # Add quadrant
            game_ax.text(quad_x_offset, 0.88, quadrant, 
                        fontsize=12, fontweight='bold', color='darkblue', 
                        ha='right', va='center')
            
            # Add win prob
            game_ax.text(prob_x_offset, 0.1, f"{bottom_right_text}{percent}", 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='right', va='center')

        plt.tight_layout()
        # Save to BytesIO and get data before cleanup
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        
        # Get image data before closing
        image_data = buf.getvalue()
        
        # Aggressive cleanup
        plt.close(fig)
        fig.clf()
        del fig
        del buf
        gc.collect()
        
        return Response(content=image_data, media_type="image/png")
        
    except HTTPException as e:
            raise e
    except Exception as e:
        print(f"Error generating team profile: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Always close the logo image
        if logo is not None:
            logo.close()
            del logo
        
        # Close all opponent logos
        for opponent, opp_logo in opponent_logos.items():
            if opp_logo is not None:
                opp_logo.close()
        opponent_logos.clear()
        
        gc.collect()

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
    
@app.get("/api/cbase/conferences")
def get_baseball_conferences():
    """Get list of all conferences"""
    try:
        modeling_stats, _ = load_baseball_data()
        conferences = sorted(modeling_stats['Conference'].unique().tolist())
        return {"conferences": conferences}
    except Exception as e:
        print(f"Error getting conferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

###################################
## CONFERENCE TOURNAMENT SECTION ##
###################################

def Baseball_PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

def get_rating(team, stats_df):
    RATING_ADJUSTMENTS = {
        "Liberty": 0.3, "Xavier": 0.3, "Southeastern La.": 0.3, "UTRGV": 0.3,
        "Yale": 0.3, "Omaha": 0.3, "Stetson": 0.3, "Duke": 0.3,
        "Georgetown": 0.3, "High Point": 0.3, "Col. of Charleston": 0.3,
        "Illinois St.": 0.3, "Cal St. Fullerton": 0.3, "Wright St.": 0.3
    }
    base_rating = stats_df.loc[stats_df["Team"] == team, "Rating"].values[0]
    return base_rating + RATING_ADJUSTMENTS.get(team, 0)

def simulate_game(team_a, team_b, ratings, location="Neutral"):
    prob = Baseball_PEAR_Win_Prob(ratings[team_a], ratings[team_b], location=location) / 100
    return team_a if random.random() < prob else team_b

def simulate_best_of_three_series(team_a, team_b, ratings, location):
    """
    Simulate a best-of-three series with team_a as the home team.
    Returns the winner.
    """
    wins = {team_a: 0, team_b: 0}
    while wins[team_a] < 2 and wins[team_b] < 2:
        winner = simulate_game(team_a, team_b, ratings, location=location)
        wins[winner] += 1
    return team_a if wins[team_a] == 2 else team_b

def plot_tournament_odds_table(final_df, row_height_multiplier, conference, title_y, subtitle_y, cell_height):
    def normalize(value, min_val, max_val):
        """Normalize values between 0 and 1 for colormap."""
        if pd.isna(value) or value == 0:
            return 0
        return (value - min_val) / (max_val - min_val)

    min_value = final_df.iloc[:, 1:].replace(0, np.nan).min().min()
    max_value = final_df.iloc[:, 1:].max().max()
    
    cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

    fig, ax = plt.subplots(figsize=(8, len(final_df) * row_height_multiplier), dpi=125)
    fig.patch.set_facecolor('#CECEB2')
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=final_df.values,
        colLabels=final_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#CECEB2'] * len(final_df.columns)
    )

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)

        if i == 0:  # Header row
            cell.set_facecolor('#CECEB2')
            cell.set_text_props(fontsize=16, weight='bold', color='black')
        elif j == 0:  # Team names column
            cell.set_facecolor('#CECEB2')
            cell.set_text_props(fontsize=16, weight='bold', color='black')
        else:
            value = final_df.iloc[i-1, j]
            normalized_value = normalize(value, min_value, max_value)
            cell.set_facecolor(cmap(normalized_value))
            cell.set_text_props(fontsize=16, weight='bold', color='black')
            if value <= 0.9:
                cell.get_text().set_text("<1%")
            else:
                cell.get_text().set_text(f"{value:.1f}%")
        
        cell.set_height(cell_height)

    plt.text(0, title_y, f'Odds to Win {conference} Tournament', fontsize=24, fontweight='bold', ha='center')
    plt.text(0, subtitle_y, "@PEARatings", fontsize=16, fontweight='bold', ha='center')
    return fig

def double_elimination_bracket(teams, stats_and_metrics, num_simulations=1000):
    """
    Simulate a 4-team double elimination bracket and return win probabilities.
    Teams must be provided in seeding order: [seed1, seed2, seed3, seed4]
    """
    results = defaultdict(int)
    r = {team: get_rating(team, stats_and_metrics) for team in teams}

    for _ in range(num_simulations):
        t1, t2, t3, t4 = teams
        w1 = simulate_game(t1, t4, r)
        l1 = t4 if w1 == t1 else t1
        w2 = simulate_game(t2, t3, r)
        l2 = t3 if w2 == t2 else t2
        w3 = simulate_game(l2, l1, r)
        w4 = simulate_game(w1, w2, r)
        l4 = w2 if w4 == w1 else w1
        w5 = simulate_game(l4, w3, r)
        final_prob = Baseball_PEAR_Win_Prob(r[w4], r[w5]) / 100
        w6 = w4 if random.random() < final_prob else w5

        # Double-elim logic: if w4 loses once, play again
        champion = w6 if w6 == w4 else (w4 if random.random() < final_prob else w5)
        results[champion] += 1

    return defaultdict(float, {team: round(results[team] / num_simulations, 3) for team in teams})

def simulate_overall_tournament(bracket_one_probs, bracket_two_probs, stats_and_metrics, num_simulations=1000):
    """
    Simulate a final between two bracket winners using weighted probabilities.
    Returns a defaultdict with tournament win percentages.
    """
    final_results = defaultdict(int)

    bracket_one_teams = list(bracket_one_probs.keys())
    bracket_two_teams = list(bracket_two_probs.keys())
    ratings = {team: get_rating(team, stats_and_metrics) for team in bracket_one_teams + bracket_two_teams}

    for _ in range(num_simulations):
        # Draw winners from each bracket based on their win probabilities
        winner_one = random.choices(bracket_one_teams, weights=[bracket_one_probs[t] for t in bracket_one_teams])[0]
        winner_two = random.choices(bracket_two_teams, weights=[bracket_two_probs[t] for t in bracket_two_teams])[0]

        # Simulate the final
        champ = simulate_best_of_three_series(winner_one, winner_two, ratings, "Neutral")
        final_results[champ] += 1

    return defaultdict(float, {team: round(wins / num_simulations, 3) for team, wins in final_results.items()})

def two_playin_games_to_four_team_double_elimination(teams, stats_and_metrics, num_simulations=1000):
    """
    Simulates a 6-team hybrid tournament:
    - Seeds 3-6 play two play-in games.
    - Two winners join seeds 1-2 in a 4-team double elimination bracket.
    Returns a DataFrame with each team's odds of reaching double elimination and winning the tournament.
    """
    made_double_elim = defaultdict(int)
    tournament_wins = defaultdict(int)
    r = {team: get_rating(team, stats_and_metrics) for team in teams}

    seeds = {i + 1: teams[i] for i in range(6)}

    for _ in range(num_simulations):
        # Play-in round
        gA_winner = simulate_game(seeds[3], seeds[6], r)
        gB_winner = simulate_game(seeds[4], seeds[5], r)

        double_elim_teams = {seeds[1], seeds[2], gA_winner, gB_winner}

        # Track double elim appearances
        for team in double_elim_teams:
            made_double_elim[team] += 1

        # Reseed play-in winners
        playin_winners = [(s, t) for s, t in seeds.items() if t in [gA_winner, gB_winner]]
        sorted_winners = sorted(playin_winners, key=lambda x: x[0])
        lowest_seed_team = sorted_winners[0][1]
        higher_seed_team = sorted_winners[1][1]

        # Simulate bracket
        bracket_result = double_elimination_bracket(
            [seeds[1], seeds[2], lowest_seed_team, higher_seed_team],
            stats_and_metrics,
            num_simulations=1
        )
        winner = max(bracket_result.items(), key=lambda x: x[1])[0]
        tournament_wins[winner] += 1

    # Final result formatting
    results = []
    for team in teams:
        reach_double = 1.0 if team in teams[:2] else made_double_elim[team] / num_simulations
        win_tourney = tournament_wins[team] / num_simulations
        results.append({
            "Team": team,
            "Make Double Elim": round(reach_double * 100, 1),
            "Win Tournament": round(win_tourney * 100, 1)
        })

    return pd.DataFrame(results)

def simulate_and_run_8_team_double_elim(teams, stats_and_metrics, num_simulations=1000): 
    results = defaultdict(int)
    ratings = {team: get_rating(team, stats_and_metrics) for team in teams}

    for _ in range(num_simulations):
        # Round 1 matchups (seed-style: 1v8, 2v7, etc.)
        matchups = [(teams[0], teams[7]), (teams[3], teams[4]), (teams[2], teams[5]), (teams[1], teams[6])]

        # Round 1 (WB)
        wb_round1_winners = []
        lb_round1_losers = []
        for t1, t2 in matchups:
            winner = simulate_game(t1, t2, ratings, location="Neutral")
            loser = t2 if winner == t1 else t1
            wb_round1_winners.append(winner)
            lb_round1_losers.append(loser)

        # Round 2A (WB)
        wb_sf1 = simulate_game(wb_round1_winners[0], wb_round1_winners[1], ratings, location="Neutral")
        wb_sf2 = simulate_game(wb_round1_winners[2], wb_round1_winners[3], ratings, location="Neutral")
        wb_losers = [t for t in wb_round1_winners if t != wb_sf1 and t != wb_sf2]

        # LB Round 1 (elimination)
        lb_r1_1 = simulate_game(lb_round1_losers[0], lb_round1_losers[1], ratings, location="Neutral")
        lb_r1_2 = simulate_game(lb_round1_losers[2], lb_round1_losers[3], ratings, location="Neutral")

        # LB Round 2
        lb_r2_1 = simulate_game(lb_r1_1, wb_losers[0], ratings, location="Neutral")
        lb_r2_2 = simulate_game(lb_r1_2, wb_losers[1], ratings, location="Neutral")

        # LB Semifinal
        lb_sf = simulate_game(lb_r2_1, lb_r2_2, ratings, location="Neutral")

        # WB Final
        wb_final = simulate_game(wb_sf1, wb_sf2, ratings, location="Neutral")
        wb_final_loser = wb_sf2 if wb_final == wb_sf1 else wb_sf1

        # LB Final
        lb_final = simulate_game(lb_sf, wb_final_loser, ratings, location="Neutral")

        # Championship
        champ = simulate_game(wb_final, lb_final, ratings, location="Neutral")

        results[champ] += 1

    # Return the results in a DataFrame
    df = pd.DataFrame([
        {"Team": team, "Win Tournament": round(results[team] / num_simulations * 100, 1)}
        for team in teams
    ])

    return df

def single_elimination_16_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Round 1", "Round 2", "Quarterfinals", "Semifinals", "Final", "Champion"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        progress = {team: None for team in seed_order}

        # Round 1 (Play-in)
        play_in_pairs = [(8, 15), (9, 14), (10, 13), (11, 12)]
        round1_winners = [simulate_game(seed_order[a], seed_order[b], ratings, location="Neutral") for a, b in play_in_pairs]
        for winner, (a, b) in zip(round1_winners, play_in_pairs):
            progress[seed_order[a]] = progress[seed_order[b]] = "Round 1"
            progress[winner] = "Round 2"

        # Round 2
        round2_winners = [simulate_game(seed_order[seed], round1_winners[i], ratings, location="Neutral") for i, seed in enumerate([4, 5, 6, 7])]
        for winner, seed in zip(round2_winners, [4, 5, 6, 7]):
            progress[seed_order[seed]] = progress[round1_winners[seed-4]] = "Round 2"
            progress[winner] = "Quarterfinals"

        # Quarterfinals
        qf_winners = [simulate_game(seed_order[seed], round2_winners[i], ratings, location="Neutral") for i, seed in enumerate([0, 1, 2, 3])]
        for winner, seed in zip(qf_winners, [0, 1, 2, 3]):
            progress[seed_order[seed]] = progress[round2_winners[seed]] = "Quarterfinals"
            progress[winner] = "Semifinals"

        # Semifinals
        sf_winners = [simulate_game(qf_winners[i], qf_winners[i+1], ratings, location="Neutral") for i in [0, 2]]
        for winner in sf_winners:
            progress[winner] = "Final"

        # Final
        winner = simulate_game(sf_winners[0], sf_winners[1], ratings, location="Neutral")
        progress[winner] = "Champion"

        # Record outcomes
        for team, reached in progress.items():
            if reached:
                for i in range(rounds.index(reached) + 1):
                    team_stats[team][rounds[i]] += 1

    # Convert counts to percentages
    result_df = pd.DataFrame.from_dict(team_stats, orient="index")
    result_df = result_df.applymap(lambda x: round(100 * x / num_simulations, 1))
    result_df = result_df.drop(columns=["Round 1"]).reset_index().rename(columns={"index": "Team"})
    
    return result_df

def single_elimination_14_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Round 1", "Quarterfinals", "Semifinals", "Final", "Champion"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        progress = {team: None for team in seed_order}

        # Round 1: Seeds 5–12 play-in
        play_in_pairs = [(4, 11), (5, 10), (6, 9), (7, 8)]
        round1_winners = [simulate_game(seed_order[a], seed_order[b], ratings, location="Neutral") for a, b in play_in_pairs]
        for winner, (a, b) in zip(round1_winners, play_in_pairs):
            progress[seed_order[a]] = progress[seed_order[b]] = "Round 1"
            progress[winner] = "Quarterfinals"

        # Quarterfinals: Seeds 1–4 vs Round 1 winners
        qf_winners = [simulate_game(seed_order[seed], round1_winners[i], ratings, location="Neutral") for i, seed in enumerate([0, 1, 2, 3])]
        for winner, seed in zip(qf_winners, [0, 1, 2, 3]):
            progress[seed_order[seed]] = progress[round1_winners[seed-4]] = "Quarterfinals"
            progress[winner] = "Semifinals"

        # Semifinals
        sf_winners = [simulate_game(qf_winners[i], qf_winners[i+1], ratings, location="Neutral") for i in [0, 2]]
        for winner in sf_winners:
            progress[winner] = "Final"

        # Final
        winner = simulate_game(sf_winners[0], sf_winners[1], ratings, location="Neutral")
        progress[winner] = "Champion"

        # Record outcomes
        for team, reached in progress.items():
            if reached:
                for i in range(rounds.index(reached) + 1):
                    team_stats[team][rounds[i]] += 1

    # Format result as DataFrame
    result_df = pd.DataFrame.from_dict(team_stats, orient="index")
    result_df = result_df.applymap(lambda x: round(100 * x / num_simulations, 1))
    result_df = result_df.drop(columns=["Round 1"]).reset_index().rename(columns={"index": "Team"})

    return result_df

def double_elimination_7_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Double Elim", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        progress = defaultdict(lambda: {"Double Elim": 0, "Win Tournament": 0})
        winners_r1 = [simulate_game(seed_order[1], seed_order[6], ratings, location="Neutral"),
                      simulate_game(seed_order[2], seed_order[5], ratings, location="Neutral"),
                      simulate_game(seed_order[3], seed_order[4], ratings, location="Neutral")]
        losers_r1 = [team for team in [seed_order[1], seed_order[6], seed_order[2], seed_order[5], seed_order[3], seed_order[4]] if team not in winners_r1]

        # Round 1 - Mark all teams
        for t in winners_r1 + losers_r1 + [seed_order[0]]:
            progress[t]["Double Elim"] += 1

        # Round 2 (Winners Bracket)
        wb2_winners = [simulate_game(seed_order[0], winners_r1[0], ratings, location="Neutral"),
                       simulate_game(winners_r1[1], winners_r1[2], ratings, location="Neutral")]

        # Loser's bracket games
        lb1 = simulate_game(losers_r1[0], losers_r1[1], ratings, location="Neutral")
        lb2 = simulate_game(losers_r1[2], wb2_winners[0], ratings, location="Neutral")
        lb3 = simulate_game(wb2_winners[1], lb1, ratings, location="Neutral")

        # Loser's Bracket Final
        lb_final = simulate_game(lb2, lb3, ratings, location="Neutral")

        # Winner's Bracket Final
        wb_final = simulate_game(wb2_winners[0], wb2_winners[1], ratings, location="Neutral")

        # Championship
        champ = simulate_game(wb_final, lb_final, ratings, location="Neutral") if wb_final != lb_final else wb_final

        progress[champ]["Win Tournament"] += 1

        # Record outcomes
        for team in progress:
            for r in rounds:
                team_stats[team][r] += progress[team][r]

    # Format result as DataFrame
    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"}).drop(columns=["Double Elim"])
    return df

def double_elimination_6_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Double Elim", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        progress = defaultdict(lambda: {"Double Elim": 0, "Win Tournament": 0})

        # Round 1: #3 vs #6, #4 vs #5
        winners_r1 = [simulate_game(seed_order[2], seed_order[5], ratings, location="Neutral"),
                      simulate_game(seed_order[3], seed_order[4], ratings, location="Neutral")]
        losers_r1 = [team for team in [seed_order[2], seed_order[5], seed_order[3], seed_order[4]] if team not in winners_r1]

        # Mark all teams in Round 1
        for t in winners_r1 + losers_r1 + [seed_order[0], seed_order[1]]:
            progress[t]["Double Elim"] += 1

        # Round 2: Winners Bracket
        wb2_winners = [simulate_game(seed_order[0], winners_r1[0], ratings, location="Neutral"),
                       simulate_game(seed_order[1], winners_r1[1], ratings, location="Neutral")]

        # Elimination games
        lb1 = simulate_game(losers_r1[0], losers_r1[1], ratings, location="Neutral")
        lb2 = simulate_game(wb2_winners[0], lb1, ratings, location="Neutral")
        lb3 = simulate_game(wb2_winners[1], lb2, ratings, location="Neutral")

        # Loser's Bracket Final
        lb_final = lb3

        # Winner's Bracket Final
        wb_final = simulate_game(wb2_winners[0], wb2_winners[1], ratings, location="Neutral")

        # Championship
        champ = simulate_game(wb_final, lb_final, ratings, location="Neutral") if wb_final != lb_final else wb_final

        progress[champ]["Win Tournament"] += 1

        # Record outcomes
        for team in progress:
            for r in rounds:
                team_stats[team][r] += progress[team][r]

    # Format result as DataFrame
    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"}).drop(columns=['Double Elim'])
    return df

def simulate_pool_play_tournament(seed_order, stats_and_metrics, num_simulations=1000):
    pools = {
        "A": [seed_order[0], seed_order[7], seed_order[11]],
        "B": [seed_order[1], seed_order[6], seed_order[10]],
        "C": [seed_order[2], seed_order[5], seed_order[9]],
        "D": [seed_order[3], seed_order[4], seed_order[8]],
    }
    rounds = ["Win Pool", "Make Final", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        pool_winners = {}

        # Simulate pool play
        for pool_name, teams in pools.items():
            wins = {team: 0 for team in teams}
            matchups = [(teams[0], teams[1]), (teams[0], teams[2]), (teams[1], teams[2])]
            for team_a, team_b in matchups:
                winner = simulate_game(team_a, team_b, ratings, location="Neutral")
                wins[winner] += 1

            pool_winner = max(wins, key=lambda team: (wins[team], -seed_order.index(team)))
            pool_winners[pool_name] = pool_winner
            team_stats[pool_winner]["Win Pool"] += 1

        # Semifinals: A vs D, B vs C
        finalists = [simulate_game(pool_winners["A"], pool_winners["D"], ratings, location="Neutral"),
                    simulate_game(pool_winners["B"], pool_winners["C"], ratings, location="Neutral")]
        for winner in finalists:
            team_stats[winner]["Make Final"] += 1

        # Final
        final_winner = simulate_game(finalists[0], finalists[1], ratings, location="Neutral")
        team_stats[final_winner]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"})
    return df

def simulate_playin_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    team_stats = {team: {"Double Elim": 0, "Win Tournament": 0} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        # Play-in between #4 and #5
        playin_winner = simulate_game(seed_order[3], seed_order[4], ratings, location="Neutral")
        team_stats[playin_winner]["Double Elim"] += 1

        # 4-team double elimination bracket
        bracket_teams = [seed_order[0], seed_order[1], seed_order[2], playin_winner]
        bracket_result = double_elimination_bracket(bracket_teams, stats_and_metrics, num_simulations=1)
        champion = max(bracket_result.items(), key=lambda x: x[1])[0]
        team_stats[champion]["Win Tournament"] += 1

    # Format results
    df = pd.DataFrame.from_dict(team_stats, orient="index").reset_index().rename(columns={"index": "Team"})
    df["Double Elim"] = df["Double Elim"].apply(lambda x: round(100 * x / num_simulations, 1))
    df["Win Tournament"] = df["Win Tournament"].apply(lambda x: round(100 * x / num_simulations, 1))
    for i in range(3):
        df.loc[df["Team"] == seed_order[i], "Double Elim"] = 100.0
    return df

def simulate_playins_to_6team_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    team_stats = {team: {"Double Elim": 0, "Win Tournament": 0} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        # Simulate play-in games
        winner_5_8 = simulate_game(seed_order[4], seed_order[7], ratings, location="Neutral")
        winner_6_7 = simulate_game(seed_order[5], seed_order[6], ratings, location="Neutral")

        # Build 6-team bracket
        advancing_teams = seed_order[:4] + [winner_5_8, winner_6_7]
        for team in advancing_teams:
            team_stats[team]["Double Elim"] += 1

        # Run one sim of 6-team double elimination
        result_df = double_elimination_6_teams(advancing_teams, stats_and_metrics, num_simulations=1)
        result_df["Win Tournament"] = result_df["Win Tournament"] / 100  # Undo percentage scaling
        champ = result_df.loc[result_df["Win Tournament"] == 1.0, "Team"].values[0]
        team_stats[champ]["Win Tournament"] += 1

    # Final formatting
    df = pd.DataFrame.from_dict(team_stats, orient="index").reset_index().rename(columns={"index": "Team"})
    df["Double Elim"] = df["Double Elim"].apply(lambda x: round(100 * x / num_simulations, 1))
    df["Win Tournament"] = df["Win Tournament"].apply(lambda x: round(100 * x / num_simulations, 1))
    return df

def simulate_mvc_tournament(seed_order, stats_and_metrics, num_simulations=1000):
    assert len(seed_order) == 8, "Seed order must have exactly 8 teams."

    def remove_team_with_two_losses(losses):
        for team, loss_count in list(losses.items()):
            if loss_count >= 2:
                del losses[team]

    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}
    
    make_de_stats = defaultdict(int)
    win_stats = defaultdict(int)

    for _ in range(num_simulations):
        # Day 1 - Single elimination: 5 vs 8 and 6 vs 7
        team5, team6, team7, team8 = seed_order[4], seed_order[5], seed_order[6], seed_order[7]

        winner_5v8 = simulate_game(team5, team8, ratings, location="Neutral")
        winner_6v7 = simulate_game(team6, team7, ratings, location="Neutral")
        if winner_5v8 == team5:
            playin_winners = [winner_5v8, winner_6v7]
        else:
            playin_winners = [winner_6v7, winner_5v8]

        for team in playin_winners:
            make_de_stats[team] += 1

        for team in seed_order[:4]:
            make_de_stats[team] += 1  # Top 4 seeds always make DE

        # Day 2 - Start double elimination (6 teams)
        de_teams = seed_order[:4] + playin_winners
        r = {team: ratings[team] for team in de_teams}
        losses = {team: 0 for team in de_teams}

        w3 = simulate_game(de_teams[2], de_teams[3], r)
        w4 = simulate_game(de_teams[0], de_teams[5], r)
        w5 = simulate_game(de_teams[1], de_teams[4], r)
        l3 = de_teams[3] if w3 == de_teams[2] else de_teams[2]
        l4 = de_teams[5] if w4 == de_teams[0] else de_teams[0]
        l5 = de_teams[4] if w5 == de_teams[1] else de_teams[1]
        losses[l3] += 1
        losses[l4] += 1
        losses[l5] += 1
        remove_team_with_two_losses(losses)

        w6 = simulate_game(l5, l4, r)
        w7 = simulate_game(l3, w4, r)
        w8 = simulate_game(w3, w5, r)
        l6 = l5 if w6 == l4 else l4
        l7 = l3 if w7 == w4 else w4
        l8 = w3 if w8 == w5 else w5
        losses[l6] += 1
        losses[l7] += 1
        losses[l8] += 1
        remove_team_with_two_losses(losses)

        if len(losses) == 4:
            w9 = simulate_game(w6, l8, r)
            l9 = w6 if w9 == l8 else l8
            losses[l9] += 1
            remove_team_with_two_losses(losses)

            w10 = simulate_game(w7, w8, r)
            l10 = w7 if w10 == w8 else w8
            losses[l10] += 1
            remove_team_with_two_losses(losses)

            w11 = simulate_game(w9, l10, r)
            l11 = w9 if w11 == l10 else l10
            losses[l11] += 1
            remove_team_with_two_losses(losses)

            w12 = simulate_game(w11, w10, r)
            l12 = w11 if w12 == w10 else w10
            losses[l12] += 1
            remove_team_with_two_losses(losses)

            if len(losses) == 1:
                champion = w12
            else:
                champion = simulate_game(w12, l12, r)
        else:
            w9 = simulate_game(l7, l8, r)
            l9 = l7 if w9 == l8 else l8
            losses[l9] += 1
            remove_team_with_two_losses(losses)

            w10 = simulate_game(w6, w7, r)
            l10 = w6 if w10 == w7 else w7
            losses[l10] += 1
            remove_team_with_two_losses(losses)

            w11 = simulate_game(w9, w8, r)
            l11 = w9 if w11 == w8 else w8
            losses[l11] += 1
            remove_team_with_two_losses(losses)

            if len(losses) == 2:
                w12 = simulate_game(w11, w10, r)
                l12 = w11 if w12 == w10 else w10
                losses[l12] += 1
                remove_team_with_two_losses(losses)

                if len(losses) == 1:
                    champion = w12
                else:
                    champion = simulate_game(w12, l12, r)  
            else:
                w12 = simulate_game(l11, w10, r)
                l12 = l11 if w12 == w10 else w10
                losses[l12] += 1
                remove_team_with_two_losses(losses)

                champion = simulate_game(w12, w11, r)

        win_stats[champion] += 1

    result = pd.DataFrame({
        "Team": seed_order,
        "Double Elim": [round(100 * make_de_stats[t] / num_simulations, 1) for t in seed_order],
        "Win Tournament": [round(100 * win_stats[t] / num_simulations, 1) for t in seed_order]
    })

    return result

def simulate_two_playin_rounds_to_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}
    tracker = {team: {"Round 2": 0, "Make Double Elim": 0, "Win Tournament": 0} for team in seed_order}

    for _ in range(num_simulations):
        # Round 1 Play-ins
        win_5v8 = simulate_game(seed_order[4], seed_order[7], ratings, location="Neutral")
        win_6v7 = simulate_game(seed_order[5], seed_order[6], ratings, location="Neutral")

        tracker[win_5v8]["Round 2"] += 1
        tracker[win_6v7]["Round 2"] += 1

        # Round 2
        win_4 = simulate_game(seed_order[3], win_5v8, ratings, location="Neutral")
        win_3 = simulate_game(seed_order[2], win_6v7, ratings, location="Neutral")

        for team in [win_3, win_4, seed_order[0], seed_order[1]]:
            tracker[team]["Make Double Elim"] += 1

        # Double elimination bracket
        de_teams = [seed_order[0], seed_order[1], win_4, win_3]
        bracket_result = double_elimination_bracket(de_teams, stats_and_metrics, num_simulations=1)
        winner = max(bracket_result.items(), key=lambda x: x[1])[0]
        tracker[winner]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(tracker, orient="index").reset_index().rename(columns={"index": "Team"})
    df["Round 2"] = df["Round 2"].astype(float) * 100 / num_simulations
    df["Make Double Elim"] = df["Make Double Elim"].astype(float) * 100 / num_simulations
    df["Win Tournament"] = df["Win Tournament"].astype(float) * 100 / num_simulations

    for team in seed_order[:4]:
        df.loc[df["Team"] == team, "Round 2"] = 100.0
    for team in seed_order[:2]:
        df.loc[df["Team"] == team, "Make Double Elim"] = 100.0

    return df

def simulate_best_of_three_tournament(seed_order, stats_and_metrics, num_simulations=1000):
    assert len(seed_order) == 4, "This format requires exactly 4 teams"

    rounds = ["Make Final", "Win Tournament"]
    stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):
        semi1 = simulate_best_of_three_series(seed_order[0], seed_order[3], ratings, "Home")
        semi2 = simulate_best_of_three_series(seed_order[1], seed_order[2], ratings, "Home")
        stats[semi1]["Make Final"] += 1
        stats[semi2]["Make Final"] += 1

        home_finalist, away_finalist = (
            (semi1, semi2) if seed_order.index(semi1) < seed_order.index(semi2)
            else (semi2, semi1)
        )
        champ = simulate_best_of_three_series(home_finalist, away_finalist, ratings, "Home")
        stats[champ]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "Team"})
    df.loc[:, rounds] = df[rounds].applymap(lambda x: round(100 * x / num_simulations, 1))
    return df

def simulate_two_playin_to_two_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    results = {team: {"Double Elim": 0, "Make Finals": 0, "Win Tournament": 0} for team in seed_order}
    ratings = {team: get_rating(team, stats_and_metrics) for team in seed_order}

    for _ in range(num_simulations):

        # Play-ins: 7 vs 10 and 8 vs 9
        win_7v10 = simulate_game(seed_order[6], seed_order[9], ratings)
        win_8v9 = simulate_game(seed_order[7], seed_order[8], ratings)

        # Higher seed is the one earlier in seed_order
        high_seed, low_seed = sorted([win_7v10, win_8v9], key=seed_order.index)

        # Assign brackets
        bracket1 = [seed_order[0], seed_order[3], seed_order[4], high_seed]
        bracket2 = [seed_order[1], seed_order[2], seed_order[5], low_seed]

        for t in bracket1 + bracket2:
            results[t]["Double Elim"] += 1

        # Simulate each double elim bracket
        bracket1 = double_elimination_bracket(bracket1, stats_and_metrics, 1)
        bracket2 = double_elimination_bracket(bracket2, stats_and_metrics, 1)
        finalist_1 = max(bracket1.items(), key=lambda x: x[1])[0]
        finalist_2 = max(bracket2.items(), key=lambda x: x[1])[0]

        results[finalist_1]["Make Finals"] += 1
        results[finalist_2]["Make Finals"] += 1

        champ = simulate_game(finalist_1, finalist_2, ratings)
        results[champ]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Team"})
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: round(100 * x / num_simulations, 1))
    return df

def get_conference_win_percentage(team, schedule_df, stats_and_metrics):
    # Map team to conference
    team_to_conf = stats_and_metrics.set_index('Team')['Conference'].to_dict()
    team_conf = team_to_conf.get(team)
    schedule_df['home_conf'] = schedule_df['home_team'].map(team_to_conf)
    schedule_df['away_conf'] = schedule_df['away_team'].map(team_to_conf)
    schedule_df["matchup"] = schedule_df["home_team"] + " vs " + schedule_df["away_team"]
    matchup = schedule_df["matchup"].values
    home_conf = schedule_df["home_conf"].values
    away_conf = schedule_df["away_conf"].values

    # Create 3-row rolling windows
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

    # Filter for conference games
    conf_games = df[
        (df['home_conf'] == team_conf) &
        (df['away_conf'] == team_conf) &
        (df['Result'].str.startswith(('W', 'L')))
    ]
    wins = conf_games['Result'].str.startswith('W')
    wins_count = int(wins.sum())
    games_count = int(len(conf_games))
    
    return wins_count / (games_count)

def simulate_conference_tournaments(conference):

    num_simulations = 1000
    stats_and_metrics, comparison_date = load_baseball_data()
    schedule_df = load_schedule_data()

    conference_teams = stats_and_metrics[stats_and_metrics['Conference'] == conference]['Team'].tolist()
    team_win_pcts = []
    for team in conference_teams:
        team_schedule = schedule_df[schedule_df['Team'] == team].reset_index(drop=True)
        win_pct = get_conference_win_percentage(team, team_schedule, stats_and_metrics)
        team_win_pcts.append((team, win_pct))
    team_win_pcts.sort(key=lambda x: x[1], reverse=True)

    if conference in ['SEC', 'ACC']:
        seed_order = [team for team, _ in team_win_pcts[:16]]
        if conference == 'ACC':
            seed_order = ['Georgia Tech', 'Florida St.', 'North Carolina', 'NC State',
                          'Clemson', 'Virginia', 'Duke', 'Wake Forest', 'Miami (FL)',
                          'Louisville', 'Notre Dame', 'Virginia Tech', 'Stanford',
                          'Boston College', 'Pittsburgh', 'California']
        elif conference == 'SEC':
            seed_order = ['Texas', 'Arkansas', 'LSU', 'Vanderbilt',
                          'Georgia', 'Auburn', 'Ole Miss', 'Tennessee',
                          'Alabama', 'Florida', 'Mississippi St.', 'Oklahoma',
                          'Kentucky', 'Texas A&M', 'South Carolina', 'Missouri']
        final_df = single_elimination_16_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.3, conference, 0.106, 0.098, 0.1)
    elif conference == "Big 12":
        seed_order = [team for team, _ in team_win_pcts[:12]]
        if conference == 'Big 12':
            seed_order = ['West Virginia', 'Kansas', 'TCU', 'Arizona',
                          'Arizona St.', 'Kansas St.', 'Oklahoma St.', 'Cincinnati',
                          'Texas Tech', 'Baylor', 'Houston', 'BYU']
        result_df = single_elimination_14_teams(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(result_df, 0.6, conference, 0.066, 0.06, 0.08)
    elif conference in ["Conference USA", "American Athletic", "Southland", "SWAC"]:
        seed_order = [team for team, _ in team_win_pcts[:8]]
        if conference == 'Southland':
            seed_order = ['Southeastern La.', 'UTRGV', 'Lamar University', 'Northwestern St.', 'McNeese', 'Houston Christian', 'A&M-Corpus Christi', 'New Orleans']
        elif conference == 'American Athletic':
            seed_order = ["UTSA", "Charlotte", "South Fla.", "Fla. Atlantic", "Tulane", "East Carolina", "Wichita St.", "Rice"]
        elif conference == 'Conference USA':
            seed_order = ['DBU', 'Western Ky.', 'Kennesaw St.', 'Jacksonville St.', 'Louisiana Tech', 'FIU', 'New Mexico St.', 'Liberty']
        elif conference == 'SWAC':
            seed_order = ['Bethune-Cookman', 'Florida A&M', 'Alabama St.', 'Ark.-Pine Bluff', 'Grambling', 'Jackson St.', 'Southern U.', 'Texas Southern']
        output = double_elimination_bracket([seed_order[0], seed_order[3], seed_order[4], seed_order[7]], stats_and_metrics, num_simulations)
        bracket_one = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
        output = double_elimination_bracket([seed_order[1], seed_order[2], seed_order[5], seed_order[6]], stats_and_metrics, num_simulations)
        bracket_two = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
        championship_results = simulate_overall_tournament(
            bracket_one.set_index("Team")["Win Regional"].to_dict(),
            bracket_two.set_index("Team")["Win Regional"].to_dict(),
            stats_and_metrics,
            num_simulations=num_simulations
        )
        championship_df = pd.DataFrame(list(championship_results.items()), columns=["Team", "Win Tournament"])
        regional_results = pd.concat([bracket_one.set_index("Team"), bracket_two.set_index("Team")], axis=0)
        final_df = pd.merge(regional_results.reset_index(), championship_df, on="Team", how="outer")
        final_df = final_df[['Team', 'Win Regional', 'Win Tournament']]
        final_df = final_df.rename(columns={'Win Regional': 'Win Group'})
        final_df[['Win Group', 'Win Tournament']] = final_df[['Win Group', 'Win Tournament']] * 100
        final_df = final_df[['Team', 'Win Group', 'Win Tournament']]
        seed_df = pd.DataFrame({'Team': seed_order})
        seed_df['Seed_Order'] = range(len(seed_order))
        final_df = pd.merge(seed_df, final_df, on='Team', how='left')
        final_df = final_df.sort_values('Seed_Order').drop(columns='Seed_Order').reset_index(drop=True)
        fig = plot_tournament_odds_table(final_df, 1, conference, 0.057, 0.052, 0.1)
    elif conference in ['America East', 'Mountain West', 'West Coast']:
        seed_order = [team for team, _ in team_win_pcts[:6]]
        if conference == 'America East':
            seed_order = ['Bryant', 'NJIT', 'Binghamton', 'Maine', 'UAlbany', 'UMBC']
        elif conference == 'Mountain West':
            seed_order = ['Nevada', 'Fresno St.', 'New Mexico', 'UNLV', 'San Diego St.', 'San Jose St.']
        elif conference == 'West Coast':
            seed_order = ['San Diego', 'Gonzaga', "Saint Mary's (CA)", 'LMU (CA)', 'Portland', 'San Francisco']
        final_df = two_playin_games_to_four_team_double_elimination(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.134, 0.122, 0.3)
    elif conference == 'ASUN':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        if conference == 'ASUN':
            seed_order = ['Austin Peay', 'Stetson', 'Lipscomb', 'Jacksonville', 'North Ala.', 'FGCU', 'Central Ark.', 'North Florida']
        final_df = simulate_and_run_8_team_double_elim(seed_order, stats_and_metrics)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.115, 0.105, 0.2)
    elif conference == "Atlantic 10":
        seed_order = [team for team, _ in team_win_pcts[:7]]
        if conference == 'Atlantic 10':
            seed_order = ['Rhode Island', 'George Mason', 'Saint Louis', 'Davidson', "Saint Joseph's", 'Fordham', 'Dayton']
        final_df = double_elimination_7_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.105, 0.093, 0.2)
    elif conference in ['Big East', 'Ivy League', 'Northeast', 'The Summit League']:
        seed_order = [team for team, _ in team_win_pcts[:4]]
        if conference == 'Ivy League':
            seed_order = ['Yale', 'Columbia', 'Penn', 'Harvard']
        elif conference == 'Big East':
            seed_order = ['Creighton', 'UConn', 'Xavier', "St. John's (NY)"]
        elif conference == 'Northeast':
            seed_order = ['LIU', 'Wagner', 'Central Conn. St.', 'FDU']
        elif conference == 'The Summit League':
            seed_order = ['Oral Roberts', 'North Dakota St.', 'Omaha', 'South Dakota St.']
        output = double_elimination_bracket([seed_order[0], seed_order[1], seed_order[2], seed_order[3]], stats_and_metrics, num_simulations)
        final_df = pd.DataFrame(list(output.items()), columns=["Team", "Win Tournament"])
        final_df["Win Tournament"] = final_df["Win Tournament"] * 100
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.143, 0.12, 0.4)
    elif conference in ['Big South', 'Coastal Athletic', 'Horizon League', 'Mid-American']:
        seed_order = [team for team, _ in team_win_pcts[:6]]
        if conference == 'Coastal Athletic':
            seed_order = ['Northeastern', 'UNCW', 'Campbell', 'Col. of Charleston', 'William & Mary', 'Elon']
        elif conference == 'Big South':
            seed_order = ['USC Upstate', 'High Point', 'Charleston So.', 'Radford', 'Winthrop', 'Presbyterian']
        elif conference == 'Horizon League':
            seed_order = ['Wright St.', 'Northern Ky.', 'Milwaukee', 'Youngstown St.', 'Oakland', 'Purdue Fort Wayne']
        elif conference == 'Mid-American':
            seed_order = ['Miami (OH)', 'Kent St.', 'Ball St.', 'Bowling Green', 'Toledo', 'Eastern Mich.']
        final_df = double_elimination_6_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.117, 0.103, 0.25)
    elif conference == 'Big Ten':
        seed_order = [team for team, _ in team_win_pcts[:12]]
        if conference == 'Big Ten':
            seed_order = ['Oregon', 'UCLA', 'Iowa', 'Southern California',
                          'Washington', 'Indiana', 'Michigan', 'Nebraska',
                          'Penn St.', 'Rutgers', 'Illinois', 'Michigan St.']
        final_df = simulate_pool_play_tournament(seed_order, stats_and_metrics, num_simulations=500)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.082, 0.075, 0.1)
    elif conference == 'Big West':
        seed_order = [team for team, _ in team_win_pcts[:5]]
        if conference == 'Big West':
            seed_order = ['UC Irvine', 'Cal Poly', 'Cal St. Fullerton', 'UC Santa Barbara', 'Hawaii']
        final_df = simulate_playin_double_elim(seed_order, stats_and_metrics)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.16, 0.14, 0.4)
    elif conference in ['MAAC', 'Southern', 'Western Athletic']:
        seed_order = [team for team, _ in team_win_pcts[:8]]
        if conference == 'MAAC':
            seed_order = ['Rider', 'Fairfield', 'Sacred Heart', 'Siena', 'Quinnipiac', "Mount St. Mary's", 'Marist', 'Niagara']
        elif conference == 'Southern':
            seed_order = ['ETSU', 'Samford', 'The Citadel', 'Mercer', 'Western Caro.', 'UNC Greensboro', 'Wofford', 'VMI']
        elif conference == 'Western Athletic':
            seed_order = ['Sacramento St.', 'Abilene Christian', 'Utah Valley', 'Grand Canyon', 'California Baptist', 'Tarleton St.', 'UT Arlington', 'Utah Tech']
        final_df = simulate_playins_to_6team_double_elim(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Missouri Valley':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        if conference == 'Missouri Valley':
            seed_order = ['Murray St.', 'Missouri St.', 'Southern Ill.', 'UIC', 'Illinois St.', 'Belmont', 'Bradley', 'Indiana St.']
        final_df = simulate_mvc_tournament(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Ohio Valley':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        if conference == 'Ohio Valley':
            seed_order = ['Eastern Ill.', 'SIUE', 'Tennessee Tech', 'Southeast Mo. St.', 'UT Martin', 'Little Rock', 'Western Ill.', 'Morehead St.']
        final_df = simulate_two_playin_rounds_to_double_elim(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Patriot League':
        seed_order = [team for team, _ in team_win_pcts[:4]]
        seed_order = ['Yale', 'Navy', 'Army West Point', 'Lehigh']
        final_df = simulate_best_of_three_tournament(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.17, 0.15, 0.5)
    elif conference == 'Sun Belt':
        seed_order = [team for team, _ in team_win_pcts[:10]]
        if conference == 'Sun Belt':
            seed_order = ['Coastal Carolina', 'Southern Miss.', 'Troy', 'Marshall', 'Louisiana', 'Old Dominion', 'Texas St.', 'Arkansas St.', 'Ga. Southern', 'App State']
        final_df = simulate_two_playin_to_two_double_elim(seed_order, stats_and_metrics, 500)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.104, 0.095, 0.15)
    return fig

@app.post("/api/cbase/simulate-conference-tournament")
def simulate_conference_tournament_endpoint(conference_data: dict):
    """Simulate a conference tournament"""
    try:
        conference = conference_data.get('conference')
        
        if not conference:
            raise HTTPException(status_code=400, detail="Conference is required")
        
        # Call your simulate_conference_tournaments function
        fig = simulate_conference_tournaments(conference)
        
        # Convert the matplotlib figure to an image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        print(f"Error simulating conference tournament: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
# ========================================
# SOFTBALL (CSOFT) ENDPOINTS
# ========================================

SOFTBALL_BASE_PATH = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Softball")
SOFTBALL_CURRENT_SEASON = 2025
SOFTBALL_HFA = 0.3  # Home field advantage in runs for softball

# print(f"Softball base path: {SOFTBALL_BASE_PATH}")
# print(f"Softball base path exists: {os.path.exists(SOFTBALL_BASE_PATH)}")

@app.get("/api/softball-logo/{team_name}")
def get_softball_logo(team_name: str):
    """Serve team logo"""
    # Replace spaces with underscores for the filename
    logo_filename = f"{team_name}.png"
    logo_path = os.path.join(SOFTBALL_BASE_PATH, "logos", logo_filename)

    # print(f"Looking for logo at: {logo_path}")
    # print(f"Logo folder: {logo_folder}")
    # print(f"File exists: {os.path.exists(logo_path)}")
    
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail=f"Logo not found at: {logo_path}")
    
    return FileResponse(logo_path, media_type="image/png")

class SoftballSpreadRequest(BaseModel):
    away_team: str
    home_team: str
    neutral: bool = False

class RegionalRequest(BaseModel):
    team_1: str
    team_2: str
    team_3: str
    team_4: str
    simulations: int = 1000

def load_softball_data():
    """Load the most recent softball data file"""
    try:
        folder_path = os.path.join(SOFTBALL_BASE_PATH, f"y{SOFTBALL_CURRENT_SEASON}", "Data")
        
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Softball data folder not found: {folder_path}")
        
        # Find all softball CSV files
        csv_files = [f for f in os.listdir(folder_path) 
                    if f.startswith("softball_") and f.endswith(".csv")]
        
        if not csv_files:
            raise HTTPException(status_code=404, detail="No softball data files found")
        
        # Extract dates and find most recent
        def extract_date(filename):
            try:
                return datetime.strptime(filename.replace("softball_", "").replace(".csv", ""), "%m_%d_%Y")
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
    
def load_softball_schedule_data():
    """Load the current season schedule"""
    try:
        schedule_path = os.path.join(SOFTBALL_BASE_PATH, f"y{SOFTBALL_CURRENT_SEASON}", 
                                     f"schedule_{SOFTBALL_CURRENT_SEASON}.csv")
        
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

@app.get("/api/softball/ratings")
def get_softball_ratings():
    """Get current softball team ratings"""
    try:
        modeling_stats, data_date = load_softball_data()
        
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
        print(f"Error in get_softball_ratings: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/softball/stats")
def get_softball_stats():
    """Get comprehensive softball team statistics"""
    try:
        modeling_stats, data_date = load_softball_data()
        
        # All the stats columns
        stats_columns = [
            'Team', 'Conference', 'Rating', 'NET', 'NET_Score', 'RPI', 'ELO', 'ELO_Rank', 'PRR', 'RQI', 
            'resume_quality', 'avg_expected_wins', 'SOS', 'SOR', 'Q1', 'Q2', 'Q3', 'Q4',
            'fWAR', 'oWAR_z', 'pWAR_z', 'WPOE', 'PYTHAG',
            'ERA', 'WHIP', 'KP7', 'RPG', 'BA', 'OBP', 'SLG', 'OPS', 'PCT'
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

@app.get("/api/softball/teams")
def get_softball_teams():
    """Get list of all softball teams"""
    try:
        modeling_stats, _ = load_softball_data()
        teams = sorted(modeling_stats['Team'].unique().tolist())
        return {"teams": teams}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_baseball_teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/softball/team-conferences")
def get_team_softball_conferences():
    """Get mapping of teams to their conferences"""
    try:
        modeling_stats, _ = load_softball_data()
        
        # Create a dictionary mapping team names to conferences
        team_conference_map = {}
        for _, row in modeling_stats[['Team', 'Conference']].drop_duplicates().iterrows():
            team_conference_map[row['Team']] = row['Conference']
        
        return {"team_conferences": team_conference_map}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in get_team_conferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/softball/calculate-spread")
def calculate_softball_matchup_spread(request: SoftballSpreadRequest):
    """Calculate spread for softball matchup"""
    try:
        modeling_stats, _ = load_softball_data()
        
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

@app.get("/api/softball/schedule/today")
def get_todays_softball_games():
    """Get today's softball games"""
    try:
        schedule_df = load_softball_schedule_data()
        
        cst = pytz.timezone('America/Chicago')
        today = datetime.now(cst).date()
        
        today_games = schedule_df[schedule_df['Date'].dt.date == today].copy()

        if len(today_games) == 0:
            # If we're past July and there are no games for today,
            # use the most recent date in the dataframe instead.
            if today.month > 7:
                last_date = schedule_df['Date'].max().date()
                today_games = schedule_df[schedule_df['Date'].dt.date == last_date].copy()
                today_games = today_games[['home_team', 'away_team', 'Location', 'PEAR', 'GQI', 'Date', 'home_win_prob', 'home_net', 'away_net']].drop_duplicates()
                return {"games": today_games.to_dict(orient="records"), "date": last_date.strftime("%B %d, %Y")}
            else:
                return {"games": [], "date": today.strftime("%B %d, %Y")}
        
        # Process results
        today_games = today_games[[
            'home_team', 'away_team', 'Location', 'PEAR', 'GQI', 'Date', 'home_win_prob', 'home_net', 'away_net'
        ]].copy().drop_duplicates()
        
        today_games = today_games.sort_values('GQI', ascending=False).reset_index(drop=True)
        
        return {
            "games": today_games.to_dict('records'),
            "date": today.strftime("%B %d, %Y"),
            "count": len(today_games)
        }
    
    except Exception as e:
        print(f"Error getting today's games: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/softball/team/{team_name}")
def get_softball_team_info(team_name: str):
    """Get detailed information for a specific softball team"""
    try:
        modeling_stats, data_date = load_softball_data()
        schedule_df = load_softball_schedule_data()
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
    
class SoftballTeamScheduleRequest(BaseModel):
    team_name: str
    
@app.post("/api/softball/profile-page")
def softball_team_schedule(request: SoftballTeamScheduleRequest):
    stats_and_metrics, comparison_date = load_softball_data()
    schedule_df = load_softball_schedule_data()
    team_name = request.team_name

    team_schedule = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True)
    metrics = stats_and_metrics[stats_and_metrics['Team'] == team_name].reset_index(drop=True)
    
    # Process schedule data
    schedule_data = []
    for idx, row in team_schedule.iterrows():
        game_data = {
            'date': row['Date'],
            'opponent': row['Opponent'] if pd.notna(row['Opponent']) else 'Non D-I',
            'location': row['Location'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
            'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
            'result': row['Result'] if pd.notna(row['Result']) else None,
            'home_win_prob': float(row['home_win_prob']) if pd.notna(row['home_win_prob']) else None,
            'resume_quality': float(row['resume_quality']) if pd.notna(row['resume_quality']) else None,
            'home_net': int(row['home_net']) if pd.notna(row['home_net']) else None,
            'away_net': int(row['away_net']) if pd.notna(row['away_net']) else None,
            'gqi': float(row['GQI']) if pd.notna(row['GQI']) else None,
            'pear': row['PEAR'] if pd.notna(row['PEAR']) else None,
        }
        
        # Determine opponent NET ranking
        if row['Team'] == row['home_team']:
            game_data['opponent_net'] = game_data['away_net']
            game_data['team_win_prob'] = game_data['home_win_prob']
        else:
            game_data['opponent_net'] = game_data['home_net']
            game_data['team_win_prob'] = 1 - game_data['home_win_prob'] if game_data['home_win_prob'] is not None else None
        
        schedule_data.append(game_data)
    
    # Get team metrics if available
    team_metrics = {}
    if len(metrics) > 0:
        team_metrics = {
            'conference': metrics.iloc[0]['Conference'] if 'Conference' in metrics.columns else None,
            'rating': float(metrics.iloc[0]['Rating']) if 'Rating' in metrics.columns and pd.notna(metrics.iloc[0]['Rating']) else None,
            'tsr': int(metrics.iloc[0]['PRR']) if 'PRR' in metrics.columns else None,
            'net': int(metrics.iloc[0]['NET']) if 'NET' in metrics.columns and pd.notna(metrics.iloc[0]['NET']) else None,
            'net_score': float(metrics.iloc[0]['NET_Score']) if 'NET_Score' in metrics.columns and pd.notna(metrics.iloc[0]['NET_Score']) else None,
            'rpi': int(metrics.iloc[0]['RPI']) if 'RPI' in metrics.columns and pd.notna(metrics.iloc[0]['RPI']) else None,
            'elo': float(metrics.iloc[0]['ELO']) if 'ELO' in metrics.columns and pd.notna(metrics.iloc[0]['ELO']) else None,
            'elo_rank': int(metrics.iloc[0]['ELO_Rank']) if 'ELO_Rank' in metrics.columns and pd.notna(metrics.iloc[0]['ELO_Rank']) else None,
            'resume_quality': float(metrics.iloc[0]['resume_quality']) if 'resume_quality' in metrics.columns and pd.notna(metrics.iloc[0]['resume_quality']) else None,
            'record': metrics.iloc[0]['Record'] if 'Record' in metrics.columns else None,
            'q1': metrics.iloc[0]['Q1'] if 'Q1' in metrics.columns else None,
            'q2': metrics.iloc[0]['Q2'] if 'Q2' in metrics.columns else None,
            'q3': metrics.iloc[0]['Q3'] if 'Q3' in metrics.columns else None,
            'q4': metrics.iloc[0]['Q4'] if 'Q4' in metrics.columns else None
        }
    
    return {
        'team_name': team_name,
        'schedule': schedule_data,
        'metrics': team_metrics,
        'data_date': comparison_date
    }

@app.get("/api/softball/conferences")
def get_softball_conferences():
    """Get list of all conferences"""
    try:
        modeling_stats, _ = load_softball_data()
        conferences = sorted(modeling_stats['Conference'].unique().tolist())
        # Remove "Independent" if present
        conferences = [c for c in conferences if c != "Independent"]
        return {"conferences": conferences}
    
    except Exception as e:
        print(f"Error getting conferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location):
    if location != "Neutral":
        home_pr += 0.3
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    spread = round(adjust_home_pr(elo_win_prob) + home_pr - away_pr, 2)
    return spread, elo_win_prob
    
def calculate_series_probabilities(win_prob):
    # Team A win probabilities
    P_A_0 = (1 - win_prob) ** 3
    P_A_1 = 3 * win_prob * (1 - win_prob) ** 2
    P_A_2 = 3 * win_prob ** 2 * (1 - win_prob)
    P_A_3 = win_prob ** 3

    # Team B win probabilities (q = 1 - p)
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

    return [P_A_at_least_1,P_A_at_least_2,P_A_3], [P_B_at_least_1,P_B_at_least_2,P_B_3]

@app.post("/api/softball/matchup-image")
def generate_softball_matchup_image(request: SoftballSpreadRequest):
    """Generate matchup comparison image"""
    home_logo = None
    away_logo = None
    
    try:

        stats_and_metrics, _ = load_softball_data()
        
        away_team = request.away_team
        home_team = request.home_team
        neutrality = "Neutral" if request.neutral else "Home"
        
            
        # Load team logos
        logo_folder = os.path.join(SOFTBALL_BASE_PATH, "logos")
        home_logo = None
        away_logo = None

        def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
            if location != "Neutral":
                home_pr += 0.3
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

        def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                            facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
            # Draw rectangle behind text
            ax.add_patch(Rectangle(
                (x - width/2, y - height/2), width, height,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=1
            ))

            # Draw text centered on top
            ax.text(x, y, text,
                    ha="center", va="center", zorder=2, **kwargs)
            
        def get_text_color(bg_color: str) -> str:
            """Determine if text should be black or white based on background color luminance"""
            import re
            
            # Handle hex colors
            if bg_color.startswith('#'):
                # Convert hex to RGB
                bg_color = bg_color.lstrip('#')
                r = int(bg_color[0:2], 16)
                g = int(bg_color[2:4], 16)
                b = int(bg_color[4:6], 16)
            else:
                # Handle rgb() format
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if not match:
                    return 'white'
                
                r = int(match.group(1))
                g = int(match.group(2))
                b = int(match.group(3))
            
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return 'black' if luminance > 0.5 else 'white'

        def rank_text_color(rank, vmin=1, vmax=300):
            """Get appropriate text color (black or white) based on rank background color"""
            if rank == "":
                return 'black'
            
            bg_color = rank_to_color(rank, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def percent_text_color(win_pct, vmin=0.0, vmax=1.0):
            """Get appropriate text color (black or white) based on win percentage background color"""
            if win_pct == "":
                return 'black'
            
            bg_color = rank_to_color(win_pct, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def plot_logo(ax, img, xy, zoom=0.2):
            """Helper to plot a logo at given xy coords."""
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy, frameon=False)
            ax.add_artist(ab)

        def rank_to_color(rank, vmin=1, vmax=300):
            """
            Map a rank (1–300) to a hex color.
            Dark blue = best (1), grey = middle, dark red = worst (300).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from blue → grey → red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rank_cmap", ["#00008B", "#00FFFF", "#D3D3D3", "#FFA500", "#8B0000"]  # dark blue, cyan, light gray, orange, dark red
            )
            
            # Normalize rank to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(rank))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def percent_to_color(win_pct, vmin=0.0, vmax=1.0):
            """
            Map a win percentage (0.0–1.0) to a hex color.
            Dark blue = best (1.0), grey = middle (0.5), dark red = worst (0.0).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from red → grey → blue
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "percent_cmap", ["#8B0000", "#FFA500", "#D3D3D3", "#00FFFF", "#00008B"]  # dark red, orange, light gray, cyan, dark blue
            )
            
            # Normalize percentage to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(win_pct))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def get_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (value, rank) for a given team and column.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and stat columns.
                team (str): Team name to look up.
                column (str): Column name to extract.
                higher_is_better (bool): If True, high values rank better (1 = highest).
                                        If False, low values rank better (1 = lowest).
            """
            ascending = not higher_is_better
            ranks = df[column].rank(ascending=ascending, method="first").astype(int)

            value = df.loc[df['Team'] == team, column].values[0]
            rank = ranks.loc[df['Team'] == team].values[0]

            return value, rank
        
        def get_record_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (record_string, win_percentage, rank) for a given team and column containing W-L records.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and record columns.
                team (str): Team name to look up.
                column (str): Column name containing records in "W-L" format.
                higher_is_better (bool): If True, high win% ranks better (1 = highest).
                                        If False, low win% ranks better (1 = lowest).
            
            Returns:
                tuple: (record_string, win_percentage as float, rank as int)
            """
            def calculate_win_pct(record):
                """Convert 'W-L' string to win percentage."""
                if pd.isna(record) or record == '':
                    return 0.0
                parts = str(record).split('-')
                wins = int(parts[0])
                losses = int(parts[1])
                total = wins + losses
                return wins / total if total > 0 else 0.0
            
            # Calculate win percentages for all teams
            win_pcts = df[column].apply(calculate_win_pct)
            
            # Calculate ranks
            ascending = not higher_is_better
            ranks = win_pcts.rank(ascending=ascending, method="first").astype(int)
            
            # Get values for specified team
            team_idx = df['Team'] == team
            record_string = df.loc[team_idx, column].values[0]
            win_pct = win_pcts.loc[team_idx].values[0]
            rank = ranks.loc[team_idx].values[0]
            
            return record_string, win_pct
        
        def add_row(x_vals, y, away_val, away_rank, away_name, home_name, home_rank, home_val, away_digits, home_digits):
            # Helper to choose text color based on rank

            # Away value
            if away_val != "":
                ax.text(x_vals[0], y, f"{away_val:.{away_digits}f}", ha='center', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            # Away rank box
            if away_rank != "":
                fixed_width_text(
                    ax, x_vals[1], y+0.007, f"{away_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(away_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(away_rank)
                )

            # Metric name
            if away_name != "":
                ax.text(x_vals[2], y, away_name, ha='left', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            if home_name != "":
                ax.text(x_vals[3], y, home_name, ha='right', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))

            # Home rank box
            if home_rank != "":
                fixed_width_text(
                    ax, x_vals[4], y+0.007, f"{home_rank}", width=0.06, height=0.04,
                    facecolor=rank_to_color(home_rank), alpha=alpha_val,
                    fontsize=16, fontweight='bold', color=rank_text_color(home_rank)
                )

            # Home value
            if home_val != "":
                ax.text(x_vals[5], y, f"{home_val:.{home_digits}f}", ha='center', fontsize=16, fontweight='bold',
                        bbox=dict(facecolor='green', alpha=0))
        
        if os.path.exists(logo_folder):
            home_logo_path = os.path.join(logo_folder, f"{home_team}.png")
            away_logo_path = os.path.join(logo_folder, f"{away_team}.png")
            
            if os.path.exists(home_logo_path):
                home_logo = Image.open(home_logo_path).convert("RGBA")
            if os.path.exists(away_logo_path):
                away_logo = Image.open(away_logo_path).convert("RGBA")

        fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
        fig.patch.set_facecolor('#CECEB2')
        ax.set_facecolor('#CECEB2')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ----------------
        # logos, score, win prob, spread, O/U
        # ----------------
        plot_logo(ax, away_logo, (0.15, 0.75), zoom=0.3)
        plot_logo(ax, home_logo, (0.85, 0.75), zoom=0.3)

        if neutrality == "Neutral":
            ax.text(0.5, 0.96, f"{away_team} (N) {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))
        else:
            ax.text(0.5, 0.96, f"{away_team} at {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))

        alpha_val = 0.9

        away_pr, away_rank = get_value_and_rank(stats_and_metrics, away_team, 'Rating')
        home_pr, home_rank = get_value_and_rank(stats_and_metrics, home_team, 'Rating')
        away_elo, away_elo_rank = get_value_and_rank(stats_and_metrics, away_team, 'ELO')
        home_elo, home_elo_rank = get_value_and_rank(stats_and_metrics, home_team, 'ELO')
        home_net_score, home_net_rank = get_value_and_rank(stats_and_metrics, home_team, 'NET_Score')
        away_net_score, away_net_rank = get_value_and_rank(stats_and_metrics, away_team, 'NET_Score')
        home_rq, home_rq_rank = get_value_and_rank(stats_and_metrics, home_team, 'resume_quality')
        away_rq, away_rq_rank = get_value_and_rank(stats_and_metrics, away_team, 'resume_quality')
        home_sos, home_sos_rank = get_value_and_rank(stats_and_metrics, home_team, 'avg_expected_wins', False)
        away_sos, away_sos_rank = get_value_and_rank(stats_and_metrics, away_team, 'avg_expected_wins', False)
        home_pythag, home_pythag_rank = get_value_and_rank(stats_and_metrics, home_team, 'PYTHAG')
        away_pythag, away_pythag_rank = get_value_and_rank(stats_and_metrics, away_team, 'PYTHAG')
        home_war, home_war_rank = get_value_and_rank(stats_and_metrics, home_team, 'fWAR')
        away_war, away_war_rank = get_value_and_rank(stats_and_metrics, away_team, 'fWAR')
        home_wpoe, home_wpoe_rank = get_value_and_rank(stats_and_metrics, home_team, 'wpoe_pct')
        away_wpoe, away_wpoe_rank = get_value_and_rank(stats_and_metrics, away_team, 'wpoe_pct')
        home_q1, home_q1_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q1')
        home_q2, home_q2_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q2')
        home_q3, home_q3_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q3')
        home_q4, home_q4_rank = get_record_value_and_rank(stats_and_metrics, home_team, 'Q4')
        away_q1, away_q1_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q1')
        away_q2, away_q2_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q2')
        away_q3, away_q3_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q3')
        away_q4, away_q4_rank = get_record_value_and_rank(stats_and_metrics, away_team, 'Q4')

        home_rpg, home_rpg_rank = get_value_and_rank(stats_and_metrics, home_team, 'RPG')
        away_rpg, away_rpg_rank = get_value_and_rank(stats_and_metrics, away_team, 'RPG')
        home_ba, home_ba_rank = get_value_and_rank(stats_and_metrics, home_team, 'BA')
        away_ba, away_ba_rank = get_value_and_rank(stats_and_metrics, away_team, 'BA')
        home_obp, home_obp_rank = get_value_and_rank(stats_and_metrics, home_team, 'OBP')
        away_obp, away_obp_rank = get_value_and_rank(stats_and_metrics, away_team, 'OBP')
        home_slg, home_slg_rank = get_value_and_rank(stats_and_metrics, home_team, 'SLG')
        away_slg, away_slg_rank = get_value_and_rank(stats_and_metrics, away_team, 'SLG')
        home_ops, home_ops_rank = get_value_and_rank(stats_and_metrics, home_team, 'OPS')
        away_ops, away_ops_rank = get_value_and_rank(stats_and_metrics, away_team, 'OPS')
        home_iso, home_iso_rank = get_value_and_rank(stats_and_metrics, home_team, 'ISO')
        away_iso, away_iso_rank = get_value_and_rank(stats_and_metrics, away_team, 'ISO')
        home_era, home_era_rank = get_value_and_rank(stats_and_metrics, home_team, 'ERA', False)
        away_era, away_era_rank = get_value_and_rank(stats_and_metrics, away_team, 'ERA', False)
        home_whip, home_whip_rank = get_value_and_rank(stats_and_metrics, home_team, 'WHIP', False)
        away_whip, away_whip_rank = get_value_and_rank(stats_and_metrics, away_team, 'WHIP', False)
        home_k9, home_k9_rank = get_value_and_rank(stats_and_metrics, home_team, 'KP7')
        away_k9, away_k9_rank = get_value_and_rank(stats_and_metrics, away_team, 'KP7')
        home_lob, home_lob_rank = get_value_and_rank(stats_and_metrics, home_team, 'LOB%')
        away_lob, away_lob_rank = get_value_and_rank(stats_and_metrics, away_team, 'LOB%')
        home_kbb, home_kbb_rank = get_value_and_rank(stats_and_metrics, home_team, 'K/BB')
        away_kbb, away_kbb_rank = get_value_and_rank(stats_and_metrics, away_team, 'K/BB')
        home_pct, home_pct_rank = get_value_and_rank(stats_and_metrics, home_team, 'PCT')
        away_pct, away_pct_rank = get_value_and_rank(stats_and_metrics, away_team, 'PCT')

        home_win_prob = PEAR_Win_Prob(home_pr, away_pr, neutrality)
        home_series, away_series = calculate_series_probabilities(home_win_prob/100)
        spread, elo_win_prob = calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, neutrality)
        if spread < 0:
            formatted_spread = f"{away_team} -{abs(spread):.2f}"
        else:
            formatted_spread = f"{home_team} -{spread:.2f}"

        max_net = 299
        w_tq = 0.70   # NET AVG
        w_wp = 0.20   # Win Probability
        w_ned = 0.10  # NET Differential
        avg_net = (home_net_rank + away_net_rank) / 2
        tq = (max_net - avg_net) / (max_net - 1)
        wp = 1 - 2 * np.abs((home_win_prob/100) - 0.5)
        ned = 1 - (np.abs(away_net_rank - home_net_rank) / (max_net - 1))
        gqi = round(10*(w_tq * tq + w_wp * wp + w_ned * ned), 1)

        bubble_team_rating = stats_and_metrics['Rating'].quantile(0.90)
        home_quality = PEAR_Win_Prob(bubble_team_rating, away_pr, neutrality) / 100
        home_win_quality, home_loss_quality = (1 - home_quality), -home_quality
        away_quality = 1-PEAR_Win_Prob(home_pr, bubble_team_rating, neutrality) / 100
        away_win_quality, away_loss_quality = (1 - away_quality), -away_quality

        ax.text(0.5, 0.57, f"{formatted_spread}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))
        ax.text(0.5, 0.625, f"GQI: {gqi}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))
        ax.text(0.6, 0.89, f"ONE GAME (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.84, f"{round(home_win_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.78, f"SERIES (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.75, f"≥1: {round(home_series[0]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.72, f"≥2: {round(home_series[1]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.6, 0.69, f"SWEEP: {round(home_series[2]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        
        ax.text(0.4, 0.89, f"ONE GAME (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.84, f"{round(100-home_win_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.78, f"SERIES (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.75, f"≥1: {round(away_series[0]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.72, f"≥2: {round(away_series[1]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
        ax.text(0.4, 0.69, f"SWEEP: {round(away_series[2]*100,1)}%", ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))

        away_record = stats_and_metrics.loc[stats_and_metrics['Team'] == away_team, 'Record'].values[0]
        ax.text(0.01, 0.53, f"{away_record}", ha='left', fontsize=16, fontweight='bold')

        home_record = stats_and_metrics.loc[stats_and_metrics['Team'] == home_team, 'Record'].values[0]
        ax.text(0.99, 0.53, f"{home_record}", ha='right', fontsize=16, fontweight='bold')

        # X positions for the 5 columns
        x_cols = [0.31, 0.378, 0.42, 0.58, 0.622, 0.69]

        ax.text(0.5, 0.528, f"{away_team} OFF vs {home_team} PCH",
                ha='center', fontsize=16, fontweight='bold',
                bbox=dict(facecolor='green', alpha=0))
        ax.hlines(y=0.518, xmin=0.29, xmax=0.71, colors='black', linewidth=1)

        # Away OFF vs Home DEF
        add_row(x_cols, 0.49, away_rpg, away_rpg_rank, "RPG", "ERA", home_era_rank, home_era, 2, 2)
        add_row(x_cols, 0.45, away_ba, away_ba_rank, "BA", "WHIP", home_whip_rank, home_whip, 3, 2)
        add_row(x_cols, 0.41, away_obp, away_obp_rank, "OBP", "K/7", home_k9_rank, home_k9, 3, 1)
        add_row(x_cols, 0.37, away_slg, away_slg_rank, "SLG", "LOB%", home_lob_rank, home_lob, 3, 2)
        add_row(x_cols, 0.33, away_ops, away_ops_rank, "OPS", "K/BB", home_kbb_rank, home_kbb, 3, 2)
        add_row(x_cols, 0.29, away_iso, away_iso_rank, "ISO", "PCT", home_pct_rank, home_pct, 3, 3)

        # Header for Away DEF vs Home OFF
        ax.text(0.5, 0.248, f"{away_team} PCH vs {home_team} OFF",
                ha='center', fontsize=16, fontweight='bold', bbox=dict(facecolor='green', alpha=0))
        ax.hlines(y=0.238, xmin=0.29, xmax=0.71, colors='black', linewidth=1)
        add_row(x_cols, 0.21, away_era, away_era_rank, "ERA", "RPG", home_rpg_rank, home_rpg, 2, 2)
        add_row(x_cols, 0.17, away_whip, away_whip_rank, "WHIP", "BA", home_ba_rank, home_ba, 2, 3)
        add_row(x_cols, 0.13, away_k9, away_k9_rank, "K/7", "OBP", home_obp_rank, home_obp, 1, 3)
        add_row(x_cols, 0.09, away_lob, away_lob_rank, "LOB%", "SLG", home_slg_rank, home_slg, 2, 3)
        add_row(x_cols, 0.05, away_kbb, away_kbb_rank, "K/BB", "OPS", home_ops_rank, home_ops, 2, 3)
        add_row(x_cols, 0.01, away_pct, away_pct_rank, "PCT", "ISO", home_iso_rank, home_iso, 2, 3)
        ax.text(0.5, -0.03, "@PEARatings", ha='center', fontsize=16, fontweight='bold',bbox=dict(facecolor='green', alpha=0))

        ### AWAY SIDE

        ax.text(0.01, 0.49, f"NET", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.478, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
        ax.text(0.19, 0.49, f"{away_net_score:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.49+0.007, f"{away_net_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_net_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_net_rank))
        
        ax.text(0.04, 0.45, f"RATING", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.45, f"{away_pr}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.45+0.007, f"{away_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_rank))

        ax.text(0.04, 0.41, f"RQI", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.41, f"{away_rq:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.41+0.007, f"{away_rq_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_rq_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_rq_rank))
        
        ax.text(0.04, 0.37, f"SOS", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.37, f"{away_sos:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.37+0.007, f"{away_sos_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_sos_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_sos_rank))

        ax.text(0.04, 0.33, f"PYTHAG", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.33, f"{away_pythag:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.33+0.007, f"{away_pythag_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_pythag_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_pythag_rank))

        ax.text(0.04, 0.29, f"WAR", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.29, f"{away_war:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.29+0.007, f"{away_war_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_war_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_war_rank))

        ax.text(0.04, 0.25, f"WPOE", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.25, f"{away_wpoe:.3f}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.25+0.007, f"{away_wpoe_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(away_wpoe_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(away_wpoe_rank))

        ax.text(0.01, 0.21, f"NET QUADS", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.198, xmin=0.01, xmax=0.26, colors='black', linewidth=1)

        ax.text(0.04, 0.17, f"Q1", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.17, f"{away_q1}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.17+0.007, f"{away_q1_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q1_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q1_rank))

        ax.text(0.04, 0.13, f"Q2", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.13, f"{away_q2}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.13+0.007, f"{away_q2_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q2_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q2_rank))
        
        ax.text(0.04, 0.09, f"Q3", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.09, f"{away_q3}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.09+0.007, f"{away_q3_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q3_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q3_rank))

        ax.text(0.04, 0.05, f"Q4", ha='left', fontsize=16, fontweight='bold')
        ax.text(0.19, 0.05, f"{away_q4}", ha='right', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.23, 0.05+0.007, f"{away_q4_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(away_q4_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(away_q4_rank))
        
        ax.text(0.01, 0.01, f"WIN QUALITY", ha='left', fontsize=16, fontweight='bold')
        ax.hlines(y=0.0, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
        ax.text(0.04, -0.03, f"{away_win_quality:.2f}", ha='left', fontsize=16, fontweight='bold', color='green')
        ax.text(0.19, -0.03, f"{away_loss_quality:.2f}", ha='right', fontsize=16, fontweight='bold', color='red')

        #### HOME SIDE

        ax.text(0.99, 0.49, f"NET", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.478, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.81, 0.49, f"{home_net_score:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.49+0.007, f"{home_net_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_net_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_net_rank))

        ax.text(0.96, 0.45, f"RATING", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.45, f"{home_pr}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.45+0.007, f"{home_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_rank))

        ax.text(0.96, 0.41, f"RQI", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.41, f"{home_rq:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.41+0.007, f"{home_rq_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_rq_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_rq_rank))

        ax.text(0.96, 0.37, f"SOS", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.37, f"{home_sos:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.37+0.007, f"{home_sos_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_sos_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_sos_rank))

        ax.text(0.96, 0.33, f"PYTHAG", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.33, f"{home_pythag:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.33+0.007, f"{home_pythag_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_pythag_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_pythag_rank))

        ax.text(0.96, 0.29, f"WAR", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.29, f"{home_war:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.29+0.007, f"{home_war_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_war_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_war_rank))
        
        ax.text(0.96, 0.25, f"WPOE", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.25, f"{home_wpoe:.3f}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.25+0.007, f"{home_wpoe_rank}", width=0.06, height=0.04,
                                facecolor=rank_to_color(home_wpoe_rank), alpha=alpha_val,
                                fontsize=16, fontweight='bold', color=rank_text_color(home_wpoe_rank))
        
        ax.text(0.99, 0.21, f"NET QUADS", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.198, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.96, 0.17, f"Q1", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.17, f"{home_q1}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.17+0.007, f"{home_q1_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q1_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q1_rank))

        ax.text(0.96, 0.13, f"Q2", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.13, f"{home_q2}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.13+0.007, f"{home_q2_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q2_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q2_rank))

        ax.text(0.96, 0.09, f"Q3", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.09, f"{home_q3}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.09+0.007, f"{home_q3_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q3_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q3_rank))

        ax.text(0.96, 0.05, f"Q4", ha='right', fontsize=16, fontweight='bold')
        ax.text(0.81, 0.05, f"{home_q4}", ha='left', fontsize=16, fontweight='bold')
        fixed_width_text(ax, 0.77, 0.05+0.007, f"{home_q4_rank:.3f}", width=0.06, height=0.04,
                                    facecolor=percent_to_color(home_q4_rank), alpha=alpha_val,
                                    fontsize=16, fontweight='bold', color=percent_text_color(home_q4_rank))

        ax.text(0.99, 0.01, f"WIN QUALITY", ha='right', fontsize=16, fontweight='bold')
        ax.hlines(y=0.0, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
        ax.text(0.81, -0.03, f"{home_win_quality:.2f}", ha='left', fontsize=16, fontweight='bold', color='green')
        ax.text(0.96, -0.03, f"{home_loss_quality:.2f}", ha='right', fontsize=16, fontweight='bold', color='red')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        
        # Get image data
        image_data = buf.getvalue()
        
        # Aggressive cleanup
        plt.close(fig)
        fig.clf()
        del fig
        del buf
        gc.collect()
        
        return Response(content=image_data, media_type="image/png")
    
    except Exception as e:
        print(f"Error generating matchup image: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        # Always close logo images
        if home_logo is not None:
            home_logo.close()
            del home_logo
        if away_logo is not None:
            away_logo.close()
            del away_logo
        gc.collect()

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

@app.get("/api/softball/tournament-outlook")
def get_softball_tournament_outlook():
    """Generate projected NCAA Tournament bracket"""
    try:
        modeling_stats, _ = load_softball_data()
        
        # Hardcoded lists from the streamlit app
        # aq_list = ["Binghamton", "East Carolina", "Stetson", "Rhode Island", "North Carolina", "Arizona",
        #         "Creighton", "USC Upstate", "Nebraska", "Cal Poly", "Northeastern", "Western Ky.", "Wright St.",
        #         "Columbia", "Fairfield", "Miami (OH)", "Murray St.", "Fresno St.",
        #         "Central Conn. St.", "Little Rock", "Holy Cross", "Vanderbilt", "Houston Christian",
        #         "ETSU", "Bethune-Cookman", "North Dakota St.", "Coastal Carolina", "Saint Mary's (CA)", "Utah Valley", "Oregon St."]
        aq_list = list(modeling_stats.loc[modeling_stats.groupby("Conference")["NET"].idxmin()]['Team'])
        
        # host_seeds_list = ["Georgia", "Auburn", "Texas", "LSU", "North Carolina", "Clemson", "Coastal Carolina", "Oregon St.",
        #             "Oregon", "Arkansas", "Southern Miss.", "Tennessee", "UCLA", "Vanderbilt", "Ole Miss", "Florida St."]
        host_seeds_list = modeling_stats.nsmallest(16, "NET")['Team'].tolist()
        
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

class SoftballRegionalSimulationRequest(BaseModel):
    seed_1: str
    seed_2: str
    seed_3: str
    seed_4: str

@app.post("/api/softball/simulate-regional")
def softball_simulate_regional(request: SoftballRegionalSimulationRequest):
    """Simulate a regional tournament and return visualization"""
    try:
        modeling_stats, _ = load_softball_data()
        
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

class SoftballTeamProfileRequest(BaseModel):
    team_name: str

def load_softball_schedule_data():
    """Load schedule data for softball"""
    try:
        softball_path = os.path.join(os.path.dirname(BACKEND_DIR), "PEAR", "PEAR Softball")
        schedule_path = os.path.join(softball_path, f"y{current_season}", f"schedule_{current_season}.csv")
        schedule_df = pd.read_csv(schedule_path)
        schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
        return schedule_df
    except Exception as e:
        print(f"Error loading schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading schedule: {str(e)}")

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

@app.post("/api/softball/team-profile")
def softball_team_profile(request: SoftballTeamProfileRequest):
    logo = None
    try:
        stats_and_metrics, comparison_date = load_softball_data()
        schedule_df = load_softball_schedule_data()
        team_name = request.team_name

        logo_folder = os.path.join(SOFTBALL_BASE_PATH, "logos")

        def plot_box(x, y, width, height, color='black', fill=False, linewidth=2, ax=None):
            if ax is None:
                fig, ax = plt.subplots()

            rect = Rectangle((x, y), width, height,
                                    linewidth=linewidth,
                                    edgecolor="black",
                                    facecolor=color if fill else 'none')
            ax.add_patch(rect)

            return ax

        def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
            if location != "Neutral":
                home_pr += 0.3
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

        def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                            facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
            # Draw rectangle behind text
            ax.add_patch(Rectangle(
                (x - width/2, y - height/2), width, height,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=1
            ))

            # Draw text centered on top
            ax.text(x, y, text,
                    ha="center", va="center", zorder=2, **kwargs)
            
        def get_text_color(bg_color: str) -> str:
            """Determine if text should be black or white based on background color luminance"""
            import re
            
            # Handle hex colors
            if bg_color.startswith('#'):
                # Convert hex to RGB
                bg_color = bg_color.lstrip('#')
                r = int(bg_color[0:2], 16)
                g = int(bg_color[2:4], 16)
                b = int(bg_color[4:6], 16)
            else:
                # Handle rgb() format
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if not match:
                    return 'white'
                
                r = int(match.group(1))
                g = int(match.group(2))
                b = int(match.group(3))
            
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return 'black' if luminance > 0.5 else 'white'

        def rank_text_color(rank, vmin=1, vmax=300):
            """Get appropriate text color (black or white) based on rank background color"""
            if rank == "":
                return 'black'
            
            bg_color = rank_to_color(rank, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def percent_text_color(win_pct, vmin=0.0, vmax=1.0):
            """Get appropriate text color (black or white) based on win percentage background color"""
            if win_pct == "":
                return 'black'
            
            bg_color = rank_to_color(win_pct, vmin=vmin, vmax=vmax)
            return get_text_color(bg_color)

        def plot_logo(ax, img, xy, zoom=0.2, zorder=3):
            """Helper to plot a logo at given xy coords."""
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy, frameon=False, zorder=zorder,
                            xycoords='axes fraction', box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

        def rank_to_color(rank, vmin=1, vmax=300):
            """
            Map a rank (1–300) to a hex color.
            Dark blue = best (1), grey = middle, dark red = worst (300).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from blue → grey → red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rank_cmap", ["#00008B", "#00FFFF", "#D3D3D3", "#FFA500", "#8B0000"]  # dark blue, cyan, light gray, orange, dark red
            )
            
            # Normalize rank to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(rank))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def percent_to_color(win_pct, vmin=0.0, vmax=1.0):
            """
            Map a win percentage (0.0–1.0) to a hex color.
            Dark blue = best (1.0), grey = middle (0.5), dark red = worst (0.0).
            Color scale: Dark Red (#8B0000) → Orange (#FFA500) → Light Gray (#D3D3D3) → Cyan (#00FFFF) → Dark Blue (#00008B)
            """
            # Define colormap from red → grey → blue
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "percent_cmap", ["#8B0000", "#FFA500", "#D3D3D3", "#00FFFF", "#00008B"]  # dark red, orange, light gray, cyan, dark blue
            )
            
            # Normalize percentage to [0,1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(win_pct))
            
            # Convert RGBA to hex
            return mcolors.to_hex(rgba)

        def get_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (value, rank) for a given team and column.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and stat columns.
                team (str): Team name to look up.
                column (str): Column name to extract.
                higher_is_better (bool): If True, high values rank better (1 = highest).
                                        If False, low values rank better (1 = lowest).
            """
            ascending = not higher_is_better
            ranks = df[column].rank(ascending=ascending, method="first").astype(int)

            value = df.loc[df['Team'] == team, column].values[0]
            rank = ranks.loc[df['Team'] == team].values[0]

            return value, rank
        
        def get_record_value_and_rank(df, team, column, higher_is_better=True):
            """
            Return (record_string, win_percentage, rank) for a given team and column containing W-L records.
            
            Args:
                df (pd.DataFrame): Data source with 'team' and record columns.
                team (str): Team name to look up.
                column (str): Column name containing records in "W-L" format.
                higher_is_better (bool): If True, high win% ranks better (1 = highest).
                                        If False, low win% ranks better (1 = lowest).
            
            Returns:
                tuple: (record_string, win_percentage as float, rank as int)
            """
            def calculate_win_pct(record):
                """Convert 'W-L' string to win percentage."""
                if pd.isna(record) or record == '':
                    return 0.0
                parts = str(record).split('-')
                wins = int(parts[0])
                losses = int(parts[1])
                total = wins + losses
                return wins / total if total > 0 else 0.0
            
            # Calculate win percentages for all teams
            win_pcts = df[column].apply(calculate_win_pct)
            
            # Calculate ranks
            ascending = not higher_is_better
            ranks = win_pcts.rank(ascending=ascending, method="first").astype(int)
            
            # Get values for specified team
            team_idx = df['Team'] == team
            record_string = df.loc[team_idx, column].values[0]
            win_pct = win_pcts.loc[team_idx].values[0]
            rank = ranks.loc[team_idx].values[0]
            
            return record_string, win_pct
        
        def get_game_color(result):
            if "W" in result:
                return "palegreen"
            elif "L" in result:
                return "lightcoral"
            else:
                return "whitesmoke"

        def get_quadrant(opponent_net, location):
            thresholds = {
                "Home": [25, 50, 100, 307],
                "Neutral": [40, 80, 160, 307],
                "Away": [60, 120, 240, 307]
            }
            
            # Get the thresholds for the given location
            location_thresholds = thresholds.get(location, thresholds["Neutral"])
            
            # Determine quadrant based on opponent NET ranking
            if opponent_net <= location_thresholds[0]:
                return "Q1"
            elif opponent_net <= location_thresholds[1]:
                return "Q2"
            elif opponent_net <= location_thresholds[2]:
                return "Q3"
            else:
                return "Q4"

        def get_location_records(team, team_schedule):
            """Get home, away, and neutral records for a team"""
            
            team_schedule['is_win'] = team_schedule['Result'].str.startswith('W')
            
            records = {}
            for loc in ['Home', 'Away', 'Neutral']:
                group = team_schedule[team_schedule['Location'] == loc]
                wins = group['is_win'].sum()
                losses = len(group) - wins
                records[loc] = f"{int(wins)}-{int(losses)}"
            
            return records

        def display_stat_row(stats_ax, x1, y1, x2, y2, x3, y3, label, value, value_format, 
                            rank, ha, fontsize=16, alpha_val=0.9):
            """
            Display a stat row with label, value, and rank badge.
            
            Parameters:
            -----------
            stats_ax : matplotlib axis
                The axis to draw on
            x1, y1 : float
                Position for the label text
            x2, y2 : float
                Position for the value text
            x3, y3 : float
                Position for the rank badge (center)
            label : str
                The stat label (e.g., "RPG", "ERA")
            value : float
                The stat value
            value_format : str
                Format string for the value (e.g., ".2f", ".3f", ".1f")
            rank : str
                The rank badge text
            ha : str
                Horizontal alignment for label and value ("left" or "right")
            fontsize : int, optional
                Font size for all text (default=16)
            alpha_val : float, optional
                Alpha transparency for rank badge (default=0.3)
            """
            # Determine alignment based on ha parameter
            if ha == "left":
                label_ha = "left"
                value_ha = "right"
            else:  # ha == "right"
                label_ha = "right"
                value_ha = "left"
            
            # Display label
            stats_ax.text(x1, y1, label, fontsize=fontsize, fontweight='bold', ha=label_ha)
            
            # Display value with proper formatting
            stats_ax.text(x2, y2, f"{value:{value_format}}", fontsize=fontsize, fontweight='bold', ha=value_ha)
            
            # Display rank badge
            fixed_width_text(stats_ax, x3, y3, f"{rank}", width=0.15, height=0.05,
                            facecolor=rank_to_color(rank), alpha=alpha_val,
                            fontsize=fontsize, fontweight='bold', color=rank_text_color(rank))

        if os.path.exists(logo_folder):
            # Try to find logo for the team (keep spaces, don't replace with underscores)
            logo_path = os.path.join(logo_folder, f"{team_name}.png")

            if os.path.exists(logo_path):
                logo = Image.open(logo_path).convert("RGBA")

        team_schedule = schedule_df[schedule_df['Team'] == team_name].reset_index(drop=True)
        
        # Calculate layout
        num_games = len(team_schedule)
        games_per_col = 10
        num_cols = (num_games + games_per_col - 1) // games_per_col

        # ----------------
        # Load opponent logos
        # ----------------
        opponent_logos = {}
        opponents_set = set(team_schedule['Opponent'].dropna())
        for opponent in opponents_set:
            logo_path = os.path.join(logo_folder, f"{opponent}.png")
            if os.path.exists(logo_path):
                opponent_logos[opponent] = Image.open(logo_path).convert("RGBA")
            else:
                opponent_logos[opponent] = None

        # ----------------
        # Create figure with GridSpec
        # ----------------
        # Calculate figure width based on number of columns
        col_width = 1  # Width per column in inches
        stats_width = 5   # Width for stats section
        total_width = (num_cols * col_width) + stats_width
        
        fig = plt.figure(figsize=(total_width, 12), dpi=200)
        fig.patch.set_facecolor('#CECEB2')
        
        # Create main grid: schedule area (left) and stats area (right)
        main_gs = gridspec.GridSpec(1, 2, figure=fig, 
                                width_ratios=[num_cols * col_width, stats_width], 
                                left=0.02, right=0.98, wspace=0.02)
        
        # Create nested grid for schedule columns
        schedule_gs = gridspec.GridSpecFromSubplotSpec(games_per_col, num_cols, 
                                                    subplot_spec=main_gs[0],
                                                    hspace=0.01, wspace=0.01)
        
        # Create stats area
        alpha_val = 0.9
        stats_ax = fig.add_subplot(main_gs[1])
        stats_ax.set_xlim(0, 1)
        stats_ax.set_ylim(0, 1)
        stats_ax.axis('off')
        stats_ax.set_facecolor('#CECEB2')
        net_score, net_rank = get_value_and_rank(stats_and_metrics, team_name, "NET_Score")
        rpg, rpg_rank = get_value_and_rank(stats_and_metrics, team_name, "RPG")
        ba, ba_rank = get_value_and_rank(stats_and_metrics, team_name, "BA")
        obp, obp_rank = get_value_and_rank(stats_and_metrics, team_name, "OBP")
        slg, slg_rank = get_value_and_rank(stats_and_metrics, team_name, "SLG")
        ops, ops_rank = get_value_and_rank(stats_and_metrics, team_name, "OPS")
        iso, iso_rank = get_value_and_rank(stats_and_metrics, team_name, "ISO")
        wOBA, wOBA_rank = get_value_and_rank(stats_and_metrics, team_name, "wOBA")
        era, era_rank = get_value_and_rank(stats_and_metrics, team_name, "ERA", False)
        whip, whip_rank = get_value_and_rank(stats_and_metrics, team_name, "WHIP", False)
        kp9, kp9_rank = get_value_and_rank(stats_and_metrics, team_name, "KP7")
        lob, lob_rank = get_value_and_rank(stats_and_metrics, team_name, "LOB%")
        kbb, kbb_rank = get_value_and_rank(stats_and_metrics, team_name, "K/BB")
        fip, fip_rank = get_value_and_rank(stats_and_metrics, team_name, "FIP", False)
        pct, pct_rank = get_value_and_rank(stats_and_metrics, team_name, "PCT")
        rating, rating_rank = get_value_and_rank(stats_and_metrics, team_name, "Rating")
        rqi, rqi_rank = get_value_and_rank(stats_and_metrics, team_name, "resume_quality")
        sos, sos_rank = get_value_and_rank(stats_and_metrics, team_name, "avg_expected_wins", False)
        war, war_rank = get_value_and_rank(stats_and_metrics, team_name, "fWAR")
        wpoe, wpoe_rank = get_value_and_rank(stats_and_metrics, team_name, "wpoe_pct")
        pythag, pythag_rank = get_value_and_rank(stats_and_metrics, team_name, "PYTHAG")
        record, record_rank = get_record_value_and_rank(stats_and_metrics, team_name, "Record")
        q1, q1_rank = get_value_and_rank(stats_and_metrics, team_name, "Q1")
        q2, q2_rank = get_value_and_rank(stats_and_metrics, team_name, "Q2")
        q3, q3_rank = get_value_and_rank(stats_and_metrics, team_name, "Q3")
        q4, q4_rank = get_value_and_rank(stats_and_metrics, team_name, "Q4")
        location_records = get_location_records(team_name, team_schedule)
        home_record = location_records.get("Home", "0-0")
        away_record = location_records.get("Away", "0-0")
        neutral_record = location_records.get("Neutral", "0-0")
        
        # Add team stats
        stats_ax.text(0.5, 0.973, f"#{net_rank} {team_name}", fontsize=24, fontweight='bold', 
                    ha='center', va='center')

        stats_ax.text(0.23, 0.923, f"OFFENSE", fontsize=16, fontweight='bold', 
                    ha='center', va='center')
        
        stats_ax.hlines(y=0.9, xmin=0.0, xmax=1, colors='black', linewidth=1)
        stats_ax.vlines(x=0.5, ymin=0.55, ymax=0.9, colors='black', linewidth=1)

        # axis, stat name coords, stat value coords, stat rank coords, ..., alignment of stat name
        display_stat_row(stats_ax, 0.0, 0.868, 0.48, 0.868, 0.23, 0.873+0.002, "RPG", rpg, ".2f", rpg_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.818, 0.48, 0.818, 0.23, 0.823+0.002, "BA", ba, ".3f", ba_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.768, 0.48, 0.768, 0.23, 0.773+0.002, "OBP", obp, ".3f", obp_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.718, 0.48, 0.718, 0.23, 0.723+0.002, "SLG", slg, ".3f", slg_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.668, 0.48, 0.668, 0.23, 0.673+0.002, "OPS", ops, ".3f", ops_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.618, 0.48, 0.618, 0.23, 0.623+0.002, "ISO", iso, ".3f", iso_rank, "left")
        display_stat_row(stats_ax, 0.0, 0.568, 0.48, 0.568, 0.23, 0.573+0.002, "wOBA", wOBA, ".3f", wOBA_rank, "left")

        # PITCHING section header
        stats_ax.text(0.77, 0.923, f"PITCHING", fontsize=16, fontweight='bold', 
                    ha='center', va='center')

        # Pitching stats
        display_stat_row(stats_ax, 1.0, 0.868, 0.52, 0.868, 0.77, 0.873+0.002, "ERA", era, ".2f", era_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.818, 0.52, 0.818, 0.77, 0.823+0.002, "WHIP", whip, ".2f", whip_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.768, 0.52, 0.768, 0.77, 0.773+0.002, "K/7", kp9, ".1f", kp9_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.718, 0.52, 0.718, 0.77, 0.723+0.002, "LOB%", lob, ".3f", lob_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.668, 0.52, 0.668, 0.77, 0.673+0.002, "K/BB", kbb, ".2f", kbb_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.618, 0.52, 0.618, 0.77, 0.623+0.002, "FIP", fip, ".2f", fip_rank, "right")
        display_stat_row(stats_ax, 1.0, 0.568, 0.52, 0.568, 0.77, 0.573+0.002, "PCT", pct, ".3f", pct_rank, "right")

        # TEAM METRICS section header
        stats_ax.text(0.5, 0.523, f"TEAM METRICS", fontsize=16, fontweight='bold', 
                    ha='center', va='center')
        stats_ax.hlines(y=0.5, xmin=0.0, xmax=1, colors='black', linewidth=1)
        stats_ax.vlines(x=0.5, ymin=0.25, ymax=0.5, colors='black', linewidth=1)

        # Team metrics - left column
        display_stat_row(stats_ax, 0.0, 0.468, 0.48, 0.468, 0.23, 0.473+0.002, 
                        "NET", net_score, ".3f", net_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.418, 0.48, 0.418, 0.23, 0.423+0.002, 
                        "RAT", rating, ".2f", rating_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.368, 0.48, 0.368, 0.23, 0.373+0.002, 
                        "RQI", rqi, ".2f", rqi_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.318, 0.48, 0.318, 0.23, 0.323+0.002, 
                        "SOS", sos, ".3f", sos_rank, "left")

        display_stat_row(stats_ax, 0.0, 0.268, 0.48, 0.268, 0.23, 0.273+0.002, 
                        "WAR", war, ".2f", war_rank, "left")

        # Team metrics - right column
        display_stat_row(stats_ax, 1.0, 0.468, 0.52, 0.468, 0.77, 0.473+0.002, 
                        "WPOE", wpoe, ".2f", wpoe_rank, "right")

        display_stat_row(stats_ax, 1.0, 0.418, 0.52, 0.418, 0.77, 0.423+0.002, 
                        "PYT", pythag, ".3f", pythag_rank, "right")
        
        stats_ax.text(0.0, 0.218, "REC", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.218, f"{record}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.168, "Q1", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.168, f"{q1}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.118, "Q2", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.118, f"{q2}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.068, "Q3", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.068, f"{q3}", fontsize=16, fontweight='bold', ha='center')

        stats_ax.text(0.0, 0.018, "Q4", fontsize=16, fontweight='bold', ha='left')
        stats_ax.text(0.23, 0.018, f"{q4}", fontsize=16, fontweight='bold', ha='center')
        
        stats_ax.text(1.0, 0.368, "HOME", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.368, f"{home_record}", fontsize=16, fontweight='bold', ha='left')

        stats_ax.text(1.0, 0.318, "AWAY", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.318, f"{away_record}", fontsize=16, fontweight='bold', ha='left')

        stats_ax.text(1.0, 0.268, "NEUTRAL", fontsize=16, fontweight='bold', ha='right')
        stats_ax.text(0.52, 0.268, f"{neutral_record}", fontsize=16, fontweight='bold', ha='left')

        # Add team logo
        if logo is not None:
            plot_logo(stats_ax, logo, (0.685, 0.125), zoom=0.2)

        # Fixed positions within each game axis (0-1 scale)
        logo_x_offset = 0.5       # Logo position
        rank_x_offset = 0.05       # NET ranking position
        loc_x_offset = 0.05       # Team name position (center-left)
        quad_x_offset = 0.95       # Quadrant position
        prob_x_offset = 0.95       # Win probability position

        # ----------------
        # Create axes for each game
        # ----------------
        for idx, (_, game_row) in enumerate(team_schedule.iterrows()):
            row = idx % games_per_col
            col = idx // games_per_col
            
            # Create axis for this game
            game_ax = fig.add_subplot(schedule_gs[row, col])
            game_ax.set_xlim(0, 1)
            game_ax.set_ylim(0, 1)
            game_ax.axis('off')
            
            if pd.notna(game_row.get('home_net')) and pd.notna(game_row.get('away_net')):
                if game_row['home_team'] == team_name:
                    opponent_net = int(game_row['away_net'])
                    if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                    else:
                        bottom_right_text = round(100 * game_row['home_win_prob'])
                        percent = "%"
                else:
                    opponent_net = int(game_row['home_net'])
                    if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                    else:
                        bottom_right_text = round(100 * (1 - game_row['home_win_prob']))
                        percent = "%"
                
                quadrant = get_quadrant(opponent_net, game_row['Location'])
                has_net_data = True
            else:
                opponent_net = ""
                if "W" in str(game_row['Result']) or "L" in str(game_row['Result']):
                        bottom_right_text = round(game_row['resume_quality'], 2)
                        percent = ""
                else:
                    bottom_right_text = round(100 * game_row['home_win_prob'])
                    percent = "%"
                quadrant = ""
                has_net_data = False

            game_color = get_game_color(game_row["Result"])
            if percent == "":
                max_val = round(team_schedule['resume_quality'].max(), 2)
                min_val = round(team_schedule['resume_quality'].min(), 2)

                if bottom_right_text == max_val:
                    game_color = 'mediumseagreen'
                elif bottom_right_text == min_val:
                    game_color = 'indianred'
            bg_rect = Rectangle((0, 0), 1, 1, transform=game_ax.transAxes,
                        facecolor=game_color, edgecolor='black', 
                        linewidth=1, zorder=0)
            game_ax.add_patch(bg_rect)
            
            # Add opponent logo
            opponent = game_row["Opponent"]
            if pd.isna(opponent):
                # handle missing opponent (e.g., Non D-I or bye week)
                game_ax.text(0.5, 0.5, "Non D-I", fontsize=10, ha='center', va='center', color='black')
            else:
                if opponent in opponent_logos and opponent_logos[opponent] is not None:
                    plot_logo(game_ax, opponent_logos[opponent], (logo_x_offset, 0.5), zoom=0.04)
                else:
                    # fallback if opponent not found or logo is None
                    game_ax.text(0.5, 0.5, opponent, fontsize=10, ha='center', va='center', color='black')
            
            # Add NET ranking
            game_ax.text(rank_x_offset, 0.88, f"#{opponent_net}", 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='left', va='center')
            
            # Add opponent name
            if game_row['Location'] == 'Home':
                loc = ""
            elif game_row['Location'] == 'Away':
                loc = "@"
            else:
                loc = "vs"
            game_ax.text(loc_x_offset, 0.1, loc, 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='left', va='center')
            
            # Add quadrant
            game_ax.text(quad_x_offset, 0.88, quadrant, 
                        fontsize=12, fontweight='bold', color='darkblue', 
                        ha='right', va='center')
            
            # Add win prob
            game_ax.text(prob_x_offset, 0.1, f"{bottom_right_text}{percent}", 
                        fontsize=12, fontweight='bold', color='black', 
                        ha='right', va='center')

        plt.tight_layout()
        # Save to BytesIO and get data before cleanup
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#CECEB2')
        buf.seek(0)
        
        # Get image data before closing
        image_data = buf.getvalue()
        
        # Aggressive cleanup
        plt.close(fig)
        fig.clf()
        del fig
        del buf
        gc.collect()
        
        return Response(content=image_data, media_type="image/png")
        
    except HTTPException as e:
            raise e
    except Exception as e:
        print(f"Error generating team profile: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Always close the logo image
        if logo is not None:
            logo.close()
            del logo
        
        # Close all opponent logos
        for opponent, opp_logo in opponent_logos.items():
            if opp_logo is not None:
                opp_logo.close()
        opponent_logos.clear()
        
        gc.collect()

@app.get("/api/softball/teams")
def get_softball_teams():
    """Get list of all teams"""
    try:
        modeling_stats, _ = load_softball_data()
        teams = sorted(modeling_stats['Team'].unique().tolist())
        return {"teams": teams}
    except Exception as e:
        print(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.get("/api/softball/conferences")
def get_softball_conferences():
    """Get list of all conferences"""
    try:
        modeling_stats, _ = load_softball_data()
        conferences = sorted(modeling_stats['Conference'].unique().tolist())
        return {"conferences": conferences}
    except Exception as e:
        print(f"Error getting conferences: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)