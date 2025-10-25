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
        spreads_path = os.path.join(BASE_PATH, f"y{year}", "Spreads", f"spreads_tracker_week{week}.xlsx")
        print(f"Attempting to load spreads from: {spreads_path}")
        print(f"Spreads file exists: {os.path.exists(spreads_path)}")
        
        if not os.path.exists(spreads_path):
            raise HTTPException(status_code=404, detail=f"Spreads file not found at: {spreads_path}")
        
        spreads = pd.read_excel(spreads_path)
        
        vegas_col = 'formattedSpread' if 'formattedSpread' in spreads.columns else 'formatted_spread'
        spreads['Vegas'] = spreads.get(vegas_col, '')
        
        required_cols = ['home_team', 'away_team', 'PEAR', 'pr_spread', 'difference', 'GQI']
        missing_cols = [col for col in required_cols if col not in spreads.columns]
        
        if missing_cols:
            print(f"Available columns: {spreads.columns.tolist()}")
            print(f"Missing columns: {missing_cols}")
            raise HTTPException(status_code=500, detail=f"Missing columns in spreads file: {missing_cols}")
        
        result = spreads[['home_team', 'away_team', 'PEAR', 'Vegas', 'difference', 'GQI', 'pr_spread']].dropna().to_dict('records')
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)