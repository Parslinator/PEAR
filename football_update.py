from scipy.optimize import minimize # type: ignore
from scipy.optimize import differential_evolution # type: ignore
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
from matplotlib.lines import Line2D # type: ignore
import cfbd # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import matplotlib.pyplot as plt # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # type: ignore
from matplotlib import gridspec # type: ignore
import os
import math # type: ignore
import matplotlib.patches as patches # type: ignore
from unittest import result
import datetime
import numpy as np # type: ignore
from PIL import ImageGrab # type: ignore
from base64 import b64decode # type: ignore
import PIL # type: ignore
import warnings
import seaborn as sns # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import pytz # type: ignore
import matplotlib.colors as mcolors # type: ignore
import matplotlib.font_manager as fm # type: ignore
checkmark_font = fm.FontProperties(family='DejaVu Sans')
# Suppress all warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
GLOBAL_HFA = 3

configuration = cfbd.Configuration(
    access_token = '7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
)
# configuration = cfbd.Configuration()
# configuration.api_key['Authorization'] = '7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
# configuration.api_key_prefix['Authorization'] = 'Bearer'
api_client = cfbd.ApiClient(configuration)
advanced_instance = cfbd.StatsApi(api_client)
games_api = cfbd.GamesApi(api_client)
betting_api = cfbd.BettingApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
teams_api = cfbd.TeamsApi(api_client)
metrics_api = cfbd.MetricsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
recruiting_api = cfbd.RecruitingApi(api_client)
drives_api = cfbd.DrivesApi(api_client)

from football_helper import modeling_data_import, build_power_ratings_multi_target, in_house_power_ratings, average_team_distribution, metric_creation, stats_formatting

outputs = modeling_data_import(9)
team_data = outputs["team_data"]
opponent_adjustment_schedule = outputs["opponent_adjustment_schedule"]
updated_metrics = outputs["updated_metrics"]
season_metrics = outputs["season_metrics"]
drive_quality = outputs["drive_quality"]
weighted_metrics = outputs["weighted_metrics"]
records = outputs["records"]
elo_ratings = outputs["elo_ratings"]
current_week = outputs["current_week"]
current_year = outputs["current_year"]
postseason = outputs["postseason"]

# --- Step 0: Select modeling features ---
core_metrics = [
    'Offense_ppa_adj',
    'Defense_ppa_adj',
    'Offense_explosiveness_adj',
    'Defense_explosiveness_adj',
    'Offense_successRate_adj',
    'Defense_successRate_adj',
    'Offense_rushing_adj',
    'Defense_rushing_adj',
    'Offense_passing_adj',
    'Defense_passing_adj'
]

team_data, opt_weights, diagnostics = in_house_power_ratings(team_data, opponent_adjustment_schedule, current_week, core_metrics, target_col='kford_rating', fixed_scale=16, home_field_advantage=3)

model_features = [
    'Offense_ppa_adj',
    'Defense_ppa_adj',
    'Offense_explosiveness_adj',
    'Defense_explosiveness_adj',
    'Offense_successRate_adj',
    'Defense_successRate_adj',
    'Offense_rushing_adj',
    'Defense_rushing_adj',
    'Offense_passing_adj',
    'Defense_passing_adj',
    'last_week',
    'avg_talent',
    'in_house'
]

team_data, diag, sys = build_power_ratings_multi_target(team_data, opponent_adjustment_schedule, model_features, ['kford_rating', 'sp_rating'], 0.4)
print(sys.print_diagnostics())

team_data, team_power_rankings = stats_formatting(team_data, current_week, current_year)
team_data, year_long_schedule, SOS, SOR, RTP, most_deserving, composite = metric_creation(team_data, records, current_week, current_year, postseason)

folder_path = f"./PEAR/PEAR Football/y{current_year}/Data"
os.makedirs(folder_path, exist_ok=True)

folder_path = f"./PEAR/PEAR Football/y{current_year}/Ratings"
os.makedirs(folder_path, exist_ok=True)

folder_path = f"./PEAR/PEAR Football/y{current_year}/Spreads"
os.makedirs(folder_path, exist_ok=True)

team_data.to_csv(f"./PEAR/PEAR Football/y{current_year}/Data/team_data_week{current_week}.csv")
team_power_rankings.to_csv(f'./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week}.csv')

print("---------- Power Ratings Done! ----------")

def PEAR_Win_Prob(home_power_rating, away_power_rating, neutral):
    if neutral == False:
        home_power_rating = home_power_rating + 1.5
    return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

games = []
for week in range(1,current_week):
    response = games_api.get_games(year=current_year, week=week,classification = 'fbs')
    games = [*games, *response]
games = [dict(
            id=g.id,
            season=g.season,
            week=g.week,
            start_date=g.start_date,
            home_team=g.home_team,
            home_elo=g.home_pregame_elo,
            away_team=g.away_team,
            away_elo=g.away_pregame_elo,
            home_points = g.home_points,
            away_points = g.away_points,
            neutral = g.neutral_site
            ) for g in games if g.home_points is not None]
schedule_info = pd.DataFrame(games)

schedule_info = schedule_info.merge(team_data[['team', 'power_rating']], 
                                    left_on='home_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'home_pr'})
schedule_info = schedule_info.drop(columns=['team'])
schedule_info = schedule_info.merge(team_data[['team', 'power_rating']], 
                                    left_on='away_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'away_pr'})
schedule_info = schedule_info.drop(columns=['team'])
fallback_value = team_data['power_rating'].mean() - 2 * team_data['power_rating'].std()
schedule_info['home_pr'] = schedule_info['home_pr'].fillna(fallback_value)
schedule_info['away_pr'] = schedule_info['away_pr'].fillna(fallback_value)
schedule_info['PEAR_win_prob'] = schedule_info.apply(
    lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral']), axis=1
)
schedule_info['home_win_prob'] = round((10**((schedule_info['home_elo'] - schedule_info['away_elo']) / 400)) / ((10**((schedule_info['home_elo'] - schedule_info['away_elo']) / 400)) + 1)*100,2)

current_year = int(current_year)
current_week = int(current_week)
print(f"Current Week: {current_week}, Current Year: {current_year}")

folder_path = f"./PEAR/PEAR Football/y{current_year}/Visuals/week_{current_week}"
os.makedirs(folder_path, exist_ok=True)

conf_folder_path = f"./PEAR/PEAR Football/y{current_year}/Visuals/week_{current_week}/Conference Projections"
os.makedirs(conf_folder_path, exist_ok=True)

from football_helper import best_and_worst, other_best_and_worst, draw_playoff_bracket_new, all_136_teams, _calculate_game_quality
from football_helper import create_conference_projection, plot_matchup_new, display_schedule_visual, conference_standings, prob_win_at_least_x

logos = outputs["logos"]

all_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Data/team_data_week{current_week}.csv")
team_data = pd.read_csv(f'./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week}.csv')

start_season_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week}.csv")
if os.path.exists(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week-1}.csv"):
    last_week_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week-1}.csv")
else:
    last_week_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week}.csv")

# Unique list of teams from your main dataset
unique_teams = all_data['team'].unique()

from concurrent.futures import ThreadPoolExecutor, as_completed # type: ignore
# Function to fetch a team's logo using the logos DataFrame
def fetch_logo(team):
    try:
        logo_url = logos.loc[logos['team'] == team, 'logo'].values[0][0]
        response = requests.get(logo_url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return team, img
    except Exception as e:
        print(f"Error loading logo for {team}: {e}")
        return team, None

import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
logo_folder = "./PEAR/PEAR Football/logos/"
team_logos = {}
def load_image(filename):
    team_name = filename[:-4].replace("_", " ")
    file_path = os.path.join(logo_folder, filename)
    try:
        img = Image.open(file_path).convert("RGBA")
        return (team_name, img)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return (team_name, None)

png_files = [f for f in os.listdir(logo_folder) if f.endswith(".png")]

with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(load_image, png_files)
    team_logos = dict(results)

try:
    top_25 = all_data.head(25).reset_index(drop=True)
    comparison = top_25.merge(
        last_week_data[['team', 'power_rating']].rename(columns={'power_rating': 'last_week_pr'}),
        on='team',
        how='left'
    )
    comparison['pr_diff'] = comparison['power_rating'] - comparison['last_week_pr']
    fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} PEAR Top 25", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "Power Rating (Δ from last week)", fontsize=12, ha='center', color='black')
    fig.text(0.5, 0.89, "@PEARatings", fontsize=12, ha='center', color='black', fontweight='bold')
    for i, ax in enumerate(axs.ravel()):
        team = comparison.loc[i, 'team']
        img = team_logos[team]
        pr = round(comparison.loc[i, 'power_rating'], 1)
        diff = comparison.loc[i, 'pr_diff']
        diff_str = f"{diff:+.1f}" if not pd.isna(diff) else "N/A"
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, -0.1,f"#{i+1} {team}\n{pr} ({diff_str})",fontsize=8,transform=ax.transAxes,ha='center',va='top')
        ax.axis('off')
    plt.savefig(os.path.join(folder_path, "top25"), bbox_inches='tight', dpi=300)
    print("Top 25 Done!")
except Exception as e:
    print(f"Error in code chunk: Top 25 Ratings. Error: {e}")

try:
    top_25 = most_deserving.head(25).reset_index(drop=True)
    fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} Most Deserving Top 25", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "AP Style Rankings", fontsize=12, ha='center', color='black')
    fig.text(0.5, 0.89, "@PEARatings", fontsize=12, ha='center', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'team']
        img = team_logos[team]
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, -0.1, f"#{i+1} {team} \n{round(top_25.loc[i, 'most_deserving_wins'], 3)}", fontsize=8, transform=ax.transAxes, ha='center', va='top')
        ax.axis('off')
    plt.savefig(os.path.join(folder_path, "most_deserving"), bbox_inches='tight', dpi=300)
    print("Most Deserving Done!")
except Exception as e:
    print(f"Error in code chunk: Most Deserving Ratings. Error: {e}")

try:
    top_25 = composite.head(25).reset_index(drop=True)
    fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} Composite Top 25", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "Combination of MD, OFF, DEF", fontsize=12, ha='center', color='black')
    fig.text(0.5, 0.89, "@PEARatings", fontsize=12, ha='center', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'team']
        img = team_logos[team]
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, -0.1, f"#{i+1} {team} \n{round(top_25.loc[i, 'composite_score'], 3)}", fontsize=8, transform=ax.transAxes, ha='center', va='top')
        ax.axis('off')
    plt.savefig(os.path.join(folder_path, "composite_ranking"), bbox_inches='tight', dpi=300)
    print("Composite Done!")
except Exception as e:
    print(f"Error in code chunk: Composite Ratings. Error: {e}")


try:
    group_of_5 = ['Conference USA', 'Mid-American', 'Sun Belt', 'American Athletic', 'Mountain West']
    top_25 = all_data[all_data['conference'].isin(group_of_5)].head(25).reset_index(drop=True)
    fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} GO5 PEAR", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "GO5 Power Rating", fontsize=12, ha='center', color='black')
    fig.text(0.5, 0.89, "@PEARatings", fontsize=12, ha='center', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'team']
        img = team_logos[team]
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, -0.1, f"#{i+1} {team} \n{round(top_25.loc[i, 'power_rating'],1)}", fontsize=8, transform=ax.transAxes, ha='center', va='top')
        ax.axis('off')
    plt.savefig(os.path.join(folder_path, "go5_top25"), bbox_inches='tight', dpi=300)
    print("GO5 Top 25 Done!")
except Exception as e:
    print(f"Error in code chunk: GO5 Ratings. Error: {e}")

try:
    all_136_teams(all_data, "power_rating", False, team_logos, 1, current_week, f"PEAR's Week {current_week} Power Ratings", folder_path, "all_power_ratings")
    print("All Power Ratings Done!")
except Exception as e:
    print(f"Error in code chunk: All Power Ratings. Error: {e}")

try:
    start_week = 1
    end_week = 17
    games_list = []
    for week in range(start_week,end_week):
        response = games_api.get_games(year=current_year, week=week,classification = 'fbs')
        games_list = [*games_list, *response]
    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_elo=g.away_pregame_elo,
                home_points = g.home_points,
                away_points = g.away_points,
                neutral = g.neutral_site
                ) for g in games_list if g.home_points is None]
    uncompleted = pd.DataFrame(games)
    uncompleted = uncompleted.merge(team_data[['team', 'power_rating']], 
                                        left_on='home_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'home_pr'})
    uncompleted = uncompleted.drop(columns=['team'])
    uncompleted = uncompleted.merge(team_data[['team', 'power_rating']], 
                                        left_on='away_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'away_pr'})
    uncompleted = uncompleted.drop(columns=['team'])
    missing_rating =round(team_data['power_rating'].mean() - 2.25*team_data['power_rating'].std(),2)
    uncompleted['home_pr'].fillna(missing_rating, inplace=True)
    uncompleted['away_pr'].fillna(missing_rating, inplace=True)
    uncompleted['PEAR_win_prob'] = uncompleted.apply(
        lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
    )
    uncompleted['home_win_prob'] = round((10**((uncompleted['home_elo'] - uncompleted['away_elo']) / 400)) / ((10**((uncompleted['home_elo'] - uncompleted['away_elo']) / 400)) + 1)*100,2)

    def adjust_home_pr(home_win_prob):
        if home_win_prob is None or (isinstance(home_win_prob, float) and math.isnan(home_win_prob)):
            return 0
        return ((home_win_prob - 50) / 50) * 1

    uncompleted['pr_spread'] = (GLOBAL_HFA + uncompleted['home_pr'] + (uncompleted['home_win_prob'].apply(adjust_home_pr)) - uncompleted['away_pr']).round(1)
    uncompleted['pr_spread'] = np.where(uncompleted['neutral'], uncompleted['pr_spread'] - GLOBAL_HFA, uncompleted['pr_spread']).round(1)
    uncompleted['PEAR'] = uncompleted.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)

    home_wins = uncompleted.groupby('home_team')['PEAR_win_prob'].sum().reset_index()
    home_wins.columns = ['team', 'home_win_exp']
    away_wins = uncompleted.groupby('away_team')['PEAR_win_prob'].apply(lambda x: (1 - x).sum()).reset_index()
    away_wins.columns = ['team', 'away_win_exp']
    total_wins = pd.merge(home_wins, away_wins, on='team', how='outer').fillna(0)
    total_wins['win_total'] = round(total_wins['home_win_exp'] + total_wins['away_win_exp'],1)
    # result = total_wins[['team', 'win_total']].sort_values('win_total', ascending=False).reset_index(drop=True)
    updated_win_total = pd.merge(total_wins[['team', 'win_total']], records[['team', 'wins']], on='team', how='left')
    updated_win_total = pd.merge(team_data[['team']], updated_win_total, on='team', how='left')
    updated_win_total['updated_win_total'] = updated_win_total['win_total'] + updated_win_total['wins']
    all_136_teams(updated_win_total, 'updated_win_total', False, team_logos, 1, current_week, f"PEAR's Updated Win Totals", folder_path, "updated_win_totals")
except Exception as e:
    print(f"Error in code chunk: Updated Win Totals. Error: {e}")

try:
    all_136_teams(all_data, 'offensive_total', False, team_logos, 2, current_week, f"Week {current_week} PEAR Offensive Efficiencies - A Measure of PPA, PPO, and Drive Quality", folder_path, "all_offensive_efficiency")
    print("All Offensive Efficiencies Done!")
except Exception as e:
    print(f"Error in code chunk: All Offensive Efficiencies. Error: {e}")

try:
    all_136_teams(all_data, 'defensive_total', False, team_logos, 2, current_week, f"Week {current_week} PEAR Defensive Efficiencies - A Measure of PPA, PPO, and Drive Quality", folder_path, "all_defensive_efficiency")
    print("All Defensive Efficiencies Done!")
except Exception as e:
    print(f"Error in code chunk: All Defensive Efficiencies. Error: {e}")

try:
    all_136_teams(all_data, 'offensive_rating', False, team_logos, 1, current_week, f"Week {current_week} PEAR Offensive Ratings", folder_path, "all_offensive_ratings")
    print("All Offensive Ratings Done!")
except Exception as e:
    print(f"Error in code chunk: All Offensive Ratings. Error: {e}")

try:
    all_136_teams(all_data, 'defensive_rating', True, team_logos, 1, current_week, f"Week {current_week} PEAR Defensive Ratings", folder_path, "all_defensive_ratings")
    print("All Defensive Ratings Done!")
except Exception as e:
    print(f"Error in code chunk: All Defensive Ratings. Error: {e}")

try:
    all_136_teams(all_data, 'most_deserving_wins', False, team_logos, 2, current_week, f"Week {current_week} PEAR Most Deserving Rankings - AP Style Rankings", folder_path, "all_most_deserving")
    print("All Most Deserving Done!")
except Exception as e:
    print(f"Error in code chunk: All Most Deserving. Error: {e}")

try:
    all_136_teams(all_data, 'composite_score', False, team_logos, 3, current_week, f"Week {current_week} PEAR Composite Rankings - Combining MD, OFF, and DEF", folder_path, "all_composite_rankings")
    print("All Composite Rankings Done!")
except Exception as e:
    print(f"Error in code chunk: All Composite Rankings. Error: {e}")

try:
    start_week = current_week
    end_week = 17
    games_list = []
    for week in range(start_week,end_week):
        response = games_api.get_games(year=2025, week=week,classification = 'fbs')
        games_list = [*games_list, *response]
    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_elo=g.away_pregame_elo,
                home_points = g.home_points,
                away_points = g.away_points,
                neutral = g.neutral_site
                ) for g in games_list]
    uncompleted_games = pd.DataFrame(games)
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='home_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'home_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='away_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'away_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    missing_rating =round(team_data['power_rating'].mean() - 2.25*team_data['power_rating'].std(),2)
    uncompleted_games['home_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['away_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['PEAR_win_prob'] = uncompleted_games.apply(
        lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
    )
    uncompleted_games['home_win_prob'] = round((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) / ((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) + 1)*100,2)

    uncompleted_games['pr_spread'] = (GLOBAL_HFA + uncompleted_games['home_pr'] + (uncompleted_games['home_win_prob'].apply(adjust_home_pr)) - uncompleted_games['away_pr']).round(1)
    uncompleted_games['pr_spread'] = np.where(uncompleted_games['neutral'], uncompleted_games['pr_spread'] - GLOBAL_HFA, uncompleted_games['pr_spread']).round(1)
    uncompleted_games['PEAR'] = uncompleted_games.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)
    results = []
    team_list = team_data['team'].tolist()
    for team in team_list:
        team_schedule = uncompleted_games[
            (uncompleted_games['home_team'] == team) |
            (uncompleted_games['away_team'] == team)
        ][['home_team', 'away_team', 'PEAR']].copy()
        
        # Count how many times team appears in PEAR column
        count_in_pear = team_schedule['PEAR'].str.startswith(team).sum()
        
        results.append([team, count_in_pear, len(team_schedule)])

    pear_df = pd.DataFrame(results, columns=['team', 'PEAR_Count', 'games_remaining'])

    mulligans = all_data[['team', 'avg_expected_wins', 'power_rating', 'conference']]
    mulligans = pd.merge(mulligans, records[['team', 'wins', 'losses']])
    mulligans = mulligans.merge(pear_df, on='team', how='left')
    mulligans['at_large_wins'] = np.ceil(mulligans['avg_expected_wins']-0.5).astype(int)
    mulligans['mulligans'] = mulligans['wins'] + mulligans['PEAR_Count'] - mulligans['at_large_wins']
    mulligans['mulligans'] = mulligans['mulligans'].where(
        mulligans['mulligans'].abs() < mulligans['games_remaining'],
        -15
    )
    power_confs = ['Big Ten', 'Big 12', 'ACC', 'SEC']
    mulligans.loc[
        (~mulligans['conference'].isin(power_confs)) & 
        (mulligans['team'] != 'Notre Dame') & 
        (mulligans['losses'] >= 1), 
        'mulligans'
    ] = -15
    mulligans = mulligans.sort_values(['mulligans', 'power_rating'], ascending=[False, False]).reset_index(drop=True)
    all_136_teams(mulligans, 'mulligans', False, team_logos, 0, current_week, f"Week {current_week} Mulligans / Upsets", folder_path, "mulligans_vs_upset")
    print("Mulligans vs. Upsets Done!")
except Exception as e:
    print(f"Error in code chunk: Mulligans vs. Upset. Error: {e}")

try:
    last_week_ratings = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week-1}.csv")
    delta = pd.merge(team_data[['team', 'power_rating']], last_week_ratings[['team', 'power_rating']], how='left', on='team')
    delta['delta'] = delta['power_rating_x'] - delta['power_rating_y']
    all_136_teams(delta, 'delta', False, team_logos, 1, current_week, f"Ratings Change From Week {current_week-1} to Week {current_week}", folder_path, "ratings_delta_last_week")
    print("Ratings Delta LW Done!")
except Exception as e:
    print(f"Error in code chunk: Ratings Delta From Last Week. Error: {e}")

try:
    preseason = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week1.csv")
    delta = pd.merge(team_data[['team', 'power_rating']], preseason[['team', 'power_rating']], how='left', on='team')
    delta['delta'] = delta['power_rating_x'] - delta['power_rating_y']
    all_136_teams(delta, 'delta', False, team_logos, 1, current_week, f"Ratings Change From Preseason to Week {current_week}", folder_path, "ratings_delta_preseason")
    print("Ratings Delta Preseason Done!")
except Exception as e:
    print(f"Error in code chunk: Ratings Delta Preseason. Error: {e}")

try:
    from scipy.stats import binom
    from math import comb

    mulligans['wins_needed'] = mulligans['at_large_wins'] - mulligans['wins']
    mulligans['prob_reach_wins'] = mulligans.apply(
        lambda row: 100*prob_win_at_least_x(row['team'], row['wins_needed'], uncompleted_games).round(2),
        axis=1
    )
    mulligans = mulligans.sort_values(['prob_reach_wins', 'power_rating'], ascending=[False, False]).reset_index(drop=True)
    all_136_teams(mulligans, 'prob_reach_wins', False, team_logos, 0, current_week, f"Week {current_week} At-Large Playoff Discussion Chances", folder_path, "at_large_playoff_chances", "Probability each team reaches the win total needed to stay in at-large contention - this is NOT Playoff Probability")
    print("At Large Playoff Done!")
except Exception as e:
    print(f"Error in code chunk: At Large Playoff Chances. Error: {e}")

try:
    all_136_teams(all_data, 'avg_expected_wins', True, team_logos, 2, current_week, f"Week {current_week} SOS Rankings", folder_path, "all_sos")
    print("All SOS Done!")
except Exception as e:
    print(f"Error in code chunk: All SOS. Error: {e}")

try:
    conference_stats = all_data.groupby('conference')['power_rating'].agg(['mean', 'min', 'max']).reset_index()
    conference_stats = conference_stats.sort_values(by='mean', ascending=False)

    plt.figure(figsize=(8, 4), facecolor='#CECEB2',dpi=125)
    bars = plt.bar(conference_stats['conference'], conference_stats['mean'], 
                    color='#A74C54', 
                    yerr=[conference_stats['mean'] - conference_stats['min'], 
                        conference_stats['max'] - conference_stats['mean']], 
                    capsize=5)
    ax = plt.gca()
    ax.set_facecolor('#CECEB2')
    for spine in ax.spines.values():
        spine.set_color('#CECEB2')  # Set the border color
        spine.set_linewidth(2)  # Adjust the border thickness if needed
    ax.xaxis.set_tick_params(color='black')  # X-axis ticks
    ax.yaxis.set_tick_params(color='black')  # Y-axis ticks
    ax.spines['bottom'].set_color('black')  # Bottom spine
    ax.spines['left'].set_color('black')  # Left spine
    ax.spines['top'].set_color('none')  # Top spine
    ax.spines['right'].set_color('none')  # Right spine

    plt.axhline(y=0, color = 'black', linestyle='--')
    plt.xlabel('Conference', fontsize=12, color='black')
    plt.ylabel('Average Power Rating', fontsize=12, color='black')
    plt.title('Average Power Rating by Conference', fontsize=14, fontweight='bold', color='black')
    plt.xticks(rotation=45, ha='right', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    plt.text(-1.8, 28.5, "@PEARatings", fontsize=12, color='black', fontweight='bold', ha='left')
    plt.tight_layout()
    file_path = os.path.join(folder_path, "conference_average")
    plt.savefig(file_path, dpi = 300)
    print("Conference Average Done!")
except Exception as e:
    print(f"Error in code chunk: Conference Average Ratings. Error: {e}")

try:
    best_and_worst(all_data, team_logos, 'total_turnovers_scaled', "Turnover Margin Percentiles", 
                    "Percentile Based: 100 is best, 1 is worst", "turnovers", folder_path)
    print("Turnovers Done!")
except Exception as e:
    print(f"Error in code chunk: Turnover Margin Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['offensive_total'] = scaler100.fit_transform(all_data[['offensive_total']])
    best_and_worst(all_data, team_logos, 'offensive_total', "PEAR Raw Offenses: Best and Worst 25", 
                    "Percentile Based: 100 is best, 1 is worst", "offenses", folder_path)
    print("Offenses Done!")
except Exception as e:
    print(f"Error in code chunk: Offenses Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['defensive_total'] = scaler100.fit_transform(all_data[['defensive_total']])
    best_and_worst(all_data, team_logos, 'defensive_total', "PEAR Raw Defenses: Best and Worst 25", 
                    "100 is the best raw defense, 1 is the worst", "defenses", folder_path)
    print("Defenses Done!")
except Exception as e:
    print(f"Error in code chunk: Defenses Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['STM_scaled'] = scaler100.fit_transform(all_data[['STM']])
    best_and_worst(all_data, team_logos, 'STM_scaled', "PEAR Special Teams", 
                    "Percentile Based: 100 is best, 1 is worst", "special_teams",folder_path)
    print("Special Teams Done!")
except Exception as e:
    print(f"Error in code chunk: Special Teams Ratings. Error: {e}")

try:
    pbr_min = all_data['PBR'].min()
    pbr_max = all_data['PBR'].max()
    all_data['PBR_scaled'] = 100 - (all_data['PBR'] - pbr_min) * (99 / (pbr_max - pbr_min))
    best_and_worst(all_data, team_logos, 'PBR_scaled', "PEAR Penalty Burden Ratio", 
                    "How Penalties Impact Success - 100 is best, 1 is worst", "penalty_burden_ratio",folder_path)
    print("PBR Done!")
except Exception as e:
    print(f"Error in code chunk: PBR Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['DCE_scaled'] = scaler100.fit_transform(all_data[['DCE']])
    best_and_worst(all_data, team_logos, 'DCE_scaled', "PEAR Drive Control Efficiency", 
                    "How Well You Control the Ball - 100 is best, 1 is worst", "drive_control_efficiency",folder_path)
    print("DCE Done!")
except Exception as e:
    print(f"Error in code chunk: DCE Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['DDE_scaled'] = scaler100.fit_transform(all_data[['DDE']])
    best_and_worst(all_data, team_logos, 'DDE_scaled', "PEAR Drive Disruption Efficiency", 
                    "How Well You Disrupt the Offense - 100 is best, 1 is worst", "drive_disruption_efficiency",folder_path)
    print("DDE Done!")
except Exception as e:
    print(f"Error in code chunk: DDE Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['offensive_total'] = scaler100.fit_transform(all_data[['offensive_total']])
    all_data['offensive_total'] = all_data['offensive_total'] - all_data['offensive_total'].mean()
    all_data['defensive_total'] = scaler100.fit_transform(all_data[['defensive_total']])
    all_data['defensive_total'] = all_data['defensive_total'] - all_data['defensive_total'].mean()
    fig, ax = plt.subplots(figsize=(15, 9),dpi=125)
    plt.gca().set_facecolor('#CECEB2')
    plt.gcf().set_facecolor('#CECEB2')
    logo_size = 2  # Half the size of the logo to create spacing
    for i in range(len(all_data)):
        img = team_logos[all_data.loc[i,'team']]
        ax.imshow(img, aspect='auto', 
                extent=(all_data['defensive_total'].iloc[i] - (logo_size-0.5),
                        all_data['defensive_total'].iloc[i] + (logo_size-0.5),
                        all_data['offensive_total'].iloc[i] - logo_size,
                        all_data['offensive_total'].iloc[i] + logo_size))
    ax.set_xlabel('Total Defense')
    ax.set_ylabel('Total Offense')
    ax.set_title('Team Offense vs Defense', fontweight='bold', fontsize=14)
    plt.xlim(all_data['defensive_total'].min() - 5, all_data['defensive_total'].max() + 5)  # Adjust x-axis limits for visibility
    plt.ylim(all_data['offensive_total'].min() - 5, all_data['offensive_total'].max() + 5)  # Adjust y-axis limits for visibility
    plt.grid(False)  # Turn off the grid
    plt.axhline(0, linestyle='--', color='black', alpha = 0.3)
    plt.axvline(0, linestyle='--', color='black', alpha = 0.3)
    plt.text(45, 50, "Good Offense, Good Defense", fontsize=10, fontweight='bold', ha='center')
    plt.text(-30, 50, "Good Offense, Bad Defense", fontsize=10, fontweight='bold', ha='center')
    plt.text(45, -50, "Bad Offense, Good Defense", fontsize=10, fontweight='bold', ha='center')
    plt.text(-30, -50, "Bad Offense, Bad Defense", fontsize=10, fontweight='bold', ha='center')
    file_path = os.path.join(folder_path, "offense_vs_defense")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    print("Offense vs. Defense Done!")
except Exception as e:
    print(f"Error in code chunk: Offense vs. Defense Ratings. Error: {e}")

try:
    performance_list = []
    completed_games = year_long_schedule.dropna(subset=["home_points", "away_points"])
    for team in team_data['team']:
        wins = records.loc[records['team'] == team, 'wins'].values[0]
        losses = records.loc[records['team'] == team, 'losses'].values[0]

        team_games = completed_games[(completed_games['home_team'] == team) | (completed_games['away_team'] == team)].copy()
        team_games['team_win_prob'] = np.where(team_games['home_team'] == team,
                                                team_games['PEAR_win_prob'],
                                                100 - team_games['PEAR_win_prob'])
        xwins = round(team_games['team_win_prob'].sum() / 100, 2)
        if len(team_games) != (wins + losses):
            xwins += 1
        performance_list.append(wins - xwins)

    current_performance = pd.DataFrame({'team': team_data['team'], 'performance': performance_list})
    other_best_and_worst(current_performance, team_logos, 'performance', 
                    f'Week {current_week} PEAR Overperformers and Underperformers', 
                    "Wins ABOVE or BELOW Your Retroactive Win Expectation", "overperformer_and_underperformer",folder_path)
    print("Achieving vs. Expectation Done!")
except Exception as e:
    print(f"Error in code chunk: Achieving vs. Expectation Ratings. Error: {e}")

try:
    other_best_and_worst(SOS, team_logos, 'avg_expected_wins', f'Week {current_week} PEAR SOS', 
                    "Efficiency of an Elite Team Against Your Opponents", "strength_of_schedule",folder_path)
    print("SOS Done!")
except Exception as e:
    print(f"Error in code chunk: SOS Ratings. Error: {e}")

try:
    other_best_and_worst(SOR, team_logos, 'wins_above_good', f'Week {current_week} PEAR SOR', 
                    "Wins Above or Below a Good Team", "strength_of_record",folder_path)
    print("SOR Done!")
except Exception as e:
    print(f"Error in code chunk: SOR Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['most_deserving_scaled'] = scaler100.fit_transform(all_data[['most_deserving_wins']])
    all_data['talent_performance'] = (all_data['most_deserving_scaled'] - all_data['avg_talent']) / math.sqrt(2)
    best_and_worst(all_data, team_logos, 'talent_performance', "PEAR Talent Performance Gap", 
                    "Is Your Team Outperforming or Underperforming Its Roster?", "talent_performance",folder_path)
    print("Talent Performance Done!")
except Exception as e:
    print(f"Error in code chunk: Talent Performance Ratings. Error: {e}")

try:
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['defensive_total'] = scaler100.fit_transform(all_data[['defensive_total']])
    all_data['offensive_total'] = scaler100.fit_transform(all_data[['offensive_total']])

    all_data['dependence_score'] = (all_data['offensive_total'] - all_data['defensive_total']) / (all_data['offensive_total'] + all_data['defensive_total'])
    other_best_and_worst(all_data, team_logos, 'dependence_score', 'PEAR Unit Dependence', 'Values near 1 indicate offensive dependence, while values near -1 indicate defensive dependence', 'dependence_score',folder_path)
    print('Dependence Score Done!')
except Exception as e:
    print(f"Error in code chunk: Dependence Score Ratings. Error: {e}")

try:
    other_best_and_worst(RTP, team_logos, 'RTP', f'PEAR Margin of Victory', "If You Are Expected to Win by 10 Points, Your Average MOV is __ Points", "mov_performance",folder_path)
    print("MOV Performance Done!")
except Exception as e:
    print(f"Error in code chunk: MOV Performance Ratings. Error: {e}")

try:
    all_136_teams(RTP, 'RTP', False, team_logos, 1, current_week, f"Adjusted Margin of Victory Performance - How Teams Do vs Projection", folder_path, "all_mov", "Difference between actual and expected margins of victory, adjusted for opponent strength, standardized by game-level variance")
    print("All MOV Done!")
except Exception as e:
    print(f"Error in code chunk: All MOV. Error: {e}")

try:
    average_pr = round(team_data['power_rating'].mean(), 2)
    good_team_pr = round(team_data['power_rating'].std() + team_data['power_rating'].mean(), 2)
    elite_team_pr = round(2 * team_data['power_rating'].std() + team_data['power_rating'].mean(), 2)
    super_team_pr = round(3 * team_data['power_rating'].std() + team_data['power_rating'].mean(), 2)
    # Merge team_data with logos to include the logo column
    team_data_logo = team_data.merge(logos, on='team', how='left')

    # Categorize teams
    super_teams = team_data_logo[team_data_logo['power_rating'] > super_team_pr].reset_index(drop=True)
    elite_teams = team_data_logo[team_data_logo['power_rating'] > elite_team_pr].reset_index(drop=True)
    good_teams = team_data_logo[
        (team_data_logo['power_rating'] > good_team_pr) & (team_data_logo['power_rating'] <= elite_team_pr)
    ].reset_index(drop=True)
    average_teams = team_data_logo[
        (team_data_logo['power_rating'] > average_pr) & (team_data_logo['power_rating'] <= good_team_pr)
    ].reset_index(drop=True)
    below_average_teams = team_data_logo[team_data_logo['power_rating'] <= average_pr].reset_index(drop=True)
    # Function to plot logos with centering and dynamic spacing
    def plot_team_logos(ax, teams, level, spacing_factor=1.5, row_spacing=0.5, max_teams_per_row=10):
        count = len(teams)
        rows = (count // max_teams_per_row) + (1 if count % max_teams_per_row != 0 else 0)
        for row in range(rows):
            # Calculate the row teams
            start_idx = row * max_teams_per_row
            end_idx = min((row + 1) * max_teams_per_row, count)
            row_teams = teams.iloc[start_idx:end_idx]
            # Center x positions for current row
            x = np.linspace(-max_teams_per_row / 2, max_teams_per_row / 2, len(row_teams)) * spacing_factor
            y = np.full(len(row_teams), -level - row * row_spacing)  # Offset rows within a tier

            for i, (team, logo) in enumerate(zip(row_teams['team'], row_teams['logo'])):
                img = team_logos[team]
                imagebox = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(imagebox, (x[i], y[i]), frameon=False)
                ax.add_artist(ab)
    # Plotting
    fig, ax = plt.subplots(figsize=(11, 15))
    fig.patch.set_facecolor('#CECEB2')
    # Plot logos with adjusted spacing
    plot_team_logos(ax, elite_teams, -0.2, spacing_factor=1, row_spacing = 0.4, max_teams_per_row=max(len(elite_teams) // 2, 1)+1)
    plot_team_logos(ax, good_teams, 0.7, spacing_factor=1, row_spacing = 0.4, max_teams_per_row=max(len(good_teams) // 3, 1)+1)
    plot_team_logos(ax, average_teams, 2, spacing_factor=1, row_spacing = 0.5, max_teams_per_row=max(len(average_teams) // 4, 1)+1)
    plot_team_logos(ax, below_average_teams, 4.1, spacing_factor=1, row_spacing = 0.5, max_teams_per_row=max(len(below_average_teams) // 4, 1)+1)
    ax.hlines(y=-0.5, xmin=-2.5, xmax=2.5, colors='black', linewidth=3)
    ax.hlines(y=-1.75, xmin=-6.5, xmax=6.5, colors='black', linewidth=3)
    ax.hlines(y=-3.8, xmin=-9, xmax=9, colors='black', linewidth=3)
    ax.hlines(y=-5.9, xmin=-9, xmax=9, colors='black', linewidth=3)
    ax.text(-3, -0.35, "Elite Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(3, -0.35, "Elite Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-7, -1.4, "Good Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(7, -1.4, "Good Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-9.5, -3.2, "Average Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(9.5, -3.2, "Average Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-10.5, -5.5, "Below Average Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(10.5, -5.5, "Below Average Teams", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    # ax.text(0, -6.3, "Below Average Teams", ha='center', fontsize=12, fontweight='bold', color='black')
    # Adjust plot limits and formatting
    ax.set_xlim(-9.5, 9.5)  # Center based on the widest row
    ax.set_ylim(-5.5, 1)  # Extend y-axis to fit all levels
    ax.set_yticks([-0.75, -2.25, -4, -6])
    ax.set_yticklabels(["Elite Teams", "Good Teams", "Average Teams", "Below Average Teams"])
    ax.text(0, 0.75, "PEAR Power Ratings Pyramid", ha='center', fontsize=24, fontweight='bold')
    ax.text(0, 0.62, "@PEARatings", fontsize=16, ha='center', fontweight='bold')
    ax.text(0, 0.50, "Teams Listed in Descending Order Within Each Row", fontsize = 16, ha='center')
    ax.axis("off")  # Remove axis lines
    plt.tight_layout()
    file_path = os.path.join(folder_path, "power_rating_team_pyramid")
    plt.savefig(file_path, dpi = 300)
    print("Power Ratings Team Pyramid Done!")
except Exception as e:
    print(f"Error occurred while drawing Power Ratings Team Pyramid: {e}")

try:
    average_pr = round(most_deserving['most_deserving_wins'].mean(), 2)
    good_team_pr = round(most_deserving['most_deserving_wins'].std() + most_deserving['most_deserving_wins'].mean(), 2)
    elite_team_pr = round(2 * most_deserving['most_deserving_wins'].std() + most_deserving['most_deserving_wins'].mean(), 2)
    # Merge team_data with logos to include the logo column
    team_data_logo = most_deserving.merge(logos, on='team', how='left')
    # Categorize teams
    elite_teams = team_data_logo[team_data_logo['most_deserving_wins'] > elite_team_pr].reset_index(drop=True)
    good_teams = team_data_logo[
        (team_data_logo['most_deserving_wins'] > good_team_pr) & (team_data_logo['most_deserving_wins'] <= elite_team_pr)
    ].reset_index(drop=True)
    average_teams = team_data_logo[
        (team_data_logo['most_deserving_wins'] > average_pr) & (team_data_logo['most_deserving_wins'] <= good_team_pr)
    ].reset_index(drop=True)
    below_average_teams = team_data_logo[team_data_logo['most_deserving_wins'] <= average_pr].reset_index(drop=True)
    # Function to plot logos with centering and dynamic spacing
    def plot_team_logos(ax, teams, level, spacing_factor=1.5, row_spacing=0.5, max_teams_per_row=10):
        count = len(teams)
        rows = (count // max_teams_per_row) + (1 if count % max_teams_per_row != 0 else 0)
        for row in range(rows):
            # Calculate the row teams
            start_idx = row * max_teams_per_row
            end_idx = min((row + 1) * max_teams_per_row, count)
            row_teams = teams.iloc[start_idx:end_idx]
            # Center x positions for current row
            x = np.linspace(-max_teams_per_row / 2, max_teams_per_row / 2, len(row_teams)) * spacing_factor
            y = np.full(len(row_teams), -level - row * row_spacing)  # Offset rows within a tier
            for i, (team, logo) in enumerate(zip(row_teams['team'], row_teams['logo'])):
                img = team_logos[team]
                imagebox = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(imagebox, (x[i], y[i]), frameon=False)
                ax.add_artist(ab)
    # Plotting
    fig, ax = plt.subplots(figsize=(11, 15))
    fig.patch.set_facecolor('#CECEB2')
    # Plot logos with adjusted spacing
    plot_team_logos(ax, elite_teams, -0.2, spacing_factor=1, row_spacing = 0.4, max_teams_per_row=max(len(elite_teams) // 2, 1)+1)
    plot_team_logos(ax, good_teams, 0.7, spacing_factor=1, row_spacing = 0.4, max_teams_per_row=max(len(good_teams) // 3, 1)+1)
    plot_team_logos(ax, average_teams, 2, spacing_factor=1, row_spacing = 0.5, max_teams_per_row=max(len(average_teams) // 4, 1)+1)
    plot_team_logos(ax, below_average_teams, 4.1, spacing_factor=1, row_spacing = 0.5, max_teams_per_row=max(len(below_average_teams) // 4, 1)+1)
    ax.hlines(y=-0.5, xmin=-2.5, xmax=2.5, colors='black', linewidth=3)
    ax.hlines(y=-1.75, xmin=-6.5, xmax=6.5, colors='black', linewidth=3)
    ax.hlines(y=-3.8, xmin=-9, xmax=9, colors='black', linewidth=3)
    ax.hlines(y=-5.9, xmin=-9, xmax=9, colors='black', linewidth=3)
    ax.text(-3, -0.35, "Elite Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(3, -0.35, "Elite Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-7, -1.4, "Good Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(7, -1.4, "Good Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-9.5, -3.2, "Average Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(9.5, -3.2, "Average Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(-10.5, -5.5, "Bad Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    ax.text(10.5, -5.5, "Bad Resume", ha='left', fontsize=16, fontweight='bold',color='black', rotation=90)
    # ax.text(0, -6.3, "Below Average Teams", ha='center', fontsize=12, fontweight='bold', color='black')
    # Adjust plot limits and formatting
    ax.set_xlim(-9.5, 9.5)  # Center based on the widest row
    ax.set_ylim(-5.5, 1)  # Extend y-axis to fit all levels
    ax.set_yticks([-0.75, -2.25, -4, -6])
    ax.set_yticklabels(["Elite Teams", "Good Teams", "Average Teams", "Below Average Teams"])
    ax.text(0, 0.75, "PEAR Most Deserving Pyramid", ha='center', fontsize=24, fontweight='bold')
    ax.text(0, 0.62, "@PEARatings", fontsize=16, ha='center', fontweight='bold')
    ax.text(0, 0.50, "Teams Listed in Descending Order Within Each Row", fontsize = 16, ha='center')
    ax.axis("off")  # Remove axis lines
    plt.tight_layout()
    file_path = os.path.join(folder_path, "most_deserving_team_pyramid")
    plt.savefig(file_path, dpi = 300)
    print("Most Deserving Team Pyramid Done!")
except Exception as e:
    print(f"Error occurred while drawing Most Deserving Team Pyramid: {e}")

try:
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 9),dpi=125)
    plt.gca().set_facecolor('#CECEB2')
    plt.gcf().set_facecolor('#CECEB2')
    # Set the size of the logos (adjust the numbers to make logos smaller or larger)
    logo_size = 0.9  # Half the size of the logo to create spacing
    # Loop through the team_data DataFrame to plot logos
    for i in range(len(all_data)):
        # Get the logo image from the URL
        img = team_logos[all_data.loc[i,'team']]
        # Calculate the extent for the logo
        # Here we use logo_size for both sides to center the logo at the specific coordinates
        ax.imshow(img, aspect='auto', 
                extent=(all_data['most_deserving_wins'].iloc[i] - (logo_size-0.7),
                        all_data['most_deserving_wins'].iloc[i] + (logo_size-0.7),
                        all_data['power_rating'].iloc[i] - logo_size,
                        all_data['power_rating'].iloc[i] + logo_size))
    # Set axis labels
    ax.set_xlabel('Resume (Record Strength)', fontweight='bold')
    ax.set_ylabel('Ratings (Team Strength)', fontweight='bold')
    ax.set_title('Resume vs. Ratings', fontweight='bold', fontsize=14)
    # Show the plot
    plt.xlim(all_data['most_deserving_wins'].min() - 1, all_data['most_deserving_wins'].max() + 1)  # Adjust x-axis limits for visibility
    plt.ylim(all_data['power_rating'].min() - 3, all_data['power_rating'].max() + 3)  # Adjust y-axis limits for visibility
    plt.grid(False)  # Turn off the grid
    elite_team_pr = all_data['power_rating'].mean() + (2*all_data['power_rating'].std())
    elite_team_resume = all_data['most_deserving_wins'].mean() + (2*all_data['most_deserving_wins'].std())
    good_team_pr = all_data['power_rating'].mean() + (1*all_data['power_rating'].std())
    good_team_resume = all_data['most_deserving_wins'].mean() + (1*all_data['most_deserving_wins'].std())
    avg_team_pr = all_data['power_rating'].mean() + (0*all_data['power_rating'].std())
    avg_team_resume = all_data['most_deserving_wins'].mean() + (0*all_data['most_deserving_wins'].std())
    below_avg_team_pr = all_data['power_rating'].mean() + (-1*all_data['power_rating'].std())
    below_avg_team_resume = all_data['most_deserving_wins'].mean() + (-1*all_data['most_deserving_wins'].std())
    # Get the data ranges for normalization
    x_min, x_max = all_data['most_deserving_wins'].min()-1, all_data['most_deserving_wins'].max()+1
    y_min, y_max = all_data['power_rating'].min()-3, all_data['power_rating'].max()+3
    plt.plot([elite_team_resume, x_max], [elite_team_pr, elite_team_pr], linestyle='--', color='darkgreen', alpha=0.6)  # Horizontal line
    plt.plot([elite_team_resume, elite_team_resume], [elite_team_pr, y_max], linestyle='--', color='darkgreen', alpha=0.6)  # Vertical line
    plt.plot([good_team_resume, x_max], [good_team_pr, good_team_pr], linestyle='--', color='yellow', alpha=0.6)  # Horizontal line
    plt.plot([good_team_resume, good_team_resume], [good_team_pr, y_max], linestyle='--', color='yellow', alpha=0.6)  # Vertical line
    # plt.plot([avg_team_resume, x_max], [avg_team_pr, avg_team_pr], linestyle='--', color='black', alpha=0.6)  # Horizontal line
    # plt.plot([avg_team_resume, avg_team_resume], [avg_team_pr, y_max], linestyle='--', color='black', alpha=0.6)  # Vertical line
    # plt.plot([below_avg_team_resume, x_max], [below_avg_team_pr, below_avg_team_pr], linestyle='--', color='red', alpha=0.6)  # Horizontal line
    # plt.plot([below_avg_team_resume, below_avg_team_resume], [below_avg_team_pr, y_max], linestyle='--', color='red', alpha=0.6)  # Vertical line
    plt.tight_layout()
    file_path = os.path.join(folder_path, "resume_vs_ratings")
    plt.savefig(file_path, dpi = 300)
    print("Resume Vs Ratings Done!")
except Exception as e:
    print(f"Error occurred while drawing Resume Vs Ratings: {e}")

try:
    file_path = "power_rating_playoff"
    draw_playoff_bracket_new(all_data, 'power_rating', team_logos, 'Current College Football Playoff Bracket via PEAR Power Rating', True, folder_path, file_path)
    print("Power Rating Playoff Done!")
except Exception as e:
    print(f"Error occurred while drawing Power Rating Playoff: {e}")

try:
    file_path = "most_deserving_playoff"
    draw_playoff_bracket_new(all_data, 'most_deserving_wins', team_logos, 'Current College Football Playoff Bracket via PEAR Most Deserving', True, folder_path, file_path)
    print("Most Deserving Playoff Done!")
except Exception as e:
    print(f"Error occurred while drawing Most Deserving Playoff: {e}")

try:
    file_path = "composite_playoff"
    draw_playoff_bracket_new(all_data, 'composite_score', team_logos, 'Current College Football Playoff Bracket via PEAR Composite Score', True, folder_path, file_path)
    print("Composite Playoff Done!")
except Exception as e:
    print(f"Error occurred while drawing Composite Playoff: {e}")

try:
    fig, ax = plt.subplots(figsize=(15, 9),dpi=125)
    plt.gca().set_facecolor('#CECEB2')
    plt.gcf().set_facecolor('#CECEB2')
    # Set the size of the logos (adjust the numbers to make logos smaller or larger)
    logo_size = 2  # Half the size of the logo to create spacing
    # Loop through the team_data DataFrame to plot logos
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    all_data['most_deserving_scaled'] = scaler100.fit_transform(all_data[['most_deserving_wins']])
    for i in range(len(all_data)):
        # Get the logo image from the URL
        img = team_logos[all_data.loc[i,'team']]

        # Calculate the extent for the logo
        # Here we use logo_size for both sides to center the logo at the specific coordinates
        ax.imshow(img, aspect='auto', 
                extent=(all_data['avg_talent'].iloc[i] - (logo_size-0.5),
                        all_data['avg_talent'].iloc[i] + (logo_size-0.5),
                        all_data['most_deserving_scaled'].iloc[i] - logo_size,
                        all_data['most_deserving_scaled'].iloc[i] + logo_size))

    ax.plot(
        [all_data['avg_talent'].min()-3, all_data['avg_talent'].max()+3],
        [all_data['avg_talent'].min()-3, all_data['avg_talent'].max()+3],
        color='black', linestyle='--'
    )

    # Set axis labels
    ax.set_xlabel('Talent Percentile', fontweight='bold')
    ax.set_ylabel('Resume Percentile', fontweight='bold')
    ax.set_title('Production vs. Talent', fontweight='bold', fontsize=14)

    # Show the plot
    plt.xlim(all_data['avg_talent'].min() - 3, all_data['avg_talent'].max() + 3)  # Adjust x-axis limits for visibility
    plt.ylim(all_data['most_deserving_scaled'].min() - 3, all_data['most_deserving_scaled'].max() + 3)  # Adjust y-axis limits for visibility
    plt.grid(False)  # Turn off the grid

    plt.text(1, 98, "Production Outperforming Talent", fontsize=10, fontweight='bold', ha='left')
    plt.text(100, 2, "Production Underperforming Talent", fontsize=10, fontweight='bold', ha='right')

    plt.tight_layout()
    file_path = os.path.join(folder_path, "production_vs_talent")
    plt.savefig(file_path, dpi = 300)
    print("Production vs Talent Done!")
except Exception as e:
    print(f"Error occurred while drawing Production Vs Talent: {e}")

try:
    # img = team_logos[this_conference_wins.loc[i, 'team']]
    start_week = current_week
    end_week = 17
    games_list = []
    for week in range(start_week,end_week):
        response = games_api.get_games(year=2025, week=week,classification = 'fbs')
        games_list = [*games_list, *response]
    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_elo=g.away_pregame_elo,
                home_points = g.home_points,
                away_points = g.away_points,
                neutral = g.neutral_site,
                conference_game = g.conference_game
                ) for g in games_list]
    uncompleted_games = pd.DataFrame(games)
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='home_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'home_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='away_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'away_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    missing_rating =round(team_data['power_rating'].mean() - 2.25*team_data['power_rating'].std(),2)
    uncompleted_games['home_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['away_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['PEAR_win_prob'] = uncompleted_games.apply(
        lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
    )
    uncompleted_games['home_win_prob'] = round((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) / ((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) + 1)*100,2)

    uncompleted_games['pr_spread'] = (GLOBAL_HFA + uncompleted_games['home_pr'] + (uncompleted_games['home_win_prob'].apply(adjust_home_pr)) - uncompleted_games['away_pr']).round(1)
    uncompleted_games['pr_spread'] = np.where(uncompleted_games['neutral'], uncompleted_games['pr_spread'] - GLOBAL_HFA, uncompleted_games['pr_spread']).round(1)
    uncompleted_games['PEAR'] = uncompleted_games.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)
    uncompleted_conference_games = uncompleted_games[uncompleted_games['conference_game'] == True].reset_index(drop=True)
    projection_dataframe = create_conference_projection(all_data, uncompleted_conference_games)
    conference_standings(projection_dataframe, records, team_data, team_logos, conf_folder_path)
    print("Conference Projections Done!")
except Exception as e:
    print(f"Error occurred while drawing Conference Projections: {e}")

start_week = 1
end_week = 17
games_list = []
for week in range(start_week,end_week):
    response = games_api.get_games(year=current_year, week=week,classification = 'fbs')
    games_list = [*games_list, *response]
if postseason:
    response = games_api.get_games(year=current_year, division = 'fbs', season_type='postseason')
    games_list = [*games_list, *response]
games = [dict(
            id=g.id,
            season=g.season,
            week=g.week,
            start_date=g.start_date,
            home_team=g.home_team,
            home_elo=g.home_pregame_elo,
            away_team=g.away_team,
            away_elo=g.away_pregame_elo,
            home_points = g.home_points,
            away_points = g.away_points,
            neutral = g.neutral_site
            ) for g in games_list if g.home_pregame_elo is not None and g.away_pregame_elo is not None]
year_long_schedule = pd.DataFrame(games)
year_long_schedule = year_long_schedule.merge(team_data[['team', 'power_rating']], 
                                    left_on='home_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'home_pr'})
year_long_schedule = year_long_schedule.drop(columns=['team'])
year_long_schedule = year_long_schedule.merge(team_data[['team', 'power_rating']], 
                                    left_on='away_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'away_pr'})
year_long_schedule = year_long_schedule.drop(columns=['team'])
fallback_value = team_data['power_rating'].mean() - 2 * team_data['power_rating'].std()
year_long_schedule['home_pr'] = year_long_schedule['home_pr'].fillna(fallback_value)
year_long_schedule['away_pr'] = year_long_schedule['away_pr'].fillna(fallback_value)

year_long_schedule['PEAR_win_prob'] = year_long_schedule.apply(
    lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
)
year_long_schedule['home_win_prob'] = round((10**((year_long_schedule['home_elo'] - year_long_schedule['away_elo']) / 400)) / ((10**((year_long_schedule['home_elo'] - year_long_schedule['away_elo']) / 400)) + 1)*100,2)

try:
    start_week = current_week
    end_week = 17
    games_list = []
    for week in range(start_week,end_week):
        response = games_api.get_games(year=2025, week=week,classification = 'fbs')
        games_list = [*games_list, *response]
    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_elo=g.away_pregame_elo,
                home_points = g.home_points,
                away_points = g.away_points,
                neutral = g.neutral_site,
                conference_game = g.conference_game
                ) for g in games_list]
    uncompleted_games = pd.DataFrame(games)
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='home_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'home_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    uncompleted_games = uncompleted_games.merge(team_data[['team', 'power_rating']], 
                                        left_on='away_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'away_pr'})
    uncompleted_games = uncompleted_games.drop(columns=['team'])
    missing_rating =round(team_data['power_rating'].mean() - 2.25*team_data['power_rating'].std(),2)
    uncompleted_games['home_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['away_pr'].fillna(missing_rating, inplace=True)
    uncompleted_games['PEAR_win_prob'] = uncompleted_games.apply(
        lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
    )
    uncompleted_games['home_win_prob'] = round((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) / ((10**((uncompleted_games['home_elo'] - uncompleted_games['away_elo']) / 400)) + 1)*100,2)

    uncompleted_games['pr_spread'] = (GLOBAL_HFA + uncompleted_games['home_pr'] + (uncompleted_games['home_win_prob'].apply(adjust_home_pr)) - uncompleted_games['away_pr']).round(1)
    uncompleted_games['pr_spread'] = np.where(uncompleted_games['neutral'], uncompleted_games['pr_spread'] - GLOBAL_HFA, uncompleted_games['pr_spread']).round(1)
    uncompleted_games['PEAR'] = uncompleted_games.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)
    uncompleted_conference_games = uncompleted_games[uncompleted_games['conference_game'] == True].reset_index(drop=True)
    start_week = 1
    end_week = 17
    games_list = []
    for week in range(start_week,end_week):
        response = games_api.get_games(year=current_year, week=week,classification = 'fbs')
        games_list = [*games_list, *response]
    if postseason:
        response = games_api.get_games(year=current_year, division = 'fbs', season_type='postseason')
        games_list = [*games_list, *response]
    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_elo=g.away_pregame_elo,
                home_points = g.home_points,
                away_points = g.away_points,
                neutral = g.neutral_site,
                conference_game = g.conference_game
                ) for g in games_list]
    full_display_schedule = pd.DataFrame(games)
    full_display_schedule = full_display_schedule.merge(team_data[['team', 'power_rating']], 
                                        left_on='home_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'home_pr'})
    full_display_schedule = full_display_schedule.drop(columns=['team'])
    full_display_schedule = full_display_schedule.merge(team_data[['team', 'power_rating']], 
                                        left_on='away_team', 
                                        right_on='team', 
                                        how='left').rename(columns={'power_rating': 'away_pr'})
    full_display_schedule = full_display_schedule.drop(columns=['team'])
    fallback_value = team_data['power_rating'].mean() - 2 * team_data['power_rating'].std()
    full_display_schedule['home_pr'] = full_display_schedule['home_pr'].fillna(fallback_value)
    full_display_schedule['away_pr'] = full_display_schedule['away_pr'].fillna(fallback_value)

    full_display_schedule['PEAR_win_prob'] = full_display_schedule.apply(
        lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
    )
    full_display_schedule['home_win_prob'] = (
        10 ** ((full_display_schedule['home_elo'] - full_display_schedule['away_elo']) / 400)
        / (10 ** ((full_display_schedule['home_elo'] - full_display_schedule['away_elo']) / 400) + 1)
        * 100
    ).round(2)

    # fill missing with rule
    full_display_schedule['home_win_prob'] = full_display_schedule.apply(
        lambda row: 1 if pd.isna(row['home_win_prob']) and row['home_team'] in team_data['team'].values
        else 0 if pd.isna(row['home_win_prob'])
        else row['home_win_prob'],
        axis=1
    )

    full_display_schedule['pr_spread'] = (GLOBAL_HFA + full_display_schedule['home_pr'] + (full_display_schedule['home_win_prob'].apply(adjust_home_pr)) - full_display_schedule['away_pr']).round(1)
    full_display_schedule['pr_spread'] = np.where(full_display_schedule['neutral'], full_display_schedule['pr_spread'] - GLOBAL_HFA, full_display_schedule['pr_spread']).round(1)
    full_display_schedule['PEAR'] = full_display_schedule.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)

    pr_min = team_data['power_rating'].min()
    pr_max = team_data['power_rating'].max()

    # Calculate game quality
    full_display_schedule['GQI'] = _calculate_game_quality(
        full_display_schedule,
        pr_min=pr_min,
        pr_max=pr_max,
        spread_cap=30
    )
    logos_info_list = []
    response = teams_api.get_teams()
    logos_info_list = [*logos_info_list, *response]
    logos_info_dict = [dict(
        team = l.school,
        color = l.color,
        alt_color = l.alternate_color,
        logo = l.logos,
        classification = l.classification
    ) for l in logos_info_list]
    logos = pd.DataFrame(logos_info_dict)
    fbs_fcs = logos[(logos['classification'] == 'fbs') | (logos['classification'] == 'fcs')].reset_index(drop=True)

    logo_folder = "./PEAR/PEAR Football/logos/"
    fbs_fcs_logos = {}
    for filename in os.listdir(logo_folder):
        if filename.endswith(".png"):
            team_name = filename[:-4].replace("_", " ")
            file_path = os.path.join(logo_folder, filename)
            try:
                img = Image.open(file_path).convert("RGBA")
                fbs_fcs_logos[team_name] = img
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                fbs_fcs_logos[team_name] = None

    all_data['at_large_wins'] = np.ceil(all_data['avg_expected_wins']-0.5).astype(int)
    all_data['wins_needed'] = all_data['at_large_wins'] - all_data['wins']
    all_data['prob_reach_wins'] = all_data.apply(
        lambda row: 100*prob_win_at_least_x(row['team'], row['wins_needed'], uncompleted_games).round(2),
        axis=1
    )
    all_data['playoff_rank'] = (
        all_data[['prob_reach_wins', 'power_rating']]
        .apply(tuple, axis=1)                       # make row-wise tuples
        .rank(method='min', ascending=False)        # rank higher = better
        .astype(int)
    )

    average_elo = elo_ratings['elo'].mean()
    average_pr = round(team_data['power_rating'].mean(), 2)
    good_team_pr = round(team_data['power_rating'].std() + team_data['power_rating'].mean(),2)
    elite_team_pr = round(2*team_data['power_rating'].std() + team_data['power_rating'].mean(),2)
    expected_wins_list = []
    conference_list = []
    for team in team_data['team']:
        this_conference = team_data[team_data['team'] == team]['conference'].values[0]
        schedule = full_display_schedule[
            (
                (full_display_schedule['home_team'] == team) | 
                (full_display_schedule['away_team'] == team)
            ) &
            (full_display_schedule['conference_game'] == True)
        ]
        df = average_team_distribution(1000, schedule, elite_team_pr, team)
        expected_wins = df['expected_wins'].values[0]
        expected_wins_list.append(expected_wins)
        conference_list.append(this_conference)
    cSOS = pd.DataFrame(zip(team_data['team'], expected_wins_list, conference_list), columns=['team', 'avg_expected_wins', 'conference'])
    cSOS = cSOS.sort_values('avg_expected_wins').reset_index(drop = True)
    cSOS['cSOS'] = cSOS.groupby('conference')['avg_expected_wins'] \
                    .rank(method='min', ascending=True).astype(int)

    i = 1
    for team in all_data['team']:
        if i % 10 == 0:
            print(i)
        display_schedule_visual(team, all_data, full_display_schedule, uncompleted_games, uncompleted_conference_games, cSOS, fbs_fcs_logos, fbs_fcs, current_year, current_week)
        i = i+1
    print("Stat Profiles Done!")
except Exception as e:
    print(f"Error occurred while drawing Stat Profiles: {e}")

try:

    media_list = games_api.get_media(year=current_year, week=current_week)
    media_dict = [dict(
                    id=g.id,
                    outlet=g.outlet
                    ) for g in media_list]
    media_info = pd.DataFrame(media_dict)
    outlet_priority = {"ABC": 1, "ESPN": 2, "FOX": 3, "NBC": 4}
    media_info_clean = (
        media_info
        .assign(priority=media_info["outlet"].map(outlet_priority).fillna(99))
        .sort_values(["id", "priority"])
        .drop_duplicates(subset=["id"], keep="first")
        .drop(columns="priority")
    )

    if postseason:
        games = []
        response = games_api.get_games(year=current_year, classification = 'fbs', season_type='postseason')
        games = [*games, *response]
    else:
        games = []
        response = games_api.get_games(year=current_year, week = current_week, classification = 'fbs')
        games = [*games, *response]

    games = [dict(
                id=g.id,
                season=g.season,
                week=g.week,
                start_date=g.start_date,
                home_team=g.home_team,
                home_conference=g.home_conference,
                home_points=g.home_points,
                home_elo=g.home_pregame_elo,
                away_team=g.away_team,
                away_conference=g.away_conference,
                away_points=g.away_points,
                away_elo=g.away_pregame_elo,
                neutral = g.neutral_site
                ) for g in games if g.home_pregame_elo is not None and g.away_pregame_elo is not None]
    week_games = pd.DataFrame(games)
    week_games["start_time"] = (
        pd.to_datetime(week_games["start_date"], utc=True)   # parse strings (handles offsets like +00:00/-05:00)
        .dt.tz_convert("America/Chicago")                 # convert to Central (handles DST)
        .dt.strftime("%A %I:%M %p")                       # e.g. "Friday 08:00 PM"
        .str.replace(r'(?<=\s)0', '', regex=True)         # remove leading zero -> "Friday 8:00 PM"
    )
    week_games = pd.merge(week_games, media_info_clean, on="id", how="left")
    week_games['time_outlet'] = week_games['start_time'] + ' - ' + week_games['outlet']

    for i, game in week_games.iterrows():
        away_team = game['away_team'].strip()
        home_team = game['home_team'].strip()
        neutral = game['neutral']
        time_outlet = game['time_outlet']
        print(f"{home_team} vs. {away_team} - {i+1}/{len(week_games)}")
        plot_matchup_new(all_data, team_logos, away_team, home_team, neutral, current_year, current_week, True, time_outlet)
    print("Matchup Visuals Done!")
except Exception as e:
    print(f"Error occurred while drawing Matchup Visuals: {e}")

print("---------- Visuals Done! ----------")

from football_helper import analyze_vegas_predictions, _calculate_pr_prediction
import matplotlib.pyplot as plt
week_games, predictions, fig = analyze_vegas_predictions(
    current_week=current_week,
    current_year=current_year,
    postseason=postseason,
    save=True
)

from concurrent.futures import ThreadPoolExecutor

logo_folder = "./PEAR/PEAR Football/logos/"
logo_cache = {}
def load_image(filename):
    team_name = filename[:-4].replace("_", " ")
    file_path = os.path.join(logo_folder, filename)
    try:
        img = Image.open(file_path).convert("RGBA")
        return (team_name, img)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return (team_name, None)

png_files = [f for f in os.listdir(logo_folder) if f.endswith(".png")]

with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(load_image, png_files)
    logo_cache = dict(results)

def PEAR_Win_Prob(home_power_rating, away_power_rating, neutral):
    if neutral == False:
        home_power_rating = home_power_rating + 1.5
    return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

visual = week_games[['week', 'start_date', 'home_team', 'away_team', 'home_pr', 'away_pr', 'neutral', 'PEAR', 'GQI']].dropna()
visual['start_date'] = pd.to_datetime(visual['start_date'], utc=True)
visual['start_date'] = visual['start_date'].dt.tz_convert('US/Central')
time_thresholds = [
    datetime.time(14,30),  # morning -> afternoon
    datetime.time(18,0),   # afternoon -> evening
    datetime.time(20,0)    # evening -> night
]
def get_time_class(t):
    if t < time_thresholds[0]:
        return 0  # Morning
    elif t < time_thresholds[1]:
        return 1  # Afternoon
    elif t < time_thresholds[2]:
        return 2  # Evening
    else:
        return 3  # Night
visual['start_time'] = visual['start_date'].dt.tz_convert('US/Central').dt.time
visual['time_class'] = visual['start_time'].apply(get_time_class)
visual['start_date'] = visual['start_date'].dt.date
visual = visual.sort_values(['start_date', 'time_class', 'GQI'], ascending=[True, True, False]).reset_index(drop=True)

visual['PEAR_win_prob'] = round(100*(visual.apply(
    lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr'], row['neutral'])/100, axis=1
)),1)

save_dir = f"PEAR/PEAR Football/y{current_year}/Visuals/Schedule"
os.makedirs(save_dir, exist_ok=True)  # create the folder if it doesn't exist

from matplotlib.lines import Line2D

def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

for date, group in visual.groupby(visual['start_date']):
    if len(group) <= 4:
        max_cols = 2
    elif len(group) <= 9:
        max_cols = 3
    elif len(group) <= 16:
        max_cols = 4
    elif len(group) <= 25:
        max_cols = 5
    else:
        max_cols = 6
    group = group.sort_values('start_time').reset_index(drop=True)
    group['time_class'] = group['start_time'].apply(get_time_class)
    group = group.sort_values(['time_class', 'GQI'], ascending=[True, False]).reset_index(drop=True)

    # First pass: compute rows needed
    rows_needed = 0
    current_class = None
    col = 0
    for _, row in group.iterrows():
        if row['time_class'] != current_class:
            rows_needed += 1
            col = 0
            current_class = row['time_class']
        elif col == max_cols:
            rows_needed += 1
            col = 0
        col += 1

    fig, axes = plt.subplots(rows_needed, max_cols, figsize=(max_cols*4.6, rows_needed*3.6), dpi=250)
    fig.patch.set_facecolor("#CECEB2")
    axes = axes.flatten()

    # Keep track of row breaks caused by time_class changes
    time_class_row_breaks = []

    # Plotting with i,j logic
    i = 0  # current row
    j = 0  # current column
    current_class = None
    row_start_idx = 0  # index of first axis in the current row

    for idx, row in group.iterrows():
        if row['time_class'] != current_class:
            # Only mark breaks when time_class changes
            if current_class is not None:
                # Save previous row for horizontal line
                time_class_row_breaks.append((row_start_idx, row_start_idx + j - 1))
                # Hide remaining unused axes in previous row
                for k in range(row_start_idx + j, row_start_idx + max_cols):
                    if k < len(axes):
                        fig.delaxes(axes[k])

            # Start new row
            i += 1 if current_class is not None else 0
            j = 0
            row_start_idx = i * max_cols
            current_class = row['time_class']

        elif j == max_cols:
            # Row wrap due to max_cols: just start new row without adding a time_class break
            i += 1
            j = 0
            row_start_idx = i * max_cols

        ax_idx = i * max_cols + j
        ax = axes[ax_idx]
        j += 1

        # Plot content
        ax.set_facecolor("#CECEB2")
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.barh([0], [row['PEAR']], color="skyblue", alpha=0)

        img = logo_cache[row['away_team']]
        imagebox = OffsetImage(img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (0.03, 0.3), frameon=False)
        ax.add_artist(ab)

        img = logo_cache[row['home_team']]
        imagebox = OffsetImage(img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (-0.03, 0.3), frameon=False)
        ax.add_artist(ab)

        if rows_needed < 4:
            # Your original logic
            fontsize_PE = 24 if 'Washington State' in row['PEAR'] else 28
            ax.text(0, -0.1, f'{row["PEAR"]}', ha='center', va='center',
                    fontsize=fontsize_PE, fontweight='bold')

            ax.text(-0.03, -0.25, f'{row["PEAR_win_prob"]}%', ha='center', va='center',
                    fontsize=24)
            ax.text(0.03, -0.25, f'{round(100-row["PEAR_win_prob"],1)}%', ha='center', va='center',
                    fontsize=24)
            ax.text(0, -0.4, f'GQI: {row["GQI"]}', ha='center', va='center',
                    fontsize=24)
        else:
            # Larger & bold for everything
            fontsize_PE = 24 if 'Washington State' in row['PEAR'] else 28
            ax.text(0, -0.1, f'{row["PEAR"]}', ha='center', va='center',
                    fontsize=fontsize_PE, fontweight='bold')

            ax.text(-0.03, -0.25, f'{row["PEAR_win_prob"]}%', ha='center', va='center',
                    fontsize=28, fontweight='bold')
            ax.text(0.03, -0.25, f'{round(100-row["PEAR_win_prob"],1)}%', ha='center', va='center',
                    fontsize=28, fontweight='bold')
            ax.text(0, -0.4, f'GQI: {row["GQI"]}', ha='center', va='center',
                    fontsize=28, fontweight='bold')

        ax.set_yticks([])
        ax.set_xticks([])

    # Hide unused axes in the last row
    for k in range(row_start_idx + j, len(axes)):
        fig.delaxes(axes[k])

    # Finalize layout first
    if max_cols == 2:
        if rows_needed == 1:
            height = 0.75
        elif rows_needed == 2:
            height = 0.85
        else:
            height = 0.9
    elif max_cols == 3:
        height = 0.9
    else:
        height = 0.95
    plt.tight_layout(rect=[0, 0, 1, height])  # leave top 10% for suptitle

    # Draw horizontal lines only for time_class changes
    for start_idx, end_idx in time_class_row_breaks:
        row_axes_positions = [axes[k].get_position().y0 for k in range(start_idx, end_idx + 1)]
        if row_axes_positions:
            if max_cols > 3:
                y_bottom = min(row_axes_positions) - 0.01  # slightly below the row
            else:
                y_bottom = min(row_axes_positions) - 0.02  # slightly below the row
            line = Line2D([0, 1], [y_bottom, y_bottom], transform=fig.transFigure, color='black', linewidth=2, linestyle='--')
            fig.add_artist(line)

    day_str = ordinal(date.day)
    date_str = date.strftime(f"%A, %B {day_str}")
    fig.suptitle(f"{date_str} Games\n@PEARatings", fontsize=32, fontweight='bold')
    filename = f"schedule_{date.strftime('%m_%d_%Y')}.png"
    fig_path = os.path.join(save_dir, filename)
    fig.savefig(fig_path, facecolor=fig.get_facecolor())
    plt.close(fig)

print("---------- Spreads Done! ----------")