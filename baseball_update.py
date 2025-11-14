import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import warnings
import os
import textwrap
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import matplotlib.offsetbox as offsetbox # type: ignore
import matplotlib.font_manager as fm # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from collections import Counter, defaultdict
from plottable import Table # type: ignore
from plottable.plots import image, circled_image # type: ignore
from plottable import ColumnDefinition # type: ignore
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import random

warnings.filterwarnings("ignore")

# --- Timezone Setup ---
cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year

# --- Team Name Replacements ---
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
    'UMass': 'Massachusetts',
    'Loyola-Marymount': 'LMU (CA)'
}

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
print("Elo Load Done")

file_paths = [
    "PEAR/PEAR Baseball/y2021/schedule_2021.csv",
    "PEAR/PEAR Baseball/y2022/schedule_2022.csv",
    "PEAR/PEAR Baseball/y2023/schedule_2023.csv",
    "PEAR/PEAR Baseball/y2024/schedule_2024.csv",
    "PEAR/PEAR Baseball/y2025/schedule_2025.csv",
]
dataframes = []
for path in file_paths:
    df = pd.read_csv(path)
    year = int(path.split("schedule_")[1].split(".csv")[0])
    df["year"] = year
    dataframes.append(df)
games = pd.concat(dataframes, ignore_index=True)[['Team', 'Date', 'home_team', 'away_team', 'home_score', 'away_score', 'Result', 'Location']]
games = games[games["home_score"] != games["away_score"]].copy()
games = games[games['Location'] != 'Neutral'].copy()
games["total_runs"] = games["home_score"] + games["away_score"]
home_runs = games.groupby("home_team")["total_runs"].mean().rename("home_runs_per_game")
away_games = games[["home_team", "away_team", "home_score", "away_score", "total_runs"]].copy()
away_games = away_games.rename(columns={"home_team": "opponent", "away_team": "team"})
away_runs = away_games.groupby("team")["total_runs"].mean().rename("away_runs_per_game")
park_factors = pd.concat([home_runs, away_runs], axis=1)
park_factors["park_factor"] = park_factors["home_runs_per_game"] / park_factors["away_runs_per_game"]
park_factors = park_factors.sort_values("park_factor", ascending=False)
park_factors = park_factors.reset_index(names='Team')
park_factors = park_factors[~park_factors['Team'].str.contains('Non Div', na=False)].reset_index(drop=True)
pf_lookup = dict(zip(park_factors['Team'], park_factors['park_factor']))

####################### Schedule Load #######################

from baseball_helper import fetch_all_schedules, convert_date, clean_team_names, get_soup, scrape_warrennolan_table, PEAR_Win_Prob

BASE_URL = "https://www.warrennolan.com"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})
schedule_data = fetch_all_schedules(elo_data, session, max_workers=12)

columns = ["Team", "Date", "Opponent", "Location", "Result", "home_team", "away_team", "home_score", "away_score"]
schedule_df = pd.DataFrame(schedule_data, columns=columns)
schedule_df = schedule_df.astype({col: 'str' for col in schedule_df.columns if col not in ['home_score', 'away_score']})
schedule_df['home_score'] = schedule_df['home_score'].astype(int, errors='ignore')
schedule_df['away_score'] = schedule_df['away_score'].astype(int, errors='ignore')
schedule_df = schedule_df.merge(elo_data[['Team', 'ELO']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'ELO': 'home_elo'}, inplace=True)
schedule_df = schedule_df.merge(elo_data[['Team', 'ELO']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'ELO': 'away_elo'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)
schedule_df = schedule_df[~(schedule_df['Result'] == 'Canceled')].reset_index(drop=True)
schedule_df = schedule_df[~(schedule_df['Result'] == 'Postponed')].reset_index(drop=True)

# Apply replacements and standardize 'State' to 'St.'
columns_to_replace = ['Team', 'home_team', 'away_team', 'Opponent']

for col in columns_to_replace:
    schedule_df[col] = schedule_df[col].str.replace('State', 'St.', regex=False)
    schedule_df[col] = schedule_df[col].replace(team_replacements)
elo_data['Team'] = elo_data['Team'].str.replace('State', 'St.', regex=False)
elo_data['Team'] = elo_data['Team'].replace(team_replacements)

print("Schedule Load Done")

# Apply function to convert date format
schedule_df["Date"] = schedule_df["Date"].astype(str).apply(convert_date)
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], format="%m-%d-%Y")
comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")

offensive_whip = schedule_df[
    (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
].reset_index(drop=True)

# --- Creating Folder Path ---
folder_path = f"./PEAR/PEAR Baseball/y{current_season}"
os.makedirs(folder_path, exist_ok=True)

# --- NCAA Stats Dropdown ---
base_url = "https://www.ncaa.com"
soup = get_soup(f"{base_url}/stats/baseball/d1")
dropdown = soup.find("select", {"id": "select-container-team"})
stat_links = {
    option.text.strip(): base_url + option["value"]
    for option in dropdown.find_all("option") if option.get("value")
}

# --- NCAA RPI Table ---
rpi_url = "https://www.ncaa.com/rankings/baseball/d1/rpi"
rpi_soup = get_soup(rpi_url)
table = rpi_soup.find("table", class_="sticky")

if table:
    headers = [th.text.strip() for th in table.find_all("th")]
    data = [
        [td.text.strip() for td in row.find_all("td")]
        for row in table.find_all("tr")[1:]
    ]
    rpi = pd.DataFrame(data, columns=headers).drop(columns=["Previous"], errors='ignore')
    rpi.rename(columns={"School": "Team"}, inplace=True)
else:
    print("NCAA RPI Table not found.")
    rpi = pd.DataFrame()

# --- Projected RPI ---
projected_rpi = scrape_warrennolan_table(
    'https://www.warrennolan.com/baseball/2025/rpi-predict',
    expected_columns=["RPI", "Team", "Conference"]
)

# --- Live RPI ---
live_rpi = scrape_warrennolan_table(
    'https://www.warrennolan.com/baseball/2025/rpi-live',
    expected_columns=["Live_RPI", "Team", "Conference"]
)

# --- ELO Ratings ---
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
    elo_data.rename(columns={'Rank': 'ELO_Rank'}, inplace=True)

else:
    print("Table not found on the page.")

# Apply team name cleanup
elo_data = clean_team_names(elo_data, team_replacements)
projected_rpi = clean_team_names(projected_rpi, team_replacements)
live_rpi = clean_team_names(live_rpi, team_replacements)

####################### Transform Config #######################

STAT_TRANSFORMS = {
    "Batting Average": lambda df: df.assign(
        HPG=df["H"] / df["G"],
        ABPG=df["AB"] / df["G"],
        HPAB=df["H"] / df["AB"]
    ).drop(columns=['Rank']),

    "Base on Balls": lambda df: df.assign(
        BBPG=df["BB"] / df["G"]
    ).drop(columns=['Rank', 'G']),

    "Earned Run Average": lambda df: df.rename(columns={"R": "RA"}).drop(columns=['Rank', 'G']),

    "Fielding Percentage": lambda df: df.assign(
        APG=df["A"] / df["G"],
        EPG=df["E"] / df["G"]
    ).drop(columns=['Rank', 'G']),

    "On Base Percentage": lambda df: df.rename(columns={"PCT": "OBP"}).assign(
        HBPPG=df["HBP"] / df["G"]
    ).drop(columns=['Rank', 'G', 'AB', 'H', 'BB', 'SF', 'SH']),

    "Runs": lambda df: df.assign(
        RPG=df["R"] / df["G"]
    ).rename(columns={"R": "RS"}).drop(columns=['Rank', 'G']),

    "Slugging Percentage": lambda df: df.rename(columns={"SLG PCT": "SLG"}).drop(columns=['Rank', 'G', 'AB']),

    "Strikeouts Per Nine Innings": lambda df: df.rename(columns={"K/9": "KP9"}).drop(columns=['Rank', 'G', 'IP', 'SO']),

    "Walks Allowed Per Nine Innings": lambda df: df.rename(columns={"PG": "WP9"}).drop(columns=['Rank', 'G', 'IP', 'BB']),

    "WHIP": lambda df: df.drop(columns=['Rank', 'HA', 'IP', 'BB']),
}

####################### Run It #######################

from baseball_helper import threaded_stat_fetch, clean_and_merge

# Example stat pull
stat_list = list(STAT_TRANSFORMS.keys())
raw_stats = threaded_stat_fetch(stat_list, stat_links, max_workers=10)
baseball_stats = clean_and_merge(raw_stats, STAT_TRANSFORMS)
baseball_stats = pd.merge(baseball_stats, elo_data, on='Team', how='left')

missing_teams = set(elo_data['Team']) - set(baseball_stats['Team'])
elo_data_sorted = elo_data.sort_values(by='ELO_Rank')
elo_data_sorted['Percentile'] = 100 - (elo_data_sorted['ELO_Rank'].rank() - 1) / (elo_data_sorted.shape[0] - 1) * 100
matched_percentiles = elo_data_sorted[elo_data_sorted['Team'].isin(missing_teams)][['Team', 'Percentile']]
percentile_dict = dict(zip(matched_percentiles['Team'], matched_percentiles['Percentile']))

####################### wOBA Stat Transforms #######################

STAT_TRANSFORMS_WOBA = {
    "Base on Balls": lambda df: df.assign(
        BBPG=df["BB"] / df["G"]
    )[["Team", "BB", "G", "BBPG"]],
    
    "Hit by Pitch": lambda df: df[["Team", "HBP"]],
    
    "Hits": lambda df: df[["Team", "AB", "H"]],
    
    "Doubles": lambda df: df[["Team", "2B"]],
    
    "Triples": lambda df: df[["Team", "3B"]],
    
    "Home Runs Per Game": lambda df: (
        lambda _df: pd.concat([
            _df[~_df["Team"].isin(
                _df[_df.duplicated("Team", keep=False)]["Team"]
            )],
            _df[_df.duplicated("Team", keep=False)].groupby("Team", as_index=False).apply(
                lambda g: g.loc[g["HR"].idxmin()]
            )
        ], ignore_index=True)
    )(df.rename(columns={"PG": "HRPG"}).drop(columns=["Rank", "G"])),
    
    "Sacrifice Flies": lambda df: df[["Team", "SF"]],
    
    "Runs": lambda df: df.assign(
        RPG=df["R"] / df["G"]
    ).rename(columns={"R": "RS"}).drop(columns=["Rank", "G"]),
    
    "Sacrifice Bunts": lambda df: df.rename(columns={"SH": "SB"}).assign(
        SBPG=lambda x: x["SB"] / x["G"]
    ).drop(columns=["Rank", "G"]),
    
    "Earned Run Average": lambda df: df.rename(columns={"R": "RA"}).drop(columns=["Rank", "G"]),
    
    "Strikeout-to-Walk Ratio": lambda df: df.rename(columns={"BB": "PBB"})[["Team", "K/BB", "PBB", "SO"]],
    
    "Hits Allowed Per Nine Innings": lambda df: df.rename(columns={"PG": "HAPG"})[["Team", "HA", "HAPG"]],
    
    "Hit Batters": lambda df: df[["Team", "HB"]],
}

####################### Fetch + Transform + Merge #######################

# Only pull stats we need
woba_stats = list(STAT_TRANSFORMS_WOBA.keys())

# Threaded fetch
raw_woba_stats = threaded_stat_fetch(woba_stats, stat_links)

# Apply transforms + merge
dfs = []
for stat, df in raw_woba_stats.items():
    if df is not None and stat in STAT_TRANSFORMS_WOBA:
        df["Team"] = df["Team"].str.strip()
        df = df.dropna(subset=["Team"])
        dfs.append(STAT_TRANSFORMS_WOBA[stat](df))

# Merge all together
wOBA = dfs[0]
for df in dfs[1:]:
    wOBA = pd.merge(wOBA, df, on="Team", how="left")

# Fill and compute final metrics
wOBA = wOBA.fillna(0)
wOBA["PA"] = wOBA["AB"] + wOBA["BB"] + wOBA["HBP"] + wOBA["SF"] + wOBA["SB"]
league_HR_per_game = wOBA["HR"].sum() / wOBA["G"].sum()
wOBA["HR_A"] = wOBA["G"] * league_HR_per_game
wOBA['1B'] = wOBA['H'] - wOBA['2B'] - wOBA['3B'] - wOBA['HR']
wOBA['wOBA'] = ((0.69 * wOBA['BB']) + (0.72 * wOBA['HBP']) + (0.88 * wOBA['1B']) + (1.24 * wOBA['2B']) + (1.56 * wOBA['3B']) + (1.95 * wOBA['HR'])) / (wOBA['PA'])
league_wOBA = (wOBA['wOBA'] * wOBA['PA']).sum() / wOBA['PA'].sum()
league_R_PA = wOBA['RS'].sum() / wOBA['PA'].sum()
wOBA_scale = league_R_PA / league_wOBA
wOBA['wRAA'] = ((wOBA['wOBA'] - league_wOBA) / wOBA_scale) * wOBA['PA']
league_RS = wOBA['RS'].sum()
league_G = wOBA['G'].sum()
RPW = 2 * (league_RS / league_G)
wOBA['oWAR'] = wOBA['wRAA'] / RPW
wOBA['ISO'] = (wOBA['2B'] + (2 * wOBA['3B']) + (3 * wOBA['HR'])) / wOBA['AB']
wOBA['wRC'] = (((wOBA['wOBA'] - league_wOBA) / wOBA_scale) + league_R_PA) * wOBA['PA']
wOBA['wRC+'] = (wOBA['wRC'] / wOBA['PA']) / league_R_PA * 100
wOBA['BB%'] = wOBA['BB'] / wOBA['PA']
wOBA['BABIP'] = (wOBA['H'] - wOBA['HR']) / (wOBA['AB'] + wOBA['SF'])
wOBA['RA9'] = (wOBA['RA'] / wOBA['IP']) * 9
wOBA['LOB%'] = (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - wOBA['RA']) / (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - (1.4*wOBA['HR_A']))
wOBA['FIP'] = ((13 * wOBA['HR_A'] + 3 * (wOBA['PBB'] + wOBA['HB']) - 2 * wOBA['SO']) / wOBA['IP'])
league_RA9 = wOBA['RA'].sum() / wOBA['G'].sum()
league_ERA = (wOBA['ER'].sum() * 9) / wOBA['IP'].sum()
league_FIP = (wOBA['FIP'] * wOBA['IP']).sum() / wOBA['IP'].sum()
replacement_level_FIP = wOBA['FIP'].quantile(0.80)
multiplier = replacement_level_FIP / league_FIP
replacement_RA9 = league_RA9 * multiplier  # Adjust RA9 to match replacement level
RPW = 9 / (2 * (wOBA['RA'].sum() / wOBA['G'].sum()))
wOBA['pWAR'] = ((replacement_RA9 - wOBA['FIP']) / RPW) * (wOBA['IP'] / 9)
mean_oWAR = wOBA['oWAR'].mean()
std_oWAR = wOBA['oWAR'].std()
mean_pWAR = wOBA['pWAR'].mean()
std_pWAR = wOBA['pWAR'].std()
wOBA['oWAR_z'] = (wOBA['oWAR'] - mean_oWAR) / std_oWAR
wOBA['pWAR_z'] = (wOBA['pWAR'] - mean_pWAR) / std_pWAR
wOBA['fWAR'] = wOBA['oWAR_z'] + wOBA['pWAR_z']
wOBA['Offensive_WHIP'] = (wOBA['H'] + wOBA['BB']) / ((wOBA['AB'] - wOBA['H']) / 3)
offensive_whip = pd.merge(offensive_whip,
    wOBA[['Team', 'Offensive_WHIP']],
    how='left',
    left_on='Opponent',
    right_on='Team'
).drop(columns=['Team_y']).rename(columns={'Team_x':'Team'})
avg_off_whip = offensive_whip.groupby('Team')['Offensive_WHIP'].mean().reset_index()
avg_off_whip.rename(columns={'Offensive_WHIP': 'Avg_Opp_Offensive_WHIP'}, inplace=True)
wOBA = wOBA.merge(avg_off_whip, how='left', on='Team')
baseball_stats = pd.merge(baseball_stats, wOBA[['Team', 'wOBA', 'wRAA', 'oWAR_z', 'pWAR_z', 'fWAR', 'ISO', 'wRC+', 'BB%', 'BABIP', 'RA9', 'FIP', 'LOB%', 'K/BB', 'Avg_Opp_Offensive_WHIP']], how='left', on='Team')
baseball_stats['WHIP+'] = 100 * (baseball_stats['WHIP'] / baseball_stats['Avg_Opp_Offensive_WHIP'])

"""
College Baseball Power Ratings System with Home Field Advantage Optimization

This system combines multiple approaches to create robust team ratings:
1. Feature Selection: Automatically selects best subset of features for each target
2. ML Models: Trains XGBoost/GradientBoosting models on selected features
3. Margin Model: Learns from game results to predict score margins
4. Ensemble: Optimally blends target-based and margin-based ratings
5. HFA Optimization: Searches for optimal home field advantage parameter

Key Features:
- Handles rank-based targets (like ELO_Rank where lower is better)
- Can optimize for multiple targets simultaneously
- Margin model can receive 0 weight if it doesn't improve target correlation
- Automatic feature selection using differential evolution
- Flexible scale based on standard deviation (not fixed range)
- Home field advantage as tunable hyperparameter

Rating Scale:
- Uses standard deviation-based scaling instead of fixed range
- Default: scale=5.0 means ~68% of teams within ±5 of center
- Allows natural expansion/compression based on team quality differences
- More flexible than forcing all datasets into same range
"""

from baseball_helper import build_baseball_power_ratings, calculate_spread_from_stats

net_2024 = pd.read_csv("./PEAR/PEAR Baseball/y2024/Data/baseball_06_25_2024.csv")[['Team', 'NET_Score']]

modeling_stats = baseball_stats[['Team', 'HPG',
                'BBPG', 'ERA', 'PCT', 
                'KP9', 'WP9', 'OPS', 'BB%', 'WHIP+',
                'WHIP', 'PYTHAG', 'fWAR', 'oWAR_z', 'pWAR_z', 'K/BB', 'wRC+', 'LOB%', 'wOBA', 'ELO_Rank']]
modeling_stats = pd.merge(modeling_stats, net_2024[['Team', 'NET_Score']], on = 'Team', how='left')
modeling_stats["ELO_Rank"] = modeling_stats["ELO_Rank"].apply(pd.to_numeric, errors='coerce')
higher_better = ["HPG", "BBPG", "BB%", "PCT", "KP9", "OPS", 'PYTHAG', 'fWAR', 'oWAR_z', 'pWAR_z', 'K/BB', 'wRC+', 'LOB%', 'wOBA', 'NET_Score']
lower_better = ["ERA", "WP9", "WHIP", "WHIP+"]
scaler = MinMaxScaler(feature_range=(1, 100))
modeling_stats[higher_better] = scaler.fit_transform(modeling_stats[higher_better])
modeling_stats[lower_better] = scaler.fit_transform(-modeling_stats[lower_better])
features = ["BB%", "PCT", "OPS", 'PYTHAG', 'fWAR', 'K/BB', 'wRC+', 'LOB%', "WHIP+", "wOBA", "NET_Score"]

# Build ratings (drop duplicates from schedule first)
model_schedule = schedule_df.drop_duplicates(
    subset=['Date', 'home_team', 'away_team', 'home_score', 'away_score']
).dropna().reset_index(drop=True)

# Fit the system
result_data, diagnostics, system = build_baseball_power_ratings(
    team_data=modeling_stats,
    schedule_df=model_schedule,
    available_features=features,
    target_columns=['ELO_Rank'],
    rating_scale=2.5,
    mae_weight=0.15,           # Pure correlation optimization
    correlation_weight=0.50,
    min_target_weight=0.20,     # At least 50% target model
    home_field_advantage=0.30              # Home field advantage weight
)
model_output = system.get_rankings(400).reset_index(drop=True)[['Team', 'Rating']]

# View results
system.print_diagnostics()

ending_data = pd.merge(baseball_stats, model_output[['Team', 'Rating']], on="Team", how="inner").sort_values('Rating', ascending=False).reset_index(drop=True)
ending_data.index = ending_data.index + 1

team_rating_quantiles = {}
for team, elo_percentile in percentile_dict.items():
    rating_at_percentile = ending_data['Rating'].quantile(elo_percentile / 100.0)
    team_rating_quantiles[team] = rating_at_percentile

missing_rating = round(ending_data['Rating'].mean() - 2.5*ending_data['Rating'].std(),2)
schedule_df = schedule_df.merge(ending_data[['Team', 'Rating']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'Rating': 'home_rating'}, inplace=True)
schedule_df = schedule_df.merge(ending_data[['Team', 'Rating']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'Rating': 'away_rating'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)
schedule_df['home_rating'].fillna(schedule_df['home_team'].map(team_rating_quantiles), inplace=True)
schedule_df['away_rating'].fillna(schedule_df['away_team'].map(team_rating_quantiles), inplace=True)
schedule_df['home_rating'].fillna(missing_rating, inplace=True)
schedule_df['away_rating'].fillna(missing_rating, inplace=True)
schedule_df['home_win_prob'] = schedule_df.apply(
    lambda row: PEAR_Win_Prob(row['home_rating'], row['away_rating'], row['Location']) / 100, axis=1
)
remaining_games = schedule_df[schedule_df["Date"] > comparison_date].reset_index(drop=True)

schedule_df['Spread'] = schedule_df.apply(
    lambda row: calculate_spread_from_stats(
        row['home_rating'], row['away_rating'],
        row['home_elo'], row['away_elo'],
        row['Location']
    )[0],  # Only take the spread
    axis=1
)
schedule_df['elo_win_prob'] = schedule_df.apply(
    lambda row: calculate_spread_from_stats(
        row['home_rating'], row['away_rating'],
        row['home_elo'], row['away_elo'],
        row['Location']
    )[1],  # Only take the win prob
    axis=1
)
schedule_df['PEAR'] = schedule_df.apply(
    lambda row: f"{row['away_team']} {-abs(row['Spread'])}" if ((row['Spread'] <= 0)) 
    else f"{row['home_team']} {-abs(row['Spread'])}", axis=1)
completed_schedule = schedule_df[
    (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
].reset_index(drop=True)
completed_schedule = completed_schedule[completed_schedule["Result"].str.startswith(("W", "L"))]

straight_up_calculator = completed_schedule.copy()

from baseball_helper import calculate_expected_wins, calculate_average_expected_wins, calculate_kpi, calculate_game_resume_quality, calculate_resume_quality, calculate_net, calculate_quadrant_records

####################### Expected Wins #######################

# Group by 'Team' and apply the calculation
team_expected_wins = completed_schedule.groupby('Team').apply(calculate_expected_wins).reset_index(drop=True)

####################### Strength of Schedule #######################

average_team = ending_data['Rating'].mean()
avg_team_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, average_team).reset_index(drop=True)

rem_avg_expected_wins = remaining_games.groupby('Team').apply(calculate_average_expected_wins, average_team).reset_index(drop=True)
if rem_avg_expected_wins.empty:
    rem_avg_expected_wins = pd.DataFrame(columns=["Team", "rem_avg_expected_wins", "rem_total_expected_wins"])
else:
    rem_avg_expected_wins.rename(columns={
        "avg_expected_wins": "rem_avg_expected_wins",
        "total_expected_wins": "rem_total_expected_wins"
    }, inplace=True)
    
####################### KPI #######################

kpi_results = calculate_kpi(completed_schedule, ending_data).sort_values('KPI_Score', ascending=False).reset_index(drop=True)

####################### Data Formatting #######################

df_1 = pd.merge(ending_data, team_expected_wins[['Team', 'expected_wins', 'Wins', 'Losses']], on='Team', how='left')
df_2 = pd.merge(df_1, avg_team_expected_wins[['Team', 'avg_expected_wins', 'total_expected_wins']], on='Team', how='left')
df_3 = pd.merge(df_2, rem_avg_expected_wins[['Team', 'rem_avg_expected_wins', 'rem_total_expected_wins']], on='Team', how='left')
df_4 = pd.merge(df_3, projected_rpi[['Team', 'RPI', 'Conference']], on='Team', how='left')
df_4.rename(columns={'RPI': 'Projected_RPI'}, inplace=True)
df_5 = pd.merge(df_4, live_rpi[['Team', 'Live_RPI']], on='Team', how='left')
df_5.rename(columns={'Live_RPI': 'RPI'}, inplace=True)
df_5['RPI'] = df_5['RPI'].astype(int)
stats_and_metrics = pd.merge(df_5, kpi_results, on='Team', how='left')
stats_and_metrics['Norm_RPI'] = stats_and_metrics['RPI'].apply(lambda x: 100 - ((x - 1) / (299 - 1)) * 99 if 299 > 1 else 100)

stats_and_metrics['wins_above_expected'] = round(stats_and_metrics['Wins'] - stats_and_metrics['total_expected_wins'],2)
stats_and_metrics['SOR'] = stats_and_metrics['wins_above_expected'].rank(method='min', ascending=False).astype(int)
max_SOR = stats_and_metrics['SOR'].max()
stats_and_metrics['SOR'].fillna(max_SOR + 1, inplace=True)
stats_and_metrics['SOR'] = stats_and_metrics['SOR'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('SOR').reset_index(drop=True)

stats_and_metrics['rem_avg_expected_wins'] = stats_and_metrics['rem_avg_expected_wins'].fillna(float(10.0))
stats_and_metrics['RemSOS'] = stats_and_metrics['rem_avg_expected_wins'].rank(method='min', ascending=True).astype(int)
max_remSOS = stats_and_metrics['RemSOS'].max()
stats_and_metrics['RemSOS'].fillna(max_remSOS + 1, inplace=True)
stats_and_metrics['RemSOS'] = stats_and_metrics['RemSOS'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('RemSOS').reset_index(drop=True)

stats_and_metrics['SOS'] = stats_and_metrics['avg_expected_wins'].rank(method='min', ascending=True).astype(int)
max_SOS = stats_and_metrics['SOS'].max()
stats_and_metrics['SOS'].fillna(max_SOS + 1, inplace=True)
stats_and_metrics['SOS'] = stats_and_metrics['SOS'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('SOS').reset_index(drop=True)

stats_and_metrics['ELO'].fillna(1200, inplace=True)

bubble_rating = stats_and_metrics.loc[(stats_and_metrics["SOR"] >= 32) & (stats_and_metrics["SOR"] <= 40), "Rating"].mean()
bubble_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, bubble_rating).reset_index(drop=True)
bubble_expected_wins.rename(columns={"avg_expected_wins": "bubble_expected_wins", "total_expected_wins":"bubble_total_expected_wins"}, inplace=True)

stats_and_metrics = pd.merge(stats_and_metrics, bubble_expected_wins, on='Team', how='left')

stats_and_metrics['wins_above_bubble'] = round(stats_and_metrics['Wins'] - stats_and_metrics['bubble_total_expected_wins'],2)
stats_and_metrics['Prelim_WAB'] = stats_and_metrics['wins_above_bubble'].rank(method='min', ascending=False).astype(int)
max_WAB = stats_and_metrics['Prelim_WAB'].max()
stats_and_metrics['Prelim_WAB'].fillna(max_WAB + 1, inplace=True)
stats_and_metrics['Prelim_WAB'] = stats_and_metrics['Prelim_WAB'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('Prelim_WAB').reset_index(drop=True)

stats_and_metrics['KPI'] = stats_and_metrics['KPI_Score'].rank(method='min', ascending=False).astype(int)
max_KPI = stats_and_metrics['KPI'].max()
stats_and_metrics['KPI'].fillna(max_KPI + 1, inplace=True)
stats_and_metrics['KPI'] = stats_and_metrics['KPI'].astype(int)

stats_and_metrics['Prelim_AVG'] = round(stats_and_metrics[['KPI', 'Prelim_WAB', 'SOR']].mean(axis=1),1)

bubble_rating = stats_and_metrics.loc[(stats_and_metrics['Prelim_AVG'] >= 32) & (stats_and_metrics['Prelim_AVG'] <= 40), "Rating"].mean()
bubble_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, bubble_rating).reset_index(drop=True)
bubble_expected_wins.rename(columns={"avg_expected_wins": "final_bubble_expected_wins", "total_expected_wins":"final_bubble_total_expected_wins"}, inplace=True)
stats_and_metrics = pd.merge(stats_and_metrics, bubble_expected_wins, on='Team', how='left')

stats_and_metrics['wins_above_bubble'] = round(stats_and_metrics['Wins'] - stats_and_metrics['final_bubble_total_expected_wins'],2)
stats_and_metrics['WAB'] = stats_and_metrics['wins_above_bubble'].rank(method='min', ascending=False).astype(int)
max_WAB = stats_and_metrics['WAB'].max()
stats_and_metrics['WAB'].fillna(max_WAB + 1, inplace=True)
stats_and_metrics['WAB'] = stats_and_metrics['WAB'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('WAB').reset_index(drop=True)

stats_and_metrics['AVG'] = round(stats_and_metrics[['KPI', 'WAB', 'SOR']].mean(axis=1),1)
stats_and_metrics['Resume'] = stats_and_metrics['AVG'].rank(method='min').astype(int)
stats_and_metrics = stats_and_metrics.sort_values('Resume').reset_index(drop=True)

####################### RQI #######################

bubble_team_rating = stats_and_metrics['Rating'].quantile(0.90)
resume_quality = completed_schedule.groupby('Team').apply(calculate_resume_quality, bubble_team_rating).reset_index(drop=True)
resume_quality['RQI'] = resume_quality['resume_quality'].rank(method='min', ascending=False).astype(int)
resume_quality = resume_quality.sort_values('RQI').reset_index(drop=True)
resume_quality['resume_quality'] = resume_quality['resume_quality'] - resume_quality.loc[15, 'resume_quality']
stats_and_metrics = pd.merge(stats_and_metrics, resume_quality, on='Team', how='left')
schedule_df["resume_quality"] = schedule_df.apply(lambda row: calculate_game_resume_quality(row, bubble_team_rating), axis=1)

####################### NET #######################

stats_and_metrics["Norm_Rating"] = (stats_and_metrics["Rating"] - stats_and_metrics["Rating"].min()) / (stats_and_metrics["Rating"].max() - stats_and_metrics["Rating"].min())
stats_and_metrics["Norm_RQI"] = (stats_and_metrics["resume_quality"] - stats_and_metrics["resume_quality"].min()) / (stats_and_metrics["resume_quality"].max() - stats_and_metrics["resume_quality"].min())
stats_and_metrics["Norm_SOS"] = 1 - (stats_and_metrics["avg_expected_wins"] - stats_and_metrics["avg_expected_wins"].min()) / (stats_and_metrics["avg_expected_wins"].max() - stats_and_metrics["avg_expected_wins"].min())  # Inverted

bounds = [(0,0.1), (0,0.15)]  
result = differential_evolution(
    calculate_net, 
    bounds, 
    args=(stats_and_metrics,),
    strategy='best1bin', 
    maxiter=500, 
    tol=1e-4, 
    seed=42
)
optimized_weights = result.x
print("NET Calculation Weights:")
print("------------------------")
print(f"Rating: {optimized_weights[0] * 100:.1f}%")
print(f"RQI: {(1 - (optimized_weights[0] + optimized_weights[1])) * 100:.1f}%")
print(f"SOS: {optimized_weights[1] * 100:.1f}%")
print(f"NET and RPI Correlation: {stats_and_metrics[['NET', 'RPI']].corr(method='spearman').iloc[0,1] * 100:.1f}%")
adj_sos_weight = (optimized_weights[1]) / ((1 - (optimized_weights[0] + optimized_weights[1])) + (optimized_weights[1]))
adj_rqi_weight = (1 - (optimized_weights[0] + optimized_weights[1])) / ((1 - (optimized_weights[0] + optimized_weights[1])) + (optimized_weights[1]))
stats_and_metrics['Norm_Resume'] = adj_rqi_weight * stats_and_metrics['Norm_RQI'] + adj_sos_weight * stats_and_metrics['Norm_SOS']
stats_and_metrics['aRQI'] = stats_and_metrics['Norm_Resume'].rank(ascending=False).astype(int)

net_row = {
    "Date": datetime.today().strftime('%Y-%m-%d'),
    "Rating": round(optimized_weights[0] * 100,1),
    "SOS": round(optimized_weights[1] * 100,1),
    "RQI": round((1 - (optimized_weights[0] + optimized_weights[1])) * 100,1),
    "NET_RPI_Correlation": round(stats_and_metrics[['NET', 'RPI']].corr(method='spearman').iloc[0, 1] * 100,1)
}
try:
    df_net = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/net_tracking.csv")
except FileNotFoundError:
    df_net = pd.DataFrame(columns=["Date", "Rating", "SOS", "RQI", "NET_RPI_Correlation"])
df_net = pd.concat([df_net, pd.DataFrame([net_row])], ignore_index=True)
df_net.to_csv(f"./PEAR/PEAR Baseball/y{current_season}/net_tracking.csv", index=False)

####################### Quadrants #######################

stats_and_metrics = calculate_quadrant_records(completed_schedule, stats_and_metrics)

stats_and_metrics.fillna(0, inplace=True)
stats_and_metrics = stats_and_metrics.sort_values('Rating', ascending=False).reset_index(drop=True)
stats_and_metrics['Rating Rank'] = stats_and_metrics.index + 1
stats_and_metrics['PRR'] = stats_and_metrics['Rating Rank']
stats_and_metrics = stats_and_metrics.sort_values('NET').reset_index(drop=True)

if "Conference" not in stats_and_metrics.columns:
    stats_and_metrics = stats_and_metrics.merge(
        projected_rpi[["Team", "Conference"]], on="Team", how="left"
    )

schedule_df = schedule_df.merge(stats_and_metrics[['Team', 'NET']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'NET': 'home_net'}, inplace=True)
schedule_df = schedule_df.merge(stats_and_metrics[['Team', 'NET']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'NET': 'away_net'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)

####################### Game Quality #######################

max_net = len(stats_and_metrics)
max_spread = 16.5

w_tq = 0.70   # NET AVG
w_wp = 0.20   # Win Probability
w_ned = 0.10  # NET Differential
schedule_df['avg_net'] = (schedule_df['home_net'] + schedule_df['away_net']) / 2
schedule_df['TQ'] = (max_net - schedule_df['avg_net']) / (max_net - 1)
schedule_df['WP'] = 1 - 2 * np.abs(schedule_df['home_win_prob'] - 0.5)
schedule_df['NED'] = 1 - (np.abs(schedule_df['home_net'] - schedule_df['away_net']) / (max_net - 1))
schedule_df['GQI'] = round(10 * (
    w_tq * schedule_df['TQ'] +
    w_wp * schedule_df['WP'] +
    w_ned * schedule_df['NED']
),1)

from baseball_helper import game_sort_key, process_result, remaining_games_rq, simulate_games

####################### Straight Up Tracking #######################

win_percentage_df = (
    completed_schedule.groupby("Team").apply(
        lambda x: pd.Series({
            "win_percentage": x["Result"].str.contains("W").sum() / len(x),
            "completed_games": len(x)
        })
    ).reset_index()
)
stats_and_metrics = pd.merge(stats_and_metrics, win_percentage_df, how='left', on='Team')
stats_and_metrics['wpoe_pct'] = stats_and_metrics['win_percentage'] - (stats_and_metrics['expected_wins'] / stats_and_metrics['completed_games'])
stats_and_metrics["WPOE"] = stats_and_metrics["wpoe_pct"].rank(ascending=False)

####################### Straight Up Tracking #######################

straight_up_calculator['Result'] = straight_up_calculator['Result'].astype(str)
straight_up_calculator = straight_up_calculator.sort_values(by="Result", key=lambda x: x.map(game_sort_key))
straight_up_calculator["Result"] = straight_up_calculator["Result"].astype(str)  # Convert to string to avoid errors
straight_up_calculator["Result"] = straight_up_calculator["Result"].apply(lambda x: x if x.startswith(("W", "L")) else "")
straight_up_calculator["Result"] = straight_up_calculator.apply(process_result, axis=1)
df = straight_up_calculator[['Result', 'PEAR', 'Date']].drop_duplicates().sort_values('Date').reset_index(drop=True)
df['Result'] = df['Result'].astype(str)
df['PEAR'] = df['PEAR'].astype(str)
def extract_team_name(text):
    match = re.match(r'^([^\d-]+)', str(text))
    return match.group(1).strip() if match else text
df['team_result'] = df['Result'].apply(extract_team_name)
df['team_pear'] = df['PEAR'].apply(extract_team_name)
df['flag'] = (df['team_result'] == df['team_pear']).astype(int)
straight_up = df.groupby('Date').agg(
    Correct=('flag', 'sum'),
    Total=('flag', 'count')
).reset_index()
previous = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/straight_up.csv").drop(columns=['Unnamed: 0'])
previous["Date"] = pd.to_datetime(previous["Date"])
merged = straight_up.merge(previous[['Date', 'Total']], on='Date', how='left', suffixes=('', '_previous'))
missing_or_mismatched = merged[
    (merged['Total_previous'].isna()) |  
    (merged['Total'] != merged['Total_previous'])
]
missing_or_mismatched = missing_or_mismatched.drop(columns=['Total_previous'])
previous = pd.concat([previous, missing_or_mismatched]).reset_index(drop=True)
previous = previous.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
previous.to_csv(f"./PEAR/PEAR Baseball/y{current_season}/straight_up.csv")

####################### Projected NET #######################

remaining_games['bubble_win_prob'] = remaining_games.apply(
    lambda row: remaining_games_rq(row, bubble_team_rating), axis=1
)

projected_wins_df = simulate_games(remaining_games)
stats_and_metrics = pd.merge(stats_and_metrics, projected_wins_df, how='left', on='Team')
stats_and_metrics['Remaining_Wins'] = stats_and_metrics['Remaining_Wins'].fillna(0)
stats_and_metrics['Remaining_Losses'] = stats_and_metrics['Remaining_Losses'].fillna(0)
stats_and_metrics['Projected_Wins'] = stats_and_metrics['Remaining_Wins'] + stats_and_metrics['Wins']
stats_and_metrics['Projected_Losses'] = stats_and_metrics['Remaining_Losses'] + stats_and_metrics['Losses']
stats_and_metrics["Projected_Record"] = stats_and_metrics.apply(
    lambda x: f"{int(x['Projected_Wins'])}-{int(x['Projected_Losses'])}", axis=1
)
stats_and_metrics['Projected_RQ'] = stats_and_metrics['resume_quality'] + stats_and_metrics['Remaining_RQ']
stats_and_metrics["Projected_Norm_RQI"] = (stats_and_metrics["Projected_RQ"] - stats_and_metrics["Projected_RQ"].min()) / (stats_and_metrics["Projected_RQ"].max() - stats_and_metrics["Projected_RQ"].min())
stats_and_metrics['Projected_NET_Score'] = (
    optimized_weights[0] * stats_and_metrics['Norm_Rating'] +
    (optimized_weights[0] + optimized_weights[1]) * stats_and_metrics['Projected_Norm_RQI'] +
    optimized_weights[1] * stats_and_metrics['Norm_SOS']
)
stats_and_metrics['Projected_NET_Score'] = stats_and_metrics['Projected_NET_Score'].fillna((optimized_weights[0] * stats_and_metrics['Norm_Rating'] + (optimized_weights[0] + optimized_weights[1]) * stats_and_metrics['Norm_RQI'] + optimized_weights[1] * stats_and_metrics['Norm_SOS']))
stats_and_metrics['Projected_NET'] = stats_and_metrics['Projected_NET_Score'].rank(ascending=False).astype(int)

####################### Percentile Calculations #######################

stats_and_metrics['pNET_Score'] = (stats_and_metrics['NET_Score'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRating'] = (stats_and_metrics['Rating'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pResume_Quality'] = (stats_and_metrics['resume_quality'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pPYTHAG'] = (stats_and_metrics['PYTHAG'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['poWAR_z'] = (stats_and_metrics['oWAR_z'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pOPS'] = (stats_and_metrics['OPS'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBA'] = (stats_and_metrics['BA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRPG'] = (stats_and_metrics['RPG'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['ppWAR_z'] = (stats_and_metrics['pWAR_z'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pKP9'] = (stats_and_metrics['KP9'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pWHIP'] = ((1 - stats_and_metrics['WHIP'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pERA'] = ((1 - stats_and_metrics['ERA'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pwOBA'] = (stats_and_metrics['wOBA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pwRAA'] = (stats_and_metrics['wRAA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pfWAR'] = (stats_and_metrics['fWAR'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pISO'] = (stats_and_metrics['ISO'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pwRC+'] = (stats_and_metrics['wRC+'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBB%'] = (stats_and_metrics['BB%'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBABIP'] = (stats_and_metrics['BABIP'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRA9'] = ((1 - stats_and_metrics['RA9'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pFIP'] = ((1 - stats_and_metrics['FIP'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pLOB%'] = (stats_and_metrics['LOB%'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pK/BB'] = (stats_and_metrics['K/BB'].rank(pct=True) * 98 + 1).round().astype(int)

todays_games = schedule_df[schedule_df["Date"] == comparison_date].dropna().sort_values('GQI', ascending=False).reset_index(drop=True)
if len(todays_games) > 20:
    todays_games = todays_games[['home_team', 'away_team', 'GQI', 'PEAR', 'home_win_prob', 'home_net', 'away_net']].drop_duplicates().reset_index(drop=True)[0:10]
else:
    todays_games = todays_games[['home_team', 'away_team', 'GQI', 'PEAR', 'home_win_prob', 'home_net', 'away_net']].drop_duplicates().reset_index(drop=True)

file_path = os.path.join(folder_path, f"Data/baseball_{formatted_date}.csv")
stats_and_metrics.to_csv(file_path)

file_path = os.path.join(folder_path, f"schedule_{current_season}.csv")
schedule_df.to_csv(file_path)

central_time_zone = pytz.timezone('US/Central')
now = datetime.now(central_time_zone)

####################### Visuals #######################

# Check if it's Monday and after 10:00 AM and before 3:00 PM
if now.hour < 23 and now.hour > 1:
    print("Starting Visuals")
    from matplotlib.patches import Rectangle
    from baseball_helper import plot_top_25, resolve_conflicts

    # --- Config & Setup ---
    BASE_URL = "https://www.warrennolan.com"
    custom_font = fm.FontProperties(fname="./PEAR/trebuc.ttf")
    plt.rcParams['font.family'] = custom_font.get_name()
    current_season = datetime.now().year
    week_1_start = datetime(current_season, 2, 10)
    today = datetime.today()
    days_since_start = (today - week_1_start).days
    current_week = (days_since_start // 7) + 1
    major_conferences = ['SEC', 'ACC', 'Independent', 'Big 12', 'Big Ten']

    import os

    logo_folder = "./PEAR/PEAR Baseball/logos/"
    logo_cache = {}
    for filename in os.listdir(logo_folder):
        if filename.endswith(".png"):
            team_name = filename[:-4].replace("_", " ")
            file_path = os.path.join(logo_folder, filename)
            try:
                img = Image.open(file_path).convert("RGBA")
                logo_cache[team_name] = img
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                logo_cache[team_name] = None

    from concurrent.futures import ThreadPoolExecutor

    logo_folder = "./PEAR/PEAR Baseball/logos/"
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

    team_images = team_logos.copy()

    # --- Generate Plots ---
    plot_top_25(
        title=f"Week {current_week} CBASE PEAR",
        subtitle="NET Ranking Incorporating Team Strength and Resume",
        team_images=team_logos,
        sorted_df=stats_and_metrics.head(25),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/NET/net_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE Resume Quality",
        subtitle="Team Performance Relative to Strength of Schedule",
        team_images=team_logos,
        sorted_df=stats_and_metrics.sort_values('RQI'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RQI/rqi_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE Team Strength",
        subtitle="Team Rating Based on Team Stats",
        team_images=team_logos,
        sorted_df=stats_and_metrics.sort_values('PRR'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/PRR/prr_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE RPI",
        subtitle="PEAR's RPI Rankings",
        team_images=team_logos,
        sorted_df=stats_and_metrics.sort_values('RPI'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RPI/rpi_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} Mid-Major CBASE PEAR",
        subtitle="NET Ranking Incorporating Team Strength and Resume",
        team_images=team_logos,
        sorted_df=stats_and_metrics[~stats_and_metrics['Conference'].isin(major_conferences)],
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Mid_Major/mid_major_{formatted_date}.png"
    )

    try:
        num_games = len(todays_games)
        num_rows = math.ceil(num_games / 2) if num_games > 0 else 1  # At least one row for layout
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(16, 1.8 * num_rows))
        fig.patch.set_facecolor('#CECEB2')
        axes = axes.flatten()
        for i in range(num_games):
            ax = axes[i]
            img = team_images.get(todays_games.loc[i, 'home_team'])
            if img:
                imagebox = OffsetImage(img, zoom=1)
                ab = AnnotationBbox(imagebox, (0.03, 0.2), frameon=False)
                ax.add_artist(ab)
            img = team_images.get(todays_games.loc[i, 'away_team'])
            if img:
                imagebox = OffsetImage(img, zoom=1)
                ab = AnnotationBbox(imagebox, (0.97, 0.2), frameon=False)
                ax.add_artist(ab)
            ax.text(0.5, 0.5, f"{todays_games.loc[i, 'PEAR']}", fontsize=24, ha='center', va='center', fontweight='bold')
            ax.text(0.5, 0.2, f"{todays_games.loc[i, 'GQI']}", fontsize=24, ha='center', va='center', fontweight='bold')
            ax.text(0.2, 0.2, f"{todays_games.loc[i, 'home_win_prob'] * 100:.1f}%", fontsize=24, ha='left', va='center')
            ax.text(0.8, 0.2, f"{(1 - todays_games.loc[i, 'home_win_prob']) * 100:.1f}%", fontsize=24, ha='right', va='center')
            ax.text(0.2, -0.1, f"#{int(todays_games.loc[i, 'home_net'])}", fontsize=24, ha='left', va='center')
            ax.text(0.8, -0.1, f"#{int(todays_games.loc[i, 'away_net'])}", fontsize=24, ha='right', va='center')
            ax.axis('off')
        for j in range(num_games, len(axes)):
            axes[j].axis('off')
        base_offset = 0.2 + (num_rows - 3) * 0.2
        plt.text(-0.11, num_rows + base_offset + 0.3, f"PEAR's {comparison_date.strftime('%m/%d')} Best Games", fontsize=32, ha='center', fontweight='bold')
        plt.text(-0.11, num_rows + base_offset, "@PEARatings", fontsize=24, ha='center', fontweight='bold')
        plt.savefig(f"./PEAR/PEAR Baseball/y2025/Visuals/Best_Games/best_games_{formatted_date}.png", bbox_inches='tight')
    except Exception as e:
        print(f"Error generating today's games plot: {e}")
    
    # ---------------------------
    # Main Logic
    # ---------------------------

    aqs = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    non_aqs = stats_and_metrics.drop(aqs.index)

    at_large = non_aqs.nsmallest(34, "NET")
    last_four_in = at_large[-8:].reset_index(drop=True)
    next_8 = non_aqs.nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    bubble_teams = non_aqs.nsmallest(50, "NET").iloc[42:].reset_index(drop=True)
    all_at_large = pd.concat([at_large, next_8, bubble_teams]).sort_values("NET").reset_index(drop=True)

    tournament = pd.concat([at_large, aqs, next_8]).sort_values("NET").reset_index(drop=True)
    sorted_aqs = aqs.sort_values("NET").reset_index(drop=True)
    last_team_in = last_four_in.loc[7, "Team"]
    last_team_in_index = all_at_large[all_at_large["Team"] == last_team_in].index[0]

    # ---------------------------
    # Bubble Distance Bar Plot
    # ---------------------------

    bubble = pd.concat([last_four_in, next_8, bubble_teams]).sort_values('NET').reset_index(drop=True)[['Team', 'NET', 'NET_Score']]
    last_net_score = bubble[bubble['Team'] == last_team_in]['NET_Score'].values[0]
    bubble["percentage_away"] = round((
        (bubble["NET_Score"] - last_net_score) / last_net_score
    ) * 100, 2)

    # Sort by percentage_away
    bubble = bubble.sort_values(by="percentage_away", ascending=False).reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(bubble) * 0.6 + 2), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_xlim(-13, 7)
    ax.set_ylim(-1, len(bubble) + 0.5)
    ax.axis('off')

    # Column positions
    logo_x = -12
    team_x = -10.5
    net_x = -5.5
    bar_center = 2.5
    bar_max_width = 10
    pct_label_offset = 0.5

    # Header
    header_y = len(bubble) + 0.2
    ax.text(team_x, header_y, 'Team', fontsize=13, weight='bold', ha='left')
    ax.text(net_x, header_y, 'NET', fontsize=13, weight='bold', ha='center')
    ax.text(bar_center, header_y, '% From Cut Line', fontsize=13, weight='bold', ha='center')

    # Draw header underline
    ax.plot([-14.5, 14.5], [header_y - 0.15, header_y - 0.15], 'k-', linewidth=2)

    # Calculate max percentage for scaling
    max_pct = max(abs(bubble['percentage_away'].min()), abs(bubble['percentage_away'].max()))

    # Get min and max NET for gradient
    min_net = bubble['NET'].min()
    max_net = bubble['NET'].max()

    # Function to get color for NET ranking box
    def get_net_box_color(net_rank):
        # Normalize NET rank to 0-1 range
        if max_net > min_net:
            normalized = (net_rank - min_net) / (max_net - min_net)
        else:
            normalized = 0.5
        
        # Dark Blue #00008B (0, 0, 139) -> Light Gray #D3D3D3 (211, 211, 211) -> Dark Red #8B0000 (139, 0, 0)
        if normalized < 0.5:
            # Interpolate between Dark Blue and Light Gray
            t = normalized * 2  # Scale to 0-1
            r = int(0 * (1 - t) + 211 * t)
            g = int(0 * (1 - t) + 211 * t)
            b = int(139 * (1 - t) + 211 * t)
        else:
            # Interpolate between Light Gray and Dark Red
            t = (normalized - 0.5) * 2  # Scale to 0-1
            r = int(211 * (1 - t) + 139 * t)
            g = int(211 * (1 - t) + 0 * t)
            b = int(211 * (1 - t) + 0 * t)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    # Draw each team
    row_height = 1.0
    start_y = len(bubble) - 0.5

    for idx, row in bubble.iterrows():
        y_pos = start_y - idx * row_height
        team = row['Team']
        net_rank = row['NET']
        pct_away = row['percentage_away']
        
        # Team logo
        logo = team_logos.get(team)
        if logo:
            imagebox = OffsetImage(logo, zoom=0.055)
            ab = AnnotationBbox(imagebox, (logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        # Team name
        ax.text(team_x, y_pos, team, fontsize=14, weight='bold', ha='left', va='center')
        
        # NET ranking box with gradient color
        net_box_width = 1.5
        net_box_height = 0.6
        box_color = get_net_box_color(net_rank)
        
        rect = Rectangle((net_x - net_box_width/2, y_pos - net_box_height/2), 
                        net_box_width, net_box_height, 
                        facecolor=box_color, edgecolor='black', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        
        # Determine text color based on background brightness
        # Extract RGB values
        r = int(box_color[1:3], 16)
        g = int(box_color[3:5], 16)
        b_val = int(box_color[5:7], 16)
        brightness = (r * 299 + g * 587 + b_val * 114) / 1000
        text_color = 'white' if brightness < 128 else 'black'
        
        ax.text(net_x, y_pos, str(int(net_rank)), fontsize=13, weight='bold', 
            ha='center', va='center', color=text_color, zorder=3)
        
        # Percentage bar
        bar_color = "#00008B" if pct_away >= 0 else "#8B0000"
        
        # Scale bar width based on percentage
        if max_pct > 0:
            scaled_width = (abs(pct_away) / max_pct) * (bar_max_width / 2)
        else:
            scaled_width = 0
        
        bar_height = 0.5
        
        if pct_away >= 0:
            # Positive (safe) - bar extends right
            bar_x = bar_center
            bar_width = scaled_width
            label_x = bar_x + bar_width + pct_label_offset
            label_ha = 'left'
        else:
            # Negative (danger) - bar extends left
            bar_x = bar_center - scaled_width
            bar_width = scaled_width
            label_x = bar_x - pct_label_offset
            label_ha = 'right'
        
        # Draw bar
        rect = Rectangle((bar_x, y_pos - bar_height/2), 
                        bar_width, bar_height, 
                        facecolor=bar_color, edgecolor='black', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        
        # Percentage label
        pct_text = f"+{pct_away:.1f}%" if pct_away >= 0 else f"{pct_away:.1f}%"
        text_color = "#00008B" if pct_away >= 0 else "#8B0000"
        ax.text(label_x, y_pos, pct_text, fontsize=12, weight='bold', 
            ha=label_ha, va='center', color=text_color, zorder=3)
        
        # Row separator
        if idx < len(bubble) - 1:
            separator_y = y_pos - row_height/2
            ax.plot([-14.5, 14.5], [separator_y, separator_y], 'k-', linewidth=0.5, alpha=0.3)

    # Draw vertical cutline at 0%
    ax.axvline(x=bar_center, color='black', linewidth=2, linestyle='--', alpha=0.7, zorder=1)
    ax.text(bar_center, -0.7, 'CUT LINE', fontsize=11, weight='bold', ha='center', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFFF', edgecolor='black'))

    # Safe/Bubble zones labels
    ax.text(bar_center + bar_max_width/4, len(bubble) + 0.7, 'SAFE', 
        fontsize=12, weight='bold', ha='center', color='#00008B', alpha=0.9)
    ax.text(bar_center - bar_max_width/4, len(bubble) + 0.7, 'BUBBLE', 
        fontsize=12, weight='bold', ha='center', color='#8B0000', alpha=0.9)

    # Title
    today = datetime.today()
    fig.text(0.02, 0.95, f"NET Score Distance From Last Team In Through {(today - timedelta(days=1)).strftime('%m/%d/%Y')}", 
            ha='left', fontweight='bold', fontsize=28)
    fig.text(0.02, 0.93, "@PEARatings", 
            ha='left', fontweight='bold', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Team_In/last_team_in_{formatted_date}.png")

    # ---------------------------
    # Hosting Distance Bar Plot
    # ---------------------------

    hosting = stats_and_metrics[0:24][['Team', 'NET', 'NET_Score']]
    last_host_score = hosting[hosting['NET'] == 16]['NET_Score'].values[0]
    hosting["percentage_away"] = round((
        (hosting["NET_Score"] - last_host_score) / last_host_score
    ) * 100, 2)

    # Sort by percentage_away
    hosting = hosting.sort_values(by="percentage_away", ascending=False).reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(hosting) * 0.6 + 2), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_xlim(-13, 10)
    ax.set_ylim(-1, len(hosting) + 0.5)
    ax.axis('off')

    # Column positions
    logo_x = -12
    team_x = -10.5
    net_x = -5.5
    bar_center = 2.5
    bar_max_width = 10
    pct_label_offset = 0.5

    # Header
    header_y = len(hosting) + 0.2
    ax.text(team_x, header_y, 'Team', fontsize=13, weight='bold', ha='left')
    ax.text(net_x, header_y, 'NET', fontsize=13, weight='bold', ha='center')
    ax.text(bar_center, header_y, '% From Cut Line', fontsize=13, weight='bold', ha='center')

    # Draw header underline
    ax.plot([-14.5, 14.5], [header_y - 0.15, header_y - 0.15], 'k-', linewidth=2)

    # Calculate max percentage for scaling
    max_pct = max(abs(hosting['percentage_away'].min()), abs(hosting['percentage_away'].max()))

    # Get min and max NET for gradient
    min_net = hosting['NET'].min()
    max_net = hosting['NET'].max()

    # Function to get color for NET ranking box
    def get_net_box_color(net_rank):
        # Normalize NET rank to 0-1 range
        if max_net > min_net:
            normalized = (net_rank - min_net) / (max_net - min_net)
        else:
            normalized = 0.5
        
        # Dark Blue #00008B (0, 0, 139) -> Light Gray #D3D3D3 (211, 211, 211) -> Dark Red #8B0000 (139, 0, 0)
        if normalized < 0.5:
            # Interpolate between Dark Blue and Light Gray
            t = normalized * 2  # Scale to 0-1
            r = int(0 * (1 - t) + 211 * t)
            g = int(0 * (1 - t) + 211 * t)
            b = int(139 * (1 - t) + 211 * t)
        else:
            # Interpolate between Light Gray and Dark Red
            t = (normalized - 0.5) * 2  # Scale to 0-1
            r = int(211 * (1 - t) + 139 * t)
            g = int(211 * (1 - t) + 0 * t)
            b = int(211 * (1 - t) + 0 * t)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    # Draw each team
    row_height = 1.0
    start_y = len(hosting) - 0.5

    for idx, row in hosting.iterrows():
        y_pos = start_y - idx * row_height
        team = row['Team']
        net_rank = row['NET']
        pct_away = row['percentage_away']
        
        # Team logo
        logo = team_logos.get(team)
        if logo:
            imagebox = OffsetImage(logo, zoom=0.055)
            ab = AnnotationBbox(imagebox, (logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        # Team name
        ax.text(team_x, y_pos, team, fontsize=14, weight='bold', ha='left', va='center')
        
        # NET ranking box with gradient color
        net_box_width = 1.5
        net_box_height = 0.6
        box_color = get_net_box_color(net_rank)
        
        rect = Rectangle((net_x - net_box_width/2, y_pos - net_box_height/2), 
                        net_box_width, net_box_height, 
                        facecolor=box_color, edgecolor='black', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        
        # Determine text color based on background brightness
        # Extract RGB values
        r = int(box_color[1:3], 16)
        g = int(box_color[3:5], 16)
        b_val = int(box_color[5:7], 16)
        brightness = (r * 299 + g * 587 + b_val * 114) / 1000
        text_color = 'white' if brightness < 128 else 'black'
        
        ax.text(net_x, y_pos, str(int(net_rank)), fontsize=13, weight='bold', 
            ha='center', va='center', color=text_color, zorder=3)
        
        # Percentage bar
        bar_color = "#00008B" if pct_away >= 0 else "#8B0000"
        
        # Scale bar width based on percentage
        if max_pct > 0:
            scaled_width = (abs(pct_away) / max_pct) * (bar_max_width / 2)
        else:
            scaled_width = 0
        
        bar_height = 0.5
        
        if pct_away >= 0:
            # Positive (safe) - bar extends right
            bar_x = bar_center
            bar_width = scaled_width
            label_x = bar_x + bar_width + pct_label_offset
            label_ha = 'left'
        else:
            # Negative (danger) - bar extends left
            bar_x = bar_center - scaled_width
            bar_width = scaled_width
            label_x = bar_x - pct_label_offset
            label_ha = 'right'
        
        # Draw bar
        rect = Rectangle((bar_x, y_pos - bar_height/2), 
                        bar_width, bar_height, 
                        facecolor=bar_color, edgecolor='black', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        
        # Percentage label
        pct_text = f"+{pct_away:.1f}%" if pct_away >= 0 else f"{pct_away:.1f}%"
        text_color = "#00008B" if pct_away >= 0 else "#8B0000"
        ax.text(label_x, y_pos, pct_text, fontsize=12, weight='bold', 
            ha=label_ha, va='center', color=text_color, zorder=3)
        
        # Row separator
        if idx < len(hosting) - 1:
            separator_y = y_pos - row_height/2
            ax.plot([-14.5, 14.5], [separator_y, separator_y], 'k-', linewidth=0.5, alpha=0.3)

    # Draw vertical cutline at 0%
    ax.axvline(x=bar_center, color='black', linewidth=2, linestyle='--', alpha=0.7, zorder=1)
    ax.text(bar_center, -0.7, 'HOST LINE', fontsize=11, weight='bold', ha='center', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFFF', edgecolor='black'))

    # Safe/Bubble zones labels
    ax.text(bar_center + bar_max_width/4, len(hosting) + 0.7, 'HOST', 
        fontsize=12, weight='bold', ha='center', color='#00008B', alpha=0.9)
    ax.text(bar_center - bar_max_width/4, len(hosting) + 0.7, 'BUBBLE', 
        fontsize=12, weight='bold', ha='center', color='#8B0000', alpha=0.9)

    # Title
    today = datetime.today()
    fig.text(0.02, 0.95, f"NET Score Distance From Last Team In Through {(today - timedelta(days=1)).strftime('%m/%d/%Y')}", 
            ha='left', fontweight='bold', fontsize=28)
    fig.text(0.02, 0.93, "@PEARatings", 
            ha='left', fontweight='bold', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Host/last_host_{formatted_date}.png")

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers])
    tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order
    conference_counts = tournament['Conference'].value_counts()
    multibid = conference_counts[conference_counts > 1]
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    formatted_df = resolve_conflicts(formatted_df, stats_and_metrics)
    formatted_df.index = formatted_df.index + 1
    # current_season = 2025
    # today = datetime.today()

    # Create a set of automatic qualifier teams for faster lookup
    automatic_teams = set(automatic_qualifiers["Team"])

    # Modify the DataFrame by appending '*' to teams in automatic qualifiers
    formatted_df_with_asterisk = formatted_df.copy()
    for col in formatted_df_with_asterisk.columns[0:]:
        formatted_df_with_asterisk[col] = formatted_df_with_asterisk[col].apply(
            lambda x: f"{x}*" if x in automatic_teams else x
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 16), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')

    # Column positions
    rank_x = 0.7
    host_logo_x = 1.5
    host_name_x = 2.2
    seed2_logo_x = 5
    seed2_name_x = 5.5
    seed3_logo_x = 8
    seed3_name_x = 8.5
    seed4_logo_x = 11
    seed4_name_x = 11.5

    # Header
    header_y = 17.8
    ax.text(host_name_x, header_y, 'Host (1 Seed)', fontsize=14, weight='bold', ha='left')
    ax.text(seed2_name_x, header_y, '2 Seed', fontsize=14, weight='bold', ha='left')
    ax.text(seed3_name_x, header_y, '3 Seed', fontsize=14, weight='bold', ha='left')
    ax.text(seed4_name_x, header_y, '4 Seed', fontsize=14, weight='bold', ha='left')

    # Draw header underline
    ax.plot([0.5, 13.8], [header_y - 0.25, header_y - 0.25], 'k-', linewidth=2)

    # Draw each regional
    row_height = 1.05
    start_y = 17.0

    for idx in range(16):
        y_pos = start_y - idx * row_height
        
        # Regional number
        ax.text(rank_x, y_pos, f"#{idx + 1}", fontsize=16, weight='bold', ha='center', va='center')
        
        # Get teams
        host_team = formatted_df.iloc[idx]['1 Seed']
        seed2_team = formatted_df.iloc[idx]['2 Seed']
        seed3_team = formatted_df.iloc[idx]['3 Seed']
        seed4_team = formatted_df.iloc[idx]['4 Seed']
        
        # Host (1 Seed)
        host_logo = team_logos.get(host_team)
        if host_logo:
            imagebox = OffsetImage(host_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (host_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        host_display = f"{host_team}*" if host_team in automatic_teams else host_team
        ax.text(host_name_x, y_pos, host_display, fontsize=16, weight='bold', ha='left', va='center')
        
        # 2 Seed
        seed2_logo = team_logos.get(seed2_team)
        if seed2_logo:
            imagebox = OffsetImage(seed2_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed2_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed2_display = f"{seed2_team}*" if seed2_team in automatic_teams else seed2_team
        ax.text(seed2_name_x, y_pos, seed2_display, fontsize=14, ha='left', va='center')
        
        # 3 Seed
        seed3_logo = team_logos.get(seed3_team)
        if seed3_logo:
            imagebox = OffsetImage(seed3_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed3_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed3_display = f"{seed3_team}*" if seed3_team in automatic_teams else seed3_team
        ax.text(seed3_name_x, y_pos, seed3_display, fontsize=14, ha='left', va='center')
        
        # 4 Seed
        seed4_logo = team_logos.get(seed4_team)
        if seed4_logo:
            imagebox = OffsetImage(seed4_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed4_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed4_display = f"{seed4_team}*" if seed4_team in automatic_teams else seed4_team
        ax.text(seed4_name_x, y_pos, seed4_display, fontsize=14, ha='left', va='center')
        
        # Row separator
        if idx < 15:
            separator_y = y_pos - row_height/2
            ax.plot([0.5, 13.8], [separator_y, separator_y], 'k-', linewidth=1, alpha=0.8)

    # Bottom info section
    info_start_y = 0.1
    info_line_height = 0.3

    # Last Four In
    last_four_in_teams = set(last_four_in["Team"])
    lfi_text = f"Last Four In: {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}"
    ax.text(0.4, info_start_y + 2 * info_line_height, lfi_text, fontsize=12, ha='left', va='top')

    # First Four Out
    ffo_text = f"First Four Out: {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}"
    ax.text(0.4, info_start_y + 1 * info_line_height, ffo_text, fontsize=12, ha='left', va='top')

    # Next Four Out
    nfo_text = f"Next Four Out: {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}"
    ax.text(0.4, info_start_y, nfo_text, fontsize=12, ha='left', va='top')

    # Multibid conferences
    conference_counts = formatted_df[['1 Seed', '2 Seed', '3 Seed', '4 Seed']].stack().value_counts()
    # Get conference for each team
    team_conferences = {}
    for team in formatted_df[['1 Seed', '2 Seed', '3 Seed', '4 Seed']].stack().unique():
        clean_team = team.rstrip('*')
        if clean_team in stats_and_metrics['Team'].values:
            conf = stats_and_metrics.loc[stats_and_metrics['Team'] == clean_team, 'Conference'].iloc[0]
            team_conferences[team] = conf

    # Count by conference
    conf_counts = {}
    for team, conf in team_conferences.items():
        conf_counts[conf] = conf_counts.get(conf, 0) + 1

    multibid = {k: v for k, v in conf_counts.items() if v > 1}
    multibid_text = " | ".join([f"{conference}: {count}" for conference, count in sorted(multibid.items(), key=lambda x: x[1], reverse=True)])
    multibid_full = f"Multibid Conferences: {multibid_text}"

    # Wrap multibid text
    wrapped_multibid = textwrap.fill(multibid_full, width=120)
    multibid_lines = wrapped_multibid.split('\n')
    multibid_y = 0.7
    for i, line in enumerate(multibid_lines):
        ax.text(13.75, multibid_y - i * 0.3, line, fontsize=11, ha='right', va='top')

    # Asterisk note
    ax.text(13.75, multibid_y - len(multibid_lines) * 0.3, "* Indicates an automatic qualifier", 
            fontsize=11, ha='right', va='top', style='italic')

    # Titles at top
    fig.text(0.05, 0.97, f"PEAR's NET Ranking {today.strftime('%m/%d')} Current Tournament Outlook", 
            ha='left', fontweight='bold', fontsize=32)
    fig.text(0.05, 0.95, f"No Considerations For Regional Proximity - Through {(today - timedelta(days=1)).strftime('%m/%d')}", 
            ha='left', fontsize=24)
    fig.text(0.05, 0.93, "@PEARatings", 
            ha='left', fontweight='bold', fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Tournament/tournament_{formatted_date}.png", bbox_inches='tight')

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["Projected_NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "Projected_NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "Projected_NET").iloc[34:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers])
    tournament = tournament.sort_values(by="Projected_NET").reset_index(drop=True)
    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order
    conference_counts = tournament['Conference'].value_counts()
    multibid = conference_counts[conference_counts > 1]
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    formatted_df = resolve_conflicts(formatted_df, stats_and_metrics)
    formatted_df.index = formatted_df.index + 1
    # current_season = 2025
    # today = datetime.today()

    # Create a set of automatic qualifier teams for faster lookup
    automatic_teams = set(automatic_qualifiers["Team"])

    # Modify the DataFrame by appending '*' to teams in automatic qualifiers
    formatted_df_with_asterisk = formatted_df.copy()
    for col in formatted_df_with_asterisk.columns[0:]:
        formatted_df_with_asterisk[col] = formatted_df_with_asterisk[col].apply(
            lambda x: f"{x}*" if x in automatic_teams else x
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 16), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')

    # Column positions
    rank_x = 0.7
    host_logo_x = 1.5
    host_name_x = 2.2
    seed2_logo_x = 5
    seed2_name_x = 5.5
    seed3_logo_x = 8
    seed3_name_x = 8.5
    seed4_logo_x = 11
    seed4_name_x = 11.5

    # Header
    header_y = 17.8
    ax.text(host_name_x, header_y, 'Host (1 Seed)', fontsize=14, weight='bold', ha='left')
    ax.text(seed2_name_x, header_y, '2 Seed', fontsize=14, weight='bold', ha='left')
    ax.text(seed3_name_x, header_y, '3 Seed', fontsize=14, weight='bold', ha='left')
    ax.text(seed4_name_x, header_y, '4 Seed', fontsize=14, weight='bold', ha='left')

    # Draw header underline
    ax.plot([0.5, 13.8], [header_y - 0.25, header_y - 0.25], 'k-', linewidth=2)

    # Draw each regional
    row_height = 1.05
    start_y = 17.0

    for idx in range(16):
        y_pos = start_y - idx * row_height
        
        # Regional number
        ax.text(rank_x, y_pos, f"#{idx + 1}", fontsize=16, weight='bold', ha='center', va='center')
        
        # Get teams
        host_team = formatted_df.iloc[idx]['1 Seed']
        seed2_team = formatted_df.iloc[idx]['2 Seed']
        seed3_team = formatted_df.iloc[idx]['3 Seed']
        seed4_team = formatted_df.iloc[idx]['4 Seed']
        
        # Host (1 Seed)
        host_logo = team_logos.get(host_team)
        if host_logo:
            imagebox = OffsetImage(host_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (host_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        host_display = f"{host_team}*" if host_team in automatic_teams else host_team
        ax.text(host_name_x, y_pos, host_display, fontsize=16, weight='bold', ha='left', va='center')
        
        # 2 Seed
        seed2_logo = team_logos.get(seed2_team)
        if seed2_logo:
            imagebox = OffsetImage(seed2_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed2_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed2_display = f"{seed2_team}*" if seed2_team in automatic_teams else seed2_team
        ax.text(seed2_name_x, y_pos, seed2_display, fontsize=14, ha='left', va='center')
        
        # 3 Seed
        seed3_logo = team_logos.get(seed3_team)
        if seed3_logo:
            imagebox = OffsetImage(seed3_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed3_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed3_display = f"{seed3_team}*" if seed3_team in automatic_teams else seed3_team
        ax.text(seed3_name_x, y_pos, seed3_display, fontsize=14, ha='left', va='center')
        
        # 4 Seed
        seed4_logo = team_logos.get(seed4_team)
        if seed4_logo:
            imagebox = OffsetImage(seed4_logo, zoom=0.07)
            ab = AnnotationBbox(imagebox, (seed4_logo_x, y_pos), frameon=False)
            ax.add_artist(ab)
        
        seed4_display = f"{seed4_team}*" if seed4_team in automatic_teams else seed4_team
        ax.text(seed4_name_x, y_pos, seed4_display, fontsize=14, ha='left', va='center')
        
        # Row separator
        if idx < 15:
            separator_y = y_pos - row_height/2
            ax.plot([0.5, 13.8], [separator_y, separator_y], 'k-', linewidth=1, alpha=0.8)

    # Bottom info section
    info_start_y = 0.1
    info_line_height = 0.3

    # Last Four In
    last_four_in_teams = set(last_four_in["Team"])
    lfi_text = f"Last Four In: {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}"
    ax.text(0.4, info_start_y + 2 * info_line_height, lfi_text, fontsize=12, ha='left', va='top')

    # First Four Out
    ffo_text = f"First Four Out: {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}"
    ax.text(0.4, info_start_y + 1 * info_line_height, ffo_text, fontsize=12, ha='left', va='top')

    # Next Four Out
    nfo_text = f"Next Four Out: {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}"
    ax.text(0.4, info_start_y, nfo_text, fontsize=12, ha='left', va='top')

    # Multibid conferences
    conference_counts = formatted_df[['1 Seed', '2 Seed', '3 Seed', '4 Seed']].stack().value_counts()
    # Get conference for each team
    team_conferences = {}
    for team in formatted_df[['1 Seed', '2 Seed', '3 Seed', '4 Seed']].stack().unique():
        clean_team = team.rstrip('*')
        if clean_team in stats_and_metrics['Team'].values:
            conf = stats_and_metrics.loc[stats_and_metrics['Team'] == clean_team, 'Conference'].iloc[0]
            team_conferences[team] = conf

    # Count by conference
    conf_counts = {}
    for team, conf in team_conferences.items():
        conf_counts[conf] = conf_counts.get(conf, 0) + 1

    multibid = {k: v for k, v in conf_counts.items() if v > 1}
    multibid_text = " | ".join([f"{conference}: {count}" for conference, count in sorted(multibid.items(), key=lambda x: x[1], reverse=True)])
    multibid_full = f"Multibid Conferences: {multibid_text}"

    # Wrap multibid text
    wrapped_multibid = textwrap.fill(multibid_full, width=120)
    multibid_lines = wrapped_multibid.split('\n')
    multibid_y = 0.7
    for i, line in enumerate(multibid_lines):
        ax.text(13.75, multibid_y - i * 0.3, line, fontsize=11, ha='right', va='top')

    # Asterisk note
    ax.text(13.75, multibid_y - len(multibid_lines) * 0.3, "* Indicates an automatic qualifier", 
            fontsize=11, ha='right', va='top', style='italic')

    # Titles at top
    fig.text(0.05, 0.97, f"PEAR's NET Ranking {today.strftime('%m/%d')} Current Tournament Outlook", 
            ha='left', fontweight='bold', fontsize=32)
    fig.text(0.05, 0.95, f"No Considerations For Regional Proximity - Through {(today - timedelta(days=1)).strftime('%m/%d')}", 
            ha='left', fontsize=24)
    fig.text(0.05, 0.93, "@PEARatings", 
            ha='left', fontweight='bold', fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Projected_Tournament/proj_tournament_{formatted_date}.png", bbox_inches='tight')

    from baseball_helper import simulate_full_tournament

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers])
    tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    formatted_df = resolve_conflicts(formatted_df, stats_and_metrics)

    tournament_sim = simulate_full_tournament(formatted_df, stats_and_metrics, 1000)
    top_25_teams = tournament_sim[0:32]
    top_25_teams.iloc[:, 1:] = top_25_teams.iloc[:, 1:] * 100

    # Determine if we're showing seeds
    show_seeds = len(top_25_teams) < 20

    # Create custom visualization
    num_teams = len(top_25_teams)
    teams_per_column = 16
    num_columns = 2

    row_height = 1
    fig_height = 1.2 + (teams_per_column * row_height)
    fig_width = 24  # Double width for two columns

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')

    # Color map for probabilities - blue gradient
    cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E8EAF6', '#1A237E'])

    # Get column names (excluding Team column)
    prob_columns = [col for col in top_25_teams.columns if col != 'Team']
    num_prob_cols = len(prob_columns)

    # Calculate min/max for normalization (excluding zeros)
    all_probs = top_25_teams[prob_columns].values.flatten()
    all_probs = all_probs[all_probs > 0]
    min_value = all_probs.min() if len(all_probs) > 0 else 0
    max_value = all_probs.max() if len(all_probs) > 0 else 100

    # Function to draw a column of teams
    def draw_column(start_idx, end_idx, column_offset_x):
        # Column positions
        rank_x = 0.5 + column_offset_x
        logo_x = 1.4 + column_offset_x
        team_x = 2.2 + column_offset_x
        
        # Distribute probability columns evenly across remaining space
        prob_start_x = 6.5 + column_offset_x
        prob_end_x = 11 + column_offset_x
        prob_spacing = (prob_end_x - prob_start_x) / (num_prob_cols - 1) if num_prob_cols > 1 else 0
        prob_x_positions = [prob_start_x + i * prob_spacing for i in range(num_prob_cols)]
        
        # Header
        header_y = fig_height - 0.7
        ax.text(rank_x, header_y, 'Rank', fontsize=20, weight='bold', ha='center')
        ax.text(team_x, header_y, 'Team', fontsize=20, weight='bold', ha='left')
        
        # Probability column headers
        for i, col in enumerate(prob_columns):
            ax.text(prob_x_positions[i], header_y, col, fontsize=20, weight='bold', ha='center')
        # Draw header underline
        ax.plot([0.2 + column_offset_x, 11.8 + column_offset_x], 
                [header_y - 0.2, header_y - 0.2], 'k-', linewidth=2)
        
        # Draw each team row
        start_y = header_y - 0.8
        for idx in range(start_idx, min(end_idx, len(top_25_teams))):
            row = top_25_teams.iloc[idx]
            local_idx = idx - start_idx
            y_pos = start_y - local_idx * row_height
            
            # Rank
            ax.text(rank_x, y_pos + 0.05, str(idx + 1), fontsize=24, weight='bold', ha='center', va='center')
            
            # Logo
            team_name = row['Team']
            # Remove seed prefix if present
            if show_seeds and team_name.startswith('#'):
                clean_team_name = team_name.split(' ', 1)[1] if ' ' in team_name else team_name
            else:
                clean_team_name = team_name
            
            logo = team_logos[clean_team_name]
            zoom_factor = 0.08
            imagebox = OffsetImage(logo, zoom=zoom_factor)
            ab = AnnotationBbox(imagebox, (logo_x, y_pos + 0.05), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)    
            display_name = clean_team_name
            
            font_size = 24
            ax.text(team_x, y_pos + 0.02, display_name, fontsize=font_size, weight='bold', ha='left', va='center')
            
            # Probabilities with color background
            box_width = 0.85 if num_prob_cols > 4 else 1.2
            box_height = 0.7
            
            for i, col in enumerate(prob_columns):
                prob = row[col]
                x_pos = prob_x_positions[i]
                
                # Normalize for color
                if prob > 0:
                    normalized = (prob - min_value) / (max_value - min_value) if max_value > min_value else 0
                else:
                    normalized = 0
                
                color = cmap(normalized)
                
                # Draw colored rectangle background
                rect = Rectangle((x_pos - box_width/2, 0.05 + y_pos - box_height/2), 
                                box_width, box_height, 
                                facecolor=color, edgecolor='black', linewidth=0.8, zorder=2)
                ax.add_patch(rect)
                
                # Display text
                if prob < 0.05 and prob > 0:
                    text = "<1%"
                elif prob == 0:
                    text = "<1%"
                else:
                    text = f"{prob:.1f}%"
                
                # Use white text for darker backgrounds
                text_color = 'white' if normalized > 0.5 else 'black'
                text_size = 24
                ax.text(x_pos, y_pos + 0.05, text, fontsize=text_size, weight='bold', 
                    ha='center', va='center', color=text_color, zorder=3)
            
            # Draw row separator line
            if local_idx < teams_per_column - 1 and idx < len(top_25_teams) - 1:
                separator_y = y_pos - row_height/2 + 0.05
                ax.plot([0.2 + column_offset_x, 11.8 + column_offset_x], 
                    [separator_y, separator_y], 'k-', linewidth=1, alpha=0.8)

    # Draw left column (teams 1-16)
    draw_column(0, teams_per_column, 0)

    # Draw right column (teams 17-32)
    draw_column(teams_per_column, teams_per_column * 2, 12)

    # Draw vertical separator between columns
    separator_x = 12
    ax.plot([separator_x, separator_x], [0.5, fig_height - 1.15], 'k-', linewidth=2, alpha=0.5)

    # Title section (centered across both columns)
    title_y = fig_height - 0.1
    brand_y = fig_height - 0.1

    ax.text(0.2, title_y, 'Odds to Win Championship - PEAR\'s Tournament Projections', fontsize=40, weight='bold', ha='left')
    ax.text(fig_width-0.2, brand_y, '@PEARatings', fontsize=40, weight='bold', ha='right')

    plt.tight_layout()
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Tournament_Odds/tournament_odds_{formatted_date}.png", bbox_inches='tight')

    # --- Utility Functions ---

    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)

    def normalize_array(values):
        values = np.array(values)
        non_zero = values[values > 0]
        min_val, max_val = non_zero.min(), non_zero.max() if len(non_zero) else (0, 1)
        return np.where(values == 0, 0, (values - min_val) / (max_val - min_val))

    # --- Simulation Functions ---

    def calculate_regional_probabilities(teams, ratings):
        """
        Calculate probabilities for a 4-team double elimination regional
        Teams: [1 seed (host), 2 seed, 3 seed, 4 seed]
        
        Tournament structure:
        Game 1: 1 vs 4
        Game 2: 2 vs 3
        Game 3: Loser G1 vs Loser G2 (elimination)
        Game 4: Winner G1 vs Winner G2 (winner to championship)
        Game 5: Loser G4 vs Winner G3 (elimination)
        Game 6: Winner G4 vs Winner G5 (championship - if G4 winner loses, game 7)
        Game 7: Winner G4 vs Winner G5 (if needed)
        """
        team_a, team_b, team_c, team_d = teams
        
        # Apply home field advantage to 1 seed
        home_advantage = 0.3
        
        def get_rating(team):
            return ratings[team] + (home_advantage if team == team_a else 0)
        
        # Initialize probabilities for each team
        team_probs = {team: 0.0 for team in teams}
        
        # Game 1: 1 seed vs 4 seed
        p_a_beats_d = PEAR_Win_Prob(get_rating(team_a), get_rating(team_d))
        
        # Game 2: 2 seed vs 3 seed  
        p_b_beats_c = PEAR_Win_Prob(get_rating(team_b), get_rating(team_c))
        
        # Iterate through all possible game 1 outcomes
        for g1_winner, g1_loser, p_g1 in [
            (team_a, team_d, p_a_beats_d),
            (team_d, team_a, 1 - p_a_beats_d)
        ]:
            # Iterate through all possible game 2 outcomes
            for g2_winner, g2_loser, p_g2 in [
                (team_b, team_c, p_b_beats_c),
                (team_c, team_b, 1 - p_b_beats_c)
            ]:
                # Game 3: Loser's bracket (elimination game)
                p_g1l_beats_g2l = PEAR_Win_Prob(get_rating(g1_loser), get_rating(g2_loser))
                
                for g3_winner, g3_loser, p_g3 in [
                    (g1_loser, g2_loser, p_g1l_beats_g2l),
                    (g2_loser, g1_loser, 1 - p_g1l_beats_g2l)
                ]:
                    # g3_loser is eliminated (doesn't win regional)
                    
                    # Game 4: Winner's bracket final
                    p_g1w_beats_g2w = PEAR_Win_Prob(get_rating(g1_winner), get_rating(g2_winner))
                    
                    for g4_winner, g4_loser, p_g4 in [
                        (g1_winner, g2_winner, p_g1w_beats_g2w),
                        (g2_winner, g1_winner, 1 - p_g1w_beats_g2w)
                    ]:
                        # g4_winner advances to championship (still undefeated)
                        # g4_loser drops to loser's bracket
                        
                        # Game 5: Loser's bracket final (elimination game)
                        p_g4l_beats_g3w = PEAR_Win_Prob(get_rating(g4_loser), get_rating(g3_winner))
                        
                        for g5_winner, g5_loser, p_g5 in [
                            (g4_loser, g3_winner, p_g4l_beats_g3w),
                            (g3_winner, g4_loser, 1 - p_g4l_beats_g3w)
                        ]:
                            # g5_loser is eliminated
                            
                            # Championship: g4_winner (undefeated) vs g5_winner (one loss)
                            p_champ = PEAR_Win_Prob(get_rating(g4_winner), get_rating(g5_winner))
                            
                            # Probability of reaching this state
                            path_prob = p_g1 * p_g2 * p_g3 * p_g4 * p_g5
                            
                            # If g4_winner wins game 6, they win the regional
                            team_probs[g4_winner] += path_prob * p_champ
                            
                            # If g5_winner wins game 6, play game 7
                            # Game 7: same matchup
                            p_g7 = PEAR_Win_Prob(get_rating(g4_winner), get_rating(g5_winner))
                            team_probs[g4_winner] += path_prob * (1 - p_champ) * p_g7
                            team_probs[g5_winner] += path_prob * (1 - p_champ) * (1 - p_g7)
        
        return team_probs

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(34, "NET")
    tournament = pd.concat([automatic_qualifiers, at_large]).sort_values("NET").reset_index(drop=True)

    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order[:len(tournament)]

    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed']

    # --- Visualization ---

    fig, axes = plt.subplots(4, 4, figsize=(12, 10), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    axes = axes.flatten()

    cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E8EAF6', '#1A237E'])

    custom_order = [0, 15, 4, 11, 1, 14, 5, 10, 2, 13, 6, 9, 3, 12, 7, 8]
    total_win_prob = 0

    for plot_idx, idx in enumerate(custom_order):
        teams = list(formatted_df.iloc[idx, 1:5])
        ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}
        
        # Calculate probabilities
        regional_probs = calculate_regional_probabilities(teams, ratings)
        
        # Create visualization for this regional
        ax = axes[plot_idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.2)
        ax.axis('off')
        
        # Column positions
        seed_x = 0.6
        logo_x = 0.5
        team_x = 1.6
        prob_x = 8.7
        
        # Regional title at top
        ax.text(5.0, 4.1, f'#{idx + 1} {teams[0]} Regional',
                fontsize=12, fontweight='bold', ha='center')
        
        # Draw each team (no header, more compact)
        win_probs = [regional_probs.get(team, 0) * 100 for team in teams]
        norm_vals = win_probs / np.max(win_probs) if np.max(win_probs) > 0 else win_probs
        
        start_y = 3.4
        row_spacing = 1
        
        for i, team in enumerate(teams):
            y_pos = start_y - i * row_spacing
            
            # Seed number
            # ax.text(seed_x, y_pos, str(i + 1), fontsize=12, weight='bold', ha='center', va='center')
            
            # Logo
            logo = team_logos[team]
            imagebox = OffsetImage(logo, zoom=0.04)
            ab = AnnotationBbox(imagebox, (logo_x, y_pos), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
            
            # Team name
            ax.text(team_x, y_pos, team, fontsize=11, weight='bold', ha='left', va='center')
            
            # Win probability box
            prob = win_probs[i]
            if i == 0:
                total_win_prob += prob
            normalized = norm_vals[i]
            color = cmap(normalized)
            
            box_width = 2.5
            box_height = 0.8
            rect = Rectangle((prob_x - box_width/2, y_pos - box_height/2), 
                            box_width, box_height, 
                            facecolor=color, edgecolor='black', linewidth=0.8, zorder=2)
            ax.add_patch(rect)
            
            text_color = 'white' if normalized > 0.5 else 'black'
            ax.text(prob_x, y_pos, f"{prob:.1f}%", fontsize=11, weight='bold', 
                ha='center', va='center', color=text_color, zorder=3)
            
            # Row separator (subtle)
            if i < 3:
                ax.plot([0.1, 9.9], [y_pos - row_spacing/2, y_pos - row_spacing/2], 
                    'k-', linewidth=0.5, alpha=0.8)

    # Overall titles
    fig.text(0.5, 0.98, "PEAR's Regional Winner Projections", ha='center', fontweight='bold', fontsize=30)
    # fig.text(0.5, 0.96, "2025 NCAA Baseball Tournament Regionals", ha='center', fontsize=14)
    fig.text(0.5, 0.95, "@PEARatings", ha='center', fontweight='bold', fontsize=20)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.02, hspace=0.1, wspace=0.1)
    plt.savefig(f"./PEAR/PEAR Baseball/y2025/Visuals/Regional_Win_Prob/regional_win_prob_{formatted_date}.png", bbox_inches='tight')
    print("Visuals Completed")