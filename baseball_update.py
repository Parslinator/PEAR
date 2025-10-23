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

BASE_URL = "https://www.warrennolan.com"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def get_park_factor(row):
    if row['Location'] == 'Neutral':
        return 1
    return pf_lookup.get(row['home_team'], 1)

def extract_schedule_data(team_name, team_url, session):
    schedule_url = BASE_URL + team_url
    team_schedule = []

    try:
        response = session.get(schedule_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[Error] {team_name} → {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    schedule_lists = soup.find_all("ul", class_="team-schedule")
    if not schedule_lists:
        return []

    schedule_list = schedule_lists[0]

    for game in schedule_list.find_all('li', class_='team-schedule'):
        try:
            # Date
            month = game.find('span', class_='team-schedule__game-date--month')
            day = game.find('span', class_='team-schedule__game-date--day')
            dow = game.find('span', class_='team-schedule__game-date--dow')
            game_date = f"{month.get_text(strip=True)} {day.get_text(strip=True)} ({dow.get_text(strip=True)})"

            # Opponent
            opponent_link = game.select_one('.team-schedule__opp-line-link')
            opponent_name = opponent_link.get_text(strip=True) if opponent_link else ""

            # Location
            location_div = game.find('div', class_='team-schedule__location')
            location_text = location_div.get_text(strip=True) if location_div else ""
            if "VS" in location_text:
                game_location = "Neutral"
            elif "AT" in location_text:
                game_location = "Away"
            else:
                game_location = "Home"

            # Result
            result_info = game.find('div', class_='team-schedule__result')
            result_text = result_info.get_text(strip=True) if result_info else "N/A"

            # Box score
            box_score_table = game.find('table', class_='team-schedule-bottom__box-score')
            home_team = away_team = home_score = away_score = "N/A"

            if box_score_table:
                rows = box_score_table.find_all('tr')
                if len(rows) > 2:
                    away_row = rows[1].find_all('td')
                    home_row = rows[2].find_all('td')
                    away_team = away_row[0].get_text(strip=True)
                    home_team = home_row[0].get_text(strip=True)
                    away_score = away_row[-3].get_text(strip=True)
                    home_score = home_row[-3].get_text(strip=True)

            team_schedule.append([
                team_name, game_date, opponent_name, game_location,
                result_text, home_team, away_team, home_score, away_score
            ])
        except Exception as e:
            print(f"[Parse Error] {team_name} game row → {e}")
            continue

    return team_schedule

# ThreadPool wrapper function
def fetch_all_schedules(elo_df, session, max_workers=12):
    schedule_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_schedule_data, row["Team"], row["Team Link"], session): row["Team"]
            for _, row in elo_df.iterrows()
        }

        for future in as_completed(futures):
            try:
                data = future.result()
                schedule_data.extend(data)
            except Exception as e:
                print(f"[Thread Error] {e}")

    return schedule_data

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
schedule_df['park_factor'] = schedule_df.apply(get_park_factor, axis=1)
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

# Mapping months to numerical values
month_mapping = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

# Function to convert "FEB 14 (FRI)" format to "mm-dd-yyyy"
def convert_date(date_str):
    # Ensure date is a string before splitting
    if isinstance(date_str, pd.Timestamp):
        date_str = date_str.strftime("%b %d (%a)").upper()  # Convert to same format
    
    parts = date_str.split()  # ["FEB", "14", "(FRI)"]
    month = month_mapping[parts[0].upper()]  # Convert month to number
    day = parts[1]  # Extract day
    return f"{month}-{day}-{current_season}"

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

# --- PEAR Win Probability ---
def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

# --- Helper Functions ---
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def scrape_warrennolan_table(url, expected_columns):
    soup = get_soup(url)
    table = soup.find('table', class_='normal-grid alternating-rows stats-table')
    data = []
    if table:
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                name_div = cells[1].find('div', class_='name-subcontainer')
                full_text = name_div.text.strip() if name_div else cells[1].text.strip()
                parts = full_text.split("\n")
                team_name = parts[0].strip()
                conference = parts[1].split("(")[0].strip() if len(parts) > 1 else ""
                data.append([cells[0].text.strip(), team_name, conference])
    return pd.DataFrame(data, columns=expected_columns)

def clean_team_names(df, column='Team'):
    df[column] = df[column].str.replace('State', 'St.', regex=False)
    df[column] = df[column].replace(team_replacements)
    return df

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
    rpi = pd.DataFrame(data, columns=headers).drop(columns=["Previous"])
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
elo_data = clean_team_names(elo_data)
projected_rpi = clean_team_names(projected_rpi)
live_rpi = clean_team_names(live_rpi)

####################### CONFIG #######################

# Must be defined elsewhere in your script:
# - stat_links: dict of stat_name -> URL
# - get_soup(url): function that returns BeautifulSoup of given URL

####################### Core Stat Fetching #######################

def get_stat_dataframe(stat_name):
    if stat_name not in stat_links:
        print(f"Stat '{stat_name}' not found. Available stats: {list(stat_links.keys())}")
        return None

    all_data = []
    page_num = 1

    while page_num < 7:
        url = stat_links[stat_name]
        if page_num > 1:
            url = f"{url}/p{page_num}"

        try:
            soup = get_soup(url)
            table = soup.find("table")
            if not table:
                break

            headers = [th.text.strip() for th in table.find_all("th")]
            data = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                data.append([col.text.strip() for col in cols])

            all_data.extend(data)

        except requests.exceptions.HTTPError:
            break
        except Exception as e:
            print(f"Error for {stat_name}, page {page_num}: {e}")
            break

        page_num += 1

    if all_data:
        df = pd.DataFrame(all_data, columns=headers)
        for col in df.columns:
            if col != "Team":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return None

####################### Threading #######################

def threaded_stat_fetch(stat_names, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stat = {
            executor.submit(get_stat_dataframe, stat): stat
            for stat in stat_names
        }
        results = {}
        for future in as_completed(future_to_stat):
            stat = future_to_stat[future]
            try:
                results[stat] = future.result()
            except Exception as e:
                print(f"Failed to fetch {stat}: {e}")
    return results

####################### Utility #######################

def clean_duplicates(df, group_col, min_col):
    duplicates = df[df.duplicated(group_col, keep=False)]
    filtered = duplicates.loc[duplicates.groupby(group_col)[min_col].idxmin()]
    cleaned = df[~df[group_col].isin(duplicates[group_col])]
    return pd.concat([cleaned, filtered], ignore_index=True)

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

####################### Merging + Final Stats #######################

def clean_and_merge(stats_raw, transforms_dict):
    dfs = []
    for stat, df in stats_raw.items():
        if df is not None and stat in transforms_dict:
            df["Team"] = df["Team"].str.strip()
            df = df.dropna(subset=["Team"])
            df_clean = transforms_dict[stat](df)
            dfs.append(df_clean)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Team", how="inner")

    merged = merged.loc[:, ~merged.columns.duplicated()].sort_values('Team').reset_index(drop=True)
    merged["OPS"] = merged["SLG"] + merged["OBP"]
    merged["PYTHAG"] = round(
        (merged["RS"] ** 1.83) / ((merged["RS"] ** 1.83) + (merged["RA"] ** 1.83)), 3
    )
    return merged

####################### Run It #######################

# Example stat pull
stat_list = list(STAT_TRANSFORMS.keys())
raw_stats = threaded_stat_fetch(stat_list, max_workers=10)
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
raw_woba_stats = threaded_stat_fetch(woba_stats)

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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to use xgboost if available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class BaseballPowerRatingSystem:
    """
    Baseball power rating system that optimizes for multiple targets including ELO_Rank
    """
    
    def __init__(self, use_xgb=HAS_XGB, home_field_advantage=0.8, random_state=42):
        self.use_xgb = use_xgb
        self.hfa = home_field_advantage
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.diagnostics = {}
        
    def _clean_round(self, values, decimals=2):
        """Helper method to ensure clean rounding without floating point artifacts"""
        return np.round(values.astype(float), decimals)
    
    def _fit_regressor(self, X, y, model_name=None):
        """Fit regressor with cross-validation"""
        if self.use_xgb:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=150,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state
            )
        
        model.fit(X, y)
        
        # Store cross-validation score
        if model_name:
            cv_score = np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error'))
            self.diagnostics[f'{model_name}_cv_mae'] = -cv_score
        
        return model
    
    def _prepare_game_data(self, team_data, schedule_df):
        """Prepare game-level data efficiently"""
        team_to_idx = {team: idx for idx, team in enumerate(team_data.index)}
        
        valid_games = (
            schedule_df['home_team'].isin(team_to_idx) & 
            schedule_df['away_team'].isin(team_to_idx)
        )
        
        schedule_clean = schedule_df[valid_games].copy()
        
        h_idx = schedule_clean['home_team'].map(team_to_idx).values
        a_idx = schedule_clean['away_team'].map(team_to_idx).values
        actual_margin = schedule_clean['home_score'].values - schedule_clean['away_score'].values
        actual_total = schedule_clean['home_score'].values + schedule_clean['away_score'].values
        
        # Check if neutral field column exists
        if 'neutral' in schedule_clean.columns:
            hfa = np.where(schedule_clean['neutral'] == False, self.hfa, 0.0)
        else:
            hfa = np.full(len(schedule_clean), self.hfa)
        
        return schedule_clean, h_idx, a_idx, actual_margin, actual_total, hfa
    
    def _optimize_feature_selection(self, team_data, available_features, target_columns):
        """
        Optimize feature selection and weights for each target using differential evolution
        Similar to original approach but with ML models
        """
        best_features = {}
        best_weights = {}
        best_scores = {}
        
        for target_col in target_columns:
            y_target = team_data[target_col].values
            n_features = len(available_features)
            
            def feature_selection_objective(params):
                # First half: binary selections, second half: continuous weights
                feature_selection = np.array(params[:n_features])
                weights = np.array(params[n_features:])
                
                # Get selected features
                selected_mask = feature_selection > 0.5
                selected_indices = np.where(selected_mask)[0]
                
                if len(selected_indices) == 0:
                    return 1.0  # Bad score if no features selected
                
                # Get selected features and normalize weights
                selected_weights = weights[selected_indices]
                selected_weights = selected_weights / np.sum(selected_weights)
                
                # Compute weighted combination of selected features
                X_selected = team_data[[available_features[i] for i in selected_indices]].values
                combined = np.sum(X_selected * selected_weights, axis=1)
                
                # For rank targets, convert to ranks
                if 'rank' in target_col.lower():
                    combined_ranks = pd.Series(combined).rank(ascending=False)
                    corr = spearmanr(combined_ranks, y_target).correlation
                else:
                    corr = spearmanr(combined, y_target).correlation
                
                if np.isnan(corr):
                    corr = 0.0
                
                return -corr  # Minimize negative correlation
            
            # Optimize feature selection
            bounds = [(0, 1)] * n_features + [(0, 1)] * n_features
            result = differential_evolution(
                feature_selection_objective,
                bounds=bounds,
                seed=self.random_state,
                maxiter=1000,
                polish=True,
                strategy='best1bin'
            )
            
            # Extract results
            feature_selection = result.x[:n_features] > 0.5
            weights = result.x[n_features:]
            
            selected_indices = np.where(feature_selection)[0]
            selected_features = [available_features[i] for i in selected_indices]
            selected_weights = weights[selected_indices]
            selected_weights = selected_weights / np.sum(selected_weights)
            
            best_features[target_col] = selected_features
            best_weights[target_col] = dict(zip(selected_features, selected_weights))
            best_scores[target_col] = -result.fun
            
        return best_features, best_weights, best_scores
    
    def _train_rank_models(self, team_data, selected_features, target_columns):
        """Train separate models for each target column using selected features"""
        target_models = {}
        target_predictions = {}
        
        for target_col in target_columns:
            features_for_target = selected_features[target_col]
            
            if len(features_for_target) == 0:
                # Fallback: use mean
                target_predictions[target_col] = np.zeros(len(team_data))
                continue
            
            X_team = team_data[features_for_target].values.astype(float)
            X_team_scaled = self.scaler.fit_transform(X_team)
            
            y_target = team_data[target_col].values
            
            model = self._fit_regressor(X_team_scaled, y_target, f'target_{target_col}')
            target_models[target_col] = model
            predictions = model.predict(X_team_scaled)
            
            # CRITICAL: If this is a rank column (lower is better), invert it to a rating (higher is better)
            if 'rank' in target_col.lower():
                # Convert ranks to ratings: invert so lower rank = higher rating
                # Use negative of the prediction so that rank 1 becomes the highest rating
                target_predictions[target_col] = -predictions
            else:
                # For rating-based targets, use as-is
                target_predictions[target_col] = predictions
            
            # Store individual correlations using the ORIGINAL target values
            if 'rank' in target_col.lower():
                # For ranks: compare predicted ranks vs actual ranks
                pred_ranks = pd.Series(predictions).rank(ascending=True)  # Lower prediction = better rank
                actual_ranks = y_target
                corr = spearmanr(pred_ranks, actual_ranks).correlation
            else:
                # For ratings: direct correlation
                corr = spearmanr(predictions, y_target).correlation
            self.diagnostics[f'{target_col}_model_correlation'] = corr
        
        self.models['target_models'] = target_models
        return target_predictions
    
    def _train_margin_model(self, team_data, selected_features, h_idx, a_idx, actual_margin, hfa):
        """Train game-level margin prediction model using union of all selected features"""
        # Use union of features selected for all targets
        all_selected_features = set()
        for features in selected_features.values():
            all_selected_features.update(features)
        
        if len(all_selected_features) == 0:
            # Fallback: no margin model
            self.models['margin_model'] = None
            return None
        
        margin_features = list(all_selected_features)
        X_team = team_data[margin_features].values.astype(float)
        X_team_scaled = self.scaler.fit_transform(X_team)
        
        game_features = X_team_scaled[h_idx] - X_team_scaled[a_idx]
        X_games = np.column_stack([game_features, hfa])
        
        margin_model = self._fit_regressor(X_games, actual_margin, 'margin')
        self.models['margin_model'] = margin_model
        self.models['margin_features'] = margin_features
        
        # Diagnostics
        margin_pred = margin_model.predict(X_games)
        self.diagnostics['margin_mae'] = mean_absolute_error(actual_margin, margin_pred)
        self.diagnostics['margin_r2'] = r2_score(actual_margin, margin_pred)
        
        return X_games
    
    def _compute_margin_ratings(self, team_data, selected_features):
        """Compute margin-based ratings for each team"""
        if self.models.get('margin_model') is None:
            return np.zeros(len(team_data))
        
        margin_features = self.models['margin_features']
        X_team = team_data[margin_features].values.astype(float)
        X_team_scaled = self.scaler.transform(X_team)
        margin_features_neutral = np.column_stack([X_team_scaled, np.zeros(len(X_team_scaled))])
        margin_ratings = self.models['margin_model'].predict(margin_features_neutral)
        return margin_ratings
    
    def _optimize_multi_target_ensemble(self, target_predictions, margin_ratings, team_data, 
                                       target_columns, h_idx, a_idx, actual_margin, hfa,
                                       mae_weight=0.1, correlation_weight=0.9, 
                                       min_target_weight=0.3):
        """
        Optimize ensemble considering multiple targets and game prediction accuracy
        
        For rank-based targets (like ELO_Rank), we optimize Spearman correlation
        For rating-based targets, we also use Spearman correlation for consistency
        
        Strategy: Maximize correlation first, then minimize MAE as tiebreaker
        
        Parameters:
        -----------
        min_target_weight : float
            Minimum weight for target model (prevents margin model from dominating)
            Default: 0.3 (at least 30% weight on target-based predictions)
        """
        baseline_mae = np.mean(np.abs(actual_margin))
        n_targets = len(target_columns)
        
        # First, evaluate each component separately for diagnostics
        component_correlations = {}
        
        # Target model correlations
        for col in target_columns:
            # target_predictions already has ranks converted to ratings (inverted)
            # So we need to rank these ratings and compare to actual ranks
            if 'rank' in col.lower():
                # Target predictions are now ratings (higher = better)
                # Convert back to ranks for comparison
                predicted_ranks = pd.Series(target_predictions[col]).rank(ascending=False)
                actual_ranks = team_data[col].values
                corr = spearmanr(predicted_ranks, actual_ranks).correlation
            else:
                corr = spearmanr(target_predictions[col], team_data[col].values).correlation
            component_correlations[f'{col}_target_only'] = corr
        
        # Margin model correlation
        for col in target_columns:
            if 'rank' in col.lower():
                # Margin ratings are ratings (higher = better)
                # Convert to ranks for comparison
                rating_ranks = pd.Series(margin_ratings).rank(ascending=False)
                corr = spearmanr(rating_ranks, team_data[col].values).correlation
            else:
                corr = spearmanr(margin_ratings, team_data[col].values).correlation
            component_correlations[f'{col}_margin_only'] = corr
        
        self.diagnostics['component_correlations'] = component_correlations
        
        def multi_objective(params):
            """
            Optimize both target weights and ensemble mixing weight
            
            params: [target_weight_1, ..., target_weight_n, ensemble_weight]
            
            Primary goal: Maximize correlation with targets
            Secondary goal: Minimize game prediction MAE
            """
            if len(params) != n_targets + 1:
                raise ValueError(f"Expected {n_targets + 1} parameters")
            
            target_weights = params[:n_targets]
            ensemble_weight = params[-1]
            
            # Enforce minimum target weight constraint
            ensemble_weight = max(min_target_weight, min(1.0, ensemble_weight))
            
            # Normalize target weights to sum to 1
            target_weights = np.array(target_weights)
            target_weights = np.abs(target_weights) / np.sum(np.abs(target_weights))
            
            # Create composite target prediction (all are now ratings where higher = better)
            composite_pred = sum(w * target_predictions[col] for w, col in zip(target_weights, target_columns))
            
            # Ensemble with margin-based ratings (also higher = better)
            ensemble_ratings = ensemble_weight * composite_pred + (1 - ensemble_weight) * margin_ratings
            
            # Calculate correlation scores with each target
            correlation_scores = []
            for col in target_columns:
                target_values = team_data[col].values
                
                # For rank targets: convert our ratings back to ranks for correlation
                if 'rank' in col.lower():
                    # ensemble_ratings are ratings (higher = better)
                    # Convert to ranks (lower = better) for comparison
                    predicted_ranks = pd.Series(ensemble_ratings).rank(ascending=False)
                    corr = spearmanr(predicted_ranks, target_values).correlation
                else:
                    # For ratings, use direct correlation
                    corr = spearmanr(ensemble_ratings, target_values).correlation
                
                if np.isnan(corr):
                    corr = 0.0
                correlation_scores.append(corr)
            
            # Average correlation across all targets
            avg_correlation = np.mean(correlation_scores)
            
            # Game prediction accuracy (secondary objective)
            game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
            game_mae = np.mean(np.abs(game_pred - actual_margin))
            normalized_mae = game_mae / baseline_mae
            
            # Add penalty for low target weight to enforce minimum
            weight_penalty = 0
            if ensemble_weight < min_target_weight:
                weight_penalty = 10.0 * (min_target_weight - ensemble_weight)
            
            # PRIMARY: Maximize correlation (larger weight)
            # SECONDARY: Minimize MAE (smaller weight as tiebreaker)
            loss = -correlation_weight * avg_correlation + mae_weight * normalized_mae + weight_penalty
            
            return loss
        
        # Set up optimization bounds
        # Target weights: 0 to 2, ensemble weight: min_target_weight to 1
        bounds = [(0, 2)] * n_targets + [(min_target_weight, 1)]
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            multi_objective,
            bounds,
            seed=self.random_state,
            maxiter=300,
            polish=True
        )
        
        optimal_params = result.x
        optimal_target_weights = optimal_params[:n_targets]
        optimal_target_weights = np.abs(optimal_target_weights) / np.sum(np.abs(optimal_target_weights))
        optimal_ensemble_weight = optimal_params[-1]
        
        # Ensure minimum weight is respected
        optimal_ensemble_weight = max(min_target_weight, optimal_ensemble_weight)
        
        return optimal_target_weights, optimal_ensemble_weight
    
    def fit(self, team_data, schedule_df, available_features, target_columns=['ELO_Rank'], 
            mae_weight=0.1, correlation_weight=0.9, rating_scale=5.0, rating_center=0.0, min_target_weight=0.3):
        """
        Fit baseball power rating system with automatic feature selection
        
        Parameters:
        -----------
        team_data : DataFrame
            Team statistics with 'Team' column and features
        schedule_df : DataFrame  
            Game results with home_team, away_team, home_score, away_score
        available_features : list
            ALL available feature columns - system will select best subset
        target_columns : list
            List of target columns to optimize for (e.g., ['ELO_Rank'])
        mae_weight : float
            Weight for game prediction accuracy in optimization
        correlation_weight : float
            Weight for target correlation in optimization
        rating_scale : float
            Target standard deviation for ratings (default 5.0 for college baseball)
            This creates a flexible scale: ~68% of teams within ±5 of center
            Higher values = more spread, lower values = more compressed
        rating_center : float
            Center point for ratings (default 0.0)
        min_target_weight : float
            Minimum weight for target-based model in ensemble (default 0.3)
            Prevents margin model from dominating when features don't correlate well
        """
        # Validate inputs
        for col in target_columns:
            if col not in team_data.columns:
                raise ValueError(f"Target column '{col}' not found in team_data")
        
        for feat in available_features:
            if feat not in team_data.columns:
                raise ValueError(f"Feature '{feat}' not found in team_data")
        
        # Check for Team column
        if 'Team' not in team_data.columns:
            raise ValueError("team_data must contain 'Team' column")
        
        # Set Team as index
        team_data_indexed = team_data.set_index('Team', drop=False)
        
        # Step 1: Feature selection for each target
        print("Optimizing feature selection...")
        selected_features, feature_weights, selection_scores = self._optimize_feature_selection(
            team_data_indexed, available_features, target_columns
        )
        
        # Store feature selection results
        self.selected_features = selected_features
        self.feature_weights = feature_weights
        
        # Print selected features
        for target_col in target_columns:
            print(f"\nSelected features for {target_col}:")
            for feat, weight in sorted(feature_weights[target_col].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {weight*100:.1f}%")
            print(f"  Initial correlation: {selection_scores[target_col]:.4f}")
        
        # Prepare game data
        schedule_clean, h_idx, a_idx, actual_margin, actual_total, hfa = self._prepare_game_data(
            team_data_indexed, schedule_df
        )
        
        # Step 2: Train ML models for each target using selected features
        print("\nTraining ML models...")
        target_predictions = self._train_rank_models(
            team_data_indexed, selected_features, target_columns
        )
        
        # Step 3: Train margin model (optional - can get 0 weight in ensemble)
        print("Training margin prediction model...")
        self._train_margin_model(
            team_data_indexed, selected_features, h_idx, a_idx, actual_margin, hfa
        )
        
        # Compute margin-based ratings
        margin_ratings = self._compute_margin_ratings(team_data_indexed, selected_features)
        
        # Step 4: Optimize ensemble (margin model can get 0 weight)
        print("Optimizing ensemble weights...")
        optimal_target_weights, optimal_ensemble_weight = self._optimize_multi_target_ensemble(
            target_predictions, margin_ratings, team_data_indexed, target_columns,
            h_idx, a_idx, actual_margin, hfa, mae_weight, correlation_weight, min_target_weight
        )
        
        # Create final ensemble ratings
        composite_pred = sum(w * target_predictions[col] 
                           for w, col in zip(optimal_target_weights, target_columns))
        
        ensemble_ratings = (optimal_ensemble_weight * composite_pred + 
                          (1 - optimal_ensemble_weight) * margin_ratings)
        
        # Scale ratings using standard deviation approach (flexible scale)
        # This allows natural expansion/compression based on team quality differences
        ensemble_ratings = ensemble_ratings - ensemble_ratings.mean()
        current_std = ensemble_ratings.std()
        
        if current_std > 0:
            # Scale to target standard deviation
            scaling_factor = rating_scale / current_std
            ensemble_ratings = ensemble_ratings * scaling_factor
        
        # Center at desired point
        ensemble_ratings = ensemble_ratings - ensemble_ratings.mean() + rating_center
        
        # Store results with clean rounding
        self.team_ratings = pd.DataFrame({
            'Team': team_data_indexed['Team'].values,
            'Rating': self._clean_round(ensemble_ratings)
        })
        
        # Add individual component ratings for analysis
        for i, col in enumerate(target_columns):
            component_scaled = target_predictions[col] - target_predictions[col].mean()
            component_std = component_scaled.std()
            if component_std > 0:
                component_scaled = (component_scaled / component_std) * rating_scale
            component_scaled = component_scaled + rating_center
            self.team_ratings[f'{col}_component'] = self._clean_round(component_scaled)
        
        margin_scaled = margin_ratings - margin_ratings.mean()
        margin_std = margin_scaled.std()
        if margin_std > 0:
            margin_scaled = (margin_scaled / margin_std) * rating_scale
        margin_scaled = margin_scaled + rating_center
        self.team_ratings['margin_component'] = self._clean_round(margin_scaled)
        
        # Final diagnostics
        final_correlations = {}
        for col in target_columns:
            if 'rank' in col.lower():
                # ensemble_ratings are ratings (higher = better)
                # Convert to ranks for comparison with actual ranks
                rating_ranks = pd.Series(ensemble_ratings).rank(ascending=False)
                corr = spearmanr(rating_ranks, team_data_indexed[col].values).correlation
            else:
                corr = spearmanr(ensemble_ratings, team_data_indexed[col].values).correlation
            final_correlations[f'final_{col}_correlation'] = corr
        
        game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
        final_mae = np.mean(np.abs(game_pred - actual_margin))
        
        self.diagnostics.update({
            'target_columns': target_columns,
            'selected_features': selected_features,
            'feature_weights': feature_weights,
            'feature_selection_scores': selection_scores,
            'optimal_target_weights': dict(zip(target_columns, optimal_target_weights)),
            'optimal_ensemble_weight': optimal_ensemble_weight,
            'final_game_mae': final_mae,
            'baseline_margin_mae': np.mean(np.abs(actual_margin)),
            'n_games': len(actual_margin),
            'rating_range': ensemble_ratings.max() - ensemble_ratings.min(),
            'rating_std': ensemble_ratings.std(),
            'rating_scale': rating_scale,
            'rating_center': rating_center,
            'home_field_advantage': self.hfa,
            **final_correlations
        })
        
        return self
    
    def predict_game(self, home_team, away_team, neutral=False):
        """Predict margin and scores for a specific game"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted before making predictions")
        
        team_lookup = dict(zip(self.team_ratings['Team'], self.team_ratings['Rating']))
        
        # Get ratings
        home_rating = team_lookup.get(home_team, 0)
        away_rating = team_lookup.get(away_team, 0)
        
        hfa = 0 if neutral else self.hfa
        
        # Calculate predicted margin
        margin = round(home_rating + hfa - away_rating, 2)
        
        # For baseball, typical game total is around 8-10 runs
        # This is a simple model; could be enhanced with offensive/defensive components
        baseline_score = 4.5  # Average runs per team
        
        home_score = round(baseline_score + margin/2, 1)
        away_score = round(baseline_score - margin/2, 1)
        
        # Format output
        if margin < 0:
            favorite = away_team
            margin_str = f'{away_team} by {abs(margin):.2f}'
        else:
            favorite = home_team
            margin_str = f'{home_team} by {margin:.2f}'
        
        return {
            'margin': margin_str,
            'home_score': home_score,
            'away_score': away_score,
            'predicted_score': f'{home_team} {home_score}, {away_team} {away_score}'
        }
    
    def get_rankings(self, n=25):
        """Get top N teams by rating"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted first")
        
        return self.team_ratings.nlargest(n, 'Rating')
    
    def print_diagnostics(self):
        """Print comprehensive model diagnostics"""
        print("\nBaseball Power Rating System Diagnostics")
        print("=" * 60)
        
        # Component correlations (individual performance)
        if 'component_correlations' in self.diagnostics:
            print("\nCOMPONENT CORRELATIONS (individual models):")
            for key, value in self.diagnostics['component_correlations'].items():
                print(f"  {key}: {value:.4f} ({value*100:.1f}%)")
        
        # Feature selection results
        print("\nFEATURE SELECTION:")
        for target_col in self.diagnostics['target_columns']:
            print(f"\n  {target_col}:")
            n_selected = len(self.diagnostics['selected_features'][target_col])
            print(f"    Features selected: {n_selected}")
            print(f"    Feature selection correlation: {self.diagnostics['feature_selection_scores'][target_col]:.4f}")
            
            for feat, weight in sorted(self.diagnostics['feature_weights'][target_col].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"      {feat}: {weight*100:.1f}%")
        
        # Target information
        print(f"\nTARGET OPTIMIZATION:")
        print(f"  Target columns: {self.diagnostics['target_columns']}")
        print("\n  Optimal target weights:")
        for target, weight in self.diagnostics['optimal_target_weights'].items():
            print(f"    {target}: {weight:.3f}")
        
        ensemble_weight = self.diagnostics['optimal_ensemble_weight']
        margin_weight = 1 - ensemble_weight
        print(f"\n  Ensemble composition:")
        print(f"    Target model weight: {ensemble_weight:.3f}")
        print(f"    Margin model weight: {margin_weight:.3f}")
        
        if margin_weight < 0.01:
            print(f"    Note: Margin model essentially not used (weight ~0)")
        elif ensemble_weight < 0.2:
            print(f"    WARNING: Target model has very low weight - features may not correlate well with target")
        
        # Model performance
        print(f"\nGAME PREDICTION PERFORMANCE:")
        print(f"  Home Field Advantage: {self.diagnostics['home_field_advantage']:.3f} runs")
        print(f"  Final MAE: {self.diagnostics['final_game_mae']:.3f} runs")
        print(f"  Baseline MAE: {self.diagnostics['baseline_margin_mae']:.3f} runs")
        improvement = (1 - self.diagnostics['final_game_mae']/self.diagnostics['baseline_margin_mae'])
        print(f"  Improvement: {improvement:.1%}")
        
        # Target correlations
        print(f"\nTARGET CORRELATIONS (Spearman) - Final Ensemble:")
        for key, value in self.diagnostics.items():
            if 'final_' in key and '_correlation' in key:
                target_name = key.replace('final_', '').replace('_correlation', '')
                print(f"  {target_name}: {value:.4f} ({value*100:.1f}%)")
        
        print(f"\nDATA SUMMARY:")
        print(f"  Games analyzed: {self.diagnostics['n_games']}")
        print(f"  Rating scale (target std): {self.diagnostics['rating_scale']:.2f}")
        print(f"  Actual rating std: {self.diagnostics['rating_std']:.2f}")
        print(f"  Rating range: {self.diagnostics['rating_range']:.2f}")
        print(f"  Rating center: {self.diagnostics['rating_center']:.2f}\n")
        
        # Show top teams
        print("=" * 60)
        print("TOP 10 TEAMS:")
        print(self.get_rankings(10).to_string(index=False))


# --- USAGE EXAMPLE ---
def build_baseball_power_ratings(team_data, schedule_df, available_features, 
                                 target_columns=['ELO_Rank'], mae_weight=0.1, 
                                 correlation_weight=0.9, rating_scale=5.0, 
                                 rating_center=0.0, min_target_weight=0.3,
                                 home_field_advantage=0.8):
    """
    Build baseball power ratings with automatic feature selection
    
    Parameters:
    -----------
    team_data : DataFrame
        Must contain 'Team' column and all available_features
    schedule_df : DataFrame
        Must contain: home_team, away_team, home_score, away_score
        Optional: neutral (True/False for neutral site games)
    available_features : list
        ALL available features - system will select best subset
        Example: ['BB%', 'OPS', 'wRC+', 'PYTHAG', 'fWAR', ...]
    target_columns : list
        Target columns to optimize for (e.g., ['ELO_Rank'])
    mae_weight : float
        Weight for game prediction accuracy in optimization
        Set to 0 to only optimize for target correlation
    correlation_weight : float
        Weight for target correlation in optimization (default 0.9)
        Higher values prioritize matching target rankings
    rating_scale : float
        Target standard deviation for ratings (default 5.0)
        Creates flexible scale: ~68% of teams within ±5 of center
        Typical values:
        - 3.0: Compressed scale (small differences matter more)
        - 5.0: Moderate scale (good for college baseball)
        - 10.0: Expanded scale (emphasizes large differences)
    rating_center : float
        Center point for ratings (default 0.0)
        Could use 100 for a 0-200 scale, 50 for 0-100, etc.
    min_target_weight : float
        Minimum weight for target-based model in ensemble (default 0.3)
    home_field_advantage : float
        Home field advantage in runs (default 0.8)
        Typical range: 0.5 to 1.2 runs for college baseball
    
    Returns:
    --------
    result_data : DataFrame
        Original team_data with Rating column added
    diagnostics : dict
        Model performance metrics and selected features
    system : BaseballPowerRatingSystem
        Fitted system for making predictions
    """
    system = BaseballPowerRatingSystem(home_field_advantage=home_field_advantage)
    system.fit(team_data, schedule_df, available_features, target_columns, 
               mae_weight=mae_weight, correlation_weight=correlation_weight,
               rating_scale=rating_scale, rating_center=rating_center, 
               min_target_weight=min_target_weight)
    
    # Merge ratings back to original team_data
    result_data = team_data.merge(
        system.team_ratings[['Team', 'Rating']], 
        on='Team', 
        how='left'
    )
    
    return result_data, system.diagnostics, system
    
def hyperparameter_search_for_mae(team_data, schedule_df, available_features,
                                   target_columns=['ELO_Rank'], n_trials=100, n_jobs=-1, verbose=True):
    """
    Perform parallelized random search to find settings that minimize game prediction MAE
    while maintaining good correlation with target
    
    Parameters:
    -----------
    team_data : DataFrame
        Team statistics with 'Team' column and features
    schedule_df : DataFrame
        Game results with home_team, away_team, home_score, away_score
    available_features : list
        ALL available features - system will select best subset
    target_columns : list
        Target columns to optimize for (e.g., ['ELO_Rank'])
    n_trials : int
        Number of random combinations to try (default: 100)
    n_jobs : int
        Number of parallel jobs (-1 uses all available cores, default: -1)
    verbose : bool
        Print progress during search
    
    Returns:
    --------
    dict with keys:
        'best_params': Best hyperparameter combination found
        'best_system': Fitted system with best parameters
        'best_result_data': Result data with ratings
        'results_df': DataFrame with all trials and their performance
        'best_mae': Best MAE achieved
    """
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr
    import numpy as np
    import pandas as pd
    import time
    
    print("PARALLELIZED RANDOM SEARCH FOR BEST GAME PREDICTION MAE")
    print("=" * 70)
    
    # Clean schedule data once
    model_schedule = schedule_df.drop_duplicates(
        subset=['Date', 'home_team', 'away_team', 'home_score', 'away_score']
    ).dropna().reset_index(drop=True)
    
    # Define search space
    search_space = {
        'rating_scale': [1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
        'mae_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'correlation_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'min_target_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'home_field_advantage': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    }
    
    print(f"Random search space: {n_trials} random combinations")
    print(f"  rating_scale: {len(search_space['rating_scale'])} values")
    print(f"  mae_weight: {len(search_space['mae_weight'])} values")
    print(f"  correlation_weight: {len(search_space['correlation_weight'])} values")
    print(f"  min_target_weight: {len(search_space['min_target_weight'])} values")
    print(f"  home_field_advantage: {len(search_space['home_field_advantage'])} values")
    
    # Generate random combinations
    np.random.seed(42)  # For reproducibility
    random_combinations = []
    for _ in range(n_trials):
        combo = (
            np.random.choice(search_space['rating_scale']),
            np.random.choice(search_space['mae_weight']),
            np.random.choice(search_space['correlation_weight']),
            np.random.choice(search_space['min_target_weight']),
            np.random.choice(search_space['home_field_advantage'])
        )
        random_combinations.append(combo)
    
    # Determine number of jobs
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} parallel jobs")
    print()
    
    def evaluate_params(idx, params_tuple):
        """Evaluate a single parameter combination"""
        rating_scale, mae_weight, correlation_weight, min_target_weight, home_field_advantage = params_tuple
        
        params = {
            'rating_scale': rating_scale,
            'mae_weight': mae_weight,
            'correlation_weight': correlation_weight,
            'min_target_weight': min_target_weight,
            'home_field_advantage': home_field_advantage
        }
        
        try:
            # Suppress print statements during parallel execution
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Build system with these parameters
            result_data, diagnostics, system = build_baseball_power_ratings(
                team_data=team_data,
                schedule_df=model_schedule,
                available_features=available_features,
                target_columns=target_columns,
                rating_scale=params['rating_scale'],
                mae_weight=params['mae_weight'],
                correlation_weight=params['correlation_weight'],
                rating_center=0.0,
                min_target_weight=params['min_target_weight'],
                home_field_advantage=params['home_field_advantage']
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Extract metrics
            mae = diagnostics['final_game_mae']
            
            # Get correlation for each target
            correlations = {}
            for col in target_columns:
                rating_ranks = result_data['Rating'].rank(ascending=False)
                if 'rank' in col.lower():
                    corr = spearmanr(rating_ranks, result_data[col]).correlation
                else:
                    corr = spearmanr(result_data['Rating'], result_data[col]).correlation
                correlations[col] = corr
            
            avg_correlation = np.mean(list(correlations.values()))
            
            # Store results
            trial_result = {
                'trial': idx + 1,
                'mae': mae,
                'avg_correlation': avg_correlation,
                **params,
                **{f'{col}_correlation': correlations[col] for col in target_columns}
            }
            
            return trial_result
            
        except Exception as e:
            # Restore stdout if error
            sys.stdout = old_stdout
            return {
                'trial': idx + 1,
                'mae': np.nan,
                'avg_correlation': np.nan,
                'error': str(e),
                **params
            }
    
    start_time = time.time()
    
    # Run parallel random search
    print("Running parallel random search...")
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_params)(idx, params) 
        for idx, params in enumerate(random_combinations)
    )
    
    elapsed_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out failed trials
    valid_results = results_df[~results_df['mae'].isna()].copy()
    valid_results = valid_results.sort_values('mae')
    
    # Find best based on MAE (with correlation threshold)
    best_candidates = valid_results[valid_results['avg_correlation'] > 0.985]
    
    if len(best_candidates) > 0:
        best_trial = best_candidates.iloc[0]
        best_params = {
            'rating_scale': best_trial['rating_scale'],
            'mae_weight': best_trial['mae_weight'],
            'correlation_weight': best_trial['correlation_weight'],
            'min_target_weight': best_trial['min_target_weight'],
            'home_field_advantage': best_trial['home_field_advantage']
        }
        best_mae = best_trial['mae']
        
        # Rebuild best system for return
        print("\nRebuilding best system...")
        best_result_data, _, best_system = build_baseball_power_ratings(
            team_data=team_data,
            schedule_df=model_schedule,
            available_features=available_features,
            target_columns=target_columns,
            **best_params,
            rating_center=0.0
        )
    else:
        best_params = None
        best_mae = None
        best_system = None
        best_result_data = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("RANDOM SEARCH COMPLETE")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Successful trials: {len(valid_results)}/{n_trials}")
    print(f"Average time per trial: {elapsed_time/n_trials:.2f} seconds")
    
    if best_params is not None:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS (Lowest MAE with Correlation > 0.985):")
        print(f"  Rating Scale:          {best_params['rating_scale']:.2f}")
        print(f"  MAE Weight:            {best_params['mae_weight']:.2f}")
        print(f"  Correlation Weight:    {best_params['correlation_weight']:.2f}")
        print(f"  Min Target Weight:     {best_params['min_target_weight']:.2f}")
        print(f"  Home Field Advantage:  {best_params['home_field_advantage']:.2f} runs")
        
        print(f"\nPERFORMANCE:")
        print(f"  Best MAE:              {best_mae:.3f} runs")
        print(f"  Avg Correlation:       {best_trial['avg_correlation']:.4f}")
        for col in target_columns:
            print(f"  {col} Correlation:    {best_trial[f'{col}_correlation']:.4f}")
        
        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY MAE:")
        display_cols = ['trial', 'mae', 'avg_correlation', 'rating_scale', 
                       'mae_weight', 'correlation_weight', 'min_target_weight', 'home_field_advantage']
        if len(valid_results) > 0:
            print(valid_results[display_cols].head(10).to_string(index=False))
        
        print("\n" + "=" * 70)
        print("PARAMETER ANALYSIS (Top 10):")
        
        # Analyze patterns in top performers
        if len(valid_results) >= 10:
            top10 = valid_results.head(10)
            
            # Show distribution of each parameter
            print("\nRating Scale distribution:")
            scale_counts = top10['rating_scale'].value_counts().sort_index()
            for val, count in scale_counts.items():
                print(f"  {val:.1f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMAE Weight distribution:")
            mae_counts = top10['mae_weight'].value_counts().sort_index()
            for val, count in mae_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nCorrelation Weight distribution:")
            corr_counts = top10['correlation_weight'].value_counts().sort_index()
            for val, count in corr_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMin Target Weight distribution:")
            target_counts = top10['min_target_weight'].value_counts().sort_index()
            for val, count in target_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nHome Field Advantage distribution:")
            hfa_counts = top10['home_field_advantage'].value_counts().sort_index()
            for val, count in hfa_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nTop 10 averages:")
            print(f"  Avg rating_scale:          {top10['rating_scale'].mean():.2f}")
            print(f"  Avg mae_weight:            {top10['mae_weight'].mean():.2f}")
            print(f"  Avg correlation_weight:    {top10['correlation_weight'].mean():.2f}")
            print(f"  Avg min_target_weight:     {top10['min_target_weight'].mean():.2f}")
            print(f"  Avg home_field_advantage:  {top10['home_field_advantage'].mean():.2f}")
    else:
        print("\nNo valid configurations found with correlation > 0.985.")
        if len(valid_results) > 0:
            print("Best configuration without correlation threshold:")
            best_any = valid_results.iloc[0]
            print(f"  MAE: {best_any['mae']:.3f}")
            print(f"  Correlation: {best_any['avg_correlation']:.4f}")
            print(f"  Home Field Advantage: {best_any['home_field_advantage']:.2f}")
    
    print("=" * 70)
    
    return {
        'best_params': best_params,
        'best_system': best_system,
        'best_result_data': best_result_data,
        'results_df': valid_results,
        'best_mae': best_mae
    }
    
    def evaluate_params(idx, params_tuple):
        """Evaluate a single parameter combination"""
        rating_scale, mae_weight, correlation_weight, min_target_weight = params_tuple
        
        params = {
            'rating_scale': rating_scale,
            'mae_weight': mae_weight,
            'correlation_weight': correlation_weight,
            'min_target_weight': min_target_weight
        }
        
        try:
            # Suppress print statements during parallel execution
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Build system with these parameters
            result_data, diagnostics, system = build_baseball_power_ratings(
                team_data=team_data,
                schedule_df=model_schedule,
                available_features=available_features,
                target_columns=target_columns,
                rating_scale=params['rating_scale'],
                mae_weight=params['mae_weight'],
                correlation_weight=params['correlation_weight'],
                rating_center=0.0,
                min_target_weight=params['min_target_weight']
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Extract metrics
            mae = diagnostics['final_game_mae']
            
            # Get correlation for each target
            correlations = {}
            for col in target_columns:
                rating_ranks = result_data['Rating'].rank(ascending=False)
                if 'rank' in col.lower():
                    corr = spearmanr(rating_ranks, result_data[col]).correlation
                else:
                    corr = spearmanr(result_data['Rating'], result_data[col]).correlation
                correlations[col] = corr
            
            avg_correlation = np.mean(list(correlations.values()))
            
            # Store results
            trial_result = {
                'trial': idx + 1,
                'mae': mae,
                'avg_correlation': avg_correlation,
                **params,
                **{f'{col}_correlation': correlations[col] for col in target_columns}
            }
            
            return trial_result
            
        except Exception as e:
            # Restore stdout if error
            sys.stdout = old_stdout
            return {
                'trial': idx + 1,
                'mae': np.nan,
                'avg_correlation': np.nan,
                'error': str(e),
                **params
            }
    
    start_time = time.time()
    
    # Run parallel grid search
    print("Running parallel grid search...")
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_params)(idx, params) 
        for idx, params in enumerate(all_combinations)
    )
    
    elapsed_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out failed trials
    valid_results = results_df[~results_df['mae'].isna()].copy()
    valid_results = valid_results.sort_values('mae')
    
    # Find best based on MAE (with correlation threshold)
    best_candidates = valid_results[valid_results['avg_correlation'] > 0.75]
    
    if len(best_candidates) > 0:
        best_trial = best_candidates.iloc[0]
        best_params = {
            'rating_scale': best_trial['rating_scale'],
            'mae_weight': best_trial['mae_weight'],
            'correlation_weight': best_trial['correlation_weight'],
            'min_target_weight': best_trial['min_target_weight']
        }
        best_mae = best_trial['mae']
        
        # Rebuild best system for return
        print("\nRebuilding best system...")
        best_result_data, _, best_system = build_baseball_power_ratings(
            team_data=team_data,
            schedule_df=model_schedule,
            available_features=available_features,
            target_columns=target_columns,
            **best_params,
            rating_center=0.0
        )
    else:
        best_params = None
        best_mae = None
        best_system = None
        best_result_data = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Successful trials: {len(valid_results)}/{n_trials}")
    print(f"Average time per trial: {elapsed_time/n_trials:.2f} seconds")
    
    if best_params is not None:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS (Lowest MAE with Correlation > 0.75):")
        print(f"  Rating Scale:        {best_params['rating_scale']:.2f}")
        print(f"  MAE Weight:          {best_params['mae_weight']:.2f}")
        print(f"  Correlation Weight:  {best_params['correlation_weight']:.2f}")
        print(f"  Min Target Weight:   {best_params['min_target_weight']:.2f}")
        
        print(f"\nPERFORMANCE:")
        print(f"  Best MAE:            {best_mae:.3f} runs")
        print(f"  Avg Correlation:     {best_trial['avg_correlation']:.4f}")
        for col in target_columns:
            print(f"  {col} Correlation:  {best_trial[f'{col}_correlation']:.4f}")
        
        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY MAE:")
        display_cols = ['trial', 'mae', 'avg_correlation', 'rating_scale', 
                       'mae_weight', 'correlation_weight', 'min_target_weight']
        if len(valid_results) > 0:
            print(valid_results[display_cols].head(10).to_string(index=False))
        
        print("\n" + "=" * 70)
        print("PARAMETER ANALYSIS (Top 10):")
        
        # Analyze patterns in top performers
        if len(valid_results) >= 10:
            top10 = valid_results.head(10)
            
            # Show distribution of each parameter
            print("\nRating Scale distribution:")
            scale_counts = top10['rating_scale'].value_counts().sort_index()
            for val, count in scale_counts.items():
                print(f"  {val:.1f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMAE Weight distribution:")
            mae_counts = top10['mae_weight'].value_counts().sort_index()
            for val, count in mae_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nCorrelation Weight distribution:")
            corr_counts = top10['correlation_weight'].value_counts().sort_index()
            for val, count in corr_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMin Target Weight distribution:")
            target_counts = top10['min_target_weight'].value_counts().sort_index()
            for val, count in target_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nTop 10 averages:")
            print(f"  Avg rating_scale:        {top10['rating_scale'].mean():.2f}")
            print(f"  Avg mae_weight:          {top10['mae_weight'].mean():.2f}")
            print(f"  Avg correlation_weight:  {top10['correlation_weight'].mean():.2f}")
            print(f"  Avg min_target_weight:   {top10['min_target_weight'].mean():.2f}")
    else:
        print("\nNo valid configurations found with correlation > 0.75.")
        if len(valid_results) > 0:
            print("Best configuration without correlation threshold:")
            best_any = valid_results.iloc[0]
            print(f"  MAE: {best_any['mae']:.3f}")
            print(f"  Correlation: {best_any['avg_correlation']:.4f}")
    
    print("=" * 70)
    
    return {
        'best_params': best_params,
        'best_system': best_system,
        'best_result_data': best_result_data,
        'results_df': valid_results,
        'best_mae': best_mae
    }

def compare_with_simple_approach(team_data, schedule_df, available_features, 
                                target_column='ELO_Rank'):
    """
    Compare the advanced ML system with a simple weighted feature approach
    
    Returns both correlations and game prediction MAE for comparison
    """
    print("COMPARISON: ML System vs Simple Weighted Approach")
    print("=" * 60)
    
    # Clean schedule data once
    model_schedule = schedule_df.drop_duplicates(
        subset=['Date', 'home_team', 'away_team', 'home_score', 'away_score']
    ).dropna().reset_index(drop=True)
    
    # Simple approach (original method)
    print("\n1. Running Simple Weighted Approach...")
    target = team_data[target_column].values
    
    def simple_objective(weights_raw):
        feature_selection = np.array(weights_raw[:len(available_features)])
        weights = np.array(weights_raw[len(available_features):])
        
        selected_features = [available_features[i] for i in range(len(available_features)) 
                           if feature_selection[i] > 0.5]
        
        if len(selected_features) == 0:
            return 1
        
        weights /= np.sum(weights)
        combined = sum(w * team_data[feat] for w, feat in zip(weights, selected_features))
        ranks = combined.rank(ascending=False)
        corr, _ = spearmanr(ranks, target)
        return -corr
    
    bounds = [(0, 1)] * len(available_features) + [(0, 1)] * len(available_features)
    result = differential_evolution(simple_objective, bounds=bounds, 
                                   strategy='best1bin', maxiter=1000, 
                                   polish=True, seed=42)
    
    feature_selection = np.array(result.x[:len(available_features)]) > 0.5
    weights = np.array(result.x[len(available_features):])
    weights /= np.sum(weights)
    
    selected_features_simple = [available_features[i] for i in range(len(available_features)) 
                               if feature_selection[i]]
    simple_corr = -result.fun
    
    # Calculate simple approach ratings (scaled to same scale for fair comparison)
    simple_combined = sum(w * team_data[feat] for w, feat in zip(weights, selected_features_simple))
    simple_ratings = simple_combined - simple_combined.mean()
    simple_std = simple_ratings.std()
    if simple_std > 0:
        simple_ratings = (simple_ratings / simple_std) * 5.0  # Scale to std=5.0
    
    # Calculate MAE for simple approach
    team_data_indexed = team_data.set_index('Team', drop=False)
    team_to_idx = {team: idx for idx, team in enumerate(team_data_indexed.index)}
    
    valid_games = (
        model_schedule['home_team'].isin(team_to_idx) & 
        model_schedule['away_team'].isin(team_to_idx)
    )
    schedule_clean = model_schedule[valid_games].copy()
    
    h_idx = schedule_clean['home_team'].map(team_to_idx).values
    a_idx = schedule_clean['away_team'].map(team_to_idx).values
    actual_margin = schedule_clean['home_score'].values - schedule_clean['away_score'].values
    
    # Check if neutral field column exists
    if 'neutral' in schedule_clean.columns:
        hfa = np.where(schedule_clean['neutral'] == False, 0.3, 0.0)
    else:
        hfa = np.full(len(schedule_clean), 0.3)
    
    simple_ratings_array = simple_ratings.values
    simple_pred_margin = simple_ratings_array[h_idx] + hfa - simple_ratings_array[a_idx]
    simple_mae = np.mean(np.abs(simple_pred_margin - actual_margin))
    
    print(f"   Features selected: {len(selected_features_simple)}")
    print(f"   Correlation: {simple_corr:.4f} ({simple_corr*100:.1f}%)")
    print(f"   Game prediction MAE: {simple_mae:.3f} runs")
    
    # Advanced ML approach
    print("\n2. Running Advanced ML System...")
    result_data, diagnostics, system = build_baseball_power_ratings(
        team_data=team_data,
        schedule_df=model_schedule,
        available_features=available_features,
        target_columns=[target_column],   
        rating_scale=4.0,
        mae_weight=0.0,           # Pure correlation optimization
        min_target_weight=0.5      # Pure correlation optimization
    )
    
    rating_ranks = result_data['Rating'].rank(ascending=False)
    ml_corr = spearmanr(rating_ranks, result_data[target_column]).correlation
    ml_mae = diagnostics['final_game_mae']
    
    # Comparison summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("\nCorrelation with Target:")
    print(f"  Simple Approach:  {simple_corr:.4f} ({simple_corr*100:.1f}%)")
    print(f"  ML System:        {ml_corr:.4f} ({ml_corr*100:.1f}%)")
    print(f"  Improvement:      {(ml_corr - simple_corr):.4f} ({(ml_corr - simple_corr)*100:.2f}%)")
    
    if ml_corr > simple_corr:
        pct_better = ((ml_corr - simple_corr) / simple_corr) * 100
        print(f"  Relative gain:    {pct_better:.2f}% better")
    
    print("\nGame Prediction MAE:")
    print(f"  Simple Approach:  {simple_mae:.3f} runs")
    print(f"  ML System:        {ml_mae:.3f} runs")
    mae_improvement = simple_mae - ml_mae
    print(f"  Improvement:      {mae_improvement:.3f} runs ({(mae_improvement/simple_mae)*100:.1f}%)")
    
    if ml_mae < simple_mae:
        print(f"  ML System predicts games {(simple_mae/ml_mae - 1)*100:.1f}% more accurately")
    else:
        print(f"  Note: Simple approach predicted games better")
    
    print("=" * 60)
    
    return {
        'simple_correlation': simple_corr,
        'ml_correlation': ml_corr,
        'correlation_improvement': ml_corr - simple_corr,
        'simple_mae': simple_mae,
        'ml_mae': ml_mae,
        'mae_improvement': mae_improvement,
        'system': system,
        'result_data': result_data
    }

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

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location):
    if location != "Neutral":
        home_pr += 0.3
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    spread = round(adjust_home_pr(elo_win_prob) + home_pr - away_pr, 2)
    return spread, elo_win_prob

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

####################### Expected Wins #######################

def calculate_expected_wins(group):
    # Initialize a variable to accumulate expected wins
    expected_wins = 0
    schedule_wins = 0
    schedule_losses = 0
    
    # Iterate over the rows of the group
    for _, row in group.iterrows():
        if row['Team'] == row['home_team']:
            expected_wins += row['home_win_prob']
            if row['home_score'] > row['away_score']:
                schedule_wins += 1
            else:
                schedule_losses += 1
        else:
            expected_wins += 1 - row['home_win_prob']
            if row['away_score'] > row['home_score']:
                schedule_wins += 1
            else:
                schedule_losses += 1
    
    # Return the total expected_wins for this group
    return pd.Series({'Team': group['Team'].iloc[0], 'expected_wins': expected_wins, 'Wins':schedule_wins, 'Losses':schedule_losses})

# Group by 'Team' and apply the calculation
team_expected_wins = completed_schedule.groupby('Team').apply(calculate_expected_wins).reset_index(drop=True)

####################### Strength of Schedule #######################

def calculate_average_expected_wins(group, average_team):
    total_expected_wins = 0

    for _, row in group.iterrows():
        if row['Team'] == row['home_team']:
            total_expected_wins += PEAR_Win_Prob(average_team, row['away_rating'], row['Location']) / 100
        else:
            total_expected_wins += 1 - PEAR_Win_Prob(row['home_rating'], average_team, row['Location']) / 100

    avg_expected_wins = total_expected_wins / len(group)

    return pd.Series({'Team': group['Team'].iloc[0], 'avg_expected_wins': avg_expected_wins, 'total_expected_wins':total_expected_wins})

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

def calculate_kpi(completed_schedule, ending_data):
    # Precompute lookup dictionaries for faster rank access
    rank_lookup = {team: rank for rank, team in enumerate(ending_data["Team"])}
    default_rank = len(ending_data)

    def get_rank(team):
        return rank_lookup.get(team, default_rank)

    total_teams = len(ending_data)
    kpi_scores = []

    for game in completed_schedule.itertuples(index=False):
        team = game.Team
        opponent = game.Opponent
        home_team = game.home_team

        team_rank = get_rank(team)
        opponent_rank = get_rank(opponent)

        # Rank-based strength (inverted)
        opponent_strength_win = 1 - (opponent_rank / (total_teams + 1))
        opponent_strength_loss = 1 - opponent_strength_win  # Equivalent to (opponent_rank / (total_teams + 1))

        # Determine if team was home or away
        is_home = team == home_team

        # Calculate margin (positive if team won)
        margin = game.home_score - game.away_score
        if not is_home:
            margin = -margin

        # Win/loss factor
        result_multiplier = 1.5 if margin > 0 else -1.5

        # Margin factor
        capped_margin = min(abs(margin), 20)
        margin_factor = 1 + (capped_margin / 20) if margin > 0 else max(0.1, 1 - (capped_margin / 20))
        opponent_strength = opponent_strength_win if margin > 0 else opponent_strength_loss

        # Team strength factor
        team_strength_adj = 1 - (team_rank / (total_teams + 1))

        # Adjusted KPI formula
        adj_grv = (opponent_strength * result_multiplier * margin_factor / 1.5) * (1 + (team_strength_adj / 2))
        kpi_scores.append((team, adj_grv))

    # Convert to DataFrame and aggregate
    kpi_df = pd.DataFrame(kpi_scores, columns=["Team", "KPI_Score"])
    kpi_avg = kpi_df.groupby("Team", as_index=False)["KPI_Score"].mean()

    return kpi_avg

kpi_results = calculate_kpi(completed_schedule, ending_data).sort_values('KPI_Score', ascending=False).reset_index(drop=True)

####################### Resume Quality #######################

def calculate_resume_quality(group, bubble_team_rating):
    results = []
    resume_quality = 0
    for _, row in group.iterrows():
        team = row['Team']
        is_home = row["home_team"] == team
        is_away = row["away_team"] == team
        opponent_rating = row["away_rating"] if is_home else row["home_rating"]
        if row["Location"] == "Away":
            win_prob = 1 - PEAR_Win_Prob(opponent_rating, bubble_team_rating, row['Location']) / 100    
        else:
            win_prob = PEAR_Win_Prob(bubble_team_rating, opponent_rating, row['Location']) / 100
        team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
        if team_won:
            resume_quality += (1-win_prob)
        else:
            resume_quality -= win_prob
    resume_quality = resume_quality / len(group)
    results.append({"Team": team, "resume_quality": resume_quality})
    return pd.DataFrame(results)

def calculate_game_resume_quality(row, one_seed_rating):
    """Calculate resume quality for a single game."""
    team = row["Team"]
    is_home = row["home_team"] == team
    is_away = row["away_team"] == team
    opponent_rating = row["away_rating"] if is_home else row["home_rating"]
    if row["Location"] == "Away":
        win_prob = 1 - PEAR_Win_Prob(opponent_rating, one_seed_rating, row['Location']) / 100    
    else:
        win_prob = PEAR_Win_Prob(one_seed_rating, opponent_rating, row['Location']) / 100

    team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
    
    return (1 - win_prob) if team_won else -win_prob

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

def calculate_net(weights):
    w_rating, w_sos = weights
    w_rqi = 1 - (w_rating + w_sos)
    
    if w_rqi < 0 or w_rqi > 1:
        return float('inf')

    stats_and_metrics['NET_Score'] = (
        w_rating * stats_and_metrics['Norm_Rating'] +
        w_rqi * stats_and_metrics['Norm_RQI'] +
        w_sos * stats_and_metrics['Norm_SOS']
    )
    stats_and_metrics['NET'] = stats_and_metrics['NET_Score'].rank(ascending=False).astype(int)
    stats_and_metrics['combined_rank'] = stats_and_metrics['ELO_Rank']
    spearman_corr = stats_and_metrics[['NET', 'combined_rank']].corr(method='spearman').iloc[0,1]

    return -spearman_corr
bounds = [(0,0.1), (0,0.15)]  
result = differential_evolution(calculate_net, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42)
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

def calculate_quadrant_records(completed_schedule, stats_and_metrics):
    # Precompute NET rankings lookup for fast access
    net_lookup = stats_and_metrics.set_index('Team')['NET'].to_dict()
    default_net = 300

    # Quadrant thresholds by location
    quadrant_thresholds = {
        "Home": [25, 50, 100, 307],
        "Neutral": [40, 80, 160, 307],
        "Away": [60, 120, 240, 307]
    }

    records = []

    # Group by team
    for team, group in completed_schedule.groupby('Team'):
        counts = {f'Q{i}_win': 0 for i in range(1, 5)}
        counts.update({f'Q{i}_loss': 0 for i in range(1, 5)})

        for row in group.itertuples(index=False):
            opponent = row.Opponent
            location = row.Location

            # Get opponent NET ranking
            opponent_net = net_lookup.get(opponent, default_net)

            # Determine if team won
            team_won = (
                (row.Team == row.home_team and row.home_score > row.away_score) or
                (row.Team == row.away_team and row.away_score > row.home_score)
            )

            # Determine quadrant
            thresholds = quadrant_thresholds[location]
            quadrant = next((i + 1 for i, val in enumerate(thresholds) if opponent_net <= val), 4)

            # Increment win/loss count
            result_key = f'Q{quadrant}_win' if team_won else f'Q{quadrant}_loss'
            counts[result_key] += 1

        # Build final formatted record: "wins-losses"
        record = {"Team": team}
        for i in range(1, 5):
            record[f"Q{i}"] = f"{counts[f'Q{i}_win']}-{counts[f'Q{i}_loss']}"

        records.append(record)

    # Convert to DataFrame and merge
    quadrant_record_df = pd.DataFrame(records)
    return pd.merge(stats_and_metrics, quadrant_record_df, on='Team', how='left')

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

def game_sort_key(result):
    if result.startswith(("W", "L")):
        return (0, None)  # Completed games
    elif result.startswith(("Bot", "Top", "Middle", "End")):
        return (1, None)  # Ongoing games
    elif result[0].isdigit():  # Upcoming games with time
        try:
            return (2, datetime.strptime(result, "%I:%M %p"))  # Convert time to sortable format
        except ValueError:
            return (2, None)  # If parsing fails, treat as unknown
    elif result.startswith("T"):  # TBA games
        return (3, None)
    elif result.startswith("C"):  # Cancelled games
        return (4, None)
    return (5, None)  # Any other cases

def process_result(row):
    result = row["Result"]
    
    if result.startswith("W"):
        # Replace 'W' with team name and space
        return re.sub(r"^W", row["Team"] + " ", result)
    
    elif result.startswith("L"):
        # Match pattern like 'L3-5', extract and swap scores
        match = re.match(r"L(\d+) - (\d+)", result)
        if match:
            return f"{row['Opponent']} {match.group(2)} - {match.group(1)}"

    return result  # In case it's not W or L

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

def PEAR_Win_Prob(home_pr, away_pr, location = "Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)  # More precision, rounded later in output

def remaining_games_rq(row, one_seed_rating):
    """Calculate resume quality for a single game."""
    team = row["Team"]
    location = row['Location']
    is_home = row["home_team"] == team
    is_away = row["away_team"] == team
    opponent_rating = row["away_rating"] if is_home else row["home_rating"]
    win_prob = PEAR_Win_Prob(one_seed_rating, opponent_rating, location)
    return win_prob

remaining_games['bubble_win_prob'] = remaining_games.apply(
    lambda row: remaining_games_rq(row, bubble_team_rating), axis=1
)

def simulate_games(df, num_simulations=100):
    projected_wins = []
    projected_resume_quality = []
    games_remaining = df.groupby("Team").size()
    for _ in range(num_simulations):
        unique_games = df.drop_duplicates(subset=["Date", "home_team", "away_team"]).copy()
        unique_games["random_val"] = np.random.rand(len(unique_games))
        unique_games["home_wins"] = unique_games["random_val"] < unique_games["home_win_prob"]
        results_map = unique_games.set_index(["Date", "home_team", "away_team"])["home_wins"].to_dict()
        df["home_wins"] = df[["Date", "home_team", "away_team"]].apply(lambda x: results_map.get(tuple(x), None), axis=1)
        df["winner"] = np.where(df["home_wins"], df["home_team"], df["away_team"])
        df["loser"] = np.where(df["home_wins"], df["away_team"], df["home_team"])
        df["win_flag"] = (df["winner"] == df["Team"]).astype(int)
        df["resume_quality_amount"] = df['win_flag'] - df['bubble_win_prob']
        wins_per_team = df.groupby("Team")["win_flag"].sum()
        total_resume_quality_per_team = df.groupby("Team")["resume_quality_amount"].sum()  # SUM instead of mean
        projected_wins.append(wins_per_team)
        projected_resume_quality.append(total_resume_quality_per_team)
    projected_wins_df = pd.DataFrame(projected_wins).mean().round().reset_index()
    projected_wins_df.columns = ["Team", "Remaining_Wins"]
    projected_resume_quality_df = pd.DataFrame(projected_resume_quality).mean().reset_index()
    projected_resume_quality_df.columns = ["Team", "Remaining_RQ"]  # Changed column name to reflect summation
    projected_wins_df["Games_Remaining"] = projected_wins_df["Team"].map(games_remaining)
    projected_wins_df["Remaining_Losses"] = projected_wins_df['Games_Remaining'] - projected_wins_df['Remaining_Wins']
    projected_wins_df = projected_wins_df.merge(projected_resume_quality_df, on="Team", how="left")
    return projected_wins_df

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
if now.hour < 13 and now.hour > 7:
    print("Starting Visuals")

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

    # --- Build Team Image Cache ---
    def fetch_team_logo(team):
        try:
            team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
            response = requests.get(team_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find("img", class_="team-menu__image")
            img_src = img_tag.get("src")
            image_url = BASE_URL + img_src
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
            return team, img
        except Exception as e:
            print(f"Error fetching {team}: {e}")
            return team, None

    # Get all top teams needed across plots
    all_top_teams = set(stats_and_metrics.head(25)['Team']) \
        | set(stats_and_metrics.sort_values('RQI').head(25)['Team']) \
        | set(stats_and_metrics.sort_values('PRR').head(25)['Team']) \
        | set(stats_and_metrics.sort_values('RPI').head(25)['Team']) \
        | set(stats_and_metrics[~stats_and_metrics['Conference'].isin(major_conferences)].head(25)['Team']) \
        | set(todays_games['home_team']) \
        | set(todays_games['away_team'])

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(fetch_team_logo, all_top_teams))

    team_images = {team: img for team, img in results if img is not None}

    # --- Plotting Function ---
    def plot_top_25(title, subtitle, sorted_df, save_path):
        top_25 = sorted_df.reset_index(drop=True).head(25)
        fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.patch.set_facecolor('#CECEB2')

        plt.suptitle(title, fontsize=20, fontweight='bold', color='black')
        fig.text(0.5, 0.92, subtitle, fontsize=10, ha='center', color='black')
        fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

        for i, ax in enumerate(axs.ravel()):
            team = top_25.loc[i, 'Team']
            img = team_images.get(team)
            if img:
                ax.imshow(img)
            ax.set_facecolor('#f0f0f0')
            ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
            ax.axis('off')

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    # --- Generate Plots ---
    plot_top_25(
        title=f"Week {current_week} CBASE PEAR",
        subtitle="NET Ranking Incorporating Team Strength and Resume",
        sorted_df=stats_and_metrics.head(25),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/NET/net_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE Resume Quality",
        subtitle="Team Performance Relative to Strength of Schedule",
        sorted_df=stats_and_metrics.sort_values('RQI'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RQI/rqi_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE Team Strength",
        subtitle="Team Rating Based on Team Stats",
        sorted_df=stats_and_metrics.sort_values('PRR'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/PRR/prr_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} CBASE RPI",
        subtitle="PEAR's RPI Rankings",
        sorted_df=stats_and_metrics.sort_values('RPI'),
        save_path=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RPI/rpi_{formatted_date}.png"
    )

    plot_top_25(
        title=f"Week {current_week} Mid-Major CBASE PEAR",
        subtitle="NET Ranking Incorporating Team Strength and Resume",
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
    # Helper Functions
    # ---------------------------

    def fetch_team_logo(team):
        try:
            team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
            response = requests.get(team_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find("img", class_="team-menu__image")
            img_src = img_tag.get("src")
            image_url = BASE_URL + img_src
            img_response = requests.get(image_url, timeout=10)
            return team, Image.open(BytesIO(img_response.content))
        except Exception as e:
            print(f"Logo fetch failed for {team}: {e}")
            return team, None

    def plot_logo_bar(data, title, filename, reference_score, zoom=0.3):
        data = data.copy()
        data["percentage_away"] = round((data["NET_Score"] - reference_score) / reference_score * 100, 2)
        data = data.sort_values("percentage_away", ascending=False)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_team_logo, data["Team"])
        team_logos = {team: img for team, img in results if img is not None}

        colors = ["#2ECC71" if x >= 0 else "#E74C3C" for x in data["percentage_away"]]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor("#CECEB2")
        ax.set_facecolor("#CECEB2")

        sns.barplot(data=data, x="percentage_away", y="Team", ax=ax, width=0.2, palette=colors, hue="Team", legend=False)
        ax.set_xlim(-data["percentage_away"].abs().max() - 0.5, data["percentage_away"].abs().max() + 0.5)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        for i, (team, pct) in enumerate(zip(data["Team"], data["percentage_away"])):
            if team in team_logos:
                ab = offsetbox.AnnotationBbox(offsetbox.OffsetImage(team_logos[team], zoom=zoom), 
                                            (pct, i), 
                                            xybox=(10 if pct >= 0 else -10, 0),
                                            boxcoords="offset points", 
                                            frameon=False)
                ax.add_artist(ab)

        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.text(-max(data["percentage_away"].abs()) - 1, -2, title, fontsize=20, ha='left', fontweight='bold')
        plt.text(-max(data["percentage_away"].abs()) - 1, -1, "@PEARatings", fontsize=12, ha='left')
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def get_conference(team, stats_df):
        return stats_df.loc[stats_df["Team"] == team, "Conference"].values[0]

    def count_conflict_conferences(teams, stats_df):
        conferences = [get_conference(team, stats_df) for team in teams]
        return sum(count - 1 for count in Counter(conferences).values() if count > 1)

    def resolve_conflicts(formatted_df, stats_df):
        seed_cols = ["2 Seed", "3 Seed", "4 Seed"]

        for seed_col in seed_cols:
            num_regionals = len(formatted_df)

            for i in range(num_regionals):
                row = formatted_df.loc[i]
                teams_i = [row["1 Seed"], row["2 Seed"], row["3 Seed"], row["4 Seed"]]
                conflict_i = count_conflict_conferences(teams_i, stats_df)

                if conflict_i == 0:
                    continue  # No conflict to resolve in this regional

                current_team = row[seed_col]

                for j in range(num_regionals):
                    if i == j:
                        continue

                    alt_team = formatted_df.at[j, seed_col]
                    if alt_team == current_team:
                        continue

                    # Simulate swap
                    row_j = formatted_df.loc[j]
                    teams_j = [row_j["1 Seed"], row_j["2 Seed"], row_j["3 Seed"], row_j["4 Seed"]]

                    # Apply the swap in test copies
                    temp_i = teams_i.copy()
                    temp_j = teams_j.copy()
                    temp_i[seed_cols.index(seed_col) + 1] = alt_team
                    temp_j[seed_cols.index(seed_col) + 1] = current_team

                    new_conflict_i = count_conflict_conferences(temp_i, stats_df)
                    new_conflict_j = count_conflict_conferences(temp_j, stats_df)

                    # Only make swap if it reduces total conflicts
                    if (new_conflict_i + new_conflict_j) < (conflict_i + count_conflict_conferences(teams_j, stats_df)):
                        formatted_df.at[i, seed_col] = alt_team
                        formatted_df.at[j, seed_col] = current_team
                        break  # Exit swap loop once we reduce conflict

        return formatted_df
    
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

    # Ratings vs Resume Scatter Plot
    fig, ax = plt.subplots(figsize=(15, 12), dpi=125)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
    logo_size = 5

    for i, row in all_at_large.iterrows():
        team = row["Team"]
        x = row["PRR"]
        y = row["aRQI"]
        above = last_team_in_index - i
        img = team_images.get(team)

        if img:
            ax.imshow(img, extent=(x - (logo_size - 1.2), x + (logo_size - 1.2), y - logo_size, y + logo_size), aspect='auto')

        if team in last_four_in["Team"].values:
            ax.add_patch(plt.Circle((x, y), logo_size, color='#2ECC71', alpha=0.3))
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')
        elif team in next_8["Team"].values:
            ax.add_patch(plt.Circle((x, y), logo_size, color='#F39C12', alpha=0.3))
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')
        elif team in bubble_teams["Team"].values:
            ax.add_patch(plt.Circle((x, y), logo_size, color='#E74C3C', alpha=0.3))
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')

    max_range = max(all_at_large["PRR"].max(), all_at_large["aRQI"].max())
    height = max_range + 4
    plt.text(max_range + 24, height + 3, "Automatic Qualifiers", ha='center', fontweight='bold', fontsize=18)
    for _, row in sorted_aqs.iterrows():
        plt.text(max_range + 24, height, row["Team"], ha='center', fontsize=16)
        height -= 3

    ax.set_xlabel("Team Strength Rank", fontsize=16)
    ax.set_ylabel("Adjusted Resume Rank", fontsize=16)
    plt.text(0, max_range + 20, "At Large Team Strength vs. Adjusted Resume", fontsize=32, fontweight='bold')
    plt.text(0, max_range + 16, f"Bubble Teams Highlighted - Through {(today - timedelta(days=1)).strftime('%m/%d')}", fontsize=24)
    plt.text(0, max_range + 12, "@PEARatings", fontsize=24, fontweight='bold')
    plt.text(max_range + 8, max_range + 7, "Projected At-Large Bids Only (Based on PEAR NET)", fontsize=16)
    plt.text(max_range + 8, max_range + 4, "Green - Last 8 In, Orange - First 8 Out, Red - Next 8 Out", fontsize=16)
    plt.text(max_range + 8, max_range + 1, "Value = Distance from Last Team In", fontsize=16)
    plt.xlim(-2, max_range + 10)
    plt.ylim(-2, max_range + 10)
    plt.grid(False)
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Ratings_vs_Resume/ratings_vs_resume_{formatted_date}.png", bbox_inches='tight')
    plt.close()

    # ---------------------------
    # Bubble Distance Bar Plot
    # ---------------------------

    bubble_df = pd.concat([last_four_in, next_8, bubble_teams]).sort_values("NET").reset_index(drop=True)[["Team", "NET", "NET_Score"]]
    last_net_score = bubble_df[bubble_df["Team"] == last_team_in]["NET_Score"].values[0]
    plot_logo_bar(
        bubble_df,
        title=f"NET Score Distance From Last Team In - Through {(today - timedelta(days=1)).strftime('%m-%d')}",
        filename=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Team_In/last_team_in_{formatted_date}.png",
        reference_score=last_net_score
    )

    # ---------------------------
    # Hosting Distance Bar Plot
    # ---------------------------

    hosting_df = stats_and_metrics.head(24)[["Team", "NET", "NET_Score"]]
    last_host_score = hosting_df[hosting_df["NET"] == 16]["NET_Score"].values[0]
    plot_logo_bar(
        hosting_df,
        title=f"NET Score Distance From Last Host Seed - Through {(today - timedelta(days=1)).strftime('%m-%d')}",
        filename=f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Host/last_host_{formatted_date}.png",
        reference_score=last_host_score
    )

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
    # Create a set of automatic qualifier teams for faster lookup
    automatic_teams = set(automatic_qualifiers["Team"])

    # Modify the DataFrame by appending '*' to teams in automatic qualifiers
    formatted_df_with_asterisk = formatted_df.copy()
    for col in formatted_df_with_asterisk.columns[0:]:  # Exclude index column
        formatted_df_with_asterisk[col] = formatted_df_with_asterisk[col].apply(lambda x: f"{x}*" if x in automatic_teams else x)
    formatted_df_with_asterisk['Host'] = formatted_df_with_asterisk.index.to_series().apply(lambda i: f"#{i} {formatted_df_with_asterisk.loc[i, '1 Seed']}")
    make_table = formatted_df_with_asterisk[['Host', '2 Seed', '3 Seed', '4 Seed']].set_index('Host')

    # Init a figure 
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#CECEB2')

    column_definitions = [
        ColumnDefinition(name='Host', # name of the column to change
                        title='Host', # new title for the column
                        textprops={"ha": "center", "weight": "bold", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='2 Seed', # name of the column to change
                        title='2 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='3 Seed', # name of the column to change
                        title='3 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='4 Seed', # name of the column to change
                        title='4 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        )
    ]

    # Create the table object with the column definitions parameter
    tab = Table(make_table, column_definitions=column_definitions, footer_divider=True, row_divider_kw={"linewidth": 1})


    # Change the color
    last_four_in_teams = set(last_four_in["Team"])
    for col in make_table.columns:
        tab.columns[col].set_facecolor("#CECEB2")

    tab.col_label_row.set_facecolor('#CECEB2')
    tab.columns["Host"].set_facecolor('#CECEB2')
    # plt.figtext(0.89, 0.09, "* Indicates an automatic qualifier", ha="right", fontsize=14, fontstyle='italic')
    plt.figtext(0.13, 0.975, f"PEAR {today.strftime('%m/%d')} Tournament Outlook", ha='left', fontsize=32, fontweight='bold')
    plt.figtext(0.13, 0.945, f"No Considerations For Regional Proximity - Through {(today - timedelta(days=1)).strftime('%m/%d')}", ha='left', fontsize=16, fontweight='bold')
    plt.figtext(0.13, 0.915, f"Regionals If Season Ended Today Based on PEAR NET Ranking", ha='left', fontsize=16)
    plt.figtext(0.13, 0.885, "@PEARatings", ha='left', fontsize=16, fontweight='bold')
    plt.figtext(0.13, 0.09, f"Last Four In - {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.06, f"First Four Out - {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.03, f"Next Four Out - {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}", ha='left', fontsize=14)
    text = " | ".join([f"{conference}: {count}" for conference, count in multibid.items()])
    final_text = "Multibid - " + text
    wrapped_text = textwrap.fill(final_text, width=100)
    lines = wrapped_text.split('\n')
    y_position = 0.00
    vertical_spacing = -0.03
    for line in lines:
        plt.figtext(0.13, y_position, line, ha='left', fontsize=14)
        y_position += vertical_spacing
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
    # Create a set of automatic qualifier teams for faster lookup
    automatic_teams = set(automatic_qualifiers["Team"])

    # Modify the DataFrame by appending '*' to teams in automatic qualifiers
    formatted_df_with_asterisk = formatted_df.copy()
    for col in formatted_df_with_asterisk.columns[0:]:  # Exclude index column
        formatted_df_with_asterisk[col] = formatted_df_with_asterisk[col].apply(lambda x: f"{x}*" if x in automatic_teams else x)
    formatted_df_with_asterisk['Host'] = formatted_df_with_asterisk.index.to_series().apply(lambda i: f"#{i} {formatted_df_with_asterisk.loc[i, '1 Seed']}")
    make_table = formatted_df_with_asterisk[['Host', '2 Seed', '3 Seed', '4 Seed']].set_index('Host')

    # Init a figure 
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#CECEB2')

    column_definitions = [
        ColumnDefinition(name='Host', # name of the column to change
                        title='Host', # new title for the column
                        textprops={"ha": "center", "weight": "bold", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='2 Seed', # name of the column to change
                        title='2 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='3 Seed', # name of the column to change
                        title='3 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='4 Seed', # name of the column to change
                        title='4 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        )
    ]

    # Create the table object with the column definitions parameter
    tab = Table(make_table, column_definitions=column_definitions, footer_divider=True, row_divider_kw={"linewidth": 1})


    # Change the color
    last_four_in_teams = set(last_four_in["Team"])
    for col in make_table.columns:
        tab.columns[col].set_facecolor("#CECEB2")

    tab.col_label_row.set_facecolor('#CECEB2')
    tab.columns["Host"].set_facecolor('#CECEB2')
    # plt.figtext(0.89, 0.09, "* Indicates an automatic qualifier", ha="right", fontsize=14, fontstyle='italic')
    plt.figtext(0.13, 0.975, f"PEAR {today.strftime('%m/%d')} Projected Tournament", ha='left', fontsize=32, fontweight='bold')
    plt.figtext(0.13, 0.945, f"No Considerations For Regional Proximity - Through {(today - timedelta(days=1)).strftime('%m/%d')}", ha='left', fontsize=16, fontweight='bold')
    plt.figtext(0.13, 0.915, f"Based on PEAR's Projected NET", ha='left', fontsize=16)
    plt.figtext(0.13, 0.885, "@PEARatings", ha='left', fontsize=16, fontweight='bold')
    plt.figtext(0.13, 0.09, f"Last Four In - {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.06, f"First Four Out - {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.03, f"Next Four Out - {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}", ha='left', fontsize=14)
    text = " | ".join([f"{conference}: {count}" for conference, count in multibid.items()])
    final_text = "Multibid - " + text
    wrapped_text = textwrap.fill(final_text, width=100)
    lines = wrapped_text.split('\n')
    y_position = 0.00
    vertical_spacing = -0.03
    for line in lines:
        plt.figtext(0.13, y_position, line, ha='left', fontsize=14)
        y_position += vertical_spacing
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Projected_Tournament/proj_tournament_{formatted_date}.png", bbox_inches='tight')

    def net_tracker(X, Y):
        folder_path = f"./PEAR/PEAR Baseball/y{current_season}/Data"
        csv_files = [f for f in os.listdir(folder_path) if f.startswith("baseball_") and f.endswith(".csv")]
        def extract_date(filename):
            try:
                return datetime.strptime(filename.replace("baseball_", "").replace(".csv", ""), "%m_%d_%Y")
            except ValueError:
                return None
            
        date_files = {extract_date(f): f for f in csv_files if extract_date(f) is not None}
        sorted_date_files = sorted(date_files.items(), key=lambda x: x[0], reverse=True)
        latest_files = sorted_date_files[:X]
        stats_and_metrics_list = []
        for _, file_name in latest_files:
            file_path = os.path.join(folder_path, file_name)
            date_of_file = extract_date(file_name).strftime('%Y-%m-%d')  # Get date in 'YYYY-MM-DD' format
            
            # Read the CSV and add the "Date" column
            df = pd.read_csv(file_path)
            df['Date'] = date_of_file  # Add a column with the date of the file
            stats_and_metrics_list.append(df)
        stats_and_metrics_combined = pd.concat(stats_and_metrics_list, ignore_index=True)
        most_recent_csv = stats_and_metrics_combined[stats_and_metrics_combined['Date'] == stats_and_metrics_combined['Date'].max()]
        top_teams = most_recent_csv.nsmallest(Y, 'NET')['Team'].tolist()
        filtered_data = stats_and_metrics_combined[stats_and_metrics_combined['Team'].isin(top_teams)]
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date']).dt.strftime('%m-%d')
        earliest = filtered_data['Date'].min()
        latest = filtered_data['Date'].max()
        pivoted_table = filtered_data.pivot_table(index='Team', columns='Date', values='NET', aggfunc='first')
        pivoted_table = pivoted_table[sorted(pivoted_table.columns)]
        pivoted_table = pivoted_table.reset_index()
        pivoted_table = pivoted_table.sort_values(by=pivoted_table.columns[-1], ascending=True)
        num_rows = math.ceil(math.sqrt(Y))
        num_cols = math.ceil(math.sqrt(Y))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig.patch.set_facecolor('#CECEB2')
        axes = axes.flatten()
        plt.text(0, 1.5, f"Top {Y} in PEAR NET Rankings", fontsize = 32, fontweight='bold', ha='left', transform=axes[0].transAxes)
        plt.text(0, 1.34, f"Past {X} Days - {earliest} to {latest}", fontsize = 16, ha='left', transform=axes[0].transAxes)
        plt.text(0, 1.19, f"@PEARatings", fontsize = 16, ha='left', fontweight='bold', transform=axes[0].transAxes)
        min_net = 0
        max_net = pivoted_table.drop(columns='Team').max().max() + 1

        for i, (team, ax) in enumerate(zip(pivoted_table['Team'], axes)):
            team_data = pivoted_table[pivoted_table['Team'] == team].drop(columns='Team').T  # Drop 'Team' column and transpose
            team_data.columns = [team]  # Set the team name as the column header
            last_value = team_data.iloc[0, 0]
            first_value = team_data.iloc[-1, 0]
            
            # Determine the color of the line based on trend
            line_color = "#2ca02c" if last_value > first_value else "#d62728"

            ax.plot(team_data.index, team_data[team], marker='o', color=line_color)
            ax.text(team_data.index[0], last_value - 1.2, f"{int(last_value)}", 
                fontsize=10, fontweight='bold', ha='center', color='black')
            ax.text(team_data.index[-1], first_value - 1.2, f"{int(first_value)}", 
                fontsize=10, fontweight='bold', ha='center', color='black')
            ax.set_title(f"#{i+1} {team}", fontweight='bold', fontsize=16)
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.set_facecolor('#CECEB2')
            ax.set_xticks("")
            ax.set_ylim(min_net, max_net)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Over_Time/over_time_{formatted_date}.png", bbox_inches='tight')
    net_tracker(14,16)

    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)  # More precision, rounded later in output

    def simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics):
        teams = [team_a, team_b, team_c, team_d]
        ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

        game1 = PEAR_Win_Prob(ratings[team_a], ratings[team_d]) 
        w1, l1 = (team_a, team_d) if random.random() < game1 else (team_d, team_a)

        game2 = PEAR_Win_Prob(ratings[team_b], ratings[team_c])
        w2, l2 = (team_b, team_c) if random.random() < game2 else (team_c, team_b)

        game3 = PEAR_Win_Prob(ratings[l2], ratings[l1])
        w3 = l2 if random.random() < game3 else l1

        game4 = PEAR_Win_Prob(ratings[w1], ratings[w2])
        w4, l4 = (w1, w2) if random.random() < game4 else (w2, w1)

        game5 = PEAR_Win_Prob(ratings[l4], ratings[w3])
        w5 = l4 if random.random() < game5 else w3

        game6 = PEAR_Win_Prob(ratings[w4], ratings[w5])
        w6 = w4 if random.random() < game6 else w5

        if w6 == w4:
            regional_winner = w6
        else:
            regional_winner = w4 if random.random() < PEAR_Win_Prob(ratings[w4], ratings[w5]) else w5

        return regional_winner

    def run_simulation(team_a, team_b, team_c, team_d, stats_and_metrics, num_simulations=1000):
        results = defaultdict(int)

        for _ in range(num_simulations):
            winner = simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics)
            results[winner] += 1

        # Sort and format results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Return a defaultdict to maintain the same structure
        formatted_results = defaultdict(float, {team: round(wins / num_simulations, 3) for team, wins in sorted_results})
        
        return formatted_results

    def select_super_regional_teams(regional_results):
        """Selects super regional teams probabilistically based on regional results."""
        super_regional_teams = []
        matchups = [(0, 15), (1, 14), (2, 13), (3, 12), 
                    (4, 11), (5, 10), (6, 9), (7, 8)]
        
        for idx1, idx2 in matchups:
            team1 = random.choices(list(regional_results[idx1].keys()), 
                                weights=list(regional_results[idx1].values()), k=1)[0]
            team2 = random.choices(list(regional_results[idx2].keys()), 
                                weights=list(regional_results[idx2].values()), k=1)[0]
            super_regional_teams.append((team1, team2, idx1))  # Keep track of region index

        return super_regional_teams

    def simulate_super_regional(team1, team2, stats_and_metrics):
        """Simulates a best-of-three series for the super regional."""
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]

        wins1, wins2 = 0, 0
        while wins1 < 2 and wins2 < 2:
            win_prob = PEAR_Win_Prob(rating1, rating2)
            if random.random() < win_prob:
                wins1 += 1
            else:
                wins2 += 1

        return team1 if wins1 == 2 else team2

    def run_super_regionals(regional_results, stats_and_metrics, num_simulations=1000):
        """Runs multiple simulations of super regionals incorporating regional probabilities."""
        results = [defaultdict(int) for _ in range(8)]  # Maintain separate regions

        for _ in range(num_simulations):
            super_regional_matchups = select_super_regional_teams(regional_results)
            
            for team1, team2, region_index in super_regional_matchups:
                winner = simulate_super_regional(team1, team2, stats_and_metrics)
                results[region_index][winner] += 1  # Track wins in the correct region

        # Normalize probabilities for each region
        for i in range(8):
            total_sims = sum(results[i].values())
            for team in results[i]:
                results[i][team] = round(results[i][team] / total_sims, 3)

        return results

    def select_team(region):
        """Selects a team probabilistically based on their make_omaha probabilities."""
        teams, weights = zip(*region.items())
        return random.choices(teams, weights=weights, k=1)[0]

    def simulate_game(team1, team2, stats_and_metrics):
        """Simulates a game using PEAR_Win_Prob."""
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]
        win_prob = PEAR_Win_Prob(rating1, rating2)
        return team1 if random.random() < win_prob else team2

    def simulate_double_elimination(teams, stats_and_metrics):
        """Runs a double-elimination tournament for 4 teams."""
        winners_bracket = [teams[0], teams[1], teams[2], teams[3]]
        losers_bracket = []

        # Round 1
        game1_winner = simulate_game(winners_bracket[0], winners_bracket[1], stats_and_metrics)
        game1_loser = winners_bracket[1] if game1_winner == winners_bracket[0] else winners_bracket[0]

        game2_winner = simulate_game(winners_bracket[2], winners_bracket[3], stats_and_metrics)
        game2_loser = winners_bracket[3] if game2_winner == winners_bracket[2] else winners_bracket[2]

        losers_bracket.extend([game1_loser, game2_loser])
        
        # Round 2 (Winners Final)
        winners_final_winner = simulate_game(game1_winner, game2_winner, stats_and_metrics)
        winners_final_loser = game1_winner if winners_final_winner == game2_winner else game2_winner

        # Losers Bracket
        losers_round1_winner = simulate_game(losers_bracket[0], losers_bracket[1], stats_and_metrics)

        # Losers Final
        losers_final_winner = simulate_game(losers_round1_winner, winners_final_loser, stats_and_metrics)

        # Championship (Double elimination rule: Losers bracket winner must win twice)
        if simulate_game(winners_final_winner, losers_final_winner, stats_and_metrics) == winners_final_winner:
            return winners_final_winner
        return simulate_game(winners_final_winner, losers_final_winner, stats_and_metrics)

    def run_college_world_series(make_omaha, stats_and_metrics, num_simulations=1000):
        """Runs two separate double-elimination tournaments and returns results in the same format as input."""
        results = [defaultdict(int), defaultdict(int)]

        for _ in range(num_simulations):
            # Tournament 1
            team1 = select_team(make_omaha[0])
            team2 = select_team(make_omaha[7])
            team3 = select_team(make_omaha[3])
            team4 = select_team(make_omaha[4])
            winner1 = simulate_double_elimination([team1, team2, team3, team4], stats_and_metrics)
            results[0][winner1] += 1

            # Tournament 2
            team5 = select_team(make_omaha[1])
            team6 = select_team(make_omaha[6])
            team7 = select_team(make_omaha[2])
            team8 = select_team(make_omaha[5])
            winner2 = simulate_double_elimination([team5, team6, team7, team8], stats_and_metrics)
            results[1][winner2] += 1

        # Normalize results to probabilities
        for i in range(2):
            total_sims = sum(results[i].values())
            for team in results[i]:
                results[i][team] = round(results[i][team] / total_sims, 3)

        return results

    def simulate_finals(make_finals, stats_and_metrics):
        """Simulates the College World Series Finals as a best-of-three series."""
        # Select a team from each defaultdict
        team1 = random.choices(list(make_finals[0].keys()), weights=list(make_finals[0].values()), k=1)[0]
        team2 = random.choices(list(make_finals[1].keys()), weights=list(make_finals[1].values()), k=1)[0]

        # Get team ratings
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]

        # Simulate best-of-three series
        wins1, wins2 = 0, 0
        while wins1 < 2 and wins2 < 2:
            win_prob = PEAR_Win_Prob(rating1, rating2)
            if random.random() < win_prob:
                wins1 += 1
            else:
                wins2 += 1

        # Return the champion
        return team1 if wins1 == 2 else team2

    def run_finals_simulation(make_finals, stats_and_metrics, num_simulations=1000):
        """Runs multiple simulations of the College World Series Finals and returns championship probabilities."""
        results = defaultdict(int)

        for _ in range(num_simulations):
            champion = simulate_finals(make_finals, stats_and_metrics)
            results[champion] += 1

        # Convert to probabilities
        total_sims = sum(results.values())
        results_dict = {team: round(wins / total_sims, 3) for team, wins in results.items()}
        
        return results_dict

    def generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals):
        teams = set()  # To store all unique teams from all simulation results

        # Collect teams from regional results (Super Regional probabilities)
        for regional in regionals_results:
            teams.update(regional.keys())

        # Collect teams from Make Omaha (Omaha probabilities)
        for omaha in make_omaha:
            teams.update(omaha.keys())

        # Collect teams from Make Finals (Finals probabilities)
        for finals in make_finals:
            teams.update(finals.keys())

        # Collect teams from Win Finals (Win NC probabilities)
        teams.update(win_finals.keys())

        # Initialize a dictionary to hold the probabilities for each team
        data = {team: {"Supers": 0, "Omaha": 0, "Finals": 0, "Win NC": 0} for team in teams}

        # Assign the probabilities directly from the simulation results
        for regional in regionals_results:
            for team, prob in regional.items():
                data[team]["Supers"] = prob  # Assign the probability for reaching Super Regionals

        for omaha in make_omaha:
            for team, prob in omaha.items():
                data[team]["Omaha"] = prob  # Assign the probability for reaching Omaha

        for finals in make_finals:
            for team, prob in finals.items():
                data[team]["Finals"] = prob  # Assign the probability for reaching Finals

        for team, prob in win_finals.items():
            data[team]["Win NC"] = prob  # Assign the probability for winning the National Championship

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')

        return df

    def simulate_full_tournament(formatted_df, stats_and_metrics, iter):
        regionals_results = []
        for i in range(len(formatted_df)):
            regionals_results.append(run_simulation(formatted_df.iloc[i,1], formatted_df.iloc[i,2], formatted_df.iloc[i,3], formatted_df.iloc[i,4], stats_and_metrics, iter))
        make_omaha = run_super_regionals(regionals_results, stats_and_metrics, iter)
        make_finals = run_college_world_series(make_omaha, stats_and_metrics, iter)
        win_finals = run_finals_simulation(make_finals, stats_and_metrics, iter)
        simulation_df = generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals)
        simulation_df = simulation_df.sort_values('Win NC', ascending=False).reset_index()
        simulation_df.rename(columns={'index': 'Team'}, inplace=True)
        return simulation_df

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
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    formatted_df = resolve_conflicts(formatted_df, stats_and_metrics)

    tournament_sim = simulate_full_tournament(formatted_df, stats_and_metrics, 1000)
    top_25_teams = tournament_sim[0:25]
    top_25_teams.iloc[:, 1:] = top_25_teams.iloc[:, 1:] * 100

    # Normalize values for color gradient (excluding 0 values)
    min_value = top_25_teams.iloc[:, 1:].replace(0, np.nan).min().min()  # Min of all probabilities
    max_value = top_25_teams.iloc[:, 1:].max().max()  # Max of all probabilities

    def normalize(value, min_val, max_val):
        """ Normalize values between 0 and 1 for colormap. """
        if pd.isna(value) or value == 0:
            return 0  # Keep 0 values at the lowest color
        return (value - min_val) / (max_val - min_val)

    # Define custom colormap (lighter green to dark green)
    cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=125)
    fig.patch.set_facecolor('#CECEB2')

    ax.axis('tight')
    ax.axis('off')

    # Add the table
    table = ax.table(
        cellText=top_25_teams.values,
        colLabels=top_25_teams.columns,
        cellLoc='center',
        loc='center',
        colColours=['#CECEB2'] * len(top_25_teams.columns)  # Set the header background color
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
            value = top_25_teams.iloc[i-1, j]  # Skip header row
            normalized_value = normalize(value, min_value, max_value)
            color = cmap(normalized_value)
            cell.set_facecolor(color)
            cell.set_text_props(fontsize=14, weight='bold', color='black')
            if value == 0:
                cell.get_text().set_text("<1%")
            else:
                cell.get_text().set_text(f"{value:.1f}%")

        cell.set_height(0.05)

    # Show the plot
    plt.text(0, 0.086, 'Odds to Win Championship', fontsize=24, fontweight='bold', ha='center')
    plt.text(0, 0.08, f"Based on Projected NET Tournament {today.strftime('%m/%d')}", fontsize=16, ha='center')
    plt.text(0, 0.074, f"@PEARatings", fontsize=16, fontweight='bold', ha='center')
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

    def simulate_tournament(teams, ratings):
        import random

        team_a, team_b, team_c, team_d = teams
        r = ratings

        def adjusted(team):
            return r[team] + 0.8 if team == team_a else r[team]

        w1, l1 = (team_a, team_d) if random.random() < PEAR_Win_Prob(adjusted(team_a), adjusted(team_d)) else (team_d, team_a)
        w2, l2 = (team_b, team_c) if random.random() < PEAR_Win_Prob(adjusted(team_b), adjusted(team_c)) else (team_c, team_b)
        w3 = l2 if random.random() < PEAR_Win_Prob(adjusted(l2), adjusted(l1)) else l1
        w4, l4 = (w1, w2) if random.random() < PEAR_Win_Prob(adjusted(w1), adjusted(w2)) else (w2, w1)
        w5 = l4 if random.random() < PEAR_Win_Prob(adjusted(l4), adjusted(w3)) else w3
        game6_prob = PEAR_Win_Prob(adjusted(w4), adjusted(w5))
        w6 = w4 if random.random() < game6_prob else w5

        return w6 if w6 == w4 else (w4 if random.random() < game6_prob else w5)

    def run_simulation(teams, stats_and_metrics, num_simulations=5000):
        ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}
        results = defaultdict(int)

        for _ in range(num_simulations):
            winner = simulate_tournament(teams, ratings)
            results[winner] += 1

        total = num_simulations
        return defaultdict(float, {team: round(count / total, 3) for team, count in results.items()})

    # --- Bracket Construction ---

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
    formatted_df = resolve_conflicts(formatted_df, stats_and_metrics)

    # --- Visualization ---

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=125)
    fig.patch.set_facecolor('#CECEB2')
    axes = axes.flatten()

    cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

    custom_order = [0, 15, 4, 11, 1, 14, 5, 10, 2, 13, 6, 9, 3, 12, 7, 8]
    for plot_idx, idx in enumerate(custom_order):
        teams = list(formatted_df.iloc[idx, 1:5])
        sim_results = run_simulation(teams, stats_and_metrics)

        seeded_teams = [f"#{i+1} {team}" for i, team in enumerate(teams)]
        # Preserve team order from input
        regional_prob = pd.DataFrame({
            "Team": seeded_teams,
            "Win": [sim_results.get(team, 0) * 100 for team in teams]
        })

        win_vals = regional_prob["Win"].values
        norm_vals = normalize_array(win_vals)

        ax = axes[plot_idx]
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=regional_prob.values,
            colLabels=["Team", "Win %"],
            cellLoc='center',
            loc='center',
            colColours=['#CECEB2'] * 2,
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(1.2)
            cell.set_height(0.15)

            if j == 0:
                cell.set_width(0.7)
            else:
                cell.set_width(0.3)

            font_props = {'fontsize': 16, 'weight': 'bold', 'color': 'black'}
            cell.set_text_props(**font_props)

            if i == 0:
                cell.set_facecolor('#CECEB2')
            else:
                if j == 1:
                    color = cmap(norm_vals[i - 1])
                    cell.set_facecolor(color)
                    cell.get_text().set_text(f"{win_vals[i - 1]:.1f}%")
                    cell.set_text_props(fontsize=16, fontweight='bold')
                else:
                    cell.set_facecolor('#CECEB2')

        ax.text(0.5, 1.05, f'#{idx + 1} {teams[0]} Regional',
                fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)

    fig.text(0.5, 0.95, "PEAR's Regional Winner Projections", ha='center', fontweight='bold', fontsize=24)
    fig.text(0.5, 0.93, "Based on PEAR's Tournament - No Consideration for Regional Proximity", 
            ha='center', fontsize=16)
    fig.text(0.5, 0.91, "@PEARatings", ha='center', fontweight='bold', fontsize=16)
    plt.savefig(f"./PEAR/PEAR Baseball/y2025/Visuals/Regional_Win_Prob/regional_win_prob_{formatted_date}.png", bbox_inches='tight')
    print("Visuals Completed")