from datetime import datetime # type: ignore
import os # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pytz # type: ignore
import re # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.ticker import MaxNLocator # type: ignore
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from io import BytesIO
from PIL import Image
import matplotlib.colors as mcolors
from plottable import Table # type: ignore
from plottable.plots import image, circled_image # type: ignore
from plottable import ColumnDefinition # type: ignore
from plottable.cmap import normed_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from collections import defaultdict
import random
import glob
from matplotlib.patches import FancyBboxPatch
font_prop = fm.FontProperties(fname="./PEAR/trebuc.ttf")
fm.fontManager.addfont("./PEAR/trebuc.ttf")
fm.fontManager.addfont("./PEAR/Trebuchet MS Bold.ttf")
plt.rcParams['font.family'] = font_prop.get_name()

cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year
schedule_df = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/schedule_{current_season}.csv")
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")
# formatted_date_dt = pd.to_datetime(comparison_date, format="%m_%d_%Y")
subset_games = schedule_df[
    (schedule_df["Date"] >= comparison_date) &
    (schedule_df["Date"] <= comparison_date + pd.Timedelta(days=0))
][['home_team', 'away_team', 'PEAR', 'GQI', 'Date', 'Team', 'Opponent', 'Result']].sort_values('Date').drop_duplicates(subset=['home_team', 'away_team'], keep = 'first').reset_index(drop=True)

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

def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

if len(subset_games) > 0:
    subset_games['Result'] = subset_games['Result'].astype(str)
    subset_games = subset_games.sort_values(by="Result", key=lambda x: x.map(game_sort_key))
    subset_games["Result"] = subset_games["Result"].astype(str)  # Convert to string to avoid errors
    subset_games["Result"] = subset_games["Result"].apply(lambda x: x if x.startswith(("W", "L")) else "")
    subset_games["Result"] = subset_games.apply(process_result, axis=1)
    subset_games = subset_games.reset_index(drop=True)
    subset_games.index = subset_games.index + 1

base_url = "https://www.ncaa.com"
stats_page = f"{base_url}/stats/baseball/d1"
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()  # Ensure request was successful
    return BeautifulSoup(response.text, "html.parser")
soup = get_soup(stats_page)
dropdown = soup.find("select", {"id": "select-container-team"})
options = dropdown.find_all("option")
stat_links = {
    option.text.strip(): base_url + option["value"]
    for option in options if option.get("value")
}
url = stat_links['Base on Balls']
response = requests.get(url)

# Parse with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

desc_div = soup.find("div", class_="stats-header__lower__desc")
desc_text = desc_div.text.strip()
# Regular expression to find the last date (format: Month Day, Year)
match = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})$", desc_text)
last_date = match.group(1)

folder_path = f"./PEAR/PEAR Baseball/y{current_season}/Data"

csv_files = [f for f in os.listdir(folder_path) if f.startswith("baseball_") and f.endswith(".csv")]

# Extract dates from filenames and find the closest one
def extract_date(filename):
    try:
        return datetime.strptime(filename.replace("baseball_", "").replace(".csv", ""), "%m_%d_%Y")
    except ValueError:
        return None
    
def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location):
    if location != "Neutral":
        home_pr += 0.3
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    spread = round(adjust_home_pr(elo_win_prob) + home_pr - away_pr, 2)
    return spread, elo_win_prob

# Get valid date files
date_files = {extract_date(f): f for f in csv_files if extract_date(f) is not None}

if date_files:
    sorted_dates = sorted(date_files.keys(), reverse=True)

    # Try latest date
    latest_date = sorted_dates[0]
    latest_file = date_files[latest_date]
    file_path = os.path.join(folder_path, latest_file)
    modeling_stats = pd.read_csv(file_path)

    # If not 299, try previous day
    if len(modeling_stats) != 299 and len(sorted_dates) > 1:
        previous_date = sorted_dates[1]
        previous_file = date_files[previous_date]
        file_path = os.path.join(folder_path, previous_file)
        modeling_stats = pd.read_csv(file_path)
        latest_date = previous_date  # Update to reflect the used date

    formatted_latest_date = latest_date.strftime("%B %d, %Y")
else:
    modeling_stats = None
    formatted_latest_date = None


base_path = "PEAR/PEAR Baseball"
seasons = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
           2018, 2019, 2021, 2022, 2023, 2024, 2025]

all_data_list = []

date_pattern = re.compile(r"baseball_(\d{2}_\d{2}_\d{4})\.csv")

for year in seasons:
    folder = os.path.join(base_path, f"y{year}", "Data")
    pattern = os.path.join(folder, "baseball_*.csv")
    files = glob.glob(pattern)

    valid_files = []
    for f in files:
        filename = os.path.basename(f)
        match = date_pattern.fullmatch(filename)
        if match:
            valid_files.append((f, pd.to_datetime(match.group(1), format="%m_%d_%Y")))

    if valid_files:
        # Pick file with latest date
        latest_file = max(valid_files, key=lambda x: x[1])[0]

        df = pd.read_csv(latest_file)[['Team', 'Conference', 'Rating', 'NET_Score']]
        df['Season'] = year
        all_data_list.append(df)
    else:
        print(f"No valid file found for {year}")

all_data = pd.concat(all_data_list, ignore_index=True)
all_data['Normalized_Rating'] = all_data.groupby('Season')['Rating'].transform(
    lambda x: (x - x.mean()) / x.std()
)

all_data['Normalized_Rating'] = all_data['Normalized_Rating'] * 2.5
all_data['Normalized_Rating'] = round(all_data['Normalized_Rating'] - all_data['Normalized_Rating'].mean(),2)
all_data['Normalized_Rating'] = round(all_data['Normalized_Rating'], 2)
all_data = all_data.sort_values('Normalized_Rating', ascending=False).reset_index(drop=True)
all_data['Season'] = all_data['Season'].astype(int)

all_data_full_list = []

date_pattern = re.compile(r"baseball_(\d{2}_\d{2}_\d{4})\.csv")

for year in seasons:
    folder = os.path.join(base_path, f"y{year}", "Data")
    pattern = os.path.join(folder, "baseball_*.csv")
    files = glob.glob(pattern)

    valid_files = []
    for f in files:
        filename = os.path.basename(f)
        match = date_pattern.fullmatch(filename)
        if match:
            valid_files.append((f, pd.to_datetime(match.group(1), format="%m_%d_%Y")))

    if valid_files:
        # Pick file with latest date
        latest_file = max(valid_files, key=lambda x: x[1])[0]

        df = pd.read_csv(latest_file)
        df['Season'] = year
        all_data_full_list.append(df)
    else:
        print(f"No valid file found for {year}")

all_data_full = pd.concat(all_data_full_list, ignore_index=True)
all_data_full['Normalized_Rating'] = all_data_full.groupby('Season')['Rating'].transform(
    lambda x: (x - x.mean()) / x.std()
)

all_data_full['Normalized_Rating'] = all_data_full['Normalized_Rating'] * 2.5
all_data_full['Normalized_Rating'] = round(all_data_full['Normalized_Rating'] - all_data_full['Normalized_Rating'].mean(),2)
all_data_full['Normalized_Rating'] = round(all_data_full['Normalized_Rating'], 2)
all_data_full = all_data_full.sort_values('Normalized_Rating', ascending=False).reset_index(drop=True)
all_data_full['Season'] = all_data_full['Season'].astype(int)


def find_spread(home_team, away_team, location = 'Neutral'):
    default_pr = modeling_stats['Rating'].mean() - 1.75 * modeling_stats['Rating'].std()
    default_elo = 1200

    home_pr = modeling_stats.loc[modeling_stats['Team'] == home_team, 'Rating']
    if location != "Neutral":
        home_pr += 0.3
    away_pr = modeling_stats.loc[modeling_stats['Team'] == away_team, 'Rating']
    home_elo = modeling_stats.loc[modeling_stats['Team'] == home_team, 'ELO']
    away_elo = modeling_stats.loc[modeling_stats['Team'] == away_team, 'ELO']
    home_pr = home_pr.iloc[0] if not home_pr.empty else default_pr
    away_pr = away_pr.iloc[0] if not away_pr.empty else default_pr
    home_elo = home_elo.iloc[0] if not home_elo.empty else default_elo
    away_elo = away_elo.iloc[0] if not away_elo.empty else default_elo
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)

    win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)
    raw_spread = adjust_home_pr(elo_win_prob) + home_pr - away_pr
    spread = round(raw_spread,2)
    if spread >= 0:
        return f"{home_team} -{spread}"
    else:
        return f"{away_team} {spread}"
    
def find_spread_matchup(home_team, away_team, modeling_stats, location):
    home_pr = modeling_stats.loc[modeling_stats['Team'] == home_team, 'Rating']
    away_pr = modeling_stats.loc[modeling_stats['Team'] == away_team, 'Rating']
    home_elo = modeling_stats.loc[modeling_stats['Team'] == home_team, 'ELO']
    away_elo = modeling_stats.loc[modeling_stats['Team'] == away_team, 'ELO']

    home_pr = home_pr.iloc[0]
    away_pr = away_pr.iloc[0]
    home_elo = home_elo.iloc[0]
    away_elo = away_elo.iloc[0]

    spread, elo_win_prob = calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location)
    # if location != "Neutral":
    #     rating_diff = home_pr + 0.5 - away_pr
    # else:
    #     rating_diff = home_pr - away_pr
    win_prob = PEAR_Win_Prob(home_pr, away_pr, location)

    if spread >= 0:
        return f"{home_team} -{spread}", win_prob
    else:
        return f"{away_team} {spread}", win_prob

def elo_load():
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
elo_data = elo_load()

def grab_team_schedule(team_name, stats_df):

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

    BASE_URL = "https://www.warrennolan.com"
    schedule_data = []
    team_link = elo_data[elo_data['Team'] == team_name]['Team Link'].values[0]

    team_schedule_url = BASE_URL + team_link
    response = requests.get(team_schedule_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    schedule_lists = soup.find_all("ul", class_="team-schedule")
    schedule_list = schedule_lists[0]

    for game in schedule_list.find_all('li', class_='team-schedule'):
        date_month = game.find('span', class_='team-schedule__game-date--month').text.strip()
        date_day = game.find('span', class_='team-schedule__game-date--day').text.strip()
        date_dow = game.find('span', class_='team-schedule__game-date--dow').text.strip()
        game_date = f"{date_month} {date_day} ({date_dow})"

        opponent_info = game.find('div', class_='team-schedule__opp')
        if opponent_info:
            opponent_link_element = opponent_info.find('a', class_='team-schedule__opp-line-link')
            opponent_name = opponent_link_element.text.strip() if opponent_link_element else ""
        else:
            opponent_name = ""

        location_div = game.find('div', class_='team-schedule__location')
        if location_div:
            location_text = location_div.text.strip()
            if "VS" in location_text:
                game_location = "Neutral"
            elif "AT" in location_text:
                game_location = "Away"
            else:
                game_location = "Home"
        else:
            game_location = "Home"

        # Extract Game Result
        result_info = game.find('div', class_='team-schedule__result')
        result_text = result_info.text.strip() if result_info else "N/A"

        # Extract Home/Away Teams from Box Score and scores
        home_score, away_score = "", ""  # Initialize scores as empty strings

        box_score_table = game.find('table', class_='team-schedule-bottom__box-score')
        if box_score_table:
            rows = box_score_table.find_all('tr')
            if len(rows) > 2:
                away_team = rows[1].find_all('td')[0].text.strip()
                home_team = rows[2].find_all('td')[0].text.strip()

                # Extracting Runs
                away_score = rows[1].find_all('td')[-3].text.strip()  # Away runs
                home_score = rows[2].find_all('td')[-3].text.strip()  # Home runs
            else:
                home_team, away_team = "N/A", "N/A"
        else:
            home_team, away_team = "N/A", "N/A"

        # Append to schedule data
        schedule_data.append([team_name, game_date, opponent_name, game_location, result_text, home_team, away_team, home_score, away_score])

    columns = ["Team", "Date", "Opponent", "Location", "Result", "home_team", "away_team", "home_score", "away_score"]
    schedule_df = pd.DataFrame(schedule_data, columns=columns)
    schedule_df = schedule_df.astype({col: 'str' for col in schedule_df.columns if col not in ['home_score', 'away_score']})
    schedule_df['home_score'] = schedule_df['home_score'].astype(int, errors='ignore')
    schedule_df['away_score'] = schedule_df['away_score'].astype(int, errors='ignore')

    columns_to_replace = ['Team', 'home_team', 'away_team', 'Opponent']
    for col in columns_to_replace:
        schedule_df[col] = schedule_df[col].str.replace('State', 'St.', regex=False)
        schedule_df[col] = schedule_df[col].replace(team_replacements)

    schedule_df = schedule_df.merge(
        stats_df[['Team', 'Rating', 'NET']],  # Keep only "Rating" and "Resume"
        left_on="Opponent",
        right_on="Team",  # Match "Opponent" with the "Rating" column (previously the index)
        how="left"  # Keep all rows from schedule_df
    )
    schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)
    schedule_df = schedule_df.drop(columns=['Team_y'])
    team_rank = int(stats_df[stats_df['Team'] == team_name]['NET'].values[0])
    team_rating = stats_df[stats_df['Team'] == team_name]['Rating'].values[0]
    # Define conditions
    conditions = [
        ((schedule_df["Location"] == "Home") & (schedule_df["NET"] <= 25)) |
        ((schedule_df["Location"] == "Neutral") & (schedule_df["NET"] <= 40)) |
        ((schedule_df["Location"] == "Away") & (schedule_df["NET"] <= 60)),

        ((schedule_df["Location"] == "Home") & (schedule_df["NET"] <= 50)) |
        ((schedule_df["Location"] == "Neutral") & (schedule_df["NET"] <= 80)) |
        ((schedule_df["Location"] == "Away") & (schedule_df["NET"] <= 120)),

        ((schedule_df["Location"] == "Home") & (schedule_df["NET"] <= 100)) |
        ((schedule_df["Location"] == "Neutral") & (schedule_df["NET"] <= 160)) |
        ((schedule_df["Location"] == "Away") & (schedule_df["NET"] <= 240))
    ]

    # Define corresponding quadrant labels
    quadrants = ["Q1", "Q2", "Q3"]

    # Assign Quadrant values
    schedule_df["Quad"] = np.select(conditions, quadrants, default="Q4")

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
    schedule_df["Comparison_Date"] = schedule_df["Date"].astype(str).apply(convert_date)
    schedule_df["Comparison_Date"] = pd.to_datetime(schedule_df["Comparison_Date"], format="%m-%d-%Y")
    formatted_date = datetime.today().strftime('%m_%d_%Y')
    comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")
    completed_schedule = schedule_df[
        (schedule_df["Comparison_Date"] <= comparison_date) & 
        (schedule_df["home_score"] != schedule_df["away_score"]) &
        (schedule_df["Result"].str.contains("W|L"))  # Check if "Result" contains "W" or "L"
    ].reset_index(drop=True)
    remaining_games = schedule_df[schedule_df["Comparison_Date"] > comparison_date].reset_index(drop=True)
    if len(remaining_games) > 0:
        def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
            if location != "Neutral":
                home_pr += 0.3
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)
        remaining_games['PEAR'] = remaining_games.apply(
            lambda row: find_spread(
                row['Opponent'], row['Team'], row['Location']
            ) if row['Location'] == 'Away' else find_spread(
                row['Team'], row['Opponent'], row['Location']
            ),
            axis=1
        )
        def clean_spread(row):
            team_name = row["Team"]
            spread = row["PEAR"]
            spread_value_str = spread.split()[-1]
            spread_value_str = spread_value_str.lstrip("-") if spread_value_str.startswith("--") else spread_value_str
            try:
                spread_value = float(spread_value_str)
            except ValueError:
                spread_value = 0.0  # Default to 0 if there's a parsing issue
            return spread_value if team_name in spread else abs(spread_value)

        remaining_games["PEAR"] = remaining_games.apply(clean_spread, axis=1)
        remaining_games['home_win_prob'] = remaining_games.apply(
            lambda row: PEAR_Win_Prob(team_rating, row['Rating'], row['Location']) / 100, axis=1
        )
        max_net = len(stats_df)
        max_spread = 16.5

        w_tq = 0.7
        w_wp = 0.2
        w_ned = 0.1

        remaining_games['avg_net'] = (team_rank + remaining_games['NET']) / 2
        remaining_games['TQ'] = (max_net - remaining_games['avg_net']) / (max_net - 1)
        remaining_games['WP'] = 1 - 2 * np.abs(remaining_games['home_win_prob'] - 0.5)
        remaining_games['NED'] = 1 - (np.abs(team_rank - remaining_games['NET']) / (max_net - 1))

        remaining_games['GQI'] = round(10 * (
            w_tq * remaining_games['TQ'] +
            w_wp * remaining_games['WP'] +
            w_ned * remaining_games['NED']
        ),1)   

    team_completed = completed_schedule[completed_schedule['Team'] == team_name].reset_index(drop=True)
    num_rows = len(team_completed)
    last_n_games = team_completed['Result'].iloc[-10 if num_rows >= 10 else -num_rows:]
    wins = last_n_games.str.count('W').sum()
    losses = (10 if num_rows >= 10 else num_rows) - wins
    last_ten = f'{wins}-{losses}'

    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        win_prob = round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)
        return win_prob


    win_rating = 500
    best_win_opponent = ""
    loss_rating = 0
    worst_loss_opponent = ""
    for _, row in completed_schedule.iterrows():
        if row['Team'] == row['home_team']:
            if row['home_score'] > row['away_score']:
                if row['NET'] < win_rating:
                    win_rating = row['NET']
                    best_win_opponent = row['Opponent']
            else:
                if row['NET'] > loss_rating:
                    loss_rating = row['NET']
                    worst_loss_opponent = row['Opponent']
        else:
            if row['away_score'] > row['home_score']:
                if row['NET'] < win_rating:
                    win_rating = row['NET']
                    best_win_opponent = row['Opponent']
            else:
                if row['NET'] > loss_rating:
                    loss_rating = row['NET']
                    worst_loss_opponent = row['Opponent']
                
    return team_rank, best_win_opponent, worst_loss_opponent, remaining_games, completed_schedule, last_ten

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import random

def get_rating(team, stats_df):
    RATING_ADJUSTMENTS = {
        "Liberty": 0.8, "Xavier": 0.8, "Southeastern La.": 0.8, "UTRGV": 0.8,
        "Yale": 0.8, "Omaha": 0.8, "Stetson": 0.8, "Duke": 0.8,
        "Georgetown": 0.8, "High Point": 0.8, "Col. of Charleston": 0.8,
        "Illinois St.": 0.8, "Cal St. Fullerton": 0.8, "Wright St.":0.8
    }
    base_rating = stats_df.loc[stats_df["Team"] == team, "Rating"].values[0]
    return base_rating + RATING_ADJUSTMENTS.get(team, 0)

def simulate_game(team_a, team_b, ratings, location="Neutral"):
    prob = PEAR_Win_Prob(ratings[team_a], ratings[team_b], location=location) / 100
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

def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

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
        final_prob = PEAR_Win_Prob(r[w4], r[w5]) / 100
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

def simulate_conference_tournaments(schedule_df, stats_and_metrics, num_simulations, conference):

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import math
from PIL import Image

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

def get_conference_record(team, schedule_df, stats_and_metrics):
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
    
    return f"{wins_count}-{games_count - wins_count}"

def get_total_record(row):
    wins = sum(int(str(row[col]).split("-")[0]) for col in ["Q1", "Q2", "Q3", "Q4"])
    losses = sum(int(str(row[col]).split("-")[1]) for col in ["Q1", "Q2", "Q3", "Q4"])
    return f"{wins}-{losses}"

def get_location_records(team, schedule_df):
    # Filter games involving the team with a valid result
    df = schedule_df[
        ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
        (schedule_df['Result'].str.startswith(('W', 'L')))
    ].copy()

    # Determine location type
    def get_loc(row):
        if row['Location'] == 'Neutral':
            return 'Neutral'
        elif row['home_team'] == team:
            return 'Home'
        else:
            return 'Away'

    df['loc'] = df.apply(get_loc, axis=1)

    # Determine win/loss from Result
    df['is_win'] = df['Result'].str.startswith('W')

    # Group by location
    records = {}
    for loc in ['Home', 'Away', 'Neutral']:
        group = df[df['loc'] == loc]
        wins = group['is_win'].sum()
        losses = len(group) - wins
        records[loc] = f"{int(wins)}-{int(losses)}"

    return records

def team_visual(team_name, stats_and_metrics, schedule_df, comparison_date):
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
    team_url = BASE_URL + elo_data[elo_data['Team'] == team_name]['Team Link'].values[0]
    response = requests.get(team_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find("img", class_="team-menu__image")
    img_src = img_tag.get("src")
    image_url = BASE_URL + img_src
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    ax_img1 = fig.add_axes([0.04, 0.85, 0.2, 0.2])
    ax_img1.imshow(img, interpolation='lanczos')
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
    data_2022 = pd.read_csv("./PEAR/PEAR Baseball/y2022/Data/baseball_06_26_2022.csv")
    team_2022 = data_2022[data_2022['Team'] == team_name]
    data_2023 = pd.read_csv("./PEAR/PEAR Baseball/y2023/Data/baseball_06_26_2023.csv")
    team_2023 = data_2023[data_2023['Team'] == team_name]
    data_2024 = pd.read_csv("./PEAR/PEAR Baseball/y2024/Data/baseball_06_25_2024.csv")
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
    return fig

import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

def get_total_record(row):
    wins = sum(int(str(row[col]).split("-")[0]) for col in ["Q1", "Q2", "Q3", "Q4"])
    losses = sum(int(str(row[col]).split("-")[1]) for col in ["Q1", "Q2", "Q3", "Q4"])
    return f"{wins}-{losses}"

def team_percentiles_chart(team_name, stats_and_metrics):
    BASE_URL = "https://www.warrennolan.com"
    team_data = stats_and_metrics[stats_and_metrics['Team'] == team_name]
    team_net = team_data['NET'].values[0]
    team_proj_record = team_data['Projected_Record'].values[0]
    team_proj_net = team_data['Projected_NET'].values[0]
    team_record = get_total_record(team_data.iloc[0])
    team_Q1 = team_data['Q1'].values[0]
    team_Q2 = team_data['Q2'].values[0]
    team_Q3 = team_data['Q3'].values[0]
    team_Q4 = team_data['Q4'].values[0]
    team_url = BASE_URL + elo_data[elo_data['Team'] == team_name]['Team Link'].values[0]
    response = requests.get(team_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find("img", class_="team-menu__image")
    img_src = img_tag.get("src")
    image_url = BASE_URL + img_src
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    percentile_columns = ['pNET_Score', 'pRating', 'pResume_Quality', 'pPYTHAG', 'pfWAR', 'pwOBA', 'pOPS', 'pISO', 'pBB%', 'pFIP', 'pWHIP', 'pLOB%', 'pK/BB']
    team_data = team_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
    cmap = plt.get_cmap('seismic')
    colors = [cmap(p / 100) for p in team_data['Percentile']]
    def darken_color(color, factor=0.3):
        color = mcolors.hex2color(color)
        darkened_color = [max(c - factor, 0) for c in color]
        return mcolors.rgb2hex(darkened_color)
    darkened_colors = [darken_color(c) for c in colors]
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
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
    # fig.text(0.93, 0.72, 'Metrics', ha='center', va='center', fontsize=16, fontweight='bold', color='black', rotation=270)
    # fig.text(0.93, 0.47, 'Offense', ha='center', va='center', fontsize=16, fontweight='bold', color='black', rotation=270)
    # fig.text(0.93, 0.245, 'Pitching', ha='center', va='center', fontsize=16, fontweight='bold', color='black', rotation=270)
    fig.text(0.5, 0.93, f'#{team_net} {team_name} Percentile Rankings', fontsize=24, fontweight='bold', ha='center')
    fig.text(0.5, 0.90, f'Including PEAR Metrics, Offensive Stats, Pitching Stats', fontsize=16, ha='center')
    fig.text(0.5, 0.87, f'@PEARatings', fontsize=16, fontweight='bold', ha='center')

    plt.text(117, 0.5, f"{team_name}", ha='center', fontsize=16, fontweight='bold')
    plt.text(117, 1.0, f"{team_record}", ha='center', fontsize=16)
    plt.text(117, 2.2, "Quadrants", ha='center', fontsize=16, fontweight='bold')
    plt.text(111, 2.7, f"Q1: {team_Q1}", ha='left', fontsize=16)
    plt.text(111, 3.2, f"Q2: {team_Q2}", ha='left', fontsize=16)
    plt.text(111, 3.7, f"Q3: {team_Q3}", ha='left', fontsize=16)
    plt.text(111, 4.2, f"Q4: {team_Q4}", ha='left', fontsize=16)
    # plt.text(117, 5.4, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    # plt.text(117, 5.9, f"#{team_proj_net}", ha='center', fontsize=16)
    plt.text(117, 5.4, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(117, 5.9, f"{team_proj_record}", ha='center', fontsize=16)


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
    ax_img1 = fig.add_axes([0.94, 0.83, 0.15, 0.15])
    ax_img1.imshow(img)
    ax_img1.axis("off")

    return fig

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

def matchup_percentiles(team_1, team_2, stats_and_metrics, location):
    BASE_URL = "https://www.warrennolan.com"
    bubble_team_rating = stats_and_metrics['Rating'].quantile(0.90)
    percentile_columns = ['pNET_Score', 'pRating', 'pResume_Quality', 'pPYTHAG', 'pfWAR', 'pwOBA', 'pOPS', 'pISO', 'pBB%', 'pFIP', 'pWHIP', 'pLOB%', 'pK/BB']
    custom_labels = ['NET', 'TSR', 'RQI', 'PWP', 'WAR', 'wOBA', 'OPS', 'ISO', 'BB%', 'FIP', 'WHIP', 'LOB%', 'K/BB']

    team1_data = stats_and_metrics[stats_and_metrics['Team'] == team_1]
    team1_record = get_total_record(team1_data.iloc[0])
    team1_proj_record = team1_data['Projected_Record'].values[0]
    team1_proj_net = team1_data['Projected_NET'].values[0]
    team1_rating = team1_data['Rating'].values[0]
    team1_Q1 = team1_data['Q1'].values[0]
    team1_Q2 = team1_data['Q2'].values[0]
    team1_Q3 = team1_data['Q3'].values[0]
    team1_Q4 = team1_data['Q4'].values[0]
    team1_net = stats_and_metrics[stats_and_metrics['Team'] == team_1]['NET'].values[0]
    team1_data = team1_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
    team1_url = BASE_URL + elo_data[elo_data['Team'] == team_1]['Team Link'].values[0]
    response = requests.get(team1_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find("img", class_="team-menu__image")
    img_src = img_tag.get("src")
    image_url = BASE_URL + img_src
    response = requests.get(image_url)
    img1 = Image.open(BytesIO(response.content))

    team2_data = stats_and_metrics[stats_and_metrics['Team'] == team_2]
    team2_record = get_total_record(team2_data.iloc[0])
    team2_proj_record = team2_data['Projected_Record'].values[0]
    team2_proj_net = team2_data['Projected_NET'].values[0]
    team2_rating = team2_data['Rating'].values[0]
    team2_Q1 = team2_data['Q1'].values[0]
    team2_Q2 = team2_data['Q2'].values[0]
    team2_Q3 = team2_data['Q3'].values[0]
    team2_Q4 = team2_data['Q4'].values[0]
    team2_net = stats_and_metrics[stats_and_metrics['Team'] == team_2]['NET'].values[0]
    team2_data = team2_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
    team2_url = BASE_URL + elo_data[elo_data['Team'] == team_2]['Team Link'].values[0]
    response = requests.get(team2_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find("img", class_="team-menu__image")
    img_src = img_tag.get("src")
    image_url = BASE_URL + img_src
    response = requests.get(image_url)
    img2 = Image.open(BytesIO(response.content))

    team2_quality = PEAR_Win_Prob(bubble_team_rating, team1_rating, location) / 100
    team2_win_quality, team2_loss_quality = (1 - team2_quality), -team2_quality

    team1_quality = 1 - PEAR_Win_Prob(team2_rating, bubble_team_rating, location) / 100
    team1_win_quality, team1_loss_quality = (1 - team1_quality), -team1_quality
    spread, team_2_win_prob = find_spread_matchup(team_2, team_1, stats_and_metrics, location)

    max_net = 299
    w_tq = 0.70   # NET AVG
    w_wp = 0.20   # Win Probability
    w_ned = 0.10  # NET Differential
    avg_net = (team1_net + team2_net) / 2
    tq = (max_net - avg_net) / (max_net - 1)
    wp = 1 - 2 * np.abs((team_2_win_prob/100) - 0.5)
    ned = 1 - (np.abs(team2_net - team1_net) / (max_net - 1))
    gqi = round(10*(w_tq * tq + w_wp * wp + w_ned * ned), 1)

    team_2_win_prob = round(team_2_win_prob / 100,3)
    team_1_win_prob = 1 - team_2_win_prob
    team_2_probs, team_1_probs = calculate_series_probabilities(team_2_win_prob)
    team_2_one_win = team_2_probs[0]
    team_2_two_win = team_2_probs[1]
    team_2_three_win = team_2_probs[2]
    team_1_one_win = team_1_probs[0]
    team_1_two_win = team_1_probs[1]
    team_1_three_win = team_1_probs[2]
    combined = pd.DataFrame({
        'Metric': team1_data['Metric'],
        'Percentile': team1_data['Percentile'] - team2_data['Percentile']
    })
    cmap = plt.get_cmap('seismic')
    colors = [cmap(abs(p) / 100) for p in combined['Percentile']]
    colors1 = [cmap(p/100) for p in team1_data['Percentile']]
    colors2 = [cmap(p/100) for p in team2_data['Percentile']]
    def darken_color(color, factor=0.3):
        color = mcolors.hex2color(color)
        darkened_color = [max(c - factor, 0) for c in color]
        return mcolors.rgb2hex(darkened_color)
    darkened_colors = [darken_color(c) for c in colors]
    darkened_colors1 = [darken_color(c) for c in colors1]
    darkened_colors2 = [darken_color(c) for c in colors2]
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
    ax.barh(combined['Metric'], 99, color='gray', height=0.1, left=0)
    ax.barh(combined['Metric'], -99, color='gray', height=0.1, left=0)
    bars = ax.barh(combined['Metric'], combined['Percentile'], color=colors, height=0.3, edgecolor=darkened_colors, linewidth=3)
    bars1 = ax.barh(team1_data['Metric'], team1_data['Percentile'], color=colors1, height=0.3, edgecolor=darkened_colors1, linewidth=3)
    bars2 = ax.barh(team2_data['Metric'], -team2_data['Percentile'], color=colors2, height=0.3, edgecolor=darkened_colors2, linewidth=3)
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars1, team1_data['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors1[i], edgecolor=darkened_colors1[i], boxstyle='circle,pad=0.4', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        ax.text(0, bar.get_y() - 0.35, custom_labels[i], fontsize=12, fontweight='bold', ha='center', va='center')
        i = i+1
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars2, team2_data['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors2[i], edgecolor=darkened_colors2[i], boxstyle='circle,pad=0.4', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        i = i+1
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars, combined['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors[i], edgecolor=darkened_colors[i], boxstyle='circle,pad=0.3', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        i = i+1

    ax.set_xlim(-104, 104)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.text(0, -1.7, f"#{team2_net} {team_2} vs. #{team1_net} {team_1}", ha='center', fontsize=24, fontweight='bold')
    # plt.text(0, -1.25, "Matchup Comparison", ha='center', fontsize=16)
    plt.text(0, -1.25, f"Game Quality: {gqi}", ha='center', fontsize=16, fontweight='bold')
    plt.text(0, -0.8, f"{spread}", ha='center', fontsize=16, fontweight='bold')
    plt.text(0, 12.8, "@PEARatings", ha='center', fontsize=16, fontweight='bold')

    plt.text(-135, 0.5, f"{team_2}", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 1.0, f"{team2_record}", ha='center', fontsize=16)
    plt.text(-135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 2.5, f"{round(team_2_win_prob*100)}%", ha='center', fontsize=16)
    plt.text(-135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 4.0, f"Win 1: {round(team_2_one_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 4.5, f"Win 2: {round(team_2_two_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 5.0, f"Win 3: {round(team_2_three_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
    plt.text(-148, 6.5, f"Q1: {team2_Q1}", ha='left', fontsize=16)
    plt.text(-148, 7.0, f"Q2: {team2_Q2}", ha='left', fontsize=16)
    plt.text(-148, 7.5, f"Q3: {team2_Q3}", ha='left', fontsize=16)
    plt.text(-148, 8.0, f"Q4: {team2_Q4}", ha='left', fontsize=16)
    # plt.text(-135, 9.0, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    # plt.text(-135, 9.5, f"#{team2_proj_net}", ha='center', fontsize=16)
    plt.text(-135, 9.0, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 9.5, f"{team2_proj_record}", ha='center', fontsize=16)
    plt.text(-135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(-155, 11.0, f"{team2_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(-115, 11.0, f"{team2_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

    plt.text(135, 0.5, f"{team_1}", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 1.0, f"{team1_record}", ha='center', fontsize=16)
    plt.text(135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 2.5, f"{round(team_1_win_prob*100)}%", ha='center', fontsize=16)
    plt.text(135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 4.0, f"Win 1: {round(team_1_one_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 4.5, f"Win 2: {round(team_1_two_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 5.0, f"Win 3: {round(team_1_three_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
    plt.text(122, 6.5, f"Q1: {team1_Q1}", ha='left', fontsize=16)
    plt.text(122, 7.0, f"Q2: {team1_Q2}", ha='left', fontsize=16)
    plt.text(122, 7.5, f"Q3: {team1_Q3}", ha='left', fontsize=16)
    plt.text(122, 8.0, f"Q4: {team1_Q4}", ha='left', fontsize=16)
    # plt.text(135, 9.0, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    # plt.text(135, 9.5, f"#{team1_proj_net}", ha='center', fontsize=16)
    plt.text(135, 9.0, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 9.5, f"{team1_proj_record}", ha='center', fontsize=16)
    plt.text(135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(115, 11.0, f"{team1_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(155, 11.0, f"{team1_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

    plt.text(-150, 13.2, "Middle Bubble is Difference Between Team Percentiles", ha='left', fontsize = 12)
    plt.text(150, 13.2, "Series Percentages are the Chance to Win __ Games", ha='right', fontsize = 12)
    plt.text(-150, 13.6, "NET - PEAR's Ranking System, Combining TSR and RQI", ha='left', fontsize = 12)
    plt.text(150, 13.6, "TSR - Team Strength Rating, How Good Your Team Is", ha='right', fontsize = 12)
    plt.text(-150, 14.0, "RQI - Resume Quality Index, How Good Your Wins Are", ha='left', fontsize = 12)
    plt.text(150, 14.0, "PWP - Pythagorean Win Percent, Expected Win Rate", ha='right', fontsize = 12)

    ax_img1 = fig.add_axes([0.94, 0.83, 0.15, 0.15])
    ax_img1.imshow(img1)
    ax_img1.axis("off")
    ax_img2 = fig.add_axes([-0.065, 0.83, 0.15, 0.15])
    ax_img2.imshow(img2)
    ax_img2.axis("off")

    return fig

def conference_team_sheets(this_conference, stats_and_metrics):
    stats_and_metrics['WAR'] = stats_and_metrics['fWAR'].rank(ascending=False).astype(int)
    stats_and_metrics['PYT'] = stats_and_metrics['PYTHAG'].rank(ascending=False).astype(int)
    conference = stats_and_metrics[stats_and_metrics['Conference'] == this_conference].reset_index(drop=True)
    conference = conference[['Team', 'NET', 'RPI', 'PRR', 'PYT', 'RQI', 'WAB']]

    cmap = LinearSegmentedColormap.from_list(
        name="green_red",
        colors=[
            '#006400',  # Dark Green (start)
            '#d5f5e3',  # Light Green (transition from green)
            '#ffb6b6',  # Light Red (transition from green to red)
            '#8b0000'   # Dark Red (end)
        ],
        N=299  # Total color bins (1 to 299)
    )

    make_table = conference.set_index('Team')
    fig, ax = plt.subplots(figsize=(10, len(conference)*0.7))
    fig.patch.set_facecolor('#CECEB2')

    column_definitions = [
        ColumnDefinition(name='Team', # name of the column to change
                        title='Team', # new title for the column
                        textprops={"ha": "left", "weight": "bold", "fontsize": 16}, width = 0.55),
        ColumnDefinition(name='NET', # name of the column to change
                        title='NET', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap, border='left'),
        ColumnDefinition(name='RPI', # name of the column to change
                        title='RPI', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap),
        ColumnDefinition(name='PRR', # name of the column to change
                        title='TSR', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap, border='left'),
        ColumnDefinition(name='PYT', # name of the column to change
                        title='PYT', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap),
        ColumnDefinition(name='RQI', # name of the column to change
                        title='RQI', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap, border='left'),
        ColumnDefinition(name='WAB', # name of the column to change
                        title='WAB', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, width = 0.2,
                        cmap=cmap, border='right')
    ]

    tab = Table(make_table, column_definitions=column_definitions, footer_divider=True, row_divider_kw={"linewidth": 1})
    tab.col_label_row.set_facecolor('#CECEB2')
    tab.columns["Team"].set_facecolor('#CECEB2')
    plt.text(0.055,-1.65, f"{this_conference} Team Sheets", fontsize = 24, fontweight='bold', ha='left')
    plt.text(0.055,-1, f"@PEARatings", fontsize = 16, ha='left')
    plt.text(0.055,len(conference)+0.6, f"NET - PEAR's Ranking", fontsize = 16, ha='left')
    plt.text(1.05,len(conference)+0.6, f"RPI - Nolan's Live RPI", fontsize = 16, ha='left')
    plt.text(0.055,len(conference)+1.2, f"PYT - Pythagorean Win Percentage", fontsize = 16, ha='left')
    plt.text(1.05,len(conference)+1.2, f"TSR - Team Strength Rating", fontsize = 16, ha='left')
    plt.text(0.055,len(conference)+1.8, f"RQI - Resume Quality Index", fontsize = 16, ha='left')
    plt.text(1.05,len(conference)+1.8, f"WAB - Wins Above Bubble", fontsize = 16, ha='left')
    return fig

def get_conference(team, stats_df):
    return stats_df.loc[stats_df["Team"] == team, "Conference"].values[0]

from collections import Counter

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

def simulate_tournament_home_field(teams, ratings):
    import random
    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)

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

def run_simulation_home_field(team_a, team_b, team_c, team_d, stats_and_metrics, num_simulations=5000):
    teams = [team_a, team_b, team_c, team_d]
    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}
    results = defaultdict(int)

    for _ in range(num_simulations):
        winner = simulate_tournament_home_field(teams, ratings)
        results[winner] += 1

    total = num_simulations
    return defaultdict(float, {team: round(count / total, 3) for team, count in results.items()})

def simulate_regional(team_a, team_b, team_c, team_d, stats_and_metrics):

    output = run_simulation_home_field(team_a, team_b, team_c, team_d, stats_and_metrics)
    regional_prob = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
    seed_map = {
        team_a: "#1 " + team_a,
        team_b: "#2 " + team_b,
        team_c: "#3 " + team_c,
        team_d: "#4 " + team_d,
    }
    regional_prob["Team"] = regional_prob["Team"].map(seed_map)
    regional_prob['Win Regional'] = regional_prob['Win Regional'] * 100
    seed_order = [seed_map[team_a], seed_map[team_b], seed_map[team_c], seed_map[team_d]]
    regional_prob["SeedOrder"] = regional_prob["Team"].apply(lambda x: seed_order.index(x))
    regional_prob = regional_prob.sort_values("SeedOrder").drop(columns="SeedOrder")

    # Normalize values for color gradient (excluding 0 values)
    min_value = regional_prob.iloc[:, 1:].replace(0, np.nan).min().min()  # Min of all probabilities
    max_value = regional_prob.iloc[:, 1:].max().max()  # Max of all probabilities

    def normalize(value, min_val, max_val):
        """ Normalize values between 0 and 1 for colormap. """
        if pd.isna(value) or value == 0:
            return 0  # Keep 0 values at the lowest color
        return (value - min_val) / (max_val - min_val)

    # Define custom colormap (lighter green to dark green)
    cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 4), dpi=125)
    fig.patch.set_facecolor('#CECEB2')

    ax.axis('tight')
    ax.axis('off')

    # Add the table
    table = ax.table(
        cellText=regional_prob.values,
        colLabels=regional_prob.columns,
        cellLoc='center',
        loc='center',
        colColours=['#CECEB2'] * len(regional_prob.columns)  # Set the header background color
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
            value = regional_prob.iloc[i-1, j]  # Skip header row
            normalized_value = normalize(value, min_value, max_value)
            color = cmap(normalized_value)
            cell.set_facecolor(color)
            cell.set_text_props(fontsize=14, weight='bold', color='black')
            if value == 0:
                cell.get_text().set_text("<1%")
            else:
                cell.get_text().set_text(f"{value:.1f}%")

        cell.set_height(0.2)

    # Show the plot
    plt.text(0, 0.07, f'{team_a} Regional', fontsize=16, fontweight='bold', ha='center')
    plt.text(0, 0.06, f"@PEARatings", fontsize=12, fontweight='bold', ha='center')
    return fig

def calculate_conference_results(schedule_df, comparison_date, stats_and_metrics, num_simulations=100):
    # Ensure "Date" is in datetime format
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])

    # --- Step 1: Filter for intra-conference games only ---

    team_conferences = stats_and_metrics.set_index("Team")["Conference"].to_dict()
    schedule_df["home_conf"] = schedule_df["home_team"].map(team_conferences)
    schedule_df["away_conf"] = schedule_df["away_team"].map(team_conferences)
    schedule_df = schedule_df[schedule_df["home_conf"] == schedule_df["away_conf"]].reset_index(drop=True).copy()

    # --- Step 2: Filter to valid 3-game intra-conference series ---

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
    schedule_df = schedule_df.iloc[valid_indices].reset_index(drop=True)

    # --- Step 3: Split into completed and remaining games based on comparison_date ---

    completed_schedule = schedule_df[
        (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
    ].reset_index(drop=True)

    remaining_games = schedule_df[
        (schedule_df["Date"] >= comparison_date) & 
        (~schedule_df["Result"].str.startswith(("W", "L")))
    ].reset_index(drop=True)
    completed_schedule = completed_schedule[completed_schedule["Result"].str.startswith(("W", "L"))]

    # --- Step 4: Compute current conference records from completed games ---

    completed_schedule["home_conf"] = completed_schedule["home_team"].map(team_conferences)
    completed_schedule["away_conf"] = completed_schedule["away_team"].map(team_conferences)
    conf_games = completed_schedule[completed_schedule["home_conf"] == completed_schedule["away_conf"]].copy()
    conf_games["Conference"] = conf_games["home_conf"]
    conf_games["winner"] = np.where(conf_games["home_score"] > conf_games["away_score"], conf_games["home_team"], conf_games["away_team"])
    conf_games["loser"] = np.where(conf_games["home_score"] > conf_games["away_score"], conf_games["away_team"], conf_games["home_team"])
    win_counts = conf_games.groupby("winner").size().reset_index(name="Conf_Wins")
    loss_counts = conf_games.groupby("loser").size().reset_index(name="Conf_Losses")
    conference_records = pd.merge(win_counts, loss_counts, left_on="winner", right_on="loser", how="outer")
    conference_records["Team"] = conference_records["winner"].combine_first(conference_records["loser"])
    conference_records = conference_records[["Team", "Conf_Wins", "Conf_Losses"]].fillna(0)
    conference_records["Conf_Wins"] = (conference_records["Conf_Wins"] / 2).astype(int)
    conference_records["Conf_Losses"] = (conference_records["Conf_Losses"] / 2).astype(int)
    conference_records["Conference"] = conference_records["Team"].map(team_conferences)

    # --- Step 5: Simulate outcomes of remaining games ---

    remaining_games = remaining_games.reset_index(drop=True).copy()
    all_teams = remaining_games["Team"].unique()
    team_idx_map = {team: idx for idx, team in enumerate(all_teams)}
    team_win_matrix = np.zeros((num_simulations, len(all_teams)), dtype=int)
    home_teams = remaining_games["home_team"].values
    away_teams = remaining_games["away_team"].values
    probs = remaining_games["home_win_prob"].values
    teams_in_game = remaining_games["Team"].values

    for sim in range(num_simulations):
        random_vals = np.random.rand(len(remaining_games))
        home_wins = random_vals < probs
        winners = np.where(home_wins, home_teams, away_teams)
        win_flags = (winners == teams_in_game).astype(int)
        for team, wins in zip(teams_in_game, win_flags):
            team_win_matrix[sim, team_idx_map[team]] += wins

    avg_wins = team_win_matrix.mean(axis=0).round().astype(int)
    projected_wins_df = pd.DataFrame({
        "Team": list(team_idx_map.keys()),
        "Remaining_Wins": avg_wins
    })
    games_remaining = remaining_games.groupby("Team").size()
    projected_wins_df["Games_Remaining"] = projected_wins_df["Team"].map(games_remaining).fillna(0).astype(int)
    projected_wins_df["Remaining_Losses"] = projected_wins_df["Games_Remaining"] - projected_wins_df["Remaining_Wins"]
    projected_wins_df["Conference"] = projected_wins_df["Team"].map(team_conferences)

    # --- Step 6: Merge current and projected records into final standings ---
    combined = pd.merge(
        conference_records,
        projected_wins_df,
        on=["Team", "Conference"],
        how="left"
    )
    combined[["Conf_Wins", "Conf_Losses", "Remaining_Wins", "Games_Remaining", "Remaining_Losses"]] = combined[["Conf_Wins", "Conf_Losses", "Remaining_Wins", "Games_Remaining", "Remaining_Losses"]].fillna(0).astype(int)
    combined["Proj_Conf_Wins"] = (combined["Conf_Wins"] + combined["Remaining_Wins"])
    combined["Proj_Conf_Losses"] = (combined["Conf_Losses"] + combined["Remaining_Losses"])

    return combined, completed_schedule, remaining_games

def conference_projected_standing(this_conference, projected_wins):
    conf_df = projected_wins[projected_wins['Conference'] == this_conference].sort_values('Proj_Conf_Wins', ascending=False).reset_index(drop=True)
    conf_df = conf_df[['Team', 'Projected_Conf_Record', 'Current_Conf_Record', 'Remaining_Conf_Record']]

    # --- Build table ---
    make_table = conf_df.set_index('Team')
    fig, ax = plt.subplots(figsize=(12, len(make_table) * 0.7))
    fig.patch.set_facecolor('#CECEB2')

    column_definitions = [
        ColumnDefinition(name='Team',
                        title='Team',
                        textprops={"ha": "left", "weight": "bold", "fontsize": 16},
                        width=0.5),
        ColumnDefinition(
            name='Projected_Conf_Record',
            title='Projected Record',
            textprops={"ha": "center", "fontsize": 16},
            width=0.5
        ),
        ColumnDefinition(
            name='Current_Conf_Record',
            title='Current Record',
            textprops={"ha": "center", "fontsize": 16},
            width=0.5
        ),
        ColumnDefinition(
            name='Remaining_Conf_Record',
            title='Remaining Record',
            textprops={"ha": "center", "fontsize": 16},
            width=0.5
        ),
    ]

    tab = Table(make_table, column_definitions=column_definitions, footer_divider=True, row_divider_kw={"linewidth": 1})
    tab.col_label_row.set_facecolor('#CECEB2')
    tab.columns["Team"].set_facecolor('#CECEB2')
    tab.columns['Projected_Conf_Record'].set_facecolor('#CECEB2')
    tab.columns['Current_Conf_Record'].set_facecolor('#CECEB2')
    tab.columns['Remaining_Conf_Record'].set_facecolor('#CECEB2')

    plt.text(0.055, -1.65, f"{this_conference} Projected Standings", fontsize=24, fontweight='bold', ha='left')
    plt.text(0.055, -1, f"@PEARatings", fontsize=16, ha='left')
    return fig

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def plot_rating_vs_net(team_name, df):
    """
    Scatter plot of Rating vs NET_Score for a given team, using distinct colors by season.
    National champions are highlighted in gold. Legend is ordered by season.
    """
    team_data = df[df['Team'] == team_name].copy()
    team_data['Season'] = team_data['Season'].astype(int)
    team_data = team_data.sort_values('Season')
    team_avg_rating = team_data['Normalized_Rating'].mean()
    team_avg_net = team_data['NET_Score'].mean()

    if team_data.empty:
        print(f"No data found for team: {team_name}")
        return

    seasons = sorted(team_data['Season'].unique())

    # Set up the figure and background color
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')

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
                            facecolor="#C8416E",  # Peach Puff
                            linewidth=2,
                            alpha=0.35,
                            mutation_scale=0.05,
                            zorder=1)
    ax.add_patch(elite_box)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Highlight national champions in gold
    champions = [
        ("Fresno St.", 2008), ("LSU", 2009), ("South Carolina", 2010), ("South Carolina", 2011),
        ('Arizona', 2012), ('UCLA', 2013), ('Vanderbilt', 2014), ('Virginia', 2015),
        ('Coastal Carolina', 2016), ('Florida', 2017), ('Oregon St.', 2018),
        ('Vanderbilt', 2019), ("Mississippi St.", 2021), ('Ole Miss', 2022), ('LSU', 2023), ('Tennessee', 2024),
        ('LSU', 2025)
    ]
    champ_df = df[df.apply(lambda row: (row['Team'], row['Season']) in champions, axis=1)]

    if not champ_df.empty:
        ax.scatter(champ_df['Normalized_Rating'], champ_df['NET_Score'],
                   color='gold', edgecolor='black', s=200, zorder=5, alpha=0.3)
    for _, row in champ_df.iterrows():
        ax.text(row['Normalized_Rating'], row['NET_Score'], str(row['Season'])[2:],
                fontsize=5, ha='center', va='center', color='black', fontweight='bold')

    # Plot each season with its own color
    for season in seasons:
        season_data = team_data[team_data['Season'] == season]
        ax.scatter(season_data['Normalized_Rating'], season_data['NET_Score'],
                   color='darkgreen', edgecolor='black', s=400, label=str(season), alpha=0.7)

    for _, row in team_data.iterrows():
        ax.text(row['Normalized_Rating'], row['NET_Score'], str(row['Season'])[2:],
                fontsize=9, ha='center', va='center', color='black', fontweight='bold')

    # Labels and styling
    plt.title(f"Team Strength vs NET for {team_name} Since 2008 (excl. '20)", 
                 fontsize=16, color='black', pad=30, fontweight='bold')
    plt.suptitle("@PEARatings", y=0.85, fontsize=14, ha='center', va='center')
    # ax.text(-1.2, 1.04, "@PEARatings", fontsize=14, ha='center', va='center')
    ax.set_xlabel('Team Strength', color='black')
    ax.set_ylabel('NET Score', color='black')
    ax.tick_params(colors='black')
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    x_min = df['Normalized_Rating'].min()
    x_max = df['Normalized_Rating'].max()
    x_mean = df['Normalized_Rating'].mean()
    y_mean = df['NET_Score'].mean()

    if team_data['Normalized_Rating'].min() < x_mean:
        ax.axvline(x_mean, linestyle='--', linewidth=1, color='red')
        ax.text(x_mean-0.02, ax.get_ylim()[1]-0.02, f'Avg Rating', 
                color='red', fontsize=9, ha='right', va='top', rotation=90, )
    if team_data['NET_Score'].min() < y_mean:
        ax.axhline(y_mean, linestyle='--', linewidth=1, color='red')
        ax.text(ax.get_xlim()[1]-0.02, y_mean-0.005, f'Avg NET', 
                color='red', fontsize=9, ha='right', va='top')
    avg_rating_patch = mpatches.Patch(color='none', label=f'Average Rating is 0', linewidth=0)
    team_rating_patch = mpatches.Patch(color='none', label=f"{team_name} avg: {team_avg_rating:.2f}")
    avg_net_patch = mpatches.Patch(color='none', label=f'Average NET is 0.56', linewidth=0)
    team_net_patch = mpatches.Patch(color='none', label=f"{team_name} avg: {team_avg_net:.2f}")

    leg = ax.legend(handles=[avg_rating_patch, team_rating_patch, avg_net_patch, team_net_patch],
                    fontsize=10,
                    title_fontsize=11,
                    loc='best',
                    frameon=True,
                    facecolor='#4B5320',  # Dark olive green for contrast
                    edgecolor='black',
                    handlelength=0,
                    handletextpad=0,
                    borderpad=1,
                    labelspacing=0.6)

    # Center the legend text labels horizontally
    for text in leg.get_texts():
        text.set_ha('center')

    plt.tight_layout()
    return fig

def find_spread_matchup_with_year(home_team, away_team, home_year, away_year, modeling_stats, location="Neutral"):
    def get_stat(team, year, stat):
        match = modeling_stats[(modeling_stats['Team'] == team) & (modeling_stats['Season'] == year)]
        if not match.empty and pd.notna(match[stat].values[0]):
            return match[stat].values[0]
        return np.nan

    def approximate_elo(net_value, net_max=300, elo_min=1000, elo_max=1600):
        # Convert NET (where 0 is best) to ELO (where higher is better)
        if pd.isna(net_value):
            return 1200
        scale = elo_max - elo_min
        normalized = 1 - (net_value / net_max)  # invert: low NET = better
        return round(elo_min + normalized * scale)

    home_pr = get_stat(home_team, home_year, 'Normalized_Rating')
    away_pr = get_stat(away_team, away_year, 'Normalized_Rating')

    home_elo = get_stat(home_team, home_year, 'ELO')
    if pd.isna(home_elo):
        home_net = get_stat(home_team, home_year, 'NET')
        home_elo = approximate_elo(home_net)

    away_elo = get_stat(away_team, away_year, 'ELO')
    if pd.isna(away_elo):
        away_net = get_stat(away_team, away_year, 'NET')
        away_elo = approximate_elo(away_net)

    spread, elo_win_prob = calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location)
    # if location != "Neutral":
    #     rating_diff = home_pr + 0.5 - away_pr
    # else:
    #     rating_diff = home_pr - away_pr
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

    if spread >= 0:
        return f"{home_year} {home_team} -{spread}", win_prob
    else:
        return f"{away_year} {away_team} {spread}", win_prob

def matchup_percentiles_with_year(team_1, team_2, team_1_year, team_2_year, stats_and_metrics, location = 'Neutral'):
    BASE_URL = "https://www.warrennolan.com"
    bubble_team_rating = stats_and_metrics['Rating'].quantile(0.90)
    percentile_columns = ['pNET_Score', 'pRating', 'pResume_Quality', 'pPYTHAG', 'pfWAR', 'pwOBA', 'pOPS', 'pISO', 'pBB%', 'pFIP', 'pWHIP', 'pLOB%', 'pK/BB']
    custom_labels = ['NET', 'TSR', 'RQI', 'PWP', 'WAR', 'wOBA', 'OPS', 'ISO', 'BB%', 'FIP', 'WHIP', 'LOB%', 'K/BB']

    team1_data = stats_and_metrics[(stats_and_metrics['Team'] == team_1) & (stats_and_metrics['Season'] == team_1_year)]
    team1_record = get_total_record(team1_data.iloc[0])
    team1_proj_record = team1_data['Projected_Record'].values[0]
    team1_proj_net = team1_data['Projected_NET'].values[0]
    team1_rating = team1_data['Rating'].values[0]
    team1_Q1 = team1_data['Q1'].values[0]
    team1_Q2 = team1_data['Q2'].values[0]
    team1_Q3 = team1_data['Q3'].values[0]
    team1_Q4 = team1_data['Q4'].values[0]
    team1_net = stats_and_metrics[(stats_and_metrics['Team'] == team_1) & (stats_and_metrics['Season'] == team_1_year)]['NET'].values[0]
    team1_data = team1_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
    img1 = None
    try:
        team_links = stats_and_metrics[stats_and_metrics['Team'] == team_1]['Team Link']
        valid_links = team_links[team_links.notna()]

        if not valid_links.empty:
            team1_link = valid_links.values[0]
            team1_url = BASE_URL + team1_link

            if len(team1_url) > len(BASE_URL):
                response = requests.get(team1_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                img_tag = soup.find("img", class_="team-menu__image")

                if img_tag and img_tag.get("src"):
                    img_src = img_tag.get("src")
                    image_url = BASE_URL + img_src
                    response = requests.get(image_url)
                    img1 = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to fetch image for {team_1}: {e}")

    team2_data = stats_and_metrics[(stats_and_metrics['Team'] == team_2) & (stats_and_metrics['Season'] == team_2_year)]
    team2_record = get_total_record(team2_data.iloc[0])
    team2_proj_record = team2_data['Projected_Record'].values[0]
    team2_proj_net = team2_data['Projected_NET'].values[0]
    team2_rating = team2_data['Rating'].values[0]
    team2_Q1 = team2_data['Q1'].values[0]
    team2_Q2 = team2_data['Q2'].values[0]
    team2_Q3 = team2_data['Q3'].values[0]
    team2_Q4 = team2_data['Q4'].values[0]
    team2_net = stats_and_metrics[(stats_and_metrics['Team'] == team_2) & (stats_and_metrics['Season'] == team_2_year)]['NET'].values[0]
    team2_data = team2_data[percentile_columns].melt(var_name='Metric', value_name='Percentile')
    img2 = None
    try:
        team_links = stats_and_metrics[stats_and_metrics['Team'] == team_2]['Team Link']
        valid_links = team_links[team_links.notna()]

        if not valid_links.empty:
            team2_link = valid_links.values[0]
            team2_url = BASE_URL + team2_link

            if len(team2_url) > len(BASE_URL):
                response = requests.get(team2_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                img_tag = soup.find("img", class_="team-menu__image")

                if img_tag and img_tag.get("src"):
                    img_src = img_tag.get("src")
                    image_url = BASE_URL + img_src
                    response = requests.get(image_url)
                    img2 = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to fetch image for {team_2}: {e}")

    team2_quality = PEAR_Win_Prob(bubble_team_rating, team1_rating, location) / 100
    team2_win_quality, team2_loss_quality = (1 - team2_quality), -team2_quality

    team1_quality = 1-PEAR_Win_Prob(team2_rating, bubble_team_rating, location) / 100
    team1_win_quality, team1_loss_quality = (1 - team1_quality), -team1_quality
    spread, team_2_win_prob = find_spread_matchup_with_year(team_2, team_1, team_2_year, team_1_year, stats_and_metrics, location)

    max_net = 299
    w_tq = 0.70   # NET AVG
    w_wp = 0.20   # Win Probability
    w_ned = 0.10  # NET Differential
    avg_net = (team1_net + team2_net) / 2
    tq = (max_net - avg_net) / (max_net - 1)
    wp = 1 - 2 * np.abs((team_2_win_prob/100) - 0.5)
    ned = 1 - (np.abs(team2_net - team1_net) / (max_net - 1))
    gqi = round(10*(w_tq * tq + w_wp * wp + w_ned * ned), 1)

    team_2_win_prob = round(team_2_win_prob / 100,3)
    team_1_win_prob = 1 - team_2_win_prob
    team_2_probs, team_1_probs = calculate_series_probabilities(team_2_win_prob)
    team_2_one_win = team_2_probs[0]
    team_2_two_win = team_2_probs[1]
    team_2_three_win = team_2_probs[2]
    team_1_one_win = team_1_probs[0]
    team_1_two_win = team_1_probs[1]
    team_1_three_win = team_1_probs[2]
    combined = pd.DataFrame({
        'Metric': team1_data['Metric'],
        'Percentile': team1_data['Percentile'] - team2_data['Percentile']
    })
    cmap = plt.get_cmap('seismic')
    colors = [cmap(abs(p) / 100) for p in combined['Percentile']]
    colors1 = [cmap(p/100) for p in team1_data['Percentile']]
    colors2 = [cmap(p/100) for p in team2_data['Percentile']]
    def darken_color(color, factor=0.3):
        color = mcolors.hex2color(color)
        darkened_color = [max(c - factor, 0) for c in color]
        return mcolors.rgb2hex(darkened_color)
    darkened_colors = [darken_color(c) for c in colors]
    darkened_colors1 = [darken_color(c) for c in colors1]
    darkened_colors2 = [darken_color(c) for c in colors2]
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
    ax.barh(combined['Metric'], 99, color='gray', height=0.1, left=0)
    ax.barh(combined['Metric'], -99, color='gray', height=0.1, left=0)
    bars = ax.barh(combined['Metric'], combined['Percentile'], color=colors, height=0.3, edgecolor=darkened_colors, linewidth=3)
    bars1 = ax.barh(team1_data['Metric'], team1_data['Percentile'], color=colors1, height=0.3, edgecolor=darkened_colors1, linewidth=3)
    bars2 = ax.barh(team2_data['Metric'], -team2_data['Percentile'], color=colors2, height=0.3, edgecolor=darkened_colors2, linewidth=3)
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars1, team1_data['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors1[i], edgecolor=darkened_colors1[i], boxstyle='circle,pad=0.4', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        ax.text(0, bar.get_y() - 0.35, custom_labels[i], fontsize=12, fontweight='bold', ha='center', va='center')
        i = i+1
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars2, team2_data['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors2[i], edgecolor=darkened_colors2[i], boxstyle='circle,pad=0.4', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        i = i+1
    i = 0
    for idx, (bar, percentile) in enumerate(zip(bars, combined['Percentile'])):
        text = ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    str(abs(percentile)), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white', zorder=2,
                    bbox=dict(facecolor=colors[i], edgecolor=darkened_colors[i], boxstyle='circle,pad=0.3', linewidth=3))
        text.set_path_effects([
            pe.withStroke(linewidth=2, foreground='black')
        ])
        i = i+1

    ax.set_xlim(-104, 104)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.text(0, -1.7, f"{team_2_year} {team_2} vs. {team_1_year} {team_1}", ha='center', fontsize=24, fontweight='bold')
    # plt.text(0, -1.25, "Matchup Comparison", ha='center', fontsize=16)
    plt.text(0, -1.25, f"Game Quality: {gqi}", ha='center', fontsize=16, fontweight='bold')
    plt.text(0, -0.8, f"{spread}", ha='center', fontsize=16, fontweight='bold')
    plt.text(0, 12.8, "@PEARatings", ha='center', fontsize=16, fontweight='bold')

    plt.text(-135, 0.5, f"{team_2}", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 1.0, f"{team2_record}", ha='center', fontsize=16)
    plt.text(-135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 2.5, f"{round(team_2_win_prob*100)}%", ha='center', fontsize=16)
    plt.text(-135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 4.0, f"Win 1: {round(team_2_one_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 4.5, f"Win 2: {round(team_2_two_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 5.0, f"Win 3: {round(team_2_three_win*100)}%", ha='center', fontsize=16)
    plt.text(-135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
    plt.text(-148, 6.5, f"Q1: {team2_Q1}", ha='left', fontsize=16)
    plt.text(-148, 7.0, f"Q2: {team2_Q2}", ha='left', fontsize=16)
    plt.text(-148, 7.5, f"Q3: {team2_Q3}", ha='left', fontsize=16)
    plt.text(-148, 8.0, f"Q4: {team2_Q4}", ha='left', fontsize=16)
    plt.text(-135, 9.0, f"{team_2_year} NET", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 9.5, f"#{team2_net}", ha='center', fontsize=16)
    # plt.text(-135, 10.5, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    # plt.text(-135, 11.0, f"{team2_proj_record}", ha='center', fontsize=16)
    plt.text(-135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(-155, 11.0, f"{team2_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(-115, 11.0, f"{team2_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

    plt.text(135, 0.5, f"{team_1}", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 1.0, f"{team1_record}", ha='center', fontsize=16)
    plt.text(135, 2.0, "Single Game", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 2.5, f"{round(team_1_win_prob*100)}%", ha='center', fontsize=16)
    plt.text(135, 3.5, "Series", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 4.0, f"Win 1: {round(team_1_one_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 4.5, f"Win 2: {round(team_1_two_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 5.0, f"Win 3: {round(team_1_three_win*100)}%", ha='center', fontsize=16)
    plt.text(135, 6.0, "NET Quads", ha='center', fontsize=16, fontweight='bold')
    plt.text(122, 6.5, f"Q1: {team1_Q1}", ha='left', fontsize=16)
    plt.text(122, 7.0, f"Q2: {team1_Q2}", ha='left', fontsize=16)
    plt.text(122, 7.5, f"Q3: {team1_Q3}", ha='left', fontsize=16)
    plt.text(122, 8.0, f"Q4: {team1_Q4}", ha='left', fontsize=16)
    plt.text(135, 9.0, f"{team_1_year} NET", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 9.5, f"#{team1_net}", ha='center', fontsize=16)
    # plt.text(135, 10.5, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    # plt.text(135, 11.0, f"{team1_proj_record}", ha='center', fontsize=16)
    plt.text(135, 10.5, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(115, 11.0, f"{team1_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(155, 11.0, f"{team1_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

    plt.text(-150, 13.2, "Middle Bubble is Difference Between Team Percentiles", ha='left', fontsize = 12)
    plt.text(150, 13.2, "Series Percentages are the Chance to Win __ Games", ha='right', fontsize = 12)
    plt.text(-150, 13.6, "NET - PEAR's Ranking System, Combining TSR and RQI", ha='left', fontsize = 12)
    plt.text(150, 13.6, "TSR - Team Strength Rating, How Good Your Team Is", ha='right', fontsize = 12)
    plt.text(-150, 14.0, "RQI - Resume Quality Index, How Good Your Wins Are", ha='left', fontsize = 12)
    plt.text(150, 14.0, "PWP - Pythagorean Win Percent, Expected Win Rate", ha='right', fontsize = 12)

    if img1 is not None:
        ax_img1 = fig.add_axes([0.94, 0.83, 0.15, 0.15])
        ax_img1.imshow(img1)
        ax_img1.axis("off")

    if img2 is not None:
        ax_img2 = fig.add_axes([-0.065, 0.83, 0.15, 0.15])
        ax_img2.imshow(img2)
        ax_img2.axis("off")

    return fig

st.title(f"{current_season} CBASE PEAR")
st.logo("./PEAR/pear_logo.jpg", size = 'large')
st.caption(f"Ratings Updated {formatted_latest_date}")
st.caption(f"Stats Through Games {last_date}")
st.caption(f"If you get an error, notice a bug, have a suggestion, or a question to ask, reach out to me!")
if len(modeling_stats) != 299:
    st.title("WARNING: Something is wrong with NCAA Data, which causes errors here. Be weary until this message is gone.")

st.sidebar.markdown("""
    <style>
        .nav-link {
            display: block;
            padding: 12px;
            margin: 4px 0;
            background-color: #262730;  /* Dark grey background */
            color: white;
            text-align: left;
            border-radius: 4px;
            text-decoration: none;
            font-weight: bold;
            font-size: 16px;
            transition: background-color 0.3s ease-in-out;
        }
        .nav-link:hover {
            background-color: #3a3b46;  /* Slightly lighter grey on hover */
        }
    </style>
    <a class="nav-link" href="#ratings-and-resume">Ratings and Resume</a>
    <a class="nav-link" href="#team-stats">Team Stats</a>
    <a class="nav-link" href="#tournament-outlook">Tournament Outlook</a>
    <a class="nav-link" href="#matchup-cards">Matchup Cards</a>
    <a class="nav-link" href="#team-profile">Team Profile</a>
    <a class="nav-link" href="#conference-team-sheets">Conference Team Sheets</a>
    <a class="nav-link" href="#simulate-regional">Simulate Regional</a>
    <a class="nav-link" href="#projected-conference-tournament">Projected Conference Tournament</a>
    <a class="nav-link" href='#historical-team-performance'>Historical Team Performance</a>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<h2 id="ratings-and-resume">Ratings and Resume</h2>', unsafe_allow_html=True)
    modeling_stats_copy = modeling_stats.copy()
    modeling_stats_copy.set_index("Team", inplace=True)
    modeling_stats_copy['TSR'] = modeling_stats_copy['PRR']
    modeling_stats_copy['ELO'] = modeling_stats_copy['ELO_Rank']
    with st.container(border=True, height=440):
        st.dataframe(modeling_stats_copy[['NET', 'RPI', 'ELO', 'TSR', 'RQI', 'SOS', 'Q1', 'Q2', 'Q3', 'Q4', 'Conference']], use_container_width=True)
    st.caption("NET - Mimicing the NCAA Evaluation Tool using TSR, RQI, SOS")
    st.caption("RPI - Warren Nolan's Live RPI, TSR - Team Strength Rank, RQI - Resume Quality Index, SOS - Strength of Schedule")
with col2:
    modeling_stats_copy['WAR'] = modeling_stats_copy['fWAR'].rank(ascending=False).astype(int)
    modeling_stats_copy['oWAR'] = modeling_stats_copy['oWAR_z'].rank(ascending=False).astype(int)
    modeling_stats_copy['pWAR'] = modeling_stats_copy['pWAR_z'].rank(ascending=False).astype(int)
    columns_to_rank = ['PYTHAG', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS']
    modeling_stats_copy[columns_to_rank] = modeling_stats_copy[columns_to_rank].rank(ascending=False, method='min').astype(int)
    modeling_stats_copy['ERA'] = modeling_stats_copy['ERA'].rank(ascending=True, method='min').astype(int)
    modeling_stats_copy['WHIP'] = modeling_stats_copy['WHIP'].rank(ascending=True, method='min').astype(int)

    st.markdown(f'<h2 id="team-stats">Team Stats</h2>', unsafe_allow_html=True)
    with st.container(border=True, height=440):
        st.dataframe(modeling_stats_copy[['WAR', 'WPOE', 'PYTHAG', 'ERA', 'WHIP', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS', 'Conference']], use_container_width=True)
    st.caption("WAR - Team WAR Rank, WPOE - Win Percentage Over Expectation (Luck), PYTHAG - Pythagorean Win Percentage, ERA - Earned Run Average, WHIP - Walks Hits Over Innings Pitched, KP9 - Strikeouts Per 9, RPG - Runs Score Per Game, BA - Batting Average, OBP - On Base Percentage, SLG - Slugging Percentage, OPS - On Base Plus Slugging")

st.divider()

col1, col2 = st.columns(2)
with col1:
    # automatic_qualifiers = modeling_stats.loc[modeling_stats.groupby("Conference")["NET"].idxmin()]
    # at_large = modeling_stats.drop(automatic_qualifiers.index)
    # at_large = at_large.nsmallest(34, "NET")
    # last_four_in = at_large[-4:].reset_index()
    # next_8_teams = modeling_stats.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    # tournament = pd.concat([at_large, automatic_qualifiers])
    # tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    # tournament["Seed"] = (tournament.index // 16) + 1

    # if "Northeastern" in tournament.loc[tournament["Seed"] == 1, "Team"].values:
    #     # Get Northeastern's row and the #17 team (first Seed 2) row
    #     northeastern_idx = tournament[(tournament["Team"] == "Northeastern") & (tournament["Seed"] == 1)].index[0]
    #     seed17_idx = tournament[tournament["Seed"] == 2].index[0]
    #     # Swap rows
    #     tournament.iloc[northeastern_idx], tournament.iloc[seed17_idx] = tournament.iloc[seed17_idx].copy(), tournament.iloc[northeastern_idx].copy()
    #     tournament["Seed"] = (tournament.index // 16) + 1

    # pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    # tournament["Host"] = pod_order
    # conference_counts = tournament['Conference'].value_counts()
    # multibid = conference_counts[conference_counts > 1]
    # formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    # formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    # formatted_df = formatted_df.reset_index()
    # formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    # formatted_df = resolve_conflicts(formatted_df, modeling_stats)
    # formatted_df.set_index('Host')
    # formatted_df.index = formatted_df.index + 1

    aq_list = ["Binghamton", "East Carolina", "Stetson", "Rhode Island", "North Carolina", "Arizona",
            "Creighton", "USC Upstate", "Nebraska", "Cal Poly", "Northeastern", "Western Ky.", "Wright St.",
            "Columbia", "Fairfield", "Miami (OH)", "Murray St.", "Fresno St.",
            "Central Conn. St.", "Little Rock", "Holy Cross", "Vanderbilt", "Houston Christian",
            "ETSU", "Bethune-Cookman", "North Dakota St.", "Coastal Carolina", "Saint Mary's (CA)", "Utah Valley", "Oregon St."]
    host_seeds_list = ["Georgia", "Auburn", "Texas", "LSU", "North Carolina", "Clemson", "Coastal Carolina", "Oregon St.",
                "Oregon", "Arkansas", "Southern Miss.", "Tennessee", "UCLA", "Vanderbilt", "Ole Miss", "Florida St."]
    automatic_qualifiers = (
        modeling_stats[modeling_stats["Team"].isin(aq_list)]
        .sort_values("NET")
    )
    host_seeds = (
        modeling_stats[modeling_stats["Team"].isin(host_seeds_list)]
        .sort_values("NET")
    )
    amount_of_at_large = 64 - len(set(automatic_qualifiers["Team"]) - set(host_seeds["Team"])) - len(host_seeds)
    at_large = modeling_stats.drop(automatic_qualifiers.index)
    at_large = at_large[~at_large["Team"].isin(host_seeds_list)]
    automatic_qualifiers = automatic_qualifiers[~automatic_qualifiers["Team"].isin(host_seeds_list)]
    at_large = at_large.nsmallest(amount_of_at_large, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8 = modeling_stats.drop(automatic_qualifiers.index)
    next_8 = next_8[~next_8["Team"].isin(host_seeds_list)]
    next_8_teams = next_8.nsmallest(amount_of_at_large+8, "NET").iloc[amount_of_at_large:].reset_index(drop=True)
    remaining_teams = pd.concat([automatic_qualifiers, at_large]).sort_values("NET").reset_index(drop=True)

    seed_1_df = host_seeds.sort_values("NET").reset_index(drop=True)
    seed_1_df = seed_1_df.assign(Seed="1 Seed")
    seed_2_df = remaining_teams.iloc[0:16].sort_values("NET", ascending=False).copy().assign(Seed="2 Seed")
    seed_3_df = remaining_teams.iloc[16:32].copy().assign(Seed="3 Seed")
    seed_4_df = remaining_teams.iloc[32:48].sort_values("NET", ascending=False).copy().assign(Seed="4 Seed")
    formatted_df = pd.DataFrame(zip(seed_1_df['Team'], seed_1_df['Team'], seed_2_df['Team'], seed_3_df['Team'], seed_4_df['Team']), columns=['Host', '1 Seed', '2 Seed', '3 Seed', '4 Seed'])
    formatted_df = resolve_conflicts(formatted_df, modeling_stats)
    formatted_df.index = formatted_df.index + 1
    all_teams = pd.unique(formatted_df[["1 Seed", "2 Seed", "3 Seed", "4 Seed"]].values.ravel())
    bracket_teams = modeling_stats[modeling_stats["Team"].isin(all_teams)]
    conference_counts = bracket_teams["Conference"].value_counts()
    multibid = conference_counts[conference_counts > 1]
    st.markdown(f'<h2 id="tournament-outlook">Tournament Outlook</h2>', unsafe_allow_html=True)
    with st.container(border=True, height=440):
        st.dataframe(formatted_df[['Host', '2 Seed', '3 Seed', '4 Seed']], use_container_width=True)
    st.caption(f"Last 4 In - {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}")
    st.caption(f"First Four Out - {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}")
    st.caption(f"Next Four Out - {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}")
    st.caption(" | ".join([f"{conference}: {count}" for conference, count in multibid.items()]))
with col2:
    st.markdown(f'<h2 id="simulate-regional">Simulate Regional</h2>', unsafe_allow_html=True)
    with st.form(key='simulate_regional'):
        team_a = st.selectbox("1 Seed", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_b = st.selectbox("2 Seed", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_c = st.selectbox("3 Seed", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_d = st.selectbox("4 Seed", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        sim_region = st.form_submit_button("Simulate Regional")
        if sim_region:
            fig = simulate_regional(team_a, team_b, team_c, team_d, modeling_stats)
            st.pyplot(fig)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<h2 id="matchup-cards">Matchup Cards</h2>', unsafe_allow_html=True)
    with st.form(key='calculate_spread'):
        away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        neutrality = st.radio(
            "Game Location",
            ["On Campus", "Neutral"]
        )
        spread_button = st.form_submit_button("Calculate Spread")
        if spread_button:
            if neutrality == 'Neutral':
                location = 'Neutral'
            else:
                location = 'Home'
            fig = matchup_percentiles(away_team, home_team, modeling_stats, location)
            st.pyplot(fig)
with col2:
    st.markdown(f'<h2 id="team-schedule">Team Profile</h2>', unsafe_allow_html=True)
    with st.form(key='team_profile'):
        team_name = st.selectbox("Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_schedule = st.form_submit_button("Team Profile")
        if team_schedule:
            fig = team_visual(team_name, modeling_stats, schedule_df, comparison_date)
            rank, best, worst, schedule, completed, last_ten = grab_team_schedule(team_name, modeling_stats)
            st.pyplot(fig)
            st.write("Upcoming Games")
            if len(schedule) > 0:
                schedule.index = schedule.index + 1
                st.dataframe(schedule[['Opponent', 'NET', 'Quad', 'GQI', 'PEAR', 'Date']], use_container_width=True)
                st.caption('PEAR - Negative Value Indicates Favorites, Positive Value Indicates Underdog')

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<h2 id="conference-team-sheets">Conference Team Sheets</h2>', unsafe_allow_html=True)
    with st.form(key='conference_team_sheets'):
        conference = st.selectbox("Conference", ["Select Conference"] + list(sorted(modeling_stats['Conference'].unique())))
        conference_team = st.form_submit_button("Team Sheets")
        if conference_team:
            fig = conference_team_sheets(conference, modeling_stats)
            st.pyplot(fig)
with col2:
    # projected_wins, clean_completed, clean_remain = calculate_conference_results(schedule_df, comparison_date, modeling_stats, num_simulations=500)
    # projected_wins[["Conf_Wins", "Conf_Losses"]] = projected_wins[["Conf_Wins", "Conf_Losses"]].fillna(0).astype(int)
    # projected_wins["Proj_Conf_Wins"] = projected_wins["Conf_Wins"] + projected_wins["Remaining_Wins"]
    # projected_wins["Proj_Conf_Losses"] = projected_wins["Conf_Losses"] + projected_wins["Remaining_Losses"]
    # projected_wins["Projected_Conf_Record"] = projected_wins.apply(
    #     lambda x: f"{int(x['Proj_Conf_Wins'])}-{int(x['Proj_Conf_Losses'])}", axis=1
    # )
    # projected_wins["Current_Conf_Record"] = projected_wins.apply(
    #     lambda x: f"{int(x['Conf_Wins'])}-{int(x['Conf_Losses'])}", axis=1
    # )
    # projected_wins["Remaining_Conf_Record"] = projected_wins.apply(
    #     lambda x: f"{int(x['Remaining_Wins'])}-{int(x['Remaining_Losses'])}", axis=1
    # )
    # st.markdown(f'<h2 id="projected-conference-standings">Projected Conference Standings</h2>', unsafe_allow_html=True)
    # with st.form(key='projected_conference_tournament'):
    #     conference = st.selectbox("Conference", ["Select Conference"] + list(sorted(modeling_stats['Conference'].unique())))
    #     conference_standings = st.form_submit_button("Projected Standings")
    #     if conference_standings:
    #         fig = conference_projected_standing(conference, projected_wins)
    #         st.pyplot(fig)

    st.markdown(f'<h2 id="projected-conference-tournament">Projected Conference Tournament</h2>', unsafe_allow_html=True)
    with st.form(key='projected_conference_tournament'):
        conference = st.selectbox(
            "Conference",
            ["Select Conference"] + sorted([c for c in modeling_stats['Conference'].unique() if c != "Independent"])
        )
        conference_tournament = st.form_submit_button("Projected Tournament")
        if conference_tournament:
            fig = simulate_conference_tournaments(schedule_df, modeling_stats, 1000, conference)
            st.pyplot(fig)

st.divider()

import os
import glob
import pandas as pd
import re

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<h2 id="historical-team-performance">Historical Team Performance</h2>', unsafe_allow_html=True)
    with st.form(key='historical-team-performance'):
        team = st.selectbox("Team", ["Select Team"] + list(sorted(all_data['Team'].unique())))
        team_performance = st.form_submit_button("Historical Performance")
        if team_performance:
            fig = plot_rating_vs_net(team, all_data)
            st.pyplot(fig)
with col2:
    st.markdown(f'<h2 id="calculate-spread-two-teams-from-different-year">Calculate Spread Between Two Teams From Different Years</h2>', unsafe_allow_html=True)
    with st.form(key='calculate_spread_two_teams'):
        team_season_pairs = sorted(all_data_full[['Team', 'Season']].drop_duplicates().values.tolist())

        # Format as "Team (Season)" for display
        options = ["Select Team & Season"] + [f"{team} ({season})" for team, season in team_season_pairs]

        # Selectboxes
        away_selection = st.selectbox("Away Team & Season", options)
        home_selection = st.selectbox("Home Team & Season", options)

        # Parse selections
        if away_selection != "Select Team & Season":
            away_team, away_season = away_selection.rsplit(" (", 1)
            away_season = int(away_season[:-1])  # Remove trailing ')'

        if home_selection != "Select Team & Season":
            home_team, home_season = home_selection.rsplit(" (", 1)
            home_season = int(home_season[:-1])
        neutrality = st.radio(
            "Game Location",
            ["On Campus", "Neutral"]
        )
        two_spread_button = st.form_submit_button("Calculate")
        if two_spread_button:
            if neutrality == 'Neutral':
                location = 'Neutral'
            else:
                location = 'Home'
            fig = matchup_percentiles_with_year(away_team, home_team, away_season, home_season, all_data_full, neutrality)
            st.pyplot(fig)

st.divider()

if len(subset_games) > 0:
    comparison_date = comparison_date.strftime("%B %d, %Y")
    st.subheader(f"{comparison_date} Games")
    subset_games['Home'] = subset_games['home_team']
    subset_games['Away'] = subset_games['away_team']
    with st.container(border=True, height=440):
        st.dataframe(subset_games[['Home', 'Away', 'GQI', 'PEAR', 'Result']], use_container_width=True)

st.divider()

# supers_df = pd.read_csv("./PEAR/PEAR Baseball/y2025/tracking_supers_odds.csv")
# cols = st.columns(2)
# for idx, supers_idx in enumerate(sorted(supers_df['Regional Number'].unique())):
#     col = cols[idx % 2]
#     supers = supers_df[supers_df['Regional Number'] == supers_idx].reset_index(drop=True)
#     host = supers['Team'].iloc[0]
#     display_df = supers[['Team', 'Day 1', 'Day 2', 'Day 3']].copy()
#     for col_name in ['Day 1', 'Day 2', 'Day 3']:
#         display_df[col_name] = (display_df[col_name]).round(1).astype(str) + "%"
#     display_df.index = display_df.index + 1
#     with col:
#         st.markdown(f"### {host} Super Regional")
#         st.dataframe(display_df, use_container_width=True)



### CODE FOR DAY-BY-DAY REGIONAL PROJECTIONS
# final_df = pd.read_csv("PEAR/PEAR Baseball/y2025/tracking_regional_simulations.csv")
# cols = st.columns(2)
# for idx, seed in enumerate(sorted(final_df['Host Seed'].unique())):
#     col = cols[idx % 2]
#     regional_df = final_df[final_df['Host Seed'] == seed].reset_index(drop=True)
#     host_team = regional_df['Team'].iloc[0]
#     display_df = regional_df[['Team', 'Day 1', 'Day 2', 'Day 3', 'Day 4']].copy()
#     for col_name in ['Day 1', 'Day 2', 'Day 3', 'Day 4']:
#         display_df[col_name] = (display_df[col_name] * 100).round(1).astype(str) + "%"
#     display_df.index = display_df.index + 1
#     with col:
#         st.markdown(f"### {host_team} Regional")
#         st.dataframe(display_df, use_container_width=True)