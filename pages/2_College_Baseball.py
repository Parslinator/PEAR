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
        home_pr += 0.8
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 7)) * 100, 2)

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

# Find the latest date
if date_files:
    latest_date = max(date_files.keys())  # Get the most recent date
    latest_file = date_files[latest_date]

    # Read the selected CSV file
    file_path = os.path.join(folder_path, latest_file)
    modeling_stats = pd.read_csv(file_path)
else:
    modeling_stats = None  # No valid files found
formatted_latest_date = latest_date.strftime("%B %d, %Y")

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
    
def find_spread_matchup(home_team, away_team, modeling_stats, location = 'Neutral'):
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
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 7.5)) * 100, 2)

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
                home_pr += 0.8
            rating_diff = home_pr - away_pr
            return round(1 / (1 + 10 ** (-rating_diff / 7)) * 100, 2)
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
        win_prob = round(1 / (1 + 10 ** (-rating_diff / 7)) * 100, 2)
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
        elif (j == 0):  # Team names column
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

def simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics):
    teams = [team_a, team_b, team_c, team_d]
    r = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

    w1, l1 = (team_a, team_d) if random.random() < PEAR_Win_Prob(r[team_a], r[team_d]) / 100 else (team_d, team_a)
    w2, l2 = (team_b, team_c) if random.random() < PEAR_Win_Prob(r[team_b], r[team_c]) / 100 else (team_c, team_b)
    w3 = l2 if random.random() < PEAR_Win_Prob(r[l2], r[l1]) / 100 else l1
    w4, l4 = (w1, w2) if random.random() < PEAR_Win_Prob(r[w1], r[w2]) / 100 else (w2, w1)
    w5 = l4 if random.random() < PEAR_Win_Prob(r[l4], r[w3]) / 100 else w3
    game6_prob = PEAR_Win_Prob(r[w4], r[w5]) / 100
    w6 = w4 if random.random() < game6_prob else w5

    return w6 if w6 == w4 else (w4 if random.random() < game6_prob else w5)

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

def simulate_overall_tournament(bracket_one_probs, bracket_two_probs, stats_and_metrics, num_simulations=1000):
    final_results = defaultdict(int)

    bracket_one_teams = list(bracket_one_probs.keys())
    bracket_two_teams = list(bracket_two_probs.keys())

    for _ in range(num_simulations):
        # Randomly pick winner from bracket one and two using probabilities
        winner_one = random.choices(bracket_one_teams, weights=[bracket_one_probs[t] for t in bracket_one_teams])[0]
        winner_two = random.choices(bracket_two_teams, weights=[bracket_two_probs[t] for t in bracket_two_teams])[0]

        # Get ratings
        r1 = stats_and_metrics.loc[stats_and_metrics["Team"] == winner_one, "Rating"].iloc[0]
        r2 = stats_and_metrics.loc[stats_and_metrics["Team"] == winner_two, "Rating"].iloc[0]

        # Simulate the championship game
        prob = PEAR_Win_Prob(r1, r2) / 100
        champ = winner_one if random.random() < prob else winner_two

        final_results[champ] += 1

    formatted_results = defaultdict(float, {team: round(wins / num_simulations, 3) for team, wins in final_results.items()})
    return formatted_results

def simulate_playin_to_double_elim(teams, stats_and_metrics):
    # Map seeds to teams
    seeds = {i+1: teams[i] for i in range(6)}
    r = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

    # Play-in round
    gA_winner = seeds[3] if random.random() < PEAR_Win_Prob(r[seeds[3]], r[seeds[6]]) / 100 else seeds[6]
    gB_winner = seeds[4] if random.random() < PEAR_Win_Prob(r[seeds[4]], r[seeds[5]]) / 100 else seeds[5]

    # Determine who makes it to double elimination
    double_elim_teams = {seeds[1], seeds[2], gA_winner, gB_winner}

    # Reseed play-in winners
    playin_winners = [(s, t) for s, t in seeds.items() if t in [gA_winner, gB_winner]]
    sorted_winners = sorted(playin_winners, key=lambda x: x[0])
    lowest_seed_team = sorted_winners[0][1]
    higher_seed_team = sorted_winners[1][1]

    # Double elimination setup
    team_a = seeds[1]
    team_b = seeds[2]
    team_c = lowest_seed_team
    team_d = higher_seed_team

    # Simulate double elimination
    champion = simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics)

    return double_elim_teams, champion

def run_hybrid_tournament(teams, stats_and_metrics, num_simulations=1000):
    made_double_elim = defaultdict(int)
    tournament_wins = defaultdict(int)

    for _ in range(num_simulations):
        double_elim_teams, winner = simulate_playin_to_double_elim(teams, stats_and_metrics)
        for team in double_elim_teams:
            made_double_elim[team] += 1
        tournament_wins[winner] += 1

    data = []
    for team in teams:
        reach_double = 1.0 if team in teams[:2] else made_double_elim[team] / num_simulations
        win_tourney = tournament_wins[team] / num_simulations
        data.append({
            "Team": team,
            "Double Elim": round(reach_double * 100, 1),
            "Win Tournament": round(win_tourney * 100, 1)
        })

    return pd.DataFrame(data)

def simulate_8_team_double_elim(teams, stats_and_metrics):
    r = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

    # Round 1 matchups (seed-style: 1v8, 2v7, etc.)
    matchups = [(teams[0], teams[7]), (teams[3], teams[4]), (teams[2], teams[5]), (teams[1], teams[6])]

    # Round 1 (WB)
    wb_round1_winners = []
    lb_round1_losers = []
    for t1, t2 in matchups:
        winner = t1 if random.random() < PEAR_Win_Prob(r[t1], r[t2]) / 100 else t2
        loser = t2 if winner == t1 else t1
        wb_round1_winners.append(winner)
        lb_round1_losers.append(loser)

    # Round 2A (WB)
    wb_sf1 = wb_round1_winners[0] if random.random() < PEAR_Win_Prob(r[wb_round1_winners[0]], r[wb_round1_winners[1]]) / 100 else wb_round1_winners[1]
    wb_sf2 = wb_round1_winners[2] if random.random() < PEAR_Win_Prob(r[wb_round1_winners[2]], r[wb_round1_winners[3]]) / 100 else wb_round1_winners[3]
    wb_losers = [t for t in wb_round1_winners if t != wb_sf1 and t != wb_sf2]

    # LB Round 1 (elimination)
    lb_r1_1 = lb_round1_losers[0] if random.random() < PEAR_Win_Prob(r[lb_round1_losers[0]], r[lb_round1_losers[1]]) / 100 else lb_round1_losers[1]
    lb_r1_2 = lb_round1_losers[2] if random.random() < PEAR_Win_Prob(r[lb_round1_losers[2]], r[lb_round1_losers[3]]) / 100 else lb_round1_losers[3]

    # LB Round 2
    lb_r2_1 = lb_r1_1 if random.random() < PEAR_Win_Prob(r[lb_r1_1], r[wb_losers[0]]) / 100 else wb_losers[0]
    lb_r2_2 = lb_r1_2 if random.random() < PEAR_Win_Prob(r[lb_r1_2], r[wb_losers[1]]) / 100 else wb_losers[1]

    # LB Semifinal
    lb_sf = lb_r2_1 if random.random() < PEAR_Win_Prob(r[lb_r2_1], r[lb_r2_2]) / 100 else lb_r2_2

    # WB Final
    wb_final = wb_sf1 if random.random() < PEAR_Win_Prob(r[wb_sf1], r[wb_sf2]) / 100 else wb_sf2
    wb_final_loser = wb_sf2 if wb_final == wb_sf1 else wb_sf1

    # LB Final
    lb_final = lb_sf if random.random() < PEAR_Win_Prob(r[lb_sf], r[wb_final_loser]) / 100 else wb_final_loser

    # Championship
    champ = wb_final if random.random() < PEAR_Win_Prob(r[wb_final], r[lb_final]) / 100 else lb_final

    return champ

def run_8_team_double_elim(teams, stats_and_metrics, num_simulations=1000):
    results = defaultdict(int)

    for _ in range(num_simulations):
        winner = simulate_8_team_double_elim(teams, stats_and_metrics)
        results[winner] += 1

    df = pd.DataFrame([
        {"Team": team, "Win Tournament": round(results[team] / num_simulations * 100, 1)}
        for team in teams
    ])

    return df

def single_elimination_16_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Round 1", "Round 2", "Quarterfinals", "Semifinals", "Final", "Champion"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    
    # Get team ratings
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in seed_order
    }

    for _ in range(num_simulations):
        progress = {team: None for team in seed_order}

        # Round 1 (Play-in)
        play_in_pairs = [(8, 15), (9, 14), (10, 13), (11, 12)]
        round1_winners = []
        for a, b in play_in_pairs:
            team_a, team_b = seed_order[a], seed_order[b]
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            round1_winners.append(winner)
            progress[team_a] = "Round 1"
            progress[team_b] = "Round 1"
            progress[winner] = "Round 2"

        # Round 2
        r2_seeds = [4, 5, 6, 7]
        round2_winners = []
        for i, seed in enumerate(r2_seeds):
            team_a = seed_order[seed]
            team_b = round1_winners[i]
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            round2_winners.append(winner)
            progress[team_a] = "Round 2"
            progress[team_b] = "Round 2"
            progress[winner] = "Quarterfinals"

        # Quarterfinals
        r3_seeds = [0, 1, 2, 3]
        qf_winners = []
        for i, seed in enumerate(r3_seeds):
            team_a = seed_order[seed]
            team_b = round2_winners[i]
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            qf_winners.append(winner)
            progress[team_a] = "Quarterfinals"
            progress[team_b] = "Quarterfinals"
            progress[winner] = "Semifinals"

        # Semifinals
        sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
        sf_winners = []
        for team_a, team_b in sf_pairs:
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            sf_winners.append(winner)
            progress[team_a] = "Semifinals"
            progress[team_b] = "Semifinals"
            progress[winner] = "Final"

        # Final
        team_a, team_b = sf_winners
        prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
        winner = team_a if np.random.rand() < prob_a else team_b
        progress[team_a] = "Final"
        progress[team_b] = "Final"
        progress[winner] = "Champion"

        # Record outcomes
        for team, reached in progress.items():
            if reached:
                reached_idx = rounds.index(reached)
                for i in range(reached_idx + 1):
                    team_stats[team][rounds[i]] += 1

    # Convert counts to percentages
    result_df = pd.DataFrame.from_dict(team_stats, orient="index")
    result_df = result_df.applymap(lambda x: round(100 * x / num_simulations, 1))
    result_df = result_df.drop(columns=["Round 1"]).reset_index().rename(columns={"index": "Team"})
    
    return result_df

def single_elimination_14_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Round 1", "Quarterfinals", "Semifinals", "Final", "Champion"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    
    # Extract ratings
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in seed_order
    }

    for _ in range(num_simulations):
        progress = {team: None for team in seed_order}

        # Round 1: Seeds 5–12 play in
        play_in_pairs = [(4, 11), (5, 10), (6, 9), (7, 8)]
        round1_winners = []
        for a, b in play_in_pairs:
            team_a, team_b = seed_order[a], seed_order[b]
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            round1_winners.append(winner)
            progress[winner] = "Quarterfinals"

        # Quarterfinals: Seeds 1–4 vs Round 1 winners
        qf_winners = []
        for i, seed in enumerate([0, 1, 2, 3]):
            team_a = seed_order[seed]
            team_b = round1_winners[i]
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            qf_winners.append(winner)
            progress[team_a] = "Quarterfinals"
            progress[team_b] = "Quarterfinals"
            progress[winner] = "Semifinals"

        # Semifinals
        sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
        sf_winners = []
        for team_a, team_b in sf_pairs:
            prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
            winner = team_a if np.random.rand() < prob_a else team_b
            sf_winners.append(winner)
            progress[winner] = "Final"

        # Final
        team_a, team_b = sf_winners
        prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
        winner = team_a if np.random.rand() < prob_a else team_b
        progress[winner] = "Champion"

        # Record outcomes
        for team in seed_order:
            reached = progress.get(team)
            if reached is not None:
                reached_idx = rounds.index(reached)
                for i in range(reached_idx + 1):
                    team_stats[team][rounds[i]] += 1

    # Format result as DataFrame
    result_df = pd.DataFrame.from_dict(team_stats, orient="index")
    result_df = result_df.applymap(lambda x: round(100 * x / num_simulations, 1))
    result_df = result_df.drop(columns=["Round 1"]).reset_index().rename(columns={"index": "Team"})

    return result_df

def double_elimination_7_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Double Elim", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0] for team in seed_order}

    def game(team_a, team_b):
        prob = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
        return team_a if np.random.rand() < prob else team_b

    for _ in range(num_simulations):
        wins = defaultdict(int)
        losses = defaultdict(int)

        # Round 1: Seeds 2 vs 7, 3 vs 6, 4 vs 5
        g1 = game(seed_order[1], seed_order[6])
        g2 = game(seed_order[2], seed_order[5])
        g3 = game(seed_order[3], seed_order[4])
        winners_r1 = [g1, g2, g3]
        losers_r1 = [team for team in [seed_order[1], seed_order[6], seed_order[2], seed_order[5], seed_order[3], seed_order[4]] if team not in winners_r1]

        for t in winners_r1 + losers_r1 + [seed_order[0]]:
            team_stats[t]["Double Elim"] += 1

        # Round 2 winners bracket
        wb2_g1 = game(seed_order[0], g1)
        wb2_g2 = game(g2, g3)
        wb2_winners = [wb2_g1, wb2_g2]
        wb2_losers = [team for team in [seed_order[0], g1, g2, g3] if team not in wb2_winners]

        # Elimination games
        lb1 = game(losers_r1[0], losers_r1[1])
        lb2 = game(losers_r1[2], wb2_losers[0])
        lb3 = game(wb2_losers[1], lb1)

        # Loser's bracket final
        lb_final = game(lb2, lb3)

        # Winner's bracket final
        wb_final = game(wb2_winners[0], wb2_winners[1])

        # Championship
        if lb_final == wb_final:
            champ = wb_final
        else:
            champ_game = game(wb_final, lb_final)
            if champ_game == wb_final:
                champ = wb_final
            else:
                champ = game(wb_final, lb_final)

        team_stats[champ]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"}).drop(columns=["Double Elim"]).sort_values('Win Tournament', ascending=False).reset_index(drop=True)
    return df

def double_elimination_6_teams(seed_order, stats_and_metrics, num_simulations=1000):
    rounds = ["Double Elim", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0] for team in seed_order}

    def game(team_a, team_b):
        prob = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
        return team_a if np.random.rand() < prob else team_b

    for _ in range(num_simulations):
        wins = defaultdict(int)
        losses = defaultdict(int)

        # Round 1: #3 vs #6, #4 vs #5
        g1 = game(seed_order[2], seed_order[5])
        g2 = game(seed_order[3], seed_order[4])
        winners_r1 = [g1, g2]
        losers_r1 = [team for team in [seed_order[2], seed_order[5], seed_order[3], seed_order[4]] if team not in winners_r1]

        for t in winners_r1 + losers_r1 + [seed_order[0], seed_order[1]]:
            team_stats[t]["Double Elim"] += 1

        # Round 2 winners bracket: #1 vs g1, #2 vs g2
        wb2_g1 = game(seed_order[0], g1)
        wb2_g2 = game(seed_order[1], g2)
        wb2_winners = [wb2_g1, wb2_g2]
        wb2_losers = [team for team in [seed_order[0], g1, seed_order[1], g2] if team not in wb2_winners]

        # Elimination games
        lb1 = game(losers_r1[0], losers_r1[1])
        lb2 = game(wb2_losers[0], lb1)
        lb3 = game(wb2_losers[1], lb2)

        # Loser's bracket final
        lb_final = lb3

        # Winner's bracket final
        wb_final = game(wb2_winners[0], wb2_winners[1])

        # Championship
        if lb_final == wb_final:
            champ = wb_final
        else:
            champ_game = game(wb_final, lb_final)
            if champ_game == wb_final:
                champ = wb_final
            else:
                champ = game(wb_final, lb_final)

        team_stats[champ]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"}).drop(columns=['Double Elim']).sort_values('Win Tournament', ascending=False).reset_index(drop=True)
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
    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0] for team in seed_order}

    def game(team_a, team_b):
        prob = PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100
        return team_a if np.random.rand() < prob else team_b

    for _ in range(num_simulations):
        pool_winners = {}

        # Simulate pool play
        for pool_name, teams in pools.items():
            wins = {team: 0 for team in teams}
            matchups = [(teams[0], teams[1]), (teams[0], teams[2]), (teams[1], teams[2])]
            for team_a, team_b in matchups:
                winner = game(team_a, team_b)
                wins[winner] += 1

            max_wins = max(wins.values())
            contenders = [team for team, w in wins.items() if w == max_wins]
            if len(contenders) == 1:
                pool_winner = contenders[0]
            else:
                pool_winner = min(contenders, key=lambda team: seed_order.index(team))  # higher seed wins tiebreaker
            pool_winners[pool_name] = pool_winner
            team_stats[pool_winner]["Win Pool"] += 1

        # Semifinals: A vs D, B vs C
        sf_matchups = [(pool_winners["A"], pool_winners["D"]), (pool_winners["B"], pool_winners["C"])]
        finalists = []
        for team_a, team_b in sf_matchups:
            winner = game(team_a, team_b)
            finalists.append(winner)
            team_stats[winner]["Make Final"] += 1

        # Final
        final_winner = game(finalists[0], finalists[1])
        team_stats[final_winner]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(team_stats, orient="index")
    df = df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"})
    return df

def simulate_playin_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    team_stats = {team: {"Double Elim": 0, "Win Tournament": 0} for team in seed_order}
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in seed_order
    }

    def win_prob(team_a, team_b):
        return PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100

    for _ in range(num_simulations):
        # Play-in between #4 and #5
        team_4, team_5 = seed_order[3], seed_order[4]
        prob_4 = win_prob(team_4, team_5)
        playin_winner = team_4 if np.random.rand() < prob_4 else team_5
        team_stats[playin_winner]["Double Elim"] += 1

        # Set up 4-team double elimination bracket
        teams = [seed_order[0], seed_order[1], seed_order[2], playin_winner]
        r = ratings

        w1, l1 = (teams[0], teams[3]) if np.random.rand() < win_prob(teams[0], teams[3]) else (teams[3], teams[0])
        w2, l2 = (teams[1], teams[2]) if np.random.rand() < win_prob(teams[1], teams[2]) else (teams[2], teams[1])
        w3 = l2 if np.random.rand() < win_prob(l2, l1) else l1
        w4, l4 = (w1, w2) if np.random.rand() < win_prob(w1, w2) else (w2, w1)
        w5 = l4 if np.random.rand() < win_prob(l4, w3) else w3
        game6_prob = win_prob(w4, w5)
        w6 = w4 if np.random.rand() < game6_prob else w5

        champion = w6 if w6 == w4 else (w4 if np.random.rand() < game6_prob else w5)
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
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in seed_order
    }

    def win_prob(team_a, team_b):
        return PEAR_Win_Prob(ratings[team_a], ratings[team_b]) / 100

    for _ in range(num_simulations):
        # Play-in games
        team_5, team_8 = seed_order[4], seed_order[7]
        team_6, team_7 = seed_order[5], seed_order[6]

        winner_5_8 = team_5 if np.random.rand() < win_prob(team_5, team_8) else team_8
        winner_6_7 = team_6 if np.random.rand() < win_prob(team_6, team_7) else team_7

        advancing_teams = seed_order[:4] + [winner_5_8, winner_6_7]
        for team in advancing_teams:
            team_stats[team]["Double Elim"] += 1

        # Now simulate the 6-team double elimination bracket
        # Seeds 1 and 2 get byes
        t1, t2, t3, t4, t5, t6 = advancing_teams

        # Game 1: 3 vs 6
        w1, l1 = (t3, t6) if np.random.rand() < win_prob(t3, t6) else (t6, t3)
        # Game 2: 4 vs 5
        w2, l2 = (t4, t5) if np.random.rand() < win_prob(t4, t5) else (t5, t4)

        # Game 3: 1 vs W2
        w3, l3 = (t1, w2) if np.random.rand() < win_prob(t1, w2) else (w2, t1)
        # Game 4: 2 vs W1
        w4, l4 = (t2, w1) if np.random.rand() < win_prob(t2, w1) else (w1, t2)

        # Elimination games
        # Game 5: L1 vs L2
        w5 = l1 if np.random.rand() < win_prob(l1, l2) else l2
        # Game 6: L3 vs W5
        w6 = l3 if np.random.rand() < win_prob(l3, w5) else w5
        # Game 7: L4 vs W6
        w7 = l4 if np.random.rand() < win_prob(l4, w6) else w6

        # Winners bracket final
        w8, l8 = (w3, w4) if np.random.rand() < win_prob(w3, w4) else (w4, w3)

        # Losers bracket final
        w9 = l8 if np.random.rand() < win_prob(l8, w7) else w7

        # Championship
        champ = w8 if np.random.rand() < win_prob(w8, w9) else w9
        if champ != w8:
            champ = w8 if np.random.rand() < win_prob(w8, w9) else w9

        team_stats[champ]["Win Tournament"] += 1

    # Final formatting
    df = pd.DataFrame.from_dict(team_stats, orient="index").reset_index().rename(columns={"index": "Team"})
    df["Double Elim"] = df["Double Elim"].apply(lambda x: round(100 * x / num_simulations, 1))
    df["Win Tournament"] = df["Win Tournament"].apply(lambda x: round(100 * x / num_simulations, 1))

    return df

def simulate_mvc_tournament(seed_order, stats_and_metrics, num_simulations=1000):
    assert len(seed_order) == 8, "Seed order must have exactly 8 teams."

    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return 1 / (1 + 10 ** (-rating_diff / 7))

    ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
               for team in seed_order}
    
    make_de_stats = defaultdict(int)
    win_stats = defaultdict(int)

    for _ in range(num_simulations):
        # Play-in games
        team5, team6, team7, team8 = seed_order[4], seed_order[5], seed_order[6], seed_order[7]

        # Game 1: 5 vs 8
        prob_5v8 = PEAR_Win_Prob(ratings[team5], ratings[team8])
        winner_5v8 = team5 if np.random.rand() < prob_5v8 else team8

        # Game 2: 6 vs 7
        prob_6v7 = PEAR_Win_Prob(ratings[team6], ratings[team7])
        winner_6v7 = team6 if np.random.rand() < prob_6v7 else team7

        playin_winners = [winner_5v8, winner_6v7]
        for team in playin_winners:
            make_de_stats[team] += 1

        for team in seed_order[:4]:
            make_de_stats[team] += 1  # Top 4 seeds always make DE

        # Begin double elimination with 6 teams:
        # Seeds 1-4 and the two play-in winners
        de_teams = seed_order[:4] + playin_winners
        r = {team: ratings[team] for team in de_teams}

        # Round 1: 3 vs playin_winner1, 4 vs playin_winner2
        w1 = de_teams[2] if np.random.rand() < PEAR_Win_Prob(r[de_teams[2]], r[de_teams[4]]) else de_teams[4]
        l1 = de_teams[4] if w1 == de_teams[2] else de_teams[2]
        w2 = de_teams[3] if np.random.rand() < PEAR_Win_Prob(r[de_teams[3]], r[de_teams[5]]) else de_teams[5]
        l2 = de_teams[5] if w2 == de_teams[3] else de_teams[3]

        # Round 2 Winners: 1 vs w1, 2 vs w2
        w3 = de_teams[0] if np.random.rand() < PEAR_Win_Prob(r[de_teams[0]], r[w1]) else w1
        l3 = w1 if w3 == de_teams[0] else de_teams[0]
        w4 = de_teams[1] if np.random.rand() < PEAR_Win_Prob(r[de_teams[1]], r[w2]) else w2
        l4 = w2 if w4 == de_teams[1] else de_teams[1]

        # Losers Bracket
        w5 = l2 if np.random.rand() < PEAR_Win_Prob(r[l2], r[l3]) else l3
        w6 = l1 if np.random.rand() < PEAR_Win_Prob(r[l1], r[l4]) else l4

        # Elimination matchups
        w7 = w5 if np.random.rand() < PEAR_Win_Prob(r[w5], r[w6]) else w6

        # Semifinal
        w8 = w3 if np.random.rand() < PEAR_Win_Prob(r[w3], r[w4]) else w4
        l8 = w4 if w8 == w3 else w3

        w9 = w7 if np.random.rand() < PEAR_Win_Prob(r[w7], r[l8]) else l8

        # Final(s)
        prob_final = PEAR_Win_Prob(r[w8], r[w9])
        champion = w8 if np.random.rand() < prob_final else w9
        if champion != w8:
            # If first loss for w8, play again
            prob_final2 = PEAR_Win_Prob(r[champion], r[w8])
            champion = champion if np.random.rand() < prob_final2 else w8

        win_stats[champion] += 1

    result = pd.DataFrame({
        "Team": seed_order,
        "Double Elim": [round(100 * make_de_stats[t] / num_simulations, 1) for t in seed_order],
        "Win Tournament": [round(100 * win_stats[t] / num_simulations, 1) for t in seed_order]
    })

    return result

def simulate_two_playin_rounds_to_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    teams = seed_order
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in teams
    }
    
    tracker = {
        team: {"Round 2": 0, "Make Double Elim": 0, "Win Tournament": 0}
        for team in teams
    }

    for _ in range(num_simulations):
        # Round 1 Play-in: #5 vs #8 and #6 vs #7
        play_in1_a, play_in1_b = seed_order[4], seed_order[7]
        prob_a = PEAR_Win_Prob(ratings[play_in1_a], ratings[play_in1_b]) / 100
        win_5v8 = play_in1_a if np.random.rand() < prob_a else play_in1_b

        play_in2_a, play_in2_b = seed_order[5], seed_order[6]
        prob_b = PEAR_Win_Prob(ratings[play_in2_a], ratings[play_in2_b]) / 100
        win_6v7 = play_in2_a if np.random.rand() < prob_b else play_in2_b

        tracker[win_5v8]["Round 2"] += 1
        tracker[win_6v7]["Round 2"] += 1

        # Round 2: #4 vs winner of 5/8, #3 vs winner of 6/7
        r2_a, r2_b = seed_order[3], win_5v8
        prob_r2a = PEAR_Win_Prob(ratings[r2_a], ratings[r2_b]) / 100
        win_4 = r2_a if np.random.rand() < prob_r2a else r2_b

        r2_c, r2_d = seed_order[2], win_6v7
        prob_r2c = PEAR_Win_Prob(ratings[r2_c], ratings[r2_d]) / 100
        win_3 = r2_c if np.random.rand() < prob_r2c else r2_d

        tracker[win_4]["Make Double Elim"] += 1
        tracker[win_3]["Make Double Elim"] += 1

        # Double elimination teams: [#1, #2, win_4, win_3]
        de_teams = [seed_order[0], seed_order[1], win_4, win_3]
        for team in [seed_order[0], seed_order[1]]:
            tracker[team]["Make Double Elim"] += 1

        # Simulate double elimination bracket
        def simulate_de(teams, ratings):
            t = teams.copy()
            r = ratings
            w1 = t[0] if np.random.rand() < PEAR_Win_Prob(r[t[0]], r[t[3]]) / 100 else t[3]
            w2 = t[1] if np.random.rand() < PEAR_Win_Prob(r[t[1]], r[t[2]]) / 100 else t[2]
            l1 = t[3] if w1 == t[0] else t[0]
            l2 = t[2] if w2 == t[1] else t[1]
            w3 = l1 if np.random.rand() < PEAR_Win_Prob(r[l1], r[l2]) / 100 else l2
            w4 = w1 if np.random.rand() < PEAR_Win_Prob(r[w1], r[w2]) / 100 else w2
            l3 = w2 if w4 == w1 else w1
            w5 = w3 if np.random.rand() < PEAR_Win_Prob(r[w3], r[l3]) / 100 else l3
            prob_f = PEAR_Win_Prob(r[w4], r[w5]) / 100
            w6 = w4 if np.random.rand() < prob_f else w5
            return w6 if w6 == w4 else (w4 if np.random.rand() < prob_f else w5)
        
        winner = simulate_de(de_teams, ratings)
        tracker[winner]["Win Tournament"] += 1

    df = pd.DataFrame.from_dict(tracker, orient="index").reset_index().rename(columns={"index": "Team"})
    df["Round 2"] = df["Round 2"].astype(float) * 100 / num_simulations
    df["Make Double Elim"] = df["Make Double Elim"].astype(float) * 100 / num_simulations
    df["Win Tournament"] = df["Win Tournament"].astype(float) * 100 / num_simulations

    for team in [seed_order[0], seed_order[1], seed_order[2], seed_order[3]]:
        df.loc[df["Team"] == team, "Round 2"] = 100.0
    for team in [seed_order[0], seed_order[1]]:
        df.loc[df["Team"] == team, "Make Double Elim"] = 100.0

    return df

def simulate_best_of_three_series(team_a, team_b, ratings):
    """Simulate a best-of-three series with team_a as the home team."""
    wins_a, wins_b = 0, 0
    while wins_a < 2 and wins_b < 2:
        prob_a = PEAR_Win_Prob(ratings[team_a], ratings[team_b], location="Home") / 100
        if random.random() < prob_a:
            wins_a += 1
        else:
            wins_b += 1
    return team_a if wins_a == 2 else team_b

def simulate_best_of_three_tournament(seed_order, stats_and_metrics, num_simulations=1000):
    assert len(seed_order) == 4, "This format requires exactly 4 teams"

    rounds = ["Semifinal", "Make Final", "Win Tournament"]
    team_stats = {team: {r: 0 for r in rounds} for team in seed_order}
    ratings = {
        team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]
        for team in seed_order
    }

    for _ in range(num_simulations):
        progress = {}

        # Semifinal 1: #1 (home) vs #4
        semi1_winner = simulate_best_of_three_series(seed_order[0], seed_order[3], ratings)
        progress[seed_order[0]] = "Semifinal"
        progress[seed_order[3]] = "Semifinal"
        progress[semi1_winner] = "Make Final"

        # Semifinal 2: #2 (home) vs #3
        semi2_winner = simulate_best_of_three_series(seed_order[1], seed_order[2], ratings)
        progress[seed_order[1]] = "Semifinal"
        progress[seed_order[2]] = "Semifinal"
        progress[semi2_winner] = "Make Final"

        # Final: Higher seed is home
        finalist_1_seed = seed_order.index(semi1_winner)
        finalist_2_seed = seed_order.index(semi2_winner)
        if finalist_1_seed < finalist_2_seed:
            final_winner = simulate_best_of_three_series(semi1_winner, semi2_winner, ratings)
        else:
            final_winner = simulate_best_of_three_series(semi2_winner, semi1_winner, ratings)

        progress[final_winner] = "Win Tournament"

        # Record results
        for team, round_reached in progress.items():
            reached_idx = rounds.index(round_reached)
            for i in range(reached_idx + 1):
                team_stats[team][rounds[i]] += 1

    result_df = pd.DataFrame.from_dict(team_stats, orient="index")
    result_df = result_df.applymap(lambda x: round(100 * x / num_simulations, 1)).reset_index().rename(columns={"index": "Team"})
    result_df = result_df.drop(columns=['Semifinal'])
    return result_df

def simulate_two_playin_to_two_double_elim(seed_order, stats_and_metrics, num_simulations=1000):
    def get_rating(team):
        return stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].values[0]

    def PEAR_Win_Prob(home_rating, away_rating, location="Neutral"):
        if location == "Home":
            home_rating += 0.8
        rating_diff = home_rating - away_rating
        return 1 / (1 + 10 ** (-rating_diff / 7))

    results = {team: {"Double Elim": 0, "Make Finals": 0, "Win Tournament": 0} for team in seed_order}

    for _ in range(num_simulations):
        ratings = {team: get_rating(team) for team in seed_order}

        # Play-in games
        playin_7_10 = seed_order[6], seed_order[9]
        playin_8_9 = seed_order[7], seed_order[8]

        winner_7_10 = playin_7_10[0] if np.random.rand() < PEAR_Win_Prob(ratings[playin_7_10[0]], ratings[playin_7_10[1]]) else playin_7_10[1]
        winner_8_9 = playin_8_9[0] if np.random.rand() < PEAR_Win_Prob(ratings[playin_8_9[0]], ratings[playin_8_9[1]]) else playin_8_9[1]

        # Determine which play-in winner is higher seed
        playin_winners = sorted([winner_7_10, winner_8_9], key=lambda x: seed_order.index(x))
        high_seed, low_seed = playin_winners

        # Assign to brackets
        bracket1_teams = [seed_order[0], seed_order[3], seed_order[4], high_seed]
        bracket2_teams = [seed_order[1], seed_order[2], seed_order[5], low_seed]

        for t in bracket1_teams:
            results[t]["Double Elim"] += 1
        for t in bracket2_teams:
            results[t]["Double Elim"] += 1

        def simulate_double_elim(t1, t2, t3, t4):
            teams = [t1, t2, t3, t4]
            r = {team: ratings[team] for team in teams}

            # Game 1: t1 vs t4
            w1, l1 = (t1, t4) if np.random.rand() < PEAR_Win_Prob(r[t1], r[t4]) else (t4, t1)
            # Game 2: t2 vs t3
            w2, l2 = (t2, t3) if np.random.rand() < PEAR_Win_Prob(r[t2], r[t3]) else (t3, t2)
            # Loser bracket: l1 vs l2
            w3 = l1 if np.random.rand() < PEAR_Win_Prob(r[l1], r[l2]) else l2
            # Winner's bracket: w1 vs w2
            w4, l4 = (w1, w2) if np.random.rand() < PEAR_Win_Prob(r[w1], r[w2]) else (w2, w1)
            # Loser bracket: w3 vs l4
            w5 = w3 if np.random.rand() < PEAR_Win_Prob(r[w3], r[l4]) else l4
            # Final: w4 vs w5
            final_game_prob = PEAR_Win_Prob(r[w4], r[w5])
            winner = w4 if np.random.rand() < final_game_prob else w5
            if winner != w4:
                winner = w4 if np.random.rand() < final_game_prob else w5
            return winner

        finalist_1 = simulate_double_elim(*bracket1_teams)
        finalist_2 = simulate_double_elim(*bracket2_teams)

        results[finalist_1]["Make Finals"] += 1
        results[finalist_2]["Make Finals"] += 1

        # Championship Game (1-game final)
        prob_final = PEAR_Win_Prob(ratings[finalist_1], ratings[finalist_2])
        champ = finalist_1 if np.random.rand() < prob_final else finalist_2
        results[champ]["Win Tournament"] += 1

    # Final formatting
    final_df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Team"})
    for col in ["Double Elim", "Make Finals", "Win Tournament"]:
        final_df[col] = final_df[col].apply(lambda x: round(100 * x / num_simulations, 1))

    return final_df

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
        final_df = single_elimination_16_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.3, conference, 0.106, 0.098, 0.1)
    elif conference == "Big 12":
        seed_order = [team for team, _ in team_win_pcts[:12]]
        result_df = single_elimination_14_teams(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(result_df, 0.6, conference, 0.066, 0.06, 0.08)
    elif conference in ["Conference USA", "American Athletic", "Southland", "SWAC"]:
        seed_order = [team for team, _ in team_win_pcts[:8]]
        output = run_simulation(seed_order[0], seed_order[3], seed_order[4], seed_order[7], stats_and_metrics)
        bracket_one = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
        output = run_simulation(seed_order[1], seed_order[2], seed_order[5], seed_order[6], stats_and_metrics)
        bracket_two = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
        championship_results = simulate_overall_tournament(
            bracket_one.set_index("Team")["Win Regional"].to_dict(),
            bracket_two.set_index("Team")["Win Regional"].to_dict(),
            stats_and_metrics,
            num_simulations=1000
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
        final_df = run_hybrid_tournament(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.134, 0.122, 0.3)
    elif conference == 'ASUN':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        final_df = run_8_team_double_elim(seed_order, stats_and_metrics)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.115, 0.105, 0.2)
    elif conference == "Atlantic 10":
        seed_order = [team for team, _ in team_win_pcts[:7]]
        final_df = double_elimination_7_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.105, 0.093, 0.2)
    elif conference in ['Big East', 'Ivy League', 'Northeast', 'The Summit League']:
        seed_order = [team for team, _ in team_win_pcts[:4]]
        output = run_simulation(seed_order[0], seed_order[1], seed_order[2], seed_order[3], stats_and_metrics)
        final_df = pd.DataFrame(list(output.items()), columns=["Team", "Win Tournament"])
        final_df["Win Tournament"] = final_df["Win Tournament"] * 100
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.143, 0.12, 0.4)
    elif conference in ['Big South', 'Coastal Athletic', 'Horizon League', 'Mid-American']:
        seed_order = [team for team, _ in team_win_pcts[:6]]
        final_df = double_elimination_6_teams(seed_order, stats_and_metrics, num_simulations=1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.117, 0.103, 0.25)
    elif conference == 'Big Ten':
        seed_order = [team for team, _ in team_win_pcts[:12]]
        final_df = simulate_pool_play_tournament(seed_order, stats_and_metrics, num_simulations=500)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.082, 0.075, 0.1)
    elif conference == 'Big West':
        seed_order = [team for team, _ in team_win_pcts[:5]]
        final_df = simulate_playin_double_elim(seed_order, stats_and_metrics)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.16, 0.14, 0.4)
    elif conference in ['MAAC', 'Southern', 'Western Athletic']:
        seed_order = [team for team, _ in team_win_pcts[:8]]
        final_df = simulate_playins_to_6team_double_elim(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Missouri Valley':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        final_df = simulate_mvc_tournament(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Ohio Valley':
        seed_order = [team for team, _ in team_win_pcts[:8]]
        final_df = simulate_two_playin_rounds_to_double_elim(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.113, 0.103, 0.2)
    elif conference == 'Patriot League':
        seed_order = [team for team, _ in team_win_pcts[:4]]
        final_df = simulate_best_of_three_tournament(seed_order, stats_and_metrics, 1000)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.17, 0.15, 0.5)
    elif conference == 'Sun Belt':
        seed_order = [team for team, _ in team_win_pcts[:10]]
        final_df = simulate_two_playin_to_two_double_elim(seed_order, stats_and_metrics, 500)
        fig = plot_tournament_odds_table(final_df, 0.5, conference, 0.104, 0.095, 0.15)
    return fig

import matplotlib.pyplot as plt # type: ignore
def create_quadrant_table(completed):
    # Clean up the 'Result' column
    def clean_result(result):
        return re.sub(r"\s\(\d+\sInnings\)", "", result)
    
    completed['Result'] = completed['Result'].apply(clean_result)

    # Count the maximum number of games in any quadrant
    quadrant_counts = completed['Quad'].value_counts()
    max_games = quadrant_counts.max()

    # Define columns for quadrants
    columns = ["Q1", "Q2", "Q3", "Q4"]
    
    # Create an empty array to store game data
    table_data = np.full((max_games, 4), '', dtype=object)
    
    # Fill table data based on 'Quadrant'
    for idx, row in completed.iterrows():
        quadrant_idx = columns.index(row["Quad"])
        if pd.notna(row['NET']):  # Check if 'Rating' exists (not NaN)
            game_info = f"{int(row['NET'])} | {row['Opponent']} | {row['Result']}"
        else:
            game_info = f"N/A | {row['Opponent']} | {row['Result']}"
        
        # Add the game info to the first available row for the respective quadrant
        for game_row in range(max_games):
            if table_data[game_row, quadrant_idx] == '':
                table_data[game_row, quadrant_idx] = game_info
                break

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Set background color of the figure
    fig.patch.set_facecolor('#CECEB2')
    
    # Hide axes for the table display
    ax.axis('off')
    
    # Add the table to the plot
    table = ax.table(cellText=table_data, colLabels=columns, loc='center')
    
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(4, 4)
    
    # Set background color for each cell
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor('#CECEB2')  # Set the background color for each cell
    
    # Style column headers and rows
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Column header row
            cell.set_text_props(weight='bold')
        else:  # Rows below the header
            cell.set_text_props(ha='left', weight='bold')  # Left align the rows
            game_result = table_data[i-1, j]
            
            # Split the game result by "|"
            result_parts = game_result.split(" | ")
            
            # Check the last element for "W" or "L"
            color = '#000000'  # Default color (black)
            if len(result_parts) > 2:  # Make sure there are enough parts (Rating | Opponent | Result)
                result = result_parts[2]  # Last element (Result)
                if "W" in result:  # If "W" is in the result, use green
                    color = '#1D4D00'
                elif "L" in result:  # If "L" is in the result, use red
                    color = '#660000'
            
            # Apply the color to the cell text
            cell.set_text_props(color=color)
    
    # Return the figure object
    return fig

def team_net_tracker(team):
    X = 14
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
    # most_recent_csv = stats_and_metrics_combined[stats_and_metrics_combined['Date'] == stats_and_metrics_combined['Date'].max()]
    filtered_data = stats_and_metrics_combined[stats_and_metrics_combined['Team'] == team]
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date']).dt.strftime('%m/%d')
    best_net = filtered_data['NET'].min()
    worst_net = filtered_data['NET'].max()
    earliest = filtered_data['Date'].min()
    latest = filtered_data['Date'].max()
    pivoted_table = filtered_data.pivot_table(index='Team', columns='Date', values='NET', aggfunc='first')
    pivoted_table = pivoted_table[sorted(pivoted_table.columns)]
    pivoted_table = pivoted_table.reset_index()
    pivoted_table = pivoted_table.sort_values(by=pivoted_table.columns[-1], ascending=True)
    import math
    num_rows = 1
    num_cols = 1
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(8, 8))  # Single axis instead of array
    fig.patch.set_facecolor('#CECEB2')

    # Title and subtitles
    plt.text(0, 1.09, f"{team} NET Ranking", fontsize=24, fontweight='bold', ha='left', transform=ax.transAxes)
    plt.text(0, 1.05, f"Past {X} Days - {earliest} to {latest}", fontsize=14, ha='left', transform=ax.transAxes)
    plt.text(0, 1.01, f"@PEARatings", fontsize=14, fontweight='bold', ha='left', transform=ax.transAxes)

    min_net = -3
    max_net = stats_and_metrics_combined['NET'].max() + 10

    team = pivoted_table['Team'].iloc[0]  # Take the first team
    team_data = pivoted_table[pivoted_table['Team'] == team].drop(columns='Team').T  # Drop 'Team' column and transpose
    team_data.columns = [team]  # Set the team name as the column header
    last_value = team_data.iloc[0, 0]
    first_value = team_data.iloc[-1, 0]

    # Determine the color of the line based on trend
    line_color = "#2ca02c" if last_value > first_value else "#d62728"
    best_net_date = filtered_data.loc[filtered_data['NET'] == best_net, 'Date'].values[0]
    worst_net_date = filtered_data.loc[filtered_data['NET'] == worst_net, 'Date'].values[0]
    best_net_index = list(team_data.index).index(best_net_date)
    worst_net_index = list(team_data.index).index(worst_net_date)

    # Annotate best_net
    ax.text(best_net_index, best_net + 15, f"{int(best_net)}", 
            fontsize=14, fontweight='bold', ha='center', color='#2ca02c', 
            bbox=dict(facecolor='#CECEB2', edgecolor='#2ca02c', boxstyle='round,pad=0.3'))

    # Annotate worst_net
    ax.text(worst_net_index, worst_net + 15, f"{int(worst_net)}", 
            fontsize=14, fontweight='bold', ha='center', color='#d62728', 
            bbox=dict(facecolor='#CECEB2', edgecolor='#d62728', boxstyle='round,pad=0.3'))

    # Annotate first_value if it is different from both best_net and worst_net
    if first_value not in [best_net, worst_net]:
        ax.text(team_data.index[-1], first_value + 15, f"{int(first_value)}", 
                fontsize=16, fontweight='bold', ha='center', color='black')

    # Annotate last_value if it is different from both best_net and worst_net
    if last_value not in [best_net, worst_net]:
        ax.text(team_data.index[0], last_value + 15, f"{int(last_value)}", 
                fontsize=16, fontweight='bold', ha='center', color='black')

    # Plotting
    ax.plot(team_data.index, team_data[team], marker='o', color=line_color, linewidth=5, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title(f"#{int(first_value)} {team}", fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    ax.set_facecolor('#CECEB2')
    ax.set_xticks([])
    ax.set_ylim(min_net, max_net)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()

    plt.tight_layout()
    return fig

import math
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

def team_schedule_quality(team, schedule_df, stats_and_metrics):
    team_schedule = schedule_df[schedule_df['Team'] == team].reset_index(drop=True)
    SOS = stats_and_metrics[stats_and_metrics['Team'] == team]['SOS'].values[0]
    RQI = stats_and_metrics[stats_and_metrics['Team'] == team]['RQI'].values[0]
    Q1 = stats_and_metrics[stats_and_metrics['Team'] == team]['Q1'].values[0]
    Q2 = stats_and_metrics[stats_and_metrics['Team'] == team]['Q2'].values[0]
    Q3 = stats_and_metrics[stats_and_metrics['Team'] == team]['Q3'].values[0]
    Q4 = stats_and_metrics[stats_and_metrics['Team'] == team]['Q4'].values[0]
    NET = stats_and_metrics[stats_and_metrics['Team'] == team]['NET'].values[0]
    RPI = stats_and_metrics[stats_and_metrics['Team'] == team]['RPI'].values[0]
    ELO = stats_and_metrics[stats_and_metrics['Team'] == team]['ELO_Rank'].values[0]
    PRR = stats_and_metrics[stats_and_metrics['Team'] == team]['PRR'].values[0]
    Record = stats_and_metrics[stats_and_metrics['Team'] == team]['Record'].values[0]
    Conf_Record = get_conference_record(team, team_schedule, stats_and_metrics)
    record = get_location_records(team, team_schedule)
    home_record = record['Home']
    away_record = record['Away']
    neutral_record = record['Neutral']

    def get_opponent_net(row, team):
        if row['home_team'] == team:
            return row['away_net']
        elif row['away_team'] == team:
            return row['home_net']
        else:
            return np.nan

    team_schedule['opponent_net'] = team_schedule.apply(lambda row: get_opponent_net(row, team), axis=1)

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
    rows = 15
    cols = math.ceil(num_items / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows*1.3))  # Adjust size as needed
    fig.patch.set_facecolor('#CECEB2')

    # Ensure axes is always 2D
    if cols == 1:
        axes = axes[:, None]
    elif rows == 1:
        axes = axes[None, :]
    # Fill subplots column-first
    for idx, (_, row) in enumerate(team_schedule.iterrows()):
        r = idx % rows
        c = idx // rows
        ax = axes[r, c]
        if row['home_team'] == team:
            opponent = row['away_team']
            net = row['away_net']
            symbol = ""
        else:
            opponent = row['home_team']
            net = row['home_net']
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
        # ax.text(0.5, 0.8, opponent, ha='center', va='center', fontsize=40, fontweight='bold', color=color)
        # ax.text(0.1, 0.3, f'#{net}', ha='left', va='center', fontsize=32)
        # ax.text(0.5, 0.5, row['Quad'], ha='right', va='center', fontsize=32, fontweight='bold')
        result_first_letter = row['Result'][0].upper() if row['Result'][0].upper() in ['W', 'L'] else ''

        if result_first_letter:
            if (row['home_team'] == team) & (row['Location'] == "Home"):
                ax.text(0.5, 0.8, f'{opponent}', ha='center', va='center', fontsize=35, fontweight='bold', color=color)
            else:
                ax.text(0.5, 0.8, f'{symbol} {opponent}', ha='center', va='center', fontsize=35, fontweight='bold', color=color)
            ax.text(0.5, 0.3, f'{row["Quad"]} {row["resume_quality"]:.2f}', ha='center', va='center', fontsize=32, fontweight='bold', color=color)
        else:
            if (row['home_team'] == team) & (row['Location'] == "Home"):
                ax.text(0.5, 0.8, f'{opponent}', ha='center', va='center', fontsize=35, fontweight='bold', color='#555555')
            else:
                ax.text(0.5, 0.8, f'{symbol} {opponent}', ha='center', va='center', fontsize=35, fontweight='bold', color='#555555')
            ax.text(0.5, 0.3, f'{row["Quad"]} {1 - abs(row["resume_quality"]):.2f}', ha='center', va='center', fontsize=32, fontweight='bold', color='#555555')
        ax.set_facecolor('#CECEB2')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # Hide any unused axes
    total_plots = rows * cols
    for idx in range(num_items, total_plots):
        r = idx % rows
        c = idx // rows
        axes[r, c].axis('off')

    if cols == 4:
        fig.lines.append(plt.Line2D([0.25, 0.25], [0, 1], transform=fig.transFigure, color='black', linewidth=2))
        fig.lines.append(plt.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color='black', linewidth=2))
        fig.lines.append(plt.Line2D([0.75, 0.75], [0, 1], transform=fig.transFigure, color='black', linewidth=2))
        fontsize = 40
    elif cols == 3:
        fig.lines.append(plt.Line2D([0.33, 0.33], [0, 1], transform=fig.transFigure, color='black', linewidth=2))
        fig.lines.append(plt.Line2D([0.66, 0.66], [0, 1], transform=fig.transFigure, color='black', linewidth=2))
        fontsize = 32

    fig.text(0.5, 1.10,f"#{NET} {team} Schedule Quality", ha='center', va='center', fontsize=48, fontweight='bold', color='black')
    fig.text(0.5, 1.06, f"@PEARatings", ha='center', va='center', fontsize=fontsize, color='black', fontweight='bold')
    fig.text(0.06, 1.06, f"H: {home_record}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.06, 1.02, f"Q1: {Q1}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.195, 1.06, f"A: {away_record}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.195, 1.02, f"Q2: {Q2}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.32, 1.06, f"N: {neutral_record}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.32, 1.02, f"RQI: {RQI}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.5, 1.02, f"{Record} ({Conf_Record})", ha='center', va='center', fontsize=fontsize, color='black', fontweight='bold')
    fig.text(0.6, 1.06, f"RPI: {RPI}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.6, 1.02, f"SOS: {SOS:}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.73, 1.06, f"ELO: {ELO}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.73, 1.02, f"Q3: {Q3}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.86, 1.06, f"TSR: {PRR}", ha='left', va='center', fontsize=fontsize, color='black')
    fig.text(0.86, 1.02, f"Q4: {Q4}", ha='left', va='center', fontsize=fontsize, color='black')
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
    plt.text(117, 5.4, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    plt.text(117, 5.9, f"#{team_proj_net}", ha='center', fontsize=16)
    plt.text(117, 7.1, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(117, 7.6, f"{team_proj_record}", ha='center', fontsize=16)


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
    plt.text(-135, 9.0, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 9.5, f"#{team2_proj_net}", ha='center', fontsize=16)
    plt.text(-135, 10.5, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(-135, 11.0, f"{team2_proj_record}", ha='center', fontsize=16)
    plt.text(-135, 12.0, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(-155, 12.5, f"{team2_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(-115, 12.5, f"{team2_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

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
    plt.text(135, 9.0, "Proj. NET", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 9.5, f"#{team1_proj_net}", ha='center', fontsize=16)
    plt.text(135, 10.5, "Proj. Record", ha='center', fontsize=16, fontweight='bold')
    plt.text(135, 11.0, f"{team1_proj_record}", ha='center', fontsize=16)
    plt.text(135, 12.0, "Win Quality", ha='center', fontsize=16, fontweight='bold')
    plt.text(115, 12.5, f"{team1_win_quality:.2f}", ha='left', fontsize=16, color='green', fontweight='bold')
    plt.text(155, 12.5, f"{team1_loss_quality:.2f}", ha='right', fontsize=16, color='red', fontweight='bold')

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

def simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics):
    teams = [team_a, team_b, team_c, team_d]
    r = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

    w1, l1 = (team_a, team_d) if random.random() < PEAR_Win_Prob(r[team_a], r[team_d]) / 100 else (team_d, team_a)
    w2, l2 = (team_b, team_c) if random.random() < PEAR_Win_Prob(r[team_b], r[team_c]) / 100 else (team_c, team_b)
    w3 = l2 if random.random() < PEAR_Win_Prob(r[l2], r[l1]) / 100 else l1
    w4, l4 = (w1, w2) if random.random() < PEAR_Win_Prob(r[w1], r[w2]) / 100 else (w2, w1)
    w5 = l4 if random.random() < PEAR_Win_Prob(r[l4], r[w3]) / 100 else w3
    game6_prob = PEAR_Win_Prob(r[w4], r[w5]) / 100
    w6 = w4 if random.random() < game6_prob else w5

    return w6 if w6 == w4 else (w4 if random.random() < game6_prob else w5)

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

def simulate_regional(team_a, team_b, team_c, team_d, stats_and_metrics):

    output = run_simulation(team_a, team_b, team_c, team_d, stats_and_metrics)
    regional_prob = pd.DataFrame(list(output.items()), columns=["Team", "Win Regional"])
    seed_map = {
        team_a: "#1 " + team_a,
        team_b: "#2 " + team_b,
        team_c: "#3 " + team_c,
        team_d: "#4 " + team_d,
    }
    regional_prob["Team"] = regional_prob["Team"].map(seed_map)
    regional_prob['Win Regional'] = regional_prob['Win Regional'] * 100

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

st.title(f"{current_season} CBASE PEAR")
st.logo("./PEAR/pear_logo.jpg", size = 'large')
st.caption(f"Ratings Updated {formatted_latest_date}")
st.caption(f"Stats Through Games {last_date}")
st.caption(f"If you get an error, notice a bug, have a suggestion, or a question to ask, reach out to me!")

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
    <a class="nav-link" href="#team-schedule">Team Schedule</a>
    <a class="nav-link" href="#team-percentiles">Team Percentiles</a>
    <a class="nav-link" href="#conference-team-sheets">Conference Team Sheets</a>
    <a class="nav-link" href="#simulate-regional">Simulate Regional</a>
    <a class="nav-link" href="#projected-conference-tournament">Projected Conference Tournament</a>
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
    automatic_qualifiers = modeling_stats.loc[modeling_stats.groupby("Conference")["NET"].idxmin()]
    at_large = modeling_stats.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = modeling_stats.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
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
    formatted_df = resolve_conflicts(formatted_df, modeling_stats)
    formatted_df.set_index('Host')
    formatted_df.index = formatted_df.index + 1
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
    st.markdown(f'<h2 id="team-schedule">Team Schedule</h2>', unsafe_allow_html=True)
    with st.form(key='team_schedule'):
        team_name = st.selectbox("Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_schedule = st.form_submit_button("Team Schedule")
        if team_schedule:
            rank, best, worst, schedule, completed, last_ten = grab_team_schedule(team_name, modeling_stats)
            wins, losses = sum(completed['Result'].str.contains('W')), sum(completed['Result'].str.contains('L'))
            record = str(wins) + "-" + str(losses)
            projected_record = modeling_stats[modeling_stats['Team'] == team_name]['Projected_Record'].values[0]
            projected_net = modeling_stats[modeling_stats['Team'] == team_name]['Projected_NET'].values[0]
            fig = team_schedule_quality(team_name, schedule_df, modeling_stats)
            # st.write(f"Record: {record}")
            # st.write(f"Projected Record: {projected_record}")
            st.write(f"NET Rank: {rank}, Best Win - {best}, Worst Loss - {worst}")
            st.write(f"Record: {record}, Last Ten: {last_ten}")
            st.write(f"Projected NET: {projected_net}")
            st.write(f"Projected Record: {projected_record}")
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

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<h2 id="team-percentiles">Team Percentiles</h2>', unsafe_allow_html=True)
    with st.form(key='team_percentile'):
        team_name = st.selectbox("Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
        team_percentile = st.form_submit_button("Team Percentiles")
        if team_percentile:
            fig = team_percentiles_chart(team_name, modeling_stats)
            st.pyplot(fig)
with col2:
    comparison_date = comparison_date.strftime("%B %d, %Y")
    st.subheader(f"{comparison_date} Games")
    subset_games['Home'] = subset_games['home_team']
    subset_games['Away'] = subset_games['away_team']
    with st.container(border=True, height=440):
        st.dataframe(subset_games[['Home', 'Away', 'GQI', 'PEAR', 'Result']], use_container_width=True)

st.divider()