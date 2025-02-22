from datetime import datetime # type: ignore
import os # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pytz # type: ignore

formatted_date = datetime.today().strftime('%m_%d_%Y')
current_season = datetime.today().year
schedule_df = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/schedule_{current_season}.csv")
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])
comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")
# formatted_date_dt = pd.to_datetime(comparison_date, format="%m_%d_%Y")
subset_games = schedule_df[
    (schedule_df["Date"] >= comparison_date) &
    (schedule_df["Date"] <= comparison_date + pd.Timedelta(days=0))
][['home_team', 'away_team', 'PEAR', 'Date']].sort_values('Date').drop_duplicates().reset_index(drop=True)

folder_path = f"./PEAR/PEAR Baseball/y{current_season}"

csv_files = [f for f in os.listdir(folder_path) if f.startswith("baseball_") and f.endswith(".csv")]

# Extract dates from filenames and find the closest one
def extract_date(filename):
    try:
        return datetime.strptime(filename.replace("baseball_", "").replace(".csv", ""), "%m_%d_%Y")
    except ValueError:
        return None

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

        location_info = game.find('div', class_='team-schedule__info')
        location = location_info.text.strip() if location_info else "Unknown"

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
        schedule_data.append([team_name, game_date, opponent_name, location, result_text, home_team, away_team, home_score, away_score])

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
        stats_df[['Team', 'Rating Rank']],  # Keep only "Rating" and "Resume"
        left_on="Opponent",
        right_on="Team",  # Match "Opponent" with the "Rating" column (previously the index)
        how="left"  # Keep all rows from schedule_df
    )
    schedule_df.rename(columns={'Team_x':'Team', 'Rating Rank':'Rating'}, inplace=True)
    schedule_df = schedule_df.drop(columns=['Team_y'])

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
    schedule_df["Compairson_Date"] = schedule_df["Date"].astype(str).apply(convert_date)
    schedule_df["Comparison_Date"] = pd.to_datetime(schedule_df["Date"], format="%m-%d-%Y")
    comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")
    completed_schedule = schedule_df[
        (schedule_df["Date"] < comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
    ].reset_index(drop=True)

    win_rating = 500
    best_win_opponent = ""
    loss_rating = 0
    worst_loss_opponent = ""
    for _, row in completed_schedule.iterrows():
        if row['Team'] == row['home_team']:
            if row['home_score'] > row['away_score']:
                if row['Rating'] < win_rating:
                    win_rating = row['Rating'].values[0]
                    best_win_opponent = row['Opponent'].values[0]
            else:
                if row['Rating'] > loss_rating:
                    loss_rating = row['Rating'].values[0]
                    worst_loss_opponent = row['Opponent'].values[0]
        else:
            if row['away_score'] > row['home_score']:
                if row['Rating'] < win_rating:
                    win_rating = row['Rating'].values[0]
                    best_win_opponent = row['Opponent'].values[0]
            else:
                if row['Rating'] > loss_rating:
                    loss_rating = row['Rating'].values[0]
                    worst_loss_opponent = row['Opponent'].values[0]
                
    return best_win_opponent, worst_loss_opponent, schedule_df

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 1.5

def find_spread(home_team, away_team):
    home_pr = modeling_stats[modeling_stats['Team'] == home_team]['Rating'].values[0]
    away_pr = modeling_stats[modeling_stats['Team'] == away_team]['Rating'].values[0]
    home_elo = modeling_stats[modeling_stats['Team'] == home_team]['ELO'].values[0]
    away_elo = modeling_stats[modeling_stats['Team'] == away_team]['ELO'].values[0]
    win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)
    raw_spread = adjust_home_pr(win_prob) + home_pr - away_pr
    spread = round(raw_spread,2)
    if spread >= 0:
        return f"{home_team} -{spread}"
    else:
        return f"{away_team} {spread}"


st.title(f"{current_season} CBASE PEAR")
st.caption(f"Last Updated {formatted_latest_date}")

st.divider()

st.subheader("CBASE Ratings and Resume")
modeling_stats.index = modeling_stats.index + 1
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'Rating', 'Resume', 'KPI', 'WAB', 'SOR', 'SOS', 'RemSOS']], use_container_width=True)
st.caption("Resume - Resume Rank based on Average from KPI, WAB, SOR, KPI - Kevin Pauga Index, WAB - Wins Above Bubble, SOR - Strength of Record, SOS - Strength of Schedule, RemSOS - Remaining Strength of Schedule")

st.divider()

st.subheader("CBASE Stats")
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'Q1', 'Q2', 'Q3', 'Q4', 'PYTHAG', 'ERA', 'WHIP', 'KP9', 'BA', 'OBP', 'SLG', 'OPS']], use_container_width=True)
st.caption("PYTHAG - Pythagorean Win Percentage, ERA - Earned Run Average, WHIP - Walks Hits Over Innings Pitched, KP9 - Strikeouts Per 9, BA - Batting Average, OBP - On Base Percentage, SLG - Slugging Percentage, OPS - On Base Plus Slugging")

st.divider()

st.subheader("Projected Spreads")
with st.form(key='calculate_spread'):
    away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
    home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
    spread_button = st.form_submit_button("Calculate Spread")
    if spread_button:
        st.write(find_spread(home_team, away_team))

st.divider()

st.subheader("Team Schedule")
with st.form(key='team_schedule'):
    team_name = st.selectbox("Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
    team_schedule = st.form_submit_button("Team Schedule")
    if team_schedule:
        best, worst, schedule = grab_team_schedule(team_name, modeling_stats)
        st.write(f"Best Win: {best}")
        st.write(f"Worst Loss: {worst}")
        st.dataframe(schedule[['Opponent', 'Rating', 'Result', 'Date']], use_container_width=True)

st.divider()

comparison_date = comparison_date.strftime("%B %d, %Y")
st.subheader(f"{comparison_date} Games")
subset_games['Home'] = subset_games['home_team']
subset_games['Away'] = subset_games['away_team']
with st.container(border=True, height=440):
    st.dataframe(subset_games[['Home', 'Away', 'PEAR']], use_container_width=True)

st.divider()