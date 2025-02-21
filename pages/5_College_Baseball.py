from datetime import datetime # type: ignore
import os # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore


formatted_date = datetime.today().strftime('%m_%d_%Y')
current_season = datetime.today().year
schedule_df = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/schedule_{current_season}.csv")
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], format="%m-%d-%Y")
# comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")
# formatted_date_dt = pd.to_datetime(comparison_date, format="%m_%d_%Y")
subset_games = schedule_df[
    (schedule_df["Date"] >= formatted_date) &
    (schedule_df["Date"] <= formatted_date + pd.Timedelta(days=0))
].reset_index(drop=True).sort_values('Date')

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

st.subheader("Calculate Spread Between Any Two Teams")
with st.form(key='calculate_spread'):
    away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
    home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(modeling_stats['Team'])))
    spread_button = st.form_submit_button("Calculate Spread")
    if spread_button:
        st.write(find_spread(home_team, away_team))

st.divider()

st.subheader("CBASE Power Ratings")
modeling_stats.index = modeling_stats.index + 1
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'Rating', 'Conf', 'Rec', 'PYTHAG', 'SOR', 'SOS', 'RemSOS', 'Q1', 'Q2', 'Q3', 'Q4']], use_container_width=True)
st.caption("PYTHAG - Pythagorean Win Percentage, SOR - Strength of Record, SOS - Strength of Schedule, RemSOS - Remaining Strength of Schedule")

st.divider()
st.subheader("CBASE Stats")
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'ERA', 'WHIP', 'KP9', 'BA', 'OBP', 'SLG', 'OPS']], use_container_width=True)
st.caption("ERA - Earned Run Average, WHIP - Walks Hits Over Innings Pitched, KP9 - Strikeouts Per 9, BA - Batting Average, OBP - On Base Percentage, SLG - Slugging Percentage, OPS - On Base Plus Slugging")

st.divider()
st.subheader(f"{formatted_latest_date} Games")
subset_games['Home'] = subset_games['home_team']
subset_games['Away'] = subset_games['away_team']
with st.container(border=True, height=440):
    st.dataframe(subset_games[['Home', 'Away', 'Pear']])