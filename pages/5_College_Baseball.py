from datetime import datetime # type: ignore
import os # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore

formatted_date = datetime.today().strftime('%m_%d_%Y')
current_season = datetime.today().year

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

def find_spread(home_team, away_team, neutrality):
    home_pr = modeling_stats[modeling_stats['Team'] == home_team]['Rating'].values[0]
    away_pr = modeling_stats[modeling_stats['Team'] == away_team]['Rating'].values[0]
    raw_spread = 0.35 + home_pr - away_pr
    if neutrality:
        raw_spread -= 0.35
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
    neutrality = st.radio(
        "Game Location",
        ["Neutral Field", "On Campus"]
    )
    spread_button = st.form_submit_button("Calculate Spread")
    if spread_button:
        if neutrality == 'Neutral Field':
            neutrality = True
        else:
            neutrality = False
        st.write(find_spread(home_team, away_team, neutrality))

st.divider()

modeling_stats.index = modeling_stats.index + 1
modeling_stats[['Wins', 'Losses']] = modeling_stats['Rec'].str.split('-', expand=True).astype(int)
modeling_stats['WIN%'] = round(modeling_stats['Wins'] / (modeling_stats['Wins'] + modeling_stats['Losses']), 3)
modeling_stats.drop(columns=['Wins', 'Losses'], inplace=True)
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'Rating', 'Conf', 'Rec', 'WIN%', 'ERA', 'WHIP', 'BA', 'OBP', 'SLG', 'OPS']], use_container_width=True)
