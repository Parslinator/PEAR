import streamlit as st # type: ignore
import pandas as pd # type: ignore
import cfbd # type: ignore
import numpy as np # type: ignore
import altair as alt # type: ignore
import statistics # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import datetime # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
import requests # type: ignore
from io import BytesIO # type: ignore
from PIL import ImageGrab # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # type: ignore
from base64 import b64decode # type: ignore
from io import BytesIO # type: ignore
from IPython import get_ipython # type: ignore
import PIL # type: ignore
import os # type: ignore
import warnings # type: ignore
import seaborn as sns # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import matplotlib.image as mpimg # type: ignore
import matplotlib.pyplot as plt # type: ignore
import requests # type: ignore
import math # type: ignore
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import matplotlib.font_manager as fm # type: ignore
import matplotlib.colors as mcolors # type: ignore
import pytz # type: ignore
import datetime
checkmark_font = fm.FontProperties(family='DejaVu Sans')
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

week_list = [9,10,11,12,13,14,15,16]

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_client = cfbd.ApiClient(configuration)
games_api = cfbd.GamesApi(api_client)
betting_api = cfbd.BettingApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
teams_api = cfbd.TeamsApi(api_client)
metrics_api = cfbd.MetricsApi(api_client)
players_api = cfbd.PlayersApi(api_client)
recruiting_api = cfbd.RecruitingApi(api_client)

current_time = datetime.datetime.now(pytz.UTC)
if current_time.month < 6:
    calendar_year = current_time.year - 1
else:
    calendar_year = current_time.year
week_start_list = [*games_api.get_calendar(year = calendar_year)]
calendar_dict = [dict(
    first_game_start = c.first_game_start,
    last_game_start = c.last_game_start,
    season = c.season,
    season_type = c.season_type,
    week = c.week
) for c in week_start_list]
calendar = pd.DataFrame(calendar_dict)
calendar['first_game_start'] = pd.to_datetime(calendar['first_game_start'])
calendar['last_game_start'] = pd.to_datetime(calendar['last_game_start'])
current_year = int(calendar.loc[0, 'season'])
first_game_start = calendar['first_game_start'].iloc[0]
last_game_start = calendar['last_game_start'].iloc[-1]
current_week = None
if current_time < first_game_start:
    current_week = 1
    postseason = False
elif current_time > last_game_start:
    current_week = calendar.iloc[-2, -1] + 1
    postseason = True
else:
    condition_1 = (calendar['first_game_start'] <= current_time) & (calendar['last_game_start'] >= current_time)
    condition_2 = (calendar['last_game_start'].shift(1) < current_time) & (calendar['first_game_start'] > current_time)

    # Combine conditions
    result = calendar[condition_1 | condition_2].reset_index(drop=True)
    if result['season_type'][0] == 'regular':
        current_week = result['week'][0]
        postseason = False
    else:
        current_week = calendar.iloc[-2, -1] + 1
        postseason = True
current_week = int(current_week)
current_year = int(current_year)

team_data = pd.read_csv(f"./PEAR/Ratings/y{current_year}/PEAR_week{current_week}.csv").drop(columns=['Unnamed: 0'])
all_data = pd.read_csv(f"./PEAR/Data/y{current_year}/team_data_week{current_week}.csv")

all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)

def date_sort(game):
    game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
    return game_date

def PEAR_Win_Prob(home_pr, away_pr):
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 20)) * 100, 2)
    return win_prob

@st.cache_data()
def get_elo():
    elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week)]
    elo_ratings_dict = [dict(
        team = e.team,
        elo = e.elo
    ) for e in elo_ratings_list]
    elo_ratings = pd.DataFrame(elo_ratings_dict)
    return elo_ratings
elo_ratings = get_elo()

@st.cache_data()
def fetch_logo_image(logo_url):
    response = requests.get(logo_url)
    return Image.open(BytesIO(response.content))
    
# Function to calculate spread
def PEAR_Win_Prob(home_pr, away_pr):
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 20)) * 100, 2)
    return win_prob

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 5

def round_to_nearest_half(x):
    return np.round(x * 2) / 2

def grab_team_rating(team):
    return team_data[team_data['team'] == team]['power_rating'].values[0]

@st.cache_data()
def grab_team_elo(team):
    if postseason == True:
        elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, team=team)]
        elo_ratings_dict = [dict(
            team=e.team,
            elo=e.elo
        ) for e in elo_ratings_list]
        elo_ratings = pd.DataFrame(elo_ratings_dict)
    else:
        elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week, team=team)]
        elo_ratings_dict = [dict(
            team=e.team,
            elo=e.elo
        ) for e in elo_ratings_list]
        elo_ratings = pd.DataFrame(elo_ratings_dict)        
    return elo_ratings['elo'].values[0]

@st.cache_data()
def find_spread(home_team, away_team, neutral=False):
    home_elo = grab_team_elo(home_team)
    away_elo = grab_team_elo(away_team)
    home_pr = grab_team_rating(home_team)
    away_pr = grab_team_rating(away_team)
    home_win_prob = round((10 ** ((home_elo - away_elo) / 400)) / ((10 ** ((home_elo - away_elo) / 400)) + 1) * 100, 2)
    HFA = 4.6
    adjustment = adjust_home_pr(home_win_prob)
    raw_spread = HFA + home_pr + adjustment - away_pr
    if neutral:
        raw_spread -= HFA

    spread = round(raw_spread,1)
    PEAR_win_prob = PEAR_Win_Prob(home_pr, away_pr)
    game_quality = round(((home_pr + away_pr) / 2) - (abs(spread + 0.01) * 0.5), 1)

    if spread >= 0:
        return f"{home_team} -{spread}"
    else:
        return f"{away_team} {spread}"

# team_data.index = team_data.index + 1
def get_week_spreads(team_data):
    import datetime
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = '7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
    configuration.api_key_prefix['Authorization'] = 'Bearer'
    api_client = cfbd.ApiClient(configuration)
    advanced_instance = cfbd.StatsApi(api_client)
    games_api = cfbd.GamesApi(api_client)
    betting_api = cfbd.BettingApi(api_client)
    ratings_api = cfbd.RatingsApi(api_client)
    teams_api = cfbd.TeamsApi(api_client)
    metrics_api = cfbd.MetricsApi(api_client)
    players_api = cfbd.PlayersApi(api_client)
    recruiting_api = cfbd.RecruitingApi(api_client)
    def date_sort(game):
        game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
        return game_date
    
    if postseason:
        games = []
        response = games_api.get_games(year=current_year, division = 'fbs', season_type="postseason")
        games = [*games, *response]
        elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week, season_type="postseason")]
        elo_ratings_dict = [dict(
            team = e.team,
            elo = e.elo
        ) for e in elo_ratings_list]
        elo_ratings = pd.DataFrame(elo_ratings_dict)
    else:
        games = []
        response = games_api.get_games(year=current_year, week = current_week, division = 'fbs')
        games = [*games, *response]
        elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week)]
        elo_ratings_dict = [dict(
            team = e.team,
            elo = e.elo
        ) for e in elo_ratings_list]
        elo_ratings = pd.DataFrame(elo_ratings_dict)
    games_dict = [dict(
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
                ) for g in games]
    games_dict.sort(key=date_sort)
    week_games = pd.DataFrame(games_dict)

    week_games['home_elo'] = week_games.apply(
    lambda row: elo_ratings.loc[elo_ratings['team'] == row['home_team'], 'elo'].values[0]
    if pd.isna(row['home_elo']) else row['home_elo'], axis=1
    )
    
    # Update `away_elo` where it is NaN or None
    week_games['away_elo'] = week_games.apply(
        lambda row: elo_ratings.loc[elo_ratings['team'] == row['away_team'], 'elo'].values[0]
        if pd.isna(row['away_elo']) else row['away_elo'], axis=1
    )

    import math

    week_games = week_games.merge(
        team_data[['team', 'power_rating']],
        left_on='home_team',
        right_on='team',
        how='left'
    ).rename(columns={'power_rating': 'home_pr'})
    week_games = week_games.merge(
        team_data[['team', 'power_rating']],
        left_on='away_team',
        right_on='team',
        how='left'
    ).rename(columns={'power_rating': 'away_pr'})
    week_games = week_games.drop(columns=['team_x', 'team_y'])


    week_games = week_games.merge(
        all_data[['team', 'offensive_total', 'defensive_total']],
        left_on='home_team',
        right_on='team',
        how='left'
    ).rename(columns={'offensive_total':'home_offense', 'defensive_total':'home_defense'})
    week_games = week_games.merge(
        all_data[['team', 'offensive_total', 'defensive_total']],
        left_on='away_team',
        right_on='team',
        how='left'
    ).rename(columns={'offensive_total':'away_offense', 'defensive_total':'away_defense'})
    week_games = week_games.drop(columns=['team_x', 'team_y'])
    week_games['xhome_points'] = round((week_games['home_offense'] - week_games['away_defense'] + (4.6/2)),1)
    week_games['xaway_points'] = round((week_games['away_offense'] - week_games['home_defense'] - (4.6/2)),1)
    week_games['predicted_over_under'] = week_games['xhome_points'] + week_games['xaway_points']

    def adjust_home_pr(home_win_prob):
        return ((home_win_prob - 50) / 50) * 5
    week_games['home_win_prob'] = round((10**((week_games['home_elo'] - week_games['away_elo']) / 400)) / ((10**((week_games['home_elo'] - week_games['away_elo']) / 400)) + 1)*100,2)
    
    week_games['pr_spread'] = (4.6 + week_games['home_pr'] + (week_games['home_win_prob'].apply(adjust_home_pr)) - week_games['away_pr']).round(1)
    week_games['pr_spread'] = np.where(week_games['neutral'], week_games['pr_spread'] - 4.6, week_games['pr_spread']).round(1)
    # week_games['pr_spread'] = week_games['pr_spread'].apply(round_to_nearest_half)

    scaler10 = MinMaxScaler(feature_range=(1,10))
    week_games['game_quality'] = ((week_games['home_pr'] + week_games['away_pr']) / 2) - abs(week_games['pr_spread'] * 0.5)
    week_games['game_quality'] = round(week_games['game_quality'], 1)

    # Comparing Prediction to Vegas Spread
    if postseason:
        betting = []
        response = betting_api.get_lines(year=current_year, season_type="postseason")
        betting.extend(response)  # Use extend for list concatenation
    else:
        betting = []
        response = betting_api.get_lines(year=current_year, week=current_week)
        betting.extend(response)  # Use extend for list concatenation

    betting_info_list = []

    for bet in betting:
        data = bet.to_dict() if hasattr(bet, 'to_dict') else vars(bet)
        lines = pd.DataFrame(data['lines'])

        if not lines.empty:
            # Try to get consensus lines first
            consensus_lines = lines[lines['provider'] == 'consensus']
            
            if consensus_lines.empty:
                consensus_lines = lines[lines['provider'] == 'DraftKings']
            if consensus_lines.empty:
                consensus_lines = lines[lines['provider'] == 'Bovada']
            if consensus_lines.empty:
                consensus_lines = lines[lines['provider'] == 'ESPN Bet']
            


            if not consensus_lines.empty:
                consensus_lines = consensus_lines[['spread', 'formatted_spread','spread_open', 'over_under']]
                combined_data = {
                    'id': data['id'],
                    'season_type': data['season_type']
                }
                df = pd.DataFrame([combined_data])
                full_df = pd.concat([df.reset_index(drop=True), consensus_lines.reset_index(drop=True)], axis=1)
                betting_info_list.append(full_df)

    betting_info = pd.concat(betting_info_list, ignore_index=True)
    week_games = pd.merge(week_games, betting_info, on='id', how='left')
    week_games['spread'] = week_games['spread'] * -1
    week_games['spread_open'] = week_games['spread_open'] * -1

    # if current_week == 7:
    #     week_games.loc[week_games['home_team'] == 'Western Kentucky', 'pr_spread'] += 0.5

    # Capping predictions that are more than 15 points away from the Vegas Spread
    threshold = 10
    capped_preds = np.clip(week_games['pr_spread'], week_games['spread'] - threshold, week_games['spread'] + threshold)
    week_games['pr_spread'] = capped_preds

    # Function to find out if PR predicts the favorite or underdog
    def calculate_pr_prediction(row, pr_spread_col, vegas_spread_col):
        if (row[vegas_spread_col] < 0) and (row[pr_spread_col] < 0) and (row[pr_spread_col] < row[vegas_spread_col]):
            return 'Favorite'
        elif (row[vegas_spread_col] > 0) and (row[pr_spread_col] > 0) and (row[pr_spread_col] > row[vegas_spread_col]):
            return 'Favorite'
        elif (row[vegas_spread_col] == row[pr_spread_col]):
            return 'Exact'
        else:
            return 'Underdog'

    week_games['formatted_open'] = week_games.apply(
        lambda row: f"{row['away_team']} {row['spread_open']}" if row['spread_open'] < 0 
                    else f"{row['home_team']} -{row['spread_open']}", axis=1
    )

    # Use the above function
    def add_pr_prediction(week_games, pr_spread_col, vegas_spread_col, prediction_col_name='pr_prediction'):
        week_games[prediction_col_name] = week_games.apply(calculate_pr_prediction, axis=1, args=(pr_spread_col,vegas_spread_col,))
        return week_games
    week_games = add_pr_prediction(week_games, 'pr_spread', 'spread', 'pr_prediction')
    week_games = add_pr_prediction(week_games, 'pr_spread', 'spread_open', 'opening_spread_prediction')

    # Formatting the KRATOS Power Rating Spread
    week_games['PEAR'] = week_games.apply(
        lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
        else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)

    week_games['difference'] = abs(week_games['pr_spread'] - week_games['spread'])
    week_games['opening_difference'] = abs(week_games['pr_spread'] - week_games['spread_open'])
    week_games['over_under_difference'] = abs(week_games['over_under'] - week_games['predicted_over_under'])
    week_games = week_games.sort_values(by=["difference", "home_win_prob"], ascending=False).reset_index(drop=True)
    week_games = week_games.drop_duplicates(subset='home_team')
    import math
    week_games['actual_margin'] = week_games['home_points'] - week_games['away_points']
    def calculate_margin_team(row):
        if row['actual_margin'] > 0:
            return f"{row['home_team']} -{row['actual_margin']}"  # If actual_margin is positive
        elif row['actual_margin'] < 0:
            return f"{row['away_team']} {row['actual_margin']}"  # If actual_margin is negative
        else:
            return ''
    week_games['actual_spread'] = week_games.apply(calculate_margin_team, axis=1)
    week_games = add_pr_prediction(week_games, 'actual_margin', 'spread', 'CLOSE ATS RESULT')
    week_games = add_pr_prediction(week_games, 'actual_margin', 'spread_open', 'OPEN ATS RESULT')

    def check_prediction_correct(row, prediction_col, ats_tester):
        if row['actual_spread'] == '':
            return ''
        if row[prediction_col] == row[ats_tester]:
            return 1
        elif 'Exact' in (row[prediction_col], row[ats_tester]):
            return 1
        else:
            return 0
    # Apply the check prediction function and store the result in a new column
    week_games['PEAR ATS CLOSE'] = week_games.apply(lambda row: check_prediction_correct(row, 'pr_prediction', 'CLOSE ATS RESULT'), axis=1)
    week_games['PEAR ATS OPEN'] = week_games.apply(lambda row: check_prediction_correct(row, 'opening_spread_prediction', 'OPEN ATS RESULT'), axis=1)

    def check_straight_up(row, prediction_col):
        if row['actual_spread'] == '':
            return ''
        if (row['actual_margin'] < 0) and (row[prediction_col] < 0):
            return 1
        elif (row['actual_margin'] > 0) and (row[prediction_col] > 0):
            return 1
        else:
            return 0
    week_games['PEAR SU'] = week_games.apply(lambda row: check_straight_up(row, 'pr_spread'), axis = 1)
    return week_games
st.title(f"{current_year} PEAR")
# st.logo("./PEAR/pear_logo.jpg", size = 'large')

# week_spreads = get_week_spreads(team_data)
# week_spreads['DK Spread'] = week_spreads['formatted_spread']
# week_spreads['PEAR Spread'] = week_spreads['PEAR']
# week_spreads.columns.values[4] = 'Home'
# week_spreads.columns.values[8] = 'Away'
# week_spreads.index = week_spreads.index + 1
# game_completion_info = week_spreads[['Home', 'Away', 'difference', 'formatted_open', 'formatted_spread', 'PEAR', 'actual_spread', 'PEAR ATS OPEN', 'PEAR ATS CLOSE', 'PEAR SU']]
# completed = game_completion_info[game_completion_info["PEAR ATS CLOSE"] != '']
# if postseason == True:
#     st.subheader("Bowl Games Projected Spreads, Ordered by Deviation")
# else:
#     st.subheader(f"Week {current_week} Projected Spreads, Ordered by Deviation")
# week_spreads['Deviation'] = week_spreads['difference']
# week_spreads['ATS'] = week_spreads['PEAR ATS CLOSE']
# with st.container(border=True, height=440):
#     st.dataframe(week_spreads[['PEAR Spread', 'DK Spread', 'Deviation','Home', 'Away', 'ATS']], use_container_width=True)
# X = 10
# if len(completed) > 0:
#     no_pushes = completed[completed['difference'] != 0.0]
#     st.markdown(f"ATS This Week: {round(100 * sum(no_pushes['PEAR ATS CLOSE']) / len(no_pushes),1)}% through {round(100*len(completed)/len(week_spreads))}% of games.")
#     st.markdown(f"SU This Week: {round(100*sum(completed['PEAR SU'] / len(completed)),1)}%")
#     # print(f'wATS: {wATS}%')
#     # print(f"MAE: {MAE}")
#     # print(f"DAE: {DAE}")
#     # print(f"RMSE: {RMSE}")
#     # print(f"MAE+: {round(100-MAE_plus,2)}%")
#     # print(f"AE < {X}: {round(count/len(completed)*100,2)}%")
# st.caption(f"Deviation is defined as the absolute difference from the DraftKings spread at the time of the data load. The current average deviation is {round(week_spreads['Deviation'].mean(),2)} points. The total deviation is {round(week_spreads['Deviation'].sum(),1)} points.")

st.divider()

st.subheader("Calculate Spread Between Any Two Teams")
with st.form(key='calculate_spread'):
    away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(team_data['team'])))
    home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(team_data['team'])))
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

st.subheader("FBS Power Ratings")
all_data['OFF'] = all_data['Offense']
all_data['DEF'] = all_data['Defense']
all_data['MD'] = all_data['most_deserving']
all_data['Rating'] = all_data['power_rating']
all_data['Team'] = all_data['team']
all_data['CONF'] = all_data['conference']
all_data['ST'] = all_data['STM_rank']
all_data['PBR'] = all_data['PBR_rank']
all_data['DCE'] = all_data['DCE_rank']
all_data['DDE'] = all_data['DDE_rank']

all_data.index = all_data.index + 1
with st.container(border=True, height=440):
    st.dataframe(all_data[['Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'ST', 'PBR', 'DCE', 'DDE', 'CONF']], use_container_width=True)
st.caption("MD - Most Deserving (PEAR's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()



# st.markdown("General Info for PEAR v2")
# st.caption(f"SU Since Week 9: {SU}%")
# st.caption(f"ATS Since Week 9: {ATS}%")
# st.caption(f"Mean Absolute Error: {MAE}")
# st.caption(f"Median Absolute Error: {DAE}")
# st.caption(f"Root Square Error: {RMSE}")
# st.caption("Made by me. Who is me? Well, I'm me. If you run into an error, please reach out to me at @PEARRatingsCFB on Twitter/X.")
st.caption("PEAR v2 came to be in Week 9, 2024. Currently powered by PEAR v2.9")