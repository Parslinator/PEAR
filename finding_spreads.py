import pandas as pd # type: ignore
import cfbd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import numpy as np # type: ignore
from scipy.optimize import minimize # type: ignore
from scipy.optimize import differential_evolution # type: ignore
from tqdm import tqdm # type: ignore
import os # type: ignore
import datetime
import pytz # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import explained_variance_score # type: ignore
import math
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

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
elif current_time > last_game_start:
    current_week = calendar.iloc[-2, -1] + 1
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

current_year = int(current_year)
current_week = int(current_week)
print(f"Current Week: {current_week}, Current Year: {current_year}")

team_data = pd.read_csv(f'./ESCAPE Ratings/Ratings/y{current_year}/ESCAPE_week{current_week}.csv').drop(columns=['Unnamed: 0'])
all_data = pd.read_csv(f"./ESCAPE Ratings/Data/y{current_year}/team_data_week{current_week}.csv").drop(columns=['Unnamed: 0'])

offensive_scaler = MinMaxScaler(feature_range=(35,70))
defensive_scaler = MinMaxScaler(feature_range=(15,40))
all_data['offensive_total'] = offensive_scaler.fit_transform(all_data[['offensive_total']])
all_data['defensive_total'] = defensive_scaler.fit_transform(all_data[['defensive_total']])

def date_sort(game):
    game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
    return game_date

if postseason:
    games = []
    response = games_api.get_games(year=current_year, division = 'fbs', season_type='postseason')
    games = [*games, *response]
else:
    games = []
    response = games_api.get_games(year=current_year, week = current_week, division = 'fbs')
    games = [*games, *response]


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

elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week)]
elo_ratings_dict = [dict(
    team = e.team,
    elo = e.elo
) for e in elo_ratings_list]
elo_ratings = pd.DataFrame(elo_ratings_dict)

week_games['home_elo'] = week_games.apply(
    lambda row: elo_ratings.loc[elo_ratings['team'] == row['home_team'], 'elo'].values[0]
    if pd.isna(row['home_elo']) else row['home_elo'], axis=1
)

# Update `away_elo` where it is NaN or None
week_games['away_elo'] = week_games.apply(
    lambda row: elo_ratings.loc[elo_ratings['team'] == row['away_team'], 'elo'].values[0]
    if pd.isna(row['away_elo']) else row['away_elo'], axis=1
)

def round_to_nearest_half(x):
    return np.round(x * 2) / 2
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
            consensus_lines = lines[lines['provider'] == 'ESPN Bet']
        if consensus_lines.empty:
            consensus_lines = lines[lines['provider'] == 'Bovada']
        


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
week_games['ESCAPE'] = week_games.apply(
    lambda row: f"{row['away_team']} {-abs(row['pr_spread'])}" if ((row['pr_spread'] <= 0)) 
    else f"{row['home_team']} {-abs(row['pr_spread'])}", axis=1)

week_games['difference'] = abs(week_games['pr_spread'] - week_games['spread'])
week_games['opening_difference'] = abs(week_games['pr_spread'] - week_games['spread_open'])
week_games['over_under_difference'] = abs(week_games['over_under'] - week_games['predicted_over_under'])
week_games = week_games.sort_values(by=["difference", "home_win_prob"], ascending=False).reset_index(drop=True)
week_games = week_games.drop_duplicates(subset='home_team')
prediction_information = week_games[['home_team', 'away_team', 'game_quality', 'home_win_prob','difference', 'formatted_open', 'formatted_spread', 'ESCAPE', 'pr_prediction', 'home_pr', 'away_pr']]
prediction_information = prediction_information.dropna()
print("Total Difference from Vegas Spread:", round(sum(prediction_information['difference']),1))
print("Average Difference from Vegas Spread:", round(sum(prediction_information['difference'])/len(prediction_information), 2))
print("Average Over Under Difference", round(sum(week_games['over_under_difference'])/len(week_games), 2))

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
week_games['ESCAPE ATS CLOSE'] = week_games.apply(lambda row: check_prediction_correct(row, 'pr_prediction', 'CLOSE ATS RESULT'), axis=1)
week_games['ESCAPE ATS OPEN'] = week_games.apply(lambda row: check_prediction_correct(row, 'opening_spread_prediction', 'OPEN ATS RESULT'), axis=1)

def check_straight_up(row, prediction_col):
    if row['actual_spread'] == '':
        return ''
    if (row['actual_margin'] < 0) and (row[prediction_col] < 0):
        return 1
    elif (row['actual_margin'] > 0) and (row[prediction_col] > 0):
        return 1
    else:
        return 0
week_games['ESCAPE SU'] = week_games.apply(lambda row: check_straight_up(row, 'pr_spread'), axis = 1)
game_completion_info = week_games[['home_team', 'away_team', 'difference', 'formatted_open', 'formatted_spread', 'ESCAPE', 'spread', 'actual_margin', 'actual_spread', 'ESCAPE ATS OPEN', 'ESCAPE ATS CLOSE', 'ESCAPE SU']]
completed = game_completion_info[game_completion_info["ESCAPE ATS CLOSE"] != '']
no_pushes = completed[completed['difference'] != 0]
no_pushes = no_pushes[no_pushes['spread'] != no_pushes['actual_margin']]

X = 10
if len(completed) > 0:
    win_difference = completed.loc[completed["ESCAPE ATS CLOSE"] == 1, "difference"].sum()
    total_difference = completed['difference'].sum()
    MAE = round(abs(week_games['actual_margin'] - week_games['pr_spread']).mean(),2)
    DAE = round(abs(week_games['actual_margin'] - week_games['pr_spread']).median(),2)
    RMSE = round(math.sqrt(((week_games['actual_margin'] - week_games['pr_spread']) ** 2).mean()),2)
    count = (abs(week_games['actual_margin'] - week_games['pr_spread']) < X).sum()
    MAE_plus = 0.5 * MAE + 0.25 * DAE + 0.25 * RMSE
    wATS = round(win_difference/total_difference * 100, 2)
    print("----------------------")
    print("Performance This Week")
    print("----------------------")
    print(f"SU: {round(100*sum(completed['ESCAPE SU'] / len(completed)),2)}%  -  {sum(completed['ESCAPE SU'])}/{len(completed)}")
    print(f"ATS: {round(100 * sum(no_pushes['ESCAPE ATS CLOSE']) / len(no_pushes),2)}%  -  {sum(no_pushes['ESCAPE ATS CLOSE'])}/{len(no_pushes)}")
    print(f'wATS: {wATS}%')
    print(f"MAE: {MAE}")
    print(f"DAE: {DAE}")
    print(f"RMSE: {RMSE}")
    print(f"MAE+: {round(100-MAE_plus,2)}%")
    print(f"AE < {X}: {round(count/len(completed)*100,2)}%")

game_completion_info.to_excel(f'./ESCAPE Ratings/Spreads/y{current_year}/spreads_tracker_week{current_week}.xlsx')