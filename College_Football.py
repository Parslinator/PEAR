import streamlit as st # type: ignore
import pandas as pd # type: ignore
import cfbd # type: ignore
import numpy as np # type: ignore
import statistics # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
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
from datetime import datetime, timedelta
checkmark_font = fm.FontProperties(family='DejaVu Sans')
warnings.filterwarnings("ignore")
GLOBAL_HFA = 3

configuration = cfbd.Configuration(
    access_token = '7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
)
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
logos = logos[logos['classification'] == 'fbs'].reset_index(drop=True)

central = pytz.timezone("US/Central")
now_ct = datetime.now(central)
start_dt = central.localize(datetime(2025, 9, 2, 9, 0, 0))

if now_ct < start_dt:
    current_week = 1
else:
    current_week = 2
    first_sunday = start_dt + timedelta(days=(6 - start_dt.weekday()))  # weekday: Mon=0, Sun=6
    first_sunday = first_sunday.replace(hour=12, minute=0, second=0, microsecond=0)
    if first_sunday <= start_dt:
        first_sunday += timedelta(weeks=1)
    if now_ct >= first_sunday:
        weeks_since = ((now_ct - first_sunday).days // 7) + 1
        current_week += weeks_since

st.set_page_config(layout="wide")

postseason = False
current_year = 2025
team_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Ratings/PEAR_week{current_week}.csv").drop(columns=['Unnamed: 0'])
all_data = pd.read_csv(f"./PEAR/PEAR Football/y{current_year}/Data/team_data_week{current_week}.csv")
spreads = pd.read_excel(f"./PEAR/PEAR Football/y{current_year}/Spreads/spreads_tracker_week{current_week}.xlsx")

def date_sort(game):
    game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
    return game_date

def PEAR_Win_Prob(home_power_rating, away_power_rating, neutral):
    if neutral == False:
        home_power_rating = home_power_rating + 1.5
    return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

def render_year(year: int, week: int, col):
    """Render a single year's ratings in the given column."""
    st.markdown(f'<h2 id="{year}-ratings">{year} Ratings</h2>', unsafe_allow_html=True)
    all_data = pd.read_csv(f"./PEAR/PEAR Football/y{year}/Data/team_data_week{week}.csv")

    # Rename + add convenience cols
    all_data.rename(columns={"offensive_rank": "Offense", "defensive_rank": "Defense"}, inplace=True)
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
        st.dataframe(
            all_data[['Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF',
                      'ST', 'PBR', 'DCE', 'DDE', 'CONF']],
            width='stretch'
        )
    st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, "
               "SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, "
               "PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")

def fetch_logo_image(logo_url):
    response = requests.get(logo_url)
    return Image.open(BytesIO(response.content))
    
def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 1

def round_to_nearest_half(x):
    return np.round(x * 2) / 2

def grab_team_rating(team):
    return team_data[team_data['team'] == team]['power_rating'].values[0]

def find_spread(home_team, away_team, neutral=False):
    # home_elo = grab_team_elo(home_team)
    # away_elo = grab_team_elo(away_team)
    home_pr = grab_team_rating(home_team)
    away_pr = grab_team_rating(away_team)
    # home_win_prob = round((10 ** ((home_elo - away_elo) / 400)) / ((10 ** ((home_elo - away_elo) / 400)) + 1) * 100, 2)
    HFA = GLOBAL_HFA
    # adjustment = adjust_home_pr(home_win_prob)
    # raw_spread = HFA + home_pr + adjustment - away_pr
    raw_spread = HFA + home_pr - away_pr
    if neutral:
        raw_spread -= HFA

    spread = round(raw_spread,1)
    PEAR_win_prob = PEAR_Win_Prob(home_pr, away_pr)
    game_quality = round(((home_pr + away_pr) / 2) - (abs(spread + 0.01) * 0.5), 1)

    if spread >= 0:
        return f"{home_team} -{spread}"
    else:
        return f"{away_team} {spread}"

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
    week_games['xhome_points'] = round((week_games['home_offense'] - week_games['away_defense'] + (GLOBAL_HFA/2)),1)
    week_games['xaway_points'] = round((week_games['away_offense'] - week_games['home_defense'] - (GLOBAL_HFA/2)),1)
    week_games['predicted_over_under'] = week_games['xhome_points'] + week_games['xaway_points']

    def adjust_home_pr(home_win_prob):
        return ((home_win_prob - 50) / 50) * 1
    week_games['home_win_prob'] = round((10**((week_games['home_elo'] - week_games['away_elo']) / 400)) / ((10**((week_games['home_elo'] - week_games['away_elo']) / 400)) + 1)*100,2)
    
    week_games['pr_spread'] = (GLOBAL_HFA + week_games['home_pr'] + (week_games['home_win_prob'].apply(adjust_home_pr)) - week_games['away_pr']).round(1)
    week_games['pr_spread'] = np.where(week_games['neutral'], week_games['pr_spread'] - GLOBAL_HFA, week_games['pr_spread']).round(1)
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

def teams_yearly_stats(team, data):
    team_df = data[data['team'] == team]
    team_df['OFF'] = team_df['offensive_rank']
    team_df['DEF'] = team_df['defensive_rank']
    team_df['MD'] = team_df['most_deserving']
    team_df['Rating'] = team_df['power_rating']
    team_df['Team'] = team_df['team']
    team_df['CONF'] = team_df['conference']
    team_df['ST'] = team_df['STM_rank']
    team_df['PBR'] = team_df['PBR_rank']
    team_df['DCE'] = team_df['DCE_rank']
    team_df['DDE'] = team_df['DDE_rank']
    team_df = team_df[['Season', 'Normalized Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'ST', 'PBR', 'DCE', 'DDE']]
    return team_df

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle

def plot_matchup_new(all_data, team_logos, away_team, home_team, neutrality, current_year, current_week):
    logo_url = logos[logos['team'] == away_team]['logo'].values[0][0]
    response = requests.get(logo_url)
    away_logo = Image.open(BytesIO(response.content))

    logo_url = logos[logos['team'] == home_team]['logo'].values[0][0]
    response = requests.get(logo_url)
    home_logo = Image.open(BytesIO(response.content))

    def fixed_width_text(ax, x, y, text, width=0.06, height=0.04,
                        facecolor="lightgrey", edgecolor="none", alpha=1.0, **kwargs):
        # Draw rectangle behind text
        ax.add_patch(Rectangle(
            (x - width/2, y - height/2), width, height,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=1
        ))

        # Draw text centered on top
        ax.text(x, y, text,
                ha="center", va="center", zorder=2, **kwargs)

    def rank_to_color(rank, vmin=1, vmax=136):
        """
        Map a rank (1–136) to a hex color.
        Dark blue = best (1), grey = middle, dark red = worst (136).
        """
        # Define colormap from blue → grey → red
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rank_cmap", ["#00008B", "#D3D3D3", "#8B0000"]  # dark blue, grey, dark red
        )
        
        # Normalize rank to [0,1]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(rank))
        
        # Convert RGBA to hex
        return mcolors.to_hex(rgba)

    def get_value_and_rank(df, team, column, higher_is_better=True):
        """
        Return (value, rank) for a given team and column.
        
        Args:
            df (pd.DataFrame): Data source with 'team' and stat columns.
            team (str): Team name to look up.
            column (str): Column name to extract.
            higher_is_better (bool): If True, high values rank better (1 = highest).
                                    If False, low values rank better (1 = lowest).
        """
        ascending = not higher_is_better
        ranks = df[column].rank(ascending=ascending, method="first").astype(int)

        value = df.loc[df['team'] == team, column].values[0]
        rank = ranks.loc[df['team'] == team].values[0]

        return value, rank

    def get_column_value(df, team, column):
        """Return the scalar value for a given team/column."""
        return df.loc[df['team'] == team, column].values[0]
    
    def adjust_home_pr(home_win_prob):
        if home_win_prob is None or (isinstance(home_win_prob, float) and math.isnan(home_win_prob)):
            return 0
        return ((home_win_prob - 50) / 50) * 1

    away_pr, away_rank = get_value_and_rank(all_data, away_team, 'power_rating')
    away_elo      = get_column_value(all_data, away_team, 'elo')
    away_offense, away_offense_rank = get_value_and_rank(all_data, away_team, 'offensive_rating')
    away_defense, away_defense_rank = get_value_and_rank(all_data, away_team, 'defensive_rating', False)
    away_off_exp, away_off_exp_rank  = get_value_and_rank(all_data, away_team, 'Offense_explosiveness_adj')
    away_off_sr, away_off_sr_rank = get_value_and_rank(all_data, away_team, 'Offense_successRate_adj')
    away_off_rus, away_off_rus_rank = get_value_and_rank(all_data, away_team, 'Offense_rushing_adj')
    away_off_pas, away_off_pas_rank = get_value_and_rank(all_data, away_team, 'Offense_passing_adj')
    away_off_ppo, away_off_ppo_rank = get_value_and_rank(all_data, away_team, 'adj_offense_ppo')
    away_off_ppa, away_off_ppa_rank = get_value_and_rank(all_data, away_team, 'Offense_ppa_adj')
    away_off_plays, away_off_plays_rank = get_value_and_rank(all_data, away_team, 'Offense_plays_adj')
    away_def_exp, away_def_exp_rank = get_value_and_rank(all_data, away_team, 'Defense_explosiveness_adj', False)
    away_def_sr, away_def_sr_rank = get_value_and_rank(all_data, away_team, 'Defense_successRate_adj', False)
    away_def_rus, away_def_rus_rank = get_value_and_rank(all_data, away_team, 'Defense_rushing_adj', False)
    away_def_pas, away_def_pas_rank = get_value_and_rank(all_data, away_team, 'Defense_passing_adj', False)
    away_def_ppo, away_def_ppo_rank = get_value_and_rank(all_data, away_team, 'adj_defense_ppo', False)
    away_def_ppa, away_def_ppa_rank = get_value_and_rank(all_data, away_team, 'Defense_ppa_adj', False)
    away_def_plays, away_def_plays_rank = get_value_and_rank(all_data, away_team, 'Defense_plays_adj', False)

    home_pr, home_rank = get_value_and_rank(all_data, home_team, 'power_rating')
    home_elo      = get_column_value(all_data, home_team, 'elo')
    home_offense, home_offense_rank = get_value_and_rank(all_data, home_team, 'offensive_rating')
    home_defense, home_defense_rank = get_value_and_rank(all_data, home_team, 'defensive_rating', False)
    home_off_exp, home_off_exp_rank = get_value_and_rank(all_data, home_team, 'Offense_explosiveness_adj')
    home_off_sr, home_off_sr_rank = get_value_and_rank(all_data, home_team, 'Offense_successRate_adj')
    home_off_rus, home_off_rus_rank = get_value_and_rank(all_data, home_team, 'Offense_rushing_adj')
    home_off_pas, home_off_pas_rank = get_value_and_rank(all_data, home_team, 'Offense_passing_adj')
    home_off_ppo, home_off_ppo_rank = get_value_and_rank(all_data, home_team, 'adj_offense_ppo')
    home_off_ppa, home_off_ppa_rank = get_value_and_rank(all_data, home_team, 'Offense_ppa_adj')
    home_off_plays, home_off_plays_rank = get_value_and_rank(all_data, home_team, 'Offense_plays_adj')
    home_def_exp, home_def_exp_rank = get_value_and_rank(all_data, home_team, 'Defense_explosiveness_adj', False)
    home_def_sr, home_def_sr_rank = get_value_and_rank(all_data, home_team, 'Defense_successRate_adj', False)
    home_def_rus, home_def_rus_rank = get_value_and_rank(all_data, home_team, 'Defense_rushing_adj', False)
    home_def_pas, home_def_pas_rank = get_value_and_rank(all_data, home_team, 'Defense_passing_adj', False)
    home_def_ppo, home_def_ppo_rank = get_value_and_rank(all_data, home_team, 'adj_defense_ppo', False)
    home_def_ppa, home_def_ppa_rank = get_value_and_rank(all_data, home_team, 'Defense_ppa_adj', False)
    home_def_plays, home_def_plays_rank = get_value_and_rank(all_data, home_team, 'Defense_plays_adj', False)

    home_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)
    PEAR_home_prob = PEAR_Win_Prob(home_pr, away_pr, neutrality)
    spread = (3+home_pr+adjust_home_pr(home_win_prob)-away_pr).round(1)
    if neutrality:
        spread = (spread-3).round(1)
    if (spread) <= 0:
        formatted_spread = (f'{away_team} {spread}')
        game_win_prob = round(100 - PEAR_home_prob,2)
    elif (spread) > 0:
        formatted_spread = (f'{home_team} -{spread}')

    home_offense_contrib = (home_offense + away_defense) / 2
    away_offense_contrib = (away_offense + home_defense) / 2
    predicted_total = round(home_offense_contrib + away_offense_contrib, 1)

    # Calculate predicted scores
    # Use spread and total to derive individual scores
    home_score = round((predicted_total + spread) / 2, 1)
    away_score = round((predicted_total - spread) / 2, 1)

    def plot_logo(ax, img, xy, zoom=0.2):
        """Helper to plot a logo at given xy coords."""
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, frameon=False)
        ax.add_artist(ab)

    # Create blank figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
    fig.patch.set_facecolor('#CECEB2')
    ax.set_facecolor('#CECEB2')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----------------
    # logos, score, win prob, spread, O/U
    # ----------------
    plot_logo(ax, away_logo, (0.15, 0.75), zoom=0.5)
    plot_logo(ax, home_logo, (0.85, 0.75), zoom=0.5)
    if neutrality:
        ax.text(0.5, 0.96, f"{away_team} (N) {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))
    else:
        ax.text(0.5, 0.96, f"{away_team} at {home_team}", ha='center', fontsize=32, fontweight='bold', bbox=dict(facecolor='red', alpha=0.0))
    ax.text(0.5, 0.57, f"{formatted_spread}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))
    ax.text(0.5, 0.625, f"O/U: {predicted_total}", ha='center', fontsize=28, fontweight='bold', bbox=dict(facecolor='blue', alpha=0.0))

    ax.text(0.4, 0.89, f"WIN PROB (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.4, 0.84, f"{round(100-PEAR_home_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.4, 0.77, f"PROJ. POINTS", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.4, 0.72, f"{away_score}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.5, 0.725, f"—", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.6, 0.89, f"WIN PROB (%)", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.6, 0.84, f"{round(PEAR_home_prob,1)}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.6, 0.77, f"PROJ. POINTS", ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))
    ax.text(0.6, 0.72, f"{home_score}", ha='center', fontsize=36, fontweight='bold', bbox=dict(facecolor='green', alpha=0.0))


    # ---------------------------
    # Helper for each row
    # ---------------------------
    def rank_text_color(rank):
        if rank != "" and (1 <= rank <= 55 or 81 <= rank <= 136):
            return 'white'
        return 'black'

    def add_row(x_vals, y, away_val, away_rank, metric_name, home_rank, home_val):
        # Helper to choose text color based on rank

        # Away value
        if away_val != "":
            ax.text(x_vals[0], y, f"{away_val:.2f}", ha='center', fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='green', alpha=0))

        # Away rank box
        if away_rank != "":
            fixed_width_text(
                ax, x_vals[1], y+0.007, f"{away_rank}", width=0.06, height=0.04,
                facecolor=rank_to_color(away_rank), alpha=alpha_val,
                fontsize=16, fontweight='bold', color=rank_text_color(away_rank)
            )

        # Metric name
        if metric_name != "":
            ax.text(x_vals[2], y, metric_name, ha='center', fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='green', alpha=0))

        # Home rank box
        if home_rank != "":
            fixed_width_text(
                ax, x_vals[3], y+0.007, f"{home_rank}", width=0.06, height=0.04,
                facecolor=rank_to_color(home_rank), alpha=alpha_val,
                fontsize=16, fontweight='bold', color=rank_text_color(home_rank)
            )

        # Home value
        if home_val != "":
            ax.text(x_vals[4], y, f"{home_val:.2f}", ha='center', fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='green', alpha=0))

    alpha_val = 0.9

    # Header
    ax.text(0.5, 0.528, f"{away_team} OFF vs {home_team} DEF",
            ha='center', fontsize=16, fontweight='bold',
            bbox=dict(facecolor='green', alpha=0))
    ax.hlines(y=0.518, xmin=0.29, xmax=0.71, colors='black', linewidth=1)

    # X positions for the 5 columns
    x_cols = [0.31, 0.378, 0.5, 0.622, 0.69]

    # Away OFF vs Home DEF
    add_row(x_cols, 0.49, away_off_exp, away_off_exp_rank, "EXPLOSIVENESS", home_def_exp_rank, home_def_exp)
    add_row(x_cols, 0.45, away_off_sr, away_off_sr_rank, "SUCCESS RATE", home_def_sr_rank, home_def_sr)
    add_row(x_cols, 0.41, away_off_rus, away_off_rus_rank, "RUSHING PPA", home_def_rus_rank, home_def_rus)
    add_row(x_cols, 0.37, away_off_pas, away_off_pas_rank, "PASSING PPA", home_def_pas_rank, home_def_pas)
    add_row(x_cols, 0.33, away_off_ppa, away_off_ppa_rank, "TOTAL PPA", home_def_ppa_rank, home_def_ppa)
    add_row(x_cols, 0.29, away_off_ppo, away_off_ppo_rank, "POINTS PER OPP", home_def_ppo_rank, home_def_ppo)

    # Header for Away DEF vs Home OFF
    ax.text(0.5, 0.248, f"{away_team} DEF vs {home_team} OFF",
            ha='center', fontsize=16, fontweight='bold', bbox=dict(facecolor='green', alpha=0))
    ax.hlines(y=0.238, xmin=0.29, xmax=0.71, colors='black', linewidth=1)

    # Away DEF vs Home OFF
    add_row(x_cols, 0.21, away_def_exp, away_def_exp_rank, "EXPLOSIVENESS", home_off_exp_rank, home_off_exp)
    add_row(x_cols, 0.17, away_def_sr, away_def_sr_rank, "SUCCESS RATE", home_off_sr_rank, home_off_sr)
    add_row(x_cols, 0.13, away_def_rus, away_def_rus_rank, "RUSHING PPA", home_off_rus_rank, home_off_rus)
    add_row(x_cols, 0.09, away_def_pas, away_def_pas_rank, "PASSING PPA", home_off_pas_rank, home_off_pas)
    add_row(x_cols, 0.05, away_def_ppa, away_def_ppa_rank, "TOTAL PPA", home_off_ppa_rank, home_off_ppa)
    add_row(x_cols, 0.01, away_def_ppo, away_def_ppo_rank, "POINTS PER OPP", home_off_ppo_rank, home_off_ppo)
    add_row(x_cols, -0.03, "", "", "@PEARatings", "", "")

    home_wins = get_column_value(all_data, home_team, 'wins')
    home_losses = get_column_value(all_data, home_team, 'losses')
    home_conf_wins = get_column_value(all_data, home_team, 'conference_wins')
    home_conf_losses = get_column_value(all_data, home_team, 'conference_losses')
    away_wins = get_column_value(all_data, away_team, 'wins')
    away_losses = get_column_value(all_data, away_team, 'losses')
    away_conf_wins = get_column_value(all_data, away_team, 'conference_wins')
    away_conf_losses = get_column_value(all_data, away_team, 'conference_losses')
    away_off_dq, away_off_dq_rank = get_value_and_rank(all_data, away_team, 'adj_offense_drive_quality')
    away_def_dq, away_def_dq_rank = get_value_and_rank(all_data, away_team, 'adj_defense_drive_quality', False)
    home_off_dq, home_off_dq_rank = get_value_and_rank(all_data, home_team, 'adj_offense_drive_quality')
    home_def_dq, home_def_dq_rank = get_value_and_rank(all_data, home_team, 'adj_defense_drive_quality', False)
    away_off_fp, away_off_fp_rank = get_value_and_rank(all_data, away_team, 'Offense_fieldPosition_averageStart', False)
    away_def_fp, away_def_fp_rank = get_value_and_rank(all_data, away_team, 'Defense_fieldPosition_averageStart')
    home_off_fp, home_off_fp_rank = get_value_and_rank(all_data, home_team, 'Offense_fieldPosition_averageStart', False)
    home_def_fp, home_def_fp_rank = get_value_and_rank(all_data, home_team, 'Defense_fieldPosition_averageStart')
    away_md, away_md_rank = get_value_and_rank(all_data, away_team, 'most_deserving_wins')
    home_md, home_md_rank = get_value_and_rank(all_data, home_team, 'most_deserving_wins')
    away_sos, away_sos_rank = get_value_and_rank(all_data, away_team, 'avg_expected_wins', False)
    home_sos, home_sos_rank = get_value_and_rank(all_data, home_team, 'avg_expected_wins', False)
    away_mov, away_mov_rank = get_value_and_rank(all_data, away_team, 'RTP')
    home_mov, home_mov_rank = get_value_and_rank(all_data, home_team, 'RTP')

    ax.text(0.01, 0.53, f"{away_wins}-{away_losses} ({away_conf_wins}-{away_conf_losses})", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.01, 0.49, f"RATING", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.478, xmin=0.01, xmax=0.26, colors='black', linewidth=1)
    ax.text(0.19, 0.49, f"{away_pr}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.49+0.007, f"{away_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_rank))
    
    ax.text(0.08, 0.45, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.45, f"{away_offense}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.45+0.007, f"{away_offense_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_offense_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_offense_rank))

    ax.text(0.08, 0.41, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.41, f"{away_defense}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.41+0.007, f"{away_defense_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_defense_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_defense_rank))

    ax.text(0.01, 0.37, f"DRIVE QUALITY", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.01, xmax=0.26, colors='black', linewidth=1)

    ax.text(0.08, 0.33, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.33, f"{away_off_dq}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.33+0.007, f"{away_off_dq_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_off_dq_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_off_dq_rank))
    
    ax.text(0.08, 0.29, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.29, f"{away_def_dq}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.29+0.007, f"{away_def_dq_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_def_dq_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_def_dq_rank))

    ax.text(0.01, 0.25, f"FIELD POSITION", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.238, xmin=0.01, xmax=0.26, colors='black', linewidth=1)

    ax.text(0.08, 0.21, f"OFF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.21, f"{75-away_off_fp:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.21+0.007, f"{away_off_fp_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_off_fp_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_off_fp_rank))
    
    ax.text(0.08, 0.17, f"DEF", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.17, f"{away_def_fp-75:.1f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.17+0.007, f"{away_def_fp_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_def_fp_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_def_fp_rank))
    
    ax.text(0.01, 0.13, f"RESUME", ha='left', fontsize=16, fontweight='bold')
    ax.hlines(y=0.118, xmin=0.01, xmax=0.26, colors='black', linewidth=1)

    ax.text(0.08, 0.09, f"MD", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.09, f"{away_md}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.09+0.007, f"{away_md_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_md_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_md_rank))

    ax.text(0.08, 0.05, f"SOS", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.05, f"{away_sos:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.05+0.007, f"{away_sos_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_sos_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_sos_rank))

    ax.text(0.08, 0.01, f"MOV", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.19, 0.01, f"{away_mov:.2f}", ha='right', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.23, 0.01+0.007, f"{away_mov_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(away_mov_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(away_mov_rank))

   
   
    ax.text(0.99, 0.53, f"{home_wins}-{home_losses} ({home_conf_wins}-{home_conf_losses})", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.99, 0.49, f"RATING", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.478, xmin=0.74, xmax=0.99, colors='black', linewidth=1)
    ax.text(0.81, 0.49, f"{home_pr}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.49+0.007, f"{home_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_rank))
    
    ax.text(0.92, 0.45, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.45, f"{home_offense}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.45+0.007, f"{home_offense_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_offense_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_offense_rank))

    ax.text(0.92, 0.41, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.41, f"{home_defense}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.41+0.007, f"{home_defense_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_defense_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_defense_rank))
    
    ax.text(0.99, 0.37, f"DRIVE QUALITY", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.358, xmin=0.74, xmax=0.99, colors='black', linewidth=1)

    ax.text(0.92, 0.33, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.33, f"{home_off_dq}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.33+0.007, f"{home_off_dq_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_off_dq_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_off_dq_rank))
    
    ax.text(0.92, 0.29, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.29, f"{home_def_dq}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.29+0.007, f"{home_def_dq_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_def_dq_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_def_dq_rank))

    ax.text(0.99, 0.25, f"FIELD POSITION", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.238, xmin=0.74, xmax=0.99, colors='black', linewidth=1)

    ax.text(0.92, 0.21, f"OFF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.21, f"{75-home_off_fp:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.21+0.007, f"{home_off_fp_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_off_fp_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_off_fp_rank))

    ax.text(0.92, 0.17, f"DEF", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.17, f"{home_def_fp-75:.1f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.17+0.007, f"{home_def_fp_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_def_fp_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_def_fp_rank))
    
    ax.text(0.99, 0.13, f"RESUME", ha='right', fontsize=16, fontweight='bold')
    ax.hlines(y=0.118, xmin=0.74, xmax=0.99, colors='black', linewidth=1)

    ax.text(0.92, 0.09, f"MD", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.09, f"{home_md}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.09+0.007, f"{home_md_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_md_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_md_rank))

    ax.text(0.92, 0.05, f"SOS", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.05, f"{home_sos:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.05+0.007, f"{home_sos_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_sos_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_sos_rank))

    ax.text(0.92, 0.01, f"MOV", ha='right', fontsize=16, fontweight='bold')
    ax.text(0.81, 0.01, f"{home_mov:.2f}", ha='left', fontsize=16, fontweight='bold')
    fixed_width_text(ax, 0.77, 0.01+0.007, f"{home_mov_rank}", width=0.06, height=0.04,
                            facecolor=rank_to_color(home_mov_rank), alpha=alpha_val,
                            fontsize=16, fontweight='bold', color=rank_text_color(home_mov_rank))
    
    return fig

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 1

st.title(f"{current_year} CFB PEAR")
st.logo("./PEAR/pear_logo.jpg", size = 'large')

st.divider()

st.markdown(f'<h2 id="fbs-power-ratings">FBS Power Ratings</h2>', unsafe_allow_html=True)
if current_week == 1:
    all_data['Rating'] = all_data['power_rating']
    all_data['Team'] = all_data['team']
    all_data.index = all_data.index + 1
    with st.container(border=True, height=440):
        st.dataframe(all_data[['Team', 'Rating', 'SOS', 'win_total']], width='stretch')
        # st.dataframe(all_data[['Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'ST', 'PBR', 'DCE', 'DDE', 'CONF']], width='stretch')
    st.caption("MD - Most Deserving (PEAR's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
    # , MD - Most Deserving Rankings
else:
    all_data['OFF'] = all_data['offensive_rank']
    all_data['DEF'] = all_data['defensive_rank']
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
        st.dataframe(all_data[['Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'PBR', 'DCE', 'DDE', 'CONF']], width='stretch')
    st.caption("MD - Most Deserving (PEAR's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
    # , MD - Most Deserving Rankings

st.divider()

col1, col2 = st.columns(2)

# --- Column 1: Game Images ---
with col1:
    folder_path = f'./PEAR/PEAR Football/y2025/Visuals/week_{current_week}/Games'
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    selected_file = st.selectbox("Select a game image:", file_list, key="game_image")
    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        st.image(file_path, caption=selected_file, width='stretch')

# --- Column 2: Stat Profiles ---
with col2:
    folder_path = f'./PEAR/PEAR Football/y2025/Visuals/week_{current_week}/Stat Profiles'
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    selected_file = st.selectbox("Select a stat profile:", file_list, key="stat_profile")
    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        st.image(file_path, caption=selected_file, width='stretch')

st.divider()

col1, col2 = st.columns(2)
spreads['Vegas'] = spreads['formattedSpread']
spreads['PEAR_raw'] = spreads['pr_spread']
spreads.index = spreads.index+1
with col1:
    st.markdown(f'<h2 id="fbs-power-ratings">Week {current_week} Spreads</h2>', unsafe_allow_html=True)
    with st.container(border=True, height=440):
        st.dataframe(spreads[['home_team', 'away_team', 'GQI', 'PEAR', 'Vegas', 'difference', 'PEAR_raw']].dropna(), width='stretch')

with col2:
    st.sidebar.markdown(f"[Calculate {current_year} Spread](#calculate-spread-between-any-two-teams)", unsafe_allow_html=True)
    st.sidebar.markdown(f"[{current_year} Power Ratings](#fbs-power-ratings)", unsafe_allow_html=True)
    st.markdown(f'<h2 id="calculate-spread-between-any-two-teams">Calculate Spread Between Any Two Teams</h2>', unsafe_allow_html=True)
    with st.form(key='calculate_spread'):
        away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(team_data['team'])))
        home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(team_data['team'])))
        neutrality = st.radio("Game Location", ["Neutral Field", "On Campus"])
        spread_button = st.form_submit_button("Calculate Spread")
        if spread_button:
            neutrality = True if neutrality == "Neutral Field" else False
            fig = plot_matchup_new(all_data, logos, away_team, home_team, neutrality, current_year, current_week)
            st.pyplot(fig)

st.divider()

team_data = pd.read_csv("./PEAR/PEAR Football/normalized_power_rating_across_years.csv")
st.sidebar.markdown(f"[Year Normalized Ratings](#year-normalized-ratings)", unsafe_allow_html=True)
st.sidebar.markdown(f"[Team Specific Stats](#view-a-specific-teams-stats)", unsafe_allow_html=True)

years = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014]
for year in years:
    st.sidebar.markdown(f"[{year} Ratings](#{year}-ratings)", unsafe_allow_html=True)

team_data.index += 1
team_data['Team'] = team_data['team']
team_data['Season'] = team_data['season'].astype(str)
team_data['Normalized Rating'] = team_data['norm_pr']

st.title("CFB Ratings Archive")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'<h2 id="year-normalized-ratings">Year Normalized Ratings</h2>', unsafe_allow_html=True)
    with st.container(border=True, height=440):
        st.dataframe(team_data[['Team', 'Normalized Rating', 'Season']], width='stretch')

with col2:
    st.markdown(f'<h2 id="view-a-specific-teams-stats">View A Specific Teams Stats</h2>', unsafe_allow_html=True)
    with st.form(key='filter_team'):
        team = st.selectbox("Team Filter", ["Select Team"] + list(sorted(team_data['team'].unique())))
        filter_button = st.form_submit_button("Filter Team")
        if filter_button:
            st.dataframe(teams_yearly_stats(team, team_data), width='stretch')

st.divider()

# --- Now render in pairs ---
# Year : Week mapping since your weeks vary
year_week_map = {
    2024: 17, 2023: 15,
    2022: 16, 2021: 16,
    2020: 17, 2019: 17,
    2018: 16, 2017: 16,
    2016: 16, 2015: 16,
    2014: 17
}

years = list(year_week_map.keys())

# Loop through pairs of years
for i in range(0, len(years), 2):
    col1, col2 = st.columns(2)

    with col1:
        render_year(years[i], year_week_map[years[i]], col1)

    if i + 1 < len(years):   # make sure we don’t go out of range
        with col2:
            render_year(years[i+1], year_week_map[years[i+1]], col2)

    st.divider()