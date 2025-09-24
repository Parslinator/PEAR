# temp_refactored.py
import os
import math
import datetime
import warnings
from io import BytesIO
from base64 import b64decode

import numpy as np
import pandas as pd
import pytz
from scipy.stats import zscore
from scipy.optimize import minimize, differential_evolution
from tqdm import tqdm

# plotting libs (kept as imported in original)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import gridspec
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import requests
from PIL import Image, ImageGrab
import PIL
import seaborn as sns

# cfbd client
import cfbd

# suppress warnings and set seed
warnings.filterwarnings("ignore")
np.random.seed(42)
checkmark_font = fm.FontProperties(family='DejaVu Sans')
GLOBAL_HFA = 3

# ---------------------------
# Configuration & API clients
# ---------------------------
configuration = cfbd.Configuration(
    access_token='7vGedNNOrnl0NGcSvt92FcVahY602p7IroVBlCA1Tt+WI/dCwtT7Gj5VzmaHrrxS'
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

# ---------------------------
# Utility helpers
# ---------------------------
def prob_win_at_least_x(team, wins_needed, uncompleted_games):
    """
    Calculate the probability a team wins at least `wins_needed` games
    given their uncompleted games and PEAR win probabilities.
    
    Args:
        team (str): Team name
        wins_needed (int): Minimum number of wins to calculate probability for
        uncompleted_games (DataFrame): DataFrame with columns 'home_team', 'away_team', 'PEAR_win_prob'
    
    Returns:
        float: Probability of winning at least `wins_needed` games
    """
    # Filter for this team's remaining games
    team_games = uncompleted_games[
        (uncompleted_games['home_team'] == team) |
        (uncompleted_games['away_team'] == team)
    ].copy()
    
    # Probability of winning each game from the team's perspective
    probs = []
    for _, row in team_games.iterrows():
        if row['home_team'] == team:
            p = row['PEAR_win_prob']
        else:
            p = 1 - row['PEAR_win_prob']
        probs.append(p)
    
    n = len(probs)
    if n == 0:
        # No remaining games: team wins 0 games from here
        return 1.0 if wins_needed <= 0 else 0.0

    # Dynamic programming: dp[k] = probability of winning exactly k games
    dp = np.zeros(n + 1)
    dp[0] = 1.0  # probability of 0 wins initially

    for p in probs:
        new_dp = np.zeros(n + 1)
        for k in range(n):
            new_dp[k] += dp[k] * (1 - p)  # lose this game
            new_dp[k + 1] += dp[k] * p    # win this game
        dp = new_dp

    # Probability of winning at least `wins_needed` games
    wins_needed = max(0, min(wins_needed, n))  # clamp to [0, n]
    return dp[wins_needed:].sum()

def prob_win_exactly_x(team, wins_needed, uncompleted_games):
    """
    Calculate the probability a team wins exactly `wins_needed` games
    given their uncompleted games and PEAR win probabilities.
    
    Args:
        team (str): Team name
        wins_needed (int): Exact number of wins to calculate probability for
        uncompleted_games (DataFrame): DataFrame with columns 'home_team', 'away_team', 'PEAR_win_prob'
    
    Returns:
        float: Probability of winning exactly `wins_needed` games
    """
    # Filter for this team's remaining games
    team_games = uncompleted_games[
        (uncompleted_games['home_team'] == team) |
        (uncompleted_games['away_team'] == team)
    ].copy()
    
    # Probability of winning each game from the team's perspective
    probs = []
    for _, row in team_games.iterrows():
        p = row['PEAR_win_prob'] if row['home_team'] == team else 1 - row['PEAR_win_prob']
        probs.append(p)
    
    n = len(probs)
    if n == 0:
        return 1.0 if wins_needed == 0 else 0.0

    # Dynamic programming: dp[k] = probability of winning exactly k games
    dp = np.zeros(n + 1)
    dp[0] = 1.0

    for p in probs:
        new_dp = np.zeros(n + 1)
        for k in range(n):
            new_dp[k] += dp[k] * (1 - p)
            new_dp[k + 1] += dp[k] * p
        dp = new_dp

    # Clamp wins_needed to [0, n]
    wins_needed = max(0, min(wins_needed, n))
    return dp[wins_needed]


def team_exact_win_probs(team, current_wins, uncompleted_games):
    """
    Returns a dictionary mapping exact wins to probabilities for all possible outcomes.
    
    Args:
        team (str): Team name
        uncompleted_games (DataFrame): DataFrame with 'home_team', 'away_team', 'PEAR_win_prob'
    
    Returns:
        dict: {wins: probability of winning exactly that many games}
    """
    team_games = uncompleted_games[
        (uncompleted_games['home_team'] == team) |
        (uncompleted_games['away_team'] == team)
    ].copy()
    n = len(team_games)
    
    probs = {}
    for wins_needed in range(0, n + 1):
        probs[wins_needed+current_wins] = prob_win_exactly_x(team, wins_needed, uncompleted_games)
    
    return probs

def team_win_probs(team, current_wins, uncompleted_games, max_wins):
    """
    Returns probability a team wins at least X total games, 
    starting from current_wins up to max_wins.
    """
    results = {}
    remaining_games = uncompleted_games[
        (uncompleted_games['home_team'] == team) |
        (uncompleted_games['away_team'] == team)
    ].copy()
    
    for wins_needed in range(current_wins, len(remaining_games) + current_wins + 1):
        # Calculate number of additional wins needed in remaining games
        additional_wins_needed = wins_needed - current_wins
        prob = prob_win_at_least_x(team, additional_wins_needed, remaining_games)
        results[wins_needed] = prob
        
    return results


def format_prob(value):
    """
    Format a probability for display.
    Args:
        value (float): probability between 0 and 1
    Returns:
        str: formatted string like '23%', '>99%', '<1%', or '' if 0
    """
    if value == 0:
        return ""
    elif value == 1.0:
        return "100%"
    elif value >= 0.99:
        return ">99%"
    elif value <= 0.01:
        return "<1%"
    else:
        return f"{int(round(value * 100))}%"
    
def transform_schedule(team_schedule: pd.DataFrame, team_name: str) -> pd.DataFrame:
    records = []

    for _, row in team_schedule.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_points = row["home_points"]
        away_points = row["away_points"]
        neutral = row["neutral"]
        win_prob_home = row["PEAR_win_prob"]
        week = row["week"]
        GQI = row['GQI']
        pear = row['PEAR']
        conf_game = row['conference_game']

        # Skip games that don't involve the target team
        if team_name not in [home_team, away_team]:
            continue

        # completed flag
        completed = not (pd.isna(home_points) or pd.isna(away_points))

        if completed:
            winning_score = max(home_points, away_points)
            losing_score = min(home_points, away_points)
            score_str = f"{int(winning_score)}-{int(losing_score)}"

            if team_name == home_team:
                win = home_points > away_points
            else:
                win = away_points > home_points
        else:
            score_str = None
            win = None

        # Location + win prob
        if neutral:
            location = "(N)"
            team_win_prob = win_prob_home if team_name == home_team else 1 - win_prob_home
        else:
            if team_name == home_team:
                location = ""
                team_win_prob = win_prob_home
            else:
                location = "AT"
                team_win_prob = 1 - win_prob_home

        records.append({
            "team": team_name,
            "opponent": away_team if team_name == home_team else home_team,
            "week": week,
            "score": score_str,
            "completed": completed,
            "win": win,
            "location": location,
            "conference_game": conf_game,
            "team_win_prob": team_win_prob,
            "PEAR":pear,
            "GQI":GQI
        })

    return pd.DataFrame(records)

def get_team_column_value(df, team, column):
    """
    Returns the value of `column` for the row where df['team'] == team.
    Returns '' if the team does not exist.
    """
    match = df[df['team'] == team]
    if not match.empty:
        return match.iloc[0][column]
    return ""

def average_team_distribution(num_simulations, schedules, average, team_name):
    # Precompute opponent probabilities
    is_home = schedules['home_team'] == team_name
    win_probs = np.where(
        is_home,
        schedules['away_pr'].apply(lambda opp_pr: PEAR_Win_Prob(average, opp_pr)),
        100 - schedules['home_pr'].apply(lambda opp_pr: PEAR_Win_Prob(opp_pr, average))
    )  # <-- remove .to_numpy()

    games_played = len(schedules)

    # Monte Carlo simulation
    random_matrix = np.random.rand(num_simulations, games_played) * 100
    wins_matrix = random_matrix < win_probs  # True = win, False = loss
    win_counts = wins_matrix.sum(axis=1)
    loss_counts = games_played - win_counts

    # Adjust for 10 or 11 game schedules
    if games_played == 11:
        win_counts = win_counts + 0.948
    elif games_played == 10:
        win_counts = win_counts + 2 * 0.948

    avg_wins = np.mean(win_counts)
    avg_losses = np.mean(loss_counts)
    projected_wins = np.bincount(win_counts.astype(int)).argmax()
    projected_losses = np.bincount(loss_counts.astype(int)).argmax()

    # Compute win thresholds
    win_distribution = np.bincount(win_counts.astype(int), minlength=13) / num_simulations
    win_thresholds = pd.DataFrame([{
        **{f'win_{i}': win_distribution[i] for i in range(13)},
        'WIN6%': win_distribution[6:13].sum(),
        'expected_wins': avg_wins,
        'expected_losses': avg_losses,
        'projected_wins': projected_wins,
        'projected_losses': projected_losses
    }])

    return win_thresholds

def simulate_game_known(home_team, away_team, home_win_prob):
    random_outcome = np.random.random() * 100
    return (home_team, away_team) if random_outcome < home_win_prob else (away_team, home_team)

def simulate_season_known(schedules, team_data):
    teams = team_data['team'].unique()
    team_wins = dict.fromkeys(teams, 0)
    team_losses = dict.fromkeys(teams, 0)

    for _, game in schedules.iterrows():
        winner, loser = simulate_game_known(game['home_team'], game['away_team'], game['PEAR_win_prob'])
        if winner in team_wins:
            team_wins[winner] += 1
        if loser in team_losses:
            team_losses[loser] += 1

    return team_wins, team_losses

def monte_carlo_simulation_known(num_simulations, schedules, team_data):
    return zip(*[simulate_season_known(schedules, team_data) for _ in range(num_simulations)])

def analyze_simulation_known(win_results, loss_results, schedules, records):
    win_df = pd.DataFrame(win_results)
    loss_df = pd.DataFrame(loss_results)

    for team in win_df.columns:
        rec = records[records['team'] == team].iloc[0]
        win_df[team] += rec['wins']
        loss_df[team] += rec['losses']
        total_games = rec['games_played'] + schedules[(schedules['home_team'] == team) | (schedules['away_team'] == team)].shape[0]

        if total_games == 11:
            win_df[team] += 1
        elif total_games == 10:
            win_df[team] += 2

    avg_wins = win_df.mean()
    avg_losses = loss_df.mean()
    most_common_wins = win_df.mode().iloc[0]
    most_common_losses = loss_df.mode().iloc[0]

    most_common_records = pd.DataFrame({'Wins': most_common_wins, 'Losses': most_common_losses})

    win_thresholds = pd.DataFrame({
        f'win_{w}': (win_df == w).mean() for w in range(13)
    })

    win_thresholds['WIN6%'] = (win_df >= 6).mean()
    win_thresholds.insert(0, 'team', win_df.columns)
    win_thresholds.reset_index(drop=True, inplace=True)

    win_thresholds['expected_wins'] = avg_wins.values
    win_thresholds['expected_loss'] = avg_losses.values
    win_thresholds['projected_wins'] = most_common_records['Wins'].values
    win_thresholds['projected_losses'] = most_common_records['Losses'].values

    return win_thresholds

def create_conference_projection(all_data, uncompleted_conference_games):
    teams = pd.unique(
        uncompleted_conference_games[['home_team', 'away_team']].values.ravel()
    )
    
    rows = []
    all_columns = set()  # to track all columns across teams
    
    for team in teams:
        team_row = {'team': team}
        team_games = uncompleted_conference_games[
            (uncompleted_conference_games['home_team'] == team) |
            (uncompleted_conference_games['away_team'] == team)
        ]
        conf_game_wins = all_data.loc[all_data['team'] == team, 'conference_wins'].values[0]
        conf_game_losses = all_data.loc[all_data['team'] == team, 'conference_losses'].values[0]
        n_remaining = len(team_games)
        
        # Probabilities for at least 0 to n_remaining games (added to current wins)
        for i in range(n_remaining + 1):
            col_name = f'win_{conf_game_wins + i}'
            prob = prob_win_at_least_x(team, i, uncompleted_conference_games)
            team_row[col_name] = prob
            all_columns.add(col_name)
        
        # Expected wins = current wins + expected from remaining games
        expected_wins = conf_game_wins + sum(
            row['PEAR_win_prob'] if row['home_team'] == team else 1 - row['PEAR_win_prob']
            for _, row in team_games.iterrows()
        )
        team_row['expected_wins'] = expected_wins
        team_row['expected_loss'] = n_remaining+conf_game_losses+conf_game_wins - expected_wins
        all_columns.add('expected_wins')
        all_columns.add('expected_loss')
        
        rows.append(team_row)
    
    # Create DataFrame (some NaNs will appear for teams with fewer columns)
    df = pd.DataFrame(rows)
    cols = ['team'] + sorted(c for c in all_columns if c.startswith('win_')) + ['expected_wins', 'expected_loss']
    df = df[cols].fillna(0)
    df = pd.merge(df, all_data[['team', 'conference']], how='left', on='team')
    return df

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dict to single-level dict with joined keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def date_sort(game):
    return game['start_date']

def PEAR_Win_Prob(home_power_rating, away_power_rating):
    return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

def standardize_team_names(df):
    """Standardize KFord team names to your map — modifies team column in place."""
    name_mapping = {
        'Appalachian State': 'App State',
        'FIU': 'Florida International',
        'Hawaii': "Hawai'i",
        'Miami (FL)': 'Miami',
        'Miami (Ohio)': 'Miami (OH)',
        'NIU': 'Northern Illinois',
        'Pitt': 'Pittsburgh',
        'San Jose State': 'San José State',
        'ULM': 'UL Monroe',
        'UMass': 'Massachusetts',
        'USF': 'South Florida',
        'WKU': 'Western Kentucky'
    }
    df['team'] = df['team'].replace(name_mapping)
    return df

# ---------------------------
# Fetch wrappers (API -> DataFrame)
# ---------------------------
def fetch_calendar(year):
    week_start_list = [*games_api.get_calendar(year=year)]
    calendar_dict = [dict(
        first_game_start=c.first_game_start,
        last_game_start=c.last_game_start,
        season=c.season,
        season_type=c.season_type,
        week=c.week
    ) for c in week_start_list]
    df = pd.DataFrame(calendar_dict)
    df['first_game_start'] = pd.to_datetime(df['first_game_start'])
    df['last_game_start'] = pd.to_datetime(df['last_game_start'])
    return df

def fetch_elo(year, week=None, postseason=False):
    if postseason:
        elo_list = [*ratings_api.get_elo(year=year)]
    else:
        elo_list = [*ratings_api.get_elo(year=year, week=week)]
    df = pd.DataFrame([dict(team=e.team, elo=e.elo) for e in elo_list])
    return df

def fetch_records(year):
    records_list = [*games_api.get_records(year=year)]
    records_dict = [dict(
        team=r.team,
        games_played=r.total.games,
        wins=r.total.wins,
        losses=r.total.losses,
        conference_games=r.conference_games.games,
        conference_wins=r.conference_games.wins,
        conference_losses=r.conference_games.losses
    ) for r in records_list]
    return pd.DataFrame(records_dict)

def fetch_drives(year, start_week, end_week_exclusive):
    drives_list = []
    for this_week in range(start_week, end_week_exclusive):
        response = drives_api.get_drives(year=year, week=this_week)
        drives_list = [*drives_list, *response]
    drives_dict = [dict(
        offense=g.offense,
        defense=g.defense,
        drive_number=g.drive_number,
        scoring=g.scoring,
        start_period=g.start_period,
        start_yardline=g.start_yardline,
        start_yards_to_goal=g.start_yards_to_goal,
        end_period=g.end_period,
        end_yardline=g.end_yardline,
        end_yards_to_goal=g.end_yards_to_goal,
        plays=g.plays,
        yards=g.yards,
        drive_result=g.drive_result,
        is_home_offense=g.is_home_offense,
        start_offense_score=g.start_offense_score,
        start_defense_score=g.start_defense_score,
        end_offense_score=g.end_offense_score,
        end_defense_score=g.end_defense_score
    ) for g in drives_list]
    return pd.DataFrame(drives_dict)

def fetch_team_fpi(year):
    fpi_list = [*ratings_api.get_fpi(year=year)]
    fpi_dict = [dict(
        team=f.team,
        fpi=f.fpi,
        fpi_rank=f.resume_ranks.fpi,
        fpi_sor=f.resume_ranks.strength_of_record,
        fpi_sos=f.resume_ranks.strength_of_schedule,
        def_eff=f.efficiencies.defense,
        off_eff=f.efficiencies.offense,
        special_eff=f.efficiencies.special_teams
    ) for f in fpi_list]
    return pd.DataFrame(fpi_dict)

def fetch_team_sp(year):
    sp_list = ratings_api.get_sp(year=year)
    sp_dict = [
        dict(
            team=t.team,
            ranking=t.ranking,
            sp_rating=t.rating,
            sp_offense_rating=t.offense.rating if t.offense else None,
            sp_defense_rating=t.defense.rating if t.defense else None,
            sp_special_teams_rating=t.special_teams.rating if t.special_teams else None,
            offense_rank=t.offense.ranking if t.offense else None,
            defense_rank=t.defense.ranking if t.defense else None
        ) for t in sp_list
    ]
    return pd.DataFrame(sp_dict).dropna(subset=["team"])

def fetch_team_info():
    team_info_list = [*teams_api.get_fbs_teams()]
    return pd.DataFrame([dict(team=t.school, conference=t.conference) for t in team_info_list])

def fetch_logos_info():
    logos_info_list = [*teams_api.get_teams()]
    logos_info_dict = [dict(team=l.school, color=l.color, alt_color=l.alternate_color, logo=l.logos, classification = l.classification) for l in logos_info_list]
    logos_info = pd.DataFrame(logos_info_dict)
    return logos_info.dropna(subset=['logo', 'color'])

def fetch_advanced_season_metrics(year):
    resp = [*advanced_instance.get_advanced_season_stats(year=year)]
    rows = []
    for item in resp:
        data = item.to_dict() if hasattr(item, 'to_dict') else vars(item)
        offense_stats = flatten_dict(data['offense'], parent_key='Offense')
        defense_stats = flatten_dict(data['defense'], parent_key='Defense')
        combined = {'team': data['team'], **offense_stats, **defense_stats}
        rows.append(combined)
    df = pd.DataFrame(rows)
    columns_to_keep = ['team', 'Offense_explosiveness', 'Defense_explosiveness', 'Offense_ppa', 'Defense_ppa', 'Defense_havoc_total',
                       'Offense_successRate','Defense_successRate','Offense_powerSuccess','Defense_powerSuccess',
                       'Offense_stuffRate','Defense_stuffRate','Offense_pointsPerOpportunity','Defense_pointsPerOpportunity',
                       'Offense_fieldPosition_averagePredictedPoints','Defense_fieldPosition_averagePredictedPoints',
                       'Offense_fieldPosition_averageStart','Defense_fieldPosition_averageStart']
    return df[columns_to_keep].copy()

def fetch_advanced_game_metrics(year):
    resp = [*advanced_instance.get_advanced_game_stats(year=year)]
    rows = []
    for item in resp:
        data = item.to_dict() if hasattr(item, 'to_dict') else vars(item)
        offense_stats = flatten_dict(data['offense'], parent_key='Offense')
        defense_stats = flatten_dict(data['defense'], parent_key='Defense')
        combined = {'team': data['team'], 'opponent': data['opponent'], 'week': data['week'], **offense_stats, **defense_stats}
        rows.append(combined)
    df = pd.DataFrame(rows)
    columns_to_keep = ['team', 'opponent', 'week', 'Offense_drives', 'Offense_plays', 'Defense_drives', 'Defense_plays', 'Offense_explosiveness', 'Defense_explosiveness', 'Offense_ppa', 'Defense_ppa',
                       'Offense_successRate','Defense_successRate','Offense_powerSuccess','Defense_powerSuccess',
                       'Offense_stuffRate','Defense_stuffRate']
    return df[columns_to_keep].copy()

def fetch_ppa_metrics(year):
    resp = [*metrics_api.get_predicted_points_added_by_game(year=year)]
    rows = []
    for item in resp:
        data = item.to_dict() if hasattr(item, 'to_dict') else vars(item)
        offense_stats = flatten_dict(data['offense'], parent_key='Offense')
        defense_stats = flatten_dict(data['defense'], parent_key='Defense')
        combined = {'team': data['team'], 'opponent': data['opponent'], 'week': data['week'], **offense_stats, **defense_stats}
        rows.append(combined)
    return pd.DataFrame(rows)

def fetch_team_stats(year):
    s_list = [*advanced_instance.get_team_stats(year=year)]
    rows = [dict(team=s.team, stat_name=s.stat_name, stat_value=s.stat_value.actual_instance) for s in s_list]
    df = pd.DataFrame(rows)
    # pivot and fill missing with 0 as original
    df = df.pivot(index='team', columns='stat_name', values='stat_value').reset_index().fillna(0)
    # derived metrics from original
    df['total_turnovers'] = df['fumblesRecovered'] + df['passesIntercepted'] - df['turnovers']
    df['thirdDownConversionRate'] = round(df['thirdDownConversions'] / df['thirdDowns'],4)
    df['fourthDownConversionRate'] = round(df['fourthDownConversions'] / df['fourthDowns'],4)
    df['thirdDownConversionRate'] = df['thirdDownConversionRate'].fillna(0)
    df['fourthDownConversionRate'] = df['fourthDownConversionRate'].fillna(0)
    df['possessionTimeMinutes'] = round(df['possessionTime'] / 60,2)
    return df

def fetch_talent(years):
    talent_list = []
    for year in years:
        response = teams_api.get_talent(year=year)
        talent_list = [*talent_list, *response]
    rows = [dict(team=t.team, season=t.year, talent=t.talent) for t in talent_list]
    return pd.DataFrame(rows)

def fetch_recruiting(years):
    rec_list = []
    for year in years:
        response = recruiting_api.get_team_recruiting_rankings(year=year)
        rec_list = [*rec_list, *response]
    rows = [dict(team=r.team, year=r.year, points=r.points) for r in rec_list]
    return pd.DataFrame(rows)

def fetch_games_so_far(year, start_week, end_week_exclusive):
    games_list = []
    for week in range(start_week, end_week_exclusive):
        response = games_api.get_games(year=year, week=week, classification='fbs')
        games_list = [*games_list, *response]
    games = [dict(
        id=g.id,
        season=g.season,
        week=g.week,
        start_date=g.start_date,
        home_team=g.home_team,
        home_elo=g.home_pregame_elo,
        away_team=g.away_team,
        away_elo=g.away_pregame_elo,
        home_points=g.home_points,
        away_points=g.away_points,
        neutral=g.neutral_site
    ) for g in games_list if g.home_points is not None]
    games.sort(key=date_sort)
    return pd.DataFrame(games)

def scrape_kford_ratings():
    """Same scraper as original; returns standardized DataFrame or empty df."""
    url = "https://kfordratings.com/power"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        iframe = soup.find('iframe', {'srcdoc': True}) or soup.find('iframe')
        if not iframe or not iframe.get('srcdoc'):
            print("Could not find iframe or srcdoc for kford")
            return pd.DataFrame()
        iframe_soup = BeautifulSoup(iframe.get('srcdoc'), 'html.parser')
        table = iframe_soup.find('table')
        if not table:
            print("Could not find table in kford iframe")
            return pd.DataFrame()
        rankings = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 6:
                rank = cells[0].get_text(strip=True)
                team = cells[1].get_text(strip=True)
                rating = cells[2].get_text(strip=True)
                wins = cells[4].get_text(strip=True)
                losses = cells[5].get_text(strip=True)
                try:
                    rank_int = int(rank) if rank.isdigit() else None
                    rating_float = float(rating) if rating.replace('-', '').replace('.', '').isdigit() else None
                    wins_int = int(wins) if wins.isdigit() else None
                    losses_int = int(losses) if losses.isdigit() else None
                    if all(v is not None for v in [rank_int, rating_float, wins_int, losses_int]):
                        rankings.append({'team': team, 'kford_rank': rank_int, 'kford_rating': rating_float})
                except Exception:
                    continue
        df = pd.DataFrame(rankings)
        if not df.empty:
            df = standardize_team_names(df)
        return df
    except Exception as e:
        print(f"Error scraping kford: {e}")
        return pd.DataFrame()

# ---------------------------
# Drive summarization & adjustments
# ---------------------------
def fix_end_yards_to_goal(drives_df):
    """Fix end_yards_to_goal when not TD and value is zero (original logic)."""
    drives_df["end_yards_to_goal"] = np.where(
        (~drives_df["drive_result"].str.contains("TD")) & (drives_df["end_yards_to_goal"] == 0),
        drives_df["start_yards_to_goal"] - drives_df["yards"],
        drives_df["end_yards_to_goal"]
    )
    return drives_df

def summarize_drives_by_side(drives_df, side):
    """Return a DataFrame with team, total drives, drives_40_or_less, 3_and_outs for the side."""
    grouped = (
        drives_df.groupby(side)
        .apply(lambda g: pd.Series({
            f"{side}_total_drives": len(g),
            f"{side}_drives_40_or_less": (g["end_yards_to_goal"] <= 40).sum(),
            f"{side}_3_and_outs": (((g["plays"] <= 3) & (g["drive_result"] == "PUNT")) | (g["end_yards_to_goal"] > 75)).sum()
        }))
        .reset_index()
        .rename(columns={side: "team"})
    )
    return grouped

def compute_drive_quality(drives_df, opponent_schedule_df):
    """Compute offense/defense drive quality and opponent-adjusted drive quality."""
    drives_df = fix_end_yards_to_goal(drives_df)
    offense_stats = summarize_drives_by_side(drives_df, "offense")
    defense_stats = summarize_drives_by_side(drives_df, "defense")
    # percent and quality
    for df, prefix in [(offense_stats, "offense"), (defense_stats, "defense")]:
        df[f"{prefix}_pct_drives_40_or_less"] = (df[f"{prefix}_drives_40_or_less"] / df[f"{prefix}_total_drives"] * 100).round(1)
        df[f"{prefix}_pct_3_and_out"] = (df[f"{prefix}_3_and_outs"] / df[f"{prefix}_total_drives"] * 100).round(1)
    combined = offense_stats.merge(defense_stats, on="team", how="outer")
    combined["offense_drive_quality"] = (combined["offense_pct_drives_40_or_less"] - combined["offense_pct_3_and_out"]).round(1)
    combined["defense_drive_quality"] = (combined["defense_pct_drives_40_or_less"] - combined["defense_pct_3_and_out"]).round(1)
    for side in ["offense", "defense"]:
        combined[f"{side}_drive_quality_count"] = (combined[f"{side}_drives_40_or_less"] - combined[f"{side}_3_and_outs"])
    # Build opponent summaries and convert counts to percent
    # Build base_df indexed by team for required columns
    base_df = combined.set_index("team")[[ "offense_drive_quality_count", "offense_total_drives", "defense_drive_quality_count", "defense_total_drives" ]]
    opp_summary_rows = []
    for _, row in opponent_schedule_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in base_df.index or a not in base_df.index:
            continue
        for team, opp in [(h, a), (a, h)]:
            opp_summary_rows.append({
                "team": team,
                "avg_opp_offense_drive_quality_count": base_df.loc[opp, "offense_drive_quality_count"],
                "avg_opp_offense_total_drives": base_df.loc[opp, "offense_total_drives"],
                "avg_opp_defense_drive_quality_count": base_df.loc[opp, "defense_drive_quality_count"],
                "avg_opp_defense_total_drives": base_df.loc[opp, "defense_total_drives"]
            })
    opp_drive_summary = pd.DataFrame(opp_summary_rows)
    if opp_drive_summary.empty:
        # create columns filled with zeros if no opponents
        combined["avg_opp_offense_drive_quality_pct"] = 0.0
        combined["avg_opp_defense_drive_quality_pct"] = 0.0
    else:
        opp_drive_summary = opp_drive_summary.groupby("team").mean().reset_index()
        opp_drive_summary["avg_opp_offense_drive_quality_pct"] = (opp_drive_summary["avg_opp_offense_drive_quality_count"] / opp_drive_summary["avg_opp_offense_total_drives"] * 100)
        opp_drive_summary["avg_opp_defense_drive_quality_pct"] = (opp_drive_summary["avg_opp_defense_drive_quality_count"] / opp_drive_summary["avg_opp_defense_total_drives"] * 100)
        combined = combined.merge(opp_drive_summary[["team", "avg_opp_offense_drive_quality_pct", "avg_opp_defense_drive_quality_pct"]], on="team", how="left")
    # adjusted drive quality
    combined["adj_offense_drive_quality"] = (combined["offense_drive_quality"] - combined["avg_opp_defense_drive_quality_pct"]).round(1)
    combined["adj_defense_drive_quality"] = (combined["defense_drive_quality"] - combined["avg_opp_offense_drive_quality_pct"]).round(1)
    # select columns exactly as original's drive_quality DF
    drive_quality = combined[[
        "team", "offense_drive_quality", "defense_drive_quality",
        "avg_opp_offense_drive_quality_pct", "avg_opp_defense_drive_quality_pct",
        "adj_offense_drive_quality", "adj_defense_drive_quality"
    ]]
    return drive_quality

# ---------------------------
# Opponent adjustment for PPA & Advanced (season metrics)
# ---------------------------
def build_opponent_summary(schedule_df, base_df, metrics_map):
    """
    Generic opponent summary builder:
      - schedule_df: has columns home_team, away_team
      - base_df: index = team; columns = metrics in metrics_map.keys()
      - metrics_map: dict mapping base_df column -> output short name
    Returns df with team, avg_opp_<shortname> columns (averaged across opponents).
    """
    rows = []
    for _, row in schedule_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in base_df.index or a not in base_df.index:
            continue
        for team, opp in [(h, a), (a, h)]:
            entry = {"team": team}
            for column_key, shortname in metrics_map.items():
                entry[f"opp_{column_key}"] = base_df.loc[opp, column_key]
            rows.append(entry)
    if not rows:
        return pd.DataFrame(columns=["team"] + [f"avg_opp_{v}" for v in metrics_map.values()])
    opp_df = pd.DataFrame(rows)
    grouped = opp_df.groupby("team").mean().reset_index()
    # rename columns to avg_opp_<shortname>
    rename_map = {f"opp_{k}": f"avg_opp_{metrics_map[k]}" for k in metrics_map}
    grouped = grouped.rename(columns=rename_map)
    return grouped

def adjust_season_metrics_for_opponents(season_metrics_df, opponent_schedule_df):
    """
    Adjust season_metrics (Offense_*/Defense_* columns) for opponent tendencies.
    Keeps original column names and produces adj_* columns as in original script.
    """
    # pick team-level ppa & related columns
    team_ppa = season_metrics_df.set_index("team")[[ "Offense_ppa", "Defense_ppa",
        "Offense_pointsPerOpportunity", "Defense_pointsPerOpportunity",
        "Offense_explosiveness", "Defense_explosiveness",
        "Offense_successRate", "Defense_successRate"
    ]]
    metrics_map = {
        "Offense_ppa": "offense_ppa",
        "Defense_ppa": "defense_ppa",
        "Offense_pointsPerOpportunity": "offense_ppo",
        "Defense_pointsPerOpportunity": "defense_ppo",
        "Offense_explosiveness": "offense_exp",
        "Defense_explosiveness": "defense_exp",
        "Offense_successRate": "offense_sr",
        "Defense_successRate": "defense_sr"
    }
    opp_ppa_summary = build_opponent_summary(opponent_schedule_df, team_ppa, metrics_map)
    merged = season_metrics_df.merge(opp_ppa_summary, on="team", how="left")
    # compute adj columns preserving names
    merged["adj_offense_ppa"] = merged["Offense_ppa"] - merged["avg_opp_defense_ppa"]
    merged["adj_defense_ppa"] = merged["Defense_ppa"] - merged["avg_opp_offense_ppa"]
    merged["adj_offense_ppo"] = merged["Offense_pointsPerOpportunity"] - merged["avg_opp_defense_ppo"]
    merged["adj_defense_ppo"] = merged["Defense_pointsPerOpportunity"] - merged["avg_opp_offense_ppo"]
    merged["adj_offense_exp"] = merged["Offense_explosiveness"] - merged["avg_opp_defense_exp"]
    merged["adj_defense_exp"] = merged["Defense_explosiveness"] - merged["avg_opp_offense_exp"]
    merged["adj_offense_sr"] = merged["Offense_successRate"] - merged["avg_opp_defense_sr"]
    merged["adj_defense_sr"] = merged["Defense_successRate"] - merged["avg_opp_offense_sr"]
    return merged

# ---------------------------
# Game-level advanced metrics -> adjusted game metrics & EWMA weighting
# ---------------------------
def build_adj_game_metrics(updated_metrics_df):
    """
    From updated_metrics (game-level metrics with team/opponent/week),
    build adjusted columns: Offense - opponent Defense and Defense - opponent Offense.
    Keeps column names: produces *_adj columns (e.g. Offense_ppa_adj).
    """
    exclude_cols = ['team', 'opponent', 'week']
    # prepare team-level opponent metrics averaged across games in updated_metrics_df
    # compute per-team averages (team_metrics_avg) used to attach to opponent side; we mimic original code
    team_metrics_avg = updated_metrics_df.drop(columns=exclude_cols).groupby(updated_metrics_df['team']).mean().reset_index()
    # split offense/defense columns
    offense_cols = [c for c in team_metrics_avg.columns if c.startswith("Offense_")]
    defense_cols = [c for c in team_metrics_avg.columns if c.startswith("Defense_")]
    # opponent metrics: rename to _opp and merge by opponent
    opponent_metrics = team_metrics_avg.rename(columns={c: c + "_opp" for c in offense_cols + defense_cols})[['team'] + [c + "_opp" for c in offense_cols + defense_cols]]
    metrics_adj = updated_metrics_df.merge(opponent_metrics, left_on='opponent', right_on='team', suffixes=('', '_opp'))
    adjusted_cols = []
    # Offense - Opponent Defense
    for col in offense_cols:
        opp_def_col = col.replace("Offense", "Defense") + "_opp"
        if opp_def_col in metrics_adj.columns:
            new_col = col + "_adj"
            metrics_adj[new_col] = metrics_adj[col] - metrics_adj[opp_def_col]
            adjusted_cols.append(new_col)
    # Defense - Opponent Offense
    for col in defense_cols:
        opp_off_col = col.replace("Defense", "Offense") + "_opp"
        if opp_off_col in metrics_adj.columns:
            new_col = col + "_adj"
            metrics_adj[new_col] = metrics_adj[col] - metrics_adj[opp_off_col]
            adjusted_cols.append(new_col)
    # keep original minimal set similar to original pipeline
    keep_cols = ['team', 'opponent', 'week'] + adjusted_cols
    return metrics_adj[keep_cols].copy()

def compute_weighted_metrics(metrics_adj_df, span=5):
    """
    Compute EWMA (span) by team for all *_adj columns, and return the last EWMA per team
    as the weighted metrics DataFrame (one row per team).
    """
    adj_cols = [c for c in metrics_adj_df.columns if c.endswith('_adj')]
    weighted_metrics_list = []
    for team, df in metrics_adj_df.groupby('team'):
        df = df.sort_values('week').reset_index(drop=True)
        if df.empty:
            continue
        ewma_df = df[adj_cols].ewm(span=span, adjust=False).mean()
        last_ewma = ewma_df.iloc[-1].to_dict()
        last_ewma['team'] = team
        weighted_metrics_list.append(last_ewma)
    return pd.DataFrame(weighted_metrics_list)

# ---------------------------
# Talent & recruiting summaries
# ---------------------------
def summarize_talent_and_recruiting(talent_df, recruiting_df):
    # last three years average talent
    last_three_rows = talent_df.groupby('team').tail(3)
    avg_talent_per_team = last_three_rows.groupby('team')['talent'].mean().reset_index()
    avg_talent_per_team.columns = ['team', 'avg_talent']
    # recruiting over last 3 seasons (sum)
    last_three_rec = recruiting_df.groupby('team').tail(3)
    recruiting_per_team = last_three_rec.groupby('team')['points'].sum().reset_index()
    recruiting_per_team.columns = ['team', 'avg_points']
    recruiting_per_team['avg_points'] = recruiting_per_team['avg_points'] + 150
    return avg_talent_per_team, recruiting_per_team

# ---------------------------
# Final team_data builder (merges everything)
# ---------------------------
def build_team_data(
    current_year,
    current_week,
    opponent_schedule_df,
    updated_metrics_df,
    season_metrics_df,
    team_stats_df,
    team_info_df,
    logos_info_df,
    team_fpi_df,
    team_sp_df,
    records_df,
    weighted_metrics_df,
    talent_avg_df,
    recruiting_per_team_df,
    conference_sp_rating_df,
    kford_df,
    elo_df
):
    # Start merging following the original merge order and column names
    intermediate_1 = pd.merge(team_info_df, talent_avg_df, how='left', on='team')
    intermediate_2 = pd.merge(intermediate_1, conference_sp_rating_df, how='left', on='conference')
    intermediate_3 = pd.merge(intermediate_2, team_stats_df, how='left', on='team')
    intermediate_4 = pd.merge(intermediate_3, logos_info_df, how='left', on='team')
    intermediate_6 = pd.merge(intermediate_4, team_fpi_df, how='left', on='team')
    intermediate_7 = pd.merge(intermediate_6, records_df, how='left', on='team')
    team_data = pd.merge(intermediate_7, weighted_metrics_df, how='left', on='team')
    team_data = pd.merge(team_data, season_metrics_df, on='team', how='left')
    team_data = pd.merge(team_data, team_sp_df, on='team', how='left')
    team_data = pd.merge(team_data, kford_df, on='team', how='left')
    team_data = pd.merge(team_data, elo_df, on='team', how='left')
    # for target teams use recruiting points as avg_talent (original logic)
    target_teams = ['Air Force', 'Army', 'Navy', 'Kennesaw State', 'Jacksonville State', 'Sam Houston', 'Delaware', 'Missouri State']
    mask = team_data['team'].isin(target_teams)
    if 'avg_points' in recruiting_per_team_df.columns:
        team_data.loc[mask, 'avg_talent'] = team_data.loc[mask, 'team'].map(recruiting_per_team_df.set_index('team')['avg_points'])
    team_data = team_data.drop_duplicates(subset=["team"]).reset_index(drop=True)
    return team_data

# ---------------------------
# Off/Def totals & ranks compute
# ---------------------------
def compute_off_def_totals_and_ranks(team_data_df, season_metrics_df):
    # The original used season_metrics columns in zscore calculation and wrote the results into team_data
    off_def_metrics = season_metrics_df[[
        "team",
        "adj_offense_ppo",
        "adj_offense_ppa",
        "adj_offense_drive_quality",
        "adj_defense_ppo",
        "adj_defense_ppa",
        "adj_defense_drive_quality"
    ]].copy()
    raw_z = {
        "off_ppa": zscore(off_def_metrics["adj_offense_ppa"], nan_policy="omit"),
        "off_ppo": zscore(off_def_metrics["adj_offense_ppo"], nan_policy="omit"),
        "off_dq": zscore(off_def_metrics["adj_offense_drive_quality"], nan_policy="omit"),
        "def_ppa": zscore(off_def_metrics["adj_defense_ppa"], nan_policy="omit"),
        "def_ppo": zscore(off_def_metrics["adj_defense_ppo"], nan_policy="omit"),
        "def_dq": zscore(off_def_metrics["adj_defense_drive_quality"], nan_policy="omit"),
    }
    results_z = {
        "off_ppa": raw_z["off_ppa"],
        "off_ppo": raw_z["off_ppo"],
        "off_dq": raw_z["off_dq"],
        "def_ppa": -raw_z["def_ppa"],
        "def_ppo": -raw_z["def_ppo"],
        "def_dq": -raw_z["def_dq"],
    }
    # merge the z-scores into team_data_df by team (preserve column names)
    team_data_df = team_data_df.copy()
    team_data_df = team_data_df.merge(
        off_def_metrics[['team']].assign(
            off_ppa=list(results_z['off_ppa']),
            off_ppo=list(results_z['off_ppo']),
            off_dq=list(results_z['off_dq']),
            def_ppa=list(results_z['def_ppa']),
            def_ppo=list(results_z['def_ppo']),
            def_dq=list(results_z['def_dq'])
        ), on='team', how='left'
    )
    team_data_df["offensive_total"] = (team_data_df["off_ppa"] + team_data_df["off_ppo"] + team_data_df["off_dq"])
    team_data_df["defensive_total"] = (team_data_df["def_ppa"] + team_data_df["def_ppo"] + team_data_df["def_dq"])
    team_data_df["offensive_rank"] = team_data_df["offensive_total"].rank(ascending=False, method="dense").astype(int)
    team_data_df["defensive_rank"] = team_data_df["defensive_total"].rank(ascending=False, method="dense").astype(int)
    return team_data_df

# ---------------------------
# last_week results
# ---------------------------
def compute_last_week(team_data_df, opponent_adjustment_schedule_df, current_week):
    last_week = opponent_adjustment_schedule_df[opponent_adjustment_schedule_df['week'] == current_week-1]
    team_data_df['last_week'] = 0
    if not last_week.empty:
        # match original logic: home_results and away_results as Series with index team names
        home_results = pd.Series(np.where(last_week['home_points'] > last_week['away_points'], 1, -1), index=last_week['home_team'])
        away_results = pd.Series(np.where(last_week['away_points'] > last_week['home_points'], 1, -1), index=last_week['away_team'])
        results = pd.concat([home_results, away_results])
        team_data_df['last_week'] = team_data_df['team'].map(results).fillna(0).astype(int)
    return team_data_df

# ---------------------------
# Main pipeline (coordinates fetch -> transforms)
# ---------------------------
def modeling_data_import(current_week_override=None):
    # --- current time and calendar / year/week logic (keeps original approach) ---
    current_time = datetime.datetime.now(pytz.UTC)
    if current_time.month < 6:
        calendar_year = current_time.year - 1
    else:
        calendar_year = current_time.year
    calendar = fetch_calendar(calendar_year)
    current_year = int(calendar.loc[0, 'season'])
    first_game_start = calendar['first_game_start'].iloc[0]
    last_game_start = calendar['last_game_start'].iloc[-1]
    # determine week/postseason similar to original code
    if current_time < first_game_start:
        current_week = 1
        postseason = False
    elif current_time > last_game_start:
        current_week = calendar.iloc[-2, -1] + 1
        postseason = True
    else:
        condition_1 = (calendar['first_game_start'] <= current_time) & (calendar['last_game_start'] >= current_time)
        condition_2 = (calendar['last_game_start'].shift(1) < current_time) & (calendar['first_game_start'] > current_time)
        result = calendar[condition_1 | condition_2].reset_index(drop=True)
        if result['season_type'][0] == 'regular':
            current_week = int(result['week'][0])
            postseason = False
        else:
            current_week = int(calendar.iloc[-2, -1] + 1)
            postseason = True
    # --- FORCE same override as original file (you can remove the next line to be dynamic) ---
    if current_week_override is not None:
        current_week = current_week_override
    current_year = int(current_year)
    current_week = int(current_week)
    print(f"Current Week: {current_week}, Current Year: {current_year}")
    print("Double Check The Current Week To Make Sure It Is Correct")

    # --- Fetch high-level datasets ---
    if postseason:
        elo_ratings_df = fetch_elo(current_year, postseason=True)
    else:
        elo_ratings_df = fetch_elo(current_year, week=current_week, postseason=False)
    records_df = fetch_records(current_year)
    drives_df = fetch_drives(current_year, start_week=1, end_week_exclusive=current_week)
    season_metrics_df = fetch_advanced_season_metrics(current_year)
    game_metrics_df = fetch_advanced_game_metrics(current_year)
    ppa_df = fetch_ppa_metrics(current_year)
    merged_metrics = pd.merge(game_metrics_df, ppa_df, how='left', on=['team', 'opponent', 'week'], suffixes=('', '_ppa'))
    conference_sp_rating_df = pd.DataFrame([dict(conference=c.conference, season=c.year, sp_conf_rating=c.rating) for c in ratings_api.get_conference_sp(year=current_year)])
    team_stats_df = fetch_team_stats(current_year)
    talent_df = fetch_talent(range(current_year-3, current_year+1))
    recruiting_df = fetch_recruiting(range(current_year-3, current_year+1))
    team_info_df = fetch_team_info()
    logos_info_df = fetch_logos_info()
    team_fpi_df = fetch_team_fpi(current_year)
    team_sp_df = fetch_team_sp(current_year)
    kford_df = scrape_kford_ratings()
    opponent_adjustment_schedule_df = fetch_games_so_far(current_year, start_week=1, end_week_exclusive=current_week)
    # keep only games with elo values similar to original
    opponent_adjustment_schedule_df = opponent_adjustment_schedule_df.dropna(subset=['home_elo', 'away_elo']).reset_index(drop=True)

    # --- compute drive_quality & adjusted season metrics ---
    drive_quality_df = compute_drive_quality(drives_df, opponent_adjustment_schedule_df)
    season_metrics_adjusted_df = adjust_season_metrics_for_opponents(season_metrics_df, opponent_adjustment_schedule_df)
    # merge drive_quality into season_metrics_adjusted_df (as original)
    season_metrics_adjusted_df = season_metrics_adjusted_df.merge(drive_quality_df, on='team', how='left')

    # --- create updated_metrics (original variable name used) ---
    updated_metrics = merged_metrics[(merged_metrics['team'].isin(team_info_df['team']))].reset_index(drop=True)

    # --- game-level adjusted metrics and weighted metrics via EWMA ---
    metrics_adj_df = build_adj_game_metrics(updated_metrics)
    weighted_metrics_df = compute_weighted_metrics(metrics_adj_df, span=5)

    # --- talent and recruiting summaries ---
    avg_talent_df, recruiting_per_team_df = summarize_talent_and_recruiting(talent_df, recruiting_df)

    # --- fetch sp conf, team_fpi, records already done above ---
    # prepare conference_sp_rating_df columns: ensure naming matches original (conference, season, sp_conf_rating)
    # Build team_data via the same merge order
    team_data_df = build_team_data(
        current_year=current_year,
        current_week=current_week,
        opponent_schedule_df=opponent_adjustment_schedule_df,
        updated_metrics_df=updated_metrics,
        season_metrics_df=season_metrics_adjusted_df,
        team_stats_df=team_stats_df,
        team_info_df=team_info_df,
        logos_info_df=logos_info_df,
        team_fpi_df=team_fpi_df,
        team_sp_df=team_sp_df,
        records_df=records_df,
        weighted_metrics_df=weighted_metrics_df,
        talent_avg_df=avg_talent_df,
        recruiting_per_team_df=recruiting_per_team_df,
        conference_sp_rating_df=conference_sp_rating_df,
        kford_df=kford_df,
        elo_df=elo_ratings_df
    )

    # --- compute off/def totals & ranks and last_week result (same column names) ---
    team_data_df = compute_off_def_totals_and_ranks(team_data_df, season_metrics_adjusted_df)
    team_data_df = compute_last_week(team_data_df, opponent_adjustment_schedule_df, current_week)

    print("Data Formatting Done")
    return {
        "team_data": team_data_df,
        "opponent_adjustment_schedule": opponent_adjustment_schedule_df,
        "updated_metrics": updated_metrics,
        "season_metrics": season_metrics_adjusted_df,
        "drive_quality": drive_quality_df,
        "weighted_metrics": weighted_metrics_df,
        "elo_ratings": elo_ratings_df,
        "logos": logos_info_df,
        "records": records_df,
        "current_week": current_week,
        "current_year": current_year,
        'postseason': postseason
    }
































############################
## IN HOUSE POWER RATINGS ##
############################
# 
# Preseason priors
#
import pandas as pd
def preseason_priors(team_data, current_week):
    if current_week < 6:
        preseason = pd.read_csv("./PEAR/PEAR Football/y2025/Ratings/PEAR_week1.csv")
        preseason.loc[preseason['team'] == 'UCLA', 'power_rating'] = -8.0
        preseason.loc[preseason['team'] == 'Oregon State', 'power_rating'] = -10.0
        preseason.loc[preseason['team'] == 'Coastal Carolina', 'power_rating'] = -15.0
        preseason.loc[preseason['team'] == 'Arkansas State', 'power_rating'] = -15.0
        preseason.loc[preseason['team'] == 'Tulsa', 'power_rating'] = -17.0
        preseason.loc[preseason['team'] == 'Massachusetts', 'power_rating'] = -25.0
        preseason.loc[preseason['team'] == 'Liberty', 'power_rating'] = -10.0
        preseason.loc[preseason['team'] == 'James Madison', 'power_rating'] = 5.0
        preseason.loc[preseason['team'] == 'UAB', 'power_rating'] = -10.0
        merged = preseason.merge(
            team_data[['team', 'power_rating', 'games_played']],
            on='team',
            how='left',
            suffixes=('_pre', '_team')
        )

        # Map games_played -> (pre_weight, season_weight)
        weight_map = {
            1: (0.85, 0.15),  # current_week == 2
            2: (0.65, 0.35),  # current_week == 3
            3: (0.35, 0.65),  # current_week == 4
            4: (0.15, 0.85),  # current_week == 5
        }

        def get_weighted_power(row):
            gp = max(row['games_played'] - 0, 0)  # already decremented naturally
            pre_w, season_w = weight_map.get(gp, (0, 1))  # default: full season rating
            if pd.notna(row['power_rating_team']):
                return pre_w * row['power_rating_pre'] + season_w * row['power_rating_team']
            else:
                return row['power_rating_pre']

        merged['weighted_power'] = merged.apply(get_weighted_power, axis=1).round(1)

        team_data = team_data.rename(columns={'power_rating': 'unweighted_power'})
        team_data = team_data.merge(
            merged[['team', 'weighted_power']],
            on='team',
            how='left'
        )
        team_data['in_house'] = team_data['weighted_power']
        team_data = team_data.drop(columns=['weighted_power'])
    return team_data

import numpy as np
import pandas as pd
from scipy.stats import rankdata, zscore, spearmanr
from scipy.optimize import differential_evolution

def in_house_power_ratings(team_data, opponent_adjustment_schedule, current_week,
                           core_metrics, target_col="kford_rating",
                           fixed_scale=16, home_field_advantage=3):
    """
    Build in-house power ratings using weighted optimization of core metrics.

    Parameters
    ----------
    team_data : pd.DataFrame
        DataFrame with 'team' column, core metric columns, 'last_week', 'avg_talent',
        and a target ratings column (default = 'kford_rating').
    opponent_adjustment_schedule : pd.DataFrame
        Schedule with columns ['home_team','away_team','home_points','away_points','neutral'].
    core_metrics : list
        List of adjusted offense/defense metric column names to include.
    target_col : str, default='kford_rating'
        Column in team_data to optimize against (for rank correlation).
    fixed_scale : float, default=16
        Scaling factor applied to ratings after weighting.
    home_field_advantage : float, default=3
        Points added to home team when not neutral.

    Returns
    -------
    team_data_out : pd.DataFrame
        team_data with a new 'power_rating' column (rounded).
    opt_weights : dict
        Mapping of model feature -> optimized weight.
    diagnostics : dict
        Final optimization diagnostics including Spearman correlation.
    """
    defense_cols = [c for c in core_metrics if c.startswith('Defense')]
    offense_cols = [c for c in core_metrics if c.startswith('Offense')]

    # Step 1: Prepare modeling input
    model_input = team_data.copy()
    model_input[defense_cols] = model_input[defense_cols] * -1
    model_features = offense_cols + defense_cols + ['last_week', 'avg_talent']

    # Standardize
    z_metrics = model_input[model_features].apply(
        lambda x: zscore(x, nan_policy='omit')
    ).fillna(0)

    # Target ranks
    target_ranks = rankdata(-team_data[target_col].values, method='ordinal')

    # Step 2: Game schedule
    team_to_idx = {t: i for i, t in enumerate(team_data['team'])}
    mask = (opponent_adjustment_schedule['home_team'].isin(team_to_idx) &
            opponent_adjustment_schedule['away_team'].isin(team_to_idx))
    schedule = opponent_adjustment_schedule.loc[mask].copy()

    h_idx = schedule['home_team'].map(team_to_idx).values
    a_idx = schedule['away_team'].map(team_to_idx).values
    actual_margin = schedule['home_points'].values - schedule['away_points'].values
    hfa = np.where(schedule['neutral'] == False, home_field_advantage, 0)

    def game_abs_error(ratings):
        if len(h_idx) == 0:
            return 0
        pred_margin = ratings[h_idx] + hfa - ratings[a_idx]
        return 2 * np.mean(np.abs(pred_margin - actual_margin))

    # Step 3: Index offense/defense positions
    off_cols_idx = [i for i, c in enumerate(z_metrics.columns) if 'off' in c.lower()]
    def_cols_idx = [i for i, c in enumerate(z_metrics.columns) if 'def' in c.lower()]

    lambda_reg = 0.01
    alpha = 0.5  # balance rank vs game spread

    # Step 4: Objective function
    def objective(x):
        weights = np.array(x)
        weights = weights / weights.sum()

        # Penalties
        if weights[-1] > 0.4:  # talent cap
            return 1e6
        if np.any(weights[:-1] > 0.15):
            return 1e6

        ratings = (z_metrics.values @ weights) * fixed_scale
        model_ranks = rankdata(-ratings, method='ordinal')

        rank_loss = np.abs(model_ranks - target_ranks).mean()
        game_loss = game_abs_error(ratings)
        l1_penalty = lambda_reg * np.sum(np.abs(weights))

        # Balance offense vs defense
        off_sum = np.sum(weights[off_cols_idx])
        def_sum = np.sum(weights[def_cols_idx])
        off_def_penalty = 100 * abs(off_sum - def_sum)

        return alpha * rank_loss + (1 - alpha) * game_loss + l1_penalty + off_def_penalty

    # Step 5: Optimize
    bounds = [(0, 1)] * len(model_features)
    result = differential_evolution(objective, bounds, strategy='best1bin',
                                    maxiter=500, seed=42)
    opt_weights_arr = result.x / sum(result.x)
    opt_weights = dict(zip(model_features, opt_weights_arr))

    # Step 6: Compute ratings
    ratings = (z_metrics.values @ opt_weights_arr) * fixed_scale
    team_data_out = team_data.copy()
    team_data_out['power_rating'] = ratings.round(1)

    # Step 7: Diagnostics
    corr, pval = spearmanr(team_data_out['power_rating'], team_data_out[target_col])
    diagnostics = {
        "spearman_corr": corr,
        "p_value": pval,
        "opt_weights": opt_weights
    }
    team_data_out = preseason_priors(team_data_out, current_week)

    return team_data_out, opt_weights, diagnostics
















































################################
## BUILD POWER RATINGS SYSTEM ##
################################
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from scipy.optimize import minimize, differential_evolution
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

class MultiTargetPowerRatingSystem:
    """
    Multi-target power rating system that can optimize for multiple rating columns
    """
    
    def __init__(self, use_xgb=HAS_XGB, home_field_advantage=3.0, random_state=42):
        self.use_xgb = use_xgb
        self.hfa = home_field_advantage
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.diagnostics = {}
        
    def _clean_round(self, values, decimals=1):
        """Helper method to ensure clean rounding without floating point artifacts"""
        return np.round(values.astype(float), decimals)
    
    def _fit_regressor(self, X, y, model_name=None):
        """Fit regressor with cross-validation"""
        if self.use_xgb:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
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
        actual_margin = schedule_clean['home_points'].values - schedule_clean['away_points'].values
        hfa = np.where(schedule_clean['neutral'] == False, self.hfa, 0.0)
        
        return schedule_clean, h_idx, a_idx, actual_margin, hfa
    
    def _create_composite_target(self, team_data, target_columns, target_weights=None):
        """Create weighted composite target from multiple columns"""
        if target_weights is None:
            target_weights = [1.0 / len(target_columns)] * len(target_columns)
        
        if len(target_weights) != len(target_columns):
            raise ValueError("Number of weights must match number of target columns")
        
        # Normalize each target column to have mean=0, std=1
        normalized_targets = []
        for col in target_columns:
            target_values = team_data[col].values
            normalized = (target_values - target_values.mean()) / target_values.std()
            normalized_targets.append(normalized)
        
        # Create weighted composite
        composite_target = sum(w * target for w, target in zip(target_weights, normalized_targets))
        
        return composite_target, normalized_targets
    
    def _train_target_models(self, team_data, model_features, target_columns):
        """Train separate models for each target column"""
        X_team = team_data[model_features].values.astype(float)
        X_team_scaled = self.scaler.fit_transform(X_team)
        
        target_models = {}
        target_predictions = {}
        
        for target_col in target_columns:
            y_target = team_data[target_col].values
            model = self._fit_regressor(X_team_scaled, y_target, f'target_{target_col}')
            target_models[target_col] = model
            target_predictions[target_col] = model.predict(X_team_scaled)
            
            # Store individual correlations
            corr = spearmanr(target_predictions[target_col], y_target).correlation
            self.diagnostics[f'{target_col}_model_correlation'] = corr
        
        self.models['target_models'] = target_models
        return X_team_scaled, target_predictions
    
    def _train_margin_model(self, team_data, model_features, h_idx, a_idx, actual_margin, hfa):
        """Train game-level margin prediction model"""
        X_team_scaled = self.scaler.transform(team_data[model_features].values.astype(float))
        
        game_features = X_team_scaled[h_idx] - X_team_scaled[a_idx]
        X_games = np.column_stack([game_features, hfa])
        
        margin_model = self._fit_regressor(X_games, actual_margin, 'margin')
        self.models['margin_model'] = margin_model
        
        # Diagnostics
        margin_pred = margin_model.predict(X_games)
        self.diagnostics['margin_mae'] = mean_absolute_error(actual_margin, margin_pred)
        self.diagnostics['margin_r2'] = r2_score(actual_margin, margin_pred)
        
        return X_games
    
    def _compute_margin_ratings(self, team_data, model_features):
        """Compute margin-based ratings for each team"""
        X_team_scaled = self.scaler.transform(team_data[model_features].values.astype(float))
        margin_features = np.column_stack([X_team_scaled, np.zeros(len(X_team_scaled))])
        margin_ratings = self.models['margin_model'].predict(margin_features)
        return margin_ratings
    
    def _optimize_multi_target_ensemble(self, target_predictions, margin_ratings, team_data, 
                                       target_columns, h_idx, a_idx, actual_margin, hfa,
                                       mae_weight=0.3, correlation_weight=0.7):
        """
        Optimize ensemble considering multiple targets and game prediction accuracy
        
        Parameters:
        -----------
        mae_weight : float
            Weight given to game prediction accuracy (0-1)
        correlation_weight : float  
            Weight given to target correlations (0-1)
        """
        baseline_mae = np.mean(np.abs(actual_margin))
        n_targets = len(target_columns)
        
        def multi_objective(params):
            """
            Optimize both target weights and ensemble mixing weight
            
            params: [target_weight_1, ..., target_weight_n, ensemble_weight]
            """
            if len(params) != n_targets + 1:
                raise ValueError(f"Expected {n_targets + 1} parameters")
            
            target_weights = params[:n_targets]
            ensemble_weight = params[-1]
            
            # Normalize target weights to sum to 1
            target_weights = np.array(target_weights)
            target_weights = np.abs(target_weights) / np.sum(np.abs(target_weights))
            
            # Create composite target prediction
            composite_pred = sum(w * target_predictions[col] for w, col in zip(target_weights, target_columns))
            
            # Ensemble with margin-based ratings
            ensemble_ratings = ensemble_weight * composite_pred + (1 - ensemble_weight) * margin_ratings
            
            # Calculate correlation scores with each target
            correlation_scores = []
            for col in target_columns:
                target_values = team_data[col].values
                corr = spearmanr(ensemble_ratings, target_values).correlation
                if np.isnan(corr):
                    corr = 0.0
                correlation_scores.append(corr)
            
            # Average correlation across all targets
            avg_correlation = np.mean(correlation_scores)
            
            # Game prediction accuracy
            game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
            game_mae = np.mean(np.abs(game_pred - actual_margin))
            normalized_mae = game_mae / baseline_mae
            
            # Multi-objective loss: maximize correlation, minimize MAE
            loss = -correlation_weight * avg_correlation + mae_weight * normalized_mae
            
            return loss
        
        # Set up optimization bounds
        # Target weights can be 0-2, ensemble weight 0-1
        bounds = [(0, 2)] * n_targets + [(0, 1)]
        
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
        
        return optimal_target_weights, optimal_ensemble_weight
    
    def fit(self, team_data, schedule_df, model_features, target_columns, 
            mae_weight=0.3, correlation_weight=0.7):
        """
        Fit multi-target power rating system
        
        Parameters:
        -----------
        team_data : DataFrame
            Team statistics with 'team' column and features
        schedule_df : DataFrame  
            Game results with home_team, away_team, home_points, away_points, neutral
        model_features : list
            Column names to use as features
        target_columns : list
            List of target rating columns to optimize for
        mae_weight : float
            Weight for game prediction accuracy in optimization
        correlation_weight : float
            Weight for target correlation in optimization
        """
        # Validate inputs
        for col in target_columns:
            if col not in team_data.columns:
                raise ValueError(f"Target column '{col}' not found in team_data")
        
        # Set team as index
        team_data_indexed = team_data.set_index('team', drop=False)
        
        # Prepare game data
        schedule_clean, h_idx, a_idx, actual_margin, hfa = self._prepare_game_data(
            team_data_indexed, schedule_df
        )
        
        # Train models for each target
        X_team_scaled, target_predictions = self._train_target_models(
            team_data_indexed, model_features, target_columns
        )
        
        # Train margin model
        self._train_margin_model(
            team_data_indexed, model_features, h_idx, a_idx, actual_margin, hfa
        )
        
        # Compute margin-based ratings
        margin_ratings = self._compute_margin_ratings(team_data_indexed, model_features)
        
        # Optimize multi-target ensemble
        optimal_target_weights, optimal_ensemble_weight = self._optimize_multi_target_ensemble(
            target_predictions, margin_ratings, team_data_indexed, target_columns,
            h_idx, a_idx, actual_margin, hfa, mae_weight, correlation_weight
        )
        
        # Create final ensemble ratings
        composite_pred = sum(w * target_predictions[col] 
                           for w, col in zip(optimal_target_weights, target_columns))
        
        ensemble_ratings = (optimal_ensemble_weight * composite_pred + 
                          (1 - optimal_ensemble_weight) * margin_ratings)
        
        # Center ratings
        ensemble_ratings = ensemble_ratings - ensemble_ratings.mean()
        
        # Store results - reset index to avoid ambiguity and ensure clean rounding
        self.team_ratings = pd.DataFrame({
            'team': team_data_indexed['team'].values,
            'power_rating': self._clean_round(ensemble_ratings)
        })
        
        # Add individual component ratings for analysis - ensure all are properly rounded  
        for i, col in enumerate(target_columns):
            self.team_ratings[f'{col}_component'] = self._clean_round(target_predictions[col])
        self.team_ratings['margin_component'] = self._clean_round(margin_ratings)
        
        # Final diagnostics
        final_correlations = {}
        for col in target_columns:
            corr = spearmanr(ensemble_ratings, team_data_indexed[col].values).correlation
            final_correlations[f'final_{col}_correlation'] = corr
        
        game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
        final_mae = np.mean(np.abs(game_pred - actual_margin))
        
        self.diagnostics.update({
            'target_columns': target_columns,
            'optimal_target_weights': dict(zip(target_columns, optimal_target_weights)),
            'optimal_ensemble_weight': optimal_ensemble_weight,
            'final_game_mae': final_mae,
            'baseline_margin_mae': np.mean(np.abs(actual_margin)),
            'n_games': len(actual_margin),
            **final_correlations
        })
        
        return self
    
    def predict_game(self, home_team, away_team, neutral=False):
        """Predict margin for a specific game"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted before making predictions")
        
        team_lookup = dict(zip(self.team_ratings['team'], self.team_ratings['power_rating']))
        
        home_rating = team_lookup.get(home_team, 0)
        away_rating = team_lookup.get(away_team, 0)
        hfa = 0 if neutral else self.hfa
        spread = round(home_rating + hfa - away_rating, 1)
        if spread < 0:
            pear = f'{away_team} {spread}'
        else:
            pear = f'{home_team} -{spread}'
        
        return pear
    
    def get_rankings(self, n=25):
        """Get top N teams by power rating"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted first")
        
        return self.team_ratings.nlargest(n, 'power_rating')
    
    def print_diagnostics(self):
        """Print comprehensive model diagnostics"""
        print("Multi-Target Power Rating System Diagnostics")
        print("=" * 50)
        
        # Target information
        print(f"Target columns: {self.diagnostics['target_columns']}")
        print("\nOptimal target weights:")
        for target, weight in self.diagnostics['optimal_target_weights'].items():
            print(f"  {target}: {weight:.3f}")
        
        print(f"\nOptimal ensemble weight: {self.diagnostics['optimal_ensemble_weight']:.3f}")
        
        # Model performance
        print(f"\nGame Prediction:")
        print(f"  Final MAE: {self.diagnostics['final_game_mae']:.3f}")
        print(f"  Baseline MAE: {self.diagnostics['baseline_margin_mae']:.3f}")
        print(f"  Improvement: {(1 - self.diagnostics['final_game_mae']/self.diagnostics['baseline_margin_mae']):.1%}")
        
        # Target correlations
        print(f"\nTarget Correlations:")
        for key, value in self.diagnostics.items():
            if 'final_' in key and '_correlation' in key:
                target_name = key.replace('final_', '').replace('_correlation', '')
                print(f"  {target_name}: {value:.4f}")
        
        print(f"\nGames analyzed: {self.diagnostics['n_games']}")

# Convenience function matching original interface
def build_power_ratings_multi_target(team_data, schedule_df, model_features, 
                                    target_columns=['fpi'], mae_weight=0.3):
    """
    Build power ratings optimizing for multiple target columns
    
    Parameters:
    -----------
    target_columns : list
        List of column names to use as optimization targets
    mae_weight : float
        Weight given to game prediction accuracy vs target correlation
    """
    system = MultiTargetPowerRatingSystem()
    system.fit(team_data, schedule_df, model_features, target_columns, mae_weight=mae_weight)
    
    # Merge ratings back to original team_data
    result_data = team_data.merge(
        system.team_ratings[['team', 'power_rating']], 
        on='team', 
        how='left'
    )
    
    return result_data, system.diagnostics, system

























#############################
## FORMATTING OF TEAM DATA ##
#############################
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# ---------------- STATS FORMATTING ---------------- #
def stats_formatting(team_data, current_week, current_year):
    """Scale and rank team statistics, create team_power_rankings table."""
    # --- Scalers
    scaler100 = MinMaxScaler(feature_range=(1, 100))
    scaler10 = MinMaxScaler(feature_range=(1, 10))
    scalerTurnovers = MinMaxScaler(feature_range=(1, 100))
    scalerThirdDown = MinMaxScaler(feature_range=(1, 100))
    scalerTalent = MinMaxScaler(feature_range=(100, 1000))
    scalerAvgFieldPosition = MinMaxScaler(feature_range=(-10, 10))
    scalerPPO = MinMaxScaler(feature_range=(1, 100))

    # --- Scale metrics
    team_data['sp_conf_scaled'] = scaler10.fit_transform(team_data[['sp_conf_rating']])
    team_data['total_turnovers_scaled'] = scalerTurnovers.fit_transform(team_data[['total_turnovers']])
    team_data['possession_scaled'] = scaler100.fit_transform(team_data[['possessionTimeMinutes']])
    team_data['third_down_scaled'] = scalerThirdDown.fit_transform(team_data[['thirdDownConversionRate']])
    team_data['offense_avg_field_position_scaled'] = -1 * scalerAvgFieldPosition.fit_transform(team_data[['Offense_fieldPosition_averageStart']])
    team_data['defense_avg_field_position_scaled'] = scalerAvgFieldPosition.fit_transform(team_data[['Defense_fieldPosition_averageStart']])
    team_data['offense_ppo_scaled'] = scalerPPO.fit_transform(team_data[['Offense_pointsPerOpportunity']])
    team_data['offense_success_scaled'] = scaler100.fit_transform(team_data[['Offense_successRate']])
    team_data['offense_explosive'] = scaler100.fit_transform(team_data[['Offense_explosiveness']])
    team_data['talent_scaled'] = scalerTalent.fit_transform(team_data[['avg_talent']])

    # --- Custom scaling
    team_data['defense_ppo_scaled'] = 100 - (
        (team_data['Defense_pointsPerOpportunity'] - team_data['Defense_pointsPerOpportunity'].min()) * 99
        / (team_data['Defense_pointsPerOpportunity'].max() - team_data['Defense_pointsPerOpportunity'].min())
    )
    team_data['penalties_scaled'] = 100 - (
        (team_data['penaltyYards'] - team_data['penaltyYards'].min()) * 99
        / (team_data['penaltyYards'].max() - team_data['penaltyYards'].min())
    )
    team_data['offense_avg_field_position_scaled'] = 100 - (
        (team_data['Offense_fieldPosition_averageStart'] - team_data['Offense_fieldPosition_averageStart'].min()) * 99
        / (team_data['Offense_fieldPosition_averageStart'].max() - team_data['Offense_fieldPosition_averageStart'].min())
    )
    team_data['offense_ppa_scaled'] = scaler100.fit_transform(team_data[['Offense_ppa']])
    team_data['defense_ppa_scaled'] = 100 - (
        (team_data['Defense_ppa'] - team_data['Defense_ppa'].min()) * 99
        / (team_data['Defense_ppa'].max() - team_data['Defense_ppa'].min())
    )
    team_data['defense_success_scaled'] = 100 - (
        (team_data['Defense_successRate'] - team_data['Defense_successRate'].min()) * 99
        / (team_data['Defense_successRate'].max() - team_data['Defense_successRate'].min())
    )
    team_data['defense_explosive'] = 100 - (
        (team_data['Defense_explosiveness'] - team_data['Defense_explosiveness'].min()) * 99
        / (team_data['Defense_explosiveness'].max() - team_data['Defense_explosiveness'].min())
    )

    # --- Composite Metrics
    team_data['PBR'] = team_data['penaltyYards'] / team_data['talent_scaled']
    team_data['PBR_rank'] = team_data['PBR'].rank(method='min', ascending=True)

    team_data['STM'] = (
        (team_data['kickReturnYards'] / team_data['kickReturns']) +
        (team_data['puntReturnYards'] / team_data['puntReturns']) -
        team_data['Offense_fieldPosition_averageStart'] +
        team_data['Defense_fieldPosition_averageStart']
    )
    team_data['STM_rank'] = team_data['STM'].rank(method='min', ascending=False)

    team_data['DCE'] = (
        (team_data['possessionTimeMinutes'] / team_data['games_played']) +
        (10 * team_data['thirdDownConversionRate']) +
        (20 * team_data['fourthDownConversionRate'])
    )
    team_data['DCE_rank'] = team_data['DCE'].rank(method='min', ascending=False)

    team_data['DefensivePossessionTime'] = (team_data['games_played'] * 60) - team_data['possessionTimeMinutes']
    team_data['DDE'] = (
        (0.6 * team_data['tacklesForLoss']) +
        (4 * team_data['interceptions']) +
        (6 * team_data['fumblesRecovered']) +
        (1.6 * team_data['sacks'])
    )
    team_data['DDE_rank'] = team_data['DDE'].rank(method='min', ascending=False)

    # --- Ranking features
    team_data['talent_scaled_rank'] = team_data['talent_scaled'].rank(method='min', ascending=False)
    team_data['offense_success_rank'] = team_data['offense_success_scaled'].rank(method='min', ascending=False)
    team_data['defense_success_rank'] = team_data['defense_success_scaled'].rank(method='min', ascending=False)
    team_data['offense_explosive_rank'] = team_data['offense_explosive'].rank(method='min', ascending=False)
    team_data['defense_explosive_rank'] = team_data['defense_explosive'].rank(method='min', ascending=False)
    team_data['total_turnovers_rank'] = team_data['total_turnovers_scaled'].rank(method='min', ascending=False)
    team_data['penalties_rank'] = team_data['penalties_scaled'].rank(method='min', ascending=False)

    # --- Sort + Rankings
    team_data = team_data.sort_values(by='power_rating', ascending=False).reset_index(drop=True)
    team_data['power_rating'] = round(team_data['power_rating'], 1)
    team_data = team_data.drop_duplicates(subset='team')

    team_power_rankings = team_data[['team', 'power_rating', 'conference']].sort_values(
        by='power_rating', ascending=False
    ).reset_index(drop=True)
    team_power_rankings.index = team_power_rankings.index + 1
    team_power_rankings['week'] = current_week
    team_power_rankings['year'] = current_year

    return team_data, team_power_rankings

from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
import pandas as pd

# --- Vectorized PEAR Win Prob function ---
def PEAR_Win_Prob_vectorized(home_pr, away_pr):
    rating_diff = np.array(home_pr) - np.array(away_pr)
    return np.round(1 / (1 + 10 ** (-rating_diff / 7.5)) * 100, 2)

# ---------------- METRIC CREATION ---------------- #
def metric_creation(team_data, records, current_week, current_year, postseason=False):
    """Create SOS, SOR, and Most Deserving metrics for teams."""
    # --- Build Year-Long Schedule
    games_list = []
    for week in range(1, 17):
        response = games_api.get_games(year=current_year, week=week, classification='fbs')
        games_list.extend(response)
    if postseason:
        games_list.extend(games_api.get_games(year=current_year, classification='fbs', season_type='postseason'))

    games = [
        dict(
            id=g.id, season=g.season, week=g.week, start_date=g.start_date,
            home_team=g.home_team, home_elo=g.home_pregame_elo,
            away_team=g.away_team, away_elo=g.away_pregame_elo,
            home_points=g.home_points, away_points=g.away_points,
            neutral=g.neutral_site
        )
        for g in games_list
    ]
    year_long_schedule = pd.DataFrame(games)

    # --- Merge power ratings
    fallback_value = team_data['power_rating'].mean() - 2 * team_data['power_rating'].std()
    for side in ['home', 'away']:
        year_long_schedule = year_long_schedule.merge(
            team_data[['team', 'power_rating']],
            left_on=f'{side}_team', right_on='team', how='left'
        ).rename(columns={'power_rating': f'{side}_pr'}).drop(columns='team')
        year_long_schedule[f'{side}_pr'] = year_long_schedule[f'{side}_pr'].fillna(fallback_value)

    # --- Add Win Probabilities
    year_long_schedule['PEAR_win_prob'] = PEAR_Win_Prob_vectorized(
        year_long_schedule['home_pr'], year_long_schedule['away_pr']
    )

    # Pre-index schedules per team
    team_schedules = {team: year_long_schedule[(year_long_schedule['home_team'] == team) |
                                               (year_long_schedule['away_team'] == team)]
                      for team in team_data['team']}

    # --- SOS Calculation ---
    average_pr = round(team_data['power_rating'].mean(), 2)
    good_team_pr = round(team_data['power_rating'].std() + average_pr, 2)
    elite_team_pr = round(2 * team_data['power_rating'].std() + average_pr, 2)

    def calc_expected(team):
        schedule = team_schedules[team]
        df = average_team_distribution(1000, schedule, elite_team_pr, team)
        return df['expected_wins'].values[0]

    expected_wins_list = Parallel(n_jobs=-1)(delayed(calc_expected)(team) for team in team_data['team'])

    SOS = pd.DataFrame({'team': team_data['team'], 'avg_expected_wins': expected_wins_list})
    SOS = SOS.sort_values('avg_expected_wins').reset_index(drop=True)
    SOS['SOS'] = SOS.index + 1
    print("SOS Calculation Done")

    # --- SOR Calculation ---
    completed_games = year_long_schedule[year_long_schedule['home_points'].notna()].copy()

    def calc_sor(team):
        games_played = records.loc[records['team'] == team, 'games_played'].values[0]
        wins = records.loc[records['team'] == team, 'wins'].values[0]
        team_games = completed_games[(completed_games['home_team'] == team) | 
                                    (completed_games['away_team'] == team)].copy()

        home_probs = PEAR_Win_Prob_vectorized(good_team_pr, team_games['away_pr'])
        away_probs = 100 - PEAR_Win_Prob_vectorized(team_games['home_pr'], good_team_pr)

        team_games['good_win_prob'] = np.where(team_games['home_team'] == team, home_probs, away_probs)

        home_probs = PEAR_Win_Prob_vectorized(average_pr, team_games['away_pr'])
        away_probs = 100 - PEAR_Win_Prob_vectorized(team_games['home_pr'], average_pr)
        team_games['avg_win_prob'] = np.where(team_games['home_team'] == team, home_probs, away_probs)

        home_probs = PEAR_Win_Prob_vectorized(elite_team_pr, team_games['away_pr'])
        away_probs = 100 - PEAR_Win_Prob_vectorized(team_games['home_pr'], elite_team_pr)
        team_games['elite_win_prob'] = np.where(team_games['home_team'] == team, home_probs, away_probs)

        current_xWins = round(team_games['avg_win_prob'].sum() / 100, 2)
        good_xWins = round(team_games['good_win_prob'].sum() / 100, 2)
        elite_xWins = round(team_games['elite_win_prob'].sum() / 100, 2)

        if games_played != len(team_games):
            current_xWins += 1
            good_xWins += 1
            elite_xWins += 1

        return round(wins - current_xWins, 2), round(wins - good_xWins, 2), round(wins - elite_xWins, 2)

    sor_results = Parallel(n_jobs=-1)(delayed(calc_sor)(team) for team in team_data['team'])
    SOR = pd.DataFrame(sor_results, columns=['wins_above_average','wins_above_good','wins_above_elite'])
    SOR.insert(0, 'team', team_data['team'])
    SOR = SOR.sort_values('wins_above_good', ascending=False).reset_index(drop=True)
    SOR['SOR'] = SOR.index + 1
    print("SOR Calculation Done")

    # --- Most Deserving Calculation ---
    num_12_pr = team_data['power_rating'].iloc[11]

    def f(mov):
        return np.clip(np.log(np.abs(mov) + 1) * np.sign(mov), -10, 10)

    completed_games['margin_of_victory'] = completed_games['home_points'] - completed_games['away_points']

    def calc_deserving(team):
        games_played = records.loc[records['team'] == team, 'games_played'].values[0]
        wins = records.loc[records['team'] == team, 'wins'].values[0]
        team_games = completed_games[(completed_games['home_team'] == team) |
                                    (completed_games['away_team'] == team)].copy()
        if current_week < 6:
            mov_adj = 0
        else:
            mov_adj = f(team_games['margin_of_victory'])

        team_games['home_input'] = np.where(~team_games['neutral'], team_games['home_pr'] + 2, team_games['home_pr'])
        team_games['home_12_pr'] = np.where(~team_games['neutral'], num_12_pr + 2, num_12_pr)
        team_games['away_12_pr'] = num_12_pr
        home_probs = PEAR_Win_Prob_vectorized(team_games['home_12_pr'], team_games['away_pr']) + mov_adj
        away_probs = 100 - PEAR_Win_Prob_vectorized(team_games['home_input'], team_games['away_12_pr']) - mov_adj
        team_games['adj_win_prob'] = np.where(team_games['home_team'] == team, home_probs, away_probs)

        xWins = round(team_games['adj_win_prob'].sum() / 100, 3)
        if games_played != len(team_games):
            xWins += 1
        return round(wins - xWins, 3)


    deserving_results = Parallel(n_jobs=-1)(delayed(calc_deserving)(team) for team in team_data['team'])
    most_deserving = pd.DataFrame({'team': team_data['team'], 'most_deserving_wins': deserving_results})
    most_deserving = most_deserving.sort_values('most_deserving_wins', ascending=False).reset_index(drop=True)
    most_deserving['most_deserving'] = most_deserving.index + 1
    print("Most Deserving Calculation Done")

    team_data = pd.merge(team_data, SOS, how='left', on='team')
    team_data = pd.merge(team_data, SOR, how='left', on='team')
    team_data = pd.merge(team_data, most_deserving, how='left', on='team')

    return team_data, year_long_schedule, SOS, SOR, most_deserving



















#########################################
## VISUALS FUNCTIONS - THESE WILL SAVE ##
#########################################

def best_and_worst(all_data, team_logos, metric, title, subtitle, visual_name, folder_path):
    if metric == 'average_metric_rank':
        top_25_best = all_data.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
        top_25_worst = all_data.sort_values(metric, ascending=False)[:25].reset_index(drop=True)
        top_25_worst = top_25_worst.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
    else:
        top_25_best = all_data.sort_values(metric, ascending=False)[:25].reset_index(drop=True)
        top_25_worst = all_data.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
        top_25_worst = top_25_worst.sort_values(metric, ascending=False)[:25].reset_index(drop=True)

    # Create a figure with 5 rows and 10 columns
    fig, axs = plt.subplots(5, 10, figsize=(20, 10), dpi=125)

    # Adjust space between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.patch.set_facecolor('#CECEB2')

    # Title and description
    plt.suptitle(title, fontsize=24, fontweight='bold', color='black')
    plt.text(0.51, 0.92, subtitle, ha='center', fontsize=18, color='black', transform=fig.transFigure)
    plt.text(0.51, 0.88, "@PEARatings", ha='center', fontsize=18, color='black', transform=fig.transFigure, fontweight='bold')

    # Fill the grid alternating Best and Worst Defenses
    for i in range(5):  # There are 5 rows
        # Best defenses: Columns 0-4 for each row (Best in every odd index)
        for j in range(5):  
            ax = axs[i, j]
            team = top_25_best.loc[i*5 + j, 'team']
            img = team_logos[team]
            ax.imshow(img)
            ax.set_facecolor('#f0f0f0')
            # ax.set_title(f"#{i*5 + j + 1} {team} \n{round(top_25_best.loc[i*5 + j, metric], 1)}", fontsize=8, fontweight='bold')
            ax.text(0.5, -0.07, f"#{i*5 + j + 1} \n{round(top_25_best.loc[i*5 + j, metric], 1)}", fontsize=14, fontweight='bold', transform=ax.transAxes, ha='center', va='top')
            ax.axis('off')

        # Worst defenses: Columns 5-9 for each row (Worst in every even index after 5)
        for j in range(5, 10):  
            ax = axs[i, j]
            team = top_25_worst.loc[i*5 + (j-5), 'team']
            img = team_logos[team]
            ax.imshow(img)
            ax.set_facecolor('#f0f0f0')
            
            # Start counting for Worst from 134 and decrement
            worst_rank = (len(all_data) - 24) + (i*5 + (j-5)) 
            ax.text(0.5, -0.07, f"#{worst_rank} \n{round(top_25_worst.loc[i*5 + (j-5), metric], 1)}", fontsize=14, fontweight='bold', transform=ax.transAxes, ha='center', va='top')
            ax.axis('off')

    fig.add_artist(Line2D([0.512, 0.512], [0.06, 0.86], color='black', lw=5))
    fig.text(0.13, 0.92, "Best", ha='left', va='center', fontsize=18, fontweight='bold', color='black')
    fig.text(0.89, 0.92, "Worst", ha='right', va='center', fontsize=18, fontweight='bold', color='black')

    file_path = os.path.join(folder_path, visual_name)
    # Show the final figure
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

def other_best_and_worst(all_data, team_logos, metric, title, subtitle, visual_name, folder_path):
    if metric == 'avg_expected_wins':
        top_25_best = all_data.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
        top_25_worst = all_data.sort_values(metric, ascending=False)[:25].reset_index(drop=True)
        top_25_worst = top_25_worst.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
        rounding = 3
        good = 'Hardest'
        bad = 'Easiest'
    else:
        top_25_best = all_data.sort_values(metric, ascending=False)[:25].reset_index(drop=True)
        top_25_worst = all_data.sort_values(metric, ascending=True)[:25].reset_index(drop=True)
        top_25_worst = top_25_worst.sort_values(metric, ascending=False)[:25].reset_index(drop=True)
        rounding = 1
        if (metric == 'wins_above_good') or (metric == 'performance'):
            rounding = 3
        if (metric == 'RTP') or (metric == 'wins_above_average'):
            rounding = 2
        good = 'Best'
        bad = 'Worst'
        if metric == 'performance':
            good='Overperformers'
            bad='Underperformers'
        if (metric == 'wins_above_average'):
            good = 'Most'
            bad = 'Least'

    # Create a figure with 5 rows and 10 columns
    fig, axs = plt.subplots(5, 10, figsize=(20, 10), dpi=125)

    # Adjust space between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.patch.set_facecolor('#CECEB2')

    # Title and description
    plt.suptitle(title, fontsize=24, fontweight='bold', color='black')
    plt.text(0.51, 0.92, subtitle, ha='center', fontsize=18, color='black', transform=fig.transFigure)
    plt.text(0.51, 0.88, "@PEARatings", ha='center', fontsize=18, color='black', transform=fig.transFigure, fontweight='bold')

    # Fill the grid alternating Best and Worst Defenses
    for i in range(5):  # There are 5 rows
        # Best defenses: Columns 0-4 for each row (Best in every odd index)
        for j in range(5):  
            ax = axs[i, j]
            team = top_25_best.loc[i*5 + j, 'team']
            img = team_logos[team]
            ax.imshow(img)
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, -0.07, f"#{i*5 + j + 1} \n{round(top_25_best.loc[i*5 + j, metric], rounding)}", fontsize=14, fontweight='bold', transform=ax.transAxes, ha='center', va='top')
            ax.axis('off')

        # Worst defenses: Columns 5-9 for each row (Worst in every even index after 5)
        for j in range(5, 10):  
            ax = axs[i, j]
            team = top_25_worst.loc[i*5 + (j-5), 'team']
            img = team_logos[team]
            ax.imshow(img)
            ax.set_facecolor('#f0f0f0')
            
            # Start counting for Worst from 134 and decrement
            worst_rank = (len(all_data) - 24) + (i*5 + (j-5)) 
            ax.text(0.5, -0.07, f"#{worst_rank} \n{round(top_25_worst.loc[i*5 + (j-5), metric], rounding)}", fontsize=14, fontweight='bold', transform=ax.transAxes, ha='center', va='top')
            ax.axis('off')

    fig.add_artist(Line2D([0.512, 0.512], [0.06, 0.86], color='black', lw=5))
    fig.text(0.13, 0.92, good, ha='left', fontsize=18, fontweight='bold', color='black')
    fig.text(0.89, 0.92, bad, ha='right', fontsize=18, fontweight='bold', color='black')

    file_path = os.path.join(folder_path, visual_name)
    # Show the final figure
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

def plot_matchup(wins_df, logos_df, team_logos, team_data, last_week_data, last_month_data, start_season_data, all_data, schedule_info, records, SOS, SOR, elo_ratings, home_team, away_team, current_year, current_week, neutrality=False):
    sns.set(style='whitegrid')
    ################################# HELPER FUNCTIONS #################################

    def PEAR_Win_Prob(home_power_rating, away_power_rating):
        return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

    def adjust_home_pr(home_win_prob):
        return ((home_win_prob - 50) / 50) * 1

    def round_to_nearest_half(x):
        return np.round(x * 2) / 2   

    def find_team_spread(game, team_data, team_name, home_team, away_team, neutral):
        fallback_value = team_data['power_rating'].mean() - 2 * team_data['power_rating'].std()
        rating_map = dict(zip(team_data['team'], team_data['power_rating']))
        home_rating = rating_map.get(home_team, fallback_value)
        away_rating = rating_map.get(away_team, fallback_value)
        home_win_prob = game['home_win_prob']
        def adjust_home_pr(home_win_prob):
            return ((home_win_prob - 50) / 50) * 1
        def round_to_nearest_half(x):
            return np.round(x * 2) / 2 
        if neutral:
            spread = round((home_rating + adjust_home_pr(home_win_prob) - away_rating),1)
        else:
            spread = round((GLOBAL_HFA + home_rating + adjust_home_pr(home_win_prob) - away_rating),1)
        if (home_team == team_name) & (spread > 0):
            output = "-" + str(spread)
        elif (home_team == team_name) & (spread < 0):
            output = "+" + str(abs(spread))
        elif (home_team != team_name) & (spread < 0):
            output = str(spread)
        elif (spread == 0.0):
            output = str(spread)
        else:
            output = "+" + str(spread)

        return output, spread
    
    def add_spreads(schedule_info, team_data, team_name):
        # Define a helper function to apply the find_team_spread to each row
        def compute_spread(row):
            # Extract relevant data from the current row
            home_team = row['home_team']
            away_team = row['away_team']
            neutral = row['neutral']
            
            # Call the find_team_spread function with this row's data
            spread_str, raw_spread = find_team_spread(row, team_data, team_name, home_team, away_team, neutral)
            
            # Return both values as a tuple, so we can unpack them into two columns
            return pd.Series([spread_str, raw_spread])

        # Apply the compute_spread function to each row of the DataFrame
        schedule_info[['spread', 'raw_spread']] = schedule_info.apply(compute_spread, axis=1)
        
        return schedule_info

    def get_color_wl(w_l):
        if w_l == "W":
            return '#1D4D00'
        else:
            return '#660000'
    
    def get_color_future(raw_spread, game_home_team, team):
        if (game_home_team == team) and (raw_spread <= 0):
            return '#660000'
        if (game_home_team == team) and (raw_spread > 0):
            return '#1D4D00'
        if (game_home_team != team) and (raw_spread <= 0):
            return '#1D4D00'
        if (game_home_team != team) and (raw_spread > 0):
            return '#660000'
        
    def get_rank_color(number):
        """
        Returns a hex color code based on a number between 1 and 134.
        
        Parameters:
            number (int): A number between 1 and 134.
        
        Returns:
            str: Hex color code corresponding to the input number.
        """
        if not 1 <= number <= 136:
            raise ValueError("Number must be between 1 and 136")
        
        # Define the color gradient points
        gradient = [
            (1, '#1D4D00'),    # Start green
            (35, '#2C5E00'),   # Midpoint green
            (67, '#808080'),   # Grey
            (105, '#8B0000'),  # Red in the middle
            (136, '#660000')   # End dark red
        ]
        
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to hex
        def rgb_to_hex(rgb):
            return '#' + ''.join(f'{int(c):02X}' for c in rgb)
        
        # Interpolate between two colors
        def interpolate_color(color1, color2, fraction):
            return tuple(
                color1[i] + (color2[i] - color1[i]) * fraction
                for i in range(3)
            )
        
        # Find the range that includes the number
        for i in range(len(gradient) - 1):
            start, end = gradient[i], gradient[i + 1]
            if start[0] <= number <= end[0]:
                fraction = (number - start[0]) / (end[0] - start[0])
                start_rgb, end_rgb = hex_to_rgb(start[1]), hex_to_rgb(end[1])
                interpolated_rgb = interpolate_color(start_rgb, end_rgb, fraction)
                return rgb_to_hex(interpolated_rgb)
        
        # Fallback (should not reach here)
        return '#000000'

    def grab_team_elo(team):        
        return elo_ratings[elo_ratings['team'] == team]['elo'].values[0]

    ################################# PREPPING DATA NEEDED #################################

    entire_schedule = schedule_info.copy()
    completed_games = schedule_info[schedule_info['home_points'].notna()]
    non_completed_games = schedule_info[schedule_info['home_points'].isna()]
    scaler100 = MinMaxScaler(feature_range=(1,100))
    all_data['talent_scaled_percentile'] = scaler100.fit_transform(all_data[['talent_scaled']])

    home_wins_df = wins_df[wins_df['team'] == home_team]
    away_wins_df = wins_df[wins_df['team'] == away_team]
    # home_xwins = round(wins_df[wins_df['team'] == home_team]['expected_wins'].values[0], 1)
    # away_xwins = round(wins_df[wins_df['team'] == away_team]['expected_wins'].values[0], 1)
    # home_xlosses = round(wins_df[wins_df['team'] == home_team]['expected_loss'].values[0], 1)
    # away_xlosses = round(wins_df[wins_df['team'] == away_team]['expected_loss'].values[0], 1)
    home_rank = team_data[team_data['team'] == home_team].index[0] + 1
    away_rank = team_data[team_data['team'] == away_team].index[0] + 1
    home_win_6 = round(100 * home_wins_df['WIN6%'].values[0], 1)
    away_win_6 = round(100 * away_wins_df['WIN6%'].values[0], 1)
    home_last_week = last_week_data[last_week_data['team'] == home_team]['power_rating'].values[0]
    away_last_week = last_week_data[last_week_data['team'] == away_team]['power_rating'].values[0]
    home_last_month = last_month_data[last_month_data['team'] == home_team]['power_rating'].values[0]
    away_last_month = last_month_data[last_month_data['team'] == away_team]['power_rating'].values[0]
    home_start_rating = start_season_data[start_season_data['team'] == home_team]['power_rating'].values[0]
    away_start_rating = start_season_data[start_season_data['team'] == away_team]['power_rating'].values[0]
    home_wins = records[records['team'] == home_team]['wins'].values[0]
    away_wins = records[records['team'] == away_team]['wins'].values[0]
    home_losses = records[records['team'] == home_team]['losses'].values[0]
    away_losses = records[records['team'] == away_team]['losses'].values[0]
    home_conference_wins = records[records['team'] == home_team]['conference_wins'].values[0]
    home_conference_losses = records[records['team'] == home_team]['conference_losses'].values[0]
    away_conference_wins = records[records['team'] == away_team]['conference_wins'].values[0]
    away_conference_losses = records[records['team'] == away_team]['conference_losses'].values[0]

    average_pr = team_data['power_rating'].mean()
    home_completed_games = completed_games[(completed_games['home_team'] == home_team) | (completed_games['away_team'] == home_team)]
    away_completed_games = completed_games[(completed_games['home_team'] == away_team) | (completed_games['away_team'] == away_team)]
    home_completed_games['team_win_prob'] = np.where(home_completed_games['home_team'] == home_team, 
                                    home_completed_games['PEAR_win_prob'], 
                                    1 - home_completed_games['PEAR_win_prob'])
    away_completed_games['team_win_prob'] = np.where(away_completed_games['home_team'] == away_team, 
                                    away_completed_games['PEAR_win_prob'], 
                                    1 - away_completed_games['PEAR_win_prob'])
    home_completed_games['avg_win_prob'] = np.where(home_completed_games['home_team'] == home_team, 
                                    PEAR_Win_Prob(average_pr, home_completed_games['away_pr']), 
                                    100 - PEAR_Win_Prob(home_completed_games['home_pr'], average_pr))
    away_completed_games['avg_win_prob'] = np.where(away_completed_games['home_team'] == away_team, 
                                    PEAR_Win_Prob(average_pr, away_completed_games['away_pr']), 
                                    100 - PEAR_Win_Prob(away_completed_games['home_pr'], average_pr))
    home_avg_xwins = round(sum(home_completed_games['avg_win_prob']) / 100, 1)
    home_avg_xlosses = round(len(home_completed_games) - home_avg_xwins, 1)
    away_avg_xwins = round(sum(away_completed_games['avg_win_prob']) / 100, 1)
    away_avg_xlosses = round(len(away_completed_games) - away_avg_xwins, 1)
    home_completed_xwins = round(sum(home_completed_games['team_win_prob']), 1)
    away_completed_xwins = round(sum(away_completed_games['team_win_prob']), 1)
    home_completed_xlosses = round(len(home_completed_games) - home_completed_xwins, 1)
    away_completed_xlosses = round(len(away_completed_games) - away_completed_xwins, 1)
    home_games_played = home_wins + home_losses
    away_games_played = away_wins + away_losses
    if len(home_completed_games) != home_games_played:
        home_completed_xwins = home_completed_xwins + 1
        home_avg_xwins = home_avg_xwins + 1
    if len(away_completed_games) != away_games_played:
        away_completed_xwins = away_completed_xwins + 1
        away_avg_xwins = away_avg_xwins + 1
    home_win_out = home_wins + 12 - home_games_played
    away_win_out = away_wins + 12 - away_games_played
    home_win_out_percent = round(home_wins_df[f'win_{home_win_out}'].values[0] * 100, 1)
    away_win_out_percent = round(away_wins_df[f'win_{away_win_out}'].values[0] * 100, 1)
    home_md = all_data[all_data['team'] == home_team]['most_deserving'].values[0]
    away_md = all_data[all_data['team'] == away_team]['most_deserving'].values[0]
    home_sor = SOR[SOR['team'] == home_team]['SOR'].values[0]
    home_sos = SOS[SOS['team'] == home_team]['SOS'].values[0]
    away_sor = SOR[SOR['team'] == away_team]['SOR'].values[0]
    away_sos = SOS[SOS['team'] == away_team]['SOS'].values[0]
    home_non_completed_games = non_completed_games[(non_completed_games['home_team'] == home_team) | (non_completed_games['away_team'] == home_team)].reset_index(drop = True)
    away_non_completed_games = non_completed_games[(non_completed_games['home_team'] == away_team) | (non_completed_games['away_team'] == away_team)].reset_index(drop = True)
    home_elo = grab_team_elo(home_team)
    away_elo = grab_team_elo(away_team)
    
    cmap = LinearSegmentedColormap.from_list('dark_gradient_orange', ['#660000', '#8B0000', '#808080', '#2C5E00', '#1D4D00'], N=100)
    def get_color(value, vmin=0, vmax=100):
        norm_value = (value - vmin) / (vmax - vmin)  # Normalize the value between 0 and 1
        return cmap(norm_value)  # Get the color from the colormap
    
    xwin = LinearSegmentedColormap.from_list('dark_gradient_pattern', ['#660000', '#8B0000', '#808080', '#2C5E00', '#1D4D00'], N=100)

    # Function to get color based on value with specified boundaries
    def get_gradient_color(value, vmin=-1.5, vmax=1.5):
        if value <= vmin:
            return '#660000'  # Lighter dark red for <= -1.5
        elif value >= vmax:
            return '#1D4D00'  # Dark green for >= 1.5
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize value between vmin and vmax
            return xwin(norm_value)  # Get color from colormap for intermediate values

    if (len(home_non_completed_games) != 0):
        home_non_completed_games = add_spreads(home_non_completed_games, team_data, home_team)
    if (len(away_non_completed_games) != 0):
        away_non_completed_games = add_spreads(away_non_completed_games, team_data, away_team)

    def safe_int_rank(df, team, col_name, fallback=None):
        val = df.loc[df['team'] == team, col_name]
        if val.empty or pd.isna(val.values[0]):
            return int(fallback if fallback is not None else df[col_name].max() + 1)
        return int(val.values[0])

    home_stm = safe_int_rank(all_data, home_team, 'STM_rank')
    home_pbr = safe_int_rank(all_data, home_team, 'PBR_rank')
    home_dce = safe_int_rank(all_data, home_team, 'DCE_rank')
    home_dde = safe_int_rank(all_data, home_team, 'DDE_rank')
    away_stm = safe_int_rank(all_data, away_team, 'STM_rank')
    away_pbr = safe_int_rank(all_data, away_team, 'PBR_rank')
    away_dce = safe_int_rank(all_data, away_team, 'DCE_rank')
    away_dde = safe_int_rank(all_data, away_team, 'DDE_rank')

    home_power_rating = round(team_data[team_data['team'] == home_team]['power_rating'].values[0], 1)
    home_talent_scaled = round(all_data[all_data['team'] == home_team]['talent_scaled_percentile'].values[0], 1)
    home_offense_success = round(all_data[all_data['team'] == home_team]['offense_success_scaled'].values[0], 1)
    home_defense_success = round(all_data[all_data['team'] == home_team]['defense_success_scaled'].values[0], 1)
    home_offense_explosive = round(all_data[all_data['team'] == home_team]['offense_explosive'].values[0], 1)
    home_defense_explosive = round(all_data[all_data['team'] == home_team]['defense_explosive'].values[0], 1)
    home_turnovers = round(all_data[all_data['team'] == home_team]['total_turnovers_scaled'].values[0], 1)
    home_penalties = round(all_data[all_data['team'] == home_team]['penalties_scaled'].values[0], 1)
    home_offensive = all_data[all_data['team'] == home_team]['offensive_rank'].values[0]
    home_defensive = all_data[all_data['team'] == home_team]['defensive_rank'].values[0]
    home_talent_rank = int(all_data[all_data['team'] == home_team]['talent_scaled_rank'].values[0])
    home_offense_success_rank = int(all_data[all_data['team'] == home_team]['offense_success_rank'].values[0])
    home_defense_success_rank = int(all_data[all_data['team'] == home_team]['defense_success_rank'].values[0])
    home_offense_explosive_rank = int(all_data[all_data['team'] == home_team]['offense_explosive_rank'].values[0])
    home_defense_explosive_rank = int(all_data[all_data['team'] == home_team]['defense_explosive_rank'].values[0])
    home_turnover_rank = int(all_data[all_data['team'] == home_team]['total_turnovers_rank'].values[0])
    home_penalties_rank = int(all_data[all_data['team'] == home_team]['penalties_rank'].values[0])

    away_power_rating = round(team_data[team_data['team'] == away_team]['power_rating'].values[0], 1)
    away_talent_scaled = round(all_data[all_data['team'] == away_team]['talent_scaled_percentile'].values[0], 1)
    away_offense_success = round(all_data[all_data['team'] == away_team]['offense_success_scaled'].values[0], 1)
    away_defense_success = round(all_data[all_data['team'] == away_team]['defense_success_scaled'].values[0], 1)
    away_offense_explosive = round(all_data[all_data['team'] == away_team]['offense_explosive'].values[0], 1)
    away_defense_explosive = round(all_data[all_data['team'] == away_team]['defense_explosive'].values[0], 1)
    away_turnovers = round(all_data[all_data['team'] == away_team]['total_turnovers_scaled'].values[0], 1)
    away_penalties = round(all_data[all_data['team'] == away_team]['penalties_scaled'].values[0], 1)
    away_offensive = all_data[all_data['team'] == away_team]['offensive_rank'].values[0]
    away_defensive = all_data[all_data['team'] == away_team]['defensive_rank'].values[0]
    away_talent_rank = int(all_data[all_data['team'] == away_team]['talent_scaled_rank'].values[0])
    away_offense_success_rank = int(all_data[all_data['team'] == away_team]['offense_success_rank'].values[0])
    away_defense_success_rank = int(all_data[all_data['team'] == away_team]['defense_success_rank'].values[0])
    away_offense_explosive_rank = int(all_data[all_data['team'] == away_team]['offense_explosive_rank'].values[0])
    away_defense_explosive_rank = int(all_data[all_data['team'] == away_team]['defense_explosive_rank'].values[0])
    away_turnover_rank = int(all_data[all_data['team'] == away_team]['total_turnovers_rank'].values[0])
    away_penalties_rank = int(all_data[all_data['team'] == away_team]['penalties_rank'].values[0])

    home_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1)*100,2)
    PEAR_home_prob = PEAR_Win_Prob(home_power_rating, away_power_rating)
    spread = (GLOBAL_HFA + home_power_rating + adjust_home_pr(home_win_prob) - away_power_rating).round(1)
    if neutrality:
        spread = (spread - GLOBAL_HFA).round(1)
    spread = round(spread,1)
    if (spread) <= 0:
        formatted_spread = (f'{away_team} {spread}')
        game_win_prob = round(100 - PEAR_home_prob,2)
    elif (spread) > 0:
        formatted_spread = (f'{home_team} -{spread}')
        game_win_prob = PEAR_home_prob

    ################################# PLOTTING LOGOS #################################

    fig, ax = plt.subplots(nrows=1, figsize=(12, 10),dpi=125)
    fig_width, fig_height = fig.get_size_inches()
    fig.patch.set_facecolor('#CECEB2')  # Set figure background color
    ax.set_facecolor('#CECEB2') 
    ax.axis('off')
    home_logo_url = logos_df[logos_df['team'] == home_team]['logo'].values[0][0]
    home_team_color = logos_df[logos_df['team'] == home_team]['color'].values[0]   
    away_logo_url = logos_df[logos_df['team'] == away_team]['logo'].values[0][0]
    away_team_color = logos_df[logos_df['team'] == away_team]['color'].values[0]   
    
    logo_img = team_logos[home_team]
    img_ax = fig.add_axes([-.1,-.05,0.4,0.4])
    img_ax.imshow(logo_img)
    img_ax.axis('off')

    logo_img = team_logos[away_team]
    img_ax = fig.add_axes([1.,-.05,0.4,0.4])
    img_ax.imshow(logo_img)
    img_ax.axis('off')

    ################################# SCHEDULE INFO ################################# 

    home_j = 0.29
    plt.text(0.65, .32, "FCS Games Not Included", fontsize = 16, va='top', ha='center', transform = ax.transAxes, fontweight='bold')
    for i, game in home_completed_games.iterrows():
        neutral = game['neutral']
        plt.text(0.295, home_j, f"{game['week']}", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
        if game['home_team'] == home_team:
            if game['home_points'] > game['away_points']:
                w_L = 'W'
            else:
                w_L = 'L'
        else:
            if game['home_points'] > game['away_points']:
                w_L = 'L'
            else:
                w_L = 'W'
        if neutral:
            if game['home_team'] == home_team:
                home_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                plt.text(0.325, home_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
                plt.text(0.355, home_j, f"{game['away_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            else:
                home_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                plt.text(0.325, home_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
                plt.text(0.355, home_j, f"{game['home_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        elif game['home_team'] == home_team:
            home_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
            plt.text(0.325, home_j, f"vs", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            plt.text(0.355, home_j, f"{game['away_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        else:
            home_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
            plt.text(0.325, home_j, f"@", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            plt.text(0.355, home_j, f"{game['home_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        if game['home_points'] > game['away_points']:
            plt.text(0.61, home_j, f"{int(game['home_points'])}-{int(game['away_points'])}", fontsize = 16, va='top', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        else:
            plt.text(0.61, home_j, f"{int(game['away_points'])}-{int(game['home_points'])}", fontsize = 16, va='top', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        home_j -= 0.03
    if len(home_non_completed_games != 0):
        for i, game in home_non_completed_games.iterrows():
            game_home_team = game['home_team']
            neutral = game['neutral']
            plt.text(0.295, home_j, f"{game['week']}", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            spread = game['spread']
            raw_spread = game['raw_spread']
            if neutral:
                if game['home_team'] == home_team:
                    home_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                    plt.text(0.325, home_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                    plt.text(0.355, home_j, f"{game['away_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                else:
                    home_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                    plt.text(0.325, home_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                    plt.text(0.355, home_j, f"{game['home_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            elif game['home_team'] == home_team:
                home_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                plt.text(0.325, home_j, f"vs", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                plt.text(0.355, home_j, f"{game['away_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            else:
                home_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                plt.text(0.325, home_j, f"@", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                plt.text(0.355, home_j, f"{game['home_team']} (#{home_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            plt.text(0.61, home_j, f"{spread}", fontsize=16, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_color_future(raw_spread, game_home_team, home_team))

            home_j -= 0.03

    away_j = 0.29
    for i, game in away_completed_games.iterrows():
        neutral = game['neutral']
        plt.text(0.675, away_j, f"{game['week']}", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
        if game['home_team'] == away_team:
            if game['home_points'] > game['away_points']:
                w_L = 'W'
            else:
                w_L = 'L'
        else:
            if game['home_points'] > game['away_points']:
                w_L = 'L'
            else:
                w_L = 'W'
        if neutral:
            if game['home_team'] == away_team:
                away_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                plt.text(0.705, away_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
                plt.text(0.735, away_j, f"{game['away_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            else:
                away_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                plt.text(0.705, away_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
                plt.text(0.735, away_j, f"{game['home_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        elif game['home_team'] == away_team:
            away_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
            plt.text(0.705, away_j, f"vs", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            plt.text(0.735, away_j, f"{game['away_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        else:
            away_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
            plt.text(0.705, away_j, f"@", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
            plt.text(0.735, away_j, f"{game['home_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        if game['home_points'] > game['away_points']:
            plt.text(0.99, away_j, f"{int(game['home_points'])}-{int(game['away_points'])}", fontsize = 16, va='top', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        else:
            plt.text(0.99, away_j, f"{int(game['away_points'])}-{int(game['home_points'])}", fontsize = 16, va='top', transform=ax.transAxes, color = get_color_wl(w_L), fontweight='bold')
        away_j -= 0.03

    if (len(away_non_completed_games) != 0):
        for i, game in away_non_completed_games.iterrows():
            neutral = game['neutral']
            game_home_team = game['home_team']
            plt.text(0.675, away_j, f"{game['week']}", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            spread = game['spread']
            raw_spread = game['raw_spread']
            if neutral:
                if game['home_team'] == away_team:
                    away_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                    plt.text(0.705, away_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                    plt.text(0.735, away_j, f"{game['away_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                else:
                    away_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                    plt.text(0.705, away_j, f"(N)", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                    plt.text(0.735, away_j, f"{game['home_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            elif game['home_team'] == away_team:
                away_opponent_rank = team_data[team_data['team'] == game['away_team']].index[0] + 1
                plt.text(0.705, away_j, f"vs", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                plt.text(0.735, away_j, f"{game['away_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            else:
                away_opponent_rank = team_data[team_data['team'] == game['home_team']].index[0] + 1
                plt.text(0.705, away_j, f"@", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
                plt.text(0.735, away_j, f"{game['home_team']} (#{away_opponent_rank})", fontsize = 16, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
            plt.text(0.99, away_j, f"{spread}", fontsize=16, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_color_future(raw_spread, game_home_team, away_team))

            away_j -= 0.03

    plt.text(-0.075, 0.99, f"Power Rating: {home_power_rating} (#{home_rank})", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_rank))
    plt.text(-0.075, 0.94, f"Current Record: {home_wins} - {home_losses} ({home_conference_wins} - {home_conference_losses})", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
    plt.text(-0.075, 0.89, f"Current xRecord: {home_completed_xwins} - {home_completed_xlosses}", fontsize = 25, va='top', ha='left', transform=ax.transAxes)
    # plt.text(-0.075, 0.84, f"Win 6%: {home_win_6}%", fontsize = 25, va='top', ha='left', transform=ax.transAxes)
    # plt.text(-0.075, 0.79, f"Win Out%: {home_win_out_percent}%", fontsize = 25, va='top', ha='left', transform=ax.transAxes)
    plt.text(-0.075, 0.79, "Percentile (Rank)", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, fontweight='bold')
    plt.text(-0.075, 0.74, f"Team Talent: {home_talent_scaled} (#{home_talent_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_talent_scaled), fontweight='bold')
    plt.text(-0.075, 0.69, f"Offense Success: {home_offense_success} (#{home_offense_success_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_offense_success), fontweight='bold')
    plt.text(-0.075, 0.64, f"Offense Explosiveness: {home_offense_explosive} (#{home_offense_explosive_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_offense_explosive), fontweight='bold')
    plt.text(-0.075, 0.59, f"Defense Success: {home_defense_success} (#{home_defense_success_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_defense_success), fontweight='bold')
    plt.text(-0.075, 0.54, f"Defense Explosiveness: {home_defense_explosive} (#{home_defense_explosive_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_defense_explosive), fontweight='bold')
    plt.text(-0.075, 0.49, f"Turnovers: {home_turnovers} (#{home_turnover_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_turnovers), fontweight='bold')
    plt.text(-0.075, 0.44, f"Penalties: {home_penalties} (#{home_penalties_rank})", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, color=get_color(home_penalties), fontweight='bold')
    plt.text(-0.075, 0.39, f"SOS: #{home_sos}", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_sos))
    plt.text(0.15, 0.39, f"SOR: #{home_sor}", fontsize=25, verticalalignment='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_sor))

    plt.text(0.45, 0.74, f"WP: {PEAR_home_prob:.1f}%", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold')
    plt.text(0.45, 0.69, f"MD: #{home_md}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_md))
    plt.text(0.45, 0.64, f"OFF: #{home_offensive}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_offensive))
    plt.text(0.45, 0.59, f"DEF: #{home_defensive}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_defensive))
    # plt.text(0.45, 0.54, f"ST: #{home_stm}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_stm))
    plt.text(0.45, 0.54, f"PBR: #{home_pbr}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_pbr))
    plt.text(0.45, 0.49, f"DCE: #{home_dce}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_dce))
    plt.text(0.45, 0.44, f"DDE: #{home_dde}", fontsize = 25, va='top', ha='left', transform=ax.transAxes, fontweight='bold', color=get_rank_color(home_dde))


    plt.text(1.38, 0.99, f"Power Rating: {away_power_rating} (#{away_rank})", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_rank))
    plt.text(1.38, 0.94, f"Current Record: {away_wins} - {away_losses} ({away_conference_wins} - {away_conference_losses})", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold')
    plt.text(1.38, 0.89, f"Current xRecord: {away_completed_xwins} - {away_completed_xlosses}", fontsize = 25, va='top', ha='right', transform=ax.transAxes)
    # plt.text(1.38, 0.84, f"Win 6%: {away_win_6}%", fontsize = 25, va='top', ha='right', transform=ax.transAxes)
    # plt.text(1.38, 0.79, f"Win Out%: {away_win_out_percent}%", fontsize = 25, va='top', ha='right', transform=ax.transAxes)
    plt.text(1.38, 0.79, "Percentile (Rank)", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, fontweight='bold')
    plt.text(1.38, 0.74, f"Team Talent: {away_talent_scaled} (#{away_talent_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_talent_scaled), fontweight='bold')
    plt.text(1.38, 0.69, f"Offense Success: {away_offense_success} (#{away_offense_success_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_offense_success), fontweight='bold')
    plt.text(1.38, 0.64, f"Offense Explosiveness: {away_offense_explosive} (#{away_offense_explosive_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_offense_explosive), fontweight='bold')
    plt.text(1.38, 0.59, f"Defense Success: {away_defense_success} (#{away_defense_success_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_defense_success), fontweight='bold')
    plt.text(1.38, 0.54, f"Defense Explosiveness: {away_defense_explosive} (#{away_defense_explosive_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_defense_explosive), fontweight='bold')
    plt.text(1.38, 0.49, f"Turnovers: {away_turnovers} (#{away_turnover_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_turnovers), fontweight='bold')
    plt.text(1.38, 0.44, f"Penalties: {away_penalties} (#{away_penalties_rank})", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, color=get_color(away_penalties), fontweight='bold')
    plt.text(1.16, 0.39, f"SOS: #{away_sos}", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_sos))
    plt.text(1.38, 0.39, f"SOR: #{away_sor}", fontsize=25, verticalalignment='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_sor))

    plt.text(0.85, 0.74, f"WP: {round(100-PEAR_home_prob,1)}%", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold')
    plt.text(0.85, 0.69, f"MD: #{away_md}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_md))
    plt.text(0.85, 0.64, f"OFF: #{away_offensive}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_offensive))
    plt.text(0.85, 0.59, f"DEF: #{away_defensive}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_defensive))
    # plt.text(0.85, 0.54, f"ST: #{away_stm}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_stm))
    plt.text(0.85, 0.54, f"PBR: #{away_pbr}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_pbr))
    plt.text(0.85, 0.49, f"DCE: #{away_dce}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_dce))
    plt.text(0.85, 0.44, f"DDE: #{away_dde}", fontsize = 25, va='top', ha='right', transform=ax.transAxes, fontweight='bold', color=get_rank_color(away_dde))

    plt.text(0.65, 0.965, f"PEAR Game Preview", fontsize = 45, va='top', ha='center', transform=ax.transAxes, fontweight='bold')
    plt.text(0.65, 0.895, f"@PEARatings", fontsize=25, va='top', ha='center', transform=ax.transAxes)
    plt.text(0.65, 0.85, f"{home_team} vs. {away_team}", fontsize = 35, va='top', ha='center', transform=ax.transAxes, fontweight='bold')
    plt.text(0.65, 0.79, f"{formatted_spread}", fontsize = 25, va='top', ha='center', transform=ax.transAxes, fontweight='bold')
    # plt.text(0.65, 0.74, f"Win Prob: {game_win_prob}%", fontsize = 25, va='top', ha='center', transform=ax.transAxes, fontweight='bold')
    plt.text(0.65, 0.35, "Team Stats are Percentile Based", fontsize = 16, va='top', ha='center', transform=ax.transAxes, fontweight='bold')

    plt.tight_layout()

    folder_path = f"./PEAR/PEAR Football/y{current_year}/Visuals/week_{current_week}/Games"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{home_team} vs {away_team}")

    plt.savefig(file_path, bbox_inches='tight', dpi = 500)

def display_schedule_visual(team_name, all_data, full_display_schedule, uncompleted_games, uncompleted_conference_games, cSOS, logo_cache, logos, current_year, current_week):

    import matplotlib.colors as mcolors

    dark_green = '#1D4D00'
    medium_green = '#3C7300'
    orange = '#D2691E'
    black = '#000000'

    def interpolate_color(c1, c2, factor):
        """Interpolate between two hex colors based on factor (0 to 1)."""
        color1 = mcolors.hex2color(c1)
        color2 = mcolors.hex2color(c2)
        return mcolors.rgb2hex([(1 - factor) * a + factor * b for a, b in zip(color1, color2)])

    def at_least_color(team_probs):
        """
        Map team probabilities to hex colors.
        Values near 0.50 are green; further away become orange → black.
        Returns dict with same keys as team_probs.
        """
        result = {}
        for team, prob in team_probs.items():
            if prob < 0:
                prob = 0
            if prob > 1:
                prob = 1
            # Distance from 0.5, normalized to [0, 1]
            dist = abs(prob - 0.5) / 0.5  

            if dist <= 0.25:  
                # 0 → 0.25 : dark green → medium green
                factor = dist / 0.25
                color = interpolate_color(dark_green, medium_green, factor)
            elif dist <= 0.6:  
                # 0.25 → 0.6 : medium green → orange
                factor = (dist - 0.25) / (0.6 - 0.25)
                color = interpolate_color(medium_green, orange, factor)
            else:  
                # 0.6 → 1.0 : orange → black
                factor = (dist - 0.6) / (1.0 - 0.6)
                color = interpolate_color(orange, black, factor)

            result[team] = color
        return result
    
    def exact_color(team_probs):
        """
        Map team probabilities to hex colors.
        The highest probability is pinned at dark green, and others scale away
        toward medium green → orange → black as they get lower.
        Returns dict with same keys as team_probs.
        """
        result = {}
        max_prob = max(team_probs.values())

        for team, prob in team_probs.items():
            # Scale relative to max (so max=0, worst=1)
            dist = (max_prob - prob) / max_prob  

            if dist <= 0.25:
                # dark green → medium green
                factor = dist / 0.25
                color = interpolate_color(dark_green, medium_green, factor)
            elif dist <= 0.6:
                # medium green → orange
                factor = (dist - 0.25) / (0.6 - 0.25)
                color = interpolate_color(medium_green, orange, factor)
            else:
                # orange → black
                factor = (dist - 0.6) / (1.0 - 0.6)
                color = interpolate_color(orange, black, factor)

            result[team] = color
        return result

    team_data = all_data[all_data['team'] == team_name].reset_index(drop=True)
    team_conference = get_team_column_value(all_data, team_name, "conference")
    team_idx = all_data[all_data['team'] == team_name].index[0] + 1
    team_uncompleted_conf_games = uncompleted_conference_games[(uncompleted_conference_games['home_team'] == team_name) | (uncompleted_conference_games['away_team'] == team_name)]
    current_wins = team_data['wins'][0]
    current_losses = team_data['losses'][0]
    current_conf_wins = team_data['conference_wins'][0]
    current_conf_losses = team_data['conference_losses'][0]
    conference_games = len(team_uncompleted_conf_games)+current_conf_wins+current_conf_losses
    team_probs = team_win_probs(team_name, current_wins, uncompleted_games, 12)
    team_exact_probs = team_exact_win_probs(team_name, current_wins, uncompleted_games)
    expected_wins = round(sum(wins * prob for wins, prob in team_exact_probs.items()),1)
    team_conf_probs = team_win_probs(team_name, current_conf_wins, uncompleted_conference_games, conference_games)
    team_exact_conf_probs = team_exact_win_probs(team_name, current_conf_wins, uncompleted_conference_games)
    conference_expected_wins = round(sum(wins * prob for wins, prob in team_exact_conf_probs.items()),1)
    team_schedule = full_display_schedule[(full_display_schedule['home_team'] == team_name) | (full_display_schedule['away_team'] == team_name)]
    display_schedule = transform_schedule(team_schedule, team_name)

    conf_exp_wins = {}
    for other_team in all_data.loc[all_data['conference'] == team_conference, 'team']:
        current_conf_wins_everyone = all_data.loc[all_data['team'] == other_team, 'conference_wins'].iloc[0]
        probs_other = team_exact_win_probs(other_team, current_conf_wins_everyone, uncompleted_conference_games)
        exp_wins = round(sum(w * p for w, p in probs_other.items()), 1)
        conf_exp_wins[other_team] = exp_wins
    conf_df = pd.DataFrame(list(conf_exp_wins.items()), columns=['team', 'expected_wins'])
    conf_df['rank'] = conf_df['expected_wins'].rank(method='min', ascending=False).astype(int)
    team_conf_rank = conf_df.loc[conf_df['team'] == team_name, 'rank'].iloc[0]

    opp_col="opponent"
    figsize=(20,10)
    zoom=0.1
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor("#CECEB2")

    # ------------------------------------------------------------
    # Columns for schedule
    # ------------------------------------------------------------
    x = 0.05  # fixed column
    y_start = len(display_schedule)  # top row
    y = y_start
    max_box=0.655 
    ax.text(0.015, y_start+0.7, f'WK', fontsize=16, ha='center', va='center', fontweight='bold')
    ax.text(x+0.03, y_start+0.7, f'LOC', fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(x+0.09, y_start+0.7, f"RK", fontsize=16, ha='right', va='center', fontweight='bold')
    ax.text(x+0.105, y_start+0.7, f"OPPONENT", fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(x+0.28, y_start+0.7, f"OFF", fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(x+0.32, y_start+0.7, f"DEF", fontsize=16, ha='left', va='center', fontweight='bold')
    ax.text(x+0.57, y_start+0.7, f"GQI", fontsize=16, ha='left', va='center', fontweight='bold')

    # ------------------------------------------------------------
    # displaying the schedule
    # ------------------------------------------------------------
    for _, row in display_schedule.iterrows():
        opp = row[opp_col]
        conf_game = row['conference_game']
        if opp in all_data['team'].values:
            opp_idx = all_data[all_data['team'] == opp].index[0] + 1
        else:
            opp_idx = ""
        week = row['week']
        location = row["location"]
        opp_off = get_team_column_value(all_data, opp, "offensive_rank")
        opp_def = get_team_column_value(all_data, opp, "defensive_rank")
        completed = row['completed']
        won = row['win']
        if completed and won:
            row_color = "palegreen"
        elif completed and not won:
            row_color = "lightcoral"
        else:
            if location == "":
                row_color = "whitesmoke"
            elif location == "AT":
                row_color = "lightblue"
            else:
                row_color = "lightgoldenrodyellow"
        img = logo_cache.get(opp)
        rect = patches.Rectangle(
            (0, y - 0.5),  # left-bottom corner of row
            max_box,             # width
            0.96,               # height of row
            linewidth=0,
            facecolor=row_color,
            zorder=0
        )
        ax.add_patch(rect)

        if img is not None and opp != "Alcorn State":
            im = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(im, (x, y-0.02), frameon=False)
            ax.add_artist(ab)
        if img is None:
            ax.text(x, y, opp, ha="center", va="center")

        loc_x, rank_x = x+0.04, x+0.09
        ax.text(0.024, y-0.04, f'{week}', fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(loc_x, y-0.04, f'{location}', fontsize=20, ha='center', va='center', fontweight='bold')
        ax.text(rank_x, y-0.04, f"{opp_idx}", fontsize=20, ha='right', va='center', fontweight='bold')

        ax.text(x+0.105, y-0.04, f"{opp}", fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(x+0.28, y-0.04, f"{opp_off}", fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(x+0.32, y-0.04, f"{opp_def}", fontsize=20, ha='left', va='center', fontweight='bold')

        if conf_game:
            box_x = loc_x - 0.015     # small padding to the left
            box_y = y - 0.321         # adjust vertical alignment
            box_width = (rank_x - loc_x) - 0.02
            box_height = 0.6
            conf_rect = patches.Rectangle(
                (box_x, box_y),
                box_width,
                box_height,
                linewidth=2,
                edgecolor="black",
                facecolor="#DDA0DD",
                zorder=2
            )
            ax.add_patch(conf_rect)

        if completed:
            ax.text(x+0.36, y-0.04, f"{row['score']}", fontsize=20, ha='left', va='center', fontweight='bold')
        else:
            text_value = row['PEAR']
            if team_name in text_value:
                pear_color = "#267326"
            else:
                pear_color = "#993d3d"
            fontsize = 20  # default size
            if "Florida International" in text_value:
                fontsize = 18
            ax.text(x + 0.36, y-0.04,text_value,fontsize=fontsize,ha="left",va="center",fontweight="bold", color=pear_color)
            ax.text(x+0.57, y-0.04, f"{row['GQI']}", fontsize=20, ha='left', va='center', fontweight='bold')

        y -= 1

    # ------------------------------------------------------------
    # displaying team name
    # ------------------------------------------------------------
    rect = patches.Rectangle((0.0, 0.48), max_box,y_start - y + 0,linewidth=2,edgecolor='black',facecolor='none')
    ax.add_patch(rect)
    logo_color = logos[logos['team'] == team_name]['color'].values[0]
    alt_color = logos[logos['team'] == team_name]['alt_color'].values[0]
    rect_width = 0.34
    rect_height = 1
    rect_x = 1 - rect_width  # small right margin
    rect_y = len(display_schedule) + 0.48 - rect_height  # slightly below top edge
    color_rect = patches.Rectangle((rect_x, rect_y),rect_width,rect_height,linewidth=2,edgecolor="black",facecolor=logo_color,zorder=4)
    ax.add_patch(color_rect)
    ax.text(rect_x + rect_width/2,rect_y + rect_height/2,f'{team_name}', color="white", ha='center', va='center', fontsize=32, fontweight='bold',zorder=5)

    # ------------------------------------------------------------
    # rectangles for overall win percentage win percentage
    # ------------------------------------------------------------
    gap_y = 0.8            # vertical gap between color_rect and down boxes
    center_gap = 0.005       # horizontal gap in the middle
    down_height = 7.9
    down_rect_y = rect_y - down_height - gap_y  # below color_rect with gap
    left_rect = patches.Rectangle(
        (rect_x, down_rect_y),                      # left starts at rect_x
        rect_width / 2 - center_gap / 2,            # shrink to leave center gap
        down_height,
        linewidth=2,
        edgecolor="black",
        facecolor="#E2CDE2",
        zorder=5
    )
    ax.add_patch(left_rect)
    numbers = ["WINS"] + list(range(13, -1, -1))
    left_x = rect_x
    left_y = down_rect_y
    left_w = rect_width / 2 - center_gap / 2
    left_h = down_height
    padding = left_h / (2 * len(numbers))  # small vertical inset
    y_positions = np.linspace(
        left_y + left_h - padding, 
        left_y + padding, 
        len(numbers)
    )

    colors = at_least_color(team_probs)
    exact_colors_dict = exact_color(team_exact_probs)

    for num, y in zip(numbers, y_positions):
        value = team_probs.get(num, 0)
        color = colors.get(num, 'black')
        exact = team_exact_probs.get(num,0)
        this_exact_color = exact_colors_dict.get(num, 'black')
        display_prob = format_prob(value)
        if str(num) == str(current_wins):
            display_prob = '100%'
        exact_prob = format_prob(exact)
        ax.text(left_x+0.03,y,str(num),ha="center",va="center",fontsize=24,color="black",fontweight='bold',zorder=6)
        ax.text(left_x + left_w / 2,y,str(display_prob),ha="center",va="center",fontsize=24,color=color,fontweight='bold',zorder=6)
        ax.text(left_x+0.14,y,str(exact_prob),ha="center",va="center",fontsize=24,color=this_exact_color,fontweight='bold',zorder=6)
        if num == "WINS":
            ax.text(left_x + left_w / 2,y,">=",ha="center",va="center",fontsize=24,color="black",fontweight='bold',zorder=6)
            ax.text(left_x+0.14,y,"=",ha="center",va="center",fontsize=24,color="black",fontweight='bold',zorder=6)

    # ------------------------------------------------------------
    # rectangles for overall expected record
    # ------------------------------------------------------------
    intermediate_height = gap_y - 0.1  # height of new boxes (slightly smaller than gap)
    intermediate_y = down_rect_y + down_height + 0.05  # just below color_rect, leaving small margin
    # Left intermediate box
    left_intermediate = patches.Rectangle(
        (rect_x, intermediate_y),                        # same left x
        rect_width / 2 - center_gap / 2,                # width same as left box
        intermediate_height,                             # height smaller than gap
        linewidth=2,
        edgecolor="black",
        facecolor="black",                             # choose a neutral fill or highlight
        zorder=5
    )
    ax.add_patch(left_intermediate)
    left_x = rect_x
    left_w = rect_width / 2 - center_gap / 2
    col_offsets = [0.03, left_w / 2, 0.11, 0.14]  # "OVR", expected wins, 12-expected
    texts = ["OVR", f"{expected_wins:.1f}", "-", f"{12 - expected_wins:.1f}"]
    y_center = intermediate_y + intermediate_height / 2
    for text, offset in zip(texts, col_offsets):
        ax.text(left_x + offset, y_center, text, ha="center", va="center",
                fontsize=24, fontweight="bold", color="white", zorder=6)

    # ------------------------------------------------------------
    # rectangles for conference win percentages
    # ------------------------------------------------------------
    right_down_height = 6  # desired height
    right_rect_top = down_rect_y + down_height - right_down_height  # align top with left
    right_rect = patches.Rectangle(
        (rect_x + rect_width / 2 + center_gap / 2, right_rect_top),  # bottom-left corner
        rect_width / 2 - center_gap / 2,                              # width
        right_down_height,                                            # height
        linewidth=2,
        edgecolor="black",
        facecolor="#E2CDE2",
        zorder=5
    )
    ax.add_patch(right_rect)
    numbers = ["WINS"] + list(range(9, -1, -1))
    right_x = rect_x + rect_width / 2 + center_gap / 2
    right_w = rect_width / 2 - center_gap / 2
    right_h = right_down_height
    right_rect_bottom = down_rect_y + down_height - right_h  # top aligned with left
    padding = right_h / (2 * len(numbers))  # small vertical inset
    # Evenly space numbers vertically
    y_positions = np.linspace(
        right_rect_bottom + right_h - padding,  # top of rectangle minus small padding
        right_rect_bottom + padding,            # bottom of rectangle plus small padding
        len(numbers)
    )
    colors = at_least_color(team_conf_probs)
    exact_colors_dict = exact_color(team_exact_conf_probs)
    for num, y in zip(numbers, y_positions):
        # Grab probability values
        value = team_conf_probs.get(num, 0)           # probability >= X conf wins
        color = colors.get(num, 'black')
        exact = team_exact_conf_probs.get(num, 0)     # probability = X conf wins
        this_exact_color = exact_colors_dict.get(num, 'black')
        display_prob = format_prob(value)
        if str(num) == str(current_conf_wins):
            display_prob = '100%'
        exact_prob = format_prob(exact)
        # Column headers
        ax.text(right_x + 0.03, y, str(num), ha="center", va="center", fontsize=24, color="black", fontweight='bold', zorder=6)
        ax.text(right_x + right_w / 2, y, str(display_prob), ha="center", va="center", fontsize=24, color=color, fontweight='bold', zorder=6)
        ax.text(right_x + 0.14, y, str(exact_prob), ha="center", va="center", fontsize=24, color=this_exact_color, fontweight='bold', zorder=6)
        # Add symbols for header row
        if num == "WINS":
            ax.text(right_x + right_w / 2, y, ">=", ha="center", va="center", fontsize=24, color="black", fontweight='bold', zorder=6)
            ax.text(right_x + 0.14, y, "=", ha="center", va="center", fontsize=24, color="black", fontweight='bold', zorder=6)

    # ------------------------------------------------------------
    # rectangles for expected conference record
    # ------------------------------------------------------------
    right_intermediate = patches.Rectangle(
        (rect_x + rect_width / 2 + center_gap / 2, intermediate_y),  # start after center gap
        rect_width / 2 - center_gap / 2,                              # width same as right box
        intermediate_height,
        linewidth=2,
        edgecolor="black",
        facecolor="black",
        zorder=5
    )
    ax.add_patch(right_intermediate)
    right_x = rect_x + rect_width / 2 + center_gap / 2
    right_w = rect_width / 2 - center_gap / 2
    col_offsets = [0.03, right_w / 2, 0.11, 0.14]  # "CONF", expected wins, "-", losses
    texts = ["CONF", f"{conference_expected_wins:.1f}", "-", f"{conference_games - conference_expected_wins:.1f}"]
    y_center = intermediate_y + intermediate_height / 2
    for text, offset in zip(texts, col_offsets):
        ax.text(right_x + offset, y_center, text, ha="center", va="center",
                fontsize=24, fontweight="bold", color="white", zorder=6)
        

    # ------------------------------------------------------------
    # rectangles and data for ranking information
    # ------------------------------------------------------------
    right_rect_y = right_rect_top
    right_rect_height = 4
    right_rect_x = rect_x + rect_width / 2 + center_gap / 2

    # Place new rectangle below it
    new_rect_height = 4.0
    new_rect_y = right_rect_y - right_rect_height - 0.05   # just below right rect
    new_rect_color = "#FFF8E1"

    right_new_rect = patches.Rectangle(
        (right_rect_x, new_rect_y),
        rect_width / 2 - center_gap / 2,   # same width as right rect
        new_rect_height,
        linewidth=2,
        edgecolor="black",
        facecolor=new_rect_color,
        zorder=5
    )
    ax.add_patch(right_new_rect)

    # Data values
    offensive = get_team_column_value(all_data, team_name, "offensive_rank")
    defensive = get_team_column_value(all_data, team_name, "defensive_rank")
    offensive_total = round(get_team_column_value(all_data, team_name, "offensive_total"), 1)
    defensive_total = round(get_team_column_value(all_data, team_name, "defensive_total"), 1)
    power_rating = get_team_column_value(all_data, team_name, "power_rating")
    most_deserving = get_team_column_value(all_data, team_name, "most_deserving")
    SOS = get_team_column_value(all_data, team_name, "SOS")
    SOS_wins = round(get_team_column_value(all_data, team_name, "avg_expected_wins"),2)
    playoff_prob = round(get_team_column_value(all_data, team_name, "prob_reach_wins"))
    playoff_rank = round(get_team_column_value(all_data, team_name, "playoff_rank"))
    cSOS_value = get_team_column_value(cSOS, team_name, 'cSOS')
    cSOS_wins = round(get_team_column_value(cSOS, team_name, 'avg_expected_wins'),2)

    # Geometry for right side
    right_x = rect_x + rect_width / 2 + center_gap / 2
    right_w = rect_width / 2 - center_gap / 2
    y_top = new_rect_y + new_rect_height

    # Column layout
    col_offsets = [0.03, right_w / 2, right_w - 0.03]
    column_names = ["", "RK", "RTG"]

    # Header row
    y_header = y_top - 0.35
    for name, offset in zip(column_names, col_offsets):
        ax.text(right_x + offset, y_header, name, ha="center", va="center",
                fontsize=18, fontweight="bold", color="black", zorder=6)

    # Data rows
    row_labels = ["OVR", "OFF", "DEF", "SOS", "cSOS", "POF", "MD", "cFIN"]
    row_values = [
        [team_idx, power_rating],
        [offensive, offensive_total],
        [defensive, defensive_total],
        [SOS, SOS_wins],
        [cSOS_value, cSOS_wins],
        [playoff_rank, f'{playoff_prob}%'],
        [most_deserving, ""],
        [team_conf_rank, ""]
    ]
    num_items = 8
    spacing = 0.42
    y_positions = [y_header - spacing - i * spacing for i in range(num_items)]

    for y, label, vals in zip(y_positions, row_labels, row_values):
        ax.text(right_x + col_offsets[0], y, label, ha="center", va="center",
                fontsize=18, fontweight="bold", color="black", zorder=6)
        for val, offset in zip(vals, col_offsets[1:]):
            ax.text(right_x + offset, y, f"{val}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color="black", zorder=6)

    img = logo_cache.get(team_name)
    zoom = 0.25  # adjust as needed
    x_img = rect_x + 0.125 # right edge of left rectangle
    y_img = new_rect_y - 0.35 # bottom of rectangle
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (x_img, y_img), frameon=False,
                        xycoords='data', box_alignment=(1, 0))  # align bottom-right
    ax.add_artist(ab)

    conference_games = display_schedule[display_schedule['conference_game'] == True]
    xWins = display_schedule[display_schedule['completed'] == True]['team_win_prob'].sum().round(1)
    xLosses = round(len(display_schedule[display_schedule['completed'] == True]) - xWins,1)
    xConfWins = conference_games[conference_games['completed'] == True]['team_win_prob'].sum().round(1)
    xConfLosses = round(len(conference_games[conference_games['completed'] == True]) - xConfWins, 1)

    ax.text(0.67, y_start+0.7, f"R: {current_wins} - {current_losses} ({current_conf_wins} - {current_conf_losses})", ha='left', va='center', fontsize=16, fontweight='bold')
    ax.text(0.99, y_start+0.7, f"xR: {xWins} - {xLosses} ({xConfWins} - {xConfLosses})", ha='right', va='center', fontsize=16, fontweight='bold')
    ax.text(0.005, 0.15, "Graphic by @PEARatings | Inspired by @KFordRatings | Underlying Data from @CFB_Data", fontsize=16, ha='left', va='center', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(display_schedule) + 1)
    ax.axis("off")
    plt.tight_layout()
    folder_path = f"./PEAR/PEAR Football/y{current_year}/Visuals/week_{current_week}/Stat Profiles"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{team_name}")
    plt.savefig(file_path, dpi = 500)
    # return fig, ax

def conference_standings(projection_dataframe, records, team_data, team_logos, conf_folder_path):
    conference_list = list(team_data['conference'].unique())
    all_figs = []
    num_conference_games = {
        'SEC':8,
        'Big Ten':9,
        'ACC':8,
        'Big 12':9,
        'Mountain West':8,
        'American Athletic':8,
        'Pac-12':1,
        'Sun Belt':8,
        'Mid-American':8,
        'Conference USA':8
    }

    import matplotlib.colors as mcolors

    dark_green = '#1D4D00'
    medium_green = '#3C7300'
    orange = '#D2691E'
    black = 'black'

    # Function to interpolate between colors
    def interpolate_color(c1, c2, factor):
        """Interpolate between two colors c1 and c2 based on factor (0 to 1)."""
        color1 = mcolors.hex2color(c1)
        color2 = mcolors.hex2color(c2)
        return mcolors.rgb2hex([(1 - factor) * a + factor * b for a, b in zip(color1, color2)])

    # Function to get color based on percentage
    def get_color_percentage(percentage):
        if 40 <= percentage <= 60:
            return dark_green  # Dark green is constant in this range
        elif 30 <= percentage < 40:
            # Transition from orange to medium green as percentage increases from 30 to 45
            factor = (percentage - 30) / (40 - 30)
            return interpolate_color(orange, medium_green, factor)
        elif 60 < percentage <= 70:
            # Transition from medium green to orange as percentage increases from 55 to 70
            factor = (percentage - 60) / (70 - 60)
            return interpolate_color(medium_green, orange, factor)
        elif 15 <= percentage < 30:
            # Transition from black to orange as percentage increases from 15 to 30
            factor = (percentage - 15) / (30 - 15)
            return interpolate_color(black, orange, factor)
        elif 70 < percentage <= 85:
            # Transition from orange to black as percentage increases from 70 to 85
            factor = (percentage - 70) / (85 - 70)
            return interpolate_color(orange, black, factor)
        else:
            return black  # Anything outside the defined ranges is black

    for conference in conference_list:
        if (conference == 'FBS Independents') | (conference == 'Pac-12'):
            continue
        this_conference_wins = projection_dataframe[projection_dataframe['conference'] == conference].sort_values('expected_wins', ascending=False).reset_index()

        this_conference = conference
        this_conference_games = num_conference_games[this_conference]
        # Create a figure with a reduced height and smaller logos
        fig, axs = plt.subplots(len(this_conference_wins), 1, figsize=(4, len(this_conference_wins) * 0.4))  # Adjusted height
        fig.patch.set_facecolor('#CECEB2')

        # Loop through each team in the conference
        for i, ax in enumerate(axs.ravel()):
            # Get the team logo URL
            img = team_logos[this_conference_wins.loc[i, 'team']]
            
            # Display the team logo with smaller size
            ax.imshow(img, extent=(1,1.01,1.01,1), alpha=0.9)  # Adjust extent for smaller logo
            ax.set_facecolor('#f0f0f0')
            ax.axis('off')
            
            # Get the actual number of conference wins for the team
            games_won = records[records['team'] == this_conference_wins.loc[i, 'team']]['conference_wins'].values[0]
            games_lost = records[records['team'] == this_conference_wins.loc[i, 'team']]['conference_losses'].values[0]
            expected_conference_wins = round(this_conference_wins.loc[i, 'expected_wins'], 1)

            # Calculate cumulative probabilities of winning at least X games
            win_columns = [f'win_{j}' for j in range(10)]  # win_0 to win_9
            cumulative_probs = this_conference_wins.loc[i, win_columns].values * 100  # Just grab the values and multiply by 100
            
            # Display cumulative win probabilities (at least X games)
            for j in range(this_conference_games, games_won - 1, -1):  # Only for win totals >= games_won
                if j == games_won:
                    ax.text(15 + 2 * (this_conference_games - j), 0.5, "✔", transform=ax.transAxes, fontsize=12, ha='center', color='green', va='center', fontproperties=checkmark_font)  # Display checkmark
                elif cumulative_probs[j] == 0:
                    ax.text(15 + 2 * (this_conference_games - j), 0.5, "X", transform=ax.transAxes, fontsize=12, ha='center', color='red', va='center', fontweight='bold')  # Display X if probability is zero
                elif round(cumulative_probs[j]) == 100:
                    continue
                elif round(cumulative_probs[j]) == 0:
                    ax.text(15 + 2 * (num_conference_games[this_conference] - j), 0.5, f"", transform=ax.transAxes, fontsize=12, ha='center', va='center', fontweight='bold')
                else:
                    ax.text(15 + 2 * (num_conference_games[this_conference] - j), 0.5, f"{round(cumulative_probs[j])}%", transform=ax.transAxes, fontweight='bold', fontsize=12, ha='center', va='center', color = get_color_percentage(cumulative_probs[j]))

            ax.text(-1.25, 0.5, f"{i+1}", transform=ax.transAxes, fontsize=12, va='center', fontweight='bold', ha='center')
            ax.text(9, 0.5, f"{games_won} - {games_lost}", transform=ax.transAxes, fontsize=12, va='center')
            ax.text(12, 0.5, f"{expected_conference_wins}", transform=ax.transAxes, fontsize=12, va='center')
            # Display the team name next to the logo
            ax.text(1.5, 0.5, f"{this_conference_wins.loc[i, 'team']}", transform=ax.transAxes, fontsize=12, va='center', fontweight='bold')
        fig.text(0.41, 0.9, f"RK", fontsize=12, fontweight='bold', ha='center')
        fig.text(0.64, 0.9, f"TEAM", fontsize=12, fontweight='bold', ha='center')
        fig.text(1.11, 0.9, f"REC", fontsize=12, fontweight='bold', ha='center')
        fig.text(1.29, 0.9, f"AVG", fontsize=12, fontweight='bold', ha='center')

        j=0
        while this_conference_games >= 0:
            fig.text(1.44 + 0.132*j, 0.9, f"{this_conference_games}", fontsize=12, fontweight='bold', ha='center')
            this_conference_games -= 1
            j+=1
        fig.text(0.38, 0.98, f"PEAR PROJECTED {this_conference.upper()} STANDINGS", fontsize=16, fontweight='bold', ha='left')
        fig.text(0.38, 0.95, "PERCENT CHANCE TO WIN AT LEAST _ CONFERENCE GAMES", fontsize =10, ha='left')
        file_path = os.path.join(conf_folder_path, f"{this_conference}")
        plt.savefig(file_path, bbox_inches='tight', dpi = 500)






















