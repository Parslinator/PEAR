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
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

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

current_year = int(current_year)
current_week = int(current_week)
print(f"Current Week: {current_week}, Current Year: {current_year}")
print("Double Check The Current Week To Make Sure It Is Correct")

def date_sort(game):
    game_date = datetime.datetime.strptime(game['start_date'], "%Y-%m-%dT%H:%M:%S.000Z")
    return game_date

def PEAR_Win_Prob(home_power_rating, away_power_rating):
    return round((1 / (1 + 10 ** ((away_power_rating - (home_power_rating)) / 20.5))) * 100, 2)

def average_team_distribution(num_simulations, schedules, average, team_name):

    def simulate_game_average(win_prob):
        random_outcome = np.random.random() * 100  # Generates a number between 0 and 100
        if random_outcome < win_prob:
            return "W"  # Home team wins, Away team loses
        else:
            return "L"  # Away team wins, Home team loses
        
    def simulate_season_average(schedules, team_name, average):
        wins = 0
        losses = 0
        for _, game in schedules.iterrows():
            if game['home_team'] == team_name:
                opponent_team = game['away_team']
                opponent_pr = game['away_pr']
                win_prob = PEAR_Win_Prob(average, opponent_pr)

                # opponent_elo = game['away_elo']
                # win_prob = round((10**((average-opponent_elo) / 400)) / ((10**((average-opponent_elo) / 400)) + 1)*100, 2)
            else:
                opponent_team = game['home_team']
                opponent_pr = game['home_pr']
                win_prob = 100 - PEAR_Win_Prob(opponent_pr, average)

                # opponent_elo = game['home_elo']
                # win_prob = 100 - round((10**((opponent_elo-average) / 400)) / ((10**((opponent_elo-average) / 400)) + 1)*100, 2)
            
            outcome = simulate_game_average(win_prob)
            if outcome == "W":
                wins += 1
            else:
                losses += 1

        return wins, losses
        
    def monte_carlo_simulation_average(num_simulations, schedules, average, team_name):
        """Runs a Monte Carlo simulation for an average team over multiple seasons."""
        win_results = []
        loss_results = []

        for _ in range(num_simulations):
            wins, losses = simulate_season_average(schedules, team_name, average)
            win_results.append(wins)
            loss_results.append(losses)
        
        return win_results, loss_results

    import statistics
    from collections import Counter
    def analyze_simulation_average(win_results, loss_results, schedules):
        games_played = len(schedules)
        if games_played == 11:
            win_results = [x + .948 for x in win_results]
        elif games_played == 10:
            win_results = [x + (2 * .948) for x in win_results]
    
        avg_wins = statistics.mean(win_results)
        avg_loss = statistics.mean(loss_results)
        most_common_win = statistics.mode(win_results)
        most_common_loss = statistics.mode(loss_results)


        win_counts = Counter(win_results)    
        total_simulations = len(win_results)
        win_percentages = {f"win_{wins}": (win_counts[wins] / total_simulations) for wins in range(13)}
        win_thresholds = pd.DataFrame([win_percentages])
        
        # win_thresholds = {}
        # for wins in range(13):  # 0 to 12 wins
        #     win_thresholds[f'win_{wins}'] = win_df.apply(lambda x: (x == wins).sum() / len(x), axis=0)

        win_thresholds['WIN6%'] = win_thresholds.loc[:, 'win_6':'win_12'].sum(axis=1)
        win_thresholds['expected_wins'] = avg_wins
        win_thresholds['expected_losses'] = avg_loss
        win_thresholds['projected_wins'] = most_common_win
        win_thresholds['projected_losses'] = most_common_loss

        return win_thresholds
    
    avg_win, avg_loss = monte_carlo_simulation_average(num_simulations, schedules, average, team_name)
    win_thresholds = analyze_simulation_average(avg_win, avg_loss,schedules)
    return win_thresholds


if postseason:
    elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year)]
else:
    elo_ratings_list = [*ratings_api.get_elo_ratings(year=current_year, week=current_week)]
elo_ratings_dict = [dict(
    team = e.team,
    elo = e.elo
) for e in elo_ratings_list]
elo_ratings = pd.DataFrame(elo_ratings_dict)

# returning production
production_list = []
response = players_api.get_returning_production(year = current_year)
production_list = [*production_list, *response]
production_dict = [dict(
    season=r.season,
    team=r.team,
    returning_ppa=r.percent_ppa,
    returning_usage=r.usage
) for r in production_list]
returning_production = pd.DataFrame(production_dict)

# team records
records_list = []
response = games_api.get_team_records(year=current_year)
records_list = [*records_list, *response]
records_dict = [dict(
    team = r.team,
    games_played = r.total.games,
    wins = r.total.wins,
    losses = r.total.losses,
    conference_games = r.conference_games.games,
    conference_wins = r.conference_games.wins,
    conference_losses = r.conference_games.losses
) for r in records_list]
records = pd.DataFrame(records_dict)

# qb ppa
qb_ppa_list = []
response = metrics_api.get_player_season_ppa(year=current_year, position = 'QB', threshold = 10)
qb_ppa_list = [*qb_ppa_list, *response]
qb_ppa_dict = [dict(
    team = q.team,
    qb_average_ppa = q.average_ppa._pass,
    qb_total_ppa = q.total_ppa._pass
) for q in qb_ppa_list]
qb_ppa = pd.DataFrame(qb_ppa_dict)
qb_ppa = qb_ppa.groupby('team', as_index=False).mean()

# fpi ranks
team_fpi_list = []
response = ratings_api.get_fpi_ratings(year = current_year)
team_fpi_list = [*team_fpi_list, *response]
team_fpi_dict = [dict(
    team = f.team,
    fpi = f.fpi,
    fpi_rank = f.resume_ranks.fpi,
    fpi_sor = f.resume_ranks.strength_of_record,
    fpi_sos = f.resume_ranks.strength_of_schedule,
    def_eff = f.efficiencies.defense,
    off_eff = f.efficiencies.offense,
    special_eff = f.efficiencies.special_teams
) for f in team_fpi_list]
team_fpi = pd.DataFrame(team_fpi_dict)

# srs ranks
team_srs_list = []
response = ratings_api.get_srs_ratings(year = current_year)
team_srs_list = [*team_srs_list, *response]
team_srs_dict = [dict(
    team = f.team,
    srs = f.rating,
    srs_rank = f.ranking
) for f in team_srs_list]
team_srs = pd.DataFrame(team_srs_dict).dropna().drop_duplicates()

# sp ranks
team_sp_list = []
response = ratings_api.get_sp_ratings(year=current_year)
team_sp_list = [*team_sp_list, *response]
team_sp_dict = [dict(
    team = t.team,
    ranking = t.ranking,
    sp_rating = t.rating
) for t in team_sp_list]
team_sp = pd.DataFrame(team_sp_dict).dropna()

# logo info
logos_info_list = []
response = teams_api.get_teams()
logos_info_list = [*logos_info_list, *response]
logos_info_dict = [dict(
    team = l.school,
    color = l.color,
    alt_color = l.alt_color,
    logo = l.logos
) for l in logos_info_list]
logos_info = pd.DataFrame(logos_info_dict)
logos_info = logos_info.dropna(subset=['logo', 'color'])

# advanced metrics
advanced_metrics_response = []
response = advanced_instance.get_advanced_team_season_stats(year = current_year)
advanced_metrics_response = [*advanced_metrics_response, *response]
advanced_metrics = pd.DataFrame()
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
for i in range(len(advanced_metrics_response)):
    data = advanced_metrics_response[i].to_dict() if hasattr(advanced_metrics_response[i], 'to_dict') else vars(advanced_metrics_response[i])

    offense_stats = flatten_dict(data['offense'], parent_key='Offense')
    defense_stats = flatten_dict(data['defense'], parent_key='Defense')
    combined_data = {
        'team':data['team'],
        **offense_stats,
        **defense_stats
    }
    df = pd.DataFrame([combined_data])
    advanced_metrics = pd.concat([advanced_metrics, df], ignore_index=True)
columns_to_keep = ['team', 'Offense_success_rate', 'Defense_success_rate', 
                'Offense_explosiveness', 'Defense_explosiveness', 'Offense_ppa', 'Offense_power_success', 'Offense_stuff_rate', 'Defense_power_success', 'Defense_stuff_rate',
                'Defense_ppa', 'Offense_points_per_opportunity', 'Defense_points_per_opportunity', 'Defense_havoc_total', 
                'Offense_field_position_average_predicted_points', 'Defense_field_position_average_predicted_points',
                'Offense_field_position_average_start', 'Defense_field_position_average_start']
metrics = advanced_metrics[columns_to_keep]

# conference sp
conference_sp = []
response = ratings_api.get_conference_sp_ratings(year=current_year)
conference_sp = [*conference_sp, *response]

sp_conf = [dict(
    conference=c.conference,
    season=c.year,
    sp_conf_rating=c.rating
) for c in conference_sp]
conference_sp_rating = pd.DataFrame(sp_conf)

# team stats
team_stats_list = []
response = advanced_instance.get_team_season_stats(year=current_year)
team_stats_list = [*team_stats_list, *response]
team_stats_dict = [dict(
    team = s.team,
    stat_name = s.stat_name,
    stat_value = s.stat_value
) for s in team_stats_list]
team_stats = pd.DataFrame(team_stats_dict)
team_stats = team_stats.pivot(index='team', columns='stat_name', values='stat_value').reset_index().fillna(0)
team_stats['total_turnovers'] = team_stats['fumblesRecovered'] + team_stats['passesIntercepted'] - team_stats['turnovers']
team_stats['thirdDownConversionRate'] = round(team_stats['thirdDownConversions'] / team_stats['thirdDowns'],4)
team_stats['fourthDownConversionRate'] = round(team_stats['fourthDownConversions'] / team_stats['fourthDowns'], 4)
team_stats['possessionTimeMinutes'] = round(team_stats['possessionTime'] / 60,2)

# team talent
talent_list = []
for year in range(current_year-3, current_year+1):
    response = teams_api.get_talent(year=year)
    talent_list = [*talent_list, *response]
talent_dict = [dict(
    team=t.school,
    season=t.year,
    talent=t.talent
) for t in talent_list]
talent = pd.DataFrame(talent_dict)

# team info
team_info_list = [*teams_api.get_fbs_teams()]
team_dict = [dict(
    team = t.school,
    conference = t.conference
) for t in team_info_list]
team_info = pd.DataFrame(team_dict)

# team recruiting
recruiting_info_list = []
for year in range(current_year-3, current_year+1):
    response = recruiting_api.get_recruiting_teams(year=year)
    recruiting_info_list = [*recruiting_info_list, *response]
recruiting_info_dict = [dict(
    team = r.team,
    year = r.year,
    points = r.points
) for r in recruiting_info_list]
recruiting = pd.DataFrame(recruiting_info_dict)

print("Data Load Done")

# entire talent profile for the team over the last three years
last_three_rows = talent.groupby('team').tail(3)
avg_talent_per_team = last_three_rows.groupby('team')['talent'].mean().reset_index()
avg_talent_per_team.columns = ['team', 'avg_talent']

# Recruiting points over the last 4 years
last_three = recruiting.groupby('team').tail(3)
recruiting_per_team = last_three.groupby('team')['points'].sum().reset_index()
recruiting_per_team.columns = ['team', 'avg_points']
recruiting_per_team['avg_points'] = recruiting_per_team['avg_points'] + 150

intermediate_1 = pd.merge(team_info, avg_talent_per_team, how='left', on='team')
intermediate_2 = pd.merge(intermediate_1, conference_sp_rating, how='left', on='conference')
intermediate_3 = pd.merge(intermediate_2, team_stats, how='left', on='team')
intermediate_4 = pd.merge(intermediate_3, logos_info, how='left', on='team')
intermediate_5 = pd.merge(intermediate_4, qb_ppa, how='left', on='team')
intermediate_6 = pd.merge(intermediate_5, team_fpi, how='left', on='team')
intermediate_7 = pd.merge(intermediate_6, records, how='left', on='team')
team_data = pd.merge(intermediate_7, metrics, how='left', on='team')

# For military schools and new FBS schools, use recruiting points instead of team talent
# New FBS Schools - you get on here for 3 years
target_teams = ['Air Force', 'Army', 'Navy', 'Kennesaw State', 'Jacksonville State', 'Sam Houston']
mask = team_data['team'].isin(target_teams)
team_data.loc[mask, 'avg_talent'] = team_data.loc[mask, 'team'].map(
    recruiting_per_team.set_index('team')['avg_points']
)

print("Data Formatting Done")
print("Starting Optimization")

############ IN HOUSE PR #################

# All the scalers used for the team data
scaler100 = MinMaxScaler(feature_range=(1, 100))
scaler60 = MinMaxScaler(feature_range=(40,98.8))
scaler10 = MinMaxScaler(feature_range=(1,10))
scalerTurnovers = MinMaxScaler(feature_range=(1, 100))
scalerPenalties = MinMaxScaler(feature_range=(1, 100))
scalerThirdDown = MinMaxScaler(feature_range=(1, 100))
scalerTalent = MinMaxScaler(feature_range=(100,1000))
scalerAvgFieldPosition = MinMaxScaler(feature_range=(-10,10))
scalerPPO = MinMaxScaler(feature_range=(1,100))

#################################################################################################################################################

# scaling all the data based on the scaler
team_data['sp_conf_scaled'] = scaler10.fit_transform(team_data[['sp_conf_rating']])
team_data['total_turnovers_scaled'] = scalerTurnovers.fit_transform(team_data[['total_turnovers']])
team_data['possession_scaled'] = scaler100.fit_transform(team_data[['possessionTimeMinutes']])
team_data['third_down_scaled'] = scalerThirdDown.fit_transform(team_data[['thirdDownConversionRate']])
team_data['offense_avg_field_position_scaled'] = -1*scalerAvgFieldPosition.fit_transform(team_data[['Offense_field_position_average_start']])
team_data['defense_avg_field_position_scaled'] = scalerAvgFieldPosition.fit_transform(team_data[['Defense_field_position_average_start']])
team_data['offense_ppo_scaled'] = scalerPPO.fit_transform(team_data[['Offense_points_per_opportunity']])
team_data['offense_success_scaled'] = scaler100.fit_transform(team_data[['Offense_success_rate']])
team_data['offense_explosive'] = scaler100.fit_transform(team_data[['Offense_explosiveness']])
team_data['talent_scaled'] = scalerTalent.fit_transform(team_data[['avg_talent']])

def_ppo_min = team_data['Defense_points_per_opportunity'].min()
def_ppo_max = team_data['Defense_points_per_opportunity'].max()
team_data['defense_ppo_scaled'] = 100 - (team_data['Defense_points_per_opportunity'] - def_ppo_min) * 99 / (def_ppo_max - def_ppo_min)

pen_min = team_data['penaltyYards'].min()
pen_max = team_data['penaltyYards'].max()
team_data['penalties_scaled'] = 100 - (team_data['penaltyYards'] - pen_min) * 99 / (pen_max - pen_min)

off_field_min = team_data['Offense_field_position_average_start'].min()
off_field_max = team_data['Offense_field_position_average_start'].max()
team_data['offense_avg_field_position_scaled'] = 100 - (team_data['Offense_field_position_average_start'] - off_field_min) * 99 / (off_field_max - off_field_min)

team_data['offense_ppa_scaled'] = scaler100.fit_transform(team_data[['Offense_ppa']])
ppa_min = team_data['Defense_ppa'].min()
ppa_max = team_data['Defense_ppa'].max()
team_data['defense_ppa_scaled'] = 100 - (team_data['Defense_ppa'] - ppa_min) * 99 / (ppa_max - ppa_min)

success_min = team_data['Defense_success_rate'].min()
success_max = team_data['Defense_success_rate'].max()
team_data['defense_success_scaled'] = 100 - (team_data['Defense_success_rate'] - success_min) * 99 / (success_max - success_min)

explosiveness_min = team_data['Defense_explosiveness'].min()
explosiveness_max = team_data['Defense_explosiveness'].max()
team_data['defense_explosive'] = 100 - (team_data['Defense_explosiveness'] - explosiveness_min) * 99 / (explosiveness_max - explosiveness_min)

#################################################################################################################################################

# calculating the adjusted metric as well as the power rating for each team
alpha = .05
team_data['adjusted_metric'] = (0.7 * (team_data['offense_success_scaled'] + team_data['defense_success_scaled']) +
                                (alpha * team_data['sp_conf_scaled']**0.5) +
                                0.25 * (team_data['offense_explosive'] + team_data['defense_explosive']) +
                                (0*team_data['talent_scaled']) + (0.4*(team_data['total_turnovers_scaled'] + team_data['penalties_scaled'] + team_data['offense_ppo_scaled'])))
team_data['average_metric'] = (team_data['offense_success_scaled'] + team_data['offense_explosive'] + team_data['offense_ppa_scaled'] + 
                            team_data['defense_success_scaled'] + team_data['defense_explosive'] + team_data['defense_ppa_scaled']) / 6
team_data['in_house_pr'] = scaler60.fit_transform(team_data[['adjusted_metric']]).round(2)
team_data['in_house_pr'] = round(team_data['in_house_pr'] - team_data['in_house_pr'].mean(), 1)

###############################################################################
###############################################################################
###############################################################################
## When adding in the pre-season ratings, I think this is a good spot for it ##
###############################################################################
###############################################################################
###############################################################################

pbar = tqdm(total=500, desc="Optimization Progress")
def progress_callback(xk, convergence):
    """Callback to update the progress bar after each iteration."""
    pbar.update(1)
    if convergence < 1e-4:  # Close bar if convergence is achieved early
        pbar.close()

######################################## SCALING THE EXTRA STATS #################################################

offensive_columns = [
    'Offense_success_rate', 'Offense_explosiveness', 'Offense_ppa', 'Offense_points_per_opportunity', 'Offense_field_position_average_predicted_points', 'Offense_power_success', 'Offense_stuff_rate'
]
defensive_columns = [
    'Defense_success_rate', 'Defense_explosiveness', 'Defense_ppa', 'Defense_points_per_opportunity', 'Defense_field_position_average_predicted_points', 'Defense_power_success', 'Defense_stuff_rate'
]
other_columns = [
    'avg_talent', 'sp_conf_rating', 'thirdDownConversionRate', 'total_turnovers', 'puntReturnTDs', 'kickReturnTDs', 'Defense_havoc_total', 'qb_average_ppa', 'qb_total_ppa'
]

# Function to scale columns between 1 and 100
def scale_columns(df, columns, reverse=False):
    scaler = MinMaxScaler(feature_range=(1, 100))
    if reverse:
        # For defensive stats, the lower value is better, so we reverse the scaling
        scaled = scaler.fit_transform(-df[columns])
    else:
        scaled = scaler.fit_transform(df[columns])
    return pd.DataFrame(scaled, columns=columns)

team_data[offensive_columns] = scale_columns(team_data, offensive_columns)
team_data[defensive_columns] = scale_columns(team_data, defensive_columns, reverse=True)
team_data[['avg_talent', 'sp_conf_rating']] = scale_columns(team_data, ['avg_talent', 'sp_conf_rating'])
team_data[['thirdDownConversionRate']] = scale_columns(team_data, ['thirdDownConversionRate'])
team_data[['fourthDownConversionRate']] = scale_columns(team_data, ['fourthDownConversionRate'])
team_data[['total_turnovers']] = scale_columns(team_data, ['total_turnovers'], reverse=True)  # Lower turnovers are better
team_data[['puntReturnTDs', 'kickReturnTDs']] = scale_columns(team_data, ['puntReturnTDs', 'kickReturnTDs'])
team_data[['Defense_havoc_total']] = scale_columns(team_data, ['Defense_havoc_total'])

merged_data = pd.merge(team_data, team_sp[['team', 'ranking']], on='team')

######################################## HERDING TO SP+ AND FPI #################################################

def objective_function(weights):
    (w_offense_sr, w_offense_expl, w_offense_ppa, w_offense_ppo,
    w_defense_sr, w_defense_expl, w_defense_ppa, w_defense_ppo,
    w_avg_talent, w_third_down, w_turnovers,
    w_special_punt, w_special_kick, w_havoc, w_in_house, 
    w_avg_qb_ppa, w_total_qb_ppa, w_defense_fp_app, 
    w_off_power, w_def_power, w_off_stuff, w_def_stuff,
    rank_weight_fpi, rank_weight_other) = weights
    
    merged_data['power_ranking'] = (
        (w_offense_sr * merged_data['Offense_success_rate'] + 
        w_offense_expl * merged_data['Offense_explosiveness'] + 
        w_offense_ppa * merged_data['Offense_ppa'] + 
        w_offense_ppo * merged_data['Offense_points_per_opportunity'] +
        w_off_power * merged_data['Offense_power_success'] +
        w_off_stuff * merged_data['Offense_stuff_rate'])
        - (w_defense_sr * merged_data['Defense_success_rate'] + 
        w_defense_expl * merged_data['Defense_explosiveness'] + 
        w_defense_ppa * merged_data['Defense_ppa'] + 
        w_defense_ppo * merged_data['Defense_points_per_opportunity'] +
        w_defense_fp_app * merged_data['Defense_field_position_average_predicted_points'] +
        w_def_power * merged_data['Defense_power_success'] +
        w_def_stuff * merged_data['Defense_stuff_rate'])
        + w_avg_talent * merged_data['avg_talent']
        + (w_third_down * merged_data['thirdDownConversionRate']
        + w_turnovers * merged_data['total_turnovers']
        + w_special_punt * merged_data['puntReturnTDs'] 
        + w_special_kick * merged_data['kickReturnTDs']
        + w_havoc * merged_data['Defense_havoc_total'])
        + w_in_house * merged_data['in_house_pr']
        + w_avg_qb_ppa * merged_data['qb_average_ppa']
        + w_total_qb_ppa * merged_data['qb_total_ppa']
    )

    # My ranking
    merged_data['calculated_rank'] = merged_data['power_ranking'].rank(ascending=False)
    # SP+/FPI combined ranking
    merged_data['combined_rank'] = (
        rank_weight_fpi * merged_data['fpi_rank'] +
        rank_weight_other * merged_data['ranking']
    )
    spearman_corr = merged_data[['calculated_rank', 'combined_rank']].corr(method='spearman').iloc[0, 1]
    
    return -spearman_corr

# Define the bounds for each weight (allowing weights to vary more widely)
bounds = [
    (-1, 1),  # Offense Success Rate
    (-1, 1),  # Offense Explosiveness
    (-1, 1),  # Offense PPA
    (-1, 1),  # Offense PPO
    (-1, 1),  # Defense Success Rate
    (-1, 1),  # Defense Explosiveness
    (-1, 1),  # Defense PPA 
    (-1, 1),  # Defense PPO
    (0, 0.5), # Avg Talent: Bound between 0 and 0.5
    (-1, 1),  # Third Down Conversion
    (-1, 1),  # Turnovers
    (-1, 1),  # Special Teams Punt
    (-1, 1),  # Special Teams Kick
    (-1, 1),  # Havoc
    (0, 0.5), # In House PR
    (-1, 1),  # Average QB PPA
    (-1, 1),  # Total QB PPA
    (-1, 1),  # Defense FP APP
    (-1, 1),  # Offense Power Success
    (-1, 1),  # Defense Power Success
    (-1, 1),  # Offense Stuff Rate
    (-1, 1),  # Defense Stuff Rate
    (0, 1),   # FPI Weight
    (0, 1)    # SP+ Weight
]

result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42, callback=progress_callback)
optimized_weights = result.x

######################################## USING OUTPUT FROM OPTIMIZATION TO CREATE POWER RANKING #################################################

# Recalculate the power ranking using the optimized weights
merged_data['power_ranking'] = (
    (optimized_weights[0] * merged_data['Offense_success_rate'] + 
    optimized_weights[1] * merged_data['Offense_explosiveness'] + 
    optimized_weights[2] * merged_data['Offense_ppa'] + 
    optimized_weights[3] * merged_data['Offense_points_per_opportunity'] +
    optimized_weights[18] * merged_data['Offense_power_success'] +
    optimized_weights[20] * merged_data['Offense_stuff_rate'])
    - (optimized_weights[4] * merged_data['Defense_success_rate'] + 
    optimized_weights[5] * merged_data['Defense_explosiveness'] + 
    optimized_weights[6] * merged_data['Defense_ppa'] + 
    optimized_weights[7] * merged_data['Defense_points_per_opportunity'] +
    optimized_weights[17] * merged_data['Defense_field_position_average_predicted_points'] +
    optimized_weights[19] * merged_data['Defense_power_success'] +
    optimized_weights[21] * merged_data['Defense_stuff_rate'])
    + optimized_weights[8] * merged_data['avg_talent']
    + (optimized_weights[9] * merged_data['thirdDownConversionRate']
    + optimized_weights[10] * merged_data['total_turnovers']
    + optimized_weights[11] * merged_data['puntReturnTDs'] 
    + optimized_weights[12] * merged_data['kickReturnTDs']
    + optimized_weights[13] * merged_data['Defense_havoc_total'])
    + optimized_weights[14] * merged_data['in_house_pr']
    + optimized_weights[15] * merged_data['qb_average_ppa']
    + optimized_weights[16] * merged_data['qb_total_ppa']
)

output_model = merged_data.copy()

team_data = output_model.copy()

######################################## TEAM STATS AND RANKINGS #################################################

team_data['PBR'] = team_data['penaltyYards'] / team_data['talent_scaled']
team_data['PBR_rank'] = team_data['PBR'].rank(method='min', ascending=True)

team_data['STM'] = (
    (team_data['kickReturnYards'] / team_data['kickReturns']) +
    (team_data['puntReturnYards'] / team_data['puntReturns']) -
    team_data['Offense_field_position_average_start'] +
    team_data['Defense_field_position_average_start']
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

team_data["offensive_total"] = team_data[offensive_columns].sum(axis=1)
team_data["offensive_rank"] = team_data["offensive_total"].rank(ascending=False, method="dense").astype(int)
team_data["defensive_total"] = team_data[defensive_columns].sum(axis=1)
team_data["defensive_rank"] = team_data["defensive_total"].rank(ascending=False, method="dense").astype(int)

team_data['talent_scaled_rank'] = team_data['talent_scaled'].rank(method='min', ascending=False)
team_data['offense_success_rank'] = team_data['offense_success_scaled'].rank(method='min', ascending=False)
team_data['defense_success_rank'] = team_data['defense_success_scaled'].rank(method='min', ascending=False)
team_data['offense_explosive_rank'] = team_data['offense_explosive'].rank(method='min', ascending=False)
team_data['defense_explosive_rank'] = team_data['defense_explosive'].rank(method='min', ascending=False)
team_data['total_turnovers_rank'] = team_data['total_turnovers_scaled'].rank(method='min', ascending=False)
team_data['penalties_rank'] = team_data['penalties_scaled'].rank(method='min', ascending=False)

######################################## POWER RATING #################################################

team_data['power_rating'] = team_data['power_ranking'] - team_data['power_ranking'].mean()
current_range = team_data['power_rating'].max() - team_data['power_rating'].min()
desired_range = 50  # The target range
scaling_factor = desired_range / current_range
team_data['power_rating'] = round(team_data['power_rating'] * scaling_factor,2)

######################################## FINAL FORMATTING #################################################

team_data = team_data.sort_values(by='power_rating', ascending=False).reset_index(drop=True)
team_data['power_rating'] = round(team_data['power_rating'], 1)
team_data = team_data.drop_duplicates(subset='team')
team_power_rankings = team_data[['team', 'power_rating', 'conference']]
team_power_rankings = team_power_rankings.sort_values(by='power_rating', ascending=False).reset_index(drop=True)

#### if top team is too high, use this
# team_power_rankings.iloc[0,1] = round(team_power_rankings.iloc[1,1] + round((team_power_rankings.iloc[0,1] - team_power_rankings.iloc[1,1]) / 2, 1),1)
# team_data.loc[0, 'power_rating'] = team_power_rankings.iloc[0, 1]

team_power_rankings.index = team_power_rankings.index + 1
team_power_rankings['week'] = current_week
team_power_rankings['year'] = current_year

## year long schedule
start_week = 1
end_week = 17

games_list = []
for week in range(start_week,end_week):
    response = games_api.get_games(year=current_year, week=week,division = 'fbs')
    games_list = [*games_list, *response]
if postseason:
    response = games_api.get_games(year=current_year, division = 'fbs', season_type='postseason')
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
            home_points = g.home_points,
            away_points = g.away_points,
            neutral = g.neutral_site
            ) for g in games_list if g.home_pregame_elo is not None and g.away_pregame_elo is not None]
games.sort(key=date_sort)
year_long_schedule = pd.DataFrame(games)

year_long_schedule = year_long_schedule.merge(team_data[['team', 'power_rating']], 
                                    left_on='home_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'home_pr'})
year_long_schedule = year_long_schedule.drop(columns=['team'])
year_long_schedule = year_long_schedule.merge(team_data[['team', 'power_rating']], 
                                    left_on='away_team', 
                                    right_on='team', 
                                    how='left').rename(columns={'power_rating': 'away_pr'})
year_long_schedule = year_long_schedule.drop(columns=['team'])

# Apply the PEAR_Win_Prob function to the schedule_info DataFrame
year_long_schedule['PEAR_win_prob'] = year_long_schedule.apply(
    lambda row: PEAR_Win_Prob(row['home_pr'], row['away_pr']), axis=1
)
year_long_schedule['home_win_prob'] = round((10**((year_long_schedule['home_elo'] - year_long_schedule['away_elo']) / 400)) / ((10**((year_long_schedule['home_elo'] - year_long_schedule['away_elo']) / 400)) + 1)*100,2)


## sos calculation
average_elo = elo_ratings['elo'].mean()
average_pr = round(team_data['power_rating'].mean(), 2)
good_team_pr = round(team_data['power_rating'].std() + team_data['power_rating'].mean(),2)
elite_team_pr = round(2*team_data['power_rating'].std() + team_data['power_rating'].mean(),2)
expected_wins_list = []
for team in team_data['team']:
    schedule = year_long_schedule[(year_long_schedule['home_team'] == team) | (year_long_schedule['away_team'] == team)]
    df = average_team_distribution(1000, schedule, elite_team_pr, team)
    expected_wins = df['expected_wins'].values[0]
    expected_wins_list.append(expected_wins)
SOS = pd.DataFrame(zip(team_data['team'], expected_wins_list), columns=['team', 'avg_expected_wins'])
SOS = SOS.sort_values('avg_expected_wins').reset_index(drop = True)
SOS['SOS'] = SOS.index + 1
print("SOS Calculation Done")


## sor calculation
completed_games = year_long_schedule[year_long_schedule['home_points'].notna()]
current_xWins_list = []
good_xWins_list = []
elite_xWins_list = []
for team in team_data['team']:
    team_completed_games = completed_games[(completed_games['home_team'] == team) | (completed_games['away_team'] == team)]
    games_played = records[records['team'] == team]['games_played'].values[0]
    wins = records[records['team'] == team]['wins'].values[0]
    team_completed_games['avg_win_prob'] = np.where(team_completed_games['home_team'] == team,
                                                    PEAR_Win_Prob(average_pr, team_completed_games['away_pr']),
                                                    100 - PEAR_Win_Prob(team_completed_games['home_pr'], average_pr))
    team_completed_games['good_win_prob'] = np.where(team_completed_games['home_team'] == team,
                                                    PEAR_Win_Prob(good_team_pr, team_completed_games['away_pr']),
                                                    100 - PEAR_Win_Prob(team_completed_games['home_pr'], good_team_pr))
    team_completed_games['elite_win_prob']  = np.where(team_completed_games['home_team'] == team,
                                                    PEAR_Win_Prob(elite_team_pr, team_completed_games['away_pr']),
                                                    100 - PEAR_Win_Prob(team_completed_games['home_pr'], elite_team_pr))

    # team_completed_games['avg_win_prob'] = np.where(team_completed_games['home_team'] == team, 
    #                             round((10**((average_elo-team_completed_games['away_elo']) / 400)) / ((10**((average_elo-team_completed_games['away_elo']) / 400)) + 1)*100, 2), 
    #                             100 - round((10**((team_completed_games['home_elo'] - average_elo) / 400)) / ((10**((team_completed_games['home_elo']- average_elo) / 400)) + 1)*100, 2))
    current_xWins = round(sum(team_completed_games['avg_win_prob']) / 100, 2)
    good_xWins = round(sum(team_completed_games['good_win_prob']) / 100, 2)
    elite_xWins = round(sum(team_completed_games['elite_win_prob']) / 100, 2)
    if games_played != len(team_completed_games):
        current_xWins += 1
        good_xWins += 1
        elite_xWins += 1
    relative_current_xWins = round(wins - current_xWins, 2)
    relative_good_xWins = round(wins - good_xWins, 2)
    relative_elite_xWins = round(wins - elite_xWins, 2)
    current_xWins_list.append(relative_current_xWins)
    good_xWins_list.append(relative_good_xWins)
    elite_xWins_list.append(relative_elite_xWins)
SOR = pd.DataFrame(zip(team_data['team'], current_xWins_list, good_xWins_list, elite_xWins_list), columns=['team','wins_above_average','wins_above_good','wins_above_elite'])
SOR = SOR.sort_values('wins_above_good', ascending=False).reset_index(drop=True)
SOR['SOR'] = SOR.index + 1
print("SOR Calculation Done")


## most deserving calculation
num_12_pr = team_data['power_rating'][11]

# Ensure MOV is calculated in the completed_games DataFrame
completed_games = year_long_schedule[year_long_schedule['home_points'].notna()]
completed_games['margin_of_victory'] = completed_games['home_points'] - completed_games['away_points']

# Function to scale MOV
def f(mov):
    return np.clip(np.log(abs(mov) + 1) * np.sign(mov), -10, 10)

current_xWins_list = []

for team in team_data['team']:
    # Filter completed games for the current team
    team_completed_games = completed_games[(completed_games['home_team'] == team) | (completed_games['away_team'] == team)]
    
    # Get the current team's record
    games_played = records[records['team'] == team]['games_played'].values[0]
    wins = records[records['team'] == team]['wins'].values[0]
    
    # Adjust win probability with MOV influence
    team_completed_games['avg_win_prob'] = np.where(
        team_completed_games['home_team'] == team,
        PEAR_Win_Prob(num_12_pr, team_completed_games['away_pr']) + f(team_completed_games['margin_of_victory']),
        100 - PEAR_Win_Prob(team_completed_games['home_pr'], num_12_pr) - f(-team_completed_games['margin_of_victory'])
    )
    
    # Calculate expected wins (xWins)
    current_xWins = round(sum(team_completed_games['avg_win_prob']) / 100, 3)
    
    # Adjust for incomplete games
    if games_played != len(team_completed_games):
        current_xWins += 1
    
    # Calculate relative xWins (wins vs. expected wins)
    relative_current_xWins = round(wins - current_xWins, 3)
    current_xWins_list.append(relative_current_xWins)

# Create the "most deserving" DataFrame
most_deserving = pd.DataFrame(zip(team_data['team'], current_xWins_list), columns=['team', 'most_deserving_wins'])
most_deserving = most_deserving.sort_values('most_deserving_wins', ascending=False).reset_index(drop=True)
most_deserving['most_deserving'] = most_deserving.index + 1
print("Most Deserving Calculation Done")

team_data = pd.merge(team_data, SOS, how='left', on='team')
team_data = pd.merge(team_data, SOR, how='left', on='team')
team_data = pd.merge(team_data, most_deserving, how='left', on='team')

folder_path = f"./PEAR/Data/y{current_year}"
os.makedirs(folder_path, exist_ok=True)

folder_path = f"./PEAR/Ratings/y{current_year}"
os.makedirs(folder_path, exist_ok=True)

folder_path = f"./PEAR/Spreads/y{current_year}"
os.makedirs(folder_path, exist_ok=True)

team_data.to_csv(f"./PEAR/Data/y{current_year}/team_data_week{current_week}_no_adjustments.csv")
team_power_rankings.to_csv(f'./PEAR/Ratings/y{current_year}/PEAR_week{current_week}_no_adjustments.csv')