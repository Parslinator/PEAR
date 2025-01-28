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
import glob
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



team_data = pd.read_csv("./ESCAPE Ratings/normalized_power_rating_across_years.csv")

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 5

def grab_team_elo_across_years(team, season):
    season = int(season)
    elo_ratings_list = [*ratings_api.get_elo_ratings(year=season, team=team)]
    elo_ratings_dict = [dict(
        team=e.team,
        elo=e.elo
    ) for e in elo_ratings_list]
    elo_ratings = pd.DataFrame(elo_ratings_dict)       
    return elo_ratings['elo'].values[0]

def spreads_across_years(team1, team1_season, team2, team2_season, data, neutrality=False):
    team1_season = int(team1_season)
    team2_season = int(team2_season)
    home_elo = grab_team_elo_across_years(team1, team1_season)
    away_elo = grab_team_elo_across_years(team2, team2_season)
    home_pr = data.loc[(data['team'] == team1) & (data['season'] == team1_season), 'norm_pr'].values[0]
    away_pr = data.loc[(data['team'] == team2) & (data['season'] == team2_season), 'norm_pr'].values[0]
    home_win_prob = round((10 ** ((home_elo - away_elo) / 400)) / ((10 ** ((home_elo - away_elo) / 400)) + 1) * 100, 2)
    adjustment = adjust_home_pr(home_win_prob)
    if neutrality:
        spread = home_pr + adjustment - away_pr
    else:
        spread = 4.6 + home_pr + adjustment - away_pr
    spread = round(spread,1)

    if spread >= 0:
        return f"{team1} -{spread}"
    else:
        return f"{team2} {spread}"

st.subheader("Calculate Spread Between Two Teams From Different Years")
with st.form(key='calculate_spread'):
    away_team = st.selectbox("Away Team", ["Select Team"] + list(sorted(team_data['team'].unique())))
    away_season = st.selectbox("Away Season", ["Select Season"] + list(sorted(team_data['season'].unique())))
    home_team = st.selectbox("Home Team", ["Select Team"] + list(sorted(team_data['team'].unique())))
    home_season = st.selectbox("Home Season", ["Select Season"] + list(sorted(team_data['season'].unique())))
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
        st.write(spreads_across_years(home_team, home_season, away_team, away_season, team_data, neutrality))


st.divider()

st.subheader("Year-Normalized Power Ratings")
team_data['Team'] = team_data['team']
team_data['Season'] = team_data['season'].astype(str)
team_data['Normalized Rating'] = team_data['norm_pr']
with st.container(border=True, height=440):
    st.dataframe(team_data[['team', 'Normalized Rating', 'Season']])

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2023/team_data_week15.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2023 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2022/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2022 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2021/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2021 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2020/team_data_week17.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2020 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2019/team_data_week17.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2019 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2018/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2018 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2017/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2017 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2016/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2016 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2015/team_data_week16.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2015 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2014/team_data_week17.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2014 Ratings")
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
st.caption("MD - Most Deserving (ESCAPE's 'AP' Ballot), SOS - Strength of Schedule, SOR - Strength of Record, OFF - Offense, DEF - Defense, ST - Special Teams, PBR - Penalty Burden Ratio, DCE - Drive Control Efficiency, DDE - Drive Disruption Efficiency")
# , MD - Most Deserving Rankings

st.divider()