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

all_data = pd.read_csv("./ESCAPE Ratings/Data/y2013/team_data_week17.csv")
all_data.rename(columns={"offensive_rank": "Offense"}, inplace=True)
all_data.rename(columns={"defensive_rank": "Defense"}, inplace=True)
st.subheader("2013 Ratings")
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