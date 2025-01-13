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
current_week = int(current_week)
current_year = int(current_year)

st.title("Team Stats")

# Construct the folder path
folder_path = f"ESCAPE Ratings/Visuals/y{current_year}/week_{current_week}/"

if os.path.exists(folder_path):
    top25_file = f"{folder_path}top25.png" # used
    go5_top25_file = f"{folder_path}go5_top25.png" # used
    power_rating_team_pyramid = f"{folder_path}power_rating_team_pyramid.png"  # used
    projected_playoff = f"{folder_path}projected_playoff.png" # used

    most_deserving_playoff = f"{folder_path}most_deserving_playoff.png"
    most_deserving_team_pyramid = f"{folder_path}most_deserving_team_pyramid.png"
    most_deserving = f"{folder_path}most_deserving.png"

    avg_metric_rank = f"{folder_path}average_metric_rank.png"
    strength_of_record = f"{folder_path}strength_of_record.png"
    strength_of_schedule = f"{folder_path}strength_of_schedule.png"
    offenses = f"{folder_path}offenses.png"
    defenses = f"{folder_path}defenses.png"
    special_teams = f"{folder_path}special_teams.png"
    turnovers = f"{folder_path}turnovers.png"
    drive_control_efficiency = f"{folder_path}drive_control_efficiency.png"
    drive_disruption_efficiency = f"{folder_path}drive_disruption_efficiency.png"
    penalty_burden_ratio = f"{folder_path}penalty_burden_ratio.png"
    mov_performance = f"{folder_path}mov_performance.png"
    overperformer_and_underperformer = f"{folder_path}overperformer_and_underperformer.png"
    conference_average = f"{folder_path}conference_average.png"

    dce_vs_dde = f"{folder_path}dce_vs_dde.png"
    offense_vs_defense = f"{folder_path}offense_vs_defense.png"

    st.subheader("Ratings Information")
    cols = st.columns(2)
    with cols[0]:
        st.image(top25_file, use_container_width=True, caption="Top 25 Ratings")
    with cols[1]:
        st.image(go5_top25_file, use_container_width=True, caption="Group of 5 Top 25 Ratings")

    cols = st.columns(2)
    with cols[0]:
        st.image(power_rating_team_pyramid, width=500, caption="Ratings Team Pyramid")
    with cols[1]:
        st.image(projected_playoff, width=500, caption="Ratings Projected Playoff - Conference Champions are Highest Rated Team")

    st.subheader("Rankings Information")
    st.image(most_deserving, width=500, caption="Most Deserving Rankings")
    cols = st.columns(2)
    with cols[0]:
        st.image(most_deserving_team_pyramid, use_container_width=True, caption="Most Deserving Team Pyramid")
    with cols[1]:
        st.image(most_deserving_playoff, use_container_width=True, caption="Rankings Projected Playoff")

    st.subheader("Team Stats Information")
    st.image(avg_metric_rank, use_container_width=True, caption="Average Metric Ranks")
    cols = st.columns(2)
    with cols[0]:
        st.image(strength_of_record, use_container_width=True, caption="Strength of Record")
    with cols[1]:
        st.image(strength_of_schedule, use_container_width=True, caption="Strength of Schedule")

    cols = st.columns(2)
    with cols[0]:
        st.image(offenses, use_container_width=True, caption="Offense Ranks")
    with cols[1]:
        st.image(defenses, use_container_width=True, caption="Defense Ranks")

    cols = st.columns(2)
    with cols[0]:
        st.image(special_teams, use_container_width=True, caption="Special Teams Ranks")
    with cols[1]:
        st.image(turnovers, use_container_width=True, caption="Turnover Ranks")

    cols = st.columns(2)
    with cols[0]:
        st.image(drive_control_efficiency, use_container_width=True, caption="Drive Control Efficiency")
    with cols[1]:
        st.image(drive_disruption_efficiency, use_container_width=True, caption="Drive Disruption Efficiency")

    cols = st.columns(2)
    with cols[0]:
        st.image(penalty_burden_ratio, use_container_width=True, caption="Penalty Burden Ratio")
    with cols[1]:
        st.image(mov_performance, use_container_width=True, caption="Margin of Victory Performance")

    cols = st.columns(2)
    with cols[0]:
        st.image(overperformer_and_underperformer, use_container_width=True, caption="Over and Underperformers")
    with cols[1]:
        st.image(conference_average, use_container_width=True, caption="Average Team Power Rating in Each Conference")

    cols = st.columns(2)
    with cols[0]:
        st.image(offense_vs_defense, use_container_width=True, caption="Offense Versus Defense")
    with cols[1]:
        st.image(dce_vs_dde, use_container_width=True, caption="DCE vs DDE")
