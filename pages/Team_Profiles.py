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

folder_path = f"ESCAPE Ratings/Visuals/y{current_year}/week_{current_week}/Stat Profiles"

if os.path.exists(folder_path):
    # Get a list of all .png files in the folder
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    # Display the images in the Streamlit app
    if png_files:
        st.subheader("All Teams Stat Profiles")

        # Loop through the .png files and display two per row
        for i in range(0, len(png_files), 2):  # Increment by 2 for two images per row
            cols = st.columns(2)  # Create two columns

            # Display first image
            with cols[0]:
                file_name = os.path.splitext(os.path.basename(png_files[i]))[0]
                st.image(png_files[i], use_container_width=True, caption=file_name)

            # Check if there's a second image in the current pair
            if i + 1 < len(png_files):
                with cols[1]:
                    file_name = os.path.splitext(os.path.basename(png_files[i+1]))[0]
                    st.image(png_files[i + 1], use_container_width=True, caption=file_name)
    else:
        st.warning(f"No .png files found in {folder_path}")