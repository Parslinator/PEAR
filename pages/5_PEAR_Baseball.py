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

st.title(f"{current_season} PEAR Baseball")
st.caption(f"Last Updated {formatted_latest_date}")
st.divider()

modeling_stats.index = modeling_stats.index + 1
with st.container(border=True, height=440):
    st.dataframe(modeling_stats[['Team', 'Rating']], use_container_width=True)
