from datetime import datetime
import os

formatted_date = datetime.today().strftime('%m_%d_%Y')
current_season = datetime.today().year

folder_path = f"./PEAR/PEAR Baseball/y{current_season}"
os.makedirs(folder_path, exist_ok=True)

import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import warnings
warnings.filterwarnings("ignore")

def PEAR_Win_Prob(home_pr, away_pr):
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 10)) * 100, 2)
    return win_prob

# Base URL for NCAA stats
base_url = "https://www.ncaa.com"
stats_page = f"{base_url}/stats/baseball/d1"

# Function to get page content
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()  # Ensure request was successful
    return BeautifulSoup(response.text, "html.parser")

# Get main page content
soup = get_soup(stats_page)

# Find the dropdown container and extract stat URLs
dropdown = soup.find("select", {"id": "select-container-team"})
options = dropdown.find_all("option")

# Extract stat names and links
stat_links = {
    option.text.strip(): base_url + option["value"]
    for option in options if option.get("value")
}

url = "https://www.ncaa.com/rankings/baseball/d1/rpi"
response = requests.get(url)
response.raise_for_status()  # Ensure request was successful
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table", class_="sticky")
if table:
    headers = [th.text.strip() for th in table.find_all("th")]
    data = []
    for row in table.find_all("tr")[1:]:  # Skip header row
        cols = row.find_all("td")
        data.append([col.text.strip() for col in cols])
    rpi = pd.DataFrame(data, columns=headers)
    rpi = rpi.drop(columns = ['Previous'])
    rpi.rename(columns={"School": "Team"}, inplace=True)
else:
    print("Table not found.")

url = "https://www.collegebaseballratings.com/"
response = requests.get(url)
response.raise_for_status()  # Raise an error for failed requests
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table", {"id": "teamList"})
headers = [th.text.strip() for th in table.find("thead").find_all("th")]
data = []
for row in table.find("tbody").find_all("tr"):
    cells = [td.text.strip() for td in row.find_all("td")]
    data.append(cells)
cbr = pd.DataFrame(data, columns=headers[1:])
cbr.rename(columns={"Rank":"CBRank"}, inplace=True)

def get_stat_dataframe(stat_name):
    """Fetches the specified stat table from multiple pages and returns a combined DataFrame,
    keeps 'Team' as string, and converts all other columns to float."""
    
    if stat_name not in stat_links:
        print(f"Stat '{stat_name}' not found. Available stats: {list(stat_links.keys())}")
        return None
    
    # Initialize the DataFrame to store all pages' data
    all_data = []
    page_num = 1  # Start from the first page

    while True:
        url = stat_links[stat_name]
        if page_num > 1:
            # Modify the URL to include the page number
            url = f"{url}/p{page_num}"
        
        # print(f"Fetching data for: {stat_name} (Page {page_num} - {url})")

        try:
            # Get stats page content
            soup = get_soup(url)

            # Locate table
            table = soup.find("table")
            if not table:
                print(f"No table found for {stat_name} on page {page_num}")
                break  # Exit the loop if no table is found (end of valid pages)

            # Extract table headers
            headers = [th.text.strip() for th in table.find_all("th")]

            # Extract table rows
            data = []
            for row in table.find_all("tr")[1:]:  # Skip header row
                cols = row.find_all("td")
                data.append([col.text.strip() for col in cols])

            all_data.extend(data)  # Add the data from this page to the list of all data
        
        except requests.exceptions.HTTPError as e:
            print(f"{stat_name} Done")
            break  # Exit the loop on HTTPError (page doesn't exist)
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Exit the loop on any other error

        page_num += 1  # Go to the next page

    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data, columns=headers)

        # Convert all columns to float except "Team"
        for col in df.columns:
            if col != "Team":
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Converts to float, invalid values become NaN

        return df
    else:
        print("No data collected.")
        return None

# Example usage
stat_name_input = "Batting Average"  # Change this to the desired stat
ba = get_stat_dataframe(stat_name_input)
ba["HPG"] = ba["H"] / ba["G"]
ba["ABPG"] = ba["AB"] / ba["G"]
ba["HPAB"] = ba["H"] / ba["AB"]
ba = ba.drop(columns=['Rank'])

stat_name_input = "Base on Balls"
bb = get_stat_dataframe(stat_name_input)
bb["BBPG"] = bb["BB"] / bb["G"]
bb = bb.drop(columns=['Rank', 'G'])

stat_name_input = "Double Plays Per Game"
dp = get_stat_dataframe(stat_name_input)
dp.rename(columns={"PG": "DPPG"}, inplace=True)
dp = dp.drop(columns=['Rank', 'G'])

stat_name_input = "Earned Run Average"
era = get_stat_dataframe(stat_name_input)
era.rename(columns={"R":"RA"}, inplace=True)
era = era.drop(columns=['Rank', 'G'])

stat_name_input = "Fielding Percentage"
fp = get_stat_dataframe(stat_name_input)
fp["APG"] = fp["A"] / fp["G"]
fp["EPG"] = fp["E"] / fp["G"]
fp = fp.drop(columns=['Rank', 'G'])

stat_name_input = "Hits Allowed Per Nine Innings"
ha = get_stat_dataframe(stat_name_input)
ha.rename(columns={"PG": "HAPG"}, inplace=True)
ha = ha.drop(columns=['Rank', 'G', 'IP'])

stat_name_input = "Home Runs Per Game"
hr = get_stat_dataframe(stat_name_input)
hr.rename(columns={"PG": "HRPG"}, inplace=True)
hr = hr.drop(columns=['Rank', 'G'])
duplicate_teams = hr[hr.duplicated('Team', keep=False)]
filtered_teams = duplicate_teams.loc[duplicate_teams.groupby('Team')["HR"].idxmin()]
hr_cleaned = hr[~hr["Team"].isin(duplicate_teams["Team"])]
hr = pd.concat([hr_cleaned, filtered_teams], ignore_index=True)

stat_name_input = "On Base Percentage"
obp = get_stat_dataframe(stat_name_input)
obp.rename(columns={"PCT": "OBP"}, inplace=True)
obp["HBPPG"] = obp["HBP"] / obp["G"]
obp = obp.drop(columns=['Rank', 'G', 'AB', 'H', 'BB', 'SF', 'SH'])

stat_name_input = "Runs"
runs = get_stat_dataframe(stat_name_input)
runs["RPG"] = runs["R"] / runs["G"]
runs.rename(columns={"R": "RS"}, inplace=True)
runs = runs.drop(columns=['Rank', 'G'])

stat_name_input = "Sacrifice Bunts"
sb = get_stat_dataframe(stat_name_input)
sb.rename(columns={"SH": "SB"}, inplace=True)
sb["SBPG"] = sb["SB"] / sb["G"]
sb = sb.drop(columns=['Rank', 'G'])

stat_name_input = "Sacrifice Flies"
sf = get_stat_dataframe(stat_name_input)
sf["SFPG"] = sf["SF"] / sf["G"]
sf = sf.drop(columns=['Rank', 'G'])

stat_name_input = "Slugging Percentage"
slg = get_stat_dataframe(stat_name_input)
slg.rename(columns={"SLG PCT": "SLG"}, inplace=True)
slg = slg.drop(columns=['Rank', 'G', 'AB'])

stat_name_input = "Stolen Bases"
stl = get_stat_dataframe(stat_name_input)
stl["STLP"] = stl["SB"] / (stl["SB"] + stl["CS"])
stl["STLPG"] = stl["SB"] / stl["G"]
stl["CSPG"] = stl["CS"] / stl["G"]
stl["SAPG"] = (stl["SB"] + stl["CS"]) / stl["G"]
stl.rename(columns={"SB": "STL"}, inplace=True)
stl = stl.drop(columns=['Rank', 'G'])

stat_name_input = "Strikeout-to-Walk Ratio"
kbb = get_stat_dataframe(stat_name_input)
kbb["IP"] = round(kbb["IP"])
kbb.rename(columns={"K/BB": "KBB"}, inplace=True)
kbb.rename(columns={"BB": "PBB"}, inplace=True)
kbb = kbb.drop(columns=['Rank', 'App', 'IP'])

stat_name_input = "Strikeouts Per Nine Innings"
kp9 = get_stat_dataframe(stat_name_input)
kp9.rename(columns={"K/9": "KP9"}, inplace=True)
kp9 = kp9.drop(columns=['Rank', 'G', 'IP', 'SO'])

stat_name_input = "Walks Allowed Per Nine Innings"
wp9 = get_stat_dataframe(stat_name_input)
wp9.rename(columns={"PG": "WP9"}, inplace=True)
wp9 = wp9.drop(columns=['Rank', 'G', 'IP', 'BB'])

stat_name_input = "WHIP"
whip = get_stat_dataframe(stat_name_input)
whip = whip.drop(columns=['Rank', 'HA', 'IP', 'BB'])

dfs = [ba, bb, era, fp, obp, runs, slg, kp9, wp9, whip, rpi, cbr]
for df in dfs:
    df["Team"] = df["Team"].str.strip()
df_combined = dfs[0]
for df in dfs[1:]:
    df_combined = pd.merge(df_combined, df, on="Team", how="inner")
baseball_stats = df_combined.loc[:, ~df_combined.columns.duplicated()].sort_values('Team').reset_index(drop=True)
baseball_stats['OPS'] = baseball_stats['SLG'] + baseball_stats['OBP']

rpi_2024 = pd.read_csv("./PEAR/PEAR Baseball/rpi_end_2024.csv")

modeling_stats = baseball_stats[['Team', 'HPG',
                'BBPG', 'ERA', 'PCT', 
                'KP9', 'WP9', 'OPS', 
                'WHIP', 'CBRank']]
modeling_stats = pd.merge(modeling_stats, rpi_2024[['Team', 'Rank']], on = 'Team', how='left')
modeling_stats["Rank"] = modeling_stats["Rank"].apply(pd.to_numeric, errors='coerce')
modeling_stats["CBRank"] = modeling_stats["CBRank"].apply(pd.to_numeric, errors='coerce')
modeling_stats['Rank_pct'] = 1 - (modeling_stats['Rank'] - 1) / (len(modeling_stats) - 1)

higher_better = ["HPG", "BBPG", "PCT", "KP9", "OPS", "Rank_pct"]
lower_better = ["ERA", "WP9", "WHIP"]

scaler = MinMaxScaler(feature_range=(1, 100))
modeling_stats[higher_better] = scaler.fit_transform(modeling_stats[higher_better])
modeling_stats[lower_better] = scaler.fit_transform(-modeling_stats[lower_better])
weights = {
    'HPG': 8, 'BBPG': 8, 'ERA': 22, 'PCT': 8,
    'KP9': 8, 'WP9': 8, 'OPS': 22, 'WHIP': 8, 'Rank_pct': 50
}
modeling_stats['in_house_pr'] = sum(modeling_stats[stat] * weight for stat, weight in weights.items())
modeling_stats['in_house_pr'] = modeling_stats['in_house_pr'] - modeling_stats['in_house_pr'].mean()
current_range = modeling_stats['in_house_pr'].max() - modeling_stats['in_house_pr'].min()
desired_range = 25
scaling_factor = desired_range / current_range
modeling_stats['in_house_pr'] = round(modeling_stats['in_house_pr'] * scaling_factor, 4)
modeling_stats['in_house_pr'] = modeling_stats['in_house_pr'] - modeling_stats['in_house_pr'].min()

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.optimize import minimize # type: ignore
import numpy as np # type: ignore
from scipy.optimize import differential_evolution # type: ignore
from tqdm import tqdm # type: ignore
pbar = tqdm(total=500, desc="Optimization Progress")
def progress_callback(xk, convergence):
    """Callback to update the progress bar after each iteration."""
    pbar.update(1)
    if convergence < 1e-4:  # Close bar if convergence is achieved early
        pbar.close()

def objective_function(weights):
    (w_hpb, w_bbpg, w_era, w_pct, w_kp9, w_wp9, w_whip, w_ops, w_in_house_pr) = weights
    
    modeling_stats['power_ranking'] = (
        w_hpb * modeling_stats['HPG'] +
        w_bbpg * modeling_stats['BBPG'] +
        w_era * modeling_stats['ERA'] +
        w_pct * modeling_stats['PCT'] +
        w_kp9 * modeling_stats['KP9'] +
        w_wp9 * modeling_stats['WP9'] +
        w_whip * modeling_stats['WHIP'] +
        w_ops * modeling_stats['OPS'] +
        w_in_house_pr * modeling_stats['in_house_pr']
    )

    modeling_stats['calculated_rank'] = modeling_stats['power_ranking'].rank(ascending=False)
    modeling_stats['combined_rank'] = (
        modeling_stats['CBRank']
    )
    spearman_corr = modeling_stats[['calculated_rank', 'combined_rank']].corr(method='spearman').iloc[0,1]

    return -spearman_corr

bounds = [(-1,1),
          (-1,1),
          (-1,1),
          (-1,1),
          (-1,1),
          (-1,1),
          (-1,1),
          (-1,1),
          (0,1)]
result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42, callback=progress_callback)
optimized_weights = result.x
modeling_stats = modeling_stats.sort_values('power_ranking', ascending=False).reset_index(drop=True)

modeling_stats['Rating'] = modeling_stats['power_ranking'] - modeling_stats['power_ranking'].mean()
current_range = modeling_stats['Rating'].max() - modeling_stats['Rating'].min()
desired_range = 25
scaling_factor = desired_range / current_range
modeling_stats['Rating'] = round(modeling_stats['Rating'] * scaling_factor, 4)
modeling_stats['Rating'] = modeling_stats['Rating'] - modeling_stats['Rating'].min()

ending_data = pd.merge(baseball_stats, modeling_stats[['Team', 'Rating']], on="Team", how="inner").sort_values('Rating', ascending=False).reset_index(drop=True)
ending_data.index = ending_data.index + 1

file_path = os.path.join(folder_path, f"baseball_{formatted_date}")
ending_data.to_csv(file_path)