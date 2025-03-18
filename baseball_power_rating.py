from datetime import datetime
import os
import pytz # type: ignore

cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year

folder_path = f"./PEAR/PEAR Baseball/y{current_season}"
os.makedirs(folder_path, exist_ok=True)

import requests # type: ignore
import re # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import warnings
warnings.filterwarnings("ignore")

def PEAR_Win_Prob(home_pr, away_pr):
    rating_diff = home_pr - away_pr
    win_prob = round(1 / (1 + 10 ** (-rating_diff / 7.5)) * 100, 2)
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

url = 'https://www.warrennolan.com/baseball/2025/rpi-predict'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', class_='normal-grid alternating-rows stats-table')
headers = [th.text.strip() for th in table.find('thead').find_all('th')]

data = []
for row in table.find('tbody').find_all('tr'):
    cells = row.find_all('td')
    if len(cells) >= 2:
        rank = cells[0].text.strip()  # First element (ranking)
        name_div = cells[1].find('div', class_='name-subcontainer')
        if name_div:
            full_text = name_div.text.strip()
        else:
            full_text = cells[1].text.strip()  # Fallback
        
        # Split the text to extract Team Name and Conference
        parts = full_text.split("\n")
        team_name = parts[0].strip()  # Extract just the team name
        conference = parts[1].split("(")[0].strip() if len(parts) > 1 else ""  # Extract conference
        data.append([rank, team_name, conference]) 
projected_rpi = pd.DataFrame(data, columns=["RPI", "Team", "Conference"])

url = 'https://www.warrennolan.com/baseball/2025/rpi-live'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', class_='normal-grid alternating-rows stats-table')
headers = [th.text.strip() for th in table.find('thead').find_all('th')]

data = []
for row in table.find('tbody').find_all('tr'):
    cells = row.find_all('td')
    if len(cells) >= 2:
        rank = cells[0].text.strip()  # First element (ranking)
        name_div = cells[1].find('div', class_='name-subcontainer')
        if name_div:
            full_text = name_div.text.strip()
        else:
            full_text = cells[1].text.strip()  # Fallback
        
        # Split the text to extract Team Name and Conference
        parts = full_text.split("\n")
        team_name = parts[0].strip()  # Extract just the team name
        conference = parts[1].split("(")[0].strip() if len(parts) > 1 else ""  # Extract conference
        data.append([rank, team_name, conference]) 
live_rpi = pd.DataFrame(data, columns=["Live_RPI", "Team", "Conference"])

# URL of the page to scrape
url = 'https://www.warrennolan.com/baseball/2025/elo'

# Fetch the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the specified class
table = soup.find('table', class_='normal-grid alternating-rows stats-table')

if table:
    # Extract table headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    headers.insert(1, "Team Link")  # Adding extra column for team link

    # Extract table rows
    data = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        row_data = []
        for i, cell in enumerate(cells):
            # If it's the first cell, extract team name and link from 'name-subcontainer'
            if i == 0:
                name_container = cell.find('div', class_='name-subcontainer')
                if name_container:
                    team_name = name_container.text.strip()
                    team_link_tag = name_container.find('a')
                    team_link = team_link_tag['href'] if team_link_tag else ''
                else:
                    team_name = cell.text.strip()
                    team_link = ''
                row_data.append(team_name)
                row_data.append(team_link)  # Add team link separately
            else:
                row_data.append(cell.text.strip())
        data.append(row_data)


    elo_data = pd.DataFrame(data, columns=[headers])
    elo_data.columns = elo_data.columns.get_level_values(0)
    elo_data = elo_data.drop_duplicates(subset='Team', keep='first')
    elo_data = elo_data.astype({col: 'str' for col in elo_data.columns if col not in ['ELO', 'Rank']})
    elo_data['ELO'] = elo_data['ELO'].astype(float, errors='ignore')
    elo_data['Rank'] = elo_data['Rank'].astype(int, errors='ignore')

else:
    print("Table not found on the page.")
print("Elo Load Done")

team_replacements = {
    'North Carolina St.': 'NC State',
    'Southern Miss': 'Southern Miss.',
    'USC': 'Southern California',
    'Dallas Baptist': 'DBU',
    'Charleston': 'Col. of Charleston',
    'Georgia Southern': 'Ga. Southern',
    'UNCG': 'UNC Greensboro',
    'East Tennessee St.': 'ETSU',
    'Lamar': 'Lamar University',
    "Saint Mary's College": "Saint Mary's (CA)",
    'Western Kentucky': 'Western Ky.',
    'FAU': 'Fla. Atlantic',
    'Connecticut': 'UConn',
    'Southeast Missouri': 'Southeast Mo. St.',
    'Alcorn St.': 'Alcorn',
    'Appalachian St.': 'App State',
    'Arkansas-Pine Bluff': 'Ark.-Pine Bluff',
    'Army': 'Army West Point',
    'Cal St. Bakersfield': 'CSU Bakersfield',
    'Cal St. Northridge': 'CSUN',
    'Central Arkansas': 'Central Ark.',
    'Central Michigan': 'Central Mich.',
    'Charleston Southern': 'Charleston So.',
    'Eastern Illinois': 'Eastern Ill.',
    'Eastern Kentucky': 'Eastern Ky.',
    'Eastern Michigan': 'Eastern Mich.',
    'Fairleigh Dickinson': 'FDU',
    'Grambling St.': 'Grambling',
    'Incarnate Word': 'UIW',
    'Long Island': 'LIU',
    'Maryland Eastern Shore': 'UMES',
    'Middle Tennessee': 'Middle Tenn.',
    'Mississippi Valley St.': 'Mississippi Val.',
    "Mount Saint Mary's": "Mount St. Mary's",
    'North Alabama': 'North Ala.',
    'North Carolina A&T': 'N.C. A&T',
    'Northern Colorado': 'Northern Colo.',
    'Northern Kentucky': 'Northern Ky.',
    'Prairie View A&M': 'Prairie View',
    'Presbyterian College': 'Presbyterian',
    'Saint Bonaventure': 'St. Bonaventure',
    "Saint John's": "St. John's (NY)",
    'Sam Houston St.': 'Sam Houston',
    'Seattle University': 'Seattle U',
    'South Carolina Upstate': 'USC Upstate',
    'South Florida': 'South Fla.',
    'Southeastern Louisiana': 'Southeastern La.',
    'Southern': 'Southern U.',
    'Southern Illinois': 'Southern Ill.',
    'Stephen F. Austin': 'SFA',
    'Tennessee-Martin': 'UT Martin',
    'Texas A&M-Corpus Christi': 'A&M-Corpus Christi',
    'UMass-Lowell': 'UMass Lowell',
    'UTA': 'UT Arlington',
    'Western Carolina': 'Western Caro.',
    'Western Illinois': 'Western Ill.',
    'Western Michigan': 'Western Mich.',
    'Albany': 'UAlbany',
    'Southern Indiana': 'Southern Ind.',
    'Queens': 'Queens (NC)',
    'Central Connecticut': 'Central Conn. St.',
    'Saint Thomas': 'St. Thomas (MN)',
    'Northern Illinois': 'NIU',
    'UMass':'Massachusetts',
    'Loyola-Marymount':'LMU (CA)'
}

elo_data['Team'] = elo_data['Team'].str.replace('State', 'St.', regex=False)
elo_data['Team'] = elo_data['Team'].replace(team_replacements)
elo_data.rename(columns={'Rank':'ELO_Rank'}, inplace=True)
projected_rpi['Team'] = projected_rpi['Team'].str.replace('State', 'St.', regex=False)
projected_rpi['Team'] = projected_rpi['Team'].replace(team_replacements)
live_rpi['Team'] = live_rpi['Team'].str.replace('State', 'St.', regex=False)
live_rpi['Team'] = live_rpi['Team'].replace(team_replacements)

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

dfs = [ba, bb, era, fp, obp, runs, slg, kp9, wp9, whip, elo_data]
for df in dfs:
    df["Team"] = df["Team"].str.strip()
df_combined = dfs[0]
for df in dfs[1:]:
    df_combined = pd.merge(df_combined, df, on="Team", how="inner")
baseball_stats = df_combined.loc[:, ~df_combined.columns.duplicated()].sort_values('Team').reset_index(drop=True)
baseball_stats['OPS'] = baseball_stats['SLG'] + baseball_stats['OBP']
baseball_stats['PYTHAG'] = round((baseball_stats['RS'] ** 1.83) / ((baseball_stats['RS'] ** 1.83) + (baseball_stats['RA'] ** 1.83)),3)


hbp = get_stat_dataframe('Hit by Pitch')[['Team', 'G', 'HBP']]
hits = get_stat_dataframe('Hits')[['Team', 'AB', 'H']]
doubles = get_stat_dataframe('Doubles')[['Team', '2B']]
triples = get_stat_dataframe('Triples')[['Team', '3B']]
sacrifice_flies = get_stat_dataframe('Sacrifice Flies')[['Team', 'SF']]
hit_batters = get_stat_dataframe('Hit Batters')[['Team', 'HB']]
stw_ratio = get_stat_dataframe('Strikeout-to-Walk Ratio')[['Team', 'K/BB', 'BB', 'SO']]
stw_ratio.rename(columns={'BB': 'PBB'}, inplace=True)
dfs = [bb, hbp, hits, doubles, triples, hr, sacrifice_flies, runs, sb, era, stw_ratio, ha, hit_batters]
for df in dfs:
    df["Team"] = df["Team"].str.strip()
wOBA = dfs[0]
for df in dfs[1:]:
    wOBA = pd.merge(wOBA, df, on="Team", how="left")
wOBA = wOBA.fillna(0)
wOBA['PA'] = wOBA['AB'] + wOBA['BB'] + wOBA['HBP'] + wOBA['SF'] + wOBA['SB']
league_HR_per_game = wOBA['HR'].sum() / wOBA['G'].sum()
wOBA['HR_A'] = wOBA['G'] * league_HR_per_game
wOBA['1B'] = wOBA['H'] - wOBA['2B'] - wOBA['3B'] - wOBA['HR']
wOBA['wOBA'] = ((0.69 * wOBA['BB']) + (0.72 * wOBA['HBP']) + (0.88 * wOBA['1B']) + (1.24 * wOBA['2B']) + (1.56 * wOBA['3B']) + (1.95 * wOBA['HR'])) / (wOBA['PA'])
league_wOBA = (wOBA['wOBA'] * wOBA['PA']).sum() / wOBA['PA'].sum()
league_R_PA = wOBA['RS'].sum() / wOBA['PA'].sum()
wOBA_scale = league_R_PA / league_wOBA
wOBA['wRAA'] = ((wOBA['wOBA'] - league_wOBA) / wOBA_scale) * wOBA['PA']
league_RS = wOBA['RS'].sum()
league_G = wOBA['G'].sum()
RPW = 2 * (league_RS / league_G)
wOBA['oWAR'] = wOBA['wRAA'] / RPW
wOBA['ISO'] = (wOBA['2B'] + (2 * wOBA['3B']) + (3 * wOBA['HR'])) / wOBA['AB']
wOBA['wRC'] = (((wOBA['wOBA'] - league_wOBA) / wOBA_scale) + league_R_PA) * wOBA['PA']
wOBA['wRC+'] = (wOBA['wRC'] / wOBA['PA']) / league_R_PA * 100
wOBA['BB%'] = wOBA['BB'] / wOBA['PA']
wOBA['BABIP'] = (wOBA['H'] - wOBA['HR']) / (wOBA['AB'] + wOBA['SF'])
wOBA['RA9'] = (wOBA['RA'] / wOBA['IP']) * 9
wOBA['LOB%'] = (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - wOBA['RA']) / (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - (1.4*wOBA['HR_A']))
wOBA['FIP'] = ((13 * wOBA['HR_A'] + 3 * (wOBA['PBB'] + wOBA['HB']) - 2 * wOBA['SO']) / wOBA['IP'])
league_RA9 = wOBA['RA'].sum() / wOBA['G'].sum()
league_ERA = (wOBA['ER'].sum() * 9) / wOBA['IP'].sum()
league_FIP = (wOBA['FIP'] * wOBA['IP']).sum() / wOBA['IP'].sum()
replacement_level_FIP = wOBA['FIP'].quantile(0.80)
multiplier = replacement_level_FIP / league_FIP
replacement_RA9 = league_RA9 * multiplier  # Adjust RA9 to match replacement level
RPW = 9 / (2 * (wOBA['RA'].sum() / wOBA['G'].sum()))
wOBA['pWAR'] = ((replacement_RA9 - wOBA['FIP']) / RPW) * (wOBA['IP'] / 9)
mean_oWAR = wOBA['oWAR'].mean()
std_oWAR = wOBA['oWAR'].std()
mean_pWAR = wOBA['pWAR'].mean()
std_pWAR = wOBA['pWAR'].std()
wOBA['oWAR_z'] = (wOBA['oWAR'] - mean_oWAR) / std_oWAR
wOBA['pWAR_z'] = (wOBA['pWAR'] - mean_pWAR) / std_pWAR
wOBA['fWAR'] = wOBA['oWAR_z'] + wOBA['pWAR_z']
baseball_stats = pd.merge(baseball_stats, wOBA[['Team', 'wOBA', 'wRAA', 'oWAR_z', 'pWAR_z', 'fWAR', 'ISO', 'wRC+', 'BB%', 'BABIP', 'RA9', 'FIP', 'LOB%', 'K/BB']], how='left', on='Team')

rpi_2024 = pd.read_csv("./PEAR/PEAR Baseball/rpi_end_2024.csv")

modeling_stats = baseball_stats[['Team', 'HPG',
                'BBPG', 'ERA', 'PCT', 
                'KP9', 'WP9', 'OPS', 
                'WHIP', 'PYTHAG', 'fWAR', 'oWAR_z', 'pWAR_z', 'K/BB', 'wRC+', 'LOB%', 'ELO_Rank']]
modeling_stats = pd.merge(modeling_stats, rpi_2024[['Team', 'Rank']], on = 'Team', how='left')
modeling_stats["Rank"] = modeling_stats["Rank"].apply(pd.to_numeric, errors='coerce')
modeling_stats["ELO_Rank"] = modeling_stats["ELO_Rank"].apply(pd.to_numeric, errors='coerce')
modeling_stats['Rank_pct'] = 1 - (modeling_stats['Rank'] - 1) / (len(modeling_stats) - 1)
higher_better = ["HPG", "BBPG", "PCT", "KP9", "OPS", "Rank_pct", 'PYTHAG', 'fWAR', 'oWAR_z', 'pWAR_z', 'K/BB', 'wRC+', 'LOB%']
lower_better = ["ERA", "WP9", "WHIP"]
scaler = MinMaxScaler(feature_range=(1, 100))
modeling_stats[higher_better] = scaler.fit_transform(modeling_stats[higher_better])
modeling_stats[lower_better] = scaler.fit_transform(-modeling_stats[lower_better])
weights = {
    'fWAR': .40, 'PYTHAG': .30, 'K/BB': .10, 'wRC+': .10, 'LOB%': .10
}
modeling_stats['in_house_pr'] = sum(modeling_stats[stat] * weight for stat, weight in weights.items())

modeling_stats['in_house_pr'] = modeling_stats['in_house_pr'] - modeling_stats['in_house_pr'].mean()
current_range = modeling_stats['in_house_pr'].max() - modeling_stats['in_house_pr'].min()
desired_range = 100
scaling_factor = desired_range / current_range
modeling_stats['in_house_pr'] = round(modeling_stats['in_house_pr'] * scaling_factor, 4)
modeling_stats['in_house_pr'] = modeling_stats['in_house_pr'] - modeling_stats['in_house_pr'].min()

import pandas as pd
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm

def objective_function(weights):
    (w_owar, w_pwar, w_kbb, w_wrc, w_pythag, w_fwar, w_lob, w_rank, w_in_house_pr) = weights
    
    modeling_stats['power_ranking'] = (
        w_fwar * modeling_stats['fWAR'] +
        w_owar * modeling_stats['oWAR_z'] +
        w_pwar * modeling_stats['pWAR_z'] +
        w_pythag * modeling_stats['PYTHAG'] + 
        w_kbb * modeling_stats['K/BB'] +
        w_wrc * modeling_stats['wRC+'] +
        w_lob * modeling_stats['LOB%'] +
        w_rank * modeling_stats['Rank_pct'] +
        w_in_house_pr * modeling_stats['in_house_pr']
    )

    modeling_stats['calculated_rank'] = modeling_stats['power_ranking'].rank(ascending=False)
    modeling_stats['combined_rank'] = (
        modeling_stats['ELO_Rank']
    )
    spearman_corr = modeling_stats[['calculated_rank', 'combined_rank']].corr(method='spearman').iloc[0,1]

    return -spearman_corr

bounds = [(0,1),
          (0,1),
          (0,1),
          (0,1),
          (0,1),
          (0,1),
          (0,1),
          (0,0.05),
          (0,1)]
result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42)
optimized_weights = result.x
modeling_stats = modeling_stats.sort_values('power_ranking', ascending=False).reset_index(drop=True)

modeling_stats['Rating'] = modeling_stats['power_ranking'] - modeling_stats['power_ranking'].mean()
current_range = modeling_stats['Rating'].max() - modeling_stats['Rating'].min()
desired_range = 15
scaling_factor = desired_range / current_range
modeling_stats['Rating'] = round(modeling_stats['Rating'] * scaling_factor, 4)
modeling_stats['Rating'] = modeling_stats['Rating'] - modeling_stats['Rating'].min()
modeling_stats['Rating'] = round(modeling_stats['Rating'] - modeling_stats['Rating'].mean(),2)
modeling_stats['Rating'] = round(modeling_stats['Rating'], 2)

ending_data = pd.merge(baseball_stats, modeling_stats[['Team', 'Rating']], on="Team", how="inner").sort_values('Rating', ascending=False).reset_index(drop=True)
# ending_data = ending_data.drop(columns=['SOR', 'SOS'])
ending_data.index = ending_data.index + 1
# ending_data[['Wins', 'Losses']] = ending_data['Rec'].str.split('-', expand=True).astype(int)
# ending_data['WIN%'] = round(ending_data['Wins'] / (ending_data['Wins'] + ending_data['Losses']), 3)
# ending_data['Wins_Over_Pythag'] = ending_data['WIN%'] - ending_data['PYTHAG']

import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore

# URL of the page to scrape
url = 'https://www.warrennolan.com/baseball/2025/elo'

# Fetch the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the specified class
table = soup.find('table', class_='normal-grid alternating-rows stats-table')

if table:
    # Extract table headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    headers.insert(1, "Team Link")  # Adding extra column for team link

    # Extract table rows
    data = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        row_data = []
        for i, cell in enumerate(cells):
            # If it's the first cell, extract team name and link from 'name-subcontainer'
            if i == 0:
                name_container = cell.find('div', class_='name-subcontainer')
                if name_container:
                    team_name = name_container.text.strip()
                    team_link_tag = name_container.find('a')
                    team_link = team_link_tag['href'] if team_link_tag else ''
                else:
                    team_name = cell.text.strip()
                    team_link = ''
                row_data.append(team_name)
                row_data.append(team_link)  # Add team link separately
            else:
                row_data.append(cell.text.strip())
        data.append(row_data)


    elo_data = pd.DataFrame(data, columns=[headers])
    elo_data.columns = elo_data.columns.get_level_values(0)
    elo_data = elo_data.drop_duplicates(subset='Team', keep='first')
    elo_data = elo_data.astype({col: 'str' for col in elo_data.columns if col not in ['ELO', 'Rank']})
    elo_data['ELO'] = elo_data['ELO'].astype(float, errors='ignore')
    elo_data['Rank'] = elo_data['Rank'].astype(int, errors='ignore')

else:
    print("Table not found on the page.")
print("Elo Load Done")

import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore
import time

# Base URL for Warren Nolan
BASE_URL = "https://www.warrennolan.com"

# Initialize storage for schedule data
schedule_data = []
schedule_counter = 1
# Iterate over each team's schedule link
for _, row in elo_data.iterrows():
    team_name = row["Team"]
    if (schedule_counter % 50 == 0):
        print(f"{schedule_counter}/{len(elo_data)}")
    schedule_counter = schedule_counter + 1
    team_schedule_url = BASE_URL + row["Team Link"]
    
    response = requests.get(team_schedule_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the team name
    # team_name = soup.find("h1").text.strip() if soup.find("h1") else "Unknown"

    # Find the team schedule list
    schedule_lists = soup.find_all("ul", class_="team-schedule")
    if not schedule_lists:
        continue  # Skip if no schedule is found

    schedule_list = schedule_lists[0]

    # Iterate over each game row in the schedule
    for game in schedule_list.find_all('li', class_='team-schedule'):
        # Extract Date
        date_month = game.find('span', class_='team-schedule__game-date--month').text.strip()
        date_day = game.find('span', class_='team-schedule__game-date--day').text.strip()
        date_dow = game.find('span', class_='team-schedule__game-date--dow').text.strip()
        game_date = f"{date_month} {date_day} ({date_dow})"

        # Extract Opponent Name (Handle missing cases)
        opponent_info = game.find('div', class_='team-schedule__opp')
        if opponent_info:
            opponent_link_element = opponent_info.find('a', class_='team-schedule__opp-line-link')
            opponent_name = opponent_link_element.text.strip() if opponent_link_element else ""
        else:
            opponent_name = ""

        # Extract Location
        location_info = game.find('div', class_='team-schedule__info')
        location = location_info.text.strip() if location_info else "Unknown"

        # Extract Game Result
        result_info = game.find('div', class_='team-schedule__result')
        result_text = result_info.text.strip() if result_info else "N/A"

        # Extract Home/Away Teams from Box Score and scores
        home_score, away_score = "", ""  # Initialize scores as empty strings

        box_score_table = game.find('table', class_='team-schedule-bottom__box-score')
        if box_score_table:
            rows = box_score_table.find_all('tr')
            if len(rows) > 2:
                away_team = rows[1].find_all('td')[0].text.strip()
                home_team = rows[2].find_all('td')[0].text.strip()

                # Extracting Runs
                away_score = rows[1].find_all('td')[-3].text.strip()  # Away runs
                home_score = rows[2].find_all('td')[-3].text.strip()  # Home runs
            else:
                home_team, away_team = "N/A", "N/A"
        else:
            home_team, away_team = "N/A", "N/A"

        # Append to schedule data
        schedule_data.append([team_name, game_date, opponent_name, location, result_text, home_team, away_team, home_score, away_score])

# Convert to DataFrame
columns = ["Team", "Date", "Opponent", "Location", "Result", "home_team", "away_team", "home_score", "away_score"]
schedule_df = pd.DataFrame(schedule_data, columns=columns)
schedule_df = schedule_df.astype({col: 'str' for col in schedule_df.columns if col not in ['home_score', 'away_score']})
schedule_df['home_score'] = schedule_df['home_score'].astype(int, errors='ignore')
schedule_df['away_score'] = schedule_df['away_score'].astype(int, errors='ignore')
schedule_df = schedule_df.merge(elo_data[['Team', 'ELO']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'ELO': 'home_elo'}, inplace=True)
schedule_df = schedule_df.merge(elo_data[['Team', 'ELO']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'ELO': 'away_elo'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)

print("Schedule Load Done")

import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore

# URL of the page to scrape
url = 'https://www.warrennolan.com/baseball/2025/elo'

# Fetch the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the specified class
table = soup.find('table', class_='normal-grid alternating-rows stats-table')

if table:
    # Extract table headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    headers.insert(1, "Team Link")  # Adding extra column for team link

    # Extract table rows
    data = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        row_data = []
        for i, cell in enumerate(cells):
            # If it's the first cell, extract team name and link from 'name-subcontainer'
            if i == 0:
                name_container = cell.find('div', class_='name-subcontainer')
                if name_container:
                    team_name = name_container.text.strip()
                    team_link_tag = name_container.find('a')
                    team_link = team_link_tag['href'] if team_link_tag else ''
                else:
                    team_name = cell.text.strip()
                    team_link = ''
                row_data.append(team_name)
                row_data.append(team_link)  # Add team link separately
            else:
                row_data.append(cell.text.strip())
        data.append(row_data)


    elo_data = pd.DataFrame(data, columns=[headers])
    elo_data.columns = elo_data.columns.get_level_values(0)
    elo_data = elo_data.drop_duplicates(subset='Team', keep='first')
    elo_data = elo_data.astype({col: 'str' for col in elo_data.columns if col not in ['ELO', 'Rank']})
    elo_data['ELO'] = elo_data['ELO'].astype(float, errors='ignore')
    elo_data['Rank'] = elo_data['Rank'].astype(int, errors='ignore')

else:
    print("Table not found on the page.")

# Define mapping for team name replacements
team_replacements = {
    'North Carolina St.': 'NC State',
    'Southern Miss': 'Southern Miss.',
    'USC': 'Southern California',
    'Dallas Baptist': 'DBU',
    'Charleston': 'Col. of Charleston',
    'Georgia Southern': 'Ga. Southern',
    'UNCG': 'UNC Greensboro',
    'East Tennessee St.': 'ETSU',
    'Lamar': 'Lamar University',
    "Saint Mary's College": "Saint Mary's (CA)",
    'Western Kentucky': 'Western Ky.',
    'FAU': 'Fla. Atlantic',
    'Connecticut': 'UConn',
    'Southeast Missouri': 'Southeast Mo. St.',
    'Alcorn St.': 'Alcorn',
    'Appalachian St.': 'App State',
    'Arkansas-Pine Bluff': 'Ark.-Pine Bluff',
    'Army': 'Army West Point',
    'Cal St. Bakersfield': 'CSU Bakersfield',
    'Cal St. Northridge': 'CSUN',
    'Central Arkansas': 'Central Ark.',
    'Central Michigan': 'Central Mich.',
    'Charleston Southern': 'Charleston So.',
    'Eastern Illinois': 'Eastern Ill.',
    'Eastern Kentucky': 'Eastern Ky.',
    'Eastern Michigan': 'Eastern Mich.',
    'Fairleigh Dickinson': 'FDU',
    'Grambling St.': 'Grambling',
    'Incarnate Word': 'UIW',
    'Long Island': 'LIU',
    'Maryland Eastern Shore': 'UMES',
    'Middle Tennessee': 'Middle Tenn.',
    'Mississippi Valley St.': 'Mississippi Val.',
    "Mount Saint Mary's": "Mount St. Mary's",
    'North Alabama': 'North Ala.',
    'North Carolina A&T': 'N.C. A&T',
    'Northern Colorado': 'Northern Colo.',
    'Northern Kentucky': 'Northern Ky.',
    'Prairie View A&M': 'Prairie View',
    'Presbyterian College': 'Presbyterian',
    'Saint Bonaventure': 'St. Bonaventure',
    "Saint John's": "St. John's (NY)",
    'Sam Houston St.': 'Sam Houston',
    'Seattle University': 'Seattle U',
    'South Carolina Upstate': 'USC Upstate',
    'South Florida': 'South Fla.',
    'Southeastern Louisiana': 'Southeastern La.',
    'Southern': 'Southern U.',
    'Southern Illinois': 'Southern Ill.',
    'Stephen F. Austin': 'SFA',
    'Tennessee-Martin': 'UT Martin',
    'Texas A&M-Corpus Christi': 'A&M-Corpus Christi',
    'UMass-Lowell': 'UMass Lowell',
    'UTA': 'UT Arlington',
    'Western Carolina': 'Western Caro.',
    'Western Illinois': 'Western Ill.',
    'Western Michigan': 'Western Mich.',
    'Albany': 'UAlbany',
    'Southern Indiana': 'Southern Ind.',
    'Queens': 'Queens (NC)',
    'Central Connecticut': 'Central Conn. St.',
    'Saint Thomas': 'St. Thomas (MN)',
    'Northern Illinois': 'NIU',
    'UMass':'Massachusetts',
    'Loyola-Marymount':'LMU (CA)'
}

# Apply replacements and standardize 'State' to 'St.'
columns_to_replace = ['Team', 'home_team', 'away_team', 'Opponent']

for col in columns_to_replace:
    schedule_df[col] = schedule_df[col].str.replace('State', 'St.', regex=False)
    schedule_df[col] = schedule_df[col].replace(team_replacements)
elo_data['Team'] = elo_data['Team'].str.replace('State', 'St.', regex=False)
elo_data['Team'] = elo_data['Team'].replace(team_replacements)

import pandas as pd # type: ignore

# Mapping months to numerical values
month_mapping = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

current_season = 2025  # Set the current season

# Function to convert "FEB 14 (FRI)" format to "mm-dd-yyyy"
def convert_date(date_str):
    # Ensure date is a string before splitting
    if isinstance(date_str, pd.Timestamp):
        date_str = date_str.strftime("%b %d (%a)").upper()  # Convert to same format
    
    parts = date_str.split()  # ["FEB", "14", "(FRI)"]
    month = month_mapping[parts[0].upper()]  # Convert month to number
    day = parts[1]  # Extract day
    return f"{month}-{day}-{current_season}"

# Apply function to convert date format
schedule_df["Date"] = schedule_df["Date"].astype(str).apply(convert_date)
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], format="%m-%d-%Y")
comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")

missing_rating = round(ending_data['Rating'].mean() - 1.5*ending_data['Rating'].std(),2)
schedule_df = schedule_df.merge(ending_data[['Team', 'Rating']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'Rating': 'home_rating'}, inplace=True)
schedule_df = schedule_df.merge(ending_data[['Team', 'Rating']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'Rating': 'away_rating'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)
schedule_df['home_rating'].fillna(missing_rating, inplace=True)
schedule_df['away_rating'].fillna(missing_rating, inplace=True)
schedule_df['home_win_prob'] = schedule_df.apply(
    lambda row: PEAR_Win_Prob(row['home_rating'], row['away_rating']) / 100, axis=1
)
remaining_games = schedule_df[schedule_df["Date"] > comparison_date].reset_index(drop=True)

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 1.5
schedule_df['elo_win_prob'] = round((10**((schedule_df['home_elo'] - schedule_df['away_elo']) / 400)) / ((10**((schedule_df['home_elo'] - schedule_df['away_elo']) / 400)) + 1)*100,2)
schedule_df['Spread'] = (schedule_df['home_rating'] + (schedule_df['elo_win_prob'].apply(adjust_home_pr)) - schedule_df['away_rating']).round(2)
schedule_df['PEAR'] = schedule_df.apply(
    lambda row: f"{row['away_team']} {-abs(row['Spread'])}" if ((row['Spread'] <= 0)) 
    else f"{row['home_team']} {-abs(row['Spread'])}", axis=1)
completed_schedule = schedule_df[
    (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
].reset_index(drop=True)
completed_schedule = completed_schedule[completed_schedule["Result"].str.startswith(("W", "L"))]

straight_up_calculator = completed_schedule.copy()

def rpi_components(completed_schedule):
    teams = set(completed_schedule['home_team']).union(set(completed_schedule['away_team']))
    team_records = {team: {'wins': 0, 'losses': 0} for team in teams}
    
    for _, row in completed_schedule.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        home_score, away_score = row['home_score'], row['away_score']
        if home_score > away_score:
            team_records[home_team]['wins'] += 1
            team_records[away_team]['losses'] += 1
        else:
            team_records[away_team]['wins'] += 1
            team_records[home_team]['losses'] += 1

    wp = {team: record['wins'] / (record['wins'] + record['losses']) if (record['wins'] + record['losses']) > 0 else 0 
          for team, record in team_records.items()}
    
    team_opponents = {team: set() for team in teams}
    for _, row in completed_schedule.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        team_opponents[home_team].add(away_team)
        team_opponents[away_team].add(home_team)
    owp = {}
    for team in teams:
        opponents = team_opponents[team]
        opponent_wps = []
        for opp in opponents:
            opp_games = completed_schedule[
                (completed_schedule['home_team'] == opp) | (completed_schedule['away_team'] == opp)
            ]
            opp_wins = 0
            opp_losses = 0
            for _, game in opp_games.iterrows():
                if game['home_team'] == opp:
                    opp_team = game['away_team']
                    if game['home_score'] > game['away_score']:
                        opp_wins += 1
                    else:
                        opp_losses += 1
                else:
                    opp_team = game['home_team']
                    if game['away_score'] > game['home_score']:
                        opp_wins += 1
                    else:
                        opp_losses += 1
                
                if opp_team == team:
                    if game['home_team'] == opp:
                        opp_wins -= 1 if game['home_score'] > game['away_score'] else 0
                        opp_losses -= 1 if game['home_score'] < game['away_score'] else 0
                    else:
                        opp_wins -= 1 if game['away_score'] > game['home_score'] else 0
                        opp_losses -= 1 if game['away_score'] < game['home_score'] else 0

            if opp_wins + opp_losses > 0:
                opponent_wps.append(opp_wins / (opp_wins + opp_losses))
        
        owp[team] = sum(opponent_wps) / len(opponent_wps) if opponent_wps else 0
    
    oowp = {team: sum(owp[opp] for opp in team_opponents[team]) / len(team_opponents[team]) if team_opponents[team] else 0 
            for team in teams}
    
    rpi_components_df = pd.DataFrame({
        "Team": list(teams),
        "WP": [wp[team] for team in teams],
        "OWP": [owp[team] for team in teams],
        "OOWP": [oowp[team] for team in teams]
    })

    return rpi_components_df

pear_rpi = rpi_components(completed_schedule)
pear_rpi = pd.merge(pear_rpi, live_rpi, on='Team', how='left')
pear_rpi = pear_rpi.dropna()

def rpi_calculation(weights):
    w_wp, w_owp = weights
    w_oowp = 1 - (w_wp + w_owp)

    if w_oowp < 0 or w_oowp > 1:
        return float('inf')

    pear_rpi['RPI_Score'] = (
        w_wp * pear_rpi['WP'] +
        w_owp * pear_rpi['OWP'] +
        w_oowp * pear_rpi['OOWP']
    )

    pear_rpi['RPI'] = pear_rpi['RPI_Score'].rank(ascending=False).astype(int)
    pear_rpi['combined_rank'] = pear_rpi['Live_RPI']
    spearman_corr = pear_rpi[['RPI', 'combined_rank']].corr(method='spearman').iloc[0,1]

    return -spearman_corr

bounds = [(0,1), (0,1)]  
result = differential_evolution(rpi_calculation, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42)
optimized_weights = result.x
print("RPI Calculation Weights:")
print("------------------------")
print(f"Win Prob: {optimized_weights[0]}")
print(f"Opp Win Prob: {optimized_weights[1]}")
print(f"Opp Opp Win Prob: {1 - (optimized_weights[0] + optimized_weights[1])}")

def calculate_expected_wins(group):
    # Initialize a variable to accumulate expected wins
    expected_wins = 0
    schedule_wins = 0
    schedule_losses = 0
    
    # Iterate over the rows of the group
    for _, row in group.iterrows():
        if row['Team'] == row['home_team']:
            expected_wins += row['home_win_prob']
            if row['home_score'] > row['away_score']:
                schedule_wins += 1
            else:
                schedule_losses += 1
        else:
            expected_wins += 1 - row['home_win_prob']
            if row['away_score'] > row['home_score']:
                schedule_wins += 1
            else:
                schedule_losses += 1
    
    # Return the total expected_wins for this group
    return pd.Series({'Team': group['Team'].iloc[0], 'expected_wins': expected_wins, 'Wins':schedule_wins, 'Losses':schedule_losses})

# Group by 'Team' and apply the calculation
team_expected_wins = completed_schedule.groupby('Team').apply(calculate_expected_wins).reset_index(drop=True)

def calculate_average_expected_wins(group, average_team):
    total_expected_wins = 0

    for _, row in group.iterrows():
        if row['Team'] == row['home_team']:
            total_expected_wins += PEAR_Win_Prob(average_team, row['away_rating']) / 100
        else:
            total_expected_wins += 1 - PEAR_Win_Prob(row['home_rating'], average_team) / 100

    avg_expected_wins = total_expected_wins / len(group)

    return pd.Series({'Team': group['Team'].iloc[0], 'avg_expected_wins': avg_expected_wins, 'total_expected_wins':total_expected_wins})

average_team = ending_data['Rating'].mean()
avg_team_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, average_team).reset_index(drop=True)

rem_avg_expected_wins = remaining_games.groupby('Team').apply(calculate_average_expected_wins, average_team).reset_index(drop=True)
rem_avg_expected_wins.rename(columns={"avg_expected_wins": "rem_avg_expected_wins", "total_expected_wins":"rem_total_expected_wins"}, inplace=True)

def calculate_kpi(completed_schedule, ending_data):
    def get_team_rank(team):
        match = ending_data.loc[ending_data["Team"] == team]
        return match.index[0] if not match.empty else len(ending_data) + 1

    def get_opponent_rank(opponent):
        match = ending_data.loc[ending_data["Team"] == opponent]
        return match.index[0] if not match.empty else len(ending_data) + 1

    kpi_scores = []

    for _, game in completed_schedule.iterrows():
        team = game["Team"]
        opponent = game["Opponent"]
        home_team = game["home_team"]
        
        # Team strength
        team_rank = get_team_rank(team)
        opponent_rank = get_opponent_rank(opponent)

        # Opponent strength calculation
        opponent_strength_win = 1 - (opponent_rank / (len(ending_data) + 1))
        opponent_strength_loss = (opponent_rank / (len(ending_data) + 1))
        
        # Determine if the team is home
        is_home = team == home_team
        
        # Scoring margin
        margin = game["home_score"] - game["away_score"]
        if not is_home:
            margin = -margin  # Flip if the team is away

        # Win or loss multiplier
        result_multiplier = 1.5 if margin > 0 else -1.5

        # Margin factor
        if margin > 0:
            margin_factor = 1 + (min(margin, 20) / 20)
            opponent_strength = opponent_strength_win
        else:
            margin_factor = max(0.1, 1 - (min(abs(margin), 20) / 20))
            opponent_strength = opponent_strength_loss

        # Team strength adjustment
        team_strength_adj = 1 - (team_rank / (len(ending_data) + 1))

        # Adjusted KPI formula
        adj_grv = (opponent_strength * result_multiplier * margin_factor / 1.5) * (1 + (team_strength_adj / 2))
        
        # Store result
        kpi_scores.append({"Team": team, "KPI_Score": adj_grv})

    # Convert to DataFrame and get average per team
    kpi_df = pd.DataFrame(kpi_scores)
    kpi_avg = kpi_df.groupby("Team")["KPI_Score"].mean().reset_index()

    return kpi_avg

# Call function
kpi_results = calculate_kpi(completed_schedule, ending_data).sort_values('KPI_Score', ascending=False).reset_index(drop=True)

def calculate_resume_quality(group, bubble_team_rating):
    results = []
    resume_quality = 0
    for _, row in group.iterrows():
        team = row['Team']
        is_home = row["home_team"] == team
        is_away = row["away_team"] == team
        opponent_rating = row["away_rating"] if is_home else row["home_rating"]
        win_prob = PEAR_Win_Prob(bubble_team_rating, opponent_rating) / 100
        team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
        if team_won:
            resume_quality += (1-win_prob)
        else:
            resume_quality -= win_prob
    # resume_quality = resume_quality / len(group)
    results.append({"Team": team, "resume_quality": resume_quality})
    return pd.DataFrame(results)

def calculate_game_resume_quality(row, one_seed_rating):
    """Calculate resume quality for a single game."""
    team = row["Team"]
    is_home = row["home_team"] == team
    is_away = row["away_team"] == team
    opponent_rating = row["away_rating"] if is_home else row["home_rating"]
    
    win_prob = PEAR_Win_Prob(one_seed_rating, opponent_rating) / 100
    team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
    
    return (1 - win_prob) if team_won else -win_prob

df_1 = pd.merge(ending_data, team_expected_wins[['Team', 'expected_wins', 'Wins', 'Losses']], on='Team', how='left')
df_2 = pd.merge(df_1, avg_team_expected_wins[['Team', 'avg_expected_wins', 'total_expected_wins']], on='Team', how='left')
df_3 = pd.merge(df_2, rem_avg_expected_wins[['Team', 'rem_avg_expected_wins', 'rem_total_expected_wins']], on='Team', how='left')
df_4 = pd.merge(df_3, projected_rpi[['Team', 'RPI', 'Conference']], on='Team', how='left')
df_4.rename(columns={'RPI': 'Projected_RPI'}, inplace=True)
df_5 = pd.merge(df_4, pear_rpi, on='Team', how='left')
stats_and_metrics = pd.merge(df_5, kpi_results, on='Team', how='left')
stats_and_metrics['RPI'] = stats_and_metrics['RPI_Score'].rank(ascending=False).astype(int)

stats_and_metrics['wins_above_expected'] = round(stats_and_metrics['Wins'] - stats_and_metrics['total_expected_wins'],2)
stats_and_metrics['SOR'] = stats_and_metrics['wins_above_expected'].rank(method='min', ascending=False).astype(int)
max_SOR = stats_and_metrics['SOR'].max()
stats_and_metrics['SOR'].fillna(max_SOR + 1, inplace=True)
stats_and_metrics['SOR'] = stats_and_metrics['SOR'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('SOR').reset_index(drop=True)

stats_and_metrics['RemSOS'] = stats_and_metrics['rem_avg_expected_wins'].rank(method='min', ascending=True).astype(int)
max_remSOS = stats_and_metrics['RemSOS'].max()
stats_and_metrics['RemSOS'].fillna(max_remSOS + 1, inplace=True)
stats_and_metrics['RemSOS'] = stats_and_metrics['RemSOS'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('RemSOS').reset_index(drop=True)

stats_and_metrics['SOS'] = stats_and_metrics['avg_expected_wins'].rank(method='min', ascending=True).astype(int)
max_SOS = stats_and_metrics['SOS'].max()
stats_and_metrics['SOS'].fillna(max_SOS + 1, inplace=True)
stats_and_metrics['SOS'] = stats_and_metrics['SOS'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('SOS').reset_index(drop=True)

stats_and_metrics['ELO'].fillna(1200, inplace=True)

bubble_rating = stats_and_metrics.loc[(stats_and_metrics["SOR"] >= 32) & (stats_and_metrics["SOR"] <= 40), "Rating"].mean()
bubble_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, bubble_rating).reset_index(drop=True)
bubble_expected_wins.rename(columns={"avg_expected_wins": "bubble_expected_wins", "total_expected_wins":"bubble_total_expected_wins"}, inplace=True)

stats_and_metrics = pd.merge(stats_and_metrics, bubble_expected_wins, on='Team', how='left')

stats_and_metrics['wins_above_bubble'] = round(stats_and_metrics['Wins'] - stats_and_metrics['bubble_total_expected_wins'],2)
stats_and_metrics['Prelim_WAB'] = stats_and_metrics['wins_above_bubble'].rank(method='min', ascending=False).astype(int)
max_WAB = stats_and_metrics['Prelim_WAB'].max()
stats_and_metrics['Prelim_WAB'].fillna(max_WAB + 1, inplace=True)
stats_and_metrics['Prelim_WAB'] = stats_and_metrics['Prelim_WAB'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('Prelim_WAB').reset_index(drop=True)

stats_and_metrics['KPI'] = stats_and_metrics['KPI_Score'].rank(method='min', ascending=False).astype(int)
max_KPI = stats_and_metrics['KPI'].max()
stats_and_metrics['KPI'].fillna(max_KPI + 1, inplace=True)
stats_and_metrics['KPI'] = stats_and_metrics['KPI'].astype(int)

stats_and_metrics['Prelim_AVG'] = round(stats_and_metrics[['KPI', 'Prelim_WAB', 'SOR']].mean(axis=1),1)

bubble_rating = stats_and_metrics.loc[(stats_and_metrics['Prelim_AVG'] >= 32) & (stats_and_metrics['Prelim_AVG'] <= 40), "Rating"].mean()
bubble_expected_wins = completed_schedule.groupby('Team').apply(calculate_average_expected_wins, bubble_rating).reset_index(drop=True)
bubble_expected_wins.rename(columns={"avg_expected_wins": "final_bubble_expected_wins", "total_expected_wins":"final_bubble_total_expected_wins"}, inplace=True)
stats_and_metrics = pd.merge(stats_and_metrics, bubble_expected_wins, on='Team', how='left')

stats_and_metrics['wins_above_bubble'] = round(stats_and_metrics['Wins'] - stats_and_metrics['final_bubble_total_expected_wins'],2)
stats_and_metrics['WAB'] = stats_and_metrics['wins_above_bubble'].rank(method='min', ascending=False).astype(int)
max_WAB = stats_and_metrics['WAB'].max()
stats_and_metrics['WAB'].fillna(max_WAB + 1, inplace=True)
stats_and_metrics['WAB'] = stats_and_metrics['WAB'].astype(int)
stats_and_metrics = stats_and_metrics.sort_values('WAB').reset_index(drop=True)

stats_and_metrics['AVG'] = round(stats_and_metrics[['KPI', 'WAB', 'SOR']].mean(axis=1),1)
stats_and_metrics['Resume'] = stats_and_metrics['AVG'].rank(method='min').astype(int)
stats_and_metrics = stats_and_metrics.sort_values('Resume').reset_index(drop=True)

bubble_team_rating = stats_and_metrics.loc[31, 'Rating']
resume_quality = completed_schedule.groupby('Team').apply(calculate_resume_quality, bubble_team_rating).reset_index(drop=True)
resume_quality['RQI'] = resume_quality['resume_quality'].rank(method='min', ascending=False).astype(int)
resume_quality = resume_quality.sort_values('RQI').reset_index(drop=True)
resume_quality['resume_quality'] = resume_quality['resume_quality'] - resume_quality.loc[15, 'resume_quality']
stats_and_metrics = pd.merge(stats_and_metrics, resume_quality, on='Team', how='left')
schedule_df["resume_quality"] = schedule_df.apply(lambda row: calculate_game_resume_quality(row, bubble_team_rating), axis=1)

stats_and_metrics["Norm_Rating"] = (stats_and_metrics["Rating"] - stats_and_metrics["Rating"].min()) / (stats_and_metrics["Rating"].max() - stats_and_metrics["Rating"].min())
stats_and_metrics["Norm_RQI"] = (stats_and_metrics["resume_quality"] - stats_and_metrics["resume_quality"].min()) / (stats_and_metrics["resume_quality"].max() - stats_and_metrics["resume_quality"].min())
stats_and_metrics["Norm_RPI"] = (stats_and_metrics["RPI_Score"] - stats_and_metrics["RPI_Score"].min()) / (stats_and_metrics["RPI_Score"].max() - stats_and_metrics["RPI_Score"].min())
stats_and_metrics["Norm_SOS"] = 1 - (stats_and_metrics["avg_expected_wins"] - stats_and_metrics["avg_expected_wins"].min()) / (stats_and_metrics["avg_expected_wins"].max() - stats_and_metrics["avg_expected_wins"].min())  # Inverted

def calculate_net(weights):
    w_rating, w_sos = weights
    w_rqi = 1 - (w_rating + w_sos)
    
    if w_rqi < 0 or w_rqi > 1:
        return float('inf')

    stats_and_metrics['NET_Score'] = (
        w_rating * stats_and_metrics['Norm_Rating'] +
        w_rqi * stats_and_metrics['Norm_RQI'] +
        w_sos * stats_and_metrics['Norm_SOS']
    )
    stats_and_metrics['NET'] = stats_and_metrics['NET_Score'].rank(ascending=False).astype(int)
    stats_and_metrics['combined_rank'] = stats_and_metrics['ELO_Rank']
    spearman_corr = stats_and_metrics[['NET', 'combined_rank']].corr(method='spearman').iloc[0,1]

    return -spearman_corr
bounds = [(0,1), (0,.05)]  
result = differential_evolution(calculate_net, bounds, strategy='best1bin', maxiter=500, tol=1e-4, seed=42)
optimized_weights = result.x
print("NET Calculation Weights:")
print("------------------------")
print(f"Rating: {optimized_weights[0]}")
print(f"RQI: {1 - (optimized_weights[0] + optimized_weights[1])}")
print(f"SOS: {optimized_weights[1]}")
adj_sos_weight = (optimized_weights[1]) / ((1 - (optimized_weights[0] + optimized_weights[1])) + (optimized_weights[1]))
adj_rqi_weight = (1 - (optimized_weights[0] + optimized_weights[1])) / ((1 - (optimized_weights[0] + optimized_weights[1])) + (optimized_weights[1]))
stats_and_metrics['Norm_Resume'] = adj_rqi_weight * stats_and_metrics['Norm_RQI'] + adj_sos_weight * stats_and_metrics['Norm_SOS']
stats_and_metrics['aRQI'] = stats_and_metrics['Norm_Resume'].rank(ascending=False).astype(int)


quadrant_records = {}

for team, group in completed_schedule.groupby('Team'):
    Q1_win, Q1_loss = 0, 0  # Initialize counters
    Q2_win, Q2_loss = 0, 0
    Q3_win, Q3_loss = 0, 0
    Q4_win, Q4_loss = 0, 0

    for _, row in group.iterrows():
        opponent = row['Opponent']
        
        if len(stats_and_metrics[stats_and_metrics['Team'] == opponent]) > 0:
            opponent_index = stats_and_metrics[stats_and_metrics['Team'] == opponent]["NET"].values[0]
        else:
            opponent_index = 300

        team_is_home = row['Team'] == row['home_team']
        team_won = (row['home_score'] > row['away_score'] and team_is_home) or \
                    (row['away_score'] > row['home_score'] and not team_is_home)

        # Apply quadrant logic
        if opponent_index <= 40:
            if team_won:
                Q1_win += 1
            else:
                Q1_loss += 1
        elif opponent_index <= 80:
            if team_won:
                Q2_win += 1
            else:
                Q2_loss += 1
        elif opponent_index <= 160:
            if team_won:
                Q3_win += 1
            else:
                Q3_loss += 1
        else:
            if team_won:
                Q4_win += 1
            else:
                Q4_loss += 1            
            

    # Store results for the team
    quadrant_records[team] = {'Team': team, 'Q1': f"{Q1_win}-{Q1_loss}", 'Q2': f"{Q2_win}-{Q2_loss}", 'Q3': f"{Q3_win}-{Q3_loss}", 'Q4': f"{Q4_win}-{Q4_loss}"}
quadrant_record_df = pd.DataFrame.from_dict(quadrant_records, orient='index').reset_index(drop=True)
stats_and_metrics = pd.merge(stats_and_metrics, quadrant_record_df, on='Team', how='left')

stats_and_metrics.fillna(0, inplace=True)
stats_and_metrics = stats_and_metrics.sort_values('Rating', ascending=False).reset_index(drop=True)
stats_and_metrics['Rating Rank'] = stats_and_metrics.index + 1
stats_and_metrics['PRR'] = stats_and_metrics['Rating Rank']
stats_and_metrics = stats_and_metrics.sort_values('NET').reset_index(drop=True)

if "Conference" not in stats_and_metrics.columns:
    stats_and_metrics = stats_and_metrics.merge(
        projected_rpi[["Team", "Conference"]], on="Team", how="left"
    )

schedule_df = schedule_df.merge(stats_and_metrics[['Team', 'NET']], left_on='home_team', right_on='Team', how='left')
schedule_df.rename(columns={'NET': 'home_net'}, inplace=True)
schedule_df = schedule_df.merge(stats_and_metrics[['Team', 'NET']], left_on='away_team', right_on='Team', how='left')
schedule_df.rename(columns={'NET': 'away_net'}, inplace=True)
schedule_df.drop(columns=['Team', 'Team_y'], inplace=True)
schedule_df.rename(columns={'Team_x':'Team'}, inplace=True)

import numpy as np # type: ignore
import pandas as pd # type: ignore

max_net = len(stats_and_metrics)
max_spread = 16.5

w_tq = 0.70   # NET AVG
w_wp = 0.20   # Win Probability
w_ned = 0.10  # NET Differential
schedule_df['avg_net'] = (schedule_df['home_net'] + schedule_df['away_net']) / 2
schedule_df['TQ'] = (max_net - schedule_df['avg_net']) / (max_net - 1)
schedule_df['WP'] = 1 - 2 * np.abs(schedule_df['home_win_prob'] - 0.5)
schedule_df['NED'] = 1 - (np.abs(schedule_df['home_net'] - schedule_df['away_net']) / (max_net - 1))
schedule_df['GQI'] = round(10 * (
    w_tq * schedule_df['TQ'] +
    w_wp * schedule_df['WP'] +
    w_ned * schedule_df['NED']
),1)

def game_sort_key(result):
    if result.startswith(("W", "L")):
        return (0, None)  # Completed games
    elif result.startswith(("Bot", "Top", "Middle", "End")):
        return (1, None)  # Ongoing games
    elif result[0].isdigit():  # Upcoming games with time
        try:
            return (2, datetime.strptime(result, "%I:%M %p"))  # Convert time to sortable format
        except ValueError:
            return (2, None)  # If parsing fails, treat as unknown
    elif result.startswith("T"):  # TBA games
        return (3, None)
    elif result.startswith("C"):  # Cancelled games
        return (4, None)
    return (5, None)  # Any other cases

def process_result(row):
    result = row["Result"]
    
    if result.startswith("W"):
        return re.sub(r"^W\s+", row["Team"] + " ", result)  # Replace "W" with Team name

    elif result.startswith("L"):
        # Extract scores and swap them
        match = re.search(r"L\s+(\d+)\s*-\s*(\d+)", result)
        if match:
            swapped_score = f"{row['Opponent']} {match.group(2)} - {match.group(1)}"
            return re.sub(r"L\s+\d+\s*-\s*\d+", swapped_score, result)  # Replace score section
    
    return result  # Leave other cases unchanged

straight_up_calculator['Result'] = straight_up_calculator['Result'].astype(str)
straight_up_calculator = straight_up_calculator.sort_values(by="Result", key=lambda x: x.map(game_sort_key))
straight_up_calculator["Result"] = straight_up_calculator["Result"].astype(str)  # Convert to string to avoid errors
straight_up_calculator["Result"] = straight_up_calculator["Result"].apply(lambda x: x if x.startswith(("W", "L")) else "")
straight_up_calculator["Result"] = straight_up_calculator.apply(process_result, axis=1)
df = straight_up_calculator[['Result', 'PEAR', 'Date']].drop_duplicates().sort_values('Date').reset_index(drop=True)
df['Result'] = df['Result'].astype(str)
df['PEAR'] = df['PEAR'].astype(str)
def extract_team_name(text):
    match = re.match(r'^([^\d-]+)', str(text))
    return match.group(1).strip() if match else text
df['team_result'] = df['Result'].apply(extract_team_name)
df['team_pear'] = df['PEAR'].apply(extract_team_name)
df['flag'] = (df['team_result'] == df['team_pear']).astype(int)
straight_up = df.groupby('Date').agg(
    Correct=('flag', 'sum'),
    Total=('flag', 'count')
).reset_index()
previous = pd.read_csv(f"./PEAR/PEAR Baseball/y{current_season}/straight_up.csv").drop(columns=['Unnamed: 0'])
previous["Date"] = pd.to_datetime(previous["Date"])
merged = straight_up.merge(previous[['Date', 'Total']], on='Date', how='left', suffixes=('', '_previous'))
missing_or_mismatched = merged[
    (merged['Total_previous'].isna()) |  
    (merged['Total'] != merged['Total_previous'])
]
missing_or_mismatched = missing_or_mismatched.drop(columns=['Total_previous'])
previous = pd.concat([previous, missing_or_mismatched]).reset_index(drop=True)
previous = previous.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
previous.to_csv(f"./PEAR/PEAR Baseball/y{current_season}/straight_up.csv")

stats_and_metrics['pNET_Score'] = (stats_and_metrics['NET_Score'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRating'] = (stats_and_metrics['Rating'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pResume_Quality'] = (stats_and_metrics['resume_quality'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pPYTHAG'] = (stats_and_metrics['PYTHAG'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['poWAR_z'] = (stats_and_metrics['oWAR_z'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pOPS'] = (stats_and_metrics['OPS'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBA'] = (stats_and_metrics['BA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRPG'] = (stats_and_metrics['RPG'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['ppWAR_z'] = (stats_and_metrics['pWAR_z'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pKP9'] = (stats_and_metrics['KP9'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pWHIP'] = ((1 - stats_and_metrics['WHIP'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pERA'] = ((1 - stats_and_metrics['ERA'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pwOBA'] = (stats_and_metrics['wOBA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pwRAA'] = (stats_and_metrics['wRAA'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pfWAR'] = (stats_and_metrics['fWAR'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pISO'] = (stats_and_metrics['ISO'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pwRC+'] = (stats_and_metrics['wRC+'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBB%'] = (stats_and_metrics['BB%'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pBABIP'] = (stats_and_metrics['BABIP'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pRA9'] = ((1 - stats_and_metrics['RA9'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pFIP'] = ((1 - stats_and_metrics['FIP'].rank(pct=True)) * 98 + 1).round().astype(int)
stats_and_metrics['pLOB%'] = (stats_and_metrics['LOB%'].rank(pct=True) * 98 + 1).round().astype(int)
stats_and_metrics['pK/BB'] = (stats_and_metrics['K/BB'].rank(pct=True) * 98 + 1).round().astype(int)

file_path = os.path.join(folder_path, f"Data/baseball_{formatted_date}.csv")
stats_and_metrics.to_csv(file_path)

file_path = os.path.join(folder_path, f"schedule_{current_season}.csv")
schedule_df.to_csv(file_path)

import datetime
central_time_zone = pytz.timezone('US/Central')
now = datetime.datetime.now(central_time_zone)

# Check if it's Monday and after 10:00 AM and before 3:00 PM
if now.hour < 13 and now.hour > 9:
    from bs4 import BeautifulSoup # type: ignore
    import pandas as pd # type: ignore
    import requests # type: ignore
    from bs4 import BeautifulSoup # type: ignore
    from PIL import Image # type: ignore
    from io import BytesIO # type: ignore
    import matplotlib.pyplot as plt # type: ignore
    import seaborn as sns # type: ignore
    import matplotlib.offsetbox as offsetbox # type: ignore
    import matplotlib.font_manager as fm # type: ignore
    from datetime import datetime, timedelta
    custom_font = fm.FontProperties(fname="./trebuc.ttf")
    plt.rcParams['font.family'] = custom_font.get_name()
    week_1_start = datetime(2025, 2, 10)
    today = datetime.today()
    days_since_start = (today - week_1_start).days
    current_week = (days_since_start // 7) + 1  # Each Monday starts a new week

    BASE_URL = "https://www.warrennolan.com"

    top_25 = stats_and_metrics[0:25]
    fig, axs = plt.subplots(5, 5, figsize=(7, 7),dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} CBASE PEAR", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "NET Ranking Incorporating Team Strength and Resume", fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/NET/net_{formatted_date}.png", bbox_inches='tight')
    print('NET Done')

    top_25 = stats_and_metrics.sort_values('RQI').reset_index(drop=True)[0:25]
    fig, axs = plt.subplots(5, 5, figsize=(7, 7),dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} CBASE Resume Quality", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "Team Performance Relative to Strength of Schedule", fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RQI/rqi_{formatted_date}.png", bbox_inches='tight')
    print('RQI Done')

    top_25 = stats_and_metrics.sort_values('PRR').reset_index(drop=True)[0:25]
    fig, axs = plt.subplots(5, 5, figsize=(7, 7),dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} CBASE Team Strength", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "Team Rating Based on Team Stats", fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/PRR/prr_{formatted_date}.png", bbox_inches='tight')
    print('PRR Done')

    top_25 = stats_and_metrics.sort_values('RPI').reset_index(drop=True)[0:25]
    fig, axs = plt.subplots(5, 5, figsize=(7, 7),dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} CBASE RPI", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "PEAR's RPI Rankings", fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/RPI/rpi_{formatted_date}.png", bbox_inches='tight')
    print('RPI Done')

    major_conferences = ['SEC', 'ACC', 'Independent', 'Big 12', 'Big Ten']
    top_25 = stats_and_metrics[~stats_and_metrics['Conference'].isin(major_conferences)].reset_index(drop=True)[0:25]
    fig, axs = plt.subplots(5, 5, figsize=(7, 7),dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')
    plt.suptitle(f"Week {current_week} Mid-Major CBASE PEAR", fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, "NET Ranking Incorporating Team Strength and Resume", fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Mid_Major/mid_major_{formatted_date}.png", bbox_inches='tight')
    print('Mid Major Done')

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-8:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    extended_bubble = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(50, "NET").iloc[42:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers, next_8_teams])
    all_at_large_teams = pd.concat([at_large, next_8_teams, extended_bubble]).sort_values(by='NET').reset_index(drop=True)
    tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    last_team_in = last_four_in.loc[len(last_four_in)-1, 'Team']
    last_team_in_index = all_at_large_teams[all_at_large_teams['Team'] == last_team_in].index.values[0]
    sorted_aqs = automatic_qualifiers.sort_values('NET').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 12),dpi=125)
    plt.gca().set_facecolor('#CECEB2')
    plt.gcf().set_facecolor('#CECEB2')
    logo_size = 5

    for i in range(len(all_at_large_teams)):
        team = all_at_large_teams.loc[i, 'Team']
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        img_src = img_tag.get("src")
        image_url = BASE_URL + img_src

        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))

        x = all_at_large_teams['PRR'].iloc[i]
        y = all_at_large_teams['aRQI'].iloc[i]
        above = last_team_in_index - all_at_large_teams.index[i]

        ax.imshow(img, aspect='auto', 
                extent=(x - (logo_size - 1.2), x + (logo_size - 1.2), 
                        y - logo_size, y + logo_size))
        
        if team in last_four_in['Team'].values:
            circle = plt.Circle((x, y), logo_size, color='#2ECC71', fill=True, alpha=0.3, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')

        if team in next_8_teams['Team'].values:
            circle = plt.Circle((x, y), logo_size, color='#F39C12', fill=True, alpha=0.3, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')

        if team in extended_bubble['Team'].values:
            circle = plt.Circle((x, y), logo_size, color='#E74C3C', fill=True, alpha=0.3, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y + logo_size, above, fontsize=14, fontweight='bold', ha='center')


    if all_at_large_teams['PRR'].max() > all_at_large_teams['aRQI'].max():
        max_range = all_at_large_teams['PRR'].max()
    else:
        max_range = all_at_large_teams['aRQI'].max()

    height = max_range + 4
    plt.text(max_range+24, height + 3, "Automatic Qualifiers", ha='center', fontweight='bold', fontsize = 18)
    for i in range(len(sorted_aqs)):
        team = sorted_aqs.loc[i, 'Team']
        conference = sorted_aqs.loc[i, 'Conference']
        plt.text(max_range+24, height, f"{team}", ha='center', fontsize=16)
        height = height - 3

    ax.set_xlabel('Team Strength Rank', fontsize = 16)
    ax.set_ylabel('Adjusted Resume Rank', fontsize = 16)
    # ax.set_title('At Large Ratings vs. Adjusted Resume', fontweight='bold', fontsize=14)
    plt.text(0, max_range + 20, "At Large Team Strength vs. Adjusted Resume", ha='left', fontsize = 32, fontweight = 'bold')
    plt.text(0, max_range + 16, f"Automatic Qualifiers Removed, Bubble Teams Highlighted - Through {(today - timedelta(days=1)).strftime('%m/%d')}", fontsize = 24)
    plt.text(0, max_range + 12, "@PEARatings", ha='left', fontsize=24, fontweight='bold')
    plt.text(max_range + 8, max_range + 7, f"Projected At Large Bids ONLY (Based on PEAR NET Rankings)", ha='right', fontsize=16)
    plt.text(max_range + 8, max_range + 4, f"Green - Last 8 In, Orange - First 8 Out, Red - Next 8 Out", ha='right', fontsize=16)
    plt.text(max_range + 8, max_range + 1, f"Value indicates distance from being the last team in", ha='right', fontsize=16)
    plt.xlim(-2, max_range + 10)
    plt.ylim(-2, max_range + 10)
    plt.grid(False)
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Ratings_vs_Resume/ratings_vs_resume_{formatted_date}.png", bbox_inches='tight')
    print('Ratings vs Resume Done')

    bubble = pd.concat([last_four_in, next_8_teams, extended_bubble]).sort_values('NET').reset_index(drop=True)[['Team', 'NET', 'NET_Score']]
    last_net_score = bubble[bubble['Team'] == last_team_in]['NET_Score'].values[0]
    bubble["percentage_away"] = round((
        (bubble["NET_Score"] - last_net_score) / last_net_score
    ) * 100, 2)

    def get_team_logo(team):
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        
        if img_tag:
            img_src = img_tag.get("src")
            image_url = BASE_URL + img_src
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content))
            return img
        return None

    team_logos = {team: get_team_logo(team) for team in bubble["Team"]}
    bubble = bubble.sort_values(by="percentage_away", ascending=False)
    max_abs_value = bubble["percentage_away"].abs().max() + 0.2
    colors = ["#2ECC71" if x >= 0 else "#E74C3C" for x in bubble["percentage_away"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor("#CECEB2")
    ax.set_facecolor("#CECEB2")
    sns.barplot(data=bubble, x="percentage_away", y="Team", ax=ax, width=0.2, palette=colors, hue='Team', legend=False)

    ax.set_xlim(-max_abs_value, max_abs_value)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add team logos
    for i, (team, percentage) in enumerate(zip(bubble["Team"], bubble["percentage_away"])):
        if team in team_logos and team_logos[team]:
            img = team_logos[team]
            xybox = (10, 0) if percentage >= 0 else (-10, 0)
            imagebox = offsetbox.OffsetImage(img, zoom=0.3)
            ab = offsetbox.AnnotationBbox(imagebox, (percentage, i), 
                                        xybox=xybox,  
                                        boxcoords="offset points", 
                                        frameon=False)
            ax.add_artist(ab)

    ax.set_xlabel("")
    # ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_title("NET Score Distance from Last Team In", fontsize = 16)
    plt.text(-max_abs_value, -2, f"NET Score Distance From Last Team In - Through {(today - timedelta(days=1)).strftime('%m-%d')}", fontsize=20, ha='left', fontweight='bold')
    plt.text(-max_abs_value, -1, "@PEARatings", ha='left', fontsize = 12)

    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Team_In/last_team_in_{formatted_date}.png", bbox_inches='tight')
    print('Last Team In Done')

    hosting = stats_and_metrics[0:24][['Team', 'NET', 'NET_Score']]
    last_host_score = hosting[hosting['NET'] == 16]['NET_Score'].values[0]
    hosting["percentage_away"] = round((
        (hosting["NET_Score"] - last_host_score) / last_host_score
    ) * 100, 2)

    def get_team_logo(team):
        team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
        response = requests.get(team_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find("img", class_="team-menu__image")
        
        if img_tag:
            img_src = img_tag.get("src")
            image_url = BASE_URL + img_src
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content))
            return img
        return None

    team_logos = {team: get_team_logo(team) for team in hosting["Team"]}
    hosting = hosting.sort_values(by="percentage_away", ascending=False)
    max_abs_value = hosting["percentage_away"].abs().max() + 0.2
    colors = ["#2ECC71" if x >= 0 else "#E74C3C" for x in hosting["percentage_away"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor("#CECEB2")
    ax.set_facecolor("#CECEB2")
    sns.barplot(data=hosting, x="percentage_away", y="Team", ax=ax, width=0.2, palette=colors, hue='Team', legend=False)

    ax.set_xlim(-max_abs_value, max_abs_value)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add team logos
    for i, (team, percentage) in enumerate(zip(hosting["Team"], hosting["percentage_away"])):
        if team in team_logos and team_logos[team]:
            img = team_logos[team]
            xybox = (10, 0) if percentage >= 0 else (-10, 0)
            imagebox = offsetbox.OffsetImage(img, zoom=0.3)
            ab = offsetbox.AnnotationBbox(imagebox, (percentage, i), 
                                        xybox=xybox,  
                                        boxcoords="offset points", 
                                        frameon=False)
            ax.add_artist(ab)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.text(-max_abs_value, -2, f"NET Score Distance From Last Host Seed - Through {(today - timedelta(days=1)).strftime('%m-%d')}", fontsize=20, ha='left', fontweight='bold')
    plt.text(-max_abs_value, -1, "@PEARatings", ha='left', fontsize = 12)
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Last_Host/last_host_{formatted_date}.png", bbox_inches='tight')
    print('Last Host Seed Done')

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers])
    tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")
    formatted_df.index = formatted_df.index + 1
    from plottable import Table # type: ignore
    from plottable.plots import image, circled_image # type: ignore
    from plottable import ColumnDefinition # type: ignore
    # Create a set of automatic qualifier teams for faster lookup
    automatic_teams = set(automatic_qualifiers["Team"])

    # Modify the DataFrame by appending '*' to teams in automatic qualifiers
    formatted_df_with_asterisk = formatted_df.copy()
    for col in formatted_df_with_asterisk.columns[0:]:  # Exclude index column
        formatted_df_with_asterisk[col] = formatted_df_with_asterisk[col].apply(lambda x: f"{x}*" if x in automatic_teams else x)
    formatted_df_with_asterisk['Host'] = formatted_df_with_asterisk.index.to_series().apply(lambda i: f"#{i} {formatted_df_with_asterisk.loc[i, '1 Seed']}")
    make_table = formatted_df_with_asterisk[['Host', '2 Seed', '3 Seed', '4 Seed']].set_index('Host')

    # Init a figure 
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#CECEB2')

    column_definitions = [
        ColumnDefinition(name='Host', # name of the column to change
                        title='Host', # new title for the column
                        textprops={"ha": "center", "weight": "bold", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='2 Seed', # name of the column to change
                        title='2 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='3 Seed', # name of the column to change
                        title='3 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        ),
        ColumnDefinition(name='4 Seed', # name of the column to change
                        title='4 Seed', # new title for the column
                        textprops={"ha": "center", "fontsize": 16}, # properties to apply
                        )
    ]

    # Create the table object with the column definitions parameter
    tab = Table(make_table, column_definitions=column_definitions, footer_divider=True, row_divider_kw={"linewidth": 1})


    # Change the color
    last_four_in_teams = set(last_four_in["Team"])
    for col in make_table.columns:
        tab.columns[col].set_facecolor("#CECEB2")

    tab.col_label_row.set_facecolor('#CECEB2')
    tab.columns["Host"].set_facecolor('#CECEB2')
    # plt.figtext(0.89, 0.09, "* Indicates an automatic qualifier", ha="right", fontsize=14, fontstyle='italic')
    plt.figtext(0.13, 0.975, f"PEAR {today.strftime('%m/%d')} Tournament Outlook", ha='left', fontsize=32, fontweight='bold')
    plt.figtext(0.13, 0.945, f"Regionals If Season Ended Today Based on PEAR NET Ranking", ha='left', fontsize=16)
    plt.figtext(0.13, 0.915, f"No Considerations For Conference or Regional Proximity - Through {(today - timedelta(days=1)).strftime('%m/%d')}", ha='left', fontsize=16)
    plt.figtext(0.13, 0.885, "@PEARatings", ha='left', fontsize=16, fontweight='bold')
    plt.figtext(0.13, 0.09, f"Last Four In - {last_four_in.loc[0, 'Team']}, {last_four_in.loc[1, 'Team']}, {last_four_in.loc[2, 'Team']}, {last_four_in.loc[3, 'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.06, f"First Four Out - {next_8_teams.loc[0,'Team']}, {next_8_teams.loc[1,'Team']}, {next_8_teams.loc[2,'Team']}, {next_8_teams.loc[3,'Team']}", ha='left', fontsize=14)
    plt.figtext(0.13, 0.03, f"Next Four Out - {next_8_teams.loc[4,'Team']}, {next_8_teams.loc[5,'Team']}, {next_8_teams.loc[6,'Team']}, {next_8_teams.loc[7,'Team']}", ha='left', fontsize=14)
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Tournament/tournament_{formatted_date}.png", bbox_inches='tight')
    print('Tournament Done')

    from matplotlib.ticker import MaxNLocator
    def net_tracker(X, Y):
        folder_path = f"./PEAR/PEAR Baseball/y{current_season}/Data"
        csv_files = [f for f in os.listdir(folder_path) if f.startswith("baseball_") and f.endswith(".csv")]
        def extract_date(filename):
            try:
                return datetime.strptime(filename.replace("baseball_", "").replace(".csv", ""), "%m_%d_%Y")
            except ValueError:
                return None
            
        date_files = {extract_date(f): f for f in csv_files if extract_date(f) is not None}
        sorted_date_files = sorted(date_files.items(), key=lambda x: x[0], reverse=True)
        latest_files = sorted_date_files[:X]
        stats_and_metrics_list = []
        for _, file_name in latest_files:
            file_path = os.path.join(folder_path, file_name)
            date_of_file = extract_date(file_name).strftime('%Y-%m-%d')  # Get date in 'YYYY-MM-DD' format
            
            # Read the CSV and add the "Date" column
            df = pd.read_csv(file_path)
            df['Date'] = date_of_file  # Add a column with the date of the file
            stats_and_metrics_list.append(df)
        stats_and_metrics_combined = pd.concat(stats_and_metrics_list, ignore_index=True)
        most_recent_csv = stats_and_metrics_combined[stats_and_metrics_combined['Date'] == stats_and_metrics_combined['Date'].max()]
        top_teams = most_recent_csv.nsmallest(Y, 'NET')['Team'].tolist()
        filtered_data = stats_and_metrics_combined[stats_and_metrics_combined['Team'].isin(top_teams)]
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date']).dt.strftime('%m-%d')
        earliest = filtered_data['Date'].min()
        latest = filtered_data['Date'].max()
        pivoted_table = filtered_data.pivot_table(index='Team', columns='Date', values='NET', aggfunc='first')
        pivoted_table = pivoted_table[sorted(pivoted_table.columns)]
        pivoted_table = pivoted_table.reset_index()
        pivoted_table = pivoted_table.sort_values(by=pivoted_table.columns[-1], ascending=True)
        import math
        num_rows = math.ceil(math.sqrt(Y))
        num_cols = math.ceil(math.sqrt(Y))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig.patch.set_facecolor('#CECEB2')
        axes = axes.flatten()
        plt.text(0, 1.5, f"Top {Y} in PEAR NET Rankings", fontsize = 32, fontweight='bold', ha='left', transform=axes[0].transAxes)
        plt.text(0, 1.34, f"Past {X} Days - {earliest} to {latest}", fontsize = 16, ha='left', transform=axes[0].transAxes)
        plt.text(0, 1.19, f"@PEARatings", fontsize = 16, ha='left', fontweight='bold', transform=axes[0].transAxes)
        min_net = 0
        max_net = pivoted_table.drop(columns='Team').max().max() + 1

        for i, (team, ax) in enumerate(zip(pivoted_table['Team'], axes)):
            team_data = pivoted_table[pivoted_table['Team'] == team].drop(columns='Team').T  # Drop 'Team' column and transpose
            team_data.columns = [team]  # Set the team name as the column header
            last_value = team_data.iloc[0, 0]
            first_value = team_data.iloc[-1, 0]
            
            # Determine the color of the line based on trend
            line_color = "#2ca02c" if last_value > first_value else "#d62728"

            ax.plot(team_data.index, team_data[team], marker='o', color=line_color)
            ax.text(team_data.index[0], last_value - 1.2, f"{int(last_value)}", 
                fontsize=10, fontweight='bold', ha='center', color='black')
            ax.text(team_data.index[-1], first_value - 1.2, f"{int(first_value)}", 
                fontsize=10, fontweight='bold', ha='center', color='black')
            ax.set_title(f"#{i+1} {team}", fontweight='bold', fontsize=16)
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.set_facecolor('#CECEB2')
            ax.set_xticks("")
            ax.set_ylim(min_net, max_net)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Over_Time/over_time_{formatted_date}.png", bbox_inches='tight')
    net_tracker(14,16)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import random
    from collections import defaultdict

    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 7.5)), 4)  # More precision, rounded later in output

    def simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics):
        teams = [team_a, team_b, team_c, team_d]
        ratings = {team: stats_and_metrics.loc[stats_and_metrics["Team"] == team, "Rating"].iloc[0] for team in teams}

        game1 = PEAR_Win_Prob(ratings[team_a], ratings[team_d]) 
        w1, l1 = (team_a, team_d) if random.random() < game1 else (team_d, team_a)

        game2 = PEAR_Win_Prob(ratings[team_b], ratings[team_c])
        w2, l2 = (team_b, team_c) if random.random() < game2 else (team_c, team_b)

        game3 = PEAR_Win_Prob(ratings[l2], ratings[l1])
        w3 = l2 if random.random() < game3 else l1

        game4 = PEAR_Win_Prob(ratings[w1], ratings[w2])
        w4, l4 = (w1, w2) if random.random() < game4 else (w2, w1)

        game5 = PEAR_Win_Prob(ratings[l4], ratings[w3])
        w5 = l4 if random.random() < game5 else w3

        game6 = PEAR_Win_Prob(ratings[w4], ratings[w5])
        w6 = w4 if random.random() < game6 else w5

        if w6 == w4:
            regional_winner = w6
        else:
            regional_winner = w4 if random.random() < PEAR_Win_Prob(ratings[w4], ratings[w5]) else w5

        return regional_winner

    def run_simulation(team_a, team_b, team_c, team_d, stats_and_metrics, num_simulations=1000):
        results = defaultdict(int)

        for _ in range(num_simulations):
            winner = simulate_tournament(team_a, team_b, team_c, team_d, stats_and_metrics)
            results[winner] += 1

        # Sort and format results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Return a defaultdict to maintain the same structure
        formatted_results = defaultdict(float, {team: round(wins / num_simulations, 3) for team, wins in sorted_results})
        
        return formatted_results

    def select_super_regional_teams(regional_results):
        """Selects super regional teams probabilistically based on regional results."""
        super_regional_teams = []
        matchups = [(0, 15), (1, 14), (2, 13), (3, 12), 
                    (4, 11), (5, 10), (6, 9), (7, 8)]
        
        for idx1, idx2 in matchups:
            team1 = random.choices(list(regional_results[idx1].keys()), 
                                weights=list(regional_results[idx1].values()), k=1)[0]
            team2 = random.choices(list(regional_results[idx2].keys()), 
                                weights=list(regional_results[idx2].values()), k=1)[0]
            super_regional_teams.append((team1, team2, idx1))  # Keep track of region index

        return super_regional_teams

    def simulate_super_regional(team1, team2, stats_and_metrics):
        """Simulates a best-of-three series for the super regional."""
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]

        wins1, wins2 = 0, 0
        while wins1 < 2 and wins2 < 2:
            win_prob = PEAR_Win_Prob(rating1, rating2)
            if random.random() < win_prob:
                wins1 += 1
            else:
                wins2 += 1

        return team1 if wins1 == 2 else team2

    def run_super_regionals(regional_results, stats_and_metrics, num_simulations=1000):
        """Runs multiple simulations of super regionals incorporating regional probabilities."""
        results = [defaultdict(int) for _ in range(8)]  # Maintain separate regions

        for _ in range(num_simulations):
            super_regional_matchups = select_super_regional_teams(regional_results)
            
            for team1, team2, region_index in super_regional_matchups:
                winner = simulate_super_regional(team1, team2, stats_and_metrics)
                results[region_index][winner] += 1  # Track wins in the correct region

        # Normalize probabilities for each region
        for i in range(8):
            total_sims = sum(results[i].values())
            for team in results[i]:
                results[i][team] = round(results[i][team] / total_sims, 3)

        return results

    def select_team(region):
        """Selects a team probabilistically based on their make_omaha probabilities."""
        teams, weights = zip(*region.items())
        return random.choices(teams, weights=weights, k=1)[0]

    def simulate_game(team1, team2, stats_and_metrics):
        """Simulates a game using PEAR_Win_Prob."""
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]
        win_prob = PEAR_Win_Prob(rating1, rating2)
        return team1 if random.random() < win_prob else team2

    def simulate_double_elimination(teams, stats_and_metrics):
        """Runs a double-elimination tournament for 4 teams."""
        winners_bracket = [teams[0], teams[1], teams[2], teams[3]]
        losers_bracket = []

        # Round 1
        game1_winner = simulate_game(winners_bracket[0], winners_bracket[1], stats_and_metrics)
        game1_loser = winners_bracket[1] if game1_winner == winners_bracket[0] else winners_bracket[0]

        game2_winner = simulate_game(winners_bracket[2], winners_bracket[3], stats_and_metrics)
        game2_loser = winners_bracket[3] if game2_winner == winners_bracket[2] else winners_bracket[2]

        losers_bracket.extend([game1_loser, game2_loser])
        
        # Round 2 (Winners Final)
        winners_final_winner = simulate_game(game1_winner, game2_winner, stats_and_metrics)
        winners_final_loser = game1_winner if winners_final_winner == game2_winner else game2_winner

        # Losers Bracket
        losers_round1_winner = simulate_game(losers_bracket[0], losers_bracket[1], stats_and_metrics)

        # Losers Final
        losers_final_winner = simulate_game(losers_round1_winner, winners_final_loser, stats_and_metrics)

        # Championship (Double elimination rule: Losers bracket winner must win twice)
        if simulate_game(winners_final_winner, losers_final_winner, stats_and_metrics) == winners_final_winner:
            return winners_final_winner
        return simulate_game(winners_final_winner, losers_final_winner, stats_and_metrics)

    def run_college_world_series(make_omaha, stats_and_metrics, num_simulations=1000):
        """Runs two separate double-elimination tournaments and returns results in the same format as input."""
        results = [defaultdict(int), defaultdict(int)]

        for _ in range(num_simulations):
            # Tournament 1
            team1 = select_team(make_omaha[0])
            team2 = select_team(make_omaha[7])
            team3 = select_team(make_omaha[3])
            team4 = select_team(make_omaha[4])
            winner1 = simulate_double_elimination([team1, team2, team3, team4], stats_and_metrics)
            results[0][winner1] += 1

            # Tournament 2
            team5 = select_team(make_omaha[1])
            team6 = select_team(make_omaha[6])
            team7 = select_team(make_omaha[2])
            team8 = select_team(make_omaha[5])
            winner2 = simulate_double_elimination([team5, team6, team7, team8], stats_and_metrics)
            results[1][winner2] += 1

        # Normalize results to probabilities
        for i in range(2):
            total_sims = sum(results[i].values())
            for team in results[i]:
                results[i][team] = round(results[i][team] / total_sims, 3)

        return results

    def simulate_finals(make_finals, stats_and_metrics):
        """Simulates the College World Series Finals as a best-of-three series."""
        # Select a team from each defaultdict
        team1 = random.choices(list(make_finals[0].keys()), weights=list(make_finals[0].values()), k=1)[0]
        team2 = random.choices(list(make_finals[1].keys()), weights=list(make_finals[1].values()), k=1)[0]

        # Get team ratings
        rating1 = stats_and_metrics.loc[stats_and_metrics["Team"] == team1, "Rating"].values[0]
        rating2 = stats_and_metrics.loc[stats_and_metrics["Team"] == team2, "Rating"].values[0]

        # Simulate best-of-three series
        wins1, wins2 = 0, 0
        while wins1 < 2 and wins2 < 2:
            win_prob = PEAR_Win_Prob(rating1, rating2)
            if random.random() < win_prob:
                wins1 += 1
            else:
                wins2 += 1

        # Return the champion
        return team1 if wins1 == 2 else team2

    def run_finals_simulation(make_finals, stats_and_metrics, num_simulations=1000):
        """Runs multiple simulations of the College World Series Finals and returns championship probabilities."""
        results = defaultdict(int)

        for _ in range(num_simulations):
            champion = simulate_finals(make_finals, stats_and_metrics)
            results[champion] += 1

        # Convert to probabilities
        total_sims = sum(results.values())
        results_dict = {team: round(wins / total_sims, 3) for team, wins in results.items()}
        
        return results_dict

    def generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals):
        teams = set()  # To store all unique teams from all simulation results

        # Collect teams from regional results (Super Regional probabilities)
        for regional in regionals_results:
            teams.update(regional.keys())

        # Collect teams from Make Omaha (Omaha probabilities)
        for omaha in make_omaha:
            teams.update(omaha.keys())

        # Collect teams from Make Finals (Finals probabilities)
        for finals in make_finals:
            teams.update(finals.keys())

        # Collect teams from Win Finals (Win NC probabilities)
        teams.update(win_finals.keys())

        # Initialize a dictionary to hold the probabilities for each team
        data = {team: {"Supers": 0, "Omaha": 0, "Finals": 0, "Win NC": 0} for team in teams}

        # Assign the probabilities directly from the simulation results
        for regional in regionals_results:
            for team, prob in regional.items():
                data[team]["Supers"] = prob  # Assign the probability for reaching Super Regionals

        for omaha in make_omaha:
            for team, prob in omaha.items():
                data[team]["Omaha"] = prob  # Assign the probability for reaching Omaha

        for finals in make_finals:
            for team, prob in finals.items():
                data[team]["Finals"] = prob  # Assign the probability for reaching Finals

        for team, prob in win_finals.items():
            data[team]["Win NC"] = prob  # Assign the probability for winning the National Championship

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')

        return df

    def simulate_full_tournament(formatted_df, stats_and_metrics, iter):
        regionals_results = []
        for i in range(len(formatted_df)):
            regionals_results.append(run_simulation(formatted_df.iloc[i,1], formatted_df.iloc[i,2], formatted_df.iloc[i,3], formatted_df.iloc[i,4], stats_and_metrics, iter))
        make_omaha = run_super_regionals(regionals_results, stats_and_metrics, iter)
        make_finals = run_college_world_series(make_omaha, stats_and_metrics, iter)
        win_finals = run_finals_simulation(make_finals, stats_and_metrics, iter)
        simulation_df = generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals)
        simulation_df = simulation_df.sort_values('Win NC', ascending=False).reset_index()
        simulation_df.rename(columns={'index': 'Team'}, inplace=True)
        return simulation_df

    automatic_qualifiers = stats_and_metrics.loc[stats_and_metrics.groupby("Conference")["NET"].idxmin()]
    at_large = stats_and_metrics.drop(automatic_qualifiers.index)
    at_large = at_large.nsmallest(34, "NET")
    last_four_in = at_large[-4:].reset_index()
    next_8_teams = stats_and_metrics.drop(automatic_qualifiers.index).nsmallest(42, "NET").iloc[34:].reset_index(drop=True)
    tournament = pd.concat([at_large, automatic_qualifiers])
    tournament = tournament.sort_values(by="NET").reset_index(drop=True)
    tournament["Seed"] = (tournament.index // 16) + 1
    pod_order = list(range(1, 17)) + list(range(16, 0, -1)) + list(range(1, 17)) + list(range(16, 0, -1))
    tournament["Host"] = pod_order
    formatted_df = tournament.pivot_table(index="Host", columns="Seed", values="Team", aggfunc=lambda x: ' '.join(x))
    formatted_df.columns = [f"{col} Seed" for col in formatted_df.columns]
    formatted_df = formatted_df.reset_index()
    formatted_df['Host'] = formatted_df['1 Seed'].apply(lambda x: f"{x}")

    tournament_sim = simulate_full_tournament(formatted_df, stats_and_metrics, 5000)
    top_25_teams = tournament_sim[0:25]
    top_25_teams.iloc[:, 1:] = top_25_teams.iloc[:, 1:] * 100

    # Normalize values for color gradient (excluding 0 values)
    min_value = top_25_teams.iloc[:, 1:].replace(0, np.nan).min().min()  # Min of all probabilities
    max_value = top_25_teams.iloc[:, 1:].max().max()  # Max of all probabilities

    def normalize(value, min_val, max_val):
        """ Normalize values between 0 and 1 for colormap. """
        if pd.isna(value) or value == 0:
            return 0  # Keep 0 values at the lowest color
        return (value - min_val) / (max_val - min_val)

    # Define custom colormap (lighter green to dark green)
    cmap = LinearSegmentedColormap.from_list('custom_green', ['#d5f5e3', '#006400'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=125)
    fig.patch.set_facecolor('#CECEB2')

    ax.axis('tight')
    ax.axis('off')

    # Add the table
    table = ax.table(
        cellText=top_25_teams.values,
        colLabels=top_25_teams.columns,
        cellLoc='center',
        loc='center',
        colColours=['#CECEB2'] * len(top_25_teams.columns)  # Set the header background color
    )
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)
        if i == 0:
            cell.set_facecolor('#CECEB2')  
            cell.set_text_props(fontsize=14, weight='bold', color='black')
        elif j == 0:
            cell.set_facecolor('#CECEB2')
            cell.set_text_props(fontsize=14, weight='bold', color='black')

        else:
            value = top_25_teams.iloc[i-1, j]  # Skip header row
            normalized_value = normalize(value, min_value, max_value)
            color = cmap(normalized_value)
            cell.set_facecolor(color)
            cell.set_text_props(fontsize=14, weight='bold', color='black')
            if value == 0:
                cell.get_text().set_text("<1%")
            else:
                cell.get_text().set_text(f"{value:.1f}%")

        cell.set_height(0.05)

    # Show the plot
    plt.text(0, 0.086, 'Odds to Win Championship', fontsize=24, fontweight='bold', ha='center')
    plt.text(0, 0.08, f"Based on Tournament Outlook {today.strftime('%m/%d')}", fontsize=16, ha='center')
    plt.text(0, 0.074, f"@PEARatings", fontsize=16, fontweight='bold', ha='center')
    plt.savefig(f"./PEAR/PEAR Baseball/y{current_season}/Visuals/Tournament_Odds/tournament_odds_{formatted_date}.png", bbox_inches='tight')
