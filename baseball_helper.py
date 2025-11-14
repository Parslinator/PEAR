import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import warnings
import os
import textwrap
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import matplotlib.offsetbox as offsetbox # type: ignore
import matplotlib.font_manager as fm # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from collections import Counter, defaultdict
from plottable import Table # type: ignore
from plottable.plots import image, circled_image # type: ignore
from plottable import ColumnDefinition # type: ignore
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import random

BASE_URL = "https://www.warrennolan.com"
cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year

def extract_schedule_data(team_name, team_url, session):
    schedule_url = BASE_URL + team_url
    team_schedule = []

    try:
        response = session.get(schedule_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[Error] {team_name} → {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    schedule_lists = soup.find_all("ul", class_="team-schedule")
    if not schedule_lists:
        return []

    schedule_list = schedule_lists[0]

    for game in schedule_list.find_all('li', class_='team-schedule'):
        try:
            # Date
            month = game.find('span', class_='team-schedule__game-date--month')
            day = game.find('span', class_='team-schedule__game-date--day')
            dow = game.find('span', class_='team-schedule__game-date--dow')
            game_date = f"{month.get_text(strip=True)} {day.get_text(strip=True)} ({dow.get_text(strip=True)})"

            # Opponent
            opponent_link = game.select_one('.team-schedule__opp-line-link')
            opponent_name = opponent_link.get_text(strip=True) if opponent_link else ""

            # Location
            location_div = game.find('div', class_='team-schedule__location')
            location_text = location_div.get_text(strip=True) if location_div else ""
            if "VS" in location_text:
                game_location = "Neutral"
            elif "AT" in location_text:
                game_location = "Away"
            else:
                game_location = "Home"

            # Result
            result_info = game.find('div', class_='team-schedule__result')
            result_text = result_info.get_text(strip=True) if result_info else "N/A"

            # Box score
            box_score_table = game.find('table', class_='team-schedule-bottom__box-score')
            home_team = away_team = home_score = away_score = "N/A"

            if box_score_table:
                rows = box_score_table.find_all('tr')
                if len(rows) > 2:
                    away_row = rows[1].find_all('td')
                    home_row = rows[2].find_all('td')
                    away_team = away_row[0].get_text(strip=True)
                    home_team = home_row[0].get_text(strip=True)
                    away_score = away_row[-3].get_text(strip=True)
                    home_score = home_row[-3].get_text(strip=True)

            team_schedule.append([
                team_name, game_date, opponent_name, game_location,
                result_text, home_team, away_team, home_score, away_score
            ])
        except Exception as e:
            print(f"[Parse Error] {team_name} game row → {e}")
            continue

    return team_schedule

# ThreadPool wrapper function
def fetch_all_schedules(elo_df, session, max_workers=12):
    schedule_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_schedule_data, row["Team"], row["Team Link"], session): row["Team"]
            for _, row in elo_df.iterrows()
        }

        for future in as_completed(futures):
            try:
                data = future.result()
                schedule_data.extend(data)
            except Exception as e:
                print(f"[Thread Error] {e}")

    return schedule_data

# Mapping months to numerical values
month_mapping = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

# Function to convert "FEB 14 (FRI)" format to "mm-dd-yyyy"
def convert_date(date_str):
    # Ensure date is a string before splitting
    if isinstance(date_str, pd.Timestamp):
        date_str = date_str.strftime("%b %d (%a)").upper()  # Convert to same format
    
    parts = date_str.split()  # ["FEB", "14", "(FRI)"]
    month = month_mapping[parts[0].upper()]  # Convert month to number
    day = parts[1]  # Extract day
    return f"{month}-{day}-{current_season}"

# --- PEAR Win Probability ---
def PEAR_Win_Prob(home_pr, away_pr, location="Neutral"):
    if location != "Neutral":
        home_pr += 0.3
    rating_diff = home_pr - away_pr
    return round(1 / (1 + 10 ** (-rating_diff / 6)) * 100, 2)

# --- Helper Functions ---
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def scrape_warrennolan_table(url, expected_columns):
    soup = get_soup(url)
    table = soup.find('table', class_='normal-grid alternating-rows stats-table')
    data = []
    if table:
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                name_div = cells[1].find('div', class_='name-subcontainer')
                full_text = name_div.text.strip() if name_div else cells[1].text.strip()
                parts = full_text.split("\n")
                team_name = parts[0].strip()
                conference = parts[1].split("(")[0].strip() if len(parts) > 1 else ""
                data.append([cells[0].text.strip(), team_name, conference])
    return pd.DataFrame(data, columns=expected_columns)

def clean_team_names(df, team_replacements, column='Team'):
    df[column] = df[column].str.replace('State', 'St.', regex=False)
    df[column] = df[column].replace(team_replacements)
    return df

####################### CONFIG #######################

# Must be defined elsewhere in your script:
# - stat_links: dict of stat_name -> URL
# - get_soup(url): function that returns BeautifulSoup of given URL

####################### Core Stat Fetching #######################

def get_stat_dataframe(stat_name, stat_links):
    if stat_name not in stat_links:
        print(f"Stat '{stat_name}' not found. Available stats: {list(stat_links.keys())}")
        return None

    all_data = []
    page_num = 1

    while page_num < 7:
        url = stat_links[stat_name]
        if page_num > 1:
            url = f"{url}/p{page_num}"

        try:
            soup = get_soup(url)
            table = soup.find("table")
            if not table:
                break

            headers = [th.text.strip() for th in table.find_all("th")]
            data = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                data.append([col.text.strip() for col in cols])

            all_data.extend(data)

        except requests.exceptions.HTTPError:
            break
        except Exception as e:
            print(f"Error for {stat_name}, page {page_num}: {e}")
            break

        page_num += 1

    if all_data:
        df = pd.DataFrame(all_data, columns=headers)
        for col in df.columns:
            if col != "Team":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return None

####################### Threading #######################

def threaded_stat_fetch(stat_names, stat_links, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stat = {
            executor.submit(get_stat_dataframe, stat, stat_links): stat
            for stat in stat_names
        }
        results = {}
        for future in as_completed(future_to_stat):
            stat = future_to_stat[future]
            try:
                results[stat] = future.result()
            except Exception as e:
                print(f"Failed to fetch {stat}: {e}")
                results[stat] = None  # Now it's inside the except block
    return results

####################### Utility #######################

def clean_duplicates(df, group_col, min_col):
    duplicates = df[df.duplicated(group_col, keep=False)]
    filtered = duplicates.loc[duplicates.groupby(group_col)[min_col].idxmin()]
    cleaned = df[~df[group_col].isin(duplicates[group_col])]
    return pd.concat([cleaned, filtered], ignore_index=True)

####################### Merging + Final Stats #######################

def clean_and_merge(stats_raw, transforms_dict):
    dfs = []
    for stat, df in stats_raw.items():
        if df is not None and stat in transforms_dict:
            df["Team"] = df["Team"].str.strip()
            df = df.dropna(subset=["Team"])
            df_clean = transforms_dict[stat](df)
            dfs.append(df_clean)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Team", how="inner")

    merged = merged.loc[:, ~merged.columns.duplicated()].sort_values('Team').reset_index(drop=True)
    merged["OPS"] = merged["SLG"] + merged["OBP"]
    merged["PYTHAG"] = round(
        (merged["RS"] ** 1.83) / ((merged["RS"] ** 1.83) + (merged["RA"] ** 1.83)), 3
    )
    return merged

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
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

class BaseballPowerRatingSystem:
    """
    Baseball power rating system that optimizes for multiple targets including ELO_Rank
    """
    
    def __init__(self, use_xgb=HAS_XGB, home_field_advantage=0.8, random_state=42):
        self.use_xgb = use_xgb
        self.hfa = home_field_advantage
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.diagnostics = {}
        
    def _clean_round(self, values, decimals=2):
        """Helper method to ensure clean rounding without floating point artifacts"""
        return np.round(values.astype(float), decimals)
    
    def _fit_regressor(self, X, y, model_name=None):
        """Fit regressor with cross-validation"""
        if self.use_xgb:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=150,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
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
        actual_margin = schedule_clean['home_score'].values - schedule_clean['away_score'].values
        actual_total = schedule_clean['home_score'].values + schedule_clean['away_score'].values
        
        # Check if neutral field column exists
        if 'neutral' in schedule_clean.columns:
            hfa = np.where(schedule_clean['neutral'] == False, self.hfa, 0.0)
        else:
            hfa = np.full(len(schedule_clean), self.hfa)
        
        return schedule_clean, h_idx, a_idx, actual_margin, actual_total, hfa
    
    def _optimize_feature_selection(self, team_data, available_features, target_columns):
        """
        Optimize feature selection and weights for each target using differential evolution
        Similar to original approach but with ML models
        """
        best_features = {}
        best_weights = {}
        best_scores = {}
        
        for target_col in target_columns:
            y_target = team_data[target_col].values
            n_features = len(available_features)
            
            def feature_selection_objective(params):
                # First half: binary selections, second half: continuous weights
                feature_selection = np.array(params[:n_features])
                weights = np.array(params[n_features:])
                
                # Get selected features
                selected_mask = feature_selection > 0.5
                selected_indices = np.where(selected_mask)[0]
                
                if len(selected_indices) == 0:
                    return 1.0  # Bad score if no features selected
                
                # Get selected features and normalize weights
                selected_weights = weights[selected_indices]
                selected_weights = selected_weights / np.sum(selected_weights)
                
                # Compute weighted combination of selected features
                X_selected = team_data[[available_features[i] for i in selected_indices]].values
                combined = np.sum(X_selected * selected_weights, axis=1)
                
                # For rank targets, convert to ranks
                if 'rank' in target_col.lower():
                    combined_ranks = pd.Series(combined).rank(ascending=False)
                    corr = spearmanr(combined_ranks, y_target).correlation
                else:
                    corr = spearmanr(combined, y_target).correlation
                
                if np.isnan(corr):
                    corr = 0.0
                
                return -corr  # Minimize negative correlation
            
            # Optimize feature selection
            bounds = [(0, 1)] * n_features + [(0, 1)] * n_features
            result = differential_evolution(
                feature_selection_objective,
                bounds=bounds,
                seed=self.random_state,
                maxiter=1000,
                polish=True,
                strategy='best1bin'
            )
            
            # Extract results
            feature_selection = result.x[:n_features] > 0.5
            weights = result.x[n_features:]
            
            selected_indices = np.where(feature_selection)[0]
            selected_features = [available_features[i] for i in selected_indices]
            selected_weights = weights[selected_indices]
            selected_weights = selected_weights / np.sum(selected_weights)
            
            best_features[target_col] = selected_features
            best_weights[target_col] = dict(zip(selected_features, selected_weights))
            best_scores[target_col] = -result.fun
            
        return best_features, best_weights, best_scores
    
    def _train_rank_models(self, team_data, selected_features, target_columns):
        """Train separate models for each target column using selected features"""
        target_models = {}
        target_predictions = {}
        
        for target_col in target_columns:
            features_for_target = selected_features[target_col]
            
            if len(features_for_target) == 0:
                # Fallback: use mean
                target_predictions[target_col] = np.zeros(len(team_data))
                continue
            
            X_team = team_data[features_for_target].values.astype(float)
            X_team_scaled = self.scaler.fit_transform(X_team)
            
            y_target = team_data[target_col].values
            
            model = self._fit_regressor(X_team_scaled, y_target, f'target_{target_col}')
            target_models[target_col] = model
            predictions = model.predict(X_team_scaled)
            
            # CRITICAL: If this is a rank column (lower is better), invert it to a rating (higher is better)
            if 'rank' in target_col.lower():
                # Convert ranks to ratings: invert so lower rank = higher rating
                # Use negative of the prediction so that rank 1 becomes the highest rating
                target_predictions[target_col] = -predictions
            else:
                # For rating-based targets, use as-is
                target_predictions[target_col] = predictions
            
            # Store individual correlations using the ORIGINAL target values
            if 'rank' in target_col.lower():
                # For ranks: compare predicted ranks vs actual ranks
                pred_ranks = pd.Series(predictions).rank(ascending=True)  # Lower prediction = better rank
                actual_ranks = y_target
                corr = spearmanr(pred_ranks, actual_ranks).correlation
            else:
                # For ratings: direct correlation
                corr = spearmanr(predictions, y_target).correlation
            self.diagnostics[f'{target_col}_model_correlation'] = corr
        
        self.models['target_models'] = target_models
        return target_predictions
    
    def _train_margin_model(self, team_data, selected_features, h_idx, a_idx, actual_margin, hfa):
        """Train game-level margin prediction model using union of all selected features"""
        # Use union of features selected for all targets
        all_selected_features = set()
        for features in selected_features.values():
            all_selected_features.update(features)
        
        if len(all_selected_features) == 0:
            # Fallback: no margin model
            self.models['margin_model'] = None
            return None
        
        margin_features = list(all_selected_features)
        X_team = team_data[margin_features].values.astype(float)
        X_team_scaled = self.scaler.fit_transform(X_team)
        
        game_features = X_team_scaled[h_idx] - X_team_scaled[a_idx]
        X_games = np.column_stack([game_features, hfa])
        
        margin_model = self._fit_regressor(X_games, actual_margin, 'margin')
        self.models['margin_model'] = margin_model
        self.models['margin_features'] = margin_features
        
        # Diagnostics
        margin_pred = margin_model.predict(X_games)
        self.diagnostics['margin_mae'] = mean_absolute_error(actual_margin, margin_pred)
        self.diagnostics['margin_r2'] = r2_score(actual_margin, margin_pred)
        
        return X_games
    
    def _compute_margin_ratings(self, team_data, selected_features):
        """Compute margin-based ratings for each team"""
        if self.models.get('margin_model') is None:
            return np.zeros(len(team_data))
        
        margin_features = self.models['margin_features']
        X_team = team_data[margin_features].values.astype(float)
        X_team_scaled = self.scaler.transform(X_team)
        margin_features_neutral = np.column_stack([X_team_scaled, np.zeros(len(X_team_scaled))])
        margin_ratings = self.models['margin_model'].predict(margin_features_neutral)
        return margin_ratings
    
    def _optimize_multi_target_ensemble(self, target_predictions, margin_ratings, team_data, 
                                       target_columns, h_idx, a_idx, actual_margin, hfa,
                                       mae_weight=0.1, correlation_weight=0.9, 
                                       min_target_weight=0.3):
        """
        Optimize ensemble considering multiple targets and game prediction accuracy
        
        For rank-based targets (like ELO_Rank), we optimize Spearman correlation
        For rating-based targets, we also use Spearman correlation for consistency
        
        Strategy: Maximize correlation first, then minimize MAE as tiebreaker
        
        Parameters:
        -----------
        min_target_weight : float
            Minimum weight for target model (prevents margin model from dominating)
            Default: 0.3 (at least 30% weight on target-based predictions)
        """
        baseline_mae = np.mean(np.abs(actual_margin))
        n_targets = len(target_columns)
        
        # First, evaluate each component separately for diagnostics
        component_correlations = {}
        
        # Target model correlations
        for col in target_columns:
            # target_predictions already has ranks converted to ratings (inverted)
            # So we need to rank these ratings and compare to actual ranks
            if 'rank' in col.lower():
                # Target predictions are now ratings (higher = better)
                # Convert back to ranks for comparison
                predicted_ranks = pd.Series(target_predictions[col]).rank(ascending=False)
                actual_ranks = team_data[col].values
                corr = spearmanr(predicted_ranks, actual_ranks).correlation
            else:
                corr = spearmanr(target_predictions[col], team_data[col].values).correlation
            component_correlations[f'{col}_target_only'] = corr
        
        # Margin model correlation
        for col in target_columns:
            if 'rank' in col.lower():
                # Margin ratings are ratings (higher = better)
                # Convert to ranks for comparison
                rating_ranks = pd.Series(margin_ratings).rank(ascending=False)
                corr = spearmanr(rating_ranks, team_data[col].values).correlation
            else:
                corr = spearmanr(margin_ratings, team_data[col].values).correlation
            component_correlations[f'{col}_margin_only'] = corr
        
        self.diagnostics['component_correlations'] = component_correlations
        
        def multi_objective(params):
            """
            Optimize both target weights and ensemble mixing weight
            
            params: [target_weight_1, ..., target_weight_n, ensemble_weight]
            
            Primary goal: Maximize correlation with targets
            Secondary goal: Minimize game prediction MAE
            """
            if len(params) != n_targets + 1:
                raise ValueError(f"Expected {n_targets + 1} parameters")
            
            target_weights = params[:n_targets]
            ensemble_weight = params[-1]
            
            # Enforce minimum target weight constraint
            ensemble_weight = max(min_target_weight, min(1.0, ensemble_weight))
            
            # Normalize target weights to sum to 1
            target_weights = np.array(target_weights)
            target_weights = np.abs(target_weights) / np.sum(np.abs(target_weights))
            
            # Create composite target prediction (all are now ratings where higher = better)
            composite_pred = sum(w * target_predictions[col] for w, col in zip(target_weights, target_columns))
            
            # Ensemble with margin-based ratings (also higher = better)
            ensemble_ratings = ensemble_weight * composite_pred + (1 - ensemble_weight) * margin_ratings
            
            # Calculate correlation scores with each target
            correlation_scores = []
            for col in target_columns:
                target_values = team_data[col].values
                
                # For rank targets: convert our ratings back to ranks for correlation
                if 'rank' in col.lower():
                    # ensemble_ratings are ratings (higher = better)
                    # Convert to ranks (lower = better) for comparison
                    predicted_ranks = pd.Series(ensemble_ratings).rank(ascending=False)
                    corr = spearmanr(predicted_ranks, target_values).correlation
                else:
                    # For ratings, use direct correlation
                    corr = spearmanr(ensemble_ratings, target_values).correlation
                
                if np.isnan(corr):
                    corr = 0.0
                correlation_scores.append(corr)
            
            # Average correlation across all targets
            avg_correlation = np.mean(correlation_scores)
            
            # Game prediction accuracy (secondary objective)
            game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
            game_mae = np.mean(np.abs(game_pred - actual_margin))
            normalized_mae = game_mae / baseline_mae
            
            # Add penalty for low target weight to enforce minimum
            weight_penalty = 0
            if ensemble_weight < min_target_weight:
                weight_penalty = 10.0 * (min_target_weight - ensemble_weight)
            
            # PRIMARY: Maximize correlation (larger weight)
            # SECONDARY: Minimize MAE (smaller weight as tiebreaker)
            loss = -correlation_weight * avg_correlation + mae_weight * normalized_mae + weight_penalty
            
            return loss
        
        # Set up optimization bounds
        # Target weights: 0 to 2, ensemble weight: min_target_weight to 1
        bounds = [(0, 2)] * n_targets + [(min_target_weight, 1)]
        
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
        
        # Ensure minimum weight is respected
        optimal_ensemble_weight = max(min_target_weight, optimal_ensemble_weight)
        
        return optimal_target_weights, optimal_ensemble_weight
    
    def fit(self, team_data, schedule_df, available_features, target_columns=['ELO_Rank'], 
            mae_weight=0.1, correlation_weight=0.9, rating_scale=5.0, rating_center=0.0, min_target_weight=0.3):
        """
        Fit baseball power rating system with automatic feature selection
        
        Parameters:
        -----------
        team_data : DataFrame
            Team statistics with 'Team' column and features
        schedule_df : DataFrame  
            Game results with home_team, away_team, home_score, away_score
        available_features : list
            ALL available feature columns - system will select best subset
        target_columns : list
            List of target columns to optimize for (e.g., ['ELO_Rank'])
        mae_weight : float
            Weight for game prediction accuracy in optimization
        correlation_weight : float
            Weight for target correlation in optimization
        rating_scale : float
            Target standard deviation for ratings (default 5.0 for college baseball)
            This creates a flexible scale: ~68% of teams within ±5 of center
            Higher values = more spread, lower values = more compressed
        rating_center : float
            Center point for ratings (default 0.0)
        min_target_weight : float
            Minimum weight for target-based model in ensemble (default 0.3)
            Prevents margin model from dominating when features don't correlate well
        """
        # Validate inputs
        for col in target_columns:
            if col not in team_data.columns:
                raise ValueError(f"Target column '{col}' not found in team_data")
        
        for feat in available_features:
            if feat not in team_data.columns:
                raise ValueError(f"Feature '{feat}' not found in team_data")
        
        # Check for Team column
        if 'Team' not in team_data.columns:
            raise ValueError("team_data must contain 'Team' column")
        
        # Set Team as index
        team_data_indexed = team_data.set_index('Team', drop=False)
        
        # Step 1: Feature selection for each target
        print("Optimizing feature selection...")
        selected_features, feature_weights, selection_scores = self._optimize_feature_selection(
            team_data_indexed, available_features, target_columns
        )
        
        # Store feature selection results
        self.selected_features = selected_features
        self.feature_weights = feature_weights
        
        # Print selected features
        for target_col in target_columns:
            print(f"\nSelected features for {target_col}:")
            for feat, weight in sorted(feature_weights[target_col].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {weight*100:.1f}%")
            print(f"  Initial correlation: {selection_scores[target_col]:.4f}")
        
        # Prepare game data
        schedule_clean, h_idx, a_idx, actual_margin, actual_total, hfa = self._prepare_game_data(
            team_data_indexed, schedule_df
        )
        
        # Step 2: Train ML models for each target using selected features
        print("\nTraining ML models...")
        target_predictions = self._train_rank_models(
            team_data_indexed, selected_features, target_columns
        )
        
        # Step 3: Train margin model (optional - can get 0 weight in ensemble)
        print("Training margin prediction model...")
        self._train_margin_model(
            team_data_indexed, selected_features, h_idx, a_idx, actual_margin, hfa
        )
        
        # Compute margin-based ratings
        margin_ratings = self._compute_margin_ratings(team_data_indexed, selected_features)
        
        # Step 4: Optimize ensemble (margin model can get 0 weight)
        print("Optimizing ensemble weights...")
        optimal_target_weights, optimal_ensemble_weight = self._optimize_multi_target_ensemble(
            target_predictions, margin_ratings, team_data_indexed, target_columns,
            h_idx, a_idx, actual_margin, hfa, mae_weight, correlation_weight, min_target_weight
        )
        
        # Create final ensemble ratings
        composite_pred = sum(w * target_predictions[col] 
                           for w, col in zip(optimal_target_weights, target_columns))
        
        ensemble_ratings = (optimal_ensemble_weight * composite_pred + 
                          (1 - optimal_ensemble_weight) * margin_ratings)
        
        # Scale ratings using standard deviation approach (flexible scale)
        # This allows natural expansion/compression based on team quality differences
        ensemble_ratings = ensemble_ratings - ensemble_ratings.mean()
        current_std = ensemble_ratings.std()
        
        if current_std > 0:
            # Scale to target standard deviation
            scaling_factor = rating_scale / current_std
            ensemble_ratings = ensemble_ratings * scaling_factor
        
        # Center at desired point
        ensemble_ratings = ensemble_ratings - ensemble_ratings.mean() + rating_center
        
        # Store results with clean rounding
        self.team_ratings = pd.DataFrame({
            'Team': team_data_indexed['Team'].values,
            'Rating': self._clean_round(ensemble_ratings)
        })
        
        # Add individual component ratings for analysis
        for i, col in enumerate(target_columns):
            component_scaled = target_predictions[col] - target_predictions[col].mean()
            component_std = component_scaled.std()
            if component_std > 0:
                component_scaled = (component_scaled / component_std) * rating_scale
            component_scaled = component_scaled + rating_center
            self.team_ratings[f'{col}_component'] = self._clean_round(component_scaled)
        
        margin_scaled = margin_ratings - margin_ratings.mean()
        margin_std = margin_scaled.std()
        if margin_std > 0:
            margin_scaled = (margin_scaled / margin_std) * rating_scale
        margin_scaled = margin_scaled + rating_center
        self.team_ratings['margin_component'] = self._clean_round(margin_scaled)
        
        # Final diagnostics
        final_correlations = {}
        for col in target_columns:
            if 'rank' in col.lower():
                # ensemble_ratings are ratings (higher = better)
                # Convert to ranks for comparison with actual ranks
                rating_ranks = pd.Series(ensemble_ratings).rank(ascending=False)
                corr = spearmanr(rating_ranks, team_data_indexed[col].values).correlation
            else:
                corr = spearmanr(ensemble_ratings, team_data_indexed[col].values).correlation
            final_correlations[f'final_{col}_correlation'] = corr
        
        game_pred = ensemble_ratings[h_idx] + hfa - ensemble_ratings[a_idx]
        final_mae = np.mean(np.abs(game_pred - actual_margin))
        
        self.diagnostics.update({
            'target_columns': target_columns,
            'selected_features': selected_features,
            'feature_weights': feature_weights,
            'feature_selection_scores': selection_scores,
            'optimal_target_weights': dict(zip(target_columns, optimal_target_weights)),
            'optimal_ensemble_weight': optimal_ensemble_weight,
            'final_game_mae': final_mae,
            'baseline_margin_mae': np.mean(np.abs(actual_margin)),
            'n_games': len(actual_margin),
            'rating_range': ensemble_ratings.max() - ensemble_ratings.min(),
            'rating_std': ensemble_ratings.std(),
            'rating_scale': rating_scale,
            'rating_center': rating_center,
            'home_field_advantage': self.hfa,
            **final_correlations
        })
        
        return self
    
    def predict_game(self, home_team, away_team, neutral=False):
        """Predict margin and scores for a specific game"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted before making predictions")
        
        team_lookup = dict(zip(self.team_ratings['Team'], self.team_ratings['Rating']))
        
        # Get ratings
        home_rating = team_lookup.get(home_team, 0)
        away_rating = team_lookup.get(away_team, 0)
        
        hfa = 0 if neutral else self.hfa
        
        # Calculate predicted margin
        margin = round(home_rating + hfa - away_rating, 2)
        
        # For baseball, typical game total is around 8-10 runs
        # This is a simple model; could be enhanced with offensive/defensive components
        baseline_score = 4.5  # Average runs per team
        
        home_score = round(baseline_score + margin/2, 1)
        away_score = round(baseline_score - margin/2, 1)
        
        # Format output
        if margin < 0:
            favorite = away_team
            margin_str = f'{away_team} by {abs(margin):.2f}'
        else:
            favorite = home_team
            margin_str = f'{home_team} by {margin:.2f}'
        
        return {
            'margin': margin_str,
            'home_score': home_score,
            'away_score': away_score,
            'predicted_score': f'{home_team} {home_score}, {away_team} {away_score}'
        }
    
    def get_rankings(self, n=25):
        """Get top N teams by rating"""
        if self.team_ratings is None:
            raise ValueError("Model must be fitted first")
        
        return self.team_ratings.nlargest(n, 'Rating')
    
    def print_diagnostics(self):
        """Print comprehensive model diagnostics"""
        print("\nBaseball Power Rating System Diagnostics")
        print("=" * 60)
        
        # Component correlations (individual performance)
        if 'component_correlations' in self.diagnostics:
            print("\nCOMPONENT CORRELATIONS (individual models):")
            for key, value in self.diagnostics['component_correlations'].items():
                print(f"  {key}: {value:.4f} ({value*100:.1f}%)")
        
        # Feature selection results
        print("\nFEATURE SELECTION:")
        for target_col in self.diagnostics['target_columns']:
            print(f"\n  {target_col}:")
            n_selected = len(self.diagnostics['selected_features'][target_col])
            print(f"    Features selected: {n_selected}")
            print(f"    Feature selection correlation: {self.diagnostics['feature_selection_scores'][target_col]:.4f}")
            
            for feat, weight in sorted(self.diagnostics['feature_weights'][target_col].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"      {feat}: {weight*100:.1f}%")
        
        # Target information
        print(f"\nTARGET OPTIMIZATION:")
        print(f"  Target columns: {self.diagnostics['target_columns']}")
        print("\n  Optimal target weights:")
        for target, weight in self.diagnostics['optimal_target_weights'].items():
            print(f"    {target}: {weight:.3f}")
        
        ensemble_weight = self.diagnostics['optimal_ensemble_weight']
        margin_weight = 1 - ensemble_weight
        print(f"\n  Ensemble composition:")
        print(f"    Target model weight: {ensemble_weight:.3f}")
        print(f"    Margin model weight: {margin_weight:.3f}")
        
        if margin_weight < 0.01:
            print(f"    Note: Margin model essentially not used (weight ~0)")
        elif ensemble_weight < 0.2:
            print(f"    WARNING: Target model has very low weight - features may not correlate well with target")
        
        # Model performance
        print(f"\nGAME PREDICTION PERFORMANCE:")
        print(f"  Home Field Advantage: {self.diagnostics['home_field_advantage']:.3f} runs")
        print(f"  Final MAE: {self.diagnostics['final_game_mae']:.3f} runs")
        print(f"  Baseline MAE: {self.diagnostics['baseline_margin_mae']:.3f} runs")
        improvement = (1 - self.diagnostics['final_game_mae']/self.diagnostics['baseline_margin_mae'])
        print(f"  Improvement: {improvement:.1%}")
        
        # Target correlations
        print(f"\nTARGET CORRELATIONS (Spearman) - Final Ensemble:")
        for key, value in self.diagnostics.items():
            if 'final_' in key and '_correlation' in key:
                target_name = key.replace('final_', '').replace('_correlation', '')
                print(f"  {target_name}: {value:.4f} ({value*100:.1f}%)")
        
        print(f"\nDATA SUMMARY:")
        print(f"  Games analyzed: {self.diagnostics['n_games']}")
        print(f"  Rating scale (target std): {self.diagnostics['rating_scale']:.2f}")
        print(f"  Actual rating std: {self.diagnostics['rating_std']:.2f}")
        print(f"  Rating range: {self.diagnostics['rating_range']:.2f}")
        print(f"  Rating center: {self.diagnostics['rating_center']:.2f}\n")
        
        # Show top teams
        print("=" * 60)
        print("TOP 10 TEAMS:")
        print(self.get_rankings(10).to_string(index=False))


# --- USAGE EXAMPLE ---
def build_baseball_power_ratings(team_data, schedule_df, available_features, 
                                 target_columns=['ELO_Rank'], mae_weight=0.1, 
                                 correlation_weight=0.9, rating_scale=5.0, 
                                 rating_center=0.0, min_target_weight=0.3,
                                 home_field_advantage=0.8):
    """
    Build baseball power ratings with automatic feature selection
    
    Parameters:
    -----------
    team_data : DataFrame
        Must contain 'Team' column and all available_features
    schedule_df : DataFrame
        Must contain: home_team, away_team, home_score, away_score
        Optional: neutral (True/False for neutral site games)
    available_features : list
        ALL available features - system will select best subset
        Example: ['BB%', 'OPS', 'wRC+', 'PYTHAG', 'fWAR', ...]
    target_columns : list
        Target columns to optimize for (e.g., ['ELO_Rank'])
    mae_weight : float
        Weight for game prediction accuracy in optimization
        Set to 0 to only optimize for target correlation
    correlation_weight : float
        Weight for target correlation in optimization (default 0.9)
        Higher values prioritize matching target rankings
    rating_scale : float
        Target standard deviation for ratings (default 5.0)
        Creates flexible scale: ~68% of teams within ±5 of center
        Typical values:
        - 3.0: Compressed scale (small differences matter more)
        - 5.0: Moderate scale (good for college baseball)
        - 10.0: Expanded scale (emphasizes large differences)
    rating_center : float
        Center point for ratings (default 0.0)
        Could use 100 for a 0-200 scale, 50 for 0-100, etc.
    min_target_weight : float
        Minimum weight for target-based model in ensemble (default 0.3)
    home_field_advantage : float
        Home field advantage in runs (default 0.8)
        Typical range: 0.5 to 1.2 runs for college baseball
    
    Returns:
    --------
    result_data : DataFrame
        Original team_data with Rating column added
    diagnostics : dict
        Model performance metrics and selected features
    system : BaseballPowerRatingSystem
        Fitted system for making predictions
    """
    system = BaseballPowerRatingSystem(home_field_advantage=home_field_advantage)
    system.fit(team_data, schedule_df, available_features, target_columns, 
               mae_weight=mae_weight, correlation_weight=correlation_weight,
               rating_scale=rating_scale, rating_center=rating_center, 
               min_target_weight=min_target_weight)
    
    # Merge ratings back to original team_data
    result_data = team_data.merge(
        system.team_ratings[['Team', 'Rating']], 
        on='Team', 
        how='left'
    )
    
    return result_data, system.diagnostics, system
    
def hyperparameter_search_for_mae(team_data, schedule_df, available_features,
                                   target_columns=['ELO_Rank'], n_trials=100, n_jobs=-1, verbose=True):
    """
    Perform parallelized random search to find settings that minimize game prediction MAE
    while maintaining good correlation with target
    
    Parameters:
    -----------
    team_data : DataFrame
        Team statistics with 'Team' column and features
    schedule_df : DataFrame
        Game results with home_team, away_team, home_score, away_score
    available_features : list
        ALL available features - system will select best subset
    target_columns : list
        Target columns to optimize for (e.g., ['ELO_Rank'])
    n_trials : int
        Number of random combinations to try (default: 100)
    n_jobs : int
        Number of parallel jobs (-1 uses all available cores, default: -1)
    verbose : bool
        Print progress during search
    
    Returns:
    --------
    dict with keys:
        'best_params': Best hyperparameter combination found
        'best_system': Fitted system with best parameters
        'best_result_data': Result data with ratings
        'results_df': DataFrame with all trials and their performance
        'best_mae': Best MAE achieved
    """
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr
    import numpy as np
    import pandas as pd
    import time
    
    print("PARALLELIZED RANDOM SEARCH FOR BEST GAME PREDICTION MAE")
    print("=" * 70)
    
    # Clean schedule data once
    model_schedule = schedule_df.drop_duplicates(
        subset=['Date', 'home_team', 'away_team', 'home_score', 'away_score']
    ).dropna().reset_index(drop=True)
    
    # Define search space
    search_space = {
        'rating_scale': [1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
        'mae_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'correlation_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'min_target_weight': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'home_field_advantage': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    }
    
    print(f"Random search space: {n_trials} random combinations")
    print(f"  rating_scale: {len(search_space['rating_scale'])} values")
    print(f"  mae_weight: {len(search_space['mae_weight'])} values")
    print(f"  correlation_weight: {len(search_space['correlation_weight'])} values")
    print(f"  min_target_weight: {len(search_space['min_target_weight'])} values")
    print(f"  home_field_advantage: {len(search_space['home_field_advantage'])} values")
    
    # Generate random combinations
    np.random.seed(42)  # For reproducibility
    random_combinations = []
    for _ in range(n_trials):
        combo = (
            np.random.choice(search_space['rating_scale']),
            np.random.choice(search_space['mae_weight']),
            np.random.choice(search_space['correlation_weight']),
            np.random.choice(search_space['min_target_weight']),
            np.random.choice(search_space['home_field_advantage'])
        )
        random_combinations.append(combo)
    
    # Determine number of jobs
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} parallel jobs")
    print()
    
    def evaluate_params(idx, params_tuple):
        """Evaluate a single parameter combination"""
        rating_scale, mae_weight, correlation_weight, min_target_weight, home_field_advantage = params_tuple
        
        params = {
            'rating_scale': rating_scale,
            'mae_weight': mae_weight,
            'correlation_weight': correlation_weight,
            'min_target_weight': min_target_weight,
            'home_field_advantage': home_field_advantage
        }
        
        try:
            # Suppress print statements during parallel execution
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Build system with these parameters
            result_data, diagnostics, system = build_baseball_power_ratings(
                team_data=team_data,
                schedule_df=model_schedule,
                available_features=available_features,
                target_columns=target_columns,
                rating_scale=params['rating_scale'],
                mae_weight=params['mae_weight'],
                correlation_weight=params['correlation_weight'],
                rating_center=0.0,
                min_target_weight=params['min_target_weight'],
                home_field_advantage=params['home_field_advantage']
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Extract metrics
            mae = diagnostics['final_game_mae']
            
            # Get correlation for each target
            correlations = {}
            for col in target_columns:
                rating_ranks = result_data['Rating'].rank(ascending=False)
                if 'rank' in col.lower():
                    corr = spearmanr(rating_ranks, result_data[col]).correlation
                else:
                    corr = spearmanr(result_data['Rating'], result_data[col]).correlation
                correlations[col] = corr
            
            avg_correlation = np.mean(list(correlations.values()))
            
            # Store results
            trial_result = {
                'trial': idx + 1,
                'mae': mae,
                'avg_correlation': avg_correlation,
                **params,
                **{f'{col}_correlation': correlations[col] for col in target_columns}
            }
            
            return trial_result
            
        except Exception as e:
            # Restore stdout if error
            sys.stdout = old_stdout
            return {
                'trial': idx + 1,
                'mae': np.nan,
                'avg_correlation': np.nan,
                'error': str(e),
                **params
            }
    
    start_time = time.time()
    
    # Run parallel random search
    print("Running parallel random search...")
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_params)(idx, params) 
        for idx, params in enumerate(random_combinations)
    )
    
    elapsed_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out failed trials
    valid_results = results_df[~results_df['mae'].isna()].copy()
    valid_results = valid_results.sort_values('mae')
    
    # Find best based on MAE (with correlation threshold)
    best_candidates = valid_results[valid_results['avg_correlation'] > 0.985]
    
    if len(best_candidates) > 0:
        best_trial = best_candidates.iloc[0]
        best_params = {
            'rating_scale': best_trial['rating_scale'],
            'mae_weight': best_trial['mae_weight'],
            'correlation_weight': best_trial['correlation_weight'],
            'min_target_weight': best_trial['min_target_weight'],
            'home_field_advantage': best_trial['home_field_advantage']
        }
        best_mae = best_trial['mae']
        
        # Rebuild best system for return
        print("\nRebuilding best system...")
        best_result_data, _, best_system = build_baseball_power_ratings(
            team_data=team_data,
            schedule_df=model_schedule,
            available_features=available_features,
            target_columns=target_columns,
            **best_params,
            rating_center=0.0
        )
    else:
        best_params = None
        best_mae = None
        best_system = None
        best_result_data = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("RANDOM SEARCH COMPLETE")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Successful trials: {len(valid_results)}/{n_trials}")
    print(f"Average time per trial: {elapsed_time/n_trials:.2f} seconds")
    
    if best_params is not None:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS (Lowest MAE with Correlation > 0.985):")
        print(f"  Rating Scale:          {best_params['rating_scale']:.2f}")
        print(f"  MAE Weight:            {best_params['mae_weight']:.2f}")
        print(f"  Correlation Weight:    {best_params['correlation_weight']:.2f}")
        print(f"  Min Target Weight:     {best_params['min_target_weight']:.2f}")
        print(f"  Home Field Advantage:  {best_params['home_field_advantage']:.2f} runs")
        
        print(f"\nPERFORMANCE:")
        print(f"  Best MAE:              {best_mae:.3f} runs")
        print(f"  Avg Correlation:       {best_trial['avg_correlation']:.4f}")
        for col in target_columns:
            print(f"  {col} Correlation:    {best_trial[f'{col}_correlation']:.4f}")
        
        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY MAE:")
        display_cols = ['trial', 'mae', 'avg_correlation', 'rating_scale', 
                       'mae_weight', 'correlation_weight', 'min_target_weight', 'home_field_advantage']
        if len(valid_results) > 0:
            print(valid_results[display_cols].head(10).to_string(index=False))
        
        print("\n" + "=" * 70)
        print("PARAMETER ANALYSIS (Top 10):")
        
        # Analyze patterns in top performers
        if len(valid_results) >= 10:
            top10 = valid_results.head(10)
            
            # Show distribution of each parameter
            print("\nRating Scale distribution:")
            scale_counts = top10['rating_scale'].value_counts().sort_index()
            for val, count in scale_counts.items():
                print(f"  {val:.1f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMAE Weight distribution:")
            mae_counts = top10['mae_weight'].value_counts().sort_index()
            for val, count in mae_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nCorrelation Weight distribution:")
            corr_counts = top10['correlation_weight'].value_counts().sort_index()
            for val, count in corr_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMin Target Weight distribution:")
            target_counts = top10['min_target_weight'].value_counts().sort_index()
            for val, count in target_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nHome Field Advantage distribution:")
            hfa_counts = top10['home_field_advantage'].value_counts().sort_index()
            for val, count in hfa_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nTop 10 averages:")
            print(f"  Avg rating_scale:          {top10['rating_scale'].mean():.2f}")
            print(f"  Avg mae_weight:            {top10['mae_weight'].mean():.2f}")
            print(f"  Avg correlation_weight:    {top10['correlation_weight'].mean():.2f}")
            print(f"  Avg min_target_weight:     {top10['min_target_weight'].mean():.2f}")
            print(f"  Avg home_field_advantage:  {top10['home_field_advantage'].mean():.2f}")
    else:
        print("\nNo valid configurations found with correlation > 0.985.")
        if len(valid_results) > 0:
            print("Best configuration without correlation threshold:")
            best_any = valid_results.iloc[0]
            print(f"  MAE: {best_any['mae']:.3f}")
            print(f"  Correlation: {best_any['avg_correlation']:.4f}")
            print(f"  Home Field Advantage: {best_any['home_field_advantage']:.2f}")
    
    print("=" * 70)
    
    return {
        'best_params': best_params,
        'best_system': best_system,
        'best_result_data': best_result_data,
        'results_df': valid_results,
        'best_mae': best_mae
    }
    
    def evaluate_params(idx, params_tuple):
        """Evaluate a single parameter combination"""
        rating_scale, mae_weight, correlation_weight, min_target_weight = params_tuple
        
        params = {
            'rating_scale': rating_scale,
            'mae_weight': mae_weight,
            'correlation_weight': correlation_weight,
            'min_target_weight': min_target_weight
        }
        
        try:
            # Suppress print statements during parallel execution
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Build system with these parameters
            result_data, diagnostics, system = build_baseball_power_ratings(
                team_data=team_data,
                schedule_df=model_schedule,
                available_features=available_features,
                target_columns=target_columns,
                rating_scale=params['rating_scale'],
                mae_weight=params['mae_weight'],
                correlation_weight=params['correlation_weight'],
                rating_center=0.0,
                min_target_weight=params['min_target_weight']
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Extract metrics
            mae = diagnostics['final_game_mae']
            
            # Get correlation for each target
            correlations = {}
            for col in target_columns:
                rating_ranks = result_data['Rating'].rank(ascending=False)
                if 'rank' in col.lower():
                    corr = spearmanr(rating_ranks, result_data[col]).correlation
                else:
                    corr = spearmanr(result_data['Rating'], result_data[col]).correlation
                correlations[col] = corr
            
            avg_correlation = np.mean(list(correlations.values()))
            
            # Store results
            trial_result = {
                'trial': idx + 1,
                'mae': mae,
                'avg_correlation': avg_correlation,
                **params,
                **{f'{col}_correlation': correlations[col] for col in target_columns}
            }
            
            return trial_result
            
        except Exception as e:
            # Restore stdout if error
            sys.stdout = old_stdout
            return {
                'trial': idx + 1,
                'mae': np.nan,
                'avg_correlation': np.nan,
                'error': str(e),
                **params
            }
    
    start_time = time.time()
    
    # Run parallel grid search
    print("Running parallel grid search...")
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(evaluate_params)(idx, params) 
        for idx, params in enumerate(all_combinations)
    )
    
    elapsed_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out failed trials
    valid_results = results_df[~results_df['mae'].isna()].copy()
    valid_results = valid_results.sort_values('mae')
    
    # Find best based on MAE (with correlation threshold)
    best_candidates = valid_results[valid_results['avg_correlation'] > 0.75]
    
    if len(best_candidates) > 0:
        best_trial = best_candidates.iloc[0]
        best_params = {
            'rating_scale': best_trial['rating_scale'],
            'mae_weight': best_trial['mae_weight'],
            'correlation_weight': best_trial['correlation_weight'],
            'min_target_weight': best_trial['min_target_weight']
        }
        best_mae = best_trial['mae']
        
        # Rebuild best system for return
        print("\nRebuilding best system...")
        best_result_data, _, best_system = build_baseball_power_ratings(
            team_data=team_data,
            schedule_df=model_schedule,
            available_features=available_features,
            target_columns=target_columns,
            **best_params,
            rating_center=0.0
        )
    else:
        best_params = None
        best_mae = None
        best_system = None
        best_result_data = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Successful trials: {len(valid_results)}/{n_trials}")
    print(f"Average time per trial: {elapsed_time/n_trials:.2f} seconds")
    
    if best_params is not None:
        print("\n" + "=" * 70)
        print("BEST PARAMETERS (Lowest MAE with Correlation > 0.75):")
        print(f"  Rating Scale:        {best_params['rating_scale']:.2f}")
        print(f"  MAE Weight:          {best_params['mae_weight']:.2f}")
        print(f"  Correlation Weight:  {best_params['correlation_weight']:.2f}")
        print(f"  Min Target Weight:   {best_params['min_target_weight']:.2f}")
        
        print(f"\nPERFORMANCE:")
        print(f"  Best MAE:            {best_mae:.3f} runs")
        print(f"  Avg Correlation:     {best_trial['avg_correlation']:.4f}")
        for col in target_columns:
            print(f"  {col} Correlation:  {best_trial[f'{col}_correlation']:.4f}")
        
        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY MAE:")
        display_cols = ['trial', 'mae', 'avg_correlation', 'rating_scale', 
                       'mae_weight', 'correlation_weight', 'min_target_weight']
        if len(valid_results) > 0:
            print(valid_results[display_cols].head(10).to_string(index=False))
        
        print("\n" + "=" * 70)
        print("PARAMETER ANALYSIS (Top 10):")
        
        # Analyze patterns in top performers
        if len(valid_results) >= 10:
            top10 = valid_results.head(10)
            
            # Show distribution of each parameter
            print("\nRating Scale distribution:")
            scale_counts = top10['rating_scale'].value_counts().sort_index()
            for val, count in scale_counts.items():
                print(f"  {val:.1f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMAE Weight distribution:")
            mae_counts = top10['mae_weight'].value_counts().sort_index()
            for val, count in mae_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nCorrelation Weight distribution:")
            corr_counts = top10['correlation_weight'].value_counts().sort_index()
            for val, count in corr_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nMin Target Weight distribution:")
            target_counts = top10['min_target_weight'].value_counts().sort_index()
            for val, count in target_counts.items():
                print(f"  {val:.2f}: {count} times ({count/10*100:.0f}%)")
            
            print("\nTop 10 averages:")
            print(f"  Avg rating_scale:        {top10['rating_scale'].mean():.2f}")
            print(f"  Avg mae_weight:          {top10['mae_weight'].mean():.2f}")
            print(f"  Avg correlation_weight:  {top10['correlation_weight'].mean():.2f}")
            print(f"  Avg min_target_weight:   {top10['min_target_weight'].mean():.2f}")
    else:
        print("\nNo valid configurations found with correlation > 0.75.")
        if len(valid_results) > 0:
            print("Best configuration without correlation threshold:")
            best_any = valid_results.iloc[0]
            print(f"  MAE: {best_any['mae']:.3f}")
            print(f"  Correlation: {best_any['avg_correlation']:.4f}")
    
    print("=" * 70)
    
    return {
        'best_params': best_params,
        'best_system': best_system,
        'best_result_data': best_result_data,
        'results_df': valid_results,
        'best_mae': best_mae
    }

def compare_with_simple_approach(team_data, schedule_df, available_features, 
                                target_column='ELO_Rank'):
    """
    Compare the advanced ML system with a simple weighted feature approach
    
    Returns both correlations and game prediction MAE for comparison
    """
    print("COMPARISON: ML System vs Simple Weighted Approach")
    print("=" * 60)
    
    # Clean schedule data once
    model_schedule = schedule_df.drop_duplicates(
        subset=['Date', 'home_team', 'away_team', 'home_score', 'away_score']
    ).dropna().reset_index(drop=True)
    
    # Simple approach (original method)
    print("\n1. Running Simple Weighted Approach...")
    target = team_data[target_column].values
    
    def simple_objective(weights_raw):
        feature_selection = np.array(weights_raw[:len(available_features)])
        weights = np.array(weights_raw[len(available_features):])
        
        selected_features = [available_features[i] for i in range(len(available_features)) 
                           if feature_selection[i] > 0.5]
        
        if len(selected_features) == 0:
            return 1
        
        weights /= np.sum(weights)
        combined = sum(w * team_data[feat] for w, feat in zip(weights, selected_features))
        ranks = combined.rank(ascending=False)
        corr, _ = spearmanr(ranks, target)
        return -corr
    
    bounds = [(0, 1)] * len(available_features) + [(0, 1)] * len(available_features)
    result = differential_evolution(simple_objective, bounds=bounds, 
                                   strategy='best1bin', maxiter=1000, 
                                   polish=True, seed=42)
    
    feature_selection = np.array(result.x[:len(available_features)]) > 0.5
    weights = np.array(result.x[len(available_features):])
    weights /= np.sum(weights)
    
    selected_features_simple = [available_features[i] for i in range(len(available_features)) 
                               if feature_selection[i]]
    simple_corr = -result.fun
    
    # Calculate simple approach ratings (scaled to same scale for fair comparison)
    simple_combined = sum(w * team_data[feat] for w, feat in zip(weights, selected_features_simple))
    simple_ratings = simple_combined - simple_combined.mean()
    simple_std = simple_ratings.std()
    if simple_std > 0:
        simple_ratings = (simple_ratings / simple_std) * 5.0  # Scale to std=5.0
    
    # Calculate MAE for simple approach
    team_data_indexed = team_data.set_index('Team', drop=False)
    team_to_idx = {team: idx for idx, team in enumerate(team_data_indexed.index)}
    
    valid_games = (
        model_schedule['home_team'].isin(team_to_idx) & 
        model_schedule['away_team'].isin(team_to_idx)
    )
    schedule_clean = model_schedule[valid_games].copy()
    
    h_idx = schedule_clean['home_team'].map(team_to_idx).values
    a_idx = schedule_clean['away_team'].map(team_to_idx).values
    actual_margin = schedule_clean['home_score'].values - schedule_clean['away_score'].values
    
    # Check if neutral field column exists
    if 'neutral' in schedule_clean.columns:
        hfa = np.where(schedule_clean['neutral'] == False, 0.3, 0.0)
    else:
        hfa = np.full(len(schedule_clean), 0.3)
    
    simple_ratings_array = simple_ratings.values
    simple_pred_margin = simple_ratings_array[h_idx] + hfa - simple_ratings_array[a_idx]
    simple_mae = np.mean(np.abs(simple_pred_margin - actual_margin))
    
    print(f"   Features selected: {len(selected_features_simple)}")
    print(f"   Correlation: {simple_corr:.4f} ({simple_corr*100:.1f}%)")
    print(f"   Game prediction MAE: {simple_mae:.3f} runs")
    
    # Advanced ML approach
    print("\n2. Running Advanced ML System...")
    result_data, diagnostics, system = build_baseball_power_ratings(
        team_data=team_data,
        schedule_df=model_schedule,
        available_features=available_features,
        target_columns=[target_column],   
        rating_scale=4.0,
        mae_weight=0.0,           # Pure correlation optimization
        min_target_weight=0.5      # Pure correlation optimization
    )
    
    rating_ranks = result_data['Rating'].rank(ascending=False)
    ml_corr = spearmanr(rating_ranks, result_data[target_column]).correlation
    ml_mae = diagnostics['final_game_mae']
    
    # Comparison summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("\nCorrelation with Target:")
    print(f"  Simple Approach:  {simple_corr:.4f} ({simple_corr*100:.1f}%)")
    print(f"  ML System:        {ml_corr:.4f} ({ml_corr*100:.1f}%)")
    print(f"  Improvement:      {(ml_corr - simple_corr):.4f} ({(ml_corr - simple_corr)*100:.2f}%)")
    
    if ml_corr > simple_corr:
        pct_better = ((ml_corr - simple_corr) / simple_corr) * 100
        print(f"  Relative gain:    {pct_better:.2f}% better")
    
    print("\nGame Prediction MAE:")
    print(f"  Simple Approach:  {simple_mae:.3f} runs")
    print(f"  ML System:        {ml_mae:.3f} runs")
    mae_improvement = simple_mae - ml_mae
    print(f"  Improvement:      {mae_improvement:.3f} runs ({(mae_improvement/simple_mae)*100:.1f}%)")
    
    if ml_mae < simple_mae:
        print(f"  ML System predicts games {(simple_mae/ml_mae - 1)*100:.1f}% more accurately")
    else:
        print(f"  Note: Simple approach predicted games better")
    
    print("=" * 60)
    
    return {
        'simple_correlation': simple_corr,
        'ml_correlation': ml_corr,
        'correlation_improvement': ml_corr - simple_corr,
        'simple_mae': simple_mae,
        'ml_mae': ml_mae,
        'mae_improvement': mae_improvement,
        'system': system,
        'result_data': result_data
    }

def adjust_home_pr(home_win_prob):
    return ((home_win_prob - 50) / 50) * 0.9

def calculate_spread_from_stats(home_pr, away_pr, home_elo, away_elo, location):
    if location != "Neutral":
        home_pr += 0.3
    elo_win_prob = round((10**((home_elo - away_elo) / 400)) / ((10**((home_elo - away_elo) / 400)) + 1) * 100, 2)
    spread = round(adjust_home_pr(elo_win_prob) + home_pr - away_pr, 2)
    return spread, elo_win_prob

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

def calculate_average_expected_wins(group, average_team):
    total_expected_wins = 0

    for _, row in group.iterrows():
        if row['Team'] == row['home_team']:
            total_expected_wins += PEAR_Win_Prob(average_team, row['away_rating'], row['Location']) / 100
        else:
            total_expected_wins += 1 - PEAR_Win_Prob(row['home_rating'], average_team, row['Location']) / 100

    avg_expected_wins = total_expected_wins / len(group)

    return pd.Series({'Team': group['Team'].iloc[0], 'avg_expected_wins': avg_expected_wins, 'total_expected_wins':total_expected_wins})

def calculate_kpi(completed_schedule, ending_data):
    # Precompute lookup dictionaries for faster rank access
    rank_lookup = {team: rank for rank, team in enumerate(ending_data["Team"])}
    default_rank = len(ending_data)

    def get_rank(team):
        return rank_lookup.get(team, default_rank)

    total_teams = len(ending_data)
    kpi_scores = []

    for game in completed_schedule.itertuples(index=False):
        team = game.Team
        opponent = game.Opponent
        home_team = game.home_team

        team_rank = get_rank(team)
        opponent_rank = get_rank(opponent)

        # Rank-based strength (inverted)
        opponent_strength_win = 1 - (opponent_rank / (total_teams + 1))
        opponent_strength_loss = 1 - opponent_strength_win  # Equivalent to (opponent_rank / (total_teams + 1))

        # Determine if team was home or away
        is_home = team == home_team

        # Calculate margin (positive if team won)
        margin = game.home_score - game.away_score
        if not is_home:
            margin = -margin

        # Win/loss factor
        result_multiplier = 1.5 if margin > 0 else -1.5

        # Margin factor
        capped_margin = min(abs(margin), 20)
        margin_factor = 1 + (capped_margin / 20) if margin > 0 else max(0.1, 1 - (capped_margin / 20))
        opponent_strength = opponent_strength_win if margin > 0 else opponent_strength_loss

        # Team strength factor
        team_strength_adj = 1 - (team_rank / (total_teams + 1))

        # Adjusted KPI formula
        adj_grv = (opponent_strength * result_multiplier * margin_factor / 1.5) * (1 + (team_strength_adj / 2))
        kpi_scores.append((team, adj_grv))

    # Convert to DataFrame and aggregate
    kpi_df = pd.DataFrame(kpi_scores, columns=["Team", "KPI_Score"])
    kpi_avg = kpi_df.groupby("Team", as_index=False)["KPI_Score"].mean()

    return kpi_avg

def calculate_resume_quality(group, bubble_team_rating):
    results = []
    resume_quality = 0
    for _, row in group.iterrows():
        team = row['Team']
        is_home = row["home_team"] == team
        is_away = row["away_team"] == team
        opponent_rating = row["away_rating"] if is_home else row["home_rating"]
        if row["Location"] == "Away":
            win_prob = 1 - PEAR_Win_Prob(opponent_rating, bubble_team_rating, row['Location']) / 100    
        else:
            win_prob = PEAR_Win_Prob(bubble_team_rating, opponent_rating, row['Location']) / 100
        team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
        if team_won:
            resume_quality += (1-win_prob)
        else:
            resume_quality -= win_prob
    resume_quality = resume_quality / len(group)
    results.append({"Team": team, "resume_quality": resume_quality})
    return pd.DataFrame(results)

def calculate_game_resume_quality(row, one_seed_rating):
    """Calculate resume quality for a single game."""
    team = row["Team"]
    is_home = row["home_team"] == team
    is_away = row["away_team"] == team
    opponent_rating = row["away_rating"] if is_home else row["home_rating"]
    if row["Location"] == "Away":
        win_prob = 1 - PEAR_Win_Prob(opponent_rating, one_seed_rating, row['Location']) / 100    
    else:
        win_prob = PEAR_Win_Prob(one_seed_rating, opponent_rating, row['Location']) / 100

    team_won = (is_home and row["home_score"] > row["away_score"]) or (is_away and row["away_score"] > row["home_score"])
    
    return (1 - win_prob) if team_won else -win_prob

def calculate_net(weights, stats_and_metrics):  # swapped order
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

def calculate_quadrant_records(completed_schedule, stats_and_metrics):
    # Precompute NET rankings lookup for fast access
    net_lookup = stats_and_metrics.set_index('Team')['NET'].to_dict()
    default_net = 300

    # Quadrant thresholds by location
    quadrant_thresholds = {
        "Home": [25, 50, 100, 307],
        "Neutral": [40, 80, 160, 307],
        "Away": [60, 120, 240, 307]
    }

    records = []

    # Group by team
    for team, group in completed_schedule.groupby('Team'):
        counts = {f'Q{i}_win': 0 for i in range(1, 5)}
        counts.update({f'Q{i}_loss': 0 for i in range(1, 5)})

        for row in group.itertuples(index=False):
            opponent = row.Opponent
            location = row.Location

            # Get opponent NET ranking
            opponent_net = net_lookup.get(opponent, default_net)

            # Determine if team won
            team_won = (
                (row.Team == row.home_team and row.home_score > row.away_score) or
                (row.Team == row.away_team and row.away_score > row.home_score)
            )

            # Determine quadrant
            thresholds = quadrant_thresholds[location]
            quadrant = next((i + 1 for i, val in enumerate(thresholds) if opponent_net <= val), 4)

            # Increment win/loss count
            result_key = f'Q{quadrant}_win' if team_won else f'Q{quadrant}_loss'
            counts[result_key] += 1

        # Build final formatted record: "wins-losses"
        record = {"Team": team}
        for i in range(1, 5):
            record[f"Q{i}"] = f"{counts[f'Q{i}_win']}-{counts[f'Q{i}_loss']}"

        records.append(record)

    # Convert to DataFrame and merge
    quadrant_record_df = pd.DataFrame(records)
    return pd.merge(stats_and_metrics, quadrant_record_df, on='Team', how='left')

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
        # Replace 'W' with team name and space
        return re.sub(r"^W", row["Team"] + " ", result)
    
    elif result.startswith("L"):
        # Match pattern like 'L3-5', extract and swap scores
        match = re.match(r"L(\d+) - (\d+)", result)
        if match:
            return f"{row['Opponent']} {match.group(2)} - {match.group(1)}"

    return result  # In case it's not W or L

def remaining_games_rq(row, one_seed_rating):
    """Calculate resume quality for a single game."""
    team = row["Team"]
    location = row['Location']
    is_home = row["home_team"] == team
    is_away = row["away_team"] == team
    opponent_rating = row["away_rating"] if is_home else row["home_rating"]
    win_prob = PEAR_Win_Prob(one_seed_rating, opponent_rating, location)
    return win_prob

def simulate_games(df, num_simulations=100):
    projected_wins = []
    projected_resume_quality = []
    games_remaining = df.groupby("Team").size()
    for _ in range(num_simulations):
        unique_games = df.drop_duplicates(subset=["Date", "home_team", "away_team"]).copy()
        unique_games["random_val"] = np.random.rand(len(unique_games))
        unique_games["home_wins"] = unique_games["random_val"] < unique_games["home_win_prob"]
        results_map = unique_games.set_index(["Date", "home_team", "away_team"])["home_wins"].to_dict()
        df["home_wins"] = df[["Date", "home_team", "away_team"]].apply(lambda x: results_map.get(tuple(x), None), axis=1)
        df["winner"] = np.where(df["home_wins"], df["home_team"], df["away_team"])
        df["loser"] = np.where(df["home_wins"], df["away_team"], df["home_team"])
        df["win_flag"] = (df["winner"] == df["Team"]).astype(int)
        df["resume_quality_amount"] = df['win_flag'] - df['bubble_win_prob']
        wins_per_team = df.groupby("Team")["win_flag"].sum()
        total_resume_quality_per_team = df.groupby("Team")["resume_quality_amount"].sum()  # SUM instead of mean
        projected_wins.append(wins_per_team)
        projected_resume_quality.append(total_resume_quality_per_team)
    projected_wins_df = pd.DataFrame(projected_wins).mean().round().reset_index()
    projected_wins_df.columns = ["Team", "Remaining_Wins"]
    projected_resume_quality_df = pd.DataFrame(projected_resume_quality).mean().reset_index()
    projected_resume_quality_df.columns = ["Team", "Remaining_RQ"]  # Changed column name to reflect summation
    projected_wins_df["Games_Remaining"] = projected_wins_df["Team"].map(games_remaining)
    projected_wins_df["Remaining_Losses"] = projected_wins_df['Games_Remaining'] - projected_wins_df['Remaining_Wins']
    projected_wins_df = projected_wins_df.merge(projected_resume_quality_df, on="Team", how="left")
    return projected_wins_df

def plot_top_25(title, subtitle, team_images, sorted_df, save_path):
    top_25 = sorted_df.reset_index(drop=True).head(25)
    fig, axs = plt.subplots(5, 5, figsize=(7, 7), dpi=125)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('#CECEB2')

    plt.suptitle(title, fontsize=20, fontweight='bold', color='black')
    fig.text(0.5, 0.92, subtitle, fontsize=10, ha='center', color='black')
    fig.text(0.9, 0.07, "@PEARatings", fontsize=12, ha='right', color='black', fontweight='bold')

    for i, ax in enumerate(axs.ravel()):
        team = top_25.loc[i, 'Team']
        img = team_images.get(team)
        if img:
            ax.imshow(img)
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f"#{i+1} {team}", fontsize=8, fontweight='bold')
        ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def get_conference(team, stats_df):
    return stats_df.loc[stats_df["Team"] == team, "Conference"].values[0]

def count_conflict_conferences(teams, stats_df):
    conferences = [get_conference(team, stats_df) for team in teams]
    return sum(count - 1 for count in Counter(conferences).values() if count > 1)

def resolve_conflicts(formatted_df, stats_df):
    seed_cols = ["2 Seed", "3 Seed", "4 Seed"]

    for seed_col in seed_cols:
        num_regionals = len(formatted_df)

        for i in range(num_regionals):
            row = formatted_df.loc[i]
            teams_i = [row["1 Seed"], row["2 Seed"], row["3 Seed"], row["4 Seed"]]
            conflict_i = count_conflict_conferences(teams_i, stats_df)

            if conflict_i == 0:
                continue  # No conflict to resolve in this regional

            current_team = row[seed_col]

            for j in range(num_regionals):
                if i == j:
                    continue

                alt_team = formatted_df.at[j, seed_col]
                if alt_team == current_team:
                    continue

                # Simulate swap
                row_j = formatted_df.loc[j]
                teams_j = [row_j["1 Seed"], row_j["2 Seed"], row_j["3 Seed"], row_j["4 Seed"]]

                # Apply the swap in test copies
                temp_i = teams_i.copy()
                temp_j = teams_j.copy()
                temp_i[seed_cols.index(seed_col) + 1] = alt_team
                temp_j[seed_cols.index(seed_col) + 1] = current_team

                new_conflict_i = count_conflict_conferences(temp_i, stats_df)
                new_conflict_j = count_conflict_conferences(temp_j, stats_df)

                # Only make swap if it reduces total conflicts
                if (new_conflict_i + new_conflict_j) < (conflict_i + count_conflict_conferences(teams_j, stats_df)):
                    formatted_df.at[i, seed_col] = alt_team
                    formatted_df.at[j, seed_col] = current_team
                    break  # Exit swap loop once we reduce conflict

    return formatted_df

def select_team(region):
    teams, weights = zip(*region.items())
    return random.choices(teams, weights=weights, k=1)[0]

def simulate_best_of_three(team1, team2, rating1, rating2):
    """Simulates a best-of-three series based on ratings."""
    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)  # More precision, rounded later in output
    wins1 = wins2 = 0
    while wins1 < 2 and wins2 < 2:
        win_prob = PEAR_Win_Prob(rating1, rating2)
        if random.random() < win_prob:
            wins1 += 1
        else:
            wins2 += 1
    return team1 if wins1 == 2 else team2

def simulate_game(team1, team2, team_ratings):
    def PEAR_Win_Prob(home_pr, away_pr):
        rating_diff = home_pr - away_pr
        return round(1 / (1 + 10 ** (-rating_diff / 6)), 4)  # More precision, rounded later in output
    rating1 = team_ratings[team1]
    rating2 = team_ratings[team2]
    return team1 if random.random() < PEAR_Win_Prob(rating1, rating2) else team2

def preprocess_team_metadata(stats_and_metrics, actual_tournament):
    seed_map = {}
    region_map = {}
    team_ratings = stats_and_metrics.set_index("Team")["Rating"].to_dict()

    for seed in actual_tournament.columns:
        for region, team in actual_tournament[seed].items():
            seed_map[team] = seed
            region_map[team] = region

    return seed_map, region_map, team_ratings

def simulate_double_elimination(teams, team_ratings, host_advantage=False):
    """Simulates a 4-team double-elimination tournament.

    Args:
        teams (list): List of 4 team names.
        team_ratings (dict): Mapping from team name to rating.
        host_advantage (bool): If True, adds +0.8 to teams[0]'s rating.
    """
    # Copy ratings locally so we don’t mutate the original dict
    ratings = {team: team_ratings[team] for team in teams}
    if host_advantage:
        ratings[teams[0]] += 0.3

    winners_bracket = [teams[0], teams[1], teams[2], teams[3]]
    losers_bracket = []

    game1_winner = simulate_game(winners_bracket[0], winners_bracket[1], ratings)
    game1_loser = winners_bracket[1] if game1_winner == winners_bracket[0] else winners_bracket[0]

    game2_winner = simulate_game(winners_bracket[2], winners_bracket[3], ratings)
    game2_loser = winners_bracket[3] if game2_winner == winners_bracket[2] else winners_bracket[2]

    losers_bracket.extend([game1_loser, game2_loser])

    winners_final_winner = simulate_game(game1_winner, game2_winner, ratings)
    winners_final_loser = game1_winner if winners_final_winner == game2_winner else game2_winner

    losers_round1_winner = simulate_game(losers_bracket[0], losers_bracket[1], ratings)
    losers_final_winner = simulate_game(losers_round1_winner, winners_final_loser, ratings)

    if simulate_game(winners_final_winner, losers_final_winner, ratings) == winners_final_winner:
        return winners_final_winner
    return simulate_game(winners_final_winner, losers_final_winner, ratings)

def run_simulation(team_a, team_b, team_c, team_d, stats_and_metrics, num_simulations=1000):
    team_ratings = stats_and_metrics.set_index("Team")["Rating"].to_dict()
    results = defaultdict(int)

    for _ in range(num_simulations):
        winner = simulate_double_elimination([team_a, team_d, team_b, team_c], team_ratings, True)
        results[winner] += 1

    return defaultdict(float, {
        team: round(wins / num_simulations, 3) for team, wins in results.items()
    })

def select_super_regional_teams(regional_results):
    matchups = [(0, 15), (1, 14), (2, 13), (3, 12),
                (4, 11), (5, 10), (6, 9), (7, 8)]

    return [
        (
            random.choices(list(regional_results[i]), weights=regional_results[i].values())[0],
            random.choices(list(regional_results[j]), weights=regional_results[j].values())[0],
            i
        )
        for i, j in matchups
    ]

def simulate_super_regional(team1, team2, team_ratings):
    return simulate_best_of_three(team1, team2, team_ratings[team1], team_ratings[team2])

def run_super_regionals(regional_results, stats_and_metrics, actual_tournament, num_simulations=1000):
    team_ratings = stats_and_metrics.set_index("Team")["Rating"].to_dict()
    results = [defaultdict(int) for _ in range(8)]

    for _ in range(num_simulations):
        matchups = select_super_regional_teams(regional_results)
        for team1, team2, region_index in matchups:
            winner = simulate_super_regional(team1, team2, team_ratings)
            results[region_index][winner] += 1

    for i in range(8):
        total = sum(results[i].values())
        for team in results[i]:
            results[i][team] = round(results[i][team] / total, 3)

    return results

def simulate_super_regional_with_hosts(team1, team2, seed_map, region_map, team_ratings):
    seed_priority = {"Host": 1, "1 Seed": 1, "2 Seed": 2, "3 Seed": 3, "4 Seed": 4}

    s1, s2 = seed_map.get(team1, ""), seed_map.get(team2, "")
    r1, r2 = region_map.get(team1, float('inf')), region_map.get(team2, float('inf'))

    prio1, prio2 = seed_priority.get(s1, float('inf')), seed_priority.get(s2, float('inf'))
    rating1, rating2 = team_ratings[team1], team_ratings[team2]

    if prio1 == prio2:
        if r1 < r2:
            rating1 += 0.3
        else:
            rating2 += 0.3
    elif prio1 < prio2:
        rating1 += 0.3
    else:
        rating2 += 0.3

    return simulate_best_of_three(team1, team2, rating1, rating2)

def run_college_world_series(make_omaha, stats_and_metrics, num_simulations=1000):
    team_ratings = stats_and_metrics.set_index("Team")["Rating"].to_dict()
    results = [defaultdict(int), defaultdict(int)]

    for _ in range(num_simulations):
        group1 = [select_team(make_omaha[i]) for i in [0, 7, 3, 4]]
        group2 = [select_team(make_omaha[i]) for i in [1, 6, 2, 5]]

        winner1 = simulate_double_elimination(group1, team_ratings)
        winner2 = simulate_double_elimination(group2, team_ratings)

        results[0][winner1] += 1
        results[1][winner2] += 1

    for i in range(2):
        total = sum(results[i].values())
        for team in results[i]:
            results[i][team] = round(results[i][team] / total, 3)

    return results

def simulate_finals(make_finals, team_ratings):
    team1 = random.choices(list(make_finals[0]), weights=make_finals[0].values())[0]
    team2 = random.choices(list(make_finals[1]), weights=make_finals[1].values())[0]
    return simulate_best_of_three(team1, team2, team_ratings[team1], team_ratings[team2])

def run_finals_simulation(make_finals, stats_and_metrics, num_simulations=1000):
    team_ratings = stats_and_metrics.set_index("Team")["Rating"].to_dict()
    results = defaultdict(int)

    for _ in range(num_simulations):
        champion = simulate_finals(make_finals, team_ratings)
        results[champion] += 1

    total = sum(results.values())
    return {team: round(wins / total, 3) for team, wins in results.items()}

def generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals):
    teams = {team for group in regionals_results + make_omaha + make_finals for team in group}
    teams |= win_finals.keys()

    data = {team: {"Supers": 0, "Omaha": 0, "Finals": 0, "Win NC": 0} for team in teams}

    for group in regionals_results:
        for team, prob in group.items():
            data[team]["Supers"] = prob

    for group in make_omaha:
        for team, prob in group.items():
            data[team]["Omaha"] = prob

    for group in make_finals:
        for team, prob in group.items():
            data[team]["Finals"] = prob

    for team, prob in win_finals.items():
        data[team]["Win NC"] = prob

    return pd.DataFrame.from_dict(data, orient='index')

def simulate_from_known_regional_winners(regional_winners, stats_and_metrics, actual_tournament, num_simulations=1000):
    seed_map, region_map, team_ratings = preprocess_team_metadata(stats_and_metrics, actual_tournament)

    super_regionals = [(0, 15), (1, 14), (2, 13), (3, 12),
                       (4, 11), (5, 10), (6, 9), (7, 8)]
    super_results = [defaultdict(int) for _ in range(8)]

    for _ in range(num_simulations):
        for i, (r1, r2) in enumerate(super_regionals):
            t1, t2 = regional_winners[r1], regional_winners[r2]
            winner = simulate_super_regional_with_hosts(t1, t2, seed_map, region_map, team_ratings)
            super_results[i][winner] += 1

    for i in range(8):
        total = sum(super_results[i].values())
        for team in super_results[i]:
            super_results[i][team] = round(super_results[i][team] / total, 3)

    make_finals = run_college_world_series(super_results, stats_and_metrics, num_simulations)
    win_finals = run_finals_simulation(make_finals, stats_and_metrics, num_simulations)

    df = generate_simulation_dataframe(super_results, super_results, make_finals, win_finals)
    return df.sort_values("Win NC", ascending=False).reset_index().rename(columns={"index": "Team"})

def simulate_full_tournament(formatted_df, stats_and_metrics, iter):
    regionals_results = []
    for i in range(len(formatted_df)):
        regionals_results.append(run_simulation(formatted_df.iloc[i,1], formatted_df.iloc[i,2], formatted_df.iloc[i,3], formatted_df.iloc[i,4], stats_and_metrics, iter))
    make_omaha = run_super_regionals(regionals_results, stats_and_metrics, formatted_df, iter)
    make_finals = run_college_world_series(make_omaha, stats_and_metrics, iter)
    win_finals = run_finals_simulation(make_finals, stats_and_metrics, iter)
    simulation_df = generate_simulation_dataframe(regionals_results, make_omaha, make_finals, win_finals)
    simulation_df = simulation_df.sort_values('Win NC', ascending=False).reset_index()
    simulation_df.rename(columns={'index': 'Team'}, inplace=True)
    return simulation_df