#!/usr/bin/env python3
"""
NFL Data Collection System

Comprehensive NFL data collector using nflfastR, ESPN API, and other sources
for advanced football prediction modeling. Collects historical and real-time
data for training and live prediction.

Author: Milkpainter
Version: 1.0
Data Sources: nflfastR, ESPN API, Pro Football Reference, Weather APIs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import requests
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
except ImportError:
    print("Installing nfl_data_py...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nfl_data_py"])
    import nfl_data_py as nfl

class NFLDataCollector:
    """
    Comprehensive NFL data collection system.
    
    Collects and processes:
    - Historical game results and statistics
    - Team performance metrics
    - Player statistics and injuries
    - Weather conditions
    - Betting odds and market data
    - Advanced analytics (EPA, DVOA, etc.)
    """
    
    def __init__(self, cache_dir: str = "./cache/nfl/"):
        """
        Initialize the NFL data collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.team_mappings = self._get_team_mappings()
        
        # API endpoints
        self.espn_api = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.weather_api = "https://api.openweathermap.org/data/2.5"
        
        print(f"🏈 NFL Data Collector initialized")
        print(f"Cache directory: {cache_dir}")
    
    def _get_team_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Get NFL team mappings for different data sources.
        
        Returns:
            Dictionary mapping team abbreviations to full names and info
        """
        teams = {
            'ARI': {'name': 'Arizona Cardinals', 'city': 'Arizona', 'conference': 'NFC', 'division': 'West'},
            'ATL': {'name': 'Atlanta Falcons', 'city': 'Atlanta', 'conference': 'NFC', 'division': 'South'},
            'BAL': {'name': 'Baltimore Ravens', 'city': 'Baltimore', 'conference': 'AFC', 'division': 'North'},
            'BUF': {'name': 'Buffalo Bills', 'city': 'Buffalo', 'conference': 'AFC', 'division': 'East'},
            'CAR': {'name': 'Carolina Panthers', 'city': 'Carolina', 'conference': 'NFC', 'division': 'South'},
            'CHI': {'name': 'Chicago Bears', 'city': 'Chicago', 'conference': 'NFC', 'division': 'North'},
            'CIN': {'name': 'Cincinnati Bengals', 'city': 'Cincinnati', 'conference': 'AFC', 'division': 'North'},
            'CLE': {'name': 'Cleveland Browns', 'city': 'Cleveland', 'conference': 'AFC', 'division': 'North'},
            'DAL': {'name': 'Dallas Cowboys', 'city': 'Dallas', 'conference': 'NFC', 'division': 'East'},
            'DEN': {'name': 'Denver Broncos', 'city': 'Denver', 'conference': 'AFC', 'division': 'West'},
            'DET': {'name': 'Detroit Lions', 'city': 'Detroit', 'conference': 'NFC', 'division': 'North'},
            'GB': {'name': 'Green Bay Packers', 'city': 'Green Bay', 'conference': 'NFC', 'division': 'North'},
            'HOU': {'name': 'Houston Texans', 'city': 'Houston', 'conference': 'AFC', 'division': 'South'},
            'IND': {'name': 'Indianapolis Colts', 'city': 'Indianapolis', 'conference': 'AFC', 'division': 'South'},
            'JAX': {'name': 'Jacksonville Jaguars', 'city': 'Jacksonville', 'conference': 'AFC', 'division': 'South'},
            'KC': {'name': 'Kansas City Chiefs', 'city': 'Kansas City', 'conference': 'AFC', 'division': 'West'},
            'LV': {'name': 'Las Vegas Raiders', 'city': 'Las Vegas', 'conference': 'AFC', 'division': 'West'},
            'LAC': {'name': 'Los Angeles Chargers', 'city': 'Los Angeles', 'conference': 'AFC', 'division': 'West'},
            'LAR': {'name': 'Los Angeles Rams', 'city': 'Los Angeles', 'conference': 'NFC', 'division': 'West'},
            'MIA': {'name': 'Miami Dolphins', 'city': 'Miami', 'conference': 'AFC', 'division': 'East'},
            'MIN': {'name': 'Minnesota Vikings', 'city': 'Minnesota', 'conference': 'NFC', 'division': 'North'},
            'NE': {'name': 'New England Patriots', 'city': 'New England', 'conference': 'AFC', 'division': 'East'},
            'NO': {'name': 'New Orleans Saints', 'city': 'New Orleans', 'conference': 'NFC', 'division': 'South'},
            'NYG': {'name': 'New York Giants', 'city': 'New York', 'conference': 'NFC', 'division': 'East'},
            'NYJ': {'name': 'New York Jets', 'city': 'New York', 'conference': 'AFC', 'division': 'East'},
            'PHI': {'name': 'Philadelphia Eagles', 'city': 'Philadelphia', 'conference': 'NFC', 'division': 'East'},
            'PIT': {'name': 'Pittsburgh Steelers', 'city': 'Pittsburgh', 'conference': 'AFC', 'division': 'North'},
            'SF': {'name': 'San Francisco 49ers', 'city': 'San Francisco', 'conference': 'NFC', 'division': 'West'},
            'SEA': {'name': 'Seattle Seahawks', 'city': 'Seattle', 'conference': 'NFC', 'division': 'West'},
            'TB': {'name': 'Tampa Bay Buccaneers', 'city': 'Tampa Bay', 'conference': 'NFC', 'division': 'South'},
            'TEN': {'name': 'Tennessee Titans', 'city': 'Tennessee', 'conference': 'AFC', 'division': 'South'},
            'WAS': {'name': 'Washington Commanders', 'city': 'Washington', 'conference': 'NFC', 'division': 'East'}
        }
        return teams
    
    def collect_historical_games(self, seasons: List[int]) -> pd.DataFrame:
        """
        Collect historical NFL game data.
        
        Args:
            seasons: List of seasons to collect (e.g., [2020, 2021, 2022])
            
        Returns:
            DataFrame with historical game data
        """
        print(f"📅 Collecting historical NFL data for seasons: {seasons}")
        
        # Collect schedules and results
        schedules = []
        for season in seasons:
            print(f"  Downloading {season} season data...")
            try:
                season_schedule = nfl.import_schedules([season])
                schedules.append(season_schedule)
            except Exception as e:
                print(f"  Error collecting {season} data: {e}")
        
        if not schedules:
            return pd.DataFrame()
        
        # Combine all seasons
        games_df = pd.concat(schedules, ignore_index=True)
        
        print(f"Collected {len(games_df)} games")
        return games_df
    
    def collect_team_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Collect team-level statistics.
        
        Args:
            seasons: List of seasons to collect
            
        Returns:
            DataFrame with team statistics
        """
        print(f"📊 Collecting NFL team statistics for seasons: {seasons}")
        
        team_stats = []
        for season in seasons:
            try:
                # Get weekly team stats
                weekly_stats = nfl.import_weekly_data(seasons=[season], columns=[
                    'season', 'week', 'season_type', 'team', 'opponent_team',
                    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
                    'carries', 'rushing_yards', 'rushing_tds', 'fantasy_points'
                ])
                team_stats.append(weekly_stats)
            except Exception as e:
                print(f"  Error collecting {season} team stats: {e}")
        
        if not team_stats:
            return pd.DataFrame()
        
        team_df = pd.concat(team_stats, ignore_index=True)
        print(f"Collected team stats for {len(team_df)} team-weeks")
        return team_df
    
    def collect_advanced_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Collect advanced NFL statistics (EPA, etc.).
        
        Args:
            seasons: List of seasons to collect
            
        Returns:
            DataFrame with advanced statistics
        """
        print(f"🧬 Collecting advanced NFL statistics for seasons: {seasons}")
        
        try:
            # Get play-by-play data for EPA calculations
            pbp_data = nfl.import_pbp_data(seasons=seasons, columns=[
                'game_id', 'season', 'week', 'home_team', 'away_team',
                'posteam', 'defteam', 'play_type', 'yards_gained', 
                'epa', 'wpa', 'down', 'ydstogo', 'yardline_100'
            ])
            
            # Aggregate advanced stats by team and game
            team_game_stats = pbp_data.groupby(['game_id', 'posteam']).agg({
                'epa': ['mean', 'sum', 'count'],
                'wpa': ['mean', 'sum'],
                'yards_gained': ['sum', 'mean'],
                'play_type': 'count'
            }).round(3)
            
            # Flatten column names
            team_game_stats.columns = ['_'.join(col) for col in team_game_stats.columns]
            team_game_stats = team_game_stats.reset_index()
            
            print(f"Collected advanced stats for {len(team_game_stats)} team-games")
            return team_game_stats
            
        except Exception as e:
            print(f"Error collecting advanced stats: {e}")
            return pd.DataFrame()
    
    def collect_injuries(self, season: int, week: int) -> pd.DataFrame:
        """
        Collect injury reports for a specific week.
        
        Args:
            season: NFL season year
            week: Week number
            
        Returns:
            DataFrame with injury information
        """
        print(f"🎥 Collecting injury data for {season} Week {week}")
        
        try:
            injuries = nfl.import_injuries([season])
            week_injuries = injuries[
                (injuries['season'] == season) & 
                (injuries['week'] == week)
            ]
            
            print(f"Found {len(week_injuries)} injury reports")
            return week_injuries
            
        except Exception as e:
            print(f"Error collecting injury data: {e}")
            return pd.DataFrame()
    
    def get_current_week_games(self) -> pd.DataFrame:
        """
        Get current week's NFL games from ESPN API.
        
        Returns:
            DataFrame with current week games
        """
        print("🔴 Fetching current week NFL games...")
        
        try:
            # Get current season scoreboard
            url = f"{self.espn_api}/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                games = []
                
                for event in data.get('events', []):
                    game_info = {
                        'game_id': event['id'],
                        'date': event['date'],
                        'home_team': event['competitions'][0]['competitors'][0]['team']['abbreviation'],
                        'away_team': event['competitions'][0]['competitors'][1]['team']['abbreviation'],
                        'home_score': event['competitions'][0]['competitors'][0].get('score', 0),
                        'away_score': event['competitions'][0]['competitors'][1].get('score', 0),
                        'status': event['status']['type']['name'],
                        'week': data.get('week', {}).get('number', 1),
                        'season': data.get('season', {}).get('year', datetime.now().year)
                    }
                    games.append(game_info)
                
                games_df = pd.DataFrame(games)
                print(f"Found {len(games_df)} games for current week")
                return games_df
            
        except Exception as e:
            print(f"Error fetching current games: {e}")
            
        return pd.DataFrame()
    
    def collect_weather_data(self, games_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
        """
        Collect weather data for outdoor NFL games.
        
        Args:
            games_df: DataFrame with game information including venues
            api_key: OpenWeatherMap API key
            
        Returns:
            DataFrame with weather information added
        """
        if not api_key:
            print("No weather API key provided, skipping weather data")
            return games_df
        
        print("☁️ Collecting weather data for outdoor games...")
        
        # Stadium locations (lat, lon) for outdoor stadiums
        outdoor_stadiums = {
            'BUF': (42.7738, -78.7870),  # Highmark Stadium
            'CHI': (41.8623, -87.6167),  # Soldier Field  
            'CLE': (41.5061, -81.6995),  # Cleveland Browns Stadium
            'DEN': (39.7439, -105.0201), # Empower Field
            'GB': (44.5013, -88.0622),   # Lambeau Field
            'KC': (39.0489, -94.4839),   # Arrowhead Stadium
            'NE': (42.0909, -71.2643),   # Gillette Stadium
            'NYG': (40.8128, -74.0742),  # MetLife Stadium
            'NYJ': (40.8128, -74.0742),  # MetLife Stadium
            'PHI': (39.9008, -75.1675),  # Lincoln Financial Field
            'PIT': (40.4468, -80.0158),  # Heinz Field
            'SEA': (47.5952, -122.3316), # Lumen Field
            'TEN': (36.1665, -86.7713),  # Nissan Stadium
            'WAS': (38.9076, -76.8645),  # FedExField
        }
        
        weather_data = []
        
        for idx, game in games_df.iterrows():
            home_team = game.get('home_team')
            
            if home_team in outdoor_stadiums:
                lat, lon = outdoor_stadiums[home_team]
                
                try:
                    # Get weather forecast/historical data
                    url = f"{self.weather_api}/weather"
                    params = {
                        'lat': lat,
                        'lon': lon,
                        'appid': api_key,
                        'units': 'imperial'
                    }
                    
                    response = requests.get(url, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        weather = response.json()
                        
                        weather_info = {
                            'game_id': game.get('game_id'),
                            'temperature': weather['main']['temp'],
                            'humidity': weather['main']['humidity'],
                            'wind_speed': weather['wind']['speed'],
                            'wind_direction': weather['wind'].get('deg', 0),
                            'weather_condition': weather['weather'][0]['main'],
                            'precipitation': weather.get('rain', {}).get('1h', 0)
                        }
                        weather_data.append(weather_info)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error getting weather for {home_team}: {e}")
            
            else:
                # Indoor stadium
                weather_data.append({
                    'game_id': game.get('game_id'),
                    'temperature': 72,  # Controlled environment
                    'humidity': 50,
                    'wind_speed': 0,
                    'wind_direction': 0,
                    'weather_condition': 'Indoor',
                    'precipitation': 0
                })
        
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            games_with_weather = games_df.merge(weather_df, on='game_id', how='left')
            print(f"Added weather data for {len(weather_df)} games")
            return games_with_weather
        
        return games_df
    
    def create_team_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced team features for modeling.
        
        Args:
            games_df: Game results DataFrame
            team_stats_df: Team statistics DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("🧬 Creating advanced team features...")
        
        features_df = games_df.copy()
        
        # Calculate recent form (last 3 games)
        def calculate_recent_form(team_df, n_games=3):
            """
            Calculate recent form metrics for a team.
            """
            if len(team_df) < n_games:
                return {}
            
            recent_games = team_df.tail(n_games)
            
            return {
                'recent_wins': len(recent_games[recent_games['result'] == 'W']),
                'recent_points_avg': recent_games['points_scored'].mean(),
                'recent_points_allowed_avg': recent_games['points_allowed'].mean(),
                'recent_turnover_diff_avg': recent_games['turnover_differential'].mean()
            }
        
        # Add team performance features
        team_features = []
        
        for team in self.team_mappings.keys():
            team_games = games_df[
                (games_df['home_team'] == team) | (games_df['away_team'] == team)
            ].copy()
            
            if len(team_games) > 0:
                # Calculate cumulative stats
                team_games['points_scored'] = np.where(
                    team_games['home_team'] == team,
                    team_games['home_score'],
                    team_games['away_score']
                )
                
                team_games['points_allowed'] = np.where(
                    team_games['home_team'] == team,
                    team_games['away_score'],
                    team_games['home_score']
                )
                
                team_games['result'] = np.where(
                    team_games['points_scored'] > team_games['points_allowed'],
                    'W', 'L'
                )
                
                # Add rolling averages
                team_games['points_scored_avg'] = team_games['points_scored'].rolling(3, min_periods=1).mean()
                team_games['points_allowed_avg'] = team_games['points_allowed'].rolling(3, min_periods=1).mean()
                
                team_features.append(team_games)
        
        if team_features:
            enhanced_df = pd.concat(team_features, ignore_index=True)
            print(f"Created features for {len(enhanced_df)} team-games")
            return enhanced_df
        
        return features_df
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save collected data to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
        filepath = os.path.join(self.cache_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"✅ Saved {len(data)} records to {filepath}")
    
    def collect_full_dataset(self, seasons: List[int], weather_api_key: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect complete NFL dataset for modeling.
        
        Args:
            seasons: List of seasons to collect
            weather_api_key: Optional weather API key
            
        Returns:
            Dictionary of collected DataFrames
        """
        print(f"
🏈 COLLECTING COMPLETE NFL DATASET")
        print("=" * 50)
        
        dataset = {}
        
        # Collect historical games
        dataset['games'] = self.collect_historical_games(seasons)
        
        # Collect team statistics
        dataset['team_stats'] = self.collect_team_stats(seasons)
        
        # Collect advanced statistics
        dataset['advanced_stats'] = self.collect_advanced_stats(seasons)
        
        # Get current week games
        dataset['current_games'] = self.get_current_week_games()
        
        # Add weather data if API key provided
        if weather_api_key and not dataset['current_games'].empty:
            dataset['current_games'] = self.collect_weather_data(
                dataset['current_games'], weather_api_key
            )
        
        # Create enhanced features
        if not dataset['games'].empty and not dataset['team_stats'].empty:
            dataset['features'] = self.create_team_features(
                dataset['games'], dataset['team_stats']
            )
        
        # Save all datasets
        for name, df in dataset.items():
            if not df.empty:
                self.save_data(df, f"nfl_{name}.csv")
        
        print(f"\n✅ NFL data collection complete!")
        print(f"Collected {sum(len(df) for df in dataset.values())} total records")
        
        return dataset


if __name__ == "__main__":
    # Example usage
    print("🏈 NFL Data Collection Demo")
    print("=" * 40)
    
    # Initialize collector
    collector = NFLDataCollector()
    
    # Collect data for recent seasons
    seasons = [2021, 2022, 2023, 2024]
    
    # Note: Add your OpenWeatherMap API key here if you want weather data
    weather_api_key = None  # Replace with your API key
    
    # Collect complete dataset
    dataset = collector.collect_full_dataset(seasons, weather_api_key)
    
    # Display summary
    print("\n📊 DATASET SUMMARY:")
    for name, df in dataset.items():
        if not df.empty:
            print(f"{name}: {len(df):,} records, {len(df.columns)} columns")
            print(f"  Sample columns: {list(df.columns[:5])}")
        else:
            print(f"{name}: No data collected")
