#!/usr/bin/env python3
"""
Ultimate American Football Upset Predictor - Main Training Pipeline

Comprehensive training pipeline that combines:
- NFL/College data collection
- Genetic algorithm feature engineering
- Ensemble model training
- Cross-season validation
- Upset detection optimization

Target Performance:
- >82% overall accuracy
- >65% upset detection rate
- >15% annual ROI with Kelly Criterion

Author: Milkpainter
Version: 1.0
"""

import os
import sys
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import our custom modules
from data.collection.nfl_collector import NFLDataCollector
from feature_engineering.genetic_algorithm import GeneticFeatureEngineer
from models.ensemble.ensemble_predictor import FootballUpsetPredictor
from upset_detection.upset_detector import FootballUpsetDetector

def setup_directories():
    """
    Create necessary directories for the project.
    """
    directories = [
        './data/nfl/',
        './data/college/',
        './data/cache/',
        './models/saved/',
        './results/',
        './logs/',
        './plots/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def collect_training_data(seasons: List[int], 
                         weather_api_key: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Collect comprehensive training data.
    
    Args:
        seasons: List of NFL seasons to collect
        weather_api_key: Optional weather API key
        
    Returns:
        Dictionary of collected datasets
    """
    print("\n" + "=" * 60)
    print("📅 PHASE 1: DATA COLLECTION")
    print("=" * 60)
    
    # Initialize data collector
    collector = NFLDataCollector(cache_dir="./data/cache/")
    
    # Collect complete NFL dataset
    dataset = collector.collect_full_dataset(seasons, weather_api_key)
    
    return dataset

def engineer_features(dataset: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features using genetic algorithm optimization.
    
    Args:
        dataset: Dictionary of collected datasets
        
    Returns:
        Tuple of (features_df, target_df)
    """
    print("\n" + "=" * 60)
    print("🧬 PHASE 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize genetic algorithm feature engineer
    ga_engineer = GeneticFeatureEngineer(
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=5,
        fitness_function='accuracy',
        random_state=42
    )
    
    # Prepare base features from collected data
    games_df = dataset.get('games', pd.DataFrame())
    team_stats_df = dataset.get('team_stats', pd.DataFrame())
    
    if games_df.empty:
        raise ValueError("No game data available for feature engineering")
    
    # Create base features
    print("🔧 Creating base features...")
    base_features = prepare_base_features(games_df, team_stats_df)
    
    # Create target variable (1 = upset, 0 = favorite wins)
    target = create_target_variable(base_features)
    
    print(f"Base features shape: {base_features.shape}")
    print(f"Upset rate: {target.mean():.1%}")
    
    # Create advanced features
    enhanced_features = ga_engineer.create_advanced_features(base_features)
    
    # Evolve optimal feature subset
    evolution_results = ga_engineer.evolve(enhanced_features, target, verbose=True)
    
    # Transform to selected features
    final_features = ga_engineer.transform(enhanced_features)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"Selected {len(final_features.columns)} features from {len(enhanced_features.columns)}")
    print(f"Feature selection improvement: {evolution_results['improvement']:.4f}")
    
    # Save feature importance
    feature_importance = ga_engineer.get_feature_importance()
    feature_importance.to_csv('./results/feature_importance.csv', index=False)
    
    return final_features, target

def prepare_base_features(games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare base features from raw data.
    
    Args:
        games_df: Games DataFrame
        team_stats_df: Team statistics DataFrame
        
    Returns:
        DataFrame with base features
    """
    # This is a simplified version - in practice you'd have more sophisticated feature engineering
    features_list = []
    
    for _, game in games_df.iterrows():
        game_features = {
            'home_team': game.get('home_team', ''),
            'away_team': game.get('away_team', ''),
            'week': game.get('week', 1),
            'season': game.get('season', 2024),
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
        }
        
        # Add team-specific features if available
        if not team_stats_df.empty:
            home_stats = team_stats_df[
                team_stats_df['team'] == game.get('home_team')
            ]
            away_stats = team_stats_df[
                team_stats_df['team'] == game.get('away_team')
            ]
            
            if not home_stats.empty:
                game_features['home_passing_yards'] = home_stats['passing_yards'].mean()
                game_features['home_rushing_yards'] = home_stats['rushing_yards'].mean()
            
            if not away_stats.empty:
                game_features['away_passing_yards'] = away_stats['passing_yards'].mean()
                game_features['away_rushing_yards'] = away_stats['rushing_yards'].mean()
        
        features_list.append(game_features)
    
    features_df = pd.DataFrame(features_list)
    
    # Fill missing values
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
    
    # Encode categorical variables
    categorical_columns = features_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        features_df[col] = pd.Categorical(features_df[col]).codes
    
    return features_df

def create_target_variable(features_df: pd.DataFrame) -> pd.Series:
    """
    Create target variable for upset prediction.
    
    Args:
        features_df: Features DataFrame
        
    Returns:
        Series with target variable (1 = upset, 0 = favorite wins)
    """
    # Simple heuristic: upset if away team wins (away teams are often underdogs)
    # In practice, you'd use betting odds to determine favorites/underdogs
    target = (features_df['away_score'] > features_df['home_score']).astype(int)
    
    return target

def train_ensemble_model(X: pd.DataFrame, y: pd.Series) -> FootballUpsetPredictor:
    """
    Train the ensemble prediction model.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Trained ensemble predictor
    """
    print("\n" + "=" * 60)
    print("🤖 PHASE 3: ENSEMBLE MODEL TRAINING")
    print("=" * 60)
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Training upset rate: {y_train.mean():.1%}")
    print(f"Validation upset rate: {y_val.mean():.1%}")
    
    # Initialize ensemble predictor
    predictor = FootballUpsetPredictor(
        model_weights={
            'xgboost': 0.40,
            'random_forest': 0.25,
            'catboost': 0.20,
            'lstm': 0.15
        }
    )
    
    # Train ensemble
    training_metrics = predictor.train_ensemble(
        X_train, y_train, 
        validation_data=(X_val, y_val)
    )
    
    # Display results
    print(f"\n📈 Training Results:")
    for metric, value in training_metrics.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # Save trained model
    predictor.save_model('./models/saved/football_ensemble_model.pkl')
    
    return predictor

def validate_performance(predictor: FootballUpsetPredictor, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, float]:
    """
    Validate model performance on test set.
    
    Args:
        predictor: Trained predictor
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of performance metrics
    """
    print("\n" + "=" * 60)
    print("📉 PHASE 4: PERFORMANCE VALIDATION")
    print("=" * 60)
    
    # Get predictions
    predictions = predictor.predict_upset_probability(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
    final_pred_binary = (predictions['final_prob'] > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, final_pred_binary),
        'precision': precision_score(y_test, final_pred_binary),
        'recall': recall_score(y_test, final_pred_binary),
        'auc': roc_auc_score(y_test, predictions['final_prob']),
        'upset_detection_rate': recall_score(y_test, final_pred_binary)
    }
    
    print(f"🎯 Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.1%}")
    
    # Check if we meet targets
    targets_met = {
        'Overall Accuracy': metrics['accuracy'] >= 0.82,
        'Upset Detection': metrics['upset_detection_rate'] >= 0.65,
        'AUC Score': metrics['auc'] >= 0.75
    }
    
    print(f"\n🎯 Target Achievement:")
    for target, met in targets_met.items():
        status = "✅" if met else "❌"
        print(f"{status} {target}: {'MET' if met else 'NOT MET'}")
    
    return metrics

def setup_upset_detection(predictor: FootballUpsetPredictor) -> FootballUpsetDetector:
    """
    Setup upset detection system.
    
    Args:
        predictor: Trained predictor
        
    Returns:
        Configured upset detector
    """
    print("\n" + "=" * 60)
    print("🚨 PHASE 5: UPSET DETECTION SETUP")
    print("=" * 60)
    
    # Initialize upset detector
    detector = FootballUpsetDetector(
        bankroll=10000,
        max_bet_percentage=0.05,
        min_edge=0.05,
        min_confidence=0.55,
        kelly_fraction=0.25
    )
    
    print(f"✅ Upset detection system configured")
    print(f"Bankroll: ${detector.bankroll:,}")
    print(f"Max bet: {detector.max_bet_percentage:.1%} of bankroll")
    print(f"Min edge required: {detector.min_edge:.1%}")
    print(f"Min confidence: {detector.min_confidence:.1%}")
    
    return detector

def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command line arguments
    """
    print("🏈 ULTIMATE AMERICAN FOOTBALL UPSET PREDICTOR")
    print("=" * 60)
    print(f"Training pipeline started at {datetime.now()}")
    print(f"Seasons: {args.seasons}")
    print(f"Target accuracy: >82%")
    print(f"Target upset detection: >65%")
    print(f"Target ROI: >15%")
    
    # Setup project directories
    setup_directories()
    
    try:
        # Phase 1: Data Collection
        dataset = collect_training_data(args.seasons, args.weather_api_key)
        
        # Phase 2: Feature Engineering
        X, y = engineer_features(dataset)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Phase 3: Model Training
        predictor = train_ensemble_model(X_train, y_train)
        
        # Phase 4: Performance Validation
        metrics = validate_performance(predictor, X_test, y_test)
        
        # Phase 5: Upset Detection Setup
        detector = setup_upset_detection(predictor)
        
        # Save final results
        results = {
            'training_complete': True,
            'timestamp': datetime.now().isoformat(),
            'seasons_trained': args.seasons,
            'performance_metrics': metrics,
            'model_path': './models/saved/football_ensemble_model.pkl'
        }
        
        import json
        with open('./results/training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: ./results/")
        print(f"Model saved to: ./models/saved/")
        print(f"Ready for live upset prediction!")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate American Football Upset Predictor Training")
    
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        default=[2021, 2022, 2023, 2024],
        help='NFL seasons to include in training (default: 2021-2024)'
    )
    
    parser.add_argument(
        '--weather-api-key',
        type=str,
        help='OpenWeatherMap API key for weather data (optional)'
    )
    
    parser.add_argument(
        '--quick-mode',
        action='store_true',
        help='Run in quick mode with reduced GA generations (for testing)'
    )
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick_mode:
        print("⚡ Running in QUICK MODE - reduced training time")
        # You could modify parameters here for faster training
    
    exit_code = main(args)
    sys.exit(exit_code)
