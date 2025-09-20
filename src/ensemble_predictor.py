"""
Advanced ML Ensemble System for Tennis Upset Prediction
Specifically designed to beat betting markets and find undervalued underdogs
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TennisUpsetPredictor:
    """
    Advanced ensemble ML system for tennis upset prediction
    Combines multiple models with sophisticated feature engineering
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature importance weights (optimized for upset detection)
        self.feature_weights = {
            'elo_gap': 0.22,                        # Reduced from 0.25 - upsets defy rankings
            'momentum_differential': 0.18,          # Increased - hot underdogs beat cold favorites
            'fatigue_differential': 0.15,           # Increased - tired favorites vulnerable
            'pressure_performance_gap': 0.12,       # Big match experience crucial
            'surface_adaptation_gap': 0.10,         # Surface transitions create opportunities
            'h2h_psychological_advantage': 0.08,    # Mental edge matters
            'market_overreaction_score': 0.15,      # Key for finding value
        }
        
        # Initialize individual models - optimized for upset detection
        self.models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,              # More trees for complex patterns
                learning_rate=0.05,            # Slower learning for stability
                max_depth=8,                   # Deeper for upset patterns
                subsample=0.7,                 # Prevent overfitting
                random_state=random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,                  # Deep trees for upset patterns
                min_samples_split=3,           # Allow smaller splits
                min_samples_leaf=1,            # Sensitive to outliers
                class_weight='balanced',       # Balance for rare upsets
                random_state=random_state
            ),
            'logistic_regression': LogisticRegression(
                C=0.5,                         # More regularization
                penalty='l1',                  # Feature selection
                solver='liblinear',
                class_weight='balanced',
                random_state=random_state,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                C=2.0,                         # Allow more complex boundaries
                gamma='auto',
                probability=True,
                class_weight='balanced',
                random_state=random_state
            )
        }
        
        # Ensemble weights (optimized for upset detection)
        self.model_weights = {
            'gradient_boosting': 0.40,  # Primary model - best at complex patterns
            'random_forest': 0.25,      # Good at outlier detection
            'logistic_regression': 0.20, # Interpretable baseline
            'svm': 0.15                 # Non-linear pattern detection
        }
        
    def create_upset_focused_training_data(self, n_samples=3000):
        """Create training data specifically designed for upset detection"""
        np.random.seed(self.random_state)
        
        # Generate realistic feature distributions
        data = {}
        
        # ELO gap - include more underdog scenarios
        elo_gaps = np.concatenate([
            np.random.normal(-50, 80, int(n_samples * 0.3)),   # Underdog scenarios
            np.random.normal(0, 120, int(n_samples * 0.4)),     # Close matches  
            np.random.normal(75, 100, int(n_samples * 0.3))     # Favorite scenarios
        ])
        data['elo_gap'] = np.random.permutation(elo_gaps)
        
        # Momentum differential - underdogs often have better recent form
        momentum_diffs = np.concatenate([
            np.random.normal(0.2, 0.3, int(n_samples * 0.4)),   # Underdog hot streaks
            np.random.normal(0, 0.2, int(n_samples * 0.3)),     # Neutral momentum
            np.random.normal(-0.15, 0.25, int(n_samples * 0.3)) # Favorite momentum
        ])
        data['momentum_differential'] = np.random.permutation(momentum_diffs)
        
        # Fatigue differential - favorites often play more, get tired
        fatigue_diffs = np.random.gamma(2, 0.15, n_samples)  # Always positive bias
        data['fatigue_differential'] = fatigue_diffs
        
        # Pressure performance gap - some underdogs perform better under pressure
        data['pressure_performance_gap'] = np.random.normal(0, 0.4, n_samples)
        
        # Surface adaptation gap - transition periods create opportunities
        data['surface_adaptation_gap'] = np.random.normal(0, 0.35, n_samples)
        
        # H2H psychological advantage
        data['h2h_psychological_advantage'] = np.random.normal(0, 0.5, n_samples)
        
        # Market overreaction score - public often overvalues big names
        market_overreactions = np.random.exponential(0.12, n_samples)
        data['market_overreaction_score'] = market_overreactions
        
        # Create DataFrame
        X = pd.DataFrame(data)
        
        # Generate target with UPSET-FOCUSED logic
        # Base probability from Elo (but reduced impact for upsets)
        base_prob = 1 / (1 + np.exp(-X['elo_gap'] / 250))  # Less steep than normal
        
        # Strong adjustments for upset factors
        momentum_effect = X['momentum_differential'] * 0.4      # Strong momentum impact
        fatigue_effect = X['fatigue_differential'] * 0.3       # Fatigue very important
        pressure_effect = X['pressure_performance_gap'] * 0.2   # Pressure situations
        surface_effect = X['surface_adaptation_gap'] * 0.15    # Surface transitions
        h2h_effect = X['h2h_psychological_advantage'] * 0.1    # Mental factors
        market_effect = -X['market_overreaction_score'] * 0.25  # Fade market overreactions
        
        # Combined probability
        adjusted_prob = (base_prob + momentum_effect + fatigue_effect + 
                        pressure_effect + surface_effect + h2h_effect + market_effect)
        adjusted_prob = np.clip(adjusted_prob, 0.02, 0.98)  # Keep realistic bounds
        
        # Generate binary outcomes
        y = np.random.binomial(1, adjusted_prob, n_samples)
        
        return X, y
    
    def fit(self, X=None, y=None, use_upset_data=True):
        """Train the ensemble model with upset-focused data"""
        if X is None or y is None:
            if use_upset_data:
                print("🔥 Generating UPSET-FOCUSED training data...")
                X, y = self.create_upset_focused_training_data()
            else:
                raise ValueError("Must provide X and y or set use_upset_data=True")
        
        print(f"📊 Training on {len(X)} matches with {np.mean(y):.1%} upset rate")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models
        for name, model in self.models.items():
            print(f"  🤖 Training {name}...")
            model.fit(X_scaled, y)
            
        self.is_fitted = True
        print("✅ UPSET DETECTION ENSEMBLE READY!")
        
    def predict_proba(self, X):
        """Predict probabilities using ensemble approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (favorite wins)
            predictions[name] = pred_proba
            
        # Weighted ensemble
        ensemble_prob = np.zeros(len(X))
        for name, weight in self.model_weights.items():
            ensemble_prob += predictions[name] * weight
            
        return ensemble_prob
    
    def predict_upset_probability(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """Predict upset probability for a single match with detailed analysis"""
        # Convert features to DataFrame
        X = pd.DataFrame([features_dict])
        
        # Get ensemble probability
        prob_favorite_wins = self.predict_proba(X)[0]
        prob_upset = 1 - prob_favorite_wins
        
        # Individual model predictions for analysis
        X_scaled = self.scaler.transform(X)
        individual_predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_scaled)[0, 1]
            individual_predictions[name] = 1 - pred_proba  # Convert to upset probability
            
        # Calculate upset confidence factors
        upset_indicators = self._analyze_upset_indicators(features_dict)
        
        return {
            'ensemble_upset_probability': prob_upset,
            'favorite_win_probability': prob_favorite_wins,
            'model_breakdown': individual_predictions,
            'confidence_score': self._calculate_confidence(individual_predictions),
            'upset_indicators': upset_indicators,
            'market_edge_score': self._calculate_market_edge(prob_upset, features_dict)
        }
    
    def _analyze_upset_indicators(self, features: Dict[str, float]) -> Dict[str, bool]:
        """Analyze specific upset indicators"""
        indicators = {}
        
        # Strong momentum for underdog
        indicators['underdog_momentum'] = features.get('momentum_differential', 0) > 0.15
        
        # Significant fatigue differential
        indicators['favorite_fatigue'] = features.get('fatigue_differential', 0) > 0.4
        
        # Pressure performance advantage
        indicators['pressure_advantage'] = features.get('pressure_performance_gap', 0) > 0.1
        
        # Surface adaptation edge
        indicators['surface_edge'] = features.get('surface_adaptation_gap', 0) > 0.2
        
        # Market overreaction
        indicators['market_overreaction'] = features.get('market_overreaction_score', 0) > 0.15
        
        # H2H psychological edge
        indicators['psychological_edge'] = features.get('h2h_psychological_advantage', 0) > 0.1
        
        return indicators
    
    def _calculate_market_edge(self, upset_probability: float, features: Dict[str, float]) -> float:
        """Calculate potential edge over betting market"""
        # Estimate market implied probability from features
        # Markets typically overweight rankings and underweight momentum/fatigue
        
        elo_gap = features.get('elo_gap', 0)
        market_implied_upset_prob = 1 / (1 + np.exp(elo_gap / 180))  # Market logic
        
        # Our edge is the difference
        edge = upset_probability - market_implied_upset_prob
        return max(0, edge)  # Only positive edges matter
    
    def _calculate_confidence(self, individual_predictions: Dict[str, float]) -> float:
        """Calculate prediction confidence based on model agreement"""
        predictions_array = np.array(list(individual_predictions.values()))
        
        # Lower standard deviation = higher agreement = higher confidence
        std_dev = np.std(predictions_array)
        max_possible_std = 0.5
        
        confidence = 1 - (std_dev / max_possible_std)
        return confidence
    
    def find_upset_opportunities(self, 
                               match_predictions: List[Dict],
                               min_upset_probability: float = 0.35,
                               min_confidence: float = 0.65,
                               min_edge: float = 0.10) -> List[Dict]:
        """Filter matches to find high-value upset opportunities"""
        
        opportunities = []
        
        for prediction in match_predictions:
            upset_prob = prediction['ensemble_upset_probability']
            confidence = prediction['confidence_score'] 
            edge = prediction['market_edge_score']
            
            # Check if this is a high-value opportunity
            if (upset_prob >= min_upset_probability and 
                confidence >= min_confidence and 
                edge >= min_edge):
                
                # Count active upset indicators
                active_indicators = sum(prediction['upset_indicators'].values())
                
                prediction['opportunity_score'] = (
                    upset_prob * 0.4 +
                    confidence * 0.3 + 
                    edge * 0.2 +
                    (active_indicators / 6) * 0.1
                )
                
                opportunities.append(prediction)
        
        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)
    
    def calculate_kelly_betting_size(self,
                                   upset_probability: float,
                                   market_odds: float,
                                   confidence: float,
                                   kelly_fraction: float = 0.25) -> float:
        """Calculate optimal bet size using fractional Kelly criterion"""
        
        # Kelly formula: f = (bp - q) / b
        # f = fraction of bankroll, b = odds - 1, p = true probability, q = 1 - p
        
        b = market_odds - 1
        p = upset_probability
        q = 1 - p
        
        if b <= 0:
            return 0
            
        # Raw Kelly percentage
        kelly_pct = (b * p - q) / b
        
        # Apply fractional Kelly and confidence adjustment
        kelly_pct = kelly_pct * kelly_fraction * confidence
        
        # Risk limits
        return max(0, min(kelly_pct, 0.05))  # Max 5% of bankroll
    
    def generate_daily_predictions(self, matches_data: List[Dict]) -> pd.DataFrame:
        """Generate predictions for all matches with upset focus"""
        
        predictions = []
        
        for match in matches_data:
            # Extract features
            features = match['features']
            
            # Get prediction
            result = self.predict_upset_probability(features)
            
            # Add match context
            prediction = {
                'match_id': match.get('match_id', 'unknown'),
                'player_a': match.get('player_a', 'Player A'),
                'player_b': match.get('player_b', 'Player B'),
                'surface': match.get('surface', 'hard'),
                'tournament': match.get('tournament', 'Unknown'),
                'round': match.get('round', 'R1'),
                'favorite_odds': match.get('favorite_odds', 1.5),
                'underdog_odds': match.get('underdog_odds', 2.5),
                **result
            }
            
            # Calculate betting recommendation
            if result['ensemble_upset_probability'] > 0.3:
                kelly_size = self.calculate_kelly_betting_size(
                    result['ensemble_upset_probability'],
                    match.get('underdog_odds', 2.5),
                    result['confidence_score']
                )
                prediction['recommended_bet_size'] = kelly_size
            else:
                prediction['recommended_bet_size'] = 0
                
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)