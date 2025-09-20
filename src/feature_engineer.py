"""
Advanced Feature Engineering for Tennis Upset Detection
Combines statistical, psychological, and market-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class TennisFeatureEngineer:
    """
    Advanced feature engineering for tennis upset detection
    Combines statistical, psychological, and market-based features
    """
    
    def __init__(self):
        self.momentum_decay = 0.95  # How quickly momentum decays
        self.pressure_weights = {
            'grand_slam': 1.5,
            'masters': 1.3,
            'atp_500': 1.1, 
            'atp_250': 1.0
        }
        
    def calculate_recent_form_momentum(self, 
                                     recent_matches: List[Dict],
                                     lookback_days: int = 90) -> Dict[str, float]:
        """Calculate momentum based on recent match results with decay"""
        if not recent_matches:
            return {'momentum_score': 0, 'win_rate': 0, 'quality_wins': 0}
            
        total_momentum = 0
        quality_wins = 0
        wins = 0
        total_weight = 0
        
        for i, match in enumerate(recent_matches):
            # Weight recent matches more heavily
            weight = self.momentum_decay ** i
            
            if match['result'] == 'W':
                wins += 1
                # Base momentum for win
                momentum_gain = 1.0 * weight
                
                # Bonus for quality wins (beating higher-rated opponents)
                if match.get('opponent_rating', 0) > match.get('player_rating', 0):
                    quality_bonus = min((match['opponent_rating'] - match['player_rating']) / 100, 0.5)
                    momentum_gain += quality_bonus
                    quality_wins += weight
                    
                # Tournament importance bonus
                tournament_bonus = self.pressure_weights.get(match.get('tournament_level', 'atp_250'), 1.0) - 1
                momentum_gain += tournament_bonus * 0.2
                
                total_momentum += momentum_gain
            else:
                # Penalty for losses, especially bad losses
                momentum_loss = -0.5 * weight
                if match.get('opponent_rating', 2000) < match.get('player_rating', 1500):
                    # Bad loss penalty
                    bad_loss_penalty = min((match['player_rating'] - match['opponent_rating']) / 100, 0.5)
                    momentum_loss -= bad_loss_penalty
                    
                total_momentum += momentum_loss
                
            total_weight += weight
            
        win_rate = wins / len(recent_matches) if recent_matches else 0
        momentum_score = total_momentum / total_weight if total_weight > 0 else 0
        
        return {
            'momentum_score': momentum_score,
            'win_rate': win_rate,
            'quality_wins': quality_wins
        }
    
    def calculate_fatigue_score(self,
                              recent_matches: List[Dict],
                              days_since_last_match: int) -> float:
        """Calculate accumulated fatigue from recent play"""
        if not recent_matches:
            return 0
            
        # Look at matches in last 30 days
        recent_matches_30d = [m for m in recent_matches if m.get('days_ago', 100) <= 30]
        
        fatigue_accumulation = 0
        
        for match in recent_matches_30d:
            days_ago = match.get('days_ago', 30)
            sets_played = match.get('sets_total', 3)
            match_duration = match.get('duration_minutes', 120)
            
            # Fatigue decays over time
            time_decay = max(0, (30 - days_ago) / 30)
            
            # Base fatigue from number of sets and duration
            match_fatigue = (sets_played - 2) * 0.1 + (match_duration - 90) / 600
            match_fatigue = max(0, match_fatigue)  # No negative fatigue
            
            fatigue_accumulation += match_fatigue * time_decay
            
        # Rest recovery bonus
        if days_since_last_match > 7:
            rest_bonus = min(days_since_last_match / 14, 0.3)
            fatigue_accumulation -= rest_bonus
            
        return max(0, fatigue_accumulation)  # Fatigue can't be negative
    
    def generate_comprehensive_features(self,
                                      player_a_data: Dict,
                                      player_b_data: Dict,
                                      match_context: Dict) -> Dict[str, float]:
        """Generate all features for upset detection"""
        
        features = {}
        
        # Feature 1: Recent Form Momentum
        momentum_a = self.calculate_recent_form_momentum(player_a_data.get('recent_matches', []))
        momentum_b = self.calculate_recent_form_momentum(player_b_data.get('recent_matches', []))
        features['momentum_differential'] = momentum_a['momentum_score'] - momentum_b['momentum_score']
        features['win_rate_differential'] = momentum_a['win_rate'] - momentum_b['win_rate']
        
        # Feature 2: Fatigue
        fatigue_a = self.calculate_fatigue_score(
            player_a_data.get('recent_matches', []), 
            player_a_data.get('days_since_last_match', 7)
        )
        fatigue_b = self.calculate_fatigue_score(
            player_b_data.get('recent_matches', []),
            player_b_data.get('days_since_last_match', 7) 
        )
        features['fatigue_differential'] = fatigue_b - fatigue_a  # Higher when opponent more tired
        
        # Feature 3: Ranking/Elo Gap (baseline)
        features['ranking_gap'] = (player_b_data.get('ranking', 100) - 
                                 player_a_data.get('ranking', 50))
        features['elo_gap'] = (player_a_data.get('elo_rating', 1500) - 
                             player_b_data.get('elo_rating', 1500))
        
        # Feature 4: Surface and pressure features (placeholder)
        features['pressure_performance_gap'] = 0.0
        features['surface_adaptation_gap'] = 0.0
        features['h2h_psychological_advantage'] = 0.0
        features['market_overreaction_score'] = match_context.get('market_overreaction', 0.0)
        
        return features