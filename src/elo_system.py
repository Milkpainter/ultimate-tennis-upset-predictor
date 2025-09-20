"""
Advanced Tennis Elo Rating System
Features:
- Surface-specific ratings (Hard, Clay, Grass)
- Set-based adjustments
- Recent form weighting
- Tournament importance scaling
- Fatigue modeling
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class AdvancedTennisElo:
    """
    Advanced Tennis Elo Rating System for upset detection
    """
    
    def __init__(self):
        self.K_FACTOR_BASE = 32
        self.surfaces = ['hard', 'clay', 'grass']
        
        # Tournament importance multipliers
        self.tournament_weights = {
            'grand_slam': 1.4,
            'masters': 1.2, 
            'atp_500': 1.1,
            'atp_250': 1.0,
            'challenger': 0.8
        }
        
        # Surface transition penalties/bonuses
        self.surface_transitions = {
            ('clay', 'grass'): -50,    # Clay to grass penalty
            ('grass', 'clay'): -30,   # Grass to clay penalty  
            ('hard', 'clay'): -15,    # Hard to clay penalty
            ('clay', 'hard'): -10,    # Clay to hard penalty
            ('hard', 'grass'): -25,   # Hard to grass penalty
            ('grass', 'hard'): -20    # Grass to hard penalty
        }
        
    def calculate_k_factor(self, rating: float, matches_played: int, days_since_last: int) -> float:
        """Dynamic K-factor based on rating, experience, and activity"""
        # Base K-factor adjustment
        if rating < 1500:
            k = self.K_FACTOR_BASE * 1.5  # Newcomers move faster
        elif rating > 2000:
            k = self.K_FACTOR_BASE * 0.8  # Elite players more stable
        else:
            k = self.K_FACTOR_BASE
            
        # Experience adjustment
        if matches_played < 50:
            k *= 1.3  # Less experienced players adjust quicker
        elif matches_played > 200:
            k *= 0.9  # Very experienced players adjust slower
            
        # Activity adjustment - rust factor
        if days_since_last > 90:
            k *= 1.2  # Inactive players adjust quicker on return
            
        return k
    
    def expected_score(self, rating_a: float, rating_b: float, surface_advantage_a: float = 0) -> float:
        """Calculate expected score with surface advantages"""
        adjusted_rating_a = rating_a + surface_advantage_a
        return 1 / (1 + math.pow(10, (rating_b - adjusted_rating_a) / 400))
    
    def predict_match_probability(self,
                                player_a_ratings: Dict[str, float],
                                player_b_ratings: Dict[str, float], 
                                surface: str,
                                player_a_specialization: Dict[str, float],
                                player_b_specialization: Dict[str, float],
                                fatigue_a: float = 0,
                                fatigue_b: float = 0,
                                h2h_advantage_a: float = 0) -> float:
        """
        Predict match probability considering all factors
        Returns probability that player A wins
        """
        
        # Base ratings for the surface
        rating_a = player_a_ratings.get(surface, player_a_ratings.get('overall', 1500))
        rating_b = player_b_ratings.get(surface, player_b_ratings.get('overall', 1500))
        
        # Surface specialization adjustments
        surface_adj_a = player_a_specialization.get(surface, 0)
        surface_adj_b = player_b_specialization.get(surface, 0)
        
        # Fatigue adjustments (negative impact on rating)
        fatigue_adj_a = -fatigue_a * 20  # Each fatigue point = -20 rating
        fatigue_adj_b = -fatigue_b * 20
        
        # Final adjusted ratings
        adjusted_rating_a = rating_a + surface_adj_a + fatigue_adj_a + h2h_advantage_a
        adjusted_rating_b = rating_b + surface_adj_b + fatigue_adj_b
        
        # Calculate probability
        return self.expected_score(adjusted_rating_a, adjusted_rating_b)