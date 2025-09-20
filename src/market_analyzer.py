"""
Advanced Market Analysis for Tennis Betting
Detects market inefficiencies and identifies profitable upset opportunities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import math

class TennisMarketAnalyzer:
    """
    Advanced market analysis system to beat tennis betting markets
    Focuses on finding mispriced matches and upset opportunities
    """
    
    def __init__(self):
        # Market inefficiency thresholds
        self.min_edge_threshold = 0.05  # 5% minimum edge
        self.high_value_threshold = 0.15  # 15% edge = high value
        
        # Kelly criterion safety factors
        self.kelly_fraction = 0.25  # Conservative fractional Kelly
        self.max_bet_size = 0.08   # Never risk more than 8% of bankroll
        self.min_bet_size = 0.01   # Minimum meaningful bet size
        
        # Market behavior patterns
        self.public_bias_factors = {
            'big_name_premium': 0.08,      # Public overvalues stars
            'recent_success_bias': 0.06,   # Recency bias in betting
            'surface_ignorance': 0.04,     # Public ignores surface specialization
            'fatigue_blindness': 0.05      # Public doesn't factor fatigue
        }
        
    def convert_odds_to_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability"""
        if odds <= 1.0:
            return 0.99  # Handle edge case
        return 1.0 / odds
    
    def convert_probability_to_odds(self, probability: float) -> float:
        """Convert probability to decimal odds"""
        if probability <= 0.01:
            return 100.0  # Handle edge case
        return 1.0 / probability
    
    def calculate_market_overround(self, odds_dict: Dict[str, float]) -> float:
        """Calculate bookmaker's overround (profit margin)"""
        total_implied_prob = sum(self.convert_odds_to_probability(odds) 
                                for odds in odds_dict.values())
        return total_implied_prob - 1.0
    
    def detect_line_movement_patterns(self,
                                    opening_odds: Dict[str, float],
                                    current_odds: Dict[str, float],
                                    betting_volume: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Analyze betting line movement to detect sharp vs public money"""
        
        patterns = {}
        
        # Calculate movement in implied probabilities
        opening_prob_favorite = self.convert_odds_to_probability(opening_odds.get('favorite', 1.8))
        current_prob_favorite = self.convert_odds_to_probability(current_odds.get('favorite', 1.8))
        
        opening_prob_underdog = self.convert_odds_to_probability(opening_odds.get('underdog', 2.2))
        current_prob_underdog = self.convert_odds_to_probability(current_odds.get('underdog', 2.2))
        
        # Line movement magnitude
        favorite_movement = current_prob_favorite - opening_prob_favorite
        underdog_movement = current_prob_underdog - opening_prob_underdog
        
        patterns['favorite_line_movement'] = favorite_movement
        patterns['underdog_line_movement'] = underdog_movement
        patterns['total_movement'] = abs(favorite_movement) + abs(underdog_movement)
        
        # Detect reverse line movement (sharp money indicator)
        if abs(favorite_movement) > 0.05:  # Significant movement
            if favorite_movement < -0.03:  # Favorite odds lengthened
                patterns['sharp_money_on_underdog'] = abs(favorite_movement)
            elif favorite_movement > 0.03:  # Favorite odds shortened
                patterns['public_money_on_favorite'] = favorite_movement
        
        # Steam move detection (rapid movement)
        if patterns['total_movement'] > 0.08:  # 8%+ total movement
            patterns['steam_move_detected'] = True
            patterns['steam_direction'] = 'underdog' if underdog_movement > 0 else 'favorite'
        
        return patterns
    
    def analyze_public_betting_bias(self,
                                  public_betting_percentages: Dict[str, float],
                                  player_rankings: Dict[str, int],
                                  player_names: Dict[str, str]) -> Dict[str, float]:
        """Detect public betting biases that create opportunities"""
        
        bias_analysis = {}
        
        favorite_public_pct = public_betting_percentages.get('favorite', 50)
        underdog_public_pct = 100 - favorite_public_pct
        
        # Detect heavy public bias (>70% on one side)
        if favorite_public_pct > 70:
            bias_analysis['public_overload_favorite'] = (favorite_public_pct - 70) / 30
            bias_analysis['contrarian_value'] = bias_analysis['public_overload_favorite'] * 0.08
        
        # Big name bias - check if favorite is a "household name"
        favorite_ranking = player_rankings.get('favorite', 50)
        underdog_ranking = player_rankings.get('underdog', 100)
        
        # If favorite is top 10 and public betting is >75%, potential bias
        if favorite_ranking <= 10 and favorite_public_pct > 75:
            bias_analysis['big_name_bias'] = 0.06
            
        # Ranking gap vs betting gap analysis
        ranking_gap = underdog_ranking - favorite_ranking
        betting_gap = favorite_public_pct - 50
        
        if ranking_gap < 20 and betting_gap > 25:  # Close in ranking but public heavily on favorite
            bias_analysis['ranking_betting_mismatch'] = 0.05
        
        return bias_analysis
    
    def calculate_expected_value(self,
                               model_probability: float,
                               market_odds: float,
                               confidence_factor: float = 1.0) -> float:
        """Calculate expected value of a bet"""
        
        # EV = (probability × odds) - 1
        raw_ev = (model_probability * market_odds) - 1
        
        # Adjust for confidence
        adjusted_ev = raw_ev * confidence_factor
        
        return adjusted_ev
    
    def find_arbitrage_opportunities(self,
                                   bookmaker_odds: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Find pure arbitrage opportunities across bookmakers"""
        
        arbitrage_ops = []
        
        if len(bookmaker_odds) < 2:
            return arbitrage_ops
        
        bookmakers = list(bookmaker_odds.keys())
        
        # Check all combinations of bookmakers
        for i in range(len(bookmakers)):
            for j in range(i + 1, len(bookmakers)):
                book1 = bookmakers[i]
                book2 = bookmakers[j]
                
                # Get best odds for each outcome
                book1_fav = bookmaker_odds[book1].get('favorite', 1.5)
                book1_dog = bookmaker_odds[book1].get('underdog', 2.5)
                book2_fav = bookmaker_odds[book2].get('favorite', 1.5) 
                book2_dog = bookmaker_odds[book2].get('underdog', 2.5)
                
                # Find best odds for each outcome
                best_fav_odds = max(book1_fav, book2_fav)
                best_dog_odds = max(book1_dog, book2_dog)
                
                # Calculate arbitrage percentage
                arb_percentage = (1/best_fav_odds) + (1/best_dog_odds)
                
                if arb_percentage < 0.98:  # Profitable arbitrage (less than 98%)
                    profit_margin = (1 - arb_percentage) * 100
                    
                    arbitrage_ops.append({
                        'profit_margin': profit_margin,
                        'favorite_bet': {
                            'bookmaker': book1 if book1_fav > book2_fav else book2,
                            'odds': best_fav_odds,
                            'stake_percentage': (1/best_fav_odds) / arb_percentage
                        },
                        'underdog_bet': {
                            'bookmaker': book1 if book1_dog > book2_dog else book2,
                            'odds': best_dog_odds,
                            'stake_percentage': (1/best_dog_odds) / arb_percentage
                        }
                    })
        
        return sorted(arbitrage_ops, key=lambda x: x['profit_margin'], reverse=True)
    
    def comprehensive_match_analysis(self,
                                   match_data: Dict,
                                   model_prediction: Dict) -> Dict:
        """Comprehensive analysis combining model predictions with market analysis"""
        
        analysis = {
            'match_info': {
                'player_a': match_data.get('player_a', 'Unknown'),
                'player_b': match_data.get('player_b', 'Unknown'),
                'surface': match_data.get('surface', 'hard'),
                'tournament': match_data.get('tournament', 'Unknown'),
                'round': match_data.get('round', 'R1')
            }
        }
        
        # Model predictions
        analysis['model_prediction'] = model_prediction
        
        # Market analysis
        market_odds = match_data.get('market_odds', {'favorite': 1.6, 'underdog': 2.4})
        analysis['market_odds'] = market_odds
        
        # Calculate market implied probabilities
        market_prob_favorite = self.convert_odds_to_probability(market_odds['favorite'])
        market_prob_underdog = self.convert_odds_to_probability(market_odds['underdog'])
        
        analysis['market_probabilities'] = {
            'favorite': market_prob_favorite,
            'underdog': market_prob_underdog,
            'overround': market_prob_favorite + market_prob_underdog - 1.0
        }
        
        # Edge calculation
        model_upset_prob = model_prediction['ensemble_upset_probability']
        edge_on_underdog = model_upset_prob - market_prob_underdog
        edge_on_favorite = (1 - model_upset_prob) - market_prob_favorite
        
        analysis['betting_edges'] = {
            'underdog_edge': edge_on_underdog,
            'favorite_edge': edge_on_favorite,
            'max_edge': max(edge_on_underdog, edge_on_favorite),
            'best_bet': 'underdog' if edge_on_underdog > edge_on_favorite else 'favorite'
        }
        
        # Expected values
        underdog_ev = self.calculate_expected_value(model_upset_prob, market_odds['underdog'])
        favorite_ev = self.calculate_expected_value(1 - model_upset_prob, market_odds['favorite'])
        
        analysis['expected_values'] = {
            'underdog_ev': underdog_ev * 100,  # Convert to percentage
            'favorite_ev': favorite_ev * 100
        }
        
        # Kelly bet sizing
        confidence = model_prediction.get('confidence_score', 0.7)
        
        if edge_on_underdog > self.min_edge_threshold:
            kelly_underdog = self.calculate_kelly_betting_size(
                model_upset_prob, market_odds['underdog'], confidence
            )
        else:
            kelly_underdog = 0
            
        if edge_on_favorite > self.min_edge_threshold:
            kelly_favorite = self.calculate_kelly_betting_size(
                1 - model_upset_prob, market_odds['favorite'], confidence
            )
        else:
            kelly_favorite = 0
            
        analysis['kelly_sizing'] = {
            'underdog_kelly_pct': kelly_underdog * 100,
            'favorite_kelly_pct': kelly_favorite * 100,
            'recommended_bet': 'underdog' if kelly_underdog > kelly_favorite and kelly_underdog > 0.01
                              else 'favorite' if kelly_favorite > 0.01
                              else 'no_bet'
        }
        
        # Opportunity classification
        max_edge = analysis['betting_edges']['max_edge']
        if max_edge > 0.15:
            opportunity_level = 'HIGH_VALUE'
        elif max_edge > 0.08:
            opportunity_level = 'MODERATE_VALUE'
        elif max_edge > 0.03:
            opportunity_level = 'LOW_VALUE'
        else:
            opportunity_level = 'NO_EDGE'
            
        analysis['opportunity_assessment'] = {
            'level': opportunity_level,
            'edge_percentage': max_edge * 100,
            'confidence_score': confidence * 100,
            'upset_potential': model_upset_prob * 100
        }
        
        return analysis
    
    def calculate_kelly_betting_size(self,
                                   win_probability: float,
                                   market_odds: float,
                                   confidence: float) -> float:
        """Calculate optimal bet size using Kelly criterion"""
        
        # Kelly formula: f = (bp - q) / b
        b = market_odds - 1  # Net odds
        p = win_probability
        q = 1 - p
        
        if b <= 0 or p <= 0:
            return 0
            
        # Raw Kelly percentage
        kelly_pct = (b * p - q) / b
        
        # Apply safety factors
        adjusted_kelly = kelly_pct * self.kelly_fraction * confidence
        
        # Apply limits
        return max(self.min_bet_size, min(adjusted_kelly, self.max_bet_size))
    
    def scan_daily_opportunities(self,
                               daily_matches: List[Dict],
                               model_predictions: List[Dict],
                               min_opportunity_score: float = 0.6) -> pd.DataFrame:
        """Scan all daily matches for betting opportunities"""
        
        opportunities = []
        
        for i, (match, prediction) in enumerate(zip(daily_matches, model_predictions)):
            # Comprehensive analysis
            analysis = self.comprehensive_match_analysis(match, prediction)
            
            # Calculate opportunity score
            edge = analysis['betting_edges']['max_edge']
            confidence = prediction.get('confidence_score', 0.7)
            upset_prob = prediction['ensemble_upset_probability']
            
            opportunity_score = (
                edge * 0.4 +                    # Edge is most important
                confidence * 0.3 +              # Confidence matters
                min(upset_prob, 0.5) * 0.2 +    # Upset potential (capped)
                (1 if edge > 0.1 else 0) * 0.1  # Bonus for high edges
            )
            
            if opportunity_score >= min_opportunity_score:
                opportunity = {
                    'match_id': f"match_{i+1}",
                    'player_a': match.get('player_a', 'Player A'),
                    'player_b': match.get('player_b', 'Player B'), 
                    'surface': match.get('surface', 'hard'),
                    'tournament': match.get('tournament', 'Unknown'),
                    'opportunity_score': opportunity_score,
                    'edge_percentage': edge * 100,
                    'upset_probability': upset_prob * 100,
                    'confidence': confidence * 100,
                    'recommended_bet': analysis['kelly_sizing']['recommended_bet'],
                    'kelly_size': max(
                        analysis['kelly_sizing']['underdog_kelly_pct'],
                        analysis['kelly_sizing']['favorite_kelly_pct']
                    ),
                    'expected_value': max(
                        analysis['expected_values']['underdog_ev'],
                        analysis['expected_values']['favorite_ev']
                    ),
                    'favorite_odds': match.get('market_odds', {}).get('favorite', 1.6),
                    'underdog_odds': match.get('market_odds', {}).get('underdog', 2.4)
                }
                
                opportunities.append(opportunity)
        
        # Convert to DataFrame and sort by opportunity score
        df = pd.DataFrame(opportunities)
        if not df.empty:
            df = df.sort_values('opportunity_score', ascending=False)
            
        return df
    
    def generate_betting_alerts(self,
                              opportunities_df: pd.DataFrame,
                              alert_threshold: float = 0.75) -> List[Dict]:
        """Generate high-priority betting alerts"""
        
        alerts = []
        
        if opportunities_df.empty:
            return alerts
            
        # Filter for high-value opportunities
        high_value = opportunities_df[opportunities_df['opportunity_score'] >= alert_threshold]
        
        for _, row in high_value.iterrows():
            alert = {
                'alert_type': 'UPSET_OPPORTUNITY' if row['upset_probability'] > 30 else 'VALUE_BET',
                'match': f"{row['player_a']} vs {row['player_b']}",
                'surface': row['surface'],
                'tournament': row['tournament'],
                'recommended_bet': row['recommended_bet'],
                'kelly_size': f"{row['kelly_size']:.1f}%",
                'expected_value': f"{row['expected_value']:+.1f}%",
                'edge': f"{row['edge_percentage']:.1f}%",
                'confidence': f"{row['confidence']:.0f}%",
                'odds': row['underdog_odds'] if row['recommended_bet'] == 'underdog' else row['favorite_odds'],
                'priority': 'HIGH' if row['opportunity_score'] > 0.85 else 'MEDIUM'
            }
            
            alerts.append(alert)
            
        return alerts
    
    def backtest_strategy(self,
                        historical_predictions: pd.DataFrame,
                        starting_bankroll: float = 1000) -> Dict[str, float]:
        """Backtest the upset detection strategy"""
        
        bankroll = starting_bankroll
        total_bets = 0
        winning_bets = 0
        total_profit = 0
        max_drawdown = 0
        peak_bankroll = starting_bankroll
        
        bet_history = []
        
        for _, prediction in historical_predictions.iterrows():
            if prediction.get('recommended_bet', 'no_bet') == 'no_bet':
                continue
                
            # Calculate bet size
            kelly_size = prediction.get('kelly_size', 0) / 100
            bet_amount = bankroll * kelly_size
            
            if bet_amount < 1:  # Skip tiny bets
                continue
                
            total_bets += 1
            
            # Determine if bet won (simulate actual outcome)
            bet_on = prediction['recommended_bet']
            actual_result = prediction.get('actual_result', 'favorite')  # Default assumption
            
            won_bet = (bet_on == actual_result)
            
            if won_bet:
                winning_bets += 1
                odds = prediction.get('underdog_odds' if bet_on == 'underdog' else 'favorite_odds', 2.0)
                profit = bet_amount * (odds - 1)
                bankroll += profit
                total_profit += profit
            else:
                bankroll -= bet_amount
                total_profit -= bet_amount
                
            # Track drawdown
            peak_bankroll = max(peak_bankroll, bankroll)
            current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)
            
            bet_history.append({
                'bet_number': total_bets,
                'bankroll': bankroll,
                'profit': profit if won_bet else -bet_amount,
                'cumulative_profit': total_profit
            })
        
        # Calculate performance metrics
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        roi = (bankroll - starting_bankroll) / starting_bankroll
        
        return {
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'roi_percentage': roi * 100,
            'win_rate': win_rate * 100,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'max_drawdown': max_drawdown * 100,
            'profit_factor': abs(total_profit / min(total_profit, -1)) if total_profit != 0 else 0,
            'bet_history': bet_history
        }