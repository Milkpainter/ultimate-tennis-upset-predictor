#!/usr/bin/env python3
"""
Ultimate Football Upset Detection Engine

Advanced system for identifying high-value upset opportunities where
model predictions (>50%) differ from market expectations (<50%).

Includes Kelly Criterion position sizing, expected value calculations,
and comprehensive risk management for profitable upset betting.

Author: Milkpainter
Version: 1.0
Target: >65% upset detection rate, >15% annual ROI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UpsetOpportunity:
    """
    Data class for upset betting opportunities.
    """
    game_id: str
    underdog: str
    favorite: str
    model_prob: float
    market_prob: float
    edge: float
    expected_value: float
    kelly_size: float
    confidence: float
    game_date: datetime
    sport: str  # 'nfl' or 'college'
    features: Dict[str, float]
    
class FootballUpsetDetector:
    """
    Advanced upset detection system for American Football.
    
    Identifies games where:
    1. Model predicts underdog wins (>50% probability)
    2. Market disagrees (underdog <50% implied probability) 
    3. Expected value is positive
    4. Kelly Criterion suggests meaningful bet size
    
    Implements comprehensive risk management and bankroll optimization.
    """
    
    def __init__(self, 
                 bankroll: float = 10000,
                 max_bet_percentage: float = 0.05,
                 min_edge: float = 0.05,
                 min_confidence: float = 0.55,
                 kelly_fraction: float = 0.25):
        """
        Initialize the upset detection system.
        
        Args:
            bankroll: Starting bankroll for bet sizing
            max_bet_percentage: Maximum % of bankroll per bet (5% default)
            min_edge: Minimum edge required for bet consideration (5% default)
            min_confidence: Minimum model confidence for upset alert (55% default)
            kelly_fraction: Fraction of full Kelly bet size (25% default for safety)
        """
        self.bankroll = bankroll
        self.max_bet_percentage = max_bet_percentage
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.kelly_fraction = kelly_fraction
        
        # Tracking variables
        self.opportunities_found = []
        self.bets_placed = []
        self.performance_metrics = {
            'total_opportunities': 0,
            'high_confidence_opportunities': 0,
            'total_expected_value': 0.0,
            'average_edge': 0.0,
            'bankroll_allocated': 0.0
        }
    
    def calculate_implied_probability(self, odds: Union[float, int], 
                                   odds_format: str = 'american') -> float:
        """
        Calculate implied probability from betting odds.
        
        Args:
            odds: Betting odds
            odds_format: 'american', 'decimal', or 'fractional'
            
        Returns:
            Implied probability (0-1)
        """
        if odds_format == 'american':
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        elif odds_format == 'decimal':
            return 1 / odds
        elif odds_format == 'fractional':
            return 1 / (odds + 1)
        else:
            raise ValueError("odds_format must be 'american', 'decimal', or 'fractional'")
    
    def calculate_expected_value(self, model_prob: float, odds: float, 
                               odds_format: str = 'american') -> float:
        """
        Calculate expected value of a bet.
        
        Args:
            model_prob: Model's probability of underdog winning
            odds: Betting odds for underdog
            odds_format: Format of odds
            
        Returns:
            Expected value as a percentage
        """
        if odds_format == 'american':
            if odds > 0:
                payout = odds / 100
            else:
                payout = 100 / abs(odds)
        elif odds_format == 'decimal':
            payout = odds - 1
        else:
            raise ValueError("Unsupported odds format")
        
        # Expected value calculation
        ev = (model_prob * payout) - (1 - model_prob)
        return ev
    
    def kelly_criterion(self, model_prob: float, odds: float, 
                       odds_format: str = 'american') -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            model_prob: Model's probability of winning
            odds: Betting odds
            odds_format: Format of odds
            
        Returns:
            Optimal fraction of bankroll to bet
        """
        if odds_format == 'american':
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
        elif odds_format == 'decimal':
            decimal_odds = odds
        else:
            raise ValueError("Unsupported odds format")
        
        # Kelly formula: f = (bp - q) / b
        # where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
        b = decimal_odds - 1
        p = model_prob
        q = 1 - model_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety fraction and maximum bet constraints
        safe_kelly = kelly_fraction * self.kelly_fraction
        return min(safe_kelly, self.max_bet_percentage)
    
    def analyze_game(self, 
                    game_data: Dict,
                    model_predictions: Dict,
                    market_odds: Dict) -> Optional[UpsetOpportunity]:
        """
        Analyze a single game for upset opportunity.
        
        Args:
            game_data: Game information (teams, date, etc.)
            model_predictions: Model's predictions
            market_odds: Current market odds
            
        Returns:
            UpsetOpportunity if criteria met, None otherwise
        """
        # Extract key information
        underdog = game_data.get('underdog')
        favorite = game_data.get('favorite') 
        game_id = game_data.get('game_id')
        game_date = game_data.get('game_date')
        sport = game_data.get('sport', 'nfl')
        
        # Model predictions
        model_prob = model_predictions.get('underdog_win_prob', 0.0)
        
        # Market odds
        underdog_odds = market_odds.get('underdog_odds')
        odds_format = market_odds.get('format', 'american')
        
        if underdog_odds is None:
            return None
        
        # Calculate market implied probability
        market_prob = self.calculate_implied_probability(underdog_odds, odds_format)
        
        # Check upset criteria: model >50% but market <50%
        is_upset_opportunity = (model_prob > 0.50) and (market_prob < 0.50)
        
        if not is_upset_opportunity:
            return None
        
        # Calculate edge and expected value
        edge = model_prob - market_prob
        expected_value = self.calculate_expected_value(model_prob, underdog_odds, odds_format)
        
        # Apply filters
        if edge < self.min_edge or model_prob < self.min_confidence:
            return None
        
        if expected_value <= 0:
            return None
        
        # Calculate optimal bet size
        kelly_size = self.kelly_criterion(model_prob, underdog_odds, odds_format)
        
        # Calculate confidence score
        confidence = min(1.0, (edge * 2) + (expected_value * 0.5))
        
        # Create opportunity object
        opportunity = UpsetOpportunity(
            game_id=game_id,
            underdog=underdog,
            favorite=favorite,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            expected_value=expected_value,
            kelly_size=kelly_size,
            confidence=confidence,
            game_date=game_date,
            sport=sport,
            features=model_predictions.get('features', {})
        )
        
        return opportunity
    
    def scan_slate(self, 
                  games_data: List[Dict],
                  model_predictions: List[Dict],
                  market_odds: List[Dict]) -> List[UpsetOpportunity]:
        """
        Scan entire slate of games for upset opportunities.
        
        Args:
            games_data: List of game information dictionaries
            model_predictions: List of model prediction dictionaries
            market_odds: List of market odds dictionaries
            
        Returns:
            List of UpsetOpportunity objects sorted by expected value
        """
        opportunities = []
        
        for game_data, predictions, odds in zip(games_data, model_predictions, market_odds):
            opportunity = self.analyze_game(game_data, predictions, odds)
            if opportunity:
                opportunities.append(opportunity)
        
        # Sort by expected value (highest first)
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        # Update tracking metrics
        self.opportunities_found.extend(opportunities)
        self.update_performance_metrics(opportunities)
        
        return opportunities
    
    def filter_opportunities(self, 
                           opportunities: List[UpsetOpportunity],
                           max_bets: int = 3,
                           correlation_threshold: float = 0.8) -> List[UpsetOpportunity]:
        """
        Filter opportunities to avoid overexposure and correlation.
        
        Args:
            opportunities: List of upset opportunities
            max_bets: Maximum number of bets to place
            correlation_threshold: Threshold for avoiding correlated bets
            
        Returns:
            Filtered list of opportunities
        """
        if len(opportunities) <= max_bets:
            return opportunities
        
        # Simple filtering: take top opportunities by expected value
        # In practice, you'd want more sophisticated correlation analysis
        filtered = opportunities[:max_bets]
        
        # Check for same-sport concentration (basic diversification)
        sports_count = {}
        final_opportunities = []
        
        for opp in filtered:
            sport_count = sports_count.get(opp.sport, 0)
            if sport_count < 2:  # Max 2 bets per sport
                final_opportunities.append(opp)
                sports_count[opp.sport] = sport_count + 1
        
        return final_opportunities
    
    def generate_betting_report(self, opportunities: List[UpsetOpportunity]) -> str:
        """
        Generate detailed betting report for opportunities.
        
        Args:
            opportunities: List of upset opportunities
            
        Returns:
            Formatted report string
        """
        if not opportunities:
            return "No upset opportunities found matching criteria."
        
        report = []
        report.append("🚨 FOOTBALL UPSET ALERT REPORT 🚨")
        report.append("=" * 50)
        report.append(f"Found {len(opportunities)} high-value upset opportunities\n")
        
        total_ev = sum(opp.expected_value for opp in opportunities)
        total_kelly = sum(opp.kelly_size * self.bankroll for opp in opportunities)
        
        report.append(f"Total Expected Value: {total_ev:.3f}")
        report.append(f"Total Kelly Allocation: ${total_kelly:,.0f}")
        report.append(f"Portfolio EV: {total_ev * 100:.1f}%\n")
        
        for i, opp in enumerate(opportunities, 1):
            bet_amount = opp.kelly_size * self.bankroll
            
            report.append(f"🎯 OPPORTUNITY #{i}")
            report.append(f"Game: {opp.underdog} vs {opp.favorite}")
            report.append(f"Sport: {opp.sport.upper()}")
            report.append(f"Date: {opp.game_date.strftime('%Y-%m-%d %H:%M')}")
            report.append(f"")
            report.append(f"Model Probability: {opp.model_prob:.1%}")
            report.append(f"Market Probability: {opp.market_prob:.1%}")
            report.append(f"Edge: {opp.edge:.1%}")
            report.append(f"Expected Value: {opp.expected_value:.3f} ({opp.expected_value*100:.1f}%)")
            report.append(f"Confidence: {opp.confidence:.1%}")
            report.append(f"")
            report.append(f"Recommended Bet: ${bet_amount:.0f} ({opp.kelly_size:.1%} of bankroll)")
            report.append(f"Potential Profit: ${bet_amount * opp.expected_value:.0f}")
            report.append("-" * 30)
        
        # Risk warnings
        report.append("\n⚠️  RISK WARNINGS:")
        report.append(f"Max single bet: {self.max_bet_percentage:.1%} of bankroll")
        report.append(f"Total exposure: {sum(opp.kelly_size for opp in opportunities):.1%} of bankroll")
        report.append("Past performance doesn't guarantee future results")
        report.append("Only bet what you can afford to lose")
        
        return "\n".join(report)
    
    def update_performance_metrics(self, opportunities: List[UpsetOpportunity]) -> None:
        """
        Update internal performance tracking metrics.
        
        Args:
            opportunities: List of new opportunities
        """
        if not opportunities:
            return
        
        self.performance_metrics['total_opportunities'] += len(opportunities)
        self.performance_metrics['high_confidence_opportunities'] += sum(
            1 for opp in opportunities if opp.confidence > 0.7
        )
        self.performance_metrics['total_expected_value'] += sum(
            opp.expected_value for opp in opportunities
        )
        
        if opportunities:
            self.performance_metrics['average_edge'] = np.mean([
                opp.edge for opp in self.opportunities_found
            ])
            
        self.performance_metrics['bankroll_allocated'] += sum(
            opp.kelly_size * self.bankroll for opp in opportunities
        )
    
    def get_performance_summary(self) -> Dict[str, Union[int, float]]:
        """
        Get summary of detector performance.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics.copy()
    
    def save_opportunities(self, opportunities: List[UpsetOpportunity], 
                         filepath: str) -> None:
        """
        Save opportunities to CSV file.
        
        Args:
            opportunities: List of opportunities to save
            filepath: Path to save CSV file
        """
        if not opportunities:
            print("No opportunities to save")
            return
        
        # Convert to DataFrame
        data = []
        for opp in opportunities:
            data.append({
                'game_id': opp.game_id,
                'underdog': opp.underdog,
                'favorite': opp.favorite,
                'model_prob': opp.model_prob,
                'market_prob': opp.market_prob,
                'edge': opp.edge,
                'expected_value': opp.expected_value,
                'kelly_size': opp.kelly_size,
                'confidence': opp.confidence,
                'game_date': opp.game_date,
                'sport': opp.sport,
                'bet_amount': opp.kelly_size * self.bankroll
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(opportunities)} opportunities to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("🏈 Football Upset Detection System Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = FootballUpsetDetector(
        bankroll=10000,
        max_bet_percentage=0.05,
        min_edge=0.05,
        min_confidence=0.55,
        kelly_fraction=0.25
    )
    
    # Example game data
    games_data = [
        {
            'game_id': 'nfl_week1_game1',
            'underdog': 'Jacksonville Jaguars',
            'favorite': 'Indianapolis Colts', 
            'game_date': datetime(2025, 9, 14, 13, 0),
            'sport': 'nfl'
        },
        {
            'game_id': 'college_week1_game1',
            'underdog': 'Central Michigan',
            'favorite': 'Michigan State',
            'game_date': datetime(2025, 9, 13, 15, 30),
            'sport': 'college'
        }
    ]
    
    # Example model predictions
    model_predictions = [
        {
            'underdog_win_prob': 0.62,  # Model thinks Jaguars win 62%
            'features': {
                'offensive_efficiency': 0.75,
                'defensive_strength': 0.68,
                'turnover_differential': 0.82
            }
        },
        {
            'underdog_win_prob': 0.58,  # Model thinks Central Michigan wins 58%
            'features': {
                'offensive_efficiency': 0.65,
                'defensive_strength': 0.72,
                'motivation_factor': 0.85
            }
        }
    ]
    
    # Example market odds (American format)
    market_odds = [
        {
            'underdog_odds': +180,  # Jaguars +180 (market thinks ~36% chance)
            'format': 'american'
        },
        {
            'underdog_odds': +210,  # Central Michigan +210 (market thinks ~32% chance)
            'format': 'american'
        }
    ]
    
    # Scan for opportunities
    opportunities = detector.scan_slate(games_data, model_predictions, market_odds)
    
    print(f"Found {len(opportunities)} upset opportunities!\n")
    
    if opportunities:
        # Filter opportunities
        filtered_opportunities = detector.filter_opportunities(opportunities, max_bets=2)
        
        # Generate report
        report = detector.generate_betting_report(filtered_opportunities)
        print(report)
        
        # Performance summary
        print("\n📊 PERFORMANCE SUMMARY:")
        metrics = detector.get_performance_summary()
        print(f"Total opportunities found: {metrics['total_opportunities']}")
        print(f"High confidence opportunities: {metrics['high_confidence_opportunities']}")
        print(f"Average edge: {metrics['average_edge']:.1%}")
        print(f"Total expected value: {metrics['total_expected_value']:.3f}")
        
        # Save opportunities
        detector.save_opportunities(filtered_opportunities, 'upset_opportunities.csv')
    
    else:
        print("No upset opportunities found matching criteria.")
