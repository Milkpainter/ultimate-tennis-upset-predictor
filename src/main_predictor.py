#!/usr/bin/env python3
"""
Ultimate Tennis Upset Predictor - Main System
Orchestrates all components to find profitable betting opportunities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import our custom modules
from elo_system import AdvancedTennisElo
from feature_engineer import TennisFeatureEngineer
from ensemble_predictor import TennisUpsetPredictor
from market_analyzer import TennisMarketAnalyzer

class UltimateTennisSystem:
    """
    Main orchestrator for the Ultimate Tennis Upset Predictor
    Combines all subsystems to identify profitable betting opportunities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize all subsystems
        self.elo_system = AdvancedTennisElo()
        self.feature_engineer = TennisFeatureEngineer()
        self.ml_predictor = TennisUpsetPredictor()
        self.market_analyzer = TennisMarketAnalyzer()
        
        # System configuration
        self.config = config or {
            'min_confidence': 0.65,
            'min_edge': 0.05,
            'max_daily_bets': 5,
            'bankroll_protection': 0.02  # Max 2% risk per day
        }
        
        # Performance tracking
        self.performance_history = []
        self.current_bankroll = 1000.0  # Starting bankroll
        
    def initialize_system(self, historical_data_path: Optional[str] = None):
        """Initialize and train the ML models"""
        print("🚀 INITIALIZING ULTIMATE TENNIS UPSET SYSTEM")
        print("=" * 60)
        
        # Train the ML ensemble
        print("🤖 Training ML Ensemble...")
        self.ml_predictor.fit(use_upset_data=True)
        
        print("✅ System initialization complete!")
        print(f"🎯 Configuration: {self.config}")
        
    def analyze_match(self,
                     player_a_data: Dict,
                     player_b_data: Dict, 
                     match_context: Dict) -> Dict:
        """Complete analysis of a single match"""
        
        # Step 1: Generate comprehensive features
        features = self.feature_engineer.generate_comprehensive_features(
            player_a_data, player_b_data, match_context
        )
        
        # Step 2: Get ML prediction
        ml_prediction = self.ml_predictor.predict_upset_probability(features)
        
        # Step 3: Market analysis
        market_analysis = self.market_analyzer.comprehensive_match_analysis(
            match_context, ml_prediction
        )
        
        # Step 4: Elo-based validation
        elo_prob = self.elo_system.predict_match_probability(
            player_a_data.get('elo_ratings', {'hard': 1500}),
            player_b_data.get('elo_ratings', {'hard': 1500}),
            match_context.get('surface', 'hard'),
            player_a_data.get('surface_specialization', {'hard': 0}),
            player_b_data.get('surface_specialization', {'hard': 0}),
            fatigue_a=features.get('fatigue_differential', 0),
            fatigue_b=0,
            h2h_advantage_a=features.get('h2h_psychological_advantage', 0)
        )
        
        # Combine all analyses
        complete_analysis = {
            'match_info': market_analysis['match_info'],
            'features': features,
            'ml_prediction': ml_prediction,
            'elo_prediction': {'player_a_win_probability': elo_prob},
            'market_analysis': market_analysis,
            'final_recommendation': self._generate_final_recommendation(market_analysis)
        }
        
        return complete_analysis
    
    def _generate_final_recommendation(self, market_analysis: Dict) -> Dict:
        """Generate final betting recommendation with risk management"""
        
        opportunity = market_analysis['opportunity_assessment']
        kelly_sizing = market_analysis['kelly_sizing']
        
        # Risk management checks
        max_edge = market_analysis['betting_edges']['max_edge']
        confidence = opportunity['confidence_score'] / 100
        
        # Apply system filters
        passes_confidence_filter = confidence >= self.config['min_confidence']
        passes_edge_filter = max_edge >= self.config['min_edge']
        
        recommended_action = 'NO_BET'
        bet_size = 0
        reasoning = []
        
        if passes_confidence_filter and passes_edge_filter:
            recommended_action = kelly_sizing['recommended_bet'].upper()
            
            if recommended_action == 'UNDERDOG':
                bet_size = min(kelly_sizing['underdog_kelly_pct'], 5.0)  # Max 5% per bet
                reasoning.append(f"Model sees {opportunity['upset_potential']:.0f}% upset chance")
                reasoning.append(f"Market edge: {opportunity['edge_percentage']:.1f}%")
            elif recommended_action == 'FAVORITE':
                bet_size = min(kelly_sizing['favorite_kelly_pct'], 5.0)
                reasoning.append(f"Strong favorite with {100-opportunity['upset_potential']:.0f}% win probability")
                reasoning.append(f"Market undervaluing favorite by {opportunity['edge_percentage']:.1f}%")
        else:
            if not passes_confidence_filter:
                reasoning.append(f"Low confidence: {confidence:.1%} < {self.config['min_confidence']:.1%}")
            if not passes_edge_filter:
                reasoning.append(f"Insufficient edge: {max_edge:.1%} < {self.config['min_edge']:.1%}")
        
        return {
            'action': recommended_action,
            'bet_size_percentage': bet_size,
            'reasoning': reasoning,
            'opportunity_level': opportunity['level'],
            'risk_reward_ratio': max_edge / max(0.01, 1 - confidence)  # Edge over uncertainty
        }
    
    def scan_daily_matches(self, matches_data: List[Dict]) -> Dict:
        """Scan all daily matches and return prioritized opportunities"""
        
        print(f"🔍 SCANNING {len(matches_data)} MATCHES FOR UPSET OPPORTUNITIES")
        print("=" * 70)
        
        all_analyses = []
        high_value_opportunities = []
        
        for i, match_data in enumerate(matches_data):
            print(f"  🎾 Analyzing Match {i+1}: {match_data.get('player_a', 'A')} vs {match_data.get('player_b', 'B')}")
            
            analysis = self.analyze_match(
                match_data.get('player_a_data', {}),
                match_data.get('player_b_data', {}),
                match_data
            )
            
            all_analyses.append(analysis)
            
            # Check if this is a high-value opportunity
            recommendation = analysis['final_recommendation']
            if (recommendation['action'] != 'NO_BET' and 
                recommendation['bet_size_percentage'] > 1.0):  # At least 1% bet size
                
                high_value_opportunities.append({
                    'match_id': i + 1,
                    'analysis': analysis
                })
        
        # Sort opportunities by potential value
        high_value_opportunities.sort(
            key=lambda x: x['analysis']['final_recommendation']['risk_reward_ratio'],
            reverse=True
        )
        
        # Generate daily summary
        summary = {
            'total_matches_analyzed': len(matches_data),
            'opportunities_found': len(high_value_opportunities),
            'opportunity_rate': len(high_value_opportunities) / len(matches_data) * 100,
            'top_opportunities': high_value_opportunities[:5],  # Top 5
            'all_analyses': all_analyses,
            'system_status': 'ACTIVE',
            'scan_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def generate_daily_report(self, daily_scan: Dict) -> str:
        """Generate human-readable daily report"""
        
        report_lines = []
        report_lines.append("🎾 ULTIMATE TENNIS UPSET PREDICTOR - DAILY REPORT")
        report_lines.append("=" * 65)
        report_lines.append(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Matches Analyzed: {daily_scan['total_matches_analyzed']}")
        report_lines.append(f"Opportunities Found: {daily_scan['opportunities_found']} ({daily_scan['opportunity_rate']:.1f}%)")
        report_lines.append("")
        
        if daily_scan['opportunities_found'] > 0:
            report_lines.append("🔥 TOP UPSET OPPORTUNITIES:")
            report_lines.append("-" * 40)
            
            for i, opp in enumerate(daily_scan['top_opportunities'][:3], 1):
                analysis = opp['analysis']
                match_info = analysis['match_info']
                recommendation = analysis['final_recommendation']
                market_analysis = analysis['market_analysis']
                
                report_lines.append(f"\n{i}. {match_info['player_a']} vs {match_info['player_b']}")
                report_lines.append(f"   Surface: {match_info['surface'].title()} | Tournament: {match_info['tournament']}")
                report_lines.append(f"   🎯 Action: {recommendation['action']} ({recommendation['bet_size_percentage']:.1f}% of bankroll)")
                report_lines.append(f"   💰 Expected Value: {max(market_analysis['expected_values']['underdog_ev'], market_analysis['expected_values']['favorite_ev']):+.1f}%")
                report_lines.append(f"   📈 Edge: {recommendation['risk_reward_ratio']:.2f}x risk-reward")
                report_lines.append(f"   🧠 Reasoning: {' | '.join(recommendation['reasoning'])}")
        else:
            report_lines.append("⚠️ No high-value opportunities detected today.")
            report_lines.append("Market appears efficient for current matches.")
        
        report_lines.append("\n" + "=" * 65)
        report_lines.append("📊 SYSTEM PERFORMANCE SUMMARY:")
        report_lines.append(f"Current Bankroll: ${self.current_bankroll:.2f}")
        report_lines.append(f"Total Opportunities Found: {daily_scan['opportunities_found']}")
        report_lines.append(f"System Status: {daily_scan['system_status']}")
        
        return "\n".join(report_lines)

def demo_system():
    """Demonstration of the complete system"""
    print("🚀 ULTIMATE TENNIS UPSET PREDICTOR - SYSTEM DEMO")
    print("=" * 65)
    
    # Initialize system
    system = UltimateTennisSystem()
    system.initialize_system()
    
    # Create sample match data
    sample_matches = [
        {
            'player_a': 'Jannik Sinner',
            'player_b': 'Novak Djokovic', 
            'surface': 'hard',
            'tournament': 'US Open',
            'round': 'Semifinals',
            'market_odds': {'favorite': 1.45, 'underdog': 2.75},
            'player_a_data': {
                'ranking': 4,
                'elo_rating': 1950,
                'recent_matches': [
                    {'result': 'W', 'opponent_rating': 1900, 'tournament_level': 'masters', 'days_ago': 7},
                    {'result': 'W', 'opponent_rating': 1850, 'tournament_level': 'grand_slam', 'days_ago': 14}
                ],
                'days_since_last_match': 7
            },
            'player_b_data': {
                'ranking': 1,
                'elo_rating': 2100,
                'recent_matches': [
                    {'result': 'W', 'opponent_rating': 1800, 'tournament_level': 'grand_slam', 'days_ago': 3,
                     'sets_total': 5, 'duration_minutes': 240},  # Long grueling match
                    {'result': 'W', 'opponent_rating': 1900, 'tournament_level': 'grand_slam', 'days_ago': 5,
                     'sets_total': 4, 'duration_minutes': 200}   # Another tough match
                ],
                'days_since_last_match': 3
            },
            'market_overreaction': 0.12  # Public loves Djokovic
        },
        {
            'player_a': 'Carlos Alcaraz',
            'player_b': 'Daniil Medvedev',
            'surface': 'clay', 
            'tournament': 'French Open',
            'round': 'Quarterfinals',
            'market_odds': {'favorite': 1.35, 'underdog': 3.20},
            'player_a_data': {
                'ranking': 2,
                'elo_rating': 2050,
                'recent_matches': [
                    {'result': 'W', 'opponent_rating': 1950, 'tournament_level': 'grand_slam', 'days_ago': 4},
                    {'result': 'W', 'opponent_rating': 1880, 'tournament_level': 'grand_slam', 'days_ago': 6}
                ],
                'days_since_last_match': 4
            },
            'player_b_data': {
                'ranking': 5,
                'elo_rating': 1920,
                'recent_matches': [
                    {'result': 'L', 'opponent_rating': 1850, 'tournament_level': 'masters', 'days_ago': 20},  # Poor clay form
                    {'result': 'W', 'opponent_rating': 1750, 'tournament_level': 'masters', 'days_ago': 25}
                ],
                'days_since_last_match': 4
            },
            'market_overreaction': 0.05
        }
    ]
    
    # Scan for opportunities
    daily_results = system.scan_daily_matches(sample_matches)
    
    # Generate report
    report = system.generate_daily_report(daily_results)
    print(report)
    
    return system, daily_results

if __name__ == "__main__":
    # Run the demo
    system, results = demo_system()