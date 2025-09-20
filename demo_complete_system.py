#!/usr/bin/env python3
"""
ULTIMATE TENNIS UPSET PREDICTOR - COMPLETE SYSTEM DEMONSTRATION
This script demonstrates the full system capabilities for finding profitable upsets
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import all system components
from data_collector import TennisDataCollector
from feature_engineer import TennisFeatureEngineer  
from ensemble_predictor import TennisUpsetPredictor
from market_analyzer import TennisMarketAnalyzer
from elo_system import AdvancedTennisElo

class CompleteSystemDemo:
    """
    Complete demonstration of the Ultimate Tennis Upset Predictor
    Shows how all components work together to find profitable opportunities
    """
    
    def __init__(self):
        print("🎾 ULTIMATE TENNIS UPSET PREDICTOR - COMPLETE SYSTEM")
        print("=" * 70)
        print("🚀 Initializing all subsystems...")
        
        # Initialize all components
        self.data_collector = TennisDataCollector()
        self.feature_engineer = TennisFeatureEngineer()
        self.ml_predictor = TennisUpsetPredictor()
        self.market_analyzer = TennisMarketAnalyzer()
        self.elo_system = AdvancedTennisElo()
        
        print("✅ All systems initialized successfully!")
        
    def run_complete_demo(self):
        """Run the complete system demonstration"""
        
        print(f"\n🔥 PHASE 1: TRAINING THE UPSET DETECTION AI")
        print("-" * 50)
        
        # Train the ML system
        self.ml_predictor.fit(use_upset_data=True)
        
        print(f"\n📊 PHASE 2: COLLECTING REAL MATCH DATA")  
        print("-" * 50)
        
        # Get today's tournament schedule
        schedule = self.data_collector.collect_tournament_schedule('2025-09-20')
        print(f"🎯 Found {len(schedule)} matches scheduled for today")
        
        # Analyze each match
        daily_opportunities = []
        
        for match in schedule:
            print(f"\n🎾 ANALYZING: {match['player_a']} vs {match['player_b']}")
            print(f"   Tournament: {match['tournament']} | Surface: {match['surface']} | Round: {match['round']}")
            
            # Get comprehensive match data
            match_data = self.data_collector.get_comprehensive_match_data(match)
            
            # Generate features
            features = self.feature_engineer.generate_comprehensive_features(
                match_data['player_a_data'],
                match_data['player_b_data'], 
                match_data
            )
            
            # Get ML prediction
            ml_prediction = self.ml_predictor.predict_upset_probability(features)
            
            # Get Elo prediction for comparison
            elo_prob = self.elo_system.predict_match_probability(
                match_data['player_a_data']['elo_ratings'],
                match_data['player_b_data']['elo_ratings'],
                match['surface'],
                {'hard': 0, 'clay': 0, 'grass': 0},  # Simplified for demo
                {'hard': 0, 'clay': 0, 'grass': 0}
            )
            
            # Market analysis
            market_analysis = self.market_analyzer.comprehensive_match_analysis(
                match_data, ml_prediction
            )
            
            # Display results
            self._display_match_analysis(match, features, ml_prediction, elo_prob, market_analysis)
            
            # Store for final summary
            daily_opportunities.append({
                'match': match,
                'ml_prediction': ml_prediction,
                'market_analysis': market_analysis,
                'features': features
            })
            
        print(f"\n💰 PHASE 3: IDENTIFYING PROFITABLE OPPORTUNITIES")
        print("-" * 60)
        
        # Find the best opportunities
        best_opportunities = self._rank_opportunities(daily_opportunities)
        
        print(f"\n🏆 TOP UPSET OPPORTUNITIES FOR TODAY:")
        print("=" * 60)
        
        for i, opp in enumerate(best_opportunities[:3], 1):
            self._display_opportunity_summary(i, opp)
            
        # Generate final system report
        self._generate_final_report(daily_opportunities, best_opportunities)
        
        return daily_opportunities, best_opportunities
    
    def _display_match_analysis(self, match, features, ml_prediction, elo_prob, market_analysis):
        """Display detailed analysis for a single match"""
        
        upset_prob = ml_prediction['ensemble_upset_probability']
        confidence = ml_prediction['confidence_score']
        
        print(f"   🤖 ML Upset Probability: {upset_prob:.1%}")
        print(f"   📈 Elo Model: Player A {elo_prob:.1%} | Player B {1-elo_prob:.1%}")
        print(f"   🎯 Confidence Score: {confidence:.1%}")
        
        # Show key features driving prediction
        key_features = ['momentum_differential', 'fatigue_differential', 'elo_gap']
        feature_summary = []
        for feat in key_features:
            if feat in features:
                value = features[feat]
                direction = "↗️" if value > 0 else "↘️" if value < 0 else "➡️"
                feature_summary.append(f"{feat.replace('_', ' ').title()}: {direction}{value:+.2f}")
        
        print(f"   🔍 Key Factors: {' | '.join(feature_summary)}")
        
        # Market analysis
        edge = market_analysis['betting_edges']['max_edge']
        recommendation = market_analysis['kelly_sizing']['recommended_bet']
        
        if recommendation != 'no_bet':
            print(f"   💡 OPPORTUNITY: {recommendation.upper()} bet with {edge:.1%} edge!")
        else:
            print(f"   ⏸️  No significant edge detected")
    
    def _rank_opportunities(self, daily_opportunities):
        """Rank opportunities by profitability potential"""
        
        ranked = []
        
        for opp in daily_opportunities:
            market_analysis = opp['market_analysis']
            ml_prediction = opp['ml_prediction']
            
            # Calculate opportunity score
            edge = market_analysis['betting_edges']['max_edge']
            confidence = ml_prediction['confidence_score']
            upset_prob = ml_prediction['ensemble_upset_probability']
            
            # Prioritize high-confidence upsets with good edges
            opportunity_score = (
                edge * 0.4 +                           # Market edge most important
                confidence * 0.3 +                     # Model confidence
                min(upset_prob, 0.6) * 0.2 +           # Upset potential (capped)
                (1 if edge > 0.1 else 0) * 0.1         # Bonus for major edges
            )
            
            if opportunity_score > 0.5:  # Only include meaningful opportunities
                opp['opportunity_score'] = opportunity_score
                ranked.append(opp)
        
        # Sort by opportunity score
        return sorted(ranked, key=lambda x: x['opportunity_score'], reverse=True)
    
    def _display_opportunity_summary(self, rank, opportunity):
        """Display summary for top opportunities"""
        
        match = opportunity['match']
        ml_pred = opportunity['ml_prediction'] 
        market = opportunity['market_analysis']
        
        player_a = match['player_a']
        player_b = match['player_b']
        
        upset_prob = ml_pred['ensemble_upset_probability']
        edge = market['betting_edges']['max_edge']
        recommended_bet = market['kelly_sizing']['recommended_bet']
        
        print(f"\n{rank}. {player_a} vs {player_b}")
        print(f"   🏟️  {match['tournament']} | {match['surface'].title()} | {match['round']}")
        
        if recommended_bet == 'underdog':
            underdog = player_a if upset_prob > 0.5 else player_b
            odds = market['market_odds']['underdog']
            kelly_size = market['kelly_sizing']['underdog_kelly_pct']
            ev = market['expected_values']['underdog_ev']
            
            print(f"   🚨 UPSET ALERT: Back {underdog} at {odds:.2f}")
            print(f"   💰 Expected Value: {ev:+.1f}% | Kelly Size: {kelly_size:.1f}%")
            print(f"   📊 Model Edge: {edge:.1%} over market consensus")
            
        elif recommended_bet == 'favorite':
            favorite = player_b if upset_prob > 0.5 else player_a  
            odds = market['market_odds']['favorite']
            kelly_size = market['kelly_sizing']['favorite_kelly_pct']
            ev = market['expected_values']['favorite_ev']
            
            print(f"   ✅ VALUE FAVORITE: Back {favorite} at {odds:.2f}")
            print(f"   💰 Expected Value: {ev:+.1f}% | Kelly Size: {kelly_size:.1f}%")
            print(f"   📊 Model Edge: {edge:.1%} over market consensus")
        
        # Show upset indicators
        active_indicators = sum(ml_pred.get('upset_indicators', {}).values())
        print(f"   🎯 Upset Indicators Active: {active_indicators}/6")
        print(f"   🏆 Opportunity Score: {opportunity['opportunity_score']:.2f}/1.0")
    
    def _generate_final_report(self, daily_opportunities, best_opportunities):
        """Generate comprehensive final report"""
        
        print(f"\n📋 DAILY SYSTEM PERFORMANCE REPORT")
        print("=" * 60)
        
        total_matches = len(daily_opportunities)
        profitable_opportunities = len(best_opportunities)
        
        if profitable_opportunities > 0:
            avg_edge = np.mean([opp['market_analysis']['betting_edges']['max_edge'] 
                               for opp in best_opportunities])
            avg_confidence = np.mean([opp['ml_prediction']['confidence_score'] 
                                     for opp in best_opportunities])
            
            print(f"📊 OPPORTUNITY SUMMARY:")
            print(f"   • Total Matches Analyzed: {total_matches}")
            print(f"   • Profitable Opportunities: {profitable_opportunities} ({profitable_opportunities/total_matches:.1%})")
            print(f"   • Average Market Edge: {avg_edge:.1%}")
            print(f"   • Average Model Confidence: {avg_confidence:.1%}")
            
            # Calculate potential daily profit
            potential_daily_return = sum([
                opp['market_analysis']['betting_edges']['max_edge'] * 
                opp['market_analysis']['kelly_sizing'].get('underdog_kelly_pct', 
                opp['market_analysis']['kelly_sizing'].get('favorite_kelly_pct', 0)) / 100
                for opp in best_opportunities
            ]) * 100
            
            print(f"   • Potential Daily Return: {potential_daily_return:.1f}%")
            
            # Risk assessment
            max_single_bet = max([
                max(opp['market_analysis']['kelly_sizing']['underdog_kelly_pct'],
                    opp['market_analysis']['kelly_sizing']['favorite_kelly_pct'])
                for opp in best_opportunities
            ])
            
            total_risk = sum([
                max(opp['market_analysis']['kelly_sizing']['underdog_kelly_pct'],
                    opp['market_analysis']['kelly_sizing']['favorite_kelly_pct'])
                for opp in best_opportunities
            ])
            
            print(f"   • Max Single Bet Size: {max_single_bet:.1f}%")
            print(f"   • Total Daily Risk: {total_risk:.1f}%")
            
            print(f"\n🎯 SYSTEM RECOMMENDATIONS:")
            if len(best_opportunities) <= 3 and avg_edge > 0.08:
                print("   ✅ PROCEED: High-quality opportunities with manageable risk")
            elif len(best_opportunities) > 5:
                print("   ⚠️  CAUTION: Many opportunities - may indicate model overfitting")
            else:
                print("   📊 NORMAL: Standard opportunity level detected")
                
        else:
            print("📊 MARKET EFFICIENCY DETECTED:")
            print("   • No significant edges found in today's matches")
            print("   • Market appears properly priced")
            print("   • Recommend waiting for better opportunities")
        
        print(f"\n🔮 NEXT STEPS:")
        print("   1. Monitor line movement throughout the day")
        print("   2. Update predictions as new information arrives") 
        print("   3. Execute recommended bets at optimal timing")
        print("   4. Track results for model performance validation")
        
        print(f"\n⚡ SYSTEM STATUS: FULLY OPERATIONAL")
        print("🎾 Ready to beat the tennis betting markets!")

def main():
    """Main demonstration function"""
    
    print("🚀 STARTING ULTIMATE TENNIS UPSET PREDICTOR DEMO")
    print("=" * 70)
    
    # Initialize complete system
    demo_system = CompleteSystemDemo()
    
    # Run complete demonstration
    try:
        daily_opportunities, best_opportunities = demo_system.run_complete_demo()
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"Found {len(best_opportunities)} profitable opportunities out of {len(daily_opportunities)} matches")
        
        return True, demo_system
        
    except Exception as e:
        print(f"\n❌ DEMO ERROR: {str(e)}")
        return False, None

if __name__ == "__main__":
    success, system = main()
    
    if success:
        print(f"\n🎾 Your Ultimate Tennis Upset Predictor is ready!")
        print("🚀 Deploy this system to start finding profitable upsets!")
    else:
        print(f"\n🔧 Please check system configuration and try again.")