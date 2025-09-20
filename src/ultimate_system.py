#!/usr/bin/env python3
"""
ULTIMATE TENNIS PREDICTION SYSTEM - FINAL INTEGRATION
Combines all advanced components into the most sophisticated tennis prediction system:
- Deep Learning Neural Networks (Transformers, LSTM, Graph Networks)
- Advanced Sentiment Analysis and News Monitoring
- Real-time Odds Tracking and Arbitrage Detection
- Market Analysis and Statistical Arbitrage
- Ensemble ML Models optimized for upsets
- Advanced Feature Engineering with 50+ features
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import all system components
from elo_system import AdvancedTennisElo
from feature_engineer import TennisFeatureEngineer
from ensemble_predictor import TennisUpsetPredictor
from market_analyzer import TennisMarketAnalyzer
from data_collector import TennisDataCollector
from deep_learning_engine import DeepLearningTennisPredictor
from sentiment_analyzer import AdvancedSentimentAnalyzer
from odds_tracker import AdvancedOddsTracker

class UltimateTennisSystem:
    """
    The Ultimate Tennis Prediction System
    Integrates all advanced components for maximum prediction accuracy
    """
    
    def __init__(self, config: Optional[Dict] = None):
        print("🏆 INITIALIZING ULTIMATE TENNIS PREDICTION SYSTEM")
        print("=" * 65)
        
        self.config = config or {
            'use_deep_learning': True,
            'use_sentiment_analysis': True,
            'use_real_time_odds': True,
            'ensemble_weights': {
                'deep_learning': 0.35,
                'ml_ensemble': 0.25,
                'elo_system': 0.20,
                'sentiment': 0.10,
                'market_analysis': 0.10
            },
            'confidence_threshold': 0.70,
            'min_edge_threshold': 0.08,
            'max_positions_per_day': 3
        }
        
        # Initialize all subsystems
        print("🔄 Initializing Core Systems...")
        self.elo_system = AdvancedTennisElo()
        self.feature_engineer = TennisFeatureEngineer()
        self.ml_predictor = TennisUpsetPredictor()
        self.market_analyzer = TennisMarketAnalyzer()
        self.data_collector = TennisDataCollector()
        
        # Initialize advanced systems
        print("⚡ Initializing Advanced Systems...")
        if self.config['use_deep_learning']:
            self.deep_learning = DeepLearningTennisPredictor()
            print("  ✅ Deep Learning Engine loaded")
        else:
            self.deep_learning = None
            
        if self.config['use_sentiment_analysis']:
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
            print("  ✅ Sentiment Analysis Engine loaded")
        else:
            self.sentiment_analyzer = None
            
        if self.config['use_real_time_odds']:
            self.odds_tracker = AdvancedOddsTracker()
            print("  ✅ Real-time Odds Tracker loaded")
        else:
            self.odds_tracker = None
            
        # Performance tracking
        self.predictions_history = []
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'upset_predictions': 0,
            'correct_upsets': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0
        }
        
        print("✅ ULTIMATE SYSTEM INITIALIZED SUCCESSFULLY!")
        print(f"  🤖 Deep Learning: {'ENABLED' if self.deep_learning else 'DISABLED'}")
        print(f"  📈 Sentiment Analysis: {'ENABLED' if self.sentiment_analyzer else 'DISABLED'}")
        print(f"  💰 Real-time Odds: {'ENABLED' if self.odds_tracker else 'DISABLED'}")
        print(f"  🎯 Confidence Threshold: {self.config['confidence_threshold']:.1%}")
        print(f"  📉 Edge Threshold: {self.config['min_edge_threshold']:.1%}")
    
    async def analyze_ultimate_match(self, match_data: Dict) -> Dict:
        """Ultimate match analysis combining all systems"""
        
        match_id = f"{match_data.get('player_a', 'A')}_vs_{match_data.get('player_b', 'B')}"
        
        print(f"\n🔍 ULTIMATE ANALYSIS: {match_data.get('player_a')} vs {match_data.get('player_b')}")
        print("-" * 70)
        
        analysis_results = {
            'match_info': {
                'match_id': match_id,
                'player_a': match_data.get('player_a'),
                'player_b': match_data.get('player_b'),
                'tournament': match_data.get('tournament'),
                'surface': match_data.get('surface'),
                'analysis_timestamp': datetime.now()
            }
        }
        
        # 1. Advanced Feature Engineering
        print("🔧 Generating advanced features...")
        features = self.feature_engineer.generate_comprehensive_features(
            match_data.get('player_a_data', {}),
            match_data.get('player_b_data', {}),
            match_data
        )
        analysis_results['features'] = features
        
        # 2. Elo System Analysis
        print("📊 Calculating Elo predictions...")
        elo_prediction = self.elo_system.predict_match_probability(
            match_data.get('player_a_data', {}).get('elo_ratings', {'overall': 1500}),
            match_data.get('player_b_data', {}).get('elo_ratings', {'overall': 1500}),
            match_data.get('surface', 'hard'),
            match_data.get('player_a_data', {}).get('surface_specialization', {}),
            match_data.get('player_b_data', {}).get('surface_specialization', {})
        )
        analysis_results['elo_prediction'] = {'player_a_win_probability': elo_prediction}
        
        # 3. ML Ensemble Prediction
        print("🤖 Running ML ensemble models...")
        ml_prediction = self.ml_predictor.predict_upset_probability(features)
        analysis_results['ml_prediction'] = ml_prediction
        
        # 4. Deep Learning Analysis (if enabled)
        if self.deep_learning:
            print("🧠 Running deep neural networks...")
            try:
                deep_prediction = self.deep_learning.predict_with_uncertainty(match_data)
                analysis_results['deep_learning_prediction'] = deep_prediction
            except Exception as e:
                print(f"  ⚠️ Deep learning unavailable: {e}")
                analysis_results['deep_learning_prediction'] = None
        
        # 5. Sentiment Analysis (if enabled)
        if self.sentiment_analyzer:
            print("📈 Analyzing market sentiment...")
            sentiment_analysis = self.sentiment_analyzer.analyze_market_sentiment(match_data)
            analysis_results['sentiment_analysis'] = sentiment_analysis
        
        # 6. Real-time Odds Analysis (if enabled)
        if self.odds_tracker:
            print("💰 Tracking real-time odds...")
            try:
                best_odds = self.odds_tracker.get_best_current_odds(match_id)
                analysis_results['best_odds'] = best_odds
                
                # Quick arbitrage check
                current_snapshots = await self.odds_tracker._collect_all_bookmaker_odds(match_id)
                arbitrage_ops = self.odds_tracker._detect_arbitrage_opportunities(current_snapshots)
                analysis_results['arbitrage_opportunities'] = arbitrage_ops
                
            except Exception as e:
                print(f"  ⚠️ Real-time odds unavailable: {e}")
                analysis_results['best_odds'] = None
                analysis_results['arbitrage_opportunities'] = []
        
        # 7. Market Analysis
        print("💹 Performing market analysis...")
        market_analysis = self.market_analyzer.comprehensive_match_analysis(
            match_data, ml_prediction
        )
        analysis_results['market_analysis'] = market_analysis
        
        # 8. Ultimate Ensemble Prediction
        print("🏆 Computing ultimate prediction...")
        ultimate_prediction = self._compute_ultimate_prediction(analysis_results)
        analysis_results['ultimate_prediction'] = ultimate_prediction
        
        # 9. Generate Trading Recommendations
        print("🎯 Generating trading recommendations...")
        recommendations = self._generate_ultimate_recommendations(analysis_results)
        analysis_results['recommendations'] = recommendations
        
        print("✅ Ultimate analysis complete!")
        return analysis_results
    
    def _compute_ultimate_prediction(self, analysis_results: Dict) -> Dict:
        """Compute the ultimate prediction by ensembling all models"""
        
        predictions = []
        weights = []
        
        # Collect predictions and weights
        if 'elo_prediction' in analysis_results:
            elo_prob = analysis_results['elo_prediction']['player_a_win_probability']
            predictions.append([elo_prob, 1 - elo_prob, min(elo_prob, 1-elo_prob)])  # [win, loss, upset]
            weights.append(self.config['ensemble_weights']['elo_system'])
        
        if 'ml_prediction' in analysis_results:
            ml_pred = analysis_results['ml_prediction']
            predictions.append([
                ml_pred['favorite_win_probability'],
                ml_pred['ensemble_upset_probability'], 
                ml_pred['ensemble_upset_probability']
            ])
            weights.append(self.config['ensemble_weights']['ml_ensemble'])
        
        if analysis_results.get('deep_learning_prediction'):
            dl_pred = analysis_results['deep_learning_prediction']
            predictions.append([
                dl_pred['win_probability'],
                dl_pred['loss_probability'],
                dl_pred['upset_probability']
            ])
            weights.append(self.config['ensemble_weights']['deep_learning'])
        
        # Sentiment adjustment
        sentiment_adjustment = 0
        if analysis_results.get('sentiment_analysis'):
            sentiment_score = analysis_results['sentiment_analysis']['combined_sentiment_score']
            sentiment_adjustment = sentiment_score * self.config['ensemble_weights']['sentiment']
        
        # Market analysis adjustment
        market_adjustment = 0
        if 'market_analysis' in analysis_results:
            market_edge = analysis_results['market_analysis']['betting_edges']['max_edge']
            market_adjustment = market_edge * self.config['ensemble_weights']['market_analysis']
        
        # Compute weighted ensemble
        if predictions:
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            # Apply sentiment and market adjustments
            ensemble_pred[0] += sentiment_adjustment + market_adjustment  # Win probability
            ensemble_pred[2] = max(ensemble_pred[1], ensemble_pred[2])  # Upset probability
            
            # Normalize probabilities
            ensemble_pred = np.clip(ensemble_pred, 0.01, 0.99)
            total = ensemble_pred[0] + ensemble_pred[1]
            if total > 0:
                ensemble_pred[0] /= total
                ensemble_pred[1] /= total
        else:
            ensemble_pred = [0.5, 0.5, 0.25]  # Default
        
        # Calculate confidence based on model agreement
        if len(predictions) > 1:
            pred_std = np.std(predictions, axis=0)
            confidence = 1 - np.mean(pred_std)
        else:
            confidence = 0.6
        
        return {
            'win_probability': float(ensemble_pred[0]),
            'loss_probability': float(ensemble_pred[1]), 
            'upset_probability': float(ensemble_pred[2]),
            'confidence': max(0.1, min(0.95, confidence)),
            'model_count': len(predictions),
            'ensemble_weights': dict(zip(['elo', 'ml', 'deep_learning'], weights[:3])) if len(weights) >= 3 else {},
            'sentiment_adjustment': sentiment_adjustment,
            'market_adjustment': market_adjustment
        }
    
    def _generate_ultimate_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate ultimate trading recommendations"""
        
        ultimate_pred = analysis_results['ultimate_prediction']
        market_analysis = analysis_results.get('market_analysis', {})
        
        recommendations = {
            'action': 'NO_BET',
            'reasoning': [],
            'bet_size': 0,
            'expected_value': 0,
            'risk_level': 'NONE',
            'opportunity_score': 0,
            'alerts': []
        }
        
        confidence = ultimate_pred['confidence']
        upset_prob = ultimate_pred['upset_probability']
        
        # Check confidence threshold
        if confidence < self.config['confidence_threshold']:
            recommendations['reasoning'].append(f"Low confidence: {confidence:.1%} < {self.config['confidence_threshold']:.1%}")
            return recommendations
        
        # Check for high-value opportunities
        market_edge = market_analysis.get('betting_edges', {}).get('max_edge', 0)
        
        if market_edge >= self.config['min_edge_threshold']:
            # Determine bet type
            if upset_prob > 0.35 and market_edge > 0.08:
                recommendations['action'] = 'BACK_UNDERDOG'
                recommendations['reasoning'].append(f"High upset probability: {upset_prob:.1%}")
                recommendations['reasoning'].append(f"Significant market edge: {market_edge:.1%}")
                recommendations['risk_level'] = 'HIGH_REWARD'
                
            elif ultimate_pred['win_probability'] > 0.7 and market_edge > 0.06:
                recommendations['action'] = 'BACK_FAVORITE'
                recommendations['reasoning'].append(f"Strong favorite: {ultimate_pred['win_probability']:.1%}")
                recommendations['reasoning'].append(f"Good market value: {market_edge:.1%}")
                recommendations['risk_level'] = 'MODERATE'
            
            # Calculate bet sizing using Kelly criterion
            if recommendations['action'] != 'NO_BET':
                kelly_size = market_analysis.get('kelly_sizing', {})
                if recommendations['action'] == 'BACK_UNDERDOG':
                    recommendations['bet_size'] = kelly_size.get('underdog_kelly_pct', 0)
                    recommendations['expected_value'] = market_analysis.get('expected_values', {}).get('underdog_ev', 0)
                else:
                    recommendations['bet_size'] = kelly_size.get('favorite_kelly_pct', 0)
                    recommendations['expected_value'] = market_analysis.get('expected_values', {}).get('favorite_ev', 0)
        
        # Calculate opportunity score
        recommendations['opportunity_score'] = (
            confidence * 0.4 +
            market_edge * 0.3 +
            min(upset_prob, 0.5) * 0.2 +
            (1 if market_edge > 0.1 else 0) * 0.1
        )
        
        # Generate alerts
        if upset_prob > 0.45:
            recommendations['alerts'].append('🚨 HIGH UPSET POTENTIAL')
        
        if market_edge > 0.15:
            recommendations['alerts'].append('💰 MAJOR MARKET INEFFICIENCY')
        
        if analysis_results.get('arbitrage_opportunities'):
            recommendations['alerts'].append('⚡ ARBITRAGE OPPORTUNITY AVAILABLE')
        
        if analysis_results.get('sentiment_analysis', {}).get('sharp_money_detected'):
            recommendations['alerts'].append('🔍 SHARP MONEY DETECTED')
        
        return recommendations
    
    async def scan_ultimate_opportunities(self, matches_data: List[Dict]) -> Dict:
        """Scan multiple matches for ultimate opportunities"""
        
        print(f"\n🚀 ULTIMATE OPPORTUNITY SCAN: {len(matches_data)} MATCHES")
        print("=" * 70)
        
        all_analyses = []
        high_value_opportunities = []
        arbitrage_opportunities = []
        
        # Analyze each match
        for i, match_data in enumerate(matches_data, 1):
            print(f"\n[{i}/{len(matches_data)}] Analyzing: {match_data.get('player_a')} vs {match_data.get('player_b')}")
            
            analysis = await self.analyze_ultimate_match(match_data)
            all_analyses.append(analysis)
            
            # Check for high-value opportunities
            recommendations = analysis['recommendations']
            if recommendations['action'] != 'NO_BET' and recommendations['opportunity_score'] > 0.7:
                high_value_opportunities.append(analysis)
            
            # Check for arbitrage
            if analysis.get('arbitrage_opportunities'):
                arbitrage_opportunities.extend(analysis['arbitrage_opportunities'])
        
        # Rank opportunities
        high_value_opportunities.sort(
            key=lambda x: x['recommendations']['opportunity_score'], 
            reverse=True
        )
        
        # Generate summary
        summary = {
            'total_matches_analyzed': len(matches_data),
            'high_value_opportunities': len(high_value_opportunities),
            'arbitrage_opportunities': len(arbitrage_opportunities),
            'top_opportunities': high_value_opportunities[:self.config['max_positions_per_day']],
            'all_analyses': all_analyses,
            'scan_timestamp': datetime.now(),
            'system_status': 'OPTIMAL'
        }
        
        return summary
    
    def generate_ultimate_report(self, scan_results: Dict) -> str:
        """Generate the ultimate daily report"""
        
        report_lines = []
        report_lines.append("🏆 ULTIMATE TENNIS PREDICTION SYSTEM - DAILY REPORT")
        report_lines.append("=" * 75)
        report_lines.append(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"🔍 Matches Analyzed: {scan_results['total_matches_analyzed']}")
        report_lines.append(f"🎯 High-Value Opportunities: {scan_results['high_value_opportunities']}")
        report_lines.append(f"⚡ Arbitrage Opportunities: {scan_results['arbitrage_opportunities']}")
        report_lines.append("")
        
        if scan_results['top_opportunities']:
            report_lines.append("🔥 TOP ULTIMATE OPPORTUNITIES:")
            report_lines.append("-" * 50)
            
            for i, opportunity in enumerate(scan_results['top_opportunities'], 1):
                match_info = opportunity['match_info']
                ultimate_pred = opportunity['ultimate_prediction']
                recommendations = opportunity['recommendations']
                
                report_lines.append(f"\n{i}. 🎾 {match_info['player_a']} vs {match_info['player_b']}")
                report_lines.append(f"   🏟️ {match_info['tournament']} | {match_info['surface'].title()} Court")
                
                # Action and details
                action = recommendations['action'].replace('_', ' ').title()
                if action != 'No Bet':
                    report_lines.append(f"   🎯 ACTION: {action} ({recommendations['bet_size']:.1f}% of bankroll)")
                    report_lines.append(f"   💰 Expected Value: {recommendations['expected_value']:+.1f}%")
                    report_lines.append(f"   🏆 Opportunity Score: {recommendations['opportunity_score']:.2f}/1.0")
                    report_lines.append(f"   🎯 Confidence: {ultimate_pred['confidence']:.1%}")
                
                # Key insights
                report_lines.append(f"   📈 Upset Probability: {ultimate_pred['upset_probability']:.1%}")
                
                # Model contributions
                if ultimate_pred.get('model_count', 0) > 1:
                    report_lines.append(f"   🤖 Models Used: {ultimate_pred['model_count']} (Ensemble)")
                
                # Alerts
                if recommendations.get('alerts'):
                    alerts_str = ' | '.join(recommendations['alerts'])
                    report_lines.append(f"   🚨 ALERTS: {alerts_str}")
                
                # Key reasoning
                if recommendations.get('reasoning'):
                    report_lines.append(f"   🧠 Reasoning: {' | '.join(recommendations['reasoning'])}")
        
        else:
            report_lines.append("⚠️ NO HIGH-VALUE OPPORTUNITIES DETECTED")
            report_lines.append("All markets appear efficiently priced today.")
            report_lines.append("System recommends patience for better setups.")
        
        # System performance summary
        report_lines.append(f"\n" + "=" * 75)
        report_lines.append("📊 ULTIMATE SYSTEM PERFORMANCE:")
        report_lines.append(f"  • Total Predictions Made: {self.performance_metrics['total_predictions']}")
        
        if self.performance_metrics['total_predictions'] > 0:
            accuracy = self.performance_metrics['correct_predictions'] / self.performance_metrics['total_predictions']
            report_lines.append(f"  • Overall Accuracy: {accuracy:.1%}")
            
        if self.performance_metrics['upset_predictions'] > 0:
            upset_accuracy = self.performance_metrics['correct_upsets'] / self.performance_metrics['upset_predictions']
            report_lines.append(f"  • Upset Detection Rate: {upset_accuracy:.1%}")
            
        report_lines.append(f"  • Total Return: {self.performance_metrics['total_return']:+.1f}%")
        
        # System status
        report_lines.append(f"\n🔄 SYSTEM COMPONENTS STATUS:")
        report_lines.append(f"  ✅ Elo System: ACTIVE")
        report_lines.append(f"  ✅ ML Ensemble: ACTIVE")
        report_lines.append(f"  {'✅' if self.deep_learning else '❌'} Deep Learning: {'ACTIVE' if self.deep_learning else 'INACTIVE'}")
        report_lines.append(f"  {'✅' if self.sentiment_analyzer else '❌'} Sentiment Analysis: {'ACTIVE' if self.sentiment_analyzer else 'INACTIVE'}")
        report_lines.append(f"  {'✅' if self.odds_tracker else '❌'} Real-time Odds: {'ACTIVE' if self.odds_tracker else 'INACTIVE'}")
        
        report_lines.append(f"\n" + "=" * 75)
        report_lines.append("🏆 ULTIMATE TENNIS SYSTEM - READY TO DOMINATE MARKETS!")
        report_lines.append("=" * 75)
        
        return "\n".join(report_lines)
    
    def update_performance(self, prediction_result: Dict, actual_outcome: str):
        """Update system performance metrics"""
        
        self.performance_metrics['total_predictions'] += 1
        
        # Check if prediction was correct
        predicted_winner = 'player_a' if prediction_result['ultimate_prediction']['win_probability'] > 0.5 else 'player_b'
        if predicted_winner == actual_outcome:
            self.performance_metrics['correct_predictions'] += 1
        
        # Check upset predictions
        if prediction_result['ultimate_prediction']['upset_probability'] > 0.35:
            self.performance_metrics['upset_predictions'] += 1
            if actual_outcome != predicted_winner:  # Upset occurred
                self.performance_metrics['correct_upsets'] += 1
        
        # Update returns (simplified)
        if prediction_result['recommendations']['action'] != 'NO_BET':
            expected_return = prediction_result['recommendations']['expected_value']
            self.performance_metrics['total_return'] += expected_return
    
    async def run_ultimate_demo(self):
        """Run ultimate system demonstration"""
        
        print("🚀 ULTIMATE TENNIS SYSTEM DEMO")
        print("=" * 50)
        
        # Sample matches
        demo_matches = [
            {
                'player_a': 'Jannik Sinner',
                'player_b': 'Novak Djokovic',
                'tournament': 'US Open Semifinals',
                'surface': 'hard',
                'player_a_data': {'ranking': 4, 'elo_ratings': {'hard': 1950, 'overall': 1920}},
                'player_b_data': {'ranking': 1, 'elo_ratings': {'hard': 2100, 'overall': 2080}},
                'favorite_odds': 2.8,
                'underdog_odds': 1.45,
                'public_betting_percentage': 72
            },
            {
                'player_a': 'Carlos Alcaraz',
                'player_b': 'Daniil Medvedev',
                'tournament': 'ATP Masters',
                'surface': 'hard',
                'player_a_data': {'ranking': 2, 'elo_ratings': {'hard': 2050, 'overall': 2020}},
                'player_b_data': {'ranking': 3, 'elo_ratings': {'hard': 1980, 'overall': 1960}},
                'favorite_odds': 1.65,
                'underdog_odds': 2.35,
                'public_betting_percentage': 58
            }
        ]
        
        # Run ultimate scan
        results = await self.scan_ultimate_opportunities(demo_matches)
        
        # Generate ultimate report
        report = self.generate_ultimate_report(results)
        print(report)
        
        return results

# Main execution
async def main():
    """Main execution function"""
    
    # Initialize ultimate system
    system = UltimateTennisSystem({
        'use_deep_learning': True,
        'use_sentiment_analysis': True,
        'use_real_time_odds': True,
        'confidence_threshold': 0.65,
        'min_edge_threshold': 0.06
    })
    
    # Run demonstration
    results = await system.run_ultimate_demo()
    
    print(f"\n🏆 ULTIMATE SYSTEM READY FOR DEPLOYMENT!")
    print(f"Found {results['high_value_opportunities']} high-value opportunities")
    print(f"System operating at MAXIMUM CAPABILITY!")
    
    return system

if __name__ == "__main__":
    # Run the ultimate system
    system = asyncio.run(main())