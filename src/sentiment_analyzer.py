"""
Advanced Sentiment Analysis and News Intelligence System
Monitors tennis news, social media, and market sentiment to detect
factors that betting markets haven't yet incorporated into odds
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
import re
import time
from collections import defaultdict

# NLP and sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    print("Tweepy not available. Install with: pip install tweepy")

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis system for tennis prediction
    Analyzes news, social media, and expert opinions to detect
    market-moving information before it's reflected in odds
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'news_sources': [
                'espn.com', 'tennischannel.com', 'atptour.com',
                'wtatennis.com', 'eurosport.com', 'tennis.com'
            ],
            'update_frequency_minutes': 30,
            'sentiment_threshold': 0.6,
            'news_lookback_hours': 24
        }
        
        self.sentiment_pipeline = None
        self.news_cache = defaultdict(list)
        self.sentiment_history = defaultdict(list)
        
        # Initialize sentiment models
        self._initialize_sentiment_models()
        
        # Tennis-specific keywords and their impact weights
        self.impact_keywords = {
            # Injury/Health
            'injury': -0.8, 'injured': -0.8, 'hurt': -0.6, 'pain': -0.5,
            'recovered': 0.6, 'healthy': 0.5, 'fitness': 0.4, 'treatment': -0.3,
            
            # Form/Performance
            'confident': 0.7, 'struggling': -0.6, 'dominating': 0.8, 'unstoppable': 0.9,
            'tired': -0.5, 'exhausted': -0.7, 'fresh': 0.6, 'motivated': 0.5,
            
            # Mental State
            'focused': 0.6, 'distracted': -0.5, 'determined': 0.7, 'frustrated': -0.6,
            'pressure': -0.4, 'relaxed': 0.5, 'nervous': -0.6, 'calm': 0.4,
            
            # Career/Personal
            'retirement': -0.9, 'comeback': 0.7, 'divorce': -0.3, 'wedding': -0.2,
            'coaching change': -0.4, 'new coach': 0.3, 'training': 0.4,
            
            # Tournament Context
            'favorite': 0.3, 'underdog': -0.2, 'upset': -0.4, 'champion': 0.6,
            'defending': 0.2, 'challenging': 0.1, 'breakthrough': 0.7
        }
    
    def _initialize_sentiment_models(self):
        """Initialize advanced sentiment analysis models"""
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use financial/sports sentiment model for better accuracy
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                print("✅ Advanced sentiment analysis models loaded")
                
            except Exception as e:
                print(f"⚠️ Fallback to basic sentiment analysis: {e}")
                self.sentiment_pipeline = None
        else:
            print("💡 Install transformers for advanced sentiment analysis")
    
    def analyze_player_news_sentiment(self, player_name: str, hours_lookback: int = 24) -> Dict[str, float]:
        """Analyze news sentiment for a specific player"""
        
        # Collect recent news
        news_articles = self._collect_player_news(player_name, hours_lookback)
        
        if not news_articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'article_count': 0,
                'sentiment_trend': 'neutral',
                'key_factors': []
            }
        
        # Analyze sentiment for each article
        sentiment_scores = []
        impact_factors = []
        
        for article in news_articles:
            article_sentiment = self._analyze_article_sentiment(article)
            sentiment_scores.append(article_sentiment)
            
            # Extract impact factors
            factors = self._extract_impact_factors(article)
            impact_factors.extend(factors)
        
        # Aggregate sentiment
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        sentiment_strength = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.5
        
        # Determine trend
        recent_sentiment = np.mean(sentiment_scores[:3]) if len(sentiment_scores) >= 3 else overall_sentiment
        older_sentiment = np.mean(sentiment_scores[3:]) if len(sentiment_scores) > 3 else overall_sentiment
        
        if recent_sentiment > older_sentiment + 0.1:
            trend = 'improving'
        elif recent_sentiment < older_sentiment - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_strength': sentiment_strength,
            'article_count': len(news_articles),
            'sentiment_trend': trend,
            'key_factors': self._rank_impact_factors(impact_factors),
            'recent_sentiment': recent_sentiment,
            'sentiment_volatility': sentiment_strength
        }
    
    def _collect_player_news(self, player_name: str, hours_lookback: int) -> List[Dict]:
        """Collect recent news articles about a player"""
        
        # For demo purposes, generate realistic news data
        sample_news = [
            {
                'title': f'{player_name} shows strong form in practice sessions',
                'content': f'{player_name} looked confident and focused during training, hitting powerful groundstrokes with precision. The player appears to be in excellent physical condition.',
                'source': 'ESPN Tennis',
                'timestamp': datetime.now() - timedelta(hours=2),
                'url': 'https://example.com/news1'
            },
            {
                'title': f'{player_name} addresses recent coaching changes',
                'content': f'In a press conference, {player_name} discussed the new training regimen and expressed optimism about the upcoming matches.',
                'source': 'Tennis Channel',
                'timestamp': datetime.now() - timedelta(hours=8),
                'url': 'https://example.com/news2'
            },
            {
                'title': f'Injury concerns for {player_name}',
                'content': f'Medical staff have been monitoring {player_name} closely after reporting some discomfort during yesterday\'s practice.',
                'source': 'ATP Tour',
                'timestamp': datetime.now() - timedelta(hours=12),
                'url': 'https://example.com/news3'
            }
        ]
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        return [article for article in sample_news if article['timestamp'] > cutoff_time]
    
    def _analyze_article_sentiment(self, article: Dict) -> float:
        """Analyze sentiment of a single article"""
        
        text = f"{article['title']} {article['content']}"
        
        if self.sentiment_pipeline:
            try:
                # Use advanced transformer model
                result = self.sentiment_pipeline(text[:512])  # Truncate for model limits
                
                # Convert to numeric score (-1 to 1)
                if result[0]['label'] == 'LABEL_2':  # Positive
                    return result[0]['score']
                elif result[0]['label'] == 'LABEL_0':  # Negative
                    return -result[0]['score']
                else:  # Neutral
                    return 0.0
                    
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                return self._basic_sentiment_analysis(text)
        else:
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic keyword-based sentiment analysis"""
        
        text_lower = text.lower()
        sentiment_score = 0.0
        
        # Check for positive keywords
        positive_words = ['confident', 'strong', 'excellent', 'optimistic', 'great', 'perfect', 'amazing', 'outstanding']
        negative_words = ['injury', 'hurt', 'struggling', 'tired', 'poor', 'weak', 'disappointed', 'concerned']
        
        for word in positive_words:
            sentiment_score += text_lower.count(word) * 0.3
            
        for word in negative_words:
            sentiment_score -= text_lower.count(word) * 0.4
            
        # Apply impact keywords with weights
        for keyword, weight in self.impact_keywords.items():
            if keyword in text_lower:
                sentiment_score += weight
        
        # Normalize to [-1, 1]
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _extract_impact_factors(self, article: Dict) -> List[Dict]:
        """Extract key factors that could impact match outcome"""
        
        text = f"{article['title']} {article['content']}".lower()
        factors = []
        
        for keyword, impact in self.impact_keywords.items():
            if keyword in text:
                factors.append({
                    'factor': keyword,
                    'impact_score': impact,
                    'source': article['source'],
                    'timestamp': article['timestamp']
                })
        
        return factors
    
    def _rank_impact_factors(self, factors: List[Dict]) -> List[Dict]:
        """Rank and aggregate impact factors by importance"""
        
        if not factors:
            return []
        
        # Aggregate by factor type
        factor_aggregates = defaultdict(list)
        for factor in factors:
            factor_aggregates[factor['factor']].append(factor['impact_score'])
        
        # Calculate average impact and recency weight
        ranked_factors = []
        for factor_name, impacts in factor_aggregates.items():
            avg_impact = np.mean(impacts)
            factor_count = len(impacts)
            
            ranked_factors.append({
                'factor': factor_name,
                'average_impact': avg_impact,
                'mention_count': factor_count,
                'importance_score': abs(avg_impact) * np.log(1 + factor_count)
            })
        
        # Sort by importance
        return sorted(ranked_factors, key=lambda x: x['importance_score'], reverse=True)[:5]
    
    def analyze_market_sentiment(self, match_data: Dict) -> Dict[str, float]:
        """Analyze overall market sentiment for a match"""
        
        player_a = match_data.get('player_a', 'Player A')
        player_b = match_data.get('player_b', 'Player B')
        
        # Get sentiment for both players
        sentiment_a = self.analyze_player_news_sentiment(player_a)
        sentiment_b = self.analyze_player_news_sentiment(player_b)
        
        # Calculate relative sentiment advantage
        sentiment_differential = sentiment_a['overall_sentiment'] - sentiment_b['overall_sentiment']
        
        # Analyze betting market sentiment
        public_betting_pct = match_data.get('public_betting_percentage', 50)
        line_movement = match_data.get('line_movement', 0)
        
        # Market sentiment indicators
        public_bias = abs(public_betting_pct - 50) / 50  # 0 to 1 scale
        sharp_money_indicator = abs(line_movement) > 0.05  # Significant line movement
        
        # Social media buzz (simulated)
        social_buzz_a = self._analyze_social_buzz(player_a)
        social_buzz_b = self._analyze_social_buzz(player_b)
        
        return {
            'sentiment_differential': sentiment_differential,
            'player_a_sentiment': sentiment_a['overall_sentiment'],
            'player_b_sentiment': sentiment_b['overall_sentiment'],
            'market_bias_strength': public_bias,
            'sharp_money_detected': sharp_money_indicator,
            'social_buzz_ratio': social_buzz_a / max(social_buzz_b, 1),
            'combined_sentiment_score': self._calculate_combined_sentiment(
                sentiment_differential, public_bias, social_buzz_a, social_buzz_b
            ),
            'sentiment_reliability': min(sentiment_a['article_count'], sentiment_b['article_count']) / 10,
            'key_factors_a': sentiment_a['key_factors'][:3],
            'key_factors_b': sentiment_b['key_factors'][:3]
        }
    
    def _analyze_social_buzz(self, player_name: str) -> float:
        """Analyze social media buzz/mentions for a player"""
        
        # Simulated social media analysis
        # In production, this would connect to Twitter API, Reddit, etc.
        
        base_buzz = np.random.uniform(50, 200)  # Base mention count
        
        # Adjust based on player popularity (simplified)
        popularity_multiplier = 1.0
        if any(name in player_name.lower() for name in ['djokovic', 'federer', 'nadal', 'alcaraz']):
            popularity_multiplier = 2.0
        elif any(name in player_name.lower() for name in ['sinner', 'medvedev', 'zverev']):
            popularity_multiplier = 1.5
        
        return base_buzz * popularity_multiplier
    
    def _calculate_combined_sentiment_score(self, 
                                          sentiment_diff: float,
                                          market_bias: float,
                                          buzz_a: float,
                                          buzz_b: float) -> float:
        """Calculate combined sentiment score for betting value"""
        
        # Weight different sentiment components
        news_weight = 0.4
        market_weight = 0.3
        social_weight = 0.3
        
        # Normalize buzz ratio
        buzz_ratio = (buzz_a - buzz_b) / (buzz_a + buzz_b + 1)
        
        combined_score = (
            sentiment_diff * news_weight +
            market_bias * market_weight * (1 if sentiment_diff > 0 else -1) +
            buzz_ratio * social_weight
        )
        
        return np.clip(combined_score, -1.0, 1.0)
    
    def detect_sentiment_arbitrage_opportunities(self, matches: List[Dict]) -> List[Dict]:
        """Detect matches where sentiment analysis reveals betting value"""
        
        opportunities = []
        
        for match in matches:
            sentiment_analysis = self.analyze_market_sentiment(match)
            
            # Look for sentiment/odds mismatches
            sentiment_score = sentiment_analysis['combined_sentiment_score']
            
            # Get market odds
            favorite_odds = match.get('favorite_odds', 1.8)
            underdog_odds = match.get('underdog_odds', 2.2)
            
            # Calculate implied probabilities
            favorite_prob = 1 / favorite_odds
            underdog_prob = 1 / underdog_odds
            
            # Detect opportunities
            opportunity_score = 0
            
            # Strong positive sentiment for underdog
            if sentiment_score > 0.3 and underdog_prob < 0.4:
                opportunity_score = sentiment_score * (0.4 - underdog_prob)
                recommendation = 'back_underdog'
                
            # Strong negative sentiment for favorite
            elif sentiment_score < -0.3 and favorite_prob > 0.6:
                opportunity_score = abs(sentiment_score) * (favorite_prob - 0.6)
                recommendation = 'fade_favorite'
            
            if opportunity_score > 0.15:  # Minimum threshold
                opportunities.append({
                    'match': f"{match.get('player_a', 'A')} vs {match.get('player_b', 'B')}",
                    'opportunity_score': opportunity_score,
                    'recommendation': recommendation,
                    'sentiment_score': sentiment_score,
                    'sentiment_analysis': sentiment_analysis,
                    'expected_edge': opportunity_score * 100
                })
        
        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)
    
    def generate_sentiment_report(self, match_data: Dict) -> str:
        """Generate comprehensive sentiment analysis report"""
        
        analysis = self.analyze_market_sentiment(match_data)
        
        player_a = match_data.get('player_a', 'Player A')
        player_b = match_data.get('player_b', 'Player B')
        
        report = f"""
📈 SENTIMENT ANALYSIS REPORT
================================
🎾 Match: {player_a} vs {player_b}
📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📰 NEWS SENTIMENT:
  • {player_a}: {analysis['player_a_sentiment']:+.2f} ({self._sentiment_label(analysis['player_a_sentiment'])})
  • {player_b}: {analysis['player_b_sentiment']:+.2f} ({self._sentiment_label(analysis['player_b_sentiment'])})
  • Sentiment Advantage: {analysis['sentiment_differential']:+.2f}

💰 MARKET INDICATORS:
  • Public Bias Strength: {analysis['market_bias_strength']:.1%}
  • Sharp Money Activity: {'YES' if analysis['sharp_money_detected'] else 'NO'}
  • Social Buzz Ratio: {analysis['social_buzz_ratio']:.1f}:1

🎯 COMBINED ANALYSIS:
  • Overall Sentiment Score: {analysis['combined_sentiment_score']:+.2f}
  • Reliability Score: {analysis['sentiment_reliability']:.1%}
"""
        
        # Add key factors
        if analysis['key_factors_a']:
            report += f"\n🔍 KEY FACTORS - {player_a}:\n"
            for factor in analysis['key_factors_a']:
                impact = "🟢" if factor['average_impact'] > 0 else "🔴" if factor['average_impact'] < 0 else "🟡"
                report += f"  {impact} {factor['factor'].title()}: {factor['average_impact']:+.2f}\n"
        
        if analysis['key_factors_b']:
            report += f"\n🔍 KEY FACTORS - {player_b}:\n"
            for factor in analysis['key_factors_b']:
                impact = "🟢" if factor['average_impact'] > 0 else "🔴" if factor['average_impact'] < 0 else "🟡"
                report += f"  {impact} {factor['factor'].title()}: {factor['average_impact']:+.2f}\n"
        
        # Trading recommendations
        report += f"\n💡 SENTIMENT-BASED RECOMMENDATIONS:\n"
        
        if abs(analysis['combined_sentiment_score']) > 0.3:
            if analysis['combined_sentiment_score'] > 0:
                report += f"  ✅ Consider backing {player_a} - positive sentiment edge\n"
            else:
                report += f"  ✅ Consider backing {player_b} - negative sentiment for opponent\n"
            report += f"  🎯 Estimated Edge: {abs(analysis['combined_sentiment_score']) * 10:.1f}%\n"
        else:
            report += "  ⚠️ No significant sentiment edge detected\n"
        
        return report
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to descriptive label"""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def monitor_breaking_news(self, players: List[str], callback=None) -> None:
        """Monitor for breaking news that could affect match odds"""
        
        print(f"📡 Monitoring breaking news for {len(players)} players...")
        
        # In production, this would set up real-time monitoring
        # For demo, simulate breaking news detection
        
        breaking_news_types = [
            "injury_update", "coaching_change", "personal_news", "training_update", "controversy"
        ]
        
        for player in players:
            # Simulate random breaking news (low probability)
            if np.random.random() < 0.1:  # 10% chance
                news_type = np.random.choice(breaking_news_types)
                
                alert = {
                    'player': player,
                    'type': news_type,
                    'timestamp': datetime.now(),
                    'impact_score': np.random.uniform(-0.8, 0.8),
                    'urgency': 'HIGH' if abs(np.random.uniform(-1, 1)) > 0.7 else 'MEDIUM'
                }
                
                if callback:
                    callback(alert)
                else:
                    print(f"🚨 BREAKING: {news_type.replace('_', ' ').title()} for {player}")
    
    def real_time_sentiment_stream(self, match_id: str, duration_minutes: int = 60):
        """Stream real-time sentiment changes during match buildup"""
        
        print(f"📶 Starting real-time sentiment monitoring for {duration_minutes} minutes...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        sentiment_timeline = []
        
        while datetime.now() < end_time:
            # Simulate real-time sentiment changes
            current_sentiment = {
                'timestamp': datetime.now(),
                'overall_sentiment': np.random.normal(0, 0.3),
                'news_volume': np.random.poisson(5),
                'social_volume': np.random.poisson(20),
                'market_movement': np.random.normal(0, 0.02)
            }
            
            sentiment_timeline.append(current_sentiment)
            
            # Check for significant changes
            if len(sentiment_timeline) > 5:
                recent_trend = np.mean([s['overall_sentiment'] for s in sentiment_timeline[-3:]])
                older_trend = np.mean([s['overall_sentiment'] for s in sentiment_timeline[-6:-3]])
                
                if abs(recent_trend - older_trend) > 0.3:
                    print(f"🚨 SENTIMENT ALERT: Trend shift detected ({recent_trend:+.2f})")
            
            time.sleep(30)  # Update every 30 seconds
        
        return sentiment_timeline

# Demo function
def demo_sentiment_system():
    """Demonstrate the advanced sentiment analysis system"""
    
    print("📈 ADVANCED SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    # Initialize system
    sentiment_analyzer = AdvancedSentimentAnalyzer()
    
    # Sample match data
    sample_match = {
        'player_a': 'Jannik Sinner',
        'player_b': 'Novak Djokovic',
        'favorite_odds': 1.65,
        'underdog_odds': 2.35,
        'public_betting_percentage': 72,
        'line_movement': -0.08
    }
    
    # Generate sentiment report
    report = sentiment_analyzer.generate_sentiment_report(sample_match)
    print(report)
    
    # Check for arbitrage opportunities
    opportunities = sentiment_analyzer.detect_sentiment_arbitrage_opportunities([sample_match])
    
    if opportunities:
        print(f"\n💰 SENTIMENT ARBITRAGE OPPORTUNITIES:")
        for opp in opportunities:
            print(f"  🎯 {opp['match']}: {opp['recommendation']} ({opp['expected_edge']:.1f}% edge)")
    else:
        print(f"\n🔍 No sentiment arbitrage opportunities detected")
    
    return sentiment_analyzer

if __name__ == "__main__":
    demo_sentiment_system()