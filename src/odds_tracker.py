"""
Advanced Real-Time Odds Tracking and Arbitrage Detection System
Tracks odds across multiple bookmakers, detects arbitrage opportunities,
monitors line movements, and identifies sharp money vs public betting patterns
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
import sqlite3
from dataclasses import dataclass
import threading

@dataclass
class OddsSnapshot:
    """Represents a single odds snapshot"""
    timestamp: datetime
    bookmaker: str
    match_id: str
    player_a_odds: float
    player_b_odds: float
    volume_indicator: float = 0.0
    market_type: str = 'match_winner'
    
@dataclass 
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    match_id: str
    profit_margin: float
    stake_distribution: Dict[str, float]
    bookmakers: List[str]
    odds: Dict[str, float]
    expiry_estimate: datetime
    confidence_score: float

class AdvancedOddsTracker:
    """
    Advanced real-time odds tracking and arbitrage detection system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'update_frequency': 30,  # seconds
            'arbitrage_threshold': 0.01,  # 1% minimum profit
            'max_history_days': 7,
            'sharp_move_threshold': 0.05,  # 5% odds change
            'volume_weight': 0.3
        }
        
        # Bookmaker configurations
        self.bookmakers = {
            'pinnacle': {'priority': 1, 'reliability': 0.95, 'limits': 'high'},
            'bet365': {'priority': 2, 'reliability': 0.90, 'limits': 'medium'},
            'betfair': {'priority': 1, 'reliability': 0.98, 'limits': 'high'},
            'william_hill': {'priority': 3, 'reliability': 0.85, 'limits': 'medium'},
            'unibet': {'priority': 3, 'reliability': 0.88, 'limits': 'medium'},
            'draftkings': {'priority': 2, 'reliability': 0.87, 'limits': 'medium'},
            'fanduel': {'priority': 2, 'reliability': 0.86, 'limits': 'medium'}
        }
        
        # Data storage
        self.odds_history = defaultdict(lambda: defaultdict(deque))
        self.arbitrage_opportunities = []
        self.line_movements = defaultdict(list)
        self.sharp_money_alerts = []
        
        # Real-time monitoring
        self.active_monitoring = False
        self.monitoring_tasks = []
        self.alert_callbacks = []
        
        # Database for persistence
        self._init_database()
        
        print("📈 Advanced Odds Tracking System Initialized")
        print(f"  🎯 Monitoring {len(self.bookmakers)} bookmakers")
        print(f"  ⚡ Update frequency: {self.config['update_frequency']}s")
        print(f"  💰 Arbitrage threshold: {self.config['arbitrage_threshold']:.1%}")
    
    def _init_database(self):
        """Initialize SQLite database for odds storage"""
        self.db_connection = sqlite3.connect('tennis_odds.db', check_same_thread=False)
        
        # Create tables
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                bookmaker TEXT,
                match_id TEXT,
                player_a_odds REAL,
                player_b_odds REAL,
                volume_indicator REAL,
                market_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                match_id TEXT,
                profit_margin REAL,
                bookmakers TEXT,
                odds_data TEXT,
                confidence_score REAL
            )
        ''')
        
        self.db_connection.commit()
        print("🖺 Database initialized for odds persistence")
    
    async def track_match_odds(self, match_id: str, duration_hours: int = 24) -> List[OddsSnapshot]:
        """Track odds for a specific match over time"""
        
        print(f"🔍 Starting {duration_hours}h odds tracking for match: {match_id}")
        
        snapshots = []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            # Collect odds from all bookmakers
            current_snapshots = await self._collect_all_bookmaker_odds(match_id)
            snapshots.extend(current_snapshots)
            
            # Store in database
            self._store_odds_snapshots(current_snapshots)
            
            # Check for arbitrage opportunities
            arbitrage_ops = self._detect_arbitrage_opportunities(current_snapshots)
            if arbitrage_ops:
                self.arbitrage_opportunities.extend(arbitrage_ops)
                await self._alert_arbitrage_opportunities(arbitrage_ops)
            
            # Detect line movements
            movements = self._detect_significant_movements(match_id, current_snapshots)
            if movements:
                self.line_movements[match_id].extend(movements)
                await self._alert_line_movements(movements)
            
            # Wait for next update
            await asyncio.sleep(self.config['update_frequency'])
        
        print(f"✅ Completed odds tracking. Collected {len(snapshots)} snapshots")
        return snapshots
    
    async def _collect_all_bookmaker_odds(self, match_id: str) -> List[OddsSnapshot]:
        """Collect current odds from all bookmakers"""
        
        snapshots = []
        timestamp = datetime.now()
        
        # Simulate collecting from multiple bookmakers
        for bookmaker, config in self.bookmakers.items():
            try:
                odds_data = await self._fetch_bookmaker_odds(bookmaker, match_id)
                
                if odds_data:
                    snapshot = OddsSnapshot(
                        timestamp=timestamp,
                        bookmaker=bookmaker,
                        match_id=match_id,
                        player_a_odds=odds_data['player_a_odds'],
                        player_b_odds=odds_data['player_b_odds'],
                        volume_indicator=odds_data.get('volume', 0.5)
                    )
                    snapshots.append(snapshot)
                    
            except Exception as e:
                print(f"⚠️ Error fetching {bookmaker} odds: {e}")
                continue
        
        return snapshots
    
    async def _fetch_bookmaker_odds(self, bookmaker: str, match_id: str) -> Optional[Dict]:
        """Fetch odds from a specific bookmaker (simulated)"""
        
        # Simulate network delay and occasional failures
        await asyncio.sleep(np.random.uniform(0.1, 0.5))
        
        if np.random.random() < 0.05:  # 5% failure rate
            raise Exception(f"Network error for {bookmaker}")
        
        # Generate realistic odds with bookmaker-specific characteristics
        base_odds_a = np.random.uniform(1.4, 3.5)
        base_odds_b = np.random.uniform(1.4, 3.5)
        
        # Adjust for bookmaker characteristics
        if bookmaker == 'pinnacle':
            # Pinnacle typically has the sharpest lines
            margin = 0.02
        elif bookmaker == 'betfair':
            # Exchange, very competitive
            margin = 0.01
        else:
            # Traditional bookmakers with higher margins
            margin = np.random.uniform(0.04, 0.08)
        
        # Apply margin to odds
        total_prob = (1/base_odds_a) + (1/base_odds_b)
        adjusted_total_prob = total_prob + margin
        
        final_odds_a = base_odds_a / (total_prob / adjusted_total_prob)
        final_odds_b = base_odds_b / (total_prob / adjusted_total_prob)
        
        return {
            'player_a_odds': round(final_odds_a, 2),
            'player_b_odds': round(final_odds_b, 2),
            'volume': np.random.uniform(0.2, 1.0),
            'timestamp': datetime.now()
        }
    
    def _store_odds_snapshots(self, snapshots: List[OddsSnapshot]):
        """Store odds snapshots in database"""
        
        cursor = self.db_connection.cursor()
        
        for snapshot in snapshots:
            cursor.execute('''
                INSERT INTO odds_snapshots 
                (timestamp, bookmaker, match_id, player_a_odds, player_b_odds, volume_indicator, market_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp,
                snapshot.bookmaker,
                snapshot.match_id,
                snapshot.player_a_odds,
                snapshot.player_b_odds,
                snapshot.volume_indicator,
                snapshot.market_type
            ))
        
        self.db_connection.commit()
    
    def _detect_arbitrage_opportunities(self, snapshots: List[OddsSnapshot]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities from current odds snapshots"""
        
        if len(snapshots) < 2:
            return []
        
        opportunities = []
        
        # Group by match_id
        matches = defaultdict(list)
        for snapshot in snapshots:
            matches[snapshot.match_id].append(snapshot)
        
        for match_id, match_snapshots in matches.items():
            # Find best odds for each outcome
            best_odds_a = max(match_snapshots, key=lambda x: x.player_a_odds)
            best_odds_b = max(match_snapshots, key=lambda x: x.player_b_odds)
            
            # Calculate arbitrage
            prob_a = 1 / best_odds_a.player_a_odds
            prob_b = 1 / best_odds_b.player_b_odds
            total_prob = prob_a + prob_b
            
            if total_prob < 1.0:
                profit_margin = 1.0 - total_prob
                
                if profit_margin >= self.config['arbitrage_threshold']:
                    # Calculate stake distribution
                    stake_a = prob_a / total_prob
                    stake_b = prob_b / total_prob
                    
                    # Assess confidence based on bookmaker reliability
                    reliability_a = self.bookmakers[best_odds_a.bookmaker]['reliability']
                    reliability_b = self.bookmakers[best_odds_b.bookmaker]['reliability']
                    confidence = (reliability_a + reliability_b) / 2
                    
                    opportunity = ArbitrageOpportunity(
                        match_id=match_id,
                        profit_margin=profit_margin,
                        stake_distribution={'player_a': stake_a, 'player_b': stake_b},
                        bookmakers=[best_odds_a.bookmaker, best_odds_b.bookmaker],
                        odds={'player_a': best_odds_a.player_a_odds, 'player_b': best_odds_b.player_b_odds},
                        expiry_estimate=datetime.now() + timedelta(minutes=5),  # Estimate
                        confidence_score=confidence
                    )
                    
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_significant_movements(self, match_id: str, current_snapshots: List[OddsSnapshot]) -> List[Dict]:
        """Detect significant line movements that may indicate sharp money"""
        
        movements = []
        
        # Compare with recent history
        for snapshot in current_snapshots:
            bookmaker = snapshot.bookmaker
            
            # Get recent history for this bookmaker/match
            history_key = f"{match_id}_{bookmaker}"
            recent_history = list(self.odds_history[history_key])[-10:]  # Last 10 snapshots
            
            if len(recent_history) >= 3:
                # Calculate movement
                avg_recent_a = np.mean([s.player_a_odds for s in recent_history[-3:]])
                avg_older_a = np.mean([s.player_a_odds for s in recent_history[:3]])
                
                movement_a = (snapshot.player_a_odds - avg_recent_a) / avg_recent_a
                
                if abs(movement_a) >= self.config['sharp_move_threshold']:
                    # Assess if this looks like sharp money
                    volume_increase = snapshot.volume_indicator > np.mean([s.volume_indicator for s in recent_history])
                    opposite_movement = movement_a * (snapshot.player_b_odds / np.mean([s.player_b_odds for s in recent_history[-3:]])) < 0
                    
                    sharp_money_probability = 0.5  # Base probability
                    if volume_increase:
                        sharp_money_probability += 0.3
                    if opposite_movement:
                        sharp_money_probability += 0.2
                    if bookmaker in ['pinnacle', 'betfair']:  # Sharp bookmakers
                        sharp_money_probability += 0.2
                    
                    movement = {
                        'match_id': match_id,
                        'bookmaker': bookmaker,
                        'movement_size': movement_a,
                        'direction': 'player_a' if movement_a > 0 else 'player_b',
                        'sharp_money_probability': min(sharp_money_probability, 0.95),
                        'timestamp': snapshot.timestamp,
                        'volume_indicator': snapshot.volume_indicator
                    }
                    
                    movements.append(movement)
            
            # Store current snapshot in history
            self.odds_history[history_key].append(snapshot)
            if len(self.odds_history[history_key]) > 100:  # Keep only recent history
                self.odds_history[history_key].popleft()
        
        return movements
    
    async def _alert_arbitrage_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Alert about arbitrage opportunities"""
        
        for opp in opportunities:
            alert_message = f"💰 ARBITRAGE ALERT: {opp.profit_margin:.2%} profit opportunity on {opp.match_id}"
            print(alert_message)
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO arbitrage_opportunities 
                (timestamp, match_id, profit_margin, bookmakers, odds_data, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                opp.match_id,
                opp.profit_margin,
                json.dumps(opp.bookmakers),
                json.dumps(opp.odds),
                opp.confidence_score
            ))
            self.db_connection.commit()
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                await callback('arbitrage', opp)
    
    async def _alert_line_movements(self, movements: List[Dict]):
        """Alert about significant line movements"""
        
        for movement in movements:
            if movement['sharp_money_probability'] > 0.7:
                alert_message = f"🚨 SHARP MONEY ALERT: {movement['movement_size']:.1%} move on {movement['match_id']} at {movement['bookmaker']}"
                print(alert_message)
                
                # Trigger callbacks
                for callback in self.alert_callbacks:
                    await callback('line_movement', movement)
    
    def calculate_closing_line_value(self, match_id: str, bet_odds: float, bet_side: str) -> float:
        """Calculate Closing Line Value (CLV) - key metric for sharp bettors"""
        
        # Get final odds before match start
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT bookmaker, player_a_odds, player_b_odds 
            FROM odds_snapshots 
            WHERE match_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 20
        ''', (match_id,))
        
        recent_odds = cursor.fetchall()
        
        if not recent_odds:
            return 0.0
        
        # Calculate average closing odds from sharp bookmakers
        sharp_bookmakers = ['pinnacle', 'betfair']
        sharp_odds = []
        
        for bookmaker, odds_a, odds_b in recent_odds:
            if bookmaker in sharp_bookmakers:
                if bet_side == 'player_a':
                    sharp_odds.append(odds_a)
                else:
                    sharp_odds.append(odds_b)
        
        if not sharp_odds:
            return 0.0
        
        avg_closing_odds = np.mean(sharp_odds)
        
        # Calculate CLV
        clv = (bet_odds - avg_closing_odds) / avg_closing_odds
        return clv
    
    def get_best_current_odds(self, match_id: str) -> Dict[str, Tuple[str, float]]:
        """Get current best odds for each outcome"""
        
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT bookmaker, player_a_odds, player_b_odds
            FROM odds_snapshots 
            WHERE match_id = ? AND timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''', (match_id,))
        
        recent_odds = cursor.fetchall()
        
        if not recent_odds:
            return {}
        
        best_odds_a = max(recent_odds, key=lambda x: x[1])
        best_odds_b = max(recent_odds, key=lambda x: x[2])
        
        return {
            'player_a': (best_odds_a[0], best_odds_a[1]),  # (bookmaker, odds)
            'player_b': (best_odds_b[0], best_odds_b[2])
        }
    
    def generate_odds_analysis_report(self, match_id: str) -> str:
        """Generate comprehensive odds analysis report"""
        
        cursor = self.db_connection.cursor()
        
        # Get odds history
        cursor.execute('''
            SELECT timestamp, bookmaker, player_a_odds, player_b_odds
            FROM odds_snapshots 
            WHERE match_id = ?
            ORDER BY timestamp
        ''', (match_id,))
        
        odds_history = cursor.fetchall()
        
        if not odds_history:
            return f"No odds data available for {match_id}"
        
        # Calculate statistics
        df = pd.DataFrame(odds_history, columns=['timestamp', 'bookmaker', 'odds_a', 'odds_b'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Best and worst odds
        best_odds_a = df['odds_a'].max()
        worst_odds_a = df['odds_a'].min()
        best_odds_b = df['odds_b'].max() 
        worst_odds_b = df['odds_b'].min()
        
        # Line movement analysis
        latest_odds = df.iloc[-1]
        earliest_odds = df.iloc[0]
        
        movement_a = (latest_odds['odds_a'] - earliest_odds['odds_a']) / earliest_odds['odds_a'] * 100
        movement_b = (latest_odds['odds_b'] - earliest_odds['odds_b']) / earliest_odds['odds_b'] * 100
        
        # Market efficiency analysis
        market_variance_a = df['odds_a'].var()
        market_variance_b = df['odds_b'].var()
        
        # Generate report
        report = f"""
📈 COMPREHENSIVE ODDS ANALYSIS REPORT
========================================
🎾 Match ID: {match_id}
📅 Analysis Period: {df['timestamp'].min()} to {df['timestamp'].max()}
📊 Data Points: {len(odds_history)} from {df['bookmaker'].nunique()} bookmakers

🎯 CURRENT BEST ODDS:
  • Player A: {best_odds_a:.2f}
  • Player B: {best_odds_b:.2f}

📉 LINE MOVEMENT:
  • Player A: {movement_a:+.1f}%
  • Player B: {movement_b:+.1f}%

💰 ODDS RANGE:
  • Player A: {worst_odds_a:.2f} - {best_odds_a:.2f} (spread: {best_odds_a - worst_odds_a:.2f})
  • Player B: {worst_odds_b:.2f} - {best_odds_b:.2f} (spread: {best_odds_b - worst_odds_b:.2f})

🏆 MARKET EFFICIENCY:
  • Player A Variance: {market_variance_a:.4f}
  • Player B Variance: {market_variance_b:.4f}
  • Market Status: {'Efficient' if market_variance_a + market_variance_b < 0.1 else 'Volatile'}
"""
        
        # Add arbitrage opportunities
        cursor.execute('''
            SELECT COUNT(*), AVG(profit_margin), MAX(profit_margin)
            FROM arbitrage_opportunities 
            WHERE match_id = ?
        ''', (match_id,))
        
        arb_stats = cursor.fetchone()
        if arb_stats[0] > 0:
            report += f"""

💰 ARBITRAGE OPPORTUNITIES:
  • Total Opportunities: {arb_stats[0]}
  • Average Profit: {arb_stats[1]:.2%}
  • Best Opportunity: {arb_stats[2]:.2%}
"""
        else:
            report += """

💰 ARBITRAGE OPPORTUNITIES:
  • No arbitrage opportunities detected
"""
        
        # Trading recommendations
        report += f"""

💡 TRADING RECOMMENDATIONS:
"""
        
        if movement_a > 5 or movement_b > 5:
            report += "  • Significant line movement detected - monitor for sharp money\n"
        
        if best_odds_a - worst_odds_a > 0.2 or best_odds_b - worst_odds_b > 0.2:
            report += "  • Large odds spreads - shop around for best value\n"
        
        if market_variance_a + market_variance_b > 0.1:
            report += "  • Volatile market - wait for line stability\n"
        else:
            report += "  • Stable market - safe to place bets\n"
        
        return report
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for real-time alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self, match_ids: List[str]):
        """Start real-time monitoring for multiple matches"""
        
        self.active_monitoring = True
        print(f"📶 Starting real-time monitoring for {len(match_ids)} matches...")
        
        # Create monitoring tasks
        for match_id in match_ids:
            task = asyncio.create_task(self.track_match_odds(match_id, duration_hours=24))
            self.monitoring_tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*self.monitoring_tasks)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.active_monitoring = False
        for task in self.monitoring_tasks:
            task.cancel()
        print("⏹️ Monitoring stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.db_connection.close()
        print("🧹 Cleanup completed")

# Demo function
async def demo_odds_tracking():
    """Demonstrate the advanced odds tracking system"""
    
    print("📈 ADVANCED ODDS TRACKING DEMO")
    print("=" * 45)
    
    # Initialize tracker
    tracker = AdvancedOddsTracker({
        'update_frequency': 2,  # Fast updates for demo
        'arbitrage_threshold': 0.005  # Lower threshold for demo
    })
    
    # Demo callback for alerts
    async def alert_handler(alert_type: str, data):
        print(f"🚨 ALERT [{alert_type.upper()}]: {data}")
    
    tracker.add_alert_callback(alert_handler)
    
    # Track sample match for short duration
    sample_match_id = "sinner_vs_djokovic_usopen2024"
    
    print(f"\n🔍 Starting 30-second odds tracking demo...")
    
    # Track for 30 seconds
    snapshots = await tracker.track_match_odds(sample_match_id, duration_hours=30/3600)  # 30 seconds
    
    print(f"\n📈 TRACKING RESULTS:")
    print(f"  • Total snapshots: {len(snapshots)}")
    print(f"  • Bookmakers covered: {len(set(s.bookmaker for s in snapshots))}")
    print(f"  • Arbitrage opportunities: {len(tracker.arbitrage_opportunities)}")
    
    # Generate analysis report
    if snapshots:
        report = tracker.generate_odds_analysis_report(sample_match_id)
        print(f"\n{report}")
    
    # Check best current odds
    best_odds = tracker.get_best_current_odds(sample_match_id)
    if best_odds:
        print(f"\n🏆 BEST CURRENT ODDS:")
        for player, (bookmaker, odds) in best_odds.items():
            print(f"  • {player}: {odds:.2f} at {bookmaker}")
    
    # Cleanup
    tracker.cleanup()
    
    return tracker

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_odds_tracking())