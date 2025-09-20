#!/usr/bin/env python3
"""
Genetic Algorithm Feature Engineering for Football Prediction

Advanced feature selection and combination using genetic algorithms
to optimize prediction accuracy. Based on research showing 15%+ improvement
in model performance through GA-optimized feature engineering.

Author: Milkpainter
Version: 1.0
Based on: Multiple research papers showing GA superiority in sports prediction
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
import random
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class GeneticFeatureEngineer:
    """
    Genetic Algorithm for automated feature engineering in football prediction.
    
    Uses evolutionary algorithms to:
    1. Select optimal feature subsets
    2. Create feature combinations
    3. Optimize feature transformations
    4. Weight feature importance
    
    Based on research showing genetic algorithms achieve 15%+ improvement
    in sports prediction accuracy through intelligent feature engineering.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 fitness_function: str = 'accuracy',
                 random_state: int = 42):
        """
        Initialize the Genetic Feature Engineer.
        
        Args:
            population_size: Number of individuals in each generation
            generations: Number of evolutionary generations
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
            elite_size: Number of best individuals to preserve each generation
            fitness_function: 'accuracy', 'auc', or 'f1'
            random_state: Random seed for reproducibility
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.fitness_function = fitness_function
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Initialize tracking variables
        self.best_features_ = None
        self.best_fitness_ = 0.0
        self.fitness_history_ = []
        self.feature_usage_history_ = []
        self.original_features_ = None
        self.scaler_ = StandardScaler()
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced football features for GA optimization.
        
        Args:
            df: Base football statistics DataFrame
            
        Returns:
            Enhanced DataFrame with advanced features
        """
        print("🧬 Creating advanced football features...")
        
        enhanced_df = df.copy()
        
        # Offensive efficiency ratios
        if 'points_scored' in df.columns and 'total_yards' in df.columns:
            enhanced_df['points_per_yard'] = df['points_scored'] / (df['total_yards'] + 1)
            enhanced_df['red_zone_efficiency'] = df.get('red_zone_tds', 0) / (df.get('red_zone_attempts', 1) + 1)
            enhanced_df['third_down_efficiency'] = df.get('third_down_conversions', 0) / (df.get('third_down_attempts', 1) + 1)
        
        # Defensive strength metrics
        if 'points_allowed' in df.columns:
            enhanced_df['defensive_efficiency'] = 1 / (df['points_allowed'] + 1)
            enhanced_df['yards_per_point_allowed'] = df.get('yards_allowed', 0) / (df['points_allowed'] + 1)
        
        # Turnover differentials
        if 'turnovers_forced' in df.columns and 'turnovers' in df.columns:
            enhanced_df['turnover_differential'] = df['turnovers_forced'] - df['turnovers']
            enhanced_df['turnover_rate'] = df['turnovers'] / (df.get('possessions', 10) + 1)
        
        # Time-weighted recent performance (last 3 games weighted more)
        if 'game_number' in df.columns:
            # Inverse distance weighting for recent games
            weights = np.exp(-0.2 * (df['game_number'].max() - df['game_number']))
            enhanced_df['weighted_points'] = df.get('points_scored', 0) * weights
            enhanced_df['weighted_yards'] = df.get('total_yards', 0) * weights
        
        # Situational performance
        enhanced_df['home_field_advantage'] = df.get('is_home', 0) * 3.5  # Average HFA
        
        # Weather adjustments (if available)
        if 'temperature' in df.columns:
            enhanced_df['cold_weather_factor'] = np.where(df['temperature'] < 32, 1, 0)
            enhanced_df['dome_advantage'] = df.get('is_dome', 0)
        
        # Rest advantages
        if 'days_rest' in df.columns:
            enhanced_df['rest_advantage'] = np.log1p(df['days_rest'])
            enhanced_df['short_rest_penalty'] = np.where(df['days_rest'] < 7, 1, 0)
        
        # Strength of schedule adjustments
        if 'opponent_rank' in df.columns:
            enhanced_df['sos_adjustment'] = 1 / (df['opponent_rank'] + 1)
            enhanced_df['playing_up'] = np.where(df['opponent_rank'] > df.get('team_rank', 50), 1, 0)
        
        # Momentum indicators
        if 'wins' in df.columns and 'losses' in df.columns:
            enhanced_df['win_percentage'] = df['wins'] / (df['wins'] + df['losses'] + 1)
            enhanced_df['recent_form'] = df.get('last_3_wins', 0) / 3.0
        
        # Coaching advantages
        if 'coach_experience' in df.columns:
            enhanced_df['coaching_edge'] = np.log1p(df['coach_experience'])
        
        # Special teams efficiency
        if 'fg_percentage' in df.columns:
            enhanced_df['special_teams_score'] = (
                df['fg_percentage'] * 0.6 + 
                df.get('punt_avg', 40) / 50 * 0.4
            )
        
        # Injury impact (if available)
        if 'key_injuries' in df.columns:
            enhanced_df['injury_impact'] = -df['key_injuries'] * 2.5
        
        # Market-based features
        if 'betting_line' in df.columns:
            enhanced_df['public_fade_factor'] = np.abs(df['betting_line'])
            enhanced_df['contrarian_value'] = df.get('public_bet_percentage', 50) - 50
        
        print(f"Created {len(enhanced_df.columns) - len(df.columns)} new features")
        return enhanced_df
    
    def initialize_population(self, n_features: int) -> List[np.ndarray]:
        """
        Create initial population of feature selection chromosomes.
        
        Args:
            n_features: Number of available features
            
        Returns:
            List of binary chromosomes representing feature selections
        """
        population = []
        
        for _ in range(self.population_size):
            # Random binary chromosome (1 = feature selected, 0 = not selected)
            # Ensure at least 10% and at most 70% of features are selected
            min_features = max(1, int(0.1 * n_features))
            max_features = min(n_features, int(0.7 * n_features))
            
            n_selected = np.random.randint(min_features, max_features + 1)
            chromosome = np.zeros(n_features, dtype=int)
            selected_indices = np.random.choice(n_features, n_selected, replace=False)
            chromosome[selected_indices] = 1
            
            population.append(chromosome)
        
        return population
    
    def calculate_fitness(self, chromosome: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate fitness of a chromosome (feature subset).
        
        Args:
            chromosome: Binary array indicating selected features
            X: Feature matrix
            y: Target variable
            
        Returns:
            Fitness score (higher is better)
        """
        # Select features based on chromosome
        selected_features = chromosome.astype(bool)
        
        if not np.any(selected_features):
            return 0.0  # No features selected
        
        X_selected = X.iloc[:, selected_features]
        
        # Use Random Forest for fitness evaluation (fast and robust)
        model = RandomForestClassifier(
            n_estimators=50,  # Smaller for speed
            max_depth=10,
            random_state=self.random_state,
            n_jobs=1  # Single job for parallel processing
        )
        
        try:
            # Cross-validation score
            if self.fitness_function == 'accuracy':
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy', n_jobs=1)
            elif self.fitness_function == 'auc':
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='roc_auc', n_jobs=1)
            elif self.fitness_function == 'f1':
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1', n_jobs=1)
            else:
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy', n_jobs=1)
            
            fitness = scores.mean()
            
            # Penalty for too many features (encourage parsimony)
            n_selected = np.sum(selected_features)
            complexity_penalty = 0.001 * n_selected / len(selected_features)
            fitness -= complexity_penalty
            
            return max(0.0, fitness)
            
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return 0.0
    
    def selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            Selected parents for reproduction
        """
        parents = []
        
        # Keep elite individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            parents.append(population[idx].copy())
        
        # Tournament selection for remaining parents
        while len(parents) < self.population_size:
            # Tournament size = 3
            tournament_indices = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
        
        return parents
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two offspring chromosomes
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Two-point crossover
        length = len(parent1)
        point1 = np.random.randint(1, length - 1)
        point2 = np.random.randint(point1, length)
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        offspring1[point1:point2] = parent2[point1:point2]
        offspring2[point1:point2] = parent1[point1:point2]
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        # Ensure at least one feature is selected
        if not np.any(mutated):
            mutated[np.random.randint(len(mutated))] = 1
        
        return mutated
    
    def evolve(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Dict:
        """
        Evolve optimal feature subset using genetic algorithm.
        
        Args:
            X: Feature matrix
            y: Target variable (1 = upset, 0 = favorite wins)
            verbose: Print progress information
            
        Returns:
            Dictionary with evolution results
        """
        if verbose:
            print(f"
🧬 Starting Genetic Algorithm Feature Engineering")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Features: {len(X.columns)}")
            print(f"Samples: {len(X)}")
            print(f"Upset rate: {y.mean():.1%}")
        
        # Store original features
        self.original_features_ = X.columns.tolist()
        
        # Standardize features
        X_scaled = pd.DataFrame(
            self.scaler_.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Initialize population
        population = self.initialize_population(len(X.columns))
        
        # Evolution loop
        self.fitness_history_ = []
        best_fitness_per_gen = []
        avg_fitness_per_gen = []
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = []
            
            # Parallel fitness evaluation for speed
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                for chromosome in population:
                    future = executor.submit(self.calculate_fitness, chromosome, X_scaled, y)
                    futures.append(future)
                
                for future in futures:
                    fitness_scores.append(future.result())
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_per_gen.append(best_fitness)
            avg_fitness_per_gen.append(avg_fitness)
            
            # Update global best
            if best_fitness > self.best_fitness_:
                self.best_fitness_ = best_fitness
                best_idx = np.argmax(fitness_scores)
                self.best_features_ = population[best_idx].copy()
            
            if verbose and (generation + 1) % 10 == 0:
                n_selected = np.sum(self.best_features_)
                print(f"Generation {generation + 1:3d}: Best fitness = {best_fitness:.4f}, "
                      f"Avg fitness = {avg_fitness:.4f}, Features = {n_selected}")
            
            # Selection
            parents = self.selection(population, fitness_scores)
            
            # Create next generation
            next_population = []
            
            # Keep elite
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(next_population) < self.population_size:
                parent1 = parents[np.random.randint(len(parents))]
                parent2 = parents[np.random.randint(len(parents))]
                
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                next_population.extend([offspring1, offspring2])
            
            # Ensure population size
            population = next_population[:self.population_size]
        
        # Final results
        selected_features = np.array(self.original_features_)[self.best_features_.astype(bool)]
        
        results = {
            'best_features': selected_features.tolist(),
            'best_fitness': self.best_fitness_,
            'n_selected': np.sum(self.best_features_),
            'selection_ratio': np.sum(self.best_features_) / len(self.best_features_),
            'fitness_history': {
                'best': best_fitness_per_gen,
                'average': avg_fitness_per_gen
            },
            'improvement': best_fitness_per_gen[-1] - best_fitness_per_gen[0]
        }
        
        if verbose:
            print(f"\n✅ Genetic Algorithm Complete!")
            print(f"Best fitness: {self.best_fitness_:.4f}")
            print(f"Selected features: {np.sum(self.best_features_)}/{len(self.best_features_)}")
            print(f"Improvement: {results['improvement']:.4f}")
        
        return results
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset using selected features.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if self.best_features_ is None:
            raise ValueError("Must run evolve() first to select features")
        
        if len(X.columns) != len(self.best_features_):
            raise ValueError("Feature count mismatch. Use same features as training.")
        
        selected_features = self.best_features_.astype(bool)
        return X.iloc[:, selected_features]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature selection frequency across evolution.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.original_features_ is None:
            raise ValueError("Must run evolve() first")
        
        importance_df = pd.DataFrame({
            'feature': self.original_features_,
            'selected': self.best_features_,
            'importance': self.best_features_.astype(float)
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def plot_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Plot evolution progress.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.fitness_history_:
            raise ValueError("Must run evolve() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fitness evolution
        generations = range(1, len(self.fitness_history_['best']) + 1)
        ax1.plot(generations, self.fitness_history_['best'], 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, self.fitness_history_['average'], 'r--', label='Average Fitness', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Genetic Algorithm Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        importance_df = self.get_feature_importance()
        selected_features = importance_df[importance_df['selected'] == 1]
        
        if len(selected_features) > 0:
            top_features = selected_features.head(15)
            ax2.barh(range(len(top_features)), [1] * len(top_features))
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'], fontsize=8)
            ax2.set_xlabel('Selected')
            ax2.set_title('Selected Features')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("🧬 Genetic Algorithm Feature Engineering Demo")
    print("=" * 50)
    
    # Create sample football data
    np.random.seed(42)
    n_samples = 1000
    
    # Base features
    base_features = {
        'points_scored': np.random.normal(24, 8, n_samples),
        'points_allowed': np.random.normal(24, 8, n_samples), 
        'total_yards': np.random.normal(350, 100, n_samples),
        'yards_allowed': np.random.normal(350, 100, n_samples),
        'turnovers': np.random.poisson(2, n_samples),
        'turnovers_forced': np.random.poisson(2, n_samples),
        'red_zone_tds': np.random.poisson(3, n_samples),
        'red_zone_attempts': np.random.poisson(4, n_samples),
        'third_down_conversions': np.random.poisson(5, n_samples),
        'third_down_attempts': np.random.poisson(12, n_samples),
        'is_home': np.random.binomial(1, 0.5, n_samples),
        'days_rest': np.random.choice([3, 7, 14], n_samples),
        'temperature': np.random.normal(65, 20, n_samples),
        'wins': np.random.randint(0, 12, n_samples),
        'losses': np.random.randint(0, 12, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(base_features)
    
    # Create target (upset = 1, favorite wins = 0)
    # Upsets more likely when points_scored > points_allowed and other factors
    upset_prob = (
        (df['points_scored'] > df['points_allowed']).astype(int) * 0.3 +
        (df['turnovers_forced'] > df['turnovers']).astype(int) * 0.2 +
        df['is_home'] * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    y = (upset_prob > 0.3).astype(int)
    
    print(f"Created dataset with {len(df)} samples, {len(df.columns)} base features")
    print(f"Upset rate: {y.mean():.1%}")
    
    # Initialize GA feature engineer
    ga_engineer = GeneticFeatureEngineer(
        population_size=30,  # Smaller for demo
        generations=20,      # Fewer generations for demo
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=3,
        fitness_function='accuracy',
        random_state=42
    )
    
    # Create advanced features
    enhanced_df = ga_engineer.create_advanced_features(df)
    print(f"Enhanced to {len(enhanced_df.columns)} features")
    
    # Evolve optimal feature subset
    results = ga_engineer.evolve(enhanced_df, y, verbose=True)
    
    # Show results
    print("\n🏆 Genetic Algorithm Results:")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Selected {results['n_selected']} out of {len(enhanced_df.columns)} features")
    print(f"Improvement: {results['improvement']:.4f}")
    
    print("\n🔥 Selected Features:")
    for feature in results['best_features'][:10]:  # Show top 10
        print(f"  • {feature}")
    
    # Transform dataset
    X_selected = ga_engineer.transform(enhanced_df)
    print(f"\nTransformed dataset shape: {X_selected.shape}")
    
    # Plot evolution (commented out for demo)
    # ga_engineer.plot_evolution()
