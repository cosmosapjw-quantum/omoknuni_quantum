"""
Optimal MCTS budget settings for different time periods.

Calculates optimal allocation of computational resources for comprehensive
statistical mechanics analysis based on 4000 sims/second performance.
"""
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np


class BudgetCalculator:
    """
    Calculate optimal MCTS parameters for different time budgets.
    
    Based on performance measurements, calculates optimal
    allocation of simulations across games for different quality levels.
    """
    
    def __init__(self, sims_per_second: float = 4000.0):
        """
        Initialize budget calculator.
        
        Args:
            sims_per_second: MCTS simulation rate
        """
        self.sims_per_second = sims_per_second
        
        # Quality level presets (optimized for RTX 3060 Ti)
        # Overhead factors account for physics analysis and memory management
        self.quality_presets = {
            'low': {'sims_per_game': 1000, 'overhead_factor': 1.2},      # Quick test
            'medium': {'sims_per_game': 2500, 'overhead_factor': 1.3},   # Early decoherence
            'high': {'sims_per_game': 5000, 'overhead_factor': 1.35},    # Peak performance
            'very_high': {'sims_per_game': 7500, 'overhead_factor': 1.4}, # All transitions
            'ultra': {'sims_per_game': 10000, 'overhead_factor': 1.5}    # Complete physics
        }
    
    def calculate_budget(self, time_hours: float, 
                        target_games: int,
                        target_quality: str = 'medium') -> Dict[str, Any]:
        """
        Calculate optimal budget allocation.
        
        Args:
            time_hours: Available time in hours
            target_games: Desired number of games
            target_quality: Quality level ('low', 'medium', 'high', 'ultra')
            
        Returns:
            Budget configuration
        """
        total_seconds = time_hours * 3600
        total_sims_available = int(total_seconds * self.sims_per_second)
        
        if target_quality not in self.quality_presets:
            target_quality = 'medium'
        
        preset = self.quality_presets[target_quality]
        desired_sims_per_game = preset['sims_per_game']
        overhead_factor = preset['overhead_factor']
        
        # Calculate effective simulation budget (accounting for overhead)
        effective_sims = total_sims_available / overhead_factor
        
        # Determine if we can achieve target with desired quality
        total_sims_needed = target_games * desired_sims_per_game
        
        if total_sims_needed <= effective_sims:
            # Can achieve target games with desired quality
            sims_per_game = desired_sims_per_game
            estimated_games = target_games
        else:
            # Need to reduce quality or games
            if effective_sims / target_games >= 500:  # Minimum viable sims per game
                sims_per_game = int(effective_sims / target_games)
                estimated_games = target_games
            else:
                # Use minimum viable sims and reduce game count
                sims_per_game = 1000
                estimated_games = int(effective_sims / sims_per_game)
        
        # Calculate timing estimates
        actual_total_sims = estimated_games * sims_per_game
        estimated_time = (actual_total_sims * overhead_factor) / self.sims_per_second
        
        return {
            'target_games': estimated_games,
            'sims_per_game': sims_per_game,
            'total_sims': actual_total_sims,
            'estimated_time': estimated_time,
            'time_budget': time_hours,
            'quality_level': target_quality,
            'utilization': estimated_time / total_seconds,
            'sims_per_second': self.sims_per_second,
            'overhead_factor': overhead_factor
        }


@dataclass
class BudgetRecommendation:
    """Recommendation for optimal budget allocation"""
    time_budget: float  # hours
    target_games: int
    sims_per_game: int
    quality_level: str
    total_sims: int
    expected_duration: float  # hours
    analysis_types: List[str]
    confidence: str  # 'high', 'medium', 'low'
    description: str


class OptimalBudgetSettings:
    """
    Provides optimal MCTS settings for different time budgets.
    
    Based on 4000 sims/second performance, calculates the best allocation
    of simulations to maximize statistical significance of results.
    """
    
    def __init__(self, sims_per_second: float = 4000.0):
        """
        Initialize with performance baseline.
        
        Args:
            sims_per_second: Measured MCTS performance
        """
        self.calculator = BudgetCalculator(sims_per_second)
        self.sims_per_second = sims_per_second
    
    def get_hourly_recommendation(self) -> BudgetRecommendation:
        """
        Get optimal settings for 1-hour analysis session.
        
        Focus: Quick validation of core thermodynamic relations
        Target: Sufficient data for basic statistical validation
        """
        budget = self.calculator.calculate_budget(
            time_hours=1.0,
            target_games=50,  # Reduced for 2.5K sims
            target_quality='medium'  # 2500 sims per game
        )
        
        return BudgetRecommendation(
            time_budget=1.0,
            target_games=budget['target_games'],
            sims_per_game=budget['sims_per_game'],
            quality_level=budget['quality_level'],
            total_sims=budget['total_sims'],
            expected_duration=budget['estimated_time'] / 3600,
            analysis_types=['thermodynamics', 'critical', 'decoherence'],
            confidence='medium',
            description=(
                "1-hour focused analysis. Validates basic thermodynamic relations "
                "(Jarzynski equality, temperature scaling) and identifies critical "
                "positions. Sufficient for proof-of-concept demonstration."
            )
        )
    
    def get_overnight_recommendation(self) -> BudgetRecommendation:
        """
        Get optimal settings for overnight analysis (8 hours).
        
        Focus: Comprehensive statistical mechanics validation
        Target: High-quality data for publication-grade analysis
        """
        budget = self.calculator.calculate_budget(
            time_hours=8.0,
            target_games=500,  # 10K sims per game for complete physics
            target_quality='ultra'  # 10000 sims per game
        )
        
        return BudgetRecommendation(
            time_budget=8.0,
            target_games=budget['target_games'],
            sims_per_game=budget['sims_per_game'],
            quality_level=budget['quality_level'],
            total_sims=budget['total_sims'],
            expected_duration=budget['estimated_time'] / 3600,
            analysis_types=['thermodynamics', 'critical', 'fdt', 
                          'decoherence', 'tunneling', 'entanglement'],
            confidence='high',
            description=(
                "Comprehensive overnight analysis. Validates all statistical "
                "mechanics relations with high confidence. Extracts critical "
                "exponents, verifies FDT, analyzes phase transitions. "
                "Includes full quantum phenomena analysis: decoherence, "
                "tunneling, and entanglement. Publication-quality dataset."
            )
        )
    
    def get_weekend_recommendation(self) -> BudgetRecommendation:
        """
        Get optimal settings for weekend analysis (48 hours).
        
        Focus: Ultra-high quality dataset for detailed research
        Target: Maximum statistical power and precision
        """
        budget = self.calculator.calculate_budget(
            time_hours=48.0,
            target_games=1000,  # Reduced from 2000, using 10K sims
            target_quality='ultra'  # 10000 sims per game max
        )
        
        return BudgetRecommendation(
            time_budget=48.0,
            target_games=budget['target_games'],
            sims_per_game=budget['sims_per_game'],
            quality_level=budget['quality_level'],
            total_sims=budget['total_sims'],
            expected_duration=budget['estimated_time'] / 3600,
            analysis_types=['thermodynamics', 'critical', 'fdt', 
                          'decoherence', 'tunneling', 'entanglement'],
            confidence='high',
            description=(
                "Ultra-comprehensive weekend analysis. Maximum quality dataset "
                "with exceptional statistical power. Ideal for detailed "
                "research, precise critical exponent extraction, and novel "
                "phenomena discovery. Complete quantum phenomena characterization "
                "with all visualizations."
            )
        )
    
    def get_quick_test_recommendation(self) -> BudgetRecommendation:
        """
        Get optimal settings for quick testing (15 minutes).
        
        Focus: Rapid validation that system is working
        Target: Basic functionality verification
        """
        budget = self.calculator.calculate_budget(
            time_hours=0.25,
            target_games=10,
            target_quality='low'  # 1000 sims per game
        )
        
        return BudgetRecommendation(
            time_budget=0.25,
            target_games=budget['target_games'],
            sims_per_game=budget['sims_per_game'],
            quality_level=budget['quality_level'],
            total_sims=budget['total_sims'],
            expected_duration=budget['estimated_time'] / 3600,
            analysis_types=['thermodynamics'],
            confidence='low',
            description=(
                "Quick test run for system validation. Verifies that data "
                "extraction and analysis pipeline works correctly. Not "
                "suitable for scientific conclusions."
            )
        )
    
    def get_custom_recommendation(self, time_hours: float,
                                target_games: int = None,
                                focus: str = 'balanced') -> BudgetRecommendation:
        """
        Get custom recommendation for specific constraints.
        
        Args:
            time_hours: Available time budget
            target_games: Desired number of games (None for automatic)
            focus: 'speed', 'quality', 'balanced'
        """
        # Determine target games if not specified
        if target_games is None:
            if time_hours <= 1:
                target_games = int(50 * time_hours)
            elif time_hours <= 8:
                target_games = int(50 * time_hours)
            else:
                target_games = int(40 * time_hours)  # Diminishing returns
        
        # Determine quality based on focus and time
        if focus == 'speed':
            quality = 'low' if time_hours < 2 else 'medium'
        elif focus == 'quality':
            quality = 'high' if time_hours > 4 else 'medium'
        else:  # balanced
            if time_hours < 1:
                quality = 'low'
            elif time_hours < 4:
                quality = 'medium'
            else:
                quality = 'high'
        
        budget = self.calculator.calculate_budget(time_hours, target_games, quality)
        
        # Determine analysis types based on time
        if time_hours < 0.5:
            analysis_types = ['thermodynamics']
        elif time_hours < 2:
            analysis_types = ['thermodynamics', 'critical', 'decoherence']
        elif time_hours < 4:
            analysis_types = ['thermodynamics', 'critical', 'fdt', 'decoherence']
        else:
            analysis_types = ['thermodynamics', 'critical', 'fdt', 
                          'decoherence', 'tunneling', 'entanglement']
        
        # Determine confidence
        if budget['target_games'] < 20:
            confidence = 'low'
        elif budget['target_games'] < 100:
            confidence = 'medium'
        else:
            confidence = 'high'
        
        return BudgetRecommendation(
            time_budget=time_hours,
            target_games=budget['target_games'],
            sims_per_game=budget['sims_per_game'],
            quality_level=budget['quality_level'],
            total_sims=budget['total_sims'],
            expected_duration=budget['estimated_time'] / 3600,
            analysis_types=analysis_types,
            confidence=confidence,
            description=f"Custom {focus}-focused analysis for {time_hours}h budget."
        )
    
    def create_generator_config_dict(self, 
                                    recommendation: BudgetRecommendation,
                                    output_dir: str = './analysis_output') -> Dict[str, Any]:
        """Create configuration dict for generator from recommendation"""
        return {
            'target_games': recommendation.target_games,
            'sims_per_game': recommendation.sims_per_game,
            'analysis_types': recommendation.analysis_types,
            'output_dir': output_dir,
            'generate_plots': True,
            'save_data': True,
            'progress_reporting': True
        }
    
    def print_recommendation_summary(self, recommendation: BudgetRecommendation):
        """Print formatted recommendation summary"""
        print(f"\n🎯 MCTS Analysis Budget Recommendation")
        print(f"{'='*50}")
        print(f"⏱️  Time Budget: {recommendation.time_budget:.1f} hours")
        print(f"🎮 Target Games: {recommendation.target_games:,}")
        print(f"🔄 Sims per Game: {recommendation.sims_per_game:,}")
        print(f"💎 Quality Level: {recommendation.quality_level.upper()}")
        print(f"🧮 Total Simulations: {recommendation.total_sims:,}")
        print(f"⏰ Expected Duration: {recommendation.expected_duration:.1f} hours")
        print(f"📊 Analysis Types: {', '.join(recommendation.analysis_types)}")
        print(f"🎯 Confidence: {recommendation.confidence.upper()}")
        print(f"\n📝 Description:")
        print(f"{recommendation.description}")
        print(f"{'='*50}")
        
        # Performance estimates
        utilization = recommendation.expected_duration / recommendation.time_budget * 100
        sims_rate = recommendation.total_sims / (recommendation.expected_duration * 3600)
        
        print(f"\n📈 Performance Estimates:")
        print(f"   • Budget Utilization: {utilization:.1f}%")
        print(f"   • Effective Sim Rate: {sims_rate:.0f} sims/sec")
        print(f"   • Time per Game: {recommendation.expected_duration*3600/recommendation.target_games:.1f} sec")
        
        # Statistical power estimates
        if recommendation.target_games >= 100:
            statistical_power = "High - suitable for publication"
        elif recommendation.target_games >= 50:
            statistical_power = "Medium - good for validation"
        elif recommendation.target_games >= 20:
            statistical_power = "Low - proof of concept only"
        else:
            statistical_power = "Very Low - testing only"
        
        print(f"   • Statistical Power: {statistical_power}")


def get_all_preset_recommendations(sims_per_second: float = 4000.0) -> Dict[str, BudgetRecommendation]:
    """Get all preset recommendations"""
    settings = OptimalBudgetSettings(sims_per_second)
    
    return {
        'quick_test': settings.get_quick_test_recommendation(),
        'hourly': settings.get_hourly_recommendation(),
        'overnight': settings.get_overnight_recommendation(),
        'weekend': settings.get_weekend_recommendation()
    }


def print_all_recommendations(sims_per_second: float = 4000.0):
    """Print all preset recommendations"""
    settings = OptimalBudgetSettings(sims_per_second)
    recommendations = get_all_preset_recommendations(sims_per_second)
    
    print(f"🚀 MCTS Analysis Budget Settings (Performance: {sims_per_second:,.0f} sims/sec)")
    print(f"{'='*80}")
    
    for name, rec in recommendations.items():
        print(f"\n🏷️  {name.upper().replace('_', ' ')} ANALYSIS")
        settings.print_recommendation_summary(rec)


# Example usage functions
def get_hourly_config(output_dir: str = './hourly_analysis') -> Dict[str, Any]:
    """Get config dict for 1-hour analysis"""
    settings = OptimalBudgetSettings()
    recommendation = settings.get_hourly_recommendation()
    return settings.create_generator_config_dict(recommendation, output_dir)


def get_overnight_config(output_dir: str = './overnight_analysis') -> Dict[str, Any]:
    """Get config dict for overnight analysis"""
    settings = OptimalBudgetSettings()
    recommendation = settings.get_overnight_recommendation()
    return settings.create_generator_config_dict(recommendation, output_dir)


if __name__ == "__main__":
    # Print all preset recommendations
    print_all_recommendations()