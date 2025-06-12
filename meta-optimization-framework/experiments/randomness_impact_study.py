#!/usr/bin/env python3
"""
Randomness Impact Study on Cognitive Optimization

This experiment investigates how different randomness strategies affect
the meta-optimization framework's ability to achieve target performance
improvements and authentic cognitive modeling.

Experimental Conditions:
1. Fixed seed (baseline reproduction)
2. Dynamic seeding (time-based variation)
3. Controlled stochastic (bounded randomness)
4. Adaptive seeding (performance-guided variation)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any

# Import framework components
import sys
sys.path.append('..')
from src.core.meta_optimization import MetaOptimizer, TaskSpecification
from src.utils.data_processing import DataProcessor
from src.utils.statistical_analysis import StatisticalAnalyzer

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

class RandomnessStrategy:
    """Base class for different randomness strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.iteration_count = 0
    
    def set_seed(self, iteration: int = None) -> int:
        """Set random seed for current iteration."""
        raise NotImplementedError
    
    def reset(self):
        """Reset strategy state."""
        self.iteration_count = 0

class FixedSeedStrategy(RandomnessStrategy):
    """Always use the same seed (baseline)."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Fixed")
        self.seed = seed
    
    def set_seed(self, iteration: int = None) -> int:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return self.seed

class DynamicSeedStrategy(RandomnessStrategy):
    """Use time-based dynamic seeding."""
    
    def __init__(self):
        super().__init__("Dynamic")
    
    def set_seed(self, iteration: int = None) -> int:
        seed = int(time.time() * 1000) % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

class ControlledStochasticStrategy(RandomnessStrategy):
    """Bounded randomness around base seed."""
    
    def __init__(self, base_seed: int = 42, variance: int = 100):
        super().__init__("Controlled")
        self.base_seed = base_seed
        self.variance = variance
    
    def set_seed(self, iteration: int = None) -> int:
        if iteration is None:
            iteration = self.iteration_count
        
        # Generate predictable but varied seed
        seed = self.base_seed + (iteration * 17 + 23) % self.variance
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.iteration_count += 1
        return seed

class AdaptiveSeedStrategy(RandomnessStrategy):
    """Performance-guided seed adaptation."""
    
    def __init__(self, base_seed: int = 42):
        super().__init__("Adaptive")
        self.base_seed = base_seed
        self.performance_history = []
        self.seed_history = []
        self.current_seed = base_seed
    
    def set_seed(self, iteration: int = None, performance: float = None) -> int:
        if iteration is None:
            iteration = self.iteration_count
        
        if performance is not None:
            self.performance_history.append(performance)
        
        # Adapt seed based on performance trend
        if len(self.performance_history) > 2:
            recent_trend = np.mean(self.performance_history[-3:]) - np.mean(self.performance_history[-6:-3]) if len(self.performance_history) >= 6 else 0
            
            if recent_trend < 0:  # Performance declining
                # Increase randomness
                self.current_seed = (self.current_seed + int(time.time()) % 1000) % 2**32
            else:  # Performance stable or improving
                # Moderate variation
                self.current_seed = (self.current_seed + iteration * 7) % 2**32
        
        torch.manual_seed(self.current_seed)
        np.random.seed(self.current_seed)
        self.seed_history.append(self.current_seed)
        self.iteration_count += 1
        return self.current_seed

def run_experiment_condition(strategy: RandomnessStrategy, 
                            num_runs: int = 5,
                            max_iterations: int = 50) -> Dict[str, Any]:
    """Run experiment with given randomness strategy."""
    
    results = {
        'strategy': strategy.name,
        'runs': [],
        'performance_gains': [],
        'authenticity_scores': [],
        'convergence_iterations': [],
        'final_parameters': [],
        'seeds_used': []
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {strategy.name} Randomness Strategy")
    print(f"{'='*60}")
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...")
        strategy.reset()
        
        # Set initial seed
        seed_used = strategy.set_seed(0)
        results['seeds_used'].append(seed_used)
        
        # Create task specification
        data_processor = DataProcessor(random_seed=seed_used)
        
        input_data, target_data = data_processor.generate_n_back_data(
            batch_size=32,  # Smaller batch for faster iteration
            sequence_length=15,
            n_back=2,
            feature_dim=64,  # Reduced complexity
            match_probability=0.3
        )
        
        task_spec = TaskSpecification(
            input_data=input_data,
            target_output=target_data,
            task_type="n_back",
            cognitive_constraints={
                "authenticity": 0.8,
                "memory_capacity": 7,
                "attention_threshold": 0.7,
                "expected_biases": 0.3
            },
            efficiency_requirements={
                "max_flops": 5e8,  # Reduced for faster testing
                "max_memory": 5e5,
                "max_complexity": 1.0
            }
        )
        
        # Initialize optimizer with strategy's seed
        optimizer = MetaOptimizer(
            cognitive_constraints=task_spec.cognitive_constraints,
            efficiency_requirements=task_spec.efficiency_requirements,
            random_seed=seed_used
        )
        
        # Run optimization with dynamic seeding if adaptive
        if isinstance(strategy, AdaptiveSeedStrategy):
            # Custom optimization loop for adaptive seeding
            result = run_adaptive_optimization(optimizer, task_spec, strategy, max_iterations)
        else:
            # Standard optimization
            result = optimizer.meta_optimize(
                task_specification=task_spec,
                max_iterations=max_iterations,
                convergence_threshold=1e-6,
                target_improvement=0.19
            )
        
        # Collect results
        run_data = {
            'performance_gain': result.performance_gain,
            'convergence_iterations': len(result.convergence_history),
            'final_alpha': result.alpha,
            'final_lambda1': result.lambda_1,
            'final_lambda2': result.lambda_2,
            'final_beta': result.beta,
            'seed_used': seed_used
        }
        
        # Get component analysis
        integration_summary = optimizer.dynamic_integrator.get_integration_summary()
        regularization_summary = optimizer.cognitive_regularizer.get_regularization_summary()
        
        run_data['authenticity_score'] = regularization_summary.get('mean_authenticity', 0)
        run_data['integration_stability'] = integration_summary.get('integration_stability', 0)
        
        results['runs'].append(run_data)
        results['performance_gains'].append(result.performance_gain)
        results['authenticity_scores'].append(run_data['authenticity_score'])
        results['convergence_iterations'].append(len(result.convergence_history))
        results['final_parameters'].append({
            'alpha': result.alpha,
            'lambda1': result.lambda_1,
            'lambda2': result.lambda_2,
            'beta': result.beta
        })
        
        print(f"   Performance gain: {result.performance_gain:.3f}")
        print(f"   Authenticity: {run_data['authenticity_score']:.3f}")
        print(f"   Iterations: {len(result.convergence_history)}")
    
    # Calculate summary statistics
    results['mean_performance'] = np.mean(results['performance_gains'])
    results['std_performance'] = np.std(results['performance_gains'])
    results['mean_authenticity'] = np.mean(results['authenticity_scores'])
    results['std_authenticity'] = np.std(results['authenticity_scores'])
    results['mean_convergence'] = np.mean(results['convergence_iterations'])
    results['target_achievement_rate'] = sum(1 for p in results['performance_gains'] if p >= 0.11) / len(results['performance_gains'])
    
    print(f"\n{strategy.name} Strategy Summary:")
    print(f"   Mean Performance: {results['mean_performance']:.3f} ± {results['std_performance']:.3f}")
    print(f"   Mean Authenticity: {results['mean_authenticity']:.3f} ± {results['std_authenticity']:.3f}")
    print(f"   Target Achievement Rate: {results['target_achievement_rate']:.1%}")
    
    return results

def run_adaptive_optimization(optimizer: MetaOptimizer, 
                            task_spec: TaskSpecification,
                            strategy: AdaptiveSeedStrategy,
                            max_iterations: int) -> Any:
    """Custom optimization loop for adaptive seeding."""
    
    # Start optimization
    current_performance = 0.0
    convergence_history = []
    
    for iteration in range(max_iterations):
        # Update seed based on performance
        if iteration > 0:
            strategy.set_seed(iteration, current_performance)
        
        # Run single optimization step
        # Note: This is a simplified version - full implementation would require
        # modifying the MetaOptimizer to support step-by-step execution
        result = optimizer.meta_optimize(
            task_specification=task_spec,
            max_iterations=5,  # Short bursts
            convergence_threshold=1e-6,
            target_improvement=0.19
        )
        
        current_performance = result.performance_gain
        convergence_history.extend(result.convergence_history)
        
        # Check for convergence
        if len(convergence_history) > 10:
            recent_improvement = abs(convergence_history[-1] - convergence_history[-10])
            if recent_improvement < 1e-6:
                break
    
    # Create final result
    class AdaptiveResult:
        def __init__(self):
            self.performance_gain = current_performance
            self.convergence_history = convergence_history
            self.alpha = result.alpha
            self.lambda_1 = result.lambda_1
            self.lambda_2 = result.lambda_2
            self.beta = result.beta
    
    return AdaptiveResult()

def analyze_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform comparative analysis across strategies."""
    
    analysis = {
        'strategy_comparison': {},
        'performance_ranking': [],
        'authenticity_ranking': [],
        'statistical_tests': {}
    }
    
    # Statistical analyzer
    analyzer = StatisticalAnalyzer()
    
    # Compare strategies
    for i, result1 in enumerate(all_results):
        for j, result2 in enumerate(all_results[i+1:], i+1):
            strategy1 = result1['strategy']
            strategy2 = result2['strategy']
            
            # T-test for performance differences
            t_stat, p_value = analyzer.independent_t_test(
                result1['performance_gains'],
                result2['performance_gains']
            )
            
            analysis['statistical_tests'][f"{strategy1}_vs_{strategy2}"] = {
                'performance_t_stat': t_stat,
                'performance_p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # Rank strategies
    performance_means = [(r['strategy'], r['mean_performance']) for r in all_results]
    analysis['performance_ranking'] = sorted(performance_means, key=lambda x: x[1], reverse=True)
    
    authenticity_means = [(r['strategy'], r['mean_authenticity']) for r in all_results]
    analysis['authenticity_ranking'] = sorted(authenticity_means, key=lambda x: x[1], reverse=True)
    
    return analysis

def create_visualizations(all_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """Create comprehensive visualizations."""
    
    # Create results directory
    results_dir = Path("../results/randomness_study")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Randomness Strategy Impact on Cognitive Optimization', fontsize=16, fontweight='bold')
    
    strategies = [r['strategy'] for r in all_results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Performance comparison (box plot)
    ax1 = axes[0, 0]
    performance_data = [r['performance_gains'] for r in all_results]
    bp1 = ax1.boxplot(performance_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(y=0.19, color='red', linestyle='--', alpha=0.7, label='Target (19%)')
    ax1.axhline(y=0.11, color='orange', linestyle='--', alpha=0.7, label='Minimum (11%)')
    ax1.set_title('Performance Gain Distribution')
    ax1.set_ylabel('Performance Gain')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Authenticity comparison
    ax2 = axes[0, 1]
    authenticity_data = [r['authenticity_scores'] for r in all_results]
    bp2 = ax2.boxplot(authenticity_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
    ax2.set_title('Authenticity Score Distribution')
    ax2.set_ylabel('Authenticity Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence speed
    ax3 = axes[0, 2]
    convergence_data = [r['convergence_iterations'] for r in all_results]
    bp3 = ax3.boxplot(convergence_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Convergence Speed')
    ax3.set_ylabel('Iterations to Convergence')
    ax3.grid(True, alpha=0.3)
    
    # 4. Target achievement rate
    ax4 = axes[1, 0]
    achievement_rates = [r['target_achievement_rate'] for r in all_results]
    bars = ax4.bar(strategies, achievement_rates, color=colors, alpha=0.7)
    ax4.set_title('Target Achievement Rate')
    ax4.set_ylabel('Proportion Achieving ≥11% Improvement')
    ax4.set_ylim(0, 1)
    for bar, rate in zip(bars, achievement_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom')
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance vs Authenticity scatter
    ax5 = axes[1, 1]
    for i, result in enumerate(all_results):
        ax5.scatter(result['authenticity_scores'], result['performance_gains'], 
                   label=result['strategy'], color=colors[i], alpha=0.7, s=50)
    ax5.axhline(y=0.19, color='red', linestyle='--', alpha=0.5, label='Performance Target')
    ax5.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Authenticity Target')
    ax5.set_xlabel('Authenticity Score')
    ax5.set_ylabel('Performance Gain')
    ax5.set_title('Performance vs Authenticity Trade-off')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical significance heatmap
    ax6 = axes[1, 2]
    strategies = [r['strategy'] for r in all_results]
    n_strategies = len(strategies)
    significance_matrix = np.eye(n_strategies)
    
    for key, test_result in analysis['statistical_tests'].items():
        strategies_pair = key.split('_vs_')
        if len(strategies_pair) == 2:
            i = strategies.index(strategies_pair[0])
            j = strategies.index(strategies_pair[1])
            p_val = test_result['performance_p_value']
            significance_matrix[i, j] = p_val
            significance_matrix[j, i] = p_val
    
    im = ax6.imshow(significance_matrix, cmap='RdYlGn', vmin=0, vmax=0.1)
    ax6.set_xticks(range(n_strategies))
    ax6.set_yticks(range(n_strategies))
    ax6.set_xticklabels(strategies, rotation=45)
    ax6.set_yticklabels(strategies)
    ax6.set_title('Statistical Significance (p-values)')
    
    # Add text annotations
    for i in range(n_strategies):
        for j in range(n_strategies):
            text = ax6.text(j, i, f'{significance_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax6, label='p-value')
    
    plt.tight_layout()
    plt.savefig(results_dir / "randomness_impact_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {results_dir}/randomness_impact_analysis.png")

def main():
    """Main experimental function."""
    
    print("=" * 80)
    print("RANDOMNESS IMPACT STUDY ON COGNITIVE OPTIMIZATION")
    print("=" * 80)
    print("\nInvestigating how different randomness strategies affect:")
    print("• Performance improvement achievement")
    print("• Cognitive authenticity modeling")
    print("• Optimization convergence patterns")
    print("• Parameter exploration diversity")
    
    # Define experimental strategies
    strategies = [
        FixedSeedStrategy(42),
        DynamicSeedStrategy(),
        ControlledStochasticStrategy(42, 100),
        AdaptiveSeedStrategy(42)
    ]
    
    # Run experiments
    all_results = []
    for strategy in strategies:
        result = run_experiment_condition(strategy, num_runs=3, max_iterations=30)
        all_results.append(result)
    
    # Analyze results
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    analysis = analyze_results(all_results)
    
    # Print rankings
    print("\nPerformance Rankings:")
    for i, (strategy, performance) in enumerate(analysis['performance_ranking'], 1):
        print(f"{i}. {strategy}: {performance:.3f}")
    
    print("\nAuthenticity Rankings:")
    for i, (strategy, authenticity) in enumerate(analysis['authenticity_ranking'], 1):
        print(f"{i}. {strategy}: {authenticity:.3f}")
    
    # Print statistical significance
    print("\nStatistical Significance Tests:")
    for comparison, test_result in analysis['statistical_tests'].items():
        significance = "***" if test_result['significant'] else "n.s."
        print(f"{comparison}: p = {test_result['performance_p_value']:.4f} {significance}")
    
    # Create visualizations
    create_visualizations(all_results, analysis)
    
    # Save detailed results
    results_dir = Path("../results/randomness_study")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization
    serializable_results = []
    for result in all_results:
        serializable_result = {k: v for k, v in result.items() if k != 'runs'}
        serializable_result['runs'] = []
        for run in result['runs']:
            serializable_run = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                              for k, v in run.items()}
            serializable_result['runs'].append(serializable_run)
        serializable_results.append(serializable_result)
    
    with open(results_dir / "detailed_results.json", 'w') as f:
        json.dump({
            'experiment_results': serializable_results,
            'analysis': {
                'performance_ranking': analysis['performance_ranking'],
                'authenticity_ranking': analysis['authenticity_ranking'],
                'statistical_tests': {k: {sk: float(sv) if isinstance(sv, (np.float64, np.float32)) else sv 
                                        for sk, sv in v.items()} 
                                     for k, v in analysis['statistical_tests'].items()}
            }
        }, f, indent=2)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    best_performance = analysis['performance_ranking'][0]
    best_authenticity = analysis['authenticity_ranking'][0]
    
    print(f"\nBest Performance Strategy: {best_performance[0]} ({best_performance[1]:.3f})")
    print(f"Best Authenticity Strategy: {best_authenticity[0]} ({best_authenticity[1]:.3f})")
    
    # Check if any strategy achieved target
    target_achievers = [(r['strategy'], r['target_achievement_rate']) 
                       for r in all_results if r['target_achievement_rate'] > 0]
    
    if target_achievers:
        print(f"\nStrategies achieving target performance:")
        for strategy, rate in sorted(target_achievers, key=lambda x: x[1], reverse=True):
            print(f"• {strategy}: {rate:.1%} success rate")
    else:
        print(f"\nNo strategy consistently achieved the 19% target improvement.")
        print(f"This suggests the need for:")
        print(f"• Architectural modifications")
        print(f"• Extended optimization periods") 
        print(f"• Alternative constraint formulations")
    
    print(f"\nDetailed results saved to: {results_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()