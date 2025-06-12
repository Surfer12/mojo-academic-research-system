"""
Meta-Optimization Framework - Evaluation System
===============================================

Comprehensive evaluation system for the cognitive-computational meta-optimization framework.
Measures the target performance improvements:
- Accuracy: 19% ± 8% improvement
- Efficiency: 12% ± 4% gains  
- Cognitive Load: 22% ± 5% reduction

Core Components:
- PerformanceEvaluator: Main evaluation orchestrator
- CognitiveMetrics: Cognitive task evaluation
- EfficiencyAnalyzer: Computational efficiency measurement
- BenchmarkSuite: Standardized benchmark tasks
- StatisticalValidator: Statistical significance testing
"""

from .performance_evaluator import PerformanceEvaluator
from .cognitive_metrics import CognitiveMetrics
from .efficiency_analyzer import EfficiencyAnalyzer
from .benchmark_suite import BenchmarkSuite
from .statistical_validator import StatisticalValidator

__all__ = [
    'PerformanceEvaluator',
    'CognitiveMetrics', 
    'EfficiencyAnalyzer',
    'BenchmarkSuite',
    'StatisticalValidator'
]

# Target performance metrics
TARGET_ACCURACY_IMPROVEMENT = 0.19  # 19%
TARGET_ACCURACY_STDDEV = 0.08      # ±8%
TARGET_EFFICIENCY_GAIN = 0.12      # 12%
TARGET_EFFICIENCY_STDDEV = 0.04    # ±4%
TARGET_COGNITIVE_REDUCTION = 0.22  # 22%
TARGET_COGNITIVE_STDDEV = 0.05     # ±5%

def get_target_metrics():
    """Get the target performance metrics for the framework."""
    return {
        'accuracy_improvement': TARGET_ACCURACY_IMPROVEMENT,
        'accuracy_stddev': TARGET_ACCURACY_STDDEV,
        'efficiency_gain': TARGET_EFFICIENCY_GAIN,
        'efficiency_stddev': TARGET_EFFICIENCY_STDDEV,
        'cognitive_reduction': TARGET_COGNITIVE_REDUCTION,
        'cognitive_stddev': TARGET_COGNITIVE_STDDEV
    }