"""
Meta-Optimization Framework - Advanced Optimization Algorithms
==============================================================

Advanced optimization algorithms that extend beyond the core meta-optimizer,
providing specialized optimization techniques for different problem domains.

Core Components:
- AdaptiveOptimizer: Self-adjusting optimization strategies
- MetaLearningOptimizer: Learning-to-optimize algorithms
- HybridOptimizer: Combines multiple optimization approaches
- CognitiveOptimizer: Cognitive science-inspired optimization
- NeuralOptimizer: Neural network-based optimization
- SymbolicOptimizer: Symbolic reasoning-based optimization
"""

from .adaptive_optimizer import AdaptiveOptimizer
from .meta_learning_optimizer import MetaLearningOptimizer
from .hybrid_optimizer import HybridOptimizer
from .cognitive_optimizer import CognitiveOptimizer
from .neural_optimizer import NeuralOptimizer
from .symbolic_optimizer import SymbolicOptimizer

__all__ = [
    'AdaptiveOptimizer',
    'MetaLearningOptimizer',
    'HybridOptimizer',
    'CognitiveOptimizer',
    'NeuralOptimizer',
    'SymbolicOptimizer'
]

# Optimization algorithm categories
OPTIMIZATION_CATEGORIES = {
    'adaptive': ['AdaptiveOptimizer'],
    'meta_learning': ['MetaLearningOptimizer'],
    'hybrid': ['HybridOptimizer'],
    'cognitive': ['CognitiveOptimizer'],
    'neural': ['NeuralOptimizer'],
    'symbolic': ['SymbolicOptimizer']
}

def get_optimizer_by_category(category: str):
    """Get optimizer classes by category."""
    if category not in OPTIMIZATION_CATEGORIES:
        available_categories = list(OPTIMIZATION_CATEGORIES.keys())
        raise ValueError(f"Category '{category}' not found. Available: {available_categories}")

    return OPTIMIZATION_CATEGORIES[category]

def get_all_optimizers():
    """Get all available optimizer classes."""
    return __all__
