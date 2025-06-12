"""
Meta-Optimization Framework: Bridging Minds and Machines

A comprehensive framework for cognitive-inspired deep learning optimization that implements
a mathematical foundation integrating symbolic reasoning, neural processing, cognitive
authenticity, computational efficiency, and human-like biases.

Core Equation:
Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt

Where:
- Ψ(x): Cognitive-computational state
- α(t): Dynamic integration parameter
- S(x): Symbolic reasoning component
- N(x): Neural processing component
- λ₁, λ₂: Regularization weights
- R_cognitive: Cognitive authenticity penalty
- R_efficiency: Computational efficiency penalty
- P(H|E,β): Bias-adjusted probability
- β: Bias modeling parameter

Target Performance Metrics:
- 19% ± 8% accuracy improvement (95% CI: [11%, 27%])
- 12% ± 4% computational efficiency gains (95% CI: [8%, 16%])
- 22% ± 5% cognitive load reduction

Author: Ryan Oates, University of California, Santa Barbara
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Ryan Oates"
__email__ = "ryan.oates@ucsb.edu"
__institution__ = "University of California, Santa Barbara"
__license__ = "MIT"

# Core framework imports
from .core.meta_optimization import MetaOptimizer, TaskSpecification, OptimizationResult
from .core.dynamic_integration import DynamicIntegrator
from .core.cognitive_regularization import CognitiveRegularizer
from .core.bias_modeling import BiasModeler, BiasType

# Utility imports
from .utils.statistical_analysis import StatisticalAnalyzer, ConfidenceInterval, EffectSize
from .utils.data_processing import DataProcessor
from .utils.failure_documentation import FailureDocumenter
from .utils.visualization import Visualizer

# Export main classes and functions
__all__ = [
    # Core components
    "MetaOptimizer",
    "TaskSpecification", 
    "OptimizationResult",
    "DynamicIntegrator",
    "CognitiveRegularizer",
    "BiasModeler",
    "BiasType",
    
    # Utilities
    "StatisticalAnalyzer",
    "ConfidenceInterval",
    "EffectSize", 
    "DataProcessor",
    "FailureDocumenter",
    "Visualizer",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__institution__",
    "__license__",
]

# Package configuration
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger for the package
logger = logging.getLogger(__name__)
logger.info(f"Meta-Optimization Framework v{__version__} initialized")

def get_version() -> str:
    """Get the current version of the package."""
    return __version__

def get_info() -> dict:
    """Get package information."""
    return {
        "name": "meta-optimization-framework",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "institution": __institution__,
        "license": __license__,
        "description": "A comprehensive framework for cognitive-inspired deep learning optimization",
    }

def print_equation():
    """Print the core mathematical equation."""
    equation = """
    Core Equation:
    Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt
    
    Where:
    - Ψ(x): Cognitive-computational state
    - α(t): Dynamic integration parameter  
    - S(x): Symbolic reasoning component
    - N(x): Neural processing component
    - λ₁, λ₂: Regularization weights
    - R_cognitive: Cognitive authenticity penalty
    - R_efficiency: Computational efficiency penalty
    - P(H|E,β): Bias-adjusted probability
    - β: Bias modeling parameter
    """
    print(equation)

def print_targets():
    """Print target performance metrics."""
    targets = """
    Target Performance Metrics:
    - Accuracy Improvement: 19% ± 8% (95% CI: [11%, 27%])
    - Efficiency Gains: 12% ± 4% (95% CI: [8%, 16%])  
    - Cognitive Load Reduction: 22% ± 5%
    """
    print(targets)