"""
Utility Components of Meta-Optimization Framework

This module contains utility classes and functions that support the core
meta-optimization framework functionality:

- StatisticalAnalyzer: Statistical validation and analysis tools
- DataProcessor: Cognitive task data generation and processing
- FailureDocumenter: Systematic failure tracking and learning
- Visualizer: Plotting and visualization tools

These utilities enable:
- Rigorous statistical validation of results
- Cognitive task benchmark generation
- Transparent failure documentation and learning
- Comprehensive visualization of optimization processes
"""

from .statistical_analysis import (
    StatisticalAnalyzer,
    ConfidenceInterval,
    EffectSize,
    HypothesisTest,
)

from .data_processing import DataProcessor

from .failure_documentation import (
    FailureDocumenter,
    FailureMode,
    FailureType,
    FailureSeverity,
)

from .visualization import Visualizer

__all__ = [
    # Statistical analysis
    "StatisticalAnalyzer",
    "ConfidenceInterval",
    "EffectSize",
    "HypothesisTest",
    
    # Data processing
    "DataProcessor",
    
    # Failure documentation
    "FailureDocumenter",
    "FailureMode",
    "FailureType",
    "FailureSeverity",
    
    # Visualization
    "Visualizer",
]