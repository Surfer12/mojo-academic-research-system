"""
Core Components of Meta-Optimization Framework

This module contains the core components that implement the mathematical foundation
of the meta-optimization framework:

Ψ(x) = ∫[α(t)S(x) + (1-α(t))N(x)] × exp(-[λ₁R_cognitive + λ₂R_efficiency]) × P(H|E,β) dt

Components:
- MetaOptimizer: Main optimization engine
- DynamicIntegrator: Adaptive symbolic-neural integration (α parameter)
- CognitiveRegularizer: Cognitive authenticity constraints (λ₁, λ₂ parameters)
- BiasModeler: Human-like bias simulation (β parameter)
"""

from .meta_optimization import (
    MetaOptimizer,
    TaskSpecification,
    OptimizationResult,
    SymbolicReasoner,
    NeuralProcessor,
    CognitiveSymbolicReasoner,
    CognitiveNeuralProcessor,
)

from .dynamic_integration import DynamicIntegrator

from .cognitive_regularization import (
    CognitiveRegularizer,
    CognitiveConstraint,
    WorkingMemoryConstraint,
    AttentionConstraint,
    ProcessingSpeedConstraint,
    BiasConsistencyConstraint,
)

from .bias_modeling import (
    BiasModeler,
    CognitiveBias,
    BiasType,
    ConfirmationBias,
    AnchoringBias,
    AvailabilityBias,
    OverconfidenceBias,
)

__all__ = [
    # Main classes
    "MetaOptimizer",
    "TaskSpecification", 
    "OptimizationResult",
    
    # Abstract base classes
    "SymbolicReasoner",
    "NeuralProcessor",
    
    # Concrete implementations
    "CognitiveSymbolicReasoner",
    "CognitiveNeuralProcessor",
    
    # Dynamic integration
    "DynamicIntegrator",
    
    # Cognitive regularization
    "CognitiveRegularizer",
    "CognitiveConstraint",
    "WorkingMemoryConstraint",
    "AttentionConstraint",
    "ProcessingSpeedConstraint",
    "BiasConsistencyConstraint",
    
    # Bias modeling
    "BiasModeler",
    "CognitiveBias",
    "BiasType",
    "ConfirmationBias",
    "AnchoringBias", 
    "AvailabilityBias",
    "OverconfidenceBias",
]