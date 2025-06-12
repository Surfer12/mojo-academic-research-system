"""
Cognitive Regularization Module

Implements cognitive plausibility regularization terms:
L_total = L_task + λ₁R_cognitive + λ₂R_efficiency

This module ensures that optimized models maintain cognitive authenticity
while achieving computational efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CognitiveConstraint(ABC):
    """Abstract base class for cognitive constraints."""
    
    @abstractmethod
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """Compute penalty for violating this cognitive constraint."""
        pass
    
    @abstractmethod
    def get_constraint_name(self) -> str:
        """Get name of this constraint."""
        pass


class WorkingMemoryConstraint(CognitiveConstraint):
    """Constraint based on working memory limitations (Miller's 7±2 rule)."""
    
    def __init__(self, capacity_limit: int = 7):
        self.capacity_limit = capacity_limit
    
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """Penalize outputs that exceed working memory capacity."""
        # Estimate "chunks" of information in output
        if len(model_output.shape) >= 2:
            # For sequence data, count distinct patterns
            unique_patterns = torch.unique(model_output, dim=0).shape[0]
        else:
            # For vector data, count significant components
            significant_components = (torch.abs(model_output) > 0.1).sum().item()
            unique_patterns = significant_components
        
        if unique_patterns > self.capacity_limit:
            penalty = (unique_patterns - self.capacity_limit) / self.capacity_limit
            return min(penalty, 1.0)  # Cap penalty at 1.0
        return 0.0
    
    def get_constraint_name(self) -> str:
        return "working_memory"


class AttentionConstraint(CognitiveConstraint):
    """Constraint based on attention limitations."""
    
    def __init__(self, focus_threshold: float = 0.8):
        self.focus_threshold = focus_threshold
    
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """Penalize outputs that violate attention focus principles."""
        # Compute attention distribution (how focused the output is)
        if len(model_output.shape) >= 2:
            # For multi-dimensional output, compute entropy across features
            probs = torch.softmax(model_output.flatten(), dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            max_entropy = np.log(len(probs))
            normalized_entropy = entropy / max_entropy
        else:
            # For 1D output, compute variance as focus measure
            normalized_entropy = torch.var(model_output).item()
        
        # High entropy (low focus) incurs penalty
        if normalized_entropy > (1 - self.focus_threshold):
            penalty = normalized_entropy - (1 - self.focus_threshold)
            return min(penalty, 1.0)
        return 0.0
    
    def get_constraint_name(self) -> str:
        return "attention_focus"


class ProcessingSpeedConstraint(CognitiveConstraint):
    """Constraint based on human processing speed limitations."""
    
    def __init__(self, max_complexity: float = 1.0):
        self.max_complexity = max_complexity
    
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """Penalize outputs that imply unrealistic processing speed."""
        # Estimate computational complexity of the output
        complexity = self._estimate_complexity(model_output)
        
        if complexity > self.max_complexity:
            penalty = (complexity - self.max_complexity) / self.max_complexity
            return min(penalty, 1.0)
        return 0.0
    
    def _estimate_complexity(self, output: torch.Tensor) -> float:
        """Estimate computational complexity of output."""
        # Simple heuristic: complexity based on output magnitude and variance
        magnitude = torch.norm(output).item()
        variance = torch.var(output).item()
        complexity = magnitude * variance
        return complexity
    
    def get_constraint_name(self) -> str:
        return "processing_speed"


class BiasConsistencyConstraint(CognitiveConstraint):
    """Constraint ensuring consistency with known cognitive biases."""
    
    def __init__(self, expected_biases: Dict[str, float]):
        self.expected_biases = expected_biases
    
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """Penalize outputs inconsistent with expected cognitive biases."""
        # This is a simplified implementation
        # In practice, would analyze output patterns for bias signatures
        
        # For now, check if output shows expected asymmetries
        if len(model_output.shape) >= 2:
            # Check for confirmation bias (preference for certain patterns)
            mean_output = torch.mean(model_output, dim=0)
            bias_score = torch.std(mean_output).item()
            
            expected_bias = self.expected_biases.get("confirmation", 0.3)
            bias_difference = abs(bias_score - expected_bias)
            
            if bias_difference > 0.2:  # Threshold for acceptable bias deviation
                return min(bias_difference, 1.0)
        
        return 0.0
    
    def get_constraint_name(self) -> str:
        return "bias_consistency"


class CognitiveRegularizer:
    """
    Main cognitive regularization system implementing R_cognitive term.
    
    Combines multiple cognitive constraints to ensure model outputs
    remain cognitively plausible while optimizing performance.
    """
    
    def __init__(self, 
                 cognitive_constraints: Dict[str, float],
                 constraint_weights: Optional[Dict[str, float]] = None):
        """
        Initialize cognitive regularizer.
        
        Args:
            cognitive_constraints: Dictionary of constraint parameters
            constraint_weights: Weights for different constraint types
        """
        self.cognitive_constraints = cognitive_constraints
        self.constraint_weights = constraint_weights or {
            "working_memory": 0.3,
            "attention_focus": 0.3,
            "processing_speed": 0.2,
            "bias_consistency": 0.2
        }
        
        # Initialize constraint objects
        self.constraints = self._initialize_constraints()
        
        # History tracking
        self.penalty_history: List[float] = []
        self.constraint_violations: Dict[str, List[float]] = {
            name: [] for name in self.constraint_weights.keys()
        }
    
    def _initialize_constraints(self) -> Dict[str, CognitiveConstraint]:
        """Initialize cognitive constraint objects."""
        constraints = {}
        
        # Working memory constraint
        if "working_memory" in self.constraint_weights:
            capacity = self.cognitive_constraints.get("memory_capacity", 7)
            constraints["working_memory"] = WorkingMemoryConstraint(int(capacity))
        
        # Attention constraint
        if "attention_focus" in self.constraint_weights:
            focus_threshold = self.cognitive_constraints.get("attention_threshold", 0.8)
            constraints["attention_focus"] = AttentionConstraint(focus_threshold)
        
        # Processing speed constraint
        if "processing_speed" in self.constraint_weights:
            max_complexity = self.cognitive_constraints.get("max_complexity", 1.0)
            constraints["processing_speed"] = ProcessingSpeedConstraint(max_complexity)
        
        # Bias consistency constraint
        if "bias_consistency" in self.constraint_weights:
            expected_biases = self.cognitive_constraints.get("expected_biases", {"confirmation": 0.3})
            constraints["bias_consistency"] = BiasConsistencyConstraint(expected_biases)
        
        return constraints
    
    def compute_penalty(self, model_output: torch.Tensor) -> float:
        """
        Compute total cognitive regularization penalty R_cognitive.
        
        Args:
            model_output: Output from the model to evaluate
            
        Returns:
            Total cognitive penalty (0-1 scale)
        """
        total_penalty = 0.0
        individual_penalties = {}
        
        for constraint_name, constraint in self.constraints.items():
            try:
                penalty = constraint.compute_penalty(model_output)
                weight = self.constraint_weights[constraint_name]
                weighted_penalty = weight * penalty
                
                total_penalty += weighted_penalty
                individual_penalties[constraint_name] = penalty
                
                # Track violations
                self.constraint_violations[constraint_name].append(penalty)
                
            except Exception as e:
                logger.warning(f"Error computing {constraint_name} penalty: {e}")
                individual_penalties[constraint_name] = 0.0
        
        # Store total penalty
        self.penalty_history.append(total_penalty)
        
        logger.debug(f"Cognitive penalties: {individual_penalties}, Total: {total_penalty:.4f}")
        
        return total_penalty
    
    def compute_authenticity_score(self, model_output: torch.Tensor) -> float:
        """
        Compute cognitive authenticity score (inverse of penalty).
        
        Args:
            model_output: Model output to evaluate
            
        Returns:
            Authenticity score (0-1, higher is more authentic)
        """
        penalty = self.compute_penalty(model_output)
        authenticity = 1.0 - penalty
        return max(0.0, authenticity)
    
    def analyze_constraint_violations(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze patterns in constraint violations.
        
        Returns:
            Dictionary with violation statistics for each constraint
        """
        analysis = {}
        
        for constraint_name, violations in self.constraint_violations.items():
            if violations:
                analysis[constraint_name] = {
                    "mean_violation": np.mean(violations),
                    "max_violation": np.max(violations),
                    "violation_rate": np.mean([v > 0 for v in violations]),
                    "trend": np.polyfit(range(len(violations)), violations, 1)[0] if len(violations) > 1 else 0.0
                }
            else:
                analysis[constraint_name] = {
                    "mean_violation": 0.0,
                    "max_violation": 0.0,
                    "violation_rate": 0.0,
                    "trend": 0.0
                }
        
        return analysis
    
    def adapt_constraint_weights(self, 
                               performance_feedback: float,
                               target_authenticity: float = 0.8) -> None:
        """
        Adapt constraint weights based on performance feedback.
        
        Args:
            performance_feedback: Current model performance (0-1)
            target_authenticity: Target cognitive authenticity level
        """
        if not self.penalty_history:
            return
        
        current_authenticity = 1.0 - self.penalty_history[-1]
        authenticity_error = target_authenticity - current_authenticity
        
        # Adaptation rate
        adaptation_rate = 0.01
        
        # If authenticity is too low, increase constraint weights
        if authenticity_error > 0.1:
            for constraint_name in self.constraint_weights:
                self.constraint_weights[constraint_name] *= (1 + adaptation_rate)
        
        # If authenticity is too high and performance is suffering, decrease weights
        elif authenticity_error < -0.1 and performance_feedback < 0.7:
            for constraint_name in self.constraint_weights:
                self.constraint_weights[constraint_name] *= (1 - adaptation_rate)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.constraint_weights.values())
        if total_weight > 0:
            for constraint_name in self.constraint_weights:
                self.constraint_weights[constraint_name] /= total_weight
        
        logger.debug(f"Adapted constraint weights: {self.constraint_weights}")
    
    def get_constraint_recommendations(self, 
                                     task_type: str,
                                     performance_requirements: Dict[str, float]) -> Dict[str, float]:
        """
        Get recommended constraint weights for specific task type.
        
        Args:
            task_type: Type of cognitive task
            performance_requirements: Performance requirements
            
        Returns:
            Recommended constraint weights
        """
        # Task-specific weight recommendations
        task_weights = {
            "n_back": {
                "working_memory": 0.5,  # Critical for working memory tasks
                "attention_focus": 0.2,
                "processing_speed": 0.2,
                "bias_consistency": 0.1
            },
            "stroop": {
                "working_memory": 0.1,
                "attention_focus": 0.5,  # Critical for attention tasks
                "processing_speed": 0.3,
                "bias_consistency": 0.1
            },
            "planning": {
                "working_memory": 0.3,
                "attention_focus": 0.2,
                "processing_speed": 0.4,  # Planning requires time
                "bias_consistency": 0.1
            },
            "pattern_recognition": {
                "working_memory": 0.2,
                "attention_focus": 0.3,
                "processing_speed": 0.2,
                "bias_consistency": 0.3  # Pattern recognition shows biases
            }
        }
        
        base_weights = task_weights.get(task_type, self.constraint_weights)
        
        # Adjust based on performance requirements
        accuracy_requirement = performance_requirements.get("accuracy", 0.8)
        speed_requirement = performance_requirements.get("speed", 0.5)
        
        # Higher accuracy requirements may need stricter cognitive constraints
        if accuracy_requirement > 0.9:
            for constraint in base_weights:
                base_weights[constraint] *= 1.2
        
        # Higher speed requirements may need relaxed constraints
        if speed_requirement > 0.8:
            base_weights["processing_speed"] *= 0.8
        
        # Normalize
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for constraint in base_weights:
                base_weights[constraint] /= total_weight
        
        return base_weights
    
    def reset_history(self) -> None:
        """Reset all history tracking."""
        self.penalty_history.clear()
        for constraint_name in self.constraint_violations:
            self.constraint_violations[constraint_name].clear()
    
    def get_regularization_summary(self) -> Dict[str, float]:
        """
        Get summary of regularization performance.
        
        Returns:
            Dictionary with regularization summary statistics
        """
        if not self.penalty_history:
            return {"status": "no_data"}
        
        return {
            "mean_penalty": np.mean(self.penalty_history),
            "max_penalty": np.max(self.penalty_history),
            "penalty_trend": np.polyfit(range(len(self.penalty_history)), self.penalty_history, 1)[0],
            "mean_authenticity": 1.0 - np.mean(self.penalty_history),
            "authenticity_stability": 1.0 - np.std(self.penalty_history),
            "num_evaluations": len(self.penalty_history)
        }