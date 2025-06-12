"""
Dynamic Integration Module

Implements the α-parameter negotiation for symbolic-neural integration:
H(x) = αS(x) + (1-α)N(x)

This module handles the dynamic weighting between symbolic and neural components
based on task requirements and performance feedback.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicIntegrator:
    """
    Handles dynamic integration between symbolic and neural components.
    
    The integration follows: H(x) = αS(x) + (1-α)N(x)
    where α is adaptively determined based on:
    - Task complexity
    - Component performance
    - Cognitive authenticity requirements
    """
    
    def __init__(self, 
                 initial_alpha: float = 0.5,
                 adaptation_rate: float = 0.01,
                 min_alpha: float = 0.1,
                 max_alpha: float = 0.9):
        """
        Initialize dynamic integrator.
        
        Args:
            initial_alpha: Initial weighting parameter (0.5 = balanced)
            adaptation_rate: Rate of α adaptation
            min_alpha: Minimum allowed α value
            max_alpha: Maximum allowed α value
        """
        self.alpha = initial_alpha
        self.adaptation_rate = adaptation_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # History tracking
        self.alpha_history: List[float] = []
        self.performance_history: List[float] = []
        self.symbolic_contributions: List[float] = []
        self.neural_contributions: List[float] = []
        
    def integrate(self, 
                 symbolic_output: torch.Tensor,
                 neural_output: torch.Tensor,
                 alpha: Optional[float] = None) -> torch.Tensor:
        """
        Integrate symbolic and neural outputs using dynamic weighting.
        
        Args:
            symbolic_output: Output from symbolic reasoning component S(x)
            neural_output: Output from neural processing component N(x)
            alpha: Optional override for α parameter
            
        Returns:
            Integrated output H(x) = αS(x) + (1-α)N(x)
        """
        if alpha is None:
            alpha = self.alpha
            
        # Ensure outputs have compatible shapes
        symbolic_output, neural_output = self._align_outputs(symbolic_output, neural_output)
        
        # Compute weighted integration
        integrated_output = alpha * symbolic_output + (1 - alpha) * neural_output
        
        # Track contributions for analysis
        self._track_contributions(symbolic_output, neural_output, alpha)
        
        return integrated_output
    
    def adapt_alpha(self, 
                   symbolic_performance: float,
                   neural_performance: float,
                   task_complexity: float,
                   cognitive_authenticity: float) -> float:
        """
        Adapt α parameter based on component performance and task requirements.
        
        Args:
            symbolic_performance: Performance of symbolic component
            neural_performance: Performance of neural component
            task_complexity: Complexity of current task (0-1)
            cognitive_authenticity: Required cognitive authenticity (0-1)
            
        Returns:
            Updated α parameter
        """
        # Performance-based adaptation
        performance_ratio = symbolic_performance / (symbolic_performance + neural_performance + 1e-8)
        
        # Task complexity influence (complex tasks may benefit from neural processing)
        complexity_bias = -0.2 * task_complexity  # Bias toward neural for complex tasks
        
        # Cognitive authenticity influence (high authenticity favors symbolic)
        authenticity_bias = 0.3 * cognitive_authenticity
        
        # Compute target α
        target_alpha = performance_ratio + complexity_bias + authenticity_bias
        target_alpha = np.clip(target_alpha, self.min_alpha, self.max_alpha)
        
        # Gradual adaptation
        alpha_update = self.adaptation_rate * (target_alpha - self.alpha)
        self.alpha = np.clip(self.alpha + alpha_update, self.min_alpha, self.max_alpha)
        
        # Store history
        self.alpha_history.append(self.alpha)
        
        logger.debug(f"α adapted: {self.alpha:.3f} (target: {target_alpha:.3f})")
        
        return self.alpha
    
    def compute_integration_quality(self, 
                                  symbolic_output: torch.Tensor,
                                  neural_output: torch.Tensor,
                                  integrated_output: torch.Tensor) -> float:
        """
        Compute quality metric for the integration.
        
        Args:
            symbolic_output: Symbolic component output
            neural_output: Neural component output
            integrated_output: Integrated output
            
        Returns:
            Integration quality score (0-1)
        """
        # Measure how well integration preserves information from both components
        symbolic_similarity = torch.cosine_similarity(
            integrated_output.flatten(), 
            symbolic_output.flatten(), 
            dim=0
        ).item()
        
        neural_similarity = torch.cosine_similarity(
            integrated_output.flatten(), 
            neural_output.flatten(), 
            dim=0
        ).item()
        
        # Quality is high when integration preserves information from both components
        quality = (symbolic_similarity + neural_similarity) / 2.0
        
        return max(0.0, quality)  # Ensure non-negative
    
    def analyze_component_synergy(self) -> dict:
        """
        Analyze synergy between symbolic and neural components.
        
        Returns:
            Dictionary with synergy analysis metrics
        """
        if len(self.symbolic_contributions) < 2:
            return {"synergy_score": 0.0, "dominant_component": "unknown"}
        
        symbolic_contrib = np.array(self.symbolic_contributions)
        neural_contrib = np.array(self.neural_contributions)
        
        # Compute correlation between contributions
        correlation = np.corrcoef(symbolic_contrib, neural_contrib)[0, 1]
        
        # Compute dominance
        avg_symbolic = np.mean(symbolic_contrib)
        avg_neural = np.mean(neural_contrib)
        
        if avg_symbolic > avg_neural:
            dominant_component = "symbolic"
            dominance_ratio = avg_symbolic / (avg_neural + 1e-8)
        else:
            dominant_component = "neural"
            dominance_ratio = avg_neural / (avg_symbolic + 1e-8)
        
        # Synergy score (high when components are complementary)
        synergy_score = 1.0 - abs(correlation)  # High synergy when low correlation
        
        return {
            "synergy_score": synergy_score,
            "dominant_component": dominant_component,
            "dominance_ratio": dominance_ratio,
            "correlation": correlation,
            "avg_alpha": np.mean(self.alpha_history) if self.alpha_history else self.alpha
        }
    
    def get_optimal_alpha_range(self, 
                               task_type: str,
                               cognitive_constraints: dict) -> Tuple[float, float]:
        """
        Get optimal α range for specific task type and constraints.
        
        Args:
            task_type: Type of cognitive task
            cognitive_constraints: Cognitive authenticity constraints
            
        Returns:
            Tuple of (min_optimal_alpha, max_optimal_alpha)
        """
        # Task-specific α ranges based on cognitive science literature
        task_ranges = {
            "n_back": (0.6, 0.8),      # Working memory benefits from symbolic reasoning
            "stroop": (0.3, 0.6),      # Attention tasks benefit from neural processing
            "planning": (0.7, 0.9),    # Executive function favors symbolic reasoning
            "pattern_recognition": (0.2, 0.5)  # Perceptual tasks favor neural processing
        }
        
        base_range = task_ranges.get(task_type, (0.4, 0.6))
        
        # Adjust based on cognitive constraints
        authenticity_requirement = cognitive_constraints.get("authenticity", 0.5)
        
        # Higher authenticity requirements shift toward symbolic reasoning
        authenticity_shift = 0.2 * (authenticity_requirement - 0.5)
        
        optimal_min = np.clip(base_range[0] + authenticity_shift, self.min_alpha, self.max_alpha)
        optimal_max = np.clip(base_range[1] + authenticity_shift, self.min_alpha, self.max_alpha)
        
        return optimal_min, optimal_max
    
    def _align_outputs(self, 
                      symbolic_output: torch.Tensor,
                      neural_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align symbolic and neural outputs to compatible shapes.
        
        Args:
            symbolic_output: Symbolic component output
            neural_output: Neural component output
            
        Returns:
            Tuple of aligned outputs
        """
        if symbolic_output.shape != neural_output.shape:
            # Handle shape mismatch by reshaping or padding
            target_shape = symbolic_output.shape
            
            if neural_output.numel() == symbolic_output.numel():
                # Same number of elements, just reshape
                neural_output = neural_output.reshape(target_shape)
            else:
                # Different number of elements, use simpler alignment
                if len(target_shape) == 3:  # Sequence data [batch, seq, features]
                    batch_size, seq_len, features = target_shape
                    
                    # Reshape neural output to match
                    if neural_output.shape[0] == batch_size:
                        # Same batch size, adjust sequence and features
                        neural_flat = neural_output.reshape(batch_size, -1)
                        target_size = seq_len * features
                        
                        if neural_flat.shape[1] > target_size:
                            # Truncate
                            neural_flat = neural_flat[:, :target_size]
                        elif neural_flat.shape[1] < target_size:
                            # Pad
                            padding = target_size - neural_flat.shape[1]
                            neural_flat = torch.nn.functional.pad(neural_flat, (0, padding))
                        
                        neural_output = neural_flat.reshape(batch_size, seq_len, features)
                    else:
                        # Different batch size, use adaptive pooling
                        neural_output = torch.nn.functional.adaptive_avg_pool1d(
                            neural_output.flatten().unsqueeze(0).unsqueeze(0),
                            symbolic_output.numel()
                        ).reshape(target_shape)
                else:
                    # For other shapes, use adaptive pooling
                    neural_output = torch.nn.functional.adaptive_avg_pool1d(
                        neural_output.flatten().unsqueeze(0).unsqueeze(0),
                        symbolic_output.numel()
                    ).reshape(target_shape)
        
        return symbolic_output, neural_output
    
    def _track_contributions(self, 
                           symbolic_output: torch.Tensor,
                           neural_output: torch.Tensor,
                           alpha: float) -> None:
        """
        Track contributions of symbolic and neural components.
        
        Args:
            symbolic_output: Symbolic component output
            neural_output: Neural component output
            alpha: Current α parameter
        """
        # Compute relative contribution magnitudes
        symbolic_magnitude = torch.norm(symbolic_output).item()
        neural_magnitude = torch.norm(neural_output).item()
        
        # Weight by α parameter
        symbolic_contribution = alpha * symbolic_magnitude
        neural_contribution = (1 - alpha) * neural_magnitude
        
        self.symbolic_contributions.append(symbolic_contribution)
        self.neural_contributions.append(neural_contribution)
    
    def reset_history(self) -> None:
        """Reset all history tracking."""
        self.alpha_history.clear()
        self.performance_history.clear()
        self.symbolic_contributions.clear()
        self.neural_contributions.clear()
    
    def get_integration_summary(self) -> dict:
        """
        Get summary of integration performance.
        
        Returns:
            Dictionary with integration summary statistics
        """
        if not self.alpha_history:
            return {"status": "no_data"}
        
        return {
            "current_alpha": self.alpha,
            "alpha_mean": np.mean(self.alpha_history),
            "alpha_std": np.std(self.alpha_history),
            "alpha_trend": np.polyfit(range(len(self.alpha_history)), self.alpha_history, 1)[0],
            "symbolic_dominance": np.mean(self.symbolic_contributions) / 
                                (np.mean(self.symbolic_contributions) + np.mean(self.neural_contributions) + 1e-8),
            "integration_stability": 1.0 - np.std(self.alpha_history),
            "num_adaptations": len(self.alpha_history)
        }