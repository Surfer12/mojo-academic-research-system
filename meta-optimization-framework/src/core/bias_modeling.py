"""
Bias Modeling Module

Implements cognitive bias modeling using the β-parameter:
P_biased(H|E) = P(H|E)^β / [P(H|E)^β + (1-P(H|E))^β]

This module simulates human cognitive biases to improve model authenticity
and human-AI collaboration effectiveness.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Enumeration of cognitive bias types."""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    REPRESENTATIVENESS = "representativeness"
    OVERCONFIDENCE = "overconfidence"
    LOSS_AVERSION = "loss_aversion"


class CognitiveBias(ABC):
    """Abstract base class for cognitive bias implementations."""
    
    @abstractmethod
    def apply_bias(self, 
                  probabilities: torch.Tensor,
                  context: Dict[str, torch.Tensor],
                  beta: float) -> torch.Tensor:
        """Apply this bias to probability distributions."""
        pass
    
    @abstractmethod
    def get_bias_name(self) -> str:
        """Get name of this bias."""
        pass


class ConfirmationBias(CognitiveBias):
    """
    Confirmation bias: tendency to favor information that confirms existing beliefs.
    """
    
    def __init__(self, prior_beliefs: Optional[torch.Tensor] = None):
        self.prior_beliefs = prior_beliefs
    
    def apply_bias(self, 
                  probabilities: torch.Tensor,
                  context: Dict[str, torch.Tensor],
                  beta: float) -> torch.Tensor:
        """Apply confirmation bias to probabilities."""
        if self.prior_beliefs is None:
            # Use current probabilities as prior if none provided
            self.prior_beliefs = probabilities.clone()
        
        # Align shapes
        if self.prior_beliefs.shape != probabilities.shape:
            self.prior_beliefs = self._align_tensors(self.prior_beliefs, probabilities)
        
        # Confirmation bias strengthens probabilities that align with priors
        alignment = torch.cosine_similarity(
            probabilities.flatten(), 
            self.prior_beliefs.flatten(), 
            dim=0
        )
        
        # Positive alignment strengthens, negative alignment weakens
        bias_factor = 1.0 + beta * alignment * 0.3  # Scale factor
        biased_probs = probabilities * bias_factor
        
        # Renormalize to maintain probability distribution
        return self._normalize_probabilities(biased_probs)
    
    def get_bias_name(self) -> str:
        return "confirmation"
    
    def _align_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Align tensor shapes for comparison."""
        if tensor1.numel() == tensor2.numel():
            return tensor1.reshape(tensor2.shape)
        else:
            # Use interpolation for different sizes
            return torch.nn.functional.interpolate(
                tensor1.unsqueeze(0).unsqueeze(0),
                size=tensor2.numel(),
                mode='linear'
            ).reshape(tensor2.shape)
    
    def _normalize_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """Normalize to valid probability distribution."""
        probs = torch.clamp(probs, min=1e-8)  # Avoid zeros
        return probs / torch.sum(probs, dim=-1, keepdim=True)


class AnchoringBias(CognitiveBias):
    """
    Anchoring bias: tendency to rely heavily on first piece of information.
    """
    
    def __init__(self):
        self.anchor = None
    
    def apply_bias(self, 
                  probabilities: torch.Tensor,
                  context: Dict[str, torch.Tensor],
                  beta: float) -> torch.Tensor:
        """Apply anchoring bias to probabilities."""
        # Set anchor on first call
        if self.anchor is None:
            self.anchor = probabilities.clone()
        
        # Bias toward anchor value
        anchor_weight = beta * 0.4  # Anchoring strength
        current_weight = 1.0 - anchor_weight
        
        # Align shapes
        if self.anchor.shape != probabilities.shape:
            anchor_aligned = self._align_tensors(self.anchor, probabilities)
        else:
            anchor_aligned = self.anchor
        
        biased_probs = current_weight * probabilities + anchor_weight * anchor_aligned
        return self._normalize_probabilities(biased_probs)
    
    def get_bias_name(self) -> str:
        return "anchoring"
    
    def _align_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Align tensor shapes."""
        if tensor1.numel() == tensor2.numel():
            return tensor1.reshape(tensor2.shape)
        else:
            return torch.nn.functional.interpolate(
                tensor1.unsqueeze(0).unsqueeze(0),
                size=tensor2.numel(),
                mode='linear'
            ).reshape(tensor2.shape)
    
    def _normalize_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """Normalize probabilities."""
        probs = torch.clamp(probs, min=1e-8)
        return probs / torch.sum(probs, dim=-1, keepdim=True)


class AvailabilityBias(CognitiveBias):
    """
    Availability bias: tendency to overweight easily recalled information.
    """
    
    def __init__(self, memory_decay: float = 0.9):
        self.memory_decay = memory_decay
        self.recent_observations = []
    
    def apply_bias(self, 
                  probabilities: torch.Tensor,
                  context: Dict[str, torch.Tensor],
                  beta: float) -> torch.Tensor:
        """Apply availability bias to probabilities."""
        # Store current observation
        self.recent_observations.append(probabilities.clone())
        
        # Decay older observations
        if len(self.recent_observations) > 10:  # Keep last 10 observations
            self.recent_observations = self.recent_observations[-10:]
        
        # Compute availability-weighted average
        if len(self.recent_observations) > 1:
            weights = torch.tensor([
                self.memory_decay ** (len(self.recent_observations) - i - 1)
                for i in range(len(self.recent_observations))
            ])
            weights = weights / torch.sum(weights)
            
            # Weighted average of recent observations
            available_memory = torch.zeros_like(probabilities)
            for i, obs in enumerate(self.recent_observations):
                if obs.shape == probabilities.shape:
                    available_memory += weights[i] * obs
                else:
                    aligned_obs = self._align_tensors(obs, probabilities)
                    available_memory += weights[i] * aligned_obs
            
            # Bias toward available memory
            memory_weight = beta * 0.3
            current_weight = 1.0 - memory_weight
            
            biased_probs = current_weight * probabilities + memory_weight * available_memory
            return self._normalize_probabilities(biased_probs)
        
        return probabilities
    
    def get_bias_name(self) -> str:
        return "availability"
    
    def _align_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Align tensor shapes."""
        if tensor1.numel() == tensor2.numel():
            return tensor1.reshape(tensor2.shape)
        else:
            return torch.nn.functional.interpolate(
                tensor1.unsqueeze(0).unsqueeze(0),
                size=tensor2.numel(),
                mode='linear'
            ).reshape(tensor2.shape)
    
    def _normalize_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """Normalize probabilities."""
        probs = torch.clamp(probs, min=1e-8)
        return probs / torch.sum(probs, dim=-1, keepdim=True)


class OverconfidenceBias(CognitiveBias):
    """
    Overconfidence bias: tendency to overestimate accuracy of beliefs.
    """
    
    def apply_bias(self, 
                  probabilities: torch.Tensor,
                  context: Dict[str, torch.Tensor],
                  beta: float) -> torch.Tensor:
        """Apply overconfidence bias to probabilities."""
        # Increase confidence by sharpening probability distribution
        sharpening_factor = 1.0 + beta * 0.5
        
        # Apply temperature scaling (lower temperature = higher confidence)
        temperature = 1.0 / sharpening_factor
        biased_probs = torch.softmax(torch.log(probabilities + 1e-8) / temperature, dim=-1)
        
        return biased_probs
    
    def get_bias_name(self) -> str:
        return "overconfidence"


class BiasModeler:
    """
    Main bias modeling system implementing P_biased(H|E) transformation.
    
    Applies multiple cognitive biases to model outputs to improve
    human-AI collaboration and cognitive authenticity.
    """
    
    def __init__(self, 
                 active_biases: Optional[List[BiasType]] = None,
                 bias_strengths: Optional[Dict[str, float]] = None):
        """
        Initialize bias modeler.
        
        Args:
            active_biases: List of bias types to apply
            bias_strengths: Strength multipliers for each bias type
        """
        self.active_biases = active_biases or [
            BiasType.CONFIRMATION,
            BiasType.ANCHORING,
            BiasType.AVAILABILITY,
            BiasType.OVERCONFIDENCE
        ]
        
        self.bias_strengths = bias_strengths or {
            "confirmation": 1.0,
            "anchoring": 0.8,
            "availability": 0.6,
            "overconfidence": 0.4
        }
        
        # Initialize bias implementations
        self.bias_implementations = self._initialize_biases()
        
        # History tracking
        self.bias_history: List[Dict[str, float]] = []
        self.beta_history: List[float] = []
    
    def _initialize_biases(self) -> Dict[str, CognitiveBias]:
        """Initialize bias implementation objects."""
        implementations = {}
        
        for bias_type in self.active_biases:
            if bias_type == BiasType.CONFIRMATION:
                implementations["confirmation"] = ConfirmationBias()
            elif bias_type == BiasType.ANCHORING:
                implementations["anchoring"] = AnchoringBias()
            elif bias_type == BiasType.AVAILABILITY:
                implementations["availability"] = AvailabilityBias()
            elif bias_type == BiasType.OVERCONFIDENCE:
                implementations["overconfidence"] = OverconfidenceBias()
        
        return implementations
    
    def apply_bias_modeling(self, 
                          model_output: torch.Tensor,
                          beta: float,
                          context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply cognitive bias modeling to model output.
        
        Implements: P_biased(H|E) = P(H|E)^β / [P(H|E)^β + (1-P(H|E))^β]
        
        Args:
            model_output: Raw model output
            beta: Bias strength parameter
            context: Additional context for bias application
            
        Returns:
            Bias-adjusted output
        """
        if context is None:
            context = {}
        
        # Convert output to probability distribution
        probabilities = self._convert_to_probabilities(model_output)
        
        # Apply core bias transformation
        biased_probs = self._apply_core_bias_formula(probabilities, beta)
        
        # Apply individual cognitive biases
        for bias_name, bias_impl in self.bias_implementations.items():
            try:
                bias_strength = self.bias_strengths.get(bias_name, 1.0)
                effective_beta = beta * bias_strength
                
                biased_probs = bias_impl.apply_bias(biased_probs, context, effective_beta)
                
            except Exception as e:
                logger.warning(f"Error applying {bias_name} bias: {e}")
        
        # Track bias application
        self._track_bias_application(probabilities, biased_probs, beta)
        
        # Convert back to original output format
        return self._convert_from_probabilities(biased_probs, model_output)
    
    def _convert_to_probabilities(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output to probability distribution."""
        if torch.all(output >= 0) and torch.allclose(torch.sum(output, dim=-1), torch.ones(output.shape[:-1])):
            # Already a probability distribution
            return output
        else:
            # Apply softmax to convert to probabilities
            return torch.softmax(output, dim=-1)
    
    def _apply_core_bias_formula(self, probabilities: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Apply core bias formula: P_biased(H|E) = P(H|E)^β / [P(H|E)^β + (1-P(H|E))^β]
        """
        p = probabilities
        p_complement = 1.0 - p
        
        # Avoid numerical issues
        p = torch.clamp(p, min=1e-8, max=1-1e-8)
        p_complement = torch.clamp(p_complement, min=1e-8, max=1-1e-8)
        
        # Apply bias formula
        numerator = torch.pow(p, beta)
        denominator = torch.pow(p, beta) + torch.pow(p_complement, beta)
        
        biased_p = numerator / (denominator + 1e-8)
        
        return biased_p
    
    def _convert_from_probabilities(self, 
                                  probabilities: torch.Tensor,
                                  original_output: torch.Tensor) -> torch.Tensor:
        """Convert probabilities back to original output format."""
        if probabilities.shape == original_output.shape:
            return probabilities
        else:
            # Use logits if original was not probabilities
            return torch.log(probabilities + 1e-8)
    
    def _track_bias_application(self, 
                              original_probs: torch.Tensor,
                              biased_probs: torch.Tensor,
                              beta: float) -> None:
        """Track bias application for analysis."""
        # Compute bias effect metrics
        kl_divergence = torch.nn.functional.kl_div(
            torch.log(biased_probs + 1e-8),
            original_probs,
            reduction='mean'
        ).item()
        
        entropy_change = (
            -torch.sum(biased_probs * torch.log(biased_probs + 1e-8)) -
            (-torch.sum(original_probs * torch.log(original_probs + 1e-8)))
        ).item()
        
        confidence_change = (
            torch.max(biased_probs) - torch.max(original_probs)
        ).item()
        
        bias_metrics = {
            "kl_divergence": kl_divergence,
            "entropy_change": entropy_change,
            "confidence_change": confidence_change,
            "beta": beta
        }
        
        self.bias_history.append(bias_metrics)
        self.beta_history.append(beta)
    
    def analyze_bias_effects(self) -> Dict[str, float]:
        """
        Analyze effects of bias modeling over time.
        
        Returns:
            Dictionary with bias effect statistics
        """
        if not self.bias_history:
            return {"status": "no_data"}
        
        kl_divergences = [h["kl_divergence"] for h in self.bias_history]
        entropy_changes = [h["entropy_change"] for h in self.bias_history]
        confidence_changes = [h["confidence_change"] for h in self.bias_history]
        
        return {
            "mean_kl_divergence": np.mean(kl_divergences),
            "mean_entropy_change": np.mean(entropy_changes),
            "mean_confidence_change": np.mean(confidence_changes),
            "bias_consistency": 1.0 - np.std(kl_divergences),
            "beta_trend": np.polyfit(range(len(self.beta_history)), self.beta_history, 1)[0] if len(self.beta_history) > 1 else 0.0,
            "num_applications": len(self.bias_history)
        }
    
    def calibrate_beta(self, 
                      human_data: torch.Tensor,
                      model_predictions: torch.Tensor,
                      target_similarity: float = 0.8) -> float:
        """
        Calibrate β parameter to match human behavior.
        
        Args:
            human_data: Human decision/judgment data
            model_predictions: Model predictions
            target_similarity: Target similarity to human behavior
            
        Returns:
            Calibrated β parameter
        """
        best_beta = 1.0
        best_similarity = 0.0
        
        # Grid search for optimal β
        for beta in np.linspace(0.1, 3.0, 30):
            biased_predictions = self.apply_bias_modeling(model_predictions, beta)
            
            # Compute similarity to human data
            similarity = self._compute_similarity(human_data, biased_predictions)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_beta = beta
        
        logger.info(f"Calibrated β = {best_beta:.3f} (similarity = {best_similarity:.3f})")
        return best_beta
    
    def _compute_similarity(self, 
                          human_data: torch.Tensor,
                          model_data: torch.Tensor) -> float:
        """Compute similarity between human and model data."""
        # Use cosine similarity as default metric
        human_flat = human_data.flatten()
        model_flat = model_data.flatten()
        
        if human_flat.shape != model_flat.shape:
            # Align shapes
            min_size = min(len(human_flat), len(model_flat))
            human_flat = human_flat[:min_size]
            model_flat = model_flat[:min_size]
        
        similarity = torch.cosine_similarity(human_flat, model_flat, dim=0).item()
        return max(0.0, similarity)  # Ensure non-negative
    
    def get_bias_recommendations(self, 
                               task_type: str,
                               collaboration_mode: str) -> Dict[str, float]:
        """
        Get recommended bias settings for specific task and collaboration mode.
        
        Args:
            task_type: Type of cognitive task
            collaboration_mode: Type of human-AI collaboration
            
        Returns:
            Recommended bias strength settings
        """
        # Task-specific bias recommendations
        task_biases = {
            "n_back": {
                "confirmation": 0.3,  # Working memory shows confirmation bias
                "anchoring": 0.2,
                "availability": 0.4,  # Recent items more available
                "overconfidence": 0.1
            },
            "stroop": {
                "confirmation": 0.2,
                "anchoring": 0.1,
                "availability": 0.2,
                "overconfidence": 0.3  # Attention tasks show overconfidence
            },
            "planning": {
                "confirmation": 0.4,  # Planning shows strong confirmation bias
                "anchoring": 0.3,     # Initial plans anchor subsequent decisions
                "availability": 0.2,
                "overconfidence": 0.2
            },
            "pattern_recognition": {
                "confirmation": 0.3,
                "anchoring": 0.1,
                "availability": 0.3,  # Recent patterns more available
                "overconfidence": 0.4  # Pattern recognition shows overconfidence
            }
        }
        
        # Collaboration mode adjustments
        collaboration_adjustments = {
            "competitive": 0.8,    # Reduce biases for competitive settings
            "cooperative": 1.2,    # Increase biases for cooperative settings
            "advisory": 1.0,       # Neutral for advisory settings
            "autonomous": 0.6      # Reduce biases for autonomous operation
        }
        
        base_biases = task_biases.get(task_type, self.bias_strengths)
        adjustment = collaboration_adjustments.get(collaboration_mode, 1.0)
        
        # Apply adjustment
        recommended_biases = {
            bias_name: strength * adjustment
            for bias_name, strength in base_biases.items()
        }
        
        return recommended_biases
    
    def reset_history(self) -> None:
        """Reset all history tracking."""
        self.bias_history.clear()
        self.beta_history.clear()
        
        # Reset bias implementations that maintain state
        for bias_impl in self.bias_implementations.values():
            if hasattr(bias_impl, 'reset'):
                bias_impl.reset()
    
    def get_bias_summary(self) -> Dict[str, float]:
        """
        Get summary of bias modeling performance.
        
        Returns:
            Dictionary with bias modeling summary statistics
        """
        if not self.bias_history:
            return {"status": "no_data"}
        
        analysis = self.analyze_bias_effects()
        
        return {
            "active_biases": len(self.bias_implementations),
            "mean_bias_effect": analysis["mean_kl_divergence"],
            "bias_stability": analysis["bias_consistency"],
            "mean_beta": np.mean(self.beta_history),
            "beta_range": np.max(self.beta_history) - np.min(self.beta_history),
            "num_applications": len(self.bias_history)
        }