"""
Adaptive Optimizer - Self-Adjusting Optimization Strategies
==========================================================

Implements adaptive optimization algorithms that automatically adjust their
parameters and strategies based on problem characteristics and performance feedback.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from ..core.meta_optimization import TaskSpecification, OptimizationResult
from ..utils.statistical_analysis import StatisticalAnalyzer


class AdaptationStrategy(Enum):
    """Types of adaptation strategies."""
    GRADIENT_BASED = "gradient_based"
    POPULATION_BASED = "population_based"
    BAYESIAN = "bayesian"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EVOLUTIONARY = "evolutionary"


@dataclass
class AdaptationState:
    """State of the adaptive optimization process."""
    iteration: int
    current_parameters: Dict[str, float]
    performance_history: List[float]
    adaptation_history: List[Dict[str, float]]
    convergence_status: str
    adaptation_rate: float


class AdaptiveOptimizer:
    """
    Adaptive optimization algorithm that self-adjusts based on problem characteristics.
    
    Automatically adapts learning rates, regularization parameters, and optimization
    strategies based on convergence patterns, problem complexity, and performance feedback.
    """
    
    def __init__(self,
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.GRADIENT_BASED,
                 learning_rate: float = 0.01,
                 adaptation_rate: float = 0.1,
                 patience: int = 10,
                 min_improvement: float = 1e-6):
        """
        Initialize adaptive optimizer.
        
        Args:
            adaptation_strategy: Strategy for parameter adaptation
            learning_rate: Initial learning rate
            adaptation_rate: Rate of parameter adaptation
            patience: Patience for convergence detection
            min_improvement: Minimum improvement threshold
        """
        self.adaptation_strategy = adaptation_strategy
        self.base_learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.logger = logging.getLogger(__name__)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Adaptive parameters
        self.adaptive_params = {
            'learning_rate': learning_rate,
            'momentum': 0.9,
            'regularization': 0.01,
            'batch_size': 32,
            'exploration_factor': 0.1
        }
        
        # Adaptation state
        self.adaptation_state = AdaptationState(
            iteration=0,
            current_parameters=self.adaptive_params.copy(),
            performance_history=[],
            adaptation_history=[],
            convergence_status='running',
            adaptation_rate=adaptation_rate
        )
        
        # Problem analysis
        self.problem_characteristics = {}
        
    def optimize(self,
                task_specification: TaskSpecification,
                max_iterations: int = 1000,
                convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Perform adaptive optimization.
        
        Args:
            task_specification: Task to optimize
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Optimization results with adaptation history
        """
        # Analyze problem characteristics
        self._analyze_problem(task_specification)
        
        # Initialize optimization state
        self._reset_optimization_state()
        
        # Main optimization loop
        for iteration in range(max_iterations):
            self.adaptation_state.iteration = iteration
            
            # Perform optimization step
            step_result = self._optimization_step(task_specification)
            
            # Record performance
            self.adaptation_state.performance_history.append(step_result['loss'])
            
            # Adapt parameters based on performance
            self._adapt_parameters(step_result)
            
            # Check convergence
            if self._check_convergence(convergence_threshold):
                self.adaptation_state.convergence_status = 'converged'
                break
            
            # Log progress
            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}: Loss = {step_result['loss']:.6f}, "
                               f"LR = {self.adaptive_params['learning_rate']:.6f}")
        
        # Create final result
        final_result = OptimizationResult(
            success=self.adaptation_state.convergence_status == 'converged',
            final_loss=self.adaptation_state.performance_history[-1] if self.adaptation_state.performance_history else float('inf'),
            iterations=self.adaptation_state.iteration + 1,
            convergence_history=self.adaptation_state.performance_history,
            final_parameters=self.adaptive_params.copy(),
            final_metrics={
                'adaptation_state': self.adaptation_state,
                'problem_characteristics': self.problem_characteristics,
                'final_adaptation_rate': self.adaptation_state.adaptation_rate
            }
        )
        
        return final_result
    
    def _analyze_problem(self, task_specification: TaskSpecification) -> None:
        """Analyze problem characteristics to inform adaptation."""
        
        # Extract problem characteristics
        data = task_specification.data
        
        characteristics = {
            'dimensionality': 0,
            'complexity': 'medium',
            'noise_level': 0.0,
            'ill_conditioning': 1.0,
            'separability': 'separable'
        }
        
        if hasattr(data, 'shape'):
            characteristics['dimensionality'] = np.prod(data.shape) if len(data.shape) > 1 else len(data)
        elif isinstance(data, (list, tuple)):
            characteristics['dimensionality'] = len(data)
        
        # Estimate problem complexity based on dimensionality
        if characteristics['dimensionality'] < 10:
            characteristics['complexity'] = 'low'
        elif characteristics['dimensionality'] < 100:
            characteristics['complexity'] = 'medium'
        else:
            characteristics['complexity'] = 'high'
        
        # Estimate noise level (simplified)
        if hasattr(data, 'std'):
            characteristics['noise_level'] = float(data.std() / data.mean()) if data.mean() != 0 else 0.0
        
        # Set initial parameters based on problem characteristics
        self._set_initial_parameters(characteristics)
        
        self.problem_characteristics = characteristics
        
        self.logger.info(f"Problem analysis: {characteristics}")
    
    def _set_initial_parameters(self, characteristics: Dict[str, Any]) -> None:
        """Set initial parameters based on problem characteristics."""
        
        # Adjust learning rate based on dimensionality
        if characteristics['complexity'] == 'low':
            self.adaptive_params['learning_rate'] = self.base_learning_rate * 2.0
        elif characteristics['complexity'] == 'high':
            self.adaptive_params['learning_rate'] = self.base_learning_rate * 0.5
        
        # Adjust regularization based on complexity
        if characteristics['complexity'] == 'high':
            self.adaptive_params['regularization'] = 0.1
        elif characteristics['complexity'] == 'low':
            self.adaptive_params['regularization'] = 0.001
        
        # Adjust batch size based on dimensionality
        if characteristics['dimensionality'] > 1000:
            self.adaptive_params['batch_size'] = 128
        elif characteristics['dimensionality'] < 10:
            self.adaptive_params['batch_size'] = 8
    
    def _reset_optimization_state(self) -> None:
        """Reset optimization state for new optimization run."""
        self.adaptation_state = AdaptationState(
            iteration=0,
            current_parameters=self.adaptive_params.copy(),
            performance_history=[],
            adaptation_history=[],
            convergence_status='running',
            adaptation_rate=self.adaptation_rate
        )
    
    def _optimization_step(self, task_specification: TaskSpecification) -> Dict[str, Any]:
        """Perform a single optimization step."""
        
        # Simulate optimization step (in real implementation, this would call the actual optimizer)
        current_loss = self._evaluate_objective(task_specification)
        
        # Compute gradients (simulated)
        gradient_norm = np.random.exponential(1.0)  # Simulated gradient norm
        
        # Update based on current strategy
        if self.adaptation_strategy == AdaptationStrategy.GRADIENT_BASED:
            update_info = self._gradient_based_update(current_loss, gradient_norm)
        elif self.adaptation_strategy == AdaptationStrategy.POPULATION_BASED:
            update_info = self._population_based_update(current_loss)
        elif self.adaptation_strategy == AdaptationStrategy.BAYESIAN:
            update_info = self._bayesian_update(current_loss)
        else:
            update_info = self._default_update(current_loss)
        
        return {
            'loss': current_loss,
            'gradient_norm': gradient_norm,
            'update_info': update_info
        }
    
    def _evaluate_objective(self, task_specification: TaskSpecification) -> float:
        """Evaluate the objective function (simulated)."""
        
        # Simulate objective evaluation with some noise and trend
        base_loss = 1.0
        
        # Add convergence trend
        if self.adaptation_state.performance_history:
            last_loss = self.adaptation_state.performance_history[-1]
            improvement = self.adaptive_params['learning_rate'] * 0.1
            base_loss = max(0.01, last_loss - improvement + np.random.normal(0, 0.01))
        
        # Add regularization effect
        regularization_penalty = self.adaptive_params['regularization'] * 0.1
        
        return base_loss + regularization_penalty
    
    def _gradient_based_update(self, current_loss: float, gradient_norm: float) -> Dict[str, Any]:
        """Gradient-based parameter update."""
        
        # Adapt learning rate based on gradient behavior
        if len(self.adaptation_state.performance_history) > 1:
            loss_change = current_loss - self.adaptation_state.performance_history[-1]
            
            if loss_change > 0:  # Loss increased
                self.adaptive_params['learning_rate'] *= 0.9  # Reduce learning rate
            elif loss_change < -self.min_improvement:  # Good improvement
                self.adaptive_params['learning_rate'] *= 1.05  # Increase learning rate slightly
        
        # Adapt momentum based on gradient norm stability
        if gradient_norm > 10.0:  # High gradient norm
            self.adaptive_params['momentum'] = min(0.99, self.adaptive_params['momentum'] * 1.02)
        elif gradient_norm < 0.1:  # Low gradient norm
            self.adaptive_params['momentum'] = max(0.5, self.adaptive_params['momentum'] * 0.98)
        
        return {
            'method': 'gradient_based',
            'gradient_norm': gradient_norm,
            'lr_adjustment': 'adaptive'
        }
    
    def _population_based_update(self, current_loss: float) -> Dict[str, Any]:
        """Population-based parameter update."""
        
        # Simulate population-based optimization
        if len(self.adaptation_state.performance_history) > 5:
            recent_performance = self.adaptation_state.performance_history[-5:]
            performance_variance = np.var(recent_performance)
            
            # High variance suggests need for exploration
            if performance_variance > 0.01:
                self.adaptive_params['exploration_factor'] *= 1.1
            else:
                self.adaptive_params['exploration_factor'] *= 0.95
            
            # Adjust population diversity (simulated through regularization)
            if performance_variance < 0.001:  # Low diversity
                self.adaptive_params['regularization'] *= 0.9
        
        return {
            'method': 'population_based',
            'exploration_factor': self.adaptive_params['exploration_factor']
        }
    
    def _bayesian_update(self, current_loss: float) -> Dict[str, Any]:
        """Bayesian optimization-based parameter update."""
        
        # Simulate Bayesian parameter optimization
        if len(self.adaptation_state.performance_history) > 3:
            # Use performance history to guide parameter selection
            performance_trend = np.polyfit(
                range(len(self.adaptation_state.performance_history)),
                self.adaptation_state.performance_history,
                1
            )[0]  # Linear trend coefficient
            
            # If improving (negative trend), maintain current strategy
            if performance_trend < 0:
                pass  # Keep current parameters
            else:
                # Try different parameter combinations
                self.adaptive_params['learning_rate'] *= np.random.uniform(0.8, 1.2)
                self.adaptive_params['regularization'] *= np.random.uniform(0.9, 1.1)
        
        return {
            'method': 'bayesian',
            'performance_trend': performance_trend if 'performance_trend' in locals() else 0.0
        }
    
    def _default_update(self, current_loss: float) -> Dict[str, Any]:
        """Default parameter update strategy."""
        
        # Simple adaptive strategy
        if len(self.adaptation_state.performance_history) > 2:
            recent_losses = self.adaptation_state.performance_history[-3:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                # Consistent improvement
                self.adaptive_params['learning_rate'] *= 1.05
            else:
                # Inconsistent performance
                self.adaptive_params['learning_rate'] *= 0.95
        
        return {
            'method': 'default',
            'adjustment': 'learning_rate_only'
        }
    
    def _adapt_parameters(self, step_result: Dict[str, Any]) -> None:
        """Adapt parameters based on optimization step results."""
        
        # Record current parameters
        current_params = self.adaptive_params.copy()
        self.adaptation_state.adaptation_history.append(current_params)
        
        # Ensure parameters stay within reasonable bounds
        self.adaptive_params['learning_rate'] = np.clip(
            self.adaptive_params['learning_rate'], 
            self.base_learning_rate * 0.01, 
            self.base_learning_rate * 10.0
        )
        
        self.adaptive_params['momentum'] = np.clip(
            self.adaptive_params['momentum'], 0.1, 0.99
        )
        
        self.adaptive_params['regularization'] = np.clip(
            self.adaptive_params['regularization'], 1e-6, 1.0
        )
        
        # Adapt the adaptation rate itself
        if len(self.adaptation_state.performance_history) > 10:
            recent_variance = np.var(self.adaptation_state.performance_history[-10:])
            if recent_variance < 1e-6:  # Very stable
                self.adaptation_state.adaptation_rate *= 0.95
            elif recent_variance > 0.1:  # Very unstable
                self.adaptation_state.adaptation_rate *= 1.05
    
    def _check_convergence(self, threshold: float) -> bool:
        """Check if optimization has converged."""
        
        if len(self.adaptation_state.performance_history) < self.patience:
            return False
        
        # Check for improvement in recent iterations
        recent_losses = self.adaptation_state.performance_history[-self.patience:]
        improvement = recent_losses[0] - recent_losses[-1]
        
        # Convergence if improvement is below threshold
        if improvement < threshold:
            return True
        
        # Check for oscillation
        if len(self.adaptation_state.performance_history) > 2 * self.patience:
            long_recent = self.adaptation_state.performance_history[-2*self.patience:-self.patience]
            short_recent = self.adaptation_state.performance_history[-self.patience:]
            
            if abs(np.mean(long_recent) - np.mean(short_recent)) < threshold:
                return True
        
        return False
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation process."""
        
        if not self.adaptation_state.performance_history:
            return {'status': 'not_started'}
        
        total_improvement = (
            self.adaptation_state.performance_history[0] - 
            self.adaptation_state.performance_history[-1]
        )
        
        # Analyze parameter evolution
        parameter_evolution = {}
        if self.adaptation_state.adaptation_history:
            for param_name in self.adaptive_params.keys():
                values = [params[param_name] for params in self.adaptation_state.adaptation_history]
                parameter_evolution[param_name] = {
                    'initial': values[0] if values else 0,
                    'final': values[-1] if values else 0,
                    'range': (min(values), max(values)) if values else (0, 0),
                    'variance': np.var(values) if values else 0
                }
        
        return {
            'status': self.adaptation_state.convergence_status,
            'total_iterations': self.adaptation_state.iteration + 1,
            'total_improvement': total_improvement,
            'final_parameters': self.adaptive_params.copy(),
            'parameter_evolution': parameter_evolution,
            'problem_characteristics': self.problem_characteristics,
            'adaptation_strategy': self.adaptation_strategy.value,
            'convergence_rate': self._estimate_convergence_rate()
        }
    
    def _estimate_convergence_rate(self) -> float:
        """Estimate the convergence rate of the optimization."""
        
        if len(self.adaptation_state.performance_history) < 10:
            return 0.0
        
        # Fit exponential decay to loss history
        losses = np.array(self.adaptation_state.performance_history)
        iterations = np.arange(len(losses))
        
        try:
            # Log-linear fit for exponential decay rate
            log_losses = np.log(losses + 1e-10)  # Avoid log(0)
            coeffs = np.polyfit(iterations, log_losses, 1)
            convergence_rate = -coeffs[0]  # Negative slope indicates convergence
            return max(0.0, convergence_rate)
        except:
            return 0.0
    
    def visualize_adaptation(self, output_file: Optional[Path] = None) -> None:
        """Visualize the adaptation process."""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss evolution
            ax1.plot(self.adaptation_state.performance_history)
            ax1.set_title('Loss Evolution')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Learning rate adaptation
            if self.adaptation_state.adaptation_history:
                lr_values = [params['learning_rate'] for params in self.adaptation_state.adaptation_history]
                ax2.plot(lr_values)
                ax2.set_title('Learning Rate Adaptation')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Learning Rate')
                ax2.grid(True)
            
            # Regularization adaptation
            if self.adaptation_state.adaptation_history:
                reg_values = [params['regularization'] for params in self.adaptation_state.adaptation_history]
                ax3.plot(reg_values)
                ax3.set_title('Regularization Adaptation')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Regularization')
                ax3.grid(True)
            
            # Momentum adaptation
            if self.adaptation_state.adaptation_history:
                momentum_values = [params['momentum'] for params in self.adaptation_state.adaptation_history]
                ax4.plot(momentum_values)
                ax4.set_title('Momentum Adaptation')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Momentum')
                ax4.grid(True)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
    
    def generate_adaptation_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive adaptation report."""
        
        summary = self.get_adaptation_summary()
        
        report_lines = [
            "# Adaptive Optimization Report\n\n",
            f"## Optimization Summary\n",
            f"- Status: {summary['status']}\n",
            f"- Total Iterations: {summary['total_iterations']}\n",
            f"- Total Improvement: {summary['total_improvement']:.6f}\n",
            f"- Adaptation Strategy: {summary['adaptation_strategy']}\n",
            f"- Convergence Rate: {summary['convergence_rate']:.6f}\n\n",
            
            "## Problem Characteristics\n"
        ]
        
        for key, value in summary['problem_characteristics'].items():
            report_lines.append(f"- {key}: {value}\n")
        
        report_lines.extend([
            "\n## Final Parameters\n"
        ])
        
        for param, value in summary['final_parameters'].items():
            report_lines.append(f"- {param}: {value:.6f}\n")
        
        if summary['parameter_evolution']:
            report_lines.extend([
                "\n## Parameter Evolution\n"
            ])
            
            for param, evolution in summary['parameter_evolution'].items():
                report_lines.extend([
                    f"### {param}\n",
                    f"- Initial: {evolution['initial']:.6f}\n",
                    f"- Final: {evolution['final']:.6f}\n",
                    f"- Range: [{evolution['range'][0]:.6f}, {evolution['range'][1]:.6f}]\n",
                    f"- Variance: {evolution['variance']:.6f}\n\n"
                ])
        
        report_content = "".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content