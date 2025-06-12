"""
Hybrid Optimizer - Combining Multiple Optimization Approaches
============================================================

Implements a hybrid optimization framework that dynamically combines
different optimization strategies based on problem characteristics and
performance feedback. Inspired by attention-recognition decoupling principles.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ..core.meta_optimization import TaskSpecification, OptimizationResult
from ..utils.statistical_analysis import StatisticalAnalyzer
from .adaptive_optimizer import AdaptiveOptimizer, AdaptationStrategy


class OptimizationMode(Enum):
    """Optimization modes for hybrid optimizer."""
    GRADIENT = "gradient"
    POPULATION = "population"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


@dataclass
class HybridState:
    """State of the hybrid optimization process."""
    current_mode: OptimizationMode
    mode_history: List[OptimizationMode]
    performance_by_mode: Dict[OptimizationMode, List[float]]
    switching_points: List[int]
    ensemble_weights: Dict[OptimizationMode, float]
    exploration_exploitation_ratio: float


class ModeSelector(nn.Module):
    """Neural network for selecting optimization mode based on problem state."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_modes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_modes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict mode probabilities."""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=-1)


class HybridOptimizer:
    """
    Hybrid optimizer that combines multiple optimization strategies.

    Dynamically switches between gradient-based, population-based, and
    Bayesian optimization based on the problem landscape and performance.
    Implements attention-like mechanisms for strategy selection.
    """

    def __init__(self,
                 initial_mode: OptimizationMode = OptimizationMode.ADAPTIVE,
                 switching_threshold: float = 0.1,
                 ensemble_size: int = 3,
                 exploration_ratio: float = 0.3,
                 use_neural_selector: bool = True,
                 device: str = "cpu"):
        """
        Initialize hybrid optimizer.

        Args:
            initial_mode: Starting optimization mode
            switching_threshold: Threshold for mode switching
            ensemble_size: Number of optimizers in ensemble mode
            exploration_ratio: Initial exploration/exploitation ratio
            use_neural_selector: Whether to use neural network for mode selection
            device: Device for computation
        """
        self.current_mode = initial_mode
        self.switching_threshold = switching_threshold
        self.ensemble_size = ensemble_size
        self.exploration_ratio = exploration_ratio
        self.device = torch.device(device)

        # Initialize mode selector
        self.use_neural_selector = use_neural_selector
        if use_neural_selector:
            self.mode_selector = ModeSelector().to(self.device)
            self.selector_optimizer = torch.optim.Adam(
                self.mode_selector.parameters(), lr=0.001
            )

        # Initialize component optimizers
        self._init_component_optimizers()

        # State tracking
        self.state = HybridState(
            current_mode=initial_mode,
            mode_history=[initial_mode],
            performance_by_mode={mode: [] for mode in OptimizationMode},
            switching_points=[],
            ensemble_weights={mode: 1.0/len(OptimizationMode) for mode in OptimizationMode},
            exploration_exploitation_ratio=exploration_ratio
        )

        # Statistical analyzer
        self.analyzer = StatisticalAnalyzer()

        # Logging
        self.logger = logging.getLogger(__name__)

    def _init_component_optimizers(self):
        """Initialize component optimization strategies."""
        # Gradient-based optimizer
        self.gradient_optimizer = AdaptiveOptimizer(
            adaptation_strategy=AdaptationStrategy.GRADIENT_BASED
        )

        # Population-based settings
        self.population_size = 50

        # Bayesian optimizer
        self.gp_kernel = Matern(length_scale=1.0, nu=2.5)
        self.acquisition_function = self._upper_confidence_bound

    def optimize(self,
                 task_specification: TaskSpecification,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Optimize using hybrid approach with dynamic mode switching.

        Args:
            task_specification: The optimization task
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold

        Returns:
            OptimizationResult containing the optimization outcome
        """
        self.logger.info(f"Starting hybrid optimization in {self.current_mode} mode")

        # Initialize optimization state
        dim = task_specification.optimization_params.get('dimension', 10)
        best_params = np.random.randn(dim) * 0.1
        best_loss = float('inf')

        loss_history = []
        mode_switches = []
        iterations_per_mode = {mode: 0 for mode in OptimizationMode}

        # Main optimization loop
        iteration = 0
        while iteration < max_iterations:
            # Determine batch size for current mode
            if self.current_mode == OptimizationMode.ENSEMBLE:
                batch_iterations = 10
            else:
                batch_iterations = min(50, max_iterations - iteration)

            # Run optimization with current mode
            mode_result = self._run_mode_optimization(
                task_specification,
                best_params,
                batch_iterations,
                convergence_threshold
            )

            # Update best solution
            if mode_result['best_loss'] < best_loss:
                best_loss = mode_result['best_loss']
                best_params = mode_result['best_params']

            # Record performance
            loss_history.extend(mode_result['loss_history'])
            self.state.performance_by_mode[self.current_mode].extend(
                mode_result['loss_history']
            )
            iterations_per_mode[self.current_mode] += len(mode_result['loss_history'])

            # Check for mode switching
            if self._should_switch_mode(loss_history, iteration):
                old_mode = self.current_mode
                new_mode = self._select_new_mode(task_specification, loss_history)

                if new_mode != old_mode:
                    self.current_mode = new_mode
                    self.state.mode_history.append(new_mode)
                    self.state.switching_points.append(iteration)
                    mode_switches.append({
                        'iteration': iteration,
                        'from': old_mode,
                        'to': new_mode,
                        'loss': best_loss
                    })

                    self.logger.info(
                        f"Switching from {old_mode} to {new_mode} at iteration {iteration}"
                    )

            # Update exploration/exploitation ratio
            self._update_exploration_ratio(iteration, max_iterations)

            iteration += batch_iterations

            # Check convergence
            if len(loss_history) > 10:
                recent_improvement = abs(loss_history[-10] - loss_history[-1])
                if recent_improvement < convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break

        # Create result
        result = OptimizationResult(
            success=best_loss < task_specification.optimization_params.get('target_loss', 0.1),
            final_loss=float(best_loss),
            optimal_parameters=best_params,
            iterations=len(loss_history),
            convergence_history=np.array(loss_history),
            metadata={
                'optimizer': 'HybridOptimizer',
                'mode_switches': mode_switches,
                'iterations_per_mode': iterations_per_mode,
                'final_mode': self.current_mode.value,
                'exploration_ratio': self.state.exploration_exploitation_ratio
            }
        )

        return result

    def _run_mode_optimization(self,
                               task_spec: TaskSpecification,
                               initial_params: np.ndarray,
                               max_iterations: int,
                               convergence_threshold: float) -> Dict[str, Any]:
        """Run optimization with current mode."""
        if self.current_mode == OptimizationMode.GRADIENT:
            return self._run_gradient_optimization(
                task_spec, initial_params, max_iterations
            )
        elif self.current_mode == OptimizationMode.POPULATION:
            return self._run_population_optimization(
                task_spec, initial_params, max_iterations
            )
        elif self.current_mode == OptimizationMode.BAYESIAN:
            return self._run_bayesian_optimization(
                task_spec, initial_params, max_iterations
            )
        elif self.current_mode == OptimizationMode.ENSEMBLE:
            return self._run_ensemble_optimization(
                task_spec, initial_params, max_iterations
            )
        else:  # ADAPTIVE
            return self._run_adaptive_optimization(
                task_spec, initial_params, max_iterations, convergence_threshold
            )

    def _run_gradient_optimization(self,
                                   task_spec: TaskSpecification,
                                   initial_params: np.ndarray,
                                   max_iterations: int) -> Dict[str, Any]:
        """Run gradient-based optimization."""
        params = torch.tensor(initial_params, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.01)

        loss_history = []
        best_loss = float('inf')
        best_params = params.detach().numpy().copy()

        for _ in range(max_iterations):
            optimizer.zero_grad()

            # Compute loss
            loss = self._compute_torch_loss(params, task_spec)
            loss_history.append(loss.item())

            # Update best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().numpy().copy()

            # Backward pass
            loss.backward()
            optimizer.step()

        return {
            'best_loss': best_loss,
            'best_params': best_params,
            'loss_history': loss_history
        }

    def _run_population_optimization(self,
                                     task_spec: TaskSpecification,
                                     initial_params: np.ndarray,
                                     max_iterations: int) -> Dict[str, Any]:
        """Run population-based optimization using differential evolution."""
        dim = len(initial_params)
        bounds = [(-5, 5)] * dim  # Default bounds

        # Define objective function
        def objective(x):
            return self._compute_numpy_loss(x, task_spec)

        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations // self.population_size,
            popsize=self.population_size,
            seed=42,
            x0=initial_params
        )

        # Extract history (approximated)
        loss_history = [result.fun] * max_iterations

        return {
            'best_loss': result.fun,
            'best_params': result.x,
            'loss_history': loss_history
        }

    def _run_bayesian_optimization(self,
                                   task_spec: TaskSpecification,
                                   initial_params: np.ndarray,
                                   max_iterations: int) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        dim = len(initial_params)

        # Initialize with random samples
        n_initial = min(10, max_iterations // 2)
        X = np.random.randn(n_initial, dim)
        y = np.array([self._compute_numpy_loss(x, task_spec) for x in X])

        best_loss = np.min(y)
        best_params = X[np.argmin(y)]
        loss_history = list(y)

        # Gaussian process model
        gp = GaussianProcessRegressor(kernel=self.gp_kernel, normalize_y=True)

        for i in range(max_iterations - n_initial):
            # Fit GP
            gp.fit(X, y)

            # Find next point using acquisition function
            next_x = self._optimize_acquisition(gp, X, dim)

            # Evaluate
            next_y = self._compute_numpy_loss(next_x, task_spec)

            # Update dataset
            X = np.vstack([X, next_x])
            y = np.append(y, next_y)
            loss_history.append(next_y)

            # Update best
            if next_y < best_loss:
                best_loss = next_y
                best_params = next_x

        return {
            'best_loss': best_loss,
            'best_params': best_params,
            'loss_history': loss_history
        }

    def _run_ensemble_optimization(self,
                                   task_spec: TaskSpecification,
                                   initial_params: np.ndarray,
                                   max_iterations: int) -> Dict[str, Any]:
        """Run ensemble of multiple optimizers."""
        # Run each optimizer for a fraction of iterations
        iterations_per_optimizer = max_iterations // self.ensemble_size

        results = []
        all_losses = []

        # Run gradient optimizer
        grad_result = self._run_gradient_optimization(
            task_spec, initial_params, iterations_per_optimizer
        )
        results.append(grad_result)
        all_losses.extend(grad_result['loss_history'])

        # Run population optimizer
        pop_result = self._run_population_optimization(
            task_spec, grad_result['best_params'], iterations_per_optimizer
        )
        results.append(pop_result)
        all_losses.extend(pop_result['loss_history'])

        # Run Bayesian optimizer
        bayes_result = self._run_bayesian_optimization(
            task_spec, pop_result['best_params'], iterations_per_optimizer
        )
        results.append(bayes_result)
        all_losses.extend(bayes_result['loss_history'])

        # Select best result
        best_idx = np.argmin([r['best_loss'] for r in results])
        best_result = results[best_idx]

        return {
            'best_loss': best_result['best_loss'],
            'best_params': best_result['best_params'],
            'loss_history': all_losses[:max_iterations]
        }

    def _run_adaptive_optimization(self,
                                   task_spec: TaskSpecification,
                                   initial_params: np.ndarray,
                                   max_iterations: int,
                                   convergence_threshold: float) -> Dict[str, Any]:
        """Run adaptive optimization using the AdaptiveOptimizer."""
        # Convert numpy params to task specification format
        modified_spec = TaskSpecification(
            data=task_spec.data,
            optimization_params={
                **task_spec.optimization_params,
                'initial_params': initial_params
            },
            cognitive_constraints=task_spec.cognitive_constraints,
            metadata=task_spec.metadata
        )

        # Run adaptive optimizer
        result = self.gradient_optimizer.optimize(
            modified_spec, max_iterations, convergence_threshold
        )

        return {
            'best_loss': result.final_loss,
            'best_params': result.optimal_parameters,
            'loss_history': list(result.convergence_history)
        }

    def _should_switch_mode(self, loss_history: List[float], iteration: int) -> bool:
        """Determine if mode switching should occur."""
        if len(loss_history) < 20:
            return False

        # Check if progress has stalled
        recent_losses = loss_history[-20:]
        improvement = (recent_losses[0] - recent_losses[-1]) / (abs(recent_losses[0]) + 1e-8)

        # Switch if improvement is below threshold
        if improvement < self.switching_threshold:
            return True

        # Periodic switching for exploration
        if iteration > 0 and iteration % 200 == 0:
            return np.random.random() < self.exploration_ratio

        return False

    def _select_new_mode(self,
                         task_spec: TaskSpecification,
                         loss_history: List[float]) -> OptimizationMode:
        """Select new optimization mode based on problem characteristics."""
        if self.use_neural_selector:
            # Extract features for mode selection
            features = self._extract_problem_features(task_spec, loss_history)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Get mode probabilities
            with torch.no_grad():
                mode_probs = self.mode_selector(features_tensor).squeeze()

            # Sample mode based on probabilities
            mode_idx = torch.multinomial(mode_probs, 1).item()
            return list(OptimizationMode)[mode_idx]
        else:
            # Rule-based selection
            return self._rule_based_mode_selection(task_spec, loss_history)

    def _rule_based_mode_selection(self,
                                   task_spec: TaskSpecification,
                                   loss_history: List[float]) -> OptimizationMode:
        """Select mode using hand-crafted rules."""
        # Analyze loss trajectory
        if len(loss_history) < 50:
            # Early stage: use gradient
            return OptimizationMode.GRADIENT

        # Check for multimodality (high variance in recent losses)
        recent_losses = loss_history[-50:]
        loss_variance = np.var(recent_losses)

        if loss_variance > 0.1:
            # High variance suggests multimodal landscape
            return OptimizationMode.POPULATION

        # Check for slow progress
        improvement_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)

        if improvement_rate < 0.001:
            # Slow progress: try Bayesian optimization
            return OptimizationMode.BAYESIAN

        # Default to ensemble for robustness
        return OptimizationMode.ENSEMBLE

    def _extract_problem_features(self,
                                  task_spec: TaskSpecification,
                                  loss_history: List[float]) -> List[float]:
        """Extract features for mode selection."""
        features = []

        # Problem dimension
        dim = task_spec.optimization_params.get('dimension', 10)
        features.append(float(dim))

        # Loss statistics
        if loss_history:
            features.extend([
                float(np.mean(loss_history[-20:])),
                float(np.std(loss_history[-20:])),
                float(np.min(loss_history)),
                float(len(loss_history))
            ])
        else:
            features.extend([0.0] * 4)

        # Mode performance statistics
        for mode in OptimizationMode:
            mode_losses = self.state.performance_by_mode[mode]
            if mode_losses:
                features.append(float(np.mean(mode_losses[-10:])))
            else:
                features.append(0.0)

        return features

    def _update_exploration_ratio(self, iteration: int, max_iterations: int):
        """Update exploration/exploitation ratio over time."""
        # Decay exploration over time
        progress = iteration / max_iterations
        self.state.exploration_exploitation_ratio = self.exploration_ratio * (1 - progress)

    def _compute_torch_loss(self, params: torch.Tensor,
                            task_spec: TaskSpecification) -> torch.Tensor:
        """Compute loss as torch tensor."""
        # Extract objective function or use default
        if 'objective_function' in task_spec.optimization_params:
            obj_func = task_spec.optimization_params['objective_function']
            if callable(obj_func):
                return obj_func(params)

        # Default quadratic loss
        return torch.sum(params ** 2)

    def _compute_numpy_loss(self, params: np.ndarray,
                            task_spec: TaskSpecification) -> float:
        """Compute loss as numpy scalar."""
        params_tensor = torch.tensor(params, dtype=torch.float32)
        loss = self._compute_torch_loss(params_tensor, task_spec)
        return loss.item()

    def _upper_confidence_bound(self, gp: GaussianProcessRegressor,
                                X: np.ndarray,
                                kappa: float = 2.0) -> np.ndarray:
        """Upper confidence bound acquisition function."""
        mean, std = gp.predict(X, return_std=True)
        return -(mean - kappa * std)  # Negative for minimization

    def _optimize_acquisition(self, gp: GaussianProcessRegressor,
                              X: np.ndarray, dim: int) -> np.ndarray:
        """Optimize acquisition function to find next point."""
        # Random search for simplicity
        n_samples = 1000
        candidates = np.random.randn(n_samples, dim)

        acquisition_values = self._upper_confidence_bound(gp, candidates)
        best_idx = np.argmin(acquisition_values)

        return candidates[best_idx]

    def save_state(self, path: Path) -> None:
        """Save optimizer state to disk."""
        state_dict = {
            'current_mode': self.current_mode.value,
            'state': self.state,
            'exploration_ratio': self.exploration_ratio
        }

        if self.use_neural_selector:
            state_dict['mode_selector'] = self.mode_selector.state_dict()

        torch.save(state_dict, path)
        self.logger.info(f"Saved hybrid optimizer state to {path}")

    def load_state(self, path: Path) -> None:
        """Load optimizer state from disk."""
        state_dict = torch.load(path, map_location=self.device)

        self.current_mode = OptimizationMode(state_dict['current_mode'])
        self.state = state_dict['state']
        self.exploration_ratio = state_dict['exploration_ratio']

        if self.use_neural_selector and 'mode_selector' in state_dict:
            self.mode_selector.load_state_dict(state_dict['mode_selector'])

        self.logger.info(f"Loaded hybrid optimizer state from {path}")
