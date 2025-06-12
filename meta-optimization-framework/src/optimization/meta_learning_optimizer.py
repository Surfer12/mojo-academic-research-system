"""
Meta-Learning Optimizer - Learning to Optimize
==============================================

Implements meta-learning algorithms that learn to optimize by leveraging
experience from previous optimization tasks. Uses neural networks and
gradient-based meta-learning to adapt quickly to new optimization problems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from src.core.meta_optimization import TaskSpecification, OptimizationResult

from ..utils.statistical_analysis import StatisticalAnalyzer


@dataclass
class MetaLearningState:
    """State of the meta-learning optimization process."""
    task_embeddings: torch.Tensor
    optimizer_state: Dict[str, torch.Tensor]
    adaptation_steps: int
    meta_gradients: List[torch.Tensor]
    performance_trajectory: List[float]


class OptimizationLSTM(nn.Module):
    """LSTM-based optimizer that learns to generate parameter updates."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 20, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Generate parameter updates based on optimization trajectory."""
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        lstm_out, hidden = self.lstm(x, hidden)
        updates = self.output_layer(lstm_out)
        return updates.squeeze(-1), hidden

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class MetaLearningOptimizer:
    """
    Meta-learning optimizer that learns to optimize from experience.

    Uses techniques like MAML (Model-Agnostic Meta-Learning) and
    learned optimization to quickly adapt to new optimization tasks.
    """

    def __init__(self,
                 meta_learning_rate: float = 0.001,
                 inner_learning_rate: float = 0.01,
                 adaptation_steps: int = 5,
                 meta_batch_size: int = 10,
                 use_lstm_optimizer: bool = True,
                 device: str = "cpu"):
        """
        Initialize meta-learning optimizer.

        Args:
            meta_learning_rate: Learning rate for meta-parameters
            inner_learning_rate: Learning rate for task-specific adaptation
            adaptation_steps: Number of gradient steps for adaptation
            meta_batch_size: Batch size for meta-training
            use_lstm_optimizer: Whether to use learned LSTM optimizer
            device: Device for computation (cpu/cuda)
        """
        self.meta_lr = meta_learning_rate
        self.inner_lr = inner_learning_rate
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size
        self.device = torch.device(device)

        # Initialize learned optimizer if requested
        self.use_lstm_optimizer = use_lstm_optimizer
        if use_lstm_optimizer:
            self.lstm_optimizer = OptimizationLSTM().to(self.device)
            self.lstm_meta_optimizer = optim.Adam(
                self.lstm_optimizer.parameters(),
                lr=meta_learning_rate
            )

        # Task memory for meta-learning
        self.task_memory = []
        self.max_memory_size = 1000

        # Statistical analyzer
        self.analyzer = StatisticalAnalyzer()

        # Logging
        self.logger = logging.getLogger(__name__)

    def optimize(self,
                 task_specification: TaskSpecification,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Optimize using meta-learned strategies.

        Args:
            task_specification: The optimization task
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold

        Returns:
            OptimizationResult containing the optimization outcome
        """
        self.logger.info("Starting meta-learning optimization")

        # Extract task characteristics
        task_features = self._extract_task_features(task_specification)

        # Initialize parameters
        params = self._initialize_parameters(task_specification)

        # Adapt to specific task using meta-learned knowledge
        adapted_params = self._adapt_to_task(
            params, task_specification, task_features
        )

        # Run optimization with adapted parameters
        result = self._run_optimization(
            adapted_params, task_specification,
            max_iterations, convergence_threshold
        )

        # Store experience for future meta-learning
        self._update_task_memory(task_specification, result)

        return result

    def _extract_task_features(self, task_spec: TaskSpecification) -> torch.Tensor:
        """Extract features that characterize the optimization task."""
        features = []

        # Basic data statistics
        data = task_spec.data
        if isinstance(data, np.ndarray):
            features.extend([
                float(data.shape[0]),
                float(np.mean(data)),
                float(np.std(data)),
                float(np.min(data)),
                float(np.max(data))
            ])
        else:
            features.extend([0.0] * 5)

        # Task parameters
        for key in ['dimension', 'constraints', 'regularization']:
            if key in task_spec.optimization_params:
                val = task_spec.optimization_params[key]
                features.append(float(val) if isinstance(val, (int, float)) else 0.0)
            else:
                features.append(0.0)

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _initialize_parameters(self, task_spec: TaskSpecification) -> torch.Tensor:
        """Initialize optimization parameters."""
        dim = task_spec.optimization_params.get('dimension', 10)

        # Use meta-learned initialization if available
        if self.task_memory:
            # Find similar tasks and use their solutions as initialization
            similar_task = self._find_similar_task(task_spec)
            if similar_task:
                return similar_task['final_params'].clone()

        # Default initialization
        return torch.randn(dim, device=self.device) * 0.1

    def _adapt_to_task(self, params: torch.Tensor,
                       task_spec: TaskSpecification,
                       task_features: torch.Tensor) -> torch.Tensor:
        """Adapt parameters to specific task using meta-learning."""
        adapted_params = params.clone().requires_grad_(True)

        # Perform few-shot adaptation
        for step in range(self.adaptation_steps):
            # Compute loss on current task
            loss = self._compute_task_loss(adapted_params, task_spec)

            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)[0]

            # Update using learned optimizer or gradient descent
            if self.use_lstm_optimizer:
                # Prepare input for LSTM optimizer
                optimizer_input = torch.stack([
                    grads,
                    adapted_params,
                    torch.full_like(grads, loss.item()),
                    torch.full_like(grads, step / self.adaptation_steps)
                ], dim=-1).unsqueeze(0)

                # Generate update using LSTM
                update, _ = self.lstm_optimizer(optimizer_input)
                adapted_params = adapted_params - update.squeeze(0)
            else:
                # Standard gradient descent
                adapted_params = adapted_params - self.inner_lr * grads

            adapted_params = adapted_params.detach().requires_grad_(True)

        return adapted_params

    def _run_optimization(self, initial_params: torch.Tensor,
                          task_spec: TaskSpecification,
                          max_iterations: int,
                          convergence_threshold: float) -> OptimizationResult:
        """Run the main optimization loop."""
        params = initial_params.clone().requires_grad_(True)

        loss_history = []
        param_history = []
        best_loss = float('inf')
        best_params = params.clone()

        # Initialize LSTM hidden state if using learned optimizer
        lstm_hidden = None
        if self.use_lstm_optimizer:
            lstm_hidden = self.lstm_optimizer.init_hidden(1)

        for iteration in range(max_iterations):
            # Compute loss
            loss = self._compute_task_loss(params, task_spec)
            loss_history.append(loss.item())
            param_history.append(params.clone().detach())

            # Track best solution
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.clone()

            # Check convergence
            if len(loss_history) > 10:
                recent_improvement = abs(loss_history[-10] - loss_history[-1])
                if recent_improvement < convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break

            # Compute gradients
            grads = torch.autograd.grad(loss, params)[0]

            # Update parameters
            if self.use_lstm_optimizer:
                # Prepare input for LSTM
                optimizer_input = torch.stack([
                    grads,
                    params,
                    torch.full_like(grads, loss.item()),
                    torch.full_like(grads, iteration / max_iterations)
                ], dim=-1).unsqueeze(0)

                # Generate update
                update, lstm_hidden = self.lstm_optimizer(optimizer_input, lstm_hidden)
                params = params - update.squeeze(0)
            else:
                # Standard gradient descent with momentum
                if not hasattr(self, '_momentum'):
                    self._momentum = torch.zeros_like(params)

                self._momentum = 0.9 * self._momentum + grads
                params = params - self.inner_lr * self._momentum

            params = params.detach().requires_grad_(True)

        # Create optimization result
        result = OptimizationResult(
            success=best_loss < task_spec.optimization_params.get('target_loss', 0.1),
            final_loss=best_loss,
            optimal_parameters=best_params.detach().cpu().numpy(),
            iterations=len(loss_history),
            convergence_history=np.array(loss_history),
            metadata={
                'optimizer': 'MetaLearningOptimizer',
                'adaptation_steps': self.adaptation_steps,
                'used_lstm': self.use_lstm_optimizer,
                'final_gradient_norm': float(torch.norm(grads))
            }
        )

        return result

    def _compute_task_loss(self, params: torch.Tensor,
                           task_spec: TaskSpecification) -> torch.Tensor:
        """Compute loss for the given task and parameters."""
        # Extract objective function or use default quadratic
        if 'objective_function' in task_spec.optimization_params:
            obj_func = task_spec.optimization_params['objective_function']
            if callable(obj_func):
                return obj_func(params)

        # Default: quadratic loss with optional constraints
        loss = torch.sum(params ** 2)

        # Add regularization if specified
        if 'regularization' in task_spec.optimization_params:
            reg_weight = task_spec.optimization_params['regularization']
            loss += reg_weight * torch.sum(torch.abs(params))

        return loss

    def _find_similar_task(self, task_spec: TaskSpecification) -> Optional[Dict]:
        """Find similar task from memory using task features."""
        if not self.task_memory:
            return None

        task_features = self._extract_task_features(task_spec)

        # Compute distances to all stored tasks
        min_distance = float('inf')
        most_similar = None

        for memory_item in self.task_memory:
            stored_features = memory_item['features']
            distance = torch.norm(task_features - stored_features).item()

            if distance < min_distance:
                min_distance = distance
                most_similar = memory_item

        # Return if sufficiently similar
        if min_distance < 1.0:  # Threshold for similarity
            return most_similar

        return None

    def _update_task_memory(self, task_spec: TaskSpecification,
                            result: OptimizationResult) -> None:
        """Update task memory with new experience."""
        memory_item = {
            'features': self._extract_task_features(task_spec),
            'final_params': torch.tensor(
                result.optimal_parameters,
                dtype=torch.float32,
                device=self.device
            ),
            'final_loss': result.final_loss,
            'success': result.success
        }

        self.task_memory.append(memory_item)

        # Limit memory size
        if len(self.task_memory) > self.max_memory_size:
            self.task_memory.pop(0)

    def meta_train(self, training_tasks: List[TaskSpecification]) -> Dict[str, Any]:
        """
        Meta-train the optimizer on a set of tasks.

        Args:
            training_tasks: List of tasks for meta-training

        Returns:
            Meta-training statistics
        """
        self.logger.info(f"Meta-training on {len(training_tasks)} tasks")

        meta_losses = []

        for epoch in range(10):  # Meta-training epochs
            epoch_loss = 0.0

            # Sample batch of tasks
            task_batch = np.random.choice(
                training_tasks,
                size=min(self.meta_batch_size, len(training_tasks)),
                replace=False
            )

            # Accumulate meta-gradients
            if self.use_lstm_optimizer:
                self.lstm_meta_optimizer.zero_grad()

            for task in task_batch:
                # Perform adaptation and optimization
                result = self.optimize(task, max_iterations=100)

                # Meta-loss is the final task loss after adaptation
                meta_loss = result.final_loss
                epoch_loss += meta_loss

                if self.use_lstm_optimizer:
                    # Backpropagate through the entire optimization trajectory
                    torch.tensor(meta_loss).backward()

            # Update meta-parameters
            if self.use_lstm_optimizer:
                self.lstm_meta_optimizer.step()

            avg_epoch_loss = epoch_loss / len(task_batch)
            meta_losses.append(avg_epoch_loss)

            self.logger.info(f"Meta-epoch {epoch}: avg loss = {avg_epoch_loss:.4f}")

        return {
            'meta_losses': meta_losses,
            'final_meta_loss': meta_losses[-1],
            'tasks_trained': len(training_tasks)
        }

    def save_state(self, path: Path) -> None:
        """Save optimizer state to disk."""
        state = {
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'adaptation_steps': self.adaptation_steps,
            'task_memory': self.task_memory
        }

        if self.use_lstm_optimizer:
            state['lstm_state'] = self.lstm_optimizer.state_dict()

        torch.save(state, path)
        self.logger.info(f"Saved meta-learning optimizer state to {path}")

    def load_state(self, path: Path) -> None:
        """Load optimizer state from disk."""
        state = torch.load(path, map_location=self.device)

        self.meta_lr = state['meta_lr']
        self.inner_lr = state['inner_lr']
        self.adaptation_steps = state['adaptation_steps']
        self.task_memory = state['task_memory']

        if self.use_lstm_optimizer and 'lstm_state' in state:
            self.lstm_optimizer.load_state_dict(state['lstm_state'])

        self.logger.info(f"Loaded meta-learning optimizer state from {path}")
