"""
Cognitive Optimizer - Cognitive Science-Inspired Optimization
============================================================

Implements optimization algorithms inspired by cognitive science principles,
including attention mechanisms, working memory constraints, schema-based
processing, and meta-awareness. Based on attention-recognition decoupling
and mind-wandering research.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from collections import deque
import networkx as nx

from src.core.meta_optimization import TaskSpecification, OptimizationResult
from ..utils.statistical_analysis import StatisticalAnalyzer


class CognitiveState(Enum):
    """Cognitive states during optimization."""
    FOCUSED = "focused"  # On-task, exploiting current solution
    OFF_FOCUS = "off_focus"  # Transitional, beginning to explore
    WANDERING = "wandering"  # Fully exploring, decoupled from task
    META_AWARE = "meta_aware"  # Aware of current cognitive state


@dataclass
class WorkingMemory:
    """Working memory representation with limited capacity."""
    capacity: int = 7  # Miller's magic number
    items: deque = field(default_factory=lambda: deque(maxlen=7))
    activation_levels: Dict[Any, float] = field(default_factory=dict)

    def add(self, item: Any, activation: float = 1.0):
        """Add item to working memory with activation level."""
        self.items.append(item)
        self.activation_levels[str(item)] = activation

    def decay(self, rate: float = 0.1):
        """Decay activation levels over time."""
        for key in self.activation_levels:
            self.activation_levels[key] *= (1 - rate)

    def retrieve(self, cue: Any = None) -> Optional[Any]:
        """Retrieve item from memory, optionally using a cue."""
        if not self.items:
            return None

        if cue is None:
            # Return most activated item
            max_key = max(self.activation_levels, key=self.activation_levels.get)
            for item in self.items:
                if str(item) == max_key:
                    return item
        return self.items[-1]  # Most recent


@dataclass
class Schema:
    """Cognitive schema for organizing optimization strategies."""
    name: str
    pattern: Dict[str, Any]
    success_rate: float = 0.5
    applications: int = 0

    def update_success(self, success: bool):
        """Update schema success rate."""
        self.applications += 1
        alpha = 1.0 / self.applications
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)


class AttentionMechanism(nn.Module):
    """Neural attention mechanism for focusing on relevant solution components."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention."""
        batch_size = x.size(0)

        # Linear projections
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        return self.output(attended)


class CognitiveOptimizer:
    """
    Cognitive science-inspired optimizer.

    Implements optimization strategies based on human cognitive processes:
    - Attention focusing and wandering
    - Working memory constraints
    - Schema-based problem solving
    - Meta-cognitive monitoring
    - Exploration-exploitation as cognitive states
    """

    def __init__(self,
                 working_memory_capacity: int = 7,
                 attention_span: int = 50,
                 mind_wandering_rate: float = 0.2,
                 meta_awareness_frequency: float = 0.1,
                 schema_learning_rate: float = 0.01,
                 device: str = "cpu"):
        """
        Initialize cognitive optimizer.

        Args:
            working_memory_capacity: Size of working memory
            attention_span: Iterations before attention may wander
            mind_wandering_rate: Probability of mind wandering
            meta_awareness_frequency: How often meta-awareness checks occur
            schema_learning_rate: Rate of schema formation
            device: Computation device
        """
        self.wm_capacity = working_memory_capacity
        self.attention_span = attention_span
        self.mind_wandering_rate = mind_wandering_rate
        self.meta_awareness_frequency = meta_awareness_frequency
        self.schema_learning_rate = schema_learning_rate
        self.device = torch.device(device)

        # Cognitive components
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.cognitive_state = CognitiveState.FOCUSED
        self.attention_mechanism = AttentionMechanism(dim=64).to(self.device)

        # Schema library
        self.schemas: Dict[str, Schema] = {
            'gradient_descent': Schema(
                'gradient_descent',
                {'type': 'local', 'step_size': 'adaptive'}
            ),
            'random_search': Schema(
                'random_search',
                {'type': 'global', 'step_size': 'large'}
            ),
            'pattern_search': Schema(
                'pattern_search',
                {'type': 'structured', 'step_size': 'fixed'}
            )
        }

        # Cognitive graph for associative thinking
        self.cognitive_graph = nx.Graph()

        # Performance tracking
        self.attention_lapses = 0
        self.successful_wanderings = 0

        # Statistical analyzer
        self.analyzer = StatisticalAnalyzer()

        # Logging
        self.logger = logging.getLogger(__name__)

    def optimize(self,
                 task_specification: TaskSpecification,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Optimize using cognitive principles.

        Args:
            task_specification: The optimization task
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold

        Returns:
            OptimizationResult with cognitive metadata
        """
        self.logger.info("Starting cognitive optimization")

        # Initialize cognitive state for new task
        self._initialize_cognitive_state(task_specification)

        # Initialize solution
        dim = task_specification.optimization_params.get('dimension', 10)
        current_solution = np.random.randn(dim) * 0.1
        best_solution = current_solution.copy()
        best_loss = float('inf')

        loss_history = []
        cognitive_state_history = []
        schema_applications = []

        attention_timer = 0

        for iteration in range(max_iterations):
            # Meta-awareness check
            if np.random.random() < self.meta_awareness_frequency:
                self._meta_awareness_check(loss_history)

            # Update cognitive state based on attention
            attention_timer += 1
            if attention_timer > self.attention_span:
                self._update_cognitive_state(loss_history)
                attention_timer = 0

            cognitive_state_history.append(self.cognitive_state.value)

            # Select and apply cognitive strategy
            if self.cognitive_state == CognitiveState.FOCUSED:
                # Exploit current solution path
                update, schema_used = self._focused_optimization(
                    current_solution, task_specification
                )
            elif self.cognitive_state == CognitiveState.OFF_FOCUS:
                # Begin exploring alternatives
                update, schema_used = self._off_focus_exploration(
                    current_solution, task_specification
                )
            elif self.cognitive_state == CognitiveState.WANDERING:
                # Free exploration
                update, schema_used = self._mind_wandering(
                    current_solution, task_specification
                )
            else:  # META_AWARE
                # Conscious strategy selection
                update, schema_used = self._meta_aware_optimization(
                    current_solution, task_specification, loss_history
                )

            schema_applications.append(schema_used)

            # Apply update with working memory constraints
            constrained_update = self._apply_working_memory_constraints(update)
            current_solution = current_solution + constrained_update

            # Evaluate solution
            current_loss = self._evaluate_solution(current_solution, task_specification)
            loss_history.append(current_loss)

            # Update best solution
            if current_loss < best_loss:
                best_loss = current_loss
                best_solution = current_solution.copy()

                # Reinforce successful schema
                if schema_used:
                    self.schemas[schema_used].update_success(True)

                # Record if wandering led to improvement
                if self.cognitive_state == CognitiveState.WANDERING:
                    self.successful_wanderings += 1

            # Update working memory
            self.working_memory.add(
                {'iteration': iteration, 'loss': current_loss, 'schema': schema_used},
                activation=1.0 / (current_loss + 1e-8)
            )
            self.working_memory.decay()

            # Check convergence
            if len(loss_history) > 10:
                recent_improvement = abs(loss_history[-10] - loss_history[-1])
                if recent_improvement < convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break

        # Create result with cognitive metadata
        result = OptimizationResult(
            success=best_loss < task_specification.optimization_params.get('target_loss', 0.1),
            final_loss=float(best_loss),
            optimal_parameters=best_solution,
            iterations=len(loss_history),
            convergence_history=np.array(loss_history),
            metadata={
                'optimizer': 'CognitiveOptimizer',
                'cognitive_states': cognitive_state_history,
                'attention_lapses': self.attention_lapses,
                'successful_wanderings': self.successful_wanderings,
                'schema_applications': schema_applications,
                'working_memory_utilization': len(self.working_memory.items) / self.wm_capacity,
                'most_successful_schema': max(
                    self.schemas.values(),
                    key=lambda s: s.success_rate
                ).name
            }
        )

        return result

    def _initialize_cognitive_state(self, task_spec: TaskSpecification):
        """Initialize cognitive state for new task."""
        self.cognitive_state = CognitiveState.FOCUSED
        self.working_memory = WorkingMemory(capacity=self.wm_capacity)
        self.attention_lapses = 0
        self.successful_wanderings = 0

        # Build cognitive graph from task structure
        self._build_cognitive_graph(task_spec)

    def _build_cognitive_graph(self, task_spec: TaskSpecification):
        """Build associative graph from task information."""
        self.cognitive_graph.clear()

        # Add nodes for different solution components
        dim = task_spec.optimization_params.get('dimension', 10)
        for i in range(dim):
            self.cognitive_graph.add_node(f'param_{i}')

        # Add schema nodes
        for schema_name in self.schemas:
            self.cognitive_graph.add_node(f'schema_{schema_name}')

        # Add edges based on correlations or domain knowledge
        # This is simplified - in practice would use actual correlations
        for i in range(dim - 1):
            self.cognitive_graph.add_edge(f'param_{i}', f'param_{i+1}', weight=0.5)

    def _update_cognitive_state(self, loss_history: List[float]):
        """Update cognitive state based on performance."""
        if len(loss_history) < 10:
            return

        # Check recent progress
        recent_improvement = (loss_history[-10] - loss_history[-1]) / (abs(loss_history[-10]) + 1e-8)

        if self.cognitive_state == CognitiveState.FOCUSED:
            # Poor progress leads to off-focus
            if recent_improvement < 0.01 or np.random.random() < self.mind_wandering_rate:
                self.cognitive_state = CognitiveState.OFF_FOCUS
                self.attention_lapses += 1
                self.logger.debug("Transitioning to OFF_FOCUS state")

        elif self.cognitive_state == CognitiveState.OFF_FOCUS:
            # Can progress to wandering or return to focus
            if np.random.random() < 0.5:
                self.cognitive_state = CognitiveState.WANDERING
                self.logger.debug("Mind wandering initiated")
            else:
                self.cognitive_state = CognitiveState.FOCUSED
                self.logger.debug("Refocused on task")

        elif self.cognitive_state == CognitiveState.WANDERING:
            # Eventually return to focus
            if np.random.random() < 0.3:
                self.cognitive_state = CognitiveState.META_AWARE
                self.logger.debug("Meta-awareness activated")

        else:  # META_AWARE
            # Always return to focused after meta-awareness
            self.cognitive_state = CognitiveState.FOCUSED

    def _meta_awareness_check(self, loss_history: List[float]):
        """Perform meta-cognitive check on optimization progress."""
        if len(loss_history) < 20:
            return

        # Analyze overall trajectory
        trajectory_slope = np.polyfit(range(20), loss_history[-20:], 1)[0]

        if trajectory_slope > -0.001:  # Not improving
            self.cognitive_state = CognitiveState.META_AWARE
            self.logger.debug("Meta-awareness triggered by lack of progress")

    def _focused_optimization(self,
                              current_solution: np.ndarray,
                              task_spec: TaskSpecification) -> Tuple[np.ndarray, str]:
        """Focused, exploitative optimization."""
        # Use gradient descent schema
        schema = self.schemas['gradient_descent']

        # Compute gradient approximation
        dim = len(current_solution)
        gradient = np.zeros(dim)
        epsilon = 0.001

        current_loss = self._evaluate_solution(current_solution, task_spec)

        for i in range(min(dim, self.wm_capacity)):  # Limited by working memory
            perturbed = current_solution.copy()
            perturbed[i] += epsilon
            gradient[i] = (self._evaluate_solution(perturbed, task_spec) - current_loss) / epsilon

        # Adaptive step size
        step_size = 0.01 / (np.linalg.norm(gradient) + 1e-8)
        update = -gradient * step_size

        return update, schema.name

    def _off_focus_exploration(self,
                               current_solution: np.ndarray,
                               task_spec: TaskSpecification) -> Tuple[np.ndarray, str]:
        """Transitional exploration state."""
        # Mix focused and random strategies
        if np.random.random() < 0.5:
            return self._focused_optimization(current_solution, task_spec)
        else:
            schema = self.schemas['pattern_search']

            # Structured exploration
            dim = len(current_solution)
            direction = np.zeros(dim)

            # Explore along coordinate axes
            axis = np.random.randint(0, dim)
            direction[axis] = np.random.choice([-1, 1]) * 0.1

            return direction, schema.name

    def _mind_wandering(self,
                        current_solution: np.ndarray,
                        task_spec: TaskSpecification) -> Tuple[np.ndarray, str]:
        """Free exploration during mind wandering."""
        schema = self.schemas['random_search']

        # Generate random associations using cognitive graph
        if self.cognitive_graph.number_of_nodes() > 0:
            # Random walk on cognitive graph
            start_node = f'param_{np.random.randint(0, len(current_solution))}'
            if start_node in self.cognitive_graph:
                neighbors = list(self.cognitive_graph.neighbors(start_node))
                if neighbors:
                    # Explore related parameters
                    related_params = [
                        int(n.split('_')[1])
                        for n in neighbors
                        if n.startswith('param_')
                    ]

                    update = np.zeros_like(current_solution)
                    for idx in related_params[:self.wm_capacity]:
                        update[idx] = np.random.randn() * 0.1

                    return update, schema.name

        # Fallback to pure random exploration
        return np.random.randn(len(current_solution)) * 0.1, schema.name

    def _meta_aware_optimization(self,
                                 current_solution: np.ndarray,
                                 task_spec: TaskSpecification,
                                 loss_history: List[float]) -> Tuple[np.ndarray, str]:
        """Conscious, deliberate strategy selection."""
        # Analyze which schemas have been most successful
        best_schema = max(self.schemas.values(), key=lambda s: s.success_rate)

        # Apply best schema with conscious effort
        if best_schema.name == 'gradient_descent':
            return self._focused_optimization(current_solution, task_spec)
        elif best_schema.name == 'pattern_search':
            return self._off_focus_exploration(current_solution, task_spec)
        else:
            return self._mind_wandering(current_solution, task_spec)

    def _apply_working_memory_constraints(self, update: np.ndarray) -> np.ndarray:
        """Apply working memory capacity constraints to update."""
        # Can only modify limited number of parameters at once
        if len(update) <= self.wm_capacity:
            return update

        # Select most important updates
        abs_update = np.abs(update)
        top_indices = np.argpartition(abs_update, -self.wm_capacity)[-self.wm_capacity:]

        constrained_update = np.zeros_like(update)
        constrained_update[top_indices] = update[top_indices]

        return constrained_update

    def _evaluate_solution(self, solution: np.ndarray,
                           task_spec: TaskSpecification) -> float:
        """Evaluate solution quality."""
        # Use provided objective function if available
        if 'objective_function' in task_spec.optimization_params:
            obj_func = task_spec.optimization_params['objective_function']
            if callable(obj_func):
                return float(obj_func(solution))

        # Default quadratic objective
        return float(np.sum(solution ** 2))

    def save_state(self, path: Path) -> None:
        """Save cognitive state to disk."""
        state = {
            'cognitive_state': self.cognitive_state.value,
            'schemas': {name: vars(schema) for name, schema in self.schemas.items()},
            'attention_lapses': self.attention_lapses,
            'successful_wanderings': self.successful_wanderings,
            'working_memory_items': list(self.working_memory.items),
            'working_memory_activations': self.working_memory.activation_levels
        }

        torch.save(state, path)
        self.logger.info(f"Saved cognitive optimizer state to {path}")

    def load_state(self, path: Path) -> None:
        """Load cognitive state from disk."""
        state = torch.load(path, map_location=self.device)

        self.cognitive_state = CognitiveState(state['cognitive_state'])

        # Restore schemas
        for name, schema_dict in state['schemas'].items():
            self.schemas[name] = Schema(**schema_dict)

        self.attention_lapses = state['attention_lapses']
        self.successful_wanderings = state['successful_wanderings']

        # Restore working memory
        self.working_memory = WorkingMemory(capacity=self.wm_capacity)
        for item in state['working_memory_items']:
            self.working_memory.add(item)
        self.working_memory.activation_levels = state['working_memory_activations']

        self.logger.info(f"Loaded cognitive optimizer state from {path}")
