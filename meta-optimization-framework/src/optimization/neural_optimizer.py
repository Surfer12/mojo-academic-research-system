"""
Neural Optimizer - Neural Network-Based Optimization
===================================================

Implements optimization algorithms that leverage neural networks for
function approximation, gradient estimation, and parameter updates.
Uses deep learning techniques for black-box optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from collections import deque

from ..core.meta_optimization import TaskSpecification, OptimizationResult
from ..utils.statistical_analysis import StatisticalAnalyzer


class NeuralArchitecture(Enum):
    """Types of neural architectures for optimization."""
    MLP = "mlp"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    VAE = "vae"
    GAN = "gan"


@dataclass
class NeuralOptimizerState:
    """State of the neural optimization process."""
    surrogate_loss: float
    true_loss: float
    gradient_approximation_error: float
    network_weights_norm: float
    exploration_noise: float
    update_magnitude: float


class SurrogateNetwork(nn.Module):
    """Neural network for approximating the objective function."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict objective value."""
        return self.network(x).squeeze(-1)


class UpdateNetwork(nn.Module):
    """Neural network that learns to generate parameter updates."""

    def __init__(self, input_dim: int, context_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim

        # Context encoder
        self.context_encoder = nn.LSTM(
            input_dim + 1,  # parameters + loss
            context_dim,
            num_layers=2,
            batch_first=True
        )

        # Update generator
        self.update_generator = nn.Sequential(
            nn.Linear(input_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()  # Bounded updates
        )

    def forward(self, params: torch.Tensor, history: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Generate parameter update based on current state and history."""
        # Encode history
        _, (h_n, c_n) = self.context_encoder(history, hidden)
        context = h_n[-1]  # Use last hidden state

        # Generate update
        combined = torch.cat([params, context], dim=-1)
        update = self.update_generator(combined)

        return update * 0.1, (h_n, c_n)  # Scale updates


class VariationalAutoencoder(nn.Module):
    """VAE for learning latent optimization space."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space."""
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class NeuralOptimizer:
    """
    Neural network-based optimizer.

    Uses neural networks for:
    - Function approximation (surrogate modeling)
    - Gradient estimation
    - Learning update rules
    - Latent space optimization
    """

    def __init__(self,
                 architecture: NeuralArchitecture = NeuralArchitecture.MLP,
                 learning_rate: float = 0.001,
                 surrogate_update_frequency: int = 10,
                 exploration_noise: float = 0.1,
                 use_experience_replay: bool = True,
                 replay_buffer_size: int = 1000,
                 device: str = "cpu"):
        """
        Initialize neural optimizer.

        Args:
            architecture: Type of neural architecture to use
            learning_rate: Learning rate for neural networks
            surrogate_update_frequency: How often to update surrogate model
            exploration_noise: Noise level for exploration
            use_experience_replay: Whether to use experience replay
            replay_buffer_size: Size of replay buffer
            device: Computation device
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.surrogate_update_freq = surrogate_update_frequency
        self.exploration_noise = exploration_noise
        self.device = torch.device(device)

        # Experience replay
        self.use_replay = use_experience_replay
        if use_experience_replay:
            self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Networks will be initialized per task
        self.surrogate_network = None
        self.update_network = None
        self.vae = None

        # Optimizers
        self.surrogate_optimizer = None
        self.update_optimizer = None
        self.vae_optimizer = None

        # Statistical analyzer
        self.analyzer = StatisticalAnalyzer()

        # Logging
        self.logger = logging.getLogger(__name__)

    def optimize(self,
                 task_specification: TaskSpecification,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Optimize using neural network-based methods.

        Args:
            task_specification: The optimization task
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold

        Returns:
            OptimizationResult with neural optimization metadata
        """
        self.logger.info(f"Starting neural optimization with {self.architecture.value} architecture")

        # Initialize networks for task
        dim = task_specification.optimization_params.get('dimension', 10)
        self._initialize_networks(dim)

        # Initialize parameters
        if self.architecture == NeuralArchitecture.VAE:
            # Start in latent space
            latent_params = torch.randn(1, 16, device=self.device)
            current_params = self.vae.decode(latent_params).squeeze(0)
        else:
            current_params = torch.randn(dim, device=self.device) * 0.1

        best_params = current_params.clone()
        best_loss = float('inf')

        loss_history = []
        neural_state_history = []

        # History for LSTM-based update network
        if self.architecture == NeuralArchitecture.LSTM:
            history_window = 10
            param_history = torch.zeros(1, history_window, dim + 1, device=self.device)
            hidden_state = None

        for iteration in range(max_iterations):
            # Evaluate true objective
            true_loss = self._evaluate_objective(current_params, task_specification)
            loss_history.append(true_loss)

            # Update best solution
            if true_loss < best_loss:
                best_loss = true_loss
                best_params = current_params.clone()

            # Store experience
            if self.use_replay:
                self.replay_buffer.append((current_params.clone(), true_loss))

            # Compute update based on architecture
            if self.architecture == NeuralArchitecture.MLP:
                update = self._mlp_update(current_params, true_loss)
            elif self.architecture == NeuralArchitecture.LSTM:
                # Update history
                history_entry = torch.cat([current_params, torch.tensor([true_loss], device=self.device)])
                param_history = torch.roll(param_history, -1, dims=1)
                param_history[0, -1] = history_entry

                update, hidden_state = self._lstm_update(current_params, param_history, hidden_state)
            elif self.architecture == NeuralArchitecture.TRANSFORMER:
                update = self._transformer_update(current_params, loss_history)
            elif self.architecture == NeuralArchitecture.VAE:
                update, latent_params = self._vae_update(current_params, latent_params, true_loss)
            else:  # GAN
                update = self._gan_update(current_params, true_loss)

            # Add exploration noise
            if self.exploration_noise > 0:
                noise = torch.randn_like(update) * self.exploration_noise
                noise *= (1 - iteration / max_iterations)  # Decay noise
                update = update + noise

            # Apply update
            current_params = current_params + update

            # Update surrogate model periodically
            if iteration % self.surrogate_update_freq == 0 and self.use_replay and len(self.replay_buffer) > 10:
                self._update_surrogate_model()

            # Record neural state
            surrogate_loss = float('inf')
            if self.surrogate_network is not None:
                with torch.no_grad():
                    surrogate_loss = self.surrogate_network(current_params.unsqueeze(0)).item()

            neural_state = NeuralOptimizerState(
                surrogate_loss=surrogate_loss,
                true_loss=true_loss,
                gradient_approximation_error=abs(surrogate_loss - true_loss),
                network_weights_norm=self._get_network_weights_norm(),
                exploration_noise=self.exploration_noise * (1 - iteration / max_iterations),
                update_magnitude=torch.norm(update).item()
            )
            neural_state_history.append(neural_state)

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
            optimal_parameters=best_params.detach().cpu().numpy(),
            iterations=len(loss_history),
            convergence_history=np.array(loss_history),
            metadata={
                'optimizer': 'NeuralOptimizer',
                'architecture': self.architecture.value,
                'final_surrogate_error': neural_state_history[-1].gradient_approximation_error if neural_state_history else 0,
                'average_update_magnitude': np.mean([s.update_magnitude for s in neural_state_history]),
                'replay_buffer_size': len(self.replay_buffer) if self.use_replay else 0
            }
        )

        return result

    def _initialize_networks(self, dim: int):
        """Initialize neural networks for the given problem dimension."""
        # Surrogate network
        self.surrogate_network = SurrogateNetwork(dim).to(self.device)
        self.surrogate_optimizer = optim.Adam(
            self.surrogate_network.parameters(),
            lr=self.learning_rate
        )

        # Architecture-specific networks
        if self.architecture == NeuralArchitecture.LSTM:
            self.update_network = UpdateNetwork(dim).to(self.device)
            self.update_optimizer = optim.Adam(
                self.update_network.parameters(),
                lr=self.learning_rate
            )
        elif self.architecture == NeuralArchitecture.VAE:
            self.vae = VariationalAutoencoder(dim).to(self.device)
            self.vae_optimizer = optim.Adam(
                self.vae.parameters(),
                lr=self.learning_rate
            )

    def _mlp_update(self, params: torch.Tensor, loss: float) -> torch.Tensor:
        """Generate update using MLP-based gradient approximation."""
        params_input = params.unsqueeze(0).requires_grad_(True)

        # Use surrogate network for gradient estimation
        surrogate_pred = self.surrogate_network(params_input)
        surrogate_pred.backward()

        # Use surrogate gradient as update direction
        grad = params_input.grad.squeeze(0)

        # Adaptive step size based on gradient magnitude
        step_size = 0.01 / (torch.norm(grad) + 1e-8)

        return -grad * step_size

    def _lstm_update(self, params: torch.Tensor, history: torch.Tensor,
                     hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate update using LSTM network."""
        update, new_hidden = self.update_network(params.unsqueeze(0), history, hidden)
        return update.squeeze(0), new_hidden

    def _transformer_update(self, params: torch.Tensor, loss_history: List[float]) -> torch.Tensor:
        """Generate update using transformer architecture (simplified)."""
        # For simplicity, use gradient approximation with attention-weighted history
        recent_losses = torch.tensor(loss_history[-10:], device=self.device)

        # Simple attention mechanism
        attention_weights = F.softmax(-recent_losses, dim=0)

        # Weighted gradient approximation
        grad = torch.zeros_like(params)
        epsilon = 0.001

        for i in range(len(params)):
            perturbed = params.clone()
            perturbed[i] += epsilon
            perturbed_loss = self._evaluate_objective(perturbed, None)
            grad[i] = (perturbed_loss - loss_history[-1]) / epsilon

        # Weight by attention
        weighted_step = torch.sum(attention_weights) * 0.01

        return -grad * weighted_step

    def _vae_update(self, params: torch.Tensor, latent: torch.Tensor,
                    loss: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate update using VAE in latent space."""
        # Optimize in latent space
        latent_grad = torch.randn_like(latent) * 0.1

        # Update latent representation
        new_latent = latent - latent_grad * (loss / 100)  # Scale by loss

        # Decode to parameter space
        new_params = self.vae.decode(new_latent).squeeze(0)

        update = new_params - params

        return update, new_latent

    def _gan_update(self, params: torch.Tensor, loss: float) -> torch.Tensor:
        """Generate update using GAN-inspired adversarial approach."""
        # Simplified: generate adversarial perturbation
        # In full implementation, would use discriminator network

        # Random direction
        direction = torch.randn_like(params)
        direction = direction / (torch.norm(direction) + 1e-8)

        # Adversarial step size
        step_size = 0.01 * np.exp(-loss)  # Larger steps when loss is high

        return direction * step_size

    def _update_surrogate_model(self):
        """Update surrogate network using replay buffer."""
        if len(self.replay_buffer) < 10:
            return

        # Sample batch from replay buffer
        batch_size = min(32, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)

        params_batch = []
        losses_batch = []

        for idx in indices:
            params, loss = self.replay_buffer[idx]
            params_batch.append(params)
            losses_batch.append(loss)

        params_tensor = torch.stack(params_batch)
        losses_tensor = torch.tensor(losses_batch, device=self.device)

        # Update surrogate
        self.surrogate_optimizer.zero_grad()
        predictions = self.surrogate_network(params_tensor)
        surrogate_loss = F.mse_loss(predictions, losses_tensor)
        surrogate_loss.backward()
        self.surrogate_optimizer.step()

    def _evaluate_objective(self, params: torch.Tensor,
                            task_spec: Optional[TaskSpecification]) -> float:
        """Evaluate the true objective function."""
        params_np = params.detach().cpu().numpy()

        if task_spec and 'objective_function' in task_spec.optimization_params:
            obj_func = task_spec.optimization_params['objective_function']
            if callable(obj_func):
                return float(obj_func(params_np))

        # Default quadratic objective
        return float(np.sum(params_np ** 2))

    def _get_network_weights_norm(self) -> float:
        """Get total norm of network weights."""
        total_norm = 0.0

        if self.surrogate_network is not None:
            for param in self.surrogate_network.parameters():
                total_norm += torch.norm(param).item() ** 2

        if self.update_network is not None:
            for param in self.update_network.parameters():
                total_norm += torch.norm(param).item() ** 2

        return np.sqrt(total_norm)

    def save_state(self, path: Path) -> None:
        """Save optimizer state to disk."""
        state = {
            'architecture': self.architecture.value,
            'exploration_noise': self.exploration_noise
        }

        if self.surrogate_network is not None:
            state['surrogate_network'] = self.surrogate_network.state_dict()

        if self.update_network is not None:
            state['update_network'] = self.update_network.state_dict()

        if self.vae is not None:
            state['vae'] = self.vae.state_dict()

        if self.use_replay:
            state['replay_buffer'] = list(self.replay_buffer)

        torch.save(state, path)
        self.logger.info(f"Saved neural optimizer state to {path}")

    def load_state(self, path: Path) -> None:
        """Load optimizer state from disk."""
        state = torch.load(path, map_location=self.device)

        self.architecture = NeuralArchitecture(state['architecture'])
        self.exploration_noise = state['exploration_noise']

        # Initialize networks if dimensions match
        if 'surrogate_network' in state and self.surrogate_network is not None:
            self.surrogate_network.load_state_dict(state['surrogate_network'])

        if 'update_network' in state and self.update_network is not None:
            self.update_network.load_state_dict(state['update_network'])

        if 'vae' in state and self.vae is not None:
            self.vae.load_state_dict(state['vae'])

        if 'replay_buffer' in state and self.use_replay:
            self.replay_buffer = deque(state['replay_buffer'], maxlen=self.replay_buffer.maxlen)

        self.logger.info(f"Loaded neural optimizer state from {path}")
