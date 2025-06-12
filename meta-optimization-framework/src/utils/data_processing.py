"""
Data Processing Utilities

Provides data handling utilities for the meta-optimization framework,
including preprocessing, normalization, and cognitive task data generation.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing class for cognitive tasks.
    
    Handles preprocessing, normalization, and generation of synthetic
    cognitive task data for training and evaluation.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data processor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    def generate_n_back_data(self, 
                           batch_size: int = 32,
                           sequence_length: int = 20,
                           n_back: int = 2,
                           feature_dim: int = 64,
                           match_probability: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic N-back task data.
        
        Args:
            batch_size: Number of sequences to generate
            sequence_length: Length of each sequence
            n_back: N-back level (e.g., 2 for 2-back)
            feature_dim: Dimensionality of features
            match_probability: Probability of n-back match
            
        Returns:
            Tuple of (input_sequences, target_labels)
        """
        # Generate random feature vectors
        sequences = torch.randn(batch_size, sequence_length, feature_dim)
        targets = torch.zeros(batch_size, sequence_length, 1)
        
        for batch_idx in range(batch_size):
            for pos in range(n_back, sequence_length):
                if np.random.random() < match_probability:
                    # Create n-back match
                    sequences[batch_idx, pos] = sequences[batch_idx, pos - n_back] + \
                                              torch.randn(feature_dim) * 0.1  # Add small noise
                    targets[batch_idx, pos] = 1.0
        
        return sequences, targets
    
    def generate_stroop_data(self,
                           batch_size: int = 32,
                           sequence_length: int = 10,
                           feature_dim: int = 64,
                           conflict_probability: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic Stroop task data.
        
        Args:
            batch_size: Number of sequences to generate
            sequence_length: Length of each sequence
            feature_dim: Dimensionality of features
            conflict_probability: Probability of conflict trials
            
        Returns:
            Tuple of (input_sequences, target_responses)
        """
        # Generate base stimuli
        sequences = torch.randn(batch_size, sequence_length, feature_dim)
        targets = torch.zeros(batch_size, sequence_length, 1)
        
        for batch_idx in range(batch_size):
            for pos in range(sequence_length):
                if np.random.random() < conflict_probability:
                    # Create conflict condition (harder to process)
                    conflict_noise = torch.randn(feature_dim) * 0.5
                    sequences[batch_idx, pos] += conflict_noise
                    targets[batch_idx, pos] = 1.0  # Conflict trial
                else:
                    # Congruent condition
                    targets[batch_idx, pos] = 0.0  # Non-conflict trial
        
        return sequences, targets
    
    def generate_planning_data(self,
                             batch_size: int = 32,
                             sequence_length: int = 15,
                             feature_dim: int = 64,
                             goal_complexity: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic planning task data.
        
        Args:
            batch_size: Number of sequences to generate
            sequence_length: Length of each sequence
            feature_dim: Dimensionality of features
            goal_complexity: Number of subgoals in planning task
            
        Returns:
            Tuple of (input_sequences, target_plans)
        """
        sequences = torch.randn(batch_size, sequence_length, feature_dim)
        targets = torch.zeros(batch_size, sequence_length, goal_complexity)
        
        for batch_idx in range(batch_size):
            # Create hierarchical goal structure
            goal_positions = np.random.choice(
                sequence_length, 
                size=min(goal_complexity, sequence_length), 
                replace=False
            )
            
            for i, pos in enumerate(sorted(goal_positions)):
                targets[batch_idx, pos, i] = 1.0
                
                # Add goal-related features to input
                goal_signal = torch.zeros(feature_dim)
                goal_signal[i * (feature_dim // goal_complexity):(i + 1) * (feature_dim // goal_complexity)] = 1.0
                sequences[batch_idx, pos] += goal_signal * 0.3
        
        return sequences, targets
    
    def generate_pattern_recognition_data(self,
                                        batch_size: int = 32,
                                        sequence_length: int = 12,
                                        feature_dim: int = 64,
                                        pattern_length: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic pattern recognition data.
        
        Args:
            batch_size: Number of sequences to generate
            sequence_length: Length of each sequence
            feature_dim: Dimensionality of features
            pattern_length: Length of patterns to recognize
            
        Returns:
            Tuple of (input_sequences, target_patterns)
        """
        sequences = torch.randn(batch_size, sequence_length, feature_dim)
        targets = torch.zeros(batch_size, sequence_length, 1)
        
        # Create base patterns
        num_patterns = 3
        base_patterns = torch.randn(num_patterns, pattern_length, feature_dim)
        
        for batch_idx in range(batch_size):
            # Randomly insert patterns
            pattern_id = np.random.randint(0, num_patterns)
            start_pos = np.random.randint(0, sequence_length - pattern_length + 1)
            
            # Insert pattern with some noise
            for i in range(pattern_length):
                sequences[batch_idx, start_pos + i] = base_patterns[pattern_id, i] + \
                                                    torch.randn(feature_dim) * 0.2
                targets[batch_idx, start_pos + i] = pattern_id + 1  # Pattern ID (1-indexed)
        
        return sequences, targets
    
    def normalize_data(self, 
                      data: torch.Tensor,
                      method: str = "z_score",
                      dim: Optional[int] = None) -> torch.Tensor:
        """
        Normalize data using specified method.
        
        Args:
            data: Input data tensor
            method: Normalization method ("z_score", "min_max", "unit_norm")
            dim: Dimension along which to normalize (None for global)
            
        Returns:
            Normalized data tensor
        """
        if method == "z_score":
            if dim is not None:
                mean = torch.mean(data, dim=dim, keepdim=True)
                std = torch.std(data, dim=dim, keepdim=True)
            else:
                mean = torch.mean(data)
                std = torch.std(data)
            
            normalized = (data - mean) / (std + 1e-8)
            
        elif method == "min_max":
            if dim is not None:
                min_val = torch.min(data, dim=dim, keepdim=True)[0]
                max_val = torch.max(data, dim=dim, keepdim=True)[0]
            else:
                min_val = torch.min(data)
                max_val = torch.max(data)
            
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            
        elif method == "unit_norm":
            if dim is not None:
                norm = torch.norm(data, dim=dim, keepdim=True)
            else:
                norm = torch.norm(data)
            
            normalized = data / (norm + 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def add_cognitive_noise(self,
                          data: torch.Tensor,
                          noise_type: str = "gaussian",
                          noise_level: float = 0.1) -> torch.Tensor:
        """
        Add cognitive-inspired noise to data.
        
        Args:
            data: Input data tensor
            noise_type: Type of noise ("gaussian", "attention", "memory_decay")
            noise_level: Strength of noise (0-1)
            
        Returns:
            Data with added noise
        """
        noisy_data = data.clone()
        
        if noise_type == "gaussian":
            # Simple Gaussian noise
            noise = torch.randn_like(data) * noise_level
            noisy_data += noise
            
        elif noise_type == "attention":
            # Attention-based noise (stronger at certain positions)
            attention_weights = torch.softmax(torch.randn(data.shape[1]), dim=0)
            for i in range(data.shape[1]):
                noise = torch.randn_like(data[:, i, :]) * noise_level * attention_weights[i]
                noisy_data[:, i, :] += noise
                
        elif noise_type == "memory_decay":
            # Memory decay noise (stronger for earlier items)
            decay_factor = 0.9
            for i in range(data.shape[1]):
                decay_strength = decay_factor ** (data.shape[1] - i - 1)
                noise = torch.randn_like(data[:, i, :]) * noise_level * (1 - decay_strength)
                noisy_data[:, i, :] += noise
                
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return noisy_data
    
    def create_cognitive_splits(self,
                              data: torch.Tensor,
                              targets: torch.Tensor,
                              split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                              stratify: bool = True) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create train/validation/test splits with cognitive considerations.
        
        Args:
            data: Input data tensor
            targets: Target tensor
            split_ratios: Ratios for (train, val, test) splits
            stratify: Whether to stratify splits based on targets
            
        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        n_samples = data.shape[0]
        indices = torch.randperm(n_samples)
        
        # Calculate split sizes
        train_size = int(n_samples * split_ratios[0])
        val_size = int(n_samples * split_ratios[1])
        
        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            "train": (data[train_indices], targets[train_indices]),
            "val": (data[val_indices], targets[val_indices]),
            "test": (data[test_indices], targets[test_indices])
        }
        
        return splits
    
    def augment_cognitive_data(self,
                             data: torch.Tensor,
                             targets: torch.Tensor,
                             augmentation_factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment cognitive task data with variations.
        
        Args:
            data: Input data tensor
            targets: Target tensor
            augmentation_factor: How many augmented versions to create
            
        Returns:
            Augmented data and targets
        """
        augmented_data = [data]
        augmented_targets = [targets]
        
        for _ in range(augmentation_factor - 1):
            # Create augmented version
            aug_data = data.clone()
            
            # Apply random transformations
            # 1. Add noise
            aug_data = self.add_cognitive_noise(aug_data, "gaussian", 0.05)
            
            # 2. Random scaling
            scale_factor = 0.9 + torch.rand(1) * 0.2  # Scale between 0.9 and 1.1
            aug_data *= scale_factor
            
            # 3. Random temporal shifts (for sequence data)
            if len(aug_data.shape) >= 3:
                shift = np.random.randint(-2, 3)  # Shift by -2 to +2 positions
                if shift != 0:
                    aug_data = torch.roll(aug_data, shift, dims=1)
            
            augmented_data.append(aug_data)
            augmented_targets.append(targets.clone())
        
        # Concatenate all versions
        final_data = torch.cat(augmented_data, dim=0)
        final_targets = torch.cat(augmented_targets, dim=0)
        
        return final_data, final_targets
    
    def compute_data_statistics(self, data: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive statistics for data analysis.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dictionary with data statistics
        """
        stats = {
            "mean": torch.mean(data).item(),
            "std": torch.std(data).item(),
            "min": torch.min(data).item(),
            "max": torch.max(data).item(),
            "median": torch.median(data).item(),
            "skewness": self._compute_skewness(data),
            "kurtosis": self._compute_kurtosis(data),
            "entropy": self._compute_entropy(data),
        }
        
        # Add shape information
        stats["shape"] = list(data.shape)
        stats["num_elements"] = data.numel()
        
        return stats
    
    def _compute_skewness(self, data: torch.Tensor) -> float:
        """Compute skewness of data."""
        mean = torch.mean(data)
        std = torch.std(data)
        skewness = torch.mean(((data - mean) / std) ** 3)
        return skewness.item()
    
    def _compute_kurtosis(self, data: torch.Tensor) -> float:
        """Compute kurtosis of data."""
        mean = torch.mean(data)
        std = torch.std(data)
        kurtosis = torch.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis.item()
    
    def _compute_entropy(self, data: torch.Tensor, bins: int = 50) -> float:
        """Compute entropy of data distribution."""
        # Convert to numpy for histogram
        data_np = data.detach().cpu().numpy().flatten()
        
        # Compute histogram
        hist, _ = np.histogram(data_np, bins=bins, density=True)
        
        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        
        return float(entropy)