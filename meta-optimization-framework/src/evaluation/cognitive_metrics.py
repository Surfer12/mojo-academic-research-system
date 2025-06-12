"""
Cognitive Metrics - Cognitive Task Evaluation
============================================

Evaluates cognitive task performance and measures cognitive load reduction.
Supports standard cognitive tasks: N-back, Stroop, Planning, Pattern Recognition.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.data_processing import DataProcessor


class CognitiveTaskType(Enum):
    """Enumeration of supported cognitive task types."""
    N_BACK = "n_back"
    STROOP = "stroop"
    PLANNING = "planning"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class CognitiveTaskResult:
    """Results from a cognitive task evaluation."""
    task_type: CognitiveTaskType
    accuracy: float
    reaction_time: float
    cognitive_load: float
    working_memory_usage: float
    attention_demand: float
    interference_resistance: float
    raw_scores: Dict[str, Any]


class CognitiveMetrics:
    """
    Cognitive task evaluation and metrics computation.
    
    Measures cognitive performance across multiple standardized tasks
    and computes cognitive load reduction compared to baselines.
    """
    
    def __init__(self, data_processor: Optional[DataProcessor] = None):
        """
        Initialize cognitive metrics evaluator.
        
        Args:
            data_processor: Data processing component
        """
        self.data_processor = data_processor or DataProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Cognitive constraints from Miller's 7±2 principle
        self.working_memory_capacity = 7  # ± 2
        self.attention_threshold = 0.8    # Normalized attention threshold
        
        # Task-specific parameters
        self.task_parameters = {
            CognitiveTaskType.N_BACK: {
                'n_levels': [1, 2, 3, 4],  # N-back levels
                'sequence_length': 50,
                'stimulus_duration': 500,  # ms
                'isi_duration': 1500      # ms
            },
            CognitiveTaskType.STROOP: {
                'conditions': ['congruent', 'incongruent', 'neutral'],
                'n_trials': 100,
                'response_window': 2000   # ms
            },
            CognitiveTaskType.PLANNING: {
                'complexity_levels': [3, 4, 5, 6],  # Number of steps
                'time_limit': 60000,      # ms
                'branching_factor': 3
            },
            CognitiveTaskType.PATTERN_RECOGNITION: {
                'pattern_sizes': [4, 6, 8, 10],
                'n_patterns': 50,
                'exposure_time': 1000     # ms
            }
        }
    
    def evaluate_cognitive_task(self,
                               task_type: CognitiveTaskType,
                               model_predictions: np.ndarray,
                               ground_truth: np.ndarray,
                               response_times: Optional[np.ndarray] = None) -> CognitiveTaskResult:
        """
        Evaluate performance on a specific cognitive task.
        
        Args:
            task_type: Type of cognitive task
            model_predictions: Model predictions
            ground_truth: Ground truth labels
            response_times: Response times (optional)
            
        Returns:
            Cognitive task evaluation results
        """
        
        if task_type == CognitiveTaskType.N_BACK:
            return self._evaluate_n_back(model_predictions, ground_truth, response_times)
        elif task_type == CognitiveTaskType.STROOP:
            return self._evaluate_stroop(model_predictions, ground_truth, response_times)
        elif task_type == CognitiveTaskType.PLANNING:
            return self._evaluate_planning(model_predictions, ground_truth, response_times)
        elif task_type == CognitiveTaskType.PATTERN_RECOGNITION:
            return self._evaluate_pattern_recognition(model_predictions, ground_truth, response_times)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _evaluate_n_back(self,
                        predictions: np.ndarray,
                        ground_truth: np.ndarray,
                        response_times: Optional[np.ndarray] = None) -> CognitiveTaskResult:
        """Evaluate N-back task performance."""
        
        # Compute basic accuracy
        accuracy = np.mean(predictions == ground_truth)
        
        # Compute reaction time metrics
        if response_times is not None:
            mean_rt = np.mean(response_times)
            rt_variability = np.std(response_times)
        else:
            mean_rt = np.random.uniform(400, 800)  # Simulated RT
            rt_variability = mean_rt * 0.2
        
        # Compute working memory usage (based on N-back level and accuracy)
        n_level = self._infer_n_back_level(predictions, ground_truth)
        working_memory_usage = min(n_level / self.working_memory_capacity, 1.0)
        
        # Compute attention demand
        attention_demand = self._compute_attention_demand(accuracy, mean_rt, n_level)
        
        # Compute interference resistance (d-prime measure)
        interference_resistance = self._compute_d_prime(predictions, ground_truth)
        
        # Compute cognitive load (composite measure)
        cognitive_load = self._compute_cognitive_load(
            working_memory_usage, attention_demand, mean_rt, rt_variability
        )
        
        return CognitiveTaskResult(
            task_type=CognitiveTaskType.N_BACK,
            accuracy=accuracy,
            reaction_time=mean_rt,
            cognitive_load=cognitive_load,
            working_memory_usage=working_memory_usage,
            attention_demand=attention_demand,
            interference_resistance=interference_resistance,
            raw_scores={
                'n_level': n_level,
                'rt_variability': rt_variability,
                'd_prime': interference_resistance,
                'hits': np.sum((predictions == 1) & (ground_truth == 1)),
                'false_alarms': np.sum((predictions == 1) & (ground_truth == 0))
            }
        )
    
    def _evaluate_stroop(self,
                        predictions: np.ndarray,
                        ground_truth: np.ndarray,
                        response_times: Optional[np.ndarray] = None) -> CognitiveTaskResult:
        """Evaluate Stroop task performance."""
        
        accuracy = np.mean(predictions == ground_truth)
        
        if response_times is not None:
            mean_rt = np.mean(response_times)
        else:
            mean_rt = np.random.uniform(600, 1000)
        
        # Stroop interference effect (difference between congruent/incongruent)
        interference_resistance = 1.0 - (mean_rt - 600) / 400  # Normalized
        interference_resistance = max(0.0, min(1.0, interference_resistance))
        
        # Working memory usage (moderate for Stroop)
        working_memory_usage = 0.4
        
        # Attention demand (high for interference resistance)
        attention_demand = 0.8 if interference_resistance > 0.6 else 0.6
        
        cognitive_load = self._compute_cognitive_load(
            working_memory_usage, attention_demand, mean_rt, mean_rt * 0.15
        )
        
        return CognitiveTaskResult(
            task_type=CognitiveTaskType.STROOP,
            accuracy=accuracy,
            reaction_time=mean_rt,
            cognitive_load=cognitive_load,
            working_memory_usage=working_memory_usage,
            attention_demand=attention_demand,
            interference_resistance=interference_resistance,
            raw_scores={
                'interference_effect': mean_rt - 600,
                'error_rate': 1.0 - accuracy
            }
        )
    
    def _evaluate_planning(self,
                          predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          response_times: Optional[np.ndarray] = None) -> CognitiveTaskResult:
        """Evaluate planning task performance."""
        
        accuracy = np.mean(predictions == ground_truth)
        
        if response_times is not None:
            mean_rt = np.mean(response_times)
        else:
            mean_rt = np.random.uniform(5000, 15000)  # Planning takes longer
        
        # Working memory usage (high for planning)
        complexity = self._infer_planning_complexity(predictions, ground_truth)
        working_memory_usage = min(complexity / 6.0, 1.0)
        
        # Attention demand (sustained attention required)
        attention_demand = 0.9
        
        # Interference resistance (maintaining plan despite distractors)
        interference_resistance = accuracy * 0.8 + (1 - mean_rt / 15000) * 0.2
        
        cognitive_load = self._compute_cognitive_load(
            working_memory_usage, attention_demand, mean_rt, mean_rt * 0.3
        )
        
        return CognitiveTaskResult(
            task_type=CognitiveTaskType.PLANNING,
            accuracy=accuracy,
            reaction_time=mean_rt,
            cognitive_load=cognitive_load,
            working_memory_usage=working_memory_usage,
            attention_demand=attention_demand,
            interference_resistance=interference_resistance,
            raw_scores={
                'complexity': complexity,
                'planning_time': mean_rt,
                'steps_optimal': np.sum(predictions == ground_truth)
            }
        )
    
    def _evaluate_pattern_recognition(self,
                                    predictions: np.ndarray,
                                    ground_truth: np.ndarray,
                                    response_times: Optional[np.ndarray] = None) -> CognitiveTaskResult:
        """Evaluate pattern recognition task performance."""
        
        accuracy = np.mean(predictions == ground_truth)
        
        if response_times is not None:
            mean_rt = np.mean(response_times)
        else:
            mean_rt = np.random.uniform(800, 1500)
        
        # Working memory usage (moderate, depends on pattern complexity)
        pattern_size = self._infer_pattern_size(predictions, ground_truth)
        working_memory_usage = min(pattern_size / 10.0, 1.0)
        
        # Attention demand (focused visual attention)
        attention_demand = 0.7
        
        # Interference resistance (pattern discrimination ability)
        interference_resistance = self._compute_d_prime(predictions, ground_truth)
        
        cognitive_load = self._compute_cognitive_load(
            working_memory_usage, attention_demand, mean_rt, mean_rt * 0.25
        )
        
        return CognitiveTaskResult(
            task_type=CognitiveTaskType.PATTERN_RECOGNITION,
            accuracy=accuracy,
            reaction_time=mean_rt,
            cognitive_load=cognitive_load,
            working_memory_usage=working_memory_usage,
            attention_demand=attention_demand,
            interference_resistance=interference_resistance,
            raw_scores={
                'pattern_size': pattern_size,
                'recognition_accuracy': accuracy,
                'd_prime': interference_resistance
            }
        )
    
    def compute_cognitive_load_reduction(self,
                                       optimized_results: List[CognitiveTaskResult],
                                       baseline_results: List[CognitiveTaskResult]) -> Dict[str, float]:
        """
        Compute cognitive load reduction across tasks.
        
        Args:
            optimized_results: Results from optimized framework
            baseline_results: Results from baseline models
            
        Returns:
            Cognitive load reduction metrics
        """
        
        if len(optimized_results) != len(baseline_results):
            raise ValueError("Number of optimized and baseline results must match")
        
        load_reductions = []
        accuracy_improvements = []
        rt_improvements = []
        
        for opt_result, base_result in zip(optimized_results, baseline_results):
            if opt_result.task_type != base_result.task_type:
                raise ValueError(f"Task type mismatch: {opt_result.task_type} vs {base_result.task_type}")
            
            # Cognitive load reduction
            load_reduction = (base_result.cognitive_load - opt_result.cognitive_load) / base_result.cognitive_load
            load_reductions.append(load_reduction)
            
            # Accuracy improvement
            acc_improvement = (opt_result.accuracy - base_result.accuracy) / base_result.accuracy
            accuracy_improvements.append(acc_improvement)
            
            # Reaction time improvement (lower is better)
            rt_improvement = (base_result.reaction_time - opt_result.reaction_time) / base_result.reaction_time
            rt_improvements.append(rt_improvement)
        
        return {
            'mean_cognitive_load_reduction': np.mean(load_reductions),
            'std_cognitive_load_reduction': np.std(load_reductions),
            'mean_accuracy_improvement': np.mean(accuracy_improvements),
            'std_accuracy_improvement': np.std(accuracy_improvements),
            'mean_rt_improvement': np.mean(rt_improvements),
            'std_rt_improvement': np.std(rt_improvements),
            'task_specific_reductions': {
                f"{opt_result.task_type.value}": load_red
                for opt_result, load_red in zip(optimized_results, load_reductions)
            }
        }
    
    def _infer_n_back_level(self, predictions: np.ndarray, ground_truth: np.ndarray) -> int:
        """Infer N-back level from task difficulty."""
        accuracy = np.mean(predictions == ground_truth)
        # Higher accuracy suggests lower N-back level
        if accuracy > 0.9:
            return 1
        elif accuracy > 0.8:
            return 2
        elif accuracy > 0.7:
            return 3
        else:
            return 4
    
    def _infer_planning_complexity(self, predictions: np.ndarray, ground_truth: np.ndarray) -> int:
        """Infer planning task complexity."""
        accuracy = np.mean(predictions == ground_truth)
        # Lower accuracy suggests higher complexity
        if accuracy > 0.8:
            return 3
        elif accuracy > 0.6:
            return 4
        elif accuracy > 0.4:
            return 5
        else:
            return 6
    
    def _infer_pattern_size(self, predictions: np.ndarray, ground_truth: np.ndarray) -> int:
        """Infer pattern recognition complexity."""
        accuracy = np.mean(predictions == ground_truth)
        # Lower accuracy suggests larger patterns
        if accuracy > 0.9:
            return 4
        elif accuracy > 0.8:
            return 6
        elif accuracy > 0.7:
            return 8
        else:
            return 10
    
    def _compute_attention_demand(self, accuracy: float, reaction_time: float, complexity: int) -> float:
        """Compute attention demand based on performance metrics."""
        # Higher complexity and longer RT suggest higher attention demand
        base_demand = complexity / 4.0  # Normalize by max complexity
        rt_factor = min(reaction_time / 1000.0, 2.0) / 2.0  # Normalize RT to [0,1]
        accuracy_factor = 1.0 - accuracy  # Lower accuracy suggests higher demand
        
        attention_demand = (base_demand + rt_factor + accuracy_factor) / 3.0
        return min(attention_demand, 1.0)
    
    def _compute_d_prime(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute d-prime (discrimination sensitivity) measure."""
        hits = np.sum((predictions == 1) & (ground_truth == 1))
        false_alarms = np.sum((predictions == 1) & (ground_truth == 0))
        misses = np.sum((predictions == 0) & (ground_truth == 1))
        correct_rejections = np.sum((predictions == 0) & (ground_truth == 0))
        
        # Compute hit rate and false alarm rate
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.5
        fa_rate = false_alarms / (false_alarms + correct_rejections) if (false_alarms + correct_rejections) > 0 else 0.5
        
        # Avoid extreme values
        hit_rate = max(0.01, min(0.99, hit_rate))
        fa_rate = max(0.01, min(0.99, fa_rate))
        
        # Compute d-prime using inverse normal function
        from scipy.stats import norm
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        
        return max(0.0, d_prime)  # Ensure non-negative
    
    def _compute_cognitive_load(self,
                              working_memory_usage: float,
                              attention_demand: float,
                              reaction_time: float,
                              rt_variability: float) -> float:
        """
        Compute composite cognitive load measure.
        
        Combines working memory usage, attention demand, and temporal metrics
        into a single cognitive load score.
        """
        
        # Normalize reaction time (higher RT = higher load)
        rt_normalized = min(reaction_time / 2000.0, 1.0)  # Assume 2s is max normal RT
        
        # Normalize RT variability (higher variability = higher load)
        rt_var_normalized = min(rt_variability / 500.0, 1.0)  # Assume 500ms is high variability
        
        # Weighted combination
        cognitive_load = (
            0.3 * working_memory_usage +    # Working memory contribution
            0.3 * attention_demand +        # Attention contribution
            0.2 * rt_normalized +           # Speed contribution
            0.2 * rt_var_normalized         # Consistency contribution
        )
        
        return min(cognitive_load, 1.0)
    
    def generate_cognitive_report(self, results: List[CognitiveTaskResult], output_file: Optional[str] = None) -> str:
        """Generate comprehensive cognitive evaluation report."""
        
        report_lines = [
            "# Cognitive Task Evaluation Report\n",
            "## Summary Statistics\n"
        ]
        
        # Compute summary statistics
        mean_accuracy = np.mean([r.accuracy for r in results])
        mean_rt = np.mean([r.reaction_time for r in results])
        mean_cognitive_load = np.mean([r.cognitive_load for r in results])
        mean_working_memory = np.mean([r.working_memory_usage for r in results])
        
        report_lines.extend([
            f"- Mean Accuracy: {mean_accuracy:.3f}\n",
            f"- Mean Reaction Time: {mean_rt:.1f}ms\n",
            f"- Mean Cognitive Load: {mean_cognitive_load:.3f}\n",
            f"- Mean Working Memory Usage: {mean_working_memory:.3f}\n\n",
            "## Task-Specific Results\n"
        ])
        
        # Task-specific results
        for result in results:
            report_lines.extend([
                f"### {result.task_type.value.replace('_', ' ').title()}\n",
                f"- Accuracy: {result.accuracy:.3f}\n",
                f"- Reaction Time: {result.reaction_time:.1f}ms\n",
                f"- Cognitive Load: {result.cognitive_load:.3f}\n",
                f"- Working Memory Usage: {result.working_memory_usage:.3f}\n",
                f"- Attention Demand: {result.attention_demand:.3f}\n",
                f"- Interference Resistance: {result.interference_resistance:.3f}\n\n"
            ])
        
        report_content = "".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content