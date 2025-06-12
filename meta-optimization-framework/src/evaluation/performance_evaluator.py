"""
Performance Evaluator - Main Evaluation Orchestrator
====================================================

Coordinates all evaluation components to measure framework's target performance metrics:
- Accuracy: 19% ± 8% improvement
- Efficiency: 12% ± 4% gains  
- Cognitive Load: 22% ± 5% reduction
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

from ..core.meta_optimization import MetaOptimizer
from ..utils.statistical_analysis import StatisticalAnalyzer
from ..utils.failure_documentation import FailureDocumenter


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy_improvement: float
    efficiency_gain: float
    cognitive_load_reduction: float
    statistical_significance: Dict[str, Any]
    execution_time: float
    memory_usage: float
    convergence_metrics: Dict[str, Any]
    
    def meets_targets(self) -> Dict[str, bool]:
        """Check if results meet target performance metrics."""
        return {
            'accuracy': self.accuracy_improvement >= 0.11,  # Lower CI bound
            'efficiency': self.efficiency_gain >= 0.08,    # Lower CI bound
            'cognitive': self.cognitive_load_reduction >= 0.17  # Lower CI bound
        }


class PerformanceEvaluator:
    """
    Main performance evaluation orchestrator for the meta-optimization framework.
    
    Coordinates comprehensive evaluation across cognitive, computational, and
    statistical dimensions to validate the framework's target improvements.
    """
    
    def __init__(self, 
                 statistical_analyzer: Optional[StatisticalAnalyzer] = None,
                 failure_documenter: Optional[FailureDocumenter] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize the performance evaluator.
        
        Args:
            statistical_analyzer: Statistical analysis component
            failure_documenter: Failure tracking component
            output_dir: Directory for evaluation outputs
        """
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.failure_documenter = failure_documenter or FailureDocumenter()
        self.output_dir = output_dir or Path("results/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Target metrics
        self.targets = {
            'accuracy_improvement': 0.19,
            'accuracy_stddev': 0.08,
            'efficiency_gain': 0.12,
            'efficiency_stddev': 0.04,
            'cognitive_reduction': 0.22,
            'cognitive_stddev': 0.05
        }
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_framework(self,
                          optimizer: MetaOptimizer,
                          tasks: List[Dict[str, Any]],
                          baseline_models: List[Any],
                          n_trials: int = 10,
                          confidence_level: float = 0.95) -> EvaluationResult:
        """
        Comprehensive framework evaluation against target metrics.
        
        Args:
            optimizer: Meta-optimizer to evaluate
            tasks: List of cognitive tasks
            baseline_models: Baseline models for comparison
            n_trials: Number of evaluation trials
            confidence_level: Statistical confidence level
            
        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()
        
        try:
            # Initialize tracking
            accuracy_improvements = []
            efficiency_gains = []
            cognitive_reductions = []
            convergence_data = []
            
            for trial in range(n_trials):
                self.logger.info(f"Starting evaluation trial {trial + 1}/{n_trials}")
                
                # Run single trial evaluation
                trial_result = self._evaluate_single_trial(
                    optimizer, tasks, baseline_models, trial
                )
                
                accuracy_improvements.append(trial_result['accuracy_improvement'])
                efficiency_gains.append(trial_result['efficiency_gain'])
                cognitive_reductions.append(trial_result['cognitive_reduction'])
                convergence_data.append(trial_result['convergence_metrics'])
            
            # Compute statistical metrics
            statistical_results = self._compute_statistical_metrics(
                accuracy_improvements, efficiency_gains, cognitive_reductions,
                confidence_level
            )
            
            # Create final result
            result = EvaluationResult(
                accuracy_improvement=np.mean(accuracy_improvements),
                efficiency_gain=np.mean(efficiency_gains),
                cognitive_load_reduction=np.mean(cognitive_reductions),
                statistical_significance=statistical_results,
                execution_time=time.time() - start_time,
                memory_usage=self._measure_memory_usage(),
                convergence_metrics=self._aggregate_convergence_metrics(convergence_data)
            )
            
            # Store and validate result
            self.evaluation_history.append(result)
            self._validate_against_targets(result)
            
            return result
            
        except Exception as e:
            self.failure_documenter.document_failure(
                "evaluation_framework",
                str(e),
                {"optimizer": str(optimizer), "n_trials": n_trials}
            )
            raise
    
    def _evaluate_single_trial(self,
                              optimizer: MetaOptimizer,
                              tasks: List[Dict[str, Any]],
                              baseline_models: List[Any],
                              trial_idx: int) -> Dict[str, Any]:
        """Evaluate a single trial."""
        trial_start = time.time()
        
        # Generate trial data
        trial_results = {
            'accuracy_improvement': 0.0,
            'efficiency_gain': 0.0,
            'cognitive_reduction': 0.0,
            'convergence_metrics': {}
        }
        
        for task_idx, task in enumerate(tasks):
            # Optimize with meta-optimizer
            optimization_result = optimizer.optimize(
                task_specification=task,
                max_iterations=100,
                convergence_threshold=1e-4
            )
            
            # Compare with baseline
            if task_idx < len(baseline_models):
                baseline_result = self._evaluate_baseline(baseline_models[task_idx], task)
                
                # Compute improvements
                accuracy_improvement = (
                    optimization_result.final_metrics['accuracy'] - 
                    baseline_result['accuracy']
                ) / baseline_result['accuracy']
                
                efficiency_improvement = (
                    baseline_result['computation_time'] - 
                    optimization_result.final_metrics['computation_time']
                ) / baseline_result['computation_time']
                
                cognitive_improvement = (
                    baseline_result['cognitive_load'] - 
                    optimization_result.final_metrics['cognitive_load']
                ) / baseline_result['cognitive_load']
                
                # Accumulate improvements
                trial_results['accuracy_improvement'] += accuracy_improvement / len(tasks)
                trial_results['efficiency_gain'] += efficiency_improvement / len(tasks)
                trial_results['cognitive_reduction'] += cognitive_improvement / len(tasks)
                
                # Track convergence
                trial_results['convergence_metrics'][f'task_{task_idx}'] = {
                    'iterations': optimization_result.iterations,
                    'final_loss': optimization_result.final_loss,
                    'convergence_time': optimization_result.final_metrics['computation_time']
                }
        
        return trial_results
    
    def _evaluate_baseline(self, baseline_model: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate baseline model performance."""
        start_time = time.time()
        
        # Simulate baseline evaluation
        # In real implementation, this would run the actual baseline
        baseline_accuracy = np.random.uniform(0.65, 0.75)  # Typical baseline range
        computation_time = time.time() - start_time + np.random.uniform(0.1, 0.5)
        cognitive_load = np.random.uniform(0.6, 0.8)  # Normalized cognitive load
        
        return {
            'accuracy': baseline_accuracy,
            'computation_time': computation_time,
            'cognitive_load': cognitive_load
        }
    
    def _compute_statistical_metrics(self,
                                   accuracy_improvements: List[float],
                                   efficiency_gains: List[float],
                                   cognitive_reductions: List[float],
                                   confidence_level: float) -> Dict[str, Any]:
        """Compute statistical significance metrics."""
        
        # Compute confidence intervals
        accuracy_ci = self.statistical_analyzer.compute_confidence_interval(
            accuracy_improvements, confidence_level
        )
        efficiency_ci = self.statistical_analyzer.compute_confidence_interval(
            efficiency_gains, confidence_level
        )
        cognitive_ci = self.statistical_analyzer.compute_confidence_interval(
            cognitive_reductions, confidence_level
        )
        
        # Compute effect sizes
        accuracy_effect = self.statistical_analyzer.compute_effect_size(
            accuracy_improvements, [0] * len(accuracy_improvements)
        )
        efficiency_effect = self.statistical_analyzer.compute_effect_size(
            efficiency_gains, [0] * len(efficiency_gains)
        )
        cognitive_effect = self.statistical_analyzer.compute_effect_size(
            cognitive_reductions, [0] * len(cognitive_reductions)
        )
        
        # Test against targets
        target_tests = self._test_against_targets(
            accuracy_improvements, efficiency_gains, cognitive_reductions
        )
        
        return {
            'confidence_intervals': {
                'accuracy': accuracy_ci,
                'efficiency': efficiency_ci,
                'cognitive': cognitive_ci
            },
            'effect_sizes': {
                'accuracy': accuracy_effect,
                'efficiency': efficiency_effect,
                'cognitive': cognitive_effect
            },
            'target_tests': target_tests,
            'sample_sizes': {
                'accuracy': len(accuracy_improvements),
                'efficiency': len(efficiency_gains),
                'cognitive': len(cognitive_reductions)
            }
        }
    
    def _test_against_targets(self,
                            accuracy_improvements: List[float],
                            efficiency_gains: List[float],
                            cognitive_reductions: List[float]) -> Dict[str, Any]:
        """Test measurements against target metrics."""
        
        # One-sample t-tests against targets
        accuracy_test = self.statistical_analyzer.one_sample_ttest(
            accuracy_improvements, self.targets['accuracy_improvement']
        )
        efficiency_test = self.statistical_analyzer.one_sample_ttest(
            efficiency_gains, self.targets['efficiency_gain']
        )
        cognitive_test = self.statistical_analyzer.one_sample_ttest(
            cognitive_reductions, self.targets['cognitive_reduction']
        )
        
        return {
            'accuracy_target_test': accuracy_test,
            'efficiency_target_test': efficiency_test,
            'cognitive_target_test': cognitive_test
        }
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _aggregate_convergence_metrics(self, convergence_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate convergence metrics across trials."""
        all_iterations = []
        all_times = []
        all_losses = []
        
        for trial_data in convergence_data:
            for task_key, task_metrics in trial_data.items():
                all_iterations.append(task_metrics['iterations'])
                all_times.append(task_metrics['convergence_time'])
                all_losses.append(task_metrics['final_loss'])
        
        return {
            'mean_iterations': np.mean(all_iterations),
            'mean_convergence_time': np.mean(all_times),
            'mean_final_loss': np.mean(all_losses),
            'std_iterations': np.std(all_iterations),
            'std_convergence_time': np.std(all_times),
            'std_final_loss': np.std(all_losses)
        }
    
    def _validate_against_targets(self, result: EvaluationResult) -> None:
        """Validate results against target metrics and log findings."""
        target_status = result.meets_targets()
        
        self.logger.info("Evaluation Results Summary:")
        self.logger.info(f"Accuracy Improvement: {result.accuracy_improvement:.3f} "
                        f"(Target: {self.targets['accuracy_improvement']:.3f}) "
                        f"{'✓' if target_status['accuracy'] else '✗'}")
        self.logger.info(f"Efficiency Gain: {result.efficiency_gain:.3f} "
                        f"(Target: {self.targets['efficiency_gain']:.3f}) "
                        f"{'✓' if target_status['efficiency'] else '✗'}")
        self.logger.info(f"Cognitive Reduction: {result.cognitive_load_reduction:.3f} "
                        f"(Target: {self.targets['cognitive_reduction']:.3f}) "
                        f"{'✓' if target_status['cognitive'] else '✗'}")
        
        # Document if targets not met
        if not all(target_status.values()):
            self.failure_documenter.document_failure(
                "target_validation",
                "Framework did not meet all target metrics",
                {
                    "target_status": target_status,
                    "actual_results": {
                        "accuracy": result.accuracy_improvement,
                        "efficiency": result.efficiency_gain,
                        "cognitive": result.cognitive_load_reduction
                    },
                    "targets": self.targets
                }
            )
    
    def generate_evaluation_report(self, output_file: Optional[Path] = None) -> Path:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_history:
            raise ValueError("No evaluation results available for report generation")
        
        report_file = output_file or self.output_dir / "evaluation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Meta-Optimization Framework Evaluation Report\n\n")
            f.write("## Target Performance Metrics\n")
            f.write(f"- Accuracy Improvement: {self.targets['accuracy_improvement']:.1%} ± {self.targets['accuracy_stddev']:.1%}\n")
            f.write(f"- Efficiency Gain: {self.targets['efficiency_gain']:.1%} ± {self.targets['efficiency_stddev']:.1%}\n")
            f.write(f"- Cognitive Load Reduction: {self.targets['cognitive_reduction']:.1%} ± {self.targets['cognitive_stddev']:.1%}\n\n")
            
            f.write("## Latest Evaluation Results\n")
            latest = self.evaluation_history[-1]
            f.write(f"- Accuracy Improvement: {latest.accuracy_improvement:.3f}\n")
            f.write(f"- Efficiency Gain: {latest.efficiency_gain:.3f}\n")
            f.write(f"- Cognitive Load Reduction: {latest.cognitive_load_reduction:.3f}\n")
            f.write(f"- Execution Time: {latest.execution_time:.2f}s\n")
            f.write(f"- Memory Usage: {latest.memory_usage:.1f}MB\n\n")
            
            target_status = latest.meets_targets()
            f.write("## Target Achievement Status\n")
            for metric, achieved in target_status.items():
                status = "✓ ACHIEVED" if achieved else "✗ NOT MET"
                f.write(f"- {metric.title()}: {status}\n")
        
        return report_file