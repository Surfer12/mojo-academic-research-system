#!/usr/bin/env python3
"""
Meta-Optimization Framework - Experiment Runner
==============================================

Automated experiment runner for systematic evaluation of different
configurations, parameters, and research scenarios.

Author: Ryan Oates, University of California Santa Barbara
"""

import argparse
import logging
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import itertools
from datetime import datetime
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.meta_optimization import MetaOptimizer, TaskSpecification
from src.optimization.adaptive_optimizer import AdaptiveOptimizer, AdaptationStrategy
from src.evaluation import PerformanceEvaluator, BenchmarkSuite, StatisticalValidator
from src.utils.data_processing import DataProcessor
from src.utils.failure_documentation import FailureDocumenter


def setup_logging(log_level: str = "INFO", experiment_name: str = "experiment") -> None:
    """Setup logging for experiments."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logging.info(f"Experiment logging initialized: {log_file}")


class ExperimentConfig:
    """Configuration for a single experiment."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict.get('name', 'unnamed_experiment')
        self.description = config_dict.get('description', '')
        self.optimizer_type = config_dict.get('optimizer_type', 'meta')
        self.optimizer_params = config_dict.get('optimizer_params', {})
        self.tasks = config_dict.get('tasks', [])
        self.evaluation_config = config_dict.get('evaluation', {})
        self.output_dir = Path(config_dict.get('output_dir', f'results/experiments/{self.name}'))
        self.metadata = config_dict.get('metadata', {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'tasks': self.tasks,
            'evaluation': self.evaluation_config,
            'output_dir': str(self.output_dir),
            'metadata': self.metadata
        }


class ExperimentRunner:
    """Main experiment runner class."""

    def __init__(self, output_base_dir: Path = Path("results/experiments")):
        self.output_base_dir = output_base_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        self.failure_documenter = FailureDocumenter()

        # Results storage
        self.experiment_results = {}

    def create_optimizer(self, optimizer_type: str, params: Dict[str, Any]):
        """Create optimizer instance based on type and parameters."""

        if optimizer_type == 'meta':
            return MetaOptimizer(**params)
        elif optimizer_type == 'adaptive':
            # Convert strategy string to enum if needed
            if 'adaptation_strategy' in params and isinstance(params['adaptation_strategy'], str):
                params['adaptation_strategy'] = AdaptationStrategy(params['adaptation_strategy'])
            return AdaptiveOptimizer(**params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def create_tasks(self, task_configs: List[Dict[str, Any]]) -> List[TaskSpecification]:
        """Create task specifications from configurations."""
        tasks = []

        for task_config in task_configs:
            task_type = task_config.get('type', 'optimization')
            task_params = task_config.get('params', {})

            # Generate task data based on type
            if task_type == 'n_back':
                data = self.data_processor.generate_n_back_data(**task_params)
            elif task_type == 'optimization':
                data = self.data_processor.generate_optimization_data(**task_params)
            elif task_type == 'matrix':
                data = self.data_processor.generate_matrix_data(**task_params)
            else:
                # Default optimization data
                data = self.data_processor.generate_optimization_data(dimension=10)

            # Create task specification
            task_spec = TaskSpecification(
                data=data,
                objective_function=task_config.get('objective_function', lambda x: sum(x**2)),
                constraints=task_config.get('constraints', []),
                cognitive_constraints=task_config.get('cognitive_constraints', {}),
                metadata=task_config
            )

            tasks.append(task_spec)

        return tasks

    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment."""

        self.logger.info(f"Starting experiment: {config.name}")
        experiment_start_time = datetime.now()

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save experiment configuration
            config_file = config.output_dir / "experiment_config.json"
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            # Create optimizer
            optimizer = self.create_optimizer(config.optimizer_type, config.optimizer_params)

            # Create tasks
            tasks = self.create_tasks(config.tasks)

            self.logger.info(f"Created {len(tasks)} tasks for experiment {config.name}")

            # Run optimization on each task
            task_results = []
            for i, task in enumerate(tasks):
                self.logger.info(f"Running task {i+1}/{len(tasks)}")

                task_start_time = datetime.now()

                # Run optimization
                optimization_result = optimizer.optimize(
                    task_specification=task,
                    max_iterations=config.evaluation_config.get('max_iterations', 1000),
                    convergence_threshold=config.evaluation_config.get('convergence_threshold', 1e-6)
                )

                task_end_time = datetime.now()
                task_duration = (task_end_time - task_start_time).total_seconds()

                # Store task result
                task_result = {
                    'task_index': i,
                    'task_metadata': task.metadata,
                    'optimization_result': {
                        'success': optimization_result.success,
                        'final_loss': optimization_result.final_loss,
                        'iterations': optimization_result.iterations,
                        'convergence_history': optimization_result.convergence_history[-10:],  # Last 10 for storage
                        'final_parameters': optimization_result.final_parameters
                    },
                    'execution_time': task_duration
                }

                task_results.append(task_result)

            # Run evaluation if configured
            evaluation_results = {}
            if config.evaluation_config.get('run_evaluation', True):
                self.logger.info("Running evaluation...")
                evaluation_results = self._run_evaluation(optimizer, tasks, config)

            # Compute experiment summary
            experiment_summary = self._compute_experiment_summary(task_results, evaluation_results)

            experiment_end_time = datetime.now()
            experiment_duration = (experiment_end_time - experiment_start_time).total_seconds()

            # Final experiment result
            experiment_result = {
                'config': config.to_dict(),
                'start_time': experiment_start_time.isoformat(),
                'end_time': experiment_end_time.isoformat(),
                'duration': experiment_duration,
                'task_results': task_results,
                'evaluation_results': evaluation_results,
                'summary': experiment_summary,
                'success': True
            }

            # Save experiment results
            results_file = config.output_dir / "experiment_results.json"
            with open(results_file, 'w') as f:
                json.dump(experiment_result, f, indent=2, default=str)

            # Generate experiment report
            self._generate_experiment_report(experiment_result, config.output_dir)

            self.logger.info(f"Experiment {config.name} completed successfully in {experiment_duration:.2f}s")

            return experiment_result

        except Exception as e:
            self.logger.error(f"Experiment {config.name} failed: {e}")

            # Document failure
            self.failure_documenter.document_failure(
                "experiment_execution",
                str(e),
                {
                    "experiment_name": config.name,
                    "config": config.to_dict()
                }
            )

            experiment_end_time = datetime.now()
            experiment_duration = (experiment_end_time - experiment_start_time).total_seconds()

            return {
                'config': config.to_dict(),
                'start_time': experiment_start_time.isoformat(),
                'end_time': experiment_end_time.isoformat(),
                'duration': experiment_duration,
                'error': str(e),
                'success': False
            }

    def _run_evaluation(self, optimizer, tasks, config: ExperimentConfig) -> Dict[str, Any]:
        """Run evaluation for an experiment."""

        evaluation_results = {}

        try:
            # Performance evaluation
            if config.evaluation_config.get('performance_evaluation', True):
                evaluator = PerformanceEvaluator(output_dir=config.output_dir)

                # Create simple baseline models
                baselines = [
                    type('Baseline', (), {
                        'optimize': lambda self, task: {
                            'accuracy': np.random.uniform(0.6, 0.8),
                            'computation_time': np.random.uniform(1.0, 2.0),
                            'cognitive_load': np.random.uniform(0.6, 0.8)
                        }
                    })() for _ in range(3)
                ]

                perf_result = evaluator.evaluate_framework(
                    optimizer=optimizer,
                    tasks=tasks,
                    baseline_models=baselines,
                    n_trials=config.evaluation_config.get('n_trials', 3)
                )

                evaluation_results['performance'] = {
                    'accuracy_improvement': perf_result.accuracy_improvement,
                    'efficiency_gain': perf_result.efficiency_gain,
                    'cognitive_load_reduction': perf_result.cognitive_load_reduction,
                    'execution_time': perf_result.execution_time
                }

            # Benchmark evaluation
            if config.evaluation_config.get('benchmark_evaluation', False):
                benchmark_suite = BenchmarkSuite()

                def model_function(task_data):
                    # Simple model function for benchmarking
                    import numpy as np
                    if isinstance(task_data, tuple) and len(task_data) > 1:
                        return np.random.randint(0, 2, len(task_data[1]))
                    else:
                        return np.random.rand(10)

                benchmark_results = benchmark_suite.run_full_benchmark_suite(
                    model_function=model_function,
                    baseline_function=model_function,  # Same for simplicity
                    categories=None  # All categories
                )

                evaluation_results['benchmarks'] = {
                    'n_benchmarks': len(benchmark_results),
                    'success_rate': sum(1 for r in benchmark_results.values() if r.meets_target) / len(benchmark_results)
                }

        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            evaluation_results['error'] = str(e)

        return evaluation_results

    def _compute_experiment_summary(self, task_results: List[Dict], evaluation_results: Dict) -> Dict[str, Any]:
        """Compute experiment summary statistics."""

        summary = {}

        # Task-level statistics
        successful_tasks = [r for r in task_results if r['optimization_result']['success']]

        summary['task_statistics'] = {
            'total_tasks': len(task_results),
            'successful_tasks': len(successful_tasks),
            'success_rate': len(successful_tasks) / len(task_results) if task_results else 0.0,
            'total_execution_time': sum(r['execution_time'] for r in task_results),
            'average_execution_time': np.mean([r['execution_time'] for r in task_results]) if task_results else 0.0
        }

        # Optimization statistics
        if successful_tasks:
            final_losses = [r['optimization_result']['final_loss'] for r in successful_tasks]
            iterations = [r['optimization_result']['iterations'] for r in successful_tasks]

            summary['optimization_statistics'] = {
                'mean_final_loss': np.mean(final_losses),
                'std_final_loss': np.std(final_losses),
                'mean_iterations': np.mean(iterations),
                'std_iterations': np.std(iterations),
                'min_final_loss': np.min(final_losses),
                'max_final_loss': np.max(final_losses)
            }

        # Evaluation statistics
        if evaluation_results:
            summary['evaluation_statistics'] = evaluation_results

        return summary

    def _generate_experiment_report(self, experiment_result: Dict[str, Any], output_dir: Path) -> None:
        """Generate experiment report."""

        report_file = output_dir / "experiment_report.md"

        with open(report_file, 'w') as f:
            config = experiment_result['config']
            summary = experiment_result['summary']

            f.write(f"# Experiment Report: {config['name']}\n\n")
            f.write(f"**Description**: {config['description']}\n\n")
            f.write(f"**Duration**: {experiment_result['duration']:.2f} seconds\n")
            f.write(f"**Status**: {'✓ SUCCESS' if experiment_result['success'] else '✗ FAILED'}\n\n")

            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- Optimizer: {config['optimizer_type']}\n")
            f.write(f"- Number of tasks: {len(config['tasks'])}\n")
            f.write(f"- Evaluation enabled: {config['evaluation'].get('run_evaluation', True)}\n\n")

            # Task statistics
            if 'task_statistics' in summary:
                stats = summary['task_statistics']
                f.write("## Task Statistics\n\n")
                f.write(f"- Total tasks: {stats['total_tasks']}\n")
                f.write(f"- Successful tasks: {stats['successful_tasks']}\n")
                f.write(f"- Success rate: {stats['success_rate']:.1%}\n")
                f.write(f"- Total execution time: {stats['total_execution_time']:.2f}s\n")
                f.write(f"- Average execution time: {stats['average_execution_time']:.2f}s\n\n")

            # Optimization statistics
            if 'optimization_statistics' in summary:
                stats = summary['optimization_statistics']
                f.write("## Optimization Statistics\n\n")
                f.write(f"- Mean final loss: {stats['mean_final_loss']:.6f} ± {stats['std_final_loss']:.6f}\n")
                f.write(f"- Mean iterations: {stats['mean_iterations']:.1f} ± {stats['std_iterations']:.1f}\n")
                f.write(f"- Best final loss: {stats['min_final_loss']:.6f}\n")
                f.write(f"- Worst final loss: {stats['max_final_loss']:.6f}\n\n")

            # Evaluation results
            if 'evaluation_statistics' in summary:
                stats = summary['evaluation_statistics']
                f.write("## Evaluation Results\n\n")

                if 'performance' in stats:
                    perf = stats['performance']
                    f.write("### Performance Metrics\n")
                    f.write(f"- Accuracy improvement: {perf['accuracy_improvement']:.3f}\n")
                    f.write(f"- Efficiency gain: {perf['efficiency_gain']:.3f}\n")
                    f.write(f"- Cognitive load reduction: {perf['cognitive_load_reduction']:.3f}\n\n")

                if 'benchmarks' in stats:
                    bench = stats['benchmarks']
                    f.write("### Benchmark Results\n")
                    f.write(f"- Number of benchmarks: {bench['n_benchmarks']}\n")
                    f.write(f"- Success rate: {bench['success_rate']:.1%}\n\n")

        self.logger.info(f"Experiment report generated: {report_file}")

    def run_parameter_sweep(self, base_config: Dict[str, Any],
                           parameter_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Run parameter sweep experiments."""

        self.logger.info("Starting parameter sweep...")

        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))

        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")

        sweep_results = []

        for i, param_combo in enumerate(param_combinations):
            # Create experiment config for this combination
            config_dict = base_config.copy()

            # Update parameters
            param_dict = dict(zip(param_names, param_combo))

            # Apply parameters to optimizer_params
            if 'optimizer_params' not in config_dict:
                config_dict['optimizer_params'] = {}
            config_dict['optimizer_params'].update(param_dict)

            # Update experiment name
            param_str = "_".join(f"{k}={v}" for k, v in param_dict.items())
            config_dict['name'] = f"{base_config.get('name', 'sweep')}_{i:03d}_{param_str}"

            # Create config object
            config = ExperimentConfig(config_dict)

            # Run experiment
            self.logger.info(f"Running sweep experiment {i+1}/{len(param_combinations)}: {config.name}")
            result = self.run_single_experiment(config)

            # Add parameter information to result
            result['parameters'] = param_dict
            sweep_results.append(result)

        # Generate sweep summary
        self._generate_sweep_summary(sweep_results, base_config.get('name', 'parameter_sweep'))

        return sweep_results

    def _generate_sweep_summary(self, sweep_results: List[Dict[str, Any]], sweep_name: str) -> None:
        """Generate parameter sweep summary."""

        summary_dir = self.output_base_dir / f"{sweep_name}_sweep_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Collect data for analysis
        successful_results = [r for r in sweep_results if r['success']]

        if not successful_results:
            self.logger.warning("No successful experiments in parameter sweep")
            return

        # Save summary data
        summary_file = summary_dir / "sweep_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(sweep_results, f, indent=2, default=str)

        # Generate analysis report
        report_file = summary_dir / "sweep_analysis.md"
        with open(report_file, 'w') as f:
            f.write(f"# Parameter Sweep Analysis: {sweep_name}\n\n")
            f.write(f"**Total experiments**: {len(sweep_results)}\n")
            f.write(f"**Successful experiments**: {len(successful_results)}\n")
            f.write(f"**Success rate**: {len(successful_results)/len(sweep_results):.1%}\n\n")

            # Best performing configuration
            if successful_results:
                # Sort by mean final loss
                best_result = min(
                    successful_results,
                    key=lambda r: r.get('summary', {}).get('optimization_statistics', {}).get('mean_final_loss', float('inf'))
                )

                f.write("## Best Configuration\n\n")
                f.write(f"**Experiment**: {best_result['config']['name']}\n")
                f.write(f"**Parameters**: {best_result.get('parameters', {})}\n")

                if 'summary' in best_result and 'optimization_statistics' in best_result['summary']:
                    stats = best_result['summary']['optimization_statistics']
                    f.write(f"**Mean final loss**: {stats['mean_final_loss']:.6f}\n")
                    f.write(f"**Mean iterations**: {stats['mean_iterations']:.1f}\n")

        self.logger.info(f"Parameter sweep summary generated: {summary_dir}")


def load_experiment_config(config_file: Path) -> Dict[str, Any]:
    """Load experiment configuration from file."""

    if config_file.suffix.lower() in ['.yaml', '.yml']:
        with open(config_file) as f:
            return yaml.safe_load(f)
    elif config_file.suffix.lower() == '.json':
        with open(config_file) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_file.suffix}")


def create_default_configs() -> Dict[str, Dict[str, Any]]:
    """Create default experiment configurations."""

    return {
        'basic_meta_optimization': {
            'name': 'basic_meta_optimization',
            'description': 'Basic meta-optimization experiment',
            'optimizer_type': 'meta',
            'optimizer_params': {},
            'tasks': [
                {
                    'type': 'optimization',
                    'params': {'dimension': 10}
                }
            ],
            'evaluation': {
                'run_evaluation': True,
                'performance_evaluation': True,
                'max_iterations': 500
            }
        },

        'adaptive_optimization': {
            'name': 'adaptive_optimization',
            'description': 'Adaptive optimization experiment',
            'optimizer_type': 'adaptive',
            'optimizer_params': {
                'adaptation_strategy': 'gradient_based',
                'learning_rate': 0.01
            },
            'tasks': [
                {
                    'type': 'optimization',
                    'params': {'dimension': 20}
                }
            ],
            'evaluation': {
                'run_evaluation': True,
                'performance_evaluation': True,
                'max_iterations': 300
            }
        },

        'cognitive_tasks': {
            'name': 'cognitive_tasks',
            'description': 'Cognitive task optimization experiment',
            'optimizer_type': 'meta',
            'optimizer_params': {},
            'tasks': [
                {
                    'type': 'n_back',
                    'params': {'n_level': 2, 'sequence_length': 50},
                    'cognitive_constraints': {
                        'working_memory_limit': 7,
                        'attention_threshold': 0.8
                    }
                }
            ],
            'evaluation': {
                'run_evaluation': True,
                'performance_evaluation': True,
                'max_iterations': 200
            }
        }
    }


def main():
    """Main experiment runner."""

    parser = argparse.ArgumentParser(description="Meta-Optimization Framework Experiment Runner")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Experiment configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=['basic_meta_optimization', 'adaptive_optimization', 'cognitive_tasks'],
        help="Pre-defined experiment to run"
    )
    parser.add_argument(
        "--parameter-sweep", "-p",
        action="store_true",
        help="Run parameter sweep experiment"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results/experiments"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    experiment_name = "experiment"
    if args.config:
        experiment_name = args.config.stem
    elif args.experiment:
        experiment_name = args.experiment

    setup_logging(args.log_level, experiment_name)

    # Initialize experiment runner
    runner = ExperimentRunner(args.output_dir)

    try:
        if args.config:
            # Load configuration from file
            config_dict = load_experiment_config(args.config)
            config = ExperimentConfig(config_dict)

            if args.parameter_sweep:
                # Extract parameter grid
                parameter_grid = config_dict.get('parameter_grid', {})
                if not parameter_grid:
                    logging.error("Parameter grid not found in configuration")
                    sys.exit(1)

                results = runner.run_parameter_sweep(config_dict, parameter_grid)
                logging.info(f"Parameter sweep completed: {len(results)} experiments")
            else:
                # Single experiment
                result = runner.run_single_experiment(config)
                if result['success']:
                    logging.info("Experiment completed successfully")
                else:
                    logging.error("Experiment failed")
                    sys.exit(1)

        elif args.experiment:
            # Use pre-defined experiment
            default_configs = create_default_configs()
            config_dict = default_configs[args.experiment]
            config = ExperimentConfig(config_dict)

            result = runner.run_single_experiment(config)
            if result['success']:
                logging.info("Experiment completed successfully")
            else:
                logging.error("Experiment failed")
                sys.exit(1)

        else:
            # Run all default experiments
            default_configs = create_default_configs()

            for exp_name, config_dict in default_configs.items():
                logging.info(f"Running default experiment: {exp_name}")
                config = ExperimentConfig(config_dict)
                result = runner.run_single_experiment(config)

                if result['success']:
                    logging.info(f"Experiment {exp_name} completed successfully")
                else:
                    logging.warning(f"Experiment {exp_name} failed")

        logging.info("All experiments completed")

    except Exception as e:
        logging.error(f"Experiment runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
