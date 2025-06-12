"""
Efficiency Analyzer - Computational Efficiency Measurement
=========================================================

Measures computational efficiency and validates the target 12% ± 4% efficiency gains.
Tracks memory usage, computation time, energy consumption, and throughput metrics.
"""

import time
import psutil
import gc
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import torch
from pathlib import Path
import logging

from ..utils.statistical_analysis import StatisticalAnalyzer


@dataclass
class EfficiencyMetrics:
    """Container for efficiency measurement results."""
    computation_time: float
    memory_usage_peak: float
    memory_usage_average: float
    cpu_usage_peak: float
    cpu_usage_average: float
    gpu_usage_peak: float
    gpu_usage_average: float
    throughput: float
    energy_estimate: float
    cache_efficiency: float
    memory_fragmentation: float


@dataclass
class EfficiencyComparison:
    """Container for efficiency comparison results."""
    baseline_metrics: EfficiencyMetrics
    optimized_metrics: EfficiencyMetrics
    efficiency_gain: float
    relative_improvements: Dict[str, float]
    statistical_significance: Dict[str, Any]


class EfficiencyAnalyzer:
    """
    Computational efficiency analyzer for the meta-optimization framework.
    
    Measures and validates the target 12% ± 4% efficiency gains through
    comprehensive computational resource monitoring and analysis.
    """
    
    def __init__(self, 
                 statistical_analyzer: Optional[StatisticalAnalyzer] = None,
                 monitoring_interval: float = 0.1):
        """
        Initialize efficiency analyzer.
        
        Args:
            statistical_analyzer: Statistical analysis component
            monitoring_interval: Monitoring interval in seconds
        """
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # Target efficiency gain: 12% ± 4%
        self.target_efficiency_gain = 0.12
        self.target_efficiency_stddev = 0.04
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_data = []
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.logger.info(f"GPU monitoring enabled: {self.gpu_count} devices")
    
    def measure_efficiency(self,
                          optimization_function: Callable,
                          *args,
                          **kwargs) -> EfficiencyMetrics:
        """
        Measure computational efficiency of an optimization function.
        
        Args:
            optimization_function: Function to measure
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Comprehensive efficiency metrics
        """
        # Clear cache and garbage collect
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()
        
        # Start monitoring
        self._start_monitoring()
        
        try:
            # Measure execution
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            # Execute function
            result = optimization_function(*args, **kwargs)
            
            # End measurement
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            computation_time = end_time - start_time
            
        finally:
            # Stop monitoring
            self._stop_monitoring()
        
        # Analyze monitoring data
        metrics = self._analyze_monitoring_data(
            computation_time, start_memory, end_memory
        )
        
        # Add throughput estimation
        if hasattr(result, 'iterations'):
            metrics.throughput = result.iterations / computation_time
        else:
            metrics.throughput = 1.0 / computation_time  # Operations per second
        
        return metrics
    
    def compare_efficiency(self,
                          baseline_function: Callable,
                          optimized_function: Callable,
                          test_cases: List[Tuple],
                          n_trials: int = 5) -> EfficiencyComparison:
        """
        Compare efficiency between baseline and optimized functions.
        
        Args:
            baseline_function: Baseline function to compare against
            optimized_function: Optimized function to evaluate
            test_cases: List of (args, kwargs) tuples for testing
            n_trials: Number of trials per test case
            
        Returns:
            Comprehensive efficiency comparison
        """
        baseline_measurements = []
        optimized_measurements = []
        
        for trial in range(n_trials):
            self.logger.info(f"Running efficiency comparison trial {trial + 1}/{n_trials}")
            
            for case_idx, (args, kwargs) in enumerate(test_cases):
                self.logger.debug(f"Testing case {case_idx + 1}/{len(test_cases)}")
                
                # Measure baseline
                baseline_metrics = self.measure_efficiency(
                    baseline_function, *args, **kwargs
                )
                baseline_measurements.append(baseline_metrics)
                
                # Measure optimized
                optimized_metrics = self.measure_efficiency(
                    optimized_function, *args, **kwargs
                )
                optimized_measurements.append(optimized_metrics)
        
        # Compute average metrics
        avg_baseline = self._average_metrics(baseline_measurements)
        avg_optimized = self._average_metrics(optimized_measurements)
        
        # Compute efficiency gain
        efficiency_gain = self._compute_efficiency_gain(avg_baseline, avg_optimized)
        
        # Compute relative improvements
        relative_improvements = self._compute_relative_improvements(
            avg_baseline, avg_optimized
        )
        
        # Statistical significance testing
        statistical_significance = self._test_efficiency_significance(
            baseline_measurements, optimized_measurements
        )
        
        return EfficiencyComparison(
            baseline_metrics=avg_baseline,
            optimized_metrics=avg_optimized,
            efficiency_gain=efficiency_gain,
            relative_improvements=relative_improvements,
            statistical_significance=statistical_significance
        )
    
    def validate_efficiency_target(self, comparison: EfficiencyComparison) -> Dict[str, Any]:
        """
        Validate efficiency gains against target metrics.
        
        Args:
            comparison: Efficiency comparison results
            
        Returns:
            Validation results with target achievement status
        """
        target_lower_bound = self.target_efficiency_gain - self.target_efficiency_stddev
        target_upper_bound = self.target_efficiency_gain + self.target_efficiency_stddev
        
        meets_target = (
            comparison.efficiency_gain >= target_lower_bound and
            comparison.efficiency_gain <= target_upper_bound
        )
        
        exceeds_minimum = comparison.efficiency_gain >= (self.target_efficiency_gain - 1.96 * self.target_efficiency_stddev / np.sqrt(10))
        
        validation_result = {
            'meets_target': meets_target,
            'exceeds_minimum': exceeds_minimum,
            'efficiency_gain': comparison.efficiency_gain,
            'target_range': (target_lower_bound, target_upper_bound),
            'target_center': self.target_efficiency_gain,
            'statistical_significance': comparison.statistical_significance,
            'detailed_improvements': comparison.relative_improvements
        }
        
        self.logger.info(f"Efficiency Validation Results:")
        self.logger.info(f"  Achieved Gain: {comparison.efficiency_gain:.3f} ({comparison.efficiency_gain:.1%})")
        self.logger.info(f"  Target Range: [{target_lower_bound:.3f}, {target_upper_bound:.3f}] ({target_lower_bound:.1%} - {target_upper_bound:.1%})")
        self.logger.info(f"  Meets Target: {'✓' if meets_target else '✗'}")
        self.logger.info(f"  Exceeds Minimum: {'✓' if exceeds_minimum else '✗'}")
        
        return validation_result
    
    def _start_monitoring(self) -> None:
        """Start system resource monitoring."""
        self.monitoring_active = True
        self.monitoring_data = []
        
        def monitor():
            while self.monitoring_active:
                timestamp = time.perf_counter()
                
                # CPU and memory monitoring
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                data_point = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_used': memory.used,
                    'memory_available': memory.available,
                    'memory_percent': memory.percent
                }
                
                # GPU monitoring
                if self.gpu_available:
                    try:
                        gpu_data = self._get_gpu_metrics()
                        data_point.update(gpu_data)
                    except Exception as e:
                        self.logger.warning(f"GPU monitoring error: {e}")
                
                self.monitoring_data.append(data_point)
                time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self) -> None:
        """Stop system resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory = psutil.virtual_memory()
        
        usage = {
            'system_memory_used': memory.used / 1024**3,  # GB
            'system_memory_percent': memory.percent
        }
        
        if self.gpu_available:
            try:
                for i in range(self.gpu_count):
                    memory_info = torch.cuda.memory_stats(i)
                    usage[f'gpu_{i}_memory_allocated'] = memory_info.get('allocated_bytes.all.current', 0) / 1024**3
                    usage[f'gpu_{i}_memory_reserved'] = memory_info.get('reserved_bytes.all.current', 0) / 1024**3
            except Exception as e:
                self.logger.warning(f"GPU memory monitoring error: {e}")
        
        return usage
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics."""
        gpu_data = {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_data[f'gpu_{i}_utilization'] = util.gpu
                gpu_data[f'gpu_{i}_memory_utilization'] = util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_data[f'gpu_{i}_temperature'] = temp
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                gpu_data[f'gpu_{i}_power'] = power
                
        except ImportError:
            # Fallback to PyTorch GPU monitoring
            for i in range(self.gpu_count):
                gpu_data[f'gpu_{i}_utilization'] = 0.0  # PyTorch doesn't provide utilization
                
        except Exception as e:
            self.logger.warning(f"Detailed GPU monitoring unavailable: {e}")
        
        return gpu_data
    
    def _analyze_monitoring_data(self,
                               computation_time: float,
                               start_memory: Dict[str, float],
                               end_memory: Dict[str, float]) -> EfficiencyMetrics:
        """Analyze collected monitoring data."""
        
        if not self.monitoring_data:
            # Fallback metrics if monitoring failed
            return EfficiencyMetrics(
                computation_time=computation_time,
                memory_usage_peak=end_memory.get('system_memory_used', 0.0),
                memory_usage_average=end_memory.get('system_memory_used', 0.0),
                cpu_usage_peak=0.0,
                cpu_usage_average=0.0,
                gpu_usage_peak=0.0,
                gpu_usage_average=0.0,
                throughput=1.0 / computation_time if computation_time > 0 else 0.0,
                energy_estimate=0.0,
                cache_efficiency=1.0,
                memory_fragmentation=0.0
            )
        
        # Extract time series data
        cpu_usage = [d['cpu_percent'] for d in self.monitoring_data]
        memory_usage = [d['memory_used'] / 1024**3 for d in self.monitoring_data]  # GB
        
        # GPU metrics
        gpu_usage = []
        gpu_power = []
        for data in self.monitoring_data:
            gpu_util = [v for k, v in data.items() if k.endswith('_utilization') and 'gpu_' in k]
            gpu_power_vals = [v for k, v in data.items() if k.endswith('_power') and 'gpu_' in k]
            
            if gpu_util:
                gpu_usage.append(np.mean(gpu_util))
            if gpu_power_vals:
                gpu_power.append(np.sum(gpu_power_vals))
        
        # Compute metrics
        memory_peak = max(memory_usage) if memory_usage else 0.0
        memory_avg = np.mean(memory_usage) if memory_usage else 0.0
        cpu_peak = max(cpu_usage) if cpu_usage else 0.0
        cpu_avg = np.mean(cpu_usage) if cpu_usage else 0.0
        gpu_peak = max(gpu_usage) if gpu_usage else 0.0
        gpu_avg = np.mean(gpu_usage) if gpu_usage else 0.0
        
        # Energy estimation (simplified)
        energy_estimate = self._estimate_energy_consumption(
            computation_time, cpu_avg, gpu_power
        )
        
        # Cache efficiency (memory access pattern analysis)
        cache_efficiency = self._estimate_cache_efficiency(memory_usage)
        
        # Memory fragmentation
        memory_fragmentation = self._estimate_memory_fragmentation(
            start_memory, end_memory
        )
        
        return EfficiencyMetrics(
            computation_time=computation_time,
            memory_usage_peak=memory_peak,
            memory_usage_average=memory_avg,
            cpu_usage_peak=cpu_peak,
            cpu_usage_average=cpu_avg,
            gpu_usage_peak=gpu_peak,
            gpu_usage_average=gpu_avg,
            throughput=1.0 / computation_time if computation_time > 0 else 0.0,
            energy_estimate=energy_estimate,
            cache_efficiency=cache_efficiency,
            memory_fragmentation=memory_fragmentation
        )
    
    def _estimate_energy_consumption(self,
                                   computation_time: float,
                                   avg_cpu_usage: float,
                                   gpu_power_readings: List[float]) -> float:
        """Estimate energy consumption in joules."""
        
        # CPU energy estimation (simplified)
        cpu_base_power = 65.0  # Watts (typical CPU TDP)
        cpu_energy = cpu_base_power * (avg_cpu_usage / 100.0) * computation_time
        
        # GPU energy estimation
        gpu_energy = 0.0
        if gpu_power_readings:
            avg_gpu_power = np.mean(gpu_power_readings)
            gpu_energy = avg_gpu_power * computation_time
        
        return cpu_energy + gpu_energy
    
    def _estimate_cache_efficiency(self, memory_usage: List[float]) -> float:
        """Estimate cache efficiency from memory access patterns."""
        if len(memory_usage) < 2:
            return 1.0
        
        # Measure memory access locality (lower variation = better cache efficiency)
        memory_variation = np.std(memory_usage) / np.mean(memory_usage) if np.mean(memory_usage) > 0 else 0.0
        
        # Convert to efficiency score (lower variation = higher efficiency)
        cache_efficiency = max(0.0, 1.0 - memory_variation)
        return min(1.0, cache_efficiency)
    
    def _estimate_memory_fragmentation(self,
                                     start_memory: Dict[str, float],
                                     end_memory: Dict[str, float]) -> float:
        """Estimate memory fragmentation."""
        start_used = start_memory.get('system_memory_used', 0.0)
        end_used = end_memory.get('system_memory_used', 0.0)
        
        memory_increase = end_used - start_used
        
        # Simple fragmentation estimate based on memory growth pattern
        if memory_increase <= 0:
            return 0.0
        
        # Higher fragmentation if memory increase is disproportionate
        fragmentation = min(memory_increase / 10.0, 1.0)  # Normalize
        return fragmentation
    
    def _average_metrics(self, metrics_list: List[EfficiencyMetrics]) -> EfficiencyMetrics:
        """Compute average of efficiency metrics."""
        if not metrics_list:
            raise ValueError("Empty metrics list")
        
        return EfficiencyMetrics(
            computation_time=np.mean([m.computation_time for m in metrics_list]),
            memory_usage_peak=np.mean([m.memory_usage_peak for m in metrics_list]),
            memory_usage_average=np.mean([m.memory_usage_average for m in metrics_list]),
            cpu_usage_peak=np.mean([m.cpu_usage_peak for m in metrics_list]),
            cpu_usage_average=np.mean([m.cpu_usage_average for m in metrics_list]),
            gpu_usage_peak=np.mean([m.gpu_usage_peak for m in metrics_list]),
            gpu_usage_average=np.mean([m.gpu_usage_average for m in metrics_list]),
            throughput=np.mean([m.throughput for m in metrics_list]),
            energy_estimate=np.mean([m.energy_estimate for m in metrics_list]),
            cache_efficiency=np.mean([m.cache_efficiency for m in metrics_list]),
            memory_fragmentation=np.mean([m.memory_fragmentation for m in metrics_list])
        )
    
    def _compute_efficiency_gain(self,
                               baseline: EfficiencyMetrics,
                               optimized: EfficiencyMetrics) -> float:
        """Compute overall efficiency gain."""
        
        # Weighted efficiency score combining multiple factors
        def efficiency_score(metrics: EfficiencyMetrics) -> float:
            # Lower is better for time, memory, energy
            # Higher is better for throughput, cache efficiency
            time_score = 1.0 / max(metrics.computation_time, 1e-6)
            memory_score = 1.0 / max(metrics.memory_usage_average, 1e-3)
            energy_score = 1.0 / max(metrics.energy_estimate, 1e-3)
            throughput_score = metrics.throughput
            cache_score = metrics.cache_efficiency
            
            # Weighted combination
            score = (
                0.3 * time_score +
                0.2 * memory_score +
                0.2 * energy_score +
                0.2 * throughput_score +
                0.1 * cache_score
            )
            return score
        
        baseline_score = efficiency_score(baseline)
        optimized_score = efficiency_score(optimized)
        
        if baseline_score == 0:
            return 0.0
        
        efficiency_gain = (optimized_score - baseline_score) / baseline_score
        return efficiency_gain
    
    def _compute_relative_improvements(self,
                                     baseline: EfficiencyMetrics,
                                     optimized: EfficiencyMetrics) -> Dict[str, float]:
        """Compute relative improvements for each metric."""
        
        def safe_improvement(baseline_val: float, optimized_val: float, lower_is_better: bool = True) -> float:
            if baseline_val == 0:
                return 0.0
            
            if lower_is_better:
                return (baseline_val - optimized_val) / baseline_val
            else:
                return (optimized_val - baseline_val) / baseline_val
        
        return {
            'computation_time': safe_improvement(baseline.computation_time, optimized.computation_time),
            'memory_usage': safe_improvement(baseline.memory_usage_average, optimized.memory_usage_average),
            'cpu_usage': safe_improvement(baseline.cpu_usage_average, optimized.cpu_usage_average),
            'gpu_usage': safe_improvement(baseline.gpu_usage_average, optimized.gpu_usage_average),
            'energy_consumption': safe_improvement(baseline.energy_estimate, optimized.energy_estimate),
            'throughput': safe_improvement(baseline.throughput, optimized.throughput, lower_is_better=False),
            'cache_efficiency': safe_improvement(baseline.cache_efficiency, optimized.cache_efficiency, lower_is_better=False),
            'memory_fragmentation': safe_improvement(baseline.memory_fragmentation, optimized.memory_fragmentation)
        }
    
    def _test_efficiency_significance(self,
                                    baseline_measurements: List[EfficiencyMetrics],
                                    optimized_measurements: List[EfficiencyMetrics]) -> Dict[str, Any]:
        """Test statistical significance of efficiency improvements."""
        
        # Extract computation times for significance testing
        baseline_times = [m.computation_time for m in baseline_measurements]
        optimized_times = [m.computation_time for m in optimized_measurements]
        
        # T-test for computation time improvement
        time_test = self.statistical_analyzer.two_sample_ttest(
            baseline_times, optimized_times
        )
        
        # Extract memory usage for testing
        baseline_memory = [m.memory_usage_average for m in baseline_measurements]
        optimized_memory = [m.memory_usage_average for m in optimized_measurements]
        
        memory_test = self.statistical_analyzer.two_sample_ttest(
            baseline_memory, optimized_memory
        )
        
        # Extract throughput for testing
        baseline_throughput = [m.throughput for m in baseline_measurements]
        optimized_throughput = [m.throughput for m in optimized_measurements]
        
        throughput_test = self.statistical_analyzer.two_sample_ttest(
            optimized_throughput, baseline_throughput  # Higher is better
        )
        
        return {
            'computation_time_test': time_test,
            'memory_usage_test': memory_test,
            'throughput_test': throughput_test,
            'sample_sizes': {
                'baseline': len(baseline_measurements),
                'optimized': len(optimized_measurements)
            }
        }
    
    def generate_efficiency_report(self,
                                 comparison: EfficiencyComparison,
                                 validation: Dict[str, Any],
                                 output_file: Optional[Path] = None) -> str:
        """Generate comprehensive efficiency analysis report."""
        
        report_lines = [
            "# Computational Efficiency Analysis Report\n\n",
            "## Target Metrics\n",
            f"- Target Efficiency Gain: {self.target_efficiency_gain:.1%} ± {self.target_efficiency_stddev:.1%}\n",
            f"- Confidence Interval: [{self.target_efficiency_gain - 1.96*self.target_efficiency_stddev:.1%}, "
            f"{self.target_efficiency_gain + 1.96*self.target_efficiency_stddev:.1%}]\n\n",
            
            "## Achieved Results\n",
            f"- Overall Efficiency Gain: {comparison.efficiency_gain:.3f} ({comparison.efficiency_gain:.1%})\n",
            f"- Meets Target: {'✓ YES' if validation['meets_target'] else '✗ NO'}\n",
            f"- Exceeds Minimum Threshold: {'✓ YES' if validation['exceeds_minimum'] else '✗ NO'}\n\n",
            
            "## Detailed Improvements\n"
        ]
        
        # Add detailed improvements
        for metric, improvement in comparison.relative_improvements.items():
            report_lines.append(f"- {metric.replace('_', ' ').title()}: {improvement:.1%}\n")
        
        report_lines.extend([
            "\n## Baseline vs Optimized Metrics\n",
            "| Metric | Baseline | Optimized | Improvement |\n",
            "|--------|----------|-----------|-------------|\n"
        ])
        
        # Add comparison table
        baseline = comparison.baseline_metrics
        optimized = comparison.optimized_metrics
        improvements = comparison.relative_improvements
        
        metrics_table = [
            ("Computation Time", f"{baseline.computation_time:.3f}s", f"{optimized.computation_time:.3f}s", f"{improvements['computation_time']:.1%}"),
            ("Memory Usage", f"{baseline.memory_usage_average:.2f}GB", f"{optimized.memory_usage_average:.2f}GB", f"{improvements['memory_usage']:.1%}"),
            ("CPU Usage", f"{baseline.cpu_usage_average:.1f}%", f"{optimized.cpu_usage_average:.1f}%", f"{improvements['cpu_usage']:.1%}"),
            ("GPU Usage", f"{baseline.gpu_usage_average:.1f}%", f"{optimized.gpu_usage_average:.1f}%", f"{improvements['gpu_usage']:.1%}"),
            ("Throughput", f"{baseline.throughput:.2f} ops/s", f"{optimized.throughput:.2f} ops/s", f"{improvements['throughput']:.1%}"),
            ("Energy Estimate", f"{baseline.energy_estimate:.1f}J", f"{optimized.energy_estimate:.1f}J", f"{improvements['energy_consumption']:.1%}")
        ]
        
        for metric, base_val, opt_val, improve in metrics_table:
            report_lines.append(f"| {metric} | {base_val} | {opt_val} | {improve} |\n")
        
        report_content = "".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content