"""
Statistical Validator - Comprehensive Statistical Validation
==========================================================

Provides rigorous statistical validation of meta-optimization framework performance
against target metrics with hypothesis testing, confidence intervals, and effect sizes.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import norm, t, chi2, f, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.statistical_analysis import StatisticalAnalyzer


class ValidationTest(Enum):
    """Types of statistical validation tests."""
    ONE_SAMPLE_TTEST = "one_sample_ttest"
    TWO_SAMPLE_TTEST = "two_sample_ttest"
    PAIRED_TTEST = "paired_ttest"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"


@dataclass
class ValidationResult:
    """Result of a statistical validation test."""
    test_type: ValidationTest
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    sample_size: int
    degrees_of_freedom: Optional[int]
    is_significant: bool
    interpretation: str
    raw_data: Dict[str, Any]


@dataclass
class ComprehensiveValidation:
    """Comprehensive validation results for all target metrics."""
    accuracy_validation: ValidationResult
    efficiency_validation: ValidationResult
    cognitive_validation: ValidationResult
    overall_validation: ValidationResult
    meets_all_targets: bool
    validation_summary: Dict[str, Any]


class StatisticalValidator:
    """
    Comprehensive statistical validator for meta-optimization framework.
    
    Provides rigorous statistical testing of performance improvements against
    target metrics with appropriate hypothesis testing, confidence intervals,
    and effect size calculations.
    """
    
    def __init__(self, 
                 statistical_analyzer: Optional[StatisticalAnalyzer] = None,
                 significance_level: float = 0.05,
                 power_threshold: float = 0.8):
        """
        Initialize statistical validator.
        
        Args:
            statistical_analyzer: Statistical analysis component
            significance_level: Statistical significance threshold
            power_threshold: Minimum required statistical power
        """
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self.logger = logging.getLogger(__name__)
        
        # Target metrics for validation
        self.targets = {
            'accuracy_improvement': 0.19,     # 19%
            'accuracy_stddev': 0.08,          # ±8%
            'efficiency_gain': 0.12,          # 12%
            'efficiency_stddev': 0.04,        # ±4%
            'cognitive_reduction': 0.22,      # 22%
            'cognitive_stddev': 0.05          # ±5%
        }
    
    def validate_accuracy_improvement(self,
                                    measured_improvements: List[float],
                                    baseline_accuracies: Optional[List[float]] = None) -> ValidationResult:
        """
        Validate accuracy improvement against target of 19% ± 8%.
        
        Args:
            measured_improvements: List of measured accuracy improvements
            baseline_accuracies: Optional baseline accuracies for paired testing
            
        Returns:
            Validation results for accuracy improvement
        """
        target_improvement = self.targets['accuracy_improvement']
        target_stddev = self.targets['accuracy_stddev']
        
        # Choose appropriate test
        if baseline_accuracies and len(baseline_accuracies) == len(measured_improvements):
            # Paired t-test
            test_result = self._paired_ttest(
                measured_improvements, baseline_accuracies, target_improvement
            )
        else:
            # One-sample t-test against target
            test_result = self._one_sample_ttest(
                measured_improvements, target_improvement
            )
        
        # Compute confidence interval
        confidence_interval = self.statistical_analyzer.compute_confidence_interval(
            measured_improvements, 1 - self.significance_level
        )
        
        # Compute effect size
        effect_size = self._compute_effect_size(measured_improvements, target_improvement)
        
        # Compute statistical power
        power = self._compute_power(
            measured_improvements, target_improvement, target_stddev
        )
        
        # Interpretation
        interpretation = self._interpret_accuracy_result(
            test_result, confidence_interval, target_improvement, target_stddev
        )
        
        return ValidationResult(
            test_type=ValidationTest.ONE_SAMPLE_TTEST if not baseline_accuracies else ValidationTest.PAIRED_TTEST,
            test_statistic=test_result['statistic'],
            p_value=test_result['p_value'],
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size=len(measured_improvements),
            degrees_of_freedom=len(measured_improvements) - 1,
            is_significant=test_result['p_value'] < self.significance_level,
            interpretation=interpretation,
            raw_data={
                'measurements': measured_improvements,
                'target': target_improvement,
                'target_stddev': target_stddev,
                'mean_improvement': np.mean(measured_improvements),
                'std_improvement': np.std(measured_improvements)
            }
        )
    
    def validate_efficiency_gain(self,
                                measured_gains: List[float],
                                baseline_efficiencies: Optional[List[float]] = None) -> ValidationResult:
        """
        Validate efficiency gain against target of 12% ± 4%.
        
        Args:
            measured_gains: List of measured efficiency gains
            baseline_efficiencies: Optional baseline efficiencies for paired testing
            
        Returns:
            Validation results for efficiency gain
        """
        target_gain = self.targets['efficiency_gain']
        target_stddev = self.targets['efficiency_stddev']
        
        # Choose appropriate test
        if baseline_efficiencies and len(baseline_efficiencies) == len(measured_gains):
            test_result = self._paired_ttest(
                measured_gains, baseline_efficiencies, target_gain
            )
        else:
            test_result = self._one_sample_ttest(
                measured_gains, target_gain
            )
        
        confidence_interval = self.statistical_analyzer.compute_confidence_interval(
            measured_gains, 1 - self.significance_level
        )
        
        effect_size = self._compute_effect_size(measured_gains, target_gain)
        power = self._compute_power(measured_gains, target_gain, target_stddev)
        
        interpretation = self._interpret_efficiency_result(
            test_result, confidence_interval, target_gain, target_stddev
        )
        
        return ValidationResult(
            test_type=ValidationTest.ONE_SAMPLE_TTEST if not baseline_efficiencies else ValidationTest.PAIRED_TTEST,
            test_statistic=test_result['statistic'],
            p_value=test_result['p_value'],
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size=len(measured_gains),
            degrees_of_freedom=len(measured_gains) - 1,
            is_significant=test_result['p_value'] < self.significance_level,
            interpretation=interpretation,
            raw_data={
                'measurements': measured_gains,
                'target': target_gain,
                'target_stddev': target_stddev,
                'mean_gain': np.mean(measured_gains),
                'std_gain': np.std(measured_gains)
            }
        )
    
    def validate_cognitive_reduction(self,
                                   measured_reductions: List[float],
                                   baseline_loads: Optional[List[float]] = None) -> ValidationResult:
        """
        Validate cognitive load reduction against target of 22% ± 5%.
        
        Args:
            measured_reductions: List of measured cognitive load reductions
            baseline_loads: Optional baseline cognitive loads for paired testing
            
        Returns:
            Validation results for cognitive load reduction
        """
        target_reduction = self.targets['cognitive_reduction']
        target_stddev = self.targets['cognitive_stddev']
        
        # Choose appropriate test
        if baseline_loads and len(baseline_loads) == len(measured_reductions):
            test_result = self._paired_ttest(
                measured_reductions, baseline_loads, target_reduction
            )
        else:
            test_result = self._one_sample_ttest(
                measured_reductions, target_reduction
            )
        
        confidence_interval = self.statistical_analyzer.compute_confidence_interval(
            measured_reductions, 1 - self.significance_level
        )
        
        effect_size = self._compute_effect_size(measured_reductions, target_reduction)
        power = self._compute_power(measured_reductions, target_reduction, target_stddev)
        
        interpretation = self._interpret_cognitive_result(
            test_result, confidence_interval, target_reduction, target_stddev
        )
        
        return ValidationResult(
            test_type=ValidationTest.ONE_SAMPLE_TTEST if not baseline_loads else ValidationTest.PAIRED_TTEST,
            test_statistic=test_result['statistic'],
            p_value=test_result['p_value'],
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size=len(measured_reductions),
            degrees_of_freedom=len(measured_reductions) - 1,
            is_significant=test_result['p_value'] < self.significance_level,
            interpretation=interpretation,
            raw_data={
                'measurements': measured_reductions,
                'target': target_reduction,
                'target_stddev': target_stddev,
                'mean_reduction': np.mean(measured_reductions),
                'std_reduction': np.std(measured_reductions)
            }
        )
    
    def comprehensive_validation(self,
                               accuracy_improvements: List[float],
                               efficiency_gains: List[float],
                               cognitive_reductions: List[float],
                               baseline_data: Optional[Dict[str, List[float]]] = None) -> ComprehensiveValidation:
        """
        Perform comprehensive validation across all target metrics.
        
        Args:
            accuracy_improvements: Measured accuracy improvements
            efficiency_gains: Measured efficiency gains
            cognitive_reductions: Measured cognitive load reductions
            baseline_data: Optional baseline data for paired testing
            
        Returns:
            Comprehensive validation results
        """
        # Extract baseline data if provided
        baseline_accuracies = baseline_data.get('accuracies') if baseline_data else None
        baseline_efficiencies = baseline_data.get('efficiencies') if baseline_data else None
        baseline_loads = baseline_data.get('cognitive_loads') if baseline_data else None
        
        # Individual validations
        accuracy_validation = self.validate_accuracy_improvement(
            accuracy_improvements, baseline_accuracies
        )
        efficiency_validation = self.validate_efficiency_gain(
            efficiency_gains, baseline_efficiencies
        )
        cognitive_validation = self.validate_cognitive_reduction(
            cognitive_reductions, baseline_loads
        )
        
        # Overall validation (multivariate test)
        overall_validation = self._multivariate_validation(
            accuracy_improvements, efficiency_gains, cognitive_reductions
        )
        
        # Check if all targets are met
        meets_all_targets = (
            accuracy_validation.is_significant and
            efficiency_validation.is_significant and
            cognitive_validation.is_significant and
            overall_validation.is_significant
        )
        
        # Summary statistics
        validation_summary = self._generate_validation_summary(
            accuracy_validation, efficiency_validation, 
            cognitive_validation, overall_validation
        )
        
        return ComprehensiveValidation(
            accuracy_validation=accuracy_validation,
            efficiency_validation=efficiency_validation,
            cognitive_validation=cognitive_validation,
            overall_validation=overall_validation,
            meets_all_targets=meets_all_targets,
            validation_summary=validation_summary
        )
    
    def bootstrap_validation(self,
                           data: List[float],
                           target_value: float,
                           n_bootstrap: int = 10000) -> ValidationResult:
        """
        Perform bootstrap validation for robust statistical inference.
        
        Args:
            data: Measurement data
            target_value: Target value to test against
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap validation results
        """
        # Bootstrap resampling
        bootstrap_means = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_means)
        bootstrap_std = np.std(bootstrap_means)
        
        # Bootstrap confidence interval
        alpha = self.significance_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        confidence_interval = (
            np.percentile(bootstrap_means, lower_percentile),
            np.percentile(bootstrap_means, upper_percentile)
        )
        
        # Bootstrap p-value (proportion of bootstrap means less than target)
        p_value = np.mean(bootstrap_means < target_value)
        p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
        
        # Effect size
        effect_size = (bootstrap_mean - target_value) / bootstrap_std
        
        # Interpretation
        interpretation = f"Bootstrap validation: {bootstrap_mean:.3f} vs target {target_value:.3f}"
        if p_value < self.significance_level:
            interpretation += " (statistically significant)"
        else:
            interpretation += " (not statistically significant)"
        
        return ValidationResult(
            test_type=ValidationTest.BOOTSTRAP,
            test_statistic=bootstrap_mean,
            p_value=p_value,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=1.0 - p_value if p_value > 0.5 else p_value,  # Approximation
            sample_size=len(data),
            degrees_of_freedom=None,
            is_significant=p_value < self.significance_level,
            interpretation=interpretation,
            raw_data={
                'original_data': data,
                'bootstrap_means': bootstrap_means.tolist(),
                'target': target_value,
                'n_bootstrap': n_bootstrap
            }
        )
    
    def _one_sample_ttest(self, data: List[float], target: float) -> Dict[str, float]:
        """Perform one-sample t-test."""
        statistic, p_value = stats.ttest_1samp(data, target)
        return {'statistic': statistic, 'p_value': p_value}
    
    def _paired_ttest(self, 
                     treatment: List[float], 
                     control: List[float], 
                     expected_difference: float) -> Dict[str, float]:
        """Perform paired t-test."""
        differences = np.array(treatment) - np.array(control)
        statistic, p_value = stats.ttest_1samp(differences, expected_difference)
        return {'statistic': statistic, 'p_value': p_value}
    
    def _multivariate_validation(self,
                                accuracy_improvements: List[float],
                                efficiency_gains: List[float],
                                cognitive_reductions: List[float]) -> ValidationResult:
        """Perform multivariate validation using Hotelling's T² test."""
        
        # Combine measurements into matrix
        data_matrix = np.column_stack([
            accuracy_improvements,
            efficiency_gains, 
            cognitive_reductions
        ])
        
        # Target vector
        target_vector = np.array([
            self.targets['accuracy_improvement'],
            self.targets['efficiency_gain'],
            self.targets['cognitive_reduction']
        ])
        
        # Sample statistics
        n_samples, n_variables = data_matrix.shape
        sample_mean = np.mean(data_matrix, axis=0)
        sample_cov = np.cov(data_matrix.T)
        
        # Hotelling's T² statistic
        mean_diff = sample_mean - target_vector
        try:
            t_squared = n_samples * mean_diff.T @ np.linalg.inv(sample_cov) @ mean_diff
        except np.linalg.LinAlgError:
            # Fallback if covariance matrix is singular
            t_squared = n_samples * np.sum(mean_diff**2)
        
        # Convert to F-statistic
        f_statistic = (n_samples - n_variables) / ((n_samples - 1) * n_variables) * t_squared
        df1 = n_variables
        df2 = n_samples - n_variables
        
        # P-value
        p_value = 1 - stats.f.cdf(f_statistic, df1, df2)
        
        # Effect size (multivariate Cohen's d approximation)
        effect_size = np.sqrt(t_squared / n_samples)
        
        # Confidence region (approximate)
        critical_value = stats.f.ppf(1 - self.significance_level, df1, df2)
        confidence_interval = (
            sample_mean - np.sqrt(critical_value * np.diag(sample_cov) / n_samples),
            sample_mean + np.sqrt(critical_value * np.diag(sample_cov) / n_samples)
        )
        
        interpretation = f"Multivariate test: F({df1},{df2}) = {f_statistic:.3f}, "
        interpretation += f"p = {p_value:.4f}"
        
        return ValidationResult(
            test_type=ValidationTest.ANOVA,  # Using ANOVA enum for multivariate
            test_statistic=f_statistic,
            p_value=p_value,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=1 - stats.f.cdf(critical_value, df1, df2),
            sample_size=n_samples,
            degrees_of_freedom=df2,
            is_significant=p_value < self.significance_level,
            interpretation=interpretation,
            raw_data={
                'data_matrix': data_matrix.tolist(),
                'target_vector': target_vector.tolist(),
                'sample_mean': sample_mean.tolist(),
                'sample_cov': sample_cov.tolist(),
                't_squared': t_squared,
                'df1': df1,
                'df2': df2
            }
        )
    
    def _compute_effect_size(self, data: List[float], target: float) -> float:
        """Compute Cohen's d effect size."""
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        
        if std_data == 0:
            return 0.0
        
        return (mean_data - target) / std_data
    
    def _compute_power(self, 
                      data: List[float], 
                      target: float, 
                      target_stddev: float) -> float:
        """Compute statistical power of the test."""
        n = len(data)
        effect_size = self._compute_effect_size(data, target)
        
        # Power calculation for one-sample t-test
        alpha = self.significance_level
        df = n - 1
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Critical value
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        # Power (probability of rejecting null when alternative is true)
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        return max(0.0, min(1.0, power))
    
    def _interpret_accuracy_result(self,
                                  test_result: Dict[str, float],
                                  confidence_interval: Tuple[float, float],
                                  target: float,
                                  target_stddev: float) -> str:
        """Interpret accuracy validation results."""
        p_value = test_result['p_value']
        ci_lower, ci_upper = confidence_interval
        
        interpretation = f"Accuracy improvement test: p = {p_value:.4f}\n"
        interpretation += f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Target: {target:.3f} ± {target_stddev:.3f}\n"
        
        if p_value < self.significance_level:
            if ci_lower > target - target_stddev:
                interpretation += "✓ Significantly achieves target accuracy improvement"
            else:
                interpretation += "✓ Significant but below target range"
        else:
            interpretation += "✗ No significant accuracy improvement detected"
        
        return interpretation
    
    def _interpret_efficiency_result(self,
                                   test_result: Dict[str, float],
                                   confidence_interval: Tuple[float, float],
                                   target: float,
                                   target_stddev: float) -> str:
        """Interpret efficiency validation results."""
        p_value = test_result['p_value']
        ci_lower, ci_upper = confidence_interval
        
        interpretation = f"Efficiency gain test: p = {p_value:.4f}\n"
        interpretation += f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Target: {target:.3f} ± {target_stddev:.3f}\n"
        
        if p_value < self.significance_level:
            if ci_lower > target - target_stddev:
                interpretation += "✓ Significantly achieves target efficiency gain"
            else:
                interpretation += "✓ Significant but below target range"
        else:
            interpretation += "✗ No significant efficiency gain detected"
        
        return interpretation
    
    def _interpret_cognitive_result(self,
                                  test_result: Dict[str, float],
                                  confidence_interval: Tuple[float, float],
                                  target: float,
                                  target_stddev: float) -> str:
        """Interpret cognitive load validation results."""
        p_value = test_result['p_value']
        ci_lower, ci_upper = confidence_interval
        
        interpretation = f"Cognitive load reduction test: p = {p_value:.4f}\n"
        interpretation += f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Target: {target:.3f} ± {target_stddev:.3f}\n"
        
        if p_value < self.significance_level:
            if ci_lower > target - target_stddev:
                interpretation += "✓ Significantly achieves target cognitive load reduction"
            else:
                interpretation += "✓ Significant but below target range"
        else:
            interpretation += "✗ No significant cognitive load reduction detected"
        
        return interpretation
    
    def _generate_validation_summary(self,
                                   accuracy_result: ValidationResult,
                                   efficiency_result: ValidationResult,
                                   cognitive_result: ValidationResult,
                                   overall_result: ValidationResult) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        
        return {
            'individual_results': {
                'accuracy': {
                    'significant': accuracy_result.is_significant,
                    'p_value': accuracy_result.p_value,
                    'effect_size': accuracy_result.effect_size,
                    'power': accuracy_result.power
                },
                'efficiency': {
                    'significant': efficiency_result.is_significant,
                    'p_value': efficiency_result.p_value,
                    'effect_size': efficiency_result.effect_size,
                    'power': efficiency_result.power
                },
                'cognitive': {
                    'significant': cognitive_result.is_significant,
                    'p_value': cognitive_result.p_value,
                    'effect_size': cognitive_result.effect_size,
                    'power': cognitive_result.power
                }
            },
            'overall_result': {
                'significant': overall_result.is_significant,
                'p_value': overall_result.p_value,
                'effect_size': overall_result.effect_size,
                'power': overall_result.power
            },
            'power_analysis': {
                'all_adequate_power': all([
                    accuracy_result.power >= self.power_threshold,
                    efficiency_result.power >= self.power_threshold,
                    cognitive_result.power >= self.power_threshold
                ]),
                'mean_power': np.mean([
                    accuracy_result.power,
                    efficiency_result.power,
                    cognitive_result.power
                ])
            },
            'effect_sizes': {
                'accuracy': self._interpret_effect_size(accuracy_result.effect_size),
                'efficiency': self._interpret_effect_size(efficiency_result.effect_size),
                'cognitive': self._interpret_effect_size(cognitive_result.effect_size)
            }
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_validation_report(self,
                                 validation: ComprehensiveValidation,
                                 output_file: Optional[Path] = None) -> str:
        """Generate comprehensive validation report."""
        
        report_lines = [
            "# Statistical Validation Report\n",
            "## Meta-Optimization Framework Performance Validation\n\n",
            "### Target Metrics\n",
            f"- Accuracy Improvement: {self.targets['accuracy_improvement']:.1%} ± {self.targets['accuracy_stddev']:.1%}\n",
            f"- Efficiency Gain: {self.targets['efficiency_gain']:.1%} ± {self.targets['efficiency_stddev']:.1%}\n",
            f"- Cognitive Load Reduction: {self.targets['cognitive_reduction']:.1%} ± {self.targets['cognitive_stddev']:.1%}\n\n",
            
            "### Overall Validation Status\n",
            f"**All Targets Met: {'✓ YES' if validation.meets_all_targets else '✗ NO'}**\n\n",
            
            "### Individual Metric Validation\n\n"
        ]
        
        # Individual results
        metrics = [
            ("Accuracy Improvement", validation.accuracy_validation),
            ("Efficiency Gain", validation.efficiency_validation),
            ("Cognitive Load Reduction", validation.cognitive_validation)
        ]
        
        for metric_name, result in metrics:
            status = "✓ SIGNIFICANT" if result.is_significant else "✗ NOT SIGNIFICANT"
            report_lines.extend([
                f"#### {metric_name}\n",
                f"- Status: {status}\n",
                f"- p-value: {result.p_value:.4f}\n",
                f"- Effect size: {result.effect_size:.3f} ({self._interpret_effect_size(result.effect_size)})\n",
                f"- Statistical power: {result.power:.3f}\n",
                f"- 95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]\n",
                f"- Sample size: {result.sample_size}\n\n"
            ])
        
        # Overall multivariate result
        report_lines.extend([
            "### Multivariate Validation\n",
            f"- Combined test p-value: {validation.overall_validation.p_value:.4f}\n",
            f"- Multivariate effect size: {validation.overall_validation.effect_size:.3f}\n",
            f"- Status: {'✓ SIGNIFICANT' if validation.overall_validation.is_significant else '✗ NOT SIGNIFICANT'}\n\n"
        ])
        
        # Power analysis
        power_summary = validation.validation_summary['power_analysis']
        report_lines.extend([
            "### Power Analysis\n",
            f"- Mean statistical power: {power_summary['mean_power']:.3f}\n",
            f"- Adequate power (≥{self.power_threshold}): {'✓ YES' if power_summary['all_adequate_power'] else '✗ NO'}\n\n"
        ])
        
        # Conclusions
        report_lines.extend([
            "### Conclusions\n"
        ])
        
        if validation.meets_all_targets:
            report_lines.append(
                "The meta-optimization framework successfully achieves all target performance metrics "
                "with statistical significance. The results provide strong evidence for the "
                "effectiveness of the cognitive-computational optimization approach.\n\n"
            )
        else:
            report_lines.append(
                "The meta-optimization framework shows mixed results against target metrics. "
                "Further optimization or larger sample sizes may be needed to achieve "
                "all performance targets.\n\n"
            )
        
        report_content = "".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content