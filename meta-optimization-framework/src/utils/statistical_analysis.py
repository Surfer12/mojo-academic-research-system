"""
Statistical Analysis Utilities

Provides statistical analysis tools for the meta-optimization framework,
including confidence intervals, effect sizes, and hypothesis testing.

Target Performance Metrics:
- 19% ± 8% accuracy improvement (95% CI: [11%, 27%])
- 12% ± 4% computational efficiency gains (95% CI: [8%, 16%])
- 22% ± 5% cognitive load reduction
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval representation."""
    lower: float
    upper: float
    confidence: float = 0.95
    
    def __str__(self) -> str:
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.confidence:.0%} CI)"
    
    def contains(self, value: float) -> bool:
        """Check if value is within confidence interval."""
        return self.lower <= value <= self.upper
    
    def width(self) -> float:
        """Get width of confidence interval."""
        return self.upper - self.lower


@dataclass
class EffectSize:
    """Effect size representation."""
    cohens_d: float
    interpretation: str
    
    def __str__(self) -> str:
        return f"Cohen's d = {self.cohens_d:.3f} ({self.interpretation})"


@dataclass
class HypothesisTest:
    """Hypothesis test result."""
    statistic: float
    p_value: float
    significant: bool
    test_name: str
    
    def __str__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return f"{self.test_name}: {self.statistic:.3f}, p = {self.p_value:.4f} ({sig_str})"


class StatisticalAnalyzer:
    """
    Main statistical analysis class for the meta-optimization framework.
    
    Provides methods for computing confidence intervals, effect sizes,
    and conducting hypothesis tests on optimization results.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        self.confidence_level = 1.0 - alpha
    
    def compute_confidence_interval(self, 
                                  data: Union[List[float], np.ndarray],
                                  confidence: float = 0.95,
                                  method: str = "t") -> ConfidenceInterval:
        """
        Compute confidence interval for data.
        
        Args:
            data: Sample data
            confidence: Confidence level (default: 0.95)
            method: Method to use ("t", "bootstrap", "normal")
            
        Returns:
            ConfidenceInterval object
        """
        data = np.array(data)
        n = len(data)
        
        if n < 2:
            logger.warning("Insufficient data for confidence interval")
            return ConfidenceInterval(0.0, 0.0, confidence)
        
        mean = np.mean(data)
        
        if method == "t":
            # Student's t-distribution (recommended for small samples)
            std_err = stats.sem(data)
            t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
            margin = t_critical * std_err
            
        elif method == "bootstrap":
            # Bootstrap confidence interval
            return self._bootstrap_confidence_interval(data, confidence)
            
        elif method == "normal":
            # Normal approximation (for large samples)
            std_err = np.std(data, ddof=1) / np.sqrt(n)
            z_critical = stats.norm.ppf((1 + confidence) / 2)
            margin = z_critical * std_err
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            confidence=confidence
        )
    
    def _bootstrap_confidence_interval(self, 
                                     data: np.ndarray,
                                     confidence: float,
                                     n_bootstrap: int = 10000) -> ConfidenceInterval:
        """Compute bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(lower, upper, confidence)
    
    def compute_effect_size(self, 
                          treatment: Union[List[float], np.ndarray],
                          control: Optional[Union[List[float], np.ndarray]] = None,
                          baseline: Optional[float] = None) -> EffectSize:
        """
        Compute Cohen's d effect size.
        
        Args:
            treatment: Treatment group data
            control: Control group data (optional)
            baseline: Baseline value for single-group comparison
            
        Returns:
            EffectSize object
        """
        treatment = np.array(treatment)
        
        if control is not None:
            # Two-group comparison
            control = np.array(control)
            mean_diff = np.mean(treatment) - np.mean(control)
            pooled_std = np.sqrt(
                ((len(treatment) - 1) * np.var(treatment, ddof=1) +
                 (len(control) - 1) * np.var(control, ddof=1)) /
                (len(treatment) + len(control) - 2)
            )
            cohens_d = mean_diff / pooled_std
            
        elif baseline is not None:
            # Single-group comparison to baseline
            mean_diff = np.mean(treatment) - baseline
            std_treatment = np.std(treatment, ddof=1)
            cohens_d = mean_diff / std_treatment
            
        else:
            # Single-group comparison to zero
            cohens_d = np.mean(treatment) / np.std(treatment, ddof=1)
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return EffectSize(cohens_d, interpretation)
    
    def t_test(self, 
              group1: Union[List[float], np.ndarray],
              group2: Optional[Union[List[float], np.ndarray]] = None,
              mu: float = 0.0,
              alternative: str = "two-sided") -> HypothesisTest:
        """
        Perform t-test.
        
        Args:
            group1: First group data
            group2: Second group data (for two-sample test)
            mu: Hypothesized mean (for one-sample test)
            alternative: Alternative hypothesis ("two-sided", "greater", "less")
            
        Returns:
            HypothesisTest result
        """
        group1 = np.array(group1)
        
        if group2 is not None:
            # Two-sample t-test
            group2 = np.array(group2)
            statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
            test_name = "Two-sample t-test"
        else:
            # One-sample t-test
            statistic, p_value = stats.ttest_1samp(group1, mu, alternative=alternative)
            test_name = "One-sample t-test"
        
        significant = p_value < self.alpha
        
        return HypothesisTest(statistic, p_value, significant, test_name)
    
    def paired_t_test(self, 
                     before: Union[List[float], np.ndarray],
                     after: Union[List[float], np.ndarray],
                     alternative: str = "two-sided") -> HypothesisTest:
        """
        Perform paired t-test.
        
        Args:
            before: Before treatment measurements
            after: After treatment measurements
            alternative: Alternative hypothesis
            
        Returns:
            HypothesisTest result
        """
        before = np.array(before)
        after = np.array(after)
        
        statistic, p_value = stats.ttest_rel(before, after, alternative=alternative)
        significant = p_value < self.alpha
        
        return HypothesisTest(statistic, p_value, significant, "Paired t-test")
    
    def wilcoxon_test(self, 
                     group1: Union[List[float], np.ndarray],
                     group2: Optional[Union[List[float], np.ndarray]] = None,
                     alternative: str = "two-sided") -> HypothesisTest:
        """
        Perform Wilcoxon test (non-parametric alternative to t-test).
        
        Args:
            group1: First group data
            group2: Second group data (for rank-sum test)
            alternative: Alternative hypothesis
            
        Returns:
            HypothesisTest result
        """
        group1 = np.array(group1)
        
        if group2 is not None:
            # Wilcoxon rank-sum test (Mann-Whitney U)
            group2 = np.array(group2)
            statistic, p_value = stats.mannwhitneyu(
                group1, group2, alternative=alternative
            )
            test_name = "Wilcoxon rank-sum test"
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(group1, alternative=alternative)
            test_name = "Wilcoxon signed-rank test"
        
        significant = p_value < self.alpha
        
        return HypothesisTest(statistic, p_value, significant, test_name)
    
    def power_analysis(self, 
                      effect_size: float,
                      sample_size: int,
                      alpha: float = 0.05,
                      test_type: str = "two-sided") -> float:
        """
        Compute statistical power for given parameters.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size
            alpha: Significance level
            test_type: Type of test
            
        Returns:
            Statistical power (0-1)
        """
        # Critical value for given alpha
        if test_type == "two-sided":
            critical_value = stats.norm.ppf(1 - alpha/2)
        else:
            critical_value = stats.norm.ppf(1 - alpha)
        
        # Standard error
        std_error = 1 / np.sqrt(sample_size)
        
        # Non-centrality parameter
        ncp = effect_size / std_error
        
        # Power calculation
        if test_type == "two-sided":
            power = (1 - stats.norm.cdf(critical_value - ncp) + 
                    stats.norm.cdf(-critical_value - ncp))
        else:
            power = 1 - stats.norm.cdf(critical_value - ncp)
        
        return power
    
    def sample_size_calculation(self, 
                              effect_size: float,
                              power: float = 0.8,
                              alpha: float = 0.05,
                              test_type: str = "two-sided") -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            test_type: Type of test
            
        Returns:
            Required sample size
        """
        # Critical values
        if test_type == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def analyze_optimization_results(self, 
                                   performance_history: List[float],
                                   baseline_performance: float = 0.5) -> Dict[str, Union[float, ConfidenceInterval, EffectSize, HypothesisTest]]:
        """
        Comprehensive analysis of optimization results.
        
        Args:
            performance_history: History of performance values
            baseline_performance: Baseline performance for comparison
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if not performance_history:
            return {"error": "No performance data provided"}
        
        performance_array = np.array(performance_history)
        
        # Basic statistics
        mean_performance = np.mean(performance_array)
        std_performance = np.std(performance_array, ddof=1)
        improvement = mean_performance - baseline_performance
        improvement_percent = (improvement / baseline_performance) * 100
        
        # Confidence interval
        ci = self.compute_confidence_interval(performance_array)
        
        # Effect size
        effect_size = self.compute_effect_size(performance_array, baseline=baseline_performance)
        
        # Hypothesis test (improvement over baseline)
        t_test_result = self.t_test(performance_array, mu=baseline_performance, alternative="greater")
        
        # Trend analysis
        if len(performance_history) > 2:
            trend_slope, _ = np.polyfit(range(len(performance_history)), performance_history, 1)
        else:
            trend_slope = 0.0
        
        # Stability analysis
        stability = 1.0 - (std_performance / mean_performance) if mean_performance > 0 else 0.0
        
        return {
            "mean_performance": mean_performance,
            "std_performance": std_performance,
            "improvement": improvement,
            "improvement_percent": improvement_percent,
            "confidence_interval": ci,
            "effect_size": effect_size,
            "significance_test": t_test_result,
            "trend_slope": trend_slope,
            "stability": stability,
            "sample_size": len(performance_history)
        }
    
    def validate_target_metrics(self, 
                              accuracy_improvements: List[float],
                              efficiency_gains: List[float],
                              cognitive_load_reductions: List[float]) -> Dict[str, bool]:
        """
        Validate against target performance metrics:
        - 19% ± 8% accuracy improvement (95% CI: [11%, 27%])
        - 12% ± 4% efficiency gains (95% CI: [8%, 16%])
        - 22% ± 5% cognitive load reduction
        
        Args:
            accuracy_improvements: List of accuracy improvement percentages
            efficiency_gains: List of efficiency gain percentages
            cognitive_load_reductions: List of cognitive load reduction percentages
            
        Returns:
            Dictionary indicating whether each target is met
        """
        results = {}
        
        # Target ranges
        targets = {
            "accuracy": {"mean": 19.0, "std": 8.0, "ci_lower": 11.0, "ci_upper": 27.0},
            "efficiency": {"mean": 12.0, "std": 4.0, "ci_lower": 8.0, "ci_upper": 16.0},
            "cognitive_load": {"mean": 22.0, "std": 5.0, "ci_lower": 17.0, "ci_upper": 27.0}
        }
        
        datasets = {
            "accuracy": accuracy_improvements,
            "efficiency": efficiency_gains,
            "cognitive_load": cognitive_load_reductions
        }
        
        for metric_name, data in datasets.items():
            if not data:
                results[f"{metric_name}_target_met"] = False
                continue
                
            data_array = np.array(data)
            mean_value = np.mean(data_array)
            ci = self.compute_confidence_interval(data_array)
            
            target = targets[metric_name]
            
            # Check if mean is within expected range
            mean_in_range = (target["mean"] - target["std"]) <= mean_value <= (target["mean"] + target["std"])
            
            # Check if confidence interval overlaps with target CI
            ci_overlap = not (ci.upper < target["ci_lower"] or ci.lower > target["ci_upper"])
            
            results[f"{metric_name}_target_met"] = mean_in_range and ci_overlap
            results[f"{metric_name}_mean"] = mean_value
            results[f"{metric_name}_ci"] = ci
        
        return results
    
    def meta_analysis(self, 
                     studies: List[Dict[str, Union[float, int]]]) -> Dict[str, float]:
        """
        Perform meta-analysis across multiple studies/experiments.
        
        Args:
            studies: List of study results with 'mean', 'std', 'n' keys
            
        Returns:
            Meta-analysis results
        """
        if not studies:
            return {"error": "No studies provided"}
        
        # Extract data
        means = np.array([study["mean"] for study in studies])
        stds = np.array([study["std"] for study in studies])
        ns = np.array([study["n"] for study in studies])
        
        # Compute weights (inverse variance)
        variances = stds ** 2
        weights = ns / variances
        total_weight = np.sum(weights)
        
        # Weighted mean
        pooled_mean = np.sum(weights * means) / total_weight
        
        # Pooled standard error
        pooled_se = np.sqrt(1 / total_weight)
        
        # Heterogeneity test (Q statistic)
        q_statistic = np.sum(weights * (means - pooled_mean) ** 2)
        df = len(studies) - 1
        p_heterogeneity = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
        
        # I² statistic (percentage of variation due to heterogeneity)
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
        
        return {
            "pooled_mean": pooled_mean,
            "pooled_se": pooled_se,
            "q_statistic": q_statistic,
            "p_heterogeneity": p_heterogeneity,
            "i_squared": i_squared,
            "num_studies": len(studies),
            "total_n": int(np.sum(ns))
        }