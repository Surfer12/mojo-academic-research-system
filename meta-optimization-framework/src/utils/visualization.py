"""
Visualization Utilities

Provides visualization tools for the meta-optimization framework,
including plots for performance analysis, parameter evolution, and
cognitive authenticity assessment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Main visualization class for the meta-optimization framework.
    
    Provides static and interactive visualizations for optimization results,
    parameter evolution, and cognitive authenticity analysis.
    """
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_convergence(self, 
                        performance_history: List[float],
                        title: str = "Optimization Convergence",
                        save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence curve.
        
        Args:
            performance_history: List of performance values over iterations
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        iterations = range(len(performance_history))
        ax.plot(iterations, performance_history, linewidth=2, marker='o', markersize=4)
        
        # Add trend line
        if len(performance_history) > 1:
            z = np.polyfit(iterations, performance_history, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), "--", alpha=0.7, color='red', 
                   label=f'Trend (slope: {z[0]:.4f})')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance statistics
        if performance_history:
            final_perf = performance_history[-1]
            max_perf = max(performance_history)
            improvement = max_perf - performance_history[0] if len(performance_history) > 1 else 0
            
            stats_text = f'Final: {final_perf:.4f}\nMax: {max_perf:.4f}\nImprovement: {improvement:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_evolution(self,
                               alpha_history: List[float],
                               lambda_history: List[Tuple[float, float]],
                               beta_history: List[float],
                               title: str = "Parameter Evolution",
                               save_path: Optional[str] = None) -> None:
        """
        Plot evolution of optimization parameters.
        
        Args:
            alpha_history: History of α parameter
            lambda_history: History of (λ₁, λ₂) parameters
            beta_history: History of β parameter
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # α parameter
        if alpha_history:
            axes[0, 0].plot(alpha_history, label='α (integration)', color='blue', linewidth=2)
            axes[0, 0].set_title('Integration Parameter (α)')
            axes[0, 0].set_ylabel('α value')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
        
        # λ parameters
        if lambda_history:
            lambda1_history = [l[0] for l in lambda_history]
            lambda2_history = [l[1] for l in lambda_history]
            
            axes[0, 1].plot(lambda1_history, label='λ₁ (cognitive)', color='green', linewidth=2)
            axes[0, 1].plot(lambda2_history, label='λ₂ (efficiency)', color='orange', linewidth=2)
            axes[0, 1].set_title('Regularization Parameters (λ)')
            axes[0, 1].set_ylabel('λ value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # β parameter
        if beta_history:
            axes[1, 0].plot(beta_history, label='β (bias)', color='red', linewidth=2)
            axes[1, 0].set_title('Bias Parameter (β)')
            axes[1, 0].set_ylabel('β value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view
        axes[1, 1].set_title('All Parameters (Normalized)')
        if alpha_history:
            axes[1, 1].plot(np.array(alpha_history), label='α', alpha=0.7)
        if lambda_history:
            lambda1_norm = np.array([l[0] for l in lambda_history])
            lambda2_norm = np.array([l[1] for l in lambda_history])
            # Normalize lambda values to [0, 1] for comparison
            if lambda1_norm.max() > 0:
                lambda1_norm = lambda1_norm / lambda1_norm.max()
            if lambda2_norm.max() > 0:
                lambda2_norm = lambda2_norm / lambda2_norm.max()
            axes[1, 1].plot(lambda1_norm, label='λ₁ (norm)', alpha=0.7)
            axes[1, 1].plot(lambda2_norm, label='λ₂ (norm)', alpha=0.7)
        if beta_history:
            beta_norm = np.array(beta_history)
            if beta_norm.max() > 0:
                beta_norm = beta_norm / beta_norm.max()
            axes[1, 1].plot(beta_norm, label='β (norm)', alpha=0.7)
        
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Normalized Value')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter evolution plot saved to {save_path}")
        
        plt.show()
    
    def plot_component_analysis(self,
                              symbolic_contributions: List[float],
                              neural_contributions: List[float],
                              title: str = "Component Contributions",
                              save_path: Optional[str] = None) -> None:
        """
        Plot analysis of symbolic vs neural component contributions.
        
        Args:
            symbolic_contributions: Symbolic component contribution history
            neural_contributions: Neural component contribution history
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Contribution over time
        axes[0, 0].plot(symbolic_contributions, label='Symbolic', linewidth=2, alpha=0.8)
        axes[0, 0].plot(neural_contributions, label='Neural', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Component Contributions Over Time')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Contribution Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Contribution ratio
        if symbolic_contributions and neural_contributions:
            total_contributions = np.array(symbolic_contributions) + np.array(neural_contributions)
            symbolic_ratio = np.array(symbolic_contributions) / (total_contributions + 1e-8)
            
            axes[0, 1].plot(symbolic_ratio, linewidth=2, color='purple')
            axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Balanced')
            axes[0, 1].set_title('Symbolic Dominance Ratio')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Symbolic Ratio')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution comparison
        axes[1, 0].hist(symbolic_contributions, alpha=0.7, label='Symbolic', bins=20)
        axes[1, 0].hist(neural_contributions, alpha=0.7, label='Neural', bins=20)
        axes[1, 0].set_title('Contribution Distributions')
        axes[1, 0].set_xlabel('Contribution Magnitude')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1, 1].scatter(symbolic_contributions, neural_contributions, alpha=0.6)
        axes[1, 1].set_title('Symbolic vs Neural Contributions')
        axes[1, 1].set_xlabel('Symbolic Contribution')
        axes[1, 1].set_ylabel('Neural Contribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        if len(symbolic_contributions) > 1:
            correlation = np.corrcoef(symbolic_contributions, neural_contributions)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 1].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Component analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_cognitive_authenticity(self,
                                  authenticity_history: List[float],
                                  constraint_violations: Dict[str, List[float]],
                                  title: str = "Cognitive Authenticity Analysis",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot cognitive authenticity metrics and constraint violations.
        
        Args:
            authenticity_history: History of overall authenticity scores
            constraint_violations: History of individual constraint violations
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall authenticity
        axes[0, 0].plot(authenticity_history, linewidth=2, color='green')
        axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
        axes[0, 0].set_title('Overall Cognitive Authenticity')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Authenticity Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Individual constraint violations
        for constraint_name, violations in constraint_violations.items():
            if violations:
                axes[0, 1].plot(violations, label=constraint_name, linewidth=2, alpha=0.8)
        
        axes[0, 1].set_title('Constraint Violations')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Violation Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Authenticity distribution
        axes[1, 0].hist(authenticity_history, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[1, 0].set_title('Authenticity Distribution')
        axes[1, 0].set_xlabel('Authenticity Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Constraint violation summary
        if constraint_violations:
            constraint_names = list(constraint_violations.keys())
            mean_violations = [np.mean(violations) if violations else 0 
                             for violations in constraint_violations.values()]
            
            bars = axes[1, 1].bar(constraint_names, mean_violations, alpha=0.7)
            axes[1, 1].set_title('Mean Constraint Violations')
            axes[1, 1].set_ylabel('Mean Violation')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Color bars based on violation severity
            for bar, violation in zip(bars, mean_violations):
                if violation > 0.5:
                    bar.set_color('red')
                elif violation > 0.2:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cognitive authenticity plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self,
                                   optimization_data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard with all optimization metrics.
        
        Args:
            optimization_data: Dictionary containing all optimization data
            save_path: Path to save HTML dashboard (optional)
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Performance Convergence', 'Parameter Evolution',
                          'Component Contributions', 'Cognitive Authenticity',
                          'Bias Effects', 'Statistical Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Performance convergence
        if 'performance_history' in optimization_data:
            fig.add_trace(
                go.Scatter(
                    y=optimization_data['performance_history'],
                    mode='lines+markers',
                    name='Performance',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Parameter evolution
        if 'alpha_history' in optimization_data:
            fig.add_trace(
                go.Scatter(
                    y=optimization_data['alpha_history'],
                    mode='lines',
                    name='α (integration)',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
        
        if 'lambda_history' in optimization_data:
            lambda1_history = [l[0] for l in optimization_data['lambda_history']]
            lambda2_history = [l[1] for l in optimization_data['lambda_history']]
            
            fig.add_trace(
                go.Scatter(
                    y=lambda1_history,
                    mode='lines',
                    name='λ₁ (cognitive)',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=2, secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    y=lambda2_history,
                    mode='lines',
                    name='λ₂ (efficiency)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2, secondary_y=True
            )
        
        # Component contributions
        if 'symbolic_contributions' in optimization_data and 'neural_contributions' in optimization_data:
            fig.add_trace(
                go.Scatter(
                    y=optimization_data['symbolic_contributions'],
                    mode='lines',
                    name='Symbolic',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    y=optimization_data['neural_contributions'],
                    mode='lines',
                    name='Neural',
                    line=dict(color='cyan', width=2)
                ),
                row=2, col=1
            )
        
        # Cognitive authenticity
        if 'authenticity_history' in optimization_data:
            fig.add_trace(
                go.Scatter(
                    y=optimization_data['authenticity_history'],
                    mode='lines+markers',
                    name='Authenticity',
                    line=dict(color='green', width=2)
                ),
                row=2, col=2
            )
            
            # Add target line
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="red",
                annotation_text="Target (0.8)",
                row=2, col=2
            )
        
        # Bias effects
        if 'bias_history' in optimization_data:
            bias_effects = [b.get('kl_divergence', 0) for b in optimization_data['bias_history']]
            fig.add_trace(
                go.Scatter(
                    y=bias_effects,
                    mode='lines',
                    name='Bias Effect (KL)',
                    line=dict(color='magenta', width=2)
                ),
                row=3, col=1
            )
        
        # Statistical summary table
        if 'summary_stats' in optimization_data:
            stats = optimization_data['summary_stats']
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=[
                        list(stats.keys()),
                        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in stats.values()]
                    ])
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Meta-Optimization Interactive Dashboard",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Iteration", row=3, col=1)
        fig.update_yaxes(title_text="Performance", row=1, col=1)
        fig.update_yaxes(title_text="Parameter Value", row=1, col=2)
        fig.update_yaxes(title_text="Contribution", row=2, col=1)
        fig.update_yaxes(title_text="Authenticity", row=2, col=2)
        fig.update_yaxes(title_text="Bias Effect", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def plot_statistical_validation(self,
                                   validation_results: Dict[str, Any],
                                   title: str = "Statistical Validation",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot statistical validation results against target metrics.
        
        Args:
            validation_results: Results from statistical validation
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target vs achieved metrics
        metrics = ['accuracy', 'efficiency', 'cognitive_load']
        targets = [19, 12, 22]  # Target percentages
        achieved = [validation_results.get(f'{metric}_mean', 0) for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x_pos - width/2, targets, width, label='Target', alpha=0.7)
        bars2 = axes[0, 0].bar(x_pos + width/2, achieved, width, label='Achieved', alpha=0.7)
        
        axes[0, 0].set_title('Target vs Achieved Performance')
        axes[0, 0].set_ylabel('Improvement (%)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars based on target achievement
        for bar, target, actual in zip(bars2, targets, achieved):
            if actual >= target * 0.8:  # Within 80% of target
                bar.set_color('green')
            elif actual >= target * 0.5:  # Within 50% of target
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Confidence intervals
        for i, metric in enumerate(metrics):
            ci = validation_results.get(f'{metric}_ci')
            if ci:
                axes[0, 1].errorbar(i, achieved[i], 
                                  yerr=[[achieved[i] - ci.lower], [ci.upper - achieved[i]]], 
                                  fmt='o', capsize=5, capthick=2)
        
        axes[0, 1].set_title('Confidence Intervals')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_xticks(range(len(metrics)))
        axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Effect sizes
        effect_sizes = [validation_results.get(f'{metric}_effect_size', 0) for metric in metrics]
        colors = ['green' if es > 0.8 else 'orange' if es > 0.5 else 'red' for es in effect_sizes]
        
        bars = axes[1, 0].bar(metrics, effect_sizes, color=colors, alpha=0.7)
        axes[1, 0].set_title('Effect Sizes (Cohen\'s d)')
        axes[1, 0].set_ylabel('Effect Size')
        axes[1, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add effect size interpretation lines
        axes[1, 0].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        axes[1, 0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
        axes[1, 0].legend()
        
        # Statistical significance
        p_values = [validation_results.get(f'{metric}_p_value', 1.0) for metric in metrics]
        significance = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                       for p in p_values]
        
        axes[1, 1].bar(metrics, [-np.log10(p) for p in p_values], alpha=0.7)
        axes[1, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        axes[1, 1].axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
        axes[1, 1].set_title('Statistical Significance (-log10 p-value)')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add significance annotations
        for i, sig in enumerate(significance):
            axes[1, 1].text(i, -np.log10(p_values[i]) + 0.1, sig, ha='center', fontweight='bold')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistical validation plot saved to {save_path}")
        
        plt.show()