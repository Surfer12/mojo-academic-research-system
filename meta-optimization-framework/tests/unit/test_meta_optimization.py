"""
Unit tests for the core meta-optimization framework.

Tests the mathematical correctness and functionality of the main
MetaOptimizer class and its components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.core.meta_optimization import (
    MetaOptimizer, 
    TaskSpecification, 
    OptimizationResult,
    CognitiveSymbolicReasoner,
    CognitiveNeuralProcessor
)
from src.utils.statistical_analysis import ConfidenceInterval, EffectSize


class TestTaskSpecification:
    """Test TaskSpecification dataclass."""
    
    def test_task_specification_creation(self):
        """Test creation of TaskSpecification."""
        input_data = torch.randn(32, 10, 64)
        target_output = torch.randn(32, 10, 64)
        
        task_spec = TaskSpecification(
            input_data=input_data,
            target_output=target_output,
            task_type="n_back",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
        
        assert task_spec.task_type == "n_back"
        assert task_spec.cognitive_constraints["authenticity"] == 0.8
        assert torch.equal(task_spec.input_data, input_data)
        assert torch.equal(task_spec.target_output, target_output)


class TestCognitiveSymbolicReasoner:
    """Test CognitiveSymbolicReasoner implementation."""
    
    @pytest.fixture
    def task_spec(self):
        """Create task specification for testing."""
        return TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="n_back",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
    
    def test_initialization(self, task_spec):
        """Test proper initialization."""
        reasoner = CognitiveSymbolicReasoner(task_spec)
        assert reasoner.task_spec == task_spec
        assert "n_back" in reasoner.reasoning_rules
    
    def test_n_back_processing(self, task_spec):
        """Test N-back task processing."""
        reasoner = CognitiveSymbolicReasoner(task_spec)
        output = reasoner.process(task_spec.input_data)
        
        # Check output shape matches input
        assert output.shape == task_spec.input_data.shape
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_stroop_processing(self):
        """Test Stroop task processing."""
        task_spec = TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="stroop",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
        
        reasoner = CognitiveSymbolicReasoner(task_spec)
        output = reasoner.process(task_spec.input_data)
        
        assert output.shape == task_spec.input_data.shape
        assert torch.isfinite(output).all()
    
    def test_unknown_task_type(self):
        """Test handling of unknown task types."""
        task_spec = TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="unknown_task",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
        
        reasoner = CognitiveSymbolicReasoner(task_spec)
        output = reasoner.process(task_spec.input_data)
        
        # Should fall back to default processing
        assert output.shape == task_spec.input_data.shape
        assert torch.isfinite(output).all()


class TestCognitiveNeuralProcessor:
    """Test CognitiveNeuralProcessor implementation."""
    
    @pytest.fixture
    def task_spec(self):
        """Create task specification for testing."""
        return TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="n_back",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
    
    def test_initialization(self, task_spec):
        """Test proper initialization."""
        processor = CognitiveNeuralProcessor(task_spec)
        assert processor.task_spec == task_spec
        assert processor.network is not None
    
    def test_n_back_network(self, task_spec):
        """Test N-back network processing."""
        processor = CognitiveNeuralProcessor(task_spec)
        output = processor.process(task_spec.input_data)
        
        # LSTM should return sequence output
        assert output.shape == task_spec.input_data.shape
        assert torch.isfinite(output).all()
    
    def test_attention_network(self):
        """Test attention network for Stroop task."""
        task_spec = TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="stroop",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
        
        processor = CognitiveNeuralProcessor(task_spec)
        output = processor.process(task_spec.input_data)
        
        assert output.shape == task_spec.input_data.shape
        assert torch.isfinite(output).all()


class TestMetaOptimizer:
    """Test MetaOptimizer main class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create MetaOptimizer instance for testing."""
        return MetaOptimizer(
            cognitive_constraints={"authenticity": 0.8, "memory_capacity": 7},
            efficiency_requirements={"max_flops": 1e9, "max_memory": 1e6},
            random_seed=42
        )
    
    @pytest.fixture
    def task_spec(self):
        """Create task specification for testing."""
        return TaskSpecification(
            input_data=torch.randn(32, 10, 64),
            target_output=torch.randn(32, 10, 64),
            task_type="n_back",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
    
    def test_initialization(self, optimizer):
        """Test proper initialization of MetaOptimizer."""
        assert optimizer.cognitive_constraints["authenticity"] == 0.8
        assert optimizer.efficiency_requirements["max_flops"] == 1e9
        assert len(optimizer.alpha_history) == 0
        assert len(optimizer.lambda_history) == 0
        assert len(optimizer.beta_history) == 0
        assert len(optimizer.performance_history) == 0
    
    def test_meta_optimize_basic(self, optimizer, task_spec):
        """Test basic meta-optimization functionality."""
        result = optimizer.meta_optimize(
            task_spec,
            max_iterations=10,  # Small number for testing
            convergence_threshold=1e-3
        )
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert 0 <= result.alpha <= 1
        assert result.lambda_1 >= 0
        assert result.lambda_2 >= 0
        assert result.beta > 0
        assert isinstance(result.confidence_interval, ConfidenceInterval)
        assert isinstance(result.effect_size, EffectSize)
        assert len(result.convergence_history) > 0
    
    def test_parameter_bounds(self, optimizer, task_spec):
        """Test that parameters stay within valid bounds."""
        result = optimizer.meta_optimize(
            task_spec,
            max_iterations=5
        )
        
        # Alpha should be between 0 and 1
        assert 0 <= result.alpha <= 1
        
        # Lambda values should be non-negative
        assert result.lambda_1 >= 0
        assert result.lambda_2 >= 0
        
        # Beta should be positive
        assert result.beta > 0
    
    def test_convergence_detection(self, optimizer, task_spec):
        """Test convergence detection mechanism."""
        # Mock convergence by making performance stable
        with patch.object(optimizer, '_evaluate_performance', return_value=0.8):
            result = optimizer.meta_optimize(
                task_spec,
                max_iterations=100,
                convergence_threshold=1e-6
            )
            
            # Should converge quickly with stable performance
            assert len(result.convergence_history) < 100
    
    def test_failure_handling(self, optimizer, task_spec):
        """Test failure handling and documentation."""
        # Mock a failure in processing
        with patch.object(CognitiveSymbolicReasoner, 'process', side_effect=RuntimeError("Test error")):
            result = optimizer.meta_optimize(
                task_spec,
                max_iterations=5
            )
            
            # Should still return a result despite failures
            assert isinstance(result, OptimizationResult)
            assert len(result.failure_modes) > 0
    
    def test_performance_improvement(self, optimizer, task_spec):
        """Test that optimization shows performance improvement."""
        result = optimizer.meta_optimize(
            task_spec,
            max_iterations=20
        )
        
        # Performance should generally improve over time
        if len(result.convergence_history) > 1:
            initial_performance = result.convergence_history[0]
            final_performance = result.convergence_history[-1]
            
            # Allow for some variation but expect general improvement
            assert final_performance >= initial_performance - 0.1
    
    def test_statistical_analysis(self, optimizer, task_spec):
        """Test statistical analysis of results."""
        result = optimizer.meta_optimize(
            task_spec,
            max_iterations=15
        )
        
        # Check confidence interval properties
        ci = result.confidence_interval
        assert ci.lower <= ci.upper
        assert 0 <= ci.confidence <= 1
        
        # Check effect size
        effect_size = result.effect_size
        assert isinstance(effect_size.cohens_d, float)
        assert effect_size.interpretation in ["negligible", "small", "medium", "large"]
    
    def test_different_task_types(self, optimizer):
        """Test optimization with different task types."""
        task_types = ["n_back", "stroop", "planning", "pattern_recognition"]
        
        for task_type in task_types:
            task_spec = TaskSpecification(
                input_data=torch.randn(16, 8, 32),  # Smaller for faster testing
                target_output=torch.randn(16, 8, 32),
                task_type=task_type,
                cognitive_constraints={"authenticity": 0.8},
                efficiency_requirements={"max_flops": 1e9}
            )
            
            result = optimizer.meta_optimize(
                task_spec,
                max_iterations=5
            )
            
            assert isinstance(result, OptimizationResult)
            assert 0 <= result.alpha <= 1
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        task_spec = TaskSpecification(
            input_data=torch.randn(16, 8, 32),
            target_output=torch.randn(16, 8, 32),
            task_type="n_back",
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
        
        # Run optimization twice with same seed
        optimizer1 = MetaOptimizer(
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9},
            random_seed=42
        )
        
        optimizer2 = MetaOptimizer(
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9},
            random_seed=42
        )
        
        result1 = optimizer1.meta_optimize(task_spec, max_iterations=5)
        result2 = optimizer2.meta_optimize(task_spec, max_iterations=5)
        
        # Results should be similar (allowing for small numerical differences)
        assert abs(result1.alpha - result2.alpha) < 0.1
        assert abs(result1.lambda_1 - result2.lambda_1) < 0.1
        assert abs(result1.lambda_2 - result2.lambda_2) < 0.1


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creation of OptimizationResult."""
        ci = ConfidenceInterval(0.1, 0.3, 0.95)
        effect_size = EffectSize(0.5, "medium")
        
        result = OptimizationResult(
            alpha=0.6,
            lambda_1=0.1,
            lambda_2=0.2,
            beta=1.2,
            performance_gain=0.15,
            confidence_interval=ci,
            effect_size=effect_size,
            convergence_history=[0.1, 0.12, 0.15],
            failure_modes=[]
        )
        
        assert result.alpha == 0.6
        assert result.performance_gain == 0.15
        assert result.confidence_interval == ci
        assert result.effect_size == effect_size
        assert len(result.convergence_history) == 3
        assert len(result.failure_modes) == 0


@pytest.mark.integration
class TestMetaOptimizationIntegration:
    """Integration tests for meta-optimization components."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create realistic task specification
        batch_size, seq_len, features = 16, 8, 32
        task_spec = TaskSpecification(
            input_data=torch.randn(batch_size, seq_len, features),
            target_output=torch.randn(batch_size, seq_len, features),
            task_type="n_back",
            cognitive_constraints={
                "authenticity": 0.8,
                "memory_capacity": 7,
                "attention_threshold": 0.7
            },
            efficiency_requirements={
                "max_flops": 1e9,
                "max_memory": 1e6,
                "max_complexity": 1.0
            }
        )
        
        # Initialize optimizer
        optimizer = MetaOptimizer(
            cognitive_constraints=task_spec.cognitive_constraints,
            efficiency_requirements=task_spec.efficiency_requirements,
            random_seed=42
        )
        
        # Run optimization
        result = optimizer.meta_optimize(
            task_spec,
            max_iterations=20,
            target_improvement=0.15
        )
        
        # Validate results
        assert isinstance(result, OptimizationResult)
        assert result.performance_gain >= 0
        assert len(result.convergence_history) > 0
        
        # Check that all components were used
        assert len(optimizer.alpha_history) > 0
        assert len(optimizer.lambda_history) > 0
        assert len(optimizer.beta_history) > 0
    
    def test_component_interaction(self):
        """Test interaction between different components."""
        task_spec = TaskSpecification(
            input_data=torch.randn(8, 5, 16),
            target_output=torch.randn(8, 5, 16),
            task_type="stroop",
            cognitive_constraints={"authenticity": 0.9},
            efficiency_requirements={"max_flops": 5e8}
        )
        
        optimizer = MetaOptimizer(
            cognitive_constraints=task_spec.cognitive_constraints,
            efficiency_requirements=task_spec.efficiency_requirements
        )
        
        # Test that components work together
        result = optimizer.meta_optimize(task_spec, max_iterations=10)
        
        # Verify component integration
        assert optimizer.dynamic_integrator is not None
        assert optimizer.cognitive_regularizer is not None
        assert optimizer.bias_modeler is not None
        assert optimizer.failure_documenter is not None
        
        # Check that parameters evolved
        assert len(optimizer.alpha_history) > 0
        assert len(optimizer.lambda_history) > 0
        assert len(optimizer.beta_history) > 0


if __name__ == "__main__":
    pytest.main([__file__])