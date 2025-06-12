# Contributing to Meta-Optimization Framework

Thank you for your interest in contributing to the Meta-Optimization Framework! This document provides guidelines for contributing to this cognitive-computational research project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Research Contributions](#research-contributions)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)

## Code of Conduct

This project adheres to principles of open science and collaborative research. We expect all contributors to:

- Maintain scientific rigor and reproducibility
- Respect diverse perspectives and approaches
- Provide constructive feedback and criticism
- Acknowledge contributions and prior work appropriately
- Follow ethical guidelines for AI research

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of cognitive science and machine learning
- Familiarity with PyTorch and scientific computing

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/meta-optimization-framework.git
   cd meta-optimization-framework
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Install Development Tools**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   python -m pytest tests/
   python -m src.core.meta_optimization --task n_back --max-iterations 10
   ```

## Contributing Guidelines

### Types of Contributions

1. **Research Contributions**
   - New cognitive constraints or bias models
   - Improved optimization algorithms
   - Novel evaluation metrics
   - Experimental validation studies

2. **Technical Contributions**
   - Bug fixes and performance improvements
   - New utility functions and tools
   - Documentation improvements
   - Test coverage expansion

3. **Community Contributions**
   - Tutorial development
   - Example implementations
   - Educational materials
   - Failure documentation and analysis

### Branch Strategy

- `main`: Stable release branch
- `develop`: Integration branch for new features
- `feature/description`: Feature development branches
- `hotfix/description`: Critical bug fixes
- `research/paper-name`: Research-specific branches

### Commit Guidelines

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `research`: Research-related changes

Example:
```
feat(bias-modeling): add availability bias implementation

Implements availability bias using memory decay mechanism
as described in cognitive psychology literature.

Closes #123
```

## Research Contributions

### Mathematical Framework

When contributing to the core mathematical framework:

1. **Theoretical Foundation**
   - Provide mathematical derivations
   - Reference relevant cognitive science literature
   - Ensure consistency with existing framework

2. **Implementation Requirements**
   - Follow the established API patterns
   - Include comprehensive docstrings
   - Provide usage examples

3. **Validation Requirements**
   - Include unit tests for mathematical correctness
   - Provide cognitive authenticity validation
   - Document expected performance characteristics

### Cognitive Constraints

New cognitive constraints should:

1. **Scientific Basis**
   - Be grounded in cognitive science literature
   - Include references to empirical studies
   - Explain psychological mechanisms

2. **Implementation Standards**
   - Inherit from `CognitiveConstraint` base class
   - Implement required methods with proper error handling
   - Include parameter validation

3. **Documentation Requirements**
   - Explain the cognitive phenomenon
   - Provide usage examples
   - Document parameter ranges and effects

### Bias Models

New bias implementations should:

1. **Psychological Accuracy**
   - Accurately model known cognitive biases
   - Include parameter calibration methods
   - Validate against human behavioral data

2. **Technical Requirements**
   - Inherit from `CognitiveBias` base class
   - Handle edge cases and numerical stability
   - Support batch processing

## Code Quality Standards

### Style Guidelines

- Follow PEP 8 style guide
- Use Black for code formatting
- Maximum line length: 127 characters
- Use type hints for all function signatures

### Code Organization

```python
"""
Module docstring explaining purpose and key concepts.

Mathematical formulations should be included where relevant.
"""

import standard_library
import third_party_packages
from local_modules import specific_functions

# Constants
CONSTANT_NAME = value

class ClassName:
    """Class docstring with mathematical notation if applicable."""
    
    def __init__(self, param: type) -> None:
        """Initialize with clear parameter descriptions."""
        pass
    
    def public_method(self, param: type) -> return_type:
        """
        Public method with comprehensive docstring.
        
        Args:
            param: Description with expected ranges/constraints
            
        Returns:
            Description of return value and its properties
            
        Raises:
            SpecificError: When and why this error occurs
        """
        pass
    
    def _private_method(self) -> None:
        """Private method for internal use."""
        pass
```

### Documentation Standards

1. **Docstring Requirements**
   - Use Google-style docstrings
   - Include mathematical formulations
   - Provide usage examples
   - Document cognitive interpretations

2. **Mathematical Notation**
   - Use LaTeX notation in docstrings
   - Be consistent with paper notation
   - Explain variable meanings

3. **Code Comments**
   - Explain complex algorithms
   - Reference relevant literature
   - Clarify cognitive interpretations

## Testing Requirements

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and methods
   - Verify mathematical correctness
   - Check edge cases and error handling

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Verify end-to-end workflows
   - Check performance characteristics

3. **Validation Tests** (`tests/validation/`)
   - Cognitive authenticity validation
   - Performance benchmark verification
   - Statistical significance testing

4. **Benchmark Tests** (`tests/benchmarks/`)
   - Performance measurement
   - Memory usage tracking
   - Scalability testing

### Test Implementation

```python
import pytest
import torch
import numpy as np
from src.core.meta_optimization import MetaOptimizer

class TestMetaOptimizer:
    """Test suite for MetaOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MetaOptimizer(
            cognitive_constraints={"authenticity": 0.8},
            efficiency_requirements={"max_flops": 1e9}
        )
    
    def test_initialization(self, optimizer):
        """Test proper initialization of optimizer."""
        assert optimizer.cognitive_constraints["authenticity"] == 0.8
        assert len(optimizer.alpha_history) == 0
    
    def test_mathematical_correctness(self, optimizer):
        """Test mathematical correctness of core algorithms."""
        # Create test data
        input_data = torch.randn(32, 10, 64)
        target_data = torch.randn(32, 10, 64)
        
        # Test optimization
        result = optimizer.meta_optimize(task_spec)
        
        # Verify mathematical properties
        assert 0 <= result.alpha <= 1
        assert result.performance_gain >= 0
        assert result.confidence_interval.lower <= result.confidence_interval.upper
    
    @pytest.mark.performance
    def test_performance_targets(self, optimizer):
        """Test that performance targets are achievable."""
        # Implementation should demonstrate target metrics
        pass
```

### Coverage Requirements

- Minimum 80% code coverage
- 100% coverage for core mathematical functions
- All public APIs must be tested
- Edge cases and error conditions must be covered

## Documentation Standards

### API Documentation

- All public functions and classes must have comprehensive docstrings
- Include mathematical formulations where relevant
- Provide usage examples
- Document cognitive interpretations

### Tutorial Development

When creating tutorials:

1. **Structure**
   - Clear learning objectives
   - Step-by-step instructions
   - Working code examples
   - Cognitive science context

2. **Content Requirements**
   - Explain both technical and cognitive aspects
   - Include visualization of results
   - Provide exercises for practice
   - Reference relevant literature

### Mathematical Documentation

- Use consistent notation across all documentation
- Provide derivations for key equations
- Explain cognitive interpretations of mathematical terms
- Include references to source literature

## Submission Process

### Pull Request Guidelines

1. **Before Submitting**
   - Ensure all tests pass
   - Run code quality checks
   - Update documentation
   - Add appropriate tests

2. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes and motivation.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Research contribution
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Mathematical/Cognitive Basis
   Explanation of theoretical foundation (if applicable).
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Performance benchmarks run
   - [ ] Cognitive validation performed
   
   ## Documentation
   - [ ] Docstrings updated
   - [ ] API documentation updated
   - [ ] Tutorial/examples added
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Tests added for new functionality
   - [ ] Documentation updated
   ```

3. **Review Process**
   - Automated CI/CD checks must pass
   - Code review by maintainers
   - Research validation for cognitive components
   - Performance impact assessment

### Research Paper Contributions

For contributions related to research papers:

1. **Coordination**
   - Discuss with research team before major changes
   - Ensure consistency with paper objectives
   - Coordinate timing with publication schedule

2. **Documentation**
   - Link implementations to paper sections
   - Provide reproducibility information
   - Include experimental validation

3. **Validation**
   - Replicate paper results
   - Provide statistical validation
   - Document any deviations or improvements

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and research discussions
- **Research Team**: Direct contact for research-related contributions

### Resources

- [Project Documentation](docs/)
- [Mathematical Framework](docs/mathematical_framework/)
- [Failure Museum](docs/failure_museum/)
- [API Reference](docs/api/)

### Mentorship

New contributors can request mentorship for:
- Understanding the cognitive science background
- Learning the mathematical framework
- Developing research contributions
- Improving code quality

## Recognition

Contributors will be acknowledged in:
- Project documentation
- Research paper acknowledgments (where appropriate)
- Release notes
- Community highlights

Significant research contributions may warrant co-authorship on related publications, subject to standard academic criteria.

---

Thank you for contributing to advancing the intersection of cognitive science and artificial intelligence!