"""
Failure Documentation System

Systematic documentation and learning from optimization failures.
Implements the "failure museum" concept for transparent science and
continuous improvement of the meta-optimization framework.
"""

import json
import datetime
import hashlib
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in the optimization process."""
    CONVERGENCE_FAILURE = "convergence_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"
    MEMORY_ERROR = "memory_error"
    PARAMETER_EXPLOSION = "parameter_explosion"
    COGNITIVE_VIOLATION = "cognitive_violation"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"
    INTEGRATION_FAILURE = "integration_failure"
    BIAS_MODELING_ERROR = "bias_modeling_error"
    DATA_CORRUPTION = "data_corruption"
    HARDWARE_FAILURE = "hardware_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Severity levels for failures."""
    LOW = "low"           # Minor issues, optimization can continue
    MEDIUM = "medium"     # Significant issues, may affect results
    HIGH = "high"         # Major issues, optimization should stop
    CRITICAL = "critical" # System-threatening issues


@dataclass
class FailureMode:
    """Detailed failure mode documentation."""
    failure_id: str
    timestamp: datetime.datetime
    failure_type: FailureType
    severity: FailureSeverity
    description: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    lessons_learned: Optional[str] = None
    prevention_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["failure_type"] = self.failure_type.value
        data["severity"] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureMode":
        """Create from dictionary."""
        data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
        data["failure_type"] = FailureType(data["failure_type"])
        data["severity"] = FailureSeverity(data["severity"])
        return cls(**data)


class FailureDocumenter:
    """
    Main failure documentation system.
    
    Provides systematic documentation, analysis, and learning from
    optimization failures to improve framework robustness.
    """
    
    def __init__(self, 
                 storage_path: str = "docs/failure_museum",
                 auto_save: bool = True):
        """
        Initialize failure documenter.
        
        Args:
            storage_path: Path to store failure documentation
            auto_save: Whether to automatically save failures to disk
        """
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.failures: List[FailureMode] = []
        
        # Create storage directory if it doesn't exist
        if self.auto_save:
            os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing failures
        self._load_existing_failures()
    
    def document_failure(self,
                        failure_type: Union[str, FailureType],
                        description: str,
                        parameters: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None,
                        severity: Union[str, FailureSeverity] = FailureSeverity.MEDIUM,
                        error_message: Optional[str] = None,
                        stack_trace: Optional[str] = None,
                        **kwargs) -> FailureMode:
        """
        Document a failure occurrence.
        
        Args:
            failure_type: Type of failure
            description: Human-readable description
            parameters: Parameters at time of failure
            context: Additional context information
            severity: Severity level
            error_message: Error message if available
            stack_trace: Stack trace if available
            **kwargs: Additional failure-specific information
            
        Returns:
            FailureMode object
        """
        # Convert string types to enums
        if isinstance(failure_type, str):
            try:
                failure_type = FailureType(failure_type)
            except ValueError:
                failure_type = FailureType.UNKNOWN
        
        if isinstance(severity, str):
            try:
                severity = FailureSeverity(severity)
            except ValueError:
                severity = FailureSeverity.MEDIUM
        
        # Generate unique failure ID
        failure_id = self._generate_failure_id(failure_type, parameters)
        
        # Create failure mode object
        failure = FailureMode(
            failure_id=failure_id,
            timestamp=datetime.datetime.now(),
            failure_type=failure_type,
            severity=severity,
            description=description,
            parameters=parameters.copy(),
            context=context or {},
            error_message=error_message,
            stack_trace=stack_trace
        )
        
        # Add any additional kwargs to context
        failure.context.update(kwargs)
        
        # Store failure
        self.failures.append(failure)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_failure(failure)
        
        logger.warning(f"Documented failure: {failure_type.value} - {description}")
        
        return failure
    
    def _generate_failure_id(self, 
                           failure_type: FailureType,
                           parameters: Dict[str, Any]) -> str:
        """Generate unique failure ID based on type and parameters."""
        # Create hash from failure type and key parameters
        hash_input = f"{failure_type.value}_{json.dumps(parameters, sort_keys=True)}"
        hash_object = hashlib.md5(hash_input.encode())
        return f"{failure_type.value}_{hash_object.hexdigest()[:8]}"
    
    def _save_failure(self, failure: FailureMode) -> None:
        """Save individual failure to disk."""
        try:
            filename = f"{failure.timestamp.strftime('%Y%m%d_%H%M%S')}_{failure.failure_id}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            with open(filepath, 'w') as f:
                json.dump(failure.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save failure documentation: {e}")
    
    def _load_existing_failures(self) -> None:
        """Load existing failure documentation from disk."""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.storage_path, filename)
                    with open(filepath, 'r') as f:
                        failure_data = json.load(f)
                        failure = FailureMode.from_dict(failure_data)
                        self.failures.append(failure)
                        
            logger.info(f"Loaded {len(self.failures)} existing failure records")
            
        except Exception as e:
            logger.error(f"Failed to load existing failures: {e}")
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in documented failures.
        
        Returns:
            Dictionary with failure pattern analysis
        """
        if not self.failures:
            return {"message": "No failures documented yet"}
        
        # Failure type distribution
        type_counts = {}
        for failure in self.failures:
            failure_type = failure.failure_type.value
            type_counts[failure_type] = type_counts.get(failure_type, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for failure in self.failures:
            severity = failure.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Temporal patterns
        failure_dates = [f.timestamp.date() for f in self.failures]
        unique_dates = list(set(failure_dates))
        daily_counts = {date: failure_dates.count(date) for date in unique_dates}
        
        # Parameter patterns (most common parameter values at failure)
        parameter_patterns = {}
        for failure in self.failures:
            for param, value in failure.parameters.items():
                if param not in parameter_patterns:
                    parameter_patterns[param] = {}
                
                # Convert value to string for counting
                value_str = str(value)
                if value_str not in parameter_patterns[param]:
                    parameter_patterns[param][value_str] = 0
                parameter_patterns[param][value_str] += 1
        
        # Most frequent failure contexts
        context_patterns = {}
        for failure in self.failures:
            for key, value in failure.context.items():
                if key not in context_patterns:
                    context_patterns[key] = {}
                
                value_str = str(value)
                if value_str not in context_patterns[key]:
                    context_patterns[key][value_str] = 0
                context_patterns[key][value_str] += 1
        
        return {
            "total_failures": len(self.failures),
            "failure_types": type_counts,
            "severity_distribution": severity_counts,
            "daily_failure_counts": daily_counts,
            "parameter_patterns": parameter_patterns,
            "context_patterns": context_patterns,
            "most_common_failure": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            "average_failures_per_day": len(self.failures) / max(1, len(unique_dates))
        }
    
    def get_failure_recommendations(self, 
                                  current_parameters: Dict[str, Any],
                                  current_context: Dict[str, Any]) -> List[str]:
        """
        Get recommendations based on historical failures.
        
        Args:
            current_parameters: Current optimization parameters
            current_context: Current optimization context
            
        Returns:
            List of recommendations to avoid failures
        """
        recommendations = []
        
        if not self.failures:
            return ["No historical failure data available for recommendations"]
        
        # Check for parameter combinations that led to failures
        risky_parameters = self._identify_risky_parameters(current_parameters)
        if risky_parameters:
            recommendations.append(
                f"Warning: Parameters {risky_parameters} have been associated with failures. "
                "Consider adjusting these values."
            )
        
        # Check for context conditions that led to failures
        risky_contexts = self._identify_risky_contexts(current_context)
        if risky_contexts:
            recommendations.append(
                f"Caution: Context conditions {risky_contexts} have been associated with failures. "
                "Monitor these conditions closely."
            )
        
        # General recommendations based on most common failures
        failure_analysis = self.analyze_failure_patterns()
        most_common = failure_analysis.get("most_common_failure")
        
        if most_common:
            specific_recommendations = self._get_specific_recommendations(most_common)
            recommendations.extend(specific_recommendations)
        
        return recommendations if recommendations else ["No specific recommendations based on current parameters"]
    
    def _identify_risky_parameters(self, current_parameters: Dict[str, Any]) -> List[str]:
        """Identify parameters that have been associated with failures."""
        risky_params = []
        
        for param, value in current_parameters.items():
            # Count failures with similar parameter values
            failure_count = 0
            total_count = 0
            
            for failure in self.failures:
                if param in failure.parameters:
                    total_count += 1
                    failure_value = failure.parameters[param]
                    
                    # Check if values are similar (within 10% for numeric values)
                    if self._values_similar(value, failure_value):
                        failure_count += 1
            
            # If more than 50% of failures with this parameter had similar values
            if total_count > 0 and failure_count / total_count > 0.5:
                risky_params.append(param)
        
        return risky_params
    
    def _identify_risky_contexts(self, current_context: Dict[str, Any]) -> List[str]:
        """Identify context conditions associated with failures."""
        risky_contexts = []
        
        for key, value in current_context.items():
            failure_count = 0
            total_count = 0
            
            for failure in self.failures:
                if key in failure.context:
                    total_count += 1
                    failure_value = failure.context[key]
                    
                    if self._values_similar(value, failure_value):
                        failure_count += 1
            
            if total_count > 0 and failure_count / total_count > 0.5:
                risky_contexts.append(key)
        
        return risky_contexts
    
    def _values_similar(self, value1: Any, value2: Any, tolerance: float = 0.1) -> bool:
        """Check if two values are similar."""
        try:
            # For numeric values
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if value1 == 0 and value2 == 0:
                    return True
                elif value1 == 0 or value2 == 0:
                    return abs(value1 - value2) < tolerance
                else:
                    return abs(value1 - value2) / max(abs(value1), abs(value2)) < tolerance
            
            # For string values
            elif isinstance(value1, str) and isinstance(value2, str):
                return value1.lower() == value2.lower()
            
            # For other types, use exact equality
            else:
                return value1 == value2
                
        except:
            return False
    
    def _get_specific_recommendations(self, failure_type: str) -> List[str]:
        """Get specific recommendations for common failure types."""
        recommendations_map = {
            "convergence_failure": [
                "Consider reducing learning rate or adaptation rate",
                "Increase convergence threshold or maximum iterations",
                "Check for parameter bounds and constraints"
            ],
            "numerical_instability": [
                "Add numerical stability checks (gradient clipping, etc.)",
                "Use more conservative parameter update rules",
                "Consider using double precision arithmetic"
            ],
            "parameter_explosion": [
                "Implement parameter bounds and clipping",
                "Reduce learning rates for parameter updates",
                "Add regularization to prevent extreme parameter values"
            ],
            "cognitive_violation": [
                "Review cognitive constraint weights and thresholds",
                "Ensure cognitive authenticity requirements are realistic",
                "Consider relaxing overly strict cognitive constraints"
            ],
            "efficiency_degradation": [
                "Monitor computational complexity during optimization",
                "Implement early stopping for efficiency violations",
                "Balance efficiency and performance trade-offs"
            ],
            "integration_failure": [
                "Check symbolic-neural component compatibility",
                "Verify input/output shape alignment",
                "Test components individually before integration"
            ],
            "bias_modeling_error": [
                "Validate bias parameter ranges (β should be > 0)",
                "Check probability distribution normalization",
                "Ensure bias implementations handle edge cases"
            ]
        }
        
        return recommendations_map.get(failure_type, ["Monitor system closely for this failure type"])
    
    def generate_failure_report(self, 
                              start_date: Optional[datetime.date] = None,
                              end_date: Optional[datetime.date] = None) -> str:
        """
        Generate comprehensive failure report.
        
        Args:
            start_date: Start date for report (optional)
            end_date: End date for report (optional)
            
        Returns:
            Formatted failure report
        """
        # Filter failures by date range if specified
        filtered_failures = self.failures
        if start_date or end_date:
            filtered_failures = []
            for failure in self.failures:
                failure_date = failure.timestamp.date()
                if start_date and failure_date < start_date:
                    continue
                if end_date and failure_date > end_date:
                    continue
                filtered_failures.append(failure)
        
        if not filtered_failures:
            return "No failures found in specified date range."
        
        # Generate report
        report_lines = [
            "# Failure Analysis Report",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Period: {start_date or 'Beginning'} to {end_date or 'Present'}",
            f"Total Failures: {len(filtered_failures)}",
            "",
            "## Failure Summary"
        ]
        
        # Failure type breakdown
        type_counts = {}
        for failure in filtered_failures:
            failure_type = failure.failure_type.value
            type_counts[failure_type] = type_counts.get(failure_type, 0) + 1
        
        for failure_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(filtered_failures)) * 100
            report_lines.append(f"- {failure_type}: {count} ({percentage:.1f}%)")
        
        # Severity breakdown
        report_lines.extend(["", "## Severity Distribution"])
        severity_counts = {}
        for failure in filtered_failures:
            severity = failure.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in sorted(severity_counts.items()):
            percentage = (count / len(filtered_failures)) * 100
            report_lines.append(f"- {severity}: {count} ({percentage:.1f}%)")
        
        # Recent failures (last 5)
        report_lines.extend(["", "## Recent Failures"])
        recent_failures = sorted(filtered_failures, key=lambda x: x.timestamp, reverse=True)[:5]
        
        for failure in recent_failures:
            report_lines.append(
                f"- {failure.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                f"{failure.failure_type.value} ({failure.severity.value}): {failure.description}"
            )
        
        # Recommendations
        report_lines.extend(["", "## Recommendations"])
        analysis = self.analyze_failure_patterns()
        most_common = analysis.get("most_common_failure")
        
        if most_common:
            recommendations = self._get_specific_recommendations(most_common)
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        return "\n".join(report_lines)
    
    def export_failures(self, 
                       filepath: str,
                       format: str = "json") -> None:
        """
        Export failure data to file.
        
        Args:
            filepath: Output file path
            format: Export format ("json", "csv")
        """
        try:
            if format == "json":
                with open(filepath, 'w') as f:
                    failure_data = [failure.to_dict() for failure in self.failures]
                    json.dump(failure_data, f, indent=2)
            
            elif format == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    if self.failures:
                        writer = csv.DictWriter(f, fieldnames=self.failures[0].to_dict().keys())
                        writer.writeheader()
                        for failure in self.failures:
                            writer.writerow(failure.to_dict())
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(self.failures)} failures to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export failures: {e}")
    
    def clear_failures(self, 
                      before_date: Optional[datetime.date] = None) -> int:
        """
        Clear failure records.
        
        Args:
            before_date: Only clear failures before this date (optional)
            
        Returns:
            Number of failures cleared
        """
        if before_date:
            original_count = len(self.failures)
            self.failures = [
                f for f in self.failures 
                if f.timestamp.date() >= before_date
            ]
            cleared_count = original_count - len(self.failures)
        else:
            cleared_count = len(self.failures)
            self.failures.clear()
        
        logger.info(f"Cleared {cleared_count} failure records")
        return cleared_count


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure Documentation System")
    parser.add_argument("--report", action="store_true", help="Generate failure report")
    parser.add_argument("--export", type=str, help="Export failures to file")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    parser.add_argument("--storage-path", type=str, default="docs/failure_museum", help="Storage path")
    
    args = parser.parse_args()
    
    documenter = FailureDocumenter(storage_path=args.storage_path)
    
    if args.report:
        report = documenter.generate_failure_report()
        print(report)
    
    if args.export:
        documenter.export_failures(args.export, args.format)
        print(f"Exported failures to {args.export}")


if __name__ == "__main__":
    main()