#!/usr/bin/env python3
"""
Configuration Loader for Meta-Optimization Framework

This module provides utilities for loading and managing configuration files
for the meta-optimization framework. It supports:

- YAML configuration file loading
- Configuration validation
- Default value handling
- Environment variable overrides
- Configuration merging and updates
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Configuration file paths."""
    default_config: str = "configs/default_config.yaml"
    experiment_config: Optional[str] = None
    user_config: Optional[str] = None
    
    
class ConfigLoader:
    """
    Configuration loader with support for multiple config files,
    environment variable overrides, and validation.
    """
    
    def __init__(self, 
                 base_path: Optional[Union[str, Path]] = None,
                 config_paths: Optional[ConfigPaths] = None):
        """
        Initialize configuration loader.
        
        Args:
            base_path: Base path for configuration files
            config_paths: Configuration file paths
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config_paths = config_paths or ConfigPaths()
        self.config = {}
        self._env_prefix = "META_OPT_"
        
    def load_config(self, 
                   config_file: Optional[str] = None,
                   load_defaults: bool = True,
                   apply_env_overrides: bool = True) -> Dict[str, Any]:
        """
        Load configuration from files and environment variables.
        
        Args:
            config_file: Specific config file to load
            load_defaults: Whether to load default configuration
            apply_env_overrides: Whether to apply environment variable overrides
            
        Returns:
            Loaded configuration dictionary
        """
        config = {}
        
        # Load default configuration
        if load_defaults:
            default_path = self.base_path / self.config_paths.default_config
            if default_path.exists():
                config.update(self._load_yaml_file(default_path))
                logger.info(f"Loaded default config from {default_path}")
            else:
                logger.warning(f"Default config file not found: {default_path}")
        
        # Load experiment configuration
        if self.config_paths.experiment_config:
            exp_path = self.base_path / self.config_paths.experiment_config
            if exp_path.exists():
                exp_config = self._load_yaml_file(exp_path)
                config = self._merge_configs(config, exp_config)
                logger.info(f"Loaded experiment config from {exp_path}")
        
        # Load user configuration
        if self.config_paths.user_config:
            user_path = self.base_path / self.config_paths.user_config
            if user_path.exists():
                user_config = self._load_yaml_file(user_path)
                config = self._merge_configs(config, user_config)
                logger.info(f"Loaded user config from {user_path}")
        
        # Load specific config file
        if config_file:
            file_path = self.base_path / config_file
            if file_path.exists():
                file_config = self._load_yaml_file(file_path)
                config = self._merge_configs(config, file_config)
                logger.info(f"Loaded config from {file_path}")
            else:
                logger.error(f"Config file not found: {file_path}")
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Apply environment variable overrides
        if apply_env_overrides:
            config = self._apply_env_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            raise
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        result = copy.deepcopy(config)
        
        # Get all environment variables with the prefix
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self._env_prefix)}
        
        for env_key, env_value in env_vars.items():
            # Convert environment variable name to config path
            config_path = env_key[len(self._env_prefix):].lower().split('_')
            
            # Navigate to the correct nested dictionary
            current = result
            for path_part in config_path[:-1]:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
            
            # Set the value with type conversion
            current[config_path[-1]] = self._convert_env_value(env_value)
            logger.info(f"Applied environment override: {env_key} = {env_value}")
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try list (comma-separated)
        if ',' in value:
            return [self._convert_env_value(item.strip()) for item in value.split(',')]
        
        # Return as string
        return value
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values."""
        # Check required sections
        required_sections = [
            'framework',
            'optimization',
            'cognitive_constraints',
            'efficiency_requirements'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing required configuration section: {section}")
        
        # Validate optimization targets
        if 'optimization' in config and 'targets' in config['optimization']:
            targets = config['optimization']['targets']
            
            # Check accuracy improvement
            if 'accuracy_improvement' in targets:
                acc_imp = targets['accuracy_improvement']
                if not 0 <= acc_imp <= 1:
                    raise ValueError(f"accuracy_improvement must be between 0 and 1, got {acc_imp}")
            
            # Check efficiency gain
            if 'efficiency_gain' in targets:
                eff_gain = targets['efficiency_gain']
                if not 0 <= eff_gain <= 1:
                    raise ValueError(f"efficiency_gain must be between 0 and 1, got {eff_gain}")
        
        # Validate alpha bounds
        if 'optimization' in config and 'dynamic_integration' in config['optimization']:
            di = config['optimization']['dynamic_integration']
            
            if 'min_alpha' in di and 'max_alpha' in di:
                if di['min_alpha'] >= di['max_alpha']:
                    raise ValueError("min_alpha must be less than max_alpha")
        
        # Validate lambda bounds
        if 'optimization' in config and 'cognitive_regularization' in config['optimization']:
            cr = config['optimization']['cognitive_regularization']
            
            if 'min_lambda' in cr and 'max_lambda' in cr:
                if cr['min_lambda'] >= cr['max_lambda']:
                    raise ValueError("min_lambda must be less than max_lambda")
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
    
    def save_config(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        self.config = self._merge_configs(self.config, updates)
        logger.info("Configuration updated from dictionary")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def print_config(self, section: Optional[str] = None) -> None:
        """Print configuration in a readable format."""
        config_to_print = self.get_section(section) if section else self.config
        print(yaml.dump(config_to_print, default_flow_style=False, indent=2))


# Global configuration loader instance
config_loader = ConfigLoader()


def load_config(config_file: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Configuration file to load
        **kwargs: Additional arguments for ConfigLoader.load_config
        
    Returns:
        Loaded configuration dictionary
    """
    return config_loader.load_config(config_file=config_file, **kwargs)


def get_config(key_path: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration value.
    
    Args:
        key_path: Dot-separated path to configuration value
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    return config_loader.get(key_path, default)