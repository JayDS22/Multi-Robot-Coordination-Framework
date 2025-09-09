#!/usr/bin/env python3
"""
Configuration Management for Multi-Robot Coordination Framework
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

class ConfigManager:
    """Manages configuration files and settings"""
    
    def __init__(self, config_file: str = "config/system_config.yaml"):
        self.config_file = config_file
        self.config_data = {}
        self.logger = logging.getLogger("config_manager")
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {self.config_file}, using defaults")
                self.config_data = self.get_default_config()
                return
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    self.config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            self.logger.info(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config_data = self.get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'coordination': {
                'max_robots': 50,
                'heartbeat_interval': 1.0,
                'task_timeout': 30.0
            },
            'learning': {
                'learning_rate': 0.01,
                'discount_factor': 0.95,
                'exploration_rate': 0.15
            },
            'communication': {
                'port': 11311,
                'timeout': 5.0,
                'max_retries': 3
            },
            'fault_tolerance': {
                'max_retries': 3,
                'failover_threshold': 2.0,
                'availability_target': 0.999
            }
        }
    
    def save_config(self, filename: Optional[str] = None):
        """Save current configuration to file"""
        save_path = filename or self.config_file
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.json'):
                    json.dump(self.config_data, f, indent=2)
                else:
                    yaml.safe_dump(self.config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")


class EnvironmentConfig:
    """Environment-specific configuration"""
    
    @staticmethod
    def get_warehouse_config():
        return {
            'boundaries': {'x_min': -50, 'x_max': 50, 'y_min': -50, 'y_max': 50},
            'task_types': ['pickup', 'delivery', 'transport', 'sorting'],
            'robot_capabilities': ['navigation', 'manipulation', 'sensing']
        }
    
    @staticmethod
    def get_factory_config():
        return {
            'boundaries': {'x_min': -100, 'x_max': 100, 'y_min': -100, 'y_max': 100},
            'task_types': ['maintenance', 'inspection', 'transport'],
            'robot_capabilities': ['navigation', 'manipulation', 'tools', 'sensing']
        }
