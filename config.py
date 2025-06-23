"""
Configuration Management
Handles application settings and configuration
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config_path = Path(config_file)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    config = json.load(file)
                    logger.info(f"Loaded configuration from {self.config_file}")
                    return config
            else:
                # Create default configuration
                default_config = self._get_default_config()
                self._save_config(default_config)
                logger.info(f"Created default configuration at {self.config_file}")
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            "app": {
                "name": "AI Personal Finance Assistant",
                "version": "1.0.0",
                "debug": False,
                "port": 8501
            },
            "data": {
                "backup_enabled": True,
                "backup_interval_days": 7,
                "auto_categorize": True,
                "data_retention_days": 365
            },
            "analysis": {
                "anomaly_detection_enabled": True,
                "anomaly_threshold": 2.5,
                "trend_analysis_window": 30,
                "prediction_horizon_months": 6
            },
            "visualization": {
                "theme": "plotly",
                "color_scheme": "default",
                "chart_height": 400,
                "show_animations": True
            },
            "notifications": {
                "spending_alerts": True,
                "budget_warnings": True,
                "goal_reminders": True,
                "alert_threshold_percentage": 0.8
            },
            "security": {
                "data_encryption": False,
                "session_timeout_minutes": 60,
                "max_file_size_mb": 10
            },
            "ai": {
                "advice_enabled": True,
                "prediction_confidence_threshold": 0.7,
                "auto_insights": True,
                "personalization_level": "medium"
            },
            "export": {
                "default_format": "csv",
                "include_metadata": True,
                "date_format": "%Y-%m-%d"
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as file:
                json.dump(config, file, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('app.debug', False)
        """
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value for {key_path}: {str(e)}")
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        Example: config.set('app.debug', True)
        """
        try:
            keys = key_path.split('.')
            config = self._config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            
            # Save to file
            self._save_config(self._config)
            logger.info(f"Configuration updated: {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"Error setting config value for {key_path}: {str(e)}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update entire configuration section"""
        try:
            if section not in self._config:
                self._config[section] = {}
            
            self._config[section].update(updates)
            self._save_config(self._config)
            logger.info(f"Configuration section '{section}' updated")
            
        except Exception as e:
            logger.error(f"Error updating config section {section}: {str(e)}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        try:
            self._config = self._get_default_config()
            self._save_config(self._config)
            logger.info("Configuration reset to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {str(e)}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        warnings = []
        
        try:
            # Validate required sections
            required_sections = ['app', 'data', 'analysis', 'visualization']
            for section in required_sections:
                if section not in self._config:
                    issues.append(f"Missing required section: {section}")
            
            # Validate specific values
            if self.get('security.session_timeout_minutes', 0) < 5:
                warnings.append("Session timeout is very short (< 5 minutes)")
            
            if self.get('security.max_file_size_mb', 0) > 100:
                warnings.append("Maximum file size is very large (> 100MB)")
            
            if self.get('data.data_retention_days', 0) < 30:
                warnings.append("Data retention period is very short (< 30 days)")
            
            # Validate port number
            port = self.get('app.port', 8501)
            if not isinstance(port, int) or port < 1000 or port > 65535:
                issues.append("Invalid port number (must be between 1000-65535)")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return {
                'valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user-specific preferences"""
        return {
            'theme': self.get('visualization.theme', 'plotly'),
            'notifications': self.get('notifications', {}),
            'auto_categorize': self.get('data.auto_categorize', True),
            'show_animations': self.get('visualization.show_animations', True),
            'advice_enabled': self.get('ai.advice_enabled', True)
        }
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user-specific preferences"""
        try:
            preference_mapping = {
                'theme': 'visualization.theme',
                'auto_categorize': 'data.auto_categorize',
                'show_animations': 'visualization.show_animations',
                'advice_enabled': 'ai.advice_enabled'
            }
            
            for pref_key, config_key in preference_mapping.items():
                if pref_key in preferences:
                    self.set(config_key, preferences[pref_key])
            
            # Handle notifications separately (nested structure)
            if 'notifications' in preferences:
                current_notifications = self.get_section('notifications')
                current_notifications.update(preferences['notifications'])
                self.update_section('notifications', current_notifications)
            
            logger.info("User preferences updated")
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to specified path"""
        try:
            with open(export_path, 'w') as file:
                json.dump(self._config, file, indent=2)
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {str(e)}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """Import configuration from specified path"""
        try:
            with open(import_path, 'r') as file:
                imported_config = json.load(file)
            
            # Validate imported config
            if not isinstance(imported_config, dict):
                raise ValueError("Invalid configuration format")
            
            # Backup current config
            backup_path = f"{self.config_file}.backup"
            self.export_config(backup_path)
            
            # Update configuration
            self._config = imported_config
            self._save_config(self._config)
            
            logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {str(e)}")
            return False
    
    @property
    def app_name(self) -> str:
        """Get application name"""
        return self.get('app.name', 'AI Personal Finance Assistant')
    
    @property
    def app_version(self) -> str:
        """Get application version"""
        return self.get('app.version', '1.0.0')
    
    @property
    def debug_mode(self) -> bool:
        """Get debug mode setting"""
        return self.get('app.debug', False)
    
    @property
    def port(self) -> int:
        """Get application port"""
        return self.get('app.port', 8501)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Config(file={self.config_file}, sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"Config(config_file='{self.config_file}', config={self._config})"
