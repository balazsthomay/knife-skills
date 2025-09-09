"""
Configuration management for the knife skills trainer.
Handles settings persistence and parameter management.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class HandDetectionConfig:
    """Configuration for hand detection parameters"""
    max_num_hands: int = 2
    model_complexity: int = 0  # Use lighter model for better FPS
    min_detection_confidence: float = 0.7  # Higher threshold for fewer false detections
    min_tracking_confidence: float = 0.7


@dataclass
class CameraConfig:
    """Configuration for camera settings"""
    camera_index: int = 0
    target_width: int = 960  # Reduce resolution for better performance
    target_height: int = 540
    target_fps: int = 30


@dataclass
class DisplayConfig:
    """Configuration for display settings"""
    window_name: str = "Knife Skills Trainer"
    show_fps: bool = True
    show_stats: bool = False  # Disable stats by default for better performance
    show_hand_connections: bool = True
    show_hand_landmarks: bool = True
    show_landmark_labels: bool = False


@dataclass
class CoordinateConfig:
    """Configuration for coordinate transformation"""
    default_pixels_per_mm: float = 3.0
    cutting_board_width_mm: float = 300.0
    cutting_board_height_mm: float = 200.0


@dataclass
class SafetyConfig:
    """Configuration for safety monitoring (Day 3)"""
    green_zone_mm: float = 50.0
    yellow_zone_mm: float = 30.0
    red_zone_mm: float = 15.0
    critical_zone_mm: float = 5.0
    enable_audio_warnings: bool = True
    prediction_time_ms: int = 300


@dataclass
class AppConfig:
    """Main application configuration"""
    hand_detection: HandDetectionConfig
    camera: CameraConfig
    display: DisplayConfig
    coordinates: CoordinateConfig
    safety: SafetyConfig
    
    # Performance settings
    target_fps: int = 20
    enable_performance_monitoring: bool = True
    
    # Debug settings
    debug_mode: bool = False
    save_debug_frames: bool = False


class ConfigManager:
    """
    Manages application configuration with persistence.
    Handles loading, saving, and updating configuration parameters.
    """
    
    def __init__(self, config_file: str = "configs/app_config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Load existing config if available
        if os.path.exists(config_file):
            self.load_config()
        else:
            self.save_config()  # Save default config
    
    def _load_default_config(self) -> AppConfig:
        """Load default configuration"""
        return AppConfig(
            hand_detection=HandDetectionConfig(),
            camera=CameraConfig(),
            display=DisplayConfig(),
            coordinates=CoordinateConfig(),
            safety=SafetyConfig()
        )
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration from loaded data
            self._update_config_from_dict(config_dict)
            print(f"✅ Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
            print("Using default configuration")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_dict = self._config_to_dict()
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
            return False
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "hand_detection": asdict(self.config.hand_detection),
            "camera": asdict(self.config.camera),
            "display": asdict(self.config.display),
            "coordinates": asdict(self.config.coordinates),
            "safety": asdict(self.config.safety),
            "target_fps": self.config.target_fps,
            "enable_performance_monitoring": self.config.enable_performance_monitoring,
            "debug_mode": self.config.debug_mode,
            "save_debug_frames": self.config.save_debug_frames
        }
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        try:
            # Update hand detection config
            if "hand_detection" in config_dict:
                hd_dict = config_dict["hand_detection"]
                self.config.hand_detection = HandDetectionConfig(**hd_dict)
            
            # Update camera config
            if "camera" in config_dict:
                cam_dict = config_dict["camera"]
                self.config.camera = CameraConfig(**cam_dict)
            
            # Update display config
            if "display" in config_dict:
                disp_dict = config_dict["display"]
                self.config.display = DisplayConfig(**disp_dict)
            
            # Update coordinates config
            if "coordinates" in config_dict:
                coord_dict = config_dict["coordinates"]
                self.config.coordinates = CoordinateConfig(**coord_dict)
            
            # Update safety config
            if "safety" in config_dict:
                safety_dict = config_dict["safety"]
                self.config.safety = SafetyConfig(**safety_dict)
            
            # Update app-level settings
            if "target_fps" in config_dict:
                self.config.target_fps = config_dict["target_fps"]
            if "enable_performance_monitoring" in config_dict:
                self.config.enable_performance_monitoring = config_dict["enable_performance_monitoring"]
            if "debug_mode" in config_dict:
                self.config.debug_mode = config_dict["debug_mode"]
            if "save_debug_frames" in config_dict:
                self.config.save_debug_frames = config_dict["save_debug_frames"]
                
        except Exception as e:
            print(f"⚠️ Error updating config from dict: {e}")
    
    def update_hand_detection(self, **kwargs):
        """Update hand detection configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.hand_detection, key):
                setattr(self.config.hand_detection, key, value)
        self.save_config()
    
    def update_camera(self, **kwargs):
        """Update camera configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.camera, key):
                setattr(self.config.camera, key, value)
        self.save_config()
    
    def update_display(self, **kwargs):
        """Update display configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.display, key):
                setattr(self.config.display, key, value)
        self.save_config()
    
    def update_coordinates(self, **kwargs):
        """Update coordinate configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.coordinates, key):
                setattr(self.config.coordinates, key, value)
        self.save_config()
    
    def update_safety(self, **kwargs):
        """Update safety configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config.safety, key):
                setattr(self.config.safety, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._load_default_config()
        self.save_config()
        print("✅ Configuration reset to defaults")
    
    def get_config_summary(self) -> str:
        """Get a summary of current configuration"""
        summary = []
        summary.append("=== Knife Skills Trainer Configuration ===")
        summary.append(f"Hand Detection: Max hands={self.config.hand_detection.max_num_hands}, "
                      f"Complexity={self.config.hand_detection.model_complexity}")
        summary.append(f"Camera: {self.config.camera.target_width}x{self.config.camera.target_height} "
                      f"@ {self.config.camera.target_fps}fps")
        summary.append(f"Display: FPS={self.config.display.show_fps}, "
                      f"Connections={self.config.display.show_hand_connections}")
        summary.append(f"Safety Zones: Green={self.config.safety.green_zone_mm}mm, "
                      f"Red={self.config.safety.red_zone_mm}mm")
        summary.append(f"Target FPS: {self.config.target_fps}")
        return "\n".join(summary)