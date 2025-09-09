#!/usr/bin/env python3
"""
Knife Skills Trainer - Main Application
Day 1: Foundation with real-time hand detection and overlay visualization.
"""

import sys
import time
from typing import Optional
import cv2

from src.detection.hand_detector import HandDetector
from src.ui.video_display import VideoCapture, VideoDisplay
from src.ui.overlay_renderer import OverlayRenderer
from src.utils.coordinates import CoordinateTransform, DistanceCalculator
from src.utils.config import ConfigManager


class KnifeSkillsTrainer:
    """
    Main application class for the Knife Skills Trainer.
    Integrates hand detection, video display, and coordinate systems.
    """
    
    def __init__(self):
        """Initialize the knife skills trainer application"""
        print("🔪 Initializing Knife Skills Trainer...")
        
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        # Initialize components
        self.hand_detector = None
        self.video_capture = None
        self.video_display = None
        self.overlay_renderer = None
        self.coordinate_transform = None
        self.distance_calculator = None
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"✅ Configuration loaded:")
        print(self.config_manager.get_config_summary())
    
    def initialize_components(self) -> bool:
        """Initialize all application components"""
        try:
            # Initialize hand detector
            self.hand_detector = HandDetector(
                max_num_hands=self.config.hand_detection.max_num_hands,
                model_complexity=self.config.hand_detection.model_complexity,
                min_detection_confidence=self.config.hand_detection.min_detection_confidence,
                min_tracking_confidence=self.config.hand_detection.min_tracking_confidence
            )
            print("✅ Hand detector initialized")
            
            # Initialize video capture
            self.video_capture = VideoCapture(
                camera_index=self.config.camera.camera_index,
                target_width=self.config.camera.target_width,
                target_height=self.config.camera.target_height
            )
            
            if not self.video_capture.is_opened:
                raise RuntimeError("Failed to initialize camera")
            print("✅ Video capture initialized")
            
            # Initialize video display
            self.video_display = VideoDisplay(
                window_name=self.config.display.window_name
            )
            self.video_display.show_fps = self.config.display.show_fps
            self.video_display.show_stats = self.config.display.show_stats
            print("✅ Video display initialized")
            
            # Initialize overlay renderer
            self.overlay_renderer = OverlayRenderer(
                show_connections=self.config.display.show_hand_connections,
                show_landmarks=self.config.display.show_hand_landmarks,
                show_labels=self.config.display.show_landmark_labels
            )
            print("✅ Overlay renderer initialized")
            
            # Initialize coordinate system
            self.coordinate_transform = CoordinateTransform(
                pixels_per_mm=self.config.coordinates.default_pixels_per_mm
            )
            self.distance_calculator = DistanceCalculator(self.coordinate_transform)
            print("✅ Coordinate system initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            return False
    
    def process_frame(self) -> bool:
        """
        Process a single frame.
        
        Returns:
            True if frame processed successfully, False otherwise
        """
        # Read frame from camera
        success, frame = self.video_capture.read_frame()
        if not success:
            return False
        
        self.frame_count += 1
        
        # Skip processing if paused
        if self.is_paused:
            self.video_display.display_frame(frame, 0.0, "PAUSED")
            return True
        
        # Detect hands
        detected_hands = self.hand_detector.detect_hands(frame)
        
        # Render hand overlays
        overlay_frame = self.overlay_renderer.render_hands(frame, detected_hands)
        
        # Prepare stats text
        stats_text = self._generate_stats_text(detected_hands)
        
        # Display frame
        current_fps = self.video_capture.get_fps()
        self.video_display.display_frame(overlay_frame, current_fps, stats_text, detected_hands)
        
        return True
    
    def _generate_stats_text(self, detected_hands) -> str:
        """Generate statistics text for display"""
        if not self.config.display.show_stats:
            return ""
        
        stats = []
        stats.append(f"Hands: {len(detected_hands)}")
        
        # Hand details
        for i, hand in enumerate(detected_hands):
            stats.append(f"  {hand.handedness}: {hand.confidence:.2f}")
        
        # Performance stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        stats.append(f"Avg FPS: {avg_fps:.1f}")
        
        # Detection stats
        detector_stats = self.hand_detector.get_stats()
        stats.append(f"Frames: {detector_stats['frames_processed']}")
        
        return "\n".join(stats)
    
    def handle_keyboard_input(self) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            True to continue running, False to exit
        """
        key = self.video_display.wait_key(1)
        
        if key == 27:  # ESC key
            return False
        elif key == ord(' '):  # Space key - pause/resume
            self.is_paused = not self.is_paused
            print(f"{'⏸️  Paused' if self.is_paused else '▶️  Resumed'}")
        elif key == ord('h'):  # Help
            self._show_help()
        elif key == ord('s'):  # Statistics toggle
            self.config.display.show_stats = not self.config.display.show_stats
            self.video_display.show_stats = self.config.display.show_stats
        elif key == ord('f'):  # FPS toggle
            self.config.display.show_fps = not self.config.display.show_fps
            self.video_display.show_fps = self.config.display.show_fps
        elif key == ord('l'):  # Labels toggle
            current_labels = self.overlay_renderer.show_labels
            self.overlay_renderer.set_display_options(show_labels=not current_labels)
        elif key == ord('c'):  # Connections toggle
            current_connections = self.overlay_renderer.show_connections
            self.overlay_renderer.set_display_options(show_connections=not current_connections)
        
        return True
    
    def _show_help(self):
        """Display help information"""
        help_text = """
🔪 Knife Skills Trainer - Controls:
  ESC     - Exit application
  SPACE   - Pause/Resume
  H       - Show this help
  S       - Toggle statistics display
  F       - Toggle FPS display
  L       - Toggle landmark labels
  C       - Toggle hand connections
        """
        print(help_text)
    
    def run(self) -> int:
        """
        Main application loop.
        
        Returns:
            Exit code (0 for success)
        """
        if not self.initialize_components():
            return 1
        
        print("\n🚀 Starting Knife Skills Trainer...")
        print("Press 'H' for help, 'ESC' to exit")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            while self.is_running:
                # Process frame
                if not self.process_frame():
                    print("⚠️ Failed to process frame")
                    break
                
                # Handle input
                if not self.handle_keyboard_input():
                    break
                
                # Remove artificial frame rate limiting for maximum performance
                # Let the system run as fast as possible
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
        
        finally:
            self.cleanup()
        
        print("👋 Application closed successfully")
        return 0
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.video_capture:
            self.video_capture.release()
        
        if self.video_display:
            self.video_display.destroy_window()
        
        # Clean up OpenCV
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    try:
        app = KnifeSkillsTrainer()
        return app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())