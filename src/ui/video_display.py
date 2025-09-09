"""
Video capture and display management.
Handles camera initialization, frame capture, and performance monitoring.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoCapture:
    """
    Manages video capture from camera with performance optimization.
    Provides frame rate monitoring and error handling.
    """
    
    def __init__(self, camera_index: int = 0, target_width: int = 1280, target_height: int = 720):
        """
        Initialize video capture.
        
        Args:
            camera_index: Camera device index (0 for default)
            target_width: Desired frame width
            target_height: Desired frame height
        """
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        
        self.cap = None
        self.is_opened = False
        
        # Performance tracking
        self._frame_count = 0
        self._start_time = time.time()
        self._last_fps_update = time.time()
        self._current_fps = 0.0
        
        self._initialize_camera()
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.is_opened = False
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success indicates if frame was read
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self._frame_count += 1
            self._update_fps()
            return True, frame
        else:
            print("⚠️ Warning: Failed to read frame from camera")
            return False, None
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self._last_fps_update >= 1.0:  # Update every second
            elapsed = current_time - self._start_time
            if elapsed > 0:
                self._current_fps = self._frame_count / elapsed
            self._last_fps_update = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self._current_fps
    
    def get_frame_count(self) -> int:
        """Get total frames captured"""
        return self._frame_count
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        if not self.is_opened or self.cap is None:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("📷 Camera released")
    
    def __del__(self):
        """Ensure camera is released on deletion"""
        self.release()


class VideoDisplay:
    """
    Manages video display window with overlay support.
    Handles window creation, frame display, and keyboard input.
    """
    
    def __init__(self, window_name: str = "Knife Skills Trainer"):
        """
        Initialize video display.
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.is_window_created = False
        
        # Display settings
        self.show_fps = True
        self.show_stats = True
        
    def create_window(self, width: int = 1280, height: int = 720):
        """Create display window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.is_window_created = True
        print(f"🖼️ Display window created: {self.window_name}")
    
    def display_frame(self, frame: np.ndarray, fps: float = 0.0, stats_text: str = "", detected_hands=None):
        """
        Display frame with optional overlays.
        
        Args:
            frame: Frame to display
            fps: Current FPS for overlay
            stats_text: Additional stats text to display
            detected_hands: List of detected hands for drawing hand info
        """
        if not self.is_window_created:
            height, width = frame.shape[:2]
            self.create_window(width, height)
        
        # Mirror the frame first for natural interaction
        mirrored_frame = cv2.flip(frame, 1)
        height, width = mirrored_frame.shape[:2]
        
        # Add hand info text (after mirroring so text is readable)
        if detected_hands:
            self._draw_hand_info_on_mirrored_frame(mirrored_frame, detected_hands, width, height)
        
        # Add FPS overlay (after mirroring so text is readable)
        if self.show_fps and fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(mirrored_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add stats overlay (after mirroring so text is readable)
        if self.show_stats and stats_text:
            y_offset = 70
            for line in stats_text.split('\n'):
                if line.strip():
                    cv2.putText(mirrored_frame, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25

        cv2.imshow(self.window_name, mirrored_frame)
    
    def _draw_hand_info_on_mirrored_frame(self, mirrored_frame, detected_hands, width, height):
        """Draw hand info text on the mirrored frame"""
        for hand in detected_hands:
            # Get wrist position (landmark 0) and mirror it
            wrist_x_normalized, wrist_y_normalized = hand.landmarks[0]
            original_wrist_x = int(wrist_x_normalized * width)
            original_wrist_y = int(wrist_y_normalized * height)
            
            # Mirror the x coordinate for the mirrored display
            mirrored_wrist_x = width - original_wrist_x
            
            # Draw hand info text
            info_text = f"{hand.handedness} ({hand.confidence:.2f})"
            text_pos = (mirrored_wrist_x + 20, original_wrist_y - 10)
            
            # Ensure text stays within frame bounds
            if text_pos[0] + 100 > width:
                text_pos = (mirrored_wrist_x - 120, original_wrist_y - 10)
            
            # Draw text background for better visibility
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            bg_rect = (text_pos[0] - 5, text_pos[1] - text_size[1] - 5,
                       text_pos[0] + text_size[0] + 5, text_pos[1] + 5)
            
            cv2.rectangle(mirrored_frame, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), 
                         (0, 0, 0), -1)
            
            # Get hand color
            hand_color = (255, 100, 100) if hand.handedness == "Left" else (100, 255, 100)
            cv2.putText(mirrored_frame, info_text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press"""
        return cv2.waitKey(delay) & 0xFF
    
    def destroy_window(self):
        """Close display window"""
        if self.is_window_created:
            cv2.destroyWindow(self.window_name)
            self.is_window_created = False
            print(f"🗑️ Window destroyed: {self.window_name}")
    
    def __del__(self):
        """Ensure window is destroyed on deletion"""
        self.destroy_window()