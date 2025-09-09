"""
Overlay rendering for hand landmarks and visual feedback.
Provides professional visualization of hand tracking data.
"""

from typing import List, Tuple
import cv2
import numpy as np
from ..detection.hand_detector import HandLandmarks


class OverlayRenderer:
    """
    Renders hand landmarks, connections, and visual feedback overlays.
    Provides color-coded visualization for left/right hands.
    """
    
    # Color scheme for visualization
    LEFT_HAND_COLOR = (255, 100, 100)   # Light blue
    RIGHT_HAND_COLOR = (100, 255, 100)  # Light green
    CONNECTION_COLOR = (200, 200, 200)  # Light gray
    LANDMARK_RADIUS = 3  # Smaller for better performance
    CONNECTION_THICKNESS = 1  # Thinner lines for better performance
    LANDMARK_THICKNESS = -1  # Filled circle
    
    # Landmark indices for finger tips (for highlighting)
    FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    
    def __init__(self, show_connections: bool = True, show_landmarks: bool = True, 
                 show_labels: bool = False):
        """
        Initialize overlay renderer.
        
        Args:
            show_connections: Whether to draw hand connections
            show_landmarks: Whether to draw landmark points
            show_labels: Whether to show landmark index labels
        """
        self.show_connections = show_connections
        self.show_landmarks = show_landmarks
        self.show_labels = show_labels
        
        # Hand landmark connections (from MediaPipe)
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
    
    def render_hands(self, frame: np.ndarray, detected_hands: List[HandLandmarks]) -> np.ndarray:
        """
        Render hand overlays on the frame.
        
        Args:
            frame: Input frame to draw on
            detected_hands: List of detected hands with landmarks
            
        Returns:
            Frame with hand overlays drawn
        """
        if not detected_hands:
            return frame
        
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        for hand in detected_hands:
            hand_color = self._get_hand_color(hand.handedness)
            
            # Convert normalized coordinates to pixel coordinates
            pixel_landmarks = self._normalize_to_pixel(hand.landmarks, width, height)
            
            # Draw connections
            if self.show_connections:
                self._draw_connections(overlay_frame, pixel_landmarks, hand_color)
            
            # Draw landmarks
            if self.show_landmarks:
                self._draw_landmarks(overlay_frame, pixel_landmarks, hand_color)
            
            # Draw labels if enabled
            if self.show_labels:
                self._draw_labels(overlay_frame, pixel_landmarks)
            
            # Skip drawing hand info here - will be drawn after mirroring in display module
        
        return overlay_frame
    
    def _get_hand_color(self, handedness: str) -> Tuple[int, int, int]:
        """Get color for hand based on handedness"""
        return self.LEFT_HAND_COLOR if handedness == "Left" else self.RIGHT_HAND_COLOR
    
    def _normalize_to_pixel(self, landmarks: List[Tuple[float, float]], 
                           width: int, height: int) -> List[Tuple[int, int]]:
        """Convert normalized coordinates to pixel coordinates"""
        pixel_coords = []
        for x, y in landmarks:
            px = int(x * width)
            py = int(y * height)
            pixel_coords.append((px, py))
        return pixel_coords
    
    def _draw_connections(self, frame: np.ndarray, pixel_landmarks: List[Tuple[int, int]], 
                         hand_color: Tuple[int, int, int]):
        """Draw connections between hand landmarks"""
        for start_idx, end_idx in self.hand_connections:
            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                start_point = pixel_landmarks[start_idx]
                end_point = pixel_landmarks[end_idx]
                
                cv2.line(frame, start_point, end_point, 
                        self.CONNECTION_COLOR, self.CONNECTION_THICKNESS)
    
    def _draw_landmarks(self, frame: np.ndarray, pixel_landmarks: List[Tuple[int, int]], 
                       hand_color: Tuple[int, int, int]):
        """Draw landmark points"""
        for i, (x, y) in enumerate(pixel_landmarks):
            # Highlight fingertips with larger circles
            radius = self.LANDMARK_RADIUS + 2 if i in self.FINGERTIP_LANDMARKS else self.LANDMARK_RADIUS
            
            cv2.circle(frame, (x, y), radius, hand_color, self.LANDMARK_THICKNESS)
            
            # Draw white border for better visibility
            cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)
    
    def _draw_labels(self, frame: np.ndarray, pixel_landmarks: List[Tuple[int, int]]):
        """Draw landmark index labels"""
        for i, (x, y) in enumerate(pixel_landmarks):
            cv2.putText(frame, str(i), (x + 8, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def _draw_hand_info(self, frame: np.ndarray, hand: HandLandmarks, wrist_pos: Tuple[int, int]):
        """Draw hand information near the wrist"""
        info_text = f"{hand.handedness} ({hand.confidence:.2f})"
        text_pos = (wrist_pos[0] + 20, wrist_pos[1] - 10)
        
        # Draw text background
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        bg_rect = (text_pos[0] - 5, text_pos[1] - text_size[1] - 5,
                   text_pos[0] + text_size[0] + 5, text_pos[1] + 5)
        
        cv2.rectangle(frame, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), 
                     (0, 0, 0), -1)
        
        # Draw text
        hand_color = self._get_hand_color(hand.handedness)
        cv2.putText(frame, info_text, text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
    
    def render_safety_zones(self, frame: np.ndarray, zones: List[dict]) -> np.ndarray:
        """
        Render safety zones (placeholder for future implementation).
        
        Args:
            frame: Input frame
            zones: List of safety zone definitions
            
        Returns:
            Frame with safety zones rendered
        """
        # Placeholder for Day 3 implementation
        return frame
    
    def render_technique_feedback(self, frame: np.ndarray, feedback: dict) -> np.ndarray:
        """
        Render technique feedback overlay (placeholder for future implementation).
        
        Args:
            frame: Input frame
            feedback: Technique feedback data
            
        Returns:
            Frame with feedback overlay
        """
        # Placeholder for Day 4 implementation
        return frame
    
    def set_display_options(self, show_connections: bool = None, 
                           show_landmarks: bool = None, show_labels: bool = None):
        """Update display options"""
        if show_connections is not None:
            self.show_connections = show_connections
        if show_landmarks is not None:
            self.show_landmarks = show_landmarks
        if show_labels is not None:
            self.show_labels = show_labels