"""
Overlay rendering for hand landmarks and visual feedback.
Provides professional visualization of hand tracking data.
"""

from typing import List, Tuple
import cv2
import numpy as np

try:
    from ..detection.hand_detector import HandLandmarks
    from ..detection.knife_detector import KnifeBoundary
except ImportError:
    # Handle imports when running from project root
    from detection.hand_detector import HandLandmarks
    from detection.knife_detector import KnifeBoundary


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
    
    # Knife visualization colors
    KNIFE_BBOX_COLOR = (0, 255, 255)    # Cyan for bounding box
    KNIFE_CONTOUR_COLOR = (0, 165, 255)  # Orange for blade contour
    KNIFE_TIP_COLOR = (0, 0, 255)       # Red for knife tip
    KNIFE_HANDLE_COLOR = (255, 0, 0)    # Blue for handle
    KNIFE_MASK_ALPHA = 0.3               # Transparency for mask overlay
    
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
    
    def render_knives(self, frame: np.ndarray, detected_knives: List[KnifeBoundary]) -> np.ndarray:
        """
        Render knife overlays on the frame.
        
        Args:
            frame: Input frame to draw on
            detected_knives: List of detected knives with boundaries
            
        Returns:
            Frame with knife overlays drawn
        """
        if not detected_knives:
            return frame
            
        overlay_frame = frame.copy()
        
        for knife in detected_knives:
            # Draw bounding box
            x1, y1, x2, y2 = knife.bbox
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), self.KNIFE_BBOX_COLOR, 2)
            
            # Draw segmentation mask with transparency
            if knife.mask is not None:
                mask_colored = np.zeros_like(overlay_frame)
                mask_colored[knife.mask > 0] = self.KNIFE_CONTOUR_COLOR
                overlay_frame = cv2.addWeighted(overlay_frame, 1.0, mask_colored, self.KNIFE_MASK_ALPHA, 0)
            
            # Draw contour outline
            if knife.contour is not None:
                cv2.drawContours(overlay_frame, [knife.contour], -1, self.KNIFE_CONTOUR_COLOR, 2)
            
            # Draw tip and handle points
            cv2.circle(overlay_frame, knife.tip_point, 6, self.KNIFE_TIP_COLOR, -1)
            cv2.circle(overlay_frame, knife.handle_point, 6, self.KNIFE_HANDLE_COLOR, -1)
            
            # Draw tip and handle labels
            cv2.putText(overlay_frame, "TIP", 
                       (knife.tip_point[0] + 10, knife.tip_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.KNIFE_TIP_COLOR, 2)
            cv2.putText(overlay_frame, "HANDLE", 
                       (knife.handle_point[0] + 10, knife.handle_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.KNIFE_HANDLE_COLOR, 2)
            
            # Draw confidence and blade length info
            info_text = f"Knife: {knife.confidence:.2f} | {knife.blade_length_pixels:.0f}px"
            text_pos = (x1, y1 - 10)
            
            # Draw text background
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay_frame, 
                         (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
                         (text_pos[0] + text_size[0] + 5, text_pos[1] + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay_frame, info_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.KNIFE_BBOX_COLOR, 2)
        
        return overlay_frame
    
    def render_safety_dangers(self, frame: np.ndarray, danger_events: List) -> np.ndarray:
        """
        Render safety danger indicators for dangerous fingertip proximities.
        
        Args:
            frame: Input frame to draw on
            danger_events: List of DangerEvent objects from SafetyMonitor
            
        Returns:
            Frame with danger indicators drawn
        """
        if not danger_events:
            return frame
            
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Group dangers by hand and fingertip for efficient rendering
        danger_fingertips = {}
        warning_messages = set()
        
        for danger in danger_events:
            hand_id = id(danger.hand)  # Use object id as unique hand identifier
            if hand_id not in danger_fingertips:
                danger_fingertips[hand_id] = {
                    'hand': danger.hand,
                    'dangerous_tips': {}
                }
            
            # Store the most severe danger for each fingertip
            if (danger.fingertip_idx not in danger_fingertips[hand_id]['dangerous_tips'] or
                self._is_more_severe(danger.severity, 
                                   danger_fingertips[hand_id]['dangerous_tips'][danger.fingertip_idx]['severity'])):
                danger_fingertips[hand_id]['dangerous_tips'][danger.fingertip_idx] = {
                    'severity': danger.severity,
                    'distance': danger.distance
                }
            
            # Collect warning messages
            warning_messages.add(f"{danger.severity.upper()}: Finger near blade!")
        
        # Render danger indicators for each dangerous fingertip
        for hand_data in danger_fingertips.values():
            hand = hand_data['hand']
            dangerous_tips = hand_data['dangerous_tips']
            
            # Convert normalized coordinates to pixel coordinates  
            pixel_landmarks = self._normalize_to_pixel(hand.landmarks, width, height)
            
            for fingertip_idx, danger_info in dangerous_tips.items():
                if fingertip_idx < len(pixel_landmarks):
                    x, y = pixel_landmarks[fingertip_idx]
                    
                    # Get danger color based on severity
                    danger_color = self._get_danger_color(danger_info['severity'])
                    
                    # Draw enlarged danger circle
                    danger_radius = self.LANDMARK_RADIUS + 4  # Larger than normal landmarks
                    cv2.circle(overlay_frame, (x, y), danger_radius, danger_color, -1)
                    
                    # Draw white border for visibility
                    cv2.circle(overlay_frame, (x, y), danger_radius + 2, (255, 255, 255), 2)
                    
                    # Add pulsing effect for critical dangers
                    if danger_info['severity'] == 'critical':
                        pulse_radius = danger_radius + 6
                        cv2.circle(overlay_frame, (x, y), pulse_radius, danger_color, 2)
        
        # Draw warning messages
        self._draw_danger_warnings(overlay_frame, warning_messages)
        
        return overlay_frame
    
    def _is_more_severe(self, severity1: str, severity2: str) -> bool:
        """Check if severity1 is more severe than severity2"""
        severity_order = {'critical': 3, 'high': 2, 'medium': 1}
        return severity_order.get(severity1, 0) > severity_order.get(severity2, 0)
    
    def _get_danger_color(self, severity: str) -> Tuple[int, int, int]:
        """Get BGR color for danger severity level"""
        colors = {
            "critical": (0, 0, 255),      # Bright red
            "high": (0, 69, 255),         # Red-orange  
            "medium": (0, 165, 255),      # Orange
        }
        return colors.get(severity, (255, 255, 255))  # White fallback
    
    def _draw_danger_warnings(self, frame: np.ndarray, warnings: set):
        """Draw danger warning messages on the frame"""
        if not warnings:
            return
            
        # Position warnings at top-center of frame
        y_start = 60
        line_height = 30
        
        for i, warning in enumerate(sorted(warnings)):
            # Draw text background
            text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = y_start + i * line_height
            
            # Background rectangle
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 10, text_y + 5),
                         (0, 0, 0), -1)
            
            # Warning text in bright red
            cv2.putText(frame, warning, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def draw_hands(self, frame: np.ndarray, hands: List[HandLandmarks]):
        """Draw hand landmarks and connections on frame"""
        if not hands:
            return
        
        rendered_frame = self.render_hands(frame, hands)
        # Copy rendered overlays back to original frame
        frame[:] = rendered_frame
    
    def draw_knives(self, frame: np.ndarray, knives: List[KnifeBoundary]):
        """Draw knife detection overlays on frame"""
        if not knives:
            return
        
        rendered_frame = self.render_knives(frame, knives)
        # Copy rendered overlays back to original frame
        frame[:] = rendered_frame
    
    def draw_danger_warning(self, frame: np.ndarray, danger_event):
        """Draw 2D danger warning for a single danger event"""
        dangers = [danger_event] if danger_event else []
        rendered_frame = self.render_safety_dangers(frame, dangers)
        # Copy rendered overlays back to original frame
        frame[:] = rendered_frame
    
    def draw_3d_danger_warning(self, frame: np.ndarray, danger_event_3d):
        """Draw 3D danger warning with depth information"""
        if not danger_event_3d:
            return
        
        # Convert 3D hand landmarks back to 2D for visualization
        hand = danger_event_3d.hand_3d.original_hand
        fingertip_idx = danger_event_3d.fingertip_idx
        severity = danger_event_3d.severity
        distance_3d = danger_event_3d.distance_3d
        distance_2d = danger_event_3d.distance_2d
        
        height, width = frame.shape[:2]
        pixel_landmarks = self._normalize_to_pixel(hand.landmarks, width, height)
        
        if fingertip_idx < len(pixel_landmarks):
            x, y = pixel_landmarks[fingertip_idx]
            
            # Get danger color
            danger_color = self._get_danger_color(severity)
            
            # Draw enhanced 3D danger indicator
            danger_radius = self.LANDMARK_RADIUS + 6  # Larger for 3D
            cv2.circle(frame, (x, y), danger_radius, danger_color, -1)
            
            # Draw white border
            cv2.circle(frame, (x, y), danger_radius + 2, (255, 255, 255), 2)
            
            # Draw depth-aware danger ring
            depth_ring_radius = danger_radius + 8
            cv2.circle(frame, (x, y), depth_ring_radius, (0, 255, 255), 2)  # Cyan for 3D
            
            # Add pulsing effect for critical 3D dangers
            if severity == 'critical':
                pulse_radius = danger_radius + 12
                cv2.circle(frame, (x, y), pulse_radius, danger_color, 3)
            
            # Draw 3D distance information
            distance_text = f"3D: {distance_3d*1000:.1f}mm"
            text_pos = (x + 15, y - 10)
            
            # Text background
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                         (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                         (0, 0, 0), -1)
            
            # Distance text
            cv2.putText(frame, distance_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Comparison text (2D vs 3D)
            if distance_2d > 0:
                comparison_text = f"2D: {distance_2d:.0f}px"
                comp_pos = (x + 15, y + 15)
                cv2.putText(frame, comparison_text, comp_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw 3D warning message
        warning_text = f"3D {severity.upper()}: {distance_3d*1000:.1f}mm to blade!"
        self._draw_3d_warning_message(frame, warning_text, severity)
    
    def _draw_3d_warning_message(self, frame: np.ndarray, warning_text: str, severity: str):
        """Draw 3D-specific warning message"""
        # Position at top-center
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 120  # Below other warnings
        
        # Background rectangle with 3D indicator color
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 10, text_y + 5),
                     (40, 40, 40), -1)  # Dark background
        
        # Border with danger color
        danger_color = self._get_danger_color(severity)
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 10, text_y + 5),
                     danger_color, 2)
        
        # 3D warning text with cyan accent
        cv2.putText(frame, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def set_display_options(self, show_connections: bool = None, 
                           show_landmarks: bool = None, show_labels: bool = None):
        """Update display options"""
        if show_connections is not None:
            self.show_connections = show_connections
        if show_landmarks is not None:
            self.show_landmarks = show_landmarks
        if show_labels is not None:
            self.show_labels = show_labels