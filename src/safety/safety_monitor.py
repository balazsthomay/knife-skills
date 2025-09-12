"""
Safety monitoring for fingertip-to-blade proximity detection.
Provides real-time danger detection for knife skills training.
"""

from typing import List, Tuple, NamedTuple
import cv2
import numpy as np

try:
    from ..detection.hand_detector import HandLandmarks
    from ..detection.knife_detector import KnifeBoundary
except ImportError:
    # Handle imports when running from project root
    from detection.hand_detector import HandLandmarks
    from detection.knife_detector import KnifeBoundary


class DangerEvent(NamedTuple):
    """Container for a detected danger event"""
    hand: HandLandmarks
    fingertip_idx: int  # Landmark index of the dangerous fingertip
    knife: KnifeBoundary
    distance: float  # Distance to blade edge in pixels
    severity: str  # "critical", "high", "medium"


class SafetyMonitor:
    """
    Monitors fingertip proximity to knife blade edges.
    Provides configurable threshold-based danger detection.
    """
    
    # Fingertip landmark indices (MediaPipe hand landmarks)
    FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    
    # Danger threshold levels (in pixels)
    CRITICAL_THRESHOLD = 30   # Immediate danger
    HIGH_THRESHOLD = 50       # High risk
    MEDIUM_THRESHOLD = 80     # Caution zone
    
    def __init__(self, 
                 critical_threshold: float = CRITICAL_THRESHOLD,
                 high_threshold: float = HIGH_THRESHOLD,
                 medium_threshold: float = MEDIUM_THRESHOLD):
        """
        Initialize safety monitor.
        
        Args:
            critical_threshold: Distance for critical danger (pixels)
            high_threshold: Distance for high danger (pixels)
            medium_threshold: Distance for medium danger (pixels)
        """
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        
        # Performance tracking
        self._total_checks = 0
        self._danger_events = 0
        
    def detect_fingertip_dangers(self, hands: List[HandLandmarks], 
                                knives: List[KnifeBoundary]) -> List[DangerEvent]:
        """
        Detect dangerous fingertip-to-blade proximities.
        
        Args:
            hands: List of detected hands with landmarks
            knives: List of detected knives with boundaries
            
        Returns:
            List of DangerEvent objects for dangerous proximities
        """
        dangers = []
        self._total_checks += 1
        
        if not hands or not knives:
            return dangers
        
        for hand in hands:
            for fingertip_idx in self.FINGERTIP_LANDMARKS:
                if fingertip_idx < len(hand.landmarks):
                    # Get fingertip position in normalized coordinates
                    fingertip_norm = hand.landmarks[fingertip_idx]
                    
                    for knife in knives:
                        # Convert normalized coordinates to pixels
                        # Note: This assumes frame dimensions, we'll get them from knife bbox
                        danger = self._check_fingertip_knife_distance(
                            hand, fingertip_idx, fingertip_norm, knife
                        )
                        
                        if danger:
                            dangers.append(danger)
                            self._danger_events += 1
        
        return dangers
    
    def _check_fingertip_knife_distance(self, hand: HandLandmarks, 
                                       fingertip_idx: int,
                                       fingertip_norm: Tuple[float, float],
                                       knife: KnifeBoundary) -> DangerEvent:
        """
        Check distance between a specific fingertip and knife blade.
        
        Args:
            hand: Hand containing the fingertip
            fingertip_idx: Index of the fingertip landmark
            fingertip_norm: Normalized (x, y) coordinates of fingertip
            knife: Knife to check distance against
            
        Returns:
            DangerEvent if dangerous, None if safe
        """
        # Estimate frame dimensions from knife bounding box
        # This is a reasonable approximation for pixel conversion
        x1, y1, x2, y2 = knife.bbox
        frame_width = max(x2, 1280)  # Fallback to common resolution
        frame_height = max(y2, 720)
        
        # Convert normalized coordinates to pixel coordinates
        fingertip_x = int(fingertip_norm[0] * frame_width)
        fingertip_y = int(fingertip_norm[1] * frame_height)
        fingertip_pixel = (fingertip_x, fingertip_y)
        
        # Calculate distance to blade edge
        distance = self._calculate_blade_distance(knife, fingertip_pixel)
        
        # Determine severity level
        severity = self._classify_danger_severity(distance)
        
        if severity:
            return DangerEvent(
                hand=hand,
                fingertip_idx=fingertip_idx,
                knife=knife,
                distance=distance,
                severity=severity
            )
        
        return None
    
    def _calculate_blade_distance(self, knife: KnifeBoundary, point: Tuple[int, int]) -> float:
        """
        Calculate minimum distance from a point to the knife blade edge.
        
        Args:
            knife: KnifeBoundary object
            point: (x, y) coordinates to measure distance from
            
        Returns:
            Minimum distance to blade edge in pixels
        """
        if knife.contour is None or len(knife.contour) == 0:
            return float('inf')
        
        # Calculate distance to contour using OpenCV
        point_array = np.array([[point[0], point[1]]], dtype=np.float32)
        distance = cv2.pointPolygonTest(knife.contour, point, True)
        
        # Return absolute distance (positive = outside, negative = inside)
        return abs(distance)
    
    def _classify_danger_severity(self, distance: float) -> str:
        """
        Classify danger severity based on distance.
        
        Args:
            distance: Distance in pixels
            
        Returns:
            Severity level string or None if safe
        """
        if distance <= self.critical_threshold:
            return "critical"
        elif distance <= self.high_threshold:
            return "high"
        elif distance <= self.medium_threshold:
            return "medium"
        else:
            return None  # Safe distance
    
    def get_danger_color(self, severity: str) -> Tuple[int, int, int]:
        """
        Get BGR color for danger severity level.
        
        Args:
            severity: Danger severity level
            
        Returns:
            BGR color tuple
        """
        colors = {
            "critical": (0, 0, 255),      # Bright red
            "high": (0, 69, 255),         # Red-orange  
            "medium": (0, 165, 255),      # Orange
        }
        return colors.get(severity, (255, 255, 255))  # White fallback
    
    def get_stats(self) -> dict:
        """Get safety monitoring statistics"""
        danger_rate = (self._danger_events / self._total_checks 
                      if self._total_checks > 0 else 0)
        
        return {
            "total_checks": self._total_checks,
            "danger_events": self._danger_events,
            "danger_rate": danger_rate,
            "thresholds": {
                "critical": self.critical_threshold,
                "high": self.high_threshold,
                "medium": self.medium_threshold
            }
        }
    
    def reset_stats(self):
        """Reset monitoring statistics"""
        self._total_checks = 0
        self._danger_events = 0