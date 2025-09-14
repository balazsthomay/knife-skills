"""
Depth-aware safety monitoring for 3D fingertip-to-blade proximity detection.
Extends 2D safety monitoring with true 3D distance measurements using depth estimation.
"""

from typing import List, Tuple, NamedTuple, Optional
import cv2
import numpy as np

try:
    from .safety_monitor import SafetyMonitor, DangerEvent
    from ..detection.hand_detector import HandLandmarks
    from ..detection.knife_detector import KnifeBoundary
    from ..vision.depth_estimator import DepthEstimator
except ImportError:
    # Handle imports when running from project root
    from safety.safety_monitor import SafetyMonitor, DangerEvent
    from detection.hand_detector import HandLandmarks
    from detection.knife_detector import KnifeBoundary
    from vision.depth_estimator import DepthEstimator


class Hand3D(NamedTuple):
    """Container for 3D hand landmark data"""
    landmarks_3d: List[Tuple[float, float, float]]  # (X, Y, Z) world coordinates in meters
    handedness: str  # "Left" or "Right"
    confidence: float  # Handedness confidence score
    original_hand: HandLandmarks  # Original 2D hand data


class Knife3D(NamedTuple):
    """Container for 3D knife boundary data"""
    tip_3d: Tuple[float, float, float]  # Knife tip in world coordinates (X, Y, Z) meters
    handle_3d: Tuple[float, float, float]  # Handle end in world coordinates
    blade_edge_3d: List[Tuple[float, float, float]]  # Blade edge points in 3D
    confidence: float  # Detection confidence score
    original_knife: KnifeBoundary  # Original 2D knife data


class DangerEvent3D(NamedTuple):
    """Container for a 3D danger event"""
    hand_3d: Hand3D
    fingertip_idx: int  # Landmark index of the dangerous fingertip
    knife_3d: Knife3D
    distance_3d: float  # True 3D distance to blade edge in meters
    distance_2d: float  # Original 2D distance in pixels (for comparison)
    severity: str  # "critical", "high", "medium"


class DepthAwareSafetyMonitor(SafetyMonitor):
    """
    3D safety monitoring using depth estimation for true spatial distance calculations.
    Extends the 2D SafetyMonitor with depth-aware 3D coordinate conversion.
    """
    
    # 3D danger thresholds (in meters - real world distances)
    CRITICAL_THRESHOLD_3D = 0.050  # 50mm - immediate danger
    HIGH_THRESHOLD_3D = 0.100      # 100mm - high risk  
    MEDIUM_THRESHOLD_3D = 0.200    # 200mm - caution zone
    
    def __init__(self,
                 depth_estimator: DepthEstimator,
                 critical_threshold_3d: float = CRITICAL_THRESHOLD_3D,
                 high_threshold_3d: float = HIGH_THRESHOLD_3D,
                 medium_threshold_3d: float = MEDIUM_THRESHOLD_3D,
                 **kwargs):
        """
        Initialize depth-aware safety monitor.
        
        Args:
            depth_estimator: Core ML depth estimator for 3D coordinate conversion
            critical_threshold_3d: 3D distance for critical danger (meters)
            high_threshold_3d: 3D distance for high danger (meters)
            medium_threshold_3d: 3D distance for medium danger (meters)
            **kwargs: Additional arguments passed to parent SafetyMonitor
        """
        # Initialize parent 2D safety monitor
        super().__init__(**kwargs)
        
        # 3D-specific components
        self.depth_estimator = depth_estimator
        self.critical_threshold_3d = critical_threshold_3d
        self.high_threshold_3d = high_threshold_3d
        self.medium_threshold_3d = medium_threshold_3d
        
        # Performance tracking for 3D detection
        self._total_3d_checks = 0
        self._danger_events_3d = 0
        self._depth_estimation_time = 0.0
        
    def detect_3d_fingertip_dangers(self, hands: List[HandLandmarks], 
                                   knives: List[KnifeBoundary],
                                   frame: np.ndarray) -> List[DangerEvent3D]:
        """
        Detect dangerous fingertip-to-blade proximities using 3D depth analysis.
        
        Args:
            hands: List of detected hands with landmarks
            knives: List of detected knives with boundaries
            frame: Current video frame for depth estimation
            
        Returns:
            List of DangerEvent3D objects for dangerous 3D proximities
        """
        dangers_3d = []
        self._total_3d_checks += 1
        
        if not hands or not knives:
            return dangers_3d
        
        # Generate depth map using Core ML (42.8ms average)
        import time
        depth_start = time.time()
        depth_map = self.depth_estimator.estimate_depth(frame)
        self._depth_estimation_time += time.time() - depth_start
        
        # Convert hands to 3D coordinates
        hands_3d = self._convert_hands_to_3d(hands, depth_map, frame.shape)
        
        # Convert knives to 3D coordinates
        knives_3d = self._convert_knives_to_3d(knives, depth_map, frame.shape)
        
        # Detect 3D dangers using true spatial distances
        for hand_3d in hands_3d:
            for fingertip_idx in self.FINGERTIP_LANDMARKS:
                if fingertip_idx < len(hand_3d.landmarks_3d):
                    fingertip_3d = hand_3d.landmarks_3d[fingertip_idx]
                    
                    for knife_3d in knives_3d:
                        danger = self._check_3d_fingertip_knife_distance(
                            hand_3d, fingertip_idx, fingertip_3d, knife_3d
                        )
                        
                        if danger:
                            dangers_3d.append(danger)
                            self._danger_events_3d += 1
        
        return dangers_3d
    
    def _convert_hands_to_3d(self, hands: List[HandLandmarks], 
                            depth_map: np.ndarray, 
                            frame_shape: Tuple[int, int, int]) -> List[Hand3D]:
        """
        Convert 2D hand landmarks to 3D world coordinates.
        
        Args:
            hands: List of 2D hand landmarks
            depth_map: Depth map from Core ML depth estimation
            frame_shape: (height, width, channels) of the video frame
            
        Returns:
            List of Hand3D objects with 3D coordinates
        """
        hands_3d = []
        frame_height, frame_width = frame_shape[:2]
        
        for hand in hands:
            landmarks_3d = []
            
            for landmark_norm in hand.landmarks:
                # Convert normalized coordinates to pixel coordinates
                pixel_x = int(landmark_norm[0] * frame_width)
                pixel_y = int(landmark_norm[1] * frame_height)
                
                # Convert to 3D world coordinates using depth map
                world_coords = self.depth_estimator.get_3d_coordinates(
                    (pixel_x, pixel_y), depth_map
                )
                landmarks_3d.append(world_coords)
            
            hand_3d = Hand3D(
                landmarks_3d=landmarks_3d,
                handedness=hand.handedness,
                confidence=hand.confidence,
                original_hand=hand
            )
            hands_3d.append(hand_3d)
        
        return hands_3d
    
    def _convert_knives_to_3d(self, knives: List[KnifeBoundary],
                             depth_map: np.ndarray,
                             frame_shape: Tuple[int, int, int]) -> List[Knife3D]:
        """
        Convert 2D knife boundaries to 3D world coordinates.
        
        Args:
            knives: List of 2D knife boundaries
            depth_map: Depth map from Core ML depth estimation  
            frame_shape: (height, width, channels) of the video frame
            
        Returns:
            List of Knife3D objects with 3D coordinates
        """
        knives_3d = []
        
        for knife in knives:
            # Convert key points to 3D
            tip_3d = self.depth_estimator.get_3d_coordinates(
                knife.tip_point, depth_map
            )
            handle_3d = self.depth_estimator.get_3d_coordinates(
                knife.handle_point, depth_map
            )
            
            # Convert blade edge contour to 3D points
            blade_edge_3d = []
            if knife.contour is not None and len(knife.contour) > 0:
                # Sample points along the contour for 3D conversion
                contour_points = knife.contour.reshape(-1, 2)
                # Subsample to avoid excessive computation (every 5th point)
                sampled_points = contour_points[::5]
                
                for point in sampled_points:
                    point_3d = self.depth_estimator.get_3d_coordinates(
                        (int(point[0]), int(point[1])), depth_map
                    )
                    blade_edge_3d.append(point_3d)
            
            knife_3d = Knife3D(
                tip_3d=tip_3d,
                handle_3d=handle_3d,
                blade_edge_3d=blade_edge_3d,
                confidence=knife.confidence,
                original_knife=knife
            )
            knives_3d.append(knife_3d)
        
        return knives_3d
    
    def _check_3d_fingertip_knife_distance(self, hand_3d: Hand3D,
                                          fingertip_idx: int,
                                          fingertip_3d: Tuple[float, float, float],
                                          knife_3d: Knife3D) -> Optional[DangerEvent3D]:
        """
        Check 3D distance between fingertip and knife blade edge.
        
        Args:
            hand_3d: 3D hand containing the fingertip
            fingertip_idx: Index of the fingertip landmark
            fingertip_3d: 3D coordinates of fingertip (X, Y, Z) in meters
            knife_3d: 3D knife to check distance against
            
        Returns:
            DangerEvent3D if dangerous, None if safe
        """
        # Calculate minimum 3D distance to blade edge
        min_distance_3d = float('inf')
        closest_blade_point = None
        
        if knife_3d.blade_edge_3d:
            for blade_point_3d in knife_3d.blade_edge_3d:
                distance = self.depth_estimator.calculate_3d_distance(
                    fingertip_3d, blade_point_3d
                )
                if distance < min_distance_3d:
                    min_distance_3d = distance
                    closest_blade_point = blade_point_3d
        
        # Also check distance to tip and handle as fallback
        tip_distance = self.depth_estimator.calculate_3d_distance(
            fingertip_3d, knife_3d.tip_3d
        )
        handle_distance = self.depth_estimator.calculate_3d_distance(
            fingertip_3d, knife_3d.handle_3d
        )
        
        # Check if tip or handle is closest
        if tip_distance < min_distance_3d:
            min_distance_3d = tip_distance
            closest_blade_point = knife_3d.tip_3d
        if handle_distance < min_distance_3d:
            min_distance_3d = handle_distance
            closest_blade_point = knife_3d.handle_3d
        
        
        # Determine 3D severity level
        severity = self._classify_3d_danger_severity(min_distance_3d)
        
        if severity:
            # Also calculate 2D distance for comparison
            distance_2d = self._calculate_blade_distance(
                knife_3d.original_knife, 
                (int(hand_3d.original_hand.landmarks[fingertip_idx][0] * 1280),  # Assume 1280 width
                 int(hand_3d.original_hand.landmarks[fingertip_idx][1] * 720))   # Assume 720 height
            )
            
            
            return DangerEvent3D(
                hand_3d=hand_3d,
                fingertip_idx=fingertip_idx,
                knife_3d=knife_3d,
                distance_3d=min_distance_3d,
                distance_2d=distance_2d,
                severity=severity
            )
        
        return None
    
    def _classify_3d_danger_severity(self, distance_3d: float) -> Optional[str]:
        """
        Classify danger severity based on 3D distance in meters.
        
        Args:
            distance_3d: Distance in meters
            
        Returns:
            Severity level string or None if safe
        """
        if distance_3d <= self.critical_threshold_3d:
            return "critical"
        elif distance_3d <= self.high_threshold_3d:
            return "high"
        elif distance_3d <= self.medium_threshold_3d:
            return "medium"
        else:
            return None  # Safe distance
    
    def get_3d_stats(self) -> dict:
        """Get 3D safety monitoring statistics"""
        danger_rate_3d = (self._danger_events_3d / self._total_3d_checks 
                         if self._total_3d_checks > 0 else 0)
        
        avg_depth_time = (self._depth_estimation_time / self._total_3d_checks
                         if self._total_3d_checks > 0 else 0)
        
        return {
            "total_3d_checks": self._total_3d_checks,
            "danger_events_3d": self._danger_events_3d,
            "danger_rate_3d": danger_rate_3d,
            "avg_depth_estimation_time_ms": avg_depth_time * 1000,
            "thresholds_3d_meters": {
                "critical": self.critical_threshold_3d,
                "high": self.high_threshold_3d,
                "medium": self.medium_threshold_3d
            }
        }
    
    def reset_3d_stats(self):
        """Reset 3D monitoring statistics"""
        self._total_3d_checks = 0
        self._danger_events_3d = 0
        self._depth_estimation_time = 0.0
    
    def compare_2d_vs_3d_detection(self, hands: List[HandLandmarks], 
                                  knives: List[KnifeBoundary],
                                  frame: np.ndarray) -> dict:
        """
        Compare 2D vs 3D danger detection for accuracy analysis.
        
        Args:
            hands: List of detected hands
            knives: List of detected knives
            frame: Current video frame
            
        Returns:
            Comparison statistics and detected dangers
        """
        # Run 2D detection
        dangers_2d = self.detect_fingertip_dangers(hands, knives)
        
        # Run 3D detection  
        dangers_3d = self.detect_3d_fingertip_dangers(hands, knives, frame)
        
        return {
            "dangers_2d": dangers_2d,
            "dangers_3d": dangers_3d,
            "count_2d": len(dangers_2d),
            "count_3d": len(dangers_3d),
            "difference": len(dangers_2d) - len(dangers_3d),
            "false_positive_reduction": max(0, len(dangers_2d) - len(dangers_3d))
        }