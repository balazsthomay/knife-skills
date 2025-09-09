"""
Coordinate transformation utilities.
Handles conversion between pixel coordinates and real-world measurements.
"""

import math
from typing import Tuple, Optional, List
import numpy as np


class CoordinateTransform:
    """
    Manages coordinate transformations between pixel and real-world coordinates.
    Provides foundation for cutting board calibration and distance measurements.
    """
    
    def __init__(self, pixels_per_mm: float = 3.0):
        """
        Initialize coordinate transform.
        
        Args:
            pixels_per_mm: Conversion factor (placeholder, will be calibrated later)
        """
        self.pixels_per_mm = pixels_per_mm
        self.is_calibrated = False
        
        # Calibration parameters (placeholders for Day 2)
        self.calibration_matrix = None
        self.cutting_board_corners = None
        self.reference_size_mm = None
    
    def pixel_to_mm(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to millimeters.
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            Distance in millimeters
        """
        return pixel_distance / self.pixels_per_mm
    
    def mm_to_pixel(self, mm_distance: float) -> float:
        """
        Convert millimeter distance to pixels.
        
        Args:
            mm_distance: Distance in millimeters
            
        Returns:
            Distance in pixels
        """
        return mm_distance * self.pixels_per_mm
    
    def euclidean_distance_2d(self, point1: Tuple[float, float], 
                             point2: Tuple[float, float]) -> float:
        """
        Calculate 2D Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance in same units as input points
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def euclidean_distance_3d(self, point1: Tuple[float, float, float], 
                             point2: Tuple[float, float, float]) -> float:
        """
        Calculate 3D Euclidean distance between two points.
        
        Args:
            point1: First point (x, y, z)
            point2: Second point (x, y, z)
            
        Returns:
            Distance in same units as input points
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        dz = point2[2] - point1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def normalize_coordinates(self, points: List[Tuple[float, float]], 
                            width: int, height: int) -> List[Tuple[float, float]]:
        """
        Normalize pixel coordinates to [0, 1] range.
        
        Args:
            points: List of (x, y) pixel coordinates
            width: Image width
            height: Image height
            
        Returns:
            List of normalized coordinates
        """
        normalized = []
        for x, y in points:
            norm_x = x / width
            norm_y = y / height
            normalized.append((norm_x, norm_y))
        return normalized
    
    def denormalize_coordinates(self, points: List[Tuple[float, float]], 
                              width: int, height: int) -> List[Tuple[int, int]]:
        """
        Convert normalized coordinates back to pixel coordinates.
        
        Args:
            points: List of normalized (x, y) coordinates [0, 1]
            width: Image width
            height: Image height
            
        Returns:
            List of pixel coordinates
        """
        pixels = []
        for norm_x, norm_y in points:
            x = int(norm_x * width)
            y = int(norm_y * height)
            pixels.append((x, y))
        return pixels
    
    def calculate_distances_from_point(self, reference_point: Tuple[float, float], 
                                     target_points: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate distances from a reference point to multiple target points.
        
        Args:
            reference_point: Reference point (x, y)
            target_points: List of target points
            
        Returns:
            List of distances in same units as input
        """
        distances = []
        for target in target_points:
            distance = self.euclidean_distance_2d(reference_point, target)
            distances.append(distance)
        return distances
    
    def find_closest_point(self, reference_point: Tuple[float, float], 
                          candidate_points: List[Tuple[float, float]]) -> Tuple[int, float]:
        """
        Find the closest point to a reference point.
        
        Args:
            reference_point: Reference point (x, y)
            candidate_points: List of candidate points
            
        Returns:
            Tuple of (closest_index, distance)
        """
        if not candidate_points:
            return -1, float('inf')
        
        distances = self.calculate_distances_from_point(reference_point, candidate_points)
        closest_idx = min(range(len(distances)), key=distances.__getitem__)
        return closest_idx, distances[closest_idx]
    
    def calibrate_from_cutting_board(self, board_corners: List[Tuple[float, float]], 
                                   board_size_mm: Tuple[float, float]):
        """
        Calibrate coordinate system using cutting board reference.
        
        Args:
            board_corners: Four corners of cutting board in pixels
            board_size_mm: Actual cutting board size (width, height) in mm
            
        Note: This is a placeholder for Day 2 implementation
        """
        # Placeholder implementation
        self.cutting_board_corners = board_corners
        self.reference_size_mm = board_size_mm
        
        # Calculate rough pixels per mm from board width
        if len(board_corners) >= 2:
            board_width_pixels = self.euclidean_distance_2d(board_corners[0], board_corners[1])
            self.pixels_per_mm = board_width_pixels / board_size_mm[0]
            self.is_calibrated = True
            print(f"✅ Coordinate system calibrated: {self.pixels_per_mm:.2f} pixels/mm")
        else:
            print("⚠️ Insufficient calibration points provided")
    
    def get_calibration_status(self) -> dict:
        """Get current calibration status and parameters"""
        return {
            "is_calibrated": self.is_calibrated,
            "pixels_per_mm": self.pixels_per_mm,
            "has_cutting_board_reference": self.cutting_board_corners is not None,
            "reference_size": self.reference_size_mm
        }


class DistanceCalculator:
    """
    Specialized class for calculating distances between hand landmarks and objects.
    Will be expanded for safety monitoring in Day 3.
    """
    
    def __init__(self, coord_transform: CoordinateTransform):
        """
        Initialize distance calculator.
        
        Args:
            coord_transform: Coordinate transformation instance
        """
        self.coord_transform = coord_transform
    
    def finger_to_point_distance(self, finger_pos: Tuple[float, float], 
                                target_pos: Tuple[float, float], 
                                in_mm: bool = True) -> float:
        """
        Calculate distance from finger to a target point.
        
        Args:
            finger_pos: Finger position (normalized or pixel coordinates)
            target_pos: Target position (same coordinate system)
            in_mm: Return distance in millimeters if True, pixels if False
            
        Returns:
            Distance in requested units
        """
        pixel_distance = self.coord_transform.euclidean_distance_2d(finger_pos, target_pos)
        
        if in_mm:
            return self.coord_transform.pixel_to_mm(pixel_distance)
        else:
            return pixel_distance
    
    def minimum_finger_distances(self, hand_landmarks: List[Tuple[float, float]], 
                                target_points: List[Tuple[float, float]], 
                                fingertip_indices: List[int] = None) -> dict:
        """
        Calculate minimum distances from fingertips to target points.
        
        Args:
            hand_landmarks: List of hand landmark positions
            target_points: List of target positions (e.g., knife edges)
            fingertip_indices: Indices of fingertip landmarks (default: [4,8,12,16,20])
            
        Returns:
            Dictionary with minimum distances and closest points
        """
        if fingertip_indices is None:
            fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        results = {
            "min_distance": float('inf'),
            "closest_finger_idx": -1,
            "closest_target_idx": -1,
            "all_distances": {}
        }
        
        for finger_idx in fingertip_indices:
            if finger_idx < len(hand_landmarks):
                finger_pos = hand_landmarks[finger_idx]
                
                for target_idx, target_pos in enumerate(target_points):
                    distance = self.finger_to_point_distance(finger_pos, target_pos)
                    
                    # Store distance
                    key = f"finger_{finger_idx}_to_target_{target_idx}"
                    results["all_distances"][key] = distance
                    
                    # Check if this is the minimum
                    if distance < results["min_distance"]:
                        results["min_distance"] = distance
                        results["closest_finger_idx"] = finger_idx
                        results["closest_target_idx"] = target_idx
        
        return results