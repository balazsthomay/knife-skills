"""
Hand detection module using MediaPipe Hands.
Provides real-time hand tracking with 21 landmarks per hand.
"""

from typing import List, Optional, Tuple, NamedTuple
import cv2
import mediapipe as mp
import numpy as np


class HandLandmarks(NamedTuple):
    """Container for hand landmark data"""
    landmarks: List[Tuple[float, float]]  # (x, y) normalized coordinates
    handedness: str  # "Left" or "Right"
    confidence: float  # Handedness confidence score


class HandDetector:
    """
    Real-time hand detection using MediaPipe Hands.
    Detects up to 2 hands and returns 21 landmarks per hand.
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            model_complexity: Model complexity (0=light, 1=full)
            min_detection_confidence: Detection confidence threshold
            min_tracking_confidence: Tracking confidence threshold
        """
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Performance tracking
        self._frame_count = 0
        self._last_detection_count = 0
    
    def detect_hands(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands in the given frame.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            List of HandLandmarks containing detected hands
        """
        self._frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Improve performance by marking image as not writeable
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        # Don't set writeable back to True since we don't modify it
        
        detected_hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))
                
                # Get handedness info and correct for camera mirroring
                # MediaPipe assumes mirrored input, so we need to swap Left/Right for natural interaction
                detected_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score
                
                # Swap handedness to match natural interaction (left hand should show as left)
                hand_label = "Left" if detected_label == "Right" else "Right"
                
                detected_hands.append(HandLandmarks(
                    landmarks=landmarks,
                    handedness=hand_label,
                    confidence=hand_confidence
                ))
        
        self._last_detection_count = len(detected_hands)
        return detected_hands
    
    def get_landmark_connections(self) -> List[Tuple[int, int]]:
        """Get hand landmark connections for drawing"""
        return self.mp_hands.HAND_CONNECTIONS
    
    def get_stats(self) -> dict:
        """Get detection statistics"""
        return {
            "frames_processed": self._frame_count,
            "last_detection_count": self._last_detection_count,
            "max_hands": self.max_num_hands
        }
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()