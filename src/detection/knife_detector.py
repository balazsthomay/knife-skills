"""
Knife detection module using trained YOLOv11 segmentation model.
Provides real-time knife detection with precise blade boundary extraction.
"""

from typing import List, Optional, Tuple, NamedTuple
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not found. Install with: pip install ultralytics")


class KnifeBoundary(NamedTuple):
    """Container for knife boundary data"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    mask: np.ndarray  # Binary segmentation mask
    contour: np.ndarray  # Knife blade contour points
    confidence: float  # Detection confidence score
    blade_length_pixels: float  # Estimated blade length in pixels
    tip_point: Tuple[int, int]  # Knife tip coordinates (x, y)
    handle_point: Tuple[int, int]  # Handle end coordinates (x, y)


class KnifeDetector:
    """
    Real-time knife detection using YOLOv11 segmentation model.
    Detects knives and extracts precise blade boundaries for safety calculations.
    """
    
    def __init__(
        self,
        model_path: str = "models/knife_seg_phase1_frozen/weights/best.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize knife detector.
        
        Args:
            model_path: Path to trained YOLOv11 model weights
            confidence_threshold: Detection confidence threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load YOLOv11 model
        try:
            self.model = YOLO(str(self.model_path))
            print(f"✅ Knife detection model loaded: {self.model_path}")
            print(f"📱 Using device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        # Performance tracking
        self._frame_count = 0
        self._last_detection_count = 0
        self._total_inference_time = 0.0
    
    def detect_knives(self, frame: np.ndarray) -> List[KnifeBoundary]:
        """
        Detect knives in the given frame.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            List of KnifeBoundary containing detected knives
        """
        import time
        start_time = time.time()
        self._frame_count += 1
        
        # Run YOLOv11 inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            imgsz=640  # Standard input size for our trained model
        )
        
        detected_knives = []
        
        # Process results
        for result in results:
            if result.boxes is not None and result.masks is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
                confidences = result.boxes.conf.cpu().numpy()
                masks = result.masks.data.cpu().numpy()  # Segmentation masks
                
                for i, (box, conf, mask) in enumerate(zip(boxes, confidences, masks)):
                    if conf >= self.confidence_threshold:
                        # Convert box to integers
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Resize mask to original frame size
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Extract blade contour and key points
                        contour, blade_length, tip, handle = self._extract_blade_info(mask_binary)
                        
                        if contour is not None:
                            detected_knives.append(KnifeBoundary(
                                bbox=(x1, y1, x2, y2),
                                mask=mask_binary,
                                contour=contour,
                                confidence=float(conf),
                                blade_length_pixels=blade_length,
                                tip_point=tip,
                                handle_point=handle
                            ))
        
        # Update performance tracking
        inference_time = time.time() - start_time
        self._total_inference_time += inference_time
        self._last_detection_count = len(detected_knives)
        
        return detected_knives
    
    def _extract_blade_info(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], float, Tuple[int, int], Tuple[int, int]]:
        """
        Extract blade contour, length, tip and handle points from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Tuple of (contour, blade_length, tip_point, handle_point)
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0, (0, 0), (0, 0)
        
        # Get the largest contour (should be the knife)
        main_contour = max(contours, key=cv2.contourArea)
        
        if len(main_contour) < 10:  # Too few points for meaningful analysis
            return None, 0.0, (0, 0), (0, 0)
        
        # Fit minimum area rectangle to estimate blade orientation
        rect = cv2.minAreaRect(main_contour)
        center, (width, height), angle = rect
        
        # Calculate blade length (longer dimension)
        blade_length = max(width, height)
        
        # Estimate tip and handle points based on the oriented rectangle
        # For a knife, the tip is typically at one end of the longer axis
        cx, cy = map(int, center)
        
        # Calculate the oriented endpoints
        angle_rad = np.radians(angle)
        if width > height:
            # Horizontal-ish knife
            half_length = width / 2
            dx = half_length * np.cos(angle_rad)
            dy = half_length * np.sin(angle_rad)
        else:
            # Vertical-ish knife
            half_length = height / 2
            dx = half_length * np.cos(angle_rad + np.pi/2)
            dy = half_length * np.sin(angle_rad + np.pi/2)
        
        # Two endpoints of the blade
        point1 = (int(cx - dx), int(cy - dy))
        point2 = (int(cx + dx), int(cy + dy))
        
        # Heuristic: tip is usually the point closer to the top of the image
        # (knives are often held with tip up/forward)
        if point1[1] < point2[1]:  # point1 is higher (smaller y)
            tip_point = point1
            handle_point = point2
        else:
            tip_point = point2
            handle_point = point1
        
        return main_contour, blade_length, tip_point, handle_point
    
    def get_blade_tip_distance(self, knife: KnifeBoundary, point: Tuple[int, int]) -> float:
        """
        Calculate distance from a point to the knife blade tip.
        
        Args:
            knife: KnifeBoundary object
            point: (x, y) coordinates to measure distance from
            
        Returns:
            Distance in pixels
        """
        tip_x, tip_y = knife.tip_point
        point_x, point_y = point
        
        return np.sqrt((tip_x - point_x)**2 + (tip_y - point_y)**2)
    
    def get_blade_edge_distance(self, knife: KnifeBoundary, point: Tuple[int, int]) -> float:
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
        
        # Calculate distance to contour
        point_array = np.array([[point[0], point[1]]], dtype=np.float32)
        distance = cv2.pointPolygonTest(knife.contour, point, True)
        
        # Return absolute distance (positive = outside, negative = inside)
        return abs(distance)
    
    def get_stats(self) -> dict:
        """Get detection statistics"""
        avg_inference_time = (self._total_inference_time / self._frame_count 
                             if self._frame_count > 0 else 0)
        
        return {
            "frames_processed": self._frame_count,
            "last_detection_count": self._last_detection_count,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold
        }
    
    def __del__(self):
        """Clean up resources"""
        # YOLOv11 handles its own cleanup
        pass