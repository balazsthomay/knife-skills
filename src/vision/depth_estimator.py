"""
Depth estimation module using Depth Anything V2.
Provides monocular depth estimation for 3D safety detection.
"""

from typing import Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image
import time

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Install with: pip install transformers")


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.
    Supports both Hugging Face pipeline and local PyTorch models.
    """
    
    def __init__(self, 
                 model_type: str = "huggingface",
                 model_size: str = "small",
                 max_depth: float = 20.0,
                 device: str = "auto"):
        """
        Initialize depth estimator.
        
        Args:
            model_type: "huggingface" or "local" 
            model_size: "small", "base", "large", or "giant"
            max_depth: Maximum depth range in meters (20 for indoor, 80 for outdoor)
            device: "auto", "cpu", "cuda", or "mps"
        """
        self.model_type = model_type
        self.model_size = model_size
        self.max_depth = max_depth
        self.device = self._get_device(device)
        
        # Performance tracking
        self._frame_count = 0
        self._total_inference_time = 0.0
        
        # Model loading
        self.model = None
        self.pipeline = None
        
        if model_type == "huggingface":
            self._load_huggingface_model()
        else:
            self._load_local_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_huggingface_model(self):
        """Load Depth Anything V2 via Hugging Face pipeline"""
        if not HF_AVAILABLE:
            raise ImportError("transformers required for Hugging Face model. Install with: pip install transformers")
        
        model_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf", 
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        model_name = model_map.get(self.model_size, model_map["small"])
        
        print(f"🔄 Loading Depth Anything V2 ({self.model_size}) via Hugging Face...")
        print(f"📱 Device: {self.device}")
        
        try:
            self.pipeline = pipeline(
                task="depth-estimation",
                model=model_name,
                device=0 if self.device == "cuda" else -1  # HF pipeline device format
            )
            print("✅ Depth Anything V2 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Depth Anything V2: {e}")
            raise
    
    def _load_local_model(self):
        """Load local PyTorch model (placeholder for Phase 3)"""
        raise NotImplementedError("Local PyTorch model loading not implemented yet. Use model_type='huggingface'")
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB frame.
        
        Args:
            frame: Input BGR frame from OpenCV
            
        Returns:
            Depth map as numpy array (HxW) with values in meters
        """
        start_time = time.time()
        self._frame_count += 1
        
        if self.model_type == "huggingface":
            depth_map = self._estimate_depth_hf(frame)
        else:
            depth_map = self._estimate_depth_local(frame)
        
        # Update performance tracking
        inference_time = time.time() - start_time
        self._total_inference_time += inference_time
        
        return depth_map
    
    def _estimate_depth_hf(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Hugging Face pipeline"""
        # Convert BGR to RGB and to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run depth estimation
        result = self.pipeline(pil_image)
        depth_pil = result["depth"]
        
        # Convert PIL depth to numpy array
        depth_array = np.array(depth_pil, dtype=np.float32)
        
        # Normalize to meters based on max_depth
        # Note: Hugging Face pipeline returns normalized depth [0,1]
        # Scale to real-world depth range
        depth_meters = depth_array * self.max_depth
        
        return depth_meters
    
    def _estimate_depth_local(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using local PyTorch model"""

        raise NotImplementedError("Local model inference not implemented yet")
    
    def get_3d_coordinates(self, point_2d: Tuple[int, int], 
                          depth_map: np.ndarray,
                          camera_intrinsics: Optional[dict] = None) -> Tuple[float, float, float]:
        """
        Convert 2D point to 3D coordinates using depth map.
        
        Args:
            point_2d: (x, y) pixel coordinates
            depth_map: Depth map from estimate_depth()
            camera_intrinsics: Camera parameters {fx, fy, cx, cy}
            
        Returns:
            (X, Y, Z) coordinates in meters
        """
        x, y = point_2d
        
        # Ensure coordinates are within bounds
        height, width = depth_map.shape
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # Get depth value
        depth = depth_map[y, x]
        
        # Use default camera intrinsics if not provided
        if camera_intrinsics is None:
            # Rough estimates for typical webcam
            camera_intrinsics = {
                'fx': width * 0.8,  # Focal length x
                'fy': height * 0.8,  # Focal length y  
                'cx': width / 2,    # Principal point x
                'cy': height / 2    # Principal point y
            }
        
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # Convert to 3D coordinates
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        
        return (X, Y, Z)
    
    def calculate_3d_distance(self, point1_3d: Tuple[float, float, float],
                             point2_3d: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            point1_3d: First 3D point (X, Y, Z)
            point2_3d: Second 3D point (X, Y, Z)
            
        Returns:
            Distance in meters
        """
        return np.sqrt(sum((a - b)**2 for a, b in zip(point1_3d, point2_3d)))
    
    def get_stats(self) -> dict:
        """Get depth estimation statistics"""
        avg_inference_time = (self._total_inference_time / self._frame_count 
                             if self._frame_count > 0 else 0)
        
        return {
            "frames_processed": self._frame_count,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "max_depth": self.max_depth,
            "device": self.device
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self._frame_count = 0
        self._total_inference_time = 0.0