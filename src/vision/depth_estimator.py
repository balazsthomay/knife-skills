"""
Depth estimation module using Depth Anything V2.
Provides monocular depth estimation for 3D safety detection.
"""

from typing import Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os
from pathlib import Path

# Add Depth Anything V2 to path
DEPTH_ANYTHING_PATH = Path(__file__).parent.parent.parent / "Depth-Anything-V2"
if DEPTH_ANYTHING_PATH.exists():
    sys.path.insert(0, str(DEPTH_ANYTHING_PATH))

try:
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  Warning: torch or depth_anything_v2 not available.")

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Install with: pip install transformers")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("⚠️  Warning: coremltools not available. Install with: pip install coremltools")


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.
    Supports both Hugging Face pipeline and local PyTorch models.
    """
    
    def __init__(self, 
                 model_type: str = "pytorch",
                 model_size: str = "small",
                 max_depth: float = 20.0,
                 device: str = "auto"):
        """
        Initialize depth estimator.
        
        Args:
            model_type: "pytorch" (direct), "huggingface" (pipeline), or "coreml" (Apple Neural Engine)
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
        
        if model_type == "pytorch":
            self._load_pytorch_model()
        elif model_type == "huggingface":
            self._load_huggingface_model()
        elif model_type == "coreml":
            self._load_coreml_model()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'pytorch', 'huggingface', or 'coreml'")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            else:
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
    
    def _load_pytorch_model(self):
        """Load direct PyTorch Depth Anything V2 model"""
        if not TORCH_AVAILABLE:
            raise ImportError("torch and depth_anything_v2 required for PyTorch model. Ensure Depth-Anything-V2 is cloned.")
        
        # Model configurations from Depth Anything V2
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if self.model_size not in model_configs:
            raise ValueError(f"Unsupported model size: {self.model_size}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[self.model_size]
        encoder = config['encoder']
        
        print(f"🔄 Loading Depth Anything V2 ({self.model_size}) via direct PyTorch...")
        print(f"📱 Device: {self.device}")
        
        try:
            # Initialize model
            self.model = DepthAnythingV2(**config)
            
            # Load pre-trained weights
            checkpoint_path = DEPTH_ANYTHING_PATH / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            print("✅ Depth Anything V2 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load Depth Anything V2: {e}")
            raise
    
    def _load_coreml_model(self):
        """Load Core ML Depth Anything V2 model for Apple Neural Engine"""
        if not COREML_AVAILABLE:
            raise ImportError("coremltools required for Core ML model. Install with: pip install coremltools")
        
        # Core ML model path
        model_path = Path(__file__).parent.parent.parent / "models" / "DepthAnythingV2SmallF16.mlpackage"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Core ML model not found: {model_path}")
        
        print(f"🔄 Loading Core ML Depth Anything V2 ({self.model_size})...")
        print(f"📱 Device: Apple Neural Engine")
        
        try:
            # Load Core ML model
            self.model = ct.models.MLModel(str(model_path))
            print("✅ Core ML Depth Anything V2 model loaded successfully")
            
            # Get model input/output info
            spec = self.model.get_spec()
            print(f"📊 Model inputs: {[input.name for input in spec.description.input]}")
            print(f"📊 Model outputs: {[output.name for output in spec.description.output]}")
            
            # Get input image size constraints
            for input_desc in spec.description.input:
                if input_desc.name == "image":
                    if hasattr(input_desc.type, 'imageType'):
                        image_type = input_desc.type.imageType
                        print(f"📊 Image input constraints: {image_type}")
                        # Store expected input size for later use
                        self._coreml_input_width = image_type.width
                        self._coreml_input_height = image_type.height
                    elif hasattr(input_desc.type, 'multiArrayType'):
                        array_type = input_desc.type.multiArrayType
                        print(f"📊 Array input shape: {list(array_type.shape)}")
                        # Store expected input size for later use
                        if len(array_type.shape) >= 2:
                            self._coreml_input_height = array_type.shape[-2]
                            self._coreml_input_width = array_type.shape[-1]
            
        except Exception as e:
            print(f"❌ Failed to load Core ML model: {e}")
            raise
    
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
        
        if self.model_type == "pytorch":
            depth_map = self._estimate_depth_pytorch(frame)
        elif self.model_type == "huggingface":
            depth_map = self._estimate_depth_hf(frame)
        elif self.model_type == "coreml":
            depth_map = self._estimate_depth_coreml(frame)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
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
    
    def _estimate_depth_pytorch(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using direct PyTorch model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Direct inference - no pipeline overhead
        with torch.no_grad():
            depth_map = self.model.infer_image(rgb_frame)
        
        # Convert to numpy if tensor
        if torch.is_tensor(depth_map):
            depth_map = depth_map.cpu().numpy()
        
        # Ensure float32 format
        depth_map = depth_map.astype(np.float32)
        
        # Scale to real-world depth range
        # Depth Anything V2 returns relative depth, scale to max_depth
        depth_normalized = depth_map / depth_map.max() if depth_map.max() > 0 else depth_map
        depth_meters = depth_normalized * self.max_depth
        
        return depth_meters
    
    def _estimate_depth_coreml(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Core ML model on Apple Neural Engine"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        original_height, original_width = rgb_frame.shape[:2]
        
        # Resize to model's expected input size (518x392 for this Core ML model)
        # Default to 518x392 if not detected from model spec
        expected_height = getattr(self, '_coreml_input_height', 392)
        expected_width = getattr(self, '_coreml_input_width', 518)
        
        rgb_resized = cv2.resize(rgb_frame, (expected_width, expected_height))
        
        # Convert to PIL Image (Core ML expects PIL Image input)
        pil_image = Image.fromarray(rgb_resized)
        
        try:
            # Run Core ML inference with PIL Image
            input_dict = {"image": pil_image}
            result = self.model.predict(input_dict)
            
            # Extract depth map
            if "depth" in result:
                depth_output = result["depth"]
            elif "output" in result:
                depth_output = result["output"]
            else:
                # Use first output
                output_name = list(result.keys())[0]
                depth_output = result[output_name]
            
            # Convert to numpy array
            if hasattr(depth_output, 'numpy'):
                depth_map = depth_output.numpy()
            else:
                depth_map = np.array(depth_output)
            
            # Remove batch dimension if present
            while len(depth_map.shape) > 2:
                depth_map = depth_map.squeeze(0)
            
            # Resize back to original frame size if needed
            if depth_map.shape[:2] != (original_height, original_width):
                depth_map = cv2.resize(depth_map, (original_width, original_height))
            
            # Ensure float32 format
            depth_map = depth_map.astype(np.float32)
            
            # Scale to real-world depth range
            # Core ML model outputs relative depth, scale to max_depth
            depth_normalized = depth_map / depth_map.max() if depth_map.max() > 0 else depth_map
            depth_meters = depth_normalized * self.max_depth
            
            return depth_meters
            
        except Exception as e:
            print(f"❌ Core ML inference failed: {e}")
            raise
    
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