#!/usr/bin/env python3
"""
Depth estimation testing demo using Depth Anything V2.
Real-time depth map visualization with performance benchmarking.
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from vision.depth_estimator import DepthEstimator


def main():
    """Main depth estimation demo"""
    print("🌊 Starting Depth Anything V2 Testing Demo")
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot + depth map")
    print("  D - Toggle depth visualization mode")
    print("  R - Reset performance statistics")
    
    # Initialize depth estimator
    try:
        print("🔄 Initializing Depth Anything V2...")
        depth_estimator = DepthEstimator(
            model_type="huggingface",
            model_size="small",
            max_depth=20.0,  # Indoor scene (kitchen environment)
            device="auto"
        )
        print("✅ Depth estimator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize depth estimator: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📹 Camera initialized")
    print("🚀 Starting depth estimation loop...")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    
    # Depth visualization modes
    depth_modes = ["grayscale", "colormap", "heatmap"]
    current_depth_mode = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Estimate depth
            start_time = time.time()
            depth_map = depth_estimator.estimate_depth(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Create depth visualization based on current mode
            if depth_modes[current_depth_mode] == "grayscale":
                # Normalize to 0-255 for grayscale visualization
                depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            elif depth_modes[current_depth_mode] == "colormap":
                # Apply colormap (VIRIDIS - closer is purple, farther is yellow)
                depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            else:  # heatmap
                # Apply heat colormap (closer is red, farther is blue)
                depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Create side-by-side display
            combined_frame = np.hstack([frame, depth_vis])
            
            # Update FPS counter
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                current_fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            else:
                current_fps = 0
        
        else:
            # Keep the last frame when paused
            pass
        
        # Get depth estimator statistics
        stats = depth_estimator.get_stats()
        
        # Draw status information
        status_y = 30
        cv2.putText(combined_frame, f"Mode: {depth_modes[current_depth_mode].upper()}", 
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Max Depth: {depth_estimator.max_depth}m", 
                   (20, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_fps > 0:
            cv2.putText(combined_frame, f"FPS: {current_fps:.1f}", 
                       (20, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Inference: {inference_time:.1f}ms", 
                   (20, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Avg Inference: {stats['avg_inference_time_ms']:.1f}ms", 
                   (20, status_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Frames Processed: {stats['frames_processed']}", 
                   (20, status_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw labels for each side
        cv2.putText(combined_frame, "RGB INPUT", (20, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined_frame, "DEPTH MAP", (frame.shape[1] + 20, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw depth scale legend (on depth side)
        legend_x = frame.shape[1] + 20
        legend_y = 200
        cv2.putText(combined_frame, "Depth Scale:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if depth_modes[current_depth_mode] == "grayscale":
            cv2.putText(combined_frame, "Dark = Close, Light = Far", (legend_x, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        elif depth_modes[current_depth_mode] == "colormap":
            cv2.putText(combined_frame, "Purple = Close, Yellow = Far", (legend_x, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(combined_frame, "Red = Close, Blue = Far", (legend_x, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw pause indicator
        if paused:
            cv2.putText(combined_frame, "PAUSED", (combined_frame.shape[1]//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw controls
        controls_text = "ESC: Exit | SPACE: Pause | S: Screenshot | D: Mode | R: Reset Stats"
        cv2.putText(combined_frame, controls_text, (20, combined_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('Depth Anything V2 Testing Demo', combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = int(time.time())
            # Save RGB frame
            rgb_filename = f"depth_test_rgb_{timestamp}.jpg"
            cv2.imwrite(rgb_filename, frame)
            # Save depth visualization
            depth_filename = f"depth_test_depth_{timestamp}.jpg"
            cv2.imwrite(depth_filename, depth_vis)
            # Save raw depth data as numpy array
            depth_raw_filename = f"depth_test_raw_{timestamp}.npy"
            np.save(depth_raw_filename, depth_map)
            print(f"📸 Screenshots saved: {rgb_filename}, {depth_filename}, {depth_raw_filename}")
        elif key == ord('d') or key == ord('D'):  # D
            current_depth_mode = (current_depth_mode + 1) % len(depth_modes)
            print(f"🎨 Depth visualization mode: {depth_modes[current_depth_mode]}")
        elif key == ord('r') or key == ord('R'):  # R
            depth_estimator.reset_stats()
            print("📊 Statistics reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    final_stats = depth_estimator.get_stats()
    print("\n📊 Final Depth Estimation Statistics:")
    print(f"  Model: {final_stats['model_type']} ({final_stats['model_size']})")
    print(f"  Device: {final_stats['device']}")
    print(f"  Max Depth: {final_stats['max_depth']}m")
    print(f"  Frames processed: {final_stats['frames_processed']}")
    print(f"  Avg inference time: {final_stats['avg_inference_time_ms']:.1f}ms")
    
    # Performance assessment
    target_fps = 15
    actual_fps = 1000 / final_stats['avg_inference_time_ms'] if final_stats['avg_inference_time_ms'] > 0 else 0
    print(f"  Estimated max FPS: {actual_fps:.1f}")
    
    if actual_fps >= target_fps:
        print(f"  ✅ Performance target met ({target_fps} FPS)")
    else:
        print(f"  ⚠️  Performance below target ({target_fps} FPS)")
        print("  💡 Consider optimizations: smaller model, lower resolution, or frame skipping")
    
    print("👋 Depth estimation demo complete!")


if __name__ == "__main__":
    main()