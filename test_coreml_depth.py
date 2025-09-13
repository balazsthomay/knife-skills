#!/usr/bin/env python3
"""
Core ML vs PyTorch vs HuggingFace depth estimation comparison.
Real-time performance benchmarking for Apple Neural Engine acceleration.
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
    """Core ML performance comparison demo"""
    print("🚀 Core ML vs PyTorch vs HuggingFace Performance Comparison")
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot + depth map")
    print("  M - Switch between Core ML/PyTorch/HuggingFace")
    print("  D - Toggle depth visualization mode")
    print("  R - Reset performance statistics")
    
    # Model configurations
    configs = {
        "coreml": {"name": "Core ML (Neural Engine)", "color": (0, 255, 255)},
        "pytorch": {"name": "Direct PyTorch", "color": (0, 255, 0)},
        "huggingface": {"name": "HuggingFace Pipeline", "color": (255, 255, 255)}
    }
    
    depth_estimators = {}
    current_model = "coreml"  # Start with Core ML
    
    # Initialize models
    for model_type, config in configs.items():
        print(f"\n🔄 Initializing {config['name']}...")
        
        try:
            depth_estimator = DepthEstimator(
                model_type=model_type,
                model_size="small",
                max_depth=20.0,  # Indoor scene (kitchen environment)
                device="auto"
            )
            depth_estimators[model_type] = depth_estimator
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to initialize {config['name']}: {e}")
            continue
    
    if not depth_estimators:
        print("❌ No models loaded successfully")
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
    print(f"🚀 Starting comparison with {configs[current_model]['name']}...")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    current_fps = 0
    
    # Depth visualization modes
    depth_modes = ["colormap", "grayscale", "heatmap"]
    current_depth_mode = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Get current estimator
            if current_model not in depth_estimators:
                current_model = list(depth_estimators.keys())[0]
            
            estimator = depth_estimators[current_model]
            
            # Estimate depth
            start_time = time.time()
            depth_map = estimator.estimate_depth(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Create depth visualization based on current mode
            if depth_modes[current_depth_mode] == "grayscale":
                # Normalize to 0-255 for grayscale visualization
                depth_normalized = (depth_map / estimator.max_depth * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            elif depth_modes[current_depth_mode] == "colormap":
                # Apply colormap (VIRIDIS - closer is purple, farther is yellow)
                depth_normalized = (depth_map / estimator.max_depth * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            else:  # heatmap
                # Apply heat colormap (closer is red, farther is blue)
                depth_normalized = (depth_map / estimator.max_depth * 255).astype(np.uint8)
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
        
        # Get estimator statistics
        stats = estimator.get_stats()
        
        # Draw status information
        status_y = 30
        model_color = configs[current_model]['color']
        
        cv2.putText(combined_frame, f"Model: {configs[current_model]['name']}", 
                   (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        cv2.putText(combined_frame, f"Mode: {depth_modes[current_depth_mode].upper()}", 
                   (20, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Max Depth: {estimator.max_depth}m", 
                   (20, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_fps > 0:
            cv2.putText(combined_frame, f"Display FPS: {current_fps:.1f}", 
                       (20, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_frame, f"Inference: {inference_time:.1f}ms", 
                   (20, status_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        cv2.putText(combined_frame, f"Avg Inference: {stats['avg_inference_time_ms']:.1f}ms", 
                   (20, status_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        cv2.putText(combined_frame, f"Est. Max FPS: {1000/stats['avg_inference_time_ms']:.1f}", 
                   (20, status_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        cv2.putText(combined_frame, f"Frames Processed: {stats['frames_processed']}", 
                   (20, status_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Performance target indicator
        target_fps = 15
        actual_fps = 1000 / stats['avg_inference_time_ms'] if stats['avg_inference_time_ms'] > 0 else 0
        target_color = (0, 255, 0) if actual_fps >= target_fps else (0, 100, 255)
        
        cv2.putText(combined_frame, f"Target: {target_fps} FPS {'✅' if actual_fps >= target_fps else '⚠️'}", 
                   (20, status_y + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, target_color, 2)
        
        # Draw labels for each side
        cv2.putText(combined_frame, "RGB INPUT", (20, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined_frame, "DEPTH MAP", (frame.shape[1] + 20, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw depth scale legend (on depth side)
        legend_x = frame.shape[1] + 20
        legend_y = 300
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
        controls_text = "ESC: Exit | SPACE: Pause | S: Screenshot | M: Switch Model | D: Mode | R: Reset"
        cv2.putText(combined_frame, controls_text, (20, combined_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('Core ML vs PyTorch vs HuggingFace Comparison', combined_frame)
        
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
            rgb_filename = f"coreml_comparison_rgb_{timestamp}.jpg"
            cv2.imwrite(rgb_filename, frame)
            # Save depth visualization
            depth_filename = f"coreml_comparison_depth_{timestamp}.jpg"
            cv2.imwrite(depth_filename, depth_vis)
            print(f"📸 Screenshots saved: {rgb_filename}, {depth_filename}")
        elif key == ord('m') or key == ord('M'):  # M
            # Switch between models
            model_list = list(depth_estimators.keys())
            if len(model_list) > 1:
                current_idx = model_list.index(current_model)
                current_model = model_list[(current_idx + 1) % len(model_list)]
                print(f"🔄 Switched to: {configs[current_model]['name']}")
        elif key == ord('d') or key == ord('D'):  # D
            current_depth_mode = (current_depth_mode + 1) % len(depth_modes)
            print(f"🎨 Depth visualization mode: {depth_modes[current_depth_mode]}")
        elif key == ord('r') or key == ord('R'):  # R
            estimator.reset_stats()
            print("📊 Statistics reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance comparison
    print("\n" + "=" * 80)
    print("🏁 FINAL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    performance_results = []
    
    for model_type, estimator in depth_estimators.items():
        final_stats = estimator.get_stats()
        config_name = configs[model_type]['name']
        
        avg_time = final_stats['avg_inference_time_ms']
        estimated_fps = 1000 / avg_time if avg_time > 0 else 0
        
        performance_results.append({
            'name': config_name,
            'type': model_type,
            'avg_time_ms': avg_time,
            'estimated_fps': estimated_fps,
            'frames_processed': final_stats['frames_processed']
        })
        
        print(f"\n📈 {config_name}:")
        print(f"  Model: {final_stats['model_type']} ({final_stats['model_size']})")
        print(f"  Device: {final_stats['device']}")
        print(f"  Max Depth: {final_stats['max_depth']}m")
        print(f"  Frames processed: {final_stats['frames_processed']}")
        print(f"  Avg inference time: {avg_time:.1f}ms")
        print(f"  Estimated max FPS: {estimated_fps:.1f}")
        
        # Performance assessment
        target_fps = 15
        if estimated_fps >= target_fps:
            print(f"  ✅ Performance target met ({target_fps} FPS)")
        else:
            print(f"  ⚠️  Performance below target ({target_fps} FPS)")
    
    # Speedup comparison
    if len(performance_results) > 1:
        print(f"\n🚀 SPEEDUP ANALYSIS:")
        
        # Filter out models with 0 inference time (not tested)
        valid_results = [r for r in performance_results if r['avg_time_ms'] > 0]
        
        if len(valid_results) > 1:
            # Sort by inference time (fastest first)
            valid_results.sort(key=lambda x: x['avg_time_ms'])
            
            fastest = valid_results[0]
            print(f"🥇 Fastest: {fastest['name']} ({fastest['avg_time_ms']:.1f}ms, {fastest['estimated_fps']:.1f} FPS)")
            
            for result in valid_results[1:]:
                speedup = result['avg_time_ms'] / fastest['avg_time_ms']
                fps_improvement = fastest['estimated_fps'] / result['estimated_fps'] if result['estimated_fps'] > 0 else float('inf')
                print(f"📊 {result['name']}: {speedup:.2f}x slower ({fps_improvement:.2f}x less FPS)")
        else:
            print("⚠️  Not enough valid results for comparison")
    
    print("\n👋 Core ML depth comparison complete!")


if __name__ == "__main__":
    main()