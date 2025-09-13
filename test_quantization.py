#!/usr/bin/env python3
"""
Quantization performance test for Depth Anything V2.
Compares non-quantized vs INT8 quantized model performance.
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from vision.depth_estimator import DepthEstimator


def test_quantization_performance():
    """Test and compare quantized vs non-quantized model performance with visual comparison"""
    print("🚀 Starting Visual Quantization Performance Test")
    print("=" * 60)
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot")
    print("  R - Reset performance statistics")
    
    # Test configurations
    configs = [
        {"quantization": "none", "name": "FP32 (Baseline)", "color": (255, 255, 255)},
        {"quantization": "int8", "name": "INT8 Quantized", "color": (0, 255, 255)}
    ]
    
    results = {}
    depth_estimators = {}
    
    # Initialize both models
    for config in configs:
        print(f"\n📊 Initializing {config['name']}...")
        
        try:
            # Initialize depth estimator
            print(f"🔄 Loading depth estimator with {config['quantization']} quantization...")
            
            depth_estimator = DepthEstimator(
                model_type="huggingface",
                model_size="small",
                max_depth=20.0,
                device="auto",
                quantization=config['quantization']
            )
            
            print("✅ Model loaded successfully")
            depth_estimators[config['quantization']] = depth_estimator
            
        except Exception as e:
            print(f"❌ Failed to initialize {config['name']}: {e}")
            continue
    
    if len(depth_estimators) != 2:
        print("❌ Need both models initialized for comparison")
        return
    
    # Initialize camera for live feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📹 Camera initialized")
    print("\n🎬 Starting visual comparison...")
    
    paused = False
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process with both models and time them
            depth_maps = {}
            inference_times = {}
            
            for quantization, estimator in depth_estimators.items():
                start_time = time.time()
                depth_maps[quantization] = estimator.estimate_depth(frame)
                inference_times[quantization] = (time.time() - start_time) * 1000
            
            # Create visualizations
            display_frames = [frame]
            
            for config in configs:
                quantization = config['quantization']
                if quantization in depth_maps:
                    # Create depth visualization
                    depth_map = depth_maps[quantization]
                    depth_normalized = (depth_map / 20.0 * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                    
                    # Add performance text overlay
                    inference_time = inference_times[quantization]
                    fps_estimate = 1000 / inference_time if inference_time > 0 else 0
                    stats = estimator.get_stats()
                    
                    # Add text with model info
                    text_color = config['color']
                    cv2.putText(depth_vis, f"{config['name']}", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    cv2.putText(depth_vis, f"Inference: {inference_time:.1f}ms", (20, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(depth_vis, f"Avg: {stats['avg_inference_time_ms']:.1f}ms", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(depth_vis, f"Est. FPS: {fps_estimate:.1f}", (20, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    display_frames.append(depth_vis)
            
            # Update FPS counter
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                current_fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
        
        # Create side-by-side display
        if len(display_frames) == 3:  # RGB + 2 depth maps
            # Resize frames to fit screen better
            display_height = 480
            display_frames_resized = []
            for frame_to_resize in display_frames:
                aspect_ratio = frame_to_resize.shape[1] / frame_to_resize.shape[0]
                display_width = int(display_height * aspect_ratio)
                resized = cv2.resize(frame_to_resize, (display_width, display_height))
                display_frames_resized.append(resized)
            
            combined_frame = np.hstack(display_frames_resized)
            
            # Add overall info
            if current_fps > 0:
                cv2.putText(combined_frame, f"Display FPS: {current_fps:.1f}", 
                           (20, combined_frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(combined_frame, "ESC: Exit | SPACE: Pause | S: Screenshot | R: Reset Stats", 
                       (20, combined_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add labels
            label_y = 30
            cv2.putText(combined_frame, "RGB INPUT", (20, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_frame, "FP32 DEPTH", (display_frames_resized[0].shape[1] + 20, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_frame, "INT8 DEPTH", (display_frames_resized[0].shape[1] + display_frames_resized[1].shape[1] + 20, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if paused:
                cv2.putText(combined_frame, "PAUSED", (combined_frame.shape[1]//2 - 100, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            cv2.imshow('Quantization Comparison', combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = int(time.time())
            screenshot_name = f"quantization_comparison_{timestamp}.jpg"
            cv2.imwrite(screenshot_name, combined_frame)
            print(f"📸 Screenshot saved: {screenshot_name}")
        elif key == ord('r') or key == ord('R'):  # R
            for estimator in depth_estimators.values():
                estimator.reset_stats()
            print("📊 Statistics reset")
    
    # Cleanup
    cap.release()
    
    cv2.destroyAllWindows()
    
    # Performance summary
    print("\n" + "=" * 60)
    print("📈 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for quantization, estimator in depth_estimators.items():
        stats = estimator.get_stats()
        config_name = next(c['name'] for c in configs if c['quantization'] == quantization)
        
        results[quantization] = {
            'name': config_name,
            'avg_inference_ms': stats['avg_inference_time_ms'],
            'estimated_fps': 1000 / stats['avg_inference_time_ms'] if stats['avg_inference_time_ms'] > 0 else 0,
            'frames_processed': stats['frames_processed']
        }
        
        print(f"📊 {config_name}:")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Avg inference: {stats['avg_inference_time_ms']:.1f}ms")
        print(f"   Estimated max FPS: {results[quantization]['estimated_fps']:.1f}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if "none" in results and "int8" in results:
        baseline = results["none"]
        quantized = results["int8"]
        
        speedup = baseline['avg_inference_ms'] / quantized['avg_inference_ms']
        fps_improvement = quantized['estimated_fps'] / baseline['estimated_fps']
        
        print(f"📊 Baseline (FP32):     {baseline['avg_inference_ms']:.1f}ms ({baseline['estimated_fps']:.1f} FPS)")
        print(f"⚡ INT8 Quantized:      {quantized['avg_inference_ms']:.1f}ms ({quantized['estimated_fps']:.1f} FPS)")
        print(f"🚀 Speedup:             {speedup:.2f}x faster")
        print(f"📈 FPS Improvement:     {fps_improvement:.2f}x")
        
        # Performance assessment
        target_fps = 15
        print(f"\n🎯 Target Performance:  {target_fps} FPS")
        
        if quantized['estimated_fps'] >= target_fps:
            print("✅ INT8 quantization achieves target performance!")
        else:
            needed_speedup = target_fps / quantized['estimated_fps']
            print(f"⚠️  Still {needed_speedup:.1f}x speedup needed for real-time performance")
            
        print(f"\n💡 Recommendation:")
        if quantized['estimated_fps'] >= target_fps:
            print("   → Use INT8 quantization for production deployment")
        elif quantized['estimated_fps'] >= target_fps * 0.7:  # Within 30% of target
            print("   → INT8 quantization + frame skipping could achieve target")
        else:
            print("   → Consider additional optimizations (resolution reduction, model distillation)")
    
    else:
        print("⚠️  Could not perform comparison - some tests failed")
        for quantization, result in results.items():
            print(f"   {result['name']}: {result['avg_inference_ms']:.1f}ms ({result['estimated_fps']:.1f} FPS)")
    
    print("\n👋 Quantization performance test complete!")


if __name__ == "__main__":
    test_quantization_performance()