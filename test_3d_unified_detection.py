#!/usr/bin/env python3
"""
3D unified hand, knife, and depth-aware safety detection demo.
Combines MediaPipe hands, YOLOv11 knife detection, and Core ML depth estimation
for true 3D spatial safety monitoring.
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.hand_detector import HandDetector
from detection.knife_detector import KnifeDetector
from ui.overlay_renderer import OverlayRenderer
from safety.safety_monitor import SafetyMonitor
from safety.depth_aware_safety import DepthAwareSafetyMonitor
from vision.depth_estimator import DepthEstimator


def main():
    """Main 3D unified detection demo"""
    print("🔪 🖐️ 🌊 Starting 3D Unified Hand + Knife + Depth Safety Demo")
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot + depth map")
    print("  2/3 - Switch between 2D and 3D safety mode")
    print("  D - Toggle depth visualization")
    print("  H - Toggle hand landmarks")
    print("  K - Toggle knife detection")
    print("  W - Toggle safety warnings")
    print("  R - Reset statistics")
    
    # Initialize all detectors and monitors
    try:
        # Hand detection
        hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ Hand detector initialized")
        
        # Knife detection
        knife_detector = KnifeDetector(
            model_path="models/knife_seg_phase1_frozen/weights/best.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
        print("✅ Knife detector initialized")
        
        # Depth estimation (Core ML for best performance)
        depth_estimator = DepthEstimator(
            model_type="coreml",  # Use Core ML for 23.4 FPS
            model_size="small",
            max_depth=3.0,  # Kitchen environment - objects 0.5-3m away
            device="auto"
        )
        print("✅ Depth estimator initialized (Core ML)")
        
        # Safety monitors
        safety_monitor_2d = SafetyMonitor(
            critical_threshold=30,
            high_threshold=50,
            medium_threshold=80
        )
        print("✅ 2D Safety monitor initialized")
        
        safety_monitor_3d = DepthAwareSafetyMonitor(
            depth_estimator=depth_estimator,
            critical_threshold_3d=0.050,  # 50mm
            high_threshold_3d=0.100,      # 100mm
            medium_threshold_3d=0.200     # 200mm
        )
        print("✅ 3D Safety monitor initialized")
        
        # Overlay renderer
        overlay_renderer = OverlayRenderer(
            show_connections=True,
            show_landmarks=True,
            show_labels=False
        )
        print("✅ Overlay renderer initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
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
    print("🚀 Starting 3D unified detection loop...")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    current_fps = 0
    
    # Display toggles
    show_hands = True
    show_knives = True
    show_safety = True
    show_depth = True
    use_3d_safety = True  # Start with 3D mode
    
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
            
            # Detect hands
            hands = hand_detector.detect_hands(frame) if show_hands else []
            
            # Detect knives
            knives = knife_detector.detect_knives(frame) if show_knives else []
            
            # Safety detection
            dangers_2d = []
            dangers_3d = []
            comparison_stats = {}
            
            if show_safety and hands and knives:
                if use_3d_safety:
                    # Run both 2D and 3D for comparison
                    comparison_stats = safety_monitor_3d.compare_2d_vs_3d_detection(
                        hands, knives, frame
                    )
                    dangers_2d = comparison_stats["dangers_2d"]
                    dangers_3d = comparison_stats["dangers_3d"]
                else:
                    # Run only 2D safety
                    dangers_2d = safety_monitor_2d.detect_fingertip_dangers(hands, knives)
            
            # Render overlays on main frame FIRST
            if show_hands:
                overlay_renderer.draw_hands(frame, hands)
            
            if show_knives:
                overlay_renderer.draw_knives(frame, knives)
            
            if show_safety:
                if use_3d_safety and dangers_3d:
                    # Render 3D safety warnings
                    for danger in dangers_3d:
                        overlay_renderer.draw_3d_danger_warning(frame, danger)
                elif dangers_2d:
                    # Render 2D safety warnings
                    for danger in dangers_2d:
                        overlay_renderer.draw_danger_warning(frame, danger)
            
            # Generate depth map for visualization
            depth_map = None
            if show_depth:
                try:
                    depth_map = depth_estimator.estimate_depth(frame)
                except Exception as e:
                    print(f"⚠️ Depth estimation failed: {e}")
            
            # Create visualization AFTER overlays are applied
            if show_depth and depth_map is not None:
                # Create depth visualization
                if depth_modes[current_depth_mode] == "grayscale":
                    depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                    depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
                elif depth_modes[current_depth_mode] == "colormap":
                    depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                else:  # heatmap
                    depth_normalized = (depth_map / depth_estimator.max_depth * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Create side-by-side display
                combined_frame = np.hstack([frame, depth_vis])
            else:
                combined_frame = frame
                depth_vis = None
            
            # Update FPS counter
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                current_fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # Draw status information
            status_y = 30
            mode_text = "3D Safety Mode" if use_3d_safety else "2D Safety Mode"
            mode_color = (0, 255, 255) if use_3d_safety else (255, 255, 255)
            
            cv2.putText(combined_frame, mode_text, (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            if current_fps > 0:
                cv2.putText(combined_frame, f"FPS: {current_fps:.1f}", 
                           (20, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Safety statistics
            if use_3d_safety and comparison_stats:
                cv2.putText(combined_frame, f"2D Dangers: {comparison_stats['count_2d']}", 
                           (20, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined_frame, f"3D Dangers: {comparison_stats['count_3d']}", 
                           (20, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if comparison_stats['false_positive_reduction'] > 0:
                    cv2.putText(combined_frame, f"False Positives Reduced: {comparison_stats['false_positive_reduction']}", 
                               (20, status_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Depth mode indicator
            if show_depth:
                cv2.putText(combined_frame, f"Depth: {depth_modes[current_depth_mode].upper()}", 
                           (20, status_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw pause indicator
            if paused:
                cv2.putText(combined_frame, "PAUSED", 
                           (combined_frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Draw controls
            controls_text = "ESC: Exit | SPACE: Pause | 2/3: 2D/3D Mode | D: Depth | S: Screenshot | R: Reset"
            cv2.putText(combined_frame, controls_text, 
                       (20, combined_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow('3D Unified Detection: Hands + Knives + Depth Safety', combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = int(time.time())
            # Save main frame
            main_filename = f"3d_unified_frame_{timestamp}.jpg"
            cv2.imwrite(main_filename, frame)
            saved_files = [main_filename]
            
            # Save depth visualization if available
            if depth_vis is not None:
                depth_filename = f"3d_unified_depth_{timestamp}.jpg"
                cv2.imwrite(depth_filename, depth_vis)
                saved_files.append(depth_filename)
            
            print(f"📸 Screenshots saved: {', '.join(saved_files)}")
        elif key == ord('2'):  # 2
            use_3d_safety = False
            print("🔄 Switched to 2D Safety Mode")
        elif key == ord('3'):  # 3
            use_3d_safety = True
            print("🔄 Switched to 3D Safety Mode")
        elif key == ord('d') or key == ord('D'):  # D
            current_depth_mode = (current_depth_mode + 1) % len(depth_modes)
            print(f"🎨 Depth visualization mode: {depth_modes[current_depth_mode]}")
        elif key == ord('h') or key == ord('H'):  # H
            show_hands = not show_hands
            print(f"👋 Hand landmarks: {'ON' if show_hands else 'OFF'}")
        elif key == ord('k') or key == ord('K'):  # K
            show_knives = not show_knives
            print(f"🔪 Knife detection: {'ON' if show_knives else 'OFF'}")
        elif key == ord('w') or key == ord('W'):  # W
            show_safety = not show_safety
            print(f"⚠️ Safety warnings: {'ON' if show_safety else 'OFF'}")
        elif key == ord('r') or key == ord('R'):  # R
            safety_monitor_2d.reset_stats()
            safety_monitor_3d.reset_stats()
            safety_monitor_3d.reset_3d_stats()
            print("📊 Statistics reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("📊 FINAL 3D UNIFIED DETECTION STATISTICS")
    print("=" * 80)
    
    # 2D safety stats
    stats_2d = safety_monitor_2d.get_stats()
    print(f"\n📈 2D Safety Monitor:")
    print(f"  Total checks: {stats_2d['total_checks']}")
    print(f"  Danger events: {stats_2d['danger_events']}")
    print(f"  Danger rate: {stats_2d['danger_rate']:.2%}")
    
    # 3D safety stats
    stats_3d = safety_monitor_3d.get_3d_stats()
    print(f"\n📈 3D Safety Monitor:")
    print(f"  Total 3D checks: {stats_3d['total_3d_checks']}")
    print(f"  3D danger events: {stats_3d['danger_events_3d']}")
    print(f"  3D danger rate: {stats_3d['danger_rate_3d']:.2%}")
    print(f"  Avg depth estimation time: {stats_3d['avg_depth_estimation_time_ms']:.1f}ms")
    
    # Depth estimator stats
    depth_stats = depth_estimator.get_stats()
    print(f"\n📈 Depth Estimator:")
    print(f"  Model type: {depth_stats['model_type']}")
    print(f"  Frames processed: {depth_stats['frames_processed']}")
    print(f"  Avg inference time: {depth_stats['avg_inference_time_ms']:.1f}ms")
    print(f"  Estimated max FPS: {1000/depth_stats['avg_inference_time_ms']:.1f}")
    
    # Performance assessment
    target_fps = 15
    actual_fps = 1000 / depth_stats['avg_inference_time_ms'] if depth_stats['avg_inference_time_ms'] > 0 else 0
    
    if actual_fps >= target_fps:
        print(f"  ✅ Real-time performance achieved ({target_fps} FPS target)")
    else:
        print(f"  ⚠️ Performance below target ({target_fps} FPS)")
    
    print("\n👋 3D unified detection demo complete!")


if __name__ == "__main__":
    main()