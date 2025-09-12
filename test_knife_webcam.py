#!/usr/bin/env python3
"""
Simple knife detection webcam demo.
Tests the trained YOLOv11 knife detection model in real-time.
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.knife_detector import KnifeDetector


def main():
    """Main demo function"""
    print("🔪 Starting Knife Detection Webcam Demo")
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot")
    
    # Initialize knife detector
    try:
        knife_detector = KnifeDetector(
            model_path="models/knife_seg_phase1_frozen/weights/best.pt",
            confidence_threshold=0.5,
            device="cpu"  # Change to "cuda" if you have CUDA available
        )
        print("✅ Knife detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize knife detector: {e}")
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
    print("🚀 Starting detection loop...")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect knives
            start_time = time.time()
            knives = knife_detector.detect_knives(frame)
            detection_time = (time.time() - start_time) * 1000
            
            # Draw knife detections
            for knife in knives:
                # Draw bounding box
                x1, y1, x2, y2 = knife.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan box
                
                # Draw segmentation mask with transparency
                if knife.mask is not None:
                    mask_colored = frame.copy()
                    mask_colored[knife.mask > 0] = [0, 165, 255]  # Orange mask
                    frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
                
                # Draw contour outline
                if knife.contour is not None:
                    cv2.drawContours(frame, [knife.contour], -1, (0, 165, 255), 2)
                
                # Draw tip and handle points
                cv2.circle(frame, knife.tip_point, 8, (0, 0, 255), -1)  # Red tip
                cv2.circle(frame, knife.handle_point, 8, (255, 0, 0), -1)  # Blue handle
                
                # Draw labels
                cv2.putText(frame, "TIP", 
                           (knife.tip_point[0] + 12, knife.tip_point[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "HANDLE", 
                           (knife.handle_point[0] + 12, knife.handle_point[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw info text
                info_text = f"Conf: {knife.confidence:.3f} | Length: {knife.blade_length_pixels:.0f}px"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
        
        # Draw status information
        status_y = 30
        cv2.putText(frame, f"Knives: {len(knives)}", (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_fps > 0:
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Inference: {detection_time:.1f}ms", (20, status_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw pause indicator
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 80, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw controls
        controls_text = "ESC: Exit | SPACE: Pause | S: Screenshot"
        cv2.putText(frame, controls_text, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('Knife Detection Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = int(time.time())
            filename = f"knife_detection_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    stats = knife_detector.get_stats()
    print("\n📊 Final Statistics:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")
    print("👋 Demo complete!")


if __name__ == "__main__":
    main()