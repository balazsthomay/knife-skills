#!/usr/bin/env python3
"""
Unified hand and knife detection webcam demo.
Combines MediaPipe hand tracking with YOLOv11 knife detection.
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.hand_detector import HandDetector
from detection.knife_detector import KnifeDetector
from ui.overlay_renderer import OverlayRenderer
from safety.safety_monitor import SafetyMonitor


def main():
    """Main unified detection demo"""
    print("🔪 🖐️ Starting Unified Hand + Knife Detection Demo")
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot")
    print("  H - Toggle hand landmarks")
    print("  K - Toggle knife detection")
    print("  W - Toggle safety warnings")
    
    # Initialize detectors
    try:
        hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ Hand detector initialized")
        
        knife_detector = KnifeDetector(
            model_path="models/knife_seg_phase1_frozen/weights/best.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
        print("✅ Knife detector initialized")
        
        overlay_renderer = OverlayRenderer(
            show_connections=True,
            show_landmarks=True,
            show_labels=False
        )
        print("✅ Overlay renderer initialized")
        
        safety_monitor = SafetyMonitor(
            critical_threshold=30,
            high_threshold=50,
            medium_threshold=80
        )
        print("✅ Safety monitor initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize detectors: {e}")
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
    print("🚀 Starting unified detection loop...")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    paused = False
    
    # Display toggles
    show_hands = True
    show_knives = True
    show_safety = True
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect hands and knives
            start_time = time.time()
            
            detected_hands = []
            detected_knives = []
            
            if show_hands:
                detected_hands = hand_detector.detect_hands(frame)
            
            if show_knives:
                detected_knives = knife_detector.detect_knives(frame)
            
            detection_time = (time.time() - start_time) * 1000
            
            # Render overlays
            if show_hands and detected_hands:
                frame = overlay_renderer.render_hands(frame, detected_hands)
            
            if show_knives and detected_knives:
                frame = overlay_renderer.render_knives(frame, detected_knives)
            
            # Detect and render safety dangers
            danger_events = []
            if show_safety and detected_hands and detected_knives:
                danger_events = safety_monitor.detect_fingertip_dangers(detected_hands, detected_knives)
                frame = overlay_renderer.render_safety_dangers(frame, danger_events)
            
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
        cv2.putText(frame, f"Hands: {len(detected_hands)}", (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Knives: {len(detected_knives)}", (20, status_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Dangers: {len(danger_events)}", (20, status_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_fps > 0:
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, status_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Detection: {detection_time:.1f}ms", (20, status_y + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw toggle status
        toggle_y = frame.shape[0] - 100
        toggle_text = f"Hands: {'ON' if show_hands else 'OFF'} | Knives: {'ON' if show_knives else 'OFF'} | Safety: {'ON' if show_safety else 'OFF'}"
        cv2.putText(frame, toggle_text, (20, toggle_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw pause indicator
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 80, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw controls
        controls_text = "ESC: Exit | SPACE: Pause | S: Screenshot | H: Hands | K: Knives | W: Safety"
        cv2.putText(frame, controls_text, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('Unified Hand + Knife Detection Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = int(time.time())
            filename = f"unified_detection_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('h') or key == ord('H'):  # H
            show_hands = not show_hands
            print(f"👋 Hand detection: {'ON' if show_hands else 'OFF'}")
        elif key == ord('k') or key == ord('K'):  # K
            show_knives = not show_knives
            print(f"🔪 Knife detection: {'ON' if show_knives else 'OFF'}")
        elif key == ord('w') or key == ord('W'):  # W
            show_safety = not show_safety
            print(f"⚠️  Safety warnings: {'ON' if show_safety else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    hand_stats = hand_detector.get_stats()
    knife_stats = knife_detector.get_stats()
    safety_stats = safety_monitor.get_stats()
    
    print("\n📊 Final Statistics:")
    print("  Hand Detection:")
    print(f"    Frames processed: {hand_stats['frames_processed']}")
    print(f"    Last detection count: {hand_stats['last_detection_count']}")
    
    print("  Knife Detection:")
    print(f"    Frames processed: {knife_stats['frames_processed']}")
    print(f"    Avg inference time: {knife_stats['avg_inference_time_ms']:.1f}ms")
    print(f"    Last detection count: {knife_stats['last_detection_count']}")
    
    print("  Safety Monitor:")
    print(f"    Total checks: {safety_stats['total_checks']}")
    print(f"    Danger events: {safety_stats['danger_events']}")
    print(f"    Danger rate: {safety_stats['danger_rate']:.2%}")
    
    print("👋 Unified demo complete!")


if __name__ == "__main__":
    main()