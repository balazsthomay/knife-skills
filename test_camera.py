#!/usr/bin/env python3
import cv2

def test_camera():
    """Quick test to verify camera access works"""
    print("Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    # Test frame capture
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from camera")
        cap.release()
        return False
    
    print(f"✅ Camera working! Frame size: {frame.shape}")
    print("Press any key to close the test window...")
    
    # Show live video feed
    print("Showing live video feed. Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
            
        cv2.imshow('Camera Test - Live Feed', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    return True

if __name__ == "__main__":
    test_camera()