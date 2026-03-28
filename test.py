"""
Simplified Real-time Violence Detection Test
"""

import cv2
import numpy as np
import sys
import time
from collections import deque

sys.path.append('.')

# Try different import paths
try:
    from fight_detection.fight_detector import ViolenceDetector
except:
    try:
        from fight_detector import ViolenceDetector
    except:
        print("❌ Cannot import ViolenceDetector")
        sys.exit(1)


def test_real_time():
    """Simple violence detection test with filtering"""
    
    print("="*60)
    print("🎥 VIOLENCE DETECTION TEST")
    print("="*60)
    
    # Initialize detector
    print("\n📦 Loading violence detector...")
    
    # Try different model paths
    model_paths = [
        "models/violence.pt",
        "models/violence.pth", 
        "violence.pt",
        "violence.pth"
    ]
    
    detector = None
    for path in model_paths:
        try:
            detector = ViolenceDetector(path)
            if detector.model is not None:
                print(f"✓ Model loaded: {path}")
                break
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
    
    if detector is None or detector.model is None:
        print("❌ Could not load any violence model!")
        print("   Please ensure models/violence.pt or models/violence.pth exists")
        return
    
    # Detection history for filtering
    detection_history = deque(maxlen=10)  # Store last 10 detections
    violence_active = False
    last_violence_time = 0
    violence_count = 0
    
    # Settings
    confidence_threshold = 0.6  # Adjust this
    consecutive_frames_needed = 3  # Need this many consecutive detections
    
    print(f"\n⚙️ Settings:")
    print(f"   Confidence Threshold: {confidence_threshold}")
    print(f"   Consecutive Frames Needed: {consecutive_frames_needed}")
    
    # Initialize camera
    print("\n🎥 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return
    
    print("\n✅ Camera opened!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("  '+' - Increase sensitivity (lower threshold)")
    print("  '-' - Decrease sensitivity (higher threshold)")
    print("  'c' - Clear history")
    print("\n⚠️ Make violent movements to test detection!")
    print("-"*60)
    
    frame_count = 0
    fps = 0
    fps_counter = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error")
            break
        
        frame_count += 1
        
        # Create test detection (full frame)
        test_persons = [
            {'id': 1, 'bbox': [0, 0, frame.shape[1], frame.shape[0]]}
        ]
        
        # Detect violence
        start_time = time.time()
        results = detector.detect_violence_in_frame(frame, test_persons)
        inference_time = (time.time() - start_time) * 1000
        
        # Update FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        # Process results
        is_violent = False
        confidence = 0
        
        for person_id, info in results.items():
            if info.get('violence_detected', False):
                confidence = info.get('confidence', 0)
                if confidence >= confidence_threshold:
                    is_violent = True
        
        # Add to history
        detection_history.append(is_violent)
        
        # Check if we have enough consecutive detections
        if len(detection_history) >= consecutive_frames_needed:
            recent_frames = list(detection_history)[-consecutive_frames_needed:]
            confirmed_violence = all(recent_frames)
        else:
            confirmed_violence = False
        
        # Update violence state
        if confirmed_violence and not violence_active:
            violence_active = True
            last_violence_time = time.time()
            violence_count += 1
            print(f"🚨 VIOLENCE CONFIRMED! (Frame {frame_count}, Confidence: {confidence:.2f})")
            
        elif not confirmed_violence and violence_active:
            if time.time() - last_violence_time > 1.0:
                violence_active = False
                print(f"✅ Violence ended (Frame {frame_count})")
        
        # Draw on frame
        if confirmed_violence:
            # Flashing red border
            border_intensity = int(127 + 128 * np.sin(time.time() * 10))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                        (0, 0, border_intensity), 5)
            
            # Warning text
            cv2.putText(frame, "⚠️ VIOLENCE DETECTED! ⚠️", 
                      (frame.shape[1]//2-200, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), 
                        (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw information panel
        panel_y = 10
        panel_height = 140
        cv2.rectangle(frame, (5, panel_y), (250, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, panel_y), (250, panel_y + panel_height), (0, 255, 0), 1)
        
        y_offset = panel_y + 25
        cv2.putText(frame, f"FPS: {fps}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        cv2.putText(frame, f"Violence Active: {'YES' if violence_active else 'NO'}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 0, 255) if violence_active else (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(frame, f"Total Alerts: {violence_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, f"History: {sum(detection_history)}/{len(detection_history)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show recent detection graph
        if detection_history:
            graph_x = 10
            graph_y = panel_y + panel_height - 30
            bar_width = 8
            for i, detected in enumerate(list(detection_history)[-15:]):
                color = (0, 0, 255) if detected else (100, 100, 100)
                cv2.rectangle(frame, (graph_x + i*bar_width, graph_y), 
                            (graph_x + i*bar_width + bar_width-1, graph_y + 20), 
                            color, -1)
        
        # Show threshold and instructions
        cv2.putText(frame, f"Threshold: {confidence_threshold:.2f}", 
                   (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 0), 1)
        cv2.putText(frame, "Q:Quit S:Save +/-:Sensitivity C:Clear", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (150, 150, 150), 1)
        
        # Show frame
        cv2.imshow("Violence Detection Test", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"violence_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('+'):
            confidence_threshold = max(0.3, confidence_threshold - 0.05)
            print(f"✓ Sensitivity increased (threshold: {confidence_threshold:.2f})")
        elif key == ord('-'):
            confidence_threshold = min(0.9, confidence_threshold + 0.05)
            print(f"✓ Sensitivity decreased (threshold: {confidence_threshold:.2f})")
        elif key == ord('c'):
            detection_history.clear()
            violence_active = False
            print("✓ History cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Violence Alerts Triggered: {violence_count}")
    print(f"Average FPS: {fps:.1f}")
    print("\n✅ Test completed!")


def check_model():
    """Check if model exists and is loadable"""
    import os
    
    print("Checking model files...")
    print("-"*40)
    
    model_paths = [
        "models/violence.pt",
        "models/violence.pth",
        "violence.pt", 
        "violence.pth"
    ]
    
    found = False
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Found: {path} ({size:.1f} MB)")
            found = True
        else:
            print(f"✗ Not found: {path}")
    
    if not found:
        print("\n❌ No violence model found!")
        print("\nPlease place your model file in one of these locations:")
        print("  - models/violence.pt")
        print("  - models/violence.pth")
        print("  - violence.pt")
        print("  - violence.pth")
        return False
    
    return True


if __name__ == "__main__":
    # First check if model exists
    if check_model():
        test_real_time()
    else:
        print("\nPlease ensure the model file is in the correct location.")