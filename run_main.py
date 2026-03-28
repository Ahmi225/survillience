#!/usr/bin/env python3
"""
Simple Main Entry Point for Weapon Detection System
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main system
try:
    from core.integrated_gun_detection_system import IntegratedGunDetectionSystem
    print("✅ Using Core System with Advanced UI")
    SYSTEM_TYPE = "CORE"
except ImportError:
    print("❌ Core system not available")
    sys.exit(1)

def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("🎯 INTELLIGENT WEAPON DETECTION SYSTEM")
    print("   Advanced UI + Firebase Integration")
    print("=" * 80)
    
    # Model path
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure YOLO model is available")
        return
    
    try:
        # Initialize system
        print(f"🚀 Initializing system with {model_path}...")
        system = IntegratedGunDetectionSystem(model_path, camera_index=0)
        
        # Add Firebase callback if available
        if hasattr(system, 'firebase_rt') and system.firebase_rt.initialized:
            def firebase_callback(detection_data):
                """Send alerts to Firebase"""
                try:
                    # Use Wazirabad camera info for single camera
                    camera_info = {
                        'id': 'CAM_WZD_001',
                        'name': 'Wazirabad Security Camera',
                        'address': 'Wazirabad, Pakistan',
                        'city': 'Wazirabad',
                        'lat': 32.245430,
                        'lng': 74.163434,
                        'type': 'Laptop Camera'
                    }
                    
                    if system.firebase_rt.send_alert(detection_data, None, camera_info):
                        print("🔥 Firebase alert sent!")
                    
                except Exception as e:
                    print(f"❌ Firebase error: {e}")
            
            system.add_detection_callback(firebase_callback)
            print("✅ Firebase callback registered")
        
        # Run the system
        print("🚀 Starting weapon detection system...")
        system.run()
        
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
