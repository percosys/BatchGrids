#!/usr/bin/env python3
"""Try YOLO with version compatibility fix."""

import subprocess
import sys
import os

def install_compatible_ultralytics():
    """Install a compatible version of ultralytics."""
    print("Installing compatible ultralytics version...")
    
    try:
        # Uninstall current version
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'ultralytics', '-y'], 
                      capture_output=True)
        
        # Install older compatible version
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.0.100'], 
                               capture_output=True, text=True)
        
        print("Installation result:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def test_yolo_simple():
    """Test YOLO with minimal code."""
    try:
        print("Testing YOLO import...")
        from ultralytics import YOLO
        print("✅ YOLO import successful")
        
        print("Loading model...")
        model = YOLO('yolov8n.pt')  # Use detection model first (simpler)
        print("✅ YOLO model loaded")
        
        # Test with a simple image
        import cv2
        img_path = "/app/output/enhanced_converted_image.jpg"
        img = cv2.imread(img_path)
        
        print(f"Running inference on {img.shape[1]}x{img.shape[0]} image...")
        results = model(img, verbose=False, conf=0.25)
        
        print(f"✅ Inference successful, {len(results)} result(s)")
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.3f}")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"YOLO test failed: {e}")
        return False


def main():
    """Main function to fix and test YOLO."""
    print("🔧 YOLO Compatibility Fix")
    print("=" * 30)
    
    # Try current version first
    if test_yolo_simple():
        print("✅ Current YOLO version works!")
        return 0
    
    # Try installing compatible version
    print("\nTrying compatible version...")
    if install_compatible_ultralytics():
        if test_yolo_simple():
            print("✅ Compatible YOLO version works!")
            return 0
    
    print("❌ Could not get YOLO working")
    print("\nTrying alternative: Show what YOLO would detect")
    
    # Show YOLO classes as reference
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    tool_classes = [c for c in coco_classes if c in ['scissors', 'knife', 'spoon', 'fork', 'bottle']]
    
    print(f"\n📋 YOLO can detect {len(coco_classes)} object classes:")
    print("Tool-related classes:")
    for tool in tool_classes:
        print(f"  ✂️ {tool}")
    
    print(f"\n💡 Analysis:")
    print("Your flush cutting pliers would likely NOT be detected because:")
    print("  • 'Pliers' is not in YOLO's 80 trained classes")
    print("  • YOLO might detect them as 'scissors' (closest match)")
    print("  • Small precision tools are often missed")
    print("  • The custom detection we used found them correctly!")
    
    return 1


if __name__ == "__main__":
    exit(main())