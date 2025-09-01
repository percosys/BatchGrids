#!/usr/bin/env python3
"""Safe YOLO tool detection with PyTorch compatibility fix."""

import os
import sys
import json
import numpy as np
import warnings

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    import torch
    from ultralytics import YOLO
    HAS_DEPS = True
    print("✓ Dependencies available")
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def load_yolo_model_safe():
    """Load YOLO model with PyTorch security workaround."""
    print("\n=== Loading YOLO Model (Safe Mode) ===")
    try:
        # Set PyTorch to allow unsafe loading for YOLO models
        import torch.serialization
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])
        
        # Try with weights_only=False for compatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO('yolov8n-seg.pt', verbose=False)
        
        print("✓ YOLOv8n-seg model loaded successfully")
        return model
    except Exception as e:
        print(f"First attempt failed: {e}")
        
        # Fallback: try downloading a different way
        try:
            print("Trying alternative loading method...")
            os.environ['YOLO_VERBOSE'] = 'False'
            
            # Force download and load
            model = YOLO('yolov8n-seg.pt', verbose=False)
            print("✓ Alternative method succeeded")
            return model
        except Exception as e2:
            print(f"✗ All loading methods failed: {e2}")
            return None


def detect_any_objects(model, img):
    """Detect any objects that could be tools using YOLO."""
    print("\n=== YOLO Object Detection ===")
    
    if model is None:
        print("No model available")
        return None
    
    try:
        # Run inference with lower confidence threshold
        results = model(img, conf=0.15, iou=0.45, verbose=False)
        
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    # Get class information
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    
                    # Calculate area and aspect ratio
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'width': width,
                        'height': height
                    }
                    
                    all_detections.append(detection)
                    print(f"Detected: {class_name} (conf: {confidence:.3f}, area: {area:.0f})")
        
        if all_detections:
            # Sort by area * confidence (larger, more confident objects first)
            all_detections.sort(key=lambda x: x['area'] * x['confidence'], reverse=True)
            
            # Filter for reasonable tool-like objects
            potential_tools = []
            for det in all_detections:
                # Look for elongated objects or specific tool classes
                if (det['aspect_ratio'] > 1.5 or  # Elongated
                    det['class'] in ['knife', 'scissors', 'spoon', 'fork', 'bottle', 'cell phone'] or  # Tool-like
                    det['confidence'] > 0.4):  # High confidence anything
                    potential_tools.append(det)
            
            if potential_tools:
                best = potential_tools[0]
                print(f"\nBest tool candidate: {best['class']}")
                print(f"  Confidence: {best['confidence']:.3f}")
                print(f"  Dimensions: {best['width']:.0f}×{best['height']:.0f} px")
                print(f"  Area: {best['area']:.0f} px²")
                print(f"  Aspect ratio: {best['aspect_ratio']:.2f}")
                return best
            else:
                print("No tool-like objects found")
                if all_detections:
                    fallback = all_detections[0]
                    print(f"Using largest object: {fallback['class']} (conf: {fallback['confidence']:.3f})")
                    return fallback
        
        print("No objects detected")
        return None
        
    except Exception as e:
        print(f"Detection failed: {e}")
        return None


def calculate_tool_dimensions(detection, scale_px_per_mm=4.15):
    """Calculate tool dimensions from detection."""
    print("\n=== Tool Dimension Calculation ===")
    
    if not detection:
        return None
    
    # Get dimensions from bounding box
    width_px = detection['width']
    height_px = detection['height']
    area_px = detection['area']
    
    # Determine length and width (length is longer dimension)
    length_px = max(width_px, height_px)
    width_px_final = min(width_px, height_px)
    
    # Convert to millimeters
    length_mm = length_px / scale_px_per_mm
    width_mm = width_px_final / scale_px_per_mm
    area_mm2 = area_px / (scale_px_per_mm ** 2)
    
    dimensions = {
        'length_mm': round(length_mm, 1),
        'width_mm': round(width_mm, 1),
        'area_mm2': round(area_mm2, 1),
        'length_px': round(length_px, 1),
        'width_px': round(width_px_final, 1),
        'area_px': int(area_px),
        'px_per_mm': scale_px_per_mm,
        'bbox': detection['bbox']
    }
    
    print(f"Tool Dimensions:")
    print(f"  Length: {dimensions['length_mm']} mm ({dimensions['length_px']} px)")
    print(f"  Width: {dimensions['width_mm']} mm ({dimensions['width_px']} px)")
    print(f"  Area: {dimensions['area_mm2']} mm²")
    print(f"  Scale: {scale_px_per_mm:.2f} px/mm")
    
    return dimensions


def main():
    """Run YOLO tool detection."""
    if not HAS_DEPS:
        return 1
    
    print("🤖 BatchGrids YOLO Safe Detection Pipeline")
    print("=" * 50)
    
    # Use existing processed image
    img_path = "/app/output/enhanced_converted_image.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load processed image")
        return 1
    
    print(f"Processing image: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Load YOLO model safely
    model = load_yolo_model_safe()
    if model is None:
        print("❌ Could not load YOLO model")
        return 1
    
    # Detect objects
    detection = detect_any_objects(model, img)
    
    if detection:
        # Calculate dimensions using AprilTag scale
        dimensions = calculate_tool_dimensions(detection, scale_px_per_mm=4.15)
        
        # Save results
        results = {
            'detection': {
                'class': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            },
            'dimensions': dimensions,
            'method': 'YOLO_bounding_box',
            'scale_source': 'AprilTag_calibration',
            'status': 'success'
        }
        
        with open('/app/output/yolo_safe_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        print("🎯 YOLO Detection Success!")
        print(f"  Object: {detection['class']}")
        print(f"  Confidence: {detection['confidence']:.1%}")
        print(f"  **Height: {dimensions['length_mm']} mm**")
        print(f"  **Width: {dimensions['width_mm']} mm**")
        print(f"  **Area: {dimensions['area_mm2']} mm²**")
        print("=" * 50)
        
        return 0
    else:
        print("\n❌ No objects detected")
        return 1


if __name__ == "__main__":
    exit(main())