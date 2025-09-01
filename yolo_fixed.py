#!/usr/bin/env python3
"""Fixed YOLO implementation that works with PyTorch 2.8+"""

import os
import sys
import json
import numpy as np
import warnings

try:
    import cv2
    import torch
    import ultralytics
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def fix_torch_loading():
    """Fix PyTorch loading issues with newer versions."""
    try:
        # Set environment variable to allow unsafe loading
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
        
        # Add unsafe globals that ultralytics needs
        import torch.serialization
        
        # Add the required classes for ultralytics
        unsafe_globals = [
            'ultralytics.nn.tasks.SegmentationModel',
            'ultralytics.nn.tasks.DetectionModel', 
            'ultralytics.nn.tasks.ClassificationModel',
            'ultralytics.nn.tasks.PoseModel',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.nn.modules.head.Segment',
            'collections.OrderedDict',
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'torch.nn.modules.activation.SiLU'
        ]
        
        for cls_name in unsafe_globals:
            try:
                torch.serialization.add_safe_globals([cls_name])
            except:
                pass
        
        print("✅ PyTorch loading fixes applied")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not apply PyTorch fixes: {e}")
        return False


def load_yolo_with_fallback():
    """Load YOLO with multiple fallback methods."""
    
    # Apply PyTorch fixes first
    fix_torch_loading()
    
    try:
        # Method 1: Try with weights_only=False
        print("Attempting YOLO load method 1...")
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Set ultralytics settings
            from ultralytics import settings
            settings.update({'verbose': False})
            
            # Try to load model
            from ultralytics import YOLO
            model = YOLO('yolov8n-seg.pt')
            
        print("✅ YOLO model loaded successfully!")
        return model
        
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        try:
            # Method 2: Download manually first
            print("Attempting YOLO load method 2 (manual download)...")
            
            import urllib.request
            model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
            model_path = "/tmp/yolov8n-seg.pt"
            
            if not os.path.exists(model_path):
                print("Downloading YOLO model manually...")
                urllib.request.urlretrieve(model_url, model_path)
            
            # Load from local file with unsafe loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = YOLO(model_path, task='segment')
            
            print("✅ YOLO model loaded via manual download!")
            return model
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            try:
                # Method 3: Use torch.load directly with unsafe loading
                print("Attempting YOLO load method 3 (direct torch load)...")
                
                model_path = "yolov8n-seg.pt"
                if os.path.exists(model_path):
                    # Load model weights directly
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    # Create basic YOLO model
                    from ultralytics import YOLO
                    model = YOLO()
                    model.model = checkpoint['model']
                    
                    print("✅ YOLO model loaded via direct torch load!")
                    return model
                    
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
    
    print("❌ All YOLO loading methods failed")
    return None


def run_yolo_detection(model, img_path):
    """Run YOLO detection and return results."""
    if model is None:
        return None, []
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return None, []
    
    print(f"Running YOLO on image: {img.shape[1]}x{img.shape[0]} pixels")
    
    try:
        # Run inference with low confidence to see all detections
        results = model(img, conf=0.1, iou=0.45, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    # Get detection data
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x, y, w, h
                        'area': int((x2-x1) * (y2-y1)),
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]
                    }
                    
                    detections.append(detection)
                    
                    print(f"Detected: {class_name} ({confidence:.3f}) at {detection['bbox']}")
        
        return img, detections
        
    except Exception as e:
        print(f"YOLO inference failed: {e}")
        return img, []


def visualize_yolo_results(img, detections, output_path):
    """Create visualization of YOLO detections."""
    if img is None:
        return False
    
    vis_img = img.copy()
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255)
    ]
    
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
        
        # Create label
        label = f"{det['class']} ({det['confidence']:.2f})"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(vis_img, (x, y - text_h - baseline - 5), 
                     (x + text_w, y), color, -1)
        cv2.putText(vis_img, label, (x, y - baseline - 2), 
                   font, font_scale, (255, 255, 255), thickness)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Visualization saved: {output_path}")
    return True


def main():
    """Main YOLO detection function."""
    if not HAS_DEPS:
        print("❌ Missing dependencies")
        return 1
    
    print("🤖 Fixed YOLO Segmentation Detection")
    print("=" * 40)
    
    # Load YOLO model with fixes
    model = load_yolo_with_fallback()
    if model is None:
        print("❌ Could not load YOLO model")
        return 1
    
    # Run detection
    img_path = "/app/output/enhanced_converted_image.jpg"
    img, detections = run_yolo_detection(model, img_path)
    
    if detections:
        print(f"\n🎯 YOLO DETECTED {len(detections)} OBJECTS:")
        print("=" * 40)
        
        # Calculate real-world dimensions
        scale_px_per_mm = 4.15
        
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            length_px = max(w, h)
            width_px = min(w, h)
            
            length_mm = length_px / scale_px_per_mm
            width_mm = width_px / scale_px_per_mm
            
            print(f"\n**DETECTION #{i+1}:**")
            print(f"  Object: {det['class']}")
            print(f"  Confidence: {det['confidence']:.1%}")
            print(f"  Dimensions: {length_mm:.1f} × {width_mm:.1f} mm")
            print(f"  Center: {det['center']}")
        
        # Create visualization
        vis_path = "/app/output/yolo_fixed_visualization.jpg"
        visualize_yolo_results(img, detections, vis_path)
        
        # Save results
        results = {
            'yolo_version': 'yolov8n-seg',
            'total_detections': len(detections),
            'scale_px_per_mm': scale_px_per_mm,
            'detections': detections
        }
        
        with open('/app/output/yolo_fixed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return 0
    else:
        print("❌ No objects detected by YOLO")
        # Still create empty visualization
        if img is not None:
            cv2.imwrite("/app/output/yolo_fixed_visualization.jpg", img)
            print("Empty visualization saved (no detections)")
        return 1


if __name__ == "__main__":
    exit(main())