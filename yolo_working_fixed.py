#!/usr/bin/env python3
"""Fixed YOLO analysis that's actually working."""

import os
import warnings
import json
import numpy as np

def setup_yolo_safely():
    """Setup YOLO with working configuration."""
    try:
        # Apply PyTorch fixes
        import torch
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
        
        import torch.serialization
        unsafe_globals = [
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.tasks.SegmentationModel',
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'collections.OrderedDict'
        ]
        
        for cls_name in unsafe_globals:
            try:
                torch.serialization.add_safe_globals([cls_name])
            except:
                pass
        
        # Patch torch.load temporarily
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        warnings.filterwarnings("ignore")
        
        from ultralytics import YOLO
        model = YOLO('yolov8n-seg.pt')
        
        # Restore torch.load
        torch.load = original_load
        
        print("✅ YOLO loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ YOLO setup failed: {e}")
        return None


def run_yolo_detection_fixed():
    """Run YOLO detection with proper error handling."""
    print("🎯 Running YOLO Detection on Your Pliers Image")
    print("=" * 50)
    
    model = setup_yolo_safely()
    if model is None:
        return False
    
    try:
        import cv2
        
        img_path = "/app/output/enhanced_converted_image.jpg"
        img = cv2.imread(img_path)
        
        if img is None:
            print("❌ Could not load image")
            return False
        
        print(f"Image loaded: {img.shape[1]}×{img.shape[0]} pixels")
        
        # Run YOLO with low confidence to catch your pliers
        print("🔍 Running YOLO inference...")
        results = model(img_path, conf=0.05, iou=0.3, verbose=False)
        
        detections = []
        vis_img = img.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    width_px = x2 - x1
                    height_px = y2 - y1
                    area_px = width_px * height_px
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Convert to mm using AprilTag scale
                    scale_px_per_mm = 4.15
                    length_mm = max(width_px, height_px) / scale_px_per_mm
                    width_mm = min(width_px, height_px) / scale_px_per_mm
                    area_mm2 = area_px / (scale_px_per_mm ** 2)
                    aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
                    
                    # Distance from center
                    img_center_x, img_center_y = img.shape[1] // 2, img.shape[0] // 2
                    distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                    
                    detection = {
                        'id': i + 1,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, width_px, height_px],
                        'center': [center_x, center_y],
                        'distance_from_center': round(distance_from_center, 1),
                        'dimensions_mm': [round(length_mm, 1), round(width_mm, 1)],
                        'area_mm2': round(area_mm2, 1),
                        'aspect_ratio': round(aspect_ratio, 2)
                    }
                    
                    detections.append(detection)
                    
                    # Draw visualization
                    color = colors[i % len(colors)]
                    
                    # Thick bounding box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 4)
                    
                    # Labels with background
                    label = f"#{i+1}: {class_name}"
                    conf_label = f"Conf: {confidence:.1%}"
                    dims_label = f"{length_mm:.1f}×{width_mm:.1f}mm"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    
                    # Background rectangles for text
                    (tw1, th1), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    (tw2, th2), _ = cv2.getTextSize(conf_label, font, 0.5, 1)
                    (tw3, th3), _ = cv2.getTextSize(dims_label, font, 0.5, 1)
                    
                    cv2.rectangle(vis_img, (x1, y1-50), (x1 + max(tw1, tw2, tw3), y1), (255, 255, 255), -1)
                    
                    cv2.putText(vis_img, label, (x1, y1-35), font, font_scale, color, thickness)
                    cv2.putText(vis_img, conf_label, (x1, y1-20), font, 0.5, color, 1)
                    cv2.putText(vis_img, dims_label, (x1, y1-5), font, 0.5, color, 1)
                    
                    # Mark center point
                    cv2.circle(vis_img, (center_x, center_y), 8, color, -1)
                    cv2.circle(vis_img, (center_x, center_y), 8, (255, 255, 255), 2)
        
        # Mark image center
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.circle(vis_img, img_center, 12, (255, 255, 255), -1)
        cv2.circle(vis_img, img_center, 12, (0, 0, 0), 3)
        cv2.putText(vis_img, "IMAGE CENTER", (img_center[0] + 20, img_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add title
        title = "YOLO Detection Results - Pliers Analysis"
        cv2.putText(vis_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(vis_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save visualization
        vis_path = "/app/output/yolo_pliers_detection.jpg"
        cv2.imwrite(vis_path, vis_img)
        
        # Save results
        results_data = {
            'model': 'yolov8n-seg',
            'total_detections': len(detections),
            'detections': detections,
            'scale_px_per_mm': 4.15,
            'analysis_notes': [
                'Scissors detection likely indicates pliers (similar tool)',
                'Low confidence is normal for specialized tools',
                'Check visualization to see actual detected areas'
            ]
        }
        
        with open('/app/output/yolo_pliers_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print detailed results
        print(f"\n🎯 YOLO DETECTION RESULTS:")
        print("=" * 50)
        
        if detections:
            for det in detections:
                print(f"\n**DETECTION #{det['id']}:**")
                print(f"  YOLO Class: {det['class']}")
                print(f"  Confidence: {det['confidence']:.1%}")
                print(f"  📏 Dimensions: {det['dimensions_mm'][0]} × {det['dimensions_mm'][1]} mm")
                print(f"  📐 Area: {det['area_mm2']} mm²")
                print(f"  📍 Location: {det['center']} (center)")
                print(f"  📏 Distance from center: {det['distance_from_center']} pixels")
                print(f"  📊 Aspect ratio: {det['aspect_ratio']}")
                
                # Pliers analysis
                length, width = det['dimensions_mm']
                is_tool_class = det['class'] in ['scissors', 'knife', 'spoon', 'fork']
                is_pliers_size = 50 < length < 300 and 5 < width < 100
                is_reasonably_centered = det['distance_from_center'] < 500
                
                print(f"\n  🎯 PLIERS ANALYSIS:")
                if is_tool_class:
                    print(f"    ✅ TOOL CLASS: '{det['class']}' is tool-related")
                else:
                    print(f"    ❓ CLASS: '{det['class']}' is not typically a tool")
                
                if is_pliers_size:
                    print(f"    ✅ SIZE: {length}×{width}mm fits pliers range")
                else:
                    print(f"    ❌ SIZE: {length}×{width}mm outside typical pliers size")
                
                if is_reasonably_centered:
                    print(f"    ✅ POSITION: Well positioned in image")
                else:
                    print(f"    ❌ POSITION: Too far from center")
                
                if is_tool_class and is_pliers_size and is_reasonably_centered:
                    print(f"    🎉 **VERDICT: LIKELY YOUR PLIERS!**")
                elif is_tool_class or is_pliers_size:
                    print(f"    🤔 **VERDICT: POSSIBLE PLIERS**")
                else:
                    print(f"    ❌ **VERDICT: UNLIKELY TO BE PLIERS**")
        else:
            print("❌ No objects detected")
            print("\nThis could mean:")
            print("  - Pliers have very low contrast")
            print("  - Pliers don't match YOLO's training data")
            print("  - Need to lower confidence threshold further")
        
        print(f"\n📁 Generated files:")
        print(f"  📷 yolo_pliers_detection.jpg - Visual detection results")
        print(f"  📄 yolo_pliers_results.json - Detailed data")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return False


if __name__ == "__main__":
    success = run_yolo_detection_fixed()
    exit(0 if success else 1)