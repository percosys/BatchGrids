#!/usr/bin/env python3
"""Show exactly what YOLO segmentation model detected in the image."""

import os
import sys
import json
import numpy as np
import warnings

try:
    import cv2
    import torch
    from ultralytics import YOLO
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def setup_yolo_safely():
    """Setup YOLO with all compatibility fixes."""
    try:
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Set environment variables for compatibility
        os.environ['YOLO_VERBOSE'] = 'False'
        
        # Try to add safe globals for PyTorch
        try:
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.SegmentationModel',
                'ultralytics.nn.tasks.DetectionModel'
            ])
        except:
            pass
        
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8n-seg.pt')
        print("✅ YOLO model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        return None


def run_yolo_detection_with_visualization(model, img_path, output_dir):
    """Run YOLO and create detailed visualization of all detections."""
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return None, []
    
    print(f"Processing image: {img.shape[1]}×{img.shape[0]} pixels")
    
    # Run YOLO with lower confidence to see more detections
    results = model(img, conf=0.1, iou=0.45, verbose=False)  # Very low confidence
    
    all_detections = []
    
    # Create visualization image
    vis_img = img.copy()
    
    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (255, 192, 203) # Pink
    ]
    
    detection_count = 0
    
    for result in results:
        boxes = result.boxes
        masks = result.masks if hasattr(result, 'masks') else None
        
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Get detection info
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                color = colors[detection_count % len(colors)]
                detection_count += 1
                
                # Draw bounding box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class and confidence
                label = f"{detection_count}. {class_name} ({confidence:.2f})"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                cv2.rectangle(vis_img, (x1, y1 - text_h - baseline - 5), 
                             (x1 + text_w, y1), color, -1)
                cv2.putText(vis_img, label, (x1, y1 - baseline - 2), 
                           font, font_scale, (255, 255, 255), thickness)
                
                # Store detection info
                detection = {
                    'id': detection_count,
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # x, y, width, height
                    'bbox_coords': [x1, y1, x2, y2],      # x1, y1, x2, y2
                    'area_px': (x2 - x1) * (y2 - y1),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'has_mask': False
                }
                
                # Add mask information if available
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    detection['has_mask'] = True
                    detection['mask_shape'] = mask.shape
                    
                    # Draw mask overlay (semi-transparent)
                    if mask.shape == (img.shape[0], img.shape[1]):
                        mask_colored = np.zeros_like(img)
                        mask_colored[mask > 0.5] = color
                        vis_img = cv2.addWeighted(vis_img, 1.0, mask_colored, 0.3, 0)
                
                all_detections.append(detection)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'yolo_detections_visualization.jpg')
    cv2.imwrite(vis_path, vis_img)
    print(f"Visualization saved: {vis_path}")
    
    # Create detailed results summary
    summary_path = os.path.join(output_dir, 'yolo_detection_summary.json')
    summary = {
        'total_detections': len(all_detections),
        'image_size': [img.shape[1], img.shape[0]],  # width, height
        'model_used': 'yolov8n-seg',
        'confidence_threshold': 0.1,
        'detections': all_detections,
        'class_counts': {}
    }
    
    # Count detections by class
    for det in all_detections:
        class_name = det['class']
        summary['class_counts'][class_name] = summary['class_counts'].get(class_name, 0) + 1
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return vis_img, all_detections


def analyze_detections(detections, scale_px_per_mm=4.15):
    """Analyze all YOLO detections and convert to real-world measurements."""
    if not detections:
        return []
    
    analyzed = []
    
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Calculate dimensions in mm
        length_px = max(w, h)
        width_px = min(w, h)
        
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = det['area_px'] / (scale_px_per_mm ** 2)
        
        analysis = {
            'detection_id': det['id'],
            'yolo_class': det['class'],
            'confidence': det['confidence'],
            'dimensions_mm': {
                'length': round(length_mm, 1),
                'width': round(width_mm, 1),
                'area': round(area_mm2, 1)
            },
            'dimensions_px': {
                'length': length_px,
                'width': width_px,
                'area': det['area_px']
            },
            'bbox_coords': det['bbox_coords'],
            'center': det['center'],
            'aspect_ratio': round(length_px / width_px if width_px > 0 else 1, 2),
            'has_segmentation_mask': det['has_mask']
        }
        
        analyzed.append(analysis)
    
    # Sort by confidence
    analyzed.sort(key=lambda x: x['confidence'], reverse=True)
    
    return analyzed


def main():
    """Run YOLO detection with full visualization."""
    if not HAS_DEPS:
        print("❌ Missing dependencies")
        return 1
    
    print("👁️  YOLO Segmentation Model Visualization")
    print("=" * 50)
    
    # Setup YOLO model
    model = setup_yolo_safely()
    if model is None:
        print("❌ Could not load YOLO model")
        return 1
    
    # Use the processed image
    img_path = "/app/output/enhanced_converted_image.jpg"
    output_dir = "/app/output"
    
    # Run detection with visualization
    vis_img, detections = run_yolo_detection_with_visualization(model, img_path, output_dir)
    
    if detections:
        print(f"\n🎯 YOLO DETECTED {len(detections)} OBJECTS:")
        print("=" * 50)
        
        # Analyze all detections
        analyzed = analyze_detections(detections)
        
        # Show results
        for analysis in analyzed:
            print(f"\n**DETECTION #{analysis['detection_id']}:**")
            print(f"  Class: {analysis['yolo_class']}")
            print(f"  Confidence: {analysis['confidence']:.1%}")
            print(f"  Dimensions: {analysis['dimensions_mm']['length']} × {analysis['dimensions_mm']['width']} mm")
            print(f"  Area: {analysis['dimensions_mm']['area']} mm²")
            print(f"  Aspect ratio: {analysis['aspect_ratio']}")
            print(f"  Segmentation mask: {'✅' if analysis['has_segmentation_mask'] else '❌'}")
            print(f"  Center position: {analysis['center']}")
        
        # Save detailed analysis
        analysis_path = "/app/output/yolo_detailed_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analyzed, f, indent=2)
        
        # Summary by class
        print(f"\n📊 DETECTION SUMMARY:")
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} detection(s)")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"  📷 yolo_detections_visualization.jpg - Visual overlay of all detections")
        print(f"  📄 yolo_detection_summary.json - Raw detection data")
        print(f"  📊 yolo_detailed_analysis.json - Analysis with measurements")
        
        return 0
    else:
        print("❌ No objects detected by YOLO")
        return 1


if __name__ == "__main__":
    exit(main())