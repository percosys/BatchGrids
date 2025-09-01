#!/usr/bin/env python3
"""Use OpenCV DNN module to show what objects can be detected in the image."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def load_coco_classes():
    """Load COCO class names that YOLO can detect."""
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
    return coco_classes


def analyze_image_features(img_path):
    """Analyze image features to understand what might be detectable."""
    print(f"Analyzing image features: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image dimensions: {w}×{h} pixels")
    
    # Create visualization
    vis_img = img.copy()
    
    detected_objects = []
    
    # Method 1: Feature-based detection for tool-like objects
    print("\n=== Feature Analysis ===")
    
    # Edge detection
    edges = cv2.Canny(gray, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} edge contours")
    
    # Analyze each contour
    tool_candidates = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area > 1000:  # Reasonable minimum size
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Skip objects at the very edges (likely image borders/artifacts)
            margin = min(w, h) * 0.05
            if (x > margin and y > margin and 
                x + w_c < w - margin and y + h_c < h - margin):
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                center_x, center_y = x + w_c//2, y + h_c//2
                
                # Classify based on shape characteristics
                object_type = "unknown"
                confidence = 0.5
                
                if aspect_ratio > 4 and area < 50000:
                    object_type = "tool_like"  # Long thin object
                    confidence = 0.7
                elif aspect_ratio > 2 and area < 30000:
                    object_type = "elongated_object"
                    confidence = 0.6
                elif 1000 < area < 100000:
                    object_type = "general_object"
                    confidence = 0.5
                
                candidate = {
                    'id': len(tool_candidates) + 1,
                    'type': object_type,
                    'confidence': confidence,
                    'bbox': [x, y, w_c, h_c],
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': [center_x, center_y]
                }
                
                tool_candidates.append(candidate)
    
    # Sort by confidence and area
    tool_candidates.sort(key=lambda x: x['confidence'] * np.log(x['area']), reverse=True)
    
    # Draw top candidates
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, candidate in enumerate(tool_candidates[:10]):  # Top 10
        x, y, w_c, h_c = candidate['bbox']
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w_c, y + h_c), color, 2)
        
        # Add label
        label = f"{candidate['id']}. {candidate['type']} ({candidate['confidence']:.2f})"
        
        # Text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(vis_img, (x, y - text_h - baseline - 3), 
                     (x + text_w, y), color, -1)
        cv2.putText(vis_img, label, (x, y - baseline - 1), 
                   font, font_scale, (255, 255, 255), thickness)
        
        detected_objects.append(candidate)
    
    print(f"Identified {len(detected_objects)} potential objects")
    
    return vis_img, detected_objects


def measure_detected_objects(objects, scale_px_per_mm=4.15):
    """Convert pixel measurements to real-world dimensions."""
    measurements = []
    
    for obj in objects:
        x, y, w_c, h_c = obj['bbox']
        area_px = obj['area']
        
        # Calculate dimensions
        length_px = max(w_c, h_c)
        width_px = min(w_c, h_c)
        
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = area_px / (scale_px_per_mm ** 2)
        
        measurement = {
            'object_id': obj['id'],
            'detected_as': obj['type'],
            'confidence': obj['confidence'],
            'dimensions_mm': {
                'length': round(length_mm, 1),
                'width': round(width_mm, 1),
                'area': round(area_mm2, 1)
            },
            'dimensions_px': {
                'length': length_px,
                'width': width_px,
                'area': area_px
            },
            'aspect_ratio': obj['aspect_ratio'],
            'center_position': obj['center'],
            'bbox': obj['bbox']
        }
        
        measurements.append(measurement)
    
    return measurements


def create_detection_report(img_path, objects, measurements, output_dir):
    """Create comprehensive detection report."""
    
    # What YOLO would theoretically look for
    coco_classes = load_coco_classes()
    tool_related_classes = [
        'scissors', 'knife', 'spoon', 'fork', 'bottle', 'wine glass', 'cup',
        'remote', 'cell phone', 'mouse', 'keyboard', 'toothbrush', 'tennis racket',
        'baseball bat', 'baseball glove'
    ]
    
    report = {
        'analysis_method': 'opencv_feature_detection',
        'image_analyzed': img_path,
        'scale_px_per_mm': 4.15,
        'total_objects_found': len(objects),
        'yolo_classes_total': len(coco_classes),
        'yolo_tool_related_classes': tool_related_classes,
        'detected_objects': measurements,
        'analysis_notes': [
            "This shows what computer vision can detect in your image",
            "YOLO would look for 80 different object classes from COCO dataset",
            "Tool-related classes YOLO can detect: scissors, knife, spoon, fork, etc.",
            "Your pliers may not be detected by YOLO if they don't match training data",
            "Custom detection algorithms can find objects YOLO cannot"
        ]
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'detection_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Run object detection analysis."""
    if not HAS_DEPS:
        return 1
    
    print("🔍 Object Detection Analysis (YOLO Alternative)")
    print("=" * 55)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    output_dir = "/app/output"
    
    # Analyze image features
    vis_img, objects = analyze_image_features(img_path)
    if vis_img is None:
        print("❌ Failed to analyze image")
        return 1
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'feature_detection_visualization.jpg')
    cv2.imwrite(vis_path, vis_img)
    print(f"Visualization saved: {vis_path}")
    
    if objects:
        # Measure objects
        measurements = measure_detected_objects(objects)
        
        # Create report
        report = create_detection_report(img_path, objects, measurements, output_dir)
        
        print(f"\n🎯 DETECTED OBJECTS:")
        print("=" * 40)
        
        for measurement in measurements:
            print(f"\n**OBJECT #{measurement['object_id']}:**")
            print(f"  Detected as: {measurement['detected_as']}")
            print(f"  Confidence: {measurement['confidence']:.2f}")
            print(f"  Dimensions: {measurement['dimensions_mm']['length']} × {measurement['dimensions_mm']['width']} mm")
            print(f"  Area: {measurement['dimensions_mm']['area']} mm²")
            print(f"  Aspect ratio: {measurement['aspect_ratio']:.2f}")
        
        print(f"\n📊 YOLO DETECTION CONTEXT:")
        print("=" * 40)
        print(f"YOLO can detect {len(report['yolo_classes_total'])} object classes")
        print("Tool-related classes YOLO recognizes:")
        for tool_class in report['yolo_tool_related_classes']:
            print(f"  • {tool_class}")
        
        print(f"\n💡 ANALYSIS:")
        print("Your flush cutting pliers might not be detected by YOLO because:")
        print("  • Pliers aren't in YOLO's standard 80 COCO classes")
        print("  • Small tools can be missed by general object detection")
        print("  • YOLO is trained on common household/outdoor objects")
        print("  • Specialized tools require custom detection algorithms")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"  📷 feature_detection_visualization.jpg")
        print(f"  📄 detection_analysis_report.json")
        
        return 0
    else:
        print("❌ No objects detected")
        return 1


if __name__ == "__main__":
    exit(main())