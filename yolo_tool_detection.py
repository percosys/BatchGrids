#!/usr/bin/env python3
"""Test YOLO segmentation for precise tool detection and measurement."""

import os
import sys
import json
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    from ultralytics import YOLO
    HAS_DEPS = True
    print("✓ OpenCV and Ultralytics available")
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def load_yolo_model():
    """Load YOLOv8 segmentation model."""
    print("\n=== Loading YOLO Model ===")
    try:
        # YOLOv8 nano segmentation model (pre-trained on COCO)
        model = YOLO('yolov8n-seg.pt')
        print("✓ YOLOv8n-seg model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return None


def yolo_detect_tools(model, img):
    """Use YOLO to detect and segment tools in the image."""
    print("\n=== YOLO Tool Detection ===")
    
    if model is None:
        print("No model available")
        return []
    
    # Run YOLO inference
    results = model(img, conf=0.25, iou=0.45, verbose=False)
    
    detections = []
    
    for result in results:
        if result.masks is not None:
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]
                
                # Get class information
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                
                # Get segmentation mask
                mask_array = mask.data.cpu().numpy()[0]
                
                # Check if this could be a tool (various object classes)
                tool_classes = [
                    'scissors', 'knife', 'spoon', 'fork', 'baseball bat',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'bowl',
                    'banana', 'apple', 'orange', 'carrot', 'hot dog',
                    'toothbrush', 'hair drier', 'cell phone', 'remote',
                    'mouse', 'keyboard', 'book', 'clock', 'vase'
                ]
                
                print(f"Detected: {class_name} (confidence: {confidence:.3f}, bbox: {bbox})")
                
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'mask': mask_array,
                    'is_tool_candidate': class_name.lower() in [c.lower() for c in tool_classes] or confidence > 0.7
                }
                
                detections.append(detection)
    
    # Filter and sort detections
    tool_detections = [d for d in detections if d['is_tool_candidate']]
    tool_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    if tool_detections:
        print(f"Found {len(tool_detections)} potential tool(s)")
        best_detection = tool_detections[0]
        print(f"Best detection: {best_detection['class']} (confidence: {best_detection['confidence']:.3f})")
        return best_detection
    else:
        print("No tools detected")
        # If no specific tools found, try the highest confidence detection of any object
        if detections:
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            fallback = detections[0]
            print(f"Using fallback detection: {fallback['class']} (confidence: {fallback['confidence']:.3f})")
            return fallback
        return None


def extract_precise_dimensions_from_mask(detection, scale_px_per_mm):
    """Extract precise dimensions using YOLO segmentation mask."""
    print("\n=== Precise Dimension Extraction from YOLO Mask ===")
    
    if not detection or 'mask' not in detection:
        print("No mask data available")
        return None
    
    mask = detection['mask']
    
    # Resize mask to match original image if needed
    if mask.shape != (1999, 1500):  # Our resized image dimensions
        mask = cv2.resize(mask.astype(np.uint8), (1500, 1999))
    
    # Convert mask to binary
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Find contours from mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in mask")
        return None
    
    # Use the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area
    area_px = cv2.contourArea(main_contour)
    
    # Get oriented bounding box for precise length/width
    rect = cv2.minAreaRect(main_contour)
    box_w, box_h = rect[1]
    
    # Length is the longer dimension
    length_px = max(box_w, box_h)
    width_px = min(box_w, box_h)
    
    # Convert to millimeters
    length_mm = length_px / scale_px_per_mm
    width_mm = width_px / scale_px_per_mm
    area_mm2 = area_px / (scale_px_per_mm ** 2)
    
    # Get bounding box from contour for verification
    x, y, w, h = cv2.boundingRect(main_contour)
    
    dimensions = {
        'length_mm': round(length_mm, 1),
        'width_mm': round(width_mm, 1),
        'area_mm2': round(area_mm2, 1),
        'length_px': round(length_px, 1),
        'width_px': round(width_px, 1),
        'area_px': int(area_px),
        'px_per_mm': scale_px_per_mm,
        'contour_bbox': (x, y, w, h),
        'mask_shape': mask.shape
    }
    
    print(f"YOLO Mask Dimensions:")
    print(f"  Length: {dimensions['length_mm']} mm ({dimensions['length_px']} px)")
    print(f"  Width: {dimensions['width_mm']} mm ({dimensions['width_px']} px)")
    print(f"  Area: {dimensions['area_mm2']} mm² ({dimensions['area_px']} px)")
    print(f"  Scale: {dimensions['px_per_mm']:.2f} px/mm")
    print(f"  Contour bbox: {dimensions['contour_bbox']}")
    
    return dimensions


def generate_yolo_svg(detection, dimensions, output_path):
    """Generate SVG with YOLO-detected tool outline."""
    print("\n=== YOLO SVG Generation ===")
    
    if not detection or not dimensions:
        return False
    
    length_mm = dimensions['length_mm']
    width_mm = dimensions['width_mm']
    area_mm2 = dimensions['area_mm2']
    class_name = detection['class']
    confidence = detection['confidence']
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 {length_mm + 30} {width_mm + 30}" 
     width="{length_mm + 30}mm" height="{width_mm + 30}mm">
  
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 2mm; fill: #333; }}
      .dimension {{ font-family: Arial, sans-serif; font-size: 1.5mm; fill: red; }}
      .metadata {{ font-family: Arial, sans-serif; font-size: 1mm; fill: gray; }}
    </style>
  </defs>
  
  <!-- Tool outline (simplified rectangle) -->
  <rect x="15" y="15" width="{length_mm}" height="{width_mm}" 
        fill="none" stroke="black" stroke-width="0.2mm"/>
  
  <!-- Title -->
  <text x="{15 + length_mm/2}" y="10" text-anchor="middle" class="title">
    {class_name.title()} - YOLO Detection
  </text>
  
  <!-- Length dimension -->
  <line x1="15" y1="8" x2="{15 + length_mm}" y2="8" stroke="red" stroke-width="0.1mm"/>
  <text x="{15 + length_mm/2}" y="6" text-anchor="middle" class="dimension">
    {length_mm} mm
  </text>
  
  <!-- Width dimension -->
  <line x1="8" y1="15" x2="8" y2="{15 + width_mm}" stroke="red" stroke-width="0.1mm"/>
  <text x="5" y="{15 + width_mm/2}" text-anchor="middle" class="dimension" 
        transform="rotate(-90, 5, {15 + width_mm/2})">
    {width_mm} mm
  </text>
  
  <!-- Metadata -->
  <text x="15" y="{width_mm + 22}" class="metadata">
    Area: {area_mm2} mm² | Confidence: {confidence:.1%} | BatchGrids YOLO Pipeline
  </text>
  
  <!-- Scale reference -->
  <line x1="15" y1="{width_mm + 25}" x2="{15 + min(50, length_mm)}" y2="{width_mm + 25}" 
        stroke="blue" stroke-width="0.1mm"/>
  <text x="15" y="{width_mm + 27}" class="metadata">
    {min(50, int(length_mm))} mm reference
  </text>
</svg>'''
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"YOLO SVG saved to: {output_path}")
    return True


def main():
    """Run YOLO-based tool detection pipeline."""
    if not HAS_DEPS:
        print("Error: Missing dependencies")
        return 1
    
    input_image = "/app/input/IMG_0756.jpg"
    output_image = "/app/output/yolo_processed_image.jpg"
    svg_output = "/app/output/yolo_tool_outline.svg"
    results_output = "/app/output/yolo_results.json"
    
    print("🤖 BatchGrids YOLO Tool Detection Pipeline")
    print("=" * 50)
    
    # Load image (reuse our resized version)
    img_path = "/app/output/enhanced_converted_image.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load processed image")
        return 1
    
    print(f"Using processed image: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Load YOLO model
    model = load_yolo_model()
    if model is None:
        print("Failed to initialize YOLO")
        return 1
    
    # Run YOLO detection
    detection = yolo_detect_tools(model, img)
    
    if detection:
        # Use our established scale from AprilTag calibration
        scale_px_per_mm = 4.15  # From previous AprilTag detection
        
        # Extract precise dimensions
        dimensions = extract_precise_dimensions_from_mask(detection, scale_px_per_mm)
        
        # Generate SVG
        svg_success = generate_yolo_svg(detection, dimensions, svg_output)
        
        # Save results
        results = {
            'yolo_detection': {
                'class': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            },
            'dimensions': dimensions,
            'scale_px_per_mm': scale_px_per_mm,
            'processing_method': 'YOLOv8_segmentation',
            'status': 'success'
        }
        
        with open(results_output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Final report
        print("\n" + "=" * 50)
        print("🎯 YOLO Detection Results:")
        print(f"  ✅ Tool detected: {detection['class']}")
        print(f"  ✅ Confidence: {detection['confidence']:.1%}")
        print(f"  ✅ Dimensions: {dimensions['length_mm']} × {dimensions['width_mm']} mm")
        print(f"  ✅ Area: {dimensions['area_mm2']} mm²")
        print(f"  ✅ SVG generated: yolo_tool_outline.svg")
        
        return 0
    else:
        print("\n❌ No tools detected by YOLO")
        print("This could mean:")
        print("  - Tool is not in YOLO's training classes")
        print("  - Confidence threshold too high") 
        print("  - Tool partially obscured or unclear")
        return 1


if __name__ == "__main__":
    exit(main())