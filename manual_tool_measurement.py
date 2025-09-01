#!/usr/bin/env python3
"""Manual tool measurement using computer vision techniques and AprilTag calibration."""

import os
import sys
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def load_image_and_detect_tool_region(img_path):
    """Load image and identify the tool region manually."""
    print(f"Loading: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"Image: {w}×{h} pixels")
    
    # Focus on center region where tool should be
    center_crop_margin = 0.25  # 25% margin from edges
    x1 = int(w * center_crop_margin)
    y1 = int(h * center_crop_margin)
    x2 = int(w * (1 - center_crop_margin))
    y2 = int(h * (1 - center_crop_margin))
    
    # Extract center region
    center_region = gray[y1:y2, x1:x2]
    
    print(f"Analyzing center region: {x2-x1}×{y2-y1} pixels")
    
    # Apply edge detection to find tool
    edges = cv2.Canny(center_region, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found")
        return img, None
    
    # Filter contours by size and aspect ratio
    min_area = (center_region.shape[0] * center_region.shape[1]) * 0.01  # 1% of center region
    
    tool_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
            
            # Tools are typically elongated
            if aspect_ratio > 1.5:
                # Adjust coordinates back to full image
                full_x = x1 + x_c
                full_y = y1 + y_c
                
                tool_contours.append({
                    'contour': contour,
                    'bbox': (full_x, full_y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
    
    if tool_contours:
        # Sort by area and take largest
        tool_contours.sort(key=lambda x: x['area'], reverse=True)
        best_tool = tool_contours[0]
        
        print(f"Tool found:")
        print(f"  Bbox: {best_tool['bbox']}")
        print(f"  Area: {best_tool['area']:.0f} px²")
        print(f"  Aspect ratio: {best_tool['aspect_ratio']:.2f}")
        
        return img, best_tool
    else:
        print("No tool-like objects found")
        return img, None


def measure_tool_with_calibration(tool_detection, scale_px_per_mm=4.15):
    """Measure tool using AprilTag calibration."""
    print("\n=== Tool Measurement ===")
    
    if not tool_detection:
        return None
    
    x, y, w, h = tool_detection['bbox']
    area_px = tool_detection['area']
    
    # Length is the longer dimension
    length_px = max(w, h)
    width_px = min(w, h)
    
    # Convert to millimeters using AprilTag scale
    length_mm = length_px / scale_px_per_mm
    width_mm = width_px / scale_px_per_mm
    area_mm2 = area_px / (scale_px_per_mm ** 2)
    
    measurements = {
        'length_mm': round(length_mm, 1),
        'width_mm': round(width_mm, 1),
        'area_mm2': round(area_mm2, 1),
        'length_px': length_px,
        'width_px': width_px,
        'area_px': int(area_px),
        'scale_px_per_mm': scale_px_per_mm,
        'bbox': tool_detection['bbox'],
        'aspect_ratio': tool_detection['aspect_ratio']
    }
    
    return measurements


def create_measurement_visualization(img, tool_detection, measurements, output_path):
    """Create visualization with measurements."""
    if not tool_detection or not measurements:
        return False
    
    vis_img = img.copy()
    x, y, w, h = tool_detection['bbox']
    
    # Draw bounding box
    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Add measurement text
    length_mm = measurements['length_mm']
    width_mm = measurements['width_mm']
    
    # Text background for readability
    text = f"Tool: {length_mm}mm x {width_mm}mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x
    text_y = y - 10 if y > 30 else y + h + 30
    
    # White background rectangle
    cv2.rectangle(vis_img, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
    
    # Text
    cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    
    # Save visualization
    cv2.imwrite(output_path, vis_img)
    print(f"Visualization saved: {output_path}")
    return True


def main():
    """Run manual tool measurement pipeline."""
    if not HAS_DEPS:
        return 1
    
    print("📐 BatchGrids Manual Tool Measurement")
    print("=" * 45)
    
    # Use existing processed image
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Detect tool region
    img, tool_detection = load_image_and_detect_tool_region(img_path)
    if img is None:
        print("Failed to load image")
        return 1
    
    if tool_detection:
        # Measure using AprilTag calibration scale
        measurements = measure_tool_with_calibration(tool_detection, scale_px_per_mm=4.15)
        
        # Create visualization
        vis_success = create_measurement_visualization(
            img, tool_detection, measurements, 
            "/app/output/tool_measurement_visualization.jpg"
        )
        
        # Save results
        results = {
            'detection_method': 'manual_computer_vision',
            'tool_detected': True,
            'measurements': measurements,
            'scale_source': 'apriltag_calibration_4.15_px_per_mm',
            'confidence': 'manual_detection'
        }
        
        with open('/app/output/manual_measurement_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 45)
        print("✅ TOOL MEASUREMENTS FOUND!")
        print("=" * 45)
        print(f"**HEIGHT (Length): {measurements['length_mm']} mm**")
        print(f"**WIDTH: {measurements['width_mm']} mm**")
        print(f"**AREA: {measurements['area_mm2']} mm²**")
        print("=" * 45)
        print(f"Scale: {measurements['scale_px_per_mm']:.2f} px/mm (AprilTag calibrated)")
        print(f"Aspect ratio: {measurements['aspect_ratio']:.2f}")
        print(f"Pixel dimensions: {measurements['length_px']}×{measurements['width_px']} px")
        
        return 0
    else:
        print("❌ No tool detected in image")
        return 1


if __name__ == "__main__":
    exit(main())