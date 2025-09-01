#!/usr/bin/env python3
"""Test the BatchGrids computer vision pipeline with a real tool image."""

import os
import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def load_and_convert_image(input_path, output_path=None):
    """Load image and convert HEIC to standard format if needed."""
    print(f"Loading image: {input_path}")
    
    # Load image with OpenCV (supports many formats)
    img = cv2.imread(input_path)
    
    if img is None:
        print("Failed to load image - trying alternative method")
        return None
    
    print(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Converted image saved to: {output_path}")
    
    return img


def detect_apriltags_simple(img):
    """Simple AprilTag detection simulation (since we don't have the full library)."""
    print("\n=== AprilTag Detection ===")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Look for black squares (AprilTag patterns)
    # This is a simplified detection - in production we'd use the apriltag library
    binary = (gray < 100).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for square-ish contours
    tag_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum size
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # Quadrilateral
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:  # Roughly square
                    tag_candidates.append({
                        'center': (x + w//2, y + h//2),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
    
    print(f"Found {len(tag_candidates)} potential AprilTag candidates")
    for i, tag in enumerate(tag_candidates):
        print(f"  Tag {i}: center {tag['center']}, bbox {tag['bbox']}")
    
    return tag_candidates


def simulate_yolo_detection(img):
    """Simulate YOLO detection (simplified version)."""
    print("\n=== YOLO Tool Detection ===")
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find tool-like objects (long, thin shapes in the center)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour in the center region
    h, w = img.shape[:2]
    center_region = (w//4, h//4, w//2, h//2)  # Center quarter
    
    tool_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Significant size
            x, y, w_c, h_c = cv2.boundingRect(contour)
            center_x, center_y = x + w_c//2, y + h_c//2
            
            # Check if center is in the middle region
            if (center_region[0] < center_x < center_region[0] + center_region[2] and
                center_region[1] < center_y < center_region[1] + center_region[3]):
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
                if aspect_ratio > 3:  # Tool-like elongated shape
                    tool_candidates.append({
                        'class': 'tool',
                        'confidence': 0.85,  # Simulated confidence
                        'bbox': (x, y, w_c, h_c),
                        'area': area,
                        'mask': contour
                    })
    
    if tool_candidates:
        # Sort by area and take the largest
        best_detection = max(tool_candidates, key=lambda x: x['area'])
        print(f"Tool detected: {best_detection['class']} (confidence: {best_detection['confidence']:.2f})")
        print(f"Bounding box: {best_detection['bbox']}")
        print(f"Area: {best_detection['area']} pixels")
        return best_detection
    else:
        print("No tool detected in center region")
        return None


def calculate_dimensions(detection, scale_px_per_mm=4.0):
    """Calculate real-world dimensions from pixel measurements."""
    print("\n=== Dimension Calculation ===")
    
    if not detection:
        print("No detection to measure")
        return None
    
    x, y, w, h = detection['bbox']
    
    # Convert pixels to millimeters
    length_mm = max(w, h) / scale_px_per_mm
    width_mm = min(w, h) / scale_px_per_mm  
    area_mm2 = detection['area'] / (scale_px_per_mm ** 2)
    
    dimensions = {
        'length_mm': round(length_mm, 1),
        'width_mm': round(width_mm, 1),
        'area_mm2': round(area_mm2, 1),
        'px_per_mm': scale_px_per_mm
    }
    
    print(f"Calculated dimensions:")
    print(f"  Length: {dimensions['length_mm']} mm")
    print(f"  Width: {dimensions['width_mm']} mm") 
    print(f"  Area: {dimensions['area_mm2']} mm²")
    print(f"  Scale: {dimensions['px_per_mm']} px/mm")
    
    return dimensions


def generate_svg_outline(detection, dimensions, output_path):
    """Generate SVG outline from detection."""
    print("\n=== SVG Generation ===")
    
    if not detection or not dimensions:
        print("No data to generate SVG")
        return False
    
    # Simplified SVG generation
    x, y, w, h = detection['bbox']
    length_mm = dimensions['length_mm']
    width_mm = dimensions['width_mm']
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 {length_mm} {width_mm}" 
     width="{length_mm}mm" height="{width_mm}mm">
  <rect x="0" y="0" width="{length_mm}" height="{width_mm}" 
        fill="none" stroke="black" stroke-width="0.1mm"/>
  <text x="{length_mm/2}" y="-2" text-anchor="middle" font-size="2mm" fill="red">
    BatchGrids Tool Outline - {length_mm}mm x {width_mm}mm
  </text>
</svg>'''
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"SVG outline saved to: {output_path}")
    return True


def main():
    """Run the complete pipeline test."""
    if not HAS_DEPS:
        print("Error: Missing OpenCV dependencies")
        return 1
    
    input_image = "/app/input/IMG_0756.jpg"
    converted_image = "/app/output/converted_image.jpg"
    svg_output = "/app/output/tool_outline.svg"
    
    print("🔍 BatchGrids Computer Vision Pipeline Test")
    print("=" * 50)
    
    # Step 1: Load and convert image
    img = load_and_convert_image(input_image, converted_image)
    if img is None:
        print("Failed to load input image")
        return 1
    
    # Step 2: AprilTag detection
    tags = detect_apriltags_simple(img)
    
    # Step 3: YOLO tool detection
    detection = simulate_yolo_detection(img)
    
    # Step 4: Dimension calculation
    dimensions = calculate_dimensions(detection, scale_px_per_mm=4.0)
    
    # Step 5: SVG generation
    svg_success = generate_svg_outline(detection, dimensions, svg_output)
    
    # Step 6: Results summary
    print("\n" + "=" * 50)
    print("🎯 Pipeline Results:")
    if tags:
        print(f"  ✓ AprilTags detected: {len(tags)}")
    else:
        print("  ⚠ No AprilTags detected")
        
    if detection:
        print(f"  ✓ Tool detected: {detection['class']} ({detection['confidence']:.2f})")
        print(f"  ✓ Dimensions: {dimensions['length_mm']}mm x {dimensions['width_mm']}mm")
    else:
        print("  ✗ No tool detected")
        
    if svg_success:
        print(f"  ✓ SVG generated: tool_outline.svg")
    else:
        print("  ✗ SVG generation failed")
    
    print("\n📋 Next Steps:")
    print("  1. Review generated files in /app/output/")
    print("  2. Check AprilTag detection accuracy")
    print("  3. Verify tool segmentation quality")
    print("  4. Validate dimensional measurements")
    
    return 0


if __name__ == "__main__":
    exit(main())