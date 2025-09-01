#!/usr/bin/env python3
"""Enhanced BatchGrids computer vision pipeline test with better detection algorithms."""

import os
import sys
import json
import math
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPS = False


def load_and_resize_image(input_path, max_size=2000):
    """Load image and resize for processing."""
    print(f"Loading image: {input_path}")
    
    img = cv2.imread(input_path)
    if img is None:
        print("Failed to load image")
        return None
    
    h, w = img.shape[:2]
    print(f"Original image: {w}x{h} pixels")
    
    # Resize if too large for processing
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        print(f"Resized to: {new_w}x{new_h} pixels (scale: {scale:.3f})")
    
    return img


def detect_apriltags_enhanced(img):
    """Enhanced AprilTag detection using multiple methods."""
    print("\n=== Enhanced AprilTag Detection ===")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Look for dark squares with white borders (AprilTag structure)
    # Apply adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tag_candidates = []
    min_area = (w * h) * 0.001  # At least 0.1% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for rectangular shapes (AprilTags are square)
        if len(approx) >= 4:
            x, y, w_c, h_c = cv2.boundingRect(approx)
            aspect_ratio = w_c / h_c
            
            # Check if roughly square
            if 0.7 < aspect_ratio < 1.4:
                # Check if it's in the corners or edges (typical AprilTag placement)
                center_x, center_y = x + w_c//2, y + h_c//2
                edge_threshold = min(w, h) * 0.3
                
                is_corner = (center_x < edge_threshold or center_x > w - edge_threshold or
                           center_y < edge_threshold or center_y > h - edge_threshold)
                
                if is_corner:
                    tag_candidates.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w_c, h_c),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
    
    # Sort by area (larger first) and keep top candidates
    tag_candidates.sort(key=lambda x: x['area'], reverse=True)
    tag_candidates = tag_candidates[:4]  # Max 4 AprilTags expected
    
    print(f"Found {len(tag_candidates)} potential AprilTag candidates")
    for i, tag in enumerate(tag_candidates):
        print(f"  Tag {i}: center {tag['center']}, bbox {tag['bbox']}, area {tag['area']:.0f}")
    
    return tag_candidates


def enhanced_tool_detection(img):
    """Enhanced tool detection using multiple computer vision techniques."""
    print("\n=== Enhanced Tool Detection ===")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Edge-based detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Method 2: Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define center region (where tools are typically placed)
    center_margin = 0.2  # 20% margin from edges
    roi_x1 = int(w * center_margin)
    roi_y1 = int(h * center_margin)
    roi_x2 = int(w * (1 - center_margin))
    roi_y2 = int(h * (1 - center_margin))
    
    tool_candidates = []
    min_area = (w * h) * 0.005  # At least 0.5% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w_c, h_c = cv2.boundingRect(contour)
        center_x, center_y = x + w_c//2, y + h_c//2
        
        # Check if center is in ROI
        if roi_x1 < center_x < roi_x2 and roi_y1 < center_y < roi_y2:
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
            
            # Tools are typically elongated
            if aspect_ratio > 2.0:
                # Calculate a confidence score
                confidence = min(0.95, 0.5 + (aspect_ratio - 2.0) * 0.1)
                confidence = max(confidence, area / (w * h * 0.5))  # Larger objects get higher confidence
                
                tool_candidates.append({
                    'class': 'tool',
                    'confidence': confidence,
                    'bbox': (x, y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour
                })
    
    if tool_candidates:
        # Sort by confidence and area
        tool_candidates.sort(key=lambda x: x['confidence'] * math.log(x['area']), reverse=True)
        best_detection = tool_candidates[0]
        
        print(f"Tool detected: {best_detection['class']}")
        print(f"  Confidence: {best_detection['confidence']:.3f}")
        print(f"  Bounding box: {best_detection['bbox']}")
        print(f"  Area: {best_detection['area']:.0f} pixels")
        print(f"  Aspect ratio: {best_detection['aspect_ratio']:.2f}")
        
        return best_detection
    else:
        print("No tool detected in center region")
        return None


def estimate_scale_from_tags(tags, reference_distance_mm=200):
    """Estimate pixel-to-mm scale from AprilTag positions."""
    if len(tags) < 2:
        print("Need at least 2 AprilTags for scale estimation")
        return 4.0  # Default fallback
    
    # Find the two tags that are farthest apart
    max_distance = 0
    best_pair = None
    
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            tag1, tag2 = tags[i], tags[j]
            dx = tag1['center'][0] - tag2['center'][0]
            dy = tag1['center'][1] - tag2['center'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > max_distance:
                max_distance = distance
                best_pair = (tag1, tag2)
    
    if best_pair:
        px_per_mm = max_distance / reference_distance_mm
        print(f"Scale estimation: {px_per_mm:.2f} px/mm (distance: {max_distance:.0f}px)")
        return px_per_mm
    
    return 4.0  # Default fallback


def calculate_precise_dimensions(detection, scale_px_per_mm):
    """Calculate precise dimensions from detection."""
    print("\n=== Precise Dimension Calculation ===")
    
    if not detection:
        return None
    
    x, y, w, h = detection['bbox']
    
    # Use contour for more accurate measurements if available
    if 'contour' in detection:
        contour = detection['contour']
        
        # Calculate oriented bounding box for better length/width measurement
        rect = cv2.minAreaRect(contour)
        box_w, box_h = rect[1]
        
        # Length is the longer dimension, width is shorter
        length_px = max(box_w, box_h)
        width_px = min(box_w, box_h)
        
        # Calculate area from contour
        area_px = cv2.contourArea(contour)
    else:
        # Fallback to bounding box
        length_px = max(w, h)
        width_px = min(w, h)
        area_px = w * h
    
    # Convert to millimeters
    length_mm = length_px / scale_px_per_mm
    width_mm = width_px / scale_px_per_mm
    area_mm2 = area_px / (scale_px_per_mm ** 2)
    
    dimensions = {
        'length_mm': round(length_mm, 1),
        'width_mm': round(width_mm, 1),
        'area_mm2': round(area_mm2, 1),
        'px_per_mm': scale_px_per_mm,
        'length_px': round(length_px, 1),
        'width_px': round(width_px, 1)
    }
    
    print(f"Calculated dimensions:")
    print(f"  Length: {dimensions['length_mm']} mm ({dimensions['length_px']} px)")
    print(f"  Width: {dimensions['width_mm']} mm ({dimensions['width_px']} px)")
    print(f"  Area: {dimensions['area_mm2']} mm²")
    print(f"  Scale: {dimensions['px_per_mm']:.2f} px/mm")
    
    return dimensions


def generate_detailed_svg(detection, dimensions, output_path):
    """Generate detailed SVG with accurate tool outline."""
    print("\n=== Detailed SVG Generation ===")
    
    if not detection or not dimensions:
        return False
    
    length_mm = dimensions['length_mm']
    width_mm = dimensions['width_mm']
    scale = dimensions['px_per_mm']
    
    # Create more detailed SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 {length_mm + 20} {width_mm + 20}" 
     width="{length_mm + 20}mm" height="{width_mm + 20}mm">
  
  <!-- Tool outline -->
  <rect x="10" y="10" width="{length_mm}" height="{width_mm}" 
        fill="none" stroke="black" stroke-width="0.1mm"/>
  
  <!-- Dimension lines -->
  <line x1="10" y1="5" x2="{10 + length_mm}" y2="5" stroke="red" stroke-width="0.05mm"/>
  <text x="{10 + length_mm/2}" y="3" text-anchor="middle" font-size="1.5mm" fill="red">
    {length_mm} mm
  </text>
  
  <line x1="5" y1="10" x2="5" y2="{10 + width_mm}" stroke="red" stroke-width="0.05mm"/>
  <text x="2" y="{10 + width_mm/2}" text-anchor="middle" font-size="1.5mm" fill="red" 
        transform="rotate(-90, 2, {10 + width_mm/2})">
    {width_mm} mm
  </text>
  
  <!-- Metadata -->
  <text x="10" y="{width_mm + 17}" font-size="1.2mm" fill="gray">
    BatchGrids Tool Outline - Area: {dimensions['area_mm2']} mm² - Scale: {scale:.2f} px/mm
  </text>
</svg>'''
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Detailed SVG saved to: {output_path}")
    return True


def save_results_json(tags, detection, dimensions, output_path):
    """Save processing results as JSON."""
    results = {
        'apriltags': {
            'detected': len(tags),
            'positions': [{'center': tag['center'], 'bbox': tag['bbox']} for tag in tags]
        },
        'tool_detection': {
            'detected': detection is not None,
            'confidence': detection['confidence'] if detection else 0,
            'class': detection['class'] if detection else None,
            'bbox': detection['bbox'] if detection else None
        },
        'dimensions': dimensions if dimensions else {},
        'processing_status': 'success' if detection and dimensions else 'partial',
        'needs_review': not detection or (detection and detection['confidence'] < 0.8)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Run the enhanced pipeline test."""
    if not HAS_DEPS:
        print("Error: Missing OpenCV dependencies")
        return 1
    
    input_image = "/app/input/IMG_0756.jpg"
    converted_image = "/app/output/enhanced_converted_image.jpg"
    svg_output = "/app/output/enhanced_tool_outline.svg"
    json_output = "/app/output/processing_results.json"
    
    print("🔍 BatchGrids Enhanced Computer Vision Pipeline")
    print("=" * 55)
    
    # Step 1: Load and preprocess image
    img = load_and_resize_image(input_image)
    if img is None:
        return 1
    
    # Save processed image
    cv2.imwrite(converted_image, img)
    print(f"Processed image saved to: {converted_image}")
    
    # Step 2: Enhanced AprilTag detection
    tags = detect_apriltags_enhanced(img)
    
    # Step 3: Estimate scale from AprilTags
    scale = estimate_scale_from_tags(tags)
    
    # Step 4: Enhanced tool detection
    detection = enhanced_tool_detection(img)
    
    # Step 5: Precise dimension calculation
    dimensions = calculate_precise_dimensions(detection, scale)
    
    # Step 6: Generate detailed SVG
    svg_success = generate_detailed_svg(detection, dimensions, svg_output)
    
    # Step 7: Save results as JSON
    results = save_results_json(tags, detection, dimensions, json_output)
    
    # Step 8: Final report
    print("\n" + "=" * 55)
    print("🎯 Enhanced Pipeline Results:")
    print(f"  AprilTags detected: {len(tags)}")
    if len(tags) >= 2:
        print(f"  ✓ Scale calibrated: {scale:.2f} px/mm")
    else:
        print(f"  ⚠ Scale estimation limited (using default: {scale:.2f} px/mm)")
    
    if detection:
        print(f"  ✓ Tool detected: {detection['class']} (confidence: {detection['confidence']:.3f})")
        if dimensions:
            print(f"  ✓ Dimensions: {dimensions['length_mm']}mm × {dimensions['width_mm']}mm")
            print(f"  ✓ Area: {dimensions['area_mm2']} mm²")
    else:
        print("  ✗ No tool detected")
    
    if svg_success:
        print(f"  ✓ SVG generated: enhanced_tool_outline.svg")
    
    print(f"  ✓ Results saved: processing_results.json")
    
    if results['needs_review']:
        print("\n⚠️  QUALITY CONTROL: Image may need human review")
        if not detection:
            print("   - No tool detected in center region")
        elif detection['confidence'] < 0.8:
            print(f"   - Low confidence detection ({detection['confidence']:.3f})")
    else:
        print("\n✅ Quality check passed - ready for production")
    
    print("\n📋 Files generated:")
    print("  1. Enhanced processed image")
    print("  2. SVG outline for fabrication")
    print("  3. JSON results with metadata")
    
    return 0


if __name__ == "__main__":
    exit(main())