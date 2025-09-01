#!/usr/bin/env python3
"""Aggressive tool detection to find any objects in the calibration mat area."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def analyze_full_image_for_objects(img_path):
    """Analyze the entire image for any objects between AprilTags."""
    print(f"Analyzing: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Full image: {w}×{h} pixels")
    
    # Try multiple detection methods
    
    # Method 1: Adaptive threshold to find any dark objects
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # Method 2: Simple threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Edge detection
    edges = cv2.Canny(gray, 20, 60)
    
    all_candidates = []
    
    # Process each method
    for method_name, binary_img in [("Adaptive", binary), ("Otsu", thresh), ("Canny", edges)]:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n{method_name} method found {len(contours)} contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Very low threshold - detect almost anything
            if area > 100:  # Very small minimum
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Check if it's not just noise at edges
                center_x, center_y = x + w_c//2, y + h_c//2
                edge_margin = min(w, h) * 0.1  # 10% edge margin
                
                if (edge_margin < center_x < w - edge_margin and 
                    edge_margin < center_y < h - edge_margin):
                    
                    aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                    
                    candidate = {
                        'method': method_name,
                        'contour_id': i,
                        'bbox': (x, y, w_c, h_c),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (center_x, center_y),
                        'contour': contour
                    }
                    
                    all_candidates.append(candidate)
    
    # Remove duplicates (similar positions) and sort by area
    unique_candidates = []
    for candidate in sorted(all_candidates, key=lambda x: x['area'], reverse=True):
        is_duplicate = False
        for existing in unique_candidates:
            # Check if centers are close
            dx = abs(candidate['center'][0] - existing['center'][0])
            dy = abs(candidate['center'][1] - existing['center'][1])
            if dx < 50 and dy < 50:  # Within 50 pixels
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    print(f"\nFound {len(unique_candidates)} unique object candidates:")
    for i, candidate in enumerate(unique_candidates[:10]):  # Show top 10
        print(f"  {i+1}. {candidate['method']} - Area: {candidate['area']:.0f}px², "
              f"Aspect: {candidate['aspect_ratio']:.2f}, Center: {candidate['center']}")
    
    return img, unique_candidates


def measure_largest_objects(candidates, scale_px_per_mm=4.15):
    """Measure the largest reasonable objects."""
    if not candidates:
        return []
    
    measurements = []
    
    for i, candidate in enumerate(candidates[:5]):  # Top 5 candidates
        x, y, w_c, h_c = candidate['bbox']
        area_px = candidate['area']
        
        # Calculate dimensions
        length_px = max(w_c, h_c)
        width_px = min(w_c, h_c)
        
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = area_px / (scale_px_per_mm ** 2)
        
        measurement = {
            'candidate_rank': i + 1,
            'detection_method': candidate['method'],
            'length_mm': round(length_mm, 1),
            'width_mm': round(width_mm, 1),
            'area_mm2': round(area_mm2, 1),
            'length_px': length_px,
            'width_px': width_px,
            'area_px': int(area_px),
            'bbox': candidate['bbox'],
            'center': candidate['center'],
            'aspect_ratio': candidate['aspect_ratio']
        }
        
        measurements.append(measurement)
    
    return measurements


def create_detection_visualization(img, candidates, measurements, output_path):
    """Create visualization showing all detected candidates."""
    vis_img = img.copy()
    
    # Draw all candidates with different colors
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (candidate, measurement) in enumerate(zip(candidates[:5], measurements)):
        x, y, w_c, h_c = candidate['bbox']
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w_c, y + h_c), color, 2)
        
        # Add label
        label = f"#{i+1}: {measurement['length_mm']}×{measurement['width_mm']}mm"
        cv2.putText(vis_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Detection visualization saved: {output_path}")
    return True


def main():
    """Run aggressive tool detection."""
    if not HAS_DEPS:
        return 1
    
    print("🔍 BatchGrids Aggressive Object Detection")
    print("=" * 50)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Analyze image for any objects
    img, candidates = analyze_full_image_for_objects(img_path)
    if img is None or not candidates:
        print("❌ No image or no objects found")
        return 1
    
    # Measure all reasonable candidates
    measurements = measure_largest_objects(candidates, scale_px_per_mm=4.15)
    
    if measurements:
        # Create visualization
        create_detection_visualization(img, candidates, measurements, 
                                     "/app/output/aggressive_detection_visualization.jpg")
        
        # Save all results
        results = {
            'detection_method': 'aggressive_multi_method',
            'total_candidates_found': len(candidates),
            'scale_px_per_mm': 4.15,
            'scale_source': 'apriltag_calibration',
            'measurements': measurements
        }
        
        with open('/app/output/aggressive_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        print("🎯 OBJECT DETECTION RESULTS:")
        print("=" * 50)
        
        for i, m in enumerate(measurements):
            print(f"\n**OBJECT #{i+1} ({m['detection_method']} method):**")
            print(f"  Height (Length): {m['length_mm']} mm")
            print(f"  Width: {m['width_mm']} mm")
            print(f"  Area: {m['area_mm2']} mm²")
            print(f"  Aspect ratio: {m['aspect_ratio']:.2f}")
            print(f"  Center position: {m['center']}")
        
        print("\n" + "=" * 50)
        print("💡 Note: These are ALL detected objects in the image.")
        print("    The actual tool is likely one of the larger, more elongated objects.")
        print("    Review the visualization image to identify which is your tool.")
        
        return 0
    else:
        print("❌ No measurable objects found")
        return 1


if __name__ == "__main__":
    exit(main())