#!/usr/bin/env python3
"""Specialized detection for small flush cutting pliers."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def detect_pliers_specifically(img_path):
    """Detect small elongated objects that match pliers characteristics."""
    print(f"Specialized pliers detection: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h} pixels")
    
    # Focus on center region where tool should be placed
    margin = 0.2  # 20% margin from edges
    roi_x1 = int(w * margin)
    roi_y1 = int(h * margin)
    roi_x2 = int(w * (1 - margin))
    roi_y2 = int(h * (1 - margin))
    
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    print(f"ROI: {roi_x2-roi_x1}×{roi_y2-roi_y1} pixels")
    
    pliers_candidates = []
    
    # Method 1: Edge detection with morphological operations
    edges = cv2.Canny(roi, 40, 120)
    
    # Clean up edges - close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} edge contours in ROI")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter for pliers-sized objects
        # Expected size: 100-150mm long, 15-25mm wide
        # At 4.15 px/mm: 415-622px long, 62-104px wide
        # Area: ~25,000-65,000 px²
        if 5000 < area < 100000:  # Reasonable area range
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Adjust coordinates back to full image
            full_x = roi_x1 + x
            full_y = roi_y1 + y
            
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
            
            # Pliers are very elongated - aspect ratio > 3
            if aspect_ratio > 2.5:
                # Check if it's roughly in the center area
                center_x = full_x + w_c//2
                center_y = full_y + h_c//2
                
                # Distance from image center
                img_center_x, img_center_y = w//2, h//2
                dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                pliers_candidates.append({
                    'method': 'edge_detection',
                    'bbox': (full_x, full_y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (center_x, center_y),
                    'dist_from_center': dist_from_center,
                    'contour': contour
                })
    
    # Method 2: Template-based approach for tool-like shapes
    # Look for dark elongated objects against lighter background
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # Adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    # Morphological operations to clean up
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove very small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_open)
    
    contours2, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours2)} binary contours in ROI")
    
    for contour in contours2:
        area = cv2.contourArea(contour)
        
        if 3000 < area < 80000:  # Pliers size range
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            full_x = roi_x1 + x
            full_y = roi_y1 + y
            
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
            
            if aspect_ratio > 2.0:  # Tool-like elongation
                center_x = full_x + w_c//2
                center_y = full_y + h_c//2
                
                img_center_x, img_center_y = w//2, h//2
                dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                pliers_candidates.append({
                    'method': 'binary_threshold',
                    'bbox': (full_x, full_y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (center_x, center_y),
                    'dist_from_center': dist_from_center,
                    'contour': contour
                })
    
    # Remove duplicates and rank by pliers-likelihood
    unique_candidates = []
    for candidate in pliers_candidates:
        is_duplicate = False
        for existing in unique_candidates:
            dx = abs(candidate['center'][0] - existing['center'][0])
            dy = abs(candidate['center'][1] - existing['center'][1])
            if dx < 30 and dy < 30:  # Within 30 pixels
                # Keep the one with better aspect ratio
                if candidate['aspect_ratio'] > existing['aspect_ratio']:
                    unique_candidates.remove(existing)
                else:
                    is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Score candidates based on pliers characteristics
    for candidate in unique_candidates:
        score = 0
        
        # Prefer high aspect ratios (elongated)
        if candidate['aspect_ratio'] > 4:
            score += 30
        elif candidate['aspect_ratio'] > 3:
            score += 20
        elif candidate['aspect_ratio'] > 2:
            score += 10
        
        # Prefer center placement
        max_dist = min(w, h) * 0.3  # 30% of image dimension
        if candidate['dist_from_center'] < max_dist:
            score += 20
        
        # Prefer reasonable size
        area = candidate['area']
        if 10000 < area < 50000:  # Sweet spot for pliers
            score += 25
        elif 5000 < area < 80000:
            score += 15
        
        candidate['pliers_score'] = score
    
    # Sort by score
    unique_candidates.sort(key=lambda x: x['pliers_score'], reverse=True)
    
    print(f"\nPliers candidates found: {len(unique_candidates)}")
    for i, candidate in enumerate(unique_candidates[:5]):
        print(f"  {i+1}. {candidate['method']} - Score: {candidate['pliers_score']}, "
              f"Aspect: {candidate['aspect_ratio']:.2f}, Area: {candidate['area']:.0f}")
    
    return img, unique_candidates


def measure_pliers(candidates, scale_px_per_mm=4.15):
    """Measure pliers candidates."""
    if not candidates:
        return []
    
    measurements = []
    
    for i, candidate in enumerate(candidates[:3]):  # Top 3 candidates
        x, y, w_c, h_c = candidate['bbox']
        area_px = candidate['area']
        
        # For pliers, length is always the longer dimension
        length_px = max(w_c, h_c)
        width_px = min(w_c, h_c)
        
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = area_px / (scale_px_per_mm ** 2)
        
        measurement = {
            'candidate_rank': i + 1,
            'detection_method': candidate['method'],
            'pliers_score': candidate['pliers_score'],
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


def create_pliers_visualization(img, candidates, measurements, output_path):
    """Create visualization highlighting pliers candidates."""
    vis_img = img.copy()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
    
    for i, (candidate, measurement) in enumerate(zip(candidates[:3], measurements)):
        x, y, w_c, h_c = candidate['bbox']
        color = colors[i % len(colors)]
        
        # Draw thick bounding box
        cv2.rectangle(vis_img, (x, y), (x + w_c, y + h_c), color, 4)
        
        # Add detailed label
        label = f"#{i+1}: {measurement['length_mm']}×{measurement['width_mm']}mm (Score:{candidate['pliers_score']})"
        
        # Background rectangle for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        text_x = x
        text_y = y - 10 if y > 40 else y + h_c + 30
        
        cv2.rectangle(vis_img, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
        cv2.putText(vis_img, label, (text_x, text_y), font, font_scale, color, thickness)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Pliers detection visualization saved: {output_path}")
    return True


def main():
    """Run specialized pliers detection."""
    if not HAS_DEPS:
        return 1
    
    print("✂️  BatchGrids Flush Cutting Pliers Detection")
    print("=" * 50)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Detect pliers specifically
    img, candidates = detect_pliers_specifically(img_path)
    if img is None or not candidates:
        print("❌ No pliers detected")
        return 1
    
    # Measure pliers candidates
    measurements = measure_pliers(candidates, scale_px_per_mm=4.15)
    
    if measurements:
        # Create visualization
        create_pliers_visualization(img, candidates, measurements, 
                                  "/app/output/pliers_detection_visualization.jpg")
        
        # Save results
        results = {
            'detection_method': 'specialized_pliers_detection',
            'total_candidates': len(candidates),
            'scale_px_per_mm': 4.15,
            'expected_pliers_size': '100-150mm length, 15-25mm width',
            'measurements': measurements
        }
        
        with open('/app/output/pliers_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        print("✂️  FLUSH CUTTING PLIERS DETECTED!")
        print("=" * 50)
        
        for i, m in enumerate(measurements):
            print(f"\n**PLIERS CANDIDATE #{i+1} (Score: {m['pliers_score']}):**")
            print(f"  Length: {m['length_mm']} mm ({m['length_mm']/25.4:.1f} inches)")
            print(f"  Width: {m['width_mm']} mm ({m['width_mm']/25.4:.1f} inches)")
            print(f"  Area: {m['area_mm2']} mm²")
            print(f"  Aspect ratio: {m['aspect_ratio']:.2f}")
            print(f"  Detection method: {m['detection_method']}")
        
        best = measurements[0]
        print(f"\n🎯 **BEST MATCH:**")
        print(f"   **Length: {best['length_mm']} mm ({best['length_mm']/25.4:.1f} inches)**")
        print(f"   **Width: {best['width_mm']} mm ({best['width_mm']/25.4:.1f} inches)**")
        
        return 0
    else:
        print("❌ No measurable pliers found")
        return 1


if __name__ == "__main__":
    exit(main())