#!/usr/bin/env python3
"""Ultra-sensitive center detection to find even very faint pliers."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def ultra_sensitive_center_detection(img_path):
    """Ultra-sensitive detection for very faint or low-contrast pliers."""
    print(f"Ultra-sensitive center detection: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h} pixels")
    
    # Even smaller center focus - just the very center
    center_margin = 0.3  # 30% margin - focus on center 40% of image
    roi_x1 = int(w * center_margin)
    roi_y1 = int(h * center_margin)
    roi_x2 = int(w * (1 - center_margin))
    roi_y2 = int(h * (1 - center_margin))
    
    print(f"Ultra-focused ROI: {roi_x2-roi_x1}×{roi_y2-roi_y1} pixels (center 40%)")
    
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Show some statistics about the ROI
    roi_mean = np.mean(roi)
    roi_std = np.std(roi)
    roi_min = np.min(roi)
    roi_max = np.max(roi)
    
    print(f"ROI intensity stats: mean={roi_mean:.1f}, std={roi_std:.1f}, range={roi_min}-{roi_max}")
    
    candidates = []
    
    # Method 1: Extremely sensitive adaptive threshold
    print("\n=== Ultra-Sensitive Adaptive Thresholds ===")
    
    # Very fine-grained thresholds
    ultra_sensitive_params = [
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 1),    # Ultra fine
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 9, 2),    # Very fine
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),   # Fine
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 13, 3),   # Medium fine
        (cv2.ADAPTIVE_THRESH_MEAN_C, 7, 1),        # Ultra fine mean
        (cv2.ADAPTIVE_THRESH_MEAN_C, 9, 2),        # Very fine mean
    ]
    
    for i, (method, block_size, C) in enumerate(ultra_sensitive_params):
        thresh = cv2.adaptiveThreshold(roi, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
        
        # Very minimal morphology to preserve fine details
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_tiny)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Ultra-adaptive {i+1}: {len(contours)} contours")
        
        # Accept MUCH smaller objects
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 100:  # Extremely low threshold
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                if min(w_c, h_c) > 2:  # Not just a line
                    aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
                    
                    # Very permissive aspect ratio
                    if aspect_ratio > 1.1:
                        center_x = full_x + w_c // 2
                        center_y = full_y + h_c // 2
                        
                        candidate = {
                            'method': f'ultra_adaptive_{i+1}',
                            'bbox': (full_x, full_y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y),
                            'contour': contour,
                            'roi_offset': (roi_x1, roi_y1)
                        }
                        candidates.append(candidate)
    
    # Method 2: Simple thresholding with multiple values
    print("\n=== Simple Threshold Sweep ===")
    
    # Try many different threshold values
    threshold_values = [50, 75, 100, 125, 150, 175, 200, 225]
    
    for thresh_val in threshold_values:
        _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Minimal cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Threshold {thresh_val}: {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 200:  # Very small objects ok
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                if min(w_c, h_c) > 3:  # At least 3 pixels wide
                    aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
                    
                    if aspect_ratio > 1.2:  # Very slight elongation
                        center_x = full_x + w_c // 2
                        center_y = full_y + h_c // 2
                        
                        candidate = {
                            'method': f'threshold_{thresh_val}',
                            'bbox': (full_x, full_y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y),
                            'threshold_value': thresh_val,
                            'contour': contour,
                            'roi_offset': (roi_x1, roi_y1)
                        }
                        candidates.append(candidate)
    
    # Method 3: Intensity variation detection
    print("\n=== Intensity Variation Detection ===")
    
    # Look for any regions that vary from the local background
    blurred = cv2.GaussianBlur(roi, (15, 15), 0)
    diff = cv2.absdiff(roi, blurred)
    
    # Multiple sensitivity levels
    diff_thresholds = [10, 15, 20, 25, 30]
    
    for thresh_val in diff_thresholds:
        _, diff_thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Diff {thresh_val}: {len(contours)} variation regions")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 300:  # Reasonable variation area
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
                
                if aspect_ratio > 1.3:
                    center_x = full_x + w_c // 2
                    center_y = full_y + h_c // 2
                    
                    candidate = {
                        'method': f'variation_{thresh_val}',
                        'bbox': (full_x, full_y, w_c, h_c),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (center_x, center_y),
                        'contour': contour,
                        'roi_offset': (roi_x1, roi_y1)
                    }
                    candidates.append(candidate)
    
    # Method 4: Laplacian edge detection (different from Canny)
    print("\n=== Laplacian Edge Detection ===")
    
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    
    # Different thresholds for Laplacian
    lap_thresholds = [5, 10, 15, 20]
    
    for thresh_val in lap_thresholds:
        _, lap_thresh = cv2.threshold(laplacian_abs, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lap_thresh = cv2.morphologyEx(lap_thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(lap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Laplacian {thresh_val}: {len(contours)} edge regions")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c)
                
                if aspect_ratio > 1.5:
                    center_x = full_x + w_c // 2
                    center_y = full_y + h_c // 2
                    
                    candidate = {
                        'method': f'laplacian_{thresh_val}',
                        'bbox': (full_x, full_y, w_c, h_c),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (center_x, center_y),
                        'contour': contour,
                        'roi_offset': (roi_x1, roi_y1)
                    }
                    candidates.append(candidate)
    
    print(f"\nFound {len(candidates)} candidates before filtering")
    
    # Remove duplicates based on overlap
    unique_candidates = []
    for candidate in candidates:
        is_duplicate = False
        x1, y1, w1, h1 = candidate['bbox']
        
        for existing in unique_candidates:
            x2, y2, w2, h2 = existing['bbox']
            
            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            union_area = w1 * h1 + w2 * h2 - overlap_area
            overlap_ratio = overlap_area / union_area if union_area > 0 else 0
            
            if overlap_ratio > 0.3:  # 30% overlap = duplicate
                # Keep the one with larger area
                if candidate['area'] > existing['area']:
                    unique_candidates.remove(existing)
                else:
                    is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Score by how close to center they are
    img_center_x, img_center_y = w // 2, h // 2
    
    for candidate in unique_candidates:
        center_x, center_y = candidate['center']
        distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        
        # Score based on centrality and size
        max_distance = min(w, h) * 0.5
        centrality_score = max(0, 100 * (1 - distance_from_center / max_distance))
        size_score = min(100, candidate['area'] / 100)  # Up to 100 points for area
        
        candidate['score'] = centrality_score + size_score
        candidate['distance_from_center'] = distance_from_center
    
    # Sort by score
    unique_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"After deduplication: {len(unique_candidates)} unique candidates")
    
    return img, unique_candidates, roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def create_ultra_sensitive_visualization(img, candidates, roi, roi_bounds, output_path):
    """Create visualization showing all ultra-sensitive detections."""
    vis_img = img.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    
    # Draw ultra-focused ROI
    cv2.rectangle(vis_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 4)
    cv2.putText(vis_img, "ULTRA-FOCUSED CENTER ROI", (roi_x1, roi_y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw exact center point
    center_point = (img.shape[1]//2, img.shape[0]//2)
    cv2.circle(vis_img, center_point, 8, (255, 0, 255), -1)
    cv2.putText(vis_img, "EXACT CENTER", (center_point[0]+15, center_point[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255)]
    
    for i, candidate in enumerate(candidates[:9]):
        color = colors[i % len(colors)]
        x, y, w_c, h_c = candidate['bbox']
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w_c, y + h_c), color, 3)
        
        # Draw contour outline
        roi_offset = candidate['roi_offset']
        contour_adjusted = candidate['contour'] + np.array([roi_offset[0], roi_offset[1]])
        cv2.drawContours(vis_img, [contour_adjusted], -1, color, 2)
        
        # Draw center point
        center_x, center_y = candidate['center']
        cv2.circle(vis_img, (center_x, center_y), 5, color, -1)
        
        # Labels with score and distance
        score = candidate.get('score', 0)
        dist = candidate.get('distance_from_center', 0)
        
        label = f"#{i+1}: {candidate['method']}"
        details = f"Score:{score:.0f} Dist:{dist:.0f}px"
        
        cv2.putText(vis_img, label, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_img, details, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Ultra-sensitive visualization saved: {output_path}")
    return True


def main():
    """Run ultra-sensitive center detection."""
    if not HAS_DEPS:
        return 1
    
    print("🔍 Ultra-Sensitive Center Detection")
    print("=" * 40)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Run ultra-sensitive detection
    img, candidates, roi, roi_bounds = ultra_sensitive_center_detection(img_path)
    if img is None:
        print("❌ Failed to load image")
        return 1
    
    if candidates:
        # Create visualization
        vis_path = "/app/output/ultra_sensitive_center.jpg"
        create_ultra_sensitive_visualization(img, candidates, roi, roi_bounds, vis_path)
        
        print(f"\n🎯 ULTRA-SENSITIVE DETECTIONS:")
        print("=" * 50)
        
        scale_px_per_mm = 4.15
        
        for i, candidate in enumerate(candidates[:10]):
            x, y, w_c, h_c = candidate['bbox']
            area = candidate['area']
            
            length_px = max(w_c, h_c)
            width_px = min(w_c, h_c)
            length_mm = length_px / scale_px_per_mm
            width_mm = width_px / scale_px_per_mm
            area_mm2 = area / (scale_px_per_mm ** 2)
            
            score = candidate.get('score', 0)
            dist = candidate.get('distance_from_center', 0)
            
            print(f"\n**DETECTION #{i+1} ({candidate['method']}):**")
            print(f"  Score: {score:.1f}")
            print(f"  Distance from center: {dist:.0f} pixels")
            print(f"  Location: {candidate['center']}")
            print(f"  Dimensions: {length_mm:.1f} × {width_mm:.1f} mm")
            print(f"  Area: {area_mm2:.1f} mm²")
            print(f"  Aspect ratio: {candidate['aspect_ratio']:.2f}")
            
            # Quick pliers assessment
            is_reasonable_size = 20 < length_mm < 300 and 2 < width_mm < 100
            is_well_centered = dist < min(img.shape[1], img.shape[0]) * 0.3
            
            if is_reasonable_size and is_well_centered:
                print(f"  **POTENTIAL PLIERS: ✅ POSSIBLE**")
            else:
                print(f"  **POTENTIAL PLIERS: ❌ UNLIKELY**")
        
        print(f"\n📁 Generated file:")
        print(f"  ultra_sensitive_center.jpg - Shows ALL detected objects")
        
        return 0
    else:
        print("\n❌ Even ultra-sensitive detection found nothing in center")
        print("\nThis suggests:")
        print("  • Pliers may have extremely low contrast")
        print("  • Pliers may be outside the center 40% area")
        print("  • Pliers may be the same color as background")
        print("  • Image quality may be limiting detection")
        
        return 1


if __name__ == "__main__":
    exit(main())