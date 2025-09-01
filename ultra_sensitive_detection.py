#!/usr/bin/env python3
"""Ultra-sensitive detection for small tools like flush cutting pliers."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def ultra_sensitive_tool_detection(img_path):
    """Ultra-sensitive detection for very small tools."""
    print(f"Ultra-sensitive detection: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h} pixels")
    
    all_candidates = []
    
    # Method 1: Very aggressive edge detection
    print("\n=== Method 1: Aggressive Edge Detection ===")
    
    # Multiple edge detection attempts with different parameters
    edge_params = [
        (20, 60, 3),   # Very sensitive
        (30, 90, 3),   # Medium sensitive  
        (40, 120, 5),  # Less sensitive
    ]
    
    for i, (low, high, aperture) in enumerate(edge_params):
        edges = cv2.Canny(gray, low, high, apertureSize=aperture)
        
        # Fill gaps with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Edge params {low}-{high}: {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Very low threshold - detect tiny objects
            if area > 200:  # Just 200 pixels minimum
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Check if it's roughly centered (not at edges)
                center_x, center_y = x + w_c//2, y + h_c//2
                edge_margin = min(w, h) * 0.15  # 15% margin
                
                if (edge_margin < center_x < w - edge_margin and 
                    edge_margin < center_y < h - edge_margin):
                    
                    aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                    
                    # Accept any elongated shape
                    if aspect_ratio > 1.2:
                        all_candidates.append({
                            'method': f'edge_{i+1}',
                            'bbox': (x, y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y)
                        })
    
    # Method 2: Multiple threshold approaches
    print("\n=== Method 2: Multiple Thresholds ===")
    
    # Otsu threshold
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Adaptive thresholds with different parameters
    adaptive_params = [
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 5),
        (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
    ]
    
    thresholded_images = [('otsu', thresh_otsu)]
    
    for i, (method, block_size, C) in enumerate(adaptive_params):
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
        thresholded_images.append((f'adaptive_{i+1}', thresh_adaptive))
    
    for thresh_name, thresh_img in thresholded_images:
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  {thresh_name}: {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 150:  # Even lower threshold
                x, y, w_c, h_c = cv2.boundingRect(contour)
                center_x, center_y = x + w_c//2, y + h_c//2
                
                # More generous center region
                edge_margin = min(w, h) * 0.1  # 10% margin
                
                if (edge_margin < center_x < w - edge_margin and 
                    edge_margin < center_y < h - edge_margin):
                    
                    aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                    
                    if aspect_ratio > 1.1:  # Very permissive
                        all_candidates.append({
                            'method': thresh_name,
                            'bbox': (x, y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y)
                        })
    
    # Method 3: Intensity-based detection
    print("\n=== Method 3: Intensity Analysis ===")
    
    # Look for regions that are darker than average
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Create mask for darker regions
    dark_threshold = mean_intensity - (std_intensity * 0.5)
    dark_mask = (gray < dark_threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  Intensity-based: {len(contours)} dark regions")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 100:  # Absolute minimum
            x, y, w_c, h_c = cv2.boundingRect(contour)
            center_x, center_y = x + w_c//2, y + h_c//2
            
            if (w*0.1 < center_x < w*0.9 and h*0.1 < center_y < h*0.9):
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                
                if aspect_ratio > 1.0:
                    all_candidates.append({
                        'method': 'intensity',
                        'bbox': (x, y, w_c, h_c),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (center_x, center_y)
                    })
    
    print(f"\nTotal candidates before filtering: {len(all_candidates)}")
    
    # Remove duplicates and filter
    unique_candidates = []
    for candidate in all_candidates:
        is_duplicate = False
        for existing in unique_candidates:
            dx = abs(candidate['center'][0] - existing['center'][0])
            dy = abs(candidate['center'][1] - existing['center'][1])
            if dx < 20 and dy < 20:  # Very close
                # Keep the one with higher aspect ratio
                if candidate['aspect_ratio'] > existing['aspect_ratio']:
                    unique_candidates.remove(existing)
                else:
                    is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Sort by a combination of aspect ratio and reasonable size
    for candidate in unique_candidates:
        # Score based on tool-like characteristics
        score = 0
        
        # Higher aspect ratio = more tool-like
        score += candidate['aspect_ratio'] * 10
        
        # Reasonable size (not too tiny, not too huge)
        area = candidate['area']
        if 500 < area < 20000:  # Sweet spot
            score += 50
        elif 200 < area < 50000:
            score += 20
        
        # Prefer center placement
        img_center_x, img_center_y = w//2, h//2
        dx = abs(candidate['center'][0] - img_center_x)
        dy = abs(candidate['center'][1] - img_center_y)
        center_distance = np.sqrt(dx*dx + dy*dy)
        max_center_dist = min(w, h) * 0.3
        
        if center_distance < max_center_dist:
            score += 30
        
        candidate['score'] = score
    
    # Sort by score
    unique_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Unique candidates after filtering: {len(unique_candidates)}")
    for i, candidate in enumerate(unique_candidates[:10]):  # Show top 10
        print(f"  {i+1}. {candidate['method']} - Score: {candidate['score']:.1f}, "
              f"Aspect: {candidate['aspect_ratio']:.2f}, Area: {candidate['area']:.0f}")
    
    return img, unique_candidates


def measure_all_candidates(candidates, scale_px_per_mm=4.15):
    """Measure all candidates to find pliers-sized objects."""
    measurements = []
    
    for i, candidate in enumerate(candidates[:15]):  # Top 15 candidates
        x, y, w_c, h_c = candidate['bbox']
        area_px = candidate['area']
        
        length_px = max(w_c, h_c)
        width_px = min(w_c, h_c)
        
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = area_px / (scale_px_per_mm ** 2)
        
        # Flag likely pliers based on expected dimensions
        is_pliers_size = (80 < length_mm < 200 and 5 < width_mm < 40)
        
        measurement = {
            'rank': i + 1,
            'method': candidate['method'],
            'score': candidate.get('score', 0),
            'length_mm': round(length_mm, 1),
            'width_mm': round(width_mm, 1),
            'area_mm2': round(area_mm2, 1),
            'length_px': length_px,
            'width_px': width_px,
            'area_px': int(area_px),
            'bbox': candidate['bbox'],
            'aspect_ratio': candidate['aspect_ratio'],
            'is_pliers_size': is_pliers_size
        }
        
        measurements.append(measurement)
    
    return measurements


def main():
    """Run ultra-sensitive detection."""
    if not HAS_DEPS:
        return 1
    
    print("🔍 Ultra-Sensitive Tool Detection")
    print("=" * 40)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Ultra-sensitive detection
    img, candidates = ultra_sensitive_tool_detection(img_path)
    if img is None or not candidates:
        print("❌ No objects detected")
        return 1
    
    # Measure all candidates
    measurements = measure_all_candidates(candidates)
    
    # Filter for pliers-like measurements
    pliers_candidates = [m for m in measurements if m['is_pliers_size']]
    
    if pliers_candidates:
        print(f"\n🎯 POTENTIAL PLIERS FOUND! ({len(pliers_candidates)} candidates)")
        print("=" * 50)
        
        for m in pliers_candidates:
            print(f"\n**PLIERS CANDIDATE #{m['rank']} ({m['method']}):**")
            print(f"  Length: {m['length_mm']} mm ({m['length_mm']/25.4:.1f}\")")
            print(f"  Width: {m['width_mm']} mm ({m['width_mm']/25.4:.1f}\")")
            print(f"  Aspect ratio: {m['aspect_ratio']:.2f}")
            print(f"  Score: {m['score']:.1f}")
        
        # Save results
        results = {
            'detection_method': 'ultra_sensitive',
            'pliers_candidates': pliers_candidates,
            'all_measurements': measurements[:10],  # Top 10
            'scale_px_per_mm': 4.15
        }
        
        with open('/app/output/ultra_sensitive_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    else:
        print("\n⚠️  No pliers-sized objects found")
        print("Showing all detected objects for reference:")
        
        for m in measurements[:10]:
            print(f"\n{m['rank']}. {m['method']} - {m['length_mm']}×{m['width_mm']}mm")
    
    return 0


if __name__ == "__main__":
    exit(main())