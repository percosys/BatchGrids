#!/usr/bin/env python3
"""Tuned detection algorithm focused on center area to capture actual pliers."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def center_focused_pliers_detection(img_path):
    """Enhanced detection focused specifically on center area objects."""
    print(f"Center-focused pliers detection: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h} pixels")
    
    # Define a smaller, more focused center region
    center_margin = 0.25  # 25% margin - focus on center 50% of image
    roi_x1 = int(w * center_margin)
    roi_y1 = int(h * center_margin)
    roi_x2 = int(w * (1 - center_margin))
    roi_y2 = int(h * (1 - center_margin))
    
    print(f"Focused ROI: {roi_x2-roi_x1}×{roi_y2-roi_y1} pixels (center 50%)")
    
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    candidates = []
    
    # Method 1: Improved adaptive threshold with multiple parameters
    print("\n=== Method 1: Multiple Adaptive Thresholds ===")
    
    adaptive_params = [
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 3),   # Fine details
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 5),   # Medium
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 21, 8),   # Coarse
        (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 3),       # Mean-based fine
        (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 5),       # Mean-based medium
    ]
    
    for i, (method, block_size, C) in enumerate(adaptive_params):
        thresh = cv2.adaptiveThreshold(roi, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
        
        # More aggressive morphological operations to connect pliers parts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Adaptive {i+1}: {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Focus on medium-sized objects (not tiny lines, not huge areas)
            if 2000 < area < 100000:  # Reasonable pliers size range
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Convert to full image coordinates
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                
                # Filter for tool-like aspect ratios (not extremely thin lines)
                if 2.0 < aspect_ratio < 15.0:  # Reasonable tool range
                    
                    # Calculate how centered it is
                    center_x = full_x + w_c // 2
                    center_y = full_y + h_c // 2
                    img_center_x, img_center_y = w // 2, h // 2
                    
                    # Distance from image center
                    center_distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                    max_center_dist = min(w, h) * 0.25  # Must be within 25% of center
                    
                    if center_distance < max_center_dist:
                        
                        # Calculate shape complexity (more complex = more likely to be real object)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        complexity = area / hull_area if hull_area > 0 else 0
                        
                        candidate = {
                            'method': f'adaptive_{i+1}',
                            'bbox': (full_x, full_y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y),
                            'center_distance': center_distance,
                            'complexity': complexity,
                            'contour': contour,
                            'roi_offset': (roi_x1, roi_y1)
                        }
                        
                        candidates.append(candidate)
    
    # Method 2: Edge-based detection with better parameters
    print("\n=== Method 2: Enhanced Edge Detection ===")
    
    # Pre-process with Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    
    edge_params = [
        (30, 100, 3),   # Medium sensitivity
        (20, 80, 3),    # High sensitivity
        (50, 150, 5),   # Lower sensitivity
    ]
    
    for i, (low, high, aperture) in enumerate(edge_params):
        edges = cv2.Canny(blurred, low, high, apertureSize=aperture)
        
        # Connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Edge {i+1}: {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 1500 < area < 80000:  # Pliers-appropriate size
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                
                if 2.5 < aspect_ratio < 12.0:  # Tool-like shape
                    center_x = full_x + w_c // 2
                    center_y = full_y + h_c // 2
                    
                    center_distance = np.sqrt((center_x - w//2)**2 + (center_y - h//2)**2)
                    
                    if center_distance < min(w, h) * 0.25:  # Must be centered
                        
                        # Calculate perimeter to area ratio (tools have specific ratios)
                        perimeter = cv2.arcLength(contour, True)
                        shape_ratio = perimeter / np.sqrt(area) if area > 0 else 0
                        
                        candidate = {
                            'method': f'edge_{i+1}',
                            'bbox': (full_x, full_y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y),
                            'center_distance': center_distance,
                            'shape_ratio': shape_ratio,
                            'contour': contour,
                            'roi_offset': (roi_x1, roi_y1)
                        }
                        
                        candidates.append(candidate)
    
    # Method 3: Intensity-based detection for dark objects
    print("\n=== Method 3: Dark Object Detection ===")
    
    # Find regions darker than the background
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)
    
    # Multiple darkness thresholds
    dark_thresholds = [
        mean_intensity - std_intensity * 0.5,
        mean_intensity - std_intensity * 0.8,
        mean_intensity - std_intensity * 1.2
    ]
    
    for i, threshold in enumerate(dark_thresholds):
        dark_mask = (roi < threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Dark {i+1} (thresh={threshold:.1f}): {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 3000 < area < 60000:  # Focus on substantial dark objects
                x, y, w_c, h_c = cv2.boundingRect(contour)
                full_x = roi_x1 + x
                full_y = roi_y1 + y
                
                aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
                
                if 1.5 < aspect_ratio < 10.0:  # Reasonable tool shape
                    center_x = full_x + w_c // 2
                    center_y = full_y + h_c // 2
                    
                    center_distance = np.sqrt((center_x - w//2)**2 + (center_y - h//2)**2)
                    
                    if center_distance < min(w, h) * 0.3:  # Somewhat centered
                        
                        candidate = {
                            'method': f'dark_{i+1}',
                            'bbox': (full_x, full_y, w_c, h_c),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': (center_x, center_y),
                            'center_distance': center_distance,
                            'darkness_threshold': threshold,
                            'contour': contour,
                            'roi_offset': (roi_x1, roi_y1)
                        }
                        
                        candidates.append(candidate)
    
    print(f"\nFound {len(candidates)} candidates before filtering")
    
    # Remove duplicates and score candidates
    unique_candidates = []
    for candidate in candidates:
        is_duplicate = False
        for existing in unique_candidates:
            dx = abs(candidate['center'][0] - existing['center'][0])
            dy = abs(candidate['center'][1] - existing['center'][1])
            if dx < 50 and dy < 50:  # Very close
                # Keep the one with better properties
                if candidate['area'] > existing['area'] and candidate.get('complexity', 0) > existing.get('complexity', 0):
                    unique_candidates.remove(existing)
                else:
                    is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Score candidates for pliers-likeness
    for candidate in unique_candidates:
        score = 0
        
        # Prefer objects in the very center
        max_center_dist = min(w, h) * 0.15
        if candidate['center_distance'] < max_center_dist:
            score += 100
        elif candidate['center_distance'] < min(w, h) * 0.25:
            score += 50
        
        # Prefer pliers-like aspect ratios
        ar = candidate['aspect_ratio']
        if 3 < ar < 8:
            score += 100
        elif 2 < ar < 10:
            score += 50
        
        # Prefer reasonable sizes
        area = candidate['area']
        if 5000 < area < 30000:  # Sweet spot for small pliers
            score += 100
        elif 2000 < area < 50000:
            score += 50
        
        # Prefer more complex shapes (not just rectangles)
        if 'complexity' in candidate and candidate['complexity'] > 0.7:
            score += 50
        
        candidate['pliers_score'] = score
    
    # Sort by score
    unique_candidates.sort(key=lambda x: x['pliers_score'], reverse=True)
    
    print(f"After filtering: {len(unique_candidates)} unique candidates")
    
    return img, unique_candidates, roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def create_center_focused_visualization(img, candidates, roi, roi_bounds, output_path):
    """Create visualization focused on center detections."""
    
    vis_img = img.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    
    # Draw the focused ROI
    cv2.rectangle(vis_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
    cv2.putText(vis_img, "FOCUSED CENTER ROI", (roi_x1, roi_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw image center point
    img_center = (img.shape[1]//2, img.shape[0]//2)
    cv2.circle(vis_img, img_center, 10, (255, 0, 255), -1)
    cv2.putText(vis_img, "CENTER", (img_center[0]+15, img_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, candidate in enumerate(candidates[:6]):
        color = colors[i % len(colors)]
        x, y, w_c, h_c = candidate['bbox']
        
        # Draw thick bounding box
        cv2.rectangle(vis_img, (x, y), (x + w_c, y + h_c), color, 4)
        
        # Draw actual contour
        roi_offset = candidate['roi_offset']
        contour_adjusted = candidate['contour'] + np.array([roi_offset[0], roi_offset[1]])
        cv2.drawContours(vis_img, [contour_adjusted], -1, color, 2)
        
        # Detailed label
        score = candidate.get('pliers_score', 0)
        center_dist = candidate['center_distance']
        
        label = f"#{i+1}: {candidate['method']} (Score:{score})"
        details = f"Dist:{center_dist:.0f}px AR:{candidate['aspect_ratio']:.1f}"
        
        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(vis_img, (x, y - text_h - baseline - 25), 
                     (x + max(text_w, 300), y), (255, 255, 255), -1)
        cv2.putText(vis_img, label, (x, y - baseline - 15), font, font_scale, color, thickness)
        cv2.putText(vis_img, details, (x, y - baseline + 5), font, 0.5, color, 1)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Center-focused visualization saved: {output_path}")
    return True


def main():
    """Run center-focused pliers detection."""
    if not HAS_DEPS:
        return 1
    
    print("🎯 Center-Focused Pliers Detection")
    print("=" * 40)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Run enhanced detection
    img, candidates, roi, roi_bounds = center_focused_pliers_detection(img_path)
    if img is None:
        print("❌ Failed to load image")
        return 1
    
    if candidates:
        # Create visualization
        vis_path = "/app/output/center_focused_detection.jpg"
        create_center_focused_visualization(img, candidates, roi, roi_bounds, vis_path)
        
        # Analyze top candidates
        print(f"\n🎯 TOP CENTER CANDIDATES:")
        print("=" * 50)
        
        scale_px_per_mm = 4.15
        
        for i, candidate in enumerate(candidates[:5]):
            x, y, w_c, h_c = candidate['bbox']
            area = candidate['area']
            
            length_px = max(w_c, h_c)
            width_px = min(w_c, h_c)
            length_mm = length_px / scale_px_per_mm
            width_mm = width_px / scale_px_per_mm
            area_mm2 = area / (scale_px_per_mm ** 2)
            
            center_dist = candidate['center_distance']
            score = candidate.get('pliers_score', 0)
            
            print(f"\n**CANDIDATE #{i+1} ({candidate['method']}):**")
            print(f"  Score: {score}")
            print(f"  Location: {candidate['center']} (distance from center: {center_dist:.0f}px)")
            print(f"  Dimensions: {length_mm:.1f} × {width_mm:.1f} mm")
            print(f"  Area: {area_mm2:.1f} mm²")
            print(f"  Aspect ratio: {candidate['aspect_ratio']:.2f}")
            
            # Pliers assessment
            is_pliers_size = 80 < length_mm < 200 and 5 < width_mm < 40
            is_centered = center_dist < min(img.shape[1], img.shape[0]) * 0.25
            is_pliers_shape = 2 < candidate['aspect_ratio'] < 12
            
            print(f"  Pliers assessment:")
            print(f"    Size (80-200mm × 5-40mm): {'✅' if is_pliers_size else '❌'}")
            print(f"    Centered (<25% from center): {'✅' if is_centered else '❌'}")
            print(f"    Tool-like shape: {'✅' if is_pliers_shape else '❌'}")
            
            if is_pliers_size and is_centered and is_pliers_shape:
                print(f"    **🎯 LIKELY PLIERS: ✅ YES**")
            else:
                print(f"    **LIKELY PLIERS: ❌ NO**")
        
        # Save results
        results = {
            'detection_method': 'center_focused_tuned',
            'roi_bounds': roi_bounds,
            'candidates': [{
                'rank': i+1,
                'method': c['method'],
                'bbox': c['bbox'],
                'center': c['center'],
                'dimensions_mm': [max(c['bbox'][2], c['bbox'][3]) / 4.15, min(c['bbox'][2], c['bbox'][3]) / 4.15],
                'score': c.get('pliers_score', 0),
                'center_distance': c['center_distance']
            } for i, c in enumerate(candidates[:10])]
        }
        
        with open('/app/output/center_focused_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Files generated:")
        print(f"  center_focused_detection.jpg - Visual results")
        print(f"  center_focused_results.json - Detailed analysis")
        
        return 0
    else:
        print("❌ No center objects detected")
        return 1


if __name__ == "__main__":
    exit(main())