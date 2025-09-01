#!/usr/bin/env python3
"""Show the actual detected object outlines to verify if we're capturing the pliers correctly."""

import os
import json
import numpy as np

try:
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def load_and_visualize_detections(img_path):
    """Load image and show detected objects with their actual outlines."""
    print(f"Analyzing detections in: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image: {w}×{h} pixels")
    
    # Re-run the same detection that found the "pliers"
    # Focus on center region
    margin = 0.15  # 15% margin
    roi_x1 = int(w * margin)
    roi_y1 = int(h * margin)
    roi_x2 = int(w * (1 - margin))
    roi_y2 = int(h * (1 - margin))
    
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Multiple detection methods
    all_detections = []
    
    # Method 1: Adaptive threshold (this found the "pliers")
    adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    adaptive_clean = cv2.morphologyEx(adaptive_binary, cv2.MORPH_CLOSE, kernel)
    
    contours_adaptive, _ = cv2.findContours(adaptive_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Adaptive threshold found {len(contours_adaptive)} contours")
    
    # Method 2: Edge detection
    edges = cv2.Canny(roi, 30, 100)
    kernel_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edges)
    
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Edge detection found {len(contours_edges)} contours")
    
    # Analyze all contours and recreate the detection logic
    candidates = []
    
    # Process adaptive threshold contours (most likely source of "pliers")
    for i, contour in enumerate(contours_adaptive):
        area = cv2.contourArea(contour)
        
        if area > 500:  # Lower threshold to see more
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Convert back to full image coordinates
            full_x = roi_x1 + x
            full_y = roi_y1 + y
            
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
            
            if aspect_ratio > 1.5:  # Any elongated object
                # Calculate center
                center_x = full_x + w_c//2
                center_y = full_y + h_c//2
                
                # Distance from image center
                img_center_x, img_center_y = w//2, h//2
                dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                # Score this detection
                score = 0
                if aspect_ratio > 15:
                    score += 400  # Very high aspect ratio
                elif aspect_ratio > 10:
                    score += 300
                elif aspect_ratio > 5:
                    score += 200
                elif aspect_ratio > 3:
                    score += 100
                
                # Size score
                if 1000 < area < 10000:
                    score += 100
                elif area > 500:
                    score += 50
                
                # Center score
                if dist_from_center < min(w, h) * 0.3:
                    score += 100
                
                candidate = {
                    'method': 'adaptive',
                    'contour_id': i,
                    'bbox': (full_x, full_y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (center_x, center_y),
                    'score': score,
                    'contour': contour,
                    'roi_offset': (roi_x1, roi_y1)
                }
                
                candidates.append(candidate)
    
    # Process edge contours too
    for i, contour in enumerate(contours_edges):
        area = cv2.contourArea(contour)
        
        if area > 300:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            full_x = roi_x1 + x
            full_y = roi_y1 + y
            
            aspect_ratio = max(w_c, h_c) / min(w_c, h_c) if min(w_c, h_c) > 0 else 1
            
            if aspect_ratio > 1.5:
                center_x = full_x + w_c//2
                center_y = full_y + h_c//2
                
                score = aspect_ratio * 10 + (50 if 500 < area < 5000 else 0)
                
                candidate = {
                    'method': 'edge',
                    'contour_id': i,
                    'bbox': (full_x, full_y, w_c, h_c),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (center_x, center_y),
                    'score': score,
                    'contour': contour,
                    'roi_offset': (roi_x1, roi_y1)
                }
                
                candidates.append(candidate)
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return img, candidates, roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def create_detailed_visualization(img, candidates, roi, roi_bounds, output_path):
    """Create detailed visualization showing what was actually detected."""
    
    # Create a large visualization with multiple views
    vis_height = img.shape[0] + roi.shape[0] + 100
    vis_width = max(img.shape[1], roi.shape[1] * 2) + 100
    vis = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255
    
    # Place original image
    vis[:img.shape[0], :img.shape[1]] = img
    
    # Draw ROI bounds on original image
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(vis, "ROI (Search Area)", (roi_x1, roi_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show detected objects on original image
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, candidate in enumerate(candidates[:6]):  # Top 6 candidates
        x, y, w_c, h_c = candidate['bbox']
        color = colors[i % len(colors)]
        
        # Draw thick bounding box
        cv2.rectangle(vis, (x, y), (x + w_c, y + h_c), color, 3)
        
        # Draw the actual contour outline
        roi_x1, roi_y1 = candidate['roi_offset']
        contour_adjusted = candidate['contour'] + np.array([roi_x1, roi_y1])
        cv2.drawContours(vis, [contour_adjusted], -1, color, 2)
        
        # Label with details
        label = f"#{i+1}: {candidate['method']}"
        details = f"AR:{candidate['aspect_ratio']:.1f} A:{candidate['area']:.0f}"
        
        cv2.putText(vis, label, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, details, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Place ROI view
    roi_start_y = img.shape[0] + 20
    roi_3ch = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    vis[roi_start_y:roi_start_y + roi.shape[0], :roi.shape[1]] = roi_3ch
    
    cv2.putText(vis, "ROI - Grayscale", (5, roi_start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add detection method visualization
    method_start_x = roi.shape[1] + 20
    if method_start_x + roi.shape[1] < vis_width:
        # Show adaptive threshold result
        adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        adaptive_3ch = cv2.cvtColor(adaptive_binary, cv2.COLOR_GRAY2BGR)
        vis[roi_start_y:roi_start_y + roi.shape[0], method_start_x:method_start_x + roi.shape[1]] = adaptive_3ch
        
        cv2.putText(vis, "Adaptive Threshold", (method_start_x, roi_start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, vis)
    print(f"Detailed visualization saved: {output_path}")
    
    return True


def analyze_top_detections(candidates, scale_px_per_mm=4.15):
    """Analyze the top detections to see if they could be pliers."""
    print(f"\n🔍 DETAILED ANALYSIS OF TOP DETECTIONS:")
    print("=" * 60)
    
    analysis = []
    
    for i, candidate in enumerate(candidates[:5]):
        x, y, w_c, h_c = candidate['bbox']
        area = candidate['area']
        aspect_ratio = candidate['aspect_ratio']
        
        # Convert to mm
        length_px = max(w_c, h_c)
        width_px = min(w_c, h_c)
        length_mm = length_px / scale_px_per_mm
        width_mm = width_px / scale_px_per_mm
        area_mm2 = area / (scale_px_per_mm ** 2)
        
        # Pliers analysis
        is_pliers_length = 80 < length_mm < 200
        is_pliers_width = 3 < width_mm < 30
        is_pliers_aspect = aspect_ratio > 8
        is_pliers_size = is_pliers_length and is_pliers_width and is_pliers_aspect
        
        result = {
            'rank': i + 1,
            'method': candidate['method'],
            'dimensions_mm': [length_mm, width_mm],
            'area_mm2': area_mm2,
            'aspect_ratio': aspect_ratio,
            'bbox': candidate['bbox'],
            'center': candidate['center'],
            'score': candidate['score'],
            'pliers_analysis': {
                'length_ok': is_pliers_length,
                'width_ok': is_pliers_width,
                'aspect_ok': is_pliers_aspect,
                'likely_pliers': is_pliers_size
            }
        }
        
        print(f"\n**DETECTION #{i+1} ({candidate['method']}):**")
        print(f"  Location: {candidate['center']} (center)")
        print(f"  Size: {length_mm:.1f} × {width_mm:.1f} mm")
        print(f"  Area: {area_mm2:.1f} mm²")
        print(f"  Aspect ratio: {aspect_ratio:.1f}")
        print(f"  Score: {candidate['score']:.1f}")
        print(f"  Pliers analysis:")
        print(f"    Length (80-200mm): {'✅' if is_pliers_length else '❌'} ({length_mm:.1f}mm)")
        print(f"    Width (3-30mm): {'✅' if is_pliers_width else '❌'} ({width_mm:.1f}mm)")
        print(f"    Aspect (>8): {'✅' if is_pliers_aspect else '❌'} ({aspect_ratio:.1f})")
        print(f"    **LIKELY PLIERS: {'✅ YES' if is_pliers_size else '❌ NO'}**")
        
        analysis.append(result)
    
    return analysis


def main():
    """Show actual detected object outlines."""
    if not HAS_DEPS:
        return 1
    
    print("👀 Showing Actual Detected Object Outlines")
    print("=" * 50)
    
    img_path = "/app/output/enhanced_converted_image.jpg"
    
    # Load and analyze detections
    img, candidates, roi, roi_bounds = load_and_visualize_detections(img_path)
    if img is None:
        print("❌ Failed to load image")
        return 1
    
    print(f"Found {len(candidates)} total candidates")
    
    if candidates:
        # Create detailed visualization
        vis_path = "/app/output/detailed_detection_outlines.jpg"
        create_detailed_visualization(img, candidates, roi, roi_bounds, vis_path)
        
        # Analyze top detections
        analysis = analyze_top_detections(candidates)
        
        # Save analysis
        with open('/app/output/detection_outline_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Summary
        pliers_candidates = [a for a in analysis if a['pliers_analysis']['likely_pliers']]
        
        print(f"\n" + "=" * 50)
        print(f"📊 SUMMARY:")
        print(f"Total detections analyzed: {len(analysis)}")
        print(f"Likely pliers candidates: {len(pliers_candidates)}")
        
        if pliers_candidates:
            best = pliers_candidates[0]
            print(f"\n🎯 BEST PLIERS CANDIDATE:")
            print(f"  Detection #{best['rank']} ({best['method']})")
            print(f"  Dimensions: {best['dimensions_mm'][0]:.1f} × {best['dimensions_mm'][1]:.1f} mm")
            print(f"  Location: {best['center']}")
        else:
            print(f"\n⚠️  NO CONVINCING PLIERS FOUND")
            print(f"The detected objects may be:")
            print(f"  • Background artifacts")
            print(f"  • Paper edges or text")
            print(f"  • Shadows or lighting effects")
            print(f"  • Not the actual pliers outline")
        
        print(f"\n📁 CHECK THE VISUALIZATION:")
        print(f"  detailed_detection_outlines.jpg - Shows exactly what was detected")
        
        return 0
    else:
        print("❌ No detections found")
        return 1


if __name__ == "__main__":
    exit(main())