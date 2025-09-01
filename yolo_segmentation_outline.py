#!/usr/bin/env python3
"""Extract YOLO segmentation masks to get precise pliers outline."""

import os
import warnings
import json
import numpy as np

def setup_yolo_segmentation():
    """Setup YOLO with segmentation capability."""
    try:
        # Apply PyTorch fixes
        import torch
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
        
        import torch.serialization
        unsafe_globals = [
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.tasks.SegmentationModel',
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'collections.OrderedDict'
        ]
        
        for cls_name in unsafe_globals:
            try:
                torch.serialization.add_safe_globals([cls_name])
            except:
                pass
        
        # Patch torch.load temporarily
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        warnings.filterwarnings("ignore")
        
        from ultralytics import YOLO
        # Use segmentation model explicitly
        model = YOLO('yolov8n-seg.pt')  # segmentation model
        
        # Restore torch.load
        torch.load = original_load
        
        print("✅ YOLO segmentation model loaded!")
        return model
        
    except Exception as e:
        print(f"❌ YOLO segmentation setup failed: {e}")
        return None


def extract_segmentation_masks(model, img_path):
    """Extract precise segmentation masks for the pliers."""
    print("🎯 Extracting YOLO Segmentation Masks")
    print("=" * 50)
    
    try:
        import cv2
        
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Could not load image")
            return None, []
        
        print(f"Processing image: {img.shape[1]}×{img.shape[0]} pixels")
        
        # Run YOLO with segmentation
        print("🔍 Running YOLO segmentation...")
        results = model(img_path, conf=0.05, iou=0.3, verbose=False)
        
        detections_with_masks = []
        vis_img = img.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is not None:
                print(f"Found {len(boxes)} detections")
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detection = {
                        'id': i + 1,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'center': [center_x, center_y],
                        'has_mask': False
                    }
                    
                    # Extract mask if available
                    if masks is not None and i < len(masks.data):
                        print(f"  Extracting mask for {class_name} detection...")
                        
                        mask = masks.data[i].cpu().numpy()
                        print(f"  Mask shape: {mask.shape}")
                        
                        # Resize mask to match image dimensions if needed
                        if mask.shape != (img.shape[0], img.shape[1]):
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                            print(f"  Resized mask to: {mask.shape}")
                        
                        # Convert to binary mask
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        
                        # Find contours from mask
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Use largest contour
                            main_contour = max(contours, key=cv2.contourArea)
                            
                            # Get precise measurements from contour
                            mask_area = cv2.contourArea(main_contour)
                            
                            # Get oriented bounding box for accurate length/width
                            rect = cv2.minAreaRect(main_contour)
                            box_w, box_h = rect[1]
                            angle = rect[2]
                            
                            # Length is always the longer dimension
                            length_px = max(box_w, box_h)
                            width_px = min(box_w, box_h)
                            
                            # Convert to mm using AprilTag scale
                            scale_px_per_mm = 4.15
                            length_mm = length_px / scale_px_per_mm
                            width_mm = width_px / scale_px_per_mm
                            area_mm2 = mask_area / (scale_px_per_mm ** 2)
                            
                            # Get mask statistics
                            mask_pixels = np.sum(binary_mask)
                            
                            # Store contour coordinates for SVG generation
                            contour_coords = main_contour.reshape(-1, 2).tolist()
                            
                            detection.update({
                                'has_mask': True,
                                'mask_area_px': int(mask_area),
                                'mask_pixels': int(mask_pixels),
                                'precise_dimensions_mm': [round(length_mm, 1), round(width_mm, 1)],
                                'precise_area_mm2': round(area_mm2, 1),
                                'oriented_angle': round(angle, 1),
                                'contour_points': contour_coords,
                                'contour_count': len(main_contour),
                                'mask_shape': mask.shape
                            })
                            
                            # Draw precise outline on visualization
                            color = colors[i % len(colors)]
                            
                            # Draw filled mask with transparency
                            mask_colored = np.zeros_like(img)
                            mask_colored[binary_mask > 0] = color
                            vis_img = cv2.addWeighted(vis_img, 0.7, mask_colored, 0.3, 0)
                            
                            # Draw contour outline
                            cv2.drawContours(vis_img, [main_contour], -1, color, 3)
                            
                            # Draw oriented bounding box
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int0(box_points)
                            cv2.drawContours(vis_img, [box_points], -1, color, 2)
                            
                            # Draw center point
                            cv2.circle(vis_img, (center_x, center_y), 8, color, -1)
                            cv2.circle(vis_img, (center_x, center_y), 8, (255, 255, 255), 2)
                            
                            # Detailed labels
                            label = f"#{i+1}: {class_name} ({confidence:.1%})"
                            mask_info = f"Mask: {length_mm:.1f}×{width_mm:.1f}mm"
                            area_info = f"Area: {area_mm2:.1f}mm² ({mask_pixels:,}px)"
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2
                            
                            # Background for text
                            (tw1, th1), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            (tw2, th2), _ = cv2.getTextSize(mask_info, font, 0.6, 1)
                            (tw3, th3), _ = cv2.getTextSize(area_info, font, 0.5, 1)
                            
                            max_width = max(tw1, tw2, tw3)
                            text_height = th1 + th2 + th3 + 30
                            
                            cv2.rectangle(vis_img, (x1, y1-text_height), (x1 + max_width, y1), (255, 255, 255), -1)
                            cv2.rectangle(vis_img, (x1, y1-text_height), (x1 + max_width, y1), color, 2)
                            
                            cv2.putText(vis_img, label, (x1+5, y1-text_height+25), font, font_scale, color, thickness)
                            cv2.putText(vis_img, mask_info, (x1+5, y1-text_height+45), font, 0.6, color, 2)
                            cv2.putText(vis_img, area_info, (x1+5, y1-text_height+60), font, 0.5, color, 1)
                            
                            print(f"  ✅ Mask extracted: {length_mm:.1f}×{width_mm:.1f}mm, area: {area_mm2:.1f}mm²")
                            
                        else:
                            print(f"  ⚠️ No contours found in mask")
                    else:
                        print(f"  ❌ No mask available for {class_name}")
                        # Draw basic bounding box for non-mask detections
                        color = colors[i % len(colors)]
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(vis_img, f"{class_name} ({confidence:.1%})", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    detections_with_masks.append(detection)
        
        # Add title and center marker
        title = "YOLO Segmentation Masks - Precise Pliers Outline"
        cv2.putText(vis_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(vis_img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Mark image center
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.circle(vis_img, img_center, 12, (255, 255, 255), -1)
        cv2.circle(vis_img, img_center, 12, (0, 0, 0), 3)
        cv2.putText(vis_img, "CENTER", (img_center[0] + 20, img_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return vis_img, detections_with_masks
        
    except Exception as e:
        print(f"❌ Segmentation extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def generate_svg_from_yolo_mask(detection, output_path):
    """Generate SVG file from YOLO segmentation mask with actual contour outline."""
    if not detection.get('has_mask', False):
        print(f"❌ No mask available for SVG generation")
        return False
    
    try:
        length_mm = detection['precise_dimensions_mm'][0]
        width_mm = detection['precise_dimensions_mm'][1]
        area_mm2 = detection['precise_area_mm2']
        class_name = detection['class']
        confidence = detection['confidence']
        scale_px_per_mm = 4.15
        
        # Get contour points and convert to mm coordinates
        contour_points = detection.get('contour_points', [])
        if not contour_points:
            print(f"❌ No contour points available for detailed outline")
            return False
        
        # Find bounding box of contour to center it in SVG
        min_x = min(point[0] for point in contour_points)
        min_y = min(point[1] for point in contour_points)
        max_x = max(point[0] for point in contour_points)
        max_y = max(point[1] for point in contour_points)
        
        # Convert contour points to mm and translate to origin
        svg_points = []
        for x, y in contour_points:
            # Convert to mm and translate so contour starts near origin
            svg_x = (x - min_x) / scale_px_per_mm + 10  # 10mm margin
            svg_y = (y - min_y) / scale_px_per_mm + 20  # 20mm margin for title
            svg_points.append(f"{svg_x:.2f},{svg_y:.2f}")
        
        # Create path string for actual pliers outline
        path_data = f"M {svg_points[0]} " + " L ".join(svg_points[1:]) + " Z"
        
        # SVG dimensions based on actual contour
        svg_width = (max_x - min_x) / scale_px_per_mm + 20  # margins
        svg_height = (max_y - min_y) / scale_px_per_mm + 30  # margins + title space
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" 
     width="{svg_width:.1f}mm" height="{svg_height:.1f}mm">
  
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 3mm; fill: #333; font-weight: bold; }}
      .dimension {{ font-family: Arial, sans-serif; font-size: 2mm; fill: red; font-weight: bold; }}
      .metadata {{ font-family: Arial, sans-serif; font-size: 1.5mm; fill: gray; }}
      .pliers-outline {{ fill: #f0f0f0; fill-opacity: 0.3; stroke: black; stroke-width: 0.3mm; }}
      .dimension-line {{ stroke: red; stroke-width: 0.15mm; }}
      .grid {{ stroke: #ddd; stroke-width: 0.05mm; }}
    </style>
  </defs>
  
  <!-- Background grid for reference -->
  <defs>
    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
      <path d="M 10 0 L 0 0 0 10" fill="none" class="grid"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)" opacity="0.3"/>
  
  <!-- Actual pliers outline from YOLO segmentation -->
  <path d="{path_data}" class="pliers-outline"/>
  
  <!-- Title -->
  <text x="{svg_width/2:.1f}" y="12" text-anchor="middle" class="title">
    {class_name.upper()} SEGMENTATION OUTLINE
  </text>
  
  <!-- Dimensions -->
  <text x="10" y="{svg_height-15:.1f}" class="dimension">
    {length_mm:.1f} × {width_mm:.1f} mm
  </text>
  
  <!-- Metadata -->
  <text x="10" y="{svg_height-10:.1f}" class="metadata">
    Area: {area_mm2:.1f} mm² | Confidence: {confidence:.1%} | YOLO Segmentation
  </text>
  
  <text x="10" y="{svg_height-6:.1f}" class="metadata">
    BatchGrids CV | Scale: {scale_px_per_mm} px/mm | Contour: {len(contour_points)} points
  </text>
  
  <text x="10" y="{svg_height-2:.1f}" class="metadata">
    Generated from mask pixels for precise fabrication template
  </text>
  
  <!-- Scale reference -->
  <line x1="10" y1="{svg_height-22:.1f}" x2="20" y2="{svg_height-22:.1f}" stroke="black" stroke-width="0.2mm"/>
  <text x="15" y="{svg_height-24:.1f}" text-anchor="middle" class="metadata">10mm</text>
  
</svg>'''
        
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        print(f"✅ SVG with actual pliers outline generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ SVG generation failed: {e}")
        return False


def main():
    """Main YOLO segmentation extraction function."""
    print("🎭 YOLO Segmentation Mask Extraction")
    print("=" * 50)
    
    # Setup YOLO segmentation
    model = setup_yolo_segmentation()
    if model is None:
        return 1
    
    # Extract segmentation masks
    img_path = "/app/output/enhanced_converted_image.jpg"
    vis_img, detections = extract_segmentation_masks(model, img_path)
    
    if vis_img is None:
        print("❌ Failed to extract segmentation masks")
        return 1
    
    # Save visualization
    vis_path = "/app/output/yolo_segmentation_masks.jpg"
    import cv2
    cv2.imwrite(vis_path, vis_img)
    
    # Process results
    print(f"\n🎯 YOLO SEGMENTATION RESULTS:")
    print("=" * 50)
    
    pliers_found = False
    
    for detection in detections:
        print(f"\n**DETECTION #{detection['id']}:**")
        print(f"  Class: {detection['class']}")
        print(f"  Confidence: {detection['confidence']:.1%}")
        
        if detection.get('has_mask', False):
            length_mm, width_mm = detection['precise_dimensions_mm']
            area_mm2 = detection['precise_area_mm2']
            
            print(f"  🎭 SEGMENTATION MASK:")
            print(f"    Precise Dimensions: {length_mm} × {width_mm} mm")
            print(f"    Precise Area: {area_mm2} mm²")
            print(f"    Mask Pixels: {detection['mask_pixels']:,}")
            print(f"    Contour Points: {detection['contour_count']}")
            print(f"    Orientation: {detection['oriented_angle']}°")
            
            # Analyze if this is the pliers
            is_tool = detection['class'] in ['scissors', 'knife', 'spoon', 'fork']
            is_reasonable_size = 50 < length_mm < 250 and 10 < width_mm < 100
            
            if is_tool and is_reasonable_size:
                print(f"    🎉 **PLIERS IDENTIFIED!**")
                pliers_found = True
                
                # Generate SVG for the pliers
                svg_path = f"/app/output/pliers_outline_{detection['class']}.svg"
                svg_success = generate_svg_from_yolo_mask(detection, svg_path)
                
                if svg_success:
                    print(f"    ✅ SVG outline generated: pliers_outline_{detection['class']}.svg")
            else:
                print(f"    ❓ May not be pliers (size or class)")
        else:
            print(f"  ❌ No segmentation mask available")
    
    # Save detailed results
    results_data = {
        'model': 'yolov8n-seg',
        'segmentation_method': 'yolo_masks',
        'total_detections': len(detections),
        'detections_with_masks': [d for d in detections if d.get('has_mask', False)],
        'pliers_found': pliers_found,
        'scale_px_per_mm': 4.15
    }
    
    with open('/app/output/yolo_segmentation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n📁 Generated files:")
    print(f"  📷 yolo_segmentation_masks.jpg - Visual masks")
    print(f"  📄 yolo_segmentation_results.json - Detailed data")
    if pliers_found:
        print(f"  🎨 pliers_outline_*.svg - Vector outline for fabrication")
    
    if pliers_found:
        print(f"\n🎉 SUCCESS! YOLO segmentation found and outlined your pliers!")
    else:
        print(f"\n⚠️  YOLO detected objects but no clear pliers segmentation")
    
    return 0 if pliers_found else 1


if __name__ == "__main__":
    exit(main())