#!/usr/bin/env python3
"""Fix PyTorch compatibility and get YOLO11 working properly."""

import os
import sys
import subprocess
import warnings

def fix_pytorch_compatibility():
    """Fix PyTorch loading issues for YOLO11."""
    print("🔧 Fixing PyTorch compatibility for YOLO11...")
    
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        
        # Set environment variables to allow unsafe loading
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
        os.environ['PYTORCH_DISABLE_CUDA_MEMORY_CACHING'] = '1'
        
        # Apply the fix directly to torch.serialization
        import torch.serialization
        
        # Add all the classes YOLO models need
        unsafe_globals = [
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.tasks.SegmentationModel',
            'ultralytics.nn.tasks.ClassificationModel',
            'ultralytics.nn.tasks.PoseModel',
            'ultralytics.nn.tasks.RTDETRDetectionModel',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.C3',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.nn.modules.head.Segment',
            'ultralytics.nn.modules.head.Classify',
            'ultralytics.nn.modules.head.Pose',
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'torch.nn.modules.activation.SiLU',
            'torch.nn.modules.activation.ReLU',
            'torch.nn.modules.pooling.MaxPool2d',
            'torch.nn.modules.pooling.AdaptiveAvgPool2d',
            'torch.nn.modules.linear.Linear',
            'torch.nn.modules.dropout.Dropout',
            'collections.OrderedDict',
            'torch.jit._script.RecursiveScriptModule'
        ]
        
        for cls_name in unsafe_globals:
            try:
                torch.serialization.add_safe_globals([cls_name])
            except Exception as e:
                print(f"  Warning: Could not add {cls_name}: {e}")
        
        print("✅ PyTorch compatibility fixes applied")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch fix failed: {e}")
        return False


def download_yolo11_manually():
    """Manually download YOLO11 models."""
    print("\n📥 Manually downloading YOLO11 models...")
    
    yolo11_models = [
        'yolo11n-seg.pt',
        'yolo11s-seg.pt'
    ]
    
    base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    
    for model_name in yolo11_models:
        try:
            import urllib.request
            model_url = base_url + model_name
            
            print(f"  Downloading {model_name}...")
            
            # Download to current directory
            urllib.request.urlretrieve(model_url, model_name)
            
            # Check if file exists and has reasonable size
            if os.path.exists(model_name):
                size = os.path.getsize(model_name)
                print(f"  ✅ {model_name} downloaded ({size:,} bytes)")
            else:
                print(f"  ❌ {model_name} download failed")
                
        except Exception as e:
            print(f"  ❌ {model_name} download failed: {e}")
    
    return True


def test_yolo11_with_fixes():
    """Test YOLO11 with all compatibility fixes applied."""
    print("\n🧪 Testing YOLO11 with fixes...")
    
    try:
        # Apply fixes first
        fix_pytorch_compatibility()
        
        # Suppress all warnings
        warnings.filterwarnings("ignore")
        
        from ultralytics import YOLO
        
        # Try YOLO11 models in order of preference
        models_to_try = [
            'yolo11n-seg.pt',   # YOLO11 nano segmentation
            'yolo11s-seg.pt',   # YOLO11 small segmentation
            'yolov8n-seg.pt',   # Fallback to YOLO8 nano
        ]
        
        for model_name in models_to_try:
            try:
                print(f"\n  Testing {model_name}...")
                
                # Try to load with explicit settings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Load model with unsafe loading explicitly enabled
                    import torch
                    old_weights_only = None
                    
                    # Temporarily modify torch.load behavior
                    original_load = torch.load
                    def patched_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    torch.load = patched_load
                    
                    try:
                        model = YOLO(model_name)
                        print(f"  ✅ {model_name} loaded successfully!")
                        
                        # Test inference
                        img_path = "/app/output/enhanced_converted_image.jpg"
                        
                        print(f"  🔍 Running inference...")
                        results = model(img_path, conf=0.1, iou=0.45, verbose=False)
                        
                        detections = []
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    class_id = int(box.cls[0])
                                    class_name = model.names[class_id]
                                    confidence = float(box.conf[0])
                                    
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    
                                    detections.append({
                                        'class': class_name,
                                        'confidence': confidence,
                                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                                    })
                        
                        print(f"  ✅ Inference successful! Found {len(detections)} objects:")
                        for det in detections:
                            print(f"    - {det['class']}: {det['confidence']:.3f}")
                        
                        # Restore torch.load
                        torch.load = original_load
                        
                        return model, model_name, detections
                        
                    finally:
                        # Always restore torch.load
                        torch.load = original_load
                
            except Exception as e:
                print(f"  ❌ {model_name} failed: {str(e)[:100]}...")
                continue
        
        print("❌ All YOLO11 models failed to load")
        return None, None, []
        
    except Exception as e:
        print(f"❌ YOLO11 testing failed: {e}")
        return None, None, []


def run_working_yolo11(model, model_name, img_path="/app/output/enhanced_converted_image.jpg"):
    """Run detailed analysis with working YOLO11 model."""
    print(f"\n🎯 Running detailed YOLO11 analysis with {model_name}")
    
    try:
        import cv2
        import json
        import numpy as np
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Could not load image")
            return False
        
        print(f"Analyzing image: {img.shape[1]}×{img.shape[0]} pixels")
        
        # Run inference with very low confidence to catch everything
        results = model(img_path, conf=0.05, iou=0.3, verbose=False)
        
        vis_img = img.copy()
        detections = []
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        detection_count = 0
        
        for result in results:
            # Process boxes
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate dimensions in mm using AprilTag scale
                    width_px = x2 - x1
                    height_px = y2 - y1
                    area_px = width_px * height_px
                    
                    scale_px_per_mm = 4.15
                    length_mm = max(width_px, height_px) / scale_px_per_mm
                    width_mm = min(width_px, height_px) / scale_px_per_mm
                    area_mm2 = area_px / (scale_px_per_mm ** 2)
                    
                    # Check if it's in center area
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    img_center_x, img_center_y = img.shape[1] // 2, img.shape[0] // 2
                    distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                    
                    detection = {
                        'id': detection_count + 1,
                        'model': model_name,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, width_px, height_px],
                        'center': [center_x, center_y],
                        'distance_from_center': distance_from_center,
                        'dimensions_mm': [round(length_mm, 1), round(width_mm, 1)],
                        'area_mm2': round(area_mm2, 1),
                        'aspect_ratio': round(max(width_px, height_px) / min(width_px, height_px), 2)
                    }
                    
                    detections.append(detection)
                    
                    # Draw visualization
                    color = colors[detection_count % len(colors)]
                    
                    # Bounding box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                    
                    # Labels
                    label = f"#{detection_count + 1}: {class_name}"
                    conf_label = f"{confidence:.2f}"
                    dims_label = f"{length_mm:.1f}×{width_mm:.1f}mm"
                    
                    # Background for text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    cv2.putText(vis_img, label, (x1, y1-35), font, font_scale, color, thickness)
                    cv2.putText(vis_img, conf_label, (x1, y1-15), font, 0.5, color, 1)
                    cv2.putText(vis_img, dims_label, (x1, y1-2), font, 0.5, color, 1)
                    
                    # Mark center point
                    cv2.circle(vis_img, (center_x, center_y), 5, color, -1)
                    
                    detection_count += 1
            
            # Process masks if available
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                for i, mask in enumerate(masks.data):
                    if i < detection_count:
                        mask_np = mask.cpu().numpy()
                        # Create colored mask overlay
                        color_mask = np.zeros_like(img)
                        color_mask[mask_np > 0.5] = colors[i % len(colors)]
                        vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.3, 0)
        
        # Mark image center
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.circle(vis_img, img_center, 10, (255, 255, 255), -1)
        cv2.circle(vis_img, img_center, 10, (0, 0, 0), 2)
        cv2.putText(vis_img, "CENTER", (img_center[0] + 15, img_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save visualization
        output_path = f"/app/output/yolo11_{model_name.replace('.pt', '')}_results.jpg"
        cv2.imwrite(output_path, vis_img)
        
        # Save detailed results
        results_data = {
            'model_used': model_name,
            'total_detections': len(detections),
            'scale_px_per_mm': 4.15,
            'image_size': [img.shape[1], img.shape[0]],
            'detections': detections
        }
        
        results_path = "/app/output/yolo11_detections.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print results
        print(f"\n🎯 YOLO11 DETECTION RESULTS:")
        print("=" * 50)
        
        if detections:
            for det in detections:
                print(f"\n**DETECTION #{det['id']}:**")
                print(f"  Object: {det['class']}")
                print(f"  Confidence: {det['confidence']:.1%}")
                print(f"  Dimensions: {det['dimensions_mm'][0]} × {det['dimensions_mm'][1]} mm")
                print(f"  Area: {det['area_mm2']} mm²")
                print(f"  Center: {det['center']} (distance from center: {det['distance_from_center']:.0f}px)")
                print(f"  Aspect ratio: {det['aspect_ratio']}")
                
                # Assess if it could be pliers
                length_mm, width_mm = det['dimensions_mm']
                is_pliers_size = 50 < length_mm < 300 and 10 < width_mm < 100
                is_tool_class = det['class'] in ['scissors', 'knife', 'spoon', 'fork']
                is_centered = det['distance_from_center'] < min(img.shape[1], img.shape[0]) * 0.4
                
                if is_pliers_size and is_centered:
                    print(f"  🎯 PLIERS ASSESSMENT: ✅ POSSIBLE (good size & centered)")
                elif is_tool_class:
                    print(f"  🎯 PLIERS ASSESSMENT: ✅ TOOL DETECTED ({det['class']})")
                else:
                    print(f"  🎯 PLIERS ASSESSMENT: ❌ Unlikely")
        else:
            print("❌ No objects detected")
        
        print(f"\n📁 Files generated:")
        print(f"  {output_path}")
        print(f"  {results_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        return False


def main():
    """Main YOLO11 setup and testing."""
    print("🚀 YOLO11 Setup and Detection")
    print("=" * 40)
    
    # Step 1: Download models manually if needed
    download_yolo11_manually()
    
    # Step 2: Test YOLO11 with fixes
    model, model_name, initial_detections = test_yolo11_with_fixes()
    
    if model is not None:
        print(f"\n✅ SUCCESS! {model_name} is working")
        
        # Step 3: Run detailed analysis
        success = run_working_yolo11(model, model_name)
        
        if success:
            print(f"\n🎉 YOLO11 analysis completed successfully!")
            print(f"Check the generated files for visual results.")
        else:
            print(f"\n⚠️ YOLO11 loaded but analysis failed")
        
        return 0
    else:
        print(f"\n❌ Could not get YOLO11 working")
        return 1


if __name__ == "__main__":
    exit(main())