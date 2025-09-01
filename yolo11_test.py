#!/usr/bin/env python3
"""Test YOLO11 for better tool detection."""

import subprocess
import sys
import os
import json

def install_yolo11():
    """Install YOLO11 (YOLOv11) from ultralytics."""
    print("🔄 Installing YOLO11...")
    
    try:
        # Update ultralytics to get YOLO11
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'ultralytics'
        ], capture_output=True, text=True)
        
        print("Installation output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def test_yolo11():
    """Test YOLO11 detection capabilities."""
    try:
        print("\n🤖 Testing YOLO11...")
        from ultralytics import YOLO
        
        # Try YOLO11 models
        yolo11_models = [
            'yolo11n-seg.pt',  # YOLO11 nano segmentation
            'yolo11s-seg.pt',  # YOLO11 small segmentation  
            'yolo11m-seg.pt',  # YOLO11 medium segmentation
        ]
        
        for model_name in yolo11_models:
            try:
                print(f"\nTrying {model_name}...")
                
                # Load model with explicit weights_only=False
                model = YOLO(model_name)
                print(f"✅ {model_name} loaded successfully!")
                
                # Test on image
                img_path = "/app/output/enhanced_converted_image.jpg"
                
                print(f"Running inference with {model_name}...")
                results = model(img_path, conf=0.1, iou=0.45, verbose=False)
                
                detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detection = {
                                'model': model_name,
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            }
                            detections.append(detection)
                
                print(f"✅ {model_name} found {len(detections)} objects!")
                
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.3f}")
                
                # If we found detections, use this model
                if detections:
                    return model, model_name, detections
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        print("❌ No YOLO11 models worked")
        return None, None, []
        
    except Exception as e:
        print(f"YOLO11 test failed: {e}")
        return None, None, []


def compare_yolo_versions():
    """Compare YOLO versions available."""
    try:
        from ultralytics import YOLO
        print("\n📊 Available YOLO Models:")
        
        # List of models to try
        models_to_test = [
            'yolov8n-seg.pt',
            'yolov8s-seg.pt', 
            'yolo11n-seg.pt',
            'yolo11s-seg.pt'
        ]
        
        working_models = []
        
        for model_name in models_to_test:
            try:
                print(f"\nTesting {model_name}...")
                
                # Quick load test
                model = YOLO(model_name)
                print(f"  ✅ {model_name}: Loading successful")
                
                # Quick inference test
                test_img = "/app/output/enhanced_converted_image.jpg"
                results = model(test_img, conf=0.25, verbose=False)
                
                detection_count = 0
                if results and len(results) > 0:
                    if results[0].boxes is not None:
                        detection_count = len(results[0].boxes)
                
                print(f"  ✅ {model_name}: {detection_count} detections")
                
                working_models.append({
                    'model': model_name,
                    'detections': detection_count,
                    'status': 'working'
                })
                
            except Exception as e:
                print(f"  ❌ {model_name}: {str(e)[:100]}...")
                working_models.append({
                    'model': model_name,
                    'detections': 0,
                    'status': f'failed: {str(e)[:50]}...'
                })
        
        return working_models
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        return []


def run_best_yolo_model():
    """Run the best available YOLO model."""
    print("\n🎯 Finding Best YOLO Model...")
    
    working_models = compare_yolo_versions()
    
    if not working_models:
        print("❌ No YOLO models available")
        return False
    
    # Find best working model (most detections)
    working_only = [m for m in working_models if m['status'] == 'working']
    
    if not working_only:
        print("❌ No working YOLO models found")
        print("\nModel Status:")
        for model in working_models:
            print(f"  {model['model']}: {model['status']}")
        return False
    
    # Sort by detection count
    best_model = max(working_only, key=lambda x: x['detections'])
    
    print(f"\n🏆 Best Model: {best_model['model']} ({best_model['detections']} detections)")
    
    # Run detailed analysis with best model
    try:
        from ultralytics import YOLO
        import cv2
        
        model = YOLO(best_model['model'])
        img_path = "/app/output/enhanced_converted_image.jpg"
        
        print(f"\nRunning detailed analysis with {best_model['model']}...")
        results = model(img_path, conf=0.1, iou=0.45, verbose=False)
        
        # Analyze results in detail
        img = cv2.imread(img_path)
        vis_img = img.copy()
        
        detections = []
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate dimensions
                    width_px = x2 - x1
                    height_px = y2 - y1
                    area_px = width_px * height_px
                    
                    # Convert to mm using AprilTag scale
                    scale_px_per_mm = 4.15
                    length_mm = max(width_px, height_px) / scale_px_per_mm
                    width_mm = min(width_px, height_px) / scale_px_per_mm
                    area_mm2 = area_px / (scale_px_per_mm ** 2)
                    
                    detection = {
                        'model_used': best_model['model'],
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, width_px, height_px],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'dimensions_mm': [length_mm, width_mm],
                        'area_mm2': area_mm2,
                        'aspect_ratio': max(width_px, height_px) / min(width_px, height_px)
                    }
                    detections.append(detection)
                    
                    # Draw on visualization
                    color = colors[i % len(colors)]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{class_name} ({confidence:.2f})"
                    dims = f"{length_mm:.1f}x{width_mm:.1f}mm"
                    
                    cv2.putText(vis_img, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(vis_img, dims, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        output_path = f"/app/output/{best_model['model'].replace('.pt', '')}_detection.jpg"
        cv2.imwrite(output_path, vis_img)
        
        # Save detailed results
        results_data = {
            'model_used': best_model['model'],
            'total_detections': len(detections),
            'detections': detections,
            'scale_px_per_mm': 4.15
        }
        
        with open('/app/output/best_yolo_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print results
        print(f"\n🎯 BEST YOLO RESULTS ({best_model['model']}):")
        print("=" * 50)
        
        for i, det in enumerate(detections):
            print(f"\n**DETECTION #{i+1}:**")
            print(f"  Object: {det['class']}")
            print(f"  Confidence: {det['confidence']:.1%}")
            print(f"  Dimensions: {det['dimensions_mm'][0]:.1f} × {det['dimensions_mm'][1]:.1f} mm")
            print(f"  Area: {det['area_mm2']:.1f} mm²")
            print(f"  Center: {det['center']}")
            print(f"  Aspect ratio: {det['aspect_ratio']:.2f}")
        
        print(f"\n📁 Files generated:")
        print(f"  {output_path}")
        print(f"  best_yolo_results.json")
        
        return True
        
    except Exception as e:
        print(f"Detailed analysis failed: {e}")
        return False


def main():
    """Main YOLO11 testing function."""
    print("🔬 YOLO11 vs YOLO8 Comparison")
    print("=" * 40)
    
    # First try with current installation
    success = run_best_yolo_model()
    
    if not success:
        print("\nTrying to upgrade to YOLO11...")
        if install_yolo11():
            print("✅ YOLO11 installation completed")
            success = run_best_yolo_model()
        else:
            print("❌ YOLO11 installation failed")
    
    if success:
        print("\n✅ YOLO model testing completed successfully!")
    else:
        print("\n❌ YOLO model testing failed")
        print("\nFallback: The ultra-sensitive custom detection we used")
        print("found multiple pliers candidates successfully.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())