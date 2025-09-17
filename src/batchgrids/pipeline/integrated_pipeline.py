"""
Integrated BatchGrids Pipeline

Orchestrates the complete pipeline:
1. Preprocessing (scale, object hints)
2. Intelligent YOLOE prompting  
3. Object detection and processing
4. Object storage and layout management

Author: BatchGrids Pipeline
"""

import cv2
import numpy as np
import hashlib
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime

from .preprocessor import ImagePreprocessor, PreprocessingResult
from .intelligent_yoloe import IntelligentYOLOEPrompting, YOLOEStrategy  
from .object_storage import ObjectDatabase, DetectedObject, ObjectTransform, GridfinityParameters, LayoutManager


class IntegratedPipeline:
    """Complete BatchGrids processing pipeline"""
    
    def __init__(self, storage_dir: Path):
        self.preprocessor = ImagePreprocessor()
        self.intelligent_yoloe = IntelligentYOLOEPrompting()
        self.object_db = ObjectDatabase(storage_dir)
        self.layout_manager = LayoutManager(self.object_db)
        
        print(f"🚀 PIPELINE: Initialized with storage at {storage_dir}")
    
    def process_image(self, 
                     image_path: Path,
                     expected_objects: Optional[List[str]] = None,
                     custom_scale_mm_per_px: Optional[float] = None) -> List[str]:
        """
        Process complete image through pipeline
        
        Args:
            image_path: Path to input image
            expected_objects: Optional list of expected object types for better prompting
            custom_scale_mm_per_px: Manual scale override
            
        Returns:
            List of object IDs created
        """
        
        print(f"🎯 PROCESSING: {image_path.name}")
        print("=" * 80)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create image hash for tracking
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        
        # Step 1: Preprocessing
        print("\\n📋 STEP 1: Preprocessing")
        preprocessing_result = self.preprocessor.process_image(image)
        
        # Override scale if provided
        if custom_scale_mm_per_px and preprocessing_result.scale_calibration:
            preprocessing_result.scale_calibration.px_per_mm = custom_scale_mm_per_px
            preprocessing_result.scale_calibration.method = 'manual_override'
            print(f"  🔧 Using manual scale: {custom_scale_mm_per_px:.2f} px/mm")
        
        # Step 2: Generate intelligent YOLOE strategy
        print("\\n🧠 STEP 2: Intelligent YOLOE Strategy")
        yoloe_strategy = self.intelligent_yoloe.generate_strategy(preprocessing_result)
        
        # Incorporate expected objects if provided
        if expected_objects:
            print(f"  🎯 Incorporating expected objects: {expected_objects}")
            yoloe_strategy.vocabulary.extend(expected_objects)
            yoloe_strategy.vocabulary = list(set(yoloe_strategy.vocabulary))  # Remove duplicates
        
        # Step 3: YOLOE Detection
        print("\\n🔍 STEP 3: YOLOE Object Detection")
        detection_results = self._run_yoloe_detection(
            preprocessing_result.enhanced_image, 
            yoloe_strategy
        )
        
        # Step 4: Refine strategy if needed
        if not detection_results or all(r.get('confidence', 0) < 0.1 for r in detection_results):
            print("\\n🔄 STEP 4: Refining Strategy")
            yoloe_strategy = self.intelligent_yoloe.refine_detection(
                yoloe_strategy, detection_results, preprocessing_result
            )
            detection_results = self._run_yoloe_detection(
                preprocessing_result.enhanced_image, 
                yoloe_strategy
            )
        
        # Step 5: Process and store objects
        print("\\n💾 STEP 5: Object Processing & Storage")
        object_ids = []
        
        for i, detection in enumerate(detection_results):
            try:
                processed_object = self._process_detection_to_object(
                    detection,
                    preprocessing_result,
                    image_hash,
                    f"detection_{i}"
                )
                
                if processed_object:
                    object_id = self.object_db.add_object(processed_object)
                    object_ids.append(object_id)
                    
            except Exception as e:
                print(f"  ⚠️ Failed to process detection {i}: {e}")
                continue
        
        print(f"\\n✅ COMPLETED: Processed {len(object_ids)} objects")
        print("=" * 80)
        
        return object_ids
    
    def create_multi_object_layout(self, 
                                   object_ids: List[str], 
                                   layout_name: str) -> str:
        """Create optimized layout for multiple objects"""
        
        print(f"🏗️ LAYOUT: Creating '{layout_name}' with {len(object_ids)} objects")
        
        layout = self.layout_manager.create_layout(layout_name, object_ids)
        
        print(f"  📐 Grid size: {layout.grid_size[0]} × {layout.grid_size[1]} units")
        print(f"  ⏱️ Est. print time: {layout.estimated_print_time_hours:.1f} hours")
        
        return layout.layout_id
    
    def rotate_object(self, object_id: str, rotation_degrees: float) -> bool:
        """Rotate object for better layout"""
        
        obj = self.object_db.get_object(object_id)
        if not obj:
            return False
        
        # Update transform
        obj.transform.rotation_degrees = rotation_degrees
        return self.object_db.update_object_transform(object_id, obj.transform)
    
    def export_gridfinity_svg(self, 
                             object_ids: List[str], 
                             output_path: Path,
                             include_layout: bool = True) -> bool:
        """Export gridfinity SVG for 3D printing"""
        
        print(f"📐 EXPORT: Generating gridfinity SVG for {len(object_ids)} objects")
        
        try:
            # Get objects
            objects = [self.object_db.get_object(oid) for oid in object_ids]
            objects = [obj for obj in objects if obj is not None]
            
            if not objects:
                print("  ❌ No valid objects to export")
                return False
            
            # Generate SVG content
            svg_content = self._generate_gridfinity_svg(objects, include_layout)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(svg_content)
            
            print(f"  ✅ Exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ Export failed: {e}")
            return False
    
    def _run_yoloe_detection(self, 
                            image: np.ndarray, 
                            strategy: YOLOEStrategy) -> List[Dict]:
        """Run YOLOE detection with given strategy"""
        
        # This would integrate with the actual YOLOE detection
        # For now, return mock results to show the architecture
        
        print(f"  🔍 Running YOLOE with {len(strategy.vocabulary)} terms")
        print(f"  🎯 Confidence threshold: {strategy.confidence_threshold:.3f}")
        
        # TODO: Implement actual YOLOE detection here
        # This would call your existing YOLOE detection code
        # but with the intelligent vocabulary and parameters
        
        # Mock detection result for architecture demonstration
        h, w = image.shape[:2]
        mock_detection = {
            'class': 'knife',
            'confidence': 0.347,
            'bbox': [540, 2112, 3311, 4143],
            'mask': np.ones((h, w), dtype=np.uint8) * 255,  # Mock mask
            'area': 1308911
        }
        
        return [mock_detection]
    
    def _process_detection_to_object(self, 
                                    detection: Dict,
                                    preprocessing_result: PreprocessingResult,
                                    image_hash: str,
                                    detection_id: str) -> Optional[DetectedObject]:
        """Convert YOLOE detection to stored object"""
        
        try:
            # Extract detection data
            bbox = detection['bbox']
            mask = detection['mask']
            
            # Calculate contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(main_contour, True)
            simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # Calculate dimensions
            scale = preprocessing_result.scale_calibration
            if scale:
                px_per_mm = scale.px_per_mm
            else:
                px_per_mm = 4.15  # Fallback
            
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            width_mm = bbox_width / px_per_mm
            height_mm = bbox_height / px_per_mm
            
            area_mm2 = detection['area'] / (px_per_mm ** 2)
            perimeter_mm = cv2.arcLength(main_contour, True) / px_per_mm
            
            # Create object
            obj = DetectedObject(
                object_id=\"\",  # Will be assigned by database
                object_type=detection['class'],
                confidence=detection['confidence'],
                bbox=tuple(bbox),
                mask=mask,
                contour=simplified_contour,
                dimensions_mm=(width_mm, height_mm),
                area_mm2=area_mm2,
                perimeter_mm=perimeter_mm,
                detection_method='yoloe_intelligent',
                processing_steps=[
                    'preprocessing',
                    'intelligent_prompting', 
                    'yoloe_detection',
                    'contour_simplification'
                ],
                quality_score=min(1.0, detection['confidence'] + 0.3),
                transform=ObjectTransform(),
                gridfinity_params=GridfinityParameters(),
                timestamp=datetime.now().isoformat(),
                source_image_hash=image_hash
            )
            
            print(f\"  ✅ Processed {obj.object_type}: {width_mm:.1f}×{height_mm:.1f}mm\")
            
            return obj
            
        except Exception as e:
            print(f\"  ⚠️ Failed to process detection: {e}\")
            return None
    
    def _generate_gridfinity_svg(self, 
                                objects: List[DetectedObject], 
                                include_layout: bool) -> str:
        \"\"\"Generate SVG content for gridfinity pocket(s)\"\"\"
        
        if len(objects) == 1:
            # Single object - use existing logic
            return self._generate_single_object_svg(objects[0])
        
        elif include_layout:
            # Multi-object layout
            return self._generate_layout_svg(objects)
        
        else:
            # Multiple separate SVGs combined
            return self._generate_combined_svg(objects)
    
    def _generate_single_object_svg(self, obj: DetectedObject) -> str:
        \"\"\"Generate SVG for single object\"\"\"
        
        # Convert contour to SVG path
        contour_points = obj.contour.reshape(-1, 2)
        
        # Apply transform
        if obj.transform.rotation_degrees != 0:
            # Apply rotation (simplified)
            angle_rad = np.radians(obj.transform.rotation_degrees)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Rotate around center
            center_x = np.mean(contour_points[:, 0])
            center_y = np.mean(contour_points[:, 1])
            
            # Translate to origin, rotate, translate back
            translated = contour_points - [center_x, center_y]
            rotated = np.array([
                translated[:, 0] * cos_a - translated[:, 1] * sin_a,
                translated[:, 0] * sin_a + translated[:, 1] * cos_a
            ]).T
            contour_points = rotated + [center_x, center_y]
        
        # Generate SVG path
        path_data = f\"M {contour_points[0, 0]:.1f} {contour_points[0, 1]:.1f} \"
        for point in contour_points[1:]:
            path_data += f\"L {point[0]:.1f} {point[1]:.1f} \"
        path_data += \"Z\"
        
        # Calculate viewBox
        min_x = np.min(contour_points[:, 0]) - 10
        min_y = np.min(contour_points[:, 1]) - 10
        max_x = np.max(contour_points[:, 0]) + 10
        max_y = np.max(contour_points[:, 1]) + 10
        
        width_mm = obj.dimensions_mm[0] + obj.gridfinity_params.clearance_mm * 2
        height_mm = obj.dimensions_mm[1] + obj.gridfinity_params.clearance_mm * 2
        
        svg_content = f'''<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<svg width=\"{width_mm:.2f}mm\" height=\"{height_mm:.2f}mm\" 
     viewBox=\"{min_x:.0f} {min_y:.0f} {max_x-min_x:.0f} {max_y-min_y:.0f}\" xmlns=\"http://www.w3.org/2000/svg\">
  <title>Gridfinity Pocket - {obj.object_type}</title>
  <desc>Intelligent pipeline detection with {obj.gridfinity_params.clearance_mm}mm clearance</desc>
  
  <defs>
    <style>
      .gridfinity-pocket {{ fill: #FF6B6B; fill-opacity: 0.3; stroke: #FF6B6B; stroke-width: 2; }}
      .info-text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #333; }}
    </style>
  </defs>
  
  <path d=\"{path_data}\" class=\"gridfinity-pocket\" title=\"Gridfinity pocket\"/>
  
  <text x=\"{min_x + 10}\" y=\"{min_y + 25}\" class=\"info-text\">Tool: {obj.object_type}</text>
  <text x=\"{min_x + 10}\" y=\"{min_y + 40}\" class=\"info-text\">Confidence: {obj.confidence:.1%}</text>
  <text x=\"{min_x + 10}\" y=\"{min_y + 55}\" class=\"info-text\">Dimensions: {obj.dimensions_mm[0]:.1f} × {obj.dimensions_mm[1]:.1f} mm</text>
  <text x=\"{min_x + 10}\" y=\"{min_y + 70}\" class=\"info-text\">Quality: {obj.quality_score:.1%}</text>
  
</svg>'''
        
        return svg_content
    
    def _generate_layout_svg(self, objects: List[DetectedObject]) -> str:
        \"\"\"Generate layout SVG for multiple objects\"\"\"
        
        # Create layout
        layout = self.layout_manager.create_layout(\"temp_layout\", [obj.object_id for obj in objects])
        
        # Generate combined SVG based on layout
        # This would be more complex, showing grid and object placements
        
        return \"<!-- Multi-object layout SVG would go here -->\"
    
    def _generate_combined_svg(self, objects: List[DetectedObject]) -> str:
        \"\"\"Generate combined SVG with separate pockets\"\"\"
        
        # Simple side-by-side arrangement
        total_width = sum(obj.dimensions_mm[0] for obj in objects) + len(objects) * 10
        max_height = max(obj.dimensions_mm[1] for obj in objects) + 20
        
        svg_content = f'''<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<svg width=\"{total_width:.2f}mm\" height=\"{max_height:.2f}mm\" 
     viewBox=\"0 0 {total_width:.0f} {max_height:.0f}\" xmlns=\"http://www.w3.org/2000/svg\">
  <title>Multiple Gridfinity Pockets</title>
  <desc>Intelligent pipeline - multiple objects</desc>
  
  <!-- Individual object pockets would be positioned here -->
  
</svg>'''
        
        return svg_content