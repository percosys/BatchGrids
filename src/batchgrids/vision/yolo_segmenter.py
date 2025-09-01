import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from ultralytics import YOLO
from shapely.geometry import Polygon
import json

from batchgrids.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """YOLO segmentation result for a single detection."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Binary mask
    polygon: List[Tuple[float, float]]  # Polygon coordinates
    area_pixels: float


@dataclass
class YOLOPrediction:
    """Complete YOLO prediction result."""
    detections: List[SegmentationResult]
    top_prediction: Optional[SegmentationResult]
    processing_time_ms: float
    model_version: str
    image_shape: Tuple[int, int]  # height, width


class YOLOSegmenter:
    """YOLO segmentation for tool detection and classification."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = settings.yolo_confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            # Check if model file exists, download if not
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.info(f"Downloading YOLO model: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def segment_image(self, image: np.ndarray, confidence_threshold: float = None) -> YOLOPrediction:
        """Segment image and detect tools."""
        if self.model is None:
            raise ValueError("YOLO model not loaded")
        
        confidence = confidence_threshold or self.confidence_threshold
        
        import time
        start_time = time.time()
        
        # Run inference
        results = self.model(image, conf=confidence, verbose=False)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Process results
        detections = []
        
        if len(results) > 0:
            result = results[0]  # Single image
            
            if result.masks is not None and len(result.masks) > 0:
                # Get class names
                class_names = result.names
                
                # Process each detection
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                masks = result.masks.data.cpu().numpy()
                
                for i, (box, conf, class_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks)):
                    # Convert mask to correct size and format
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Extract polygon from mask
                    polygon = self._mask_to_polygon(binary_mask)
                    
                    # Calculate area
                    area_pixels = np.sum(binary_mask)
                    
                    detection = SegmentationResult(
                        class_name=class_names[class_id],
                        confidence=float(conf),
                        bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        mask=binary_mask,
                        polygon=polygon,
                        area_pixels=float(area_pixels)
                    )
                    detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        top_prediction = detections[0] if detections else None
        
        return YOLOPrediction(
            detections=detections,
            top_prediction=top_prediction,
            processing_time_ms=processing_time_ms,
            model_version=self.model_path,
            image_shape=(image.shape[0], image.shape[1])
        )
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Convert binary mask to polygon coordinates."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce points
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of tuples
        polygon = [(float(point[0][0]), float(point[0][1])) for point in simplified]
        
        return polygon
    
    def extract_tool_outline(self, mask: np.ndarray, px_per_mm: float) -> Dict:
        """Extract tool outline and dimensions from mask."""
        if np.sum(mask) == 0:
            return {"outline": [], "dimensions": {}, "area_mm2": 0.0}
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"outline": [], "dimensions": {}, "area_mm2": 0.0}
        
        # Get largest contour (main tool)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour for outline
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)
        outline = [(float(p[0][0]), float(p[0][1])) for p in simplified]
        
        # Calculate dimensions
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Convert to mm
        width_mm = w / px_per_mm
        height_mm = h / px_per_mm
        area_mm2 = cv2.contourArea(main_contour) / (px_per_mm ** 2)
        
        dimensions = {
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),
            "length_mm": float(max(width_mm, height_mm)),  # Length is the longer dimension
            "area_mm2": float(area_mm2),
            "bbox_mm": {
                "x": float(x / px_per_mm),
                "y": float(y / px_per_mm),
                "width": float(width_mm),
                "height": float(height_mm)
            }
        }
        
        return {
            "outline": outline,
            "dimensions": dimensions,
            "area_mm2": float(area_mm2)
        }
    
    def create_svg_outline(self, outline: List[Tuple[float, float]], 
                          dimensions: Dict, px_per_mm: float) -> str:
        """Create SVG representation of tool outline."""
        if not outline:
            return ""
        
        # Convert to mm coordinates
        outline_mm = [(x / px_per_mm, y / px_per_mm) for x, y in outline]
        
        # Find bounds
        min_x = min(p[0] for p in outline_mm)
        min_y = min(p[1] for p in outline_mm)
        max_x = max(p[0] for p in outline_mm)
        max_y = max(p[1] for p in outline_mm)
        
        # Create SVG path
        path_data = f"M {outline_mm[0][0]:.2f} {outline_mm[0][1]:.2f}"
        for point in outline_mm[1:]:
            path_data += f" L {point[0]:.2f} {point[1]:.2f}"
        path_data += " Z"
        
        # SVG template
        svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="{min_x:.2f} {min_y:.2f} {max_x - min_x:.2f} {max_y - min_y:.2f}"
     width="{max_x - min_x:.2f}mm" 
     height="{max_y - min_y:.2f}mm">
  <path d="{path_data}" 
        fill="none" 
        stroke="black" 
        stroke-width="0.1mm"/>
</svg>"""
        return svg
    
    def visualize_segmentation(self, image: np.ndarray, prediction: YOLOPrediction) -> np.ndarray:
        """Visualize segmentation results on image."""
        result = image.copy()
        
        for detection in prediction.detections:
            # Draw mask overlay
            mask = detection.mask
            colored_mask = np.zeros_like(result)
            colored_mask[mask > 0] = [0, 255, 0]  # Green overlay
            result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw polygon outline
            if detection.polygon:
                polygon_points = np.array(detection.polygon, dtype=np.int32)
                cv2.polylines(result, [polygon_points], True, (0, 0, 255), 2)
        
        return result
    
    def needs_review(self, prediction: YOLOPrediction) -> Tuple[bool, str]:
        """Determine if prediction needs human review."""
        if not prediction.top_prediction:
            return True, "No detections found"
        
        if prediction.top_prediction.confidence < self.confidence_threshold:
            return True, f"Low confidence: {prediction.top_prediction.confidence:.3f}"
        
        if len(prediction.detections) > 1:
            # Check if there are multiple high-confidence detections
            high_conf_count = sum(1 for d in prediction.detections if d.confidence > self.confidence_threshold)
            if high_conf_count > 1:
                return True, "Multiple high-confidence detections"
        
        return False, "Prediction is confident"