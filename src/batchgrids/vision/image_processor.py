import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json

from batchgrids.vision.apriltag_detector import AprilTagDetector, CalibrationResult
from batchgrids.vision.yolo_segmenter import YOLOSegmenter, YOLOPrediction
from batchgrids.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete image processing result."""
    # Calibration
    calibration: CalibrationResult
    rectified_image: Optional[np.ndarray] = None
    
    # Segmentation
    yolo_prediction: Optional[YOLOPrediction] = None
    
    # Tool extraction
    tool_outline: Optional[Dict] = None
    tool_dimensions: Optional[Dict] = None
    svg_outline: Optional[str] = None
    
    # Status
    success: bool = False
    error_message: Optional[str] = None
    needs_review: bool = False
    review_reason: Optional[str] = None
    processing_time_ms: float = 0.0


class ImageProcessor:
    """Main image processing pipeline coordinator."""
    
    def __init__(self):
        self.apriltag_detector = AprilTagDetector()
        self.yolo_segmenter = YOLOSegmenter()
    
    def process_image(self, image: np.ndarray) -> ProcessingResult:
        """Complete image processing pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Calibrate with AprilTags
            logger.info("Starting AprilTag calibration")
            calibration = self.apriltag_detector.calibrate_image(image)
            
            if not calibration.is_valid:
                return ProcessingResult(
                    calibration=calibration,
                    success=False,
                    error_message=f"Calibration failed: {calibration.error_message}",
                    processing_time_ms=self._get_processing_time_ms(start_time)
                )
            
            logger.info(f"Calibration successful: {calibration.px_per_mm:.2f} px/mm")
            
            # Step 2: Rectify image
            rectified_image = self.apriltag_detector.rectify_image(
                image, calibration.homography_matrix
            )
            
            # Step 3: Segment with YOLO
            logger.info("Running YOLO segmentation")
            yolo_prediction = self.yolo_segmenter.segment_image(rectified_image)
            
            # Step 4: Check if needs review
            needs_review, review_reason = self.yolo_segmenter.needs_review(yolo_prediction)
            
            # Step 5: Extract tool information (if we have a good prediction)
            tool_outline = None
            tool_dimensions = None
            svg_outline = None
            
            if yolo_prediction.top_prediction:
                # Extract tool outline and dimensions
                tool_data = self.yolo_segmenter.extract_tool_outline(
                    yolo_prediction.top_prediction.mask,
                    calibration.px_per_mm
                )
                
                tool_outline = tool_data["outline"]
                tool_dimensions = tool_data["dimensions"]
                
                # Create SVG outline
                svg_outline = self.yolo_segmenter.create_svg_outline(
                    tool_outline,
                    tool_dimensions,
                    calibration.px_per_mm
                )
                
                logger.info(f"Tool extracted: {tool_dimensions.get('length_mm', 0):.1f}mm length")
            
            processing_time_ms = self._get_processing_time_ms(start_time)
            
            return ProcessingResult(
                calibration=calibration,
                rectified_image=rectified_image,
                yolo_prediction=yolo_prediction,
                tool_outline=tool_outline,
                tool_dimensions=tool_dimensions,
                svg_outline=svg_outline,
                success=True,
                needs_review=needs_review,
                review_reason=review_reason,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                calibration=CalibrationResult(
                    px_per_mm=0.0,
                    homography_matrix=np.eye(3),
                    rectified_corners={},
                    detected_tags=[],
                    is_valid=False,
                    error_message="Processing failed"
                ),
                success=False,
                error_message=str(e),
                processing_time_ms=self._get_processing_time_ms(start_time)
            )
    
    def _get_processing_time_ms(self, start_time: datetime) -> float:
        """Calculate processing time in milliseconds."""
        return (datetime.now() - start_time).total_seconds() * 1000
    
    def create_debug_visualization(self, image: np.ndarray, result: ProcessingResult) -> np.ndarray:
        """Create debug visualization showing all processing steps."""
        if not result.success:
            # Just show original with error text
            debug_image = image.copy()
            cv2.putText(debug_image, f"ERROR: {result.error_message}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return debug_image
        
        # Create composite visualization
        h, w = image.shape[:2]
        
        # Create 2x2 grid
        debug_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Original with AprilTags
        original_with_tags = self.apriltag_detector.draw_detections(
            image, result.calibration.detected_tags
        )
        debug_image[:h, :w] = original_with_tags
        
        # Top-right: Rectified image
        if result.rectified_image is not None:
            rectified_resized = cv2.resize(result.rectified_image, (w, h))
            if len(rectified_resized.shape) == 2:
                rectified_resized = cv2.cvtColor(rectified_resized, cv2.COLOR_GRAY2BGR)
            debug_image[:h, w:] = rectified_resized
        
        # Bottom-left: YOLO segmentation
        if result.yolo_prediction and result.rectified_image is not None:
            yolo_vis = self.yolo_segmenter.visualize_segmentation(
                result.rectified_image, result.yolo_prediction
            )
            yolo_vis_resized = cv2.resize(yolo_vis, (w, h))
            if len(yolo_vis_resized.shape) == 2:
                yolo_vis_resized = cv2.cvtColor(yolo_vis_resized, cv2.COLOR_GRAY2BGR)
            debug_image[h:, :w] = yolo_vis_resized
        
        # Bottom-right: Tool outline
        if result.tool_outline and result.rectified_image is not None:
            outline_vis = result.rectified_image.copy()
            if len(outline_vis.shape) == 2:
                outline_vis = cv2.cvtColor(outline_vis, cv2.COLOR_GRAY2BGR)
            
            # Draw outline
            if result.tool_outline:
                outline_points = np.array(result.tool_outline, dtype=np.int32)
                cv2.polylines(outline_vis, [outline_points], True, (0, 255, 0), 3)
            
            outline_vis_resized = cv2.resize(outline_vis, (w, h))
            debug_image[h:, w:] = outline_vis_resized
        
        # Add labels
        cv2.putText(debug_image, "Original + AprilTags", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "Rectified", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "YOLO Segmentation", (10, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "Tool Outline", (w + 10, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add processing info
        info_text = f"Scale: {result.calibration.px_per_mm:.2f} px/mm"
        if result.yolo_prediction and result.yolo_prediction.top_prediction:
            pred = result.yolo_prediction.top_prediction
            info_text += f" | {pred.class_name}: {pred.confidence:.3f}"
        if result.tool_dimensions:
            dims = result.tool_dimensions
            info_text += f" | {dims.get('length_mm', 0):.1f}x{dims.get('width_mm', 0):.1f}mm"
        
        cv2.putText(debug_image, info_text, (10, h * 2 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return debug_image
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize if too large
        h, w = image.shape[:2]
        max_size = settings.max_image_size
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        
        return image
    
    def prepare_metadata(self, result: ProcessingResult) -> Dict:
        """Prepare metadata for database storage."""
        metadata = {
            "calibration": {
                "px_per_mm": result.calibration.px_per_mm,
                "is_valid": result.calibration.is_valid,
                "detected_tags": [
                    {
                        "tag_id": tag.tag_id,
                        "center": tag.center,
                        "corners": tag.corners.tolist()
                    }
                    for tag in result.calibration.detected_tags
                ],
                "homography": result.calibration.homography_matrix.tolist() if result.calibration.is_valid else None
            },
            "segmentation": None,
            "tool": None,
            "processing": {
                "success": result.success,
                "needs_review": result.needs_review,
                "review_reason": result.review_reason,
                "processing_time_ms": result.processing_time_ms
            }
        }
        
        if result.yolo_prediction:
            metadata["segmentation"] = {
                "model_version": result.yolo_prediction.model_version,
                "processing_time_ms": result.yolo_prediction.processing_time_ms,
                "detections": [
                    {
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                        "area_pixels": det.area_pixels
                    }
                    for det in result.yolo_prediction.detections
                ]
            }
        
        if result.tool_dimensions:
            metadata["tool"] = {
                "dimensions": result.tool_dimensions,
                "outline_points": len(result.tool_outline) if result.tool_outline else 0
            }
        
        return metadata