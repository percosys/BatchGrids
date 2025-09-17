import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json

from batchgrids.vision.apriltag_detector import AprilTagDetector, CalibrationResult
from batchgrids.vision.yolo_segmenter import YOLOSegmenter, YOLOPrediction
from batchgrids.vision.ruler_calibrator import RulerCalibrator
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
    # Ruler fallback info
    ruler_px_per_mm: Optional[float] = None
    ruler_roi_bbox: Optional[Tuple[int, int, int, int]] = None


class ImageProcessor:
    """Main image processing pipeline coordinator."""
    
    def __init__(self):
        self.apriltag_detector = AprilTagDetector()
        self.yolo_segmenter = YOLOSegmenter()
        self.ruler_calibrator = RulerCalibrator()
    
    def process_image(self, image: np.ndarray, px_per_mm_override: Optional[float] = None) -> ProcessingResult:
        """Complete image processing pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Calibrate with AprilTags (unless manual override provided)
            logger.info("Starting calibration")
            if px_per_mm_override and px_per_mm_override > 0:
                # Synthesize a minimal calibration result; no rectification
                calibration = CalibrationResult(
                    px_per_mm=px_per_mm_override,
                    homography_matrix=np.eye(3),
                    rectified_corners={},
                    detected_tags=[],
                    is_valid=False,
                    error_message="Manual scale override"
                )
            else:
                logger.info("Starting AprilTag calibration")
                calibration = self.apriltag_detector.calibrate_image(image)
            px_per_mm = calibration.px_per_mm
            rectified_image = image

            # Defaults for optional ruler info
            ruler_px_per_mm = None
            ruler_roi_bbox = None
            
            # Initialize review flags
            needs_review = False
            review_reason = None

            manual_override = bool(px_per_mm_override and px_per_mm_override > 0)
            if not calibration.is_valid and not manual_override:
                # Try ruler-based fallback for scale
                px_per_mm = 1.0
                needs_review = True
                review_reason = f"Calibration failed: {calibration.error_message} (using px units)"
                # Explicitly skip ruler fallback (reverted to prior behavior)
            else:
                # Only rectify if we have a valid calibration (not manual override)
                if calibration.is_valid:
                    logger.info(f"Calibration successful: {calibration.px_per_mm:.2f} px/mm")
                    h, w = image.shape[:2]
                    rectified_image = self.apriltag_detector.rectify_image(
                        image, calibration.homography_matrix, (w, h)
                    )
                else:
                    # Manual scale override: do not rectify, keep original image coordinates
                    rectified_image = image
            
            # Step 3: Segment with YOLO
            logger.info("Running YOLO segmentation")
            yolo_prediction = self.yolo_segmenter.segment_image(rectified_image)
            
            # Step 4: Check if needs review
            # Determine if review is needed (retain calibration fail reason if set)
            if not calibration.is_valid:
                # Keep the review flags set above
                pass
            else:
                needs_review, review_reason = self.yolo_segmenter.needs_review(yolo_prediction)
            
            # Step 5: Extract tool information (if we have a good prediction)
            tool_outline = None
            tool_dimensions = None
            svg_outline = None
            
            if yolo_prediction.top_prediction:
                # Extract tool outline and dimensions
                tool_data = self.yolo_segmenter.extract_tool_outline(
                    yolo_prediction.top_prediction.mask,
                    px_per_mm
                )
                
                tool_outline = tool_data["outline"]
                tool_dimensions = tool_data["dimensions"]
                
                # Create SVG outline
                svg_outline = self.yolo_segmenter.create_svg_outline(
                    tool_outline,
                    tool_dimensions,
                    px_per_mm
                )
                
                logger.info(f"Tool extracted: {tool_dimensions.get('length_mm', 0):.1f}mm length")
            else:
                # Fallback: classical segmentation if YOLO found nothing
                logger.info("YOLO found no detections; attempting fallback edge-based segmentation")
                fallback_mask = self._segment_by_edges(rectified_image)
                if np.sum(fallback_mask) > 0:
                    tool_data = self.yolo_segmenter.extract_tool_outline(fallback_mask, px_per_mm)
                    tool_outline = tool_data["outline"]
                    tool_dimensions = tool_data["dimensions"]
                    svg_outline = self.yolo_segmenter.create_svg_outline(tool_outline, tool_dimensions, px_per_mm)
                    needs_review = True
                    review_reason = (review_reason + "; " if review_reason else "") + "YOLO missed object; used fallback segmentation"
            
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
                processing_time_ms=processing_time_ms,
                ruler_px_per_mm=(px_per_mm if not calibration.is_valid else None),
                ruler_roi_bbox=(ruler_roi_bbox if not calibration.is_valid else None)
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

    def _segment_by_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhanced background-agnostic segmentation optimized for tool detection."""
        if image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        img = image.copy()
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Light blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # More sensitive edge detection for tools
        edges = cv2.Canny(gray, 30, 100)  # Lowered thresholds
        
        # Larger kernel and more iterations for better tool shape recovery
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges = cv2.dilate(edges, kernel, iterations=2)
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray, dtype=np.uint8)
        
        # Filter out very small contours
        min_area = gray.shape[0] * gray.shape[1] * 0.001  # 0.1% of image area
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not contours:
            return np.zeros_like(gray, dtype=np.uint8)
            
        # Get the largest contour (most likely the tool)
        main = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [main], -1, 1, thickness=cv2.FILLED)
        
        # Final morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        
        return mask.astype(np.uint8)
    
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
