import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
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
    
    def __init__(self, model_path: str = None, load_model: bool = True):
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = settings.yolo_confidence_threshold
        # Tunable parameters for mask cleanup and outline smoothing
        self.mask_threshold = settings.yolo_mask_threshold
        self.min_area_fraction = settings.yolo_min_area_fraction
        self.morph_kernel_px = max(1, int(settings.yolo_morph_kernel_px))
        self.morph_iterations = max(0, int(settings.yolo_morph_iterations))
        self.simplify_epsilon_ratio = settings.outline_simplify_epsilon_ratio
        self.smooth_buffer_px = max(0, int(settings.outline_smooth_buffer_px))
        self.svg_use_smooth_curves = settings.svg_use_smooth_curves
        # Additional smoothing controls
        self.chaikin_iterations = max(0, int(getattr(settings, 'outline_chaikin_iterations', 1)))
        self.max_outline_points = max(50, int(getattr(settings, 'outline_max_points', 400)))
        self.smooth_buffer_max_px = max(1, int(getattr(settings, 'outline_smooth_buffer_max_px', 10)))
        # Do not smooth the base outline by default; only the offset will be rounded
        self.apply_base_rounding = False
        self.apply_base_overlay_smoothing = False
        self.model = None
        if load_model:
            self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            # Resolve model path, prefer local files if configured path missing
            model_file = Path(self.model_path)
            if not model_file.exists():
                for fallback in [
                    Path("yolo11s-seg.pt"),
                    Path("yolo11n-seg.pt"),
                    Path("yoloe-11s-seg.pt"),
                ]:
                    if fallback.exists():
                        logger.info(f"Configured model not found. Using fallback: {fallback}")
                        model_file = fallback
                        break
            self.model = YOLO(str(model_file))
            logger.info(f"Loaded YOLO model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def segment_image(self, image: np.ndarray, confidence_threshold: float = None) -> YOLOPrediction:
        """Segment image and detect tools."""
        if self.model is None:
            # Lazy-load model only when needed for segmentation
            self._load_model()
        
        # Use configured inference threshold (lower for better detection) unless overridden
        confidence = confidence_threshold if confidence_threshold is not None else settings.yolo_inference_confidence
        
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
                    # Threshold and clean mask
                    binary_mask = (mask_resized > float(self.mask_threshold)).astype(np.uint8)
                    if self.morph_iterations > 0 and self.morph_kernel_px > 0:
                        binary_mask = self._postprocess_mask(binary_mask)
                    
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
        """Convert binary mask to a smoothed polygon exterior (in pixels)."""
        img_h, img_w = mask.shape[:2]
        min_area = float(self.min_area_fraction) * float(img_h * img_w)

        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        # Filter tiny contours and keep the largest
        contours = [c for c in contours if cv2.contourArea(c) >= max(1.0, min_area)]
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        pts = largest.reshape(-1, 2)

        # Shapely polygon smoothing via buffer in/out to round corners and clean self-intersections
        poly = Polygon(pts)
        try:
            area_orig = abs(poly.area)
        except Exception:
            area_orig = 0.0
        if not poly.is_valid:
            poly = poly.buffer(0)
        if self.apply_base_rounding and self.smooth_buffer_px > 0:
            # Dynamic buffer based on perimeter for robust corner rounding
            perim = float(cv2.arcLength(largest, True))
            dyn = max(float(self.smooth_buffer_px), max(1.0, 0.003 * perim))
            dyn = min(float(self.smooth_buffer_max_px), float(dyn))
            poly = poly.buffer(dyn, join_style=1).buffer(-dyn, join_style=1)

        # Simplify with RDP equivalent to reduce noise while preserving shape
        epsilon = max(1e-6, float(self.simplify_epsilon_ratio) * float(cv2.arcLength(largest, True)))
        poly = poly.simplify(epsilon, preserve_topology=True)
        # Guardrail: if smoothing/distortion changed area too much, fall back to original
        try:
            if area_orig > 0:
                area_new = abs(poly.area)
                if abs(area_new - area_orig) / area_orig > 0.2:
                    poly = Polygon(pts)
        except Exception:
            pass
        if poly.is_empty:
            return []

        # Return exterior coordinates (drop duplicate last point)
        coords = list(poly.exterior.coords)
        return [(float(x), float(y)) for (x, y) in coords[:-1]]
    
    def extract_tool_outline(self, mask: np.ndarray, px_per_mm: float) -> Dict:
        """Extract tool outline and dimensions from mask."""
        if np.sum(mask) == 0:
            return {"outline": [], "dimensions": {}, "area_mm2": 0.0}

        outline = self._mask_to_polygon(mask)
        if not outline:
            return {"outline": [], "dimensions": {}, "area_mm2": 0.0}

        # Do not smooth the base outline for overlay by default
        if self.apply_base_overlay_smoothing and self.chaikin_iterations > 0 and len(outline) >= 4:
            outline = self._chaikin_smooth(outline, iterations=self.chaikin_iterations)
            if len(outline) > self.max_outline_points:
                step = max(1, len(outline) // self.max_outline_points)
                outline = outline[::step]

        cnt = np.array(outline, dtype=np.float32).reshape(-1, 1, 2)
        # Calculate dimensions
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Convert to mm
        width_mm = w / px_per_mm
        height_mm = h / px_per_mm
        area_mm2 = cv2.contourArea(cnt) / (px_per_mm ** 2)
        
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
        
        return {"outline": outline, "dimensions": dimensions, "area_mm2": float(area_mm2)}
    
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
        
        # Ensure closed ring
        if outline_mm[0] != outline_mm[-1]:
            outline_mm.append(outline_mm[0])

        # Pre-smooth outline for nicer curves
        if self.svg_use_smooth_curves and len(outline_mm) >= 4:
            pts_mm = outline_mm
            # Limit vertex count to keep path manageable
            if len(pts_mm) > self.max_outline_points:
                step = max(1, len(pts_mm) // self.max_outline_points)
                pts_mm = pts_mm[::step]
                if pts_mm[0] != pts_mm[-1]:
                    pts_mm.append(pts_mm[0])
            # Chaikin corner cutting for a few iterations
            if self.chaikin_iterations > 0:
                pts_mm = self._chaikin_smooth(pts_mm, iterations=self.chaikin_iterations)
            path_data = self._points_to_cubic_bezier_path(pts_mm)
        else:
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
        stroke-width="0.1mm"
        stroke-linejoin="round"
        stroke-linecap="round"/>
</svg>"""
        return svg

    def _offset_outline(self, outline: List[Tuple[float, float]], offset_mm: float, px_per_mm: float) -> List[Tuple[float, float]]:
        """Offset an outline by offset_mm (mm). Positive grows outward, negative inward.

        Uses shapely buffer to round corners and close small gaps. Returns exterior ring coords.
        """
        if not outline or px_per_mm <= 0 or abs(offset_mm) < 1e-6:
            return outline
        try:
            poly = Polygon(outline)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return outline
            offset_px = float(offset_mm) * float(px_per_mm)

            # 1) Pre-simplify in pixel space to remove jagged steps while preserving straight runs
            pre_mm = 0.5  # remove sub-mm noise from the base before offsetting
            pre_eps = max(1.0, pre_mm * float(px_per_mm))
            try:
                poly = poly.simplify(pre_eps, preserve_topology=True)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                pass

            # 2) Offset with round joins and moderate resolution for smooth arcs
            #    resolution=8 keeps arcs soft without over-segmentation
            grown = poly.buffer(offset_px, resolution=8, join_style=1)
            if grown.is_empty:
                return outline
            if isinstance(grown, MultiPolygon):
                # choose largest area component
                grown = max(grown.geoms, key=lambda g: g.area)
            # 3) Post-simplify gently with mm-based epsilon to prefer straight lines
            post_mm = 0.75  # bias toward straighter segments in the offset result
            post_eps = max(1.0, post_mm * float(px_per_mm))
            try:
                grown = grown.simplify(post_eps, preserve_topology=True)
            except Exception:
                pass
            if grown.is_empty:
                return outline
            coords = list(grown.exterior.coords)
            # Drop duplicate last point
            return [(float(x), float(y)) for (x, y) in coords[:-1]]
        except Exception:
            return outline
    
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

    # ----------------------------
    # Helpers: mask and path utils
    # ----------------------------
    def _postprocess_mask(self, mask_binary: np.ndarray) -> np.ndarray:
        """Clean mask with open/close to remove noise and fill small gaps."""
        mask_binary = (mask_binary > 0).astype(np.uint8)
        if self.morph_kernel_px <= 0 or self.morph_iterations <= 0:
            return mask_binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_px, self.morph_kernel_px))
        cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        # Optional final shrink to tighten the outline
        shrink_px = getattr(settings, 'yolo_refine_shrink_px', 0)
        if isinstance(shrink_px, int) and shrink_px > 0:
            k = max(1, shrink_px * 2 + 1)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            cleaned = cv2.erode(cleaned, erode_kernel, iterations=1)
        return (cleaned > 0).astype(np.uint8)

    def _points_to_cubic_bezier_path(self, pts: List[Tuple[float, float]]) -> str:
        """Create a smooth closed path using cubic Bezier segments (Catmull-Rom-like)."""
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]

        def add(p, q):
            return (p[0] + q[0], p[1] + q[1])

        def sub(p, q):
            return (p[0] - q[0], p[1] - q[1])

        def mul(p, s: float):
            return (p[0] * s, p[1] * s)

        n = len(pts)
        path = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
        # Use uniform Catmull-Rom to Bezier conversion
        for i in range(n - 1):
            p0 = pts[(i - 1) % (n - 1)]
            p1 = pts[i]
            p2 = pts[(i + 1) % (n - 1)]
            p3 = pts[(i + 2) % (n - 1)]
            c1 = add(p1, mul(sub(p2, p0), 1.0 / 6.0))
            c2 = sub(p2, mul(sub(p3, p1), 1.0 / 6.0))
            path += f" C {c1[0]:.2f} {c1[1]:.2f}, {c2[0]:.2f} {c2[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f}"
        path += " Z"
        return path

    def _chaikin_smooth(self, pts: List[Tuple[float, float]], iterations: int = 1) -> List[Tuple[float, float]]:
        """Chaikin's corner cutting to smooth a closed polygon.
        Returns a new list of points; may increase the number of vertices.
        """
        if not pts or iterations <= 0:
            return pts
        closed = pts[0] == pts[-1]
        p = pts[:-1] if closed else pts[:]
        for _ in range(iterations):
            new_pts: List[Tuple[float, float]] = []
            n = len(p)
            for i in range(n):
                p0 = p[i]
                p1 = p[(i + 1) % n]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])
            p = new_pts
        if closed:
            if p[0] != p[-1]:
                p.append(p[0])
        return p
