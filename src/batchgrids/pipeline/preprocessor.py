"""
Image Preprocessor Module

Handles initial image analysis:
1. Paper detection and scale calibration
2. General object type discovery
3. Image quality assessment and enhancement

Author: BatchGrids Pipeline
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScaleCalibration:
    """Scale calibration information"""
    px_per_mm: float
    confidence: float
    method: str  # 'apriltag', 'paper_detection', 'manual'
    reference_object: Optional[str] = None
    measurements: Optional[Dict] = None


@dataclass
class ObjectHint:
    """Object type hint from preprocessing"""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    characteristics: Dict[str, any]


@dataclass
class PreprocessingResult:
    """Complete preprocessing result"""
    scale_calibration: Optional[ScaleCalibration]
    object_hints: List[ObjectHint]
    enhanced_image: np.ndarray
    metadata: Dict[str, any]
    processing_time: float


class ImagePreprocessor:
    """Intelligent image preprocessing for gridfinity pipeline"""
    
    def __init__(self):
        self.paper_sizes = {
            'US_LETTER': (216, 279),  # 8.5" × 11" in mm
            'A4': (210, 297),         # A4 in mm
            'US_LEGAL': (216, 356),   # 8.5" × 14" in mm
        }
    
    def process_image(self, image: np.ndarray) -> PreprocessingResult:
        """
        Complete image preprocessing pipeline
        
        Args:
            image: Input BGR image
            
        Returns:
            PreprocessingResult with scale, objects, and enhanced image
        """
        start_time = datetime.now()
        
        print("🔍 PREPROCESSING: Analyzing image...")
        
        # Step 1: Detect and calibrate scale
        scale_calibration = self._detect_scale(image)
        
        # Step 2: Enhance image quality
        enhanced_image = self._enhance_image(image)
        
        # Step 3: General object detection
        object_hints = self._detect_object_types(enhanced_image)
        
        # Step 4: Compile metadata
        metadata = {
            'original_size': image.shape,
            'enhanced_size': enhanced_image.shape,
            'timestamp': datetime.now().isoformat(),
            'num_objects': len(object_hints),
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PreprocessingResult(
            scale_calibration=scale_calibration,
            object_hints=object_hints,
            enhanced_image=enhanced_image,
            metadata=metadata,
            processing_time=processing_time
        )
    
    def _detect_scale(self, image: np.ndarray) -> Optional[ScaleCalibration]:
        """Detect scale using multiple methods"""
        
        # Method 1: AprilTag detection
        apriltag_result = self._detect_apriltag_scale(image)
        if apriltag_result:
            return apriltag_result
        
        # Method 2: Paper boundary detection
        paper_result = self._detect_paper_scale(image)
        if paper_result:
            return paper_result
        
        print("  ⚠️ No scale reference found")
        return None
    
    def _detect_apriltag_scale(self, image: np.ndarray) -> Optional[ScaleCalibration]:
        """Detect AprilTags for precise scale calibration"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple AprilTag dictionaries
            dictionaries = [
                cv2.aruco.DICT_APRILTAG_36h11,
                cv2.aruco.DICT_APRILTAG_25h9,
                cv2.aruco.DICT_APRILTAG_16h5,
            ]
            
            for dict_type in dictionaries:
                dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
                parameters = cv2.aruco.DetectorParameters()
                
                # Aggressive detection parameters
                parameters.minMarkerPerimeterRate = 0.01
                parameters.maxMarkerPerimeterRate = 10.0
                parameters.polygonalApproxAccuracyRate = 0.2
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                
                detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
                
                if ids is not None and len(ids) >= 4:
                    # Calculate scale from AprilTag size
                    # Assume 20mm AprilTags (standard calibration size)
                    tag_size_mm = 20.0
                    
                    # Average marker size in pixels
                    marker_sizes_px = []
                    for corner in corners:
                        # Calculate marker side length
                        side1 = np.linalg.norm(corner[0][0] - corner[0][1])
                        side2 = np.linalg.norm(corner[0][1] - corner[0][2])
                        marker_sizes_px.append((side1 + side2) / 2)
                    
                    avg_size_px = np.mean(marker_sizes_px)
                    px_per_mm = avg_size_px / tag_size_mm
                    
                    print(f"  ✅ AprilTag scale: {px_per_mm:.2f} px/mm ({len(ids)} tags)")
                    
                    return ScaleCalibration(
                        px_per_mm=px_per_mm,
                        confidence=0.95,
                        method='apriltag',
                        reference_object='20mm_apriltag',
                        measurements={'num_tags': len(ids), 'avg_size_px': avg_size_px}
                    )
            
        except Exception as e:
            print(f"  ⚠️ AprilTag detection failed: {e}")
        
        return None
    
    def _detect_paper_scale(self, image: np.ndarray) -> Optional[ScaleCalibration]:
        """Detect paper boundaries for scale estimation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection to find paper boundaries
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular paper-like contours
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) == 4:
                    # Calculate dimensions
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Check if it's reasonable paper size (not tiny, not huge)
                    area = w * h
                    image_area = gray.shape[0] * gray.shape[1]
                    
                    if 0.1 * image_area < area < 0.8 * image_area:
                        # Estimate paper type by aspect ratio
                        aspect_ratio = max(w, h) / min(w, h)
                        
                        best_match = None
                        best_score = float('inf')
                        
                        for paper_name, (width_mm, height_mm) in self.paper_sizes.items():
                            paper_aspect = max(width_mm, height_mm) / min(width_mm, height_mm)
                            score = abs(aspect_ratio - paper_aspect)
                            
                            if score < best_score:
                                best_score = score
                                best_match = (paper_name, width_mm, height_mm)
                        
                        if best_match and best_score < 0.2:  # Good aspect ratio match
                            paper_name, width_mm, height_mm = best_match
                            
                            # Calculate scale (use longer dimension)
                            paper_long_px = max(w, h)
                            paper_long_mm = max(width_mm, height_mm)
                            px_per_mm = paper_long_px / paper_long_mm
                            
                            print(f"  ✅ Paper scale: {px_per_mm:.2f} px/mm ({paper_name})")
                            
                            return ScaleCalibration(
                                px_per_mm=px_per_mm,
                                confidence=0.7,
                                method='paper_detection',
                                reference_object=paper_name,
                                measurements={'paper_size_px': (w, h), 'aspect_ratio': aspect_ratio}
                            )
            
        except Exception as e:
            print(f"  ⚠️ Paper detection failed: {e}")
        
        return None
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better object detection"""
        
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Optional: bilateral filtering for noise reduction
        enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
        
        return enhanced_image
    
    def _detect_object_types(self, image: np.ndarray) -> List[ObjectHint]:
        """Detect general object types to guide YOLOE prompting"""
        
        object_hints = []
        
        # Method 1: Basic shape analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find significant contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out small contours
            if area < 5000:
                continue
            
            # Analyze shape characteristics
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Classify by shape
            object_type = self._classify_by_shape(contour, aspect_ratio)
            
            if object_type:
                object_hints.append(ObjectHint(
                    object_type=object_type,
                    confidence=0.6,  # Moderate confidence from shape analysis
                    bbox=(x, y, x+w, y+h),
                    area=int(area),
                    characteristics={
                        'aspect_ratio': aspect_ratio,
                        'area': area,
                        'perimeter': cv2.arcLength(contour, True)
                    }
                ))
        
        print(f"  🔍 Found {len(object_hints)} object hints")
        return object_hints
    
    def _classify_by_shape(self, contour, aspect_ratio: float) -> Optional[str]:
        """Classify object type by shape characteristics"""
        
        # Elongated objects - likely tools
        if aspect_ratio > 3.0:
            return 'elongated_tool'  # knives, screwdrivers, etc.
        
        # Square-ish objects
        elif 0.8 < aspect_ratio < 1.2:
            return 'compact_tool'  # pliers, small tools
        
        # Moderately elongated
        elif 1.5 < aspect_ratio < 3.0:
            return 'medium_tool'  # wrenches, etc.
        
        return None