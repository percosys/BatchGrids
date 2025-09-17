import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    import apriltag  # type: ignore
    _APRILTAG_BACKEND = 'apriltag'
except Exception:
    try:
        from pupil_apriltags import Detector as PupilDetector  # type: ignore
        _APRILTAG_BACKEND = 'pupil'
    except Exception:
        # No AprilTag backend - will use fallback methods
        _APRILTAG_BACKEND = 'none'
from dataclasses import dataclass

from batchgrids.config import settings


@dataclass
class AprilTagDetection:
    """AprilTag detection result."""
    tag_id: int
    center: Tuple[float, float]
    corners: np.ndarray  # 4x2 array of corner coordinates
    pose_R: Optional[np.ndarray] = None  # Rotation matrix (if camera calibrated)
    pose_t: Optional[np.ndarray] = None  # Translation vector (if camera calibrated)


@dataclass
class CalibrationResult:
    """Calibration result from AprilTag detection."""
    px_per_mm: float
    homography_matrix: np.ndarray
    rectified_corners: Dict[int, Tuple[float, float]]
    detected_tags: List[AprilTagDetection]
    is_valid: bool
    error_message: Optional[str] = None


class AprilTagDetector:
    """AprilTag detector for scan mat calibration."""
    
    def __init__(self, tag_family: str = None):
        self.tag_family = tag_family or settings.apriltag_family
        if _APRILTAG_BACKEND == 'apriltag':
            self.detector = apriltag.Detector()
        elif _APRILTAG_BACKEND == 'pupil':
            # pupil-apriltags uses different init args; at minimum we set families
            self.detector = PupilDetector(families=self.tag_family)
        else:
            # No backend available - detection will always fail gracefully
            self.detector = None
        
        # Standard calibration mat setup (in mm)
        # Tags at corners of a known rectangle
        self.CALIBRATION_DISTANCE_MM = 200.0  # Distance between tag centers
        self.REQUIRED_TAG_IDS = {0, 1, 2, 3}  # Corner tag IDs
        
        # Expected positions for tags in rectified coordinate system (mm)
        self.REFERENCE_POSITIONS = {
            0: (0.0, 0.0),      # Bottom-left
            1: (200.0, 0.0),    # Bottom-right  
            2: (200.0, 200.0),  # Top-right
            3: (0.0, 200.0),    # Top-left
        }
    
    def detect_tags(self, image: np.ndarray) -> List[AprilTagDetection]:
        """Detect AprilTags in image."""
        if self.detector is None:
            # No backend available - return empty list
            return []
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect tags
        detections = self.detector.detect(gray)
        
        results = []
        for detection in detections:
            # pupil-apriltags returns numpy arrays similar to apriltag
            center = detection.center
            corners = detection.corners
            tag_detection = AprilTagDetection(
                tag_id=int(detection.tag_id),
                center=(float(center[0]), float(center[1])),
                corners=np.asarray(corners, dtype=np.float32),
            )
            results.append(tag_detection)
        
        return results
    
    def validate_calibration_tags(self, detections: List[AprilTagDetection]) -> Tuple[bool, str]:
        """Validate that we have all required calibration tags."""
        detected_ids = {det.tag_id for det in detections}
        
        if not self.REQUIRED_TAG_IDS.issubset(detected_ids):
            missing = self.REQUIRED_TAG_IDS - detected_ids
            return False, f"Missing required tags: {missing}"
        
        if len(detected_ids) > len(self.REQUIRED_TAG_IDS):
            extra = detected_ids - self.REQUIRED_TAG_IDS
            return False, f"Extra tags detected (remove from scan area): {extra}"
        
        return True, "All required tags detected"
    
    def compute_homography(self, detections: List[AprilTagDetection]) -> CalibrationResult:
        """Compute homography matrix from AprilTag detections."""
        # Filter to only calibration tags
        calib_detections = [d for d in detections if d.tag_id in self.REQUIRED_TAG_IDS]
        
        # Validate tags
        is_valid, error_msg = self.validate_calibration_tags(calib_detections)
        if not is_valid:
            return CalibrationResult(
                px_per_mm=0.0,
                homography_matrix=np.eye(3),
                rectified_corners={},
                detected_tags=detections,
                is_valid=False,
                error_message=error_msg
            )
        
        # Extract corner coordinates
        src_points = []
        dst_points = []
        
        for detection in calib_detections:
            tag_id = detection.tag_id
            center = detection.center
            
            # Source point (detected center in image)
            src_points.append([center[0], center[1]])
            
            # Destination point (reference position)
            ref_pos = self.REFERENCE_POSITIONS[tag_id]
            dst_points.append([ref_pos[0], ref_pos[1]])
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography
        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        
        if homography is None:
            return CalibrationResult(
                px_per_mm=0.0,
                homography_matrix=np.eye(3),
                rectified_corners={},
                detected_tags=detections,
                is_valid=False,
                error_message="Failed to compute homography"
            )
        
        # Calculate scale (pixels per mm)
        # Use the distance between two known points
        tag0_center = next(d.center for d in calib_detections if d.tag_id == 0)
        tag1_center = next(d.center for d in calib_detections if d.tag_id == 1)
        
        pixel_distance = np.linalg.norm(np.array(tag1_center) - np.array(tag0_center))
        px_per_mm = pixel_distance / self.CALIBRATION_DISTANCE_MM
        
        # Rectified corner positions
        rectified_corners = {}
        for detection in calib_detections:
            rect_pos = self.REFERENCE_POSITIONS[detection.tag_id]
            rectified_corners[detection.tag_id] = rect_pos
        
        return CalibrationResult(
            px_per_mm=px_per_mm,
            homography_matrix=homography,
            rectified_corners=rectified_corners,
            detected_tags=detections,
            is_valid=True
        )
    
    def rectify_image(self, image: np.ndarray, homography: np.ndarray, 
                     output_size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """Apply homography to rectify image."""
        return cv2.warpPerspective(image, homography, output_size)
    
    def calibrate_image(self, image: np.ndarray) -> CalibrationResult:
        """Complete calibration pipeline for an image."""
        detections = self.detect_tags(image)
        return self.compute_homography(detections)
    
    def draw_detections(self, image: np.ndarray, detections: List[AprilTagDetection]) -> np.ndarray:
        """Draw detected tags on image for visualization."""
        result = image.copy()
        
        for detection in detections:
            # Draw corners
            corners = detection.corners.astype(int)
            cv2.polylines(result, [corners], True, (0, 255, 0), 2)
            
            # Draw center
            center = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            # Draw tag ID
            cv2.putText(result, str(detection.tag_id), 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return result
