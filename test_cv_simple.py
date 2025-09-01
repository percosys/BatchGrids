#!/usr/bin/env python3
"""Simple test of computer vision components without running the full pipeline."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if we can import the computer vision modules."""
    try:
        print("Testing imports...")
        
        # Test basic imports first
        import numpy as np
        print("✓ numpy imported")
        
        import cv2
        print(f"✓ OpenCV imported (version: {cv2.__version__})")
        
        # Test our modules
        from batchgrids.vision.apriltag_detector import AprilTagDetector
        print("✓ AprilTagDetector imported")
        
        from batchgrids.vision.yolo_segmenter import YOLOSegmenter  
        print("✓ YOLOSegmenter imported")
        
        from batchgrids.vision.image_processor import ImageProcessor
        print("✓ ImageProcessor imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_apriltag_detector():
    """Test AprilTag detector with dummy data."""
    try:
        print("\nTesting AprilTag detector...")
        
        import numpy as np
        from batchgrids.vision.apriltag_detector import AprilTagDetector
        
        detector = AprilTagDetector()
        print("✓ AprilTagDetector initialized")
        
        # Create dummy image  
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Test detection (should return empty list for blank image)
        detections = detector.detect_tags(dummy_image)
        print(f"✓ Tag detection works (found {len(detections)} tags in blank image)")
        
        return True
        
    except Exception as e:
        print(f"✗ AprilTag test failed: {e}")
        return False


def test_yolo_segmenter():
    """Test YOLO segmenter initialization."""
    try:
        print("\nTesting YOLO segmenter...")
        
        # This will download the model if needed
        from batchgrids.vision.yolo_segmenter import YOLOSegmenter
        
        segmenter = YOLOSegmenter()
        print("✓ YOLOSegmenter initialized")
        print(f"  Model path: {segmenter.model_path}")
        print(f"  Confidence threshold: {segmenter.confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== BatchGrids Computer Vision Pipeline Test ===\n")
    
    success = True
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed. Cannot proceed.")
        return 1
    
    # Test AprilTag detector
    if not test_apriltag_detector():
        success = False
    
    # Test YOLO segmenter  
    if not test_yolo_segmenter():
        success = False
    
    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed!")
        print("\nComputer vision pipeline is ready for use.")
        print("To test with real images:")
        print("1. Generate calibration mat: docker-compose run api python src/batchgrids/cli/generate_calibration_mat.py")
        print("2. Test pipeline: docker-compose run api python src/batchgrids/cli/test_vision.py <image>")
    else:
        print("✗ Some tests failed. Check the errors above.")
        
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())