#!/usr/bin/env python3
"""Simple script to generate test calibration mat and test image."""

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available, creating simple test image instead")


def create_simple_test_image():
    """Create a simple test image with mock AprilTags."""
    # Create a 800x600 image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw mock AprilTags as black squares with white borders
    tag_size = 60
    positions = [
        (200, 150),  # Tag 0 - Top-left  
        (600, 150),  # Tag 1 - Top-right
        (600, 450),  # Tag 2 - Bottom-right
        (200, 450),  # Tag 3 - Bottom-left
    ]
    
    for i, (x, y) in enumerate(positions):
        # Draw white border
        cv2.rectangle(img, (x-30, y-30), (x+30, y+30), (255, 255, 255), -1)
        # Draw black center
        cv2.rectangle(img, (x-25, y-25), (x+25, y+25), (0, 0, 0), -1)
        # Add some pattern
        cv2.rectangle(img, (x-15, y-15), (x-5, y-5), (255, 255, 255), -1)
        cv2.rectangle(img, (x+5, y+5), (x+15, y+15), (255, 255, 255), -1)
        
        # Label
        cv2.putText(img, str(i), (x-40, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add a mock tool in the center (screwdriver shape)
    center_x, center_y = 400, 300
    
    # Handle (rectangle)
    cv2.rectangle(img, (center_x-5, center_y-80), (center_x+5, center_y+20), (139, 69, 19), -1)
    
    # Blade (thin rectangle)  
    cv2.rectangle(img, (center_x-2, center_y+20), (center_x+2, center_y+80), (192, 192, 192), -1)
    
    # Add instructions
    cv2.putText(img, "TEST IMAGE - Mock AprilTags & Tool", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Tags: 0=TL, 1=TR, 2=BR, 3=BL", (50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return img


def main():
    if not HAS_CV2:
        print("OpenCV not installed. Please install with: pip install opencv-python")
        return 1
    
    print("Creating test calibration image...")
    
    # Create test image
    test_img = create_simple_test_image()
    
    # Save image
    cv2.imwrite("test_calibration_image.jpg", test_img)
    
    print("✓ Test image saved: test_calibration_image.jpg")
    print("\nThis is a simplified test image with mock AprilTags.")
    print("For production, use proper AprilTag generation.")
    print("\nTo test the vision pipeline:")
    print("  python -m pytest or run the test manually")
    
    return 0


if __name__ == "__main__":
    exit(main())