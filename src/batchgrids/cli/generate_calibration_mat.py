#!/usr/bin/env python3
"""Generate a printable AprilTag calibration mat for BatchGrids."""

import argparse
import cv2
import numpy as np
from pathlib import Path


def generate_apriltag_image(tag_id: int, size: int = 200) -> np.ndarray:
    """Generate an AprilTag image using OpenCV."""
    # Create AprilTag detector to get the tag dictionary
    detector = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    
    # Generate tag
    tag_img = cv2.aruco.drawMarker(detector, tag_id, size)
    
    return tag_img


def create_calibration_mat(output_path: str, mat_size_mm: int = 300, tag_size_mm: int = 40, dpi: int = 300):
    """Create a printable calibration mat with AprilTags at the corners."""
    # Calculate dimensions in pixels
    pixels_per_mm = dpi / 25.4
    mat_size_px = int(mat_size_mm * pixels_per_mm)
    tag_size_px = int(tag_size_mm * pixels_per_mm)
    
    # Create white background
    mat = np.ones((mat_size_px, mat_size_px, 3), dtype=np.uint8) * 255
    
    # Tag positions (200mm apart, centered)
    center = mat_size_px // 2
    offset_px = int(100 * pixels_per_mm)  # 100mm from center
    
    positions = {
        0: (center - offset_px, center + offset_px),  # Bottom-left
        1: (center + offset_px, center + offset_px),  # Bottom-right
        2: (center + offset_px, center - offset_px),  # Top-right
        3: (center - offset_px, center - offset_px),  # Top-left
    }
    
    # Generate and place tags
    for tag_id, (x, y) in positions.items():
        tag_img = generate_apriltag_image(tag_id, tag_size_px)
        
        # Convert to 3-channel
        if len(tag_img.shape) == 2:
            tag_img = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
        
        # Calculate placement position (center the tag)
        start_x = x - tag_size_px // 2
        start_y = y - tag_size_px // 2
        end_x = start_x + tag_size_px
        end_y = start_y + tag_size_px
        
        # Ensure within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(mat_size_px, end_x)
        end_y = min(mat_size_px, end_y)
        
        # Place tag
        mat[start_y:end_y, start_x:end_x] = tag_img[:end_y-start_y, :end_x-start_x]
    
    # Add grid lines for reference (every 10mm)
    grid_spacing_px = int(10 * pixels_per_mm)
    grid_color = (200, 200, 200)  # Light gray
    
    # Vertical lines
    for x in range(0, mat_size_px, grid_spacing_px):
        cv2.line(mat, (x, 0), (x, mat_size_px), grid_color, 1)
    
    # Horizontal lines
    for y in range(0, mat_size_px, grid_spacing_px):
        cv2.line(mat, (0, y), (mat_size_px, y), grid_color, 1)
    
    # Add center crosshair
    center_color = (150, 150, 150)
    cv2.line(mat, (center - 20, center), (center + 20, center), center_color, 2)
    cv2.line(mat, (center, center - 20), (center, center + 20), center_color, 2)
    
    # Add labels and instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(2 * pixels_per_mm / 100)  # Scale with DPI
    font_color = (0, 0, 0)
    thickness = max(1, int(pixels_per_mm / 100))
    
    # Title
    title = "BatchGrids Calibration Mat"
    title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    title_x = (mat_size_px - title_size[0]) // 2
    cv2.putText(mat, title, (title_x, 50), font, font_scale, font_color, thickness)
    
    # Instructions
    instructions = [
        "1. Print this page at actual size (no scaling)",
        "2. Ensure AprilTags 0-3 are clearly visible",
        "3. Place tool in center area for scanning",
        f"4. Tag spacing: 200mm, Mat size: {mat_size_mm}mm"
    ]
    
    y_pos = mat_size_px - 120
    for instruction in instructions:
        cv2.putText(mat, instruction, (20, y_pos), font, font_scale * 0.6, font_color, thickness)
        y_pos += int(25 * pixels_per_mm / 100)
    
    # Add corner labels
    label_positions = [
        (positions[0][0] - 30, positions[0][1] + tag_size_px // 2 + 40, "0"),
        (positions[1][0] - 10, positions[1][1] + tag_size_px // 2 + 40, "1"),
        (positions[2][0] - 10, positions[2][1] - tag_size_px // 2 - 20, "2"),
        (positions[3][0] - 30, positions[3][1] - tag_size_px // 2 - 20, "3"),
    ]
    
    for x, y, label in label_positions:
        cv2.putText(mat, f"Tag {label}", (x, y), font, font_scale * 0.8, font_color, thickness)
    
    return mat


def main():
    parser = argparse.ArgumentParser(description="Generate AprilTag calibration mat for BatchGrids")
    parser.add_argument("--output", "-o", default="calibration_mat.png", help="Output file path")
    parser.add_argument("--size", "-s", type=int, default=300, help="Mat size in mm (default: 300)")
    parser.add_argument("--tag-size", "-t", type=int, default=40, help="Tag size in mm (default: 40)")
    parser.add_argument("--dpi", "-d", type=int, default=300, help="Print resolution (default: 300)")
    
    args = parser.parse_args()
    
    try:
        print(f"Generating calibration mat...")
        print(f"  Mat size: {args.size}mm")
        print(f"  Tag size: {args.tag_size}mm")
        print(f"  Resolution: {args.dpi} DPI")
        
        mat = create_calibration_mat(args.output, args.size, args.tag_size, args.dpi)
        
        # Save image
        success = cv2.imwrite(args.output, mat)
        if not success:
            print(f"Error: Failed to save image to {args.output}")
            return 1
        
        print(f"✓ Calibration mat saved: {args.output}")
        print(f"\nPrint Instructions:")
        print(f"  1. Print at ACTUAL SIZE (no scaling)")
        print(f"  2. Use high-quality paper (matte preferred)")
        print(f"  3. Ensure tags are crisp and black/white")
        print(f"  4. Verify tag spacing with ruler (should be 200mm)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())