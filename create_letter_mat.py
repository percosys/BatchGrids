#!/usr/bin/env python3
"""Create a calibration mat optimized for US Letter paper (8.5" x 11")."""

import os

try:
    import numpy as np
    import cv2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def create_apriltag_pattern(tag_id, size):
    """Create a simple AprilTag-like pattern since we don't have the full library."""
    # Create a simple black and white pattern that resembles AprilTag
    tag = np.ones((size, size), dtype=np.uint8) * 255
    
    # Create border
    border_width = size // 8
    tag[:border_width, :] = 0  # top
    tag[-border_width:, :] = 0  # bottom
    tag[:, :border_width] = 0  # left
    tag[:, -border_width:] = 0  # right
    
    # Create inner pattern based on tag ID
    inner_size = size - 2 * border_width
    inner_start = border_width
    quarter_size = inner_size // 4
    
    # Different patterns for each tag ID
    patterns = {
        0: [(0, 0), (2, 2)],  # corners
        1: [(0, 2), (2, 0)],  # opposite corners
        2: [(1, 0), (1, 2)],  # vertical
        3: [(0, 1), (2, 1)]   # horizontal
    }
    
    # Fill pattern
    if tag_id in patterns:
        for row, col in patterns[tag_id]:
            r_start = inner_start + row * quarter_size
            r_end = r_start + quarter_size
            c_start = inner_start + col * quarter_size  
            c_end = c_start + quarter_size
            tag[r_start:r_end, c_start:c_end] = 0
    
    return tag


def create_letter_calibration_mat():
    """Create calibration mat for US Letter paper (8.5" x 11")."""
    
    # US Letter dimensions
    paper_width_inches = 8.5
    paper_height_inches = 11.0
    dpi = 300
    
    # Convert to pixels
    width_px = int(paper_width_inches * dpi)
    height_px = int(paper_height_inches * dpi)
    
    print(f"Creating calibration mat: {width_px}x{height_px} pixels")
    print(f"Paper size: {paper_width_inches}\" x {paper_height_inches}\"")
    
    # Create white background
    mat = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    
    # Calculate tag positions for maximum usable area
    # Leave 0.5" margins on all sides
    margin_inches = 0.5
    margin_px = int(margin_inches * dpi)
    
    usable_width = width_px - 2 * margin_px
    usable_height = height_px - 2 * margin_px
    
    # Tag size (about 0.6 inches)
    tag_size_px = int(0.6 * dpi)
    
    # Center the calibration area
    center_x = width_px // 2
    center_y = height_px // 2
    
    # 200mm spacing between tag centers (about 7.87 inches)
    spacing_inches = 200 / 25.4  # 200mm to inches
    spacing_px = int(spacing_inches * dpi)
    half_spacing = spacing_px // 2
    
    # Tag positions (200mm apart in a square)
    tag_positions = {
        0: (center_x - half_spacing, center_y + half_spacing),  # Bottom-left
        1: (center_x + half_spacing, center_y + half_spacing),  # Bottom-right
        2: (center_x + half_spacing, center_y - half_spacing),  # Top-right
        3: (center_x - half_spacing, center_y - half_spacing)   # Top-left
    }
    
    # Generate and place tags
    for tag_id, (x, y) in tag_positions.items():
        tag_img = create_apriltag_pattern(tag_id, tag_size_px)
        
        # Convert to 3-channel
        tag_img_3ch = np.stack([tag_img, tag_img, tag_img], axis=2)
        
        # Calculate placement
        start_x = x - tag_size_px // 2
        start_y = y - tag_size_px // 2
        end_x = start_x + tag_size_px
        end_y = start_y + tag_size_px
        
        # Ensure within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(width_px, end_x)
        end_y = min(height_px, end_y)
        
        # Place tag
        mat[start_y:end_y, start_x:end_x] = tag_img_3ch[:end_y-start_y, :end_x-start_x]
    
    # Add grid lines (every 10mm = ~0.4")
    grid_spacing_inches = 10 / 25.4
    grid_spacing_px = int(grid_spacing_inches * dpi)
    grid_color = (200, 200, 200)  # Light gray
    
    # Draw grid in the calibration area only
    grid_left = center_x - half_spacing - tag_size_px
    grid_right = center_x + half_spacing + tag_size_px
    grid_top = center_y - half_spacing - tag_size_px
    grid_bottom = center_y + half_spacing + tag_size_px
    
    # Vertical grid lines
    x = grid_left
    while x <= grid_right:
        cv2.line(mat, (int(x), int(grid_top)), (int(x), int(grid_bottom)), grid_color, 1)
        x += grid_spacing_px
    
    # Horizontal grid lines  
    y = grid_top
    while y <= grid_bottom:
        cv2.line(mat, (int(grid_left), int(y)), (int(grid_right), int(y)), grid_color, 1)
        y += grid_spacing_px
    
    # Add center crosshair
    crosshair_size = int(0.2 * dpi)  # 0.2 inch crosshair
    cv2.line(mat, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (100, 100, 100), 2)
    cv2.line(mat, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (100, 100, 100), 2)
    
    # Add title and instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 0, 0)
    thickness = 2
    
    # Title
    title = "BatchGrids Calibration Mat - US Letter"
    title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    title_x = (width_px - title_size[0]) // 2
    cv2.putText(mat, title, (title_x, int(0.3 * dpi)), font, font_scale, font_color, thickness)
    
    # Instructions
    instructions = [
        "PRINT INSTRUCTIONS:",
        "1. Print at ACTUAL SIZE (no scaling)",
        "2. Measure tag spacing = 200mm (7.87\")",
        "3. Place tool in center area for scanning",
        "4. Ensure all 4 corner tags (0,1,2,3) are visible"
    ]
    
    y_pos = int(10.2 * dpi)  # Near bottom
    for i, instruction in enumerate(instructions):
        font_scale_inst = 0.6 if i == 0 else 0.5
        thickness_inst = 2 if i == 0 else 1
        cv2.putText(mat, instruction, (margin_px, y_pos), font, font_scale_inst, font_color, thickness_inst)
        y_pos += int(0.2 * dpi)
    
    # Add measurement verification box
    verification_text = f"Verification: Tag spacing should measure exactly 200mm (7.87\")"
    cv2.putText(mat, verification_text, (margin_px, int(9.5 * dpi)), font, 0.5, (255, 0, 0), 1)
    
    # Add corner labels
    for tag_id, (x, y) in tag_positions.items():
        label_text = f"Tag {tag_id}"
        # Position labels outside the tags
        if tag_id == 0:  # Bottom-left
            label_pos = (x - tag_size_px//2 - 40, y + tag_size_px//2 + 30)
        elif tag_id == 1:  # Bottom-right
            label_pos = (x + tag_size_px//2 + 10, y + tag_size_px//2 + 30)
        elif tag_id == 2:  # Top-right
            label_pos = (x + tag_size_px//2 + 10, y - tag_size_px//2 - 10)
        else:  # Top-left
            label_pos = (x - tag_size_px//2 - 40, y - tag_size_px//2 - 10)
        
        cv2.putText(mat, label_text, label_pos, font, 0.6, (255, 0, 0), 2)
    
    # Add scale information
    scale_text = f"Scale: 200mm tag spacing @ {dpi} DPI"
    cv2.putText(mat, scale_text, (margin_px, int(0.8 * dpi)), font, 0.4, (100, 100, 100), 1)
    
    return mat


def main():
    """Create and save the calibration mat."""
    if not HAS_DEPS:
        print("Error: OpenCV and numpy not installed.")
        print("Install with: pip install opencv-python numpy")
        return 1
    
    print("Creating BatchGrids calibration mat for US Letter paper...")
    
    try:
        # Create the mat
        mat = create_letter_calibration_mat()
        
        # Save to output directory (or desktop if not available)
        output_path = "/app/output/BatchGrids_Calibration_Mat_Letter.png"
        if not os.path.exists("/app/output"):
            output_path = "/Users/alexperez/Desktop/BatchGrids_Calibration_Mat_Letter.png"
        success = cv2.imwrite(output_path, mat)
        
        if success:
            print(f"✓ Calibration mat saved to: {output_path}")
            print("\nPRINT INSTRUCTIONS:")
            print("1. Open the PNG file and print at ACTUAL SIZE")
            print("2. Do NOT scale or fit to page - must be exact size")
            print("3. Use high-quality paper (matte preferred)")
            print("4. Verify with ruler: tag spacing = 200mm (7.87\")")
            print("\nUSAGE:")
            print("1. Place tool in center area between tags")
            print("2. Ensure all 4 corner tags are visible in photo")
            print("3. Upload photo to BatchGrids for processing")
            return 0
        else:
            print("✗ Failed to save image")
            return 1
            
    except Exception as e:
        print(f"Error creating calibration mat: {e}")
        return 1


if __name__ == "__main__":
    exit(main())