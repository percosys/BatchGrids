#!/usr/bin/env python3
"""CLI tool for testing the computer vision pipeline."""

import argparse
import cv2
import sys
from pathlib import Path

from batchgrids.vision.image_processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Test BatchGrids computer vision pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", "-o", help="Output directory for results", default="./cv_output")
    parser.add_argument("--show", "-s", action="store_true", help="Show debug visualization")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate processing steps")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize processor
        print("Initializing computer vision pipeline...")
        processor = ImageProcessor()
        
        # Load and process image
        print(f"Loading image: {args.image_path}")
        image = processor.load_image(args.image_path)
        
        print("Processing image...")
        result = processor.process_image(image)
        
        # Print results
        print(f"\n=== PROCESSING RESULTS ===")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Error: {result.error_message}")
            sys.exit(1)
        
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        print(f"Needs review: {result.needs_review}")
        if result.needs_review:
            print(f"Review reason: {result.review_reason}")
        
        # Calibration results
        print(f"\n=== CALIBRATION ===")
        print(f"Scale: {result.calibration.px_per_mm:.2f} px/mm")
        print(f"AprilTags detected: {len(result.calibration.detected_tags)}")
        for tag in result.calibration.detected_tags:
            print(f"  Tag {tag.tag_id}: center=({tag.center[0]:.1f}, {tag.center[1]:.1f})")
        
        # YOLO results
        if result.yolo_prediction:
            print(f"\n=== YOLO SEGMENTATION ===")
            print(f"Model: {result.yolo_prediction.model_version}")
            print(f"Processing time: {result.yolo_prediction.processing_time_ms:.1f}ms")
            print(f"Detections: {len(result.yolo_prediction.detections)}")
            
            for i, detection in enumerate(result.yolo_prediction.detections[:3]):  # Top 3
                print(f"  {i+1}. {detection.class_name}: {detection.confidence:.3f}")
                print(f"     Area: {detection.area_pixels:.0f} pixels")
        
        # Tool dimensions
        if result.tool_dimensions:
            print(f"\n=== TOOL DIMENSIONS ===")
            dims = result.tool_dimensions
            print(f"Length: {dims.get('length_mm', 0):.1f}mm")
            print(f"Width: {dims.get('width_mm', 0):.1f}mm")
            print(f"Area: {dims.get('area_mm2', 0):.1f}mm²")
        
        # Save results
        base_name = Path(args.image_path).stem
        
        # Save debug visualization
        debug_viz = processor.create_debug_visualization(image, result)
        debug_path = output_dir / f"{base_name}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_viz)
        print(f"\nDebug visualization saved: {debug_path}")
        
        # Save individual steps if requested
        if args.save_steps:
            # Original with AprilTags
            if result.calibration.detected_tags:
                original_tagged = processor.apriltag_detector.draw_detections(
                    image, result.calibration.detected_tags
                )
                cv2.imwrite(str(output_dir / f"{base_name}_apriltags.jpg"), original_tagged)
            
            # Rectified image
            if result.rectified_image is not None:
                cv2.imwrite(str(output_dir / f"{base_name}_rectified.jpg"), result.rectified_image)
            
            # YOLO visualization
            if result.yolo_prediction and result.rectified_image is not None:
                yolo_viz = processor.yolo_segmenter.visualize_segmentation(
                    result.rectified_image, result.yolo_prediction
                )
                cv2.imwrite(str(output_dir / f"{base_name}_yolo.jpg"), yolo_viz)
            
            print("Individual processing steps saved to output directory")
        
        # Save SVG outline
        if result.svg_outline:
            svg_path = output_dir / f"{base_name}_outline.svg"
            with open(svg_path, 'w') as f:
                f.write(result.svg_outline)
            print(f"SVG outline saved: {svg_path}")
        
        # Show visualization if requested
        if args.show:
            cv2.imshow("BatchGrids Vision Pipeline Debug", debug_viz)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"\n✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()