import logging
from celery import shared_task
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from batchgrids.database import engine
from batchgrids.models.image import Image, ImageStatus
from batchgrids.models.tool import Tool
from batchgrids.models.prediction import Prediction
from batchgrids.vision.image_processor import ImageProcessor
from batchgrids.storage.s3_client import s3_client

logger = logging.getLogger(__name__)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@shared_task
def process_image_task(image_id: int) -> dict:
    """Process uploaded image with AprilTag detection and YOLO segmentation."""
    logger.info(f"Starting image processing for image {image_id}")
    
    with SessionLocal() as db:
        try:
            # Get image record
            image = db.get(Image, image_id)
            if not image:
                return {"status": "error", "message": "Image not found", "image_id": image_id}
            
            # Update status to processing
            image.status = ImageStatus.PROCESSING
            db.commit()
            
            # Load image from S3
            try:
                image_array = s3_client.download_image(image.uri.split('/')[-1])  # Extract key from URI
            except Exception as e:
                logger.error(f"Failed to load image from S3: {e}")
                image.status = ImageStatus.FAILED
                db.commit()
                return {"status": "error", "message": "Failed to load image", "image_id": image_id}
            
            # Process image
            processor = ImageProcessor()
            result = processor.process_image(image_array)
            
            if not result.success:
                logger.error(f"Image processing failed: {result.error_message}")
                image.status = ImageStatus.FAILED
                db.commit()
                return {
                    "status": "error", 
                    "message": result.error_message, 
                    "image_id": image_id
                }
            
            # Update image with processing results
            image.px_per_mm = result.calibration.px_per_mm
            image.homography = result.calibration.homography_matrix.tolist()
            image.bbox = None  # Will be set if we create a tool
            
            # Save mask to S3 if we have segmentation
            if result.yolo_prediction and result.yolo_prediction.top_prediction:
                mask_key = s3_client.generate_unique_key("masks", ".png")
                mask_uri = s3_client.upload_image(
                    result.yolo_prediction.top_prediction.mask * 255,  # Convert to 0-255
                    mask_key,
                    "image/png"
                )
                image.mask_uri = mask_uri
            
            # Save prediction data
            if result.yolo_prediction:
                prediction = Prediction(
                    image_id=image_id,
                    model=result.yolo_prediction.model_version,
                    top1=result.yolo_prediction.top_prediction.class_name if result.yolo_prediction.top_prediction else "unknown",
                    p_top1=result.yolo_prediction.top_prediction.confidence if result.yolo_prediction.top_prediction else 0.0,
                    topk={
                        det.class_name: det.confidence 
                        for det in result.yolo_prediction.detections[:5]  # Top 5
                    },
                    reviewed=False
                )
                db.add(prediction)
            
            # Create tool record if we have good results and don't need review
            tool = None
            if result.yolo_prediction and result.yolo_prediction.top_prediction and not result.needs_review:
                tool = Tool(
                    user_id=image.user_id,
                    canonical_name=result.yolo_prediction.top_prediction.class_name,
                    dims_mm=result.tool_dimensions,
                    shape_polygon={
                        "type": "Polygon",
                        "coordinates": [result.tool_outline] if result.tool_outline else []
                    }
                )
                db.add(tool)
                db.flush()  # Get tool ID
                
                # Link image to tool
                image.tool_id = tool.id
                
                # Store SVG outline in S3
                if result.svg_outline:
                    svg_key = s3_client.generate_unique_key("outlines", ".svg")
                    s3_client.upload_string(result.svg_outline, svg_key, "image/svg+xml")
            
            # Save debug visualization
            debug_viz = processor.create_debug_visualization(image_array, result)
            debug_key = s3_client.generate_unique_key("debug", ".jpg")
            s3_client.upload_image(debug_viz, debug_key, "image/jpeg")
            
            # Update status
            image.status = ImageStatus.PROCESSED
            db.commit()
            
            response = {
                "status": "success",
                "image_id": image_id,
                "calibration": {
                    "px_per_mm": result.calibration.px_per_mm,
                    "is_valid": result.calibration.is_valid
                },
                "needs_review": result.needs_review,
                "review_reason": result.review_reason,
                "processing_time_ms": result.processing_time_ms,
                "debug_uri": s3_client._get_object_url(debug_key)
            }
            
            if tool:
                response["tool_id"] = tool.id
                response["tool_name"] = tool.canonical_name
            
            if result.yolo_prediction and result.yolo_prediction.top_prediction:
                response["prediction"] = {
                    "class_name": result.yolo_prediction.top_prediction.class_name,
                    "confidence": result.yolo_prediction.top_prediction.confidence
                }
            
            logger.info(f"Successfully processed image {image_id}")
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error processing image {image_id}: {e}")
            
            # Update status to failed
            try:
                image = db.get(Image, image_id)
                if image:
                    image.status = ImageStatus.FAILED
                    db.commit()
            except Exception:
                pass
            
            return {
                "status": "error",
                "message": str(e),
                "image_id": image_id
            }


@shared_task
def export_bin_task(bin_id: int, format: str) -> dict:
    """Generate export file for bin in specified format."""
    # TODO: Implement export generation
    # 1. Load bin and tool assignments
    # 2. Generate file based on format:
    #    - SVG/DXF: 2D foam insert patterns
    #    - STL/STEP/3MF: 3D Gridfinity bin models
    # 3. Store file in S3
    # 4. Create Export record
    
    return {"status": "not_implemented", "bin_id": bin_id, "format": format}


@shared_task
def auto_pack_bin_task(bin_id: int) -> dict:
    """Auto-pack tools into bin using packing algorithm."""
    # TODO: Implement auto-packing
    # 1. Load bin dimensions and available tools
    # 2. Run packing algorithm (start with row packing)
    # 3. Create Assignment records
    # 4. Handle overflow by creating new bins
    
    return {"status": "not_implemented", "bin_id": bin_id}