import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Annotated, Optional

from batchgrids.api.routers.auth import get_current_user
from batchgrids.database import get_async_db
from batchgrids.models.user import User
from batchgrids.models.image import Image, ImageStatus
from batchgrids.models.prediction import Prediction
from batchgrids.storage.s3_client import s3_client
from batchgrids.worker.tasks import process_image_task

router = APIRouter()

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageResponse(BaseModel):
    id: int
    uri: str
    mask_uri: Optional[str] = None
    px_per_mm: Optional[float] = None
    status: ImageStatus
    tool_id: Optional[int] = None
    created_at: str
    
    class Config:
        from_attributes = True


class ProcessingStatus(BaseModel):
    status: str
    message: Optional[str] = None
    needs_review: Optional[bool] = None
    review_reason: Optional[str] = None
    debug_uri: Optional[str] = None
    tool_id: Optional[int] = None
    prediction: Optional[dict] = None


@router.post("/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    auto_process: bool = True,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Upload a tool image for processing."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    # Validate image format
    nparr = np.frombuffer(contents, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if cv_image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        # Generate unique key and upload to S3
        key = s3_client.generate_unique_key("images", file_ext)
        uri = s3_client.upload_bytes(contents, key, file.content_type or "image/jpeg")
        
        # Create image record
        image = Image(
            user_id=current_user.id,
            uri=uri,
            status=ImageStatus.UPLOADED
        )
        
        db.add(image)
        await db.commit()
        await db.refresh(image)
        
        # Queue processing if requested
        if auto_process:
            process_image_task.delay(image.id)
            image.status = ImageStatus.PROCESSING
            await db.commit()
        
        return ImageResponse(
            id=image.id,
            uri=image.uri,
            status=image.status,
            created_at=image.created_at.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/{image_id}/process", response_model=ProcessingStatus)
async def process_image(
    image_id: int,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Trigger processing of an uploaded image."""
    # Get image
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if image.status == ImageStatus.PROCESSING:
        return ProcessingStatus(
            status="processing",
            message="Image is already being processed"
        )
    
    if image.status == ImageStatus.PROCESSED:
        return ProcessingStatus(
            status="completed",
            message="Image has already been processed"
        )
    
    # Queue processing task
    process_image_task.delay(image_id)
    
    # Update status
    image.status = ImageStatus.PROCESSING
    await db.commit()
    
    return ProcessingStatus(
        status="queued",
        message="Image processing has been queued"
    )


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get image details and processing results."""
    # Get image with prediction
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return ImageResponse(
        id=image.id,
        uri=image.uri,
        mask_uri=image.mask_uri,
        px_per_mm=image.px_per_mm,
        status=image.status,
        tool_id=image.tool_id,
        created_at=image.created_at.isoformat()
    )


@router.get("/{image_id}/prediction")
async def get_image_prediction(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get YOLO prediction results for an image."""
    # Verify image ownership
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get prediction
    pred_result = await db.execute(
        select(Prediction).where(Prediction.image_id == image_id)
    )
    prediction = pred_result.scalar_one_or_none()
    
    if not prediction:
        return {"message": "No prediction available"}
    
    return {
        "model": prediction.model,
        "top_prediction": {
            "class_name": prediction.top1,
            "confidence": prediction.p_top1
        },
        "all_predictions": prediction.topk,
        "reviewed": prediction.reviewed,
        "created_at": prediction.created_at.isoformat()
    }


@router.get("/{image_id}/download")
async def download_image(
    image_id: int,
    image_type: str = "original",  # original, mask, debug
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get download URL for image or processed results."""
    # Verify image ownership
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get appropriate URI
    if image_type == "original":
        uri = image.uri
    elif image_type == "mask" and image.mask_uri:
        uri = image.mask_uri
    else:
        raise HTTPException(status_code=404, detail=f"Image type '{image_type}' not available")
    
    # Extract key from URI and generate presigned URL
    key = uri.split('/')[-1]
    download_url = s3_client.generate_presigned_url(key, expiration=3600)
    
    return {"download_url": download_url}