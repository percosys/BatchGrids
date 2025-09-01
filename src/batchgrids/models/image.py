from enum import Enum
from typing import Dict, Optional

from sqlalchemy import JSON, Enum as SQLEnum, ForeignKey, String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class ImageStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Image(Base):
    __tablename__ = "images"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tools.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Storage URIs
    uri: Mapped[str] = mapped_column(String(500))  # Original image URI
    mask_uri: Mapped[Optional[str]] = mapped_column(String(500))  # Segmentation mask URI
    
    # Processing metadata
    px_per_mm: Mapped[Optional[float]] = mapped_column(Float)  # Scale factor from AprilTag calibration
    homography: Mapped[Optional[Dict]] = mapped_column(JSON)  # Homography matrix for rectification
    bbox: Mapped[Optional[Dict]] = mapped_column(JSON)  # Bounding box {"x": 0, "y": 0, "width": 100, "height": 100}
    
    status: Mapped[ImageStatus] = mapped_column(SQLEnum(ImageStatus), default=ImageStatus.UPLOADED)
    
    # Relationships
    user = relationship("User", back_populates="images")
    tool = relationship("Tool", back_populates="images")
    predictions = relationship("Prediction", back_populates="image")