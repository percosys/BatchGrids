from typing import Dict, Optional

from sqlalchemy import JSON, Boolean, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    model: Mapped[str] = mapped_column(String(100))  # Model name/version used
    
    # Top prediction
    top1: Mapped[str] = mapped_column(String(100))  # Top class prediction
    p_top1: Mapped[float] = mapped_column(Float)  # Confidence score for top prediction
    
    # All predictions
    topk: Mapped[Optional[Dict]] = mapped_column(JSON)  # All class predictions with scores
    
    # Review status
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    image = relationship("Image", back_populates="predictions")