from typing import Dict, Optional

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class Tool(Base):
    __tablename__ = "tools"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    class_id: Mapped[Optional[int]] = mapped_column(ForeignKey("classes.id"), index=True)
    canonical_name: Mapped[str] = mapped_column(String(200), index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Dimensions in millimeters
    dims_mm: Mapped[Optional[Dict]] = mapped_column(JSON)  # {"length": 100.5, "width": 25.0, "thickness": 5.0}
    
    # Geometry as GeoJSON polygon
    shape_polygon: Mapped[Optional[Dict]] = mapped_column(JSON)  # GeoJSON polygon
    
    # Relationships
    user = relationship("User", back_populates="tools")
    class_ = relationship("Class", back_populates="tools")
    images = relationship("Image", back_populates="tool")
    assignments = relationship("Assignment", back_populates="tool")