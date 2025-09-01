from enum import Enum
from typing import Optional

from sqlalchemy import Enum as SQLEnum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class ExportFormat(str, Enum):
    SVG = "svg"
    DXF = "dxf"
    STL = "stl"
    STEP = "step"
    THREEMF = "3mf"


class Export(Base):
    __tablename__ = "exports"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    bin_id: Mapped[int] = mapped_column(ForeignKey("grid_bins.id"), index=True)
    
    format: Mapped[ExportFormat] = mapped_column(SQLEnum(ExportFormat))
    uri: Mapped[Optional[str]] = mapped_column(String(500))  # S3 URI of generated file
    version: Mapped[int] = mapped_column(Integer, default=1)
    
    # Export metadata
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Relationships
    user = relationship("User", back_populates="exports")
    bin = relationship("GridBin", back_populates="exports")