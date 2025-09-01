from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class Assignment(Base):
    __tablename__ = "assignments"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    bin_id: Mapped[int] = mapped_column(ForeignKey("grid_bins.id"), index=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tools.id"), index=True)
    
    # Position within bin (in mm from bottom-left corner)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    
    # Size allocation (in mm)
    w: Mapped[float] = mapped_column(Float)  # Width
    h: Mapped[float] = mapped_column(Float)  # Height
    
    # Rotation in degrees (0, 90, 180, 270)
    rotation: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    bin = relationship("GridBin", back_populates="assignments")
    tool = relationship("Tool", back_populates="assignments")