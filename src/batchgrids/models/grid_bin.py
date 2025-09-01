from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class GridBin(Base):
    __tablename__ = "grid_bins"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    drawer_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drawers.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    
    # Gridfinity dimensions (in grid units, 42mm each)
    x_units: Mapped[int] = mapped_column(Integer)
    y_units: Mapped[int] = mapped_column(Integer)
    z_units: Mapped[int] = mapped_column(Integer)
    
    # Physical constraints in mm
    inner_clearance_mm: Mapped[float] = mapped_column(Float, default=40.0)  # Usable space per grid unit
    wall_mm: Mapped[float] = mapped_column(Float, default=1.0)  # Wall thickness
    
    # Tile label for inventory
    label_code: Mapped[Optional[str]] = mapped_column(String(50), index=True)  # e.g., "T1", "T2"
    
    # Relationships
    user = relationship("User", back_populates="grid_bins")
    drawer = relationship("Drawer", back_populates="grid_bins")
    assignments = relationship("Assignment", back_populates="bin")
    exports = relationship("Export", back_populates="bin")
    
    @property
    def usable_width_mm(self) -> float:
        """Usable width in millimeters."""
        return self.x_units * self.inner_clearance_mm
    
    @property
    def usable_height_mm(self) -> float:
        """Usable height in millimeters."""
        return self.y_units * self.inner_clearance_mm