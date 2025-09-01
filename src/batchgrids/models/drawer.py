from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class Drawer(Base):
    __tablename__ = "drawers"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    label_code: Mapped[str] = mapped_column(String(50), unique=True, index=True)  # e.g., "D1", "DRAWER_A"
    
    # Relationships
    user = relationship("User", back_populates="drawers")
    grid_bins = relationship("GridBin", back_populates="drawer")