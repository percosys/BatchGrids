from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    
    # Relationships
    tools = relationship("Tool", back_populates="user")
    images = relationship("Image", back_populates="user")
    drawers = relationship("Drawer", back_populates="user")
    grid_bins = relationship("GridBin", back_populates="user")
    exports = relationship("Export", back_populates="user")
    tool_svgs = relationship("ToolSvg", back_populates="user")