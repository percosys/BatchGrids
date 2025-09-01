from typing import List, Optional

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class Class(Base):
    __tablename__ = "classes"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    synonyms: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    description: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Relationships
    tools = relationship("Tool", back_populates="class_")