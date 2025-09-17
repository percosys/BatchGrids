from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from batchgrids.models.base import Base


class ToolSvg(Base):
    """Persisted SVG asset generated for a tool outline."""

    __tablename__ = "tool_svgs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    name: Mapped[str] = mapped_column(String(200))
    svg: Mapped[str] = mapped_column(Text)
    ai_description: Mapped[str] = mapped_column(Text)

    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    user = relationship("User", back_populates="tool_svgs")
