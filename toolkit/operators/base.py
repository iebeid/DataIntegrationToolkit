from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from toolkit.catalog.table import TableMetadata


class BaseOperator(ABC):
    """Abstract base class for all Lakehouse ingestion and transformation operators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, context: Any, **kwargs) -> TableMetadata:
        """Executes the operator and returns the resulting TableMetadata."""
        pass
