import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pyarrow as pa


@dataclass
class TableMetadata:
    name: str
    tier: str  # 'bronze', 'silver', 'gold'
    uri: str
    schema: pa.Schema
    row_count: int
    created_at: float = field(default_factory=time.time)
    properties: Dict[str, Any] = field(default_factory=dict)
    format: str = "parquet"
    version: int = 0

    @property
    def qualified_name(self) -> str:
        if "." in self.name:
            return self.name
        return f"{self.tier}.{self.name}"

    def summary(self) -> Dict[str, Any]:
        return {
            "table": self.qualified_name,
            "tier": self.tier,
            "rows": self.row_count,
            "columns": len(self.schema.names),
            "format": self.format,
            "version": self.version,
            "uri": self.uri,
        }
