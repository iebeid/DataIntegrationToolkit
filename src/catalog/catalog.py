from typing import Dict, List, Optional, Tuple
from src.catalog.table import TableMetadata


class LakehouseCatalog:
    """
    Catalog to track tables across Bronze, Silver, and Gold medallion tiers.
    """

    @staticmethod
    def parse_table_name(name: str, default_tier: str = "bronze") -> Tuple[str, str]:
        if "." in name:
            parts = name.split(".", 1)
            return parts[0].lower(), parts[1]
        return default_tier.lower(), name

    def __init__(self):
        # Maps qualified name (e.g. 'bronze.customers') -> TableMetadata
        self._tables: Dict[str, TableMetadata] = {}

    def register_table(self, metadata: TableMetadata) -> None:
        self._tables[metadata.qualified_name] = metadata
        # Also alias by short name if unique
        if "." not in metadata.name:
            self._tables[metadata.name] = metadata

    def get_table(self, name: str) -> Optional[TableMetadata]:
        if name in self._tables:
            return self._tables[name]
        # Try checking across tiers if short name provided
        for tier in ["gold", "silver", "bronze"]:
            qname = f"{tier}.{name}"
            if qname in self._tables:
                return self._tables[qname]
        return None

    def list_tables(self, tier: Optional[str] = None) -> List[TableMetadata]:
        tables = [t for k, t in self._tables.items() if "." in k]
        if tier:
            tables = [t for t in tables if t.tier.lower() == tier.lower()]
        return tables

    def drop_table(self, name: str) -> bool:
        target = self.get_table(name)
        if target:
            if target.qualified_name in self._tables:
                del self._tables[target.qualified_name]
            if target.name in self._tables:
                del self._tables[target.name]
            return True
        return False
