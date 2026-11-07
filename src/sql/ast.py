from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ASTNode:
    pass


@dataclass
class IngestStatement(ASTNode):
    source_uri: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolveStatement(ASTNode):
    records_table: str
    pairs_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectGraphStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShowTablesStatement(ASTNode):
    tier: Optional[str] = None


@dataclass
class DescribeTableStatement(ASTNode):
    table_name: str


@dataclass
class SelectStatement(ASTNode):
    table_name: str
    columns: List[str] = field(default_factory=lambda: ["*"])
    where_clause: Optional[str] = None
    order_by: Optional[str] = None
    order_desc: bool = False
    limit: Optional[int] = None


@dataclass
class EvaluateStatement(ASTNode):
    target_table: str
    truth_file: str
    pairs_table: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileStatement(ASTNode):
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateEmbeddingsStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainGNNStatement(ASTNode):
    source_table: str
    graph_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestFastaStatement(ASTNode):
    source_uri: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestIDMappingStatement(ASTNode):
    source_uri: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapIdentifiersStatement(ASTNode):
    source_table: str
    target_table: str
    mapping_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoarsenGraphStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestPPIStatement(ASTNode):
    source_uri: str
    target_table: str
    mapping_table: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractEntitiesStatement(ASTNode):
    source_table: str
    target_table: str
    annotations_table: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapOntologyStatement(ASTNode):
    source_table: str
    target_table: str
    ontology_triples: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstructKnowledgeGraphStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileGraphStatement(ASTNode):
    source_table: str
    target_table: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DropTableStatement(ASTNode):
    table_name: str
    if_exists: bool = False
    purge: bool = False


@dataclass
class VacuumStatement(ASTNode):
    table_name: str
    retain_hours: int = 168


@dataclass
class ExplainStatement(ASTNode):
    inner_statement: ASTNode


@dataclass
class AIProfileStatement(ASTNode):
    source_uri: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIRouteStatement(ASTNode):
    source_uri: str
    target_table: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)





