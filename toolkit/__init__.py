"""
Data Integration Toolkit (DITK)
A Universal Data Ingestion & Entity Resolution Lakehouse Engine.
"""

from toolkit.context import IngestionContext, LakehouseContext
from toolkit.catalog.catalog import LakehouseCatalog
from toolkit.storage.fs import StorageManager
from toolkit.operators.silver import (
    StagerOperator,
    BlockingOperator,
    CharacterMapsOperator,
    BlockCleanerOperator,
    RecordFeatureComparatorOperator,
    ClusterEntropyCalculator,
)
from toolkit.operators.gold import (
    TransitiveClosureOperator,
    GraphProjectorOperator,
    GraphOGAlgorithms,
    CliqueRankOperator,
    HITSOperator,
    BipartiteGraphConstructor,
)
from toolkit.operators.gnn import (
    TextVectorizationOperator,
    GNNEncoderOperator,
    Word2VecOperator,
    Doc2VecOperator,
    FastTextOperator,
    LSTMDataBuilder,
    XGCNDataBuilder,
)
from toolkit.operators.eval import (
    EvaluateResolutionOperator,
    DataQualityProfilerOperator,
    EvaluationReport,
    ErrorLookupProfiler,
    BenchmarkDatasetLoader,
    ConfigER,
)

__version__ = "1.0.0"
__all__ = [
    "IngestionContext",
    "LakehouseContext",
    "LakehouseCatalog",
    "StorageManager",
    "StagerOperator",
    "BlockingOperator",
    "CharacterMapsOperator",
    "BlockCleanerOperator",
    "RecordFeatureComparatorOperator",
    "ClusterEntropyCalculator",
    "TransitiveClosureOperator",
    "GraphProjectorOperator",
    "GraphOGAlgorithms",
    "CliqueRankOperator",
    "HITSOperator",
    "BipartiteGraphConstructor",
    "TextVectorizationOperator",
    "GNNEncoderOperator",
    "Word2VecOperator",
    "Doc2VecOperator",
    "FastTextOperator",
    "LSTMDataBuilder",
    "XGCNDataBuilder",
    "EvaluateResolutionOperator",
    "DataQualityProfilerOperator",
    "EvaluationReport",
    "ErrorLookupProfiler",
    "BenchmarkDatasetLoader",
    "ConfigER",
]
