"""
Data Integration Toolkit (DITK)
A Universal Data Ingestion & Entity Resolution Lakehouse Engine.
"""

from src.context import IngestionContext, LakehouseContext
from src.catalog.catalog import LakehouseCatalog
from src.storage.fs import StorageManager
from src.operators.silver import (
    StagerOperator,
    BlockingOperator,
    CharacterMapsOperator,
    BlockCleanerOperator,
    RecordFeatureComparatorOperator,
    ClusterEntropyCalculator,
)
from src.operators.gold import (
    TransitiveClosureOperator,
    GraphProjectorOperator,
    GraphOGAlgorithms,
    CliqueRankOperator,
    HITSOperator,
    BipartiteGraphConstructor,
)
from src.operators.gnn import (
    TextVectorizationOperator,
    GNNEncoderOperator,
    Word2VecOperator,
    Doc2VecOperator,
    FastTextOperator,
    LSTMDataBuilder,
    XGCNDataBuilder,
)
from src.operators.eval import (
    EvaluateResolutionOperator,
    DataQualityProfilerOperator,
    EvaluationReport,
    ErrorLookupProfiler,
    BenchmarkDatasetLoader,
    ConfigER,
)
from src.operators.synthetic import (
    SyntheticDemographicsGeneratorOperator,
    SyntheticSeedGenerator,
    SyntheticErrorCorruptor,
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
    "SyntheticDemographicsGeneratorOperator",
    "SyntheticSeedGenerator",
    "SyntheticErrorCorruptor",
]
