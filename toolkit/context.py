import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pyarrow as pa

from toolkit.storage.fs import StorageManager
from toolkit.catalog.catalog import LakehouseCatalog
from toolkit.catalog.table import TableMetadata
from toolkit.operators.bronze import BatchIngestOperator
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
from toolkit.operators.bio import (
    FastaIngestOperator,
    IDMappingIngestOperator,
    IDMapLookupOperator,
    GraphCoarseningOperator,
    PPIIngestOperator,
    NgramGraphBuilderOperator,
    SequenceDeduplicationOperator,
)
from toolkit.operators.nlp import (
    ExtractEntitiesOperator,
    OntologyMapperOperator,
    ConstructKnowledgeGraphOperator,
    AuthorProfilerOperator,
    KnowledgeGraphEmbeddingOperator,
)


class IngestionContext:
    """
    Main Lakehouse Session / Ingestion Context.
    Manages the Bronze -> Silver -> Gold storage tiers, tracks metadata,
    and runs declarative data ingestion and entity resolution pipelines.
    """

    def __init__(
        self,
        lakehouse_root: str = "./lakehouse",
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        self.lakehouse_root = lakehouse_root.replace("\\", "/")
        self.storage = StorageManager(storage_options=storage_options)
        self.catalog = LakehouseCatalog()

        # Initialize Medallion directory tree
        for tier in ["bronze", "silver", "gold"]:
            tier_dir = os.path.join(self.lakehouse_root, tier).replace("\\", "/")
            self.storage.makedirs(tier_dir, exist_ok=True)

    def ingest(
        self,
        source_uri: str,
        target_name: str,
        file_format: str = "auto",
        delimiter: str = ",",
        has_header: bool = True,
        engine: str = "pyarrow",
        table_format: str = "parquet",
        **kwargs,
    ) -> TableMetadata:
        """Runs the Bronze Ingest Operator (supports pyarrow and dask engines, parquet and delta formats)."""
        options = dict(kwargs)
        options["engine"] = engine
        options["table_format"] = table_format
        op = BatchIngestOperator(
            source_uri=source_uri,
            target_name=target_name,
            file_format=file_format,
            delimiter=delimiter,
            has_header=has_header,
            options=options,
        )
        return op.execute(self, **kwargs)

    def stage(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        feature_columns: Optional[List[str]] = None,
        stopword_threshold: float = 0.05,
        min_token_len: int = 2,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Silver Stager Operator (Tokenization + Beta Stopword Pruning)."""
        op = StagerOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            feature_columns=feature_columns,
            stopword_threshold=stopword_threshold,
            min_token_len=min_token_len,
        )
        return op.execute(self, **kwargs)

    def block(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        method: str = "token_inverted_index",
        max_block_size: int = 200,
        num_perm: int = 64,
        threshold: float = 0.4,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Silver Blocking Operator (Candidate Pair Generation via Index or MinHash LSH)."""
        op = BlockingOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            method=method,
            max_block_size=max_block_size,
            num_perm=num_perm,
            threshold=threshold,
        )
        return op.execute(self, **kwargs)

    def resolve(
        self,
        records_table: str,
        pairs_table: str,
        target_name: str,
        key_column: str = "RecID",
        similarity_metric: str = "jaccard",
        threshold: float = 0.5,
        clustering: str = "transitive_closure",
        attribute_weights: Optional[Dict[str, float]] = None,
        attribute_metrics: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Gold Entity Resolution Operator (Transitive Closure, Louvain, Label Propagation, etc.)."""
        op = TransitiveClosureOperator(
            records_table=records_table,
            pairs_table=pairs_table,
            target_name=target_name,
            key_column=key_column,
            similarity_metric=similarity_metric,
            threshold=threshold,
            clustering=clustering,
            attribute_weights=attribute_weights,
            attribute_metrics=attribute_metrics,
        )
        return op.execute(self, **kwargs)

    def project_graph(
        self,
        source_table: str,
        target_name: str,
        sequence_column: str = "sequence",
        n: int = 3,
        directed: bool = True,
        nodes_target: Optional[str] = None,
        compute_probabilities: bool = True,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Gold Graph Projector / Ngram Graph Builder Operator."""
        op = NgramGraphBuilderOperator(
            source_table=source_table,
            target_name=target_name,
            sequence_column=sequence_column,
            id_column=kwargs.get("id_column", "sequence_id"),
            n=n,
            directed=directed,
            nodes_target=nodes_target,
            compute_probabilities=compute_probabilities,
            batch_size=kwargs.get("batch_size", 50000),
            export_matrices_dir=kwargs.get("export_matrices_dir"),
            export_dense=kwargs.get("export_dense", False),
            export_directgcn=kwargs.get("export_directgcn", True),
        )
        return op.execute(self, **kwargs)

    def construct_ngram_graph(
        self,
        source_table: str,
        target_name: str,
        sequence_column: str = "sequence",
        id_column: str = "sequence_id",
        n: int = 3,
        directed: bool = True,
        nodes_target: Optional[str] = None,
        compute_probabilities: bool = True,
        **kwargs,
    ) -> TableMetadata:
        """Builds an n-gram transition graph and optional node catalog from sequence data."""
        return self.project_graph(
            source_table=source_table,
            target_name=target_name,
            sequence_column=sequence_column,
            n=n,
            directed=directed,
            nodes_target=nodes_target,
            compute_probabilities=compute_probabilities,
            id_column=id_column,
            **kwargs,
        )

    def create_embeddings(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        method: str = "tfidf",
        dimension: int = 128,
        columns: Optional[List[str]] = None,
        subword: bool = False,
        max_df: float = 0.5,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Text Vectorization Operator (TF-IDF, n-gram hashing)."""
        op = TextVectorizationOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            method=method,
            dimension=dimension,
            columns=columns,
            subword=subword,
            max_df=max_df,
        )
        return op.execute(self, **kwargs)

    def train_gnn(
        self,
        source_table: str,
        graph_table: str,
        target_name: str,
        key_column: str = "node_id",
        src_col: str = "src",
        dst_col: str = "dst",
        weight_col: Optional[str] = "weight",
        model: str = "dgcn",
        dimension: int = 32,
        layers: int = 2,
        epochs: int = 20,
        directed: bool = True,
        **kwargs,
    ) -> TableMetadata:
        """Runs the GNN Encoder Operator (Directed GCN, GCN, DGI)."""
        op = GNNEncoderOperator(
            source_table=source_table,
            graph_table=graph_table,
            target_name=target_name,
            key_column=key_column,
            src_col=src_col,
            dst_col=dst_col,
            weight_col=weight_col,
            model=model,
            dimension=dimension,
            layers=layers,
            epochs=epochs,
            directed=directed,
        )
        return op.execute(self, **kwargs)


    def evaluate(
        self,
        target_table: str,
        truth_file: str,
        pairs_table: Optional[str] = None,
        key_column: str = "RecID",
        cluster_column: str = "canonical_entity_id",
        output_table: Optional[str] = None,
        **kwargs,
    ) -> EvaluationReport:
        """Runs the Benchmark Evaluation Operator against Ground Truth."""
        op = EvaluateResolutionOperator(
            target_table=target_table,
            truth_file=truth_file,
            pairs_table=pairs_table,
            key_column=key_column,
            cluster_column=cluster_column,
            output_table=output_table,
        )
        return op.execute(self, **kwargs)

    def profile(self, target_table: str, **kwargs) -> Dict[str, Any]:
        """Runs the Data Quality Profiler Operator on any Lakehouse table."""
        op = DataQualityProfilerOperator(target_table=target_table)
        return op.execute(self, **kwargs)

    def ingest_fasta(
        self,
        source_uri: str,
        target_name: str,
        sample: Optional[int] = None,
        limit: Optional[int] = None,
        random_seed: int = 42,
        target_ids: Optional[Union[Set[str], List[str], str]] = None,
        chunk_size: int = 50000,
        clean_sequences: bool = False,
        remove_non_standard: bool = False,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Bioinformatics FASTA Ingest Operator."""
        op = FastaIngestOperator(
            source_uri=source_uri,
            target_name=target_name,
            sample=sample,
            limit=limit,
            random_seed=random_seed,
            target_ids=target_ids,
            chunk_size=chunk_size,
            clean_sequences=clean_sequences,
            remove_non_standard=remove_non_standard,
        )
        return op.execute(self, **kwargs)

    def deduplicate_sequences(
        self,
        source_table: str,
        target_name: str,
        sequence_column: str = "sequence",
        id_column: str = "sequence_id",
        **kwargs,
    ) -> TableMetadata:
        """Runs the Silver-tier Sequence Entity Resolution / Deduplication Operator."""
        op = SequenceDeduplicationOperator(
            source_table=source_table,
            target_name=target_name,
            sequence_column=sequence_column,
            id_column=id_column,
        )
        return op.execute(self, **kwargs)

    def ingest_idmapping(
        self,
        source_uri: str,
        target_name: str,
        id_types: Optional[Union[List[str], Set[str], str]] = None,
        delimiter: str = "\t",
        has_header: bool = False,
        **kwargs,
    ) -> TableMetadata:
        """Runs the UniProt/TSV ID Mapping Ingest Operator."""
        op = IDMappingIngestOperator(
            source_uri=source_uri,
            target_name=target_name,
            id_types=id_types,
            delimiter=delimiter,
            has_header=has_header,
        )
        return op.execute(self, **kwargs)

    def map_identifiers(
        self,
        source_table: str,
        mapping_table: str,
        target_name: str,
        src_col: str,
        target_col: str = "uniprot_ac",
        id_type: Optional[str] = None,
        join_type: str = "inner",
        **kwargs,
    ) -> TableMetadata:
        """Runs the ID Map Lookup Operator translating cross-database IDs."""
        op = IDMapLookupOperator(
            source_table=source_table,
            mapping_table=mapping_table,
            target_name=target_name,
            src_col=src_col,
            target_col=target_col,
            id_type=id_type,
            join_type=join_type,
        )
        return op.execute(self, **kwargs)

    def coarsen_graph(
        self,
        source_table: str,
        target_name: str,
        src_col: str = "src",
        dst_col: str = "dst",
        weight_col: Optional[str] = "weight",
        levels: int = 1,
        method: str = "graclus",
        mapping_table: Optional[str] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Graph Coarsening Operator (Graclus / Heavy-Edge Matching)."""
        op = GraphCoarseningOperator(
            source_table=source_table,
            target_name=target_name,
            src_col=src_col,
            dst_col=dst_col,
            weight_col=weight_col,
            levels=levels,
            method=method,
            mapping_table=mapping_table,
        )
        return op.execute(self, **kwargs)

    def ingest_ppi(
        self,
        source_uri: str,
        target_name: str,
        source_format: str = "biogrid",
        experimental_system: Optional[str] = "physical",
        min_score: float = 400.0,
        mapping_table: Optional[str] = None,
        delimiter: Optional[str] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Protein-Protein Interaction (PPI) Ground Truth Ingestion Operator."""
        op = PPIIngestOperator(
            source_uri=source_uri,
            target_name=target_name,
            source_format=source_format,
            experimental_system=experimental_system,
            min_score=min_score,
            mapping_table=mapping_table,
            delimiter=delimiter,
        )
        return op.execute(self, **kwargs)

    def extract_entities(
        self,
        source_table: str,
        target_name: str,
        text_col: str = "abstract",
        key_col: str = "pmid",
        annotations_table: Optional[str] = None,
        vocab_path: Optional[str] = None,
        entity_types: Optional[Union[List[str], str]] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Biomedical Entity Extraction Operator."""
        op = ExtractEntitiesOperator(
            source_table=source_table,
            target_name=target_name,
            text_col=text_col,
            key_col=key_col,
            annotations_table=annotations_table,
            vocab_path=vocab_path,
            entity_types=entity_types,
        )
        return op.execute(self, **kwargs)

    def map_ontology(
        self,
        source_table: str,
        target_name: str,
        ontology: str = "chem2bio2rdf",
        drug_mapping: Optional[Union[str, Dict[str, str]]] = None,
        gene_mapping: Optional[Union[str, Dict[str, str]]] = None,
        ontology_triples: Optional[str] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Semantic Ontology Mapping & Normalization Operator."""
        op = OntologyMapperOperator(
            source_table=source_table,
            target_name=target_name,
            ontology=ontology,
            drug_mapping=drug_mapping,
            gene_mapping=gene_mapping,
            ontology_triples=ontology_triples,
        )
        return op.execute(self, **kwargs)

    def construct_knowledge_graph(
        self,
        articles_table: str,
        target_name: str,
        entities_table: Optional[str] = None,
        authors_table: Optional[str] = None,
        citations_table: Optional[str] = None,
        triples_table: Optional[str] = None,
        ontology_table: Optional[str] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Heterogeneous Multi-Relational Knowledge Graph Construction Operator."""
        op = ConstructKnowledgeGraphOperator(
            articles_table=articles_table,
            target_name=target_name,
            entities_table=entities_table,
            authors_table=authors_table,
            citations_table=citations_table,
            triples_table=triples_table,
            ontology_table=ontology_table,
        )
        return op.execute(self, **kwargs)

    def profile_graph(
        self,
        source_table: str,
        target_name: str,
        target_type: str = "author",
        top_n: int = 20,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Knowledge Graph Profiler & Entity Association Operator."""
        op = AuthorProfilerOperator(
            source_table=source_table,
            target_name=target_name,
            target_type=target_type,
            top_n=top_n,
        )
        return op.execute(self, **kwargs)

    def clean_blocks(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        alias_dict: Optional[Dict[str, str]] = None,
        edit_threshold: float = 0.85,
        typo_distance_max: int = 2,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Silver Block Cleaner & Global Token Replacement Operator (DWM45/DWM25)."""
        op = BlockCleanerOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            alias_dict=alias_dict,
            edit_threshold=edit_threshold,
            typo_distance_max=typo_distance_max,
        )
        return op.execute(self, **kwargs)

    def compare_record_features(
        self,
        source_pairs_table: str,
        records_table: str,
        target_name: str,
        key_column: str = "RecID",
        digit_tolerance: int = 6,
        typo_threshold: int = 2,
        **kwargs,
    ) -> TableMetadata:
        """Runs the 21-feature Record Feature Comparator with Digit Discrimination (DWM68/DWM69)."""
        op = RecordFeatureComparatorOperator(
            source_pairs_table=source_pairs_table,
            records_table=records_table,
            target_name=target_name,
            key_column=key_column,
            digit_tolerance=digit_tolerance,
            typo_threshold=typo_threshold,
        )
        return op.execute(self, **kwargs)

    def character_maps(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        feature_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> TableMetadata:
        """Runs the CharacterMaps Operator for character-level transition and ASCII modeling."""
        op = CharacterMapsOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            feature_columns=feature_columns,
        )
        return op.execute(self, **kwargs)

    @staticmethod
    def calculate_cluster_entropy(cluster_tokens: List[List[str]], epsilon: float = 0.01) -> float:
        """Calculates cluster purity via Shannon entropy over token distributions (DWM95)."""
        return ClusterEntropyCalculator.calculate_entropy(cluster_tokens, epsilon=epsilon)

    def clique_rank(
        self,
        source_pairs_table: str,
        target_name: str,
        alpha: float = 0.85,
        max_iter: int = 100,
        **kwargs,
    ) -> TableMetadata:
        """Runs the CliqueRank PageRank / Random Surfer Operator (GER_CliqueRank_Runner)."""
        op = CliqueRankOperator(
            source_pairs_table=source_pairs_table,
            target_name=target_name,
            alpha=alpha,
            max_iter=max_iter,
        )
        return op.execute(self, **kwargs)

    def hits(
        self,
        source_pairs_table: str,
        target_name: str,
        max_iter: int = 100,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Kleinberg HITS (Hubs & Authorities) Operator on candidate entity graphs."""
        op = HITSOperator(
            source_pairs_table=source_pairs_table,
            target_name=target_name,
            max_iter=max_iter,
        )
        return op.execute(self, **kwargs)

    def bipartite_graph(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        token_column: str = "_staged_tokens",
        **kwargs,
    ) -> TableMetadata:
        """Constructs bipartite incidence graphs between records and tokens/attributes."""
        op = BipartiteGraphConstructor(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            token_column=token_column,
        )
        return op.execute(self, **kwargs)

    def embed_word2vec(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        dimension: int = 128,
        window: int = 5,
        min_count: int = 2,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Word2Vec / SGNS Embedding Operator (er2word2vec)."""
        op = Word2VecOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            dimension=dimension,
            window=window,
            min_count=min_count,
        )
        return op.execute(self, **kwargs)

    def embed_doc2vec(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        dimension: int = 128,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Doc2Vec Paragraph Vector Embedding Operator (er2doc2vec)."""
        op = Doc2VecOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            dimension=dimension,
        )
        return op.execute(self, **kwargs)

    def embed_fasttext(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        dimension: int = 128,
        **kwargs,
    ) -> TableMetadata:
        """Runs the FastText Subword N-gram Embedding Operator (er2fasttext2vec)."""
        op = FastTextOperator(
            source_table=source_table,
            target_name=target_name,
            key_column=key_column,
            dimension=dimension,
        )
        return op.execute(self, **kwargs)

    def embed_kg(
        self,
        source_table: str,
        target_name: str,
        dimension: int = 128,
        walk_length: int = 30,
        walks_per_node: int = 5,
        p: float = 0.5,
        q: float = 2.0,
        **kwargs,
    ) -> TableMetadata:
        """Runs the Node2Vec / MetaPath2Vec Knowledge Graph Embedding Operator (MERGE_08)."""
        op = KnowledgeGraphEmbeddingOperator(
            source_table=source_table,
            target_name=target_name,
            dimension=dimension,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
        )
        return op.execute(self, **kwargs)

    def load_benchmark(
        self,
        benchmark_name: str,
        source_path: str,
        target_name: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        **kwargs,
    ) -> TableMetadata:
        """Ingests and standardizes classic ER benchmark datasets (DBLP-ACM, DBLP-Google, Affiliations)."""
        op = BenchmarkDatasetLoader(
            benchmark_name=benchmark_name,
            source_path=source_path,
            target_name=target_name,
            delimiter=delimiter,
            encoding=encoding,
        )
        return op.execute(self, **kwargs)

    def profile_error_lookup(
        self,
        target_table: str,
        truth_file: str,
        records_table: Optional[str] = None,
        key_column: str = "RecID",
        cluster_col: str = "canonical_entity_id",
        **kwargs,
    ) -> Dict[str, Any]:
        """Performs error lookup diagnostics comparing resolved entities against ground truth (QuantifyQuality)."""
        op = ErrorLookupProfiler(
            target_table=target_table,
            truth_file=truth_file,
            records_table=records_table,
            key_column=key_column,
            cluster_col=cluster_col,
        )
        return op.execute(self, **kwargs)


    def read_table(self, name: str, version: Optional[int] = None) -> pa.Table:
        """Reads any registered lakehouse table directly into an Arrow Table, with optional Delta time-travel version."""
        meta = self.catalog.get_table(name)
        if not meta:
            raise KeyError(f"Table '{name}' not found in catalog.")
        if meta.format == "delta" or version is not None:
            return self.storage.read_delta(meta.uri, version=version)
        return self.storage.read_tabular(meta.uri, file_format=meta.format)

    def read_dask_table(self, name: str):
        """Reads a lakehouse table as a Dask DataFrame for out-of-core distributed analysis."""
        import dask.dataframe as dd
        meta = self.catalog.get_table(name)
        if not meta:
            raise KeyError(f"Table '{name}' not found in catalog.")
        return dd.read_parquet(meta.uri)

    def drop_table(self, name: str, purge: bool = False) -> bool:
        """Drops table from lakehouse catalog, with optional physical file deletion."""
        meta = self.catalog.get_table(name)
        if not meta:
            return False
        if purge:
            fs, path = self.storage.get_fs_and_path(meta.uri)
            candidates = [path]
            if path.endswith(".parquet"):
                candidates.append(path[:-8])
            for p in candidates:
                try:
                    if fs.isdir(p):
                        fs.rm(p, recursive=True)
                    elif fs.exists(p):
                        fs.rm(p)
                except Exception:
                    pass
        return self.catalog.drop_table(name)

    def vacuum(self, name: str, retain_hours: int = 168) -> int:
        """Purges old Delta commit files or orphaned Parquet parts older than retention threshold."""
        meta = self.catalog.get_table(name)
        if not meta:
            raise KeyError(f"Table '{name}' not found in catalog.")
        fs, path = self.storage.get_fs_and_path(meta.uri)
        if path.endswith(".parquet"):
            path = path[:-8]
        log_dir = f"{path.rstrip('/')}/_delta_log"
        if not fs.exists(log_dir):
            return 0
        active_files = set(self.storage._get_active_delta_files(fs, log_dir))
        all_parts = [os.path.basename(f) for f in fs.ls(path) if f.endswith(".parquet")]
        purged = 0
        for p in all_parts:
            if p not in active_files:
                try:
                    fs.rm(f"{path.rstrip('/')}/{p}")
                    purged += 1
                except Exception:
                    pass
        return purged

    def ai_profile(self, source_uri: str, api_key: Optional[str] = None) -> Any:
        """
        Profiles an unknown dataset assuming zero prior knowledge using Gemini AI / LLM.
        Returns a structured IngestionPlan with domain, format, and recommended SQL.
        """
        from toolkit.ai.router import AIIngestionRouter

        router = AIIngestionRouter(context=self, api_key=api_key)
        return router.plan(source_uri)

    def ai_route(
        self,
        source_uri: str,
        auto_execute: bool = False,
        target_table: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """
        Scans, profiles with Gemini AI, and routes an unknown dataset to the optimal
        Bronze preparation operator.
        """
        from toolkit.ai.router import AIIngestionRouter

        router = AIIngestionRouter(context=self, api_key=api_key)
        if auto_execute:
            return router.route_and_execute(source_uri, target_table=target_table)
        return router.plan(source_uri)

    def explain(self, query: str) -> str:
        """Explains the query plan for a declarative SQL statement."""
        from toolkit.sql.parser import SQLParser
        from toolkit.sql.planner import QueryPlanner

        ast = SQLParser().parse(query)
        planner = QueryPlanner(self)
        return planner.explain(ast)

    def show_tables(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Summarizes all registered tables in the lakehouse catalog."""
        return [t.summary() for t in self.catalog.list_tables(tier=tier)]

    def sql(self, query: str) -> Any:
        """
        Executes a declarative SQL statement against the lakehouse context.
        """
        from toolkit.sql.parser import SQLParser
        from toolkit.sql.planner import QueryPlanner

        ast = SQLParser().parse(query)
        planner = QueryPlanner(self)
        return planner.execute(ast)


# Alias LakehouseContext for IngestionContext
LakehouseContext = IngestionContext

