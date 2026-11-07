# ==============================================================================
# MODULE: data_ingestion/pipeline.py
# PURPOSE: Stage 1 Orchestrator - Executes Dask data ingestion, UniRef FASTA &
#          ID mapping processing, and constructs/saves the n-gram graphs on disk.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from pathlib import Path
from typing import List, Optional

from configuration.config import Config
from data_ingestion.data_manager import DataManager
from data_ingestion.graph_builder import ProtGramDataBuilder
from data_ingestion.id_mapper import IDMapper
from data_ingestion.ngram_graph import DirectedNgramGraph
from data_ingestion.data_utils import DataUtils
from data_ingestion.fs.file_system_manager import fs_manager


class DataIngestionPipeline:
    """
    Stage 1 Pipeline (`data_ingestion`):
    1. Validates/downloads raw UniRef FASTA, UniProt ID mappings, and PPI datasets.
    2. Runs Dask to convert `idmapping.dat` into a partitioned Parquet dataset.
    3. Runs Dask over UniRef FASTA sequences to build and save the directed
       hierarchical n-gram graphs (`ngram_graph_n1` ... `ngram_graph_nN`) on disk.
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            project_root = Path(__file__).resolve().parent.parent
            config_yaml = project_root / "configuration" / "config.yaml"
            config = Config(config_yaml_path=config_yaml if config_yaml.exists() else None)
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent

    def verify_and_prepare_raw_data(self) -> None:
        """Verifies raw files and runs Dask ID-mapping conversion to Parquet."""
        DataUtils.print_header("STAGE 1.1: VERIFYING RAW DATA & DASK PARQUET INGESTION")
        data_manager = DataManager(self.config, self.project_root)

        if self.config.PERFORM_HASH_CHECK_ON_STARTUP:
            if not data_manager.check_and_prepare_data():
                raise RuntimeError("Data integrity check and preparation failed.")

        fs, _ = fs_manager.get_fs_and_path(self.config.RAW_DATA_DIR)
        required_uris = [self.config.UNIREF_FASTA_PATH]
        if self.config.PPI_GROUND_TRUTH_SOURCE in ["biogrid", "both"]:
            required_uris.extend([self.config.BIOGRID_PATH, self.config.IDMAPPING_SELECTED_PATH])
        if self.config.PPI_GROUND_TRUTH_SOURCE in ["string", "both"]:
            required_uris.extend([self.config.STRING_DB_PATH, self.config.STRING_ID_MAPPING_PATH])

        for uri in required_uris:
            _, path = fs_manager.get_fs_and_path(uri)
            if not fs.exists(path):
                raise FileNotFoundError(f"Required raw data file not found: {uri}")

        if self.config.PPI_GROUND_TRUTH_SOURCE in ["biogrid", "both"]:
            if not IDMapper.convert_dat_to_parquet(
                self.config.IDMAPPING_SELECTED_PATH,
                self.config.IDMAPPING_PARQUET_PATH,
                blocksize=self.config.DASK_BLOCKSIZE
            ):
                raise RuntimeError("Failed to convert ID mapping file to Parquet via Dask.")

    def build_ngram_graphs(self) -> List[DirectedNgramGraph]:
        """Builds and persists the hierarchical n-gram graphs on disk using Dask."""
        builder = ProtGramDataBuilder(self.config)
        return builder.build_and_save_graphs()

    def run(self, skip_raw_verification: bool = False) -> List[DirectedNgramGraph]:
        """Runs the complete Stage 1 Data Ingestion workflow."""
        DataUtils.print_header("RUNNING STAGE 1: DATA INGESTION PIPELINE")
        if not skip_raw_verification:
            self.verify_and_prepare_raw_data()
        graphs = self.build_ngram_graphs()
        print(f"--- Stage 1 Complete: {len(graphs)} N-Gram Graph(s) saved to '{self.config.PROCESSED_DATA_DIR}' ---")
        return graphs


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run()
