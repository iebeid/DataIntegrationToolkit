# ==============================================================================
# MODULE: data_ingestion/id_mapper.py
# PURPOSE: Dask-accelerated ID mapping utilities (idmapping.dat -> Parquet).
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Set, Union, List

import dask
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from data_ingestion.data_utils import DataUtils
from data_ingestion.fs.file_utils import FileUtils
from data_ingestion.fs.file_system_manager import fs_manager


class IDMapper:
    """Handles conversion and loading of protein ID mapping files using Dask."""

    @staticmethod
    def convert_dat_to_parquet(dat_uri: Union[str, Path], parquet_uri: Union[str, Path], blocksize: str = "128MB") -> bool:
        """
        Converts the massive idmapping.dat file to a partitioned Parquet dataset using Dask.
        This is a one-time operation that dramatically speeds up subsequent lookups.
        """
        DataUtils.print_header("Step: Pre-processing ID Mapping File to Parquet")
        fs, dat_path = fs_manager.get_fs_and_path(str(dat_uri))
        _, parquet_path = fs_manager.get_fs_and_path(str(parquet_uri))

        if fs.exists(parquet_path):
            try:
                parquet_files = fs.glob(f"{parquet_path.rstrip('/')}/*.parquet")
                if parquet_files:
                    print(f"  Found valid, non-empty Parquet dataset at '{parquet_uri}'. Skipping conversion.")
                    return True
                else:
                    print(f"  Warning: Parquet directory exists at '{parquet_uri}' but contains no .parquet files. Re-running conversion.")
                    shutil.rmtree(parquet_path)
            except Exception as e:
                print(f"  Warning: Could not verify existing Parquet dataset ({e}). Re-running conversion.")
                shutil.rmtree(parquet_path, ignore_errors=True)

        if not fs.exists(dat_path):
            print(f"  ERROR: Source ID mapping file not found at '{dat_uri}'. Cannot convert to Parquet.")
            return False

        print(f"  Converting '{os.path.basename(dat_path)}' to Parquet format at '{parquet_uri}'.")
        print("  This is a one-time operation and may take several minutes...")
        start_time = time.time()

        try:
            ddf = dd.read_csv(
                dat_uri,
                sep='\t',
                header=None,
                names=['uniprot_ac', 'id_type', 'external_id'],
                dtype={'uniprot_ac': 'object', 'id_type': 'category', 'external_id': 'object'},
                blocksize=blocksize,
                on_bad_lines='warn',
                engine='python'
            )

            with ProgressBar():
                ddf.to_parquet(
                    parquet_uri,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=False
                )

            duration = time.time() - start_time
            print(f"  Successfully converted to Parquet in {duration:.2f} seconds.")
            return True
        except Exception as e:
            print(f"  ERROR: Failed to convert '{dat_uri}' to Parquet: {e}")
            if fs.exists(parquet_path):
                fs.rm(parquet_path, recursive=True)
            return False

    @staticmethod
    def load_id_Mappings(parquet_uri: Union[str, Path], cache_uri: Union[str, Path], id_types_to_keep: List[str], blocksize_gb: float = 10.0) -> Dict[str, Set[str]]:
        """
        Creates a mapping from various biological database IDs to UniProt ACs by reading
        from the pre-processed Parquet dataset in a memory-safe, chunked manner using Dask.
        """
        fs, cache_path = fs_manager.get_fs_and_path(str(cache_uri))
        if fs.exists(cache_path):
            print(f"  Loading pre-computed ID map from cache: {cache_uri}")
            cached_map = FileUtils.load_object(cache_uri)
            if cached_map is not None:
                return cached_map
            print("  Warning: Failed to load cached ID map. Recomputing from Parquet...")

        _, parquet_path = fs_manager.get_fs_and_path(str(parquet_uri))
        if not fs.exists(parquet_path):
            print(f"  ERROR: Pre-processed Parquet dataset not found at '{parquet_uri}'.")
            return {}

        print(f"  Building BioGRID/External ID to UniProt map from Parquet dataset: '{os.path.basename(parquet_path)}'...")

        try:
            ddf = dd.read_parquet(parquet_uri, columns=['uniprot_ac', 'id_type', 'external_id'])
            filtered_ddf = ddf[ddf['id_type'].isin(id_types_to_keep)]
            grouped_series = filtered_ddf.groupby('external_id')['uniprot_ac'].apply(set, meta=('uniprot_ac', 'object'))

            print("  Computing final ID map in a single parallel pass...")
            with dask.config.set(scheduler='threads'), ProgressBar():
                final_map = grouped_series.compute().to_dict()

            print(f"  Successfully built map with {len(final_map):,} external IDs.")
            FileUtils.save_object(final_map, cache_uri)
            return final_map

        except Exception as e:
            print(f"  ERROR: An unexpected error occurred while reading the Parquet dataset: {e}")
            return {}

    @staticmethod
    def create_string_to_uniprot_map(mapping_file_uri: Union[str, Path], cache_uri: Union[str, Path]) -> Dict[str, str]:
        """
        Parses a custom mapping file to create a direct dictionary from STRING IDs
        to UniProt Accession numbers, using a cache to avoid re-computation.
        """
        fs, cache_path = fs_manager.get_fs_and_path(str(cache_uri))
        if fs.exists(cache_path):
            print(f"  Loading cached STRING-to-UniProt map from: {cache_uri}")
            cached_map = FileUtils.load_object(cache_uri)
            if cached_map is not None:
                return cached_map
            print("  Warning: Cached map was corrupted. Re-computing...")

        _, mapping_path = fs_manager.get_fs_and_path(str(mapping_file_uri))
        if not fs.exists(mapping_path):
            print(f"  ERROR: STRING mapping file not found at '{mapping_file_uri}'.")
            return {}

        print(f"  Building STRING-to-UniProt map from '{os.path.basename(mapping_path)}'...")
        string_to_uniprot_map = {}
        try:
            with fs.open(mapping_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="  Parsing STRING mapping file"):
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        uniprot_ac_raw = parts[0]
                        uniprot_ac = uniprot_ac_raw.split('|')[0]
                        string_id = parts[2]
                        string_to_uniprot_map[string_id] = uniprot_ac

            print(f"  Successfully built map with {len(string_to_uniprot_map):,} STRING-to-UniProt mappings.")
            FileUtils.save_object(string_to_uniprot_map, cache_uri)
            return string_to_uniprot_map
        except Exception as e:
            print(f"  ERROR: Failed to parse STRING mapping file '{mapping_file_uri}': {e}")
            return {}
