# ==============================================================================
# MODULE: data_ingestion/ground_truth_loader.py
# PURPOSE: Loads and processes ground truth interaction files (BioGRID, STRING).
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from pathlib import Path
from typing import Set, Tuple, Union, List

import pandas as pd
from tqdm.auto import tqdm

from data_ingestion.data_utils import DataUtils
from data_ingestion.fs.file_utils import FileUtils
from data_ingestion.id_mapper import IDMapper
from data_ingestion.fs.file_system_manager import fs_manager


class GroundTruthLoader:
    """Loads and caches ground-truth protein-protein interactions."""

    @staticmethod
    def load_biogrid_interactions_mapped(biogrid_uri: Union[str, Path], idmapping_parquet_uri: Union[str, Path], cache_dir_uri: Union[str, Path], biogrid_cache_uri: Union[str, Path],
                                         id_types_to_keep: List[str], chunk_size: int = 100000, id_map_blocksize_gb: float = 10.0) -> Set[Tuple[str, str]]:
        """
        Loads protein-protein interactions from a BioGRID TAB3 file, mapping IDs to UniProt ACs.
        """
        DataUtils.print_header("Step: Loading and Mapping BioGRID Ground Truth Interactions")

        fs, biogrid_cache_path = fs_manager.get_fs_and_path(str(biogrid_cache_uri))
        if fs.exists(biogrid_cache_path):
            print(f"  Loading pre-computed BioGRID interactions from cache: {biogrid_cache_uri}")
            cached_interactions = FileUtils.load_object(biogrid_cache_uri)
            if cached_interactions is not None:
                print(f"  Successfully loaded {len(cached_interactions):,} cached interactions.")
                return cached_interactions
            print("  Warning: Failed to load cached BioGRID interactions. Recomputing...")

        _, biogrid_path = fs_manager.get_fs_and_path(str(biogrid_uri))
        if not fs.exists(biogrid_path):
            print(f"  Warning: BioGRID file not found at '{biogrid_uri}'. Cannot load ground truth.")
            return set()

        id_map_cache_uri = f"{str(cache_dir_uri).rstrip('/')}/biogrid_to_uniprot_id_map.pkl"
        id_map = IDMapper.load_id_Mappings(idmapping_parquet_uri, id_map_cache_uri, id_types_to_keep, blocksize_gb=id_map_blocksize_gb)
        if not id_map:
            print("  ERROR: Failed to load or build the ID mapping dictionary. Cannot map BioGRID interactions.")
            return set()

        interactions = set()
        print(f"  Processing BioGRID file '{os.path.basename(biogrid_path)}' in chunks...")

        cols_to_use = [
            'Official Symbol Interactor A', 'Official Symbol Interactor B',
            'Synonyms Interactor A', 'Synonyms Interactor B',
            'Experimental System Type'
        ]

        try:
            with fs.open(biogrid_path, 'rt', encoding='utf-8') as f:
                chunk_iterator = pd.read_csv(f, sep='\t', iterator=True, chunksize=chunk_size, usecols=lambda c: c in cols_to_use, low_memory=False, dtype=str)

                for chunk in tqdm(chunk_iterator, desc="  Mapping BioGRID chunks"):
                    physical_chunk = chunk[chunk['Experimental System Type'] == 'physical'].copy()
                    if physical_chunk.empty:
                        continue

                    for _, row in physical_chunk.iterrows():
                        ids_a = {str(row['Official Symbol Interactor A'])} | set(str(row['Synonyms Interactor A']).split('|'))
                        ids_b = {str(row['Official Symbol Interactor B'])} | set(str(row['Synonyms Interactor B']).split('|'))

                        uniprots_a = {up_id for bg_id in ids_a if bg_id in id_map for up_id in id_map[bg_id]}
                        uniprots_b = {up_id for bg_id in ids_b if bg_id in id_map for up_id in id_map[bg_id]}

                        for u_a in uniprots_a:
                            for u_b in uniprots_b:
                                if u_a != u_b:
                                    interactions.add(tuple(sorted((u_a, u_b))))

        except Exception as e:
            print(f"  ERROR: An error occurred while processing the BioGRID file: {e}")
            return set()

        print(f"  Successfully mapped and loaded {len(interactions):,} unique UniProt interaction pairs from BioGRID.")
        FileUtils.save_object(interactions, biogrid_cache_uri)
        return interactions

    @staticmethod
    def load_string_ground_truth(string_links_uri: Union[str, Path], string_mapping_uri: Union[str, Path], cache_dir_uri: Union[str, Path], string_cache_uri: Union[str, Path], score_threshold: float) -> Set[Tuple[str, str]]:
        """
        Loads protein-protein interactions from a STRING DB links file, mapping IDs
        to UniProt ACs using a dedicated mapping file and filtering by a confidence score.
        """
        DataUtils.print_header("Step: Loading and Mapping STRING Ground Truth Interactions")

        fs, string_cache_path = fs_manager.get_fs_and_path(str(string_cache_uri))
        if fs.exists(string_cache_path):
            print(f"  Loading pre-computed STRING interactions from cache: {string_cache_uri}")
            cached_interactions = FileUtils.load_object(string_cache_uri)
            if cached_interactions is not None:
                print(f"  Successfully loaded {len(cached_interactions):,} cached interactions.")
                return cached_interactions
            print("  Warning: Cached STRING interactions file was corrupted. Re-computing...")

        id_map_cache_uri = f"{str(cache_dir_uri).rstrip('/')}/string_to_uniprot_id_map.pkl"
        string_to_uniprot_map = IDMapper.create_string_to_uniprot_map(string_mapping_uri, id_map_cache_uri)
        if not string_to_uniprot_map:
            print("  ERROR: Failed to build STRING-to-UniProt ID map. Cannot load STRING interactions.")
            return set()

        _, links_path = fs_manager.get_fs_and_path(str(string_links_uri))
        if not fs.exists(links_path):
            print(f"  ERROR: STRING links file not found at '{string_links_uri}'.")
            return set()

        interactions = set()
        print(f"  Processing STRING links file '{os.path.basename(links_path)}' with score threshold >= {score_threshold}...")
        try:
            with fs.open(links_path, 'rt', encoding='utf-8') as f:
                header = f.readline()
                for line in tqdm(f, desc="  Mapping STRING interactions"):
                    parts = line.strip().split(' ')
                    if len(parts) < 3:
                        continue
                    try:
                        score = int(parts[2])
                        if score >= score_threshold:
                            p1_string, p2_string = parts[0], parts[1]
                            u1 = string_to_uniprot_map.get(p1_string)
                            u2 = string_to_uniprot_map.get(p2_string)
                            if u1 and u2 and u1 != u2:
                                interactions.add(tuple(sorted((u1, u2))))
                    except ValueError:
                        continue

            print(f"  Successfully mapped and loaded {len(interactions):,} unique UniProt interaction pairs from STRING.")
            FileUtils.save_object(interactions, string_cache_uri)
            return interactions
        except Exception as e:
            print(f"  ERROR: An error occurred while processing the STRING links file: {e}")
            return set()
