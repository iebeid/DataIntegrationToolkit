# ==============================================================================
# MODULE: data_ingestion/fasta_loader.py
# PURPOSE: Utilities for loading and sampling protein sequences from UniRef FASTA files.
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import random
from pathlib import Path
from typing import Dict, Set, Union, Optional

from Bio import SeqIO
from tqdm.auto import tqdm

from data_ingestion.data_utils import DataUtils
from data_ingestion.fs.file_system_manager import fs_manager


class FastaUtils:
    """A collection of static methods for processing FASTA files."""

    @staticmethod
    def load_sequences_from_fasta(fasta_uri: Union[str, Path], max_proteins: Optional[int] = None, random_seed: int = 42, target_ids: Optional[Set[str]] = None) -> Dict[str, str]:
        """
        Loads protein sequences from a FASTA file into a dictionary from any fsspec-supported location.
        If `max_proteins` is specified, it performs reservoir sampling to select a
        random subset of sequences in a single, memory-efficient pass.
        If `target_ids` is provided, it will only load sequences matching those IDs.
        """
        fs, path = fs_manager.get_fs_and_path(str(fasta_uri))
        DataUtils.print_header("Step: Loading Protein Sequences from FASTA")
        if not fs.exists(path):
            print(f"Error: FASTA file not found at {fasta_uri}")
            return {}

        if target_ids is not None:
            print(f"  Loading sequences for a specific set of {len(target_ids):,} target proteins from '{os.path.basename(path)}'...")
            sequences = {}
            found_count = 0
            num_targets = len(target_ids)
            with fs.open(path, "rt", encoding="utf-8") as handle:
                for record in tqdm(SeqIO.parse(handle, "fasta"), desc="  Scanning FASTA for targets"):
                    parts = record.id.split('|')
                    protein_id = parts[1] if len(parts) > 1 else parts[0].split('_')[1] if '_' in parts[0] else parts[0]
                    if protein_id in target_ids:
                        sequences[protein_id] = str(record.seq).upper()
                        found_count += 1
                        if found_count == num_targets:
                            print("  All target proteins found. Stopping scan early.")
                            break
            print(f"  Successfully loaded {len(sequences):,} of {num_targets:,} target protein sequences.")
            return sequences

        use_sampling = max_proteins is not None and max_proteins > 0
        if use_sampling:
            print(f"  Sampling up to {max_proteins:,} random sequences from '{os.path.basename(path)}' using Reservoir Sampling (seed={random_seed})...")
            random.seed(random_seed)
            reservoir = []
        else:
            print(f"  Loading all sequences from '{os.path.basename(path)}'...")
            sequences = {}

        try:
            with fs.open(path, "rt", encoding="utf-8") as handle:
                for i, record in enumerate(tqdm(SeqIO.parse(handle, "fasta"), desc="  Reading FASTA")):
                    parts = record.id.split('|')
                    protein_id = parts[1] if len(parts) > 1 else parts[0].split('_')[1] if '_' in parts[0] else parts[0]
                    seq = str(record.seq).upper()
                    if not seq:
                        continue

                    if use_sampling:
                        if len(reservoir) < max_proteins:
                            reservoir.append((protein_id, seq))
                        else:
                            j = random.randint(0, i)
                            if j < max_proteins:
                                reservoir[j] = (protein_id, seq)
                    else:
                        sequences[protein_id] = seq

            if use_sampling:
                sequences = dict(reservoir)
            print(f"  Successfully loaded {len(sequences):,} unique protein sequences.")
            return sequences
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred while reading FASTA file {fasta_uri}: {e}")
            return {}
