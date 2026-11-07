# ==============================================================================
# MODULE: data_ingestion/dask_helpers.py
# PURPOSE: Static helper methods for Dask-based n-gram graph construction.
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

from collections import Counter
from typing import Dict

import pandas as pd


class ProtgramDaskHelpers:
    """
    A collection of static methods designed to be mapped over Dask Bag or
    DataFrame partitions for parallel graph construction.
    """

    @staticmethod
    def extract_nodes_from_partition(df_partition: pd.DataFrame, n: int) -> pd.DataFrame:
        """Extracts unique n-grams from a partition of sequences."""
        ngrams = set()
        for seq in df_partition['seq']:
            if len(seq) >= n:
                for i in range(len(seq) - n + 1):
                    ngrams.add(seq[i:i + n])
        return pd.DataFrame({'ngram': list(ngrams)})

    @staticmethod
    def extract_edges_and_associations_from_partition(df_partition: pd.DataFrame, n: int, node_to_idx: Dict[str, int]):
        """
        Processes a partition of sequences to extract edges and node-to-protein
        associations using a pre-computed node_to_idx map.
        """
        edge_counts = Counter()
        associations = {}

        for _, row in df_partition.iterrows():
            prot_id, seq = row['id'], row['seq']
            if len(seq) < n:
                continue

            ngrams_in_seq = [seq[i:i + n] for i in range(len(seq) - n + 1)]

            for ngram in set(ngrams_in_seq):
                node_idx = node_to_idx.get(ngram)
                if node_idx is not None:
                    if node_idx not in associations:
                        associations[node_idx] = set()
                    associations[node_idx].add(prot_id)

            if len(ngrams_in_seq) >= 2:
                for i in range(len(ngrams_in_seq) - 1):
                    u_idx = node_to_idx.get(ngrams_in_seq[i])
                    v_idx = node_to_idx.get(ngrams_in_seq[i + 1])
                    if u_idx is not None and v_idx is not None:
                        edge_counts[(u_idx, v_idx)] += 1

        return [(edge_counts, associations)]
