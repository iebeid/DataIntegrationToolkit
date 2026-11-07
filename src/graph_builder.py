# ==============================================================================
# MODULE: data_ingestion/graph_builder.py
# PURPOSE: Builds a hierarchy of DirectedNgramGraph objects from UniRef FASTA
#          protein sequences using Dask and saves them to disk.
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

from collections import Counter
from typing import Dict, Set, List

import dask
import dask.bag as db
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

from configuration.config import Config
from data_ingestion.ngram_graph import DirectedNgramGraph
from data_ingestion.data_utils import DataUtils
from data_ingestion.fasta_loader import FastaUtils
from data_ingestion.fs.file_utils import FileUtils
from data_ingestion.dask_helpers import ProtgramDaskHelpers
from data_ingestion.fs.file_system_manager import fs_manager


class ProtGramDataBuilder:
    """
    A dedicated builder class to construct the hierarchy of n-gram graphs
    required for the ProtGram-DirectGCN pipeline. It uses Dask to parallelize
    the graph construction process for scalability and saves the graphs to disk.
    """

    def __init__(self, config: Config):
        self.config = config
        self.sequences: Dict[str, str] = {}
        self.graphs: List[DirectedNgramGraph] = []

    def build_and_save_graphs(self) -> List[DirectedNgramGraph]:
        """
        Orchestrates the end-to-end process of loading sequences, building
        each n-gram graph in the hierarchy using Dask, and saving them to disk.
        """
        DataUtils.print_header("STAGE 1 (DATA INGESTION): BUILDING N-GRAM GRAPH HIERARCHY")

        all_graphs_exist = True
        loaded_graphs = []
        fs, _ = fs_manager.get_fs_and_path(self.config.PROCESSED_DATA_DIR)

        for n in range(1, self.config.N_GRAM_MAX_N + 1):
            graph_dir_uri = f"{self.config.PROCESSED_DATA_DIR.rstrip('/')}/ngram_graph_n{n}"
            _, graph_dir_path = fs_manager.get_fs_and_path(graph_dir_uri)
            if not fs.isdir(graph_dir_path):
                all_graphs_exist = False
                break

        if all_graphs_exist:
            print("  Found complete set of pre-computed n-gram graphs on disk. Loading them directly...")
            try:
                for n in range(1, self.config.N_GRAM_MAX_N + 1):
                    graph_dir_uri = f"{self.config.PROCESSED_DATA_DIR.rstrip('/')}/ngram_graph_n{n}"
                    graph = DirectedNgramGraph.load(graph_dir_uri)
                    loaded_graphs.append(graph)
                print("  Successfully loaded all pre-computed graphs.")
                return loaded_graphs
            except Exception as e:
                print(f"  Warning: Failed to load pre-computed graphs ({e}). Re-building from scratch...")

        self.sequences = FastaUtils.load_sequences_from_fasta(
            self.config.UNIREF_FASTA_PATH,
            max_proteins=self.config.MAX_PROTEINS_FOR_TRAINING,
            random_seed=self.config.RANDOM_SEED
        )
        if not self.sequences:
            raise RuntimeError("Failed to load any sequences from the FASTA file. Cannot build graphs.")

        seq_df = pd.DataFrame(list(self.sequences.items()), columns=['id', 'seq'])
        ddf = dd.from_pandas(seq_df, npartitions=self.config.DASK_NPARTITIONS)
        print(f"  Converted {len(self.sequences):,} sequences into a Dask DataFrame with {ddf.npartitions} partitions.")

        all_node_associations: Dict[int, Dict[int, Set[str]]] = {}

        for n in range(1, self.config.N_GRAM_MAX_N + 1):
            DataUtils.print_header(f"Building {n}-gram Graph")

            print("  Step 1/3: Discovering unique n-gram nodes across all sequences...")
            with dask.config.set(scheduler='threads'), ProgressBar():
                nodes_ddf = ddf.map_partitions(
                    ProtgramDaskHelpers.extract_nodes_from_partition, n=n, meta={'ngram': 'object'}
                )
                unique_nodes = sorted(nodes_ddf['ngram'].drop_duplicates().compute().tolist())

            if not unique_nodes:
                print(f"  Warning: No {n}-grams found. Creating an empty graph for n={n}.")
                graph = DirectedNgramGraph(n=n, nodes=[], Associated_proteins={})
                self.graphs.append(graph)
                continue

            print(f"  Found {len(unique_nodes):,} unique {n}-gram nodes.")
            node_to_idx = {node: i for i, node in enumerate(unique_nodes)}

            print("  Step 2/3: Counting directed edges and mapping nodes to proteins in parallel...")
            with dask.config.set(scheduler='threads'), ProgressBar():
                results_bag = db.from_delayed([
                    dask.delayed(ProtgramDaskHelpers.extract_edges_and_associations_from_partition)(part, n=n, node_to_idx=node_to_idx)
                    for part in ddf.to_delayed()
                ]).flatten()

                def combine_counters_and_dicts(tuple1, tuple2):
                    edges1, assoc1 = tuple1
                    edges2, assoc2 = tuple2
                    edges1.update(edges2)
                    for k, v in assoc2.items():
                        assoc1.setdefault(k, set()).update(v)
                    return edges1, assoc1

                final_edges, final_associations = results_bag.fold(
                    combine_counters_and_dicts, initial=(Counter(), {})
                ).compute()

            print(f"  Aggregated {len(final_edges):,} unique directed edges.")
            all_node_associations[n] = {k: list(v) for k, v in final_associations.items()}

            print("  Step 3/3: Initializing DirectedNgramGraph and computing normalized matrices...")
            graph = DirectedNgramGraph(
                n=n,
                nodes=unique_nodes,
                Associated_proteins=final_associations,
                edges=final_edges,
                build_homo_hetero=self.config.DIRECTGCN_USE_HOMO_HETERO_PATHS
            )
            graph.print_Summary()
            graph_save_uri = f"{self.config.PROCESSED_DATA_DIR.rstrip('/')}/ngram_graph_n{n}"
            graph.save(graph_save_uri)
            self.graphs.append(graph)

        FileUtils.save_json(all_node_associations, self.config.NODE_TO_PROTEINS_MAP_PATH)
        return self.graphs
