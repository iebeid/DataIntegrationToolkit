# ==============================================================================
# MODULE: data_ingestion/base_graph.py
# PURPOSE: Defines the abstract base class for graph data structures and disk I/O.
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Set, Union

import numpy as np
import pandas as pd
import torch

from data_ingestion.data_utils import DataUtils
from data_ingestion.fs.file_system_manager import fs_manager


class Graph(ABC):
    """
    Abstract base class for graph representations. It defines the common
    interface and attributes for all graph types in the project and handles
    disk persistence (nodes.parquet, *_indices.npy, *_values.npy, metadata.json).
    """

    def __init__(self, n: int, nodes: List[str], Associated_proteins: Dict[int, Set[str]], **kwargs):
        self.n = n
        self.nodes: List[str] = nodes
        self.node_to_idx: Dict[str, int] = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node: Dict[int, str] = {i: node for i, node in enumerate(self.nodes)}
        self.Associated_proteins: Dict[int, Set[str]] = Associated_proteins
        self.node_attributes: Dict[int, Dict[str, Any]] = {}

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._initialize_matrices(**kwargs)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @abstractmethod
    def _initialize_matrices(self, **kwargs):
        pass

    @abstractmethod
    def get_STATISTICS(self) -> Dict[str, Any]:
        pass

    def print_Summary(self):
        DataUtils.print_header(f"SUMMARY FOR {self.n}-GRAM GRAPH")
        if self.num_nodes == 0:
            print("  Graph is empty.")
            return
        stats = self.get_STATISTICS()
        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                print(f"  - {key.replace('_', ' ').title()}: {value:,}")
            elif isinstance(value, (float, np.floating)):
                print(f"  - {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  - {key.replace('_', ' ').title()}: {value}")

    def save(self, directory_uri: Union[str, Path]):
        """
        Saves the graph's data to a directory in a scalable, language-agnostic format.
        """
        fs, final_path = fs_manager.get_fs_and_path(str(directory_uri))
        print(f"  Saving {self.n}-gram graph to directory: {directory_uri}")

        temp_dir = Path(tempfile.mkdtemp())
        try:
            nodes_df = pd.DataFrame({'node_idx': range(self.num_nodes), 'ngram': self.nodes})
            nodes_df.to_parquet(temp_dir / "nodes.parquet", index=False)

            matrices_to_save = self._get_matrices_to_save()
            for name, tensor in matrices_to_save.items():
                if tensor is not None and tensor._nnz() > 0:
                    coalesced_tensor = tensor.coalesce().cpu()
                    np.save(temp_dir / f"{name}_indices.npy", coalesced_tensor.indices().numpy())
                    np.save(temp_dir / f"{name}_values.npy", coalesced_tensor.values().numpy())

            metadata = self.get_STATISTICS()
            metadata['n'] = self.n
            for k, v in metadata.items():
                if isinstance(v, (np.integer, np.floating)):
                    metadata[k] = v.item()
            with open(temp_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)

            if fs.exists(final_path):
                fs.rm(final_path, recursive=True)
            fs.put(str(temp_dir), final_path, recursive=True)
            print(f"  Graph successfully saved to {directory_uri}")
        except Exception as e:
            print(f"  ERROR: Failed to save graph to {directory_uri}: {e}")
        finally:
            shutil.rmtree(temp_dir)

    @classmethod
    def load(cls, directory_uri: Union[str, Path]) -> 'Graph':
        """Loads a graph from a directory created by the `save` method."""
        fs, path = fs_manager.get_fs_and_path(str(directory_uri))
        if not fs.isdir(path):
            raise FileNotFoundError(f"Graph directory not found at: {directory_uri}")

        print(f"  Loading graph from directory: {directory_uri}")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            fs.get(f"{path.rstrip('/')}/*", str(temp_dir), recursive=True)

            with open(temp_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            n = metadata['n']

            nodes_df = pd.read_parquet(temp_dir / "nodes.parquet")
            nodes = nodes_df.sort_values('node_idx')['ngram'].tolist()
            num_nodes = len(nodes)
            shape = (num_nodes, num_nodes)

            loaded_matrices = {}
            for matrix_name in cls._get_matrix_names_to_load():
                indices_path = temp_dir / f"{matrix_name}_indices.npy"
                values_path = temp_dir / f"{matrix_name}_values.npy"
                if indices_path.exists() and values_path.exists():
                    indices = torch.from_numpy(np.load(indices_path))
                    values = torch.from_numpy(np.load(values_path))
                    loaded_matrices[matrix_name] = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
                else:
                    loaded_matrices[matrix_name] = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], shape, dtype=torch.float32)

            return cls._from_loaded_data(n=n, nodes=nodes, loaded_matrices=loaded_matrices, metadata=metadata)
        finally:
            shutil.rmtree(temp_dir)

    @abstractmethod
    def _get_matrices_to_save(self) -> Dict[str, torch.Tensor]:
        pass

    @classmethod
    @abstractmethod
    def _get_matrix_names_to_load(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def _from_loaded_data(cls, n: int, nodes: List[str], loaded_matrices: Dict[str, torch.Tensor], metadata: Dict) -> 'Graph':
        pass
