# ==============================================================================
# MODULE: data_ingestion/ngram_graph.py
# PURPOSE: Directed weighted n-gram graph representation and matrix normalizations.
# VERSION: 2.0 (Reorganized into data_ingestion stage)
# AUTHOR: Islam Ebeid
# ==============================================================================

from collections import Counter
from typing import Dict, Any, List, Set

import numpy as np
import torch
from tqdm.auto import tqdm

from data_ingestion.base_graph import Graph


class DirectedNgramGraph(Graph):
    """
    Represents a directed, weighted n-gram graph. This class computes and stores
    all adjacency and degree matrices required by the DirectGCN model using
    memory-efficient sparse tensors.
    """

    def __init__(self, n: int, nodes: List[str], Associated_proteins: Dict[int, Set[str]], **kwargs):
        super().__init__(n, nodes, Associated_proteins, **kwargs)
        print(f"    (DirectedNgramGraph initialization complete for n={self.n})")

    @property
    def num_edges(self) -> int:
        return self.adj_raw._nnz()

    def _initialize_matrices(self, **kwargs):
        """
        Computes all required graph matrices using sparse tensor operations.
        Can be initialized from raw `edges` or from `preloaded_matrices`.
        """
        if 'preloaded_matrices' in kwargs:
            print("    Initializing graph from pre-loaded matrices...")
            matrices = kwargs['preloaded_matrices']
            self.adj_raw = matrices.get('adj_raw').to(self.device)
            self.mathcal_A_in = matrices.get('mathcal_A_in').to(self.device)
            self.mathcal_A_out = matrices.get('mathcal_A_out').to(self.device)
            self.adj_undirected_norm = matrices.get('adj_undirected_norm').to(self.device)
            self.adj_homo_norm = matrices.get('adj_homo_norm').to(self.device)
            self.adj_hetero_norm = matrices.get('adj_hetero_norm').to(self.device)
            self.edge_Stats = kwargs.get('metadata', {}).get('edge_weight_stats', {})
            return

        edges: Counter = kwargs.get('edges', Counter())
        build_homo_hetero: bool = kwargs.get('build_homo_hetero', True)

        if self.num_nodes == 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            empty_shape = (0, 0)
            self.adj_raw = torch.sparse_coo_tensor(empty_indices, empty_values, empty_shape, device=self.device)
            self.mathcal_A_in = self.adj_raw
            self.mathcal_A_out = self.adj_raw
            self.adj_undirected_norm = self.adj_raw
            self.adj_homo_norm = self.adj_raw
            self.adj_hetero_norm = self.adj_raw
            self.edge_Stats = {}
            return

        shape = (self.num_nodes, self.num_nodes)

        print("    1/4: Building raw sparse adjacency matrix (A)...")
        if edges:
            edge_list = list(edges.keys())
            values = list(edges.values())
            indices = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
            values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
            self.adj_raw = torch.sparse_coo_tensor(indices, values_tensor, shape, device=self.device).coalesce()
            weights = values_tensor.cpu().numpy()
            self.edge_Stats = {
                "min": float(np.min(weights)), "max": float(np.max(weights)),
                "mean": float(np.mean(weights)), "std": float(np.std(weights))
            }
        else:
            self.adj_raw = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], shape, device=self.device)
            self.edge_Stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        print("    2/4: Building normalized directed matrices (mathcal_A_in, mathcal_A_out)...")
        eye_indices = torch.arange(0, self.num_nodes, dtype=torch.long, device=self.device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(self.num_nodes, dtype=torch.float32, device=self.device)
        I = torch.sparse_coo_tensor(eye_indices, eye_values, shape, device=self.device)
        adj_with_self_loops = (self.adj_raw + I).coalesce()

        self.out_degree = torch.sparse.sum(adj_with_self_loops, dim=1).to_dense()
        self.in_degree = torch.sparse.sum(adj_with_self_loops, dim=0).to_dense()

        in_deg_inv_sqrt = self.in_degree.pow(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)

        out_deg_inv_sqrt = self.out_degree.pow(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)

        rows, cols = adj_with_self_loops.indices()
        vals = adj_with_self_loops.values()

        vals_in_norm = vals * in_deg_inv_sqrt[rows] * in_deg_inv_sqrt[cols]
        self.mathcal_A_in = torch.sparse_coo_tensor(adj_with_self_loops.indices(), vals_in_norm, shape, device=self.device).coalesce()

        vals_out_norm = vals * out_deg_inv_sqrt[rows] * out_deg_inv_sqrt[cols]
        self.mathcal_A_out = torch.sparse_coo_tensor(adj_with_self_loops.indices(), vals_out_norm, shape, device=self.device).coalesce()

        print("    3/4: Building normalized undirected matrix (A_undir_norm)...")
        adj_raw_t = self.adj_raw.transpose(0, 1).coalesce()
        adj_undirected = (self.adj_raw + adj_raw_t).coalesce()
        self.adj_undirected_norm = self._normalize_undirected_sparse(adj_undirected, I)

        if build_homo_hetero:
            print("    4/4: Building normalized homophilic and heterophilic matrices...")
            if self.adj_raw._nnz() > 0 and self.Associated_proteins:
                homo_edges = Counter()
                hetero_edges = Counter()
                for (u, v), weight in tqdm(edges.items(), desc="      Calculating Jaccard similarity for edges", leave=False):
                    prots_u = self.Associated_proteins.get(u, set())
                    prots_v = self.Associated_proteins.get(v, set())
                    intersection = len(prots_u.intersection(prots_v))
                    union = len(prots_u.union(prots_v))
                    jaccard = intersection / union if union > 0 else 0.0

                    if jaccard > 0:
                        homo_edges[(u, v)] += weight * jaccard
                        homo_edges[(v, u)] += weight * jaccard
                    if (1.0 - jaccard) > 0:
                        hetero_edges[(u, v)] += weight * (1.0 - jaccard)
                        hetero_edges[(v, u)] += weight * (1.0 - jaccard)

                adj_homo = self._counter_to_sparse(homo_edges, shape)
                adj_hetero = self._counter_to_sparse(hetero_edges, shape)

                self.adj_homo_norm = self._normalize_undirected_sparse(adj_homo, I)
                self.adj_hetero_norm = self._normalize_undirected_sparse(adj_hetero, I)
            else:
                self.adj_homo_norm = self.adj_raw
                self.adj_hetero_norm = self.adj_raw
        else:
            print("    4/4: Skipping homophilic and heterophilic matrix construction as configured.")
            empty_sparse = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], shape, device=self.device)
            self.adj_homo_norm = empty_sparse
            self.adj_hetero_norm = empty_sparse

    def _counter_to_sparse(self, edge_counter: Counter, shape: tuple) -> torch.Tensor:
        if not edge_counter:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], shape, device=self.device)
        indices = torch.tensor(list(edge_counter.keys()), dtype=torch.long, device=self.device).t()
        values = torch.tensor(list(edge_counter.values()), dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(indices, values, shape, device=self.device).coalesce()

    def _normalize_undirected_sparse(self, adj_matrix: torch.Tensor, identity_matrix: torch.Tensor) -> torch.Tensor:
        if adj_matrix._nnz() == 0:
            return adj_matrix
        adj_with_loops = (adj_matrix + identity_matrix).coalesce()
        degree = torch.sparse.sum(adj_with_loops, dim=1).to_dense()
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        rows, cols = adj_with_loops.indices()
        vals = adj_with_loops.values()
        norm_vals = vals * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
        return torch.sparse_coo_tensor(adj_with_loops.indices(), norm_vals, adj_matrix.shape, device=self.device).coalesce()

    def get_STATISTICS(self) -> Dict[str, Any]:
        if self.num_nodes == 0:
            return {"num_nodes": 0, "num_edges": 0}

        if not hasattr(self, 'in_degree') or not hasattr(self, 'out_degree'):
            eye_indices = torch.arange(0, self.num_nodes, dtype=torch.long, device=self.device).unsqueeze(0).repeat(2, 1)
            eye_values = torch.ones(self.num_nodes, dtype=torch.float32, device=self.device)
            I = torch.sparse_coo_tensor(eye_indices, eye_values, (self.num_nodes, self.num_nodes), device=self.device)
            adj_with_self_loops = (self.adj_raw + I).coalesce()
            in_degrees = (torch.sparse.sum(adj_with_self_loops, dim=0).to_dense() - 1).cpu().numpy()
            out_degrees = (torch.sparse.sum(adj_with_self_loops, dim=1).to_dense() - 1).cpu().numpy()
        else:
            in_degrees = (self.in_degree - 1).cpu().numpy()
            out_degrees = (self.out_degree - 1).cpu().numpy()

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.num_edges / (self.num_nodes ** 2) if self.num_nodes > 0 else 0.0,
            "in_degree_stats": {
                "min": np.min(in_degrees), "max": np.max(in_degrees),
                "mean": np.mean(in_degrees), "std": np.std(in_degrees)
            },
            "out_degree_stats": {
                "min": np.min(out_degrees), "max": np.max(out_degrees),
                "mean": np.mean(out_degrees), "std": np.std(out_degrees)
            },
            "edge_weight_stats": self.edge_Stats
        }

    def _get_matrices_to_save(self) -> Dict[str, torch.Tensor]:
        return {
            'adj_raw': self.adj_raw,
            'mathcal_A_in': self.mathcal_A_in,
            'mathcal_A_out': self.mathcal_A_out,
            'adj_undirected_norm': self.adj_undirected_norm,
            'adj_homo_norm': self.adj_homo_norm,
            'adj_hetero_norm': self.adj_hetero_norm,
        }

    @classmethod
    def _get_matrix_names_to_load(cls) -> List[str]:
        return [
            'adj_raw', 'mathcal_A_in', 'mathcal_A_out',
            'adj_undirected_norm', 'adj_homo_norm', 'adj_hetero_norm'
        ]

    @classmethod
    def _from_loaded_data(cls, n: int, nodes: List[str], loaded_matrices: Dict[str, torch.Tensor], metadata: Dict) -> 'DirectedNgramGraph':
        return cls(n=n, nodes=nodes, Associated_proteins={}, preloaded_matrices=loaded_matrices, metadata=metadata)
