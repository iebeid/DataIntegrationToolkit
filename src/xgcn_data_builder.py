# ==============================================================================
# MODULE: data_ingestion/xgcn_data_builder.py
# PURPOSE: Prepares initial node features and targets for XGCN training from
#          DirectedNgramGraph instances.
# VERSION: 2.1 (Moved to data_ingestion stage for strict data-prep decoupling)
# AUTHOR: Islam Ebeid
# ==============================================================================

from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import diags, lil_matrix
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm

from configuration.config import Config
from data_ingestion.ngram_graph import DirectedNgramGraph


class XGCNDataBuilder:
    """
    A static helper class responsible for preparing the initial node features (X)
    and target labels (Y) required to train the GNN at each level of the hierarchy.
    """

    @staticmethod
    def prepare_gcn_task_and_features(
        current_graph: DirectedNgramGraph,
        n_val: int,
        prev_n_embeddings_dict: Optional[Dict[str, np.ndarray]],
        prev_n_val: Optional[int],
        config: Config,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, str]:
        num_nodes = current_graph.num_nodes
        if num_nodes == 0:
            return None, None, 0, "empty_graph"

        print(f"    Preparing initial node features for {n_val}-gram graph ({num_nodes:,} nodes)...")
        if n_val == 1 or not prev_n_embeddings_dict:
            dim = config.GCN_INITIAL_FEATURE_DIM_N1
            print(f"      - Level 1: Initializing random normal features of dimension {dim}.")
            initial_features = torch.randn((num_nodes, dim), dtype=torch.float32, device=device)
        else:
            prev_dim = next(iter(prev_n_embeddings_dict.values())).shape[0]
            print(f"      - Level {n_val}: Pooling features from {prev_n_val}-gram embeddings (dim={prev_dim})...")
            pooled_features = np.zeros((num_nodes, prev_dim), dtype=np.float32)
            default_vec = np.zeros(prev_dim, dtype=np.float32)

            for idx, ngram_node in enumerate(
                tqdm(current_graph.nodes, desc=f"      Pooling n={prev_n_val} features", leave=False)
            ):
                sub_ngrams = [ngram_node[i : i + prev_n_val] for i in range(len(ngram_node) - prev_n_val + 1)]
                sub_vecs = [prev_n_embeddings_dict.get(sub, default_vec) for sub in sub_ngrams]
                if sub_vecs:
                    if config.SUB_NGRAM_POOLING_STRATEGY == "mean":
                        pooled_features[idx] = np.mean(sub_vecs, axis=0)
                    elif config.SUB_NGRAM_POOLING_STRATEGY == "max":
                        pooled_features[idx] = np.max(sub_vecs, axis=0)
                    else:
                        pooled_features[idx] = np.sum(sub_vecs, axis=0)

            initial_features = torch.from_numpy(pooled_features).to(device)

        task = config.NGRAM_GNN_TRAINING_TASK
        print(f"    Preparing target labels for self-supervised task: '{task}'...")

        if task == "next_Save_distribution":
            adj_coo = current_graph.adj_raw.coalesce().cpu()
            rows, cols = adj_coo.indices().numpy()
            vals = adj_coo.values().numpy()

            adj_lil = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            adj_lil[rows, cols] = vals
            adj_csr = adj_lil.tocsr()

            row_sums = np.array(adj_csr.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_row_sums = 1.0 / row_sums
            prob_matrix_csr = diags(inv_row_sums) @ adj_csr

            target_dim = min(config.SVD_TARGET_COMPONENTS, num_nodes - 1)
            if target_dim > 0 and num_nodes > target_dim:
                print(
                    f"      - Compressing ({num_nodes}x{num_nodes}) transition matrix to {target_dim} dims using TruncatedSVD..."
                )
                svd = TruncatedSVD(n_components=target_dim, random_state=config.RANDOM_SEED)
                compressed_targets = svd.fit_transform(prob_matrix_csr)
                targets = torch.from_numpy(compressed_targets.astype(np.float32)).to(device)
                num_classes = target_dim
                actual_task = "svd_Target_regression"
            else:
                print("      - Graph is small; using full transition probability matrix as target.")
                targets = torch.from_numpy(prob_matrix_csr.toarray()).to(device)
                num_classes = num_nodes
                actual_task = "next_Save_distribution"

        elif task == "degree_binned_classification":
            num_bins = config.DEGREE_BINNING_NUM_BINS
            degrees = (current_graph.in_degree + current_graph.out_degree).cpu().numpy()
            _, bin_edges = np.histogram(degrees, bins=num_bins)
            labels = np.digitize(degrees, bin_edges[:-1]) - 1
            labels = np.clip(labels, 0, num_bins - 1)
            targets = torch.from_numpy(labels).long().to(device)
            num_classes = num_bins
            actual_task = "degree_binned_classification"

        elif task == "contextual_nextjs_prediction":
            top_k = min(config.CONTEXTUAL_PREDICTION_TOP_K, num_nodes)
            adj_coo = current_graph.adj_raw.coalesce().cpu()
            cols = adj_coo.indices()[1].numpy()
            vals = adj_coo.values().numpy()
            col_counts = Counter()
            for c, v in zip(cols, vals):
                col_counts[c] += v
            most_common_nodes = [node_idx for node_idx, _ in col_counts.most_common(top_k)]
            top_node_to_class = {node_idx: i for i, node_idx in enumerate(most_common_nodes)}

            multi_hot_targets = np.zeros((num_nodes, top_k), dtype=np.float32)
            rows = adj_coo.indices()[0].numpy()
            for r, c in zip(rows, cols):
                if c in top_node_to_class:
                    multi_hot_targets[r, top_node_to_class[c]] = 1.0

            targets = torch.from_numpy(multi_hot_targets).to(device)
            num_classes = top_k
            actual_task = "contextual_nextjs_prediction"

        else:
            raise ValueError(f"Unknown NGRAM_GNN_TRAINING_TASK: '{task}'")

        print(f"    Task preparation complete. Target shape: {targets.shape}, Output dim: {num_classes}")
        return initial_features, targets, num_classes, actual_task
