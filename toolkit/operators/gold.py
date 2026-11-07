import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolkit.operators.base import BaseOperator
from toolkit.catalog.table import TableMetadata


# ==============================================================================
# SIMILARITY FUNCTIONS (Synthesized from SimilER, DWM65, DWM68)
# ==============================================================================

def compute_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_levenshtein(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
    dist = matrix[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def compute_exact_match(s1: str, s2: str, numeric_rule: bool = True) -> float:
    """Exact match comparison, honoring the numeric token rule from DWM65."""
    clean1 = str(s1).strip().lower()
    clean2 = str(s2).strip().lower()
    if clean1 == clean2:
        return 1.0
    # Clean non-digits if checking numeric-only (e.g. SSN or zip)
    if numeric_rule:
        digits1 = re.sub(r"\D", "", clean1)
        digits2 = re.sub(r"\D", "", clean2)
        if digits1 and digits2 and digits1 == digits2:
            return 1.0
    return 0.0


def compute_composite_similarity(
    rec_a: Dict[str, Any],
    rec_b: Dict[str, Any],
    weights: Dict[str, float],
    metrics: Optional[Dict[str, str]] = None,
) -> float:
    """
    Computes weighted multi-attribute similarity (from DWM68 and SimilER).
    """
    metrics = metrics or {}
    total_weight = 0.0
    accumulated_sim = 0.0

    for attr, weight in weights.items():
        val_a = str(rec_a.get(attr, "")).strip()
        val_b = str(rec_b.get(attr, "")).strip()
        if not val_a or not val_b:
            continue

        metric = metrics.get(attr, "levenshtein").lower()
        if metric == "exact":
            sim = compute_exact_match(val_a, val_b)
        elif metric == "jaccard":
            sim = compute_jaccard(set(val_a.lower().split()), set(val_b.lower().split()))
        else:
            sim = compute_levenshtein(val_a.lower(), val_b.lower())

        accumulated_sim += sim * weight
        total_weight += weight

    return (accumulated_sim / total_weight) if total_weight > 0 else 0.0


# ==============================================================================
# RESOLUTION OPERATOR
# ==============================================================================

class TransitiveClosureOperator(BaseOperator):
    """
    Gold-tier operator: Synthesized from GraphOG, GraphOG_Algorithms, SimilER, and DWM80.
    Calculates pairwise similarity across candidate pairs, builds a similarity graph,
    and applies graph partitioning (Transitive Closure, Louvain, Label Propagation,
    or Greedy Modularity) to resolve entities into canonical clusters.
    """

    def __init__(
        self,
        name: str = "transitive_closure",
        records_table: str = "",
        pairs_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        similarity_metric: str = "jaccard",  # 'jaccard', 'levenshtein', or 'composite'
        threshold: float = 0.5,
        clustering: str = "transitive_closure",  # 'transitive_closure', 'louvain', 'label_propagation', 'greedy_modularity'
        attribute_weights: Optional[Dict[str, float]] = None,
        attribute_metrics: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name)
        self.records_table = records_table
        self.pairs_table = pairs_table
        self.target_name = target_name
        self.key_column = key_column
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        self.clustering = clustering
        self.attribute_weights = attribute_weights
        self.attribute_metrics = attribute_metrics

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        rec_meta = context.catalog.get_table(kwargs.get("records_table", self.records_table))
        pairs_meta = context.catalog.get_table(kwargs.get("pairs_table", self.pairs_table))

        if not rec_meta or not pairs_meta:
            raise ValueError("Records table or pairs table missing from catalog.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "")
        key_col = kwargs.get("key_column", self.key_column)
        threshold = kwargs.get("threshold", self.threshold)
        clustering_algo = kwargs.get("clustering", self.clustering).lower()
        sim_metric = kwargs.get("similarity_metric", self.similarity_metric).lower()
        weights = kwargs.get("attribute_weights", self.attribute_weights)
        attr_metrics = kwargs.get("attribute_metrics", self.attribute_metrics)

        # 1. Load records and candidate pairs
        records_arrow = fs_manager.read_tabular(rec_meta.uri, file_format="parquet")
        pairs_arrow = fs_manager.read_tabular(pairs_meta.uri, file_format="parquet")

        rec_dict = records_arrow.to_pydict()
        pairs_dict = pairs_arrow.to_pydict()
        num_records = records_arrow.num_rows

        if key_col not in records_arrow.column_names:
            key_col = "_row_id" if "_row_id" in records_arrow.column_names else records_arrow.column_names[0]

        # Map RecID -> record representation
        record_map: Dict[str, Dict[str, Any]] = {}
        record_tokens_map: Dict[str, Set[str]] = {}
        record_embedding_map: Dict[str, np.ndarray] = {}
        token_col = "_staged_tokens" if "_staged_tokens" in records_arrow.column_names else None
        has_embeddings = "embedding" in records_arrow.column_names

        for i in range(num_records):
            rid = str(rec_dict[key_col][i])
            record_map[rid] = {c: rec_dict[c][i] for c in records_arrow.column_names}
            if has_embeddings and rec_dict["embedding"][i] is not None:
                record_embedding_map[rid] = np.array(rec_dict["embedding"][i], dtype=np.float32)
            if token_col and rec_dict[token_col][i]:
                record_tokens_map[rid] = set(rec_dict[token_col][i].split())
            else:
                text_content = " ".join([str(rec_dict[c][i]) for c in records_arrow.column_names if not c.startswith("_")])
                record_tokens_map[rid] = set(text_content.lower().split())

        # 2. Evaluate candidate pairs and build similarity graph
        graph = nx.Graph()
        for rid in record_map.keys():
            graph.add_node(rid)

        num_pairs = pairs_arrow.num_rows
        id_a_list = pairs_dict["id_a"]
        id_b_list = pairs_dict["id_b"]
        matching_edges = 0

        for i in range(num_pairs):
            a = str(id_a_list[i])
            b = str(id_b_list[i])
            if a not in record_map or b not in record_map:
                continue

            # Compute similarity based on requested metric
            if sim_metric == "cosine":
                vec_a = record_embedding_map.get(a)
                vec_b = record_embedding_map.get(b)
                if vec_a is not None and vec_b is not None:
                    norm_a = float(np.linalg.norm(vec_a))
                    norm_b = float(np.linalg.norm(vec_b))
                    if norm_a > 1e-9 and norm_b > 1e-9:
                        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                    else:
                        sim = 0.0
                else:
                    sim = 0.0
            elif sim_metric == "euclidean":
                vec_a = record_embedding_map.get(a)
                vec_b = record_embedding_map.get(b)
                if vec_a is not None and vec_b is not None:
                    dist = float(np.linalg.norm(vec_a - vec_b))
                    sim = 1.0 / (1.0 + dist)
                else:
                    sim = 0.0
            elif sim_metric == "composite" and weights:
                sim = compute_composite_similarity(record_map[a], record_map[b], weights, attr_metrics)
            elif sim_metric == "levenshtein":
                text_a = " ".join([str(v) for k, v in record_map[a].items() if not k.startswith("_")])
                text_b = " ".join([str(v) for k, v in record_map[b].items() if not k.startswith("_")])
                sim = compute_levenshtein(text_a, text_b)
            else:
                # Default: Jaccard on tokens
                sim = compute_jaccard(record_tokens_map[a], record_tokens_map[b])

            if sim >= threshold:
                graph.add_edge(a, b, weight=sim)
                matching_edges += 1

        # 3. Graph Partitioning / Community Detection
        if clustering_algo == "louvain":
            # Louvain Community Detection (modularity maximization)
            # Add small self-loops or weights so disconnected nodes form singleton communities
            try:
                communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42))
            except Exception:
                communities = list(nx.connected_components(graph))
        elif clustering_algo == "label_propagation":
            # Label Propagation
            try:
                communities = list(nx.community.label_propagation_communities(graph))
            except Exception:
                communities = list(nx.connected_components(graph))
        elif clustering_algo == "greedy_modularity":
            # Clauset-Newman-Moore greedy modularity
            try:
                communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
            except Exception:
                communities = list(nx.connected_components(graph))
        else:
            # Default: Transitive Closure / Connected Components (GraphOG & Talburt)
            communities = list(nx.connected_components(graph))

        # 4. Map records to canonical cluster IDs
        rec_to_canonical = {}
        cluster_sizes = {}
        for cluster_idx, comp in enumerate(communities):
            canonical_id = f"ENT_{cluster_idx + 1:06d}"
            c_size = len(comp)
            for member in comp:
                rec_to_canonical[member] = canonical_id
                cluster_sizes[member] = c_size

        # 5. Emit Gold Resolved Arrow Table
        canonical_ids = [rec_to_canonical.get(str(rec_dict[key_col][i]), "ENT_UNK") for i in range(num_records)]
        c_sizes = [cluster_sizes.get(str(rec_dict[key_col][i]), 1) for i in range(num_records)]

        gold_table = records_arrow.append_column("canonical_entity_id", pa.array(canonical_ids, type=pa.string()))
        gold_table = gold_table.append_column("entity_cluster_size", pa.array(c_sizes, type=pa.int64()))

        target_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(gold_table, target_uri)

        multi_record_clusters = sum(1 for c in communities if len(c) > 1)

        meta = TableMetadata(
            name=clean_target,
            tier="gold",
            uri=target_uri,
            schema=gold_table.schema,
            row_count=num_records,
            properties={
                "clustering_algorithm": clustering_algo,
                "similarity_metric": sim_metric,
                "threshold": threshold,
                "distinct_resolved_entities": len(communities),
                "matching_edges": matching_edges,
                "multi_record_clusters": multi_record_clusters,
            },
        )
        context.catalog.register_table(meta)
        return meta


class GraphProjectorOperator(BaseOperator):
    """
    Gold-tier operator: Synthesized from DirectedNgramGraph and base_graph.
    Projects tabular or sequence data into a directed n-gram graph topology.
    """

    def __init__(
        self,
        name: str = "graph_projector",
        source_table: str = "",
        target_name: str = "",
        sequence_column: str = "sequence",
        n: int = 3,
        directed: bool = True,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.sequence_column = sequence_column
        self.n = n
        self.directed = directed

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "")
        seq_col = kwargs.get("sequence_column", self.sequence_column)
        n = kwargs.get("n", self.n)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()

        if seq_col not in arrow_table.column_names:
            raise ValueError(f"Sequence column '{seq_col}' not found in table.")

        edge_counter = Counter()
        for seq in pydict[seq_col]:
            if not seq or len(str(seq)) < n + 1:
                continue
            s = str(seq)
            for i in range(len(s) - n):
                src_ngram = s[i : i + n]
                tgt_ngram = s[i + 1 : i + 1 + n]
                edge_counter[(src_ngram, tgt_ngram)] += 1

        src_list = [e[0] for e in edge_counter.keys()]
        tgt_list = [e[1] for e in edge_counter.keys()]
        weights = list(edge_counter.values())

        graph_table = pa.Table.from_arrays(
            [
                pa.array(src_list, type=pa.string()),
                pa.array(tgt_list, type=pa.string()),
                pa.array(weights, type=pa.int64()),
            ],
            names=["source", "target", "weight"],
        )

        target_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(graph_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="gold",
            uri=target_uri,
            schema=graph_table.schema,
            row_count=graph_table.num_rows,
            properties={"n": n, "directed": self.directed, "edges_count": graph_table.num_rows},
        )
        context.catalog.register_table(meta)
        return meta


# ==============================================================================
# GRAPH-BASED ENTITY RESOLUTION & GRAPHOG SUITE (Synthesized from GraphOG,
# GraphOG_Algorithms, GER_CliqueRank_Runner, HITS, and DWM100)
# ==============================================================================

class GraphOGAlgorithms:
    """
    Foundational graph algorithms for entity resolution synthesized from
    src/GraphOG_Algorithms.py, src/GraphOG.py, and src/HITS.py.
    """

    @staticmethod
    def talburt_transitive_closure(edges: List[Tuple[str, str]]) -> List[List[str]]:
        """
        Iterative MapReduce-style key-group transitive closure algorithm
        (Kolb et al. / Talburt Transitive Closure from src/GraphOG_Algorithms.py).
        """
        if not edges:
            return []

        # Bootstrap: add (u, v), (v, u), (u, u), (v, v)
        cluster_list = []
        for u, v in edges:
            cluster_list.append((u, v))
            cluster_list.append((v, u))
            cluster_list.append((u, u))
            cluster_list.append((v, v))

        pair_list = sorted(list(set(cluster_list)))
        more_work_to_do = True

        while more_work_to_do:
            more_work_to_do = False
            caboose = ("\uffff", "\uffff")
            pair_list.append(caboose)
            key_group = []
            cluster_list = []

            for j in range(len(pair_list) - 1):
                cur_pair = pair_list[j]
                next_pair = pair_list[j + 1]
                key_group.append(cur_pair)

                if cur_pair[0] != next_pair[0]:
                    first_group_pair = key_group[0]
                    first_key = first_group_pair[0]
                    first_val = first_group_pair[1]
                    kg_size = len(key_group)

                    if first_key > first_val:
                        if kg_size > 1:
                            more_work_to_do = True
                            for k in range(kg_size):
                                gp_val = key_group[k][1]
                                cluster_list.append((first_val, gp_val))
                                cluster_list.append((gp_val, first_val))
                            last_val = key_group[kg_size - 1][1]
                            if first_key < last_val:
                                cluster_list.append(first_group_pair)
                    else:
                        cluster_list.extend(key_group)
                    key_group = []

            pair_list = sorted(list(set(cluster_list)))

        cluster_ids = {}
        for p in pair_list:
            cluster_ids[p[1]] = p[0]
        node_label_profile = defaultdict(list)
        for key, val in sorted(cluster_ids.items()):
            node_label_profile[val].append(key)
        return [sorted(list(set(members))) for members in node_label_profile.values() if members]

    @staticmethod
    def akef_quick_connected_components(edges: List[Tuple[str, str]], all_nodes: Optional[List[str]] = None) -> List[List[str]]:
        """Disjoint-Set Union (DSU) fast connected component detection."""
        parent: Dict[str, str] = {}

        def find(i: str) -> str:
            path = []
            while parent.get(i, i) != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i: str, j: str):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        if all_nodes:
            for n in all_nodes:
                parent[n] = n

        for u, v in edges:
            if u not in parent:
                parent[u] = u
            if v not in parent:
                parent[v] = v
            union(u, v)

        comp_dict = defaultdict(list)
        for n in parent.keys():
            comp_dict[find(n)].append(n)
        return [sorted(c) for c in comp_dict.values()]

    @staticmethod
    def akef_unbiased_random_walk(graph: nx.Graph, start_node: str, walk_length: int = 10, seed: int = 42) -> List[str]:
        """Unbiased random walker on graph topology."""
        rng = np.random.RandomState(seed)
        walk = [start_node]
        cur = start_node
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(cur))
            if not neighbors:
                break
            cur = rng.choice(neighbors)
            walk.append(cur)
        return walk

    @staticmethod
    def akef_weighted_biased_random_walk(
        graph: nx.Graph,
        start_node: str,
        walk_length: int = 10,
        weight_attribute: str = "weight",
        seed: int = 42,
    ) -> List[str]:
        """Transition-probability weighted random walk."""
        rng = np.random.RandomState(seed)
        walk = [start_node]
        cur = start_node
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(cur))
            if not neighbors:
                break
            weights = np.array([graph[cur][nbr].get(weight_attribute, 1.0) for nbr in neighbors], dtype=np.float64)
            total = np.sum(weights)
            probs = weights / total if total > 0 else np.ones(len(neighbors)) / len(neighbors)
            cur = rng.choice(neighbors, p=probs)
            walk.append(cur)
        return walk

    @staticmethod
    def spectral_fiedler_clustering(
        graph: nx.Graph,
        threshold_mode: str = "sign",  # 'sign' (> 0 vs <= 0) or 'median'
    ) -> List[List[str]]:
        """
        Spectral Fiedler graph partitioning: Computes normalized Laplacian,
        finds eigenvector corresponding to the second smallest eigenvalue (Fiedler vector),
        and partitions the graph into clusters.
        """
        nodes = list(graph.nodes())
        if len(nodes) <= 2:
            return [nodes]

        try:
            import scipy.linalg
            L = nx.laplacian_matrix(graph, nodelist=nodes).todense()
            eigenvals, eigenvecs = scipy.linalg.eigh(L)
            sorted_indices = np.argsort(eigenvals.real)
            fiedler_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            fiedler_vec = eigenvecs[:, fiedler_idx].real

            if threshold_mode == "median":
                cut = float(np.median(fiedler_vec))
            else:
                cut = 0.0

            comp_a = [nodes[i] for i in range(len(nodes)) if fiedler_vec[i] > cut]
            comp_b = [nodes[i] for i in range(len(nodes)) if fiedler_vec[i] <= cut]

            results = []
            if comp_a:
                results.append(comp_a)
            if comp_b:
                results.append(comp_b)
            return results if results else [nodes]
        except Exception:
            return list(nx.connected_components(graph))

    @staticmethod
    def akef_modified_brim_bipartite_clustering(
        entity_token_edges: List[Tuple[str, str]],
        max_iter: int = 20,
    ) -> List[List[str]]:
        """
        Modified Bipartite Recursively Induced Modules (BRIM) for record-token networks.
        Maximizes bipartite modularity through recursive degree-weighted assignment.
        """
        entities = sorted(list({e[0] for e in entity_token_edges}))
        tokens = sorted(list({e[1] for e in entity_token_edges}))

        if not entities or not tokens:
            return [entities]

        ent_to_idx = {e: i for i, e in enumerate(entities)}
        tok_to_idx = {t: j for j, t in enumerate(tokens)}

        p, q = len(entities), len(tokens)
        B = np.zeros((p, q), dtype=np.float64)
        for u, v in entity_token_edges:
            B[ent_to_idx[u], tok_to_idx[v]] = 1.0

        M = np.sum(B)
        if M == 0:
            return [entities]

        # Row and column degrees
        k_in = np.sum(B, axis=1, keepdims=True)
        d_out = np.sum(B, axis=0, keepdims=True)
        null_model = (k_in @ d_out) / M
        B_tilde = B - null_model

        # Initialize c modules
        c = min(10, max(2, p // 5))
        # Initial random assignment for tokens
        R = np.zeros((q, c), dtype=np.float64)
        for j in range(q):
            R[j, j % c] = 1.0

        for _ in range(max_iter):
            # Update entity assignments: T = B_tilde @ R
            T = B_tilde @ R
            T_new = np.zeros_like(T)
            for i in range(p):
                best_mod = np.argmax(T[i])
                T_new[i, best_mod] = 1.0

            # Update token assignments: R = B_tilde.T @ T_new
            R_cand = B_tilde.T @ T_new
            R_new = np.zeros_like(R)
            for j in range(q):
                best_mod = np.argmax(R_cand[j])
                R_new[j, best_mod] = 1.0

            if np.array_equal(R, R_new):
                break
            R = R_new

        # Extract entity clusters
        ent_clusters = defaultdict(list)
        for i in range(p):
            mod_idx = int(np.argmax(T[i]))
            ent_clusters[mod_idx].append(entities[i])

        return [sorted(members) for members in ent_clusters.values() if members]


class CliqueRankOperator(BaseOperator):
    """
    Refactored from src/GER_CliqueRank_Runner.py.
    Runs random surfer / power iteration PageRank across bipartite record-token
    or pairwise similarity graphs to rank and resolve high-confidence entity links.
    """

    def __init__(
        self,
        name: str = "clique_rank",
        source_pairs_table: str = "",
        target_name: str = "",
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        super().__init__(name)
        self.source_pairs_table = source_pairs_table
        self.target_name = target_name
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_pairs_table", self.source_pairs_table))
        if not source_meta:
            raise ValueError(f"Source pairs table {self.source_pairs_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "")
        alpha = kwargs.get("alpha", self.alpha)
        max_iter = kwargs.get("max_iter", self.max_iter)
        tol = kwargs.get("tol", self.tol)

        pairs_arrow = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = pairs_arrow.to_pydict()

        id_a_list = [str(x) for x in pydict["id_a"]]
        id_b_list = [str(x) for x in pydict["id_b"]]
        weight_col = "jaccard_sim" if "jaccard_sim" in pydict else ("weight" if "weight" in pydict else None)
        weights = [float(pydict[weight_col][i]) for i in range(len(id_a_list))] if weight_col else [1.0] * len(id_a_list)

        graph = nx.Graph()
        for i in range(len(id_a_list)):
            graph.add_edge(id_a_list[i], id_b_list[i], weight=weights[i])

        try:
            pagerank_scores = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight")
        except Exception:
            nodes = list(graph.nodes())
            pagerank_scores = {n: 1.0 / len(nodes) for n in nodes}

        # Assign CliqueRank scores to pairs
        pair_scores = []
        for i in range(len(id_a_list)):
            score_a = pagerank_scores.get(id_a_list[i], 0.0)
            score_b = pagerank_scores.get(id_b_list[i], 0.0)
            # Harmonic mean / combined centrality of pair
            pair_score = (2.0 * score_a * score_b / (score_a + score_b)) if (score_a + score_b) > 0 else 0.0
            pair_scores.append(float(pair_score))

        out_table = pairs_arrow.append_column("clique_rank_score", pa.array(pair_scores, type=pa.float64()))

        target_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="gold",
            uri=target_uri,
            schema=out_table.schema,
            row_count=out_table.num_rows,
            properties={"alpha": alpha, "max_iter": max_iter},
        )
        context.catalog.register_table(meta)
        return meta


class HITSOperator(BaseOperator):
    """
    Refactored from src/HITS.py.
    Executes Kleinberg's Hyperlink-Induced Topic Search (HITS) algorithm on entity networks,
    computing Hub and Authority centrality scores via power iteration or eigensolver.
    """

    def __init__(
        self,
        name: str = "hits",
        source_pairs_table: str = "",
        target_name: str = "",
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        super().__init__(name)
        self.source_pairs_table = source_pairs_table
        self.target_name = target_name
        self.max_iter = max_iter
        self.tol = tol

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_pairs_table", self.source_pairs_table))
        if not source_meta:
            raise ValueError(f"Source pairs table {self.source_pairs_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "")
        max_iter = kwargs.get("max_iter", self.max_iter)
        tol = kwargs.get("tol", self.tol)

        pairs_arrow = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = pairs_arrow.to_pydict()

        id_a_list = [str(x) for x in pydict["id_a"]]
        id_b_list = [str(x) for x in pydict["id_b"]]

        dgraph = nx.DiGraph()
        for i in range(len(id_a_list)):
            dgraph.add_edge(id_a_list[i], id_b_list[i])

        try:
            hubs, authorities = nx.hits(dgraph, max_iter=max_iter, tol=tol, normalized=True)
        except Exception:
            nodes = list(dgraph.nodes())
            unif = 1.0 / len(nodes) if nodes else 0.0
            hubs = {n: unif for n in nodes}
            authorities = {n: unif for n in nodes}

        hub_scores = [hubs.get(id_a_list[i], 0.0) for i in range(len(id_a_list))]
        auth_scores = [authorities.get(id_b_list[i], 0.0) for i in range(len(id_b_list))]

        out_table = pairs_arrow.append_column("hub_score", pa.array(hub_scores, type=pa.float64()))
        out_table = out_table.append_column("authority_score", pa.array(auth_scores, type=pa.float64()))

        target_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="gold",
            uri=target_uri,
            schema=out_table.schema,
            row_count=out_table.num_rows,
            properties={"max_iter": max_iter, "nodes_evaluated": dgraph.number_of_nodes()},
        )
        context.catalog.register_table(meta)
        return meta


class BipartiteGraphConstructor(BaseOperator):
    """
    Refactored from src/GER_Bipartite_Graph_Constructor.py.
    Constructs weighted bipartite incidence graphs between entity records
    and tokens or blocks, exporting edge incidence tables with degrees.
    """

    def __init__(
        self,
        name: str = "bipartite_constructor",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        token_column: str = "_staged_tokens",
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.token_column = token_column

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "")
        key_col = kwargs.get("key_column", self.key_column)
        tok_col = kwargs.get("token_column", self.token_column)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        if key_col not in arrow_table.column_names:
            key_col = "_row_id" if "_row_id" in arrow_table.column_names else arrow_table.column_names[0]

        records = []
        tokens = []
        for i in range(num_rows):
            rec_id = str(pydict[key_col][i])
            tok_text = pydict[tok_col][i] if tok_col in arrow_table.column_names and pydict[tok_col][i] else ""
            tok_list = tok_text.split() if tok_text else []
            for t in tok_list:
                records.append(rec_id)
                tokens.append(f"TOK_{t}")

        edge_counts = Counter(zip(records, tokens))
        src_edges = [k[0] for k in edge_counts.keys()]
        tgt_edges = [k[1] for k in edge_counts.keys()]
        weights = list(edge_counts.values())

        bipartite_table = pa.Table.from_arrays(
            [
                pa.array(src_edges, type=pa.string()),
                pa.array(tgt_edges, type=pa.string()),
                pa.array(weights, type=pa.int64()),
            ],
            names=["record_id", "token_node", "weight"],
        )

        target_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(bipartite_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="gold",
            uri=target_uri,
            schema=bipartite_table.schema,
            row_count=bipartite_table.num_rows,
            properties={"distinct_records": len(set(src_edges)), "distinct_tokens": len(set(tgt_edges))},
        )
        context.catalog.register_table(meta)
        return meta

