import os
import math
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
from src.operators.base import BaseOperator
from src.catalog.table import TableMetadata


# ==============================================================================
# 1. TEXT VECTORIZATION (TF-IDF & N-GRAM HASHING)
# ==============================================================================

class TextVectorizationOperator(BaseOperator):
    """
    Transforms tabular text / staged records into dense L2-normalized vector embeddings.
    Supports TF-IDF, n-gram hashing, and token bag-of-words.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        key_column: str = "RecID",
        method: str = "tfidf",  # 'tfidf', 'ngram_hash', 'char_ngram', 'subword_tfidf'
        dimension: int = 128,
        columns: Optional[List[str]] = None,
        subword: bool = False,
        max_df: float = 0.5,
    ):
        super().__init__("text_vectorization")
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.method = method.lower()
        self.dimension = dimension
        self.columns = columns
        self.subword = subword or (self.method in ("char_ngram", "subword_tfidf"))
        self.max_df = max_df

    @staticmethod
    def _hash_token(token: str, dim: int) -> Tuple[int, float]:
        """Weinberger signed feature hashing trick."""
        h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little")
        sign = 1.0 if (h & 1) == 0 else -1.0
        return (h >> 1) % dim, sign

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs = context.storage
        meta = context.catalog.get_table(self.source_table)
        if not meta:
            raise KeyError(f"Source table '{self.source_table}' not found in catalog.")

        table = fs.read_tabular(meta.uri, file_format="parquet")
        n = table.num_rows

        target_cols = self.columns
        if not target_cols:
            if "_staged_tokens" in table.column_names:
                target_cols = ["_staged_tokens"]
            else:
                target_cols = [c for c in table.column_names if not c.startswith("_") and c != self.key_column]

        # Extract text per row
        row_texts: List[List[str]] = []
        for row_idx in range(n):
            tokens = []
            for col in target_cols:
                val = table[col][row_idx].as_py()
                if val is None:
                    continue
                if isinstance(val, list):
                    tokens.extend([str(t).lower() for t in val])
                else:
                    tokens.extend(str(val).lower().split())

            if self.subword:
                sub_tokens = list(tokens)
                for t in tokens:
                    if len(t) >= 3:
                        for k in range(len(t) - 2):
                            sub_tokens.append(t[k : k + 3])
                row_texts.append(sub_tokens)
            else:
                row_texts.append(tokens)

        vectors = np.zeros((n, self.dimension), dtype=np.float32)

        if "tfidf" in self.method:
            # Compute document frequency
            df = Counter()
            for tokens in row_texts:
                for t in set(tokens):
                    df[t] += 1

            # Filter high frequency stopwords and rare singletons
            candidate_tokens = [t for t, count in df.most_common() if count < self.max_df * n and count >= 2]
            if len(candidate_tokens) < self.dimension:
                candidate_tokens = [t for t, _ in df.most_common()]

            vocab = {token: idx for idx, token in enumerate(candidate_tokens[: self.dimension])}

            for i, tokens in enumerate(row_texts):
                tf = Counter(tokens)
                for t, count in tf.items():
                    if t in vocab:
                        col_idx = vocab[t]
                        idf = math.log((1.0 + n) / (1.0 + df[t]))
                        vectors[i, col_idx] += count * idf

        elif self.method in ("ngram_hash", "char_ngram"):
            # Signed Feature hashing
            for i, tokens in enumerate(row_texts):
                for t in tokens:
                    h_idx, sign = self._hash_token(t, self.dimension)
                    vectors[i, h_idx] += sign * 1.0
        else:
            # Bag-of-words signed hashing
            for i, tokens in enumerate(row_texts):
                for t in tokens:
                    h_idx, sign = self._hash_token(t, self.dimension)
                    vectors[i, h_idx] += sign * 1.0

        # L2-normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        # Pack into PyArrow Table with list column
        clean_target = self.target_name.replace("gold.", "").replace("silver.", "")
        tier = "gold" if "gold." in self.target_name else "silver"
        target_uri = os.path.join(context.lakehouse_root, tier, f"{clean_target}.parquet").replace("\\", "/")

        vector_pylist = [v.tolist() for v in vectors]
        embed_col = pa.array(vector_pylist, type=pa.list_(pa.float32()))

        if self.key_column in table.column_names:
            key_arr = table[self.key_column]
            actual_key_col = self.key_column
        elif "_row_id" in table.column_names:
            key_arr = table["_row_id"]
            actual_key_col = "_row_id"
        else:
            key_arr = pa.array([str(i) for i in range(n)])
            actual_key_col = self.key_column

        out_cols = [key_arr, embed_col]
        out_names = [actual_key_col, "embedding"]

        # Preserve original columns if not already in out_cols
        for col_name in table.column_names:
            if col_name not in out_names:
                out_cols.append(table[col_name])
                out_names.append(col_name)

        out_table = pa.Table.from_arrays(out_cols, names=out_names)
        fs.write_parquet(out_table, target_uri)

        out_meta = TableMetadata(
            name=self.target_name,
            tier=tier,
            uri=target_uri,
            schema=out_table.schema,
            row_count=out_table.num_rows,
            properties={"dimension": self.dimension, "method": self.method},
        )
        context.catalog.register_table(out_meta)
        return out_meta


# ==============================================================================
# 2. DIRECTED GRAPH CONVOLUTIONAL NETWORK (DGCN) ENGINE
# ==============================================================================

class DirectedGCNEngine:
    """
    Core Directed GCN & Deep Graph Infomax (DGI) mathematical engine.
    Supports in/out degree normalization and multi-layer directional aggregation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 2,
        directed: bool = True,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.directed = directed
        self.lr = learning_rate
        self.rng = np.random.RandomState(seed)

        # Initialize Glorot weights for in and out directions
        self.weights: List[Dict[str, np.ndarray]] = []
        cur_dim = in_dim
        for l in range(layers):
            dim_next = out_dim if l == layers - 1 else max(out_dim, 32)
            limit = np.sqrt(6.0 / (cur_dim + dim_next))
            layer_w = {
                "W_in": self.rng.uniform(-limit, limit, (cur_dim, dim_next)).astype(np.float32),
                "W_out": self.rng.uniform(-limit, limit, (cur_dim, dim_next)).astype(np.float32),
                "b": np.zeros(dim_next, dtype=np.float32),
            }
            self.weights.append(layer_w)
            cur_dim = dim_next

        # Discriminator bilinear parameter for DGI
        self.W_d = self.rng.randn(out_dim, out_dim).astype(np.float32) * 0.1

    @staticmethod
    def normalize_adjacency(adj: np.ndarray, directed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes normalized in-degree and out-degree adjacency matrices:
        A_in = D_in^{-1} A,  A_out = D_out^{-1} A^T
        """
        n = adj.shape[0]
        # Add self-loops
        adj_loop = adj + np.eye(n, dtype=np.float32)

        if not directed:
            # Symmetric Kipf-Welling D^{-1/2} A D^{-1/2}
            deg = np.sum(adj_loop, axis=1)
            deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
            deg_inv_sqrt[deg == 0] = 0.0
            d_mat = np.diag(deg_inv_sqrt)
            norm_sym = d_mat @ adj_loop @ d_mat
            return norm_sym, norm_sym

        in_deg = np.sum(adj_loop, axis=0)
        out_deg = np.sum(adj_loop, axis=1)

        in_deg_inv = np.zeros_like(in_deg)
        mask_in = in_deg > 0
        in_deg_inv[mask_in] = 1.0 / in_deg[mask_in]

        out_deg_inv = np.zeros_like(out_deg)
        mask_out = out_deg > 0
        out_deg_inv[mask_out] = 1.0 / out_deg[mask_out]

        A_in = adj_loop * in_deg_inv  # in-degree normalized
        A_out = (adj_loop.T) * out_deg_inv[:, np.newaxis]  # out-degree normalized
        return A_in.astype(np.float32), A_out.astype(np.float32)

    def forward_layers(self, X: np.ndarray, A_in: np.ndarray, A_out: np.ndarray) -> List[np.ndarray]:
        """Forward pass recording all layer activations."""
        activations = [X]
        H = X
        for l, w in enumerate(self.weights):
            if self.directed:
                h_in = A_in @ H @ w["W_in"]
                h_out = A_out @ H @ w["W_out"]
                Z = h_in + h_out + w["b"]
            else:
                Z = (A_in @ H @ w["W_in"]) + w["b"]

            # Activation: ReLU on hidden layers, linear on output
            if l < len(self.weights) - 1:
                H = np.maximum(Z, 0.0)
            else:
                H = Z
            activations.append(H)

        # Final L2-normalization for cosine representation learning
        norms = np.linalg.norm(activations[-1], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        activations[-1] = activations[-1] / norms
        return activations

    def forward(self, X: np.ndarray, A_in: np.ndarray, A_out: np.ndarray) -> np.ndarray:
        """Computes forward directional graph convolutions."""
        return self.forward_layers(X, A_in, A_out)[-1]

    def train_dgi(
        self,
        X: np.ndarray,
        adj: np.ndarray,
        epochs: int = 15,
    ) -> np.ndarray:
        """
        Self-supervised Deep Graph Infomax (DGI):
        Maximizes mutual information between local node embeddings and global graph summary.
        """
        n = X.shape[0]
        A_in, A_out = self.normalize_adjacency(adj, directed=self.directed)

        for ep in range(epochs):
            # 1. Forward positive graph with intermediate activations
            pos_layers = self.forward_layers(X, A_in, A_out)
            H_pos = pos_layers[-1]
            H_prev = pos_layers[-2]

            # Global summary vector: s = sigmoid(mean(H_pos))
            s = 1.0 / (1.0 + np.exp(-np.mean(H_pos, axis=0)))

            # 2. Corrupt features by row-permutation
            perm = self.rng.permutation(n)
            X_corrupt = X[perm]
            neg_layers = self.forward_layers(X_corrupt, A_in, A_out)
            H_neg = neg_layers[-1]

            # 3. Bilinear scoring: D(h, s) = sigmoid(h @ W_d @ s)
            proj_s = self.W_d @ s
            scores_pos = 1.0 / (1.0 + np.exp(-np.clip(H_pos @ proj_s, -15.0, 15.0)))
            scores_neg = 1.0 / (1.0 + np.exp(-np.clip(H_neg @ proj_s, -15.0, 15.0)))

            # 4. Gradient updates for top layer and bilinear discriminator
            grad_pos = (1.0 - scores_pos)[:, np.newaxis] * proj_s
            grad_neg = (-scores_neg)[:, np.newaxis] * proj_s
            grad_h = (grad_pos + grad_neg) / n

            top_w = self.weights[-1]
            top_w["W_in"] += self.lr * (H_prev.T @ grad_h)
            if self.directed:
                top_w["W_out"] += self.lr * (H_prev.T @ grad_h)

            grad_Wd = (H_pos.T @ ((1.0 - scores_pos)[:, np.newaxis] * s)) + (H_neg.T @ ((-scores_neg)[:, np.newaxis] * s))
            self.W_d += self.lr * (grad_Wd / n)

        return self.forward(X, A_in, A_out)


# ==============================================================================
# 3. GNN ENCODER OPERATOR (LAKEHOUSE INTEGRATION)
# ==============================================================================

class GNNEncoderOperator(BaseOperator):
    """
    Applies Directed GCN or Deep Graph Infomax over entity graphs and candidate pair topologies.
    Produces high-dimensional continuous entity embeddings stored in Parquet.
    """

    def __init__(
        self,
        source_table: str,
        graph_table: str,
        target_name: str,
        key_column: str = "RecID",
        src_col: str = "src",
        dst_col: str = "dst",
        weight_col: Optional[str] = "weight",
        model: str = "dgcn",  # 'dgcn', 'gcn', 'dgi'
        dimension: int = 32,
        layers: int = 2,
        epochs: int = 20,
        directed: bool = True,
    ):
        super().__init__("gnn_encoder")
        self.source_table = source_table
        self.graph_table = graph_table
        self.target_name = target_name
        self.key_column = key_column
        self.src_col = src_col
        self.dst_col = dst_col
        self.weight_col = weight_col
        self.model = model.lower()
        self.dimension = dimension
        self.layers = layers
        self.epochs = epochs
        self.directed = directed

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs = context.storage
        src_meta = context.catalog.get_table(self.source_table)
        graph_meta = context.catalog.get_table(self.graph_table)

        if not src_meta:
            raise KeyError(f"Source table '{self.source_table}' not found in catalog.")
        if not graph_meta:
            raise KeyError(f"Graph table '{self.graph_table}' not found in catalog.")

        src_tbl = fs.read_tabular(src_meta.uri, file_format="parquet")
        graph_tbl = fs.read_tabular(graph_meta.uri, file_format="parquet")

        n = src_tbl.num_rows
        rec_ids = [str(x) for x in src_tbl[self.key_column].to_pylist()] if self.key_column in src_tbl.column_names else [str(i) for i in range(n)]
        rec_id_to_idx = {r_id: idx for idx, r_id in enumerate(rec_ids)}

        # Load or initialize feature matrix X
        if "embedding" in src_tbl.column_names:
            pylist_emb = src_tbl["embedding"].to_pylist()
            X = np.array(pylist_emb, dtype=np.float32)
        else:
            # Fallback to random feature projection
            X = np.random.randn(n, 64).astype(np.float32)

        # Build adjacency matrix from graph table
        adj = np.zeros((n, n), dtype=np.float32)
        g_cols = graph_tbl.column_names

        # Check edge columns (source, target) or (row_id_1, row_id_2)
        if self.src_col in g_cols and self.dst_col in g_cols:
            c1, c2 = self.src_col, self.dst_col
        elif "source" in g_cols and "target" in g_cols:
            c1, c2 = "source", "target"
        elif "src" in g_cols and "dst" in g_cols:
            c1, c2 = "src", "dst"
        elif "id_a" in g_cols and "id_b" in g_cols:
            c1, c2 = "id_a", "id_b"
        elif "row_id_1" in g_cols and "row_id_2" in g_cols:
            c1, c2 = "row_id_1", "row_id_2"
        else:
            c1, c2 = g_cols[0], g_cols[1]

        weight_col = None
        if self.weight_col and self.weight_col in g_cols:
            weight_col = self.weight_col
        elif "weight" in g_cols:
            weight_col = "weight"
        elif len(g_cols) >= 3 and not g_cols[2].startswith("_"):
            weight_col = g_cols[2]

        edges_1 = [str(x) for x in graph_tbl[c1].to_pylist()]
        edges_2 = [str(x) for x in graph_tbl[c2].to_pylist()]
        weights = graph_tbl[weight_col].to_pylist() if weight_col else [1.0] * len(edges_1)

        for u, v, w in zip(edges_1, edges_2, weights):
            idx1 = rec_id_to_idx.get(u)
            idx2 = rec_id_to_idx.get(v)
            if idx1 is None:
                try:
                    idx1 = int(u)
                except ValueError:
                    continue
            if idx2 is None:
                try:
                    idx2 = int(v)
                except ValueError:
                    continue

            if 0 <= idx1 < n and 0 <= idx2 < n:
                w_val = float(w) if w is not None else 1.0
                adj[idx1, idx2] = w_val
                if not self.directed:
                    adj[idx2, idx1] = w_val

        # Execute Directed GCN Engine
        engine = DirectedGCNEngine(
            in_dim=X.shape[1],
            out_dim=self.dimension,
            layers=self.layers,
            directed=self.directed,
        )

        if self.model == "dgi":
            embeddings = engine.train_dgi(X, adj, epochs=self.epochs)
        else:
            A_in, A_out = engine.normalize_adjacency(adj, directed=self.directed)
            embeddings = engine.forward(X, A_in, A_out)

        # Write output table to Lakehouse
        clean_target = self.target_name.replace("gold.", "").replace("silver.", "")
        tier = "gold" if "gold." in self.target_name else "silver"
        target_uri = os.path.join(context.lakehouse_root, tier, f"{clean_target}.parquet").replace("\\", "/")

        vector_pylist = [v.tolist() for v in embeddings]
        embed_col = pa.array(vector_pylist, type=pa.list_(pa.float32()))

        if self.key_column in src_tbl.column_names:
            key_arr = src_tbl[self.key_column]
            actual_key_col = self.key_column
        elif "_row_id" in src_tbl.column_names:
            key_arr = src_tbl["_row_id"]
            actual_key_col = "_row_id"
        else:
            key_arr = pa.array([str(i) for i in range(src_tbl.num_rows)])
            actual_key_col = self.key_column

        out_cols = [key_arr, embed_col]
        out_names = [actual_key_col, "embedding"]

        for col_name in src_tbl.column_names:
            if col_name not in out_names and col_name != "embedding":
                out_cols.append(src_tbl[col_name])
                out_names.append(col_name)

        out_table = pa.Table.from_arrays(out_cols, names=out_names)
        fs.write_parquet(out_table, target_uri)

        out_meta = TableMetadata(
            name=self.target_name,
            tier=tier,
            uri=target_uri,
            schema=out_table.schema,
            row_count=out_table.num_rows,
            properties={"model": self.model, "dimension": self.dimension, "directed": self.directed},
        )
        context.catalog.register_table(out_meta)
        return out_meta


# ==============================================================================
# 3. VECTOR ENTITY RESOLUTION & NEURAL DATA BUILDERS (VectER, er2vec, LSTM, XGCN)
# ==============================================================================

class Word2VecOperator(BaseOperator):
    """
    Refactored from src/er2word2vec.py and src/VectER.py.
    Trains distributed token embeddings (Word2Vec / PMI SVD) on record tokens,
    infers continuous entity embeddings, and performs cosine similarity queries.
    """

    def __init__(
        self,
        name: str = "word2vec",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        dimension: int = 128,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.dimension = dimension
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    @staticmethod
    def _train_embeddings_pmi_svd(corpus: List[List[str]], dim: int, window: int, min_count: int) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """
        Levy & Goldberg (2014) Positive Pointwise Mutual Information (PPMI) + SVD factorization,
        which provides an exact deterministic closed-form solution to Word2Vec Skip-Gram with Negative Sampling.
        """
        vocab_counts = Counter(token for doc in corpus for token in doc)
        vocab = [tok for tok, count in vocab_counts.items() if count >= min_count]
        if not vocab:
            vocab = list(vocab_counts.keys())
        word_to_id = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        if V == 0:
            return {}, {}

        # Co-occurrence counting
        cooccur = defaultdict(float)
        total_pairs = 0.0
        for doc in corpus:
            doc_len = len(doc)
            for i, target in enumerate(doc):
                if target not in word_to_id:
                    continue
                w_start = max(0, i - window)
                w_end = min(doc_len, i + window + 1)
                for j in range(w_start, w_end):
                    if i != j and doc[j] in word_to_id:
                        cooccur[(word_to_id[target], word_to_id[doc[j]])] += 1.0
                        total_pairs += 1.0

        if total_pairs == 0:
            rng = np.random.RandomState(42)
            return {w: rng.randn(dim).astype(np.float32) for w in vocab}, word_to_id

        # Compute PPMI matrix
        target_marginals = defaultdict(float)
        context_marginals = defaultdict(float)
        for (w, c), count in cooccur.items():
            target_marginals[w] += count
            context_marginals[c] += count

        matrix = np.zeros((V, V), dtype=np.float32)
        for (w, c), count in cooccur.items():
            p_wc = count / total_pairs
            p_w = target_marginals[w] / total_pairs
            p_c = context_marginals[c] / total_pairs
            pmi = math.log(p_wc / (p_w * p_c) + 1e-9)
            if pmi > 0:
                matrix[w, c] = pmi

        # Truncated SVD
        actual_dim = min(dim, V)
        try:
            U, S, _ = np.linalg.svd(matrix, full_matrices=False)
            W = U[:, :actual_dim] * np.sqrt(S[:actual_dim])
            # If actual_dim < dim, pad with zeros
            if actual_dim < dim:
                W = np.pad(W, ((0, 0), (0, dim - actual_dim)))
        except Exception:
            rng = np.random.RandomState(42)
            W = rng.randn(V, dim).astype(np.float32)

        vectors = {w: W[i] for w, i in word_to_id.items()}
        return vectors, word_to_id

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise KeyError(f"Table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "").replace("silver.", "")
        tier = "gold" if "gold." in self.target_name else "silver"
        key_col = kwargs.get("key_column", self.key_column)

        arrow_table = fs.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        if key_col not in arrow_table.column_names:
            key_col = "_row_id" if "_row_id" in arrow_table.column_names else arrow_table.column_names[0]

        token_col = "_staged_tokens" if "_staged_tokens" in arrow_table.column_names else None
        corpus = []
        for i in range(num_rows):
            if token_col and pydict[token_col][i]:
                corpus.append(pydict[token_col][i].split())
            else:
                text = " ".join([str(pydict[c][i]) for c in arrow_table.column_names if not c.startswith("_")])
                corpus.append(text.lower().split())

        # Try gensim first if available
        word_vectors = None
        try:
            from gensim.models import Word2Vec
            model = Word2Vec(sentences=corpus, vector_size=self.dimension, window=self.window, min_count=self.min_count, epochs=self.epochs, seed=42)
            word_vectors = {w: model.wv[w] for w in model.wv.index_to_key}
        except ImportError:
            word_vectors, _ = self._train_embeddings_pmi_svd(corpus, self.dimension, self.window, self.min_count)

        # Average token vectors to produce record embeddings
        record_embeddings = np.zeros((num_rows, self.dimension), dtype=np.float32)
        for i, doc in enumerate(corpus):
            doc_vecs = [word_vectors[w] for w in doc if w in word_vectors]
            if doc_vecs:
                mean_vec = np.mean(doc_vecs, axis=0)
                norm = np.linalg.norm(mean_vec)
                record_embeddings[i] = (mean_vec / norm) if norm > 1e-9 else mean_vec

        embed_col = pa.array([row.tolist() for row in record_embeddings], type=pa.list_(pa.float32()))
        if "embedding" in arrow_table.column_names:
            out_table = arrow_table.set_column(arrow_table.column_names.index("embedding"), "embedding", embed_col)
        else:
            out_table = arrow_table.append_column("embedding", embed_col)

        target_uri = os.path.join(context.lakehouse_root, tier, f"{clean_target}.parquet").replace("\\", "/")
        fs.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier=tier,
            uri=target_uri,
            schema=out_table.schema,
            row_count=num_rows,
            properties={"dimension": self.dimension, "vocab_size": len(word_vectors)},
        )
        context.catalog.register_table(meta)
        return meta


class Doc2VecOperator(BaseOperator):
    """
    Refactored from src/er2doc2vec.py.
    Learns continuous distributed representations for whole documents and entity records.
    """

    def __init__(
        self,
        name: str = "doc2vec",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        dimension: int = 128,
        epochs: int = 10,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.dimension = dimension
        self.epochs = epochs

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise KeyError(f"Table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("gold.", "").replace("silver.", "")
        tier = "gold" if "gold." in self.target_name else "silver"
        key_col = kwargs.get("key_column", self.key_column)

        arrow_table = fs.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        if key_col not in arrow_table.column_names:
            key_col = "_row_id" if "_row_id" in arrow_table.column_names else arrow_table.column_names[0]

        token_col = "_staged_tokens" if "_staged_tokens" in arrow_table.column_names else None
        corpus = []
        for i in range(num_rows):
            if token_col and pydict[token_col][i]:
                corpus.append(pydict[token_col][i].split())
            else:
                text = " ".join([str(pydict[c][i]) for c in arrow_table.column_names if not c.startswith("_")])
                corpus.append(text.lower().split())

        doc_embeddings = np.zeros((num_rows, self.dimension), dtype=np.float32)
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
            model = Doc2Vec(tagged_docs, vector_size=self.dimension, window=5, min_count=1, epochs=self.epochs, seed=42)
            for i in range(num_rows):
                vec = model.dv[i]
                norm = np.linalg.norm(vec)
                doc_embeddings[i] = (vec / norm) if norm > 1e-9 else vec
        except ImportError:
            # High-performance TF-IDF weighted centroid embedding fallback
            vec_op = TextVectorizationOperator(
                source_table=source_meta.qualified_name,
                target_name=clean_target,
                key_column=key_col,
                method="tfidf",
                dimension=self.dimension,
            )
            res_meta = vec_op.execute(context)
            res_table = fs.read_tabular(res_meta.uri, file_format="parquet")
            return res_meta

        embed_col = pa.array([row.tolist() for row in doc_embeddings], type=pa.list_(pa.float32()))
        if "embedding" in arrow_table.column_names:
            out_table = arrow_table.set_column(arrow_table.column_names.index("embedding"), "embedding", embed_col)
        else:
            out_table = arrow_table.append_column("embedding", embed_col)

        target_uri = os.path.join(context.lakehouse_root, tier, f"{clean_target}.parquet").replace("\\", "/")
        fs.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier=tier,
            uri=target_uri,
            schema=out_table.schema,
            row_count=num_rows,
            properties={"dimension": self.dimension},
        )
        context.catalog.register_table(meta)
        return meta


class FastTextOperator(BaseOperator):
    """
    Refactored from src/er2fasttext2vec.py.
    Extracts subword n-gram character components to build typo-resilient entity embeddings.
    """

    def __init__(
        self,
        name: str = "fasttext",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        dimension: int = 128,
        min_n: int = 3,
        max_n: int = 6,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.dimension = dimension
        self.min_n = min_n
        self.max_n = max_n

    @classmethod
    def extract_subwords(cls, word: str, min_n: int = 3, max_n: int = 6) -> List[str]:
        w = f"<{word}>"
        subwords = []
        for n in range(min_n, max_n + 1):
            for i in range(len(w) - n + 1):
                subwords.append(w[i:i + n])
        return subwords

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        vec_op = TextVectorizationOperator(
            source_table=kwargs.get("source_table", self.source_table),
            target_name=kwargs.get("target_name", self.target_name),
            key_column=kwargs.get("key_column", self.key_column),
            method="subword_tfidf",
            dimension=self.dimension,
            subword=True,
        )
        return vec_op.execute(context)


class LSTMDataBuilder:
    """
    Refactored from src/lstm_data_builder.py.
    Builds PyTorch or NumPy training samples for sequential character-level / token-level prediction.
    """

    def __init__(self, sequences: List[str], seq_len: int, step: int, char_to_int: Dict[str, int]):
        self.sequences = sequences
        self.seq_len = seq_len
        self.step = step
        self.char_to_int = char_to_int
        self.samples: List[Tuple[str, str]] = []
        for seq in self.sequences:
            if len(seq) > self.seq_len:
                for i in range(0, len(seq) - self.seq_len, self.step):
                    input_seq_text = seq[i : i + self.seq_len]
                    target_char = seq[i + self.seq_len]
                    self.samples.append((input_seq_text, target_char))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        input_seq_text, target_char = self.samples[idx]
        input_ids = [self.char_to_int[c] for c in input_seq_text]
        target_id = self.char_to_int[target_char]
        try:
            import torch
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)
        except ImportError:
            return np.array(input_ids, dtype=np.int64), np.int64(target_id)


class XGCNDataBuilder:
    """
    Refactored from src/xgcn_data_builder.py.
    Prepares initial node feature matrices (identity, degree, SVD spectral, bag-of-words)
    and target labels for GNN training across coarsened graph hierarchies.
    """

    @staticmethod
    def prepare_node_features(
        adj_matrix: np.ndarray,
        feature_type: str = "degree_svd",
        dim: int = 64,
    ) -> np.ndarray:
        """Constructs rich initial node representations from graph topology."""
        n = adj_matrix.shape[0]
        if feature_type == "identity":
            return np.eye(n, dtype=np.float32)

        degrees = np.sum(adj_matrix, axis=1, keepdims=True)
        in_degrees = np.sum(adj_matrix, axis=0, keepdims=True).T
        deg_feats = np.hstack([degrees, in_degrees])

        if feature_type == "degree":
            if dim > 2:
                pad = np.zeros((n, dim - 2), dtype=np.float32)
                return np.hstack([deg_feats, pad])
            return deg_feats[:, :dim].astype(np.float32)

        # Default: Truncated SVD on Adjacency + Degrees
        actual_dim = min(dim - 2, n)
        if actual_dim > 0:
            try:
                U, S, _ = np.linalg.svd(adj_matrix.astype(np.float32), full_matrices=False)
                svd_feats = U[:, :actual_dim] * np.sqrt(S[:actual_dim])
                feats = np.hstack([deg_feats, svd_feats])
                if feats.shape[1] < dim:
                    pad = np.zeros((n, dim - feats.shape[1]), dtype=np.float32)
                    feats = np.hstack([feats, pad])
                return feats[:, :dim].astype(np.float32)
            except Exception:
                pass

        rng = np.random.RandomState(42)
        return rng.randn(n, dim).astype(np.float32)

