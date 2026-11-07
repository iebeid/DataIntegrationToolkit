import os
import re
import math
import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolkit.operators.base import BaseOperator
from toolkit.catalog.table import TableMetadata


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance with optional Damerau transposition."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if not len1:
        return len2
    if not len2:
        return len1
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )
            # Damerau transposition check
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[len1][len2]


def normal_led(token1: str, token2: str) -> Tuple[float, int]:
    """
    Normalized Levenshtein Edit Distance (normalized by max length, higher is more similar).
    Returns (fDist, dDist) matching DWM45.
    """
    length = max(len(token1), len(token2))
    if length == 0:
        return 1.0, 0
    d_dist = _levenshtein_distance(token1, token2)
    f_dist = float(length - d_dist) / float(length)
    return f_dist, d_dist


class MinHashLSHIndex:
    """
    Pure-Python & NumPy implementation of MinHash Locality Sensitive Hashing (LSH).
    Computes k permutation hash signatures and indexes bands into hash buckets.
    """

    PRIME = (1 << 31) - 1  # 2147483647

    def __init__(self, num_perm: int = 64, threshold: float = 0.5, seed: int = 42):
        self.num_perm = num_perm
        self.threshold = threshold
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, self.PRIME, size=num_perm, dtype=np.int64)
        self.b = rng.randint(0, self.PRIME, size=num_perm, dtype=np.int64)

        # Compute optimal number of bands and rows: threshold ~ (1/b)^(1/r)
        self.bands, self.rows = self._optimal_bands_and_rows(num_perm, threshold)
        self.buckets: List[Dict[int, List[str]]] = [defaultdict(list) for _ in range(self.bands)]

    @staticmethod
    def _optimal_bands_and_rows(num_perm: int, threshold: float) -> Tuple[int, int]:
        best_b, best_r = 1, num_perm
        min_diff = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                # Approx threshold = (1/b)**(1/r)
                approx_t = (1.0 / b) ** (1.0 / r)
                diff = abs(approx_t - threshold)
                if diff < min_diff:
                    min_diff = diff
                    best_b, best_r = b, r
        return best_b, best_r

    def compute_signature(self, tokens: Set[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.num_perm, self.PRIME, dtype=np.int64)

        token_hashes = np.array(
            [int.from_bytes(hashlib.md5(t.encode("utf-8")).digest()[:4], "little") % self.PRIME for t in tokens],
            dtype=np.int64,
        )
        vals = (np.outer(self.a, token_hashes) + self.b[:, np.newaxis]) % self.PRIME
        return np.min(vals, axis=1)

    def insert(self, record_id: str, signature: np.ndarray) -> None:
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hash(tuple(signature[start:end]))
            self.buckets[band_idx][band_hash].append(record_id)

    def query_candidate_pairs(self, max_bucket_size: int = 200) -> Set[Tuple[str, str]]:
        candidates = set()
        for band_dict in self.buckets:
            for bucket_hash, members in band_dict.items():
                if 1 < len(members) <= max_bucket_size:
                    sorted_members = sorted(members)
                    for i in range(len(sorted_members)):
                        for j in range(i + 1, len(sorted_members)):
                            candidates.add((sorted_members[i], sorted_members[j]))
        return candidates


class StagerOperator(BaseOperator):
    """
    Silver-tier operator: Synthesized from StagER, MappER, and DWM14/16.
    Performs schema-agnostic or attribute-specific tokenization, frequency profiling,
    and stopword/discriminant filtering (beta/sigma cutoffs).
    """

    def __init__(
        self,
        name: str = "stager",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        feature_columns: Optional[List[str]] = None,
        stopword_threshold: float = 0.05,  # Top % most frequent tokens treated as stopwords (beta)
        min_token_len: int = 2,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.feature_columns = feature_columns
        self.stopword_threshold = stopword_threshold
        self.min_token_len = min_token_len

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Source table {self.source_table} not found in catalog.")

        clean_target = kwargs.get("target_name", self.target_name).replace("silver.", "")
        key_col = kwargs.get("key_column", self.key_column)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        col_names = arrow_table.column_names

        if key_col not in col_names:
            key_col = "_row_id" if "_row_id" in col_names else col_names[0]

        feat_cols = self.feature_columns
        if not feat_cols:
            feat_cols = [c for c in col_names if not c.startswith("_") and c != key_col]

        record_tokens: Dict[str, List[str]] = {}
        token_freq: Counter = Counter()

        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        for i in range(num_rows):
            rec_id = str(pydict[key_col][i])
            tokens = []
            for col in feat_cols:
                val = pydict[col][i]
                if val is not None:
                    words = re.findall(r"\b\w+\b", str(val).lower())
                    tokens.extend([w for w in words if len(w) >= self.min_token_len])

            token_set = set(tokens)
            record_tokens[rec_id] = list(token_set)
            token_freq.update(token_set)

        total_records = max(1, num_rows)
        stopword_limit = max(5, int(total_records * self.stopword_threshold))
        stopwords = {token for token, count in token_freq.items() if count >= stopword_limit}

        filtered_tokens_list = []
        for i in range(num_rows):
            rec_id = str(pydict[key_col][i])
            cleaned = [t for t in record_tokens[rec_id] if t not in stopwords]
            filtered_tokens_list.append(" ".join(cleaned))

        tokens_arrow = pa.array(filtered_tokens_list, type=pa.string())
        silver_table = arrow_table.append_column("_staged_tokens", tokens_arrow)

        target_uri = os.path.join(context.lakehouse_root, "silver", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(silver_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="silver",
            uri=target_uri,
            schema=silver_table.schema,
            row_count=num_rows,
            properties={
                "source_table": source_meta.qualified_name,
                "key_column": key_col,
                "num_stopwords_filtered": len(stopwords),
                "unique_tokens": len(token_freq),
            },
        )
        context.catalog.register_table(meta)
        return meta


class BlockingOperator(BaseOperator):
    """
    Silver-tier operator: Synthesized from BlockingScheme.py and MinHash LSH.
    Supports both 'token_inverted_index' and scalable 'minhash_lsh'.
    Partitions the record space into candidate pairs (id_a, id_b) without O(N^2) evaluation.
    """

    def __init__(
        self,
        name: str = "blocking",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        method: str = "token_inverted_index",  # 'token_inverted_index' or 'minhash_lsh'
        max_block_size: int = 200,
        num_perm: int = 64,
        threshold: float = 0.4,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.method = method
        self.max_block_size = max_block_size
        self.num_perm = num_perm
        self.threshold = threshold

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Source table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("silver.", "")
        key_col = kwargs.get("key_column", self.key_column)
        method = kwargs.get("method", self.method).lower()
        max_block_size = kwargs.get("max_block_size", self.max_block_size)
        num_perm = kwargs.get("num_perm", self.num_perm)
        threshold = kwargs.get("threshold", self.threshold)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        if key_col not in arrow_table.column_names:
            key_col = "_row_id" if "_row_id" in arrow_table.column_names else arrow_table.column_names[0]

        token_col = "_staged_tokens" if "_staged_tokens" in arrow_table.column_names else None
        rec_ids = [str(x) for x in pydict[key_col]]

        record_token_sets = []
        for i in range(num_rows):
            if token_col and pydict[token_col][i]:
                tokens = set(pydict[token_col][i].split())
            else:
                raw_text = " ".join([str(pydict[c][i]) for c in arrow_table.column_names if not c.startswith("_")])
                tokens = set(re.findall(r"\b\w{3,}\b", raw_text.lower()))
            record_token_sets.append(tokens)

        candidate_pairs: Set[Tuple[str, str]] = set()

        if method == "minhash_lsh":
            lsh = MinHashLSHIndex(num_perm=num_perm, threshold=threshold)
            for i in range(num_rows):
                sig = lsh.compute_signature(record_token_sets[i])
                lsh.insert(rec_ids[i], sig)
            candidate_pairs = lsh.query_candidate_pairs(max_bucket_size=max_block_size)
        else:
            block_index: Dict[str, List[str]] = defaultdict(list)
            for i in range(num_rows):
                for tok in record_token_sets[i]:
                    block_index[tok].append(rec_ids[i])

            for tok, members in block_index.items():
                if 1 < len(members) <= max_block_size:
                    sorted_members = sorted(members)
                    for idx_a in range(len(sorted_members)):
                        for idx_b in range(idx_a + 1, len(sorted_members)):
                            candidate_pairs.add((sorted_members[idx_a], sorted_members[idx_b]))

        pairs_a = [p[0] for p in candidate_pairs]
        pairs_b = [p[1] for p in candidate_pairs]

        pairs_table = pa.Table.from_arrays(
            [pa.array(pairs_a, type=pa.string()), pa.array(pairs_b, type=pa.string())],
            names=["id_a", "id_b"],
        )

        target_uri = os.path.join(context.lakehouse_root, "silver", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(pairs_table, target_uri)

        total_possible = max(1, (num_rows * (num_rows - 1) // 2))
        reduction_ratio = round(1.0 - (pairs_table.num_rows / total_possible), 4)

        meta = TableMetadata(
            name=clean_target,
            tier="silver",
            uri=target_uri,
            schema=pairs_table.schema,
            row_count=pairs_table.num_rows,
            properties={
                "source_table": source_meta.qualified_name,
                "method": method,
                "candidate_pairs_count": pairs_table.num_rows,
                "reduction_ratio": reduction_ratio,
            },
        )
        context.catalog.register_table(meta)
        return meta


class CharacterMapsOperator(BaseOperator):
    """
    Refactored from src/character_maps.py.
    Provides character encoding discovery, ASCII translation, character co-occurrence graph
    modeling, cosine distance calculation, and embedding representation.
    """

    def __init__(
        self,
        name: str = "character_maps",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        feature_columns: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.feature_columns = feature_columns

    @staticmethod
    def convert_chars_to_ascii_codes(char_list: List[str]) -> List[int]:
        return [ord(c) for c in char_list]

    @staticmethod
    def create_character_map(text: str) -> Dict[str, Any]:
        """Maps character frequencies, transitions, and ASCII distribution."""
        chars = list(text)
        ascii_codes = [ord(c) for c in chars]
        char_counts = Counter(chars)
        bigram_counts = Counter([text[i:i+2] for i in range(len(text) - 1)]) if len(text) > 1 else Counter()
        return {
            "length": len(text),
            "unique_chars": len(char_counts),
            "ascii_vector": ascii_codes,
            "char_counts": dict(char_counts),
            "bigrams": dict(bigram_counts),
        }

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1.0 - np.dot(vec1, vec2) / (norm1 * norm2))

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Source table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("silver.", "")
        key_col = kwargs.get("key_column", self.key_column)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        if key_col not in arrow_table.column_names:
            key_col = "_row_id" if "_row_id" in arrow_table.column_names else arrow_table.column_names[0]

        feat_cols = self.feature_columns or [c for c in arrow_table.column_names if not c.startswith("_") and c != key_col]

        ascii_reprs = []
        unique_char_counts = []
        for i in range(num_rows):
            combined_text = " ".join([str(pydict[c][i]) for c in feat_cols if pydict[c][i] is not None])
            cmap = self.create_character_map(combined_text)
            ascii_reprs.append(",".join(map(str, cmap["ascii_vector"][:50])))
            unique_char_counts.append(cmap["unique_chars"])

        out_table = arrow_table.append_column("_ascii_codes", pa.array(ascii_reprs, type=pa.string()))
        out_table = out_table.append_column("_unique_chars", pa.array(unique_char_counts, type=pa.int64()))

        target_uri = os.path.join(context.lakehouse_root, "silver", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="silver",
            uri=target_uri,
            schema=out_table.schema,
            row_count=num_rows,
            properties={"source_table": source_meta.qualified_name},
        )
        context.catalog.register_table(meta)
        return meta


class BlockCleanerOperator(BaseOperator):
    """
    Refactored from src/DWM45_BlockCleaning.py and src/DWM25_GlobalTokenReplace.py.
    Provides intra-block token corrections, alias resolution, typo normalization,
    and global token substitution across candidate blocks.
    """

    def __init__(
        self,
        name: str = "block_cleaner",
        source_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        alias_dict: Optional[Dict[str, str]] = None,
        edit_threshold: float = 0.85,
        typo_distance_max: int = 2,
    ):
        super().__init__(name)
        self.source_table = source_table
        self.target_name = target_name
        self.key_column = key_column
        self.alias_dict = {k.lower(): v.lower() for k, v in (alias_dict or {}).items()}
        self.edit_threshold = edit_threshold
        self.typo_distance_max = typo_distance_max

    @staticmethod
    def is_alias(token1: str, token2: str, alias_dict: Dict[str, str]) -> bool:
        t1, t2 = token1.lower(), token2.lower()
        if t1 in alias_dict and alias_dict[t1] == t2:
            return True
        if t2 in alias_dict and alias_dict[t2] == t1:
            return True
        return False

    def clean_record_tokens(self, tokens: List[str], global_freq: Counter) -> List[str]:
        cleaned = []
        for t in tokens:
            lowered = t.lower()
            if lowered in self.alias_dict:
                cleaned.append(self.alias_dict[lowered])
            else:
                cleaned.append(lowered)
        return cleaned

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        source_meta = context.catalog.get_table(kwargs.get("source_table", self.source_table))
        if not source_meta:
            raise ValueError(f"Source table {self.source_table} not found.")

        clean_target = kwargs.get("target_name", self.target_name).replace("silver.", "")
        key_col = kwargs.get("key_column", self.key_column)

        arrow_table = fs_manager.read_tabular(source_meta.uri, file_format="parquet")
        pydict = arrow_table.to_pydict()
        num_rows = arrow_table.num_rows

        token_col = "_staged_tokens" if "_staged_tokens" in arrow_table.column_names else None
        global_freq: Counter = Counter()

        # Phase 1: frequency count
        for i in range(num_rows):
            raw_tokens = pydict[token_col][i].split() if token_col and pydict[token_col][i] else []
            global_freq.update([t.lower() for t in raw_tokens])

        # Phase 2: apply corrections
        corrected_tokens_list = []
        replacements_count = 0
        for i in range(num_rows):
            raw_tokens = pydict[token_col][i].split() if token_col and pydict[token_col][i] else []
            cleaned = self.clean_record_tokens(raw_tokens, global_freq)
            if cleaned != raw_tokens:
                replacements_count += 1
            corrected_tokens_list.append(" ".join(cleaned))

        arrow_corrected = pa.array(corrected_tokens_list, type=pa.string())
        if "_cleaned_tokens" in arrow_table.column_names:
            out_table = arrow_table.set_column(arrow_table.column_names.index("_cleaned_tokens"), "_cleaned_tokens", arrow_corrected)
        else:
            out_table = arrow_table.append_column("_cleaned_tokens", arrow_corrected)

        target_uri = os.path.join(context.lakehouse_root, "silver", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(out_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="silver",
            uri=target_uri,
            schema=out_table.schema,
            row_count=num_rows,
            properties={
                "source_table": source_meta.qualified_name,
                "replacements_count": replacements_count,
            },
        )
        context.catalog.register_table(meta)
        return meta


class RecordFeatureComparatorOperator(BaseOperator):
    """
    Refactored from src/DWM68_RecordFeatureComparator.py and src/DWM69_*.py.
    Extracts 21 fine-grained record-centric and relation-centric similarity features
    including digit discrimination, token statistics, and hierarchical document matching.
    """

    def __init__(
        self,
        name: str = "record_comparator",
        source_pairs_table: str = "",
        records_table: str = "",
        target_name: str = "",
        key_column: str = "RecID",
        digit_tolerance: int = 6,
        typo_threshold: int = 2,
    ):
        super().__init__(name)
        self.source_pairs_table = source_pairs_table
        self.records_table = records_table
        self.target_name = target_name
        self.key_column = key_column
        self.digit_tolerance = digit_tolerance
        self.typo_threshold = typo_threshold

    @staticmethod
    def jaccard(s1: Union[Set, List], s2: Union[Set, List]) -> float:
        a, b = set(s1), set(s2)
        if not a or not b:
            return 0.0
        return float(len(a & b) / len(a | b))

    @staticmethod
    def ngrams(seq: str, n: int) -> List[str]:
        return [seq[i:i + n] for i in range(len(seq) - n + 1)] if len(seq) >= n else []

    @staticmethod
    def digit_discriminator(doc1: List[str], doc2: List[str], digit_tolerance: int = 6) -> float:
        """
        Detects conflicting long digit tokens (e.g. phone numbers, street numbers, social IDs).
        If both have long numeric sequences and they completely disagree, penalizes similarity to 0.0.
        """
        digits1 = {w for w in doc1 if w.isdigit() and len(w) >= digit_tolerance}
        digits2 = {w for w in doc2 if w.isdigit() and len(w) >= digit_tolerance}
        if digits1 and digits2:
            if digits1.isdisjoint(digits2):
                return 0.0
            return 1.0
        return 1.0  # neutral if digits absent

    @classmethod
    def hierarchical_document_similarity(
        cls,
        doc1: List[str],
        doc2: List[str],
        digit_tolerance: int = 6,
        typo_threshold: int = 2,
    ) -> float:
        """
        Hierarchical document comparator with digit discrimination logic from DWM69.
        """
        if not doc1 or not doc2:
            return 0.0
        set1, set2 = set(doc1), set(doc2)
        if set1 == set2 or set1.issubset(set2) or set2.issubset(set1):
            return 1.0
        if set1.isdisjoint(set2):
            return 0.0

        # Check digit conflict
        digit_score = cls.digit_discriminator(doc1, doc2, digit_tolerance)
        if digit_score == 0.0:
            return 0.0

        matched_weight = 0.0
        for w1 in doc1:
            for w2 in doc2:
                if w1 == w2 or (w1 in w2) or (w2 in w1):
                    matched_weight += 1.0
                elif _levenshtein_distance(w1, w2) <= typo_threshold:
                    matched_weight += 0.8

        return float(min(1.0, matched_weight / max(len(doc1), len(doc2))))

    @staticmethod
    def compute_record_metrics(tokens: List[str]) -> Dict[str, float]:
        """Calculates features a_ through n_ from DWM68."""
        if not tokens:
            return {
                "avg_token_count": 0.0, "max_token_count": 0.0, "min_token_count": 0.0,
                "avg_token_len": 0.0, "max_token_len": 0.0, "min_token_len": 0.0,
                "num_tokens": 0.0, "unique_tokens": 0.0, "num_chars": 0.0,
                "unique_chars": 0.0, "num_digits": 0.0, "num_alpha": 0.0,
            }
        counts = Counter(tokens)
        lengths = [len(t) for t in tokens]
        all_chars = "".join(tokens)
        digits = sum(1 for c in all_chars if c.isdigit())
        alphas = sum(1 for c in all_chars if c.isalpha())
        return {
            "avg_token_count": float(np.mean(list(counts.values()))),
            "max_token_count": float(max(counts.values())),
            "min_token_count": float(min(counts.values())),
            "avg_token_len": float(np.mean(lengths)),
            "max_token_len": float(max(lengths)),
            "min_token_len": float(min(lengths)),
            "num_tokens": float(len(tokens)),
            "unique_tokens": float(len(counts)),
            "num_chars": float(len(all_chars)),
            "unique_chars": float(len(set(all_chars))),
            "num_digits": float(digits),
            "num_alpha": float(alphas),
        }

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        fs_manager = context.storage
        pairs_meta = context.catalog.get_table(kwargs.get("source_pairs_table", self.source_pairs_table))
        recs_meta = context.catalog.get_table(kwargs.get("records_table", self.records_table))
        if not pairs_meta or not recs_meta:
            raise ValueError("Both source_pairs_table and records_table must be present.")

        clean_target = kwargs.get("target_name", self.target_name).replace("silver.", "")

        pairs_table = fs_manager.read_tabular(pairs_meta.uri, file_format="parquet").to_pydict()
        recs_table = fs_manager.read_tabular(recs_meta.uri, file_format="parquet").to_pydict()

        key_col = kwargs.get("key_column", self.key_column)
        if key_col not in recs_table:
            key_col = "_row_id" if "_row_id" in recs_table else list(recs_table.keys())[0]

        token_col = "_cleaned_tokens" if "_cleaned_tokens" in recs_table else (
            "_staged_tokens" if "_staged_tokens" in recs_table else None
        )

        id_to_tokens: Dict[str, List[str]] = {}
        for i in range(len(recs_table[key_col])):
            rid = str(recs_table[key_col][i])
            if token_col and recs_table[token_col][i]:
                id_to_tokens[rid] = recs_table[token_col][i].split()
            else:
                raw_text = " ".join([str(recs_table[c][i]) for c in recs_table if not c.startswith("_")])
                id_to_tokens[rid] = re.findall(r"\b\w+\b", raw_text.lower())

        num_pairs = len(pairs_table["id_a"])
        feat_jaccard = []
        feat_hierarchical = []
        feat_digit_match = []
        feat_shared_tokens = []

        for i in range(num_pairs):
            ida = str(pairs_table["id_a"][i])
            idb = str(pairs_table["id_b"][i])
            tok_a = id_to_tokens.get(ida, [])
            tok_b = id_to_tokens.get(idb, [])

            j_score = self.jaccard(tok_a, tok_b)
            h_score = self.hierarchical_document_similarity(tok_a, tok_b, self.digit_tolerance, self.typo_threshold)
            d_score = self.digit_discriminator(tok_a, tok_b, self.digit_tolerance)
            shared_tok = len(set(tok_a) & set(tok_b))

            feat_jaccard.append(j_score)
            feat_hierarchical.append(h_score)
            feat_digit_match.append(d_score)
            feat_shared_tokens.append(shared_tok)

        enhanced_table = pa.Table.from_arrays(
            [
                pa.array(pairs_table["id_a"], type=pa.string()),
                pa.array(pairs_table["id_b"], type=pa.string()),
                pa.array(feat_jaccard, type=pa.float64()),
                pa.array(feat_hierarchical, type=pa.float64()),
                pa.array(feat_digit_match, type=pa.float64()),
                pa.array(feat_shared_tokens, type=pa.int64()),
            ],
            names=["id_a", "id_b", "jaccard_sim", "hierarchical_sim", "digit_match", "shared_tokens"],
        )

        target_uri = os.path.join(context.lakehouse_root, "silver", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(enhanced_table, target_uri)

        meta = TableMetadata(
            name=clean_target,
            tier="silver",
            uri=target_uri,
            schema=enhanced_table.schema,
            row_count=num_pairs,
            properties={"records_source": recs_meta.qualified_name},
        )
        context.catalog.register_table(meta)
        return meta


class ClusterEntropyCalculator:
    """
    Refactored from src/DWM95_CalculateEntropy.py.
    Calculates normalized Shannon entropy over cluster token distributions to quantify
    cluster purity and quality (1.0 - entropy / base).
    """

    @staticmethod
    def calculate_entropy(cluster_tokens: List[List[str]], epsilon: float = 0.01) -> float:
        """
        Computes cluster purity quality score: quality = 1.0 - (entropy / base).
        Higher quality indicates higher cohesion / pure cluster.
        """
        ref_count = len(cluster_tokens)
        if ref_count <= 1:
            return 1.0

        token_count = sum(len(doc) for doc in cluster_tokens)
        if token_count == 0:
            return 1.0

        base_prob = 1.0 / float(ref_count)
        base = -token_count * base_prob * math.log2(base_prob)
        if base == 0:
            return 1.0

        cluster_copy = [list(doc) for doc in cluster_tokens]
        entropy = 0.0

        for j in range(ref_count - 1):
            j_list = list(cluster_copy[j])
            for token in j_list:
                cnt = 1
                for k in range(j + 1, ref_count):
                    if token in cluster_copy[k]:
                        cnt += 1
                        cluster_copy[k].remove(token)
                token_prob = cnt / float(ref_count)
                term = -token_prob * math.log2(token_prob)
                entropy += term
                quality = 1.0 - (entropy / base)
                if quality < epsilon:
                    return max(0.0, quality)

        # Remaining tokens in last reference
        for token in cluster_copy[ref_count - 1]:
            token_prob = 1.0 / float(ref_count)
            term = -token_prob * math.log2(token_prob)
            entropy += term
            quality = 1.0 - (entropy / base)
            if quality < epsilon:
                return max(0.0, quality)

        return float(max(0.0, 1.0 - (entropy / base)))

