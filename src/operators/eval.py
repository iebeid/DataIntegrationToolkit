import os
import re
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

import pyarrow as pa
from src.operators.base import BaseOperator


def _combinations_2(n: int) -> int:
    """Computes n * (n - 1) / 2."""
    if n < 2:
        return 0
    return n * (n - 1) // 2


def _compute_adjusted_rand_index(labels_true: List[Any], labels_pred: List[Any]) -> float:
    """
    Computes Adjusted Rand Index (ARI) without strictly requiring scikit-learn.
    Uses sklearn if available, otherwise exact contingency table formulation.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_true, labels_pred))
    except ImportError:
        pass

    n = len(labels_true)
    if n <= 1:
        return 1.0

    # Build contingency table
    contingency = defaultdict(lambda: defaultdict(int))
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)

    for t, p in zip(labels_true, labels_pred):
        contingency[t][p] += 1

    sum_comb_nij = sum(
        _combinations_2(count)
        for t_dict in contingency.values()
        for count in t_dict.values()
    )
    sum_comb_ai = sum(_combinations_2(c) for c in true_counts.values())
    sum_comb_bj = sum(_combinations_2(c) for c in pred_counts.values())
    comb_n2 = _combinations_2(n)

    if comb_n2 == 0:
        return 1.0

    expected_index = (sum_comb_ai * sum_comb_bj) / comb_n2
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)

    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if sum_comb_nij == expected_index else 0.0

    return float((sum_comb_nij - expected_index) / denominator)


@dataclass
class EvaluationReport:
    """Encapsulates benchmark entity resolution and blocking metrics."""

    record_count: int
    universe_pairs: int
    linked_pairs: int  # L
    equiv_pairs: int   # E
    true_positives: int  # TP
    false_positives: int  # FP
    false_negatives: int  # FN
    true_negatives: int  # TN
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    balanced_accuracy: float
    fpr: float
    tpr: float
    tnr: float
    cluster_purity: float
    adjusted_rand_index: float
    cluster_count: int
    singleton_count: int
    max_cluster_size: int
    cluster_profile: Dict[int, int] = field(default_factory=dict)
    
    # Optional blocking metrics
    candidate_pairs: Optional[int] = None
    reduction_ratio: Optional[float] = None
    pairs_completeness: Optional[float] = None
    pairs_quality: Optional[float] = None

    def print_report(self) -> None:
        """Prints a comprehensive benchmark evaluation report in Talburt EvaluatER format."""
        sep = "-" * 42
        print(sep)
        print("   LAKEHOUSE ENTITY RESOLUTION EVALUATION  ")
        print(sep)
        print(f"Evaluated Records (N)    : {self.record_count}")
        print(f"Universe Pairs (N*(N-1)/2): {self.universe_pairs}")
        print(f"Predicted Linked Pairs (L): {self.linked_pairs}")
        print(f"Ground Truth Pairs (E)    : {self.equiv_pairs}")
        print(sep)
        print(f"True Positives (TP)       : {self.true_positives}")
        print(f"False Positives (FP)      : {self.false_positives}")
        print(f"False Negatives (FN)      : {self.false_negatives}")
        print(f"True Negatives (TN)       : {self.true_negatives}")
        print(sep)
        print(f"Precision                 : {self.precision:.4f}")
        print(f"Recall (Sensitivity)      : {self.recall:.4f}")
        print(f"F1-Score                  : {self.f1_score:.4f}")
        print(f"Accuracy                  : {self.accuracy:.4f}")
        print(f"Balanced Accuracy         : {self.balanced_accuracy:.4f}")
        print(f"False Positive Rate (FPR) : {self.fpr:.4f}")
        print(f"True Positive Rate (TPR)  : {self.tpr:.4f}")
        print(f"True Negative Rate (TNR)  : {self.tnr:.4f}")
        print(f"Cluster Purity            : {self.cluster_purity:.4f}")
        print(f"Adjusted Rand Index (ARI) : {self.adjusted_rand_index:.4f}")
        print(sep)
        print(f"Total Predicted Clusters  : {self.cluster_count}")
        print(f"Singletons (Size 1)       : {self.singleton_count}")
        print(f"Largest Cluster Size      : {self.max_cluster_size}")

        if self.candidate_pairs is not None:
            print(sep)
            print("         BLOCKING METRICS           ")
            print(sep)
            print(f"Candidate Pairs (C)       : {self.candidate_pairs}")
            print(f"Reduction Ratio (RR)      : {self.reduction_ratio:.4f}")
            print(f"Pairs Completeness (PC)   : {self.pairs_completeness:.4f}")
            print(f"Pairs Quality (PQ)        : {self.pairs_quality:.4f}")

        print(sep)
        print("          CLUSTER PROFILE           ")
        print("Size\tCount\tTotal Records")
        total_records = 0
        for sz in sorted(self.cluster_profile.keys()):
            cnt = self.cluster_profile[sz]
            tot = sz * cnt
            total_records += tot
            print(f"{sz}\t{cnt}\t{tot}")
        print(f"Total\t\t{total_records}")
        print(sep)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics into a dictionary."""
        d = {
            "record_count": self.record_count,
            "universe_pairs": self.universe_pairs,
            "linked_pairs": self.linked_pairs,
            "equiv_pairs": self.equiv_pairs,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "fpr": round(self.fpr, 4),
            "tpr": round(self.tpr, 4),
            "tnr": round(self.tnr, 4),
            "cluster_purity": round(self.cluster_purity, 4),
            "adjusted_rand_index": round(self.adjusted_rand_index, 4),
            "cluster_count": self.cluster_count,
            "singleton_count": self.singleton_count,
            "max_cluster_size": self.max_cluster_size,
            "cluster_profile": self.cluster_profile,
        }
        if self.candidate_pairs is not None:
            d["candidate_pairs"] = self.candidate_pairs
            d["reduction_ratio"] = round(self.reduction_ratio, 4) if self.reduction_ratio is not None else None
            d["pairs_completeness"] = round(self.pairs_completeness, 4) if self.pairs_completeness is not None else None
            d["pairs_quality"] = round(self.pairs_quality, 4) if self.pairs_quality is not None else None
        return d

    def to_table(self) -> pa.Table:
        """Converts summary metrics into a structured PyArrow Table."""
        metrics_dict = self.to_dict()
        names = []
        values = []
        for k, v in metrics_dict.items():
            if k == "cluster_profile":
                continue
            names.append(k)
            values.append(str(v))
        return pa.Table.from_arrays(
            [pa.array(names), pa.array(values)],
            names=["metric_name", "metric_value"],
        )


class EvaluateResolutionOperator(BaseOperator):
    """
    Evaluates resolved entity clusters against benchmark ground truth files.
    Calculates pairwise metrics (TP, FP, FN, TN, Precision, Recall, F1),
    cluster metrics (Purity, ARI, Size Profile), and blocking metrics (Reduction Ratio, Pairs Completeness).
    """

    def __init__(
        self,
        target_table: str,
        truth_file: str,
        pairs_table: Optional[str] = None,
        key_column: str = "RecID",
        cluster_column: str = "canonical_entity_id",
        output_table: Optional[str] = None,
    ):
        self.target_table = target_table
        self.truth_file = truth_file.replace("\\", "/")
        self.pairs_table = pairs_table
        self.key_column = key_column
        self.cluster_column = cluster_column
        self.output_table = output_table

    def _read_truth_file(self) -> Dict[str, str]:
        """Loads truth file into a dictionary of {rec_id: truth_cluster_id}."""
        truth_dict = {}
        resolved_truth = self.truth_file
        if not os.path.exists(resolved_truth):
            filename = os.path.basename(resolved_truth)
            for root, _, files in os.walk("data"):
                if filename in files:
                    resolved_truth = os.path.join(root, filename).replace("\\", "/")
                    break

        if not os.path.exists(resolved_truth):
            raise FileNotFoundError(f"Ground truth file not found: {self.truth_file}")

        with open(resolved_truth, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                rec_id, truth_id = parts[0], parts[1]
                # Skip header if present
                if idx == 0 and (rec_id.lower() in ("recid", "id", "record_id") or not truth_id.isdigit()):
                    if rec_id.lower() == "recid":
                        continue
                truth_dict[rec_id] = truth_id
        return truth_dict

    def execute(self, context: Any, **kwargs) -> EvaluationReport:
        table = context.read_table(self.target_table)
        cols = table.column_names

        if self.key_column not in cols:
            raise ValueError(f"Key column '{self.key_column}' not found in table '{self.target_table}'. Available: {cols}")

        actual_cluster_col = self.cluster_column
        if actual_cluster_col not in cols:
            # Fallback checks
            if "canonical_entity_id" in cols:
                actual_cluster_col = "canonical_entity_id"
            elif "entity_cluster_id" in cols:
                actual_cluster_col = "entity_cluster_id"
            elif "cluster_id" in cols:
                actual_cluster_col = "cluster_id"
            else:
                raise ValueError(f"Cluster column '{self.cluster_column}' not found in table '{self.target_table}'. Available: {cols}")

        rec_ids = [str(x) for x in table[self.key_column].to_pylist()]
        cluster_ids = [str(x) for x in table[actual_cluster_col].to_pylist()]

        pred_map = dict(zip(rec_ids, cluster_ids))
        truth_map = self._read_truth_file()

        n = len(rec_ids)
        if n == 0:
            raise ValueError("Target table contains 0 records.")

        universe_pairs = _combinations_2(n)

        # Build erDict: recID -> (clusterID, truthID)
        er_dict = {}
        for r_id in rec_ids:
            c_id = pred_map[r_id]
            t_id = truth_map.get(r_id, f"_singleton_{r_id}")
            er_dict[r_id] = (c_id, t_id)

        linked_pairs_dict = Counter()
        equiv_pairs_dict = Counter()
        true_pos_dict = Counter()

        labels_true = []
        labels_pred = []

        for pair in er_dict.values():
            c_id, t_id = pair
            linked_pairs_dict[c_id] += 1
            equiv_pairs_dict[t_id] += 1
            true_pos_dict[pair] += 1
            labels_pred.append(c_id)
            labels_true.append(t_id)

        L = sum(_combinations_2(cnt) for cnt in linked_pairs_dict.values())
        E = sum(_combinations_2(cnt) for cnt in equiv_pairs_dict.values())
        TP = sum(_combinations_2(cnt) for cnt in true_pos_dict.values())
        FP = L - TP
        FN = E - TP
        TN = max(0, universe_pairs - (TP + FP + FN))

        precision = (TP / L) if L > 0 else 1.0
        recall = (TP / E) if E > 0 else 1.0
        f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        fpr = (FP / (FP + TN)) if (FP + TN) > 0 else 0.0
        tpr = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        tnr = 1.0 - fpr

        total_pairs = TP + TN + FP + FN
        accuracy = ((TP + TN) / total_pairs) if total_pairs > 0 else 1.0
        balanced_accuracy = (tpr + tnr) / 2.0

        # Cluster Purity: (1 / n) * sum_{c} max_{t} |C_c \cap E_t|
        cluster_truth_counts = defaultdict(lambda: Counter())
        for c_id, t_id in er_dict.values():
            cluster_truth_counts[c_id][t_id] += 1

        purity_sum = sum(
            counts.most_common(1)[0][1]
            for counts in cluster_truth_counts.values()
            if counts
        )
        cluster_purity = purity_sum / n if n > 0 else 1.0

        # Adjusted Rand Index
        ari = _compute_adjusted_rand_index(labels_true, labels_pred)

        # Cluster Profile
        cluster_profile = Counter()
        for cnt in linked_pairs_dict.values():
            cluster_profile[cnt] += 1

        cluster_count = len(linked_pairs_dict)
        singleton_count = cluster_profile.get(1, 0)
        max_cluster_size = max(linked_pairs_dict.values()) if linked_pairs_dict else 0

        # Blocking metrics if pairs_table is provided
        candidate_pairs = None
        reduction_ratio = None
        pairs_completeness = None
        pairs_quality = None

        if self.pairs_table:
            blocks_table = context.read_table(self.pairs_table)
            candidate_pairs = blocks_table.num_rows

            if universe_pairs > 0:
                reduction_ratio = 1.0 - (candidate_pairs / universe_pairs)
            else:
                reduction_ratio = 1.0

            # Count true matches captured in blocks
            p_cols = blocks_table.column_names
            c1 = "row_id_1" if "row_id_1" in p_cols else p_cols[0]
            c2 = "row_id_2" if "row_id_2" in p_cols else p_cols[1]

            pairs_r1 = [str(x) for x in blocks_table[c1].to_pylist()]
            pairs_r2 = [str(x) for x in blocks_table[c2].to_pylist()]

            # Determine if candidate pairs store rec_ids or row_index integers
            is_rec_id = any(p in truth_map for p in pairs_r1[:min(10, len(pairs_r1))])

            true_matches_captured = 0
            for r1, r2 in zip(pairs_r1, pairs_r2):
                if is_rec_id:
                    k1, k2 = r1, r2
                else:
                    try:
                        k1 = rec_ids[int(r1)]
                        k2 = rec_ids[int(r2)]
                    except (ValueError, IndexError):
                        continue

                t1 = truth_map.get(k1)
                t2 = truth_map.get(k2)
                if t1 is not None and t2 is not None and t1 == t2:
                    true_matches_captured += 1

            pairs_completeness = (true_matches_captured / E) if E > 0 else 1.0
            pairs_quality = (true_matches_captured / candidate_pairs) if candidate_pairs > 0 else 0.0

        report = EvaluationReport(
            record_count=n,
            universe_pairs=universe_pairs,
            linked_pairs=L,
            equiv_pairs=E,
            true_positives=TP,
            false_positives=FP,
            false_negatives=FN,
            true_negatives=TN,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            fpr=fpr,
            tpr=tpr,
            tnr=tnr,
            cluster_purity=cluster_purity,
            adjusted_rand_index=ari,
            cluster_count=cluster_count,
            singleton_count=singleton_count,
            max_cluster_size=max_cluster_size,
            cluster_profile=dict(cluster_profile),
            candidate_pairs=candidate_pairs,
            reduction_ratio=reduction_ratio,
            pairs_completeness=pairs_completeness,
            pairs_quality=pairs_quality,
        )

        if self.output_table:
            # Register in Lakehouse catalog
            out_tbl = report.to_table()
            target_uri = os.path.join(context.lakehouse_root, "gold", f"{self.output_table}.parquet").replace("\\", "/")
            context.storage.write_parquet(out_tbl, target_uri)
            from src.catalog.table import TableMetadata
            meta = TableMetadata(
                name=self.output_table,
                tier="gold",
                uri=target_uri,
                schema=out_tbl.schema,
                row_count=out_tbl.num_rows,
                properties={"evaluated_target": self.target_table, "truth_file": self.truth_file},
            )
            context.catalog.register_table(meta)

        return report


class DataQualityProfilerOperator(BaseOperator):
    """
    Profiles table data quality: column completeness, distinctness, entropy,
    and token distributions across Bronze/Silver/Gold tiers.
    """

    def __init__(self, target_table: str):
        self.target_table = target_table

    def execute(self, context: Any, **kwargs) -> Dict[str, Any]:
        table = context.read_table(self.target_table)
        n = table.num_rows
        profile = {
            "table_name": self.target_table,
            "row_count": n,
            "column_count": table.num_columns,
            "columns": {},
        }

        for col_name in table.column_names:
            col_arr = table[col_name].to_pylist()
            null_count = sum(1 for x in col_arr if x is None or str(x).strip() == "")
            completeness = (n - null_count) / n if n > 0 else 1.0
            distinct_values = set(str(x) for x in col_arr if x is not None)
            distinct_count = len(distinct_values)

            col_prof = {
                "null_count": null_count,
                "completeness_ratio": round(completeness, 4),
                "distinct_count": distinct_count,
                "cardinality_ratio": round(distinct_count / n, 4) if n > 0 else 0.0,
            }
            profile["columns"][col_name] = col_prof

        return profile


# ==============================================================================
# BENCHMARK DATASET LOADERS & ERROR LOOKUP PROFILING (Synthesized from
# QuantifyQualityUsingErrorLookup, PrepareDBLPACMDataset, and ConfigER)
# ==============================================================================

@dataclass
class ConfigER:
    """Canonical Entity Resolution configuration parameters matching src/ConfigER.py."""
    input_file_name: str = ""
    truth_filename: str = ""
    log_filename: str = ""
    delimiter: str = ","
    percentile_of_blocking_words: float = 90.0
    percentile_of_stop_words: float = 100.0
    percentile_of_discriminate_words_lengths: float = 80.0
    lsh_permutations: int = 128
    lsh_threshold: float = 0.5
    typo_count: int = 2


class ErrorLookupProfiler(BaseOperator):
    """
    Refactored from src/QuantifyQualityUsingErrorLookup.py.
    Provides diagnostic error breakdown for entity resolution predictions against ground truth,
    isolating False Positives (over-clustering) and False Negatives (missed true links).
    """

    def __init__(
        self,
        target_table: str,
        truth_file: str,
        records_table: Optional[str] = None,
        key_column: str = "RecID",
        cluster_col: str = "canonical_entity_id",
    ):
        super().__init__("error_lookup")
        self.target_table = target_table
        self.truth_file = truth_file
        self.records_table = records_table
        self.key_column = key_column
        self.cluster_col = cluster_col

    def execute(self, context: Any, **kwargs) -> Dict[str, Any]:
        pred_tbl = context.read_table(self.target_table)
        pydict = pred_tbl.to_pydict()
        n = pred_tbl.num_rows

        key_col = self.key_column
        if key_col not in pred_tbl.column_names:
            key_col = "_row_id" if "_row_id" in pred_tbl.column_names else pred_tbl.column_names[0]
        c_col = self.cluster_col if self.cluster_col in pred_tbl.column_names else "canonical_entity_id"

        # Load truth map
        truth_map: Dict[str, str] = {}
        if os.path.exists(self.truth_file):
            with open(self.truth_file, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",") if "," in line else line.split()
                    if len(parts) >= 2:
                        truth_map[str(parts[0]).strip()] = str(parts[1]).strip()

        # Build clusters
        pred_clusters: Dict[str, List[str]] = defaultdict(list)
        for i in range(n):
            rid = str(pydict[key_col][i])
            cid = str(pydict[c_col][i])
            pred_clusters[cid].append(rid)

        false_positive_pairs = []
        true_positive_pairs = []
        for cid, members in pred_clusters.items():
            if len(members) > 1:
                for idx1 in range(len(members)):
                    for idx2 in range(idx1 + 1, len(members)):
                        m1, m2 = members[idx1], members[idx2]
                        t1, t2 = truth_map.get(m1), truth_map.get(m2)
                        if t1 is not None and t2 is not None:
                            if t1 == t2:
                                true_positive_pairs.append((m1, m2, cid))
                            else:
                                false_positive_pairs.append((m1, m2, cid, t1, t2))

        # Check false negatives
        truth_clusters = defaultdict(list)
        for rid, tid in truth_map.items():
            truth_clusters[tid].append(rid)

        rec_to_pred_cid = {str(pydict[key_col][i]): str(pydict[c_col][i]) for i in range(n)}
        false_negative_pairs = []
        for tid, members in truth_clusters.items():
            if len(members) > 1:
                for idx1 in range(len(members)):
                    for idx2 in range(idx1 + 1, len(members)):
                        m1, m2 = members[idx1], members[idx2]
                        if m1 in rec_to_pred_cid and m2 in rec_to_pred_cid:
                            if rec_to_pred_cid[m1] != rec_to_pred_cid[m2]:
                                false_negative_pairs.append((m1, m2, tid, rec_to_pred_cid[m1], rec_to_pred_cid[m2]))

        return {
            "evaluated_table": self.target_table,
            "total_records": n,
            "true_positive_count": len(true_positive_pairs),
            "false_positive_count": len(false_positive_pairs),
            "false_negative_count": len(false_negative_pairs),
            "sample_false_positives": false_positive_pairs[:10],
            "sample_false_negatives": false_negative_pairs[:10],
        }


class BenchmarkDatasetLoader(BaseOperator):
    """
    Refactored from src/PrepareDBLPACMDataset.py, src/PrepareDBLPGoogleDataset.py,
    and src/PrepareAffiliationsDataset.py.
    Standardized ingestion and schema normalization for classic entity resolution benchmarks.
    """

    def __init__(
        self,
        benchmark_name: str,
        source_path: str,
        target_name: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ):
        super().__init__("benchmark_loader")
        self.benchmark_name = benchmark_name.lower()
        self.source_path = source_path
        self.target_name = target_name
        self.delimiter = delimiter
        self.encoding = encoding

    def execute(self, context: Any, **kwargs) -> Any:
        import csv
        clean_target = self.target_name.replace("bronze.", "")
        target_uri = os.path.join(context.lakehouse_root, "bronze", f"{clean_target}.parquet").replace("\\", "/")

        rows = []
        with open(self.source_path, "r", encoding=self.encoding, errors="ignore") as fp:
            reader = csv.reader(fp, delimiter=self.delimiter)
            header = next(reader, None)
            if not header:
                raise ValueError(f"Empty benchmark file: {self.source_path}")
            clean_headers = [re.sub(r"\W+", "_", h.strip().lower()) for h in header]
            for row in reader:
                if row:
                    rows.append(row)

        col_dict = defaultdict(list)
        for row in rows:
            for j, val in enumerate(row):
                h = clean_headers[j] if j < len(clean_headers) else f"col_{j}"
                col_dict[h].append(val)

        arrays = [pa.array(col_dict[h], type=pa.string()) for h in clean_headers]
        arrow_table = pa.Table.from_arrays(arrays, names=clean_headers)
        context.storage.write_parquet(arrow_table, target_uri)

        from src.catalog.table import TableMetadata
        meta = TableMetadata(
            name=clean_target,
            tier="bronze",
            uri=target_uri,
            schema=arrow_table.schema,
            row_count=arrow_table.num_rows,
            properties={"benchmark": self.benchmark_name, "source_path": self.source_path},
        )
        context.catalog.register_table(meta)
        return meta

