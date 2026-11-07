import os
import shutil
import tempfile
import pytest
from src.context import IngestionContext
from src.operators.eval import (
    EvaluateResolutionOperator,
    DataQualityProfilerOperator,
    _compute_adjusted_rand_index,
)


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_eval_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_adjusted_rand_index_computation():
    labels_true = ["c1", "c1", "c2", "c2", "c3"]
    labels_pred = ["c1", "c1", "c2", "c2", "c3"]
    ari_perfect = _compute_adjusted_rand_index(labels_true, labels_pred)
    assert round(ari_perfect, 4) == 1.0

    labels_noisy = ["c1", "c2", "c1", "c2", "c3"]
    ari_noisy = _compute_adjusted_rand_index(labels_true, labels_noisy)
    assert -1.0 <= ari_noisy <= 1.0


def test_evaluate_entities_operator(temp_lakehouse):
    ctx, root = temp_lakehouse

    # Resolved clusters table
    clusters_csv = os.path.join(root, "resolved.csv")
    with open(clusters_csv, "w", encoding="utf-8") as f:
        f.write("RecID,canonical_entity_id\n")
        f.write("r1,c1\n")
        f.write("r2,c1\n")
        f.write("r3,c2\n")
        f.write("r4,c2\n")
        f.write("r5,c3\n")

    # Ground truth mapping
    truth_csv = os.path.join(root, "truth.csv")
    with open(truth_csv, "w", encoding="utf-8") as f:
        f.write("RecID,TruthID\n")
        f.write("r1,1\n")
        f.write("r2,1\n")
        f.write("r3,2\n")
        f.write("r4,2\n")
        f.write("r5,3\n")

    ctx.ingest(source_uri=clusters_csv, target_name="gold.resolved", file_format="csv")

    report = ctx.evaluate(
        target_table="gold.resolved",
        truth_file=truth_csv,
        key_column="RecID",
        cluster_column="canonical_entity_id",
    )
    assert report.record_count == 5
    assert report.true_positives == 2
    assert report.false_positives == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
    assert report.cluster_purity == 1.0


def test_data_quality_profiler(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_path = os.path.join(root, "customers.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,name,email,score\n")
        f.write("1,Alice,alice@example.com,100\n")
        f.write("2,Bob,,85\n")
        f.write("3,Charlie,charlie@example.com,\n")

    ctx.ingest(source_uri=csv_path, target_name="bronze.customers", file_format="csv")
    profile = ctx.profile("bronze.customers")

    assert profile["row_count"] == 3
    assert profile["column_count"] == 7  # 4 original + 3 audit
    col_profiles = profile["columns"]
    assert col_profiles["id"]["null_count"] == 0
    assert col_profiles["email"]["null_count"] == 1
    assert col_profiles["score"]["null_count"] == 1
    assert round(col_profiles["email"]["completeness_ratio"], 2) == round(2 / 3, 2)
