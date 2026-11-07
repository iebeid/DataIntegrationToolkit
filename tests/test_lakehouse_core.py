import os
import shutil
import tempfile
import pytest
from src.context import IngestionContext


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_core_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_bronze_ingest_and_audit_columns(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_path = os.path.join(root, "sample.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("RecID,title,author\n")
        f.write("r1,Graph Theory,Euler\n")
        f.write("r2,Calculus,Newton\n")
        f.write("r3,Graph Theory Revised,Euler L\n")

    meta = ctx.ingest(
        source_uri=csv_path,
        target_name="bronze.books",
        file_format="csv",
    )
    assert meta.row_count == 3
    assert meta.tier == "bronze"

    table = ctx.read_table("bronze.books")
    assert "_lakehouse_ingest_time" in table.column_names
    assert "_source_uri" in table.column_names
    assert "_row_id" in table.column_names
    assert table.num_rows == 3


def test_silver_staging_and_blocking(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_path = os.path.join(root, "entities.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("RecID,name,city\n")
        f.write("e1,Acme Corporation,New York\n")
        f.write("e2,Acme Corp,New York City\n")
        f.write("e3,Beta Tech,San Francisco\n")
        f.write("e4,Beta Technologies,SF\n")

    ctx.ingest(source_uri=csv_path, target_name="bronze.entities", file_format="csv")

    # Staging
    stage_meta = ctx.stage(
        source_table="bronze.entities",
        target_name="silver.staged_entities",
        key_column="RecID",
        stopword_threshold=0.1,
    )
    assert stage_meta.tier == "silver"
    staged = ctx.read_table("silver.staged_entities")
    assert "_staged_tokens" in staged.column_names

    # Inverted Index Blocking
    block_meta = ctx.block(
        source_table="silver.staged_entities",
        target_name="silver.candidate_pairs",
        key_column="RecID",
        method="token_inverted_index",
        max_block_size=50,
    )
    pairs = ctx.read_table("silver.candidate_pairs")
    assert pairs.num_rows > 0
    assert "id_a" in pairs.column_names
    assert "id_b" in pairs.column_names


def test_gold_entity_resolution(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_path = os.path.join(root, "pubs.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("RecID,title\n")
        f.write("p1,Machine Learning Fundamentals\n")
        f.write("p2,Machine Learning Fundamentals 2nd Ed\n")
        f.write("p3,Quantum Physics Primer\n")

    ctx.ingest(source_uri=csv_path, target_name="bronze.pubs", file_format="csv")
    ctx.stage(source_table="bronze.pubs", target_name="silver.staged_pubs", key_column="RecID")
    ctx.block(source_table="silver.staged_pubs", target_name="silver.pairs", key_column="RecID")

    # Resolve using Jaccard & Transitive Closure
    gold_meta = ctx.resolve(
        records_table="silver.staged_pubs",
        pairs_table="silver.pairs",
        target_name="gold.resolved_clusters",
        key_column="RecID",
        similarity_metric="jaccard",
        threshold=0.4,
        clustering="transitive_closure",
    )
    assert gold_meta.tier == "gold"
    clusters = ctx.read_table("gold.resolved_clusters")
    assert "canonical_entity_id" in clusters.column_names
    assert clusters.num_rows == 3
