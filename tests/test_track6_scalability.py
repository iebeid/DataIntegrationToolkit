import os
import sys
import shutil
import json
import pyarrow as pa
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.context import IngestionContext


def test_track6_scalability_and_ecosystem():
    test_root = os.path.join(os.path.dirname(__file__), "lakehouse_test_track6")
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    ctx = IngestionContext(lakehouse_root=test_root)

    # Prepare sample CSV
    csv_file = os.path.join(test_root, "sample_data.csv")
    os.makedirs(test_root, exist_ok=True)
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("id,name,score,status\n")
        f.write("1,Alice,95.5,active\n")
        f.write("2,Bob,82.0,active\n")
        f.write("3,Charlie,71.5,pending\n")
        f.write("4,Diana,99.0,active\n")
        f.write("5,Evan,64.0,inactive\n")

    print("=" * 70)
    print("STEP 1: Testing Delta Lake ACID Table Format & Time Travel")
    print("=" * 70)

    # 1. Ingest as Delta table
    sql_delta = f"""
    INGEST bronze.delta_users
    FROM '{csv_file}'
    WITH (
        format = 'csv',
        table_format = 'delta'
    )
    """
    meta_v0 = ctx.sql(sql_delta)
    print(f"Ingested Delta table: {meta_v0.name}, format: {meta_v0.format}, rows: {meta_v0.row_count}")
    assert meta_v0.format == "delta"

    # Check that _delta_log directory and 00000000000000000000.json exist
    delta_dir = meta_v0.uri
    if delta_dir.endswith(".parquet"):
        delta_dir = delta_dir[:-8]
    log_file_v0 = os.path.join(delta_dir, "_delta_log", "00000000000000000000.json")
    assert os.path.exists(log_file_v0), f"Expected Delta log file at {log_file_v0}"
    print(f"ACID Transaction Log Verified at: {log_file_v0}")

    # Read back Delta table at version 0
    tbl_v0 = ctx.read_table("bronze.delta_users", version=0)
    assert tbl_v0.num_rows == 5, f"Expected 5 rows, got {tbl_v0.num_rows}"
    print(f"Read Delta snapshot v0: {tbl_v0.num_rows} rows")

    # Overwrite Delta table with additional row to produce version 1
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("6,Fiona,88.5,active\n")

    meta_v1 = ctx.sql(sql_delta)
    log_file_v1 = os.path.join(delta_dir, "_delta_log", "00000000000000000001.json")
    assert os.path.exists(log_file_v1), f"Expected Delta log file v1 at {log_file_v1}"
    print(f"ACID Commit v1 Log Verified at: {log_file_v1}")

    # Test Time-Travel: Version 0 has 5 rows, Version 1 has 6 rows
    snap_v0 = ctx.read_table("bronze.delta_users", version=0)
    snap_v1 = ctx.read_table("bronze.delta_users", version=1)
    print(f"Time-Travel Verification: v0 rows = {snap_v0.num_rows}, v1 rows = {snap_v1.num_rows}")
    assert snap_v0.num_rows == 5, f"Expected 5 rows in v0, got {snap_v0.num_rows}"
    assert snap_v1.num_rows == 6, f"Expected 6 rows in v1, got {snap_v1.num_rows}"

    # Test Vacuum: purges orphaned files
    purged_count = ctx.vacuum("bronze.delta_users")
    print(f"Vacuum completed. Purged unreferenced data files: {purged_count}")

    print("\n" + "=" * 70)
    print("STEP 2: Testing Dask Out-of-Core Execution Engine")
    print("=" * 70)

    # 2. Ingest using Dask engine
    sql_dask = f"""
    INGEST bronze.dask_users
    FROM '{csv_file}'
    WITH (
        engine = 'dask',
        npartitions = 2,
        format = 'csv'
    )
    """
    meta_dask = ctx.sql(sql_dask)
    print(f"Ingested via Dask: {meta_dask.name}, rows: {meta_dask.row_count}, engine: {meta_dask.properties.get('engine')}")
    assert meta_dask.row_count == 6

    # Read as Dask DataFrame
    ddf = ctx.read_dask_table("bronze.dask_users")
    print(f"Dask DataFrame loaded. Partitions: {ddf.npartitions}")
    computed_rows = int(ddf.shape[0].compute())
    assert computed_rows == 6, f"Expected 6 rows computed via Dask, got {computed_rows}"

    # Test Dask N-Gram Helpers
    from toolkit.operators.dask_engine import DaskProtgramHelpers
    sample_seq_df = pd.DataFrame({
        "id": ["p1", "p2"],
        "seq": ["ACDEFGH", "ACDYYZZ"],
    })
    ngrams_part = DaskProtgramHelpers.extract_ngrams_from_partition(sample_seq_df, n=3, seq_col="seq")
    assert len(ngrams_part) > 0
    print(f"Dask N-gram helper extracted {len(ngrams_part)} unique 3-grams from partition.")

    node_map = {ng: i for i, ng in enumerate(ngrams_part["ngram"])}
    edges_part = DaskProtgramHelpers.extract_edges_from_partition(sample_seq_df, n=3, node_to_idx=node_map)
    assert len(edges_part[0][0]) > 0
    print(f"Dask edge helper extracted {len(edges_part[0][0])} unique transitions.")

    print("\n" + "=" * 70)
    print("STEP 3: Testing SQL Grammar: SELECT with WHERE, EXPLAIN, and DROP TABLE")
    print("=" * 70)

    # 3.1 EXPLAIN query
    sql_explain = "EXPLAIN INGEST bronze.test FROM 'data.csv' WITH (format = 'csv')"
    plan_str = ctx.sql(sql_explain)
    print("Execution Plan:")
    print(plan_str)
    assert "=== Lakehouse Query Plan:" in plan_str

    # 3.2 SELECT with WHERE filter (=, >, LIKE)
    res_eq = ctx.sql("SELECT id, name, score FROM bronze.dask_users WHERE status = 'active'")
    print(f"SELECT WHERE status = 'active': {len(res_eq)} rows returned")
    assert len(res_eq) == 4

    res_gt = ctx.sql("SELECT id, name, score FROM bronze.dask_users WHERE score > 85.0")
    print(f"SELECT WHERE score > 85.0: {len(res_gt)} rows returned (Alice: 95.5, Diana: 99.0, Fiona: 88.5)")
    assert len(res_gt) == 3

    res_like = ctx.sql("SELECT id, name FROM bronze.dask_users WHERE name LIKE 'A%'")
    print(f"SELECT WHERE name LIKE 'A%': {res_like}")
    assert len(res_like) == 1 and res_like[0]["name"] == "Alice"

    # 3.3 DROP TABLE [IF EXISTS] [PURGE]
    drop_res = ctx.sql("DROP TABLE IF EXISTS bronze.delta_users PURGE")
    print(f"DROP TABLE Result: {drop_res}")
    assert ctx.catalog.get_table("bronze.delta_users") is None
    assert not os.path.exists(delta_dir), "Expected purged delta directory removed from disk"

    # Cleanup
    shutil.rmtree(test_root)
    print("\n" + "=" * 70)
    print("ALL TRACK 6 SCALABILITY & ECOSYSTEM TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_track6_scalability_and_ecosystem()
