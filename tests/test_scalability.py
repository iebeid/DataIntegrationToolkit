import os
import shutil
import tempfile
import pytest
import pandas as pd
from toolkit.context import IngestionContext
from toolkit.operators.dask_engine import DaskProtgramHelpers


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_scale_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_delta_lake_acid_and_time_travel(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_file = os.path.join(root, "users.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,82.0\n")

    # Ingest Delta v0
    ctx.sql(f"""
    INGEST bronze.delta_users
    FROM '{csv_file}'
    WITH (format = 'csv', table_format = 'delta')
    """)
    v0_meta = ctx.catalog.get_table("bronze.delta_users")
    assert v0_meta.format == "delta"

    tbl_v0 = ctx.read_table("bronze.delta_users", version=0)
    assert tbl_v0.num_rows == 2

    # Append row and ingest Delta v1
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("3,Charlie,75.0\n")

    ctx.sql(f"""
    INGEST bronze.delta_users
    FROM '{csv_file}'
    WITH (format = 'csv', table_format = 'delta')
    """)
    tbl_v1 = ctx.read_table("bronze.delta_users", version=1)
    assert tbl_v1.num_rows == 3

    # Vacuum
    purged = ctx.vacuum("bronze.delta_users")
    assert purged >= 0


def test_dask_out_of_core_and_helpers(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_file = os.path.join(root, "large.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("id,val\n")
        for i in range(20):
            f.write(f"{i},{i*10}\n")

    ctx.sql(f"""
    INGEST bronze.dask_large
    FROM '{csv_file}'
    WITH (engine = 'dask', npartitions = 2, format = 'csv')
    """)
    ddf = ctx.read_dask_table("bronze.dask_large")
    assert int(ddf.shape[0].compute()) == 20

    # Test Dask N-gram helper
    sample_df = pd.DataFrame({"seq": ["ACDEF", "ACDYY"]})
    ngrams = DaskProtgramHelpers.extract_ngrams_from_partition(sample_df, n=3, seq_col="seq")
    assert len(ngrams) > 0


def test_sql_explain_where_drop(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_file = os.path.join(root, "items.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("id,name,score\n")
        f.write("1,Alpha,100\n")
        f.write("2,Beta,50\n")
        f.write("3,Apex,90\n")

    ctx.ingest(source_uri=csv_file, target_name="bronze.items", file_format="csv")

    # EXPLAIN
    plan = ctx.sql("EXPLAIN SELECT * FROM bronze.items")
    assert "=== Lakehouse Query Plan:" in plan

    # WHERE filter (=, >, LIKE)
    res_eq = ctx.sql("SELECT id, name FROM bronze.items WHERE name = 'Beta'")
    assert len(res_eq) == 1 and res_eq[0]["name"] == "Beta"

    res_gt = ctx.sql("SELECT id, name FROM bronze.items WHERE score > 80")
    assert len(res_gt) == 2

    res_like = ctx.sql("SELECT id, name FROM bronze.items WHERE name LIKE 'A%'")
    assert len(res_like) == 2

    # DROP TABLE PURGE
    drop_res = ctx.sql("DROP TABLE IF EXISTS bronze.items PURGE")
    assert drop_res["dropped"] is True
    assert ctx.catalog.get_table("bronze.items") is None
