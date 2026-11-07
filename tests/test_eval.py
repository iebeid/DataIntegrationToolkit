import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.context import IngestionContext


def run_benchmark_tests():
    test_lakehouse = os.path.join(os.path.dirname(__file__), "lakehouse_eval_test")
    if os.path.exists(test_lakehouse):
        shutil.rmtree(test_lakehouse)

    ctx = IngestionContext(lakehouse_root=test_lakehouse)

    print("=================================================================")
    print("  TEST 1: S1G Benchmark (50 records against truthABCgoodDQ.txt)  ")
    print("=================================================================")

    # 1. Ingest
    ctx.sql("INGEST FROM 'data/S1G.txt' INTO bronze.s1g WITH (format = 'csv', delimiter = ',');")

    # 2. Stage
    ctx.sql("CREATE STAGE silver.s1g_staged FROM bronze.s1g WITH (key = 'RecID', stopword_threshold = 0.05);")

    # 3. Block with MinHash LSH
    ctx.sql("CREATE BLOCKS silver.s1g_blocks FROM silver.s1g_staged WITH (method = 'minhash_lsh', num_perm = 64, threshold = 0.4);")

    # 4. Resolve Entities using Louvain
    ctx.sql("""
        RESOLVE ENTITIES INTO gold.s1g_resolved
        FROM silver.s1g_staged
        USING BLOCKS silver.s1g_blocks
        WITH (
            clustering = 'louvain',
            similarity = 'jaccard',
            threshold = 0.5,
            weights = 'fname:0.3;lname:0.4;address:0.2;zip:0.1'
        );
    """)

    # 5. Evaluate against ground truth
    report_s1g = ctx.sql("""
        EVALUATE gold.s1g_resolved
        AGAINST TRUTH 'data/truthABCgoodDQ.txt'
        USING BLOCKS silver.s1g_blocks
        WITH (key = 'RecID');
    """)

    report_s1g.print_report()

    assert report_s1g.record_count == 50
    assert report_s1g.f1_score >= 0.80, f"Expected S1G F1 >= 0.80, got {report_s1g.f1_score}"
    assert report_s1g.reduction_ratio >= 0.95, f"Expected RR >= 0.95, got {report_s1g.reduction_ratio}"
    print("[PASS] Test 1 S1G benchmark successfully validated!")

    print("\n=================================================================")
    print("  TEST 2: S3Rest Benchmark (869 restaurants against truthRestaurant.txt) ")
    print("=================================================================")

    # 1. Ingest
    ctx.sql("INGEST FROM 'data/S3Rest.txt' INTO bronze.restaurants WITH (format = 'csv', delimiter = ',');")

    # 2. Stage
    ctx.sql("CREATE STAGE silver.restaurants_staged FROM bronze.restaurants WITH (key = 'RecID', stopword_threshold = 0.05);")

    # 3. Block using inverted index
    ctx.sql("CREATE BLOCKS silver.restaurants_blocks FROM silver.restaurants_staged WITH (method = 'token_inverted_index', max_block_size = 100);")

    # 4. Resolve with Transitive Closure
    ctx.sql("""
        RESOLVE ENTITIES INTO gold.restaurants_resolved
        FROM silver.restaurants_staged
        USING BLOCKS silver.restaurants_blocks
        WITH (
            clustering = 'transitive_closure',
            similarity = 'jaccard',
            threshold = 0.55
        );
    """)

    # 5. Evaluate against ground truth
    report_rest = ctx.sql("""
        EVALUATE gold.restaurants_resolved
        AGAINST TRUTH 'data/truthRestaurant.txt'
        USING BLOCKS silver.restaurants_blocks
        WITH (key = 'RecID', output_table = 'eval_report_restaurants');
    """)

    report_rest.print_report()

    assert report_rest.record_count in (868, 869)
    assert report_rest.f1_score >= 0.70, f"Expected Restaurant F1 >= 0.70, got {report_rest.f1_score}"
    assert report_rest.reduction_ratio >= 0.95, f"Expected Restaurant RR >= 0.95, got {report_rest.reduction_ratio}"
    print("[PASS] Test 2 S3Rest benchmark successfully validated!")

    print("\n=================================================================")
    print("  TEST 3: Table Data Quality Profiler                              ")
    print("=================================================================")
    profile = ctx.sql("PROFILE TABLE bronze.restaurants;")
    print(f"Table Profile for: {profile['table_name']}")
    print(f"Total rows: {profile['row_count']}, Total columns: {profile['column_count']}")
    for col, stats in profile["columns"].items():
        print(f"  Column '{col}': Completeness={stats['completeness_ratio']}, Distinct={stats['distinct_count']}, CardinalityRatio={stats['cardinality_ratio']}")

    assert profile["row_count"] in (868, 869)
    assert "RecID" in profile["columns"]
    print("[PASS] Test 3 Data Quality Profiler successfully validated!")

    # Verify report registered in catalog
    catalog_tables = [t["table"] for t in ctx.show_tables()]
    assert any("eval_report_restaurants" in t for t in catalog_tables), f"Catalog tables: {catalog_tables}"
    print("[PASS] Evaluation report successfully persisted to Lakehouse catalog!")

    # Cleanup
    if os.path.exists(test_lakehouse):
        shutil.rmtree(test_lakehouse)
    print("\n>>> ALL BENCHMARK EVALUATION SUITE TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_benchmark_tests()
