import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context import LakehouseContext


def test_uniref_ngram_graph_pipeline():
    test_root = "tests/test_lakehouse_uniref"
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    ctx = LakehouseContext(lakehouse_root=test_root)
    sample_fasta = "data/bioinformatics/uniref50_sample.fasta"

    # Step 1: Ingest 50 sequences from sample
    meta1 = ctx.sql(f"INGEST FASTA bronze.uniref_sequences FROM '{sample_fasta}' WITH (sample = 50);")
    assert meta1.row_count == 50, f"Expected 50 sequences, got {meta1.row_count}"
    
    seq_tbl = ctx.read_table("bronze.uniref_sequences")
    assert "sequence_id" in seq_tbl.column_names
    assert "sequence" in seq_tbl.column_names
    assert "length" in seq_tbl.column_names

    # Step 2: Build N-gram graph (tri-peptides n=3)
    meta2 = ctx.sql("PROJECT NGRAM GRAPH gold.uniref_ngram_graph FROM bronze.uniref_sequences WITH (n = 3, nodes_target = 'silver.uniref_ngram_nodes', compute_probabilities = true);")
    assert meta2.row_count > 0, "Graph should have edges"
    assert meta2.properties["num_nodes"] > 0
    assert meta2.properties["num_edges"] > 0

    edges_tbl = ctx.read_table("gold.uniref_ngram_graph")
    nodes_tbl = ctx.read_table("silver.uniref_ngram_nodes")

    assert set(["source", "target", "weight", "transition_prob"]).issubset(edges_tbl.column_names)
    assert set(["node_id", "ngram", "in_degree", "out_degree", "total_degree", "protein_count"]).issubset(nodes_tbl.column_names)

    # Step 3: Run Coarsening on N-gram graph
    meta3 = ctx.sql("COARSEN GRAPH gold.uniref_coarse_graph FROM gold.uniref_ngram_graph WITH (src_col = 'source', dst_col = 'target', weight_col = 'weight', levels = 1, method = 'graclus', mapping_table = 'silver.uniref_cluster_map');")
    assert meta3.row_count > 0
    assert meta3.properties["coarsened_nodes"] <= meta3.properties["orig_nodes"]
    assert meta3.properties["node_reduction_ratio"] >= 0.0

    # Step 4: SQL queries on resulting graphs
    top_edges = ctx.sql("SELECT source, target, weight, transition_prob FROM gold.uniref_ngram_graph ORDER BY weight DESC LIMIT 5;")
    assert len(top_edges) == 5
    for edge in top_edges:
        assert edge["weight"] >= 1
        assert 0.0 <= edge["transition_prob"] <= 1.0

    top_nodes = ctx.sql("SELECT ngram, in_degree, out_degree, total_degree, protein_count FROM silver.uniref_ngram_nodes ORDER BY total_degree DESC LIMIT 5;")
    assert len(top_nodes) == 5
    for node in top_nodes:
        assert node["total_degree"] > 0
        assert node["protein_count"] >= 1

    shutil.rmtree(test_root, ignore_errors=True)
    print("ALL UNIREF NGRAM PIPELINE CHECKS PASSED!")


def test_decoupled_ai_profiling_and_workflow():
    from examples.run_uniref_ngram_usecase import perform_ai_profiling, formulate_pipeline_plan, execute_pipeline

    test_root = "tests/test_lakehouse_decoupled"
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    ctx = LakehouseContext(lakehouse_root=test_root)
    sample_fasta = "data/bioinformatics/uniref50_sample.fasta"

    # 1. AI Profiling
    profile_report = perform_ai_profiling(sample_fasta, ctx=ctx)
    assert profile_report["domain"] == "bioinformatics"
    assert profile_report["file_format"] == "fasta"
    assert profile_report["confidence"] > 0.9
    assert profile_report["is_uniref_clusters"] is True

    # 2. Pipeline plan formulation in autonomous mode
    plan = formulate_pipeline_plan(
        profile_report=profile_report,
        interactive=False,
        cli_overrides={"limit": 40, "n": 3},
    )
    assert plan["clean_sequences"] is True
    assert plan["run_deduplication"] is True
    assert plan["tokenizer"] == "ngram_graph"
    assert plan["n"] == 3

    # 3. Execution & delivery to test_root
    manifest = execute_pipeline(
        dataset_path=sample_fasta,
        output_dir=test_root,
        plan=plan,
        profile_report=profile_report,
        ctx=ctx,
    )

    assert "stages" in manifest
    assert "bronze_ingestion" in manifest["stages"]
    assert "entity_resolution" in manifest["stages"]
    assert "graph_construction" in manifest["stages"]

    # Verify manifest on disk
    manifest_file = os.path.join(test_root, "pipeline_manifest.json")
    assert os.path.exists(manifest_file)

    # Verify matrices on disk
    mat_dir = os.path.join(test_root, "matrices")
    assert os.path.exists(mat_dir)
    assert os.path.exists(os.path.join(mat_dir, "adj_raw_sparse_n3.npz"))
    assert os.path.exists(os.path.join(mat_dir, "adj_mathcal_A_in_n3.npz"))
    assert os.path.exists(os.path.join(mat_dir, "adj_mathcal_A_out_n3.npz"))
    assert os.path.exists(os.path.join(mat_dir, "node_vocabulary_n3.json"))

    shutil.rmtree(test_root, ignore_errors=True)
    print("ALL DECOUPLED AI PROFILING & WORKFLOW CHECKS PASSED!")


if __name__ == "__main__":
    test_uniref_ngram_graph_pipeline()
    test_decoupled_ai_profiling_and_workflow()
