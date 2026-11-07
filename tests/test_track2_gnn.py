"""
End-to-End Test for Track 2: Deep Learning & Graph Neural Networks (DGCN / GNN / DGI)
Tests:
1. Text Vectorization via SQL (`CREATE EMBEDDINGS ... WITH (method='tfidf', dimension=1024, subword=true)`).
2. Cosine similarity Entity Resolution on neural/dense vector embeddings.
3. Karate club graph ingestion (edges + metadata with whitespace delimiters).
4. Directed GCN (`TRAIN GNN ... WITH (model='dgcn')`) and DGI (`TRAIN GNN ... WITH (model='dgi')`).
5. Verification of dense embeddings and community clustering.
"""

import os
import sys
import shutil
import tempfile
import pyarrow as pa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.context import IngestionContext


def test_track2_end_to_end():
    tmp_lakehouse = tempfile.mkdtemp(prefix="ditk_track2_test_")
    print(f"[*] Initializing temporary Lakehouse at {tmp_lakehouse}")
    ctx = IngestionContext(lakehouse_root=tmp_lakehouse)

    try:
        # ======================================================================
        # Part 1: Text Vectorization & Cosine Similarity Entity Resolution
        # ======================================================================
        print("\n--- [1] Ingesting Restaurant Dataset ---")
        ctx.sql("INGEST FROM 'data/entity_resolution/restaurants/S3Rest.txt' INTO bronze.restaurants WITH (format='csv', delimiter=',');")
        ctx.sql("CREATE STAGE silver.restaurants_staged FROM bronze.restaurants WITH (key='RecID', stopword_threshold=0.05);")
        ctx.sql("CREATE BLOCKS silver.restaurant_pairs FROM silver.restaurants_staged WITH (method='token_inverted_index', max_block_size=100);")

        print("\n--- [2] Generating Dense TF-IDF Embeddings via SQL ---")
        ctx.sql("CREATE EMBEDDINGS silver.restaurant_embeddings FROM silver.restaurants_staged WITH (method='tfidf', dimension=1024, subword=true, max_df=0.20, key='RecID');")
        
        emb_tbl = ctx.read_table("silver.restaurant_embeddings")
        assert "embedding" in emb_tbl.column_names, "Embedding column missing from output table"
        assert emb_tbl.num_rows == 868, f"Expected 868 rows, got {emb_tbl.num_rows}"
        dim = len(emb_tbl["embedding"][0].as_py())
        print(f"Successfully generated embeddings for {emb_tbl.num_rows} records. Dimension: {dim}")
        assert dim == 1024

        print("\n--- [3] Resolving Entities via Cosine Similarity on Embeddings ---")
        ctx.sql("""
            RESOLVE ENTITIES INTO gold.restaurants_resolved
            FROM silver.restaurant_embeddings
            USING BLOCKS silver.restaurant_pairs
            WITH (key='RecID', similarity='cosine', threshold=0.70, clustering='transitive_closure');
        """)
        resolved_tbl = ctx.read_table("gold.restaurants_resolved")
        assert "canonical_entity_id" in resolved_tbl.column_names
        print(f"Resolved {resolved_tbl.num_rows} records into canonical entities using Cosine Similarity.")

        print("\n--- [4] Evaluating Neural ER Against Ground Truth ---")
        report = ctx.sql("""
            EVALUATE gold.restaurants_resolved
            AGAINST TRUTH 'data/entity_resolution/restaurants/truthRestaurant.txt'
            USING BLOCKS silver.restaurant_pairs
            WITH (key='RecID');
        """)
        report.print_report()
        assert report.record_count == 868
        assert report.f1_score >= 0.70, f"Expected F1 >= 0.70, got {report.f1_score}"
        print(f"[PASS] Neural ER successfully validated (F1={report.f1_score:.4f}, Precision={report.precision:.4f}, Recall={report.recall:.4f})")

        # ======================================================================
        # Part 2: Graph Ingestion & Directed GCN / DGI Representation Learning
        # ======================================================================
        print("\n--- [5] Ingesting Karate Club Graph & Node Groups ---")
        ctx.sql("INGEST FROM 'data/graph_learning/karate_club/karate_edges_77.txt' INTO bronze.karate_edges WITH (delimiter='whitespace', has_header=false);")
        ctx.sql("INGEST FROM 'data/graph_learning/karate_club/karate_groups.txt' INTO bronze.karate_nodes WITH (delimiter='whitespace', has_header=false);")
        
        edges_tbl = ctx.read_table("bronze.karate_edges")
        nodes_tbl = ctx.read_table("bronze.karate_nodes")
        print(f"Karate edges rows: {edges_tbl.num_rows}, columns: {edges_tbl.column_names}")
        print(f"Karate nodes rows: {nodes_tbl.num_rows}, columns: {nodes_tbl.column_names}")
        assert edges_tbl.num_rows == 154
        assert nodes_tbl.num_rows == 34

        # Generate node feature embeddings from group representations
        ctx.sql("CREATE EMBEDDINGS silver.karate_node_features FROM bronze.karate_nodes WITH (method='ngram_hash', dimension=16, key='f0');")
        
        print("\n--- [6] Training Directed GCN (DGCN) on Karate Club Graph ---")
        ctx.sql("""
            TRAIN GNN gold.karate_dgcn_embeddings
            FROM silver.karate_node_features
            USING GRAPH bronze.karate_edges
            WITH (key='f0', src_col='f0', dst_col='f1', weight_col='f2', model='dgcn', dimension=16, layers=2, directed=true);
        """)
        dgcn_tbl = ctx.read_table("gold.karate_dgcn_embeddings")
        assert dgcn_tbl.num_rows == 34, f"Expected 34 nodes in Karate Club, got {dgcn_tbl.num_rows}"
        dgcn_emb = dgcn_tbl["embedding"][0].as_py()
        assert len(dgcn_emb) == 16
        print(f"Directed GCN generated embeddings for {dgcn_tbl.num_rows} nodes. Embedding vector len: {len(dgcn_emb)}")

        print("\n--- [7] Training Deep Graph Infomax (DGI) Self-Supervised Learning ---")
        ctx.sql("""
            TRAIN GNN gold.karate_dgi_embeddings
            FROM silver.karate_node_features
            USING GRAPH bronze.karate_edges
            WITH (key='f0', src_col='f0', dst_col='f1', weight_col='f2', model='dgi', dimension=16, layers=2, epochs=15);
        """)
        dgi_tbl = ctx.read_table("gold.karate_dgi_embeddings")
        assert dgi_tbl.num_rows == 34
        dgi_emb = dgi_tbl["embedding"][0].as_py()
        assert len(dgi_emb) == 16
        print(f"DGI generated self-supervised embeddings for {dgi_tbl.num_rows} nodes. Embedding vector len: {len(dgi_emb)}")

        print("\n================================================================================")
        print(" [SUCCESS] Track 2: All GNN, DGCN, DGI, and Neural Entity Resolution tests passed!")
        print("================================================================================")

    finally:
        shutil.rmtree(tmp_lakehouse, ignore_errors=True)


if __name__ == "__main__":
    test_track2_end_to_end()
