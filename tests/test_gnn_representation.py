import os
import shutil
import tempfile
import pytest
from toolkit.context import IngestionContext


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_gnn_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_tfidf_embeddings(temp_lakehouse):
    ctx, root = temp_lakehouse
    csv_path = os.path.join(root, "docs.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("RecID,text\n")
        f.write("d1,Deep learning for molecular graph structures\n")
        f.write("d2,Graph neural networks and deep representation learning\n")
        f.write("d3,Astronomical telescope observation data\n")

    ctx.ingest(source_uri=csv_path, target_name="bronze.docs", file_format="csv")
    emb_meta = ctx.create_embeddings(
        source_table="bronze.docs",
        target_name="silver.doc_embeddings",
        key_column="RecID",
        method="tfidf",
        dimension=32,
    )
    assert emb_meta.tier == "silver"
    table = ctx.read_table("silver.doc_embeddings")
    assert "embedding" in table.column_names
    assert table.num_rows == 3


def test_dgcn_training(temp_lakehouse):
    ctx, root = temp_lakehouse
    nodes_csv = os.path.join(root, "nodes.csv")
    edges_csv = os.path.join(root, "edges.csv")

    with open(nodes_csv, "w", encoding="utf-8") as f:
        f.write("node_id,text\n")
        f.write("n1,Protein kinase A alpha catalytic subunit\n")
        f.write("n2,Protein kinase inhibitor regulatory domain\n")
        f.write("n3,Tumor suppressor protein p53\n")
        f.write("n4,Cellular tumor antigen p53 mutant\n")

    with open(edges_csv, "w", encoding="utf-8") as f:
        f.write("src,dst,weight\n")
        f.write("n1,n2,1.0\n")
        f.write("n2,n1,0.8\n")
        f.write("n3,n4,1.0\n")
        f.write("n4,n3,1.0\n")

    ctx.ingest(source_uri=nodes_csv, target_name="bronze.nodes", file_format="csv")
    ctx.ingest(source_uri=edges_csv, target_name="bronze.graph_edges", file_format="csv")

    ctx.create_embeddings(
        source_table="bronze.nodes",
        target_name="silver.node_feats",
        key_column="node_id",
        method="tfidf",
        dimension=16,
    )

    gnn_meta = ctx.train_gnn(
        source_table="silver.node_feats",
        graph_table="bronze.graph_edges",
        target_name="silver.gnn_embeddings",
        key_column="node_id",
        src_col="src",
        dst_col="dst",
        weight_col="weight",
        model="dgcn",
        dimension=16,
        layers=2,
        epochs=10,
        directed=True,
    )
    assert gnn_meta.tier == "silver"
    gnn_tbl = ctx.read_table("silver.gnn_embeddings")
    assert "embedding" in gnn_tbl.column_names
    assert gnn_tbl.num_rows == 4
    emb0 = gnn_tbl["embedding"][0].as_py()
    assert len(emb0) == 16


def test_dgi_self_supervised(temp_lakehouse):
    ctx, root = temp_lakehouse
    nodes_csv = os.path.join(root, "nodes.csv")
    edges_csv = os.path.join(root, "edges.csv")

    with open(nodes_csv, "w", encoding="utf-8") as f:
        f.write("node_id,text\n")
        f.write("a,Alpha entity\n")
        f.write("b,Beta entity\n")
        f.write("c,Gamma entity\n")

    with open(edges_csv, "w", encoding="utf-8") as f:
        f.write("src,dst,weight\n")
        f.write("a,b,1.0\n")
        f.write("b,c,1.0\n")
        f.write("c,a,1.0\n")

    ctx.ingest(source_uri=nodes_csv, target_name="bronze.dgi_nodes", file_format="csv")
    ctx.ingest(source_uri=edges_csv, target_name="bronze.dgi_edges", file_format="csv")

    ctx.create_embeddings(
        source_table="bronze.dgi_nodes",
        target_name="silver.dgi_feats",
        key_column="node_id",
        method="tfidf",
        dimension=8,
    )

    dgi_meta = ctx.train_gnn(
        source_table="silver.dgi_feats",
        graph_table="bronze.dgi_edges",
        target_name="silver.dgi_embeddings",
        key_column="node_id",
        src_col="src",
        dst_col="dst",
        weight_col="weight",
        model="dgi",
        dimension=8,
        epochs=5,
    )
    tbl = ctx.read_table("silver.dgi_embeddings")
    assert tbl.num_rows == 3
    assert len(tbl["embedding"][0].as_py()) == 8
