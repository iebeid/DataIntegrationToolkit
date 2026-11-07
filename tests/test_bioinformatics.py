import os
import shutil
import tempfile
import pytest
from toolkit.context import IngestionContext


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_bio_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_fasta_ingestion(temp_lakehouse):
    ctx, root = temp_lakehouse
    fasta_path = os.path.join(root, "sample.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens\n")
        f.write("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\n")
        f.write(">sp|P01116|RASH_HUMAN GTPase HRas OS=Homo sapiens\n")
        f.write("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG\n")

    meta = ctx.ingest_fasta(
        source_uri=fasta_path,
        target_name="bronze.human_proteins",
    )
    assert meta.row_count == 2
    table = ctx.read_table("bronze.human_proteins")
    assert "sequence_id" in table.column_names
    assert "sequence" in table.column_names
    assert "length" in table.column_names
    assert table["length"][0].as_py() == 60


def test_id_mapping_and_ppi_coarsening(temp_lakehouse):
    ctx, root = temp_lakehouse

    # 1. ID Mapping
    map_tsv = os.path.join(root, "idmap.tsv")
    with open(map_tsv, "w", encoding="utf-8") as f:
        f.write("P04637\tGene_Name\tTP53\n")
        f.write("P04637\tBioGRID\t112233\n")
        f.write("P01116\tGene_Name\tHRAS\n")
        f.write("P01116\tBioGRID\t102030\n")

    ctx.sql(f"INGEST IDMAPPING silver.uniprot_map FROM '{map_tsv}' WITH (id_types = 'Gene_Name,BioGRID');")
    tbl_map = ctx.read_table("silver.uniprot_map")
    assert tbl_map.num_rows == 4

    # 2. Map Identifiers
    genes_csv = os.path.join(root, "genes.csv")
    with open(genes_csv, "w", encoding="utf-8") as f:
        f.write("gene_symbol\nTP53\nHRAS\n")

    ctx.sql(f"INGEST FROM '{genes_csv}' INTO bronze.my_genes;")
    ctx.sql("MAP IDENTIFIERS silver.mapped_genes FROM bronze.my_genes USING MAPPING silver.uniprot_map WITH (src_col = 'gene_symbol', target_col = 'uniprot_ac', id_type = 'Gene_Name');")
    mapped = ctx.read_table("silver.mapped_genes")
    assert mapped.num_rows == 2
    assert "uniprot_ac" in mapped.column_names

    # 3. Graph Coarsening
    graph_csv = os.path.join(root, "graph.csv")
    with open(graph_csv, "w", encoding="utf-8") as f:
        f.write("src,dst,weight\n")
        f.write("n1,n2,5.0\n")
        f.write("n2,n3,4.0\n")
        f.write("n3,n4,3.0\n")
        f.write("n4,n5,6.0\n")

    ctx.sql(f"INGEST FROM '{graph_csv}' INTO bronze.raw_graph;")
    ctx.sql("COARSEN GRAPH silver.coarse_graph FROM bronze.raw_graph WITH (src_col = 'src', dst_col = 'dst', weight_col = 'weight', ratio = 0.5);")
    coarse_tbl = ctx.read_table("silver.coarse_graph")
    assert coarse_tbl.num_rows > 0

    # 4. STRING PPI Ingest
    string_tsv = os.path.join(root, "string.tsv")
    with open(string_tsv, "w", encoding="utf-8") as f:
        f.write("protein1 protein2 combined_score\n")
        f.write("P04637 P01116 950\n")
        f.write("P04637 P12345 200\n")

    ctx.sql(f"INGEST PPI gold.string_ppi FROM '{string_tsv}' WITH (source = 'string', min_score = 400);")
    ppi_tbl = ctx.read_table("gold.string_ppi")
    assert ppi_tbl.num_rows == 1
