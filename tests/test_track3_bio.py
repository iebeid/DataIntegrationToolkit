import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.context import IngestionContext


def setup_test_environment(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(f"{base_dir}/data", exist_ok=True)
    ctx = IngestionContext(lakehouse_root=f"{base_dir}/lakehouse")
    return ctx, base_dir


def test_track3_bio_pipeline():
    ctx, base_dir = setup_test_environment()
    data_dir = f"{base_dir}/data"

    print("\n" + "=" * 80)
    print("TRACK 3 BIOINFORMATICS INTEGRATION TEST: FASTA, ID MAPPING, COARSENING, PPI")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. FASTA INGESTION & SAMPLING (Task 3.1)
    # --------------------------------------------------------------------------
    fasta_path = f"{data_dir}/sample_proteins.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(""">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
>sp|P01116|RASH_HUMAN GTPase HRas OS=Homo sapiens OX=9606 GN=HRAS PE=1 SV=1
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG
QEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDL
>tr|A0A024R1R8|A0A024R1R8_HUMAN Protein kinase C alpha OS=Homo sapiens
MADVYPANDSTASQDVANRFARKGALRQKNVHEVKDHKFIARFFKQPTFCSHCTDFIWGF
GKQGFQCQVCCFVVHKRCHEFVTFSCPGADKGPDTDDPRSKHKFKIHTYGSPTFCDHCGS
>UniRef50_P12345 Glucose-6-phosphate isomerase OS=Homo sapiens
MAALTRDPQFQKLQQWYREHRSELNLRRLFDANKDRFNHFSLTLNTNHGHILVDYSKNLV
""")

    print("\n[Step 1.1] Ingesting FASTA into Bronze tier via SQL...")
    res = ctx.sql(f"INGEST FASTA bronze.proteins FROM '{fasta_path}';")
    assert res.row_count == 4, f"Expected 4 sequences, got {res.row_count}"

    tbl = ctx.read_table("bronze.proteins")
    ids = tbl["sequence_id"].to_pylist()
    assert "P04637" in ids, "P04637 missing from extracted IDs"
    assert "P01116" in ids, "P01116 missing from extracted IDs"
    assert "A0A024R1R8" in ids, "A0A024R1R8 missing from extracted IDs"
    assert "P12345" in ids, "P12345 missing from extracted IDs"
    lengths = tbl["length"].to_pylist()
    assert all(l > 0 for l in lengths)
    print(f"  Successfully ingested 4 protein sequences. IDs: {ids}")

    # Reservoir sampling test
    print("\n[Step 1.2] Testing Reservoir Sampling (sample=2)...")
    res_sample = ctx.sql(f"INGEST FASTA bronze.proteins_sample FROM '{fasta_path}' WITH (sample = 2, seed = 42);")
    assert res_sample.row_count == 2, f"Expected 2 sampled sequences, got {res_sample.row_count}"
    print("  Reservoir sampling successfully yielded exactly 2 sequences.")

    # Target ID filtering test
    print("\n[Step 1.3] Testing Target ID filtering (P04637, P01116)...")
    res_filt = ctx.sql(f"INGEST FASTA bronze.proteins_target FROM '{fasta_path}' WITH (target_ids = 'P04637,P01116');")
    assert res_filt.row_count == 2
    filtered_ids = ctx.read_table("bronze.proteins_target")["sequence_id"].to_pylist()
    assert set(filtered_ids) == {"P04637", "P01116"}
    print(f"  Target filtering successfully loaded target sequences: {filtered_ids}")

    # --------------------------------------------------------------------------
    # 2. DISTRIBUTED ID MAPPER (Task 3.2)
    # --------------------------------------------------------------------------
    idmap_path = f"{data_dir}/sample_idmapping.tsv"
    with open(idmap_path, "w", encoding="utf-8") as f:
        f.write("P04637\tGene_Name\tTP53\n")
        f.write("P04637\tBioGRID\t112233\n")
        f.write("P04637\tSTRING\t9606.ENSP00000269305\n")
        f.write("P01116\tGene_Name\tHRAS\n")
        f.write("P01116\tBioGRID\t102030\n")
        f.write("A0A024R1R8\tGene_Name\tPRKCA\n")
        f.write("A0A024R1R8\tBioGRID\t115599\n")

    print("\n[Step 2.1] Ingesting ID Mapping Table via SQL...")
    res_map = ctx.sql(f"INGEST IDMAPPING silver.uniprot_map FROM '{idmap_path}' WITH (id_types = 'Gene_Name,BioGRID,STRING');")
    assert res_map.row_count == 7, f"Expected 7 mapping rows, got {res_map.row_count}"
    print(f"  Successfully ingested {res_map.row_count} ID mapping tuples.")

    # Create external gene list to map
    genes_path = f"{data_dir}/my_genes.csv"
    with open(genes_path, "w", encoding="utf-8") as f:
        f.write("gene_symbol,description\n")
        f.write("TP53,Tumor Protein P53\n")
        f.write("HRAS,Harvey Rat Sarcoma Viral Oncogene Homolog\n")
        f.write("PRKCA,Protein Kinase C Alpha\n")

    ctx.sql(f"INGEST FROM '{genes_path}' INTO bronze.my_genes;")

    print("\n[Step 2.2] Translating gene symbols to UniProt ACs using MAP IDENTIFIERS...")
    res_trans = ctx.sql("MAP IDENTIFIERS silver.mapped_genes FROM bronze.my_genes USING MAPPING silver.uniprot_map WITH (src_col = 'gene_symbol', target_col = 'uniprot_ac', id_type = 'Gene_Name');")
    mapped_tbl = ctx.read_table("silver.mapped_genes")
    assert mapped_tbl.num_rows == 3
    assert "uniprot_ac" in mapped_tbl.column_names

    gene_to_ac = dict(zip(mapped_tbl["gene_symbol"].to_pylist(), mapped_tbl["uniprot_ac"].to_pylist()))
    assert gene_to_ac["TP53"] == "P04637"
    assert gene_to_ac["HRAS"] == "P01116"
    assert gene_to_ac["PRKCA"] == "A0A024R1R8"
    print(f"  Successfully translated identifiers: {gene_to_ac}")

    # --------------------------------------------------------------------------
    # 3. GRAPH COARSENING OPERATOR (Task 3.3)
    # --------------------------------------------------------------------------
    graph_path = f"{data_dir}/sample_graph.csv"
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write("src,dst,weight\n")
        f.write("N1,N2,5.0\n")
        f.write("N2,N3,4.0\n")
        f.write("N1,N3,3.0\n")
        f.write("N4,N5,6.0\n")
        f.write("N5,N6,4.0\n")
        f.write("N4,N6,5.0\n")
        f.write("N3,N4,1.0\n")  # Bridge between {N1,N2,N3} and {N4,N5,N6}

    ctx.sql(f"INGEST FROM '{graph_path}' INTO bronze.sample_graph;")

    print("\n[Step 3.1] Running Graclus Graph Coarsening Operator...")
    coarse_res = ctx.sql("COARSEN GRAPH silver.coarse_graph FROM bronze.sample_graph WITH (src_col = 'src', dst_col = 'dst', levels = 1, method = 'graclus', mapping_table = 'silver.cluster_map');")

    props = coarse_res.properties
    orig_nodes = props["orig_nodes"]
    coarsened_nodes = props["coarsened_nodes"]
    node_rr = props["node_reduction_ratio"]
    print(f"  Coarsening Results: {orig_nodes} nodes -> {coarsened_nodes} nodes (Reduction Ratio: {node_rr:.2%})")
    assert orig_nodes == 6
    assert coarsened_nodes < orig_nodes, "Coarsened graph should have fewer nodes than original"
    assert node_rr > 0.0

    map_tbl = ctx.read_table("silver.cluster_map")
    assert map_tbl.num_rows == 6
    print(f"  Cluster mapping table created with {map_tbl.num_rows} node-to-cluster mappings.")

    # --------------------------------------------------------------------------
    # 4. PPI GROUND TRUTH INGESTION (Task 3.4)
    # --------------------------------------------------------------------------
    biogrid_path = f"{data_dir}/sample_biogrid.tab3.txt"
    with open(biogrid_path, "w", encoding="utf-8") as f:
        # Minimal BioGRID format line with official symbols & experimental system
        f.write("#BioGRID ID Interactor A\tBioGRID ID Interactor B\tEntrez A\tEntrez B\tSystematic A\tSystematic B\tOfficial Symbol Interactor A\tOfficial Symbol Interactor B\tSynonyms Interactor A\tSynonyms Interactor B\tExperimental System\tExperimental System Type\n")
        f.write("1\t2\t100\t200\t-\t-\tTP53\tHRAS\t-\t-\tTwo-hybrid\tphysical\n")
        f.write("1\t3\t100\t300\t-\t-\tTP53\tPRKCA\t-\t-\tAffinity Capture-MS\tphysical\n")
        f.write("2\t3\t200\t300\t-\t-\tHRAS\tPRKCA\t-\t-\tSynthetic Lethality\tgenetic\n")

    print("\n[Step 4.1] Ingesting BioGRID TAB3 with UniProt Mapping...")
    ppi_res = ctx.sql(f"INGEST PPI gold.ppi_physical FROM '{biogrid_path}' USING MAPPING silver.uniprot_map WITH (source = 'biogrid', system = 'physical');")
    ppi_tbl = ctx.read_table("gold.ppi_physical")

    assert ppi_tbl.num_rows == 2, f"Expected 2 physical interactions, got {ppi_tbl.num_rows}"
    interactors_a = ppi_tbl["interactor_a"].to_pylist()
    interactors_b = ppi_tbl["interactor_b"].to_pylist()

    # Check mapped UniProt pairs
    pairs = set(zip(interactors_a, interactors_b))
    expected_pairs = {("P01116", "P04637"), ("A0A024R1R8", "P04637")}
    assert pairs == expected_pairs, f"Expected pairs {expected_pairs}, got {pairs}"
    print(f"  Successfully ingested & mapped BioGRID interactions to UniProt: {pairs}")

    # STRING DB Ingestion
    string_path = f"{data_dir}/sample_string.txt"
    with open(string_path, "w", encoding="utf-8") as f:
        f.write("protein1 protein2 combined_score\n")
        f.write("9606.ENSP00000269305 9606.ENSP00000304636 950\n") # high confidence
        f.write("9606.ENSP00000269305 9606.ENSP00000222222 200\n") # low confidence (< 400)
        f.write("9606.ENSP00000304636 9606.ENSP00000333333 750\n") # high confidence

    print("\n[Step 4.2] Ingesting STRING network with score filtering...")
    str_res = ctx.sql(f"INGEST PPI gold.string_interactions FROM '{string_path}' WITH (source = 'string', min_score = 400);")
    str_tbl = ctx.read_table("gold.string_interactions")
    assert str_tbl.num_rows == 2, f"Expected 2 interactions with score >= 400, got {str_tbl.num_rows}"
    print(f"  Successfully ingested STRING network with {str_tbl.num_rows} high-confidence interactions.")

    # Real dataset test on data/bioinformatics if present
    real_sim_path = "data/bioinformatics/protein_sequence_similarity.txt"
    if os.path.exists(real_sim_path):
        print("\n[Step 5.1] Testing real protein sequence similarity dataset ingestion...")
        # Create a sample slice for rapid verification
        sample_slice_path = f"{data_dir}/protein_sim_slice.txt"
        with open(real_sim_path, "r", encoding="utf-8") as rf, open(sample_slice_path, "w", encoding="utf-8") as wf:
            for _ in range(1000):
                line = rf.readline()
                if not line:
                    break
                wf.write(line)

        ctx.sql(f"INGEST FROM '{sample_slice_path}' INTO bronze.protein_sim WITH (delimiter = '\\t', has_header = false);")
        coarse_real = ctx.sql("COARSEN GRAPH silver.coarse_protein_sim FROM bronze.protein_sim WITH (src_col = 'f0', dst_col = 'f1', levels = 1, method = 'graclus');")
        print(f"  Real protein similarity graph coarsening: {coarse_real.properties['orig_nodes']} nodes -> {coarse_real.properties['coarsened_nodes']} nodes (Reduction: {coarse_real.properties['node_reduction_ratio']:.2%})")

    print("\n" + "=" * 80)
    print("ALL TRACK 3 TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_track3_bio_pipeline()
