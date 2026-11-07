import os
import sys
import shutil
import json
import pyarrow as pa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.context import IngestionContext
from toolkit.catalog.table import TableMetadata


def setup_test_data(ctx: IngestionContext):
    # 1. Literature Articles table
    articles_data = [
        (
            "1001",
            "Targeting TP53 and Sorafenib Resistance in Hepatocellular Carcinoma",
            "Hepatocellular carcinoma (HCC) is often complicated by liver cirrhosis. Here we evaluate Sorafenib and Carmustine combination therapy in patients with TP53 and HRAS mutations.",
            "2021",
        ),
        (
            "1002",
            "Efficacy of Lenvatinib and Regorafenib in Advanced Carcinoma",
            "Patients with refractory adenocarcinoma were treated with Lenvatinib. Downregulation of PRKCA and BRCA1 was observed following tumor regression.",
            "2022",
        ),
        (
            "1003",
            "Immune Checkpoint Inhibition and Cisplatin in Leukemia",
            "We investigated PD-L1 expression and Cisplatin sensitivity in acute leukemia. Co-targeting CD274 and Doxorubicin induced significant tumour necrosis.",
            "2023",
        ),
    ]

    schema_art = pa.schema([
        ("pmid", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("year", pa.string()),
    ])
    tbl_art = pa.Table.from_arrays(
        [
            pa.array([x[0] for x in articles_data]),
            pa.array([x[1] for x in articles_data]),
            pa.array([x[2] for x in articles_data]),
            pa.array([x[3] for x in articles_data]),
        ],
        schema=schema_art,
    )
    art_path = os.path.join(ctx.lakehouse_root, "silver", "pubmed_articles.parquet").replace("\\", "/")
    ctx.storage.write_tabular(tbl_art, art_path, file_format="parquet")
    from toolkit.catalog.table import TableMetadata
    ctx.catalog.register_table(TableMetadata(name="pubmed_articles", tier="silver", uri=art_path, schema=schema_art, row_count=len(tbl_art)))

    # 2. Authors table
    authors_data = [
        ("1001", "Wang_J"),
        ("1001", "Chen_L"),
        ("1002", "Wang_J"),
        ("1002", "Smith_A"),
        ("1003", "Chen_L"),
        ("1003", "Taylor_R"),
    ]
    tbl_auth = pa.Table.from_arrays(
        [
            pa.array([x[0] for x in authors_data]),
            pa.array([x[1] for x in authors_data]),
        ],
        schema=pa.schema([("pmid", pa.string()), ("author_id", pa.string())]),
    )
    auth_path = os.path.join(ctx.lakehouse_root, "silver", "pubmed_authors.parquet").replace("\\", "/")
    ctx.storage.write_tabular(tbl_auth, auth_path, file_format="parquet")
    ctx.catalog.register_table(TableMetadata(name="pubmed_authors", tier="silver", uri=auth_path, schema=tbl_auth.schema, row_count=len(tbl_auth)))

    # 3. Citations table
    citations_data = [
        ("1002", "1001"),
        ("1003", "1001"),
    ]
    tbl_cit = pa.Table.from_arrays(
        [
            pa.array([x[0] for x in citations_data]),
            pa.array([x[1] for x in citations_data]),
        ],
        schema=pa.schema([("citing_pmid", pa.string()), ("cited_pmid", pa.string())]),
    )
    cit_path = os.path.join(ctx.lakehouse_root, "silver", "pubmed_citations.parquet").replace("\\", "/")
    ctx.storage.write_tabular(tbl_cit, cit_path, file_format="parquet")
    ctx.catalog.register_table(TableMetadata(name="pubmed_citations", tier="silver", uri=cit_path, schema=tbl_cit.schema, row_count=len(tbl_cit)))


def test_track4_nlp_pipeline():
    test_root = os.path.join(os.path.dirname(__file__), "lakehouse_test_track4")
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    ctx = IngestionContext(lakehouse_root=test_root)
    setup_test_data(ctx)

    print("=" * 70)
    print("STEP 1: Testing EXTRACT ENTITIES (Gazetteer & Vocab NER)")
    print("=" * 70)

    # 1.1 Extract entities using SQL syntax with local biomedical vocabularies
    sql_extract = """
    EXTRACT ENTITIES silver.extracted_entities
    FROM silver.pubmed_articles
    WITH (
        text_col = 'abstract',
        key_col = 'pmid',
        vocab_path = 'data/biomedical_nlp/vocab_entities'
    )
    """
    meta_extract = ctx.sql(sql_extract)
    print(f"Extracted Entities Table: {meta_extract.name} in tier {meta_extract.tier}")
    print(f"Row count: {meta_extract.row_count}")

    tbl_extracted = ctx.read_table("silver.extracted_entities")
    df_extracted = tbl_extracted.to_pydict()
    print("Sample Extracted Entities:")
    for i in range(min(10, tbl_extracted.num_rows)):
        print(f"  PMID {df_extracted['doc_id'][i]}: [{df_extracted['entity_type'][i]}] {df_extracted['mention'][i]} (ID: {df_extracted['entity_id'][i]})")

    assert tbl_extracted.num_rows > 0, "Expected entities extracted from abstracts"
    extracted_types = set(df_extracted["entity_type"])
    assert "disease" in extracted_types, "Expected disease entities extracted"
    assert "drug" in extracted_types, "Expected drug entities extracted"
    assert "gene" in extracted_types, "Expected gene entities extracted"

    # Verify specific key entities were captured
    mentions_lower = [m.lower() for m in df_extracted["mention"]]
    assert "sorafenib" in mentions_lower, "Expected Sorafenib extracted"
    assert "tp53" in mentions_lower, "Expected TP53 extracted"
    assert any("carcinoma" in m or "cirrhosis" in m for m in mentions_lower), "Expected cancer/cirrhosis extracted"

    print("\n" + "=" * 70)
    print("STEP 2: Testing MAP ONTOLOGY (Chem2Bio2RDF URIs)")
    print("=" * 70)

    # 2. Map extracted entities to canonical Chem2Bio2RDF URIs
    sql_map = """
    MAP ONTOLOGY silver.mapped_entities
    FROM silver.extracted_entities
    WITH (
        ontology = 'chem2bio2rdf'
    )
    """
    meta_map = ctx.sql(sql_map)
    print(f"Mapped Entities Table: {meta_map.name}, rows: {meta_map.row_count}")

    tbl_mapped = ctx.read_table("silver.mapped_entities")
    df_mapped = tbl_mapped.to_pydict()
    print("Sample Mapped URIs:")
    for i in range(min(10, tbl_mapped.num_rows)):
        print(f"  [{df_mapped['entity_type'][i]}] {df_mapped['mention'][i]} -> {df_mapped['canonical_uri'][i]}")

    assert tbl_mapped.num_rows > 0, "Expected mapped entity URIs"
    uris = df_mapped["canonical_uri"]
    assert any("pubchem_compound" in u for u in uris), "Expected PubChem compound URIs"
    assert any("uniprot/resource/gene" in u for u in uris), "Expected UniProt gene URIs"
    assert any("omim/resource/disease" in u for u in uris), "Expected OMIM disease URIs"

    print("\n" + "=" * 70)
    print("STEP 3: Testing CONSTRUCT KNOWLEDGE GRAPH (Heterogeneous Multi-Relational Triples)")
    print("=" * 70)

    # 3. Construct knowledge graph
    sql_kg = """
    CONSTRUCT KNOWLEDGE GRAPH gold.biomedical_kg
    FROM silver.pubmed_articles
    WITH (
        entities = 'silver.extracted_entities',
        authors = 'silver.pubmed_authors',
        citations = 'silver.pubmed_citations',
        ontology = 'silver.mapped_entities'
    )
    """
    meta_kg = ctx.sql(sql_kg)
    print(f"Knowledge Graph Table: {meta_kg.name}, total triples: {meta_kg.row_count}")

    tbl_kg = ctx.read_table("gold.biomedical_kg")
    df_kg = tbl_kg.to_pydict()
    predicates = set(df_kg["predicate"])
    print(f"Unique Relation Types in KG: {predicates}")

    assert "mentionedIn" in predicates, "Expected mentionedIn relations"
    assert "wrote" in predicates, "Expected wrote relations"
    assert "cites" in predicates, "Expected cites relations"
    assert "sameAs" in predicates, "Expected sameAs canonical ontology relations"

    # Display sample triples
    print("Sample Triples:")
    for i in range(min(12, tbl_kg.num_rows)):
        print(f"  ({df_kg['subject'][i]}, {df_kg['predicate'][i]}, {df_kg['object'][i]}) [{df_kg['subject_type'][i]} -> {df_kg['object_type'][i]}]")

    print("\n" + "=" * 70)
    print("STEP 4: Testing PROFILE GRAPH (Author Centrality & Entity Profiler)")
    print("=" * 70)

    # 4. Profile authors from the knowledge graph
    sql_profile = """
    PROFILE GRAPH gold.author_profiles
    FROM gold.biomedical_kg
    WITH (
        target_type = 'author',
        top_n = 10
    )
    """
    meta_profile = ctx.sql(sql_profile)
    print(f"Author Profile Table: {meta_profile.name}, ranked authors: {meta_profile.row_count}")

    tbl_profile = ctx.read_table("gold.author_profiles")
    df_profile = tbl_profile.to_pydict()
    print("Top Author Rankings & Centrality:")
    for i in range(tbl_profile.num_rows):
        assoc = json.loads(df_profile["top_associations"][i])
        print(f"  Rank {i+1}: {df_profile['node_id'][i]} | Centrality: {df_profile['degree_centrality'][i]:.4f} | InDeg: {df_profile['in_degree'][i]} | OutDeg: {df_profile['out_degree'][i]}")
        print(f"    Associated Diseases: {assoc.get('diseases', [])[:3]}")
        print(f"    Associated Genes: {assoc.get('genes', [])[:3]}")
        print(f"    Associated Drugs: {assoc.get('drugs', [])[:3]}")

    assert tbl_profile.num_rows > 0, "Expected ranked authors"
    assert df_profile["degree_centrality"][0] >= df_profile["degree_centrality"][-1], "Expected descending centrality ordering"

    print("\n" + "=" * 70)
    print("STEP 5: Testing Pre-computed Annotations Branch (BERN/BioBERT)")
    print("=" * 70)

    # Create dummy pre-computed BERN annotation table
    ann_data = [
        ("2001", "disease/106939701", "disease", "hepatocellular carcinoma"),
        ("2001", "gene/7157", "gene", "TP53"),
        ("2002", "drug/216239", "drug", "Sorafenib"),
    ]
    tbl_ann = pa.Table.from_arrays(
        [
            pa.array([x[0] for x in ann_data]),
            pa.array([x[1] for x in ann_data]),
            pa.array([x[2] for x in ann_data]),
            pa.array([x[3] for x in ann_data]),
        ],
        schema=pa.schema([
            ("PMID", pa.string()),
            ("EntityID", pa.string()),
            ("Type", pa.string()),
            ("Mention", pa.string()),
        ]),
    )
    ann_path = os.path.join(test_root, "silver", "bern_annotations.parquet").replace("\\", "/")
    ctx.storage.write_tabular(tbl_ann, ann_path, file_format="parquet")
    ctx.catalog.register_table(TableMetadata(name="bern_annotations", tier="silver", uri=ann_path, schema=tbl_ann.schema, row_count=len(tbl_ann)))

    sql_extract_ann = """
    EXTRACT ENTITIES silver.bern_extracted
    FROM silver.pubmed_articles
    USING ANNOTATIONS silver.bern_annotations
    """
    meta_bern = ctx.sql(sql_extract_ann)
    tbl_bern = ctx.read_table("silver.bern_extracted")
    assert tbl_bern.num_rows == 3, f"Expected 3 rows from BERN annotations, got {tbl_bern.num_rows}"
    print(f"Pre-computed Annotation Extraction Successful: {tbl_bern.num_rows} annotations registered.")

    # Cleanup
    shutil.rmtree(test_root)
    print("\n" + "=" * 70)
    print("ALL TRACK 4 BIOMEDICAL NLP & KNOWLEDGE GRAPH TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_track4_nlp_pipeline()
