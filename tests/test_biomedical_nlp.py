import os
import shutil
import tempfile
import json
import pytest
from src.context import IngestionContext


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_lakehouse_nlp_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_biomedical_nlp_pipeline(temp_lakehouse):
    ctx, root = temp_lakehouse

    # 1. Gazetteer dictionary
    vocab_dir = os.path.join(root, "vocab_entities")
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "Chemical.tsv"), "w", encoding="utf-8") as f:
        f.write("Aspirin\t1\nIbuprofen\t2\n")
    with open(os.path.join(vocab_dir, "Gene.tsv"), "w", encoding="utf-8") as f:
        f.write("TP53\t10\nBRCA1\t20\n")
    with open(os.path.join(vocab_dir, "Disease.tsv"), "w", encoding="utf-8") as f:
        f.write("Melanoma\t100\nCarcinoma\t200\n")

    # 2. Sample articles
    abstracts_csv = os.path.join(root, "articles.csv")
    with open(abstracts_csv, "w", encoding="utf-8") as f:
        f.write("pmid,title,abstract,year\n")
        f.write("1001,Title 1,Aspirin targets TP53 in Melanoma patients.,2021\n")
        f.write("1002,Title 2,Ibuprofen and BRCA1 regulation in Carcinoma.,2022\n")

    # Authors
    authors_csv = os.path.join(root, "authors.csv")
    with open(authors_csv, "w", encoding="utf-8") as f:
        f.write("pmid,author_id\n")
        f.write("1001,Dr_Smith\n")
        f.write("1002,Dr_Smith\n")
        f.write("1002,Dr_Lee\n")

    # Citations
    citations_csv = os.path.join(root, "citations.csv")
    with open(citations_csv, "w", encoding="utf-8") as f:
        f.write("citing_pmid,cited_pmid\n")
        f.write("1002,1001\n")

    ctx.ingest(source_uri=abstracts_csv, target_name="silver.pubmed_articles", file_format="csv")
    ctx.ingest(source_uri=authors_csv, target_name="silver.pubmed_authors", file_format="csv")
    ctx.ingest(source_uri=citations_csv, target_name="silver.pubmed_citations", file_format="csv")

    # 3. Extract Entities via SQL
    ctx.sql(f"""
    EXTRACT ENTITIES silver.extracted_entities
    FROM silver.pubmed_articles
    WITH (
        text_col = 'abstract',
        key_col = 'pmid',
        vocab_path = '{vocab_dir}'
    )
    """)
    ent_tbl = ctx.read_table("silver.extracted_entities")
    assert ent_tbl.num_rows > 0
    assert "mention" in ent_tbl.column_names
    assert "entity_type" in ent_tbl.column_names

    # 4. Map Ontology via SQL
    ctx.sql("""
    MAP ONTOLOGY silver.mapped_entities
    FROM silver.extracted_entities
    WITH (
        ontology = 'chem2bio2rdf'
    )
    """)
    mapped_tbl = ctx.read_table("silver.mapped_entities")
    assert "canonical_uri" in mapped_tbl.column_names

    # 5. Construct Knowledge Graph via SQL
    ctx.sql("""
    CONSTRUCT KNOWLEDGE GRAPH gold.biomedical_kg
    FROM silver.pubmed_articles
    WITH (
        entities = 'silver.extracted_entities',
        authors = 'silver.pubmed_authors',
        citations = 'silver.pubmed_citations',
        ontology = 'silver.mapped_entities'
    )
    """)
    kg_tbl = ctx.read_table("gold.biomedical_kg")
    assert "subject" in kg_tbl.column_names
    assert "predicate" in kg_tbl.column_names
    assert "object" in kg_tbl.column_names
    preds = set(kg_tbl["predicate"].to_pylist())
    assert "mentionedIn" in preds
    assert "wrote" in preds
    assert "cites" in preds
    assert "sameAs" in preds

    # 6. Profile Author via SQL
    ctx.sql("""
    PROFILE GRAPH gold.author_profiles
    FROM gold.biomedical_kg
    WITH (target_type = 'author', top_n = 10)
    """)
    prof_tbl = ctx.read_table("gold.author_profiles")
    assert "node_id" in prof_tbl.column_names
    assert "degree_centrality" in prof_tbl.column_names
    assert "top_associations" in prof_tbl.column_names

    p0 = json.loads(prof_tbl["top_associations"][0].as_py())
    assert "diseases" in p0 and "genes" in p0 and "drugs" in p0
