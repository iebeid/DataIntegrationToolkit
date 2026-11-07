
"""
Turnkey PubMed Lakehouse Pipeline Use Case.
Demonstrates end-to-end execution of:
  0. AI Entry Point: Zero-assumption dataset scanning & AI/Gemini profiling
  1. Bronze: Autonomous AI routing & ingestion of PubMed citations & authors into Bronze Parquet
  2. Silver: Named Entity Recognition (NER) for diseases, genes, and drugs via vocabulary gazetteers
  3. Silver: Semantic ontology mapping to Chem2Bio2RDF URIs
  4. Gold: Heterogeneous Knowledge Graph construction (RDF Triples: wrote, mentionedIn, sameAs)
  5. Gold: Graph centrality profiling and author disease/drug association mining
  6. SQL Analytics: Querying author profiles and entity relationships via Lakehouse SQL engine
"""

import os
import sys
import time
import argparse

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import xml.etree.ElementTree as ET
import pyarrow as pa
from toolkit.context import LakehouseContext, IngestionContext
from toolkit.catalog.table import TableMetadata


def resolve_path(p: str) -> str:
    """Resolves relative paths against repository root if not found in current working directory."""
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    repo_p = os.path.join(REPO_ROOT, p)
    if os.path.exists(repo_p):
        return repo_p
    return repo_p


def run_pubmed_pipeline(
    xml_path: str = "data/biomedical_nlp/pubmed/pubmed23n0001.xml",
    vocab_path: str = "data/biomedical_nlp/vocab_entities",
    lakehouse_root: str = "lakehouse_pubmed",
    max_articles: int = 1500,
    use_ai: bool = True,
    use_ai_sql: bool = False,
    api_key: str = None,
):
    xml_path = resolve_path(xml_path)
    vocab_path = resolve_path(vocab_path)
    lakehouse_root = resolve_path(lakehouse_root)

    print("=" * 80)
    print("  PUBMED BIOMEDICAL KNOWLEDGE GRAPH LAKEHOUSE PIPELINE")
    print("=" * 80)
    print(f"[*] PubMed Source XML : {xml_path}")
    print(f"[*] Entity Gazetteers : {vocab_path}")
    print(f"[*] Lakehouse Storage : {lakehouse_root}")
    print(f"[*] Article Limit     : {max_articles}")
    print(f"[*] AI Profiling/Route: {'Enabled' if (use_ai or use_ai_sql) else 'Manual Baseline'}")
    if use_ai_sql:
        print(f"[*] AI Routing Mode   : Lakehouse SQL (AI ROUTE ...)")
    elif use_ai:
        print(f"[*] AI Routing Mode   : Python API (ctx.ai_route)")
    print("-" * 80)

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"PubMed XML source file not found at: {xml_path}")

    # Initialize Lakehouse Context
    ctx = LakehouseContext(lakehouse_root=lakehouse_root)

    # -------------------------------------------------------------------------
    # STEP 0 & 1: BRONZE TIER - AI PROFILING & AUTONOMOUS INGESTION ROUTING
    # -------------------------------------------------------------------------
    if use_ai_sql:
        print("\n[STEP 0 & 1] Ingesting via Lakehouse SQL AI Layer (AI ROUTE)...")
        t0 = time.time()
        
        # 1. Profile with Lakehouse SQL
        print(" -> Executing SQL: AI PROFILE ...")
        profile_query = f"AI PROFILE '{xml_path}'"
        if api_key:
            profile_query += f" WITH (api_key = '{api_key}')"
        profile_res = ctx.sql(profile_query)
        print(f"    AI Format Detected : {profile_res.get('file_format')}")
        print(f"    Confidence Score   : {profile_res.get('confidence', 0):.2f}")
        print(f"    Recommended Op     : {profile_res.get('suggested_operator')}")
        print(f"    AI Provider        : {profile_res.get('ai_provider')}")

        # 2. Route & Execute with Lakehouse SQL
        print(" -> Executing SQL: AI ROUTE ... INTO bronze.pubmed_articles WITH (execute = true)")
        route_query = f"AI ROUTE '{xml_path}' INTO bronze.pubmed_articles WITH (execute = true"
        if api_key:
            route_query += f", api_key = '{api_key}'"
        route_query += ")"
        route_res = ctx.sql(route_query)
        print(f" -> SQL AI Routing status: {route_res.get('execution_status')} in {time.time() - t0:.2f}s.")

    elif use_ai:
        print("\n[STEP 0] Zero-Assumption Dataset Scanning & AI/Gemini Profiling...")
        t0 = time.time()

        # Step 0: Profile without prior knowledge
        decision = ctx.ai_profile(xml_path, api_key=api_key)
        print(f" -> Detected Format     : {decision.file_format}")
        print(f" -> Biological Domain   : {decision.domain}")
        print(f" -> Confidence Score    : {decision.confidence:.2f}")
        print(f" -> Recommended Operator: {decision.suggested_operator}")
        print(f" -> Recommended Target  : {decision.target_bronze_table}")
        print(f" -> AI Provider Engine  : {decision.ai_provider}")
        print(f" -> Reasoning Summary   : {decision.reasoning}")
        print(f" -> Estimated Rows      : {decision.estimated_rows}")

        print("\n[STEP 1] Autonomous AI Ingestion Routing into Bronze Tier...")
        plan = ctx.ai_route(
            xml_path,
            target_table="bronze.pubmed_articles",
            auto_execute=True,
            api_key=api_key,
        )
        print(f" -> Ingestion Plan Executed: {plan.execution_status} in {time.time() - t0:.2f}s.")
        print(f" -> Materialized Target    : {plan.target_bronze_table}")

    else:
        # Fallback manual parsing baseline
        print("\n[STEP 1] Ingesting PubMed Citations into Bronze Tier (Manual Baseline)...")
        t0 = time.time()

        pmids, titles, abstracts, pub_years = [], [], [], []
        auth_pmids, auth_names = [], []

        count = 0
        for event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "PubmedArticle":
                count += 1
                pmid = elem.findtext(".//MedlineCitation/PMID")
                title = elem.findtext(".//MedlineCitation/Article/ArticleTitle") or ""
                year = elem.findtext(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/Year") or "1975"

                abstract_nodes = elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
                abstract = " ".join([n.text for n in abstract_nodes if n.text])

                author_nodes = elem.findall(".//MedlineCitation/Article/AuthorList/Author")
                for a in author_nodes:
                    fore = a.findtext("ForeName") or a.findtext("Initials") or ""
                    last = a.findtext("LastName") or a.findtext("CollectiveName") or ""
                    name = f"{fore} {last}".strip()
                    if name and pmid:
                        auth_pmids.append(str(pmid))
                        auth_names.append(name.replace(" ", "_"))

                if pmid and abstract:
                    pmids.append(str(pmid))
                    titles.append(title)
                    abstracts.append(abstract)
                    pub_years.append(int(year) if year.isdigit() else 1975)

                elem.clear()
                if count >= max_articles:
                    break

        # Materialize Bronze Tables in Parquet
        tbl_articles = pa.Table.from_arrays(
            [
                pa.array(pmids, type=pa.string()),
                pa.array(titles, type=pa.string()),
                pa.array(abstracts, type=pa.string()),
                pa.array(pub_years, type=pa.int64()),
            ],
            names=["pmid", "title", "abstract", "year"],
        )

        tbl_authors = pa.Table.from_arrays(
            [
                pa.array(auth_pmids, type=pa.string()),
                pa.array(auth_names, type=pa.string()),
            ],
            names=["pmid", "author_id"],
        )

        art_uri = os.path.join(lakehouse_root, "bronze/pubmed_articles.parquet").replace("\\", "/")
        auth_uri = os.path.join(lakehouse_root, "bronze/pubmed_authors.parquet").replace("\\", "/")

        ctx.storage.write_tabular(tbl_articles, art_uri, file_format="parquet")
        ctx.catalog.register_table(
            TableMetadata(name="pubmed_articles", tier="bronze", uri=art_uri, schema=tbl_articles.schema, row_count=len(tbl_articles))
        )

        ctx.storage.write_tabular(tbl_authors, auth_uri, file_format="parquet")
        ctx.catalog.register_table(
            TableMetadata(name="pubmed_authors", tier="bronze", uri=auth_uri, schema=tbl_authors.schema, row_count=len(tbl_authors))
        )
        print(f" -> Scanned {count} raw XML entries in {time.time() - t0:.2f}s.")

    # Inspect Bronze catalog tables
    tbl_articles = ctx.read_table("bronze.pubmed_articles")
    tbl_authors = ctx.read_table("bronze.pubmed_authors")
    print(f" -> Verified 'bronze.pubmed_articles': {tbl_articles.num_rows} articles with abstracts.")
    print(f" -> Verified 'bronze.pubmed_authors' : {tbl_authors.num_rows} authorship relations.")

    # -------------------------------------------------------------------------
    # STEP 2: SILVER TIER - BIOMEDICAL ENTITY EXTRACTION (NER)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Running Named Entity Recognition (NER) via Lakehouse SQL...")
    t1 = time.time()

    sql_ner = f"""
    EXTRACT ENTITIES silver.extracted_entities
    FROM bronze.pubmed_articles
    WITH (
        text_col = 'abstract',
        key_col = 'pmid',
        vocab_path = '{vocab_path}'
    );
    """
    ctx.sql(sql_ner)
    ent_tbl = ctx.read_table("silver.extracted_entities")
    n_docs = len(set(ent_tbl["doc_id"].to_pylist()))
    print(f" -> Extracted {ent_tbl.num_rows} bio-entity mentions across {n_docs} distinct articles in {time.time() - t1:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 3: SILVER TIER - ONTOLOGY MAPPING TO CHEM2BIO2RDF
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Mapping Extracted Entities to Chem2Bio2RDF URIs...")
    t2 = time.time()

    sql_map = """
    MAP ONTOLOGY silver.mapped_entities
    FROM silver.extracted_entities
    WITH (
        ontology = 'chem2bio2rdf'
    );
    """
    ctx.sql(sql_map)
    mapped_tbl = ctx.read_table("silver.mapped_entities")
    print(f" -> Mapped {mapped_tbl.num_rows} entities to canonical ontology URIs in {time.time() - t2:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 4: GOLD TIER - HETEROGENEOUS KNOWLEDGE GRAPH CONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Constructing Heterogeneous Knowledge Graph (RDF Triples)...")
    t3 = time.time()

    sql_kg = """
    CONSTRUCT KNOWLEDGE GRAPH gold.biomedical_kg
    FROM bronze.pubmed_articles
    WITH (
        entities = 'silver.extracted_entities',
        authors = 'bronze.pubmed_authors',
        ontology = 'silver.mapped_entities'
    );
    """
    ctx.sql(sql_kg)
    kg_tbl = ctx.read_table("gold.biomedical_kg")
    preds = set(kg_tbl["predicate"].to_pylist())
    print(f" -> Generated {kg_tbl.num_rows} multi-relational triples in {time.time() - t3:.2f}s.")
    print(f" -> Predicates included: {sorted(list(preds))}")

    # -------------------------------------------------------------------------
    # STEP 5: GOLD TIER - AUTHOR CENTRALITY & CO-OCCURRENCE PROFILING
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Profiling Author Degree Centrality and Disease/Drug Associations...")
    t4 = time.time()

    sql_profile = """
    PROFILE GRAPH gold.author_profiles
    FROM gold.biomedical_kg
    WITH (
        target_type = 'author',
        top_n = 20
    );
    """
    ctx.sql(sql_profile)
    prof_tbl = ctx.read_table("gold.author_profiles")
    print(f" -> Profiling completed for {prof_tbl.num_rows} nodes in {time.time() - t4:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 6: LAKEHOUSE SQL ANALYTICS & INSIGHTS
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Executing Lakehouse SQL Queries on Graph Profiles...")
    print("-" * 80)
    top_authors = ctx.sql("""
        SELECT node_id, degree_centrality, out_degree, top_associations 
        FROM gold.author_profiles 
        ORDER BY degree_centrality DESC 
        LIMIT 10
    """)

    print(f"{'Author ID':<35} | {'Degree Centrality':<18} | {'Papers Authored':<15} | {'Associated Entities'}")
    print("-" * 110)
    for row in top_authors:
        node = row["node_id"].replace("author/", "")
        cent = f"{row['degree_centrality']:.4f}"
        papers = row["out_degree"]
        assocs = row["top_associations"][:50] + "..." if len(str(row["top_associations"])) > 50 else row["top_associations"]
        print(f"{node:<35} | {cent:<18} | {papers:<15} | {assocs}")

    # Query Knowledge Graph Triples
    print("-" * 80)
    print("\n[STEP 7] Sample RDF Triples from 'gold.biomedical_kg':")
    sample_triples = ctx.sql("""
        SELECT subject, predicate, object 
        FROM gold.biomedical_kg 
        LIMIT 8
    """)
    for row in sample_triples:
        print(f"  ({row['subject']}) --[{row['predicate']}]--> ({row['object']})")

    print("\n" + "=" * 80)
    print("  PUBMED PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PubMed Knowledge Graph Lakehouse Pipeline with AI Profiling")
    parser.add_argument("--xml", default="data/biomedical_nlp/pubmed/pubmed23n0001.xml", help="Path to PubMed XML")
    parser.add_argument("--vocab", default="data/biomedical_nlp/vocab_entities", help="Path to entity vocabularies")
    parser.add_argument("--storage", default="lakehouse_pubmed", help="Lakehouse root directory")
    parser.add_argument("--limit", type=int, default=1500, help="Maximum articles to parse")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI profiling and use manual ingestion")
    parser.add_argument("--ai-sql", action="store_true", help="Use Lakehouse SQL AI ROUTE syntax instead of Python API")
    parser.add_argument("--api-key", default=None, help="Google Gemini API key (optional, falls back to heuristic)")
    args = parser.parse_args()

    run_pubmed_pipeline(
        xml_path=args.xml,
        vocab_path=args.vocab,
        lakehouse_root=args.storage,
        max_articles=args.limit,
        use_ai=not args.no_ai,
        use_ai_sql=args.ai_sql,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
