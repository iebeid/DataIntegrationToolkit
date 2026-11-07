"""
Decoupled Turnkey UniRef Protein Sequence N-gram Graph Lakehouse Pipeline.

Workflow:
  1. Dataset Drop-Zone: Detects dataset dropped in `data/inbox/` or workspace directory.
  2. Decoupled AI Profiling: Scans dataset assuming zero prior knowledge (format, domain,
     structural markers, non-standard characters, cluster tags, duplicate ratio).
  3. Interactive Step-by-Step Prompting: Prompts the user on recommended actions:
     - Sequence cleaning (standard IUPAC 20 amino acids vs non-standard characters)
     - Entity resolution / deduplication of identical sequences across clusters
     - Tokenization strategy (N-gram directed transitions vs WordPiece/subword vs char)
     - Feature representations (transition probabilities P(v|u), TF-IDF, Count/BoW)
     - Machine learning labels & targets (TaxID, Organism, Cluster size)
     - Graph topology & adjacency matrix (Sparse COO/CSR, DirectGCN A_in/A_out, Dense NumPy)
     - Hierarchical Graclus graph coarsening
  4. Execution & Delivery: Executes the pipeline across Medallion tiers (Bronze -> Silver -> Gold)
     and drops all outputs and a complete JSON manifest to the known output destination.
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from toolkit.context import LakehouseContext
from toolkit.ai.scanner import DatasetScanner, DatasetScanResult
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


# ==============================================================================
# 1. DATASET DISCOVERY (INBOX / WORKSPACE SCANNER)
# ==============================================================================

def discover_input_dataset(specified_path: Optional[str] = None) -> str:
    """
    Finds the candidate dataset:
      1. Explicit path if provided by user.
      2. Files inside workspace inbox: `data/inbox/`.
      3. Bioinformatics datasets in `data/bioinformatics/`.
    """
    if specified_path:
        resolved = resolve_path(specified_path)
        if os.path.exists(resolved):
            return resolved
        raise FileNotFoundError(f"Specified input dataset not found: {specified_path}")

    # Check inbox first
    inbox_dir = resolve_path("data/inbox")
    if os.path.exists(inbox_dir):
        inbox_files = [
            os.path.join(inbox_dir, f)
            for f in os.listdir(inbox_dir)
            if f.endswith((".fasta", ".fa", ".faa", ".xml", ".tsv", ".csv", ".parquet"))
        ]
        if inbox_files:
            inbox_files.sort(key=os.path.getmtime, reverse=True)
            return inbox_files[0]

    # Check bioinformatics default folder
    bio_dir = resolve_path("data/bioinformatics")
    if os.path.exists(bio_dir):
        sample_path = os.path.join(bio_dir, "uniref50_sample.fasta")
        if os.path.exists(sample_path):
            return sample_path
        full_path = os.path.join(bio_dir, "uniref50.fasta")
        if os.path.exists(full_path):
            return full_path
        for f in os.listdir(bio_dir):
            if f.endswith((".fasta", ".fa")):
                return os.path.join(bio_dir, f)

    raise FileNotFoundError("No dataset found in 'data/inbox/' or 'data/bioinformatics/'.")


# ==============================================================================
# 2. DECOUPLED AI PROFILING PHASE
# ==============================================================================

def perform_ai_profiling(dataset_path: str, ctx: LakehouseContext, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Decoupled AI Dataset Profiler.
    Inspects raw physical bytes, delimiter distributions, structural markers,
    character sets, and header metadata without prior assumptions.
    """
    print("=" * 85)
    print("      PHASE 1: DECOUPLED AI DATASET PROFILING & STRUCTURAL INSPECTION")
    print("=" * 85)
    print(f"[*] Scanning Raw Input File : {dataset_path}")

    file_size_bytes = os.path.getsize(dataset_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"[*] Physical File Size      : {file_size_mb:,.2f} MB ({file_size_bytes:,} bytes)")

    # Run Lakehouse zero-assumption scanner & AI profile
    t0 = time.time()
    scanner = DatasetScanner(max_sample_lines=60, max_sample_bytes=65536)
    scan_result: DatasetScanResult = scanner.scan(dataset_path)
    profile_decision = ctx.ai_profile(dataset_path, api_key=api_key)

    scan_time = time.time() - t0

    # Check for specific protein / UniRef markers
    headers = [l for l in scan_result.sample_lines if l.startswith(">")]
    seq_lines = [l for l in scan_result.sample_lines if not l.startswith(">") and l.strip()]

    sample_seq_str = "".join(seq_lines).upper()
    total_sample_chars = len(sample_seq_str)
    all_sample_chars = set(sample_seq_str)

    # Standard 20 amino acids
    standard_20 = set("ACDEFGHIKLMNPQRSTVWY")
    non_standard_found = sorted(list(all_sample_chars - standard_20))

    has_tax_tags = any("taxid=" in h.lower() or "tax=" in h.lower() for h in headers)
    has_cluster_tags = any("repid=" in h.lower() or " n=" in h.lower() for h in headers)
    is_uniref = any("uniref" in h.lower() for h in headers) or "uniref" in os.path.basename(dataset_path).lower()

    profile_report = {
        "dataset_path": dataset_path,
        "filename": scan_result.filename,
        "file_size_mb": round(file_size_mb, 2),
        "domain": profile_decision.domain,
        "file_format": profile_decision.file_format,
        "confidence": profile_decision.confidence,
        "ai_provider": profile_decision.ai_provider,
        "reasoning": profile_decision.reasoning,
        "estimated_records": scan_result.estimated_row_count,
        "is_uniref_clusters": is_uniref,
        "detected_headers_count_sample": len(headers),
        "sample_total_chars": total_sample_chars,
        "has_non_standard_chars": len(non_standard_found) > 0,
        "non_standard_chars": non_standard_found,
        "has_taxonomic_labels": has_tax_tags,
        "has_cluster_metadata": has_cluster_tags,
        "structural_markers": scan_result.structural_markers,
        "scan_duration_sec": round(scan_time, 3),
    }

    print(f" -> AI Detected Domain      : {profile_report['domain'].upper()}")
    print(f" -> AI Detected File Format : {profile_report['file_format'].upper()} (Confidence: {profile_report['confidence']:.2f})")
    print(f" -> AI Provider / Engine    : {profile_report['ai_provider']}")
    print(f" -> Estimated Total Records : ~{profile_report['estimated_records']:,} rows")
    print(f" -> Structural Markers      : {', '.join(profile_report['structural_markers']) if profile_report['structural_markers'] else 'None'}")
    print(f" -> Non-Standard Amino Acids: {', '.join(non_standard_found) if non_standard_found else 'None (100% Standard IUPAC)'}")
    print(f" -> Taxonomic Tags Found    : {'Yes (TaxID, Organism Name)' if has_tax_tags else 'None'}")
    print(f" -> Cluster Metadata Found  : {'Yes (RepID, Cluster Size n)' if has_cluster_tags else 'None'}")
    print(f" -> AI Reasoning Summary    : {profile_report['reasoning']}")
    print("-" * 85)

    return profile_report


# ==============================================================================
# 3. INTERACTIVE STEP-BY-STEP PROMPTING & RECOMMENDATION SYSTEM
# ==============================================================================

def prompt_choice(prompt_text: str, options: List[str], default_idx: int = 0, interactive: bool = True) -> int:
    """Presents a numbered choice to the user and returns the 0-based selected index."""
    print(f"\n{prompt_text}")
    for idx, opt in enumerate(options):
        prefix = " [*]" if idx == default_idx else "    "
        print(f"{prefix} [{idx + 1}] {opt}")

    if not interactive:
        print(f" -> [Autonomous Mode] Auto-selected Recommended Option: [{default_idx + 1}] {options[default_idx]}")
        return default_idx

    while True:
        try:
            choice = input(f"Select option [1-{len(options)}] (Enter for default [{default_idx + 1}]): ").strip()
            if not choice:
                return default_idx
            val = int(choice) - 1
            if 0 <= val < len(options):
                return val
            print(f"Please enter a valid number between 1 and {len(options)}.")
        except (ValueError, EOFError):
            return default_idx


def prompt_int(prompt_text: str, default_val: int, interactive: bool = True) -> int:
    """Prompts the user for an integer parameter."""
    if not interactive:
        print(f"{prompt_text} -> [Autonomous Mode] Using default: {default_val}")
        return default_val
    try:
        val = input(f"{prompt_text} (Enter for default {default_val}): ").strip()
        return int(val) if val.isdigit() and int(val) > 0 else default_val
    except (ValueError, EOFError):
        return default_val


def formulate_pipeline_plan(
    profile_report: Dict[str, Any],
    interactive: bool = True,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Presents the interactive step-by-step decision tree to the user based on profiling.
    Allows user to decide cleaning, deduplication, tokenization, features, ML labels, and adjacency.
    """
    cli = cli_overrides or {}
    print("\n" + "=" * 85)
    print("      PHASE 2: INTERACTIVE STEP-BY-STEP ACTION RECOMMENDATIONS")
    print("=" * 85)
    print("The AI profiler detected dataset characteristics. Please configure each processing step:")

    plan = {}

    # -------------------------------------------------------------------------
    # STEP A: Sequence Cleaning
    # -------------------------------------------------------------------------
    non_std = profile_report.get("non_standard_chars", [])
    has_non_std = len(non_std) > 0
    clean_prompt = (
        f"Step 1/7: Sequence Sanitization & Cleaning\n"
        f"AI Finding: {'Non-standard characters detected: ' + ', '.join(non_std) if has_non_std else 'Sequences conform to standard alphabet.'}\n"
        f"Do you want to clean/sanitize sequences from unwanted characters?"
    )
    clean_options = [
        "Sanitize to standard 20 IUPAC amino acids (strip gaps/asterisks, map ambiguous B->D, Z->E, J->L, X->A) (Recommended)",
        "Strip only whitespace, gaps (-), and stop asterisks (*); keep ambiguous characters",
        "Keep raw sequences exactly as-is without modification",
    ]
    c_idx = prompt_choice(clean_prompt, clean_options, default_idx=0, interactive=interactive)
    plan["clean_sequences"] = c_idx in (0, 1)
    plan["remove_non_standard"] = (c_idx == 0)

    # -------------------------------------------------------------------------
    # STEP B: Sequence Entity Resolution & Deduplication
    # -------------------------------------------------------------------------
    dedup_prompt = (
        "Step 2/7: Entity Resolution & Deduplication\n"
        "AI Finding: In cluster datasets (UniRef/SwissProt), redundant identical sequences may exist across clusters.\n"
        "Should Entity Resolution be performed to collapse exact duplicate sequences and preserve cluster IDs?"
    )
    dedup_options = [
        "Yes: Run sequence deduplication & aggregate cluster accessions (Recommended)",
        "No: Retain every FASTA entry as a separate record",
    ]
    d_idx = prompt_choice(dedup_prompt, dedup_options, default_idx=0, interactive=interactive)
    plan["run_deduplication"] = (d_idx == 0)

    # -------------------------------------------------------------------------
    # STEP C: Tokenization Strategy
    # -------------------------------------------------------------------------
    tok_prompt = (
        "Step 3/7: Sequence Tokenization Strategy\n"
        "How do you want to tokenize the protein sequences into tokens / graph nodes?"
    )
    tok_options = [
        "N-gram Graph Tokenizer (Directed k-mer transitions: s[i:i+n] -> s[i+1:i+1+n]) (Recommended for DirectGCN / Graph Learning)",
        "Subword / WordPiece BPE Tokenizer (128-dim subword embeddings)",
        "Single Amino Acid Character Tokenizer (n=1, 20 canonical amino acid nodes)",
    ]
    t_idx = prompt_choice(tok_prompt, tok_options, default_idx=0, interactive=interactive)
    if t_idx == 0:
        plan["tokenizer"] = "ngram_graph"
        default_n = cli.get("n", 3)
        plan["n"] = prompt_int("   Enter N-gram length k (e.g. 2 for di-peptides, 3 for tri-peptides, 4 for 4-mers):", default_n, interactive)
    elif t_idx == 1:
        plan["tokenizer"] = "wordpiece_subword"
        plan["n"] = 3
    else:
        plan["tokenizer"] = "char"
        plan["n"] = 1

    # -------------------------------------------------------------------------
    # STEP D: Feature Representations
    # -------------------------------------------------------------------------
    feat_prompt = (
        "Step 4/7: Feature Representations & Statistical Modeling\n"
        "Which feature representations should be computed from the tokenized sequences?"
    )
    feat_options = [
        "Directed transition weights W(u,v) and transition probabilities P(v|u) (Recommended for Markovian / Graph workflows)",
        "Sequence-level TF-IDF vectorization (128-dim dense L2-normalized embeddings)",
        "Both Graph Transition Probabilities AND TF-IDF Sequence Embeddings",
        "Token Count / Bag-of-Words (BoW) frequency vectors",
        "Raw co-occurrence transition counts only",
    ]
    f_idx = prompt_choice(feat_prompt, feat_options, default_idx=0, interactive=interactive)
    plan["compute_probabilities"] = f_idx in (0, 2)
    plan["compute_tfidf"] = f_idx in (1, 2)
    plan["compute_bow"] = (f_idx == 3)

    # -------------------------------------------------------------------------
    # STEP E: Machine Learning Labels & Targets
    # -------------------------------------------------------------------------
    has_labels = profile_report.get("has_taxonomic_labels", False)
    ml_prompt = (
        f"Step 5/7: Machine Learning Labels & Supervised Metadata\n"
        f"AI Finding: {'Embedded taxonomic annotations (TaxID, Organism) detected in headers.' if has_labels else 'No explicit taxonomic tags detected.'}\n"
        f"How should target labels be prepared for downstream Machine Learning?"
    )
    ml_options = [
        "Extract taxonomy IDs (TaxID) and organism names as classification labels (Recommended for Taxonomy Prediction)" if has_labels else "Extract accession IDs as class labels",
        "Unsupervised Representation (No target labels; graph pre-training / structure embeddings)",
        "Extract cluster sizes (n=...) as continuous regression targets",
    ]
    m_idx = prompt_choice(ml_prompt, ml_options, default_idx=0, interactive=interactive)
    plan["ml_label_mode"] = "taxonomy" if m_idx == 0 else ("unsupervised" if m_idx == 1 else "cluster_size")

    # -------------------------------------------------------------------------
    # STEP F: Graph Topology & Adjacency Matrix Export
    # -------------------------------------------------------------------------
    adj_prompt = (
        "Step 6/7: Graph Adjacency Matrix Representation & Export\n"
        "How should the graph adjacency matrix be computed and saved?"
    )
    adj_options = [
        "Sparse COO/CSR matrix (.npz) + DirectGCN normalized matrices (A_in, A_out) (Recommended for GNNs / ProtGram)",
        "Dense adjacency matrix (NumPy .npy format)",
        "Both Sparse (.npz) and Dense (.npy) matrices",
        "Lakehouse Graph Edge List table only (Parquet format)",
    ]
    a_idx = prompt_choice(adj_prompt, adj_options, default_idx=0, interactive=interactive)
    plan["export_sparse"] = a_idx in (0, 2)
    plan["export_directgcn"] = a_idx in (0, 2)
    plan["export_dense"] = a_idx in (1, 2)

    # -------------------------------------------------------------------------
    # STEP G: Graph Coarsening (Graclus / HEM)
    # -------------------------------------------------------------------------
    coarsen_prompt = (
        "Step 7/7: Graph Coarsening (Topological Compression for Scalable GNNs)\n"
        "Do you want to run Graclus Heavy-Edge Matching (HEM) coarsening on the resulting graph?"
    )
    coarsen_options = [
        "Yes: 1-level Graclus coarsening (~48% node reduction, preserves motif topology) (Recommended)",
        "Yes: 2-level hierarchical coarsening (~75% node reduction)",
        "No: Preserve uncoarsened graph only",
    ]
    g_idx = prompt_choice(coarsen_prompt, coarsen_options, default_idx=0, interactive=interactive)
    plan["coarsen"] = (g_idx in (0, 1))
    plan["coarsen_levels"] = 1 if g_idx == 0 else (2 if g_idx == 1 else 0)

    # Sequence limits
    plan["limit"] = cli.get("limit", 1000)
    plan["sample"] = cli.get("sample", None)

    print("\n" + "=" * 85)
    print("  CONFIRMED PIPELINE EXECUTION CONFIGURATION:")
    print("=" * 85)
    print(f"  * Sequence Sanitization  : {'Enabled (IUPAC Standard 20 AA)' if plan['remove_non_standard'] else ('Basic cleaning' if plan['clean_sequences'] else 'Raw')}")
    print(f"  * Entity Resolution/Dedup: {'Enabled (Deduplicate identical sequences)' if plan['run_deduplication'] else 'Disabled'}")
    print(f"  * Tokenizer Mode         : {plan['tokenizer']} (n={plan['n']})")
    print(f"  * Feature Representations: Probabilities={plan['compute_probabilities']}, TF-IDF={plan['compute_tfidf']}, BoW={plan['compute_bow']}")
    print(f"  * ML Labels Target       : {plan['ml_label_mode']}")
    print(f"  * Adjacency Matrices     : Sparse={plan['export_sparse']}, DirectGCN={plan['export_directgcn']}, Dense={plan['export_dense']}")
    print(f"  * Graclus Coarsening     : {'Enabled (Levels=' + str(plan['coarsen_levels']) + ')' if plan['coarsen'] else 'Disabled'}")
    lim_str = f"First {plan['limit']:,} sequences (Prefix)" if plan.get("limit") else (f"{plan['sample']:,} sequences (Reservoir)" if plan.get("sample") else "All")
    print(f"  * Ingestion Limit/Sample : {lim_str}")
    print("-" * 85)

    return plan


# ==============================================================================
# 4. EXECUTION & OUTPUT MATERIALIZATION
# ==============================================================================

def execute_pipeline(
    dataset_path: str,
    output_dir: str,
    plan: Dict[str, Any],
    profile_report: Dict[str, Any],
    ctx: Optional[LakehouseContext] = None,
) -> Dict[str, Any]:
    """
    Executes all configured pipeline stages and deposits all materialized
    tables, matrices, and metadata manifest to the designated output location.
    """
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if ctx is None:
        ctx = LakehouseContext(lakehouse_root=output_dir)

    print("\n" + "=" * 85)
    print("      PHASE 3: PIPELINE EXECUTION & OUTPUT MATERIALIZATION")
    print("=" * 85)
    print(f"[*] Target Known Output Location: {output_dir}")

    execution_manifest: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dataset": dataset_path,
        "output_directory": output_dir,
        "ai_profile": profile_report,
        "pipeline_config": plan,
        "stages": {},
    }

    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. BRONZE TIER: FASTA INGESTION & SANITIZATION
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Ingesting & Sanitizing FASTA into Bronze Tier...")
    t0 = time.time()
    bronze_meta = ctx.ingest_fasta(
        source_uri=dataset_path,
        target_name="bronze.uniref_sequences",
        sample=plan.get("sample"),
        limit=plan.get("limit"),
        clean_sequences=plan.get("clean_sequences", False),
        remove_non_standard=plan.get("remove_non_standard", False),
    )
    t_bronze = time.time() - t0
    print(f" -> Bronze Table Created: 'bronze.uniref_sequences' ({bronze_meta.row_count:,} rows in {t_bronze:.2f}s)")
    execution_manifest["stages"]["bronze_ingestion"] = {
        "table": "bronze.uniref_sequences",
        "rows": bronze_meta.row_count,
        "duration_sec": round(t_bronze, 2),
        "uri": bronze_meta.uri,
    }

    current_source_table = "bronze.uniref_sequences"

    # -------------------------------------------------------------------------
    # 2. SILVER TIER: ENTITY RESOLUTION / DEDUPLICATION
    # -------------------------------------------------------------------------
    if plan.get("run_deduplication", False):
        print("\n[STEP 2] Running Sequence Entity Resolution & Deduplication...")
        t1 = time.time()
        dedup_meta = ctx.deduplicate_sequences(
            source_table=current_source_table,
            target_name="silver.uniref_deduplicated_sequences",
            sequence_column="sequence",
            id_column="sequence_id",
        )
        t_dedup = time.time() - t1
        props = dedup_meta.properties
        print(f" -> Deduplication Results: {props.get('original_rows', 0):,} -> {props.get('deduplicated_rows', 0):,} sequences")
        print(f" -> Redundant Records Cleared: {props.get('duplicates_removed', 0):,} ({props.get('deduplication_ratio', 0.0):.2%} reduction in {t_dedup:.2f}s)")
        execution_manifest["stages"]["entity_resolution"] = {
            "table": "silver.uniref_deduplicated_sequences",
            "original_rows": props.get("original_rows"),
            "deduplicated_rows": props.get("deduplicated_rows"),
            "duplicates_removed": props.get("duplicates_removed"),
            "duration_sec": round(t_dedup, 2),
            "uri": dedup_meta.uri,
        }
        current_source_table = "silver.uniref_deduplicated_sequences"
        seq_col = "sequence"
        id_col = "canonical_id"
    else:
        seq_col = "sequence"
        id_col = "sequence_id"

    # -------------------------------------------------------------------------
    # 3. SILVER TIER: OPTIONAL TF-IDF / EMBEDDINGS
    # -------------------------------------------------------------------------
    if plan.get("compute_tfidf", False):
        print("\n[STEP 3] Computing Sequence TF-IDF Embeddings...")
        t2 = time.time()
        tfidf_meta = ctx.create_embeddings(
            source_table=current_source_table,
            target_name="silver.uniref_tfidf_embeddings",
            key_column=id_col,
            columns=[seq_col],
            method="tfidf",
            dimension=128,
            subword=True,
        )
        t_tfidf = time.time() - t2
        print(f" -> TF-IDF Vector Table: 'silver.uniref_tfidf_embeddings' ({tfidf_meta.row_count:,} embeddings in {t_tfidf:.2f}s)")
        execution_manifest["stages"]["tfidf_embeddings"] = {
            "table": "silver.uniref_tfidf_embeddings",
            "rows": tfidf_meta.row_count,
            "dimension": 128,
            "duration_sec": round(t_tfidf, 2),
            "uri": tfidf_meta.uri,
        }

    # -------------------------------------------------------------------------
    # 4. GOLD TIER: DIRECTED N-GRAM GRAPH PROJECTION & ADJACENCY MATRICES
    # -------------------------------------------------------------------------
    n_val = plan.get("n", 3)
    print(f"\n[STEP 4] Constructing Directed {n_val}-gram Graph Topology & Matrices...")
    t3 = time.time()

    matrices_dir = os.path.join(output_dir, "matrices") if (plan.get("export_sparse") or plan.get("export_dense")) else None

    graph_meta = ctx.construct_ngram_graph(
        source_table=current_source_table,
        target_name="gold.uniref_ngram_graph",
        sequence_column=seq_col,
        id_column=id_col,
        n=n_val,
        directed=True,
        nodes_target="silver.uniref_ngram_nodes",
        compute_probabilities=plan.get("compute_probabilities", True),
        export_matrices_dir=matrices_dir,
        export_dense=plan.get("export_dense", False),
        export_directgcn=plan.get("export_directgcn", True),
    )
    t_graph = time.time() - t3
    g_props = graph_meta.properties
    num_nodes = g_props.get("num_nodes", 0)
    num_edges = g_props.get("num_edges", graph_meta.row_count)
    density = g_props.get("density", 0.0)

    print(f" -> Gold Graph Table: 'gold.uniref_ngram_graph' ({num_edges:,} edges in {t_graph:.2f}s)")
    print(f" -> Node Vocabulary Table: 'silver.uniref_ngram_nodes' ({num_nodes:,} unique {n_val}-grams)")
    print(f" -> Graph Metrics: |V| = {num_nodes:,}, |E| = {num_edges:,}, Density = {density:.6e}")
    if matrices_dir and os.path.exists(matrices_dir):
        exported_files = os.listdir(matrices_dir)
        print(f" -> Exported Adjacency Files: {', '.join(exported_files)} in '{matrices_dir}'")

    execution_manifest["stages"]["graph_construction"] = {
        "table": "gold.uniref_ngram_graph",
        "nodes_table": "silver.uniref_ngram_nodes",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "matrices_directory": matrices_dir,
        "exported_matrix_files": os.listdir(matrices_dir) if matrices_dir and os.path.exists(matrices_dir) else [],
        "duration_sec": round(t_graph, 2),
        "uri": graph_meta.uri,
    }

    # -------------------------------------------------------------------------
    # 5. GOLD TIER: GRAPH COARSENING (GRACLUS / HEM)
    # -------------------------------------------------------------------------
    if plan.get("coarsen", False):
        levels = plan.get("coarsen_levels", 1)
        print(f"\n[STEP 5] Compressing Graph Topology via Graclus Coarsening (Levels = {levels})...")
        t4 = time.time()
        coarse_meta = ctx.coarsen_graph(
            source_table="gold.uniref_ngram_graph",
            target_name="gold.uniref_coarse_graph",
            src_col="source",
            dst_col="target",
            weight_col="weight",
            levels=levels,
            method="graclus",
            mapping_table="silver.uniref_cluster_map",
        )
        t_coarse = time.time() - t4
        c_props = coarse_meta.properties
        orig_v = c_props.get("orig_nodes", num_nodes)
        coarse_v = c_props.get("coarsened_nodes", 0)
        red_v = c_props.get("node_reduction_ratio", 0.0)
        orig_e = c_props.get("orig_edges", num_edges)
        coarse_e = c_props.get("coarsened_edges", coarse_meta.row_count)
        red_e = c_props.get("edge_reduction_ratio", 0.0)

        print(f" -> Coarsened Graph: 'gold.uniref_coarse_graph' ({coarse_e:,} edges in {t_coarse:.2f}s)")
        print(f" -> Node Reduction: {orig_v:,} -> {coarse_v:,} ({red_v:.2%})")
        print(f" -> Edge Reduction: {orig_e:,} -> {coarse_e:,} ({red_e:.2%})")
        print(f" -> Node Cluster Map: 'silver.uniref_cluster_map'")

        execution_manifest["stages"]["graph_coarsening"] = {
            "table": "gold.uniref_coarse_graph",
            "cluster_map_table": "silver.uniref_cluster_map",
            "levels": levels,
            "original_nodes": orig_v,
            "coarsened_nodes": coarse_v,
            "node_reduction_ratio": red_v,
            "original_edges": orig_e,
            "coarsened_edges": coarse_e,
            "edge_reduction_ratio": red_e,
            "duration_sec": round(t_coarse, 2),
            "uri": coarse_meta.uri,
        }

    # -------------------------------------------------------------------------
    # 6. LAKEHOUSE SQL ANALYTICS & INSIGHTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("      PHASE 4: LAKEHOUSE SQL ANALYTICS & RESULTS")
    print("=" * 85)

    print(" -> Top 8 Most Frequent Directed Amino Acid Transitions:")
    top_trans = ctx.sql("""
        SELECT source, target, weight, transition_prob
        FROM gold.uniref_ngram_graph
        ORDER BY weight DESC
        LIMIT 8;
    """)
    print(f"   {'Source':<12} | {'Target':<12} | {'Frequency Weight':<20} | {'Transition Prob P(v|u)'}")
    print("   " + "-" * 70)
    for row in top_trans:
        prob = f"{row.get('transition_prob', 0.0):.4f}"
        print(f"   {row['source']:<12} | {row['target']:<12} | {row['weight']:<20,} | {prob}")

    print("\n -> Top 8 Hub N-grams by Total Degree (In-Degree + Out-Degree):")
    top_nodes = ctx.sql("""
        SELECT ngram, in_degree, out_degree, total_degree, protein_count
        FROM silver.uniref_ngram_nodes
        ORDER BY total_degree DESC
        LIMIT 8;
    """)
    print(f"   {'N-gram Motif':<15} | {'In-Degree':<12} | {'Out-Degree':<12} | {'Total Degree':<15} | {'Proteins'}")
    print("   " + "-" * 75)
    for row in top_nodes:
        in_d = f"{int(row.get('in_degree', 0)):,}"
        out_d = f"{int(row.get('out_degree', 0)):,}"
        tot_d = f"{int(row.get('total_degree', 0)):,}"
        p_cnt = f"{int(row.get('protein_count', 0)):,}"
        print(f"   {row['ngram']:<15} | {in_d:<12} | {out_d:<12} | {tot_d:<15} | {p_cnt}")

    total_pipeline_time = time.time() - t_start
    execution_manifest["total_duration_sec"] = round(total_pipeline_time, 2)

    # Save execution manifest to output directory
    manifest_path = os.path.join(output_dir, "pipeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(execution_manifest, mf, indent=2)
    print(f"\n[*] Execution Manifest Saved to: {manifest_path}")

    print("-" * 85)
    print(" -> Lakehouse Catalog Inventory at Destination:")
    for tier in ["bronze", "silver", "gold"]:
        tbls = ctx.catalog.list_tables(tier=tier)
        for meta in tbls:
            print(f"    [{meta.tier.upper():<6}] {meta.qualified_name:<35} | Rows: {meta.row_count:<8,} | URI: {meta.uri}")

    print("\n" + "=" * 85)
    print(f"  ALL WORKFLOW PHASES COMPLETED SUCCESSFULLY IN {total_pipeline_time:.2f}s!")
    print("=" * 85)

    return execution_manifest


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="UniRef Sequence AI Profiling & Decoupled Interactive N-gram Pipeline")
    parser.add_argument("--input", "--fasta", default=None, help="Path to input dataset (if omitted, auto-discovers in data/inbox/ or data/bioinformatics/)")
    parser.add_argument("--output", "--output-dir", "--storage", default="lakehouse_uniref", help="Output directory location for all lakehouse tiers and matrices")
    parser.add_argument("--n", type=int, default=3, help="Default N-gram size (k-mer length)")
    parser.add_argument("--limit", type=int, default=1000, help="Prefix sequence limit for rapid testing on large files (0 for unlimited)")
    parser.add_argument("--sample", type=int, default=0, help="Reservoir sample size (0 to use --limit)")
    parser.add_argument("--api-key", default=None, help="Google Gemini API key (optional; defaults to heuristic classifier if omitted)")
    parser.add_argument("--auto", "--yes", action="store_true", help="Autonomous mode: accept all AI-recommended defaults without interactive prompts")
    args = parser.parse_args()

    # Step 1: Discover input file
    input_file = discover_input_dataset(args.input)

    # Step 2: Initialize Lakehouse context & run decoupled AI profiling
    output_dir = resolve_path(args.output)
    ctx = LakehouseContext(lakehouse_root=output_dir)
    profile_report = perform_ai_profiling(input_file, ctx=ctx, api_key=args.api_key)

    # Step 3: Interactive step-by-step decision tree
    is_interactive = not args.auto and sys.stdin.isatty()
    cli_overrides = {
        "n": args.n,
        "limit": args.limit if args.limit > 0 else None,
        "sample": args.sample if args.sample > 0 else None,
    }
    plan = formulate_pipeline_plan(
        profile_report=profile_report,
        interactive=is_interactive,
        cli_overrides=cli_overrides,
    )

    # Step 4: Execute configured pipeline and deposit to known output location
    execute_pipeline(
        dataset_path=input_file,
        output_dir=output_dir,
        plan=plan,
        profile_report=profile_report,
        ctx=ctx,
    )


if __name__ == "__main__":
    main()
