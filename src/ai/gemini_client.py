"""
Gemini API Client and AI Dataset Profiler with Heuristic Fallback.
Leverages Google Gemini LLM API (gemini-2.0-flash / gemini-1.5-flash) to semantically
profile unknown datasets and determine the optimal Medallion Bronze ingestion strategy.
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.ai.scanner import DatasetScanResult


@dataclass
class ProfilingDecision:
    """Structured AI decision regarding dataset domain and ingestion routing."""
    domain: str
    file_format: str
    target_bronze_table: str
    suggested_operator: str
    recommended_sql: str
    confidence: float
    reasoning: str
    provider: str = "gemini-2.0-flash"


class GeminiClient:
    """
    Interfaces with Google Gemini LLM API to analyze dataset samples.
    Falls back gracefully to a deterministic heuristic classifier when offline
    or when GEMINI_API_KEY is not configured.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model

    def classify_and_plan(self, scan: DatasetScanResult) -> ProfilingDecision:
        """
        Profiles the dataset and returns an Ingestion Plan.
        Attempts Gemini API call if API key is present; otherwise falls back to heuristics.
        """
        if self.api_key:
            try:
                return self._call_gemini_api(scan)
            except Exception as exc:
                decision = self._heuristic_classify(scan)
                decision.reasoning += f" (Gemini API attempted but failed: {str(exc)}; used heuristic classifier)"
                return decision

        return self._heuristic_classify(scan)

    def _call_gemini_api(self, scan: DatasetScanResult) -> ProfilingDecision:
        """Calls Google Gemini REST API with structured JSON output instructions."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        prompt = f"""
You are an autonomous AI data architect and ingestion router for an enterprise Medallion Lakehouse engine.
Analyze the following unknown raw dataset sample and determine its domain, format, and optimal Bronze ingestion pipeline.

DATASET PROFILE:
- File Name: {scan.filename}
- File Size (Bytes): {scan.file_size_bytes}
- Dominant Delimiter: {repr(scan.dominant_delimiter)}
- Structural Markers: {', '.join(scan.structural_markers) if scan.structural_markers else 'None detected'}
- Candidate Headers: {scan.candidate_headers}
- First Sample Lines:
{scan.sample_text}

JSON RESPONSE INSTRUCTIONS:
Respond with a strict JSON object with these exact keys:
{{
    "domain": "<bioinformatics | biomedical_nlp | entity_resolution | graph_network | generic_tabular>",
    "file_format": "<fasta | xml | idmapping | ppi_biogrid | ppi_string | csv | tsv | parquet | json>",
    "target_bronze_table": "<bronze.table_name>",
    "suggested_operator": "<IngestFastaOperator | ExtractEntitiesOperator | BronzeIngestOperator | IngestPPIOperator | IngestIDMappingOperator>",
    "recommended_sql": "<valid Lakehouse SQL string to ingest into Bronze tier>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<short explanation of structural patterns detected>"
}}
"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        raw_json_str = body["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(raw_json_str)

        return ProfilingDecision(
            domain=data.get("domain", "generic_tabular"),
            file_format=data.get("file_format", "csv"),
            target_bronze_table=data.get("target_bronze_table", "bronze.raw_dataset"),
            suggested_operator=data.get("suggested_operator", "BronzeIngestOperator"),
            recommended_sql=data.get("recommended_sql", f"INGEST FROM '{scan.path}' INTO bronze.raw_dataset;"),
            confidence=float(data.get("confidence", 0.95)),
            reasoning=data.get("reasoning", "Classified via Gemini multimodal LLM."),
            provider=self.model,
        )

    def _heuristic_classify(self, scan: DatasetScanResult) -> ProfilingDecision:
        """Deterministic heuristic classifier when Gemini API key is absent or offline."""
        fn = scan.filename.lower()
        markers = set(scan.structural_markers)

        # 1. Bioinformatics FASTA
        if "FASTA_HEADER" in markers or fn.endswith(".fasta") or fn.endswith(".fa"):
            clean_name = os.path.splitext(scan.filename)[0].lower().replace("-", "_").replace(" ", "_")
            if "uniref" in clean_name or "UNIREF_CLUSTER_FASTA" in markers:
                tbl = "bronze.uniref_sequences"
                reason = f"Detected UniRef protein cluster FASTA with sequence records, cluster IDs, and taxonomic tags in '{scan.filename}'."
            elif "protein" in clean_name:
                tbl = f"bronze.{clean_name}"
                reason = f"Detected protein FASTA sequence entries in '{scan.filename}'."
            else:
                tbl = "bronze.proteins"
                reason = f"Detected FASTA header marker '>' and biological sequence strings in '{scan.filename}'."

            return ProfilingDecision(
                domain="bioinformatics",
                file_format="fasta",
                target_bronze_table=tbl,
                suggested_operator="IngestFastaOperator",
                recommended_sql=f"INGEST FASTA FROM '{scan.path}' INTO {tbl};",
                confidence=0.98,
                reasoning=reason,
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 2. Biomedical NLP PubMed XML
        if "PUBMED_XML_CITATION" in markers or (fn.endswith(".xml") and "pubmed" in fn):
            return ProfilingDecision(
                domain="biomedical_nlp",
                file_format="xml",
                target_bronze_table="bronze.pubmed_articles",
                suggested_operator="BronzeIngestOperator",
                recommended_sql=f"INGEST FROM '{scan.path}' INTO bronze.pubmed_articles WITH (format = 'xml');",
                confidence=0.96,
                reasoning=f"Detected Medline/PubMed XML article structure (<PubmedArticleSet>) in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 3. UniProt ID Mapping TSV
        if "UNIPROT_IDMAPPING" in markers or "idmapping" in fn:
            return ProfilingDecision(
                domain="bioinformatics",
                file_format="idmapping",
                target_bronze_table="bronze.uniprot_idmapping",
                suggested_operator="IngestIDMappingOperator",
                recommended_sql=f"INGEST IDMAPPING FROM '{scan.path}' INTO bronze.uniprot_idmapping WITH (delimiter = '\\t');",
                confidence=0.95,
                reasoning=f"Detected UniProt identifier mapping column signatures (UniProtKB-AC/Gene_Name) in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 4. Protein-Protein Interaction (BioGRID)
        if "BIOGRID_PPI" in markers or "biogrid" in fn:
            return ProfilingDecision(
                domain="bioinformatics",
                file_format="ppi_biogrid",
                target_bronze_table="bronze.ppi_biogrid",
                suggested_operator="IngestPPIOperator",
                recommended_sql=f"INGEST PPI FROM '{scan.path}' INTO bronze.ppi_biogrid WITH (source = 'biogrid');",
                confidence=0.94,
                reasoning=f"Detected BioGRID physical interaction format headers (INTERACTOR_A, OFFICIAL_SYMBOL) in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 5. Protein-Protein Interaction (STRING)
        if "STRING_PPI_NETWORK" in markers or "string" in fn:
            return ProfilingDecision(
                domain="bioinformatics",
                file_format="ppi_string",
                target_bronze_table="bronze.ppi_string",
                suggested_operator="IngestPPIOperator",
                recommended_sql=f"INGEST PPI FROM '{scan.path}' INTO bronze.ppi_string WITH (source = 'string');",
                confidence=0.92,
                reasoning=f"Detected STRING organism-protein interaction weights format in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 6. Graph Network Edge List
        if "GRAPH_EDGE_LIST" in markers or fn.endswith(".edges") or "graph" in fn:
            clean_name = os.path.splitext(scan.filename)[0].lower().replace("-", "_").replace(" ", "_")
            return ProfilingDecision(
                domain="graph_network",
                file_format="csv",
                target_bronze_table=f"bronze.{clean_name}",
                suggested_operator="BronzeIngestOperator",
                recommended_sql=f"INGEST FROM '{scan.path}' INTO bronze.{clean_name} WITH (format = 'csv', delimiter = ' ');",
                confidence=0.88,
                reasoning=f"Detected pair/triplet node adjacency edge list in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 7. Entity Resolution Benchmark Dataset
        if "ENTITY_RESOLUTION_RECORD" in markers or any(k in fn for k in ("s1g", "restaurant", "er_", "records")):
            delim = scan.dominant_delimiter or ","
            clean_name = os.path.splitext(scan.filename)[0].lower().replace("-", "_").replace(" ", "_")
            return ProfilingDecision(
                domain="entity_resolution",
                file_format="csv",
                target_bronze_table=f"bronze.{clean_name}",
                suggested_operator="BronzeIngestOperator",
                recommended_sql=f"INGEST FROM '{scan.path}' INTO bronze.{clean_name} WITH (format = 'csv', delimiter = '{delim}');",
                confidence=0.91,
                reasoning=f"Detected Entity Resolution benchmark schema with record identifiers in '{scan.filename}'.",
                provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
            )

        # 8. Generic Tabular
        delim = scan.dominant_delimiter or ","
        clean_name = os.path.splitext(scan.filename)[0].lower().replace("-", "_").replace(" ", "_")
        return ProfilingDecision(
            domain="generic_tabular",
            file_format="csv" if delim == "," else "tsv",
            target_bronze_table=f"bronze.{clean_name}",
            suggested_operator="BronzeIngestOperator",
            recommended_sql=f"INGEST FROM '{scan.path}' INTO bronze.{clean_name} WITH (format = 'csv', delimiter = '{delim}');",
            confidence=0.80,
            reasoning=f"Detected structured tabular data with delimiter {repr(delim)} in '{scan.filename}'.",
            provider="heuristic-fallback (set GEMINI_API_KEY to activate cloud LLM)",
        )
