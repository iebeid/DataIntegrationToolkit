"""
AI Ingestion Router.
Coordinates dataset scanning, AI/Gemini profiling, and routes raw datasets
directly into the appropriate Lakehouse Bronze tier ingestion operator.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pyarrow as pa

from src.ai.scanner import DatasetScanner, DatasetScanResult
from src.ai.gemini_client import GeminiClient, ProfilingDecision
from src.catalog.table import TableMetadata


@dataclass
class IngestionPlan:
    """Execution plan formulated by the AI profiler for Bronze ingestion."""
    dataset_path: str
    domain: str
    file_format: str
    target_bronze_table: str
    suggested_operator: str
    recommended_sql: str
    confidence: float
    reasoning: str
    ai_provider: str
    estimated_rows: int
    execution_status: str = "PLANNED"
    executed_metadata: Optional[Any] = None

    @property
    def format_type(self) -> str:
        return self.file_format

    @property
    def recommended_operator(self) -> str:
        return self.suggested_operator

    @property
    def provider(self) -> str:
        return self.ai_provider

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "domain": self.domain,
            "file_format": self.file_format,
            "target_bronze_table": self.target_bronze_table,
            "suggested_operator": self.suggested_operator,
            "recommended_sql": self.recommended_sql,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "ai_provider": self.ai_provider,
            "estimated_rows": self.estimated_rows,
            "execution_status": self.execution_status,
            "row_count": getattr(self.executed_metadata, "row_count", None),
        }


class AIIngestionRouter:
    """
    Entry point layer for zero-assumption raw dataset profiling and
    autonomous routing to Lakehouse Bronze preparation operators.
    """

    def __init__(self, context: Optional[Any] = None, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.context = context
        self.scanner = DatasetScanner()
        self.gemini = GeminiClient(api_key=api_key, model=model)

    def scan(self, file_path: str) -> DatasetScanResult:
        """Scans dataset without prior assumptions."""
        return self.scanner.scan(file_path)

    def plan(self, file_path: str) -> IngestionPlan:
        """Profiles unknown dataset and produces an autonomous ingestion plan."""
        scan_res = self.scanner.scan(file_path)
        decision: ProfilingDecision = self.gemini.classify_and_plan(scan_res)

        return IngestionPlan(
            dataset_path=file_path,
            domain=decision.domain,
            file_format=decision.file_format,
            target_bronze_table=decision.target_bronze_table,
            suggested_operator=decision.suggested_operator,
            recommended_sql=decision.recommended_sql,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            ai_provider=decision.provider,
            estimated_rows=scan_res.estimated_row_count,
            execution_status="PLANNED",
        )

    def route_and_execute(
        self,
        file_path: str,
        target_table: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> IngestionPlan:
        """
        Scans, profiles with AI, and routes directly to the correct
        ingestion/bronze preparation operator.
        """
        ctx = context or self.context
        plan = self.plan(file_path)
        target = target_table or plan.target_bronze_table

        if ctx is None:
            return plan

        scan_res = self.scanner.scan(file_path)

        # 1. Bioinformatics FASTA Ingestion
        if plan.file_format == "fasta" or plan.suggested_operator == "IngestFastaOperator":
            meta = ctx.ingest_fasta(source_uri=file_path, target_name=target)
            plan.execution_status = "EXECUTED"
            plan.executed_metadata = meta
            return plan

        # 2. UniProt ID Mapping Ingestion
        if plan.file_format == "idmapping" or plan.suggested_operator == "IngestIDMappingOperator":
            delim = scan_res.dominant_delimiter or "\t"
            meta = ctx.ingest_idmapping(source_uri=file_path, target_name=target, delimiter=delim)
            plan.execution_status = "EXECUTED"
            plan.executed_metadata = meta
            return plan

        # 3. Protein-Protein Interaction (BioGRID or STRING)
        if "ppi" in plan.file_format or plan.suggested_operator == "IngestPPIOperator":
            source_fmt = "biogrid" if "biogrid" in plan.file_format else "string"
            delim = scan_res.dominant_delimiter
            meta = ctx.ingest_ppi(source_uri=file_path, target_name=target, source_format=source_fmt, delimiter=delim)
            plan.execution_status = "EXECUTED"
            plan.executed_metadata = meta
            return plan

        # 4. PubMed / Biomedical XML
        if plan.file_format == "xml" or "PUBMED_XML_CITATION" in scan_res.structural_markers:
            meta = self._ingest_pubmed_xml(ctx, file_path, target)
            plan.execution_status = "EXECUTED"
            plan.executed_metadata = meta
            return plan

        # 5. Default Tabular / Graph Edge List Ingestion (CSV / TSV)
        delim = scan_res.dominant_delimiter or ","
        meta = ctx.ingest(source_uri=file_path, target_name=target, file_format="csv", delimiter=delim)
        plan.execution_status = "EXECUTED"
        plan.executed_metadata = meta
        return plan

    def _ingest_pubmed_xml(self, ctx: Any, file_path: str, target_name: str) -> TableMetadata:
        """Parses PubMed XML citations and materializes into Bronze table."""
        tier, tbl = ctx.catalog.parse_table_name(target_name, default_tier="bronze")
        target_path = os.path.join(ctx.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        pmids, titles, abstracts, pub_years = [], [], [], []
        auth_pmids, auth_names = [], []
        count = 0

        for event, elem in ET.iterparse(file_path, events=("end",)):
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
                if count >= 5000:
                    break

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(pmids, type=pa.string()),
                pa.array(titles, type=pa.string()),
                pa.array(abstracts, type=pa.string()),
                pa.array(pub_years, type=pa.int64()),
            ],
            names=["pmid", "title", "abstract", "year"],
        )

        ctx.storage.write_tabular(arrow_table, target_path, file_format="parquet")
        meta = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=arrow_table.schema,
            row_count=len(arrow_table),
            properties={"source_uri": file_path, "format": "xml"},
        )
        ctx.catalog.register_table(meta)

        # Companion authorship table for knowledge graph construction if authors found
        if auth_pmids:
            auth_tbl_name = f"{tbl.replace('_articles', '')}_authors" if "_articles" in tbl else "pubmed_authors"
            auth_path = os.path.join(ctx.lakehouse_root, tier, f"{auth_tbl_name}.parquet").replace("\\", "/")
            auth_arrow = pa.Table.from_arrays(
                [
                    pa.array(auth_pmids, type=pa.string()),
                    pa.array(auth_names, type=pa.string()),
                ],
                names=["pmid", "author_id"],
            )
            ctx.storage.write_tabular(auth_arrow, auth_path, file_format="parquet")
            ctx.catalog.register_table(
                TableMetadata(
                    name=auth_tbl_name,
                    tier=tier,
                    uri=auth_path,
                    schema=auth_arrow.schema,
                    row_count=len(auth_arrow),
                    properties={"source_uri": file_path, "companion_to": tbl},
                )
            )

        return meta
