"""
Zero-Assumption Dataset Scanner and Content Inspector.
Inspects arbitrary raw files without prior assumptions, extracting structural
samples, token distributions, delimiter signatures, and format markers.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetScanResult:
    """Contains low-level physical and structural profile of an unknown dataset."""
    path: str
    filename: str
    file_size_bytes: int
    sample_text: str
    sample_lines: List[str] = field(default_factory=list)
    detected_encoding: str = "utf-8"
    dominant_delimiter: Optional[str] = None
    delimiter_counts: Dict[str, int] = field(default_factory=dict)
    structural_markers: List[str] = field(default_factory=list)
    estimated_row_count: int = 0
    candidate_headers: Optional[List[str]] = None


class DatasetScanner:
    """
    Scans an arbitrary file assuming zero prior knowledge.
    Extracts lightweight physical samples and structural signatures.
    """

    def __init__(self, max_sample_lines: int = 50, max_sample_bytes: int = 32768):
        self.max_sample_lines = max_sample_lines
        self.max_sample_bytes = max_sample_bytes

    def scan(self, file_path: str) -> DatasetScanResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset path not found: {file_path}")

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)

        # Detect encoding and read sample bytes
        encoding = "utf-8"
        raw_bytes = b""
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read(self.max_sample_bytes)
        except Exception as e:
            raw_bytes = b""

        # Try decoding
        for enc in ("utf-8", "latin-1", "ascii"):
            try:
                sample_text = raw_bytes.decode(enc)
                encoding = enc
                break
            except UnicodeDecodeError:
                continue
        else:
            sample_text = raw_bytes.decode("utf-8", errors="replace")

        lines = [line.strip() for line in sample_text.splitlines() if line.strip()]
        sample_lines = lines[: self.max_sample_lines]

        # Estimate row count
        if sample_lines and file_size > 0:
            avg_line_len = max(1, sum(len(l) for l in sample_lines) // len(sample_lines))
            estimated_rows = max(len(sample_lines), file_size // avg_line_len)
        else:
            estimated_rows = len(sample_lines)

        # Delimiter analysis on non-empty lines
        delimiter_counts = {"\t": 0, ",": 0, "|": 0, ";": 0, " ": 0}
        for l in sample_lines:
            for delim in delimiter_counts.keys():
                delimiter_counts[delim] += l.count(delim)

        # Filter out space if other delimiters are strongly present
        delims_no_space = {k: v for k, v in delimiter_counts.items() if k != " "}
        sorted_delims = sorted(delims_no_space.items(), key=lambda x: x[1], reverse=True)
        dominant_delimiter = sorted_delims[0][0] if sorted_delims and sorted_delims[0][1] > 0 else None

        # Structural markers identification
        markers = []

        # 1. FASTA markers
        fasta_headers = [l for l in sample_lines if l.startswith(">")]
        if fasta_headers:
            markers.append("FASTA_HEADER")
            # Check if sequence lines follow
            seq_lines = [l for l in sample_lines if not l.startswith(">") and re.match(r"^[A-Za-z* \t-]+$", l)]
            if seq_lines:
                markers.append("PROTEIN_OR_NUCLEOTIDE_SEQUENCE")
                # Detect non-standard amino acid characters
                all_chars = set("".join(seq_lines).upper())
                non_std = all_chars.intersection(set("BZXJUO*-"))
                if non_std:
                    markers.append("NON_STANDARD_AMINO_ACIDS_DETECTED")

            # Check UniRef / cluster markers
            if any("uniref" in h.lower() for h in fasta_headers):
                markers.append("UNIREF_CLUSTER_FASTA")
            if any("taxid=" in h.lower() or "tax=" in h.lower() for h in fasta_headers):
                markers.append("TAXONOMIC_LABELS_PRESENT")
            if any("repid=" in h.lower() or re.search(r"\bn=\d+", h) for h in fasta_headers):
                markers.append("UNIREF_CLUSTER_METADATA_PRESENT")

        # 2. XML markers
        if any("<PubmedArticle" in l for l in sample_lines):
            markers.append("PUBMED_XML_CITATION")
        elif any(l.startswith("<?xml") or ("<" in l and ">" in l and "/" in l) for l in sample_lines):
            markers.append("XML_MARKUP")

        # 3. ID Mapping / UniProt markers
        if any("UniProtKB" in l or "Gene_Name" in l or "Ensembl" in l for l in sample_lines):
            markers.append("UNIPROT_IDMAPPING")

        # 4. PPI Network markers
        if any("BioGRID" in l or "INTERACTOR_A" in l or "OFFICIAL_SYMBOL" in l for l in sample_lines):
            markers.append("BIOGRID_PPI")
        elif any(re.match(r"^\d+\.[A-Za-z0-9_-]+\s+\d+\.[A-Za-z0-9_-]+\s+\d+", l) for l in sample_lines):
            markers.append("STRING_PPI_NETWORK")

        # 5. Graph Edge List markers
        edge_like = [l for l in sample_lines if len(l.split()) in (2, 3) and not l.startswith("#") and not l.startswith(">")]
        if len(edge_like) >= max(3, len(sample_lines) // 2):
            markers.append("GRAPH_EDGE_LIST")

        # 6. Entity Resolution Tabular / Benchmark markers
        first_line = sample_lines[0] if sample_lines else ""
        candidate_headers = None
        if dominant_delimiter and dominant_delimiter in first_line:
            candidate_headers = [h.strip().strip('"') for h in first_line.split(dominant_delimiter)]
            if any(h.lower() in ("recid", "record", "cluster_id", "entity_id", "id") for h in candidate_headers):
                markers.append("ENTITY_RESOLUTION_RECORD")

        return DatasetScanResult(
            path=file_path,
            filename=filename,
            file_size_bytes=file_size,
            sample_text="\n".join(sample_lines[:20]),
            sample_lines=sample_lines,
            detected_encoding=encoding,
            dominant_delimiter=dominant_delimiter,
            delimiter_counts=delimiter_counts,
            structural_markers=markers,
            estimated_row_count=int(estimated_rows),
            candidate_headers=candidate_headers,
        )
