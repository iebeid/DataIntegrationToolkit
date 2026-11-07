import os
import gzip
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from toolkit.catalog.table import TableMetadata
from toolkit.operators.base import BaseOperator


class FastaIngestOperator(BaseOperator):
    """
    Bioinformatics FASTA Ingest Operator.
    Streams and samples nucleotide / protein sequences from FASTA (.fasta, .fa, .faa, .fna, .gz)
    directly into Parquet lakehouse tables with metadata extraction and reservoir sampling.
    """

    def __init__(
        self,
        source_uri: str,
        target_name: str,
        sample: Optional[int] = None,
        limit: Optional[int] = None,
        random_seed: int = 42,
        target_ids: Optional[Union[Set[str], List[str], str]] = None,
        chunk_size: int = 50000,
        clean_sequences: bool = False,
        remove_non_standard: bool = False,
    ):
        super().__init__(name="FastaIngestOperator")
        self.source_uri = source_uri
        self.target_name = target_name
        self.sample = sample
        self.limit = limit
        self.random_seed = random_seed
        self.chunk_size = chunk_size
        self.clean_sequences = clean_sequences
        self.remove_non_standard = remove_non_standard

        if isinstance(target_ids, str):
            self.target_ids = {i.strip() for i in target_ids.split(",") if i.strip()}
        elif isinstance(target_ids, (list, set)):
            self.target_ids = set(target_ids)
        else:
            self.target_ids = None

    def _parse_fasta_stream(self, handle):
        """Generator yielding (sequence_id, header, description, sequence, tax_id, tax_name, cluster_size, rep_id)."""
        import re
        current_id = None
        current_header = None
        current_desc = ""
        current_tax_id = ""
        current_tax_name = ""
        current_cluster_size = 1
        current_rep_id = ""
        seq_chunks = []

        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    raw_seq = "".join(seq_chunks).upper()
                    if self.clean_sequences:
                        raw_seq = re.sub(r"[^A-Za-z]", "", raw_seq)
                        if self.remove_non_standard:
                            # Standard 20: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
                            # Replace ambiguous: B->D, Z->E, J->L, X->A, U->C, O->K
                            trans = str.maketrans({"B": "D", "Z": "E", "J": "L", "X": "A", "U": "C", "O": "K"})
                            raw_seq = raw_seq.translate(trans)
                    yield (current_id, current_header, current_desc, raw_seq, current_tax_id, current_tax_name, current_cluster_size, current_rep_id)

                raw_header = line[1:].strip()
                current_header = raw_header
                parts = raw_header.split(maxsplit=1)
                id_token = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""

                # Extract embedded tags
                m_tax_id = re.search(r"TaxID=(\d+)", raw_header)
                current_tax_id = m_tax_id.group(1) if m_tax_id else ""

                m_tax = re.search(r"Tax=([^\s]+(?:\s+[^\s=]+)*?)(?=\s+\w+=|$)", raw_header)
                current_tax_name = m_tax.group(1) if m_tax else ""

                m_size = re.search(r"\bn=(\d+)", raw_header)
                current_cluster_size = int(m_size.group(1)) if m_size else 1

                m_rep = re.search(r"RepID=([^\s]+)", raw_header)
                current_rep_id = m_rep.group(1) if m_rep else ""

                # Parse accession / identifier
                if "|" in id_token:
                    pipe_parts = id_token.split("|")
                    if len(pipe_parts) >= 2 and pipe_parts[0] in ["sp", "tr", "gi", "ref"]:
                        current_id = pipe_parts[1]
                    else:
                        current_id = pipe_parts[1] if len(pipe_parts) > 1 else id_token
                elif "_" in id_token and id_token.startswith(("UniRef50_", "UniRef90_", "UniRef100_")):
                    current_id = id_token.split("_", 1)[1]
                else:
                    current_id = id_token
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if current_id is not None:
            raw_seq = "".join(seq_chunks).upper()
            if self.clean_sequences:
                raw_seq = re.sub(r"[^A-Za-z]", "", raw_seq)
                if self.remove_non_standard:
                    trans = str.maketrans({"B": "D", "Z": "E", "J": "L", "X": "A", "U": "C", "O": "K"})
                    raw_seq = raw_seq.translate(trans)
            yield (current_id, current_header, current_desc, raw_seq, current_tax_id, current_tax_name, current_cluster_size, current_rep_id)

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="bronze")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        fs, norm_path = context.storage.get_fs_and_path(self.source_uri)
        is_gz = self.source_uri.endswith(".gz")

        records: List[Dict[str, Any]] = []
        ingest_time = int(time.time() * 1000)

        # Handle Reservoir Sampling vs Target Filtering vs Full Scan
        if self.sample is not None and self.sample > 0:
            random.seed(self.random_seed)
            reservoir: List[Tuple[str, str, str, str]] = []
            total_seen = 0

            with fs.open(norm_path, "rb" if is_gz else "rt", encoding=None if is_gz else "utf-8") as raw_f:
                f = gzip.open(raw_f, "rt", encoding="utf-8") if is_gz else raw_f
                for item in self._parse_fasta_stream(f):
                    seq_id, hdr, desc, seq, tax_id, tax_name, c_size, rep_id = item
                    if self.target_ids and seq_id not in self.target_ids:
                        continue

                    if total_seen < self.sample:
                        reservoir.append(item)
                    else:
                        j = random.randint(0, total_seen)
                        if j < self.sample:
                            reservoir[j] = item
                    total_seen += 1

            for row_id, (seq_id, hdr, desc, seq, tax_id, tax_name, c_size, rep_id) in enumerate(reservoir):
                records.append({
                    "sequence_id": seq_id,
                    "header": hdr,
                    "description": desc,
                    "sequence": seq,
                    "length": len(seq),
                    "taxonomy_id": tax_id,
                    "taxonomy_name": tax_name,
                    "cluster_size": c_size,
                    "representative_id": rep_id,
                    "_lakehouse_ingest_time": ingest_time,
                    "_source_uri": self.source_uri,
                    "_row_id": row_id,
                })
        else:
            row_id = 0
            found_targets = 0
            num_targets = len(self.target_ids) if self.target_ids else 0

            with fs.open(norm_path, "rb" if is_gz else "rt", encoding=None if is_gz else "utf-8") as raw_f:
                f = gzip.open(raw_f, "rt", encoding="utf-8") if is_gz else raw_f
                for seq_id, hdr, desc, seq, tax_id, tax_name, c_size, rep_id in self._parse_fasta_stream(f):
                    if self.target_ids:
                        if seq_id not in self.target_ids:
                            continue
                        found_targets += 1

                    records.append({
                        "sequence_id": seq_id,
                        "header": hdr,
                        "description": desc,
                        "sequence": seq,
                        "length": len(seq),
                        "taxonomy_id": tax_id,
                        "taxonomy_name": tax_name,
                        "cluster_size": c_size,
                        "representative_id": rep_id,
                        "_lakehouse_ingest_time": ingest_time,
                        "_source_uri": self.source_uri,
                        "_row_id": row_id,
                    })
                    row_id += 1

                    if self.limit and row_id >= self.limit:
                        break
                    if self.target_ids and found_targets == num_targets:
                        break

        schema = pa.schema([
            ("sequence_id", pa.string()),
            ("header", pa.string()),
            ("description", pa.string()),
            ("sequence", pa.string()),
            ("length", pa.int64()),
            ("taxonomy_id", pa.string()),
            ("taxonomy_name", pa.string()),
            ("cluster_size", pa.int64()),
            ("representative_id", pa.string()),
            ("_lakehouse_ingest_time", pa.int64()),
            ("_source_uri", pa.string()),
            ("_row_id", pa.int64()),
        ])

        if not records:
            arrow_table = pa.Table.from_arrays(
                [pa.array([], type=f.type) for f in schema],
                schema=schema,
            )
        else:
            cols = {f.name: [r[f.name] for r in records] for f in schema}
            arrow_table = pa.Table.from_pydict(cols, schema=schema)

        context.storage.write_tabular(arrow_table, target_path, file_format="parquet")

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=len(arrow_table),
            properties={
                "source_uri": self.source_uri,
                "format": "fasta",
                "sample_size": len(arrow_table),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class IDMappingIngestOperator(BaseOperator):
    """
    Ingests UniProt idmapping.dat or TSV ID mapping files into Lakehouse Parquet tables.
    Format: uniprot_ac \t id_type \t external_id
    """

    def __init__(
        self,
        source_uri: str,
        target_name: str,
        id_types: Optional[Union[List[str], Set[str], str]] = None,
        delimiter: str = "\t",
        has_header: bool = False,
    ):
        super().__init__(name="IDMappingIngestOperator")
        self.source_uri = source_uri
        self.target_name = target_name
        self.delimiter = delimiter
        self.has_header = has_header

        if isinstance(id_types, str):
            self.id_types = {t.strip() for t in id_types.split(",") if t.strip()}
        elif isinstance(id_types, (list, set)):
            self.id_types = set(id_types)
        else:
            self.id_types = None

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        fs, norm_path = context.storage.get_fs_and_path(self.source_uri)
        is_gz = self.source_uri.endswith(".gz")

        uniprot_acs = []
        id_types_list = []
        external_ids = []
        row_ids = []
        ingest_time = int(time.time() * 1000)

        with fs.open(norm_path, "rb" if is_gz else "rt", encoding=None if is_gz else "utf-8") as raw_f:
            f = gzip.open(raw_f, "rt", encoding="utf-8") if is_gz else raw_f
            first_line = True
            row_id = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue
                if first_line and self.has_header:
                    first_line = False
                    continue
                first_line = False

                parts = line.split(self.delimiter)
                if len(parts) < 3:
                    continue

                ac = parts[0].strip()
                id_type = parts[1].strip()
                ext_id = parts[2].strip()

                if self.id_types and id_type not in self.id_types:
                    continue

                uniprot_acs.append(ac)
                id_types_list.append(id_type)
                external_ids.append(ext_id)
                row_ids.append(row_id)
                row_id += 1

        schema = pa.schema([
            ("uniprot_ac", pa.string()),
            ("id_type", pa.string()),
            ("external_id", pa.string()),
            ("_lakehouse_ingest_time", pa.int64()),
            ("_source_uri", pa.string()),
            ("_row_id", pa.int64()),
        ])

        n_rows = len(uniprot_acs)
        times = [ingest_time] * n_rows
        uris = [self.source_uri] * n_rows

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(uniprot_acs, type=pa.string()),
                pa.array(id_types_list, type=pa.string()),
                pa.array(external_ids, type=pa.string()),
                pa.array(times, type=pa.int64()),
                pa.array(uris, type=pa.string()),
                pa.array(row_ids, type=pa.int64()),
            ],
            schema=schema,
        )

        context.storage.write_tabular(arrow_table, target_path, file_format="parquet")

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=n_rows,
            properties={"source_uri": self.source_uri, "id_types_filter": list(self.id_types) if self.id_types else "all"},
        )
        context.catalog.register_table(metadata)
        return metadata


class IDMapLookupOperator(BaseOperator):
    """
    Translates cross-database identifiers in a source table using an ID mapping catalog table.
    Supports Gene_Name, BioGRID, STRING, Ensembl, RefSeq, UniProt AC translations.
    """

    def __init__(
        self,
        source_table: str,
        mapping_table: str,
        target_name: str,
        src_col: str,
        target_col: str = "uniprot_ac",
        id_type: Optional[str] = None,
        join_type: str = "inner",
    ):
        super().__init__(name="IDMapLookupOperator")
        self.source_table = source_table
        self.mapping_table = mapping_table
        self.target_name = target_name
        self.src_col = src_col
        self.target_col = target_col
        self.id_type = id_type
        self.join_type = join_type.lower()

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        src_tbl = context.read_table(self.source_table)
        map_tbl = context.read_table(self.mapping_table)

        if self.src_col not in src_tbl.column_names:
            raise KeyError(f"Source column '{self.src_col}' not found in '{self.source_table}'.")

        # Filter mapping table by id_type if specified
        if self.id_type and "id_type" in map_tbl.column_names:
            mask = pc.equal(map_tbl["id_type"], self.id_type)
            map_filtered = map_tbl.filter(mask)
        else:
            map_filtered = map_tbl

        # Build in-memory lookup dictionary (ext_id -> set of uniprot_ac)
        ext_arr = map_filtered["external_id"].to_pylist()
        ac_arr = map_filtered["uniprot_ac"].to_pylist()
        lookup: Dict[str, str] = {}
        for ext, ac in zip(ext_arr, ac_arr):
            if ext and ac:
                lookup[str(ext)] = str(ac)

        src_dict = src_tbl.to_pydict()
        src_values = [str(x) if x is not None else "" for x in src_dict[self.src_col]]

        mapped_values: List[Optional[str]] = []
        keep_mask: List[bool] = []

        for val in src_values:
            target_val = lookup.get(val)
            if target_val is not None:
                mapped_values.append(target_val)
                keep_mask.append(True)
            else:
                mapped_values.append(None if self.join_type == "left" else "")
                keep_mask.append(self.join_type == "left")

        # Apply join filtering
        out_dict = {}
        for col_name, col_data in src_dict.items():
            if self.join_type == "left":
                out_dict[col_name] = col_data
            else:
                out_dict[col_name] = [val for val, keep in zip(col_data, keep_mask) if keep]

        if self.join_type == "left":
            out_dict[self.target_col] = mapped_values
        else:
            out_dict[self.target_col] = [val for val, keep in zip(mapped_values, keep_mask) if keep]

        out_table = pa.Table.from_pydict(out_dict)
        context.storage.write_tabular(out_table, target_path, file_format="parquet")

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=out_table.schema,
            row_count=len(out_table),
            properties={
                "source_table": self.source_table,
                "mapping_table": self.mapping_table,
                "id_type": self.id_type or "all",
                "join_type": self.join_type,
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class GraphCoarseningOperator(BaseOperator):
    """
    Graph Coarsening Operator.
    Synthesizes coarsener.py into a high-performance Graclus / Heavy-Edge Matching (HEM)
    graph compression engine that reduces node and edge counts while preserving structural topology.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        src_col: str = "src",
        dst_col: str = "dst",
        weight_col: Optional[str] = "weight",
        levels: int = 1,
        method: str = "graclus",
        mapping_table: Optional[str] = None,
    ):
        super().__init__(name="GraphCoarseningOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.src_col = src_col
        self.dst_col = dst_col
        self.weight_col = weight_col
        self.levels = max(1, levels)
        self.method = method.lower()
        self.mapping_table = mapping_table

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        edge_table = context.read_table(self.source_table)
        src_data = [str(x) for x in edge_table[self.src_col].to_pylist()]
        dst_data = [str(x) for x in edge_table[self.dst_col].to_pylist()]

        if self.weight_col and self.weight_col in edge_table.column_names:
            w_raw = edge_table[self.weight_col].to_pylist()
            weights = [float(x) if x is not None else 1.0 for x in w_raw]
        else:
            weights = [1.0] * len(src_data)

        # Build initial unique nodes
        all_nodes = sorted(list(set(src_data) | set(dst_data)))
        node_to_cluster = {node: node for node in all_nodes}

        curr_src = list(src_data)
        curr_dst = list(dst_data)
        curr_w = list(weights)

        orig_nodes = len(all_nodes)
        orig_edges = len(curr_src)

        for lvl in range(self.levels):
            # Adjacency and degree accumulation
            adj: Dict[str, Dict[str, float]] = {}
            degrees: Dict[str, float] = {}

            for u, v, w in zip(curr_src, curr_dst, curr_w):
                if u == v:
                    continue
                if u not in adj:
                    adj[u] = {}
                    degrees[u] = 0.0
                if v not in adj:
                    adj[v] = {}
                    degrees[v] = 0.0

                adj[u][v] = adj[u].get(v, 0.0) + w
                adj[v][u] = adj[v].get(u, 0.0) + w
                degrees[u] += w
                degrees[v] += w

            nodes_in_level = list(adj.keys())
            matched = set()
            level_cluster_map: Dict[str, str] = {}
            cluster_id_counter = 0

            # Deterministic sorting (descending degree)
            nodes_in_level.sort(key=lambda n: degrees.get(n, 0.0), reverse=True)

            for u in nodes_in_level:
                if u in matched:
                    continue

                best_neighbor = None
                best_score = -1.0

                deg_u = max(degrees.get(u, 1.0), 1e-9)

                for v, w in adj[u].items():
                    if v in matched or v == u:
                        continue
                    deg_v = max(degrees.get(v, 1.0), 1e-9)

                    if self.method == "graclus":
                        score = (w / deg_u) + (w / deg_v)
                    else:  # heavy edge matching
                        score = w

                    if score > best_score:
                        best_score = score
                        best_neighbor = v

                c_name = f"c_{lvl}_{cluster_id_counter}"
                cluster_id_counter += 1

                level_cluster_map[u] = c_name
                matched.add(u)

                if best_neighbor is not None:
                    level_cluster_map[best_neighbor] = c_name
                    matched.add(best_neighbor)

            # Map any isolated nodes that had no edges in this level
            for u in set(curr_src) | set(curr_dst):
                if u not in level_cluster_map:
                    level_cluster_map[u] = f"c_{lvl}_{cluster_id_counter}"
                    cluster_id_counter += 1

            # Update overall node_to_cluster mapping
            for orig_node, curr_c in node_to_cluster.items():
                node_to_cluster[orig_node] = level_cluster_map.get(curr_c, curr_c)

            # Contract edges
            edge_agg: Dict[Tuple[str, str], float] = {}
            for u, v, w in zip(curr_src, curr_dst, curr_w):
                cu = level_cluster_map.get(u, u)
                cv = level_cluster_map.get(v, v)
                if cu == cv:
                    continue
                pair = (min(cu, cv), max(cu, cv))
                edge_agg[pair] = edge_agg.get(pair, 0.0) + w

            curr_src = [pair[0] for pair in edge_agg.keys()]
            curr_dst = [pair[1] for pair in edge_agg.keys()]
            curr_w = list(edge_agg.values())

        # Construct coarsened edge table
        coarsened_nodes = len(set(curr_src) | set(curr_dst))
        coarsened_edges = len(curr_src)

        coarse_table = pa.Table.from_pydict({
            self.src_col: curr_src,
            self.dst_col: curr_dst,
            self.weight_col or "weight": curr_w,
        })
        context.storage.write_tabular(coarse_table, target_path, file_format="parquet")

        node_rr = 1.0 - (coarsened_nodes / max(orig_nodes, 1))
        edge_rr = 1.0 - (coarsened_edges / max(orig_edges, 1))

        # Optionally persist mapping table
        if self.mapping_table:
            m_tier, m_tbl = context.catalog.parse_table_name(self.mapping_table, default_tier="silver")
            m_path = os.path.join(context.lakehouse_root, m_tier, f"{m_tbl}.parquet").replace("\\", "/")
            map_arrow = pa.Table.from_pydict({
                "node_id": list(node_to_cluster.keys()),
                "cluster_id": list(node_to_cluster.values()),
                "levels": [self.levels] * len(node_to_cluster),
            })
            context.storage.write_tabular(map_arrow, m_path, file_format="parquet")
            m_meta = TableMetadata(
                name=m_tbl,
                tier=m_tier,
                uri=m_path,
                schema=map_arrow.schema,
                row_count=len(map_arrow),
                properties={"source_graph": self.source_table, "coarsening_levels": self.levels},
            )
            context.catalog.register_table(m_meta)

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=coarse_table.schema,
            row_count=len(coarse_table),
            properties={
                "source_table": self.source_table,
                "method": self.method,
                "levels": self.levels,
                "orig_nodes": orig_nodes,
                "coarsened_nodes": coarsened_nodes,
                "node_reduction_ratio": round(node_rr, 4),
                "orig_edges": orig_edges,
                "coarsened_edges": coarsened_edges,
                "edge_reduction_ratio": round(edge_rr, 4),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class PPIIngestOperator(BaseOperator):
    """
    Protein-Protein Interaction (PPI) Ground Truth Ingestion Operator.
    Ingests BioGRID TAB3 or STRING network files, optionally resolves symbols/synonyms
    to UniProt ACs using a Lakehouse mapping table, and outputs canonical interaction graphs.
    """

    def __init__(
        self,
        source_uri: str,
        target_name: str,
        source_format: str = "biogrid",
        experimental_system: Optional[str] = "physical",
        min_score: float = 400.0,
        mapping_table: Optional[str] = None,
        delimiter: Optional[str] = None,
    ):
        super().__init__(name="PPIIngestOperator")
        self.source_uri = source_uri
        self.target_name = target_name
        self.source_format = source_format.lower()
        self.experimental_system = experimental_system.lower() if experimental_system else None
        self.min_score = min_score
        self.mapping_table = mapping_table
        self.delimiter = delimiter or ("\t" if self.source_format == "biogrid" else " ")

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="bronze")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        fs, norm_path = context.storage.get_fs_and_path(self.source_uri)
        is_gz = self.source_uri.endswith(".gz")

        # Load ID mapping if provided
        id_map: Dict[str, Set[str]] = {}
        if self.mapping_table:
            m_tbl = context.read_table(self.mapping_table)
            ext_col = "external_id" if "external_id" in m_tbl.column_names else m_tbl.column_names[1]
            target_col = "uniprot_ac" if "uniprot_ac" in m_tbl.column_names else m_tbl.column_names[0]
            for ext, ac in zip(m_tbl[ext_col].to_pylist(), m_tbl[target_col].to_pylist()):
                if ext and ac:
                    s_ext = str(ext).strip()
                    s_ac = str(ac).strip()
                    if s_ext not in id_map:
                        id_map[s_ext] = set()
                    id_map[s_ext].add(s_ac)

        interactions: Dict[Tuple[str, str], Dict[str, Any]] = {}

        with fs.open(norm_path, "rb" if is_gz else "rt", encoding=None if is_gz else "utf-8") as raw_f:
            f = gzip.open(raw_f, "rt", encoding="utf-8") if is_gz else raw_f

            if self.source_format == "biogrid":
                header_indices = {}
                first_line = True

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        line = line[1:].strip()

                    parts = line.split("\t")
                    if first_line:
                        first_line = False
                        # Try to detect column names
                        header_indices = {col.strip(): i for i, col in enumerate(parts)}
                        # If first line was header, proceed to next
                        if "Official Symbol Interactor A" in header_indices or "BioGRID ID Interactor A" in header_indices:
                            continue

                    idx_a = header_indices.get("Official Symbol Interactor A", 7)
                    idx_b = header_indices.get("Official Symbol Interactor B", 8)
                    idx_syn_a = header_indices.get("Synonyms Interactor A", 9)
                    idx_syn_b = header_indices.get("Synonyms Interactor B", 10)
                    idx_sys = header_indices.get("Experimental System Type", header_indices.get("Experimental System", 11))

                    if len(parts) <= max(idx_a, idx_b):
                        continue

                    raw_sys = parts[idx_sys].strip().lower() if len(parts) > idx_sys else "physical"
                    # Determine physical vs genetic
                    physical_terms = {"physical", "two-hybrid", "affinity capture-ms", "reconstituted complex", "co-crystal structure", "fret"}
                    genetic_terms = {"genetic", "synthetic lethality", "dosage lethality", "phenotypic suppression"}

                    if raw_sys in physical_terms:
                        exp_type = "physical"
                    elif raw_sys in genetic_terms:
                        exp_type = "genetic"
                    else:
                        exp_type = raw_sys

                    if self.experimental_system and self.experimental_system != "all":
                        if exp_type != self.experimental_system and raw_sys != self.experimental_system:
                            continue

                    sym_a = parts[idx_a].strip()
                    sym_b = parts[idx_b].strip()
                    syn_a = parts[idx_syn_a].strip().split("|") if len(parts) > idx_syn_a and parts[idx_syn_a] != "-" else []
                    syn_b = parts[idx_syn_b].strip().split("|") if len(parts) > idx_syn_b and parts[idx_syn_b] != "-" else []

                    ids_a = {sym_a} | {s.strip() for s in syn_a if s.strip()}
                    ids_b = {sym_b} | {s.strip() for s in syn_b if s.strip()}

                    if id_map:
                        resolved_a = {up for item in ids_a if item in id_map for up in id_map[item]}
                        resolved_b = {up for item in ids_b if item in id_map for up in id_map[item]}
                    else:
                        resolved_a = {sym_a}
                        resolved_b = {sym_b}

                    for p_a in resolved_a:
                        for p_b in resolved_b:
                            if p_a != p_b:
                                pair = (min(p_a, p_b), max(p_a, p_b))
                                if pair not in interactions:
                                    interactions[pair] = {"weight": 1.0, "system": exp_type, "source": "biogrid"}
                                else:
                                    interactions[pair]["weight"] += 1.0

            elif self.source_format == "string":
                first_line = True
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    if first_line and not parts[2].replace(".", "", 1).isdigit():
                        first_line = False
                        continue
                    first_line = False

                    p_a = parts[0].strip()
                    p_b = parts[1].strip()
                    try:
                        score = float(parts[2].strip())
                    except ValueError:
                        continue

                    if score < self.min_score:
                        continue

                    # Strip taxon e.g. 9606.ENSP... -> ENSP...
                    if "." in p_a and p_a.split(".", 1)[0].isdigit():
                        p_a = p_a.split(".", 1)[1]
                    if "." in p_b and p_b.split(".", 1)[0].isdigit():
                        p_b = p_b.split(".", 1)[1]

                    if id_map:
                        resolved_a = id_map.get(p_a, {p_a})
                        resolved_b = id_map.get(p_b, {p_b})
                    else:
                        resolved_a = {p_a}
                        resolved_b = {p_b}

                    for u_a in resolved_a:
                        for u_b in resolved_b:
                            if u_a != u_b:
                                pair = (min(u_a, u_b), max(u_a, u_b))
                                if pair not in interactions or score > interactions[pair]["weight"]:
                                    interactions[pair] = {"weight": score, "system": "physical", "source": "string"}

        interactors_a = [pair[0] for pair in interactions.keys()]
        interactors_b = [pair[1] for pair in interactions.keys()]
        weights = [data["weight"] for data in interactions.values()]
        systems = [data["system"] for data in interactions.values()]
        sources = [data["source"] for data in interactions.values()]

        schema = pa.schema([
            ("interactor_a", pa.string()),
            ("interactor_b", pa.string()),
            ("weight", pa.float64()),
            ("experimental_system", pa.string()),
            ("source", pa.string()),
        ])

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(interactors_a, type=pa.string()),
                pa.array(interactors_b, type=pa.string()),
                pa.array(weights, type=pa.float64()),
                pa.array(systems, type=pa.string()),
                pa.array(sources, type=pa.string()),
            ],
            schema=schema,
        )

        context.storage.write_tabular(arrow_table, target_path, file_format="parquet")

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=len(arrow_table),
            properties={
                "source_uri": self.source_uri,
                "source_format": self.source_format,
                "interactions_count": len(arrow_table),
                "experimental_system": self.experimental_system or "all",
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class SequenceDeduplicationOperator(BaseOperator):
    """
    Silver-tier Sequence Entity Resolution and Deduplication Operator.
    Identifies duplicate protein sequences, collapses redundant records,
    aggregates associated identifiers, and preserves cluster and taxonomy metadata.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        sequence_column: str = "sequence",
        id_column: str = "sequence_id",
    ):
        super().__init__(name="SequenceDeduplicationOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.sequence_column = sequence_column
        self.id_column = id_column

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        src_tbl = context.read_table(self.source_table)
        pydict = src_tbl.to_pydict()

        seqs = pydict[self.sequence_column]
        ids = pydict[self.id_column]
        tax_ids = pydict.get("taxonomy_id", [""] * len(seqs))
        tax_names = pydict.get("taxonomy_name", [""] * len(seqs))
        cluster_sizes = pydict.get("cluster_size", [1] * len(seqs))

        grouped: Dict[str, Dict[str, Any]] = {}
        for i in range(len(seqs)):
            s = seqs[i]
            if not s:
                continue
            if s not in grouped:
                grouped[s] = {
                    "canonical_id": ids[i],
                    "sequence": s,
                    "length": len(s),
                    "duplicate_count": 1,
                    "associated_ids": [ids[i]],
                    "taxonomy_id": tax_ids[i] if i < len(tax_ids) else "",
                    "taxonomy_name": tax_names[i] if i < len(tax_names) else "",
                    "cluster_size": cluster_sizes[i] if i < len(cluster_sizes) else 1,
                }
            else:
                grouped[s]["duplicate_count"] += 1
                grouped[s]["associated_ids"].append(ids[i])
                grouped[s]["cluster_size"] += cluster_sizes[i] if i < len(cluster_sizes) else 1

        out_records = list(grouped.values())
        schema = pa.schema([
            ("canonical_id", pa.string()),
            ("sequence", pa.string()),
            ("length", pa.int64()),
            ("duplicate_count", pa.int64()),
            ("associated_ids_count", pa.int64()),
            ("associated_ids", pa.string()),
            ("taxonomy_id", pa.string()),
            ("taxonomy_name", pa.string()),
            ("total_cluster_size", pa.int64()),
        ])

        arrow_tbl = pa.Table.from_arrays(
            [
                pa.array([r["canonical_id"] for r in out_records], type=pa.string()),
                pa.array([r["sequence"] for r in out_records], type=pa.string()),
                pa.array([r["length"] for r in out_records], type=pa.int64()),
                pa.array([r["duplicate_count"] for r in out_records], type=pa.int64()),
                pa.array([len(r["associated_ids"]) for r in out_records], type=pa.int64()),
                pa.array([",".join(r["associated_ids"][:50]) for r in out_records], type=pa.string()),
                pa.array([r["taxonomy_id"] for r in out_records], type=pa.string()),
                pa.array([r["taxonomy_name"] for r in out_records], type=pa.string()),
                pa.array([r["cluster_size"] for r in out_records], type=pa.int64()),
            ],
            schema=schema,
        )
        context.storage.write_tabular(arrow_tbl, target_path, file_format="parquet")

        orig_count = len(seqs)
        dedup_count = len(arrow_tbl)
        reduction = 1.0 - (dedup_count / max(orig_count, 1))

        meta = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=dedup_count,
            properties={
                "source_table": self.source_table,
                "original_rows": orig_count,
                "deduplicated_rows": dedup_count,
                "duplicates_removed": orig_count - dedup_count,
                "deduplication_ratio": round(reduction, 4),
            },
        )
        context.catalog.register_table(meta)
        return meta


class NgramGraphBuilderOperator(BaseOperator):
    """
    High-Performance Directed N-gram Graph Builder Operator.
    Parses biological sequences into n-gram transition graphs with frequency weights,
    out-degree transition probabilities, vocabulary indexing, and optional sparse/dense matrix exports.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        sequence_column: str = "sequence",
        id_column: str = "sequence_id",
        n: int = 3,
        directed: bool = True,
        nodes_target: Optional[str] = None,
        compute_probabilities: bool = True,
        batch_size: int = 50000,
        export_matrices_dir: Optional[str] = None,
        export_dense: bool = False,
        export_directgcn: bool = True,
    ):
        super().__init__(name="NgramGraphBuilderOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.sequence_column = sequence_column
        self.id_column = id_column
        self.n = max(1, n)
        self.directed = directed
        self.nodes_target = nodes_target
        self.compute_probabilities = compute_probabilities
        self.batch_size = batch_size
        self.export_matrices_dir = export_matrices_dir
        self.export_dense = export_dense
        self.export_directgcn = export_directgcn

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="gold")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        src_tbl = context.read_table(self.source_table)
        if self.sequence_column not in src_tbl.column_names:
            raise KeyError(f"Sequence column '{self.sequence_column}' not found in '{self.source_table}'.")

        has_id = self.id_column in src_tbl.column_names
        sequences = src_tbl[self.sequence_column].to_pylist()
        seq_ids = src_tbl[self.id_column].to_pylist() if has_id else [f"seq_{i}" for i in range(len(sequences))]

        edge_counter: Dict[Tuple[str, str], int] = {}
        node_out_degree: Dict[str, int] = {}
        node_in_degree: Dict[str, int] = {}
        node_protein_set: Dict[str, Set[str]] = {}

        n = self.n

        for seq, p_id in zip(sequences, seq_ids):
            if not seq:
                continue
            s = str(seq).strip().upper()
            seq_len = len(s)
            if seq_len < n:
                continue

            ngrams_in_seq = [s[i : i + n] for i in range(seq_len - n + 1)]
            unique_in_seq = set(ngrams_in_seq)

            for ng in unique_in_seq:
                if ng not in node_protein_set:
                    node_protein_set[ng] = set()
                node_protein_set[ng].add(str(p_id))

            if len(ngrams_in_seq) >= 2:
                for i in range(len(ngrams_in_seq) - 1):
                    u = ngrams_in_seq[i]
                    v = ngrams_in_seq[i + 1]

                    pair = (u, v)
                    edge_counter[pair] = edge_counter.get(pair, 0) + 1
                    node_out_degree[u] = node_out_degree.get(u, 0) + 1
                    node_in_degree[v] = node_in_degree.get(v, 0) + 1

                    if not self.directed:
                        rev_pair = (v, u)
                        edge_counter[rev_pair] = edge_counter.get(rev_pair, 0) + 1
                        node_out_degree[v] = node_out_degree.get(v, 0) + 1
                        node_in_degree[u] = node_in_degree.get(u, 0) + 1

        all_nodes = sorted(list(node_protein_set.keys()))
        node_to_idx = {ng: idx for idx, ng in enumerate(all_nodes)}

        src_list: List[str] = []
        tgt_list: List[str] = []
        weights: List[int] = []
        probs: List[float] = []

        for (u, v), w in edge_counter.items():
            src_list.append(u)
            tgt_list.append(v)
            weights.append(w)
            out_d = node_out_degree.get(u, 1)
            prob = float(w) / max(out_d, 1)
            probs.append(round(prob, 6))

        edge_cols = [
            pa.array(src_list, type=pa.string()),
            pa.array(tgt_list, type=pa.string()),
            pa.array(weights, type=pa.int64()),
        ]
        edge_names = ["source", "target", "weight"]

        if self.compute_probabilities:
            edge_cols.append(pa.array(probs, type=pa.float64()))
            edge_names.append("transition_prob")

        edge_table = pa.Table.from_arrays(edge_cols, names=edge_names)
        context.storage.write_tabular(edge_table, target_path, file_format="parquet")

        # Materialize nodes table if requested
        if self.nodes_target:
            n_tier, n_tbl = context.catalog.parse_table_name(self.nodes_target, default_tier="silver")
            n_path = os.path.join(context.lakehouse_root, n_tier, f"{n_tbl}.parquet").replace("\\", "/")

            node_indices = [node_to_idx[ng] for ng in all_nodes]
            in_degs = [float(node_in_degree.get(ng, 0)) for ng in all_nodes]
            out_degs = [float(node_out_degree.get(ng, 0)) for ng in all_nodes]
            tot_degs = [in_d + out_d for in_d, out_d in zip(in_degs, out_degs)]
            prot_counts = [len(node_protein_set.get(ng, set())) for ng in all_nodes]

            nodes_table = pa.Table.from_arrays(
                [
                    pa.array(node_indices, type=pa.int64()),
                    pa.array(all_nodes, type=pa.string()),
                    pa.array(in_degs, type=pa.float64()),
                    pa.array(out_degs, type=pa.float64()),
                    pa.array(tot_degs, type=pa.float64()),
                    pa.array(prot_counts, type=pa.int64()),
                ],
                names=["node_id", "ngram", "in_degree", "out_degree", "total_degree", "protein_count"],
            )
            context.storage.write_tabular(nodes_table, n_path, file_format="parquet")
            nodes_meta = TableMetadata(
                name=n_tbl,
                tier=n_tier,
                uri=n_path,
                schema=nodes_table.schema,
                row_count=len(nodes_table),
                properties={"n": self.n, "total_unique_ngrams": len(all_nodes)},
            )
            context.catalog.register_table(nodes_meta)

        # Optional DirectGCN / SciPy sparse matrix & dense export
        if self.export_matrices_dir:
            try:
                import json
                import scipy.sparse as sp
                os.makedirs(self.export_matrices_dir, exist_ok=True)
                row_idx = [node_to_idx[u] for u in src_list]
                col_idx = [node_to_idx[v] for v in tgt_list]
                shape = (len(all_nodes), len(all_nodes))
                adj_raw = sp.coo_matrix((weights, (row_idx, col_idx)), shape=shape, dtype=np.float32).tocsr()
                sp.save_npz(os.path.join(self.export_matrices_dir, f"adj_raw_sparse_n{self.n}.npz"), adj_raw)

                # DirectGCN normalized adjacency matrices
                if self.export_directgcn and len(all_nodes) > 0:
                    eye = sp.eye(len(all_nodes), dtype=np.float32, format="csr")
                    A_tilde = (adj_raw + eye).tocsr()
                    out_deg = np.array(A_tilde.sum(axis=1)).flatten()
                    in_deg = np.array(A_tilde.sum(axis=0)).flatten()

                    with np.errstate(divide="ignore"):
                        out_inv = np.power(out_deg, -0.5)
                        out_inv[np.isinf(out_inv)] = 0.0
                        in_inv = np.power(in_deg, -0.5)
                        in_inv[np.isinf(in_inv)] = 0.0

                    D_out = sp.diags(out_inv, format="csr")
                    D_in = sp.diags(in_inv, format="csr")
                    A_in = D_out.dot(A_tilde).dot(D_in)
                    A_out = D_in.dot(A_tilde.transpose()).dot(D_out)

                    sp.save_npz(os.path.join(self.export_matrices_dir, f"adj_mathcal_A_in_n{self.n}.npz"), A_in)
                    sp.save_npz(os.path.join(self.export_matrices_dir, f"adj_mathcal_A_out_n{self.n}.npz"), A_out)

                # Dense NumPy matrix
                if self.export_dense and len(all_nodes) <= 15000:
                    np.save(os.path.join(self.export_matrices_dir, f"adj_dense_n{self.n}.npy"), adj_raw.toarray())

                # Vocabulary mapping
                with open(os.path.join(self.export_matrices_dir, f"node_vocabulary_n{self.n}.json"), "w", encoding="utf-8") as vf:
                    json.dump(node_to_idx, vf, indent=2)

            except Exception as e:
                pass

        num_nodes = len(all_nodes)
        num_edges = len(edge_table)
        density = float(num_edges) / max(num_nodes * num_nodes, 1)

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=edge_table.schema,
            row_count=num_edges,
            properties={
                "source_table": self.source_table,
                "n": self.n,
                "directed": self.directed,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": round(density, 6),
                "nodes_table": self.nodes_target or "none",
            },
        )
        context.catalog.register_table(metadata)
        return metadata
