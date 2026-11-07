import os
import hashlib
import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import fsspec
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pyarrow.compute as pc


class StorageManager:
    """
    Storage-agnostic filesystem & serialization manager leveraging fsspec and pyarrow.
    Seamlessly handles local paths, S3 (s3://), GCS (gs://), memory (memory://), etc.
    """

    def __init__(self, storage_options: Optional[Dict[str, Any]] = None):
        self.storage_options = storage_options or {}

    @staticmethod
    def resolve_uri(uri: str) -> str:
        """
        Resolves local data URIs, searching data/ use-case subdirectories if needed.
        """
        if os.path.exists(uri):
            return uri
        normalized = uri.replace("\\", "/")
        if "data/" in normalized or normalized.startswith("data"):
            filename = os.path.basename(normalized)
            for root, _, files in os.walk("data"):
                if filename in files:
                    return os.path.join(root, filename).replace("\\", "/")
        return uri

    def get_fs_and_path(self, uri: str) -> Tuple[fsspec.AbstractFileSystem, str]:
        """Resolves URI into an fsspec filesystem instance and a normalized path."""
        resolved = self.resolve_uri(uri)
        fs, path = fsspec.core.url_to_fs(resolved, **self.storage_options)
        return fs, path

    def exists(self, uri: str) -> bool:
        fs, path = self.get_fs_and_path(uri)
        return fs.exists(path)

    def makedirs(self, uri: str, exist_ok: bool = True) -> None:
        fs, path = self.get_fs_and_path(uri)
        fs.makedirs(path, exist_ok=exist_ok)

    def read_tabular(
        self,
        uri: str,
        file_format: str = "auto",
        delimiter: str = ",",
        has_header: bool = True,
        encoding: str = "utf-8",
    ) -> pa.Table:
        """
        Reads tabular data from any supported filesystem directly into an Apache Arrow Table.
        """
        fs, path = self.get_fs_and_path(uri)

        # Detect Delta Lake table format
        delta_log_dir = f"{path.rstrip('/')}/_delta_log"
        if file_format == "delta" or (fs.isdir(path) and fs.exists(delta_log_dir)):
            return self.read_delta(uri)

        # Detect format if auto
        if file_format == "auto":
            ext = os.path.splitext(path)[1].lower()
            if ext in [".parquet", ".pq"]:
                file_format = "parquet"
            elif ext in [".csv", ".txt"]:
                file_format = "csv"
            elif ext in [".tsv", ".tab"]:
                file_format = "tsv"
            elif ext in [".json", ".jsonl"]:
                file_format = "json"
            else:
                file_format = "csv"

        if file_format == "parquet":
            if fs.isdir(path):
                return pq.read_table(path)
            with fs.open(path, mode="rb") as f:
                return pq.read_table(f)

        with fs.open(path, mode="rb") as f:
            if file_format in ["csv", "tsv"]:
                if delimiter in ["whitespace", "\\s+", "space"]:
                    text_content = f.read().decode(encoding)
                    lines = [line.strip().split() for line in text_content.splitlines() if line.strip()]
                    if not lines:
                        return pa.table({})
                    if has_header:
                        headers = lines[0]
                        rows = lines[1:]
                    else:
                        num_cols = len(lines[0])
                        headers = [f"f{i}" for i in range(num_cols)]
                        rows = lines

                    cols_data: Dict[str, list] = {h: [] for h in headers}
                    for row in rows:
                        for idx, h in enumerate(headers):
                            val = row[idx] if idx < len(row) else None
                            cols_data[h].append(val)

                    arrow_arrays = {}
                    for h, vals in cols_data.items():
                        try:
                            int_vals = [int(v) if v is not None else None for v in vals]
                            arrow_arrays[h] = pa.array(int_vals)
                        except (ValueError, TypeError):
                            try:
                                flt_vals = [float(v) if v is not None else None for v in vals]
                                arrow_arrays[h] = pa.array(flt_vals)
                            except (ValueError, TypeError):
                                arrow_arrays[h] = pa.array(vals)
                    return pa.Table.from_pydict(arrow_arrays)

                actual_delim = "\t" if file_format == "tsv" else delimiter
                if actual_delim == "\\t":
                    actual_delim = "\t"
                parse_opts = pcsv.ParseOptions(delimiter=actual_delim)
                read_opts = pcsv.ReadOptions(
                    autogenerate_column_names=not has_header,
                    encoding=encoding,
                )
                return pcsv.read_csv(f, read_options=read_opts, parse_options=parse_opts)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    def write_parquet(
        self,
        table: pa.Table,
        uri: str,
        compression: str = "snappy",
    ) -> str:
        """
        Writes an Apache Arrow Table to a Parquet file via fsspec.
        """
        fs, path = self.get_fs_and_path(uri)
        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent:
            fs.makedirs(parent, exist_ok=True)

        with fs.open(path, mode="wb") as f:
            pq.write_table(table, f, compression=compression)
        return uri

    def write_tabular(
        self,
        table: pa.Table,
        uri: str,
        file_format: str = "parquet",
        compression: str = "snappy",
        mode: str = "overwrite",
    ) -> str:
        """
        Writes an Apache Arrow Table to tabular format (Parquet or Delta Lake).
        """
        if file_format == "parquet":
            return self.write_parquet(table, uri, compression=compression)
        elif file_format == "delta":
            return self.write_delta(table, uri, mode=mode, compression=compression)
        raise ValueError(f"Unsupported write format: {file_format}")

    def write_delta(
        self,
        table: pa.Table,
        uri: str,
        mode: str = "overwrite",
        compression: str = "snappy",
    ) -> str:
        """
        Writes an Apache Arrow Table into Delta Lake table format with ACID transaction logs.
        """
        fs, path = self.get_fs_and_path(uri)
        if path.endswith(".parquet"):
            path = path[:-8]
        fs.makedirs(path, exist_ok=True)
        log_dir = f"{path.rstrip('/')}/_delta_log"
        fs.makedirs(log_dir, exist_ok=True)

        commit_files = []
        try:
            for item in fs.ls(log_dir):
                fname = os.path.basename(item)
                if fname.endswith(".json"):
                    commit_files.append(fname)
        except Exception:
            pass

        existing_versions = []
        for fn in commit_files:
            try:
                existing_versions.append(int(fn.replace(".json", "")))
            except ValueError:
                pass

        curr_version = max(existing_versions) + 1 if existing_versions else 0
        part_filename = f"part-{curr_version:05d}-{uuid.uuid4().hex[:8]}.parquet"
        part_path = f"{path.rstrip('/')}/{part_filename}"

        with fs.open(part_path, "wb") as f:
            pq.write_table(table, f, compression=compression)
        part_size = fs.size(part_path)
        now_ms = int(time.time() * 1000)

        actions = []
        if curr_version == 0:
            actions.append({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}})
            schema_json = [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in table.schema]
            actions.append({
                "metaData": {
                    "id": str(uuid.uuid4()),
                    "format": {"provider": "parquet", "options": {}},
                    "schemaString": json.dumps({"type": "struct", "fields": schema_json}),
                    "partitionColumns": [],
                    "configuration": {},
                    "createdTime": now_ms,
                }
            })

        if mode == "overwrite" and curr_version > 0:
            prev_table_files = self._get_active_delta_files(fs, log_dir, max_version=curr_version - 1)
            for prev_f in prev_table_files:
                actions.append({
                    "remove": {
                        "path": prev_f,
                        "deletionTimestamp": now_ms,
                        "dataChange": True,
                    }
                })

        actions.append({
            "add": {
                "path": part_filename,
                "size": part_size,
                "modificationTime": now_ms,
                "dataChange": True,
            }
        })

        commit_log_path = f"{log_dir}/{curr_version:020d}.json"
        with fs.open(commit_log_path, "wt", encoding="utf-8") as f:
            for act in actions:
                f.write(json.dumps(act) + "\n")

        return uri

    def _get_active_delta_files(self, fs: Any, log_dir: str, max_version: Optional[int] = None) -> List[str]:
        """Scans Delta commit logs up to max_version and returns list of active relative file paths."""
        commit_files = []
        try:
            for item in fs.ls(log_dir):
                fname = os.path.basename(item)
                if fname.endswith(".json"):
                    commit_files.append(fname)
        except Exception:
            return []

        commit_files.sort()
        active_files = set()

        for fn in commit_files:
            try:
                v = int(fn.replace(".json", ""))
            except ValueError:
                continue
            if max_version is not None and v > max_version:
                continue
            log_path = f"{log_dir}/{fn}"
            with fs.open(log_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        action = json.loads(line)
                    except Exception:
                        continue
                    if "add" in action:
                        active_files.add(action["add"]["path"])
                    elif "remove" in action:
                        active_files.discard(action["remove"]["path"])

        return sorted(list(active_files))

    def read_delta(self, uri: str, version: Optional[int] = None) -> pa.Table:
        """
        Reads a Delta Lake table at specified snapshot version (or latest).
        """
        fs, path = self.get_fs_and_path(uri)
        if path.endswith(".parquet"):
            path = path[:-8]
        log_dir = f"{path.rstrip('/')}/_delta_log"
        if not fs.exists(log_dir):
            raise FileNotFoundError(f"Delta log not found at '{log_dir}'.")

        active_files = self._get_active_delta_files(fs, log_dir, max_version=version)
        if not active_files:
            return pa.table({})

        tables = []
        for rel_f in active_files:
            full_f = f"{path.rstrip('/')}/{rel_f}"
            with fs.open(full_f, "rb") as f:
                tables.append(pq.read_table(f))

        return pa.concat_tables(tables) if tables else pa.table({})

    @staticmethod
    def compute_checksum(uri: str) -> str:
        """Computes SHA-256 checksum for a file."""
        fs, path = fsspec.core.url_to_fs(uri)
        hasher = hashlib.sha256()
        with fs.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
