import os
import time
from typing import Any, Dict, Optional
import pyarrow as pa
import pyarrow.compute as pc

from src.operators.base import BaseOperator
from src.catalog.table import TableMetadata


class BatchIngestOperator(BaseOperator):
    """
    Bronze-tier operator: Ingests raw data from external sources into standardized
    lakehouse Parquet storage with automated auditing metadata.
    """

    def __init__(
        self,
        name: str = "batch_ingest",
        source_uri: str = "",
        target_name: str = "",
        file_format: str = "auto",
        delimiter: str = ",",
        has_header: bool = True,
        encoding: str = "utf-8",
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name)
        self.source_uri = source_uri
        self.target_name = target_name
        self.file_format = file_format
        self.delimiter = delimiter
        self.has_header = has_header
        self.encoding = encoding
        self.options = options or {}
        self.engine = self.options.get("engine", "pyarrow")
        self.table_format = self.options.get("table_format", "parquet")

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        engine = kwargs.get("engine", self.engine)
        table_format = kwargs.get("table_format", self.table_format)

        if engine == "dask":
            from src.operators.dask_engine import DaskOutofCoreIngestOperator
            op = DaskOutofCoreIngestOperator(
                source_uri=kwargs.get("source_uri", self.source_uri),
                target_name=kwargs.get("target_name", self.target_name),
                file_format=kwargs.get("file_format", self.file_format),
                delimiter=kwargs.get("delimiter", self.delimiter),
                has_header=kwargs.get("has_header", self.has_header),
                npartitions=int(self.options.get("npartitions", 0)) if self.options.get("npartitions") else None,
                table_format=table_format,
            )
            return op.execute(context, **kwargs)

        fs_manager = context.storage
        source_uri = kwargs.get("source_uri", self.source_uri)
        target_name = kwargs.get("target_name", self.target_name)
        clean_target = target_name.replace("bronze.", "")

        # 1. Read source data
        raw_table = fs_manager.read_tabular(
            uri=source_uri,
            file_format=kwargs.get("file_format", self.file_format),
            delimiter=kwargs.get("delimiter", self.delimiter),
            has_header=kwargs.get("has_header", self.has_header),
            encoding=kwargs.get("encoding", self.encoding),
        )

        # 2. Append lakehouse audit columns
        now_ts = time.time()
        num_rows = raw_table.num_rows

        ingest_col = pa.array([now_ts] * num_rows, type=pa.float64())
        source_col = pa.array([source_uri] * num_rows, type=pa.string())
        row_id_col = pa.array(list(range(num_rows)), type=pa.int64())

        enriched_table = raw_table.append_column("_lakehouse_ingest_time", ingest_col)
        enriched_table = enriched_table.append_column("_source_uri", source_col)
        enriched_table = enriched_table.append_column("_row_id", row_id_col)

        # 3. Write Parquet or Delta Lake to Bronze lakehouse storage
        if table_format == "delta":
            target_uri = os.path.join(context.lakehouse_root, "bronze", clean_target).replace("\\", "/")
            fs_manager.write_delta(enriched_table, target_uri)
        else:
            target_uri = os.path.join(context.lakehouse_root, "bronze", f"{clean_target}.parquet").replace("\\", "/")
            fs_manager.write_parquet(enriched_table, target_uri)

        # 4. Register in Catalog
        meta = TableMetadata(
            name=clean_target,
            tier="bronze",
            uri=target_uri,
            schema=enriched_table.schema,
            row_count=num_rows,
            format=table_format,
            properties={
                "source_uri": source_uri,
                "file_format": self.file_format,
                "engine": engine,
            },
        )
        context.catalog.register_table(meta)
        return meta
