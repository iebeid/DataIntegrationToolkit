from typing import Any, Dict, List, Optional
import pyarrow as pa
from src.sql.ast import (
    ASTNode,
    IngestStatement,
    StageStatement,
    BlockStatement,
    ResolveStatement,
    ProjectGraphStatement,
    ShowTablesStatement,
    DescribeTableStatement,
    SelectStatement,
    EvaluateStatement,
    ProfileStatement,
    CreateEmbeddingsStatement,
    TrainGNNStatement,
    IngestFastaStatement,
    IngestIDMappingStatement,
    MapIdentifiersStatement,
    CoarsenGraphStatement,
    IngestPPIStatement,
    ExtractEntitiesStatement,
    MapOntologyStatement,
    ConstructKnowledgeGraphStatement,
    ProfileGraphStatement,
    DropTableStatement,
    VacuumStatement,
    ExplainStatement,
    AIProfileStatement,
    AIRouteStatement,
)


def _parse_mapping_string(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parses key-value strings like 'fname:0.3;lname:0.4;address:0.3'."""
    if not raw or not isinstance(raw, str):
        return None
    res = {}
    for part in raw.split(";"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                res[k] = float(v)
            except ValueError:
                res[k] = v
    return res if res else None


class QueryPlanner:
    """
    Translates parsed SQL AST nodes into executable Lakehouse operators
    and executes them within the IngestionContext.
    """

    def __init__(self, context: Any):
        self.context = context

    def execute(self, node: ASTNode) -> Any:
        if isinstance(node, IngestStatement):
            opts = node.options
            return self.context.ingest(
                source_uri=node.source_uri,
                target_name=node.target_table,
                file_format=opts.get("format", "auto"),
                delimiter=opts.get("delimiter", ","),
                has_header=opts.get("has_header", True),
                encoding=opts.get("encoding", "utf-8"),
                engine=opts.get("engine", "pyarrow"),
                table_format=opts.get("table_format", "parquet"),
            )

        elif isinstance(node, StageStatement):
            opts = node.options
            return self.context.stage(
                source_table=node.source_table,
                target_name=node.target_table,
                key_column=opts.get("key", "RecID"),
                stopword_threshold=opts.get("stopword_threshold", 0.05),
                min_token_len=opts.get("min_token_len", 2),
            )

        elif isinstance(node, BlockStatement):
            opts = node.options
            return self.context.block(
                source_table=node.source_table,
                target_name=node.target_table,
                key_column=opts.get("key", "RecID"),
                method=opts.get("method", "token_inverted_index"),
                max_block_size=opts.get("max_block_size", 200),
                num_perm=opts.get("num_perm", 64),
                threshold=opts.get("threshold", 0.4),
            )

        elif isinstance(node, CreateEmbeddingsStatement):
            opts = node.options
            cols = opts.get("columns")
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",")]
            return self.context.create_embeddings(
                source_table=node.source_table,
                target_name=node.target_table,
                key_column=opts.get("key", "RecID"),
                method=opts.get("method", "tfidf"),
                dimension=int(opts.get("dimension", 128)),
                columns=cols,
                subword=opts.get("subword", False),
                max_df=float(opts.get("max_df", 0.5)),
            )

        elif isinstance(node, TrainGNNStatement):
            opts = node.options
            return self.context.train_gnn(
                source_table=node.source_table,
                graph_table=node.graph_table,
                target_name=node.target_table,
                key_column=opts.get("key", "node_id"),
                src_col=opts.get("src_col", "src"),
                dst_col=opts.get("dst_col", "dst"),
                weight_col=opts.get("weight_col", "weight"),
                model=opts.get("model", "dgcn"),
                dimension=int(opts.get("dimension", 32)),
                layers=int(opts.get("layers", 2)),
                epochs=int(opts.get("epochs", 20)),
                directed=opts.get("directed", True),
            )

        elif isinstance(node, ResolveStatement):
            opts = node.options
            weights = _parse_mapping_string(opts.get("weights"))
            attr_metrics = _parse_mapping_string(opts.get("metrics"))

            return self.context.resolve(
                records_table=node.records_table,
                pairs_table=node.pairs_table,
                target_name=node.target_table,
                key_column=opts.get("key", "RecID"),
                similarity_metric=opts.get("similarity", "jaccard"),
                threshold=opts.get("threshold", 0.5),
                clustering=opts.get("clustering", "transitive_closure"),
                attribute_weights=weights,
                attribute_metrics=attr_metrics,
            )

        elif isinstance(node, ProjectGraphStatement):
            opts = node.options
            return self.context.project_graph(
                source_table=node.source_table,
                target_name=node.target_table,
                sequence_column=opts.get("sequence_column", "sequence"),
                id_column=opts.get("id_column", "sequence_id"),
                n=int(opts.get("n", 3)),
                directed=opts.get("directed", True) if isinstance(opts.get("directed"), bool) else str(opts.get("directed", "true")).lower() == "true",
                nodes_target=opts.get("nodes_target"),
                compute_probabilities=opts.get("compute_probabilities", True) if isinstance(opts.get("compute_probabilities"), bool) else str(opts.get("compute_probabilities", "true")).lower() == "true",
                export_matrices_dir=opts.get("export_matrices_dir"),
                export_dense=str(opts.get("export_dense", "false")).lower() == "true",
                export_directgcn=str(opts.get("export_directgcn", "true")).lower() == "true",
            )

        elif isinstance(node, ShowTablesStatement):
            return self.context.show_tables(tier=node.tier)

        elif isinstance(node, DescribeTableStatement):
            meta = self.context.catalog.get_table(node.table_name)
            if not meta:
                raise ValueError(f"Table '{node.table_name}' does not exist.")
            return {
                "table": meta.qualified_name,
                "tier": meta.tier,
                "rows": meta.row_count,
                "columns": [
                    {"name": field.name, "type": str(field.type)}
                    for field in meta.schema
                ],
                "properties": meta.properties,
                "uri": meta.uri,
            }

        elif isinstance(node, SelectStatement):
            table = self.context.read_table(node.table_name)
            if node.where_clause:
                import re
                cond_m = re.match(r"(\w+)\s*(=|!=|>|<|>=|<=|LIKE)\s*(.*)", node.where_clause.strip(), re.IGNORECASE)
                if cond_m:
                    col_name = cond_m.group(1).strip()
                    op = cond_m.group(2).strip().upper()
                    raw_val = cond_m.group(3).strip()
                    if (raw_val.startswith("'") and raw_val.endswith("'")) or (raw_val.startswith('"') and raw_val.endswith('"')):
                        val = raw_val[1:-1]
                    else:
                        try:
                            val = float(raw_val) if "." in raw_val else int(raw_val)
                        except ValueError:
                            val = raw_val

                    if col_name in table.column_names:
                        col_vals = table[col_name].to_pylist()
                        keep_indices = []
                        for idx, v in enumerate(col_vals):
                            matched = False
                            if v is not None:
                                try:
                                    if op == "=":
                                        matched = (str(v) == str(val))
                                    elif op == "!=":
                                        matched = (str(v) != str(val))
                                    elif op == ">":
                                        matched = (float(v) > float(val))
                                    elif op == "<":
                                        matched = (float(v) < float(val))
                                    elif op == ">=":
                                        matched = (float(v) >= float(val))
                                    elif op == "<=":
                                        matched = (float(v) <= float(val))
                                    elif op == "LIKE":
                                        pattern = "^" + re.escape(str(val)).replace("%", ".*").replace("_", ".") + "$"
                                        matched = bool(re.match(pattern, str(v), re.IGNORECASE))
                                except (ValueError, TypeError):
                                    pass
                            if matched:
                                keep_indices.append(idx)
                        table = table.take(pa.array(keep_indices, type=pa.int64()))

            if node.order_by and node.order_by in table.column_names:
                import pyarrow.compute as pc
                order_dir = "descending" if node.order_desc else "ascending"
                sort_indices = pc.sort_indices(table[node.order_by], sort_keys=[(node.order_by, order_dir)])
                table = table.take(sort_indices)

            if node.columns != ["*"]:
                selected_cols = [c for c in node.columns if c in table.column_names]
                table = table.select(selected_cols)
            if node.limit is not None:
                table = table.slice(0, node.limit)
            return table.to_pylist()

        elif isinstance(node, EvaluateStatement):
            opts = node.options
            return self.context.evaluate(
                target_table=node.target_table,
                truth_file=node.truth_file,
                pairs_table=node.pairs_table,
                key_column=opts.get("key", "RecID"),
                cluster_column=opts.get("cluster_id", "canonical_entity_id"),
                output_table=opts.get("output_table"),
            )

        elif isinstance(node, ProfileStatement):
            return self.context.profile(target_table=node.target_table)

        elif isinstance(node, IngestFastaStatement):
            opts = node.options
            sample = opts.get("sample")
            sample_val = int(sample) if sample is not None else None
            limit = opts.get("limit")
            limit_val = int(limit) if limit is not None else None
            clean_seq = str(opts.get("clean_sequences", "false")).lower() == "true"
            rem_non_std = str(opts.get("remove_non_standard", "false")).lower() == "true"
            return self.context.ingest_fasta(
                source_uri=node.source_uri,
                target_name=node.target_table,
                sample=sample_val,
                limit=limit_val,
                random_seed=int(opts.get("seed", 42)),
                target_ids=opts.get("target_ids"),
                chunk_size=int(opts.get("chunk_size", 50000)),
                clean_sequences=clean_seq,
                remove_non_standard=rem_non_std,
            )

        elif isinstance(node, IngestIDMappingStatement):
            opts = node.options
            return self.context.ingest_idmapping(
                source_uri=node.source_uri,
                target_name=node.target_table,
                id_types=opts.get("id_types"),
                delimiter=opts.get("delimiter", "\t"),
                has_header=opts.get("has_header", False),
            )

        elif isinstance(node, MapIdentifiersStatement):
            opts = node.options
            return self.context.map_identifiers(
                source_table=node.source_table,
                mapping_table=node.mapping_table,
                target_name=node.target_table,
                src_col=opts.get("src_col", "external_id"),
                target_col=opts.get("target_col", "uniprot_ac"),
                id_type=opts.get("id_type"),
                join_type=opts.get("join_type", "inner"),
            )

        elif isinstance(node, CoarsenGraphStatement):
            opts = node.options
            return self.context.coarsen_graph(
                source_table=node.source_table,
                target_name=node.target_table,
                src_col=opts.get("src_col", "src"),
                dst_col=opts.get("dst_col", "dst"),
                weight_col=opts.get("weight_col", "weight"),
                levels=int(opts.get("levels", 1)),
                method=opts.get("method", "graclus"),
                mapping_table=opts.get("mapping_table"),
            )

        elif isinstance(node, IngestPPIStatement):
            opts = node.options
            return self.context.ingest_ppi(
                source_uri=node.source_uri,
                target_name=node.target_table,
                source_format=opts.get("source", opts.get("format", "biogrid")),
                experimental_system=opts.get("system", "physical"),
                min_score=float(opts.get("min_score", 400.0)),
                mapping_table=node.mapping_table or opts.get("mapping_table"),
                delimiter=opts.get("delimiter"),
            )

        elif isinstance(node, ExtractEntitiesStatement):
            opts = node.options
            return self.context.extract_entities(
                source_table=node.source_table,
                target_name=node.target_table,
                text_col=opts.get("text_col", "abstract"),
                key_col=opts.get("key_col", "pmid"),
                annotations_table=node.annotations_table or opts.get("annotations_table"),
                vocab_path=opts.get("vocab_path"),
                entity_types=opts.get("entity_types"),
            )

        elif isinstance(node, MapOntologyStatement):
            opts = node.options
            return self.context.map_ontology(
                source_table=node.source_table,
                target_name=node.target_table,
                ontology=opts.get("ontology", "chem2bio2rdf"),
                drug_mapping=opts.get("drug_mapping"),
                gene_mapping=opts.get("gene_mapping"),
                ontology_triples=node.ontology_triples or opts.get("ontology_triples"),
            )

        elif isinstance(node, ConstructKnowledgeGraphStatement):
            opts = node.options
            return self.context.construct_knowledge_graph(
                articles_table=node.source_table,
                target_name=node.target_table,
                entities_table=opts.get("entities"),
                authors_table=opts.get("authors"),
                citations_table=opts.get("citations"),
                triples_table=opts.get("triples"),
                ontology_table=opts.get("ontology"),
            )

        elif isinstance(node, ProfileGraphStatement):
            opts = node.options
            return self.context.profile_graph(
                source_table=node.source_table,
                target_name=node.target_table,
                target_type=opts.get("target_type", "author"),
                top_n=int(opts.get("top_n", 20)),
            )

        elif isinstance(node, DropTableStatement):
            dropped = self.context.drop_table(node.table_name, purge=node.purge)
            if not dropped and not node.if_exists:
                raise ValueError(f"Table '{node.table_name}' does not exist.")
            return {"dropped": dropped, "table": node.table_name, "purged": node.purge}

        elif isinstance(node, VacuumStatement):
            purged = self.context.vacuum(node.table_name, retain_hours=node.retain_hours)
            return {"table": node.table_name, "files_purged": purged}

        elif isinstance(node, ExplainStatement):
            return self.explain(node.inner_statement)

        elif isinstance(node, AIProfileStatement):
            opts = node.options
            plan = self.context.ai_profile(source_uri=node.source_uri, api_key=opts.get("api_key"))
            return plan.to_dict()

        elif isinstance(node, AIRouteStatement):
            opts = node.options
            auto_exec = str(opts.get("auto_execute", opts.get("execute", "true"))).lower() in ("true", "1", "yes")
            plan = self.context.ai_route(
                source_uri=node.source_uri,
                auto_execute=auto_exec,
                target_table=node.target_table or opts.get("target"),
                api_key=opts.get("api_key"),
            )
            return plan.to_dict()

        else:
            raise TypeError(f"Unknown AST node type: {type(node)}")

    def explain(self, node: ASTNode) -> str:
        lines = [
            f"=== Lakehouse Query Plan: {node.__class__.__name__} ===",
        ]
        tbl = getattr(node, "target_table", getattr(node, "table_name", ""))
        tier = "bronze"
        if "." in tbl:
            tier, tbl = tbl.split(".", 1)

        if tbl:
            lines.append(f"Target Tier     : {tier.upper()}")
            lines.append(f"Target Table    : {tbl}")
        if hasattr(node, "source_table"):
            lines.append(f"Source Table    : {getattr(node, 'source_table')}")
        if hasattr(node, "source_uri"):
            lines.append(f"Source URI      : {getattr(node, 'source_uri')}")
        if hasattr(node, "where_clause") and getattr(node, "where_clause"):
            lines.append(f"Filter (WHERE)  : {getattr(node, 'where_clause')}")
        if hasattr(node, "order_by") and getattr(node, "order_by"):
            dir_str = "DESC" if getattr(node, "order_desc", False) else "ASC"
            lines.append(f"Order By        : {getattr(node, 'order_by')} {dir_str}")
        if hasattr(node, "limit") and getattr(node, "limit") is not None:
            lines.append(f"Limit           : {getattr(node, 'limit')}")
        if hasattr(node, "columns") and getattr(node, "columns") != ["*"]:
            lines.append(f"Columns         : {', '.join(getattr(node, 'columns'))}")
        if hasattr(node, "options") and getattr(node, "options"):
            lines.append("Plan Options    :")
            for k, v in getattr(node, "options").items():
                lines.append(f"  * {k}: {v}")
        return "\n".join(lines)
