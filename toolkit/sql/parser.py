import re
from typing import Any, Dict, List, Optional
from toolkit.sql.ast import (
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


class SQLParser:
    """
    Parses Lakehouse Ingestion & Entity Resolution SQL dialect into AST nodes.
    """

    def parse(self, sql: str) -> ASTNode:
        clean_sql = sql.strip().rstrip(";")

        # Normalize spaces
        tokens = re.split(r"\s+", clean_sql)
        if not tokens or not tokens[0]:
            raise ValueError("Empty SQL command.")

        verb = tokens[0].upper()

        if verb == "EXPLAIN":
            inner_sql = clean_sql[len("EXPLAIN"):].strip()
            inner_ast = self.parse(inner_sql)
            return ExplainStatement(inner_statement=inner_ast)

        if verb == "AI" and len(tokens) > 1:
            action = tokens[1].upper()
            if action == "PROFILE":
                return self._parse_ai_profile(clean_sql)
            elif action == "ROUTE":
                return self._parse_ai_route(clean_sql)

        if verb == "DROP" and len(tokens) > 1 and tokens[1].upper() == "TABLE":
            return self._parse_drop_table(clean_sql)

        if verb == "VACUUM":
            return self._parse_vacuum(clean_sql)

        if verb == "SHOW" and len(tokens) >= 2 and tokens[1].upper() == "TABLES":
            tier = tokens[2].lower() if len(tokens) > 2 else None
            return ShowTablesStatement(tier=tier)

        if verb == "DESCRIBE" or (verb == "DESC" and len(tokens) >= 2):
            table_idx = 2 if tokens[1].upper() == "TABLE" else 1
            table_name = tokens[table_idx]
            return DescribeTableStatement(table_name=table_name)

        if verb == "SELECT":
            return self._parse_select(clean_sql)

        if verb == "INGEST":
            second = tokens[1].upper() if len(tokens) > 1 else ""
            if second == "FASTA":
                return self._parse_ingest_fasta(clean_sql)
            elif second == "IDMAPPING":
                return self._parse_ingest_idmapping(clean_sql)
            elif second == "PPI":
                return self._parse_ingest_ppi(clean_sql)
            else:
                return self._parse_ingest(clean_sql)

        if verb == "CREATE":
            second = tokens[1].upper() if len(tokens) > 1 else ""
            if second == "SOURCE":
                return self._parse_create_source(clean_sql)
            elif second == "STAGE":
                return self._parse_create_stage(clean_sql)
            elif second == "BLOCKS":
                return self._parse_create_blocks(clean_sql)
            elif second == "EMBEDDINGS":
                return self._parse_create_embeddings(clean_sql)

        if verb == "TRAIN" and len(tokens) > 1 and tokens[1].upper() == "GNN":
            return self._parse_train_gnn(clean_sql)

        if verb == "MAP" and len(tokens) > 1:
            second = tokens[1].upper()
            if second == "IDENTIFIERS":
                return self._parse_map_identifiers(clean_sql)
            elif second == "ONTOLOGY":
                return self._parse_map_ontology(clean_sql)

        if verb == "EXTRACT" and len(tokens) > 1 and tokens[1].upper() == "ENTITIES":
            return self._parse_extract_entities(clean_sql)

        if verb == "CONSTRUCT" and len(tokens) > 2 and tokens[1].upper() == "KNOWLEDGE" and tokens[2].upper() == "GRAPH":
            return self._parse_construct_kg(clean_sql)

        if verb == "COARSEN" and len(tokens) > 1 and tokens[1].upper() == "GRAPH":
            return self._parse_coarsen_graph(clean_sql)

        if verb == "RESOLVE":
            return self._parse_resolve(clean_sql)

        if verb == "EVALUATE":
            return self._parse_evaluate(clean_sql)

        if verb == "PROFILE":
            if len(tokens) > 1 and tokens[1].upper() == "GRAPH":
                return self._parse_profile_graph(clean_sql)
            return self._parse_profile(clean_sql)

        if verb == "PROJECT" and len(tokens) > 1 and (tokens[1].upper() in ("GRAPH", "NGRAM")):
            return self._parse_project_graph(clean_sql)

        raise ValueError(f"Unsupported SQL command: {clean_sql}")

    def _parse_options(self, sql: str) -> Dict[str, Any]:
        """Extracts key-value pairs from WITH (...) clause, respecting quoted strings."""
        match = re.search(r"WITH\s*\((.*?)\)", sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return {}

        raw_opts = match.group(1).strip()
        opts = {}
        # Find key = value pairs, handling single/double quotes properly
        pattern = re.compile(r"(\w+)\s*=\s*('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[^,]+)")
        for m in pattern.finditer(raw_opts):
            k = m.group(1).strip()
            v = m.group(2).strip()

            # Parse types
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                opts[k] = v[1:-1]
            elif v.lower() == "true":
                opts[k] = True
            elif v.lower() == "false":
                opts[k] = False
            else:
                try:
                    if "." in v:
                        opts[k] = float(v)
                    else:
                        opts[k] = int(v)
                except ValueError:
                    opts[k] = v
        return opts

    def _parse_ingest(self, sql: str) -> IngestStatement:
        # INGEST FROM 'path' INTO table WITH (...) OR INGEST table FROM 'path' WITH (...)
        match1 = re.search(
            r"INGEST\s+FROM\s+['\"](.*?)['\"]\s+INTO\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if match1:
            source_uri = match1.group(1)
            target_table = match1.group(2)
        else:
            match2 = re.search(
                r"INGEST\s+([\w\.\-]+)\s+FROM\s+['\"](.*?)['\"]",
                sql,
                re.IGNORECASE,
            )
            if not match2:
                raise ValueError("Syntax: INGEST FROM '<source_uri>' INTO <target_table> [WITH (...)] or INGEST <target_table> FROM '<source_uri>' [WITH (...)]")
            target_table = match2.group(1)
            source_uri = match2.group(2)
        options = self._parse_options(sql)
        return IngestStatement(source_uri=source_uri, target_table=target_table, options=options)

    def _parse_create_source(self, sql: str) -> IngestStatement:
        # CREATE SOURCE table FROM 'path' WITH (...)
        match = re.search(
            r"CREATE\s+SOURCE\s+([\w\.\-]+)\s+FROM\s+['\"](.*?)['\"]",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: CREATE SOURCE <target_table> FROM '<source_uri>' [WITH (...)]")
        target_table = match.group(1)
        source_uri = match.group(2)
        options = self._parse_options(sql)
        return IngestStatement(source_uri=source_uri, target_table=target_table, options=options)

    def _parse_create_stage(self, sql: str) -> StageStatement:
        # CREATE STAGE target_table FROM source_table WITH (...)
        match = re.search(
            r"CREATE\s+STAGE\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: CREATE STAGE <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return StageStatement(source_table=source_table, target_table=target_table, options=options)

    def _parse_create_blocks(self, sql: str) -> BlockStatement:
        # CREATE BLOCKS target_table FROM source_table WITH (...)
        match = re.search(
            r"CREATE\s+BLOCKS\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: CREATE BLOCKS <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return BlockStatement(source_table=source_table, target_table=target_table, options=options)

    def _parse_create_embeddings(self, sql: str) -> CreateEmbeddingsStatement:
        # CREATE EMBEDDINGS target_table FROM source_table [WITH (...)]
        match = re.search(
            r"CREATE\s+EMBEDDINGS\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: CREATE EMBEDDINGS <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return CreateEmbeddingsStatement(
            source_table=source_table, target_table=target_table, options=options
        )

    def _parse_train_gnn(self, sql: str) -> TrainGNNStatement:
        # TRAIN GNN target_table FROM source_table USING GRAPH graph_table [WITH (...)]
        match = re.search(
            r"TRAIN\s+GNN\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)\s+USING\s+GRAPH\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: TRAIN GNN <target_table> FROM <source_table> USING GRAPH <graph_table> [WITH (...)]"
            )
        target_table = match.group(1)
        source_table = match.group(2)
        graph_table = match.group(3)
        options = self._parse_options(sql)
        return TrainGNNStatement(
            source_table=source_table,
            graph_table=graph_table,
            target_table=target_table,
            options=options,
        )


    def _parse_resolve(self, sql: str) -> ResolveStatement:
        # RESOLVE ENTITIES INTO target_table FROM records_table USING BLOCKS pairs_table WITH (...)
        match = re.search(
            r"RESOLVE\s+ENTITIES\s+INTO\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)\s+USING\s+BLOCKS\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: RESOLVE ENTITIES INTO <target_table> FROM <records_table> USING BLOCKS <pairs_table> [WITH (...)]"
            )
        target_table = match.group(1)
        records_table = match.group(2)
        pairs_table = match.group(3)
        options = self._parse_options(sql)
        return ResolveStatement(
            records_table=records_table,
            pairs_table=pairs_table,
            target_table=target_table,
            options=options,
        )

    def _parse_project_graph(self, sql: str) -> ProjectGraphStatement:
        # PROJECT [NGRAM] GRAPH target_table FROM source_table WITH (...)
        match = re.search(
            r"PROJECT\s+(?:NGRAM\s+)?GRAPH\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: PROJECT [NGRAM] GRAPH <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return ProjectGraphStatement(source_table=source_table, target_table=target_table, options=options)

    def _parse_select(self, sql: str) -> SelectStatement:
        match = re.search(
            r"SELECT\s+(.*?)\s+FROM\s+([\w\.\-]+)(?:\s+WHERE\s+(.*?))?(?:\s+ORDER\s+BY\s+([\w\.\-]+)(?:\s+(ASC|DESC))?)?(?:\s+LIMIT\s+(\d+))?$",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            raise ValueError("Syntax: SELECT <columns> FROM <table_name> [WHERE <condition>] [ORDER BY <col> [ASC|DESC]] [LIMIT <n>]")
        raw_cols = match.group(1).strip()
        table_name = match.group(2).strip()
        where_clause = match.group(3).strip() if match.group(3) else None
        order_by = match.group(4).strip() if match.group(4) else None
        order_dir = (match.group(5) or "ASC").upper()
        order_desc = order_dir == "DESC"
        limit_str = match.group(6)

        cols = [c.strip() for c in raw_cols.split(",")] if raw_cols != "*" else ["*"]
        limit = int(limit_str) if limit_str else None
        return SelectStatement(
            table_name=table_name,
            columns=cols,
            where_clause=where_clause,
            order_by=order_by,
            order_desc=order_desc,
            limit=limit,
        )

    def _parse_drop_table(self, sql: str) -> DropTableStatement:
        match = re.search(
            r"DROP\s+TABLE\s+(IF\s+EXISTS\s+)?([\w\.\-]+)(?:\s+(PURGE))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: DROP TABLE [IF EXISTS] <table_name> [PURGE]")
        if_exists = bool(match.group(1))
        table_name = match.group(2)
        purge = bool(match.group(3))
        return DropTableStatement(table_name=table_name, if_exists=if_exists, purge=purge)

    def _parse_vacuum(self, sql: str) -> VacuumStatement:
        match = re.search(r"VACUUM\s+([\w\.\-]+)", sql, re.IGNORECASE)
        if not match:
            raise ValueError("Syntax: VACUUM <table_name> [WITH (retain_hours = 168)]")
        table_name = match.group(1)
        options = self._parse_options(sql)
        retain_hours = int(options.get("retain_hours", 168))
        return VacuumStatement(table_name=table_name, retain_hours=retain_hours)

    def _parse_evaluate(self, sql: str) -> EvaluateStatement:
        match = re.search(
            r"EVALUATE\s+(?:ENTITIES\s+)?([\w\.\-]+)\s+AGAINST\s+TRUTH\s+['\"](.*?)['\"](?:\s+USING\s+BLOCKS\s+([\w\.\-]+))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: EVALUATE [ENTITIES] <target_table> AGAINST TRUTH '<truth_path>' [USING BLOCKS <pairs_table>] [WITH (...)]"
            )
        target_table = match.group(1)
        truth_file = match.group(2)
        pairs_table = match.group(3)
        options = self._parse_options(sql)
        return EvaluateStatement(
            target_table=target_table,
            truth_file=truth_file,
            pairs_table=pairs_table,
            options=options,
        )

    def _parse_profile(self, sql: str) -> ProfileStatement:
        match = re.search(r"PROFILE\s+(?:TABLE\s+)?([\w\.\-]+)", sql, re.IGNORECASE)
        if not match:
            raise ValueError("Syntax: PROFILE [TABLE] <target_table> [WITH (...)]")
        target_table = match.group(1)
        options = self._parse_options(sql)
        return ProfileStatement(target_table=target_table, options=options)

    def _parse_ingest_fasta(self, sql: str) -> IngestFastaStatement:
        # INGEST FASTA target_table FROM 'source_uri' WITH (...)
        match = re.search(
            r"INGEST\s+FASTA\s+([\w\.\-]+)\s+FROM\s+['\"](.*?)['\"]",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: INGEST FASTA <target_table> FROM '<source_uri>' [WITH (...)]")
        target_table = match.group(1)
        source_uri = match.group(2)
        options = self._parse_options(sql)
        return IngestFastaStatement(source_uri=source_uri, target_table=target_table, options=options)

    def _parse_ingest_idmapping(self, sql: str) -> IngestIDMappingStatement:
        # INGEST IDMAPPING target_table FROM 'source_uri' WITH (...)
        match = re.search(
            r"INGEST\s+IDMAPPING\s+([\w\.\-]+)\s+FROM\s+['\"](.*?)['\"]",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: INGEST IDMAPPING <target_table> FROM '<source_uri>' [WITH (...)]")
        target_table = match.group(1)
        source_uri = match.group(2)
        options = self._parse_options(sql)
        return IngestIDMappingStatement(source_uri=source_uri, target_table=target_table, options=options)

    def _parse_map_identifiers(self, sql: str) -> MapIdentifiersStatement:
        # MAP IDENTIFIERS target_table FROM source_table USING MAPPING mapping_table WITH (...)
        match = re.search(
            r"MAP\s+IDENTIFIERS\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)\s+USING\s+MAPPING\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: MAP IDENTIFIERS <target_table> FROM <source_table> USING MAPPING <mapping_table> [WITH (...)]"
            )
        target_table = match.group(1)
        source_table = match.group(2)
        mapping_table = match.group(3)
        options = self._parse_options(sql)
        return MapIdentifiersStatement(
            source_table=source_table,
            target_table=target_table,
            mapping_table=mapping_table,
            options=options,
        )

    def _parse_coarsen_graph(self, sql: str) -> CoarsenGraphStatement:
        # COARSEN GRAPH target_table FROM source_table WITH (...)
        match = re.search(
            r"COARSEN\s+GRAPH\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: COARSEN GRAPH <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return CoarsenGraphStatement(source_table=source_table, target_table=target_table, options=options)

    def _parse_ingest_ppi(self, sql: str) -> IngestPPIStatement:
        # INGEST PPI target_table FROM 'source_uri' [USING MAPPING mapping_table] WITH (...)
        match = re.search(
            r"INGEST\s+PPI\s+([\w\.\-]+)\s+FROM\s+['\"](.*?)['\"](?:\s+USING\s+MAPPING\s+([\w\.\-]+))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: INGEST PPI <target_table> FROM '<source_uri>' [USING MAPPING <mapping_table>] [WITH (...)]"
            )
        target_table = match.group(1)
        source_uri = match.group(2)
        mapping_table = match.group(3)
        options = self._parse_options(sql)
        return IngestPPIStatement(
            source_uri=source_uri,
            target_table=target_table,
            mapping_table=mapping_table,
            options=options,
        )

    def _parse_extract_entities(self, sql: str) -> ExtractEntitiesStatement:
        # EXTRACT ENTITIES target_table FROM source_table [USING ANNOTATIONS ann_table] [WITH (...)]
        match = re.search(
            r"EXTRACT\s+ENTITIES\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)(?:\s+USING\s+ANNOTATIONS\s+([\w\.\-]+))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: EXTRACT ENTITIES <target_table> FROM <source_table> [USING ANNOTATIONS <ann_table>] [WITH (...)]"
            )
        target_table = match.group(1)
        source_table = match.group(2)
        annotations_table = match.group(3)
        options = self._parse_options(sql)
        return ExtractEntitiesStatement(
            source_table=source_table,
            target_table=target_table,
            annotations_table=annotations_table,
            options=options,
        )

    def _parse_map_ontology(self, sql: str) -> MapOntologyStatement:
        # MAP ONTOLOGY target_table FROM source_table [USING ONTOLOGY triples_table] [WITH (...)]
        match = re.search(
            r"MAP\s+ONTOLOGY\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)(?:\s+USING\s+ONTOLOGY\s+([\w\.\-]+))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: MAP ONTOLOGY <target_table> FROM <source_table> [USING ONTOLOGY <triples_table>] [WITH (...)]"
            )
        target_table = match.group(1)
        source_table = match.group(2)
        ontology_triples = match.group(3)
        options = self._parse_options(sql)
        return MapOntologyStatement(
            source_table=source_table,
            target_table=target_table,
            ontology_triples=ontology_triples,
            options=options,
        )

    def _parse_construct_kg(self, sql: str) -> ConstructKnowledgeGraphStatement:
        # CONSTRUCT KNOWLEDGE GRAPH target_table FROM source_table [WITH (...)]
        match = re.search(
            r"CONSTRUCT\s+KNOWLEDGE\s+GRAPH\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "Syntax: CONSTRUCT KNOWLEDGE GRAPH <target_table> FROM <source_table> [WITH (...)]"
            )
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return ConstructKnowledgeGraphStatement(
            source_table=source_table,
            target_table=target_table,
            options=options,
        )

    def _parse_profile_graph(self, sql: str) -> ProfileGraphStatement:
        # PROFILE GRAPH target_table FROM source_table [WITH (...)]
        match = re.search(
            r"PROFILE\s+GRAPH\s+([\w\.\-]+)\s+FROM\s+([\w\.\-]+)",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: PROFILE GRAPH <target_table> FROM <source_table> [WITH (...)]")
        target_table = match.group(1)
        source_table = match.group(2)
        options = self._parse_options(sql)
        return ProfileGraphStatement(
            source_table=source_table,
            target_table=target_table,
            options=options,
        )

    def _parse_ai_profile(self, sql: str) -> AIProfileStatement:
        match = re.search(r"AI\s+PROFILE\s+['\"](.*?)['\"]", sql, re.IGNORECASE)
        if not match:
            raise ValueError("Syntax: AI PROFILE '<source_uri>' [WITH (...)]")
        source_uri = match.group(1).strip()
        options = self._parse_options(sql)
        return AIProfileStatement(source_uri=source_uri, options=options)

    def _parse_ai_route(self, sql: str) -> AIRouteStatement:
        match = re.search(
            r"AI\s+ROUTE\s+['\"](.*?)['\"](?:\s+INTO\s+([\w\.\-]+))?",
            sql,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Syntax: AI ROUTE '<source_uri>' [INTO <target_table>] [WITH (...)]")
        source_uri = match.group(1).strip()
        target_table = match.group(2).strip() if match.group(2) else None
        options = self._parse_options(sql)
        return AIRouteStatement(source_uri=source_uri, target_table=target_table, options=options)




