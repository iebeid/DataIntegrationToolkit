import os
import re
import time
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pyarrow as pa
import pyarrow.compute as pc

from toolkit.catalog.table import TableMetadata
from toolkit.operators.base import BaseOperator


class ExtractEntitiesOperator(BaseOperator):
    """
    Biomedical Named Entity Recognition (NER) Operator.
    Extracts disease, gene/protein, and drug/chemical entities from unstructured
    literature abstracts or text columns, supporting vocabulary gazetteers,
    regex boundary matchers, and pre-computed BioBERT / BERN annotation tables.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        text_col: str = "abstract",
        key_col: str = "pmid",
        annotations_table: Optional[str] = None,
        vocab_path: Optional[str] = None,
        entity_types: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(name="ExtractEntitiesOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.text_col = text_col
        self.key_col = key_col
        self.annotations_table = annotations_table
        self.vocab_path = vocab_path

        if isinstance(entity_types, str):
            self.entity_types = {t.strip().lower() for t in entity_types.split(",") if t.strip()}
        elif isinstance(entity_types, list):
            self.entity_types = {t.strip().lower() for t in entity_types}
        else:
            self.entity_types = {"disease", "gene", "drug"}

    def _load_vocabularies(self, context: Any) -> Dict[str, Dict[str, str]]:
        """Loads gazetteers mapping lowercase terms -> (entity_id, entity_type)."""
        vocab: Dict[str, Dict[str, str]] = {}  # term -> {"id": ..., "type": ...}

        # Default canonical biomedical terms
        defaults = {
            "disease": [
                ("hepatocellular carcinoma", "disease/106939701"),
                ("cirrhosis", "disease/107405101"),
                ("adenocarcinoma", "disease/107418901"),
                ("leukemia", "disease/107405102"),
                ("carcinoma", "disease/106939702"),
                ("cancer", "disease/106939700"),
                ("tumour", "disease/106939703"),
                ("tumor", "disease/106939703"),
            ],
            "gene": [
                ("tp53", "gene/7157"),
                ("p53", "gene/7157"),
                ("hras", "gene/3265"),
                ("brca1", "gene/672"),
                ("brca2", "gene/675"),
                ("prkca", "gene/5578"),
                ("apc", "gene/324"),
                ("pd-l1", "gene/29126"),
                ("cd274", "gene/29126"),
                ("arid1a", "gene/8289"),
                ("ccne1", "gene/898"),
                ("cdh1", "gene/999"),
            ],
            "drug": [
                ("sorafenib", "drug/216239"),
                ("lenvatinib", "drug/9823820"),
                ("regorafenib", "drug/11167602"),
                ("carmustine", "drug/4660903"),
                ("cisplatin", "drug/441203"),
                ("doxorubicin", "drug/31703"),
                ("fluorouracil", "drug/3385"),
                ("5-fluorouracil", "drug/3385"),
            ],
        }

        for etype, items in defaults.items():
            if etype in self.entity_types:
                for term, eid in items:
                    vocab[term.lower()] = {"id": eid, "type": etype}

        if not self.vocab_path:
            return vocab

        fs, norm_path = context.storage.get_fs_and_path(self.vocab_path)
        if not fs.exists(norm_path):
            return vocab

        # Check if directory with individual lists or single file
        file_map = {
            "disease": ["disease-list.txt", "diseases.txt"],
            "gene": ["gene-list.txt", "genes.txt"],
            "drug": ["drug-list.txt", "drugs.txt"],
        }

        if fs.isdir(norm_path):
            for etype, fnames in file_map.items():
                if etype not in self.entity_types:
                    continue
                for fn in fnames:
                    sub_p = f"{norm_path.rstrip('/')}/{fn}"
                    if fs.exists(sub_p):
                        with fs.open(sub_p, "rt", encoding="utf-8") as f:
                            for idx, line in enumerate(f):
                                term = line.strip().lower()
                                if term and len(term) >= 2:
                                    vocab[term] = {"id": f"{etype}/{abs(hash(term)) % 100000000}", "type": etype}

            # Check for biobert-entities.txt
            biobert_p = f"{norm_path.rstrip('/')}/biobert-entities.txt"
            if fs.exists(biobert_p):
                with fs.open(biobert_p, "rt", encoding="utf-8") as f:
                    curr_id = None
                    curr_type = None
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if "/" in line and not line.startswith("["):
                            curr_id = line
                            curr_type = line.split("/")[0].lower()
                        elif curr_id and curr_type in self.entity_types and line.startswith("["):
                            try:
                                import ast
                                mentions = ast.literal_eval(line)
                                for m in mentions:
                                    m_low = str(m).strip().lower()
                                    if m_low:
                                        vocab[m_low] = {"id": curr_id, "type": curr_type}
                            except Exception:
                                pass
        return vocab

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        doc_ids: List[str] = []
        entity_ids: List[str] = []
        entity_types: List[str] = []
        mentions: List[str] = []
        start_chars: List[int] = []
        end_chars: List[int] = []
        row_ids: List[int] = []
        ingest_time = int(time.time() * 1000)
        curr_row = 0

        # Branch 1: Ingest from pre-computed annotation table (e.g. BERN / BioBERT)
        if self.annotations_table:
            ann_tbl = context.read_table(self.annotations_table)
            pydict = ann_tbl.to_pydict()
            n_ann = ann_tbl.num_rows

            pmid_col = self.key_col if self.key_col in ann_tbl.column_names else "PMID"
            ent_id_col = "EntityID" if "EntityID" in ann_tbl.column_names else "entity_id"
            type_col = "Type" if "Type" in ann_tbl.column_names else "entity_type"
            mention_col = "Mention" if "Mention" in ann_tbl.column_names else "mention"

            type_code_map = {"2": "disease", "3": "gene", "4": "drug"}

            for i in range(n_ann):
                pmid = str(pydict[pmid_col][i])
                raw_eid = str(pydict[ent_id_col][i])
                raw_type = str(pydict[type_col][i]).lower()
                m = str(pydict[mention_col][i])

                norm_type = type_code_map.get(raw_type, raw_type)
                if norm_type not in self.entity_types or raw_eid == "0":
                    continue

                full_eid = raw_eid if "/" in raw_eid else f"{norm_type}/{raw_eid}"

                doc_ids.append(pmid)
                entity_ids.append(full_eid)
                entity_types.append(norm_type)
                mentions.append(m)
                start_chars.append(0)
                end_chars.append(len(m))
                row_ids.append(curr_row)
                curr_row += 1

        else:
            # Branch 2: Unstructured Text NER using Gazetteers & Regex
            src_tbl = context.read_table(self.source_table)
            src_dict = src_tbl.to_pydict()
            n_docs = src_tbl.num_rows

            if self.key_col not in src_tbl.column_names:
                key_col = "_row_id" if "_row_id" in src_tbl.column_names else src_tbl.column_names[0]
            else:
                key_col = self.key_col

            if self.text_col not in src_tbl.column_names:
                raise KeyError(f"Text column '{self.text_col}' not found in '{self.source_table}'.")

            vocab = self._load_vocabularies(context)
            # Sort terms by descending length to match longest multi-word phrases first
            sorted_terms = sorted(vocab.keys(), key=len, reverse=True)

            # Build regex pattern for vocabulary matching
            escaped_terms = [re.escape(t) for t in sorted_terms if len(t) >= 2]
            if escaped_terms:
                pattern = re.compile(r"\b(" + "|".join(escaped_terms) + r")\b", re.IGNORECASE)
            else:
                pattern = None

            for i in range(n_docs):
                doc_id = str(src_dict[key_col][i])
                text = str(src_dict[self.text_col][i] or "")
                if not text or not pattern:
                    continue

                for match in pattern.finditer(text):
                    m_text = match.group(0)
                    m_low = m_text.lower()
                    meta = vocab.get(m_low)
                    if meta:
                        doc_ids.append(doc_id)
                        entity_ids.append(meta["id"])
                        entity_types.append(meta["type"])
                        mentions.append(m_text)
                        start_chars.append(match.start())
                        end_chars.append(match.end())
                        row_ids.append(curr_row)
                        curr_row += 1

        schema = pa.schema([
            ("doc_id", pa.string()),
            ("entity_id", pa.string()),
            ("entity_type", pa.string()),
            ("mention", pa.string()),
            ("start_char", pa.int64()),
            ("end_char", pa.int64()),
            ("_lakehouse_ingest_time", pa.int64()),
            ("_source_table", pa.string()),
            ("_row_id", pa.int64()),
        ])

        n_rows = len(doc_ids)
        times = [ingest_time] * n_rows
        sources = [self.source_table] * n_rows

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(doc_ids, type=pa.string()),
                pa.array(entity_ids, type=pa.string()),
                pa.array(entity_types, type=pa.string()),
                pa.array(mentions, type=pa.string()),
                pa.array(start_chars, type=pa.int64()),
                pa.array(end_chars, type=pa.int64()),
                pa.array(times, type=pa.int64()),
                pa.array(sources, type=pa.string()),
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
            properties={
                "source_table": self.source_table,
                "text_col": self.text_col,
                "extracted_entities_count": n_rows,
                "unique_entities_count": len(set(entity_ids)),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class OntologyMapperOperator(BaseOperator):
    """
    Maps extracted bio-entities to canonical Chem2Bio2RDF and standard semantic ontologies
    (PubChem compounds, UniProt gene targets, MeSH / OMIM diseases).
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        ontology: str = "chem2bio2rdf",
        drug_mapping: Optional[Union[str, Dict[str, str]]] = None,
        gene_mapping: Optional[Union[str, Dict[str, str]]] = None,
        ontology_triples: Optional[str] = None,
    ):
        super().__init__(name="OntologyMapperOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.ontology = ontology.lower()
        self.drug_mapping = drug_mapping
        self.gene_mapping = gene_mapping
        self.ontology_triples = ontology_triples

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="silver")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        src_tbl = context.read_table(self.source_table)
        src_dict = src_tbl.to_pydict()
        n_rows = src_tbl.num_rows

        eids = [str(x) for x in src_dict["entity_id"]]
        etypes = [str(x) for x in src_dict["entity_type"]]
        mentions = [str(x) for x in src_dict["mention"]]

        # 1. Resolve drug mapping (mention -> pubchem cid)
        drug_map: Dict[str, str] = {
            "sorafenib": "216239",
            "lenvatinib": "9823820",
            "regorafenib": "11167602",
            "carmustine": "4660903",
            "cisplatin": "441203",
            "doxorubicin": "31703",
        }
        if isinstance(self.drug_mapping, str):
            d_tbl = context.read_table(self.drug_mapping)
            c1, c2 = d_tbl.column_names[0], d_tbl.column_names[1]
            for m, cid in zip(d_tbl[c1].to_pylist(), d_tbl[c2].to_pylist()):
                if m and cid:
                    drug_map[str(m).strip().lower()] = str(cid).strip()
        elif isinstance(self.drug_mapping, dict):
            drug_map.update({k.lower(): str(v) for k, v in self.drug_mapping.items()})

        # 2. Resolve gene mapping (mention -> canonical symbol/uniprot)
        gene_map: Dict[str, str] = {
            "p53": "TP53",
            "tp53": "TP53",
            "hras": "HRAS",
            "brca1": "BRCA1",
            "brca2": "BRCA2",
            "prkca": "PRKCA",
            "apc": "APC",
            "pd-l1": "CD274",
        }
        if isinstance(self.gene_mapping, str):
            g_tbl = context.read_table(self.gene_mapping)
            c1, c2 = g_tbl.column_names[0], g_tbl.column_names[1]
            for m, sym in zip(g_tbl[c1].to_pylist(), g_tbl[c2].to_pylist()):
                if m and sym:
                    gene_map[str(m).strip().lower()] = str(sym).strip().upper()
        elif isinstance(self.gene_mapping, dict):
            gene_map.update({k.lower(): str(v).upper() for k, v in self.gene_mapping.items()})

        # 3. Optional ontology validation nodes
        valid_nodes: Optional[Set[str]] = None
        if self.ontology_triples:
            triples_tbl = context.read_table(self.ontology_triples)
            s_col = triples_tbl.column_names[0]
            o_col = triples_tbl.column_names[2] if len(triples_tbl.column_names) >= 3 else triples_tbl.column_names[1]
            s_nodes = set(triples_tbl[s_col].to_pylist())
            o_nodes = set(triples_tbl[o_col].to_pylist())
            valid_nodes = {str(n) for n in (s_nodes | o_nodes) if n}

        # Build mappings
        mapped_eids = []
        mapped_types = []
        mapped_mentions = []
        canonical_uris = []
        ontologies = []

        seen_entities = set()

        for eid, etype, m in zip(eids, etypes, mentions):
            pair_key = (eid, m.lower())
            if pair_key in seen_entities:
                continue
            seen_entities.add(pair_key)

            m_low = m.lower()
            uri = None

            if etype == "drug":
                cid = drug_map.get(m_low) or (eid.split("/")[1] if "/" in eid else None)
                if cid:
                    uri = f"http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/{cid}"
            elif etype == "gene":
                symbol = gene_map.get(m_low, m.upper())
                uri = f"http://chem2bio2rdf.org/uniprot/resource/gene/{symbol}"
            elif etype == "disease":
                code = eid.split("/")[1] if "/" in eid else abs(hash(m_low)) % 1000000
                uri = f"http://chem2bio2rdf.org/omim/resource/disease/{code}"

            if uri:
                if valid_nodes is not None and uri not in valid_nodes:
                    continue

                mapped_eids.append(eid)
                mapped_types.append(etype)
                mapped_mentions.append(m)
                canonical_uris.append(uri)
                ontologies.append(self.ontology)

        schema = pa.schema([
            ("entity_id", pa.string()),
            ("entity_type", pa.string()),
            ("mention", pa.string()),
            ("canonical_uri", pa.string()),
            ("ontology", pa.string()),
        ])

        out_table = pa.Table.from_arrays(
            [
                pa.array(mapped_eids, type=pa.string()),
                pa.array(mapped_types, type=pa.string()),
                pa.array(mapped_mentions, type=pa.string()),
                pa.array(canonical_uris, type=pa.string()),
                pa.array(ontologies, type=pa.string()),
            ],
            schema=schema,
        )

        context.storage.write_tabular(out_table, target_path, file_format="parquet")

        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=len(out_table),
            properties={
                "source_table": self.source_table,
                "ontology": self.ontology,
                "mapped_entities_count": len(out_table),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class ConstructKnowledgeGraphOperator(BaseOperator):
    """
    Constructs and merges heterogeneous multi-relational Knowledge Graphs (RDF Triples)
    combining literature articles, bio-entities, authors, citations, and semantic domain relations.
    Synthesizes MERGE_06_Construct_And_Merge_PubMed_C2B2RDF_Graphs.py.
    """

    def __init__(
        self,
        source_table: Optional[str] = None,
        target_name: str = "",
        entities_table: Optional[str] = None,
        authors_table: Optional[str] = None,
        citations_table: Optional[str] = None,
        ontology_triples: Optional[str] = None,
        ontology_mapping_table: Optional[str] = None,
        articles_table: Optional[str] = None,
        triples_table: Optional[str] = None,
        ontology_table: Optional[str] = None,
    ):
        super().__init__(name="ConstructKnowledgeGraphOperator")
        self.source_table = articles_table or source_table or ""
        self.target_name = target_name
        self.entities_table = entities_table or ""
        self.authors_table = authors_table
        self.citations_table = citations_table
        self.ontology_triples = ontology_triples or triples_table
        self.ontology_mapping_table = ontology_mapping_table or ontology_table

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="gold")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        triples: Set[Tuple[str, str, str, str, str]] = set()

        # 1. Literature Articles
        art_tbl = context.read_table(self.source_table)
        id_col = "pmid" if "pmid" in art_tbl.column_names else art_tbl.column_names[0]
        articles = {str(x) for x in art_tbl[id_col].to_pylist() if x is not None}

        # 2. Entity Mentions
        ent_tbl = context.read_table(self.entities_table)
        doc_col = "doc_id" if "doc_id" in ent_tbl.column_names else "pmid"
        eid_col = "entity_id" if "entity_id" in ent_tbl.column_names else "EntityID"
        etype_col = "entity_type" if "entity_type" in ent_tbl.column_names else "Type"

        for doc_id, eid, etype in zip(ent_tbl[doc_col].to_pylist(), ent_tbl[eid_col].to_pylist(), ent_tbl[etype_col].to_pylist()):
            if doc_id and eid:
                s_node = f"bioentity/{eid}"
                o_node = f"article/pmid/{doc_id}"
                triples.add((s_node, "mentionedIn", o_node, str(etype), "article"))

        # 3. Authors (if available)
        if self.authors_table:
            auth_tbl = context.read_table(self.authors_table)
            a_doc_col = auth_tbl.column_names[0]
            a_col = auth_tbl.column_names[1]
            for doc_id, auth_id in zip(auth_tbl[a_doc_col].to_pylist(), auth_tbl[a_col].to_pylist()):
                if doc_id and auth_id:
                    s_node = f"author/{auth_id}"
                    o_node = f"article/pmid/{doc_id}"
                    triples.add((s_node, "wrote", o_node, "author", "article"))

        # 4. Citations (if available)
        if self.citations_table:
            cit_tbl = context.read_table(self.citations_table)
            c1, c2 = cit_tbl.column_names[0], cit_tbl.column_names[1]
            for src_id, dst_id in zip(cit_tbl[c1].to_pylist(), cit_tbl[c2].to_pylist()):
                if src_id and dst_id:
                    s_node = f"article/pmid/{src_id}"
                    o_node = f"article/pmid/{dst_id}"
                    triples.add((s_node, "cites", o_node, "article", "article"))

        # 5. Canonical Equivalence & Ontology Mapping
        if self.ontology_mapping_table:
            ont_tbl = context.read_table(self.ontology_mapping_table)
            for eid, uri, etype in zip(ont_tbl["entity_id"].to_pylist(), ont_tbl["canonical_uri"].to_pylist(), ont_tbl["entity_type"].to_pylist()):
                if eid and uri:
                    s_node = f"bioentity/{eid}"
                    triples.add((s_node, "sameAs", str(uri), str(etype), "uri"))

        # 6. Domain Ontology Triples (e.g. Chem2Bio2RDF links)
        if self.ontology_triples:
            dom_tbl = context.read_table(self.ontology_triples)
            c_s = dom_tbl.column_names[0]
            c_p = dom_tbl.column_names[1]
            c_o = dom_tbl.column_names[2]
            for s, p, o in zip(dom_tbl[c_s].to_pylist(), dom_tbl[c_p].to_pylist(), dom_tbl[c_o].to_pylist()):
                if s and p and o:
                    triples.add((str(s), str(p), str(o), "uri", "uri"))

        subjects = [t[0] for t in triples]
        predicates = [t[1] for t in triples]
        objects = [t[2] for t in triples]
        s_types = [t[3] for t in triples]
        o_types = [t[4] for t in triples]
        weights = [1.0] * len(triples)

        schema = pa.schema([
            ("subject", pa.string()),
            ("predicate", pa.string()),
            ("object", pa.string()),
            ("subject_type", pa.string()),
            ("object_type", pa.string()),
            ("weight", pa.float64()),
        ])

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(subjects, type=pa.string()),
                pa.array(predicates, type=pa.string()),
                pa.array(objects, type=pa.string()),
                pa.array(s_types, type=pa.string()),
                pa.array(o_types, type=pa.string()),
                pa.array(weights, type=pa.float64()),
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
                "source_articles": self.source_table,
                "entities_table": self.entities_table,
                "total_triples": len(arrow_table),
                "unique_subjects": len(set(subjects)),
                "unique_objects": len(set(objects)),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class AuthorProfilerOperator(BaseOperator):
    """
    Knowledge Graph & Author Profiler Operator.
    Synthesizes MERGE_09_Author_Profiler.py to analyze degree centrality,
    citation influence, and biomedical entity associations (diseases, genes, drugs)
    for researchers and domain entities across the heterogeneous knowledge graph.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        target_type: str = "author",
        top_n: int = 50,
    ):
        super().__init__(name="AuthorProfilerOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.target_type = target_type.lower()
        self.top_n = top_n

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="gold")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        kg_table = context.read_table(self.source_table)
        subjects = [str(x) for x in kg_table["subject"].to_pylist()]
        predicates = [str(x) for x in kg_table["predicate"].to_pylist()]
        objects = [str(x) for x in kg_table["object"].to_pylist()]
        s_types = [str(x) for x in kg_table["subject_type"].to_pylist()]
        o_types = [str(x) for x in kg_table["object_type"].to_pylist()]

        # Build adjacency and node types
        adj: Dict[str, Set[str]] = defaultdict(set)
        in_degrees: Counter = Counter()
        out_degrees: Counter = Counter()
        node_types: Dict[str, str] = {}

        # Co-occurrence bipartite graph (e.g. author -> article -> entities)
        author_articles: Dict[str, Set[str]] = defaultdict(set)
        article_entities: Dict[str, Set[str]] = defaultdict(set)

        for s, p, o, st, ot in zip(subjects, predicates, objects, s_types, o_types):
            adj[s].add(o)
            adj[o].add(s)
            out_degrees[s] += 1
            in_degrees[o] += 1

            if s not in node_types:
                node_types[s] = st
            if o not in node_types:
                node_types[o] = ot

            # Infer node types from URI/prefix if generic
            if s.startswith("author/"):
                node_types[s] = "author"
            elif s.startswith("article/"):
                node_types[s] = "article"
            elif s.startswith("bioentity/"):
                node_types[s] = st if st != "uri" else "bioentity"

            if o.startswith("author/"):
                node_types[o] = "author"
            elif o.startswith("article/"):
                node_types[o] = "article"
            elif o.startswith("bioentity/"):
                node_types[o] = ot if ot != "uri" else "bioentity"

            # Track authorship and mentions
            if p == "wrote" and s.startswith("author/") and o.startswith("article/"):
                author_articles[s].add(o)
            elif p == "mentionedIn" and s.startswith("bioentity/") and o.startswith("article/"):
                article_entities[o].add(s)

        total_nodes = len(adj)
        scale = max(total_nodes - 1, 1)

        # Compute degree centrality: deg / (N - 1)
        nodes_to_profile = []
        for n, ntype in node_types.items():
            if self.target_type == "all" or ntype == self.target_type or (self.target_type == "author" and n.startswith("author/")):
                nodes_to_profile.append(n)

        # Sort by total degree
        nodes_to_profile.sort(key=lambda n: len(adj[n]), reverse=True)
        selected_nodes = nodes_to_profile[: self.top_n]

        out_node_ids = []
        out_types = []
        out_centralities = []
        out_in_deg = []
        out_out_deg = []
        out_associations = []

        for n in selected_nodes:
            deg = len(adj[n])
            centrality = deg / scale
            ntype = node_types.get(n, "entity")

            # Collect top associations (for authors: top co-occurring bio-entities)
            if n.startswith("author/"):
                papers = author_articles.get(n, set())
                ent_counts: Dict[str, Counter] = {"disease": Counter(), "gene": Counter(), "drug": Counter(), "other": Counter()}
                for p in papers:
                    for ent in article_entities.get(p, set()):
                        etype = node_types.get(ent, "other")
                        if etype not in ent_counts:
                            etype = "other"
                        ent_counts[etype][ent.replace("bioentity/", "")] += 1
                assoc_data = {
                    "diseases": [f"{e}:{c}" for e, c in ent_counts["disease"].most_common(5)],
                    "genes": [f"{e}:{c}" for e, c in ent_counts["gene"].most_common(5)],
                    "drugs": [f"{e}:{c}" for e, c in ent_counts["drug"].most_common(5)],
                }
                assoc_str = json.dumps(assoc_data)
            else:
                top_neighbors = [neighbor for neighbor in adj[n] if not neighbor.startswith("author/")][:5]
                assoc_str = json.dumps({"neighbors": top_neighbors})

            out_node_ids.append(n)
            out_types.append(ntype)
            out_centralities.append(float(round(centrality, 6)))
            out_in_deg.append(int(in_degrees[n]))
            out_out_deg.append(int(out_degrees[n]))
            out_associations.append(assoc_str)

        schema = pa.schema([
            ("node_id", pa.string()),
            ("node_type", pa.string()),
            ("degree_centrality", pa.float64()),
            ("in_degree", pa.int64()),
            ("out_degree", pa.int64()),
            ("top_associations", pa.string()),
        ])

        arrow_table = pa.Table.from_arrays(
            [
                pa.array(out_node_ids, type=pa.string()),
                pa.array(out_types, type=pa.string()),
                pa.array(out_centralities, type=pa.float64()),
                pa.array(out_in_deg, type=pa.int64()),
                pa.array(out_out_deg, type=pa.int64()),
                pa.array(out_associations, type=pa.string()),
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
                "source_kg": self.source_table,
                "target_type": self.target_type,
                "profiled_nodes_count": len(arrow_table),
            },
        )
        context.catalog.register_table(metadata)
        return metadata


class KnowledgeGraphEmbeddingOperator(BaseOperator):
    """
    Refactored from src/MERGE_08_Graph_Embedding.py.
    Computes Node2Vec / MetaPath2Vec biased random walk embeddings over Knowledge Graphs,
    projecting heterogeneous entities (authors, articles, genes, drugs, diseases) into continuous space.
    """

    def __init__(
        self,
        source_table: str,
        target_name: str,
        dimension: int = 128,
        walk_length: int = 30,
        walks_per_node: int = 5,
        p: float = 0.5,
        q: float = 2.0,
    ):
        super().__init__(name="KnowledgeGraphEmbeddingOperator")
        self.source_table = source_table
        self.target_name = target_name
        self.dimension = dimension
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q

    def execute(self, context: Any, **kwargs) -> TableMetadata:
        tier, tbl = context.catalog.parse_table_name(self.target_name, default_tier="gold")
        target_path = os.path.join(context.lakehouse_root, tier, f"{tbl}.parquet").replace("\\", "/")

        kg_table = context.read_table(self.source_table)
        subjects = [str(x) for x in kg_table["subject"].to_pylist()]
        objects = [str(x) for x in kg_table["object"].to_pylist()]

        import networkx as nx
        G = nx.Graph()
        for s, o in zip(subjects, objects):
            G.add_edge(s, o)

        nodes = list(G.nodes())
        if not nodes:
            raise ValueError(f"No nodes found in KG table {self.source_table}")

        # Biased random walk generator
        walks = []
        rng = np.random.RandomState(42)
        for _ in range(self.walks_per_node):
            rng.shuffle(nodes)
            for node in nodes:
                walk = [node]
                cur = node
                prev = None
                for _ in range(self.walk_length - 1):
                    nbrs = list(G.neighbors(cur))
                    if not nbrs:
                        break
                    if prev is None:
                        next_node = rng.choice(nbrs)
                    else:
                        # Node2Vec 2nd order bias: p = return, q = in-out
                        probs = []
                        for nbr in nbrs:
                            if nbr == prev:
                                probs.append(1.0 / self.p)
                            elif G.has_edge(prev, nbr):
                                probs.append(1.0)
                            else:
                                probs.append(1.0 / self.q)
                        probs = np.array(probs, dtype=np.float64)
                        probs /= np.sum(probs)
                        next_node = rng.choice(nbrs, p=probs)
                    prev = cur
                    cur = next_node
                    walk.append(cur)
                walks.append(walk)

        # Train Word2Vec on random walks
        try:
            from gensim.models import Word2Vec
            model = Word2Vec(sentences=walks, vector_size=self.dimension, window=5, min_count=1, sg=1, epochs=5, seed=42)
            node_embeddings = {n: model.wv[n] for n in model.wv.index_to_key}
        except ImportError:
            # Levy & Goldberg SVD PMI factorization on walks
            from toolkit.operators.gnn import Word2VecOperator
            node_embeddings, _ = Word2VecOperator._train_embeddings_pmi_svd(walks, self.dimension, 5, 1)

        node_list = []
        embed_list = []
        for n in nodes:
            vec = node_embeddings.get(n)
            if vec is None:
                vec = np.zeros(self.dimension, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            node_list.append(n)
            embed_list.append(vec.tolist())

        schema = pa.schema([
            ("node_id", pa.string()),
            ("embedding", pa.list_(pa.float32())),
        ])

        arrow_table = pa.Table.from_arrays(
            [pa.array(node_list, type=pa.string()), pa.array(embed_list, type=pa.list_(pa.float32()))],
            schema=schema,
        )

        context.storage.write_tabular(arrow_table, target_path, file_format="parquet")
        metadata = TableMetadata(
            name=tbl,
            tier=tier,
            uri=target_path,
            schema=schema,
            row_count=len(arrow_table),
            properties={"dimension": self.dimension, "walk_count": len(walks)},
        )
        context.catalog.register_table(metadata)
        return metadata

