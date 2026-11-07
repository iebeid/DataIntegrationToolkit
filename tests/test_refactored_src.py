import os
import shutil
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import networkx as nx

from toolkit.context import LakehouseContext
from toolkit.operators.silver import (
    normal_led,
    BlockCleanerOperator,
    CharacterMapsOperator,
    RecordFeatureComparatorOperator,
    ClusterEntropyCalculator,
)
from toolkit.operators.gold import (
    GraphOGAlgorithms,
    CliqueRankOperator,
    HITSOperator,
    BipartiteGraphConstructor,
)
from toolkit.operators.gnn import (
    Word2VecOperator,
    Doc2VecOperator,
    FastTextOperator,
    LSTMDataBuilder,
    XGCNDataBuilder,
)
from toolkit.operators.eval import (
    ErrorLookupProfiler,
    BenchmarkDatasetLoader,
    ConfigER,
)


class TestRefactoredSrcCoverage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ctx = LakehouseContext(lakehouse_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal_led_and_block_cleaner(self):
        # 1. Test normal_led
        f_dist, d_dist = normal_led("robert", "rober")
        self.assertAlmostEqual(f_dist, 5.0 / 6.0, places=3)
        self.assertEqual(d_dist, 1)

        # 2. Test BlockCleanerOperator
        schema = pa.schema([
            ("RecID", pa.string()),
            ("_staged_tokens", pa.string()),
        ])
        tbl = pa.Table.from_arrays(
            [
                pa.array(["R1", "R2", "R3"]),
                pa.array(["univ corp inc", "caltech univ corp", "dept comp sci"]),
            ],
            schema=schema,
        )
        self.ctx.storage.write_parquet(tbl, f"{self.temp_dir}/silver/staged.parquet")
        from toolkit.catalog.table import TableMetadata
        self.ctx.catalog.register_table(
            TableMetadata(name="staged", tier="silver", uri=f"{self.temp_dir}/silver/staged.parquet", schema=schema, row_count=3)
        )

        meta = self.ctx.clean_blocks(
            source_table="silver.staged",
            target_name="silver.cleaned",
            alias_dict={"corp": "corporation", "sci": "science"},
        )
        out_table = self.ctx.read_table("silver.cleaned")
        cleaned_col = out_table["_cleaned_tokens"].to_pylist()
        self.assertIn("corporation", cleaned_col[0])
        self.assertIn("science", cleaned_col[2])

    def test_character_maps_operator(self):
        schema = pa.schema([
            ("RecID", pa.string()),
            ("name", pa.string()),
        ])
        tbl = pa.Table.from_arrays(
            [pa.array(["1", "2"]), pa.array(["Alan Turing", "Ada Lovelace"])],
            schema=schema,
        )
        self.ctx.storage.write_parquet(tbl, f"{self.temp_dir}/silver/recs.parquet")
        from toolkit.catalog.table import TableMetadata
        self.ctx.catalog.register_table(
            TableMetadata(name="recs", tier="silver", uri=f"{self.temp_dir}/silver/recs.parquet", schema=schema, row_count=2)
        )

        meta = self.ctx.character_maps(source_table="silver.recs", target_name="silver.cmaps")
        out_table = self.ctx.read_table("silver.cmaps")
        self.assertIn("_ascii_codes", out_table.column_names)
        self.assertIn("_unique_chars", out_table.column_names)

    def test_record_feature_comparator_and_digit_discriminator(self):
        # Doc digit discrimination: agreeing vs conflicting digits
        doc_agree_1 = ["christopher", "abbott", "137", "columbine", "27106", "173616732"]
        doc_agree_2 = ["christopher", "abbott", "lee", "137", "columbine", "27106", "173616732"]
        doc_conflict = ["christopher", "abbott", "sr", "2300", "mckinnon", "27012", "184858365"]

        sim_agree = RecordFeatureComparatorOperator.hierarchical_document_similarity(doc_agree_1, doc_agree_2)
        sim_conflict = RecordFeatureComparatorOperator.hierarchical_document_similarity(doc_agree_1, doc_conflict)
        self.assertGreater(sim_agree, 0.8)
        self.assertEqual(sim_conflict, 0.0)  # Penalized by digit discriminator!

    def test_cluster_entropy_calculator(self):
        # Pure cluster (identical tokens)
        pure_cluster = [["john", "smith", "chicago"], ["john", "smith", "chicago"]]
        impure_cluster = [["john", "smith", "chicago"], ["alex", "jones", "austin"], ["david", "miller", "seattle"]]

        quality_pure = ClusterEntropyCalculator.calculate_entropy(pure_cluster)
        quality_impure = ClusterEntropyCalculator.calculate_entropy(impure_cluster)
        self.assertGreater(quality_pure, quality_impure)

    def test_graphog_algorithms(self):
        edges = [("A", "B"), ("B", "C"), ("D", "E")]
        # Talburt Transitive Closure
        clusters_tc = GraphOGAlgorithms.talburt_transitive_closure(edges)
        cluster_sets = [set(c) for c in clusters_tc]
        self.assertIn({"A", "B", "C"}, cluster_sets)
        self.assertIn({"D", "E"}, cluster_sets)

        # Quick CC
        clusters_qcc = GraphOGAlgorithms.akef_quick_connected_components(edges)
        qcc_sets = [set(c) for c in clusters_qcc]
        self.assertIn({"A", "B", "C"}, qcc_sets)
        self.assertIn({"D", "E"}, qcc_sets)

        # Random Walk
        g = nx.Graph()
        g.add_edges_from(edges)
        walk = GraphOGAlgorithms.akef_unbiased_random_walk(g, "A", walk_length=5)
        self.assertEqual(len(walk), 5)
        self.assertEqual(walk[0], "A")

        # Spectral Fiedler Clustering
        bipartite_edges = [("R1", "T1"), ("R1", "T2"), ("R2", "T1"), ("R3", "T3"), ("R4", "T3")]
        brim_clusters = GraphOGAlgorithms.akef_modified_brim_bipartite_clustering(bipartite_edges)
        self.assertTrue(len(brim_clusters) >= 1)

    def test_clique_rank_and_hits_operators(self):
        schema = pa.schema([
            ("id_a", pa.string()),
            ("id_b", pa.string()),
        ])
        tbl = pa.Table.from_arrays(
            [pa.array(["R1", "R2", "R3"]), pa.array(["R2", "R3", "R1"])],
            schema=schema,
        )
        self.ctx.storage.write_parquet(tbl, f"{self.temp_dir}/silver/pairs.parquet")
        from toolkit.catalog.table import TableMetadata
        self.ctx.catalog.register_table(
            TableMetadata(name="pairs", tier="silver", uri=f"{self.temp_dir}/silver/pairs.parquet", schema=schema, row_count=3)
        )

        # CliqueRank
        self.ctx.clique_rank(source_pairs_table="silver.pairs", target_name="gold.clique_ranked")
        cr_table = self.ctx.read_table("gold.clique_ranked")
        self.assertIn("clique_rank_score", cr_table.column_names)

        # HITS
        self.ctx.hits(source_pairs_table="silver.pairs", target_name="gold.hits_scores")
        hits_table = self.ctx.read_table("gold.hits_scores")
        self.assertIn("hub_score", hits_table.column_names)
        self.assertIn("authority_score", hits_table.column_names)

    def test_word2vec_fasttext_lstm_xgcn(self):
        # 1. Word2Vec operator
        schema = pa.schema([
            ("RecID", pa.string()),
            ("_staged_tokens", pa.string()),
        ])
        tbl = pa.Table.from_arrays(
            [
                pa.array(["1", "2", "3", "4"]),
                pa.array([
                    "protein kinase binding domain",
                    "kinase domain active site",
                    "binding domain receptor kinase",
                    "cellular membrane receptor binding",
                ]),
            ],
            schema=schema,
        )
        self.ctx.storage.write_parquet(tbl, f"{self.temp_dir}/silver/tokens.parquet")
        from toolkit.catalog.table import TableMetadata
        self.ctx.catalog.register_table(
            TableMetadata(name="tokens", tier="silver", uri=f"{self.temp_dir}/silver/tokens.parquet", schema=schema, row_count=4)
        )

        meta_w2v = self.ctx.embed_word2vec(source_table="silver.tokens", target_name="gold.w2v_embeds", dimension=16)
        w2v_table = self.ctx.read_table("gold.w2v_embeds")
        self.assertIn("embedding", w2v_table.column_names)
        self.assertEqual(len(w2v_table["embedding"][0].as_py()), 16)

        # 2. FastText subword extraction
        subwords = FastTextOperator.extract_subwords("kinase", min_n=3, max_n=4)
        self.assertIn("<ki", subwords)
        self.assertIn("kin", subwords)

        # 3. LSTMDataBuilder
        vocab = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        lstm_builder = LSTMDataBuilder(sequences=["ACDEFGHIKLM", "DEFGHIKLMNP"], seq_len=4, step=2, char_to_int=vocab)
        self.assertGreater(len(lstm_builder), 0)
        x_sample, y_sample = lstm_builder[0]
        self.assertEqual(len(x_sample), 4)

        # 4. XGCNDataBuilder
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        node_feats = XGCNDataBuilder.prepare_node_features(adj, feature_type="degree_svd", dim=8)
        self.assertEqual(node_feats.shape, (3, 8))

    def test_error_lookup_and_benchmark_loader(self):
        # 1. ConfigER
        config = ConfigER(input_file_name="test.txt", lsh_permutations=64)
        self.assertEqual(config.lsh_permutations, 64)

        # 2. Benchmark CSV
        csv_path = os.path.join(self.temp_dir, "benchmark.csv")
        with open(csv_path, "w", encoding="utf-8") as fp:
            fp.write("id,title,authors,year\n")
            fp.write("1,Data Integration,Smith,2020\n")
            fp.write("2,Entity Resolution,Jones,2021\n")

        self.ctx.load_benchmark(benchmark_name="dblp_acm", source_path=csv_path, target_name="bronze.bench")
        bench_tbl = self.ctx.read_table("bronze.bench")
        self.assertEqual(bench_tbl.num_rows, 2)
        self.assertIn("title", bench_tbl.column_names)

        # 3. ErrorLookupProfiler
        truth_path = os.path.join(self.temp_dir, "truth.csv")
        with open(truth_path, "w", encoding="utf-8") as fp:
            fp.write("1,CLUSTER_A\n")
            fp.write("2,CLUSTER_B\n")

        # Fake resolved table
        schema = pa.schema([
            ("RecID", pa.string()),
            ("canonical_entity_id", pa.string()),
        ])
        resolved_tbl = pa.Table.from_arrays(
            [pa.array(["1", "2"]), pa.array(["ENT_001", "ENT_001"])],  # Forced false positive!
            schema=schema,
        )
        self.ctx.storage.write_parquet(resolved_tbl, f"{self.temp_dir}/gold/resolved.parquet")
        from toolkit.catalog.table import TableMetadata
        self.ctx.catalog.register_table(
            TableMetadata(name="resolved", tier="gold", uri=f"{self.temp_dir}/gold/resolved.parquet", schema=schema, row_count=2)
        )

        diag = self.ctx.profile_error_lookup(target_table="gold.resolved", truth_file=truth_path)
        self.assertEqual(diag["false_positive_count"], 1)
        self.assertEqual(diag["true_positive_count"], 0)


if __name__ == "__main__":
    unittest.main()
