import os
import shutil
import tempfile
import unittest
import pyarrow as pa

from src.context import LakehouseContext
from src.operators.synthetic import (
    read_xlsx_table,
    SyntheticSeedGenerator,
    SyntheticErrorCorruptor,
    SyntheticDemographicsGeneratorOperator,
)


class TestSyntheticDemographicsGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ctx = LakehouseContext(lakehouse_root=self.temp_dir)
        self.seed_dir = "data/entity_resolution/synthetic_demographics"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_xlsx_table(self):
        # Test reading LastName_Top.xlsx if exists
        p = os.path.join(self.seed_dir, "LastName_Top.xlsx")
        if os.path.exists(p):
            rows = read_xlsx_table(p, max_rows=10)
            self.assertGreater(len(rows), 1)
            self.assertEqual(rows[0][0], "Lname")

    def test_seed_generator_fallback_and_custom(self):
        # 1. Fallback generation (no seed dir)
        gen_fallback = SyntheticSeedGenerator(seed_dir=None)
        records_fb = gen_fallback.generate_seed_records(n=20, seed=42)
        self.assertEqual(len(records_fb), 20)
        self.assertIn("fname", records_fb[0])
        self.assertIn("ssn", records_fb[0])
        self.assertIn("credit_card_num", records_fb[0])

        # 2. Generation using seed directory
        gen_custom = SyntheticSeedGenerator(seed_dir=self.seed_dir)
        records_cust = gen_custom.generate_seed_records(n=50, seed=123)
        self.assertEqual(len(records_cust), 50)
        self.assertIn("address_full", records_cust[0])
        self.assertIn(records_cust[0]["gender"], ["F", "M"])

    def test_all_corruption_functions(self):
        gen = SyntheticSeedGenerator(seed_dir=None)
        base_records = gen.generate_seed_records(n=20, seed=42)
        corruptor = SyntheticErrorCorruptor()

        # 1. Duplicate rows
        dupl = corruptor.duplicate_rows(base_records, duplication_factor=0.5, seed=42)
        self.assertGreaterEqual(len(dupl), len(base_records))

        # 2. Swap columns
        recs, lk_swap = corruptor.swap_cols(base_records, n_errors=5, col1="fname", col2="lname", seed=42)
        self.assertEqual(len(lk_swap), 5)
        self.assertEqual(lk_swap[0]["error_type"], "swap_cols")

        # 3. Remove hyphens
        recs, lk_hyphen = corruptor.remove_hyphens(base_records, n_errors=5, col="ssn", seed=42)
        self.assertGreaterEqual(len(lk_hyphen), 1)
        self.assertNotIn("-", lk_hyphen[0]["after"])

        # 4. Remove parentheses
        recs, lk_paren = corruptor.remove_parentheses(base_records, n_errors=5, col="homephone", seed=42)
        self.assertGreaterEqual(len(lk_paren), 1)
        self.assertNotIn("(", lk_paren[0]["after"])

        # 5. Transpose characters
        recs, lk_trans = corruptor.transpose_ch(base_records, n_errors=5, col="dob", seed=42)
        self.assertEqual(len(lk_trans), 5)
        self.assertEqual(lk_trans[0]["error_type"], "transpose_ch")

        # 6. Repeat character
        recs, lk_rep = corruptor.rep_onech(base_records, n_errors=5, col="fname", seed=42)
        self.assertEqual(len(lk_rep), 5)
        self.assertGreater(len(lk_rep[0]["after"]), len(lk_rep[0]["before"]))

        # 7. Initials
        recs, lk_init = corruptor.initials(base_records, n_errors=5, col="fname", seed=42)
        self.assertEqual(len(lk_init), 5)
        self.assertEqual(len(lk_init[0]["after"]), 1)

        # 8. Married name change
        recs, lk_marr = corruptor.married(base_records, n_errors=5, lname_col="lname", gender_col="gender", seed=42)
        self.assertGreaterEqual(len(lk_marr), 1)

        # 9. Twins
        recs, lk_twins = corruptor.create_twins(base_records, n_errors=3, fname_col="fname", seed=42)
        self.assertEqual(len(lk_twins), 3)

        # 10. Change address
        recs, lk_addr = corruptor.change_address(base_records, n_errors=5, seed=42)
        self.assertEqual(len(lk_addr), 5)

        # 11. Random delete (missing value)
        recs, lk_del = corruptor.random_delete(base_records, n_errors=5, col="gender", seed=42)
        self.assertEqual(len(lk_del), 5)
        self.assertEqual(lk_del[0]["after"], "")

        # 12. Nicknames
        recs, lk_nick = corruptor.to_nicknames(base_records, n_errors=5, col="fname", seed=42)
        self.assertGreaterEqual(len(lk_nick), 1)

    def test_end_to_end_lakehouse_synthetic_generation(self):
        # Run via LakehouseContext
        results = self.ctx.generate_synthetic_demographics(
            target_name="bronze.synthetic_test",
            truth_target_name="gold.synthetic_test_truth",
            error_lookup_target_name="gold.synthetic_test_lookup",
            n_seed_records=50,
            duplication_factor=0.3,
            seed_dir=self.seed_dir,
            seed=42,
        )

        self.assertIn("records", results)
        self.assertIn("truth", results)
        self.assertIn("error_lookup", results)

        # Read tables from Medallion storage
        records_tbl = self.ctx.read_table("bronze.synthetic_test")
        truth_tbl = self.ctx.read_table("gold.synthetic_test_truth")
        lookup_tbl = self.ctx.read_table("gold.synthetic_test_lookup")

        self.assertGreaterEqual(records_tbl.num_rows, 50)
        self.assertEqual(records_tbl.num_rows, truth_tbl.num_rows)
        self.assertGreater(lookup_tbl.num_rows, 0)

        # Check required schema
        for col in ["recid", "fname", "lname", "gender", "streetnbr", "address", "city", "state", "zip", "ssn"]:
            self.assertIn(col, records_tbl.column_names)

        # Downstream Lakehouse integration: Stage and Block the synthetic dataset
        self.ctx.stage(
            source_table="bronze.synthetic_test",
            target_name="silver.staged_synthetic",
            key_column="recid",
            feature_columns=["fname", "lname", "streetnbr", "address", "city", "state", "zip"],
        )
        staged_tbl = self.ctx.read_table("silver.staged_synthetic")
        self.assertIn("_staged_tokens", staged_tbl.column_names)

        self.ctx.block(
            source_table="silver.staged_synthetic",
            target_name="silver.blocked_synthetic",
            key_column="recid",
            method="token_inverted_index",
        )
        blocked_tbl = self.ctx.read_table("silver.blocked_synthetic")
        self.assertIn("id_a", blocked_tbl.column_names)
        self.assertIn("id_b", blocked_tbl.column_names)


if __name__ == "__main__":
    unittest.main()
