import os
import re
import math
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pyarrow as pa

from src.operators.base import BaseOperator
from src.catalog.table import TableMetadata


# ==============================================================================
# FAST ZERO-DEPENDENCY XLSX TABLE READER
# ==============================================================================

def read_xlsx_table(path: str, max_rows: Optional[int] = None) -> List[List[str]]:
    """
    Fast zero-dependency reader for .xlsx files using standard library zipfile and xml.
    Falls back gracefully if openpyxl or pandas is installed, or parses XML directly.
    """
    if not os.path.exists(path):
        return []

    try:
        with zipfile.ZipFile(path) as z:
            # 1. Parse shared strings table if present
            strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in tree.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                    # Check for direct <t> or nested runs <r><t>
                    text_parts = [t.text for t in si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t") if t.text]
                    strings.append("".join(text_parts))

            # 2. Parse sheet1
            sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
            rows = []
            for i, row in enumerate(sheet.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row")):
                row_vals = []
                for c in row.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                    t_attr = c.get("t")
                    v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if v is not None and v.text is not None:
                        val = v.text
                        if t_attr == "s" and strings:
                            idx = int(val)
                            val = strings[idx] if idx < len(strings) else val
                        row_vals.append(val)
                    else:
                        row_vals.append("")
                rows.append(row_vals)
                if max_rows and len(rows) >= max_rows:
                    break
            return rows
    except Exception:
        pass

    # Fallback to pandas if installed
    try:
        import pandas as pd
        df = pd.read_excel(path)
        header = list(df.columns)
        values = df.astype(str).values.tolist()
        return [header] + values
    except Exception:
        return []


# ==============================================================================
# DEFAULT POPULATION DISTRIBUTIONS (Fallback when Excel seed files not present)
# ==============================================================================

DEFAULT_LAST_NAMES = [
    ("SMITH", 2442977), ("JOHNSON", 1932812), ("WILLIAMS", 1625252), ("BROWN", 1437026),
    ("JONES", 1425470), ("GARCIA", 1166120), ("MILLER", 1161437), ("DAVIS", 1116357),
    ("RODRIGUEZ", 1094924), ("MARTINEZ", 1060159), ("HERNANDEZ", 1043281), ("LOPEZ", 874523),
    ("GONZALEZ", 841025), ("WILSON", 801882), ("ANDERSON", 784404), ("THOMAS", 756142),
    ("TAYLOR", 751209), ("MOORE", 724374), ("JACKSON", 708099), ("MARTIN", 672711),
    ("LEE", 605860), ("PEREZ", 588506), ("THOMPSON", 555675), ("WHITE", 545620),
    ("HARRIS", 543075), ("SANCHEZ", 527552), ("CLARK", 512210), ("RAMIREZ", 511650),
    ("LEWIS", 509933), ("ROBINSON", 508272), ("WALKER", 488755), ("YOUNG", 484447),
    ("ALLEN", 482607), ("KING", 465822), ("WRIGHT", 458980), ("SCOTT", 439530),
    ("TORRES", 437813), ("NGUYEN", 437645), ("HILL", 434427), ("FLORES", 433969),
    ("GREEN", 430182), ("ADAMS", 427865), ("NELSON", 424958), ("BAKER", 419586),
    ("HALL", 414008), ("RIVERA", 391440), ("CAMPBELL", 386157), ("MITCHELL", 384486),
    ("CARTER", 376963), ("ROBERTS", 376774),
]

DEFAULT_FIRST_NAMES_F = [
    ("EMMA", 19738), ("OLIVIA", 18632), ("AVA", 15738), ("ISABELLA", 15574),
    ("SOPHIA", 15231), ("CHARLOTTE", 13030), ("MIA", 12642), ("AMELIA", 12301),
    ("HARPER", 10582), ("EVELYN", 10376), ("ABIGAIL", 9793), ("EMILY", 8656),
    ("ELIZABETH", 8513), ("MILA", 8126), ("ELLA", 8055), ("AVERY", 8053),
    ("SOFIA", 7621), ("CAMILA", 7473), ("ARIA", 7373), ("SCARLETT", 7131),
    ("VICTORIA", 7089), ("MADISON", 7036), ("LUNA", 6897), ("GRACE", 6724),
    ("CHLOE", 6654), ("PENELOPE", 6633), ("LAYLA", 6474), ("RILEY", 6328),
    ("ZOEY", 5899), ("NORA", 5873), ("LILY", 5624), ("ELEANOR", 5610),
    ("HANNAH", 5590), ("LILLIAN", 5576), ("ADDISON", 5530), ("AUBREY", 5434),
    ("ELLIE", 5334), ("STELLA", 5296), ("NATALIE", 5258), ("ZOE", 5125),
    ("LEAH", 5084), ("HAZEL", 4956), ("VIOLET", 4867), ("AURORA", 4794),
    ("SAVANNAH", 4771), ("AUDREY", 4725), ("BROOKLYN", 4684), ("BELLA", 4670),
    ("CLAIRE", 4647), ("SKYLAR", 4639),
]

DEFAULT_FIRST_NAMES_M = [
    ("LIAM", 19837), ("NOAH", 18267), ("WILLIAM", 14516), ("JAMES", 13525),
    ("OLIVER", 13389), ("BENJAMIN", 13273), ("ELIJAH", 13028), ("LUCAS", 12585),
    ("MASON", 12435), ("LOGAN", 12352), ("ALEXANDER", 11299), ("ETHAN", 10924),
    ("JACOB", 10852), ("MICHAEL", 10571), ("DANIEL", 10469), ("HENRY", 10217),
    ("JACKSON", 10090), ("SEBASTIAN", 9969), ("AIDEN", 9678), ("MATTHEW", 9665),
    ("SAMUEL", 9555), ("DAVID", 9505), ("JOSEPH", 9427), ("CARTER", 9193),
    ("OWEN", 9070), ("WYATT", 8946), ("JOHN", 8834), ("JACK", 8744),
    ("LUKE", 8724), ("JAYDEN", 8540), ("DYLAN", 8408), ("GRAYSON", 8252),
    ("LEVI", 8235), ("ISAAC", 7990), ("GABRIEL", 7847), ("JULIAN", 7731),
    ("MATEO", 7635), ("ANTHONY", 7604), ("JAXON", 7591), ("LINCOLN", 7531),
    ("JOSHUA", 7425), ("CHRISTOPHER", 7393), ("ANDREW", 7356), ("THEODORE", 7312),
    ("CALEB", 7268), ("RYAN", 7264), ("ASHER", 7149), ("NATHAN", 7122),
    ("THOMAS", 7058), ("LEO", 6980),
]

DEFAULT_NICKNAMES = {
    "WILLIAM": ["BILL", "WILL", "BILLY", "WILLIE", "LIAM"],
    "ROBERT": ["BOB", "BOBBY", "ROB", "BERT"],
    "JAMES": ["JIM", "JIMMY", "JEM"],
    "JOHN": ["JACK", "JOHNNY", "IAN"],
    "MICHAEL": ["MIKE", "MIKEY", "MICK"],
    "ELIZABETH": ["BETH", "LIZ", "BETTY", "ELIZA", "LIZZY"],
    "MARGARET": ["MEG", "PEGGY", "MAGGIE", "MARGE"],
    "CATHERINE": ["CATHY", "KATE", "KATIE", "KITTY"],
    "THOMAS": ["TOM", "TOMMY"],
    "RICHARD": ["DICK", "RICK", "RICH", "RICKY"],
    "EDWARD": ["ED", "EDDIE", "NED", "TED"],
    "CHARLES": ["CHARLIE", "CHUCK"],
    "ALEXANDER": ["ALEX", "ALEC", "SANDY"],
    "CHRISTOPHER": ["CHRIS", "KIT"],
    "DANIEL": ["DAN", "DANNY"],
    "JOSEPH": ["JOE", "JOEY"],
    "BENJAMIN": ["BEN", "BENNY"],
    "SAMUEL": ["SAM", "SAMMY"],
    "ANTHONY": ["TONY"],
    "PATRICIA": ["PAT", "PATTY", "TRISH"],
    "JENNIFER": ["JEN", "JENNY"],
    "BARBARA": ["BARB", "BARBIE"],
}

DEFAULT_ADDRESSES = [
    ("14646", "RANCHERO RD", "HESPERIA", "CA", "92345"),
    ("517", "N MOUNTAIN AVE # 202", "UPLAND", "CA", "91786"),
    ("2731", "PONKAN MEADOW DR", "APOPKA", "FL", "32712"),
    ("237", "SAINT TROPEZ LN", "LINCOLN", "CA", "95648"),
    ("969", "PO BOX", "ALTURAS", "CA", "96101"),
    ("1204", "WILLOW CREEK RD", "LITTLE ROCK", "AR", "72211"),
    ("742", "EVERGREEN TERRACE", "SPRINGFIELD", "OR", "97477"),
    ("1060", "W ADDISON ST", "CHICAGO", "IL", "60613"),
    ("350", "FIFTH AVE", "NEW YORK", "NY", "10118"),
    ("221", "BAKER ST", "BOSTON", "MA", "02115"),
    ("1600", "PENNSYLVANIA AVE NW", "WASHINGTON", "DC", "20500"),
    ("420", "MAIN ST # 4B", "SEATTLE", "WA", "98101"),
    ("88", "COLUMBUS BLVD", "SAN FRANCISCO", "CA", "94133"),
    ("1301", "CONGRESS AVE", "AUSTIN", "TX", "78701"),
    ("500", "PEACHTREE ST NE", "ATLANTA", "GA", "30308"),
]


# ==============================================================================
# SEED DATA GENERATOR
# ==============================================================================

class SyntheticSeedGenerator:
    """
    Synthesizes clean demographic population seeds matching SOG mimic (all).R.
    Uses cumulative frequency sampling over surnames, first names by gender, addresses,
    and synthetic identifiers (SSN, DOB, credit card, phone).
    """

    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir = seed_dir
        self.last_names, self.last_name_weights = self._load_last_names()
        self.first_names_f, self.first_names_f_weights = self._load_first_names(gender="F")
        self.first_names_m, self.first_names_m_weights = self._load_first_names(gender="M")
        self.addresses = self._load_addresses()
        self.nicknames = self._load_nicknames()

    def _load_last_names(self) -> Tuple[List[str], np.ndarray]:
        if self.seed_dir:
            p = os.path.join(self.seed_dir, "LastName_Top.xlsx")
            rows = read_xlsx_table(p)
            if len(rows) > 1:
                names = [r[0].strip().upper() for r in rows[1:] if r and len(r) >= 2 and r[0].strip()]
                weights = [float(r[1]) if r[1].isdigit() else 1.0 for r in rows[1:] if r and len(r) >= 2 and r[0].strip()]
                w_arr = np.array(weights, dtype=np.float64)
                return names, w_arr / np.sum(w_arr)
        names = [x[0] for x in DEFAULT_LAST_NAMES]
        weights = np.array([x[1] for x in DEFAULT_LAST_NAMES], dtype=np.float64)
        return names, weights / np.sum(weights)

    def _load_first_names(self, gender: str) -> Tuple[List[str], np.ndarray]:
        if self.seed_dir:
            p = os.path.join(self.seed_dir, "FirstName_Top.xlsx")
            rows = read_xlsx_table(p)
            if len(rows) > 1:
                filtered = [r for r in rows[1:] if len(r) >= 3 and r[1].strip().upper() == gender]
                if filtered:
                    names = [r[0].strip().upper() for r in filtered]
                    weights = [float(r[2]) if r[2].isdigit() else 1.0 for r in filtered]
                    w_arr = np.array(weights, dtype=np.float64)
                    return names, w_arr / np.sum(w_arr)
        defaults = DEFAULT_FIRST_NAMES_F if gender == "F" else DEFAULT_FIRST_NAMES_M
        names = [x[0] for x in defaults]
        weights = np.array([x[1] for x in defaults], dtype=np.float64)
        return names, weights / np.sum(weights)

    def _load_addresses(self) -> List[Tuple[str, str, str, str, str]]:
        if self.seed_dir:
            p = os.path.join(self.seed_dir, "AddressList.xlsx")
            rows = read_xlsx_table(p, max_rows=5000)
            if len(rows) > 1:
                addrs = []
                for r in rows[1:]:
                    if len(r) >= 6:
                        # Street number, street name, city, state, zip
                        num = r[1].strip()
                        street = " ".join([x.strip() for x in r[2:len(r)-3] if x.strip()])
                        city = r[-3].strip()
                        state = r[-2].strip()
                        zip_code = r[-1].strip()
                        if num and city and state:
                            addrs.append((num, street, city, state, zip_code))
                if addrs:
                    return addrs
        return DEFAULT_ADDRESSES

    def _load_nicknames(self) -> Dict[str, List[str]]:
        if self.seed_dir:
            p = os.path.join(self.seed_dir, "nicknames.xlsx")
            rows = read_xlsx_table(p)
            if len(rows) > 1:
                nmap = defaultdict(list)
                for r in rows[1:]:
                    if len(r) >= 3:
                        name = r[1].strip().upper()
                        alt = r[2].strip().upper()
                        if name and alt:
                            nmap[name].append(alt)
                if nmap:
                    return dict(nmap)
        return DEFAULT_NICKNAMES

    def generate_seed_records(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        records = []

        genders = rng.choice(["F", "M"], size=n, p=[0.5, 0.5])
        lnames = rng.choice(self.last_names, size=n, p=self.last_name_weights)

        # First names by gender
        f_idx = np.where(genders == "F")[0]
        m_idx = np.where(genders == "M")[0]

        fnames = np.empty(n, dtype=object)
        if len(f_idx) > 0:
            fnames[f_idx] = rng.choice(self.first_names_f, size=len(f_idx), p=self.first_names_f_weights)
        if len(m_idx) > 0:
            fnames[m_idx] = rng.choice(self.first_names_m, size=len(m_idx), p=self.first_names_m_weights)

        addr_indices = rng.choice(len(self.addresses), size=n)

        for i in range(n):
            rec_id = f"SEED_{i + 1:07d}"
            num, st, city, state, zip_c = self.addresses[addr_indices[i]]
            full_addr = f"{num} {st}, {city}, {state}, {zip_c}".strip()

            # Generate synthetic identifiers
            ssn = f"{rng.randint(100, 999):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"
            year = rng.randint(1945, 2005)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            dob = f"{year:04d}-{month:02d}-{day:02d}"
            cc = f"{rng.randint(4000, 4999):04d}-{rng.randint(1000, 9999):04d}-{rng.randint(1000, 9999):04d}-{rng.randint(1000, 9999):04d}"
            phone = f"({rng.randint(200, 999):03d}) {rng.randint(200, 999):03d}-{rng.randint(1000, 9999):04d}"

            records.append({
                "recid": rec_id,
                "idtruth": str(i + 1),
                "fname": str(fnames[i]),
                "lname": str(lnames[i]),
                "gender": str(genders[i]),
                "streetnbr": num,
                "address": st,
                "city": city,
                "state": state,
                "zip": zip_c,
                "address_full": full_addr,
                "ssn": ssn,
                "dob": dob,
                "credit_card_num": cc,
                "homephone": phone,
            })
        return records


# ==============================================================================
# CONTROLLED ERROR CORRUPTION FUNCTIONS (From SOG mimic (all).R)
# ==============================================================================

class SyntheticErrorCorruptor:
    """
    Executes the 16 systematic error corruption operations from src/SOG mimic (all).R,
    logging all modifications into an error lookup table.
    """

    @staticmethod
    def duplicate_rows(
        records: List[Dict[str, Any]],
        duplication_factor: float = 0.25,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Generates duplicate records simulating clustered real-world entities.
        Uses exponential / geometric repetition rate matching R's `rexp(n, 0.05)`.
        """
        rng = np.random.RandomState(seed)
        duplicated = []
        for r in records:
            # Poisson/exponential duplicate count
            reps = int(rng.exponential(duplication_factor)) + 1
            for rep in range(reps):
                new_r = dict(r)
                duplicated.append(new_r)
        return duplicated

    @staticmethod
    def swap_cols(
        records: List[Dict[str, Any]],
        n_errors: int,
        col1: str = "fname",
        col2: str = "lname",
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 2: Swaps values between two attributes (e.g. First Name and Last Name)."""
        rng = np.random.RandomState(seed)
        n = len(records)
        n_errors = min(n_errors, n)
        selected_idx = rng.choice(n, size=n_errors, replace=False)
        lookup = []

        for idx in selected_idx:
            r = records[idx]
            v1 = str(r.get(col1, ""))
            v2 = str(r.get(col2, ""))
            r[col1] = v2
            r[col2] = v1
            lookup.append({
                "recid": r["recid"],
                "error_type": "swap_cols",
                "column": f"{col1}, {col2}",
                "before": f"{v1}, {v2}",
                "after": f"{v2}, {v1}",
            })
        return records, lookup

    @staticmethod
    def remove_hyphens(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Steps 3-6: Removes hyphens from SSN, DOB, credit card, or homephone."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if "-" in str(r.get(col, ""))]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            after = before.replace("-", "")
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "remove_hyphens",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def remove_parentheses(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str = "homephone",
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 7: Strips parentheses from phone numbers."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if "(" in str(r.get(col, "")) or ")" in str(r.get(col, ""))]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            after = before.replace("(", "").replace(")", "")
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "remove_parentheses",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def transpose_ch(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Steps 9, 11-13: Transposes two adjacent characters (typographical error)."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if len(str(r.get(col, ""))) >= 2]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            pos = rng.randint(0, len(before) - 1)
            ch_list = list(before)
            ch_list[pos], ch_list[pos + 1] = ch_list[pos + 1], ch_list[pos]
            after = "".join(ch_list)
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "transpose_ch",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def rep_onech(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Steps 8, 14, 17: Randomly repeats / doubles one character in a name or string."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if len(str(r.get(col, ""))) >= 1]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            pos = rng.randint(0, len(before))
            ch = before[pos]
            after = before[:pos] + ch + before[pos:]
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "rep_onech",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def initials(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str = "fname",
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 10: Truncates full first name to initial only."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if len(str(r.get(col, ""))) > 1]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            after = before[0]
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "initials",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def married(
        records: List[Dict[str, Any]],
        n_errors: int,
        lname_col: str = "lname",
        gender_col: str = "gender",
        new_last_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 15: Simulates female marital surname change."""
        rng = np.random.RandomState(seed)
        candidates = [i for i, r in enumerate(records) if str(r.get(gender_col, "")).upper() == "F"]
        n_errors = min(n_errors, len(candidates))
        if n_errors == 0:
            return records, []

        ln_pool = new_last_names or [x[0] for x in DEFAULT_LAST_NAMES]
        selected_idx = rng.choice(candidates, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(lname_col, ""))
            after = rng.choice(ln_pool)
            r[lname_col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "married",
                "column": lname_col,
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def create_twins(
        records: List[Dict[str, Any]],
        n_errors: int,
        fname_col: str = "fname",
        first_names_pool: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 12: Injects twin identities sharing surname, address, and DOB but with distinct first names."""
        rng = np.random.RandomState(seed)
        n = len(records)
        n_errors = min(n_errors, n)
        if n_errors == 0:
            return records, []

        fn_pool = first_names_pool or [x[0] for x in DEFAULT_FIRST_NAMES_M + DEFAULT_FIRST_NAMES_F]
        selected_idx = rng.choice(n, size=n_errors, replace=False)
        lookup = []

        max_truth_id = max([int(r["idtruth"]) for r in records if str(r["idtruth"]).isdigit()] or [n])

        for idx in selected_idx:
            base_r = records[idx]
            twin_r = dict(base_r)
            twin_recid = f"TWIN_{len(records) + len(lookup) + 1:07d}"
            max_truth_id += 1

            old_fname = str(base_r.get(fname_col, ""))
            # Pick a twin name starting with the same initial if possible
            same_initial = [w for w in fn_pool if w.startswith(old_fname[:1]) and w != old_fname]
            new_fname = rng.choice(same_initial) if same_initial else rng.choice(fn_pool)

            twin_r["recid"] = twin_recid
            twin_r[fname_col] = new_fname
            twin_r["ssn"] = f"{rng.randint(100, 999):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"
            twin_r["idtruth"] = str(max_truth_id)  # Distinct ground truth entity!

            records.append(twin_r)
            lookup.append({
                "recid": twin_recid,
                "error_type": "create_twins",
                "column": "all columns",
                "before": f"{base_r['recid']}, {old_fname}",
                "after": f"{twin_recid}, {new_fname}",
            })
        return records, lookup

    @staticmethod
    def change_address(
        records: List[Dict[str, Any]],
        n_errors: int,
        addresses_pool: Optional[List[Tuple[str, str, str, str, str]]] = None,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 16: Simulates residential address change (relocation)."""
        rng = np.random.RandomState(seed)
        n = len(records)
        n_errors = min(n_errors, n)
        if n_errors == 0:
            return records, []

        pool = addresses_pool or DEFAULT_ADDRESSES
        selected_idx = rng.choice(n, size=n_errors, replace=False)
        lookup = []

        for idx in selected_idx:
            r = records[idx]
            before = str(r.get("address_full", ""))
            num, st, city, state, zip_c = pool[rng.randint(len(pool))]
            after = f"{num} {st}, {city}, {state}, {zip_c}"
            r["streetnbr"] = num
            r["address"] = st
            r["city"] = city
            r["state"] = state
            r["zip"] = zip_c
            r["address_full"] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "change_address",
                "column": "address_full",
                "before": before,
                "after": after,
            })
        return records, lookup

    @staticmethod
    def random_delete(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Steps 18-23: Missing value injection."""
        rng = np.random.RandomState(seed)
        eligible = [i for i, r in enumerate(records) if str(r.get(col, "")).strip() != ""]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, ""))
            r[col] = ""
            lookup.append({
                "recid": r["recid"],
                "error_type": "random_delete",
                "column": col,
                "before": before,
                "after": "",
            })
        return records, lookup

    @staticmethod
    def to_nicknames(
        records: List[Dict[str, Any]],
        n_errors: int,
        col: str = "fname",
        nicknames_dict: Optional[Dict[str, List[str]]] = None,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 14: Substitutes formal given names with familiar nicknames."""
        rng = np.random.RandomState(seed)
        ndict = nicknames_dict or DEFAULT_NICKNAMES
        eligible = [i for i, r in enumerate(records) if str(r.get(col, "")).upper() in ndict]
        n_errors = min(n_errors, len(eligible))
        if n_errors == 0:
            return records, []

        selected_idx = rng.choice(eligible, size=n_errors, replace=False)
        lookup = []
        for idx in selected_idx:
            r = records[idx]
            before = str(r.get(col, "")).upper()
            after = rng.choice(ndict[before])
            r[col] = after
            lookup.append({
                "recid": r["recid"],
                "error_type": "to_nicknames",
                "column": col,
                "before": before,
                "after": after,
            })
        return records, lookup


# ==============================================================================
# INTEGRATED LAKEHOUSE MEDALLION OPERATOR
# ==============================================================================

class SyntheticDemographicsGeneratorOperator(BaseOperator):
    """
    Gold/Bronze Lakehouse Operator:
    Synthesized from src/SOG mimic (all).R, src/Install.R, and src/Test.R.
    Generates synthetic population datasets with controlled real-world error corruptions,
    producing Bronze raw records, Gold ground-truth links, and Gold diagnostic error tables.
    """

    def __init__(
        self,
        name: str = "synthetic_demographics_generator",
        target_name: str = "bronze.synthetic_demographics",
        truth_target_name: str = "gold.synthetic_truth",
        error_lookup_target_name: str = "gold.synthetic_error_lookup",
        n_seed_records: int = 1000,
        duplication_factor: float = 0.25,
        seed_dir: Optional[str] = "data/entity_resolution/synthetic_demographics",
        corruption_config: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        super().__init__(name)
        self.target_name = target_name
        self.truth_target_name = truth_target_name
        self.error_lookup_target_name = error_lookup_target_name
        self.n_seed_records = n_seed_records
        self.duplication_factor = duplication_factor
        self.seed_dir = seed_dir
        self.corruption_config = corruption_config or {}
        self.seed = seed

    def execute(self, context: Any, **kwargs) -> Dict[str, TableMetadata]:
        fs_manager = context.storage
        n_seeds = kwargs.get("n_seed_records", self.n_seed_records)
        seed_dir = kwargs.get("seed_dir", self.seed_dir)
        dup_factor = kwargs.get("duplication_factor", self.duplication_factor)
        seed_val = kwargs.get("seed", self.seed)

        # 1. Generate clean seed records
        seed_gen = SyntheticSeedGenerator(seed_dir=seed_dir)
        clean_records = seed_gen.generate_seed_records(n=n_seeds, seed=seed_val)

        # 2. Step 1: Duplicate records
        records = SyntheticErrorCorruptor.duplicate_rows(clean_records, duplication_factor=dup_factor, seed=seed_val)

        # Assign unique recid (Axxxxx)
        for i, r in enumerate(records):
            r["recid"] = f"A{805296 + i}"

        # 3. Apply error corruption pipeline
        full_lookup = []
        corruptor = SyntheticErrorCorruptor()

        # Ratio-based default error scale
        total_recs = len(records)
        scale = total_recs / 10000.0

        c_cfg = self.corruption_config
        n_swap = c_cfg.get("swap_cols", max(1, int(100 * scale)))
        n_hyphen = c_cfg.get("remove_hyphens", max(1, int(500 * scale)))
        n_paren = c_cfg.get("remove_parentheses", max(1, int(500 * scale)))
        n_rep = c_cfg.get("rep_onech", max(1, int(150 * scale)))
        n_trans = c_cfg.get("transpose_ch", max(1, int(250 * scale)))
        n_initials = c_cfg.get("initials", max(1, int(250 * scale)))
        n_married = c_cfg.get("married", max(1, int(150 * scale)))
        n_twins = c_cfg.get("create_twins", max(1, int(50 * scale)))
        n_addr = c_cfg.get("change_address", max(1, int(250 * scale)))
        n_nick = c_cfg.get("to_nicknames", max(1, int(200 * scale)))
        n_delete = c_cfg.get("random_delete", max(1, int(300 * scale)))

        # Step 2: swap fname & lname
        records, lk = corruptor.swap_cols(records, n_swap, "fname", "lname", seed=seed_val)
        full_lookup.extend(lk)

        # Steps 3-6: remove hyphens
        for col in ["ssn", "dob", "credit_card_num", "homephone"]:
            records, lk = corruptor.remove_hyphens(records, n_hyphen, col, seed=seed_val)
            full_lookup.extend(lk)

        # Step 7: remove parentheses
        records, lk = corruptor.remove_parentheses(records, n_paren, "homephone", seed=seed_val)
        full_lookup.extend(lk)

        # Step 8: repeat character in fname
        records, lk = corruptor.rep_onech(records, n_rep, "fname", seed=seed_val)
        full_lookup.extend(lk)

        # Step 9: transpose ssn
        records, lk = corruptor.transpose_ch(records, n_trans, "ssn", seed=seed_val)
        full_lookup.extend(lk)

        # Step 10: keep initials
        records, lk = corruptor.initials(records, n_initials, "fname", seed=seed_val)
        full_lookup.extend(lk)

        # Step 11-13: transpose dob, credit_card_num, homephone
        for col in ["dob", "credit_card_num", "homephone"]:
            records, lk = corruptor.transpose_ch(records, n_trans, col, seed=seed_val)
            full_lookup.extend(lk)

        # Step 14: repeat char in lname
        records, lk = corruptor.rep_onech(records, n_rep, "lname", seed=seed_val)
        full_lookup.extend(lk)

        # Step 15: female marital surname change
        records, lk = corruptor.married(records, n_married, "lname", "gender", new_last_names=seed_gen.last_names, seed=seed_val)
        full_lookup.extend(lk)

        # Step 16: change address
        records, lk = corruptor.change_address(records, n_addr, addresses_pool=seed_gen.addresses, seed=seed_val)
        full_lookup.extend(lk)

        # Step 17: nicknames
        records, lk = corruptor.to_nicknames(records, n_nick, "fname", nicknames_dict=seed_gen.nicknames, seed=seed_val)
        full_lookup.extend(lk)

        # Step 18: twins
        records, lk = corruptor.create_twins(records, n_twins, "fname", first_names_pool=seed_gen.first_names_m, seed=seed_val)
        full_lookup.extend(lk)

        # Steps 19-23: random deletes
        for col in ["gender", "ssn", "dob", "credit_card_num", "homephone"]:
            records, lk = corruptor.random_delete(records, n_delete, col, seed=seed_val)
            full_lookup.extend(lk)

        # 4. Shuffle records
        rng = np.random.RandomState(seed_val)
        rng.shuffle(records)

        # 5. Build Arrow Tables
        rec_ids = [r["recid"] for r in records]
        id_truths = [r["idtruth"] for r in records]
        fnames = [r["fname"] for r in records]
        lnames = [r["lname"] for r in records]
        genders = [r["gender"] for r in records]
        streetnbrs = [r["streetnbr"] for r in records]
        addresses = [r["address"] for r in records]
        cities = [r["city"] for r in records]
        states = [r["state"] for r in records]
        zips = [r["zip"] for r in records]
        ssns = [r["ssn"] for r in records]
        dobs = [r["dob"] for r in records]
        ccs = [r["credit_card_num"] for r in records]
        phones = [r["homephone"] for r in records]

        records_arrow = pa.Table.from_arrays(
            [
                pa.array(rec_ids, type=pa.string()),
                pa.array(fnames, type=pa.string()),
                pa.array(lnames, type=pa.string()),
                pa.array(genders, type=pa.string()),
                pa.array(streetnbrs, type=pa.string()),
                pa.array(addresses, type=pa.string()),
                pa.array(cities, type=pa.string()),
                pa.array(states, type=pa.string()),
                pa.array(zips, type=pa.string()),
                pa.array(ssns, type=pa.string()),
                pa.array(dobs, type=pa.string()),
                pa.array(ccs, type=pa.string()),
                pa.array(phones, type=pa.string()),
            ],
            names=[
                "recid", "fname", "lname", "gender", "streetnbr", "address",
                "city", "state", "zip", "ssn", "dob", "credit_card_num", "homephone"
            ],
        )

        truth_arrow = pa.Table.from_arrays(
            [pa.array(rec_ids, type=pa.string()), pa.array(id_truths, type=pa.string())],
            names=["recid", "idtruth"],
        )

        lk_recids = [item["recid"] for item in full_lookup]
        lk_types = [item["error_type"] for item in full_lookup]
        lk_cols = [item["column"] for item in full_lookup]
        lk_befores = [item["before"] for item in full_lookup]
        lk_afters = [item["after"] for item in full_lookup]

        lookup_arrow = pa.Table.from_arrays(
            [
                pa.array(lk_recids, type=pa.string()),
                pa.array(lk_types, type=pa.string()),
                pa.array(lk_cols, type=pa.string()),
                pa.array(lk_befores, type=pa.string()),
                pa.array(lk_afters, type=pa.string()),
            ],
            names=["recid", "error_type", "column", "before", "after"],
        )

        # 6. Save Tables to Medallion Tiers
        clean_target = self.target_name.replace("bronze.", "")
        target_uri = os.path.join(context.lakehouse_root, "bronze", f"{clean_target}.parquet").replace("\\", "/")
        fs_manager.write_parquet(records_arrow, target_uri)

        clean_truth = self.truth_target_name.replace("gold.", "")
        truth_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_truth}.parquet").replace("\\", "/")
        fs_manager.write_parquet(truth_arrow, truth_uri)

        clean_lookup = self.error_lookup_target_name.replace("gold.", "")
        lookup_uri = os.path.join(context.lakehouse_root, "gold", f"{clean_lookup}.parquet").replace("\\", "/")
        fs_manager.write_parquet(lookup_arrow, lookup_uri)

        meta_records = TableMetadata(
            name=clean_target,
            tier="bronze",
            uri=target_uri,
            schema=records_arrow.schema,
            row_count=records_arrow.num_rows,
            properties={"n_seed_records": n_seeds, "error_count": len(full_lookup)},
        )
        context.catalog.register_table(meta_records)

        meta_truth = TableMetadata(
            name=clean_truth,
            tier="gold",
            uri=truth_uri,
            schema=truth_arrow.schema,
            row_count=truth_arrow.num_rows,
            properties={"distinct_entities": len(set(id_truths))},
        )
        context.catalog.register_table(meta_truth)

        meta_lookup = TableMetadata(
            name=clean_lookup,
            tier="gold",
            uri=lookup_uri,
            schema=lookup_arrow.schema,
            row_count=lookup_arrow.num_rows,
            properties={"error_events": len(full_lookup)},
        )
        context.catalog.register_table(meta_lookup)

        return {
            "records": meta_records,
            "truth": meta_truth,
            "error_lookup": meta_lookup,
        }
