import os
import shutil
import tempfile
import pytest
from src.context import IngestionContext
from src.ai.scanner import DatasetScanner
from src.ai.gemini_client import GeminiClient
from src.ai.router import AIIngestionRouter


@pytest.fixture(autouse=True)
def ensure_sample_data():
    data_dir = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data")
    os.makedirs(data_dir, exist_ok=True)
    fasta_p = os.path.join(data_dir, "sample_proteins.fasta")
    if not os.path.exists(fasta_p) or os.path.getsize(fasta_p) < 400:
        with open(fasta_p, "w", encoding="utf-8") as f:
            f.write(""">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
>sp|P01116|RASH_HUMAN GTPase HRas OS=Homo sapiens OX=9606 GN=HRAS PE=1 SV=1
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG
QEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDL
>tr|A0A024R1R8|A0A024R1R8_HUMAN Protein kinase C alpha OS=Homo sapiens
MADVYPANDSTASQDVANRFARKGALRQKNVHEVKDHKFIARFFKQPTFCSHCTDFIWGF
GKQGFQCQVCCFVVHKRCHEFVTFSCPGADKGPDTDDPRSKHKFKIHTYGSPTFCDHCGS
>UniRef50_P12345 Glucose-6-phosphate isomerase OS=Homo sapiens
MAALTRDPQFQKLQQWYREHRSELNLRRLFDANKDRFNHFSLTLNTNHGHILVDYSKNLV
""")
    idmap_p = os.path.join(data_dir, "sample_idmapping.tsv")
    if not os.path.exists(idmap_p):
        with open(idmap_p, "w", encoding="utf-8") as f:
            f.write("P04637\tGene_Name\tTP53\nP04637\tBioGRID\t112233\nP01116\tGene_Name\tHRAS\n")
    biogrid_p = os.path.join(data_dir, "sample_biogrid.tab3.txt")
    if not os.path.exists(biogrid_p):
        with open(biogrid_p, "w", encoding="utf-8") as f:
            f.write("#BioGRID\tID Interactor A\tID Interactor B\tScore\n1\t112233\t102030\t0.95\n")


@pytest.fixture
def temp_lakehouse():
    temp_dir = tempfile.mkdtemp(prefix="test_ai_lakehouse_")
    ctx = IngestionContext(lakehouse_root=temp_dir)
    yield ctx, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_scanner():
    scanner = DatasetScanner()

    # 1. FASTA scan
    fasta_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_proteins.fasta")
    res_fasta = scanner.scan(fasta_p)
    assert res_fasta.filename == "sample_proteins.fasta"
    assert "FASTA_HEADER" in res_fasta.structural_markers
    assert "PROTEIN_OR_NUCLEOTIDE_SEQUENCE" in res_fasta.structural_markers

    # 2. TSV IDMapping scan
    idmap_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_idmapping.tsv")
    res_idmap = scanner.scan(idmap_p)
    assert res_idmap.dominant_delimiter == "\t"
    assert "UNIPROT_IDMAPPING" in res_idmap.structural_markers

    # 3. BioGRID PPI scan
    biogrid_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_biogrid.tab3.txt")
    res_biogrid = scanner.scan(biogrid_p)
    assert "BIOGRID_PPI" in res_biogrid.structural_markers


def test_ai_router_planning():
    router = AIIngestionRouter()

    # Profile FASTA
    fasta_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_proteins.fasta")
    plan_fasta = router.plan(fasta_p)
    assert plan_fasta.domain == "bioinformatics"
    assert plan_fasta.file_format == "fasta"
    assert plan_fasta.suggested_operator == "IngestFastaOperator"
    assert "INGEST FASTA" in plan_fasta.recommended_sql

    # Profile BioGRID
    biogrid_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_biogrid.tab3.txt")
    plan_biogrid = router.plan(biogrid_p)
    assert plan_biogrid.domain == "bioinformatics"
    assert "ppi" in plan_biogrid.file_format
    assert plan_biogrid.suggested_operator == "IngestPPIOperator"


def test_ai_router_execution(temp_lakehouse):
    ctx, root = temp_lakehouse
    router = AIIngestionRouter(context=ctx)

    fasta_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_proteins.fasta")
    plan = router.route_and_execute(fasta_p, target_table="bronze.auto_proteins")

    assert plan.execution_status == "EXECUTED"
    tbl = ctx.read_table("bronze.auto_proteins")
    assert tbl.num_rows > 0
    assert "sequence_id" in tbl.column_names
    assert "sequence" in tbl.column_names


def test_ai_sql_profile_and_route(temp_lakehouse):
    ctx, root = temp_lakehouse
    fasta_p = os.path.join(os.path.dirname(__file__), "test_lakehouse_track3/data/sample_proteins.fasta")

    # SQL: AI PROFILE
    res_prof = ctx.sql(f"AI PROFILE '{fasta_p}';")
    assert res_prof["domain"] == "bioinformatics"
    assert res_prof["file_format"] == "fasta"

    # SQL: AI ROUTE INTO bronze.sql_proteins WITH (auto_execute = true)
    res_route = ctx.sql(f"AI ROUTE '{fasta_p}' INTO bronze.sql_proteins WITH (auto_execute = true);")
    assert res_route["execution_status"] == "EXECUTED"

    tbl = ctx.read_table("bronze.sql_proteins")
    assert tbl.num_rows == 4
    assert ctx.catalog.get_table("bronze.sql_proteins") is not None
