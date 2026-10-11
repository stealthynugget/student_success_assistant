from pathlib import Path

from ssa.cli import quality_gate, secret_scan

ROOT = Path(__file__).resolve().parents[1]


def test_no_secrets_in_source_tree():
    assert secret_scan(ROOT) == []


def test_env_files_excluded_from_git_and_docker():
    for name in (".gitignore", ".dockerignore"):
        lines = (ROOT / name).read_text().splitlines()
        assert ".env" in lines, name


def test_quality_gate_passes_on_fixture(built):
    res = quality_gate(run_tests=False)
    assert res["profile"] == "smoke"
    assert res["passed"], [c for c in res["checks"] if not c["passed"]]
