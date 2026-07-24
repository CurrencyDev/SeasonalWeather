from tools.quality.architecture_check import scan
from tools.quality.governance import ROOT, load_toml

FIXTURES = ROOT / "tests/architecture/fixtures"
CONFIG = load_toml(ROOT / "quality/architecture.toml")


def test_valid_architecture_fixture_passes():
    assert scan(FIXTURES / "valid", CONFIG) == []


def test_invalid_architecture_fixture_proves_rules_fail_closed():
    findings = scan(FIXTURES / "invalid", CONFIG)

    assert {finding.rule for finding in findings} >= {
        "SWARCH001",
        "SWARCH003",
        "SWARCH006",
        "SWARCH009",
        "SWARCH010",
    }
    assert any("filesystem mutation" in finding.message for finding in findings)


def test_control_module_has_no_duplicate_job_repository_or_scheduler_authority():
    source = (ROOT / "seasonalweather/control.py").read_text(encoding="utf-8")

    assert "seasonalweather.job_store" not in source
    assert "JobRepository(" not in source
    assert "JobScheduler(" not in source
    assert "sqlite3" not in source
