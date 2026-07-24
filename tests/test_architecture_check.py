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
        "SWARCH011",
        "SWARCH012",
    }
    assert any("filesystem mutation" in finding.message for finding in findings)


def test_control_module_has_no_duplicate_job_repository_or_scheduler_authority():
    source = (ROOT / "seasonalweather/control.py").read_text(encoding="utf-8")

    assert "seasonalweather.job_store" not in source
    assert "JobRepository(" not in source
    assert "JobScheduler(" not in source
    assert "sqlite3" not in source


def test_control_and_api_have_no_swwp_or_simulation_authority():
    control = (ROOT / "seasonalweather/control.py").read_text(encoding="utf-8")
    api = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "seasonalweather/api").glob("*.py"))

    for source in (control, api):
        assert "seasonalweather.swwp" not in source
        assert "swwp_simulation" not in source
        assert "SimulatedPeers" not in source
