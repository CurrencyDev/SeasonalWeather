from __future__ import annotations

from tools.quality.governance import ROOT, load_toml, validate_governed_record


def validate() -> list[str]:
    errors: list[str] = []

    baseline = load_toml(ROOT / "quality/baselines.toml")
    checks = baseline.get("checks", {})
    if not isinstance(checks, dict) or not checks:
        errors.append("quality/baselines.toml: checks table is required")
    else:
        for name, raw in checks.items():
            context = f"quality/baselines.toml checks.{name}"
            if not isinstance(raw, dict):
                errors.append(f"{context}: must be a table")
                continue
            errors.extend(validate_governed_record(raw, context=context))
            maximum = raw.get("maximum")
            if not isinstance(maximum, int) or maximum < 0:
                errors.append(f"{context}: maximum must be a non-negative integer")

    exceptions = load_toml(ROOT / "quality/exceptions.toml").get("exceptions", [])
    if not isinstance(exceptions, list):
        errors.append("quality/exceptions.toml: exceptions must be an array of tables")
    else:
        seen: set[tuple[object, object]] = set()
        for index, raw in enumerate(exceptions):
            context = f"quality/exceptions.toml exceptions[{index}]"
            if not isinstance(raw, dict):
                errors.append(f"{context}: must be a table")
                continue
            errors.extend(validate_governed_record(raw, context=context))
            if not raw.get("rule"):
                errors.append(f"{context}: rule is required")
            key = (raw.get("rule"), raw.get("scope"))
            if key in seen:
                errors.append(f"{context}: duplicate rule/scope exception")
            seen.add(key)
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Quality baselines and architecture exceptions are valid and unexpired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
