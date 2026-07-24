from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_GOVERNANCE_FIELDS = {
    "owner",
    "rationale",
    "scope",
    "review_date",
    "removal_condition",
}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def parse_review_date(value: object, *, context: str) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{context}: review_date must use YYYY-MM-DD") from exc
    raise ValueError(f"{context}: review_date is required")


def validate_governed_record(record: dict[str, Any], *, context: str) -> list[str]:
    errors: list[str] = []
    missing = sorted(field for field in REQUIRED_GOVERNANCE_FIELDS if not record.get(field))
    if missing:
        errors.append(f"{context}: missing {', '.join(missing)}")
    try:
        review_date = parse_review_date(record.get("review_date"), context=context)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if review_date < dt.date.today():
            errors.append(f"{context}: review_date {review_date.isoformat()} has expired")
    return errors
