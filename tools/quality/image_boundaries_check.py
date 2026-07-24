from __future__ import annotations

import datetime as dt

from tools.quality.governance import ROOT, load_toml, parse_review_date


def main() -> int:
    config = load_toml(ROOT / "quality/image-boundaries.toml")
    definitions = [
        path
        for pattern in ("Dockerfile*", "docker-compose*.yml", "docker-compose*.yaml", "compose*.yml", "compose*.yaml")
        for path in ROOT.glob(pattern)
    ]
    if config.get("status") != "not_applicable":
        print("image-boundaries-check: unsupported declaration status")
        return 1
    try:
        review_date = parse_review_date(config.get("review_date"), context="quality/image-boundaries.toml")
    except ValueError:
        review_date = None
    if review_date is None or review_date < dt.date.today():
        print("image-boundaries-check: declaration review_date is missing or expired")
        return 1
    if definitions:
        print("image-boundaries-check: image definitions now exist; replace the P1-01 not-applicable declaration")
        for path in definitions:
            print(path.relative_to(ROOT))
        return 1
    print("image-boundaries-check: no image definitions; governed not-applicable declaration remains valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
