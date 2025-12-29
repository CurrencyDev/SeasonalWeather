from __future__ import annotations

# =========================================================================================
#      MP"""""`MM                                                       dP              MM'"""'YMM
#      M  mmmmm..M                                                       88              M' .mmm. `M
#      M.      `YM .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88              M  MMMMMooM dP    dP 88d888b. 88d888b. .d8888b. 88d888b. .d8888b. dP    dP
#      MMMMMMM.  M 88ooood8 88'  `88 Y8ooooo. 88'  `88 88'  `88 88'  `88 88              M  MMMMMMMM 88    88 88'  `88 88'  `88 88ooood8 88'  `88 88'  `"" 88    88
#      M. .MMM'  M 88.  ... 88.  .88       88 88.  .88 88    88 88.  .88 88              M. `MMM' .M 88.  .88 88       88       88.  ... 88    88 88.  ... 88.  .88
#      Mb.     .dM `88888P' `88888P8 `88888P' `88888P' dP    dP `88888P8 dP              MM.     .dM `88888P' dP       dP       `88888P' dP    dP `88888P' `8888P88
#      MMMMMMMMMMM                                                Seasonal_Currency      MMMMMMMMMMM                                                            .88
#                                                                                                                                                           d8888P.
# =========================================================================================

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def _parse_iso(s: str) -> Optional[dt.datetime]:
    try:
        # Accept "Z" or offset forms
        s2 = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None


@dataclass
class CapLedger:
    """
    Tiny persistent "seen" ledger to prevent CAP restart spam.

    Keys are typically "{logical_alert_id}|{sent_iso}" so updates still emit once.
    """
    path: Path
    max_age_days: int = 14

    _seen: Dict[str, str] = None  # type: ignore[assignment]
    _loaded: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._seen = {}

        try:
            if not self.path.exists():
                return
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            seen = data.get("seen", {})
            if isinstance(seen, dict):
                self._seen = {str(k): str(v) for k, v in seen.items()}
        except Exception:
            # If ledger corrupt, start fresh (but don't explode your service)
            self._seen = {}

        self.cleanup()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")

        data = {
            "version": 1,
            "updated_utc": _utcnow().isoformat(),
            "seen": self._seen,
        }

        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(str(tmp), str(self.path))

    @staticmethod
    def make_key(alert_id: str, sent: str | None) -> str:
        a = (alert_id or "").strip()
        s = (sent or "").strip()
        return f"{a}|{s}"

    def has(self, key: str) -> bool:
        self._load()
        return key in self._seen

    def mark(self, key: str) -> None:
        self._load()
        self._seen[key] = _utcnow().isoformat()

    def cleanup(self) -> None:
        self._load()
        cutoff = _utcnow() - dt.timedelta(days=max(3, int(self.max_age_days)))
        out: Dict[str, str] = {}
        for k, v in self._seen.items():
            t = _parse_iso(v) or _utcnow()
            if t >= cutoff:
                out[k] = v
        self._seen = out

    def flush(self) -> None:
        self.cleanup()
        self._save()
