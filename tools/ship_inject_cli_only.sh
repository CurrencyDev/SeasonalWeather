#!/usr/bin/env bash
set -euo pipefail

SRC="/usr/local/bin/seasonalweather-inject"
REPO_ROOT="$(pwd)"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC not found"
  exit 1
fi

mkdir -p "$REPO_ROOT/seasonalweather/cli"
touch "$REPO_ROOT/seasonalweather/cli/__init__.py"

cp -f "$SRC" "$REPO_ROOT/seasonalweather/cli/inject_tool.py"

python3 - <<'PY'
from pathlib import Path
import re
p = Path("seasonalweather/cli/inject_tool.py")
s = p.read_text(encoding="utf-8", errors="replace")
s = re.sub(r"\A#![^\n]*\n", "", s)
s = re.sub(
    r'\n?APP_DIR\s*=\s*".*?"\s*\nif\s+APP_DIR\s+not\s+in\s+sys\.path:\s*\n\s*sys\.path\.insert\(0,\s*APP_DIR\)\s*\n',
    "\n",
    s,
    flags=re.MULTILINE,
)
p.write_text(s.rstrip() + "\n", encoding="utf-8")
PY

cat > "$REPO_ROOT/seasonalweather/cli/inject.py" <<'PY'
import runpy
def main() -> None:
    runpy.run_module("seasonalweather.cli.inject_tool", run_name="__main__")
PY

echo "Done. Added:"
echo "  seasonalweather/cli/inject_tool.py"
echo "  seasonalweather/cli/inject.py"
echo ""
echo "NOTE: To make `seasonalweather-inject` a real command for users, add an entrypoint in your packaging later."
