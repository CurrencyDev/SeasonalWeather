#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: tools/release.sh X.Y.Z

Creates a SeasonalWeather release commit and annotated vX.Y.Z tag.

Environment:
  SKIP_TESTS=1        skip pytest during release preparation
  ALLOW_NON_MAIN=1    allow releasing from a branch other than main
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

version="$1"
tag="v${version}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python3 tools/semver_guard.py check-version "$version" >/dev/null

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" && "${ALLOW_NON_MAIN:-0}" != "1" ]]; then
  echo "error: releases must be cut from main; current branch is $branch" >&2
  echo "       set ALLOW_NON_MAIN=1 only for an intentional maintenance release" >&2
  exit 1
fi

git update-index -q --refresh
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "error: working tree is dirty; commit or stash changes before releasing" >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  echo "error: local tag already exists: ${tag}" >&2
  exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
  echo "error: remote tag already exists on origin: ${tag}" >&2
  exit 1
fi

python3 tools/semver_guard.py check-newer "$version"
python3 tools/semver_guard.py replace-version "$version"
python3 tools/semver_guard.py check-working

python3 -m compileall -q seasonalweather tools

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  python3 -m pytest
else
  echo "warning: SKIP_TESTS=1 set; pytest was not run" >&2
fi

git add seasonalweather/__init__.py
git commit -m "ver: release ${tag}"
git tag -a "$tag" -m "SeasonalWeather ${tag}"

cat <<EOF
Created release commit and annotated tag ${tag}.

Verify with:
  git show --no-patch --oneline --decorate ${tag}^{}
  git show ${tag}^{}:seasonalweather/__init__.py

Push with:
  git push origin main ${tag}
EOF
