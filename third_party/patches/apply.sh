#!/usr/bin/env bash
# Apply this repo's patches to the third-party submodules.
#
# The submodules are pinned to their upstream commits and their working trees are
# patched in place - the superproject records only a commit SHA, so nothing under
# third_party/<name>/ is version-controlled here. These .patch files are, and this
# script is what turns a fresh `git submodule update --init` into a working checkout.
#
# Each patch's base commit is asserted before it is applied: a patch built against a
# different revision would either fail loudly or, worse, apply to the wrong lines.
#
#   ./third_party/patches/apply.sh          # apply
#   ./third_party/patches/apply.sh --check  # report whether each is already applied
#   ./third_party/patches/apply.sh --revert # undo
set -euo pipefail

cd "$(dirname "$0")/../.."
MODE="${1:-apply}"

# name:base-commit - the upstream revision each patch was generated against.
MODULES=(
  "FastGS:f6210e44dbb104b2e88259f3767ac95ffe0e0057"
)

status=0
for entry in "${MODULES[@]}"; do
  name="${entry%%:*}"
  base="${entry##*:}"
  dir="third_party/$name"
  patch="third_party/patches/$name.patch"

  if [ ! -d "$dir/.git" ] && [ ! -f "$dir/.git" ]; then
    echo "SKIP  $name - not checked out (git submodule update --init $dir)"
    status=1
    continue
  fi
  head=$(git -C "$dir" rev-parse HEAD)
  if [ "$head" != "$base" ]; then
    echo "SKIP  $name - at ${head:0:12}, patch was built against ${base:0:12}"
    status=1
    continue
  fi

  case "$MODE" in
    --check)
      if git -C "$dir" apply --reverse --check "../../$patch" 2>/dev/null; then
        echo "OK    $name - already applied"
      elif git -C "$dir" apply --check "../../$patch" 2>/dev/null; then
        echo "TODO  $name - clean tree, patch applies"
        status=1
      else
        echo "DIRTY $name - neither applies nor reverses cleanly; inspect by hand"
        status=1
      fi
      ;;
    --revert)
      git -C "$dir" apply --reverse "../../$patch"
      echo "REVERTED $name"
      ;;
    apply)
      if git -C "$dir" apply --reverse --check "../../$patch" 2>/dev/null; then
        echo "OK    $name - already applied"
      else
        git -C "$dir" apply "../../$patch"
        echo "APPLIED $name"
      fi
      ;;
    *)
      echo "usage: $0 [--check|--revert]" >&2
      exit 2
      ;;
  esac
done
exit $status
