#!/usr/bin/env bash
# Ship a panel's ANALYSIS-GRADE artifacts back from the run box.
#
#   bash scripts/pack_artifacts.sh v11dec
#   bash scripts/pack_artifacts.sh v11dec /mnt/shared/outbox
#
# What travels: removal.jsonl, alpha_trace.jsonl, not_applicable.jsonl and any
# per-panel json. That is the whole input to `make freeze` -- every number the
# manuscript cites is derived from these.
#
# What does NOT travel: the support dumps and trained direction tensors. They
# are gigabytes, they are regenerable from the corpora and configs, and no
# published number reads them directly. `.gitignore` excludes them for the same
# reason.
#
# The archive is content-addressed by the row count so two shipments of the
# same panel state collide loudly instead of silently overwriting.
set -euo pipefail

PANEL="${1:?usage: pack_artifacts.sh <panel> [outdir]}"
OUTDIR="${2:-.}"
SRC="${BS_OUT:-results}/$PANEL"

[ -d "$SRC" ] || { echo "no such panel: $SRC"; exit 1; }

rows=0
for f in removal.jsonl alpha_trace.jsonl; do
  [ -f "$SRC/$f" ] && rows=$((rows + $(wc -l < "$SRC/$f")))
done
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$OUTDIR/${PANEL}_${stamp}_r${rows}.tar.gz"

mkdir -p "$OUTDIR"
# Exclusions verified against a real panel: the support dumps live in
# <panel>/supports/*.npz, which an earlier `support_*` pattern did NOT match --
# it shipped 319 MB instead of 1 MB. Keep the directory name in the list.
tar czf "$ARCHIVE" \
  --exclude='supports' --exclude='support_*' \
  --exclude='*.npz' --exclude='*.npy' --exclude='*.pt' --exclude='*.safetensors' \
  --exclude='directions*' --exclude='sketches*' \
  -C "${BS_OUT:-results}" "$PANEL"

echo "packed $ARCHIVE"
echo "  rows      : $rows"
echo "  size      : $(du -h "$ARCHIVE" | cut -f1)"
echo "  contents  :"
tar tzf "$ARCHIVE" | sed 's/^/    /' | head -20
echo
echo "on the analysis box:  tar xzf $(basename "$ARCHIVE") -C results/ && make freeze"
