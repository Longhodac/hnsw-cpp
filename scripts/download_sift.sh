#!/usr/bin/env bash
# Download TEXMEX SIFT datasets into data/.
#   scripts/download_sift.sh [siftsmall|sift1m|all]   (default: all)
# siftsmall: 10K base vectors (~5 MB).  sift1m: 1M base vectors (~160 MB compressed).
set -euo pipefail

BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

fetch() {  # fetch <tarball-name> <extracted-dir>
    local tarball="$1" dir="$2"
    if [[ -f "$DATA_DIR/$dir/${dir}_base.fvecs" ]]; then
        echo "$dir already present, skipping"
        return
    fi
    echo "Downloading $tarball ..."
    curl --fail --location --retry 3 --continue-at - -o "$DATA_DIR/$tarball" "$BASE_URL/$tarball"
    tar -xzf "$DATA_DIR/$tarball" -C "$DATA_DIR"
    rm -f "${DATA_DIR:?}/${tarball:?}"
    echo "$dir ready in $DATA_DIR/$dir"
}

case "${1:-all}" in
    siftsmall) fetch siftsmall.tar.gz siftsmall ;;
    sift1m)    fetch sift.tar.gz sift ;;
    all)       fetch siftsmall.tar.gz siftsmall; fetch sift.tar.gz sift ;;
    *) echo "usage: $0 [siftsmall|sift1m|all]" >&2; exit 2 ;;
esac
