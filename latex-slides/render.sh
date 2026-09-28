#!/bin/sh
# Render contact sheets of the deck for visual review: ./render.sh <outdir> [dpi]
out=${1:-/tmp/lab2-sheets}; dpi=${2:-55}
rm -rf "$out"; mkdir -p "$out"
pdftoppm -r "$dpi" -png build/lab2-slides.pdf "$out/p"
cd "$out" || exit 1
ls p-*.png | split -l 12 - grp_
for g in grp_*; do montage $(cat "$g") -tile 3x -geometry +6+6 -background '#888' "sheet_$g.png"; done
ls sheet_*.png
