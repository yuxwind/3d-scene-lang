#!/bin/bash
# Small copies of the 50 VSI-Bench ScanNet++ videos for scenelang_atlas.html: 320 px wide, 6 fps, H.264 CRF 34, no audio.
#   bash make_atlas_videos.sh <ffmpeg binary> <VSI-Bench dir> <out dir>    (ffmpeg: e.g. pip install imageio-ffmpeg)
set -euo pipefail
FF=$1; SRC=$2/scannetpp; OUT=$3
mkdir -p "$OUT"
ls "$SRC"/*.mp4 | xargs -P 4 -I{} bash -c 'f={}; o="'"$OUT"'/$(basename $f)"; [[ -s $o ]] || \
  "'"$FF"'" -hide_banner -loglevel error -y -i "$f" -an -vf "scale=320:-2,fps=6" -c:v libx264 -preset medium \
  -crf 34 -pix_fmt yuv420p -movflags +faststart "$o.tmp.mp4" && mv "$o.tmp.mp4" "$o" && echo "$(basename $f)"'
du -sh "$OUT"
