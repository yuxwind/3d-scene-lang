#!/bin/bash
# Build the submission PDF locally (Overleaf-like) with Tectonic (brew install tectonic). Output: build/<name>.pdf
#   ./build.sh            build main.tex once -> build/main.pdf
#   ./build.sh watch      open the PDF, then rebuild whenever a .tex/.bib/image here changes (Ctrl-C to stop)
#   ./build.sh one NAME   build a single fragment NAME.tex with main.tex's preamble -> build/NAME.pdf
#   ./build.sh open       open build/main.pdf
# Errors: the last lines of the TeX log are printed; the full log is build/<name>.log.
set -uo pipefail
cd "$(dirname "$0")"
command -v tectonic >/dev/null || { echo "tectonic missing: brew install tectonic" >&2; exit 1; }
mkdir -p build

compile() {  # $1 = tex file, $2 = output name
  local log="build/$2.build.txt"
  if tectonic -X compile "$1" --outdir build --keep-logs --synctex >"$log" 2>&1; then
    [ "$1" = "build/$2.tex" ] || true
    echo "$(date +%T) OK   build/$2.pdf ($(grep -c -i 'warning' "$log") warnings)"
  else
    echo "$(date +%T) FAIL $1"; grep -E "^(error|!)|\.tex:[0-9]+:" "$log" | head -n 12
    return 1
  fi
}

one() {  # wrap a fragment in main.tex's preamble
  local n=${1%.tex}
  [ -f "$n.tex" ] || { echo "no $n.tex" >&2; exit 1; }
  sed -n '1,/\\begin{document}/p' main.tex | grep -v -E '\\(maketitle|tableofcontents)' > "build/$n.tex"
  printf '\\input{../%s}\n\\end{document}\n' "$n" >> "build/$n.tex"
  # graphics referenced relative to this folder
  sed -i '' 's|\\begin{document}|\\graphicspath{{../}}\n\\begin{document}|' "build/$n.tex"
  compile "build/$n.tex" "$n"
}

stamp() { find . -maxdepth 1 \( -name '*.tex' -o -name '*.bib' -o -name '*.jpg' -o -name '*.png' -o -name '*.pdf' \) -exec stat -f '%m %N' {} + 2>/dev/null | sort | md5 -q; }

case "${1:-build}" in
  build) compile main.tex main ;;
  one)   one "${2:?fragment name}" ;;
  open)  open build/main.pdf ;;
  watch)
    compile main.tex main; open build/main.pdf
    last=$(stamp); echo "watching *.tex *.bib images (Ctrl-C to stop)"
    while sleep 1; do now=$(stamp); [ "$now" != "$last" ] && { last=$now; compile main.tex main; }; done ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
