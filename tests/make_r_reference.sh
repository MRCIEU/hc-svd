#!/usr/bin/env bash
# Regenerates the R reference outputs in tests/data/r, which tests/r_reference.rs compares hcsvd against.
# They come from the R step that hcsvd replaces: calculate_haplotype_components.R (R Matrix + sparsesvd) from
# MRCIEU/pipeline-nf-haplotype-components, dependencies/calculate_HCs/; the current references used commit
# 54acfcb. It runs in rocker/r-ver:4.5.2, whose CRAN snapshot is fixed. Needs docker. Run from anywhere:
#
#   tests/make_r_reference.sh <path to calculate_haplotype_components.R>

set -euo pipefail

IMAGE=rocker/r-ver:4.5.2
[ $# -eq 1 ] && [ -f "$1" ] || { echo "usage: $0 <path to calculate_haplotype_components.R>" >&2; exit 1; }
r_script=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")

cd "$(dirname "$0")/data"
work=$(mktemp -d "${PWD}/.r_reference.XXXXXX")
trap 'rm -rf "${work}"' EXIT
cp "${r_script}" "${work}/calculate_haplotype_components.R"

# the case names, and their options beyond the common ones, must match tests/r_reference.rs
cat > "${work}/run.sh" <<'EOF'
set -euo pipefail
Rscript -e 'install.packages(c("data.table", "Matrix", "sparsesvd", "optparse", "R.utils"), lib = Sys.getenv("R_LIBS_USER"), quiet = TRUE)' > /dev/null
Rscript -e 'cat(R.version.string, sprintf("%s %s", c("Matrix", "sparsesvd", "data.table"), sapply(c("Matrix", "sparsesvd", "data.table"), function(p) as.character(packageVersion(p)))), sep = "\n")' > r/VERSIONS
run() {
    local name=$1; shift
    Rscript "${WORK}/calculate_haplotype_components.R" --matrix combined.chunklengths.txt.gz --samples samples.txt \
        --sums combined.chunklengths.sums --nhcs 20 --out "${WORK}/${name}.csv" --diagnostics "${WORK}/${name}.tsv" "$@"
}
run plain
run pairs_reweight --remove_pairs related_pairs.txt --reweight
run fit_third --fit_samples fit_third.txt
run fit_unrelated_project --fit_samples fit_unrelated.txt --project --remove_pairs related_pairs.txt --reweight
EOF
docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" -e HOME=/tmp -e WORK="${work}" -e R_LIBS_USER=/tmp/lib \
    -v "${PWD}":"${PWD}" -w "${PWD}" "${IMAGE}" bash -c 'mkdir -p "${R_LIBS_USER}" && bash "${WORK}/run.sh"'

for f in "${work}"/*.csv "${work}"/*.tsv; do
    gzip -n -9 -c "${f}" > "r/$(basename "${f}").gz"
done
echo "R reference outputs written to $(pwd)/r:"
cat r/VERSIONS
