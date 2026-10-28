#!/usr/bin/env bash
# Run one or both regression subjects using a Docker image, then compare exactly.
# Usage: tests/run-longitudinal.sh DATASET IMAGE OUTPUT_DIR [sub-test01|sub-test02]
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=$(cd -- "${1:?Dataset directory required}" && pwd)
image=${2:?Image required}
output=${3:?Output directory required}
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
case "$output" in *' '*) echo 'Output path must not contain spaces' >&2; exit 2;; esac
subjects=(sub-test01 sub-test02)
if [ "$#" -ge 4 ]; then
    case "$4" in sub-test01|sub-test02) subjects=("$4");; *) echo 'Unknown subject' >&2; exit 2;; esac
fi
(cd "$root" && sha256sum --check SHA256SUMS)
for subject in "${subjects[@]}"; do
    if [ -e "$output/$subject" ]; then
        echo "Refusing to overwrite $output/$subject" >&2
        exit 2
    fi
    mkdir "$output/$subject"
    docker run --rm --platform linux/amd64 --cpus 4 \
        --user "$(id -u):$(id -g)" \
        --env USER=nonroot \
        --mount "type=bind,src=$root/inputs,dst=/inputs,readonly" \
        --mount "type=bind,src=$output/$subject,dst=/output" \
        "$image" \
        --dwi "/inputs/$subject/ses-1/${subject}_ses-1_dwi.nii.gz" \
              "/inputs/$subject/ses-2/${subject}_ses-2_dwi.nii.gz" \
        --tp ses-1 ses-2 --id "$subject" -o /output \
        --bRange 800 1200 --threads 4 --para 1 --qc 1 \
        2>&1 | tee "$output/$subject/console.log"
    docker image inspect "$image" > "$output/$subject/image-inspect.json"
    "${PYTHON:-python3}" "$repo/tools/compare_results.py" \
        "$root/reference/$subject/delta-svd_results.csv" \
        "$output/$subject/delta-svd_results.csv" \
        2>&1 | tee "$output/$subject/comparison.txt"
    "${PYTHON:-python3}" "$repo/tools/longitudinal_changes.py" \
        "$output/$subject/delta-svd_results.csv" "$output/$subject/longitudinal-changes.csv"
done
