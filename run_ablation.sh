#!/usr/bin/env bash
set -euo pipefail

# Sweep initialization, representation distance, Gemma layer, and concept
# type. Override the space-separated variables to make a smaller/larger grid.
# Example:
#   LAYERS="1 12" STEPS=200 ./run_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/ablation_vqgan_gemma}"
STEPS="${STEPS:-900}"
SAVE_EVERY="${SAVE_EVERY:-300}"
INIT_SCALE="${INIT_SCALE:-0.5}"
LAYERS="${LAYERS:-1 8 16 24}"
INIT_METHODS="${INIT_METHODS:-gaussian}"
DISTANCES="${DISTANCES:-cosine}"
CONCEPT_SET="${CONCEPT_SET:-abstract}" # concrete, abstract, or both
DRY_RUN="${DRY_RUN:-0}"

read -r -a LAYER_LIST <<< "$LAYERS"
read -r -a INIT_LIST <<< "$INIT_METHODS"
read -r -a DISTANCE_LIST <<< "$DISTANCES"

CONCRETE_CONCEPTS=(castle volcano astronaut)
ABSTRACT_CONCEPTS=(justice memory intelligence beauty chaos freedom )

case "$CONCEPT_SET" in
  concrete) CONCEPTS=("${CONCRETE_CONCEPTS[@]}") ;;
  abstract) CONCEPTS=("${ABSTRACT_CONCEPTS[@]}") ;;
  both) CONCEPTS=("${CONCRETE_CONCEPTS[@]}" "${ABSTRACT_CONCEPTS[@]}") ;;
  *)
    echo "CONCEPT_SET must be concrete, abstract, or both" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"
run_count=0

for concept in "${CONCEPTS[@]}"; do
  for init_method in "${INIT_LIST[@]}"; do
    for distance in "${DISTANCE_LIST[@]}"; do
      for layer in "${LAYER_LIST[@]}"; do
        output_dir="${OUTPUT_ROOT}/${concept}/init-${init_method}/distance-${distance}/layer-${layer}"
        cmd=(
          "$PYTHON_BIN" "$SCRIPT_DIR/vqgan_gemma.py"
          --target "$concept"
          --layer "$layer"
          --init-method "$init_method"
          --init-scale "$INIT_SCALE"
          --representation-distance "$distance"
          --steps "$STEPS"
          --save-every "$SAVE_EVERY"
          --out "$output_dir"
        )

        printf '[%s] %s\n' "$((run_count + 1))" "${cmd[*]}"
        run_count=$((run_count + 1))
        if [[ "$DRY_RUN" != "1" ]]; then
          mkdir -p "$output_dir"
          "${cmd[@]}"
        fi
      done
    done
  done
done

echo "Ablation runs: ${run_count}"
echo "Results: ${OUTPUT_ROOT}"
