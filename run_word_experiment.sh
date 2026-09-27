#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/word_experiment_vqgan_gemma}"
LAYERS=(5 15 25)

while IFS='|' read -r category target; do
  slug="${target// /_}"
  for layer in "${LAYERS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${category}/${slug}/layer-${layer}"

    echo "Generating ${category}: ${target} (layer ${layer})"
    start_time=$(date +%s)
    "$PYTHON_BIN" "$SCRIPT_DIR/vqgan_gemma.py" \
      --target "$target" \
      --layer "$layer" \
      --out "$output_dir"
    end_time=$(date +%s)
    echo "Finished ${category}: ${target} (layer ${layer}) in $((end_time - start_time)) seconds"
  done
done <<'WORDS'
Animals|giraffe
Animals|octopus
Celebrations|Halloween
Celebrations|Christmas
People|Einstein
People|Marilyn Monroe
Sports|soccer
Sports|chess
Activities|sleeping
Activities|swimming
Emotions|fear
Emotions|happiness
Physical_objects|bicycle
Physical_objects|kettle
WORDS

# Animals|octopus
# Animals|frog
# Animals|squirrel
# Animals|giraffe
# Animals|bee
# Animals|dog
# Animals|lion
# Animals|elephant
# Animals|parrot
# Animals|T-rex
# Seasons|spring (season)
# Seasons|summer
# Seasons|autumn
# Seasons|winter
# Celebrations|Christmas
# Celebrations|Halloween
# Celebrations|Easter
# Celebrations|birthday
# Celebrations|wedding
# Celebrations|funeral
# Subjects|mathematics
# Subjects|philosophy
# Subjects|geometry
# Subjects|history
# Subjects|physics
# Subjects|chemistry
# Subjects|biology
# Subjects|computer science
# Subjects|geography
# Subjects|music (subject)
# Sensory|loud
# Sensory|silent
# Sensory|smooth
# Sensory|rough
# Sensory|sweet
# People|Cleopatra
# People|Caesar
# People|Napoleon
# People|Marilyn Monroe
# People|Frida Kahlo
# People|Elvis Presley
# People|Einstein
# People|William Shakespeare
# People|Wolfgang Amadeus Mozart
# People|Winston Churchill
# Cities|new york
# Cities|san francisco
# Cities|paris
# Cities|rome
# Cities|london
# Sports|soccer
# Sports|poker
# Sports|basketball
# Sports|chess
# Sports|hockey
# Sports|rugby
# Sports|tennis
# Sports|golf
# Sports|judo
# Sports|boxing
# Nationalities|french
# Nationalities|italian
# Nationalities|egyptian
# Nationalities|czech
# Nationalities|chinese
# Nationalities|greek
# Nationalities|american
# Nationalities|indian
# Nationalities|german
# Nationalities|japanese
# Activities|swimming
# Activities|running
# Activities|reading
# Activities|eating
# Activities|sleeping
# Activities|crying
# Activities|smiling
# Activities|flying
# Activities|screaming
# Activities|dancing
# Emotions|love
# Emotions|fear
# Emotions|anger
# Emotions|sadness
# Emotions|happiness
# Emotion_adjectives|loving person
# Emotion_adjectives|fearful person
# Emotion_adjectives|angry person
# Emotion_adjectives|sad person
# Emotion_adjectives|happy person
# LLM_tasks|programming
# LLM_tasks|translating
# LLM_tasks|refusing
# LLM_tasks|summarizing
# LLM_tasks|formatting
# Physical_objects|kettle
# Physical_objects|toaster
# Physical_objects|jupiter
# Physical_objects|armchair
# Physical_objects|bicycle
# Physical_objects|tree
# Physical_objects|camera
# Physical_objects|key
# Physical_objects|radio
# Physical_objects|phone
