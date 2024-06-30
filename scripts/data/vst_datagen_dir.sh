#!/bin/bash

# Define an array of directories
directories=(
    # instruments
    "/import/c4dm-datasets/MUSDB18HQ"
    "/import/c4dm-datasets/ENST-drums"
    "/import/c4dm-datasets/medley-solos-db/audio"
    "/import/c4dm-datasets/GuitarSet/audio_mono-mic"
    "/import/c4dm-datasets-ext/IDMT-SMT-DRUMS-V2/audio"
    "/import/c4dm-datasets-ext/IDMT-PIANO-MM/audio"
    # music 
    "/import/c4dm-datasets-ext/mtg-jamendo"
    # singing voice
    "/import/c4dm-datasets/URSing/data"
    # speech
    "/import/c4dm-datasets-ext/LibriSpeech48k"
    # sound effects
    "/import/c4dm-datasets/FSD50K"
)

names=(
    "musdb18"
    "enst-drums"
    "medley-solos-db"
    "guitarset"
    "mtg-jamendo"
    "ursing"
    "librispeech"
    "fsd50k"
)

# Loop over the array of directories
#for directory in "${directories[@]}"; do
for ((i=0; i<${#directories[@]}; i++)); do
 
    echo "${directories[$i]}"
    echo "${names[$i]}"

    python scripts/vst_datagen_dir.py \
    "${directories[$i]}" \
    "/homes/cjs01/code/lcap/plugins/valid" \
    --presets "/homes/cjs01/code/lcap/plugins/presets" \
    --length 524288 \
    --dataset_name "${names[$i]}" \
    -o "/import/c4dm-datasets-ext/lcap-datasets/20k/${names[$i]}-vst3-presets" &

done