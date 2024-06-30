#!/bin/bash

# Define an array of directories
directories=(
    # instruments
    "/import/c4dm-datasets/MUSDB18HQ"
    "/import/c4dm-datasets/ENST-drums"
    "/import/c4dm-datasets/medley-solos-db/audio"
    "/import/c4dm-datasets/GuitarSet"
    # music 
    "/import/c4dm-datasets-ext/mtg-jamendo"
    "/import/c4dm-datasets/gtzan"
    # singing voice
    "/import/c4dm-datasets/URSing/data"
    "/import/c4dm-datasets-ext/MedleyVox"
    "/import/c4dm-datasets-ext/OpenSinger"
    # speech
    "/import/c4dm-datasets-ext/LibriSpeech48k"
    # sound effects
    "/import/c4dm-datasets/FSD50K"
)

# for testing we hold out
# speech: DAPS
# singing: VocalSet
# music: FMA
# instruments: 

# Loop over the array of directories
for directory in "${directories[@]}"; do
    # Check if the item is a directory
    if [ -d "$directory" ]; then
        # Call your Python script with the directory path as an argument
        python scripts/vst_datagen.py \
        "$directory" \
        "/homes/cjs01/code/lcap/plugins/valid" \
        --presets "/homes/cjs01/code/lcap/plugins/presets" \
        --length 524288 \
        -o "/import/c4dm-datasets-ext/lcap-datasets/vst" &
    fi
done


python scripts/vst_datagen.py \
"/import/c4dm-datasets/GuitarSet/audio_mono-mic" \
"/homes/cjs01/code/lcap/plugins/valid" \
--presets "/homes/cjs01/code/lcap/plugins/presets" \
--fixed_preset \
--length 524288 \
--dataset_name "guitarset" \
-o "/import/c4dm-datasets-ext/lcap-datasets/guitarset-vst2"