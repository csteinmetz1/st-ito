#!/bin/bash

# Define an array of directories
directories=(
    # instruments
    "/import/c4dm-datasets/MUSDB18HQ"
    "/import/c4dm-datasets/ENST-drums"
    "/import/c4dm-datasets/medley-solos-db/audio"
    "/import/c4dm-datasets/GuitarSet/audio_mono-mic"
    "/import/c4dm-datasets-ext/IDMT-PIANO-MM/audio"
    "/import/c4dm-datasets-ext/IDMT-SMT-DRUMS-V2/audio"
    # music 
    "/import/c4dm-datasets-ext/mtg-jamendo"
    # singing voice
    "/import/c4dm-datasets/URSing/data"
    "/import/c4dm-datasets/VocalSet1-2/data_by_singer"
    "/import/c4dm-datasets-ext/OpenSinger"
    # speech
    "/import/c4dm-datasets-ext/LibriSpeech48k"
    "/import/c4dm-datasets/daps_dataset/clean"
    "/import/c4dm-datasets/VCTK-Corpus-0.92/wav48_silence_trimmed"
    # sound effects
    "/import/c4dm-datasets/FSD50K"
)

names=(
    "musdb18"
    "enst-drums"
    "medley-solos-db"
    "guitarset"
    "idmt-piano"
    "idmt-drums"
    "mtg-jamendo"
    "ursing"
    "vocalset"
    "opensinger"
    "librispeech"
    "daps"
    "vctk"
    "fsd50k"
)

# Loop over the array of directories
#for directory in "${directories[@]}"; do
for ((i=0; i<${#directories[@]}; i++)); do
 
    echo "${directories[$i]}"
    echo "${names[$i]}"

    python scripts/data/vst_datagen_style.py \
    "${directories[$i]}" \
    "/homes/cjs01/code/lcap/scripts/data/vst-chains/eq+multiband-comp+limiter.json" \
    --length 524288 \
    --dataset_name "${names[$i]}" \
    --num_examples 4000 \
    -o "/import/c4dm-datasets-ext/lcap-datasets/style-eq+multiband-comp+limiter/${names[$i]}-vst3-style" &

done

module load singularity
singularity exec --bind /import/c4dm-datasets-ext:/import/c4dm-datasets-ext,/import/c4dm-datasets:/import/c4dm-datasets /homes/cjs01/containers/mycontainer.sif /bin/bash
source env_gen/bin/activate

python scripts/data/vst_datagen_style.py \
"/import/c4dm-datasets-ext/IDMT-PIANO-MM/audio" \
"/homes/cjs01/code/lcap/scripts/data/vst-chains/eq+multiband-comp+limiter.json" \
--length 524288 \
--dataset_name "idmt-piano" \
--num_examples 4000 \
-o "/import/c4dm-datasets-ext/lcap-datasets/style-eq+multiband-comp+limiter/idmt-piano-vst3-style"

python scripts/data/vst_datagen_style.py \
"/import/c4dm-datasets-ext/IDMT-SMT-DRUMS-V2/audio" \
"/homes/cjs01/code/lcap/scripts/data/vst-chains/eq+multiband-comp+limiter.json" \
--length 524288 \
--dataset_name "idmt-drums" \
--num_examples 4000 \
-o "/import/c4dm-datasets-ext/lcap-datasets/style-eq+multiband-comp+limiter/idmt-drums-vst3-style"