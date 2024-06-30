#!/bin/bash

# Define an array of directories
directories=(
    # instruments
    "/import/c4dm-datasets/MUSDB18HQ"
    "/import/c4dm-datasets/ENST-drums"
    "/import/c4dm-datasets/medley-solos-db/audio"
    "/import/c4dm-datasets/GuitarSet/audio_mono-mic"
    # music 
    "/import/c4dm-datasets-ext/mtg-jamendo"
    #"/import/c4dm-datasets/gtzan"
    # singing voice
    "/import/c4dm-datasets/URSing/data"
    #"/import/c4dm-datasets-ext/MedleyVox"
    # speech
    "/import/c4dm-datasets-ext/LibriSpeech48k"
    #"/import/c4dm-datasets/daps_dataset/produced"
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

# for testing we hold out
# speech: DAPS
# singing: VocalSet
# music: FMA
# instruments: 

# Specify the chunk size
# number of processes = number of chunks
N=8

# Maximum number of files to process
max_files=3000

# Function to split an array into subarrays
split_array() {
    local array=("$@")
    local size=$1
    shift
    local index=0

    while (( ${#array[@]} > 0 )); do
        subarray=("${array[@]:0:size}")
        array=("${array[@]:size}")
        subarray_name="subarray_${index}"
        eval "${subarray_name}=(\"${subarray[@]}\")"
        index=$((index+1))
    done
}


# Loop over the array of directories
for directory in "${directories[@]}"; do
    # find all audio files in the directory recursively
    #audio_files=($(find "$directory"  -type f  -name "*.wav"))

    # Use find to locate all WAV files in the directory and its subdirectories
    while IFS= read -r -d '' file; do
        # Append the file path to the array
        audio_files+=("$file")
    done < <(find "$directory" -type f -name "*.mp3" -print0)

    echo "Found ${#audio_files[@]} audio files in $directory"

    # if the length of audio_files array is greater than max_files
    # we will truncate the array and take the first max_files elements
    if [ ${#audio_files[@]} -gt $max_files ]; then
        echo "Truncating array to $max_files elements"
        audio_files=("${audio_files[@]:0:$max_files}")
    fi

    # Loop over the chunks and print each file
    for ((i=0; i<${#audio_files[@]}/N; i++)); do
        start=$((i * N))
        end=$(($start + N))
        echo "${i}==== ${start} ${end}"

        # split the array into subarrays
        sub_audio_files=("${audio_files[@]:$start:$N}")
        
        # iterate over the subarray and print
        process=0
        for file in "${sub_audio_files[@]}"; do
            echo "$file"
            
            # check if process is last in subarray
            if [ $process -eq $(($N - 1)) ]; then
                echo "$process - last process. blocking..."
                python scripts/vst_datagen_file.py \
                "$file" \
                "/homes/cjs01/code/lcap/plugins/valid" \
                --presets "/homes/cjs01/code/lcap/plugins/presets" \
                --length 524288 \
                -o "/import/c4dm-datasets-ext/lcap-datasets/${names[$i]}-vst3-presets"
            else
                echo "$process - not last process. non-blocking..."
                python scripts/vst_datagen_file.py \
                "$file" \
                "/homes/cjs01/code/lcap/plugins/valid" \
                --presets "/homes/cjs01/code/lcap/plugins/presets" \
                --length 524288 \
                -o "/import/c4dm-datasets-ext/lcap-datasets/${names[$i]}-vst3-presets"
            fi
            # increment process
            process=$((process + 1))
        done
    done
done
