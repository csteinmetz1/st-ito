import os
import glob
import uuid
import torch
import random
import argparse
import torchaudio
import pedalboard
import numpy as np


def get_nonsilent_region(
    audio_input: torch.Tensor,
    length: int,
    max_tries=100,
    threshold: float = 1e-4,
    min_start: float = 0.25,
):
    # always crop the audio to the first 25% of the audio
    min_start = int(audio_input.shape[-1] * min_start)
    audio_input = audio_input[:, min_start:]

    # get number of frames
    num_frames = audio_input.shape[-1]

    # check if audio is long enough
    if (num_frames) < length:
        # if too short, repeat pad the audio
        num_repeats = int(np.ceil(length / num_frames))
        audio_input = audio_input.repeat(1, num_repeats)

    num_frames = audio_input.shape[1]

    silent = True
    tries = 0
    while silent and tries < max_tries:
        start = random.randint(0, num_frames - (length))

        # get audio segment
        segment = audio_input[:, start : start + length]

        # check for silence
        energy = torch.mean(torch.abs(segment))
        if energy > threshold:
            silent = False

        tries += 1

    if silent:
        raise ValueError(f"Could not find a non-silent region in {audio_filepath}.")

    return segment


def process(
    x: torch.Tensor,
    sample_rate: int,
    plugin: pedalboard.Pedalboard,
    parameters: dict = None,
):
    # make dual mono if necessary
    if x.shape[0] == 1:
        x = torch.cat((x, x), axis=0)

    # set parameters if necessary
    if parameters is not None:
        for name, value in parameters.items():
            if hasattr(plugin, name):
                setattr(plugin, name, value)
            else:
                raise ValueError(f"Parameter {name} not found in {plugin} plugin.")

    x_processed = plugin.process(x.numpy(), sample_rate)
    x_processed = torch.from_numpy(x_processed)

    # peak normalize
    x_processed /= torch.max(torch.abs(x_processed)).clamp(1e-8)

    return x_processed


EFFECTS = {
    "compressor": {
        "plugin": pedalboard.Compressor(),
        "parameters": {
            "threshold_db": [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0],
            "ratio": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            "attack_ms": [0.1, 1.0, 10.0, 25.0, 100.0, 250.0],
            "release_ms": [10.0, 25.0, 100.0, 250.0, 500.0],
        },
    },
    "delay": {
        "plugin": pedalboard.Delay(),
        "parameters": {
            "delay_seconds": [0.025, 0.05, 0.10, 0.25, 0.3, 0.4, 0.5],
            "feedback": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            "mix": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    },
    "distortion": {
        "plugin": pedalboard.Distortion(),
        "parameters": {
            "drive_db": [9.0, 12.0, 16.0, 18.0, 22.0, 24.0],
        },
    },
    "chorus": {
        "plugin": pedalboard.Chorus(),
        "parameters": {
            "centre_delay_ms": [0.5, 1.0, 2.5, 5.0, 7.5, 10.0],
            "depth": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "rate_hz": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "mix": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    },
    "phaser": {
        "plugin": pedalboard.Phaser(),
        "parameters": {
            "centre_frequency_hz": [
                400.0,
                600.0,
                800.0,
                1000.0,
                1200.0,
                1600.0,
                1800.0,
                2000.0,
                2400.0,
            ],
            "depth": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "rate_hz": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "mix": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    },
    "reverb": {
        "plugin": pedalboard.Reverb(),
        "parameters": {
            "room_size": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "damping": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "wet_level": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "dry_level": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
    },
    "limiter": {
        "plugin": pedalboard.Limiter(),
        "parameters": {
            "threshold_db": [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0],
            "release_ms": [10.0, 25.0, 100.0, 250.0, 500.0],
        },
    },
    "bitcrusher": {
        "plugin": pedalboard.Bitcrush(),
        "parameters": {
            "bit_depth": [4.0, 6.0, 8.0, 10.0],
        },
    },
    "highpass": {
        "plugin": pedalboard.HighpassFilter(),
        "parameters": {
            "cutoff_frequency_hz": [
                200.0,
                400.0,
                800.0,
                1600.0,
                2400.0,
                4800.0,
                9600.0,
            ],
        },
    },
    "lowpass": {
        "plugin": pedalboard.LowpassFilter(),
        "parameters": {
            "cutoff_frequency_hz": [
                25.0,
                50.0,
                100.0,
                200.0,
                400.0,
                800.0,
                1600.0,
                2400.0,
                4800.0,
            ],
        },
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_styles", type=str, default=10)
    parser.add_argument("--num_examples_per_style", type=str, default=100)
    parser.add_argument("--min_length", type=int, default=262144)
    parser.add_argument("--max_length", type=int, default=524288)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--max_num_effects", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-vis-dataset",
    )
    args = parser.parse_args()

    audio_dirs = [
        "/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/Ibanez 2820",
        "/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/acoustic_mic",
        "/import/c4dm-datasets/VocalSet1-2/data_by_singer",
        "/import/c4dm-datasets/daps_dataset/clean",
        "/import/c4dm-datasets/ENST-drums-mixes",
    ]
    dataset_types = [
        "electric-guitar",
        "acoustic-guitar",
        "vocals",
        "speech",
        "drums",
    ]

    # find all audio files in the audio directory
    audio_filepaths = {}
    for dataset_type, audio_dir in zip(dataset_types, audio_dirs):
        audio_filepaths[dataset_type] = []
        for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
            audio_filepaths[dataset_type] += glob.glob(
                os.path.join(audio_dir, "**", ext), recursive=True
            )
        print(
            f"Found {len(audio_filepaths[dataset_type])} audio files for {dataset_type}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    effects = EFFECTS

    for style_idx in range(args.num_styles):
        # create the style randomly
        style_dir = os.path.join(args.output_dir, f"style_{style_idx:03d}")
        os.makedirs(style_dir, exist_ok=True)

        # sample the number of effects for this style
        example_num_effects = random.randint(1, args.max_num_effects)

        # randomly select N effects
        random_effects = np.random.choice(
            list(effects.keys()),
            size=example_num_effects,
            replace=False,
        )

        print(style_idx, random_effects)
        effect_names = "-".join(random_effects)

        parameters = {}
        # randomly sample parameters for each effect
        for effect_idx, effect_name in enumerate(random_effects):
            effect_key = f"{effect_name}-{effect_idx}"
            effect = effects[effect_name]

            # randomly select parameters
            parameters[effect_key] = {}
            for param_name, values in effect["parameters"].items():
                parameters[effect_key][param_name] = np.random.choice(values)

        # now generate the audio examples by applying this style to each audio file
        for example_idx in range(args.num_examples_per_style):
            # randomly select dataset type
            dataset_type = random.choice(dataset_types)

            # randomly select an audio file
            audio_filepath = random.choice(audio_filepaths[dataset_type])

            # load and resample
            x, sr = torchaudio.load(audio_filepath, backend="soundfile")
            if sr != args.sample_rate:
                x = torchaudio.functional.resample(x, sr, args.sample_rate)

            # randomly sample the segment length
            segment_length = random.randint(args.min_length, args.max_length)

            # find a non-silent segment of length
            x_segment = get_nonsilent_region(
                x, segment_length, threshold=args.threshold
            )

            # apply the effects in series
            x_segment_processed = x_segment
            for effect_key, effect_parameters in parameters.items():
                effect_name = effect_key.split("-")[0]
                effect = effects[effect_name]
                x_segment_processed = process(
                    x_segment_processed,
                    args.sample_rate,
                    effect["plugin"],
                    parameters=parameters[effect_key],
                )

            # check for NaN
            if torch.isnan(x_segment_processed).any():
                print("NaN detected")
                continue

            # save result
            filepath = os.path.join(
                style_dir,
                f"{example_idx:03d}_{style_idx:03d}_{dataset_type}_{effect_names}.flac",
            )

            torchaudio.save(
                filepath,
                x_segment_processed,
                int(args.sample_rate),
                backend="soundfile",
            )
