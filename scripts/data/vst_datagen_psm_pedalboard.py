import os
import glob
import uuid
import torch
import random
import argparse
import torchaudio
import pedalboard
import numpy as np
import pyloudnorm as pyln

from tqdm import tqdm
from typing import List, Dict, Callable

from lcap.utils import load_mfcc_feature_extractor, get_mfcc_feature_embeds


def find_distinct_parameters(
    func: Callable,
    effect_dict: Dict,
    audio_segment: torch.tensor,
    sample_rate: int,
    num_iters: int = 100,
):
    model = load_mfcc_feature_extractor(use_gpu=False)

    # first sample parameter set 1
    parameters1 = {}
    for param_name, ranges in effect_dict["parameters"].items():
        parameters1[param_name] = np.random.choice(ranges)

    # process with parameters1
    x_segment_processed1 = func(
        audio_segment,
        sample_rate,
        effect_dict["plugin"],
        parameters=parameters1,
    )

    # compute mfcc stats
    embeddings1 = get_mfcc_feature_embeds(
        x_segment_processed1.unsqueeze(0), model, sample_rate
    )

    max_distance = float("-inf")

    pbar = tqdm(range(num_iters))
    for i in pbar:
        # sample parameter set 2
        parameters2 = {}
        for param_name, ranges in effect_dict["parameters"].items():
            parameters2[param_name] = np.random.choice(ranges)

        # process with parameters2
        x_segment_processed2 = func(
            audio_segment,
            sample_rate,
            effect_dict["plugin"],
            parameters=parameters2,
        )

        embeddings2 = get_mfcc_feature_embeds(
            x_segment_processed2.unsqueeze(0), model, sample_rate
        )

        # measure the difference
        diff = torch.mean(torch.abs(embeddings1 - embeddings2))

        if diff > max_distance:
            max_distance = diff
            parameters2_best = parameters2

        pbar.set_description(
            f"current: {diff.item():0.4f} - max: {max_distance.item():.4f}"
        )

    return parameters1, parameters2_best


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

# when defining the directory structure we need to conisder that we will want to read
# these files from our listening test platform. this means that we should use a directory
# structure where the uuids are at the top-level directory and the files within that directory
# form one example. this will make it easier to read the files in the listening test platform.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir", type=str, help="Path to root directory of audio files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory.",
    )
    parser.add_argument("--sample_rate", type=float, default=48000)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--min_length", type=int, default=131072)
    parser.add_argument("--max_length", type=int, default=262144)
    parser.add_argument("--target_lufs_db", type=float, default=-23.0)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--intra_effect_hard", action="store_true")
    parser.add_argument("--intra_effect_easy", action="store_true")
    parser.add_argument("--inter_effect", action="store_true")
    parser.add_argument("--multi_effect", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # find all audio files in the input directory based on file extensions
    audio_filepaths = []
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
        audio_filepaths += glob.glob(
            os.path.join(args.input_dir, "**", ext), recursive=True
        )

    # set effects
    effects = EFFECTS

    # loudness meter
    meter = pyln.Meter(args.sample_rate)

    num_classes = 25

    # in this mode we will apply multiple effects to the same audio
    if args.multi_effect:
        for num_effects in range(1, 6):
            for example_idx in range(args.num_examples):
                # randomly select 2 audio files
                random_audio_filepaths = np.random.choice(audio_filepaths, size=2)
                print(random_audio_filepaths)

                multi_effect_dir = os.path.join(
                    args.output_dir,
                    f"multi-effects",
                )
                os.makedirs(multi_effect_dir, exist_ok=True)

                multi_effect_subdir = os.path.join(
                    multi_effect_dir,
                    f"effects={num_effects}",
                )

                # generate a uuid for this example
                uid = str(uuid.uuid4())
                example_dir = os.path.join(multi_effect_subdir, uid)
                os.makedirs(example_dir, exist_ok=True)

                audio_segments = []
                for audio_filepath in random_audio_filepaths:
                    # load audio
                    x, sr = torchaudio.load(audio_filepath, backend="soundfile")

                    if sr != args.sample_rate:
                        x = torchaudio.functional.resample(x, sr, args.sample_rate)

                    # randomly sample the segment length
                    segment_length = random.randint(args.min_length, args.max_length)

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, segment_length, threshold=args.threshold
                    )

                    audio_segments.append(x_segment)

                # first create the reference
                # then create the other segments
                final_audio_segments = []
                final_audio_segments.append(audio_segments[0])
                segment_names = []
                segment_names.append("ref")
                for n in range(num_classes):
                    final_audio_segments.append(audio_segments[1].clone())
                    segment_names.append(chr(97 + n))

                for segment_idx, (segment_name, audio_segment) in enumerate(
                    zip(segment_names, final_audio_segments)
                ):
                    # sample the number of effects for this example
                    example_num_effects = random.randint(1, num_effects)

                    # randomly select N effects
                    random_effects = np.random.choice(
                        list(effects.keys()),
                        size=example_num_effects,
                        replace=False,
                    )

                    if segment_idx == 0:
                        ref_random_effects = random_effects
                    elif segment_idx == 1:
                        random_effects = ref_random_effects

                    print(random_effects)

                    parameters = {}
                    # randomly sample parameters for each effect
                    for effect_idx, effect_name in enumerate(random_effects):
                        effect_key = f"{effect_name}-{effect_idx}"
                        effect = effects[effect_name]

                        # randomly select parameters
                        parameters[effect_key] = {}
                        for param_name, values in effect["parameters"].items():
                            parameters[effect_key][param_name] = np.random.choice(
                                values
                            )

                    if segment_idx == 0:
                        ref_parameters = parameters
                    elif segment_idx == 1:
                        parameters = ref_parameters

                    print(parameters)

                    # apply the effects in series
                    x_segment_processed = audio_segment
                    for effect_key, effect_parameters in parameters.items():
                        effect_name = effect_key.split("-")[0]
                        effect = effects[effect_name]
                        x_segment_processed = process(
                            x_segment_processed,
                            args.sample_rate,
                            effect["plugin"],
                            parameters=parameters[effect_key],
                        )

                    # loudness normalize the audio
                    loudness = meter.integrated_loudness(
                        x_segment_processed.permute(1, 0).numpy()
                    )
                    loudness_delta = args.target_lufs_db - loudness
                    gain_db = loudness_delta
                    x_segment_processed *= 10.0 ** (gain_db / 20.0)

                    effect_names = "-".join(random_effects)

                    # check for NaN
                    if torch.isnan(x_segment_processed).any():
                        print("nan detected")
                        continue

                    # save result
                    filepath = os.path.join(
                        example_dir, f"{segment_name}-{effect_names}.flac"
                    )
                    torchaudio.save(
                        filepath,
                        x_segment_processed,
                        int(args.sample_rate),
                        backend="soundfile",
                    )

    # generate the first mode: intra-effect
    # in this mode, we randomly select one plugin and three random audios
    # then we apply the plugin to each audio but with unique random parameters
    # (x) audio 1 + plugin 1 + params 1
    # (a) audio 2 + plugin 1 + params 2
    # (b) audio 3 + plugin 1 + params 3
    # (c) audio 1 + plugin 1 + params 4
    # In this case, we do not know the correct answer, so we will be asking humans.
    if args.intra_effect_hard:
        intra_effect_dir = os.path.join(args.output_dir, "intra-effect-hard")
        os.makedirs(intra_effect_dir, exist_ok=True)

        # iterate over vst plugins
        for instance_name, effect in tqdm(effects.items()):
            plugin_dir = os.path.join(intra_effect_dir, instance_name)
            os.makedirs(plugin_dir, exist_ok=True)
            for n in range(args.num_examples):
                # generate a uuid for this example
                uid = str(uuid.uuid4())
                example_dir = os.path.join(plugin_dir, uid)
                os.makedirs(example_dir, exist_ok=True)

                # randomly select three audio files
                random_audio_filepaths = np.random.choice(audio_filepaths, size=3)
                names = ["x", "a", "b", "c"]

                audio_segments = []
                for audio_filepath in random_audio_filepaths:
                    # load audio
                    x, sr = torchaudio.load(audio_filepath, backend="soundfile")

                    if sr != args.sample_rate:
                        x = torchaudio.functional.resample(x, sr, args.sample_rate)

                    # randomly sample the segment length
                    segment_length = random.randint(args.min_length, args.max_length)

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, segment_length, threshold=args.threshold
                    )

                    # make dual mono if necessary
                    if x_segment.shape[0] == 1:
                        x_segment = torch.cat((x_segment, x_segment), axis=0)

                    audio_segments.append(x_segment)

                # repeat the first audio segment
                audio_segments.append(audio_segments[0])

                for name, x_segment in zip(names, audio_segments):
                    # process with random parameters

                    # randomly select parameters
                    parameters = {}
                    for param_name, ranges in effect["parameters"].items():
                        parameters[param_name] = np.random.uniform(ranges[0], ranges[1])

                    x_segment_processed = process(
                        x_segment,
                        args.sample_rate,
                        effect["plugin"],
                        parameters=parameters,
                    )

                    # loudness normalize the audio
                    loudness = meter.integrated_loudness(
                        x_segment_processed.permute(1, 0).numpy()
                    )
                    loudness_delta = args.target_lufs_db - loudness
                    gain_db = loudness_delta
                    x_segment_processed *= 10.0 ** (gain_db / 20.0)

                    # save result
                    filepath = os.path.join(example_dir, f"{name}-{instance_name}.flac")
                    torchaudio.save(filepath, x_segment_processed, sr)

    # generate the second mode: inter-effect
    # in this mode we want to test if the model is focused more on content or style
    # we select two different audios and two different plugins
    # (x) audio 1 + plugin 1
    # (a) audio 2 + plugin 1
    # (b) audio 3 + plugin 2
    # (c) audio 1 + plugin 2
    # In this case, the human and model should say that (x) and (a) are more similar
    # since they have the same plugin applied to them. (b) should be more different
    if args.inter_effect:
        inter_effect_dir = os.path.join(args.output_dir, "inter-effect")
        os.makedirs(inter_effect_dir, exist_ok=True)

        for instance_name, effect in tqdm(effects.items()):
            plugin_dir = os.path.join(inter_effect_dir, instance_name)
            os.makedirs(plugin_dir, exist_ok=True)
            for n in range(args.num_examples):
                # generate a uuid for this example
                uid = str(uuid.uuid4())
                example_dir = os.path.join(plugin_dir, uid)
                os.makedirs(example_dir, exist_ok=True)

                # randomly select three audio files
                random_audio_filepaths = np.random.choice(audio_filepaths, size=3)
                names = ["x", "a", "b", "c"]

                audio_segments = []
                for audio_filepath in random_audio_filepaths:
                    # load audio
                    x, sr = torchaudio.load(audio_filepath, backend="soundfile")

                    if sr != args.sample_rate:
                        x = torchaudio.functional.resample(x, sr, args.sample_rate)

                    # randomly sample the segment length
                    segment_length = random.randint(args.min_length, args.max_length)

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, segment_length, threshold=args.threshold
                    )

                    audio_segments.append(x_segment)

                # repeat the first and second audio segments
                audio_segments = [
                    audio_segments[0],
                    audio_segments[1],
                    audio_segments[2],
                    audio_segments[0],
                ]

                # select the distractor plugin (different from the current plugin)
                other_effects = [e for e in list(effects.keys()) if e != instance_name]
                distractor_instance_name = np.random.choice(other_effects)
                effect_distractor = effects[distractor_instance_name]

                # todo: we need to randomize the parameters for both plugins here
                plugin1_parameters = {}
                plugin2_parameters = {}

                for param_name, ranges in effect["parameters"].items():
                    plugin1_parameters[param_name] = np.random.choice(ranges)

                for param_name, ranges in effect_distractor["parameters"].items():
                    plugin2_parameters[param_name] = np.random.choice(ranges)

                plugins = [
                    effect["plugin"],
                    effect["plugin"],
                    effect_distractor["plugin"],
                    effect_distractor["plugin"],
                ]

                plugin_parameters = [
                    plugin1_parameters,
                    plugin1_parameters,
                    plugin2_parameters,
                    plugin2_parameters,
                ]

                plugin_names = [
                    instance_name,
                    instance_name,
                    distractor_instance_name,
                    distractor_instance_name,
                ]

                for (
                    name,
                    audio_segment,
                    plugin,
                    plugin_name,
                    parameters,
                ) in zip(
                    names,
                    audio_segments,
                    plugins,
                    plugin_names,
                    plugin_parameters,
                ):
                    print(name, plugin_name, parameters)
                    x_segment_processed = process(
                        audio_segment,
                        args.sample_rate,
                        plugin,
                        parameters,
                    )

                    # loudness normalize the audio
                    loudness = meter.integrated_loudness(
                        x_segment_processed.permute(1, 0).numpy()
                    )
                    loudness_delta = args.target_lufs_db - loudness
                    gain_db = loudness_delta
                    x_segment_processed *= 10.0 ** (gain_db / 20.0)

                    # save result
                    filepath = os.path.join(example_dir, f"{name}-{plugin_name}.flac")
                    torchaudio.save(
                        filepath, x_segment_processed, int(args.sample_rate)
                    )

    # generate the third mode: intra-effect
    # in this mode we apply the same plugin to multiple audios
    # (x) audio 1 + plugin 1 + params 1
    # (a) audio 2 + plugin 1 + params 1
    # (b) audio 3 + plugin 1 + params 2
    # (c) audio 1 + plugin 1 + params 2
    # In this case, the human and model should say that (x) and (a) are more similar
    if args.intra_effect_easy:
        intra_effect_dir = os.path.join(args.output_dir, "intra-effect-easy")
        os.makedirs(intra_effect_dir, exist_ok=True)

        for instance_name, effect in tqdm(effects.items()):
            plugin_dir = os.path.join(intra_effect_dir, instance_name)
            os.makedirs(plugin_dir, exist_ok=True)

            for n in range(args.num_examples):
                # generate a uuid for this example
                uid = str(uuid.uuid4())
                example_dir = os.path.join(plugin_dir, uid)
                os.makedirs(example_dir, exist_ok=True)

                # randomly select three audio files
                random_audio_filepaths = np.random.choice(audio_filepaths, size=3)
                names = ["x", "a", "b", "c"]

                audio_segments = []
                for audio_filepath in random_audio_filepaths:
                    # load audio
                    x, sr = torchaudio.load(audio_filepath, backend="soundfile")

                    if sr != args.sample_rate:
                        x = torchaudio.functional.resample(x, sr, args.sample_rate)

                    # randomly sample the segment length
                    segment_length = random.randint(args.min_length, args.max_length)

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, segment_length, threshold=args.threshold
                    )

                    audio_segments.append(x_segment)

                # repeat the first and second audio segments
                audio_segments = [
                    audio_segments[0],
                    audio_segments[1],
                    audio_segments[2],
                    audio_segments[0],
                ]

                # create two sets of parameters for the plugin
                parameters1 = {}
                parameters2 = {}
                param_names = list(effect["parameters"].keys())

                parameters1, parameters2 = find_distinct_parameters(
                    process, effect, audio_segments[0], args.sample_rate, 10
                )

                for name, audio_segment in zip(names, audio_segments):
                    # process with random parameters
                    if name in ["x", "a"]:
                        parameter_name = "params1"
                        x_segment_processed = process(
                            audio_segment,
                            args.sample_rate,
                            effect["plugin"],
                            parameters=parameters1,
                        )
                    else:
                        parameter_name = "params2"
                        x_segment_processed = process(
                            audio_segment,
                            args.sample_rate,
                            effect["plugin"],
                            parameters=parameters2,
                        )

                    # loudness normalize the audio
                    loudness = meter.integrated_loudness(
                        x_segment_processed.permute(1, 0).numpy()
                    )
                    loudness_delta = args.target_lufs_db - loudness
                    gain_db = loudness_delta
                    x_segment_processed *= 10.0 ** (gain_db / 20.0)

                    x_segment_processed = torch.nan_to_num(x_segment_processed)
                    # check for nans
                    if torch.isnan(x_segment_processed).any():
                        print(x_segment_processed)
                        print("nan detected")

                    # save result
                    filepath = os.path.join(
                        example_dir, f"{name}-{instance_name}-{parameter_name}.flac"
                    )
                    torchaudio.save(
                        filepath, x_segment_processed, int(args.sample_rate)
                    )
