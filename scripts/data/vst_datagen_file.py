import os
import time
import uuid
import json
import glob
import torch
import random
import argparse
import itertools
import torchaudio
import pedalboard
import numpy as np

from tqdm import tqdm


def process_function(
    audio_input: torch.Tensor,
    vst_plugin: dict,
    preset_index: int,
    warmup: int,
    sample_rate: int,
    silence_threshold: float,
    output_dir: str,
):
    """

    Args:
        audio_input (torch.Tensor): Audio segment to process.
        vst_plugin (dict): Plugin instances and its presets.
        warmup (int): Number of warmup samples during processing.
        sample_rate (int): Sample rate.
        silence_threshold (float): Silence threshold.
        output_dir (str): Output directory.

    """
    plugin = vst_plugin["plugin"]  # get the plugin instance
    print(f"Processing {instance_name} {plugin}...")
    num_channels = vst_plugin["num_channels"]
    presets = vst_plugin["presets"]
    num_presets = len(presets)
    preset = presets[preset_index]

    # create segments with correct channel counts
    # make dual mono if necessary
    if audio_input.shape[0] == 1 and num_channels == 2:
        audio_input = torch.cat((audio_input, audio_input), axis=0)
    if audio_input.shape[0] == 2 and num_channels == 1:
        audio_input = audio_input[0:1, :]

    # configure the plugin with the preset parameters
    for name, parameter in plugin.parameters.items():
        if "bypass" in name.lower():
            parameter.raw_value = 0
        elif name in preset:  # set the parameter to the preset value
            parameter.raw_value = preset[name]
        else:
            print(f"Parameter {name} not found in preset.")

    # process with the plugin
    silent = True
    tries = 0
    max_tries = 10
    while silent and tries < max_tries:
        gain_db = random.uniform(-12, 12)
        gain_lin = 10 ** (gain_db / 20)
        audio_output = plugin.process(gain_lin * audio_input.numpy(), sample_rate)
        audio_output = torch.from_numpy(audio_output)

        if torch.mean(torch.abs(audio_output)) > silence_threshold:
            silent = False

        tries += 1

    # remove warmup samples
    audio_input = audio_input[:, warmup:]
    audio_output = audio_output[:, warmup:]

    if torch.max(torch.abs(audio_output)) > 1.0:
        audio_output /= torch.max(torch.abs(audio_output)).clamp(1e-8)

    # check if output is not silent and save
    #
    details = {
        "instance": instance_name,
        "preset": preset_index,
        "dataset": args.dataset_name,
    }

    # output filepaths
    output_filepath = os.path.join(output_dir, f"{uid}", f"{instance_name}.flac")
    json_filepath = os.path.join(output_dir, f"{uid}", f"{instance_name}.json")

    torchaudio.save(
        output_filepath,
        audio_output,
        sample_rate,
        backend="soundfile",
    )

    # save json file
    with open(json_filepath, "w") as fp:
        json.dump(details, fp, indent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to process.", type=str)
    parser.add_argument("vst_dir", help="Directory containing VST3 plugins.", type=str)
    parser.add_argument(
        "--presets",
        help="Directory containing presets.",
    )
    parser.add_argument(
        "--fixed_preset",
        help="Use only one preset per effect.",
        action="store_true",
    )
    parser.add_argument(
        "--length",
        help="Length of audio in samples",
        type=int,
        default=524288,
    )
    parser.add_argument(
        "--sample_rate",
        help="Sample rate",
        type=int,
        default=48000,
    )
    parser.add_argument(
        "--warmup",
        help="Number of warmup samples during processing.",
        type=int,
        default=48000,
    )
    parser.add_argument(
        "--silence_threshold",
        help="Silence threshold",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--dataset_name", help="Dataset name", type=str, default=None)
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all audio files in the input directory based on file extensions
    audio_filepaths = [args.input_file]

    print(f"Found {len(audio_filepaths)} file in {args.input_file}")

    if len(audio_filepaths) == 0:
        raise ValueError(f"No audio files found in {args.input_file}.")

    # find all vst3 files
    vst_filepaths = []
    for ext in ["*.vst3"]:
        vst_filepaths += glob.glob(os.path.join(args.vst_dir, ext), recursive=True)

    vst_filepaths = sorted(vst_filepaths)
    print(f"Found {len(vst_filepaths)} VST3 files in {args.vst_dir}.")

    vst_plugins = {}
    print("Loading plugins...")
    for filepath in tqdm(vst_filepaths):
        instance_name = os.path.basename(filepath).replace(".vst3", "")
        print(f"Loading {instance_name}...")
        # try to load each plugin
        try:
            plugin = pedalboard.load_plugin(filepath)
        except Exception:
            print(f"Error loading {filepath}.")
            continue

        print(f"Loaded {os.path.basename(filepath)} successfully.")

        presets = []
        if args.presets:
            # find all presets for this plugin
            preset_filepath = os.path.join(
                args.presets, instance_name, f"{instance_name}.json"
            )

            # check if preset file exists
            if not os.path.exists(preset_filepath):
                print(f"No preset file found for {instance_name}.")
                continue

            # load presets
            with open(preset_filepath, "r") as f:
                plugin_info = json.load(f)

        vst_plugins[instance_name] = {
            "filepath": filepath,
            "plugin": plugin,
            "num_channels": plugin_info["num_channels"],
            "presets": plugin_info["presets"],
        }

    for idx, (instance_name, vst_plugin) in enumerate(vst_plugins.items()):
        print(idx + 1, instance_name)

    # generate examples
    # iterate over each audio file that was located
    # split into non-overlapping frames and process with a random plugin and preset
    for file_idx, audio_filepath in enumerate(tqdm(audio_filepaths)):
        print(audio_filepath)
        # load the entire audio file into memory
        audio, sr = torchaudio.load(audio_filepath)

        # peak normalize the entire file
        audio /= torch.max(torch.abs(audio)).clamp(1e-8)

        # check sample rate and resample if necessary
        if sr != args.sample_rate:
            audio = torchaudio.transforms.Resample(sr, args.sample_rate)(audio)

        # get number of channels and sequence length
        chs, seq_len = audio.shape

        # if the audio is too short, repeat pad it
        if seq_len < args.length + args.warmup:
            num_repeats = int(np.ceil((args.length + args.warmup) / seq_len))
            audio = audio.repeat(1, num_repeats)

        # compute number of segments given the length
        num_segments = seq_len // args.length

        audio_segments = []
        # iterate over audio file and pick out segments
        for segment_idx in range(num_segments):
            start_idx = segment_idx * args.length
            end_idx = (segment_idx + 1) * args.length + args.warmup
            audio_segment = audio[:, start_idx:end_idx]

            # check if the segment is silent
            if torch.mean(torch.abs(audio_segment)) < args.silence_threshold:
                continue  # if so, skip this segment

            audio_segments.append(audio_segment)

        if len(audio_segments) == 0:
            raise ValueError("No audio segments found.")

        # pick at random one segment
        audio_segment = random.choice(audio_segments)

        # create a unique id for this example
        uid = uuid.uuid4()

        # save the input file
        os.makedirs(os.path.join(args.output_dir, f"{uid}"), exist_ok=True)
        input_filepath = os.path.join(args.output_dir, f"{uid}", "input.flac")
        torchaudio.save(
            input_filepath,
            audio_segment,
            args.sample_rate,
            backend="soundfile",
        )

        # iterate over plugin instance names
        for instance_name in vst_plugins.keys():
            print(instance_name)
            if args.fixed_preset:  # pick the first preset
                preset_index = 0
            else:  # pick a random preset
                preset_index = random.randint(
                    0, len(vst_plugins[instance_name]["presets"]) - 1
                )

            # process the audio segment with the plugin
            process_function(
                audio_segment,
                vst_plugins[instance_name],
                preset_index,
                args.warmup,
                args.sample_rate,
                args.silence_threshold,
                args.output_dir,
            )
