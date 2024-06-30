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
from typing import List, Dict
import multiprocessing as mp


def get_nonsilent_region(
    audio_filepath: str,
    length: int,
    max_tries=100,
    threshold: float = 1e-4,
):
    # get number of frames
    num_frames = torchaudio.info(audio_filepath).num_frames
    silent = True
    tries = 0
    while silent and tries < max_tries:
        start = random.randint(0, num_frames - (length))
        audio_input, sr = torchaudio.load(
            audio_filepath,
            frame_offset=start,
            num_frames=length,
        )

        tries += 1

    if silent:
        raise ValueError(f"Could not find a non-silent region in {audio_filepath}.")

    return audio_input, sr


def process_with_preset(
    plugin: pedalboard.VST3Plugin,
    audio: torch.Tensor,
    sample_rate: int,
    preset: dict,
):
    # configure the plugin based on the preset
    for name, parameter in plugin.parameters.items():
        if name in preset:
            # set the parameter to the preset value
            parameter.raw_value = preset[name]

    # now process audio
    audio_output = plugin.process(audio.numpy(), sample_rate)
    audio_output = torch.from_numpy(audio_output)

    # peak normalize if out of range
    if torch.max(torch.abs(audio_output)) > 1.0:
        audio_output /= torch.max(torch.abs(audio_output)).clamp(1e-8)

    return audio_output


def process_audio(x: torch.Tensor, w: torch.Tensor, sr: int, plugins: List[dict]):
    """Process audio with plugins and provided parameters on [0, 1].

    Args:
        x (torch.Tensor): Audio tensor of shape (1, num_samples)
        w (torch.Tensor): Parameter tensor of shape (num_params,)
        sr (int): Sample rate
        plugins (List[dict]): List of plugin dicts
    """
    widx = 0
    params = {}
    for plugin_name, plugin in plugins.items():
        for name, parameter in plugin["instance"].parameters.items():
            parameter_key = f"{plugin_name}-{name}"
            if name in plugin["fixed_parameters"]:
                params[parameter_key] = plugin["fixed_parameters"][name]
                parameter.raw_value = plugin["fixed_parameters"][name]
            else:
                params[parameter_key] = w[widx].item()
                parameter.raw_value = w[widx]
            widx += 1

        if plugin["num_channels"] == 2:
            x = x.repeat(2, 1)

        # process audio
        x = plugin["instance"].process(x.cpu().numpy(), sample_rate=sr)
        x = torch.from_numpy(x)

        if plugin["num_channels"] == 2:
            x = x[0:1, :]

    return x, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory", type=str)
    parser.add_argument(
        "vst_json", help="JSON file specifying the VST effect chain.", type=str
    )
    parser.add_argument(
        "--num_examples",
        help="Number of examples to generate",
        type=int,
        default=1000,
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
        default=1e-2,
    )
    parser.add_argument("--dataset_name", help="Dataset name", type=str, default=None)
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))

    # find all audio files in the input directory based on file extensions
    audio_filepaths = []
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
        audio_filepaths += glob.glob(
            os.path.join(args.input_dir, "**", ext), recursive=True
        )

    print(f"Found {len(audio_filepaths)} files in {args.input_dir}")

    if len(audio_filepaths) == 0:
        raise ValueError(f"No audio files found in {args.input_dir}.")

    # load the json file specifying the VST effect chain
    with open(args.vst_json, "r") as fp:
        plugins = json.load(fp)

    total_num_params = 0
    init_params = []
    for plugin_name, plugin in plugins.items():
        plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
        num_params = 0
        for name, parameter in plugin_instance.parameters.items():
            num_params += 1
            print(f"{plugin_name}: {name} = {parameter.raw_value}")
            init_params.append(parameter.raw_value)
        print()

        plugin["num_params"] = num_params
        plugin["instance"] = plugin_instance
        total_num_params += num_params

    audio_filepath_iter = itertools.cycle(audio_filepaths)

    # generate examples
    # iterate over each audio file that was located
    finished = False
    num_generated = 0
    while not finished:
        audio_filepath = next(audio_filepath_iter)
        print(audio_filepath)
        # load the entire audio file into memory
        audio, sr = torchaudio.load(audio_filepath, backend="soundfile")

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
            audio_segment = audio[0:1, start_idx:end_idx]
            # note: we crop to mono for now...

            # check if the segment is silent
            if torch.mean(torch.abs(audio_segment)) < args.silence_threshold:
                continue  # if so, skip this segment

            audio_segments.append(audio_segment)

        if len(audio_segments) == 0:
            continue

        # pick at random one segment
        audio_segment = random.choice(audio_segments)

        # create a unique id for this example
        uid = uuid.uuid4()

        # save the input file
        os.makedirs(os.path.join(args.output_dir, f"{uid}"), exist_ok=True)
        input_filepath = os.path.join(args.output_dir, f"{uid}", "input.wav")
        torchaudio.save(
            input_filepath,
            audio_segment,
            args.sample_rate,
            backend="soundfile",
        )

        # process the audio with a random parameters
        output_segment, params = process_audio(
            audio_segment, torch.rand(total_num_params), args.sample_rate, plugins
        )

        # save the output file
        output_filepath = os.path.join(args.output_dir, f"{uid}", "output.wav")
        torchaudio.save(
            output_filepath,
            output_segment,
            args.sample_rate,
            backend="soundfile",
        )

        # save the parameters
        with open(os.path.join(args.output_dir, f"{uid}", "params.json"), "w") as fp:
            json.dump(params, fp, indent=4)

        num_generated += 1  # increment the number of generated examples
        if num_generated >= args.num_examples:
            finished = True
