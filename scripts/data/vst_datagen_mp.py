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


def worker_function(
    vst_plugins: dict,
    warmup: int,
    sample_rate: int,
    silence_threshold: float,
    output_dir: str,
    pipe,
):
    """

    Args:
        vst_plugins (dict): Dictionary of plugin instances and their presets.
        warmup (int): Number of warmup samples during processing.
        sample_rate (int): Sample rate.
        silence_threshold (float): Silence threshold.
        output_dir (str): Output directory.
        pipe (mp.Pipe): Pipe for communication between processes.

    We create each working passing in a dict of the plugins and their presets.
    Then, when the worker is called, we will pass in the plugin instance name ,audio data, a uuid, and an exit flag.
    The worker will select the appropriate plugin instance and process the audio data with a random preset.
    The worker will then save the audio data and a json file containing the instance name, preset, and dataset name.

    All arguments to this function are the same across all examples.
    Any variable arguments (audio data, uid, etc.) are passed through the pipe.

    """

    # create instances of all plugins for this worker
    worker_vst_plugins = {}
    for instance_name, vst_plugin in tqdm(vst_plugins.items()):
        worker_vst_plugins[instance_name] = {
            "plugin": pedalboard.load_plugin(vst_plugin["filepath"]),
            "num_channels": vst_plugin["num_channels"],
            "presets": vst_plugin["presets"],
        }

    while True:
        # Receive data from the parent thread
        instance_name, preset_index, audio_input, uid, exit_flag = pipe.recv()

        if exit_flag:
            print("Worker thread is exiting gracefully.")
            return

        vst_plugin = worker_vst_plugins[instance_name]  # get the plugin details
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
            if name in preset:  # set the parameter to the preset value
                parameter.raw_value = preset[name]

        # process with the plugin
        try:
            audio_output = plugin.process(audio_input.numpy(), sample_rate)
            audio_output = torch.from_numpy(audio_output)
        except Exception as e:
            print(f"Error processing {instance_name} {preset_index}.")
            print(e)
            raise RuntimeError()

        # remove warmup samples
        audio_input = audio_input[:, warmup:]
        audio_output = audio_output[:, warmup:]

        # check if output is not silent and save
        if torch.mean(torch.abs(audio_output)) > silence_threshold:
            details = {
                "instance": instance_name,
                "preset": preset_index,
                "dataset": dataset_name,
            }

            # output filepaths
            input_filepath = os.path.join(output_dir, f"{uid}.input.wav")
            output_filepath = os.path.join(output_dir, f"{uid}.output.wav")
            json_filepath = os.path.join(output_dir, f"{uid}.json")

            if False:
                # save audio
                torchaudio.save(
                    input_filepath,
                    audio_input,
                    sample_rate,
                    backend="soundfile",
                )
                torchaudio.save(
                    output_filepath,
                    audio_output,
                    sample_rate,
                    backend="soundfile",
                )

                # save json file
                with open(json_filepath, "w") as fp:
                    json.dump(details, fp, indent=True)

            result = True
        else:
            result = False

        # Send the result back to the parent thread
        pipe.send(result)

        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory", type=str)
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
        default=1e-2,
    )
    parser.add_argument("--num_workers", help="Number of workers", type=int, default=4)
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

    # find all vst3 files
    vst_filepaths = []
    for ext in ["*.vst3"]:
        vst_filepaths += glob.glob(os.path.join(args.vst_dir, ext), recursive=True)

    vst_filepaths = sorted(vst_filepaths)
    print(f"Fount {len(vst_filepaths)} VST3 files in {args.input_dir}.")

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
            "num_channels": plugin_info["num_channels"],
            "presets": plugin_info["presets"],
        }

    for idx, (instance_name, vst_plugin) in enumerate(vst_plugins.items()):
        print(idx + 1, instance_name)

    # use multiprocessing to spawn multiple workers
    # each worker will have a copy of all the plugin instances
    processes = []
    for n in range(args.num_workers):
        print(f"Spawning worker process n={n}...")
        # load plugins in each process

        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=worker_function,
            args=(
                vst_plugins,
                args.warmup,
                args.sample_rate,
                args.silence_threshold,
                args.output_dir,
                child_conn,
            ),
        )
        p.start()
        processes.append((p, parent_conn, child_conn))

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

        # pick at random one segment
        audio_segment = random.choice(audio_segments)

        process_index = 0
        # iterate over plugin instance names
        for instance_name in vst_plugins.keys():
            print(process_index, instance_name)
            if args.fixed_preset:  # pick the first preset
                preset_index = 0
            else:  # pick a random preset
                preset_index = random.randint(
                    0, len(vst_plugins[instance_name]["presets"])
                )

            # send data to the running workers
            if process_index < len(processes):
                p, parent_conn, child_conn = processes[process_index]
                uid = uuid.uuid4()  # generate uid in the parent process
                parent_conn.send(
                    (
                        instance_name,
                        preset_index,
                        audio_segment,
                        uid,
                        False,
                    )
                )
                process_index += 1

            # collect the results from the workers
            if process_index == len(processes):
                for idx, process in enumerate(processes):
                    p, parent_conn, child_conn = process
                    instance_name = list(vst_plugins.keys())[idx]
                    print(f"waiting on {instance_name}...")
                    result = parent_conn.recv()
                    print(result)
                process_index = 0  # reset

    # close all processes
    print("Closing processes...")
    for p, parent_conn, child_conn in processes:
        parent_conn.send(
            (audio_segment, None, True)
        )  # send dummy data to unblock the process
        # result = parent_conn.recv()
        p.join()
