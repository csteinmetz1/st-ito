import os
import torch
import argparse
import torchaudio
import pedalboard
import numpy as np

from typing import List

from lcap.effects import (
    BasicParametricEQ,
    BasicCompressor,
    BasicDistortion,
    BasicDelay,
    BasicReverb,
)


def process_audio(x: np.ndarray, w: np.ndarray, sr: int, plugins: List[dict]):
    """Process audio with plugins and provided parameters on [0, 1].

    Args:
        x (np.ndarray): Audio vector of shape (1, num_samples)
        w (np.ndarray): Parameter vector of shape (num_params,)
        sr (int): Sample rate
        plugins (List[dict]): List of plugin dicts
    """
    widx = 0
    for plugin_name, plugin in plugins.items():
        for name, parameter in plugin["instance"].parameters.items():
            if name in plugin["fixed_parameters"]:
                parameter.raw_value = plugin["fixed_parameters"][name]
                widx += 1
            else:
                parameter.raw_value = w[widx]
                widx += 1

        if plugin["num_channels"] == 2 and x.shape[0] == 1:
            x = np.concatenate((x, x), axis=0)

        # process audio
        x = plugin["instance"].process(x, sample_rate=sr)

    # crop early transient and make mono
    x = x[:, 1024:]

    # peak normalize
    x /= np.clip(np.max(np.abs(x)), a_min=1e-8, a_max=None)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize audio with random effects."
    )
    parser.add_argument("input_audio", type=str, help="Path to input audio file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="audio/synth",
        help="Path to output directory.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # define plugins
    plugins = {
        "ParametricEQ": {
            "class_path": BasicParametricEQ,
            "num_params": None,
            "num_channels": 1,
            "fixed_parameters": {},
        },
        "Compressor": {
            "class_path": BasicCompressor,
            "num_params": None,
            "num_channels": 1,
            "fixed_parameters": {},
        },
        "Distortion": {
            "class_path": BasicDistortion,
            "num_params": None,
            "num_channels": 1,
            "fixed_parameters": {},
        },
        "Delay": {
            "class_path": BasicDelay,
            "num_params": None,
            "num_channels": 2,
            "fixed_parameters": {},
        },
        "Reverb": {
            "class_path": BasicReverb,
            "num_params": None,
            "num_channels": 2,
            "fixed_parameters": {},
        },
    }

    # load plugins
    total_num_params = 0
    init_params = []
    for plugin_name, plugin in plugins.items():
        if "vst_filepath" in plugin:
            plugin_instance = pedalboard.load_plugin(plugin["vst_filepath"])
        elif "class_path" in plugin:
            plugin_instance = plugin["class_path"]()
        else:
            raise ValueError(f"Plugin must contain 'vst_filepath' or 'class_path'.")

        num_params = 0
        for name, parameter in plugin_instance.parameters.items():
            num_params += 1
            print(f"{plugin_name}: {name} = {parameter.raw_value}")
            init_params.append(parameter.raw_value)
        print()

        plugin["num_params"] = num_params
        plugin["instance"] = plugin_instance
        total_num_params += num_params

    # load audio
    file_id = os.path.basename(args.input_audio).split(".")[0]
    input_audio, sr = torchaudio.load(args.input_audio)

    # resample audio
    if sr != 48000:
        input_audio = torchaudio.transforms.resample(input_audio, sr, 48000)

    # apply effects with random parameters
    w = torch.rand(total_num_params)

    output_audio = process_audio(input_audio.numpy(), w.numpy(), 48000, plugins)

    # save audio
    output_path = os.path.join(args.output_dir, f"target_{file_id}.wav")
    input_path = os.path.join(args.output_dir, f"input_{file_id}.wav")

    torchaudio.save(output_path, torch.tensor(output_audio), 48000)
    torchaudio.save(input_path, input_audio, 48000)
