import os
import glob
import uuid
import torch
import random
import argparse
import torchaudio
import pedalboard
import numpy as np

from tqdm import tqdm
from typing import List, Dict


def get_nonsilent_region(
    audio_input: torch.Tensor,
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


def load_plugins(plugin_dir: str, sample_rate: float) -> List[Dict]:
    vst_filepaths = glob.glob(os.path.join(plugin_dir, "**", "*.vst3"), recursive=True)

    # find all vst3 files
    vst_filepaths = []
    for ext in ["*.vst3"]:
        vst_filepaths += glob.glob(os.path.join(plugin_dir, ext), recursive=True)

    vst_filepaths = sorted(vst_filepaths)
    print(f"Found {len(vst_filepaths)} VST3 files in {plugin_dir}.")

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

        # try to pass audio through the plugin
        failed = True
        try:
            num_channels = 2
            audio_input = torch.randn(2, sample_rate * 5)
            audio_output = plugin(audio_input.numpy(), sample_rate)
            failed = False
        except Exception as e:
            num_channels = 1
            print(e)
            print(f"Error processing audio with 2 channels.")

        try:
            audio_input = torch.randn(1, sample_rate * 5)
            audio_output = plugin(audio_input.numpy(), sample_rate)
            failed = False
        except Exception as e:
            num_channels = 2
            print(e)
            print(f"Failed processing audio with 1 channel.")

        vst_plugins[instance_name] = {
            "filepath": filepath,
            "plugin": plugin,
            "num_channels": num_channels,
        }

    return vst_plugins


def process_with_random_parameters(
    plugin,
    audio_input: torch.Tensor,
    sample_rate: float,
    diff_threshold: float = 1e-2,
    silence_threshold: float = 1e-2,
    max_tries: int = 100,
):
    same = True
    silent = True
    tries = 0

    audio_input /= torch.max(torch.abs(audio_input)).clamp(1e-8)

    while (same or silent) and tries < max_tries:
        parameters = {}
        parameters_normalized = {}

        # randomly set parameters
        for name, parameter in plugin.parameters.items():
            # if bypass is in name
            if "bypass" in name.lower():
                parameter.raw_value = 0
            elif name in ["buffer_size_frames", "sample_rate_frames"]:
                pass
            else:
                random_value = random.uniform(0, 1)
                # set the parameter to the random value
                parameter.raw_value = random_value

                # store results
                parameters[name] = random_value
                parameters_normalized[name] = parameter.raw_value

                print(parameter)

        # render audio with effect
        audio_output = plugin.process(audio_input.numpy(), sample_rate)
        audio_output = torch.from_numpy(audio_output)

        tries += 1  # increment tries

        # check for silence
        energy = torch.mean(torch.abs(audio_output))
        if energy > silence_threshold:
            silent = False

        # check if the difference is below the threshold
        # measure difference (L1 error in time domain)
        audio_input_norm = audio_input / torch.max(torch.abs(audio_input)).clamp(1e-8)
        audio_output_norm = audio_output / torch.max(torch.abs(audio_output)).clamp(
            1e-8
        )
        diff = torch.sum(torch.abs(audio_input_norm - audio_output_norm))
        if diff > diff_threshold:
            same = False

        print(f"tries: {tries}  diff: {diff:0.3f}  silent: {energy:0.3f}   {plugin}")

    return audio_output, parameters, parameters_normalized


def process(
    x: torch.Tensor,
    sample_rate: int,
    plugin: pedalboard.Pedalboard,
    num_channels: int,
    parameters: dict = None,
):
    # make dual mono if necessary
    if x.shape[0] == 1 and num_channels == 2:
        x = torch.cat((x, x), axis=0)
    if x.shape[0] == 2 and num_channels == 1:
        x = x[0:1, :]

    # make sure the effect is not bypassed
    for name, parameter in plugin.parameters.items():
        if "bypass" in name.lower():
            parameter.raw_value = 0

    # set parameters if necessary
    if parameters is not None:
        for name, value in parameters.items():
            if name in plugin.parameters:
                parameter.raw_value = value
            else:
                raise ValueError(f"Parameter {name} not found in plugin.")

    for name, parameter in plugin.parameters.items():
        print(f"{name}: {parameter.raw_value}")

    x_processed = plugin.process(x.numpy(), sample_rate)
    x_processed = torch.from_numpy(x_processed)

    # peak normalize
    x_processed /= torch.max(torch.abs(x_processed)).clamp(1e-8)

    return x_processed


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
        "plugin_dir", type=str, help="Path to root directory of VST3 plugins."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory.",
    )
    parser.add_argument("--sample_rate", type=float, default=48000)
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--intra_effect_hard", action="store_true")
    parser.add_argument("--intra_effect_easy", action="store_true")
    parser.add_argument("--inter_effect", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all audio files in the input directory based on file extensions
    audio_filepaths = []
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
        audio_filepaths += glob.glob(
            os.path.join(args.input_dir, "**", ext), recursive=True
        )

    # find all VST3 plugins in the plugin directory
    vst_plugins = load_plugins(args.plugin_dir, args.sample_rate)

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
        for instance_name, vst_plugin in tqdm(vst_plugins.items()):
            num_channels = vst_plugin["num_channels"]
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

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, args.length, threshold=args.threshold
                    )

                    # make dual mono if necessary
                    if x_segment.shape[0] == 1 and num_channels == 2:
                        x_segment = torch.cat((x_segment, x_segment), axis=0)
                    if x_segment.shape[0] == 2 and num_channels == 1:
                        x_segment = x_segment[0:1, :]

                    audio_segments.append(x_segment)

                # repeat the first audio segment
                audio_segments.append(audio_segments[0])

                for name, x_segment in zip(names, audio_segments):
                    # process with random parameters
                    (
                        x_segment_processed,
                        parameters,
                        parameters_normalized,
                    ) = process_with_random_parameters(
                        vst_plugin["plugin"],
                        x_segment,
                        sr,
                    )

                    # peak normalize
                    if torch.max(torch.abs(x_segment_processed)) > 1.0:
                        x_segment_processed /= torch.max(
                            torch.abs(x_segment_processed)
                        ).clamp(1e-8)

                    # save result
                    filepath = os.path.join(example_dir, f"{name}-{instance_name}.flac")
                    torchaudio.save(filepath, x_segment_processed, sr)

    # generate the second mode: inter-effect
    # in this mode we want to test if the model is focused more on content or style
    # we select two different audios and two different plugins
    # (x) audio 1 + plugin 1
    # (a) audio 1 + plugin 2
    # (b) audio 2 + plugin 1
    # (c) audio 2 + plugin 2
    # In this case, the human and model should say that (a) and (c) are more similar
    # since they have the same plugin applied to them. (b) should be more different
    if args.inter_effect:
        inter_effect_dir = os.path.join(args.output_dir, "inter-effect")
        os.makedirs(inter_effect_dir, exist_ok=True)

        for instance_name, vst_plugin in tqdm(vst_plugins.items()):
            num_channels = vst_plugin["num_channels"]
            plugin_dir = os.path.join(inter_effect_dir, instance_name)
            os.makedirs(plugin_dir, exist_ok=True)
            for n in range(args.num_examples):
                # generate a uuid for this example
                uid = str(uuid.uuid4())
                example_dir = os.path.join(plugin_dir, uid)
                os.makedirs(example_dir, exist_ok=True)

                # randomly select three audio files
                random_audio_filepaths = np.random.choice(audio_filepaths, size=2)
                names = ["x", "a", "b", "c"]

                audio_segments = []
                for audio_filepath in random_audio_filepaths:
                    # load audio
                    x, sr = torchaudio.load(audio_filepath, backend="soundfile")

                    if sr != args.sample_rate:
                        x = torchaudio.functional.resample(x, sr, args.sample_rate)

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, args.length, threshold=args.threshold
                    )

                    audio_segments.append(x_segment)

                # repeat the first and second audio segments
                audio_segments = [
                    audio_segments[0],
                    audio_segments[0],
                    audio_segments[1],
                    audio_segments[1],
                ]

                # select the distractor plugin
                distractor_instance_name = np.random.choice(list(vst_plugins.keys()))
                vst_plugin_distractor = vst_plugins[distractor_instance_name]

                # todo: we need to randomize the parameters for both plugins here
                plugin1_parameters = {}
                plugin2_parameters = {}

                for name, parameter in vst_plugin["plugin"].parameters.items():
                    if "bypass" in name.lower():
                        pass
                    else:
                        plugin1_parameters[name] = np.random.rand()

                for name, parameter in vst_plugin_distractor[
                    "plugin"
                ].parameters.items():
                    if "bypass" in name.lower():
                        pass
                    else:
                        plugin2_parameters[name] = np.random.rand()

                plugins = [
                    vst_plugin["plugin"],
                    vst_plugin_distractor["plugin"],
                    vst_plugin["plugin"],
                    vst_plugin_distractor["plugin"],
                ]

                channel_counts = [
                    vst_plugin["num_channels"],
                    vst_plugin_distractor["num_channels"],
                    vst_plugin["num_channels"],
                    vst_plugin_distractor["num_channels"],
                ]

                plugin_parameters = [
                    plugin1_parameters,
                    plugin2_parameters,
                    plugin1_parameters,
                    plugin2_parameters,
                ]

                plugin_names = [
                    instance_name,
                    distractor_instance_name,
                    instance_name,
                    distractor_instance_name,
                ]

                for (
                    name,
                    audio_segment,
                    plugin,
                    plugin_name,
                    num_channels,
                    parameters,
                ) in zip(
                    names,
                    audio_segments,
                    plugins,
                    plugin_names,
                    channel_counts,
                    plugin_parameters,
                ):
                    x_segment_processed = process(
                        audio_segment,
                        args.sample_rate,
                        plugin,
                        num_channels,
                        parameters,
                    )
                    # save result
                    filepath = os.path.join(example_dir, f"{name}-{plugin_name}.flac")
                    torchaudio.save(filepath, x_segment_processed, sr)

    # generate the third mode: intra-effect
    # in this mode we apply the same plugin to multiple audios
    # (x) audio 1 + plugin 1 + params 1
    # (a) audio 2 + plugin 1 + params 1
    # (b) audio 3 + plugin 1 + params 2
    # (c) audio 1 + plugin 1 + params 2
    # In this case, the human and model should say that (a) and (b) are more similar
    if args.intra_effect_easy:
        intra_effect_dir = os.path.join(args.output_dir, "intra-effect-easy")
        os.makedirs(intra_effect_dir, exist_ok=True)

        for instance_name, vst_plugin in tqdm(vst_plugins.items()):
            num_channels = vst_plugin["num_channels"]
            plugin_dir = os.path.join(intra_effect_dir, instance_name)
            os.makedirs(plugin_dir, exist_ok=True)

            # create two sets of parameters for the plugin
            parameters1 = {}
            parameters2 = {}
            for name, parameter in vst_plugin["plugin"].parameters.items():
                if "bypass" in name.lower():
                    pass
                elif name in ["buffer_size_frames", "sample_rate_frames"]:
                    pass
                else:
                    parameters1[name] = np.random.rand()
                    parameters2[name] = np.random.rand()

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

                    # find a non-silent segment of length
                    x_segment = get_nonsilent_region(
                        x, args.length, threshold=args.threshold
                    )

                    audio_segments.append(x_segment)

                # repeat the first and second audio segments
                audio_segments = [
                    audio_segments[0],
                    audio_segments[1],
                    audio_segments[2],
                    audio_segments[0],
                ]

                for name, audio_segment in zip(names, audio_segments):
                    # process with random parameters
                    if name in ["x", "a"]:
                        parameter_name = "params1"
                        x_segment_processed = process(
                            audio_segment,
                            args.sample_rate,
                            vst_plugin["plugin"],
                            num_channels,
                            parameters=parameters1,
                        )
                    else:
                        parameter_name = "params2"
                        x_segment_processed = process(
                            audio_segment,
                            args.sample_rate,
                            vst_plugin["plugin"],
                            num_channels,
                            parameters=parameters2,
                        )

                    # save result
                    filepath = os.path.join(
                        example_dir, f"{name}-{instance_name}-{parameter_name}.flac"
                    )
                    torchaudio.save(filepath, x_segment_processed, sr)
