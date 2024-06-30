import os
import json
import glob
import torch
import random
import argparse
import torchaudio
import pedalboard

torchaudio.set_audio_backend("soundfile")


def process_with_random_plugin_parameters(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vst_filepath", help="Path to VST under test.", type=str)
    parser.add_argument("--sample_rate", help="Sample rate", type=int, default=48000)
    args = parser.parse_args()

    os.makedirs("debug", exist_ok=True)

    # load a random audio file
    x, sr = torchaudio.load("audio/pop_5_90BPM.wav")

    if sr != args.sample_rate:
        x = torchaudio.transforms.Resample(sr, args.sample_rate)(x)
        sr = args.sample_rate

    x = x[:, 131072 + int(sr * 10) : 131072 + int(sr * 20)]

    result = {
        "name": os.path.basename(args.vst_filepath),
        "filepath": args.vst_filepath,
    }
    print(f"Testing {args.vst_filepath}...")
    # load the plugin
    try:
        plugin = pedalboard.load_plugin(args.vst_filepath)
        print("Loaded.")
        result["status"] = "Loaded"
    except Exception as e:
        print(e)
        result["status"] = "Error"

    # try to pass audio through the plugin
    failed = True
    try:
        num_channels = 2
        audio_input = torch.randn(2, args.sample_rate * 5)
        audio_output = plugin(audio_input.numpy(), args.sample_rate)
        failed = False
    except Exception as e:
        num_channels = 1
        print(e)
        print(f"Error processing audio with 2 channels.")

    try:
        audio_input = torch.randn(1, args.sample_rate * 5)
        audio_output = plugin(audio_input.numpy(), args.sample_rate)
        failed = False
    except Exception as e:
        num_channels = 2
        print(e)
        print(f"Failed processing audio with 1 channel.")

    if failed:
        sys.exit()

    print(f"Loaded {os.path.basename(args.vst_filepath)} successfully.")

    # get the valid values for each parameter
    parameters = []
    for name, parameter in plugin.parameters.items():
        valid_values = parameter.valid_values
        parameters.append({"name": name, "valid_values": valid_values})
        print(f"{name}: {parameter}")

    instance_name = os.path.basename(args.vst_filepath).replace(".vst3", "")

    item = {
        "num_channels": num_channels,
        "parameters": parameters,
    }

    # add plugin instance
    item["plugin"] = plugin

    # now render a few examples with random parameters
    for example_idx in range(10):
        plugin = item["plugin"]
        num_channels = item["num_channels"]

        if num_channels == 1:
            audio_input = x[0:1, :]
        elif num_channels == 2:
            audio_input = x.repeat(2, 1)

        # process audio with random parameters
        (
            audio_output,
            parameters,
            parameters_normalized,
        ) = process_with_random_plugin_parameters(plugin, audio_input, sr)

        print(example_idx)

        # for name, parameter in plugin.parameters.items():
        #    print(f"{name}: {parameter}")

        print(audio_output.shape)
        print(audio_output.min(), audio_output.max())

        # peak normalize
        audio_output /= torch.max(torch.abs(audio_output)).clamp(1e-8)

        # save the audio
        torchaudio.save(
            f"debug/{instance_name}-{example_idx:03d}.wav",
            audio_output,
            sr,
        )
