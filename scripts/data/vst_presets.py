import os
import glob
import json
import random
import librosa
import pedalboard
import numpy as np
import soundfile as sf

from tqdm import tqdm
from sklearn.cluster import KMeans


def process_with_random_plugin_parameters(
    plugin,
    audio_input: np.ndarray,
    sample_rate: float,
    diff_threshold: float = 1e-2,
    silence_threshold: float = 1e-2,
    max_tries: int = 100,
    random_gain: bool = True,
    clip: bool = False,
):
    same = True
    silent = True
    tries = 0

    if random_gain:
        # randomly set gain between -12 and 0 dB
        gain_db = random.uniform(-12, 0)
        gain_lin = 10 ** (gain_db / 20)
        audio_input = audio_input * gain_lin

    while (same or silent) and tries < max_tries:
        parameters = {}
        parameters_normalized = {}
        parameters_index = {}

        # randomly set parameters
        for name, parameter in plugin.parameters.items():
            # check if the string "bypass" is in the name
            if "bypass" in name.lower():
                parameter.raw_value = 0
            elif name in ["buffer_size_frames", "sample_rate_frames"]:
                pass
            else:
                # get all of the valid values for the parameter
                valid_values = parameter.valid_values

                # uniformly sample a random value from the valid values
                # random_value = random.choice(valid_values)
                random_value = np.random.rand()

                # set the parameter to the random value
                # setattr(plugin, name, random_value)

                parameter.raw_value = random_value

                # store results
                parameters[name] = random_value
                parameters_normalized[name] = parameter.raw_value
                # parameters_index[name] = valid_values.index(random_value)

        # render audio with effect
        audio_output = plugin.process(audio_input, sample_rate)

        if clip:
            audio_output = np.clip(audio_output, a_min=-1.0, a_max=1.0)

        tries += 1  # increment tries

        # check for silence
        energy = np.mean(np.abs(audio_output))
        if energy > silence_threshold:
            silent = False

        # check if the difference is below the threshold
        # measure difference (L1 error in time domain)
        audio_input_norm = audio_input / np.clip(
            np.max(np.abs(audio_input)),
            a_min=1e-8,
            a_max=None,
        )
        audio_output_norm = audio_output / np.clip(
            np.max(np.abs(audio_output)),
            a_min=1e-8,
            a_max=None,
        )
        diff = np.mean(np.abs(audio_input_norm - audio_output_norm))
        if diff > diff_threshold:
            same = False

        print(f"tries: {tries}  diff: {diff:0.3f}  silent: {energy:0.3f}   {plugin}")

    return audio_output, parameters


def is_silent(audio, threshold=1e-4):
    energy = np.mean(np.abs(audio))
    if energy > threshold:
        return False
    else:
        return True


def is_different(audio_input, audio_output, threshold=1e-4):
    audio_input_norm = audio_input / np.clip(
        np.max(np.abs(audio_input)),
        a_min=1e-8,
        a_max=None,
    )
    audio_output_norm = audio_output / np.clip(
        np.max(np.abs(audio_output)),
        a_min=1e-8,
        a_max=None,
    )
    diff = np.sum(np.abs(audio_input_norm - audio_output_norm))
    if diff > threshold:
        return True
    else:
        return False


def create_adjusted_nd_grid(total_samples, dimensions):
    # Calculate the number of points per dimension
    # The nth root of total_samples gives an equal distribution of points across dimensions
    points_per_dim = int(total_samples ** (1 / dimensions))
    print(points_per_dim)

    # Generate linspace for each dimension
    linspace = np.linspace(0, 1, points_per_dim)

    # Generate the ND grid
    grid = np.meshgrid(*[linspace for _ in range(dimensions)], indexing="ij")

    # Reshape and combine into a list of vectors
    grid_shape = grid[0].shape
    vectors = [
        np.array([grid[dim][index] for dim in range(dimensions)])
        for index in np.ndindex(grid_shape)
    ]

    return vectors


def grid_sample_from_plugin(
    plugin,
    audio_input: np.ndarray,
    sample_rate: float,
    total_samples: int,
    diff_threshold: float = 1e-4,
    silence_threshold: float = 1e-4,
    random_gain: bool = False,
    clip: bool = False,
    subsampling_factor: int = 4,
    frame_size: int = 1024,
):
    # find number of parameters
    num_params = len(plugin.parameters)
    print(num_params)

    # first create a grid of parameters
    grid = create_adjusted_nd_grid(total_samples, num_params)

    # storage for outputs
    audio_outputs = []

    for params in grid:
        print(params)
        for idx, (name, parameter) in enumerate(plugin.parameters.items()):
            # check if the string "bypass" is in the name
            if "bypass" in name.lower():
                parameter.raw_value = 0
            elif name in ["buffer_size_frames", "sample_rate_frames"]:
                pass
            else:
                parameter.raw_value = params[idx]

        # render audio with effect
        audio_output = plugin.process(audio_input, sample_rate)

        # check conditions
        silent = is_silent(audio_output, threshold=silence_threshold)
        different = is_different(audio_input, audio_output, threshold=diff_threshold)

        if silent or not different:
            continue

        audio_output = np.reshape(audio_output, (-1))
        audio_output = audio_output / np.clip(
            np.max(np.abs(audio_output)),
            a_min=1e-8,
            a_max=None,
        )
        # subsample the output into chunks
        num_frames = audio_output.shape[-1] // (frame_size * subsampling_factor)
        tmp_audio_output = np.zeros((int(frame_size * num_frames)))
        for n in range(num_frames):
            write_start_idx = n * frame_size
            write_stop_idx = write_start_idx + frame_size

            read_start_idx = subsampling_factor * write_start_idx
            read_stop_idx = read_start_idx + frame_size

            frame = audio_output[read_start_idx:read_stop_idx]

            if frame.shape[0] == frame_size:
                tmp_audio_output[write_start_idx:write_stop_idx]
            else:
                break

        # store results
        audio_outputs.append(tmp_audio_output)

    return audio_outputs


if __name__ == "__main__":
    root_dir = "plugins/valid"
    preset_dir = "plugins/presets"
    sample_rate = 48000
    num_generations = 1000
    num_presets = 10
    random_sampling = True

    # find all vst3 plugins
    vst_filepaths = glob.glob(os.path.join(root_dir, "*.vst3"), recursive=True)
    vst_filepaths = sorted(vst_filepaths)
    print(f"Fount {len(vst_filepaths)} VST3 files in {root_dir}.")

    for vst_filepath in vst_filepaths:
        instance_name = os.path.basename(vst_filepath).replace(".vst3", "")

        # check if we already have generated presets for this plugin
        preset_filepath = os.path.join(
            preset_dir, instance_name, f"{instance_name}.json"
        )
        if os.path.exists(preset_filepath):
            print(f"Skipping {instance_name}...")
            continue

        print(f"Loading {instance_name}...")
        # try to load each plugin
        try:
            plugin = pedalboard.load_plugin(vst_filepath)
        except Exception:
            print(f"Error loading {vst_filepath}.")

        # try to pass audio through the plugin
        failed = True
        try:
            num_channels = 2
            audio_input = np.random.randn(2, sample_rate * 5)
            audio_output = plugin(audio_input, sample_rate)
            failed = False
        except Exception as e:
            num_channels = 1
            print(e)
            print(f"Error processing audio with 2 channels.")

        try:
            audio_input = np.random.randn(1, sample_rate * 5)
            audio_output = plugin(audio_input, sample_rate)
            failed = False
        except Exception as e:
            num_channels = 2
            print(e)
            print(f"Failed processing audio with 1 channel.")

        if failed:
            continue

        print(f"Loaded {os.path.basename(vst_filepath)} successfully.")

        # load input audio for processing
        audio_input, sr = sf.read("audio/pop_5_90BPM.wav", always_2d=True)
        audio_input = audio_input.T
        start_idx = int(sr * 10)
        stop_idx = start_idx + int(sr * 3)
        audio_input = audio_input[:, start_idx:stop_idx]

        # make dual mono if necessary
        if audio_input.shape[0] == 1 and num_channels == 2:
            audio_input = np.concatenate((audio_input, audio_input), axis=0)

        if audio_input.shape[0] == 2 and num_channels == 1:
            audio_input = audio_input[0:1, :]

        # get the valid values for each parameter
        parameters = []
        for name, parameter in plugin.parameters.items():
            valid_values = parameter.valid_values
            parameters.append({"name": name, "valid_values": valid_values})

        # store plugin details
        details = {
            "instance": instance_name,
            "parameters": parameters,
            "num_channels": num_channels,
        }

        preset_parameters = []
        audio_outputs = []
        features = []
        # generate random outputs from the plugin storing parameters and outputs
        if random_sampling:
            for n in tqdm(range(num_generations)):
                # process audio with random parameters
                (
                    audio_output,
                    params,
                ) = process_with_random_plugin_parameters(
                    plugin,
                    audio_input,
                    sample_rate,
                )

                # store parameters and output
                preset_parameters.append(params)

                # peak normalize all outputs before clustering
                audio_output = audio_output / np.clip(
                    np.max(np.abs(audio_output)),
                    a_min=1e-8,
                    a_max=None,
                )

                audio_outputs.append(audio_output)

                # flatten stereo audio into single vector
                audio_output = np.reshape(audio_output, (-1))

                # compute mfcc features
                mfccs = librosa.feature.mfcc(
                    y=audio_output,
                    sr=sample_rate,
                )
                mfccs = np.reshape(mfccs, (-1))
                features.append(mfccs)

        else:
            # use grid method
            audio_outputs = grid_sample_from_plugin(
                plugin, audio_input, sample_rate, num_generations
            )

        X = np.stack(features, axis=0)
        print(X.shape)

        # perform clustering to discover presets
        print("Clustering...")
        kmeans = KMeans(n_clusters=num_presets, random_state=0, n_init="auto").fit(X)
        preset_indices = kmeans.labels_

        # save preset parameters
        pruned_preset_parameters = []
        pruned_preset_audio_outputs = []

        for preset_index in range(num_presets):
            cluster_index = np.where(preset_indices == preset_index)[0][0]
            preset_params = preset_parameters[cluster_index]
            audio_output = audio_outputs[cluster_index]
            pruned_preset_parameters.append(preset_params)
            pruned_preset_audio_outputs.append(audio_output)
            print(preset_index, preset_params)

        details["presets"] = pruned_preset_parameters

        preset_plugin_dir = os.path.join(preset_dir, instance_name)
        os.makedirs(preset_plugin_dir, exist_ok=True)
        json_filepath = os.path.join(preset_plugin_dir, f"{instance_name}.json")
        with open(json_filepath, "w") as f:
            json.dump(details, f, indent=4)

        # save preset audio
        for preset_index, audio_output in enumerate(pruned_preset_audio_outputs):
            audio_output = np.reshape(audio_output, (num_channels, -1))
            audio_output = audio_output.T
            audio_output_filepath = os.path.join(
                preset_plugin_dir, f"{instance_name}_{preset_index}.wav"
            )
            sf.write(audio_output_filepath, audio_output, sample_rate)
