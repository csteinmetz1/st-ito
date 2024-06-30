# in this case study we will load an audio file and use this for the case stuudy
# we will apply some reverb to the audio file incremently increasing one parameter, such as the reverb size.
# then for each incrememnt we will run the ES style transfer process and see how the predicted reverb size compares to the actual reverb size.
# we can then create a plot of the true parameter value and the predicted value, while they may not align perfectly, we would expect to see a positive correlation between the two values.
import os
import glob
import json
import torch
import torchaudio
import pedalboard
import numpy as np
import pytorch_lightning as pl

from st_ito.style_transfer import run_es, process_audio, load_plugins
from st_ito.utils import (
    load_param_model,
    get_param_embeds,
    load_clap_model,
    get_clap_embeds,
)

from st_ito.effects import (
    BasicParametricEQ,
    BasicCompressor,
    BasicDistortion,
    BasicDelay,
    BasicReverb,
    BasicChorus,
)

if __name__ == "__main__":
    pl.seed_everything(42)

    num_runs = 3
    num_steps = 4
    max_iters = 5
    save_audio = True
    audio_dirs = [
        "/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/acoustic_mic",
        "/import/c4dm-datasets-ext/IDMT_SMT_GUITAR/dataset4/Ibanez 2820",
        "/import/c4dm-datasets/VocalSet1-2/data_by_singer",
        "/import/c4dm-datasets/daps_dataset/clean",
        "/import/c4dm-datasets/ENST-drums-mixes",
    ]
    modes = ["different"]

    # find all wav files in audio_dir
    audio_subsets = []
    for audio_dir in audio_dirs:
        audio_filepaths = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        print(audio_dir, len(audio_filepaths))
        audio_subsets.append(audio_filepaths)

    # load model
    methods = {
        "param-panns": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
            ),
            "embed_func": get_param_embeds,
        },
        "clap": {
            "model": load_clap_model(),
            "embed_func": get_clap_embeds,
        },
    }

    plugin_names = [
        # "vst_RoughRider3",
        # "vst_DragonflyPlateReverb",
        # "vst_3BandEQ",
        # "vst_MaGigaverb",
        # "vst_MetalTone",
        # "vst_TAL-Chorus-LX",
        # "pb_Chorus",
        # "pb_Reverb",
        # "pb_Delay",
        "pb_Distortion",
        "pb_Compressor",
        "pb_ParametricEQ",
    ]

    for plugin_name in plugin_names:
        os.makedirs(f"output/new_case_study/{plugin_name}/audio", exist_ok=True)
        if plugin_name == "vst_RoughRider3":
            parameter_under_test = "sensitivity_db"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "vst_RoughRider3": {
                    "vst_filepath": "plugins/valid/RoughRider3.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "sc_hpf_hz": 0.0,
                        "input_lvl_db": 0.9230769276618958,
                        "ratio": 0.5,
                        "attack_ms": 0.09990999102592468,
                        "release_ms": 0.24242424964904785,
                        "makeup_db": 0.1666666716337204,
                        "mix": 1.0,
                        "output_lvl_db": 0.9230769276618958,
                        "sc_active": 0.0,
                        "full_bandwidth": 1.0,
                        "bypass": 0.0,
                        "program": 0.0,
                    },
                }
            }
        elif plugin_name == "vst_DragonflyPlateReverb":
            parameter_under_test = "decay_s"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "vst_DragonflyPlateReverb": {
                    "vst_filepath": "plugins/valid/DragonflyPlateReverb.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "buffer_size_frames": 0.015625,
                        "sample_rate_frames": 0.11484374850988388,
                        "dry_level": 0.800000011920929,
                        "wet_level": 0.20000000298023224,
                        "algorithm": 0.5,
                        "width": 0.5,
                        "predelay_ms": 0.0,
                        "low_cut_hz": 1.0,
                        "high_cut_hz": 1.0,
                        "dampen_hz": 0.800000011920929,
                    },
                }
            }
        elif plugin_name == "vst_3BandEQ":
            parameter_under_test = "high_db"
            min_param_value = 0
            max_param_value = 1
            plugins = {
                "vst_3BandEQ": {
                    "vst_filepath": "plugins/valid/3BandEQ.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "buffer_size_frames": 0.015625,
                        "sample_rate_frames": 0.11484374850988388,
                        "current_program": 0.0,
                        "low_db": 0.5,
                        "mid_db": 0.5,
                        "master_db": 0.5,
                        "low_mid_freq_hz": 0.2199999988079071,
                        "mid_high_freq_hz": 0.05263157933950424,
                    },
                }
            }
        elif plugin_name == "vst_MaGigaverb":
            parameter_under_test = "roomsize"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "vst_MaGigaverb": {
                    "vst_filepath": "plugins/valid/MaGigaverb.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "damping": 0.699999988079071,
                        "revtime": 0.030286191031336784,
                        # "roomsize": 0.24974992871284485,
                        "spread": 0.23000000417232513,
                        "bandwidth": 0.5,
                        "tail": 0.25,
                        "dry": 1.0,
                        "early": 0.25,
                    },
                }
            }
        elif plugin_name == "vst_MetalTone":
            parameter_under_test = "dist"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "vst_MetalTone": {
                    "vst_filepath": "plugins/valid/MetalTone.vst3",
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "buffer_size_frames": 0.015625,
                        "sample_rate_frames": 0.11484374850988388,
                        "high": 0.5,
                        "level": 0.30000001192092896,
                        "low": 0.5,
                        "midfreq": 0.5,
                        "middle": 0.5,
                        "bypass": 0.0,
                    },
                }
            }
        elif plugin_name == "vst_TAL-Chorus-LX":
            parameter_under_test = "dry_wet"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "vst_TAL-Chorus-LX": {
                    "vst_filepath": "plugins/valid/TAL-Chorus-LX.vst3",
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "volume": 0.5,
                        "stereo": 1.0,
                        "chorus_1": 1.0,
                        "chorus_2": 0.0,
                        "compatible_with_version_1_3_1": 0.0,
                        "bypass": 0.0,
                    },
                }
            }
        elif plugin_name == "pb_ParametricEQ":
            parameter_under_test = "low_shelf_gain_db"
            min_param_value = 0
            max_param_value = 1
            plugins = {
                "pb_ParametricEQ": {
                    "class_path": BasicParametricEQ,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "low_shelf_cutoff_freq": 120.0,
                        "low_shelf_q_factor": 0.707,
                        "band0_gain_db": 0.0,
                        "band0_cutoff_freq": 300.0,
                        "band0_q_factor": 0.707,
                        "band1_gain_db": 0.0,
                        "band1_cutoff_freq": 1000.0,
                        "band1_q_factor": 0.707,
                        "band2_gain_db": 0.0,
                        "band2_cutoff_freq": 3000.0,
                        "band2_q_factor": 0.707,
                        "band3_gain_db": 0.0,
                        "band3_cutoff_freq": 10000.0,
                        "band3_q_factor": 0.707,
                        "high_shelf_gain_db": 0.0,
                        "high_shelf_cutoff_freq": 10000.0,
                        "high_shelf_q_factor": 0.707,
                    },
                },
            }
        elif plugin_name == "pb_Chorus":
            parameter_under_test = "mix"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "pb_Chorus": {
                    "class_path": BasicChorus,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "rate_hz": 1.0,
                        "centre_delay_ms": 7.0,
                        "depth": 0.1,
                        "feedback": 0.5,
                        "our_bypass": 0.0,
                    },
                }
            }
        elif plugin_name == "pb_Compressor":
            min_param_value = 0.0
            max_param_value = 1.0
            parameter_under_test = "threshold_db"
            plugins = {
                "pb_Compressor": {
                    "class_path": BasicCompressor,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "ratio": 4.0,
                        "attack_ms": 1.0,
                        "release_ms": 100.0,
                    },
                },
            }
        elif plugin_name == "pb_Distortion":
            min_param_value = 0.5
            max_param_value = 1.0
            parameter_under_test = "drive_db"
            plugins = {
                "pb_Distortion": {
                    "class_path": BasicDistortion,
                    "num_params": None,
                    "num_channels": 1,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "output_gain_db": 0.0,
                    },
                },
            }
        elif plugin_name == "pb_Delay":
            parameter_under_test = "mix"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "pb_Delay": {
                    "class_path": BasicDelay,
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "delay_seconds": 0.1,
                        "feedback": 0.5,
                    },
                },
            }
        elif plugin_name == "pb_Reverb":
            parameter_under_test = "room_size"
            min_param_value = 0.0
            max_param_value = 1.0
            plugins = {
                "pb_Reverb": {
                    "class_path": BasicReverb,
                    "num_params": None,
                    "num_channels": 2,
                    "fixed_parameters": {
                        "our_bypass": 0.0,
                        "damping": 0.4,
                        "wet_dry": 0.4,
                        "width": 0.8,
                    },
                },
            }

        results = {}

        for mode in modes:

            results[mode] = {}

            for method_name, method in methods.items():
                results[mode][method_name] = {parameter_under_test: {}}

            for parameter_value in np.linspace(
                min_param_value, max_param_value, num_steps
            ):
                for method_name, method in methods.items():
                    results[mode][method_name][parameter_under_test][
                        parameter_value
                    ] = []

                for n in range(num_runs):
                    print(f"Run {n+1}/{num_runs}")

                    # load plugins (reload each time)
                    plugins, total_num_params, init_params = load_plugins(plugins)

                    # get the index of the parameter under test
                    for param_idx, (param_name) in enumerate(
                        plugins[plugin_name]["parameter_names"]
                    ):
                        if param_name == parameter_under_test:
                            parameter_under_test_index = param_idx
                            break

                    # set the parameter in init_params to the parameter value
                    test_init_params = init_params.copy()
                    test_init_params[parameter_under_test_index] = parameter_value

                    # process dummy audio to set parameter
                    _ = process_audio(
                        np.random.randn(2, 131072), test_init_params, 48000, plugins
                    )

                    # denormalize the target parameter value
                    if "pb" in plugin_name:
                        denorm_target_param_value = (
                            plugins[plugin_name]["instance"]
                            .parameters[parameter_under_test]
                            .get_value()
                        )
                    else:
                        denorm_target_param_value = (
                            plugins[plugin_name]["instance"]
                            .parameters[parameter_under_test]
                            .raw_value
                        )

                    print(f"Norm Target Param Value: {parameter_value}")
                    print(f"Denorm Target Param Value: {denorm_target_param_value}")

                    # get a random audio subset
                    audio_subset_filepaths = np.random.choice(audio_subsets)

                    # get input audio
                    valid_audio = False
                    while valid_audio is False:
                        input_filepath = np.random.choice(audio_subset_filepaths)
                        input_audio, input_sr = torchaudio.load(
                            input_filepath, backend="soundfile"
                        )
                        if input_audio.shape[-1] >= 524288 + 48000:
                            valid_audio = True

                    if mode == "different":  # load new audio
                        # get target audio
                        valid_audio = False
                        while valid_audio is False:
                            target_filepath = np.random.choice(audio_subset_filepaths)
                            target_audio, target_sr = torchaudio.load(
                                target_filepath, backend="soundfile"
                            )
                            if target_audio.shape[-1] >= 524288 + 48000:
                                valid_audio = True
                    else:  # use the same audio
                        target_audio = input_audio.clone()
                        target_sr = input_sr

                    # resample audio to 48kHz
                    sr = 48000
                    if input_sr != sr:
                        input_audio = torchaudio.functional.resample(
                            input_audio, input_sr, sr
                        )
                    if target_sr != sr:
                        target_audio = torchaudio.functional.resample(
                            target_audio, target_sr, sr
                        )

                    # randomly crop the audio
                    input_len = np.random.randint(262144, 524288)
                    target_len = np.random.randint(262144, 524288)

                    start_idx = np.random.randint(
                        48000, input_audio.shape[1] - input_len - 48000
                    )
                    input_audio = input_audio[:, start_idx : start_idx + input_len]
                    input_audio /= input_audio.abs().max()

                    start_idx = np.random.randint(
                        48000, target_audio.shape[1] - target_len - 48000
                    )
                    target_audio = target_audio[:, start_idx : start_idx + target_len]
                    target_audio /= target_audio.abs().max()

                    # peak noramlize

                    print(input_audio.shape, target_audio.shape)

                    # ensure stereo
                    if input_audio.shape[0] == 1:
                        input_audio = input_audio.repeat(2, 1)

                    if target_audio.shape[0] == 1:
                        target_audio = target_audio.repeat(2, 1)

                    # apply the effect to the reference audio
                    audio_output = process_audio(
                        target_audio.numpy(), test_init_params, sr, plugins
                    )
                    audio_output = torch.tensor(audio_output)

                    if save_audio:
                        torchaudio.save(
                            f"output/new_case_study/{plugin_name}/audio/{parameter_under_test}_{parameter_value:0.2f}_{n}.wav",
                            audio_output,
                            sr,
                        )

                    # run the ES style transfer process for each method configuration
                    for method_name, method in methods.items():
                        print(f"Method: {method_name}")

                        model = method["model"]
                        embed_func = method["embed_func"]

                        result = run_es(
                            input_audio.unsqueeze(0),
                            audio_output.unsqueeze(0),
                            sr,
                            plugins,
                            model,
                            embed_func,
                            max_iters=max_iters,
                            w0=None,
                            find_w0=False,
                            sigma0=0.33,
                            distance="cosine",
                            random_crop=True,
                            popsize=128,
                            parallel=False,
                            dropout=0.0,
                        )

                        estimated_params = result["params"]
                        estimated_audio_output = result["output_audio"]
                        estimated_param = estimated_params[plugin_name][
                            parameter_under_test
                        ]
                        estimated_param = result["wopt"][parameter_under_test_index]

                        fopt = result["fopt"]

                        results[mode][method_name][parameter_under_test][
                            parameter_value
                        ].append((estimated_param, fopt))

                        if save_audio:
                            # save audio output
                            torchaudio.save(
                                f"output/new_case_study/{plugin_name}/audio/{parameter_under_test}_{parameter_value:0.2f}_{n}_{method_name}_{estimated_param:0.2f}.wav",
                                estimated_audio_output,
                                sr,
                            )

                    # save results as json
                    with open(
                        f"output/new_case_study/{plugin_name}/case_study_results.json",
                        "w",
                    ) as f:
                        json.dump(results, f, indent=4)
