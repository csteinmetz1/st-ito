import os
import glob
import json
import time
import torch
import auraloss
import argparse
import pedalboard
import torchaudio
import numpy as np
import pyloudnorm as pyln

from st_ito.utils import (
    load_param_model,
    load_deepafx_st_model,
    get_deepafx_st_embeds,
    get_param_embeds,
    load_mfcc_feature_extractor,
    get_mfcc_feature_embeds,
    load_fx_encoder_model,
    get_fx_encoder_embeds,
    load_clap_model,
    get_clap_embeds,
    apply_fade_in,
)

from st_ito.style_transfer import (
    run_deepafx_st,
    run_es,
    run_input,
    run_random,
    run_rule_based,
    load_plugins,
)

from st_ito.effects import (
    BasicParametricEQ,
    BasicCompressor,
    BasicDistortion,
    BasicDelay,
    BasicReverb,
)


def get_source_type(filename: str):
    if "music" in filename:
        source_type = "music"
    elif "vocals" in filename or "straight" in filename:
        source_type = "vocals"
    elif "speech" in filename:
        source_type = "speech"
    else:
        raise ValueError(f"Unknown source type for {filename}")
    return source_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/import/c4dm-datasets-ext/lcap-datasets/output/synthetic-06042024",
    )
    parser.add_argument("--fade_samples", type=int, default=32768)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    meter = pyln.Meter(48000)

    # loss function
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss()

    test_cases = [
        "easy-1",
        "easy-2",
        "medium-1",
        "medium-2",
        "hard-1",
        "hard-2",
    ]

    # setup the plugins
    buffer_size_frames = 0.03125
    sample_rate_frames = 0.11484374850988388
    current_program = 0.0

    vst_plugins = {
        "Reverb": {
            "num_channels": 2,
            "fixed_parameters": {
                "ducking": 0.0,
                "side_chain": 0.0,
                "bits": 1.0,
                "sample_rate_divider": 1.0,
                "bypass": 0.0,
                "harmonic_tune": 0.0,
            },
        },
    }

    pb_plugins = {
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

    # setup metric
    # ------------------ Set up metrics ------------------
    metric = {
        "model": load_param_model(
            "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
        ),
        "embed_func": get_param_embeds,
    }

    def style_loss_fn(output_audio: torch.Tensor, target_audio: torch.Tensor):
        output_embed = metric["embed_func"](output_audio, metric["model"], 48000)
        target_embed = metric["embed_func"](target_audio, metric["model"], 48000)

        dists = []
        for output_embed_i, target_embed_i in zip(
            output_embed.values(), target_embed.values()
        ):
            dists.append(
                torch.nn.functional.cosine_similarity(output_embed_i, target_embed_i)
            )
        dists = torch.stack(dists)
        return dists.mean()

    # setup methods under test
    # ------------------ Set up methods ------------------
    methods = {
        "input": {
            "func": run_input,
            "params": {"model": None},
        },
        "random": {
            "func": run_random,
            "params": {"model": None},
        },
        "rule-based": {
            "func": run_rule_based,
            "params": {"model": None},
        },
        "deepafx-st": {
            "func": run_deepafx_st,
            "params": {
                "model": load_deepafx_st_model(
                    "/import/c4dm-datasets-ext/lcap/style/lcap-style/p8qjwc4g/checkpoints/last.ckpt"
                ),
            },
        },
        "deepafx-st+": {
            "func": run_deepafx_st,
            "params": {
                "model": load_deepafx_st_model(
                    "/import/c4dm-datasets-ext/lcap/style/lcap-style/n9y2bgn2/checkpoints/last.ckpt"
                ),
            },
        },
        "style-es (param-panns)": {
            "func": run_es,
            "params": {
                "model": load_param_model(
                    "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
                ),
                "embed_func": get_param_embeds,
                "normalization": "peak",
                "max_iters": 32,
                "sigma0": 0.33,
                "distance": "cosine",
                "popsize": 128,
                "dropout": 0.0,
                "save_pop": False,
                "find_w0": False,
                "random_crop": False,
            },
        },
        # "style-es (param-panns+clap)": {
        #    "func": run_es,
        #    "params": {
        #        "model": load_param_model(
        #            "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
        #        ),
        #        "embed_func": get_param_embeds,
        #        "content_model": load_clap_model(),
        #        "content_embed_func": get_clap_embeds,
        #        "normalization": "peak",
        #        "max_iters": 25,
        #        "sigma0": 0.33,
        #        "distance": "cosine",
        #        "popsize": 64,
        #        "dropout": 0.0,
        #        "save_pop": False,
        #        "find_w0": False,
        #        "random_crop": True,
        #    },
        # },
        # "style-es (param-panns-adv-32)": {
        #    "func": run_es,
        #    "params": {
        #        "model": load_param_model(
        #            "/import/c4dm-datasets-ext/lcap/param/lcap-param/uobr3qvz/checkpoints/last.ckpt"  # param-panns-adv
        #        ),
        #        "embed_func": get_param_embeds,
        #        "normalization": "peak",
        #        "max_iters": 25,
        #        "sigma0": 0.33,
        #        "distance": "cosine",
        #        "popsize": 64,
        #        "dropout": 0.0,
        #        "save_pop": False,
        #        "find_w0": False,
        #        "random_crop": True,
        #    },
        # },
        # "style-es (clap)": {
        #    "func": run_es,
        #    "params": {
        #        "model": load_clap_model(),
        #        "embed_func": get_clap_embeds,
        #        "normalization": "peak",
        #       "max_iters": 25,
        #       "sigma0": 0.33,
        #       "distance": "cosine",
        #        "popsize": 64,
        #        "dropout": 0.0,
        #        "save_pop": False,
        #        "find_w0": False,
        #        "random_crop": True,
        #    },
        # },
    }

    # load all dry audio first
    dry_examples = {"music": {}, "speech": {}, "vocals": {}}
    dry_filepaths = glob.glob(os.path.join(args.input_dir, "dry", "*.wav"))
    for dry_filepath in dry_filepaths:
        filepath = os.path.basename(dry_filepath)
        filename = filepath.replace(".wav", "")
        source_type = get_source_type(filename)

        dry_audio, dry_sr = torchaudio.load(dry_filepath, backend="soundfile")

        if dry_sr != 48000:
            dry_audio = torchaudio.functional.resample(dry_audio, dry_sr, 48000)

        # fade the dry audio
        dry_audio = apply_fade_in(dry_audio, args.fade_samples)
        dry_examples[source_type][filename] = dry_audio.unsqueeze(0)

    test_examples = {}
    for test_case in test_cases:
        if test_case not in test_examples:
            test_examples[test_case] = {"music": {}, "speech": {}, "vocals": {}}
        # load all audio for this test case
        test_filepaths = glob.glob(os.path.join(args.input_dir, test_case, "*.wav"))

        for test_filepath in test_filepaths:
            filepath = os.path.basename(test_filepath)
            filename = filepath.replace(".wav", "")
            source_type = get_source_type(filename)
            test_audio, test_sr = torchaudio.load(test_filepath, backend="soundfile")

            if test_sr != 48000:
                test_audio = torchaudio.functional.resample(test_audio, test_sr, 48000)

            # fade the test audio
            test_audio_fade = apply_fade_in(test_audio, args.fade_samples)
            test_examples[test_case][source_type][filename] = test_audio.unsqueeze(0)

    print(test_examples)

    # now convert the style of each dry example to each test example
    results = {}
    for source_type, dry_examples in dry_examples.items():
        print("Processing", source_type)
        for dry_filename, dry_audio in dry_examples.items():
            for test_case, test_case_source_types in test_examples.items():
                # get test cases with this source type
                source_type_test_examples = test_case_source_types[source_type]
                test_filenames = list(source_type_test_examples.keys())

                test_filename = dry_filename
                # select one test example at random
                while dry_filename == test_filename:
                    # print(test_filenames)
                    test_filename = np.random.choice(test_filenames)
                    test_audio = source_type_test_examples[test_filename]

                print(f"Processing {dry_filename}->{test_case}-{test_filename}")

                if test_case not in results:
                    results[test_case] = {}

                example_id = f"{dry_filename}->{test_case}-{test_filename}"
                results[test_case][example_id] = {}

                example_dir = os.path.join(args.output_dir, example_id)
                os.makedirs(example_dir, exist_ok=True)

                # add w0 to methods
                for method_name, method in methods.items():
                    method["params"]["w0"] = None

                print(dry_audio.shape, test_audio.shape)

                # each method should take as input input and target and then return audio
                for method_name, method in methods.items():
                    # time the method

                    if "style-es" in method_name or "random" in method_name:
                        plugin_set = [vst_plugins, pb_plugins]
                        plugin_set_names = ["vst", "pb"]
                    else:
                        plugin_set = [vst_plugins]
                        plugin_set_names = [None]

                    for plugin_set_name, plugins in zip(plugin_set_names, plugin_set):
                        # reload plugins
                        if plugin_set_name is not None:
                            plugins, total_num_params, init_params = load_plugins(
                                plugins
                            )
                            plugin_set_str = f"_{plugin_set_name}"
                        else:
                            total_num_params = 0
                            init_params = []
                            plugin_set_str = ""

                        # run method
                        start_time = time.time()
                        result = method["func"](
                            dry_audio,
                            test_audio,
                            48000,
                            plugins,
                            **(method["params"]),
                        )
                        elapsed_time = time.time() - start_time
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")

                        # extract output audio
                        output_audio = result["output_audio"]

                        if output_audio.ndim == 2:
                            output_audio = output_audio.unsqueeze(0)

                        # measure against ground truth
                        gt_audio = test_examples[test_case][source_type][dry_filename]

                        # peak normnalize first
                        output_audio_norm = output_audio / output_audio.abs().max()
                        gt_audio_norm = gt_audio / gt_audio.abs().max()

                        error = loss_fn(output_audio, gt_audio)
                        error_norm = loss_fn(output_audio_norm, gt_audio_norm)
                        style_error_gt = style_loss_fn(output_audio, gt_audio)
                        style_error_target = style_loss_fn(output_audio, test_audio)
                        print(error, error_norm, style_error_gt, style_error_target)

                        results[test_case][example_id][method_name + plugin_set_str] = {
                            "elapsed_time": elapsed_time,
                            "mrstft_error": error.item(),
                            "mrstft_error_norm": error_norm.item(),
                            "style_error_gt": style_error_gt.item(),
                            "style_error_target": style_error_target.item(),
                        }

                        # remove batch dim and loudness normalize to -22
                        output_audio = output_audio.squeeze(0)
                        test_audio_out = test_audio.squeeze(0)
                        gt_audio_out = gt_audio.squeeze(0)

                        # crop all audios to the min length of the three
                        min_length = min(
                            output_audio.shape[-1],
                            test_audio_out.shape[-1],
                            gt_audio_out.shape[-1],
                        )
                        output_audio = output_audio[..., :min_length]
                        test_audio_out = test_audio_out[..., :min_length]
                        gt_audio_out = gt_audio_out[..., :min_length]

                        # now loudness normalize
                        output_audio_lufs = meter.integrated_loudness(
                            output_audio.permute(1, 0).numpy()
                        )
                        delta_lufs = -22 - output_audio_lufs
                        output_audio = output_audio * 10 ** (delta_lufs / 20)

                        test_audio_lufs = meter.integrated_loudness(
                            test_audio_out.permute(1, 0).numpy()
                        )
                        delta_lufs = -22 - test_audio_lufs
                        test_audio_out = test_audio_out * 10 ** (delta_lufs / 20)

                        gt_audio_lufs = meter.integrated_loudness(
                            gt_audio_out.permute(1, 0).numpy()
                        )
                        delta_lufs = -22 - gt_audio_lufs
                        gt_audio_out = gt_audio_out * 10 ** (delta_lufs / 20)

                        # save audio to disk
                        filepath = os.path.join(
                            example_dir, f"{example_id}_{method_name}"
                        )
                        if plugin_set_name == "vst":
                            filepath += "_vst.wav"
                        elif plugin_set_name == "pb":
                            filepath += "_pb.wav"
                        else:
                            filepath += ".wav"

                        torchaudio.save(filepath, output_audio, 48000)

                        # save the test case
                        filepath = os.path.join(example_dir, f"{example_id}_target.wav")
                        torchaudio.save(filepath, test_audio_out, 48000)

                        # save the reference
                        filepath = os.path.join(example_dir, f"{example_id}_gt.wav")
                        torchaudio.save(filepath, gt_audio_out, 48000)

                        with open(
                            os.path.join(args.output_dir, "results.json"), "w"
                        ) as fp:
                            json.dump(results, fp, indent=2)
