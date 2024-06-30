import os
import cma
import json
import time
import yaml
import torch
import torchaudio
import pedalboard
import numpy as np
import pyloudnorm as pyln
import multiprocessing as mp

from typing import List
from datetime import datetime
from importlib import import_module

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
from st_ito.methods.style import StyleTransferSystem

from st_ito.features import (
    compute_barkspectrum,
    compute_crest_factor,
    compute_rms_energy,
    compute_lufs,
    compute_spectral_centroid,
)
from st_ito.effects import (
    BasicParametricEQ,
    BasicCompressor,
    BasicDistortion,
    BasicDelay,
    BasicReverb,
)

# ------------------ Method functions ------------------

from st_ito.style_transfer import (
    run_input,
    run_random,
    run_rule_based,
    run_es,
    run_deepafx_st,
    process_audio,
    load_plugins,
)


# ------------------- audio feature wrappers --------------------
def compute_barkspectrum_wrapper(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    *args,
    **kwargs,
):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    return compute_barkspectrum(x, sample_rate=sample_rate, mode="mono")


def compute_rms_energy_wrapper(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    *args,
    **kwargs,
):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    return compute_rms_energy(x)


def compute_lufs_wrapper(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    *args,
    **kwargs,
):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    return compute_lufs(x, sample_rate=sample_rate)


def compute_spectral_centroid_wrapper(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    *args,
    **kwargs,
):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    return compute_spectral_centroid(x, sample_rate=sample_rate)


def get_param_embeds_wrapper(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    *args,
    **kwargs,
):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    return get_param_embeds(x, model, sample_rate)


def get_contrived_examples(mode: str):

    if mode == "music":
        dataset_name = "musdb18_44100"
        root_dir = "musdb18_44100_styles_100/test"
    elif mode == "speech":
        dataset_name = "cleanraw"
        root_dir = f"daps_24000_styles_100/test"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    examples = []

    for style_name in ["bright", "broadcast", "telephone", "warm"]:
        for n in range(80, 100):
            basename = f"{dataset_name}_test.wav"
            examples.append(
                (
                    f"{root_dir}/neutral/{n:03d}_neutral_{basename}",
                    f"{root_dir}/{style_name}/{n:03d}_{style_name}_{basename}",
                )
            )

    return examples


def get_real_examples(mode: str):
    # input / target
    if mode == "speech":
        examples = [
            ["speech/0YY7K7Xa5rE.wav", "speech/ASU_wpcB-1I.wav"],  # 0
            ["speech/GqPGXG5TlZw.wav", "speech/bPYtvBDMoT8.wav"],  # 1
            ["speech/Hd9pHZC7Sak.wav", "speech/sNDyQ5wdV7Y.wav"],  # 2
            ["speech/kCc8FmEb1nY.wav", "speech/-QqTwJzi7Wo.wav"],  # 3
            ["speech/njqx9QHqQnI.wav", "speech/rp18gXVZiws.wav"],  # 4
            ["speech/VkBEQDYCpeo.wav", "speech/KmHe_QUCATQ.wav"],  # 5
            ["speech/dtp6b76pMak.wav", "speech/505UazMNgLg.wav"],  # 6
            ["speech/tI0uvIgh3e8.wav", "speech/DxOIZ4sOQAw.wav"],  # 7
            ["speech/YxLm0jmazq8.wav", "speech/nyxcO2vdcCg.wav"],  # 8
            ["speech/6PZGOzYUMh4.wav", "speech/4aSHovdnCyY.wav"],  # 9
        ]
        indices = [0, 1, 4, 5]
    elif mode == "guitar":
        examples = [
            ["guitar/q7dd3PAUpqE.wav", "guitar/1MxfbKkX7Zg.wav"],  # 0
            ["guitar/q7dd3PAUpqE.wav", "guitar/5Az0vI2kU8o.wav"],  # 2
            ["guitar/9uH5GvurJYc.wav", "guitar/8-lQhm67ZxE.wav"],  # 3
            ["guitar/DPGanZQH6L4.wav", "guitar/8_tM8HPkR5w.wav"],  # 4
            ["guitar/YDiUYW8gPbE.wav", "guitar/KqNrQw_Ne8w.wav"],  # 5
            ["guitar/4cH_Q-uqJhU.wav", "guitar/7Mv-Et66FS4.wav"],  # 6
            ["guitar/_xybjiuD9K0.wav", "guitar/DPGanZQH6L4.wav"],  # 7
            ["guitar/MmUX2ZKhn_Q.wav", "guitar/KqNrQw_Ne8w.wav"],  # 8
            ["guitar/BLrJSfrgYGI.wav", "guitar/ko8G5hkGqvc.wav"],  # 9
            ["guitar/Fwnj5n1SdxY.wav", "guitar/wglmFyQPL4o.wav"],  # 10
        ]
        indices = [0, 2, 3, 4]
    elif mode == "vocals":
        examples = [
            ["vocals/I_QWegHp-r0.wav", "vocals/-o_MW5vifL8.wav"],  # 0
            ["vocals/n8cRTh4GEYg.wav", "vocals/CI2a5BxEIV0.wav"],  # 1
            ["vocals/IyJ34F3tjG0.wav", "vocals/UGiEw22GI-4.wav"],  # 2
            ["vocals/PGS0UvbCwGk.wav", "vocals/U1kifTk5xsU.wav"],  # 3
            ["vocals/QP37fZmj-XY.wav", "vocals/CI2a5BxEIV0.wav"],  # 4
            ["vocals/ScQISlpnjoQ.wav", "vocals/-o_MW5vifL8.wav"],  # 5
            ["vocals/Slhrbuil8Yo.wav", "vocals/w1vxWWD1j50.wav"],  # 6
            ["vocals/U1kifTk5xsU.wav", "vocals/w1vxWWD1j50.wav"],  # 7
            ["vocals/UKyuxmgir2w.wav", "vocals/uOWK-ArhziU.wav"],  # 8
            ["vocals/uOWK-ArhziU.wav", "vocals/Wbuj60Ew2p4.wav"],  # 9
        ]
        indices = [0, 2, 3, 9]
    elif mode == "music":
        examples = [
            ["music/wXhTHyIgQ_U.wav", "music/PAa2KuxXSYw.wav"],  # 0
            ["music/TUVcZfQe-Kw.wav", "music/qku2WZ7aRYw.wav"],  # 1
            ["music/1JNmz17gnMw.wav", "music/R-MSfd2S7lo.wav"],  # 2
            ["music/UqyT8IEBkvY.wav", "music/TUVcZfQe-Kw.wav"],  # 3
            ["music/wXhTHyIgQ_U.wav", "music/UqyT8IEBkvY.wav"],  # 4
            ["music/ylXk1LBvIqU.wav", "music/ORxKWb8kKz8.wav"],  # 5
            ["music/7nJRGARveVc.wav", "music/1JNmz17gnMw.wav"],  # 6
            ["music/HAIDqt2aUek.wav", "music/dhNfddJRulQ.wav"],  # 7
            ["music/HMO-gn2qrnc.wav", "music/HAIDqt2aUek.wav"],  # 8
            ["music/IL-6hwW4ViA.wav", "music/LwHWGnhg3o4.wav"],  # 9
        ]
        indices = [5, 6, 7, 8]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return examples, indices


def get_plugins(chain_type: str):
    # ------------------ Load plugins ------------------
    buffer_size_frames = 0.03125
    sample_rate_frames = 0.11484374850988388
    current_program = 0.0

    if chain_type == "general-vst":
        plugins = {
            "TubeScreamer": {
                "vst_filepath": "plugins/valid/TubeScreamer.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZamCompX2": {
                "vst_filepath": "plugins/valid/ZamCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "ZamDelay": {
                "vst_filepath": "plugins/valid/ZamDelay.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "TAL-Reverb-4": {
                "vst_filepath": "plugins/valid/TAL-Reverb-4.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "ducking": 0.0,
                    "side_chain": 0.0,
                    "bits": 1.0,
                    "sample_rate_divider": 1.0,
                    "bypass": 0.0,
                    "modulation_depth": 0.0,
                },
            },
        }
    elif chain_type == "general-pb":
        plugins = {
            "Distortion": {
                "class_path": BasicDistortion,
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {},
            },
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
    elif chain_type == "simple":
        plugins = {
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZaMultiCompX2": {
                "vst_filepath": "plugins/valid/ZaMultiCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "zamcomp_1_on": 1.0,
                    "zamcomp_2_on": 1.0,
                    "zamcomp_3_on": 1.0,
                    "listen_1": 0.0,
                    "listen_2": 0.0,
                    "listen_3": 0.0,
                    "detection_max_avg": 0.0,
                },
            },
            "ZaMaximX2": {
                "vst_filepath": "plugins/valid/ZaMaximX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "latency_frames": 0.012500000186264515,
                    "current_program": current_program,
                },
            },
        }
    elif chain_type == "speech-vst":
        plugins = {
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZaMultiCompX2": {
                "vst_filepath": "plugins/valid/ZaMultiCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "zamcomp_1_on": 1.0,
                    "zamcomp_2_on": 1.0,
                    "zamcomp_3_on": 1.0,
                    "listen_1": 0.0,
                    "listen_2": 0.0,
                    "listen_3": 0.0,
                    "detection_max_avg": 0.0,
                },
            },
            "TAL-Reverb-4": {
                "vst_filepath": "plugins/valid/TAL-Reverb-4.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {"bypass": 0.0, "on_off": 1.0},
            },
        }
    elif chain_type == "mastering-vst":
        plugins = {
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZaMultiCompX2": {
                "vst_filepath": "plugins/valid/ZaMultiCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "zamcomp_1_on": 1.0,
                    "zamcomp_2_on": 1.0,
                    "zamcomp_3_on": 1.0,
                    "listen_1": 0.0,
                    "listen_2": 0.0,
                    "listen_3": 0.0,
                    "detection_max_avg": 0.0,
                },
            },
            "TubeScreamer": {
                "vst_filepath": "plugins/valid/TubeScreamer.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "ZaMaximX2": {
                "vst_filepath": "plugins/valid/ZaMaximX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "latency_frames": 0.012500000186264515,
                    "current_program": current_program,
                },
            },
        }
    elif chain_type == "vocals-vst":
        plugins = {
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZamComp": {
                "vst_filepath": "plugins/valid/ZaMultiCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "TubeScreamer": {
                "vst_filepath": "plugins/valid/TubeScreamer.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "ZamDelay": {
                "vst_filepath": "plugins/valid/ZamDelay.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "DragonflyPlateReverb": {
                "vst_filepath": "plugins/valid/DragonflyPlateReverb.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": 0.015625,
                    "sample_rate_frames": 0.11484374850988388,
                    "bypass": 0.0,
                },
            },
            # "TAL-Reverb-4": {
            #    "vst_filepath": "plugins/valid/TAL-Reverb-4.vst3",
            #    "num_params": None,
            #    "num_channels": 2,
            #    "fixed_parameters": {"bypass": 0.0, "on_off": 1.0},
            # },
        }
    elif chain_type == "guitar-vst":
        plugins = {
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "ZaMultiCompX2": {
                "vst_filepath": "plugins/valid/ZaMultiCompX2.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "zamcomp_1_on": 1.0,
                    "zamcomp_2_on": 1.0,
                    "zamcomp_3_on": 1.0,
                    "listen_1": 0.0,
                    "listen_2": 0.0,
                    "listen_3": 0.0,
                    "detection_max_avg": 0.0,
                },
            },
            "STR-X": {
                "vst_filepath": "plugins/valid/STR-X.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {"bypass": 0.0},
            },
            "TubeScreamer": {
                "vst_filepath": "plugins/valid/TubeScreamer.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                },
            },
            "ZamEQ2": {
                "vst_filepath": "plugins/valid/ZamEQ2.vst3",
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {
                    "buffer_size_frames": buffer_size_frames,
                    "sample_rate_frames": sample_rate_frames,
                    "current_program": current_program,
                    "peaks_on": 0,
                },
            },
            "TAL-Reverb-4": {
                "vst_filepath": "plugins/valid/TAL-Reverb-4.vst3",
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {"bypass": 0.0, "on_off": 1.0},
            },
        }
    elif chain_type == "mastering-pb":
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
            "Reverb": {
                "class_path": BasicReverb,
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {},
            },
        }
    elif chain_type == "vocals-pb":
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
    elif chain_type == "guitar-pb":
        plugins = {
            "Compressor": {
                "class_path": BasicCompressor,
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {},
            },
            "ParametricEQ": {
                "class_path": BasicParametricEQ,
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
            "ParametricEQ": {
                "class_path": BasicParametricEQ,
                "num_params": None,
                "num_channels": 1,
                "fixed_parameters": {},
            },
            "Reverb": {
                "class_path": BasicReverb,
                "num_params": None,
                "num_channels": 2,
                "fixed_parameters": {},
            },
        }
    else:
        raise ValueError(f"Unknown chain_type: {chain_type}")

    return plugins


def run_pst_benchmark(
    metrics: dict,
    methods: dict,
    mode: str,
    dataset_type: str,
    effect_type: str,
    plugins: dict,
    root_dir: str,
    w0: torch.Tensor = None,
    fade_samples: int = 32768,
):
    """Run the PST benchmark.

    Args:
        metrics (dict): Dictionary of metrics
        methods (dict): Dictionary of methods
        mode (str): Mode one of ["speech", "guitar", "vocals", "music"]
        dataset_type (str): Dataset type one of ["real", "contrived"]
        effect_type (str): Effect type one of ["vst", "pedalboard"]
        plugins (dict): Dictionary of plugins
        root_dir (str): Root directory of the PST dataset
        w0 (torch.Tensor): Initial parameters
        dropout (float): Dropout rate
    """
    # get inputs -> targets
    if dataset_type == "real":
        examples, indices = get_real_examples(mode)
    elif dataset_type == "contrived":
        examples = get_contrived_examples(mode)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # ------------------ Run evaluation ------------------
    results = {}  # list of dicts
    for method_name in methods.keys():
        results[method_name] = {"time_elapsed": []}
        for metric_name in metrics.keys():
            results[method_name][metric_name] = []

    for idx, example in enumerate(examples):

        if idx not in indices:  # skip examples not in indices
            continue

        input_filepath, target_filepath = example

        input_filepath = os.path.join(root_dir, input_filepath)
        target_filepath = os.path.join(root_dir, target_filepath)

        # load input and target
        input_audio, input_sr = torchaudio.load(input_filepath, backend="soundfile")
        target_audio, target_sr = torchaudio.load(target_filepath, backend="soundfile")

        # ensure mono when processing speech, guitar, and vocals
        # if mode in ["speech", "guitar", "vocals"]:
        #    input_audio = input_audio.mean(0, keepdim=True)
        #    target_audio = target_audio.mean(0, keepdim=True)

        # resample to 48000 kHz
        if input_sr != 48000:
            input_audio = torchaudio.functional.resample(input_audio, input_sr, 48000)

        if target_sr != 48000:
            target_audio = torchaudio.functional.resample(
                target_audio, target_sr, 48000
            )

        sr = 48000

        # crop to the same length
        min_len = min(input_audio.shape[1], target_audio.shape[1], 262144)
        # input_audio = input_audio[:, :min_len]
        # target_audio = target_audio[:, :min_len]

        # ensure stereo
        if input_audio.shape[0] == 1:
            input_audio = input_audio.repeat(2, 1)

        if target_audio.shape[0] == 1:
            target_audio = target_audio.repeat(2, 1)

        chs, seq_len = input_audio.shape

        # setup output dir
        os.makedirs("output", exist_ok=True)
        os.makedirs(os.path.join("output", "pst"), exist_ok=True)
        os.makedirs(os.path.join("output", "pst", f"{dataset_type}"), exist_ok=True)
        os.makedirs(
            os.path.join("output", "pst", f"{dataset_type}", f"{mode}"), exist_ok=True
        )

        # add batch dim to input and target
        input_audio = input_audio.unsqueeze(0)
        target_audio = target_audio.unsqueeze(0)

        # apply fade in to avoid transients
        input_audio = apply_fade_in(input_audio, fade_samples)
        target_audio = apply_fade_in(target_audio, fade_samples)

        # add w0 to methods
        for method_name, method in methods.items():
            method["params"]["w0"] = w0

        # each method should take as input input and target and then return audio
        for method_name, method in methods.items():
            # time the method
            start_time = time.time()
            result = method["func"](
                input_audio,
                target_audio,
                sr,
                plugins,
                **(method["params"]),
            )
            elapsed_time = time.time() - start_time

            results[method_name]["time_elapsed"].append(elapsed_time)

            # extract output audio
            output_audio = result["output_audio"]

            # get parameters
            if "params" in result:
                params = result["params"]
                param_filepath = os.path.join(
                    "output",
                    "pst",
                    f"{dataset_type}",
                    f"{mode}",
                    f"{idx:02d}_{method_name}_{mode}_{effect_type}.json",
                )
                with open(param_filepath, "w") as f:
                    json.dump(params, f, indent=2)

            if output_audio.ndim == 2:
                output_audio = output_audio.unsqueeze(0)

            print(
                method_name,
                input_audio.shape,
                target_audio.shape,
                output_audio.shape,
            )

            # evaluate with different metrics
            for metric_name, metric in metrics.items():
                # extract embeddings
                output_embeds = metric["func"](
                    output_audio,
                    metric["model"],
                    sr,
                )
                target_embeds = metric["func"](
                    target_audio,
                    metric["model"],
                    sr,
                )

                dists = []
                for embed_name, output_embed in output_embeds.items():
                    target_embed = target_embeds[embed_name]

                    # compute distance
                    with torch.no_grad():
                        if metric["distance"] == "cosine":
                            dist = torch.cosine_similarity(
                                output_embed, target_embed, dim=1
                            )
                        elif metric["distance"] == "l2":
                            dist = torch.nn.functional.mse_loss(
                                output_embed, target_embed
                            )

                        dists.append(dist)
                dists = torch.stack(dists)
                dist = dists.mean()

                # save results
                results[method_name][metric_name].append(dist.item())

            # crop the length to min_len
            output_audio = output_audio[:, :, :min_len]

            # loudness normalize output audio
            meter = pyln.Meter(sr)
            output_lufs = meter.integrated_loudness(
                output_audio.squeeze(0).permute(1, 0).cpu().numpy()
            )
            delta_lufs = -22.0 - output_lufs
            gain_lin = 10 ** (delta_lufs / 20)
            output_audio = output_audio * gain_lin

            # save output audio
            output_filepath = os.path.join(
                "output",
                "pst",
                f"{dataset_type}",
                f"{mode}",
                f"{idx:02d}_{method_name}_{mode}_{effect_type}.wav",
            )
            torchaudio.save(
                output_filepath,
                output_audio.squeeze(0),
                sample_rate=sr,
                backend="soundfile",
            )

        # crop the target to min length
        target_audio = target_audio[:, :, :min_len]

        # loudness normalize target
        target_audio = target_audio.squeeze(0)
        meter = pyln.Meter(sr)
        target_lufs = meter.integrated_loudness(
            target_audio.permute(1, 0).cpu().numpy()
        )
        delta_lufs = -22.0 - target_lufs
        gain_lin = 10 ** (delta_lufs / 20)
        target_audio = target_audio * gain_lin

        output_filepath = os.path.join(
            "output",
            "pst",
            f"{dataset_type}",
            f"{mode}",
            f"{idx:02d}_target_{mode}.wav",
        )
        torchaudio.save(
            output_filepath,
            target_audio,
            sample_rate=sr,
            backend="soundfile",
        )

    # ------------------ Print results ------------------
    print()
    for method_name, method in methods.items():
        print(f"{method_name}:")
        for metric_name, metric in metrics.items():
            print(f"\t{metric_name}: {np.mean(results[method_name][metric_name])}")

    # save results to json
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_filepath = os.path.join(
        "output",
        "pst",
        f"{dataset_type}",
        f"{mode}",
        f"results_{timestamp}.json",
    )
    with open(json_filepath, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":

    # ------------------ Set up metrics ------------------
    metrics = {
        # "barkspectrum": {
        #    "model": None,
        #    "func": compute_barkspectrum_wrapper,
        #    "distance": "l2",
        # },
        # "rms-energy": {
        #    "model": None,
        #    "func": compute_rms_energy_wrapper,
        #    "distance": "l2",
        # },
        # "lufs": {
        #    "model": None,
        #    "func": compute_lufs_wrapper,
        #    "distance": "l2",
        # },
        # "spectral-centroid": {
        #    "model": None,
        #    "func": compute_spectral_centroid_wrapper,
        #    "distance": "l2",
        # },
        # "mfccs": {
        #    "model": load_mfcc_feature_extractor(),
        #    "func": get_mfcc_feature_embeds,
        #    "distance": "cosine",
        # },
        "style_features": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"
            ),
            "func": get_param_embeds_wrapper,
            "distance": "cosine",
        },
    }

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
                "random_crop": True,
            },
        },
        # "style-es (param-panns+clap)": {
        #    "func": run_es,
        #    "params": {
        #        "model": load_param_model(
        #            "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
        #        ),
        ##        "embed_func": get_param_embeds,
        #       "content_model": load_clap_model(),
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
    }

    # ------------------ Run PST benchmark ------------------
    for dataset_type in ["real"]:  # , "contrived"

        if dataset_type == "real":
            root_dir = "/import/c4dm-datasets-ext/lcap-datasets/pst-benchmark-dataset"
            modes = ["vocals", "speech", "music"]  # "vocals",
        elif dataset_type == "contrived":
            root_dir = "/import/c4dm-datasets/deepafx2/"
            modes = ["speech", "music"]

        for mode in modes:
            for effect_type in ["vst", "pedalboard"]:  # , ,

                if effect_type == "pedalboard":
                    methods_subset = {
                        "style-es (param-panns)": methods["style-es (param-panns)"],
                    }
                else:
                    methods_subset = methods

                if effect_type == "vst":
                    chain_type = "general-vst"
                elif effect_type == "pedalboard":
                    chain_type = "general-pb"

                print(f"Running PST benchmark on {mode} with {chain_type}")

                # ------------------ Load plugins ------------------
                plugins = get_plugins(chain_type)
                plugins, total_num_params, init_params = load_plugins(plugins)
                print(chain_type, total_num_params)
                continue

                # create vector of initial parameter values
                # w0 = torch.zeros(total_num_params)
                # for idx, param in enumerate(init_params):
                #    w0[idx] = param
                w0 = None

                run_pst_benchmark(
                    metrics=metrics,
                    methods=methods_subset,
                    mode=mode,
                    dataset_type=dataset_type,
                    effect_type=effect_type,
                    plugins=plugins,
                    root_dir=root_dir,
                    w0=w0,
                )
