import torch
import argparse
import torchaudio
import pedalboard

from tqdm import tqdm
from st_ito.utils import (
    get_clap_embeds,
    load_clap_model,
    get_param_embeds,
    load_param_model,
)

SOURCES = {
    "di-guitar": {
        "input_filepaths": [
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0075/Gtr_Lead.wav",  # 1
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0019/cleanGTR_L.wav",  # 2
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0038/Cln_Gtr_06.wav",  # 3
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0021/GTR_-_0100_-_Audio_-_GTR_L2.wav",  # 4
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0084/Main_R.wav",  # 5
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0041/Ld_Gtr_25.wav",  # 6
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0024/Clean_Lead_DI.wav",  # 7
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0042/LEAD.wav",  # 8
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0094/Lead_Guitar.wav",  # 9
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0004/Gtr_Clean_1.wav",  # 10
        ],
        "target_filepaths": [
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0019/cleanGTR_L.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0059/Gtr_Lead_Left.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0056/Rhythm_R.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0104/Rhythm_R.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0061/Gtr_DI_CleanL.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0038/Ld_Gtr_01.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0048/Lead_Guitar_2_L.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0074/Lead_Reverb_R.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0109/Guitar_Fast_Lead_R_DI.wav",
            "/import/c4dm-datasets-ext/private-guitar-processed-48khz/0061/Gtr_DI_CleanL.wav",
        ],
    },
    "speech": {
        "input_filepaths": [
            "/import/c4dm-datasets/daps_dataset/cleanraw/f10_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m1_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f3_script2_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m5_script4_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f7_script4_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m10_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f8_script4_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m2_script2_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f4_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m6_script2_cleanraw.wav",
        ],
        "target_filepaths": [
            "/import/c4dm-datasets/daps_dataset/cleanraw/f1_script5_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m4_script4_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f4_script3_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m7_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f9_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m1_script2_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f7_script3_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m3_script1_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/f5_script4_cleanraw.wav",
            "/import/c4dm-datasets/daps_dataset/cleanraw/m7_script5_cleanraw.wav",
        ],
    },
}

EFFECTS = {
    "distortion": {
        "plugin": pedalboard.Distortion(),
        "param-name": "drive_db",
        "param-range": [0, 40],
    },
    "compressor": {
        "plugin": pedalboard.Compressor(),
        "param-name": "threshold_db",
        "param-range": [-60, 0],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="speech")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    # get sources to evaluate
    if args.source not in SOURCES.keys():
        raise ValueError(
            f"Source {args.source} not found. Please choose from {SOURCES.keys()}"
        )
    source = SOURCES[args.source]
    input_filepaths = source["input_filepaths"]
    target_filepaths = source["target_filepaths"]

    # load models
    print("Loading models...")
    models = {
        "param-panns": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/lcap/txraplm1/checkpoints/epoch=94-step=207765.ckpt",
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        "clap": {
            "model": load_clap_model(use_gpu=args.use_gpu),
            "embed_fn": get_clap_embeds,
        },
    }

    # iterate over effects
    for effect_name, effect in EFFECTS.items():
        # iterate over sources
        for input_filepath, target_filepath in zip(input_filepaths, target_filepaths):
            input_audio, _ = torchaudio.load(input_filepath, backend="soundfile")
            target_audio, _ = torchaudio.load(target_filepath, backend="soundfile")
