import os
import glob
import yaml
import json
import torch
import argparse
import torchaudio
import laion_clap
import numpy as np

from tqdm import tqdm
from importlib import import_module
from typing import Dict, List, Tuple, Union

from st_ito.utils import (
    load_param_model,
    load_clap_model,
    load_deepafx_st_model,
    get_deepafx_st_embeds,
    load_fx_encoder_model,
    get_fx_encoder_embeds,
    get_param_embeds,
    get_clap_embeds,
    load_wav2vec2_model,
    get_wav2vec2_embeds,
    load_wav2clip_model,
    get_wav2clip_embeds,
    load_vggish_model,
    get_vggish_embeds,
    load_mfcc_feature_extractor,
    get_mfcc_feature_embeds,
    load_mir_feature_extractor,
    get_mir_feature_embeds,
    load_beats_model,
    get_beats_embeds,
)


def is_lowest(value, array):
    for element in array:
        if element < value:
            return False
    return True


def is_highest(value, array):
    for element in array:
        if element > value:
            return False
    return True


def load_audios(example_dir: str, min_length: int = 131072, random_crop: bool = False):
    # find all the audio examples
    audio_filepaths = glob.glob(os.path.join(example_dir, "*.flac"))
    audio_filepaths = sorted(audio_filepaths, reverse=True)
    # x, c, b, a

    audios = {}
    for idx, audio_filepath in enumerate(audio_filepaths):

        try:
            audio_segment, sr = torchaudio.load(audio_filepath, backend="soundfile")
        except:
            print(f"Error loading {audio_filepath}")
            continue

        # take a random crop of random length
        if idx == 3 or not random_crop:
            pass  # use the same length for X
        else:
            crop_length = np.random.randint(min_length, audio_segment.shape[1])
            crop_start = np.random.randint(0, audio_segment.shape[1] - crop_length)
            audio_segment = audio_segment[:, crop_start : crop_start + crop_length]

        # add 12 db of headroom
        audio_segment /= audio_segment.abs().max()
        # audio_segment *= 10 ** (-12 / 20)

        audio_filename = os.path.basename(audio_filepath)
        audio_id = audio_filename.split("-")[0]
        audios[audio_id] = audio_segment.unsqueeze(0)  # add batch dimension

    return audios, sr


def evaluate_model(
    model_details: dict,
    audios: dict,
    sample_rate: int,
    include_c: bool = True,
):
    embeddings = {}
    similarities = {}

    # always embed ref first
    audio_ref = audios["ref"]
    audio_ref_embed = model_details["embed_fn"](
        audio_ref, model_details["model"], sample_rate
    )
    embeddings["ref"] = audio_ref_embed  # store

    # get embeddings
    for audio_name, audio in audios.items():
        if audio_name == "ref":
            continue
        audio_embed = model_details["embed_fn"](
            audio, model_details["model"], sample_rate
        )
        embeddings[audio_name] = audio_embed  # store

        # measure distance from reference (x)
        cos_sims = []
        for embed_name, output_embed in audio_embed.items():
            target_embed = embeddings["ref"][embed_name]
            cos_sim = torch.nn.functional.cosine_similarity(output_embed, target_embed)
            cos_sims.append(cos_sim)
        cos_sim = torch.mean(torch.stack(cos_sims))
        similarities[audio_name] = cos_sim

    incorrect_similarities = similarities.copy()
    del incorrect_similarities["a"]

    results = {}

    # compute for a random subset of the embeddings
    for num_embeds in np.arange(1, len(similarities)):
        similarities_subset = {}

        # always include the correct similarity (a)
        similarities_subset["a"] = similarities["a"]

        # select the first (num_embeds) incorrect similarities
        incorrect_similarities_subset = dict(
            list(incorrect_similarities.items())[:num_embeds]
        )

        # add these to the subset
        similarities_subset.update(incorrect_similarities_subset)

        print(similarities_subset)

        # find the most similar
        model_closer_cos = max(similarities_subset, key=similarities_subset.get)

        # now check if this agrees with human
        if model_closer_cos == "a":
            result = True
        else:
            result = False

        results[f"{num_embeds}"] = result

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dirs",
        nargs="+",
        help="List of paths to root directory of audio files.",
        default=[
            "/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/idmt-smt-guitar-acoustic-pedalboard",
            "/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/idmt-smt-guitar-electric-pedalboard",
            "/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/daps-pedalboard",
            "/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/enst-drums-pedalboard",
            "/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-dataset/vocalset-pedalboard",
        ],
    )
    parser.add_argument(
        "--human_dir",
        type=str,
        help="Path to root directory of human evaluations JSON files.",
        default="/import/c4dm-datasets-ext/lcap-datasets/human/submissions",
    )
    parser.add_argument(
        "--include_c",
        action="store_true",
    )
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    # first load human evaluations (find all JSON files)
    # human_eval_json_filepaths = glob.glob(os.path.join(args.human_dir, "*.json"))
    # print(f"Found {len(human_eval_json_filepaths)} human evaluation files.")

    # human_evals = []
    # print("Loading human evaluations...")
    # for human_eval_json_filepath in tqdm(human_eval_json_filepaths):
    #    with open(human_eval_json_filepath) as f:
    #        human_eval = json.load(f)
    #    human_evals.append(human_eval)

    # /import/c4dm-datasets-ext/lcap/lcap/qprhifks5/checkpoints/epoch=27-step=351652.ckpt

    # load models
    print("Loading models...")
    models = {
        # --------------- self-supervised parameter models --------------- #
        # "param-panns-blind": {
        #    "model": load_param_model(
        #        "/import/c4dm-datasets-ext/lcap/lcap/dsus0kmd/checkpoints/epoch=146-step=321489.ckpt",
        #        use_gpu=args.use_gpu,
        #    ),
        #    "embed_fn": get_param_embeds,
        # },
        # "param-panns-concat": {
        #    "model": load_param_model(
        #        "/import/c4dm-datasets-ext/lcap/lcap/yrq0f9t1/checkpoints/epoch=217-step=476766.ckpt",
        #        use_gpu=args.use_gpu,
        #    ),
        #    "embed_fn": get_param_embeds,
        # },
        # "param-panns-diff": {
        #    "model": load_param_model(
        #        "/import/c4dm-datasets-ext/lcap/lcap/5xtn9al0/checkpoints/epoch=15-step=34992.ckpt",
        #        use_gpu=args.use_gpu,
        #    ),
        #    "embed_fn": get_param_embeds,
        # },
        # "param-panns-l2-diff": {
        #    "model": load_param_model(
        #        "/import/c4dm-datasets-ext/lcap/lcap/plfl3q9x/checkpoints/epoch=173-step=380538.ckpt",
        #        use_gpu=args.use_gpu,
        #    ),
        #    "embed_fn": get_param_embeds,
        # },
        "param-panns-l2-concat": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        "param-panns-l2-concat-adv": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/bn2u0cil/checkpoints/last.ckpt",  # 1.0
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        "param-panns-l2-concat-adv-32": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/uobr3qvz/checkpoints/last.ckpt",  # 5.0
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        # "param-dstcn": {
        #    "model": load_param_model(
        #        "/import/c4dm-datasets-ext/lcap/lcap/9axg7erk/checkpoints/epoch=317-step=695466.ckpt",
        #        use_gpu=args.use_gpu,
        #    ),
        #    "embed_fn": get_param_embeds,
        # },
        "param-clap": {
            "model": load_param_model(
                "/import/c4dm-datasets-ext/lcap/param/lcap-param/cxdl1pga/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
            ),
            "embed_fn": get_param_embeds,
        },
        # ---------------- style transfer models ---------------- #
        "deepafx-st": {
            "model": load_deepafx_st_model(
                "/import/c4dm-datasets-ext/lcap/style/lcap-style/k7c29vsa/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
                encoder_only=True,
            ),
            "embed_fn": get_deepafx_st_embeds,
        },
        "deepafx-st+": {
            "model": load_deepafx_st_model(
                "/import/c4dm-datasets-ext/lcap/style/lcap-style/87h3sb6n/checkpoints/last.ckpt",
                use_gpu=args.use_gpu,
                encoder_only=True,
            ),
            "embed_fn": get_deepafx_st_embeds,
        },
        # --------------- audio features --------------- #
        "mfccs": {  # these are mfccs
            "model": load_mfcc_feature_extractor(use_gpu=args.use_gpu),
            "embed_fn": get_mfcc_feature_embeds,
        },
        "mir-features": {
            "model": load_mir_feature_extractor(use_gpu=args.use_gpu),
            "embed_fn": get_mir_feature_embeds,
        },
        # --------------- pretrained models --------------- #
        "fx-encoder": {
            "model": load_fx_encoder_model(use_gpu=args.use_gpu),
            "embed_fn": get_fx_encoder_embeds,
        },
        "clap": {
            "model": load_clap_model(use_gpu=args.use_gpu),
            "embed_fn": get_clap_embeds,
        },
        "wav2vec2": {
            "model": load_wav2vec2_model(use_gpu=args.use_gpu),
            "embed_fn": get_wav2vec2_embeds,
        },
        "wav2clip": {
            "model": load_wav2clip_model(use_gpu=args.use_gpu),
            "embed_fn": get_wav2clip_embeds,
        },
        "vggish": {
            "model": load_vggish_model(use_gpu=args.use_gpu),
            "embed_fn": get_vggish_embeds,
        },
        "beats": {
            "model": load_beats_model(use_gpu=args.use_gpu),
            "embed_fn": get_beats_embeds,
        },
    }

    results = {}

    for case_name in ["multi-effects"]:  # "inter-effect", "intra-effect-easy"
        for model_name, model_details in models.items():
            if model_name not in results:
                results[model_name] = {}
            if case_name not in results[model_name]:
                results[model_name][case_name] = {}

        # do this for each dataset
        for audio_dir in args.audio_dirs:
            print(f"Processing {audio_dir}...")
            # find all plugin directories
            plugin_dirs = glob.glob(
                os.path.join(audio_dir, case_name, "*"), recursive=True
            )
            plugin_dirs = [d for d in plugin_dirs if os.path.isdir(d)]
            plugin_dirs = sorted(plugin_dirs)
            print(f"Found {len(plugin_dirs)} plugin directories in {case_name}.")

            # iterate over human evaluations and score each model
            for plugin_dir in tqdm(plugin_dirs):
                plugin_name = os.path.basename(plugin_dir)
                # find all examples within this plugin directory
                example_dirs = glob.glob(os.path.join(plugin_dir, "*"), recursive=True)
                example_dirs = [d for d in example_dirs if os.path.isdir(d)]
                print(plugin_dir)

                for example_dir in example_dirs:
                    audios, sample_rate = load_audios(example_dir)

                    if "ref" not in audios:
                        print(f"Skipping {example_dir} as it has no reference audio.")
                        continue

                    if "a" not in audios:
                        print(f"Skipping {example_dir} as it has no audio a.")
                        continue

                    if args.use_gpu:
                        for key, audio in audios.items():
                            audios[key] = audio.cuda()

                    for model_name, model_details in models.items():
                        model_results = evaluate_model(
                            model_details, audios, sample_rate, include_c=args.include_c
                        )
                        # compute a running mean
                        if plugin_name not in results[model_name][case_name]:
                            results[model_name][case_name][plugin_name] = {}

                        for num_embeds, result in model_results.items():
                            if (
                                num_embeds
                                not in results[model_name][case_name][plugin_name]
                            ):
                                results[model_name][case_name][plugin_name][
                                    num_embeds
                                ] = []

                            results[model_name][case_name][plugin_name][
                                num_embeds
                            ].append(result)

                            mean_result = np.mean(
                                results[model_name][case_name][plugin_name][num_embeds]
                            )
                            num_results = len(
                                results[model_name][case_name][plugin_name][num_embeds]
                            )
                            print(
                                f"{model_name} {case_name} {plugin_name} {num_embeds} {mean_result} {num_results}"
                            )
                    print()

            # for model_name in models.keys():
            # model_scores = []
            # for plugin_name in results[model_name][case_name].keys():
            #    score = np.mean(results[model_name][case_name][plugin_name])
            #    # replace the list with the mean
            #    results[model_name][case_name][plugin_name] = score
            #    model_scores.append(score)
            # results[model_name][case_name]["overall"] = np.mean(model_scores)

            print()
            output_dir = "output/psm"

            if args.include_c:
                output_dir += "+c"

            os.makedirs(output_dir, exist_ok=True)
            dataset_names = "+".join(
                [os.path.basename(audio_dir) for audio_dir in args.audio_dirs]
            )
            filename = f"{output_dir}/results-{dataset_names}.json"
            with open(f"{output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)
