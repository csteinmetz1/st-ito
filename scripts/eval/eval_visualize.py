import os
import glob
import yaml
import json
import umap
import torch
import argparse
import torchaudio
import laion_clap
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from importlib import import_module
from sklearn.decomposition import PCA

from st_ito.utils import (
    load_param_model,
    load_clap_model,
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
    load_deepafx_st_model,
    get_deepafx_st_embeds,
    load_fx_encoder_model,
    get_fx_encoder_embeds,
    load_beats_model,
    get_beats_embeds,
)


def embed(
    audio_dir: str,
    use_gpu: bool = False,
    output_dir: str = "output",
    include_c: bool = False,
):
    # find all style dirs
    style_dirs = glob.glob(os.path.join(audio_dir, "style_*"))

    # example_dirs = example_dirs[:10]
    print(f"Found {len(style_dirs)} examples.")
    print(style_dirs)

    # load models
    print("Loading models...")
    if False:
        models = {
            "param-panns-concat": {
                "model": load_param_model(
                    "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt",
                    use_gpu=args.use_gpu,
                ),
                "embed_fn": get_param_embeds,
            },
            "param-panns-concat-adv": {
                "model": load_param_model(
                    "/import/c4dm-datasets-ext/lcap/param/lcap-param/bn2u0cil/checkpoints/last.ckpt",  # 1.0
                    use_gpu=args.use_gpu,
                ),
                "embed_fn": get_param_embeds,
            },
            "param-panns-concat-adv-32": {
                "model": load_param_model(
                    "/import/c4dm-datasets-ext/lcap/param/lcap-param/uobr3qvz/checkpoints/last.ckpt",  # 5.0
                    use_gpu=args.use_gpu,
                ),
                "embed_fn": get_param_embeds,
            },
            "fx-encoder": {
                "model": load_fx_encoder_model(use_gpu=args.use_gpu),
                "embed_fn": get_fx_encoder_embeds,
            },
            "mfccs": {
                "model": load_mfcc_feature_extractor(use_gpu=use_gpu),
                "embed_fn": get_mfcc_feature_embeds,
            },
            "clap": {
                "model": load_clap_model(use_gpu=use_gpu),
                "embed_fn": get_clap_embeds,
            },
            "BEATs": {
                "model": load_beats_model(use_gpu=use_gpu),
                "embed_fn": get_beats_embeds,
            },
            # "param-clap-ft": {
            #    "model": load_param_model(
            #        "/import/c4dm-datasets-ext/lcap/lcap/oci5e3e5/checkpoints/epoch=51-step=113724.ckpt",
            #        use_gpu=args.use_gpu,
            #    ),
            #    "embed_fn": get_param_embeds,
            # 1},
            # ---------------- style transfer models ---------------- #
            "deepafx-st+": {
                "model": load_deepafx_st_model(
                    "/import/c4dm-datasets-ext/lcap/style/lcap-style/n9y2bgn2/checkpoints/last.ckpt",
                    use_gpu=args.use_gpu,
                    encoder_only=True,
                ),
                "embed_fn": get_deepafx_st_embeds,
            },
        }

    models = {
        "fx-encoder": {
            "model": load_fx_encoder_model(use_gpu=args.use_gpu),
            "embed_fn": get_fx_encoder_embeds,
        },
    }

    # get embeddings from each model
    print("Getting embeddings...")
    embeddings = {}
    effect_names = {}

    for idx, (model_name, model_dict) in enumerate(models.items()):
        model = model_dict["model"]
        embed_fn = model_dict["embed_fn"]

        embedding_list = []
        effect_list = []
        content_list = []

        for style_dir in tqdm(style_dirs):
            audio_filepaths = glob.glob(os.path.join(style_dir, "*.flac"))
            for audio_filepath in audio_filepaths:
                # load audio
                output_audio, sample_rate = torchaudio.load(
                    audio_filepath, backend="soundfile"
                )

                # move to GPU
                if args.use_gpu:
                    output_audio = output_audio.cuda()

                # add batch dim
                output_audio = output_audio.unsqueeze(0)

                # get embeddings
                output_embed = embed_fn(output_audio, model, sample_rate)

                # for now, just take the first embed in the dict
                output_embed = output_embed[list(output_embed.keys())[0]]

                # save use the style name as the effect name
                effect_name = os.path.basename(audio_filepath).split("_")[1]
                content_name = os.path.basename(audio_filepath).split("_")[2]

                embedding_list.append(output_embed.cpu().numpy())
                effect_list.append(effect_name)
                content_list.append(content_name)

        # convert to a single numpy array and create a list that indexes the source effect
        # stack
        embedding_list = np.vstack(embedding_list)
        effect_list = np.hstack(effect_list)
        content_list = np.hstack(content_list)

        print(embedding_list.shape)
        print(effect_list.shape)
        print(content_list.shape)

        embed_dir = os.path.join(output_dir, "embeds")

        os.makedirs(embed_dir, exist_ok=True)

        # save
        np.save(os.path.join(embed_dir, f"{model_name}_embeddings.npy"), embedding_list)
        np.save(os.path.join(embed_dir, f"{model_name}_effects.npy"), effect_list)
        np.save(os.path.join(embed_dir, f"{model_name}_content.npy"), content_list)


def plot(output_dir: str = "output", l2_norm: bool = False):

    embed_dir = "embeds"

    # find all embedding files
    embedding_filepaths = glob.glob(
        os.path.join(output_dir, embed_dir, "*_embeddings.npy")
    )
    effect_filepaths = glob.glob(os.path.join(output_dir, embed_dir, "*_effects.npy"))
    content_filepaths = glob.glob(os.path.join(output_dir, embed_dir, "*_content.npy"))

    assert len(embedding_filepaths) == len(
        effect_filepaths
    )  # must have the same number of files

    for embedding_filepath, effect_filepath, content_filepath in zip(
        embedding_filepaths, effect_filepaths, content_filepaths
    ):

        embedding_model_name = os.path.basename(embedding_filepath).replace(
            "_embeddings.npy", ""
        )
        print(embedding_model_name)

        # load
        embedding_list = np.load(embedding_filepath)
        effect_list = np.load(effect_filepath)
        content_list = np.load(content_filepath)

        content_names = sorted(np.unique(content_list))
        effect_names = sorted(np.unique(effect_list))

        # convert each string in effect_list to a integer index
        effect_list_idx = [effect_names.index(effect_str) for effect_str in effect_list]

        content_list_idx = [
            content_names.index(content_str) for content_str in content_list
        ]

        num_effects = len(np.unique(effect_list))

        if l2_norm:
            embedding_list = (
                embedding_list / np.linalg.norm(embedding_list, axis=1)[:, None]
            )

        for embed_type in ["umap", "pca"]:
            # create new plot for each embedding model
            fig, axs = plt.subplots(
                figsize=(10, 5),
                ncols=2,
            )

            if embed_type == "umap":
                # create 2d projection with umap once
                proj_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.4).fit_transform(
                    embedding_list
                )
            else:
                # create 2d projection with pca
                pca = PCA(n_components=2)
                proj_embeddings = pca.fit_transform(embedding_list)

            print(proj_embeddings.shape)

            for row_idx, plot_mode in enumerate(["style", "content"]):
                # create colors
                if plot_mode == "style":
                    effect_colors = [
                        "#a6cee3",
                        "#1f78b4",
                        "#b2df8a",
                        "#33a02c",
                        "#fb9a99",
                        "#e31a1c",
                        "#fdbf6f",
                        "#ff7f00",
                        "#cab2d6",
                        "#999999",
                    ]
                    color_list = [effect_colors[e_idx] for e_idx in effect_list_idx]

                elif plot_mode == "content":
                    content_colors = [
                        "#66c2a5",
                        "#fc8d62",
                        "#8da0cb",
                        "#e78ac3",
                        "#a6d854",
                    ]
                    color_list = [content_colors[c_idx] for c_idx in content_list_idx]

                axs[row_idx].set_title("Effects" if plot_mode == "style" else "Content")
                axs[row_idx].scatter(
                    proj_embeddings[:, 0],
                    proj_embeddings[:, 1],
                    c=color_list,
                    s=10,
                    alpha=1.0,
                    linewidths=0.2,
                    edgecolors="white",
                )

                # turn axis off
                axs[row_idx].axis("off")

            # add a lengend to the first plot
            axs[0].legend(
                [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                    )
                    for color in effect_colors
                ],
                [effect_name.capitalize() for effect_name in effect_names],
                loc="upper left",
                frameon=False,
                bbox_to_anchor=(1, 1),
            )
            plt.subplots_adjust(right=0.75)

            full_content_names = {
                "speech": "Speech",
                "drums": "Drums",
                "acoustic-guitar": "AcGuitar",
                "electric-guitar": "EGuitar",
                "vocals": "Vocals",
            }

            content_name_labels = [
                full_content_names[c_name] for c_name in content_names
            ]

            print(content_name_labels)

            axs[1].legend(
                [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                    )
                    for color in content_colors
                ],
                content_name_labels,
                loc="upper left",
                frameon=False,
                bbox_to_anchor=(1, 1),
            )
            plt.subplots_adjust(right=0.75)

            plt.tight_layout(w_pad=7, h_pad=7)

            if l2_norm:
                filepath = os.path.join(
                    output_dir, f"{embedding_model_name}_{embed_type}_l2_norm"
                )
            else:
                filepath = os.path.join(
                    output_dir, f"{embedding_model_name}_{embed_type}"
                )

            plt.savefig(filepath + ".png", dpi=300)
            plt.savefig(filepath + ".pdf")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        help="Paths to root directory of audio files.",
        default="/import/c4dm-datasets-ext/lcap-datasets/psm-benchmark-vis-dataset",
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--l2_norm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, "psm-vis")
    os.makedirs(output_dir, exist_ok=True)

    if args.embed:
        print("Embedding...")
        embed(
            args.audio_dir,
            use_gpu=args.use_gpu,
            output_dir=output_dir,
        )

    if args.plot:
        print("Plotting...")
        plot(
            output_dir=output_dir,
            l2_norm=args.l2_norm,
        )
