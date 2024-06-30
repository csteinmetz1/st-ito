import os
import json
import glob
import torch
import torchaudio
import numpy as np

from lcap.utils import load_param_model, get_param_embeds

if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/lcap-datasets/listening-test-v2"
    content_types = ["music", "speech", "vocals"]

    # load similarity metric
    model = load_param_model(
        "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"  # param-panns
    )
    embed_func = get_param_embeds

    results = {}

    # load the audio files from each listening test example
    for content_type in content_types:
        # find all directories
        directories = glob.glob(os.path.join(root_dir, content_type, "*"))
        for directory in directories:
            # find all audio files
            audio_files = {}
            audio_filepaths = glob.glob(os.path.join(directory, "*.wav"))
            for audio_filepath in audio_filepaths:
                audio_filename = os.path.basename(audio_filepath).replace(".wav", "")
                x, sr = torchaudio.load(audio_filepath, backend="soundfile")
                audio_files[audio_filename] = x

            # compute the similarity between each audio file and the target
            for audio_filename, audio_file in audio_files.items():
                if "target" not in audio_filename:
                    continue
                target_audio = audio_file
                target_embeds = embed_func(target_audio.unsqueeze(0), model, 48000)

            for audio_filename, audio_file in audio_files.items():
                if audio_filename == "target":
                    continue

                method_name = audio_filename.split("_")[-1]
                print(method_name)

                audio_embeds = embed_func(audio_file.unsqueeze(0), model, 48000)

                similarities = []
                for embed_name, audio_embed in audio_embeds.items():
                    target_embed = target_embeds[embed_name]

                    similarity = torch.nn.functional.cosine_similarity(
                        target_embed, audio_embed
                    ).item()
                    print(embed_name, similarity)
                    similarities.append(similarity)
                similarity = np.mean(similarities)
                print(f"{content_type} - {audio_filename}: {similarity}")
                results[audio_filename] = similarity

    # save the final results to json file for loading in boxplots
    with open("output/listen-similarity-results.json", "w") as f:
        json.dump(results, f)
