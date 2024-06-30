# last minute script to evaluate objective metric
import os
import glob
import json
import torch
import torchaudio
import numpy as np

from st_ito.utils import (
    load_param_model,
    get_param_embeds,
)

if __name__ == "__main__":

    results = {}

    model = load_param_model(
        "/import/c4dm-datasets-ext/lcap/param/lcap-param/myjhyqlz/checkpoints/last.ckpt"
    )  # param-panns
    embed_func = get_param_embeds

    # load audio examples that have been rendered
    root_dir = "/import/c4dm-datasets-ext/lcap-datasets/output/synthetic-06042024"

    example_dirs = glob.glob(os.path.join(root_dir, "*"))
    example_dirs = [
        example_dir for example_dir in example_dirs if os.path.isdir(example_dir)
    ]
    print(example_dirs)

    for example_dir in example_dirs:

        example_id = os.path.basename(example_dir)
        print(example_id)
        example_test_case_name = example_id.split("->")[-1].split("-")[0]
        example_test_case_id = example_id.split("->")[-1].split("-")[1]
        example_test_case = f"{example_test_case_name}-{example_test_case_id}"
        print(example_test_case)

        if example_test_case not in results:
            results[example_test_case] = {}

        if example_id not in results[example_test_case]:
            results[example_test_case][example_id] = {}

        audio_data = {}
        # load audio files
        audio_filepaths = glob.glob(os.path.join(example_dir, "*.wav"))
        for audio_filepath in audio_filepaths:
            audio_filename = os.path.basename(audio_filepath)
            audio_filename = os.path.splitext(audio_filename)[0]
            x, sr = torchaudio.load(audio_filepath, backend="soundfile")
            audio_data[audio_filename] = x

        # extract embedding from the target
        for audio_filename, audio_tensor in audio_data.items():
            if "target" in audio_filename:
                target_embeds = embed_func(audio_tensor.unsqueeze(0), model, 48000)

        # extract embedding from the other audio files and compare cos sim to target
        for audio_filename, audio_tensor in audio_data.items():
            if "random_pb" in audio_filename:
                method_name = "random_pb"
            elif "random_vst" in audio_filename:
                method_name = "random_vst"
            elif "style-es (param-panns)_pb" in audio_filename:
                method_name = "style-es (param-panns)_pb"
            elif "style-es (param-panns)_vst" in audio_filename:
                method_name = "style-es (param-panns)_vst"
            else:
                method_name = audio_filename.split("_")[-1]
            if "target" not in audio_filename:
                other_embeds = embed_func(audio_tensor.unsqueeze(0), model, 48000)

                cos_sims = []
                for embed_name, embed in other_embeds.items():
                    target_embed = target_embeds[embed_name]
                    cos_sim = torch.nn.functional.cosine_similarity(target_embed, embed)
                    cos_sims.append(cos_sim.item())

                avg_cos_sim = np.mean(cos_sims)
                print(f"{audio_filename}: {avg_cos_sim}")
                results[example_test_case][example_id][method_name] = avg_cos_sim

    result_filepath = "output/synthetic-06042024.json"
    with open(result_filepath, "w") as f:
        json.dump(results, f)
