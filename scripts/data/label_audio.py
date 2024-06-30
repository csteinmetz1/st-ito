import os
import glob
import torch
import random
import tarfile
import torchaudio
import transformers

from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def get_classifier_logits(extractor, model, audio: torch.Tensor, sr: int):
    """Get the logits from a pretrained audio classifier (AST trained on AudioSet) for a given audio file."""
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    audio = audio.mean(dim=0, keepdim=False)  # must be mono

    # Extract features
    outputs = extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
    features = outputs.input_values

    with torch.no_grad():
        logits = model(features).logits

    # here is how you get the classes if you want that...
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    return logits, predicted_class_ids, predicted_label


def torchaudio_decode(b):
    x, sr = torchaudio.load(b, format="FLAC")
    return x


if __name__ == "__main__":

    # Load model directly
    extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    num_classes = len(model.config.id2label)

    tar_files = [
        "/import/c4dm-datasets-ext/lcap-datasets/20k/musdb18-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/mtg-jamendo-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/ursing-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/fsd50k-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/librispeech-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/medley-solos-db-vst3-presets.tar",
        "/import/c4dm-datasets-ext/lcap-datasets/20k/guitarset-vst3-presets.tar",
    ]

    out_dir = "/import/c4dm-datasets-ext/lcap-datasets/20k"
    counter = 0

    for tar_file in tar_files:
        tar_filename = os.path.basename(tar_file).split(".")[0]
        tar_handle = tarfile.open(tar_file, "r:")
        tar_handle.next()  # skip the first root directory
        tar_handle.next()  # skip the first example directory

        # create a dictionary to store class labels
        # check if the labels file already exists
        if os.path.exists(os.path.join(out_dir, f"{tar_filename}-labels.pt")):
            labels = torch.load(os.path.join(out_dir, f"{tar_filename}-labels.pt"))
        else:
            labels = {}

        members = {}
        done = False
        handle_reset = False
        pbar = tqdm()
        while not done:
            # get next member
            member = tar_handle.next()

            if member is None:
                # we have reached the end of this tar file
                # so, we will close it and remove open tar handle
                print(f"End of tar file: {tar_filename}")
                # close this tar handle
                tar_handle.close()
                handle_reset = True
                break

            if "input.flac" in member.name:
                # get input file
                input_file = tar_handle.extractfile(member)
                input = torchaudio_decode(input_file)

                # get the directory name one level up
                parent_dir = os.path.dirname(member.name).split("/")[-1]

                if parent_dir in labels:
                    # we have already classified this audio
                    logits = labels[parent_dir]["logits"]
                    predicted_class_ids = labels[parent_dir]["predicted_class_ids"]
                    predicted_label = labels[parent_dir]["predicted_label"]
                else:
                    # classify this audio
                    logits, predicted_class_ids, predicted_label = (
                        get_classifier_logits(extractor, model, input, 48000)
                    )

                    # store the result in new file alongside tarfile
                    labels[parent_dir] = {
                        "predicted_label": predicted_label,
                        "predicted_class_ids": predicted_class_ids,
                        "logits": logits,
                    }

                counter += 1
                print(
                    f"idx: {counter+1} {tar_filename} {parent_dir} label: {predicted_label} class_id: {predicted_class_ids}"
                )
            else:
                continue

            if counter % 100 == 0:
                filepath = os.path.join(out_dir, f"{tar_filename}-labels.pt")
                torch.save(labels, filepath)

        # when done with this tarfile save the labels
        filepath = os.path.join(out_dir, f"{tar_filename}-labels.pt")
        torch.save(labels, filepath)
