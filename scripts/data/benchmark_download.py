import os
import yt_dlp
import argparse
import torchaudio
import pandas as pd
from tqdm import tqdm


def get_seconds(time_str: str):
    """Get seconds from time."""
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "pst-benchmark-dataset")
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in ["speech", "vocals", "guitar", "music"]:
        os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
        df = pd.read_csv(args.csv_path)

        for index, row in tqdm(df.iterrows()):
            url = row["url"]

            audio_type = row["audio_type"]

            if audio_type != dataset_name:
                continue

            start = get_seconds(row["start"])
            end = get_seconds(row["end"])

            ytid = url.split("=")[-1]

            # check if file already exists
            if os.path.exists(os.path.join(output_dir, dataset_name, f"{ytid}.wav")):
                print(f"File {ytid}.wav already exists. Skipping download.")
                continue

            print(f"Downloading {url}")

            # download audio save to output_dir
            ydl_opts = {
                "format": "m4a/bestaudio/best",
                "postprocessors": [
                    {  # Extract audio using ffmpeg
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                    }
                ],
                "outtmpl": os.path.join(output_dir, dataset_name, "%(id)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            audio, sr = torchaudio.load(
                os.path.join(output_dir, dataset_name, f"{ytid}.mp3"),
                backend="soundfile",
            )
            audio = torchaudio.functional.resample(audio, sr, args.sample_rate)

            # extract clip based on start and end
            start_frame = int(start * args.sample_rate)
            end_frame = int(end * args.sample_rate)
            audio = audio[:, start_frame:end_frame]

            # peak normalization
            audio /= audio.abs().max().clamp(min=1e-8)

            # save to output_dir as wav
            torchaudio.save(
                os.path.join(output_dir, dataset_name, f"{ytid}.wav"),
                audio,
                sample_rate=args.sample_rate,
                backend="soundfile",
            )

            # delete m4a file
            os.remove(os.path.join(output_dir, dataset_name, f"{ytid}.mp3"))
