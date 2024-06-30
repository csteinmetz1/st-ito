import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("speech_input", type=str)
    parser.add_argument("music_input", type=str)
    parser.add_argument("--combine", action="store_true")

    args = parser.parse_args()

    with open(args.music_input, "r") as f:
        music_results = json.load(f)

    with open(args.speech_input, "r") as f:
        speech_results = json.load(f)

    style_names = ["telephone", "bright", "warm", "broadcast", "neutral"]
    model_names = [
        "mfccs",
        "mir-features",
        "clap",
        "wav2vec2",
        "wav2clip",
        "vggish",
        "fx_encoder",
        "deepafx-st",
        "deepafx-st+",
        "param-panns-l2-concat",
        "param-clap",
        "beats",
    ]

    full_model_names = {
        "mfccs": "MFCCs",
        "mir-features": "MIR Feats.",
        "clap": "CLAP",
        "wav2vec2": "Wav2Vec2",
        "wav2clip": "Wav2Clip",
        "vggish": "VGGish",
        "fx_encoder": "FX Encoder",
        "deepafx-st": "DeepAFx-ST",
        "deepafx-st+": "DeepAFx-ST+",
        "param-panns-l2-concat": "Param-Enc (ours)",
        "param-clap": "CLAP-Param",
        "beats": "BEATs",
    }

    if args.combine:
        # create combined results between speech and input in a single table
        combined_results = {}
        for model_name in model_names:
            if model_name not in combined_results:
                combined_results[model_name] = {}

            for style_name in style_names:
                speech_result = speech_results[model_name][style_name]
                music_result = music_results[model_name][style_name]
                # average the results
                avg_result = (speech_result["avg"] + music_result["avg"]) / 2
                combined_results[model_name][style_name] = avg_result

        for key, val in combined_results.items():
            val["overall"] = sum(val.values()) / len(val)

        for key, val in combined_results.items():
            print(key, val)

        table = "\\toprule \n"
        table += "Rep. & TL & BR & WM & BC & NT & AVG \\\\ \\midrule \n"

        for model_name in model_names:
            table += f"{full_model_names[model_name]} & "

            for style_name in style_names:
                table += f"{combined_results[model_name][style_name]:.2f} & "
            table += f"{combined_results[model_name]['overall']:.2f} \\\\ \n"

        print(table)

    else:
        table = "\\toprule \n"
        table += "Feature & Telephone & Bright & Warm & Broadcast & Neutral & Average & Telephone & Bright & Warm & Broadcast & Neutral & Average \\\\ \\midrule \n"

        for model_name in model_names:
            table += f"{full_model_names[model_name]} & "

            for style_name in style_names:
                table += f"{speech_results[model_name][style_name]['avg']:.2f} & "
            table += f"{speech_results[model_name]['overall']:.2f} & "

            for style_name in style_names:
                table += f"{music_results[model_name][style_name]['avg']:.2f} & "
            table += f"{music_results[model_name]['overall']:.2f} \\\\ \n"

        print(table)
