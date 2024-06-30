import os
import glob
import json

if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/lcap-datasets/MUSDB18HQ+vst/plugins"
    # find all json files
    json_filepaths = glob.glob(os.path.join(root_dir, "*.json"), recursive=True)

    overall_max = 0

    for json_filepath in json_filepaths:
        with open(json_filepath, "r") as f:
            data = json.load(f)

        for parameter in data["parameters"]:
            name = parameter["name"]
            valid_values = parameter["valid_values"]
            max_value = len(valid_values)

            if max_value > overall_max:
                overall_max = max_value
            print(name, max_value, overall_max)
